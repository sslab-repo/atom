"""Optional PyTorch deep tier (ADR-0008): sequence classifiers for raw
time-series packages (`atom pack --type timeseries --ts-layout raw`).

Registers ONLY when torch is importable — on a no-torch machine this module
adds nothing and the CPU tier still runs. Modules reshape the flat feature
matrix X (n, C*L) into (n, C, L) using the `seq_shape` handed in via the FIT
context, train on the resolved device (cuda/mps/cpu), and predict probabilities.
"""

from __future__ import annotations

from atom.contract import (
    Declaration,
    Modality,
    Module,
    ModuleKind,
    Operation,
    Parameter,
    ResourceHints,
    RunContext,
    RunResult,
    SearchSpace,
    TaskFamily,
    UnsupportedOperation,
)
from atom.registries import register

try:  # pragma: no cover - depends on environment
    import numpy as np
    import torch
    import torch.nn as nn

    from atom.core.device import resolve_device

    _TAB = frozenset({Modality.TABULAR})
    _FIDELITY = (0.1, 0.33, 1.0)

    class _Conv1DBody(nn.Module):
        def __init__(self, n_channels: int, hidden: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten())

        def forward(self, x):
            return self.net(x)

    class _LSTMBody(nn.Module):
        def __init__(self, n_channels: int, hidden: int):
            super().__init__()
            self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)

        def forward(self, x):                # (n, C, L) -> (n, L, C) for the LSTM
            out, _ = self.lstm(x.transpose(1, 2))
            return out[:, -1, :]             # last timestep

    class _SeqNet(nn.Module):
        """Standardizes internally (buffers) so the exported graph is
        self-contained: raw (n, C*L) -> reshape -> body -> softmax proba."""

        def __init__(self, body: nn.Module, head_in: int, n_classes: int,
                     n_channels: int, seq_len: int, mean, std):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
            self.C, self.L = n_channels, seq_len
            self.body = body
            self.head = nn.Linear(head_in, n_classes)

        def forward(self, x):
            # complete, self-contained graph: NaN-fill + standardize + reshape are
            # all IN the graph, so the exported ONNX takes RAW features.
            x = torch.nan_to_num(x)
            x = (x - self.mean) / self.std
            x = x.reshape(-1, self.C, self.L)
            return torch.softmax(self.head(self.body(x)), dim=1)

    class _TorchSeqClassifier(Module):
        NAME: str
        CATEGORY: str
        FAMILY = TaskFamily.CLASSIFICATION
        SELF_PREPROCESSING = True  # net does NaN-fill + standardize; skip sklearn chain

        def declares(self) -> Declaration:
            return Declaration(
                name=self.NAME, version="1.0", kind=ModuleKind.METHOD,
                task_families=frozenset({TaskFamily.CLASSIFICATION}), modalities=_TAB,
                category=self.CATEGORY, exportable=True)

        def hints(self) -> ResourceHints:
            return ResourceHints(cpu=1, gpu=0, fidelity_levels=_FIDELITY)

        def space(self) -> SearchSpace:
            return SearchSpace((
                Parameter("hidden", "categorical", (32, 64, 128), 64),
                Parameter("lr", "log_float", (1e-3, 1e-2), 3e-3),
                Parameter("epochs", "int", (20, 80), 40),
            ))

        def _body(self, config, C, L) -> tuple[nn.Module, int]:
            raise NotImplementedError

        def run(self, ctx: RunContext) -> RunResult:
            if ctx.operation is Operation.FIT:
                return self._fit(ctx)
            if ctx.operation is Operation.SCORE:
                return self._score(ctx)
            raise UnsupportedOperation(ctx.operation)

        def _fit(self, ctx: RunContext) -> RunResult:
            X = np.asarray(ctx.data["X"], dtype=np.float32)
            y_raw = np.asarray(ctx.data["y"]).astype(str)
            classes = sorted(set(y_raw.tolist()))
            cidx = {c: i for i, c in enumerate(classes)}
            y = np.array([cidx[v] for v in y_raw], dtype=np.int64)
            shape = ctx.data.get("seq_shape")
            if shape is None:  # not a raw-sequence package -> treat as 1 channel
                shape = (1, X.shape[1])
            C, L = int(shape[0]), int(shape[1])
            seed = int(ctx.config.get("_seed", 0))
            torch.manual_seed(seed)
            X = np.nan_to_num(X, copy=False)
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-6
            dev = resolve_device()

            body, head_in = self._body(ctx.config, C, L)
            net = _SeqNet(body, head_in, len(classes), C, L, mean, std).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=float(ctx.config.get("lr", 3e-3)))
            lossf = nn.CrossEntropyLoss()
            xt = torch.tensor(X, device=dev)
            yt = torch.tensor(y, device=dev)
            net.train()
            n = len(X)
            bs = min(256, n)
            for _ in range(int(ctx.config.get("epochs", 40))):
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad()
                    lossf(torch.log(net(xt[idx]) + 1e-9), yt[idx]).backward()
                    opt.step()
            net.eval()
            net = net.to("cpu")  # inference + ONNX export on CPU (train used the GPU)
            return RunResult(artifacts={"net": net, "classes": classes, "n_features": X.shape[1]})

        def _score(self, ctx: RunContext) -> RunResult:
            net = ctx.artifacts["net"]
            classes = ctx.artifacts["classes"]
            X = np.asarray(ctx.data["X"], dtype=np.float32)  # net NaN-fills internally
            with torch.no_grad():
                proba = net(torch.tensor(X)).numpy()
            pred = np.array([classes[i] for i in proba.argmax(axis=1)], dtype=object)
            return RunResult(outputs={"pred": pred, "proba": proba, "classes": classes})

    @register
    class Conv1DClassifierM(_TorchSeqClassifier):
        NAME, CATEGORY = "conv1d-classifier", "neural-network"

        def _body(self, config, C, L):
            h = int(config.get("hidden", 64))
            return _Conv1DBody(C, h), h

    @register
    class LSTMClassifierM(_TorchSeqClassifier):
        NAME, CATEGORY = "lstm-classifier", "neural-network"

        def _body(self, config, C, L):
            h = int(config.get("hidden", 64))
            return _LSTMBody(C, h), h

except ImportError:  # torch absent: register nothing (CPU tier still works)
    pass
