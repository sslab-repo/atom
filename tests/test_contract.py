"""Contract + registry conformance tests, incl. ADR-0005 v1 bars."""

import pytest

from atom.contract import (
    Adaptation,
    Declaration,
    DetectionSetting,
    Modality,
    Module,
    ModuleKind,
    Operation,
    Paradigm,
    RunContext,
    RunResult,
    SearchSpace,
    TaskFamily,
    UnsupportedOperation,
)
from atom.registries import InvalidDeclarationError, find, register


def _decl(**kw):
    base = dict(
        name="m",
        version="0.1",
        kind=ModuleKind.METHOD,
        task_families=frozenset({TaskFamily.CLASSIFICATION}),
        modalities=frozenset({Modality.TABULAR}),
    )
    base.update(kw)
    return Declaration(**base)


class _Stub(Module):
    DECL = None

    def declares(self):
        return self.DECL

    def space(self):
        return SearchSpace()

    def run(self, ctx: RunContext) -> RunResult:
        if ctx.operation is Operation.FIT:
            return RunResult(artifacts={"ok": True})
        raise UnsupportedOperation(ctx.operation)


def test_register_and_find():
    class M(_Stub):
        DECL = _decl(name="stub-clf", category="linear")

    register(M)
    matches = find(ModuleKind.METHOD, TaskFamily.CLASSIFICATION, Modality.TABULAR)
    assert any(m.declares().name == "stub-clf" for m in matches)
    assert not find(ModuleKind.METHOD, TaskFamily.CLASSIFICATION, Modality.IMAGE)


def test_foundation_requires_adaptation():
    class M(_Stub):
        DECL = _decl(name="fm-bad", paradigm=Paradigm.FOUNDATION)

    with pytest.raises(InvalidDeclarationError, match="adaptation"):
        register(M)


def test_v1_barred_adaptations_rejected():
    class M(_Stub):
        DECL = _decl(name="fm-ft", paradigm=Paradigm.FOUNDATION, adaptation=Adaptation.FULL_FINETUNE)

    with pytest.raises(InvalidDeclarationError, match="barred"):
        register(M)


def test_anomaly_requires_setting():
    class Bad(_Stub):
        DECL = _decl(name="ad-bad", task_families=frozenset({TaskFamily.ANOMALY_DETECTION}))

    with pytest.raises(InvalidDeclarationError, match="setting"):
        register(Bad)

    class Good(_Stub):
        DECL = _decl(
            name="ad-good",
            task_families=frozenset({TaskFamily.ANOMALY_DETECTION}),
            setting=DetectionSetting.OUTLIER,
        )

    register(Good)  # no raise


def test_unsupported_operation():
    class M(_Stub):
        DECL = _decl(name="op-stub")

    with pytest.raises(UnsupportedOperation):
        M().run(RunContext(operation=Operation.GENERATE, data=None))
