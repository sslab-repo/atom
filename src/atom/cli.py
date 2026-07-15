"""ATOM CLI. M1 commands: inspect, pack."""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_inspect(args: argparse.Namespace) -> int:
    from atom.core.ingest import fingerprint
    from atom.data import DatasetPackage

    with DatasetPackage.open(args.package) as pkg:
        fp = fingerprint(pkg, sample_rows=args.sample_rows)
    if args.json:
        print(json.dumps(fp.to_dict(), indent=2))
        return 0

    m = pkg.manifest
    print(f"package    : {m.name}  [{m.manifest_version}]")
    print(f"id         : {fp.package_id[:19]}…")
    print(f"modality   : {fp.modality} / {fp.dataset_type}")
    print("samples    : " + "  ".join(f"{s}={c:,}" for s, c in fp.counts.items()))
    print(f"columns    : {fp.n_columns}  (profiled {fp.sampled_rows:,} train rows)")
    roles = ", ".join(f"{k}={v}" for k, v in fp.roles.items() if v) or "(none declared)"
    print(f"roles      : {roles}")
    if fp.target_classes:
        top = list(fp.target_classes.items())[:8]
        rest = len(fp.target_classes) - len(top)
        line = "  ".join(f"{k or '∅'}:{v:,}" for k, v in top)
        print(f"target dist: {line}" + (f"  (+{rest} more)" if rest > 0 else ""))
    if fp.quality_flags:
        print("flags      : " + "; ".join(fp.quality_flags))
    n_show = min(len(fp.columns), args.columns)
    if n_show:
        print(f"--- columns (first {n_show}) ---")
        for c in fp.columns[:n_show]:
            extra = f"  inf={c.inf_rate:.1%}" if c.inf_rate else ""
            print(
                f"  {c.name:<40} {c.dtype:<8} missing={c.missing_rate:.1%}"
                f" distinct≈{c.distinct_sampled}{extra}"
            )
    return 0


def _confirm_gate(task, fp, assume_yes: bool) -> bool:
    print("=== inferred task (confirm gate) ===")
    print(f"  family   : {task.family.value}"
          + (f" / {task.setting.value}" if task.setting else ""))
    print(f"  target   : {task.target}   metric: {task.primary_metric}"
          + (f"   classes: {task.n_classes}" if task.n_classes else ""))
    if task.imbalanced:
        print("  note     : severe class imbalance (rare-class classification)")
    for n in task.notes:
        print(f"  note     : {n}")
    if assume_yes or not sys.stdin.isatty():
        print("  proceeding (--yes)")
        return True
    return input("proceed with this task? [y/N] ").strip().lower() in ("y", "yes")


def _cmd_run(args: argparse.Namespace) -> int:
    from atom.core.run import run_package

    outcome = run_package(
        args.package,
        target=args.target,
        wall_clock_s=args.time_budget,
        max_trials=args.max_trials,
        min_trials=args.min_trials,
        max_rows=args.max_rows,
        out_root=args.out,
        kb_root=args.kb,
        force_task=args.task,
        include_experimental=args.include_experimental,
        seed=args.seed,
        confirm=lambda task, fp: _confirm_gate(task, fp, args.yes),
        progress=lambda s: print(f"  {s}"),
    )
    print("=== result ===")
    print(f"  final    : {outcome.final_kind}   trials: {outcome.n_trials}"
          f"   elapsed: {outcome.elapsed_s:.0f}s")
    print(f"  val      : {outcome.task.primary_metric}={abs(outcome.val_score):.4f}")
    print("  test     : " + "  ".join(f"{k}={v:.4f}" for k, v in outcome.test_metrics.items()))
    print(f"  artifacts: {outcome.run_dir}")
    return 0


def _cmd_modules(args: argparse.Namespace) -> int:
    from atom.registries import all_modules, discover, lifecycle_of
    from atom.registries.builtins import load_builtins

    load_builtins()
    n_ext = discover()
    if n_ext:
        print(f"(discovered {n_ext} external module(s) via entry points)")
    modules = sorted(all_modules(), key=lambda m: (m.declares().kind.value, m.declares().name))
    if args.action == "list":
        for m in modules:
            d = m.declares()
            fams = ",".join(sorted(f.value for f in d.task_families))
            print(f"  {d.kind.value:<14} {d.name}@{d.version:<6} [{lifecycle_of(m)}] "
                  f"{d.category:<20} {fams}")
        return 0
    failures = 0
    for m in modules:
        status = _smoke(m)
        print(f"  {'PASS' if status == 'PASS' else 'FAIL':<5} {m.declares().name}"
              + ("" if status == "PASS" else f"  ({status})"))
        failures += status != "PASS"
    print(f"{len(modules) - failures}/{len(modules)} modules pass the smoke gate")
    return 1 if failures else 0


def _smoke(module) -> str:
    """Contract-conformance smoke: declaration valid + a tiny synthetic run."""
    import numpy as np

    from atom.contract import ModuleKind, Operation, RunContext, TaskFamily

    d = module.declares()
    problems = d.validate()
    if problems:
        return "; ".join(problems)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    try:
        if d.kind is ModuleKind.PREPROCESSING:
            Xn = X.copy()
            Xn[0, 0] = np.nan
            r = module.run(RunContext(Operation.FIT, {"X": Xn}))
            module.run(RunContext(Operation.TRANSFORM, {"X": Xn}, artifacts=r.artifacts))
        elif d.kind is ModuleKind.METHOD:
            fam = next(iter(d.task_families))
            if fam in (TaskFamily.CLASSIFICATION, TaskFamily.REGRESSION):
                y = ((X[:, 0] > 0).astype(str) if fam is TaskFamily.CLASSIFICATION
                     else X[:, 0] * 2 + 1)
                r = module.run(RunContext(Operation.FIT, {"X": X, "y": y}))
                module.run(RunContext(Operation.SCORE, {"X": X}, artifacts=r.artifacts))
            elif fam is TaskFamily.DIMENSION_REDUCTION:
                r = module.run(RunContext(Operation.FIT, {"X": X, "y": None}))
                module.run(RunContext(Operation.TRANSFORM, {"X": X}, artifacts=r.artifacts))
            else:  # clustering, anomaly
                r = module.run(RunContext(Operation.FIT, {"X": X, "y": None}))
                module.run(RunContext(Operation.SCORE, {"X": X}, artifacts=r.artifacts))
        elif d.kind is ModuleKind.METRIC:
            fam = next(iter(d.task_families))
            pred = (X[:, 0] > 0).astype(str)
            data = {"y_true": pred, "pred": pred, "X": X}
            if fam is TaskFamily.REGRESSION:
                data = {"y_true": X[:, 0], "pred": X[:, 0]}
            module.run(RunContext(Operation.SCORE, data))
        return "PASS"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch a remote dataset and convert it into an ADP (packager source)."""
    if not args.source.startswith("kaggle:"):
        print("supported sources: kaggle:<owner/dataset>", file=sys.stderr)
        return 2
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed — pip install 'atom-ai[kaggle]'", file=sys.stderr)
        return 2
    import os

    slug = args.source.split(":", 1)[1]
    path = kagglehub.dataset_download(slug)
    csvs = sorted((os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs
                   if f.lower().endswith(".csv")), key=os.path.getsize, reverse=True)
    if not csvs:
        print(f"no CSV files in {slug} ({path}) — pack manually", file=sys.stderr)
        return 1
    if args.file:
        csvs = [c for c in csvs if os.path.basename(c) == args.file] or csvs
    from atom.data import pack_csv

    name = args.name or slug.replace("/", "-")
    root = pack_csv(csvs[0], args.out, name=name, target=args.target)
    print(f"fetched {slug} ({os.path.basename(csvs[0])}) -> ADP: {root}")
    return 0


def _cmd_pack(args: argparse.Namespace) -> int:
    from atom.data import pack_csv

    root = pack_csv(args.csv, args.out, name=args.name, target=args.target)
    print(f"wrote ADP: {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atom", description="ATOM — AuTO ai Machine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="profile a dataset package (zip or folder)")
    p_inspect.add_argument("package")
    p_inspect.add_argument("--json", action="store_true", help="emit the full fingerprint as JSON")
    p_inspect.add_argument("--sample-rows", type=int, default=50_000)
    p_inspect.add_argument("--columns", type=int, default=12, help="max columns to display")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_run = sub.add_parser("run", help="AutoAI run: package -> trained model + provenance")
    p_run.add_argument("package")
    p_run.add_argument("--target", help="target column (overrides/completes manifest roles)")
    p_run.add_argument("--time-budget", type=float, default=120.0, metavar="SECONDS")
    p_run.add_argument("--max-trials", type=int)
    p_run.add_argument("--min-trials", type=int)
    p_run.add_argument("--max-rows", type=int, default=100_000)
    p_run.add_argument("--out", default="runs")
    p_run.add_argument("--kb", help="meta-KB root (default: $ATOM_HOME/metakb or ~/.atom/metakb)")
    p_run.add_argument("--task", choices=["classification", "regression", "clustering",
                                          "anomaly-detection"],
                       help="override the inferred task family")
    p_run.add_argument("--include-experimental", action="store_true",
                       help="let experimental (unpromoted) modules join the search")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--yes", "-y", action="store_true", help="skip the confirm gate")
    p_run.set_defaults(func=_cmd_run)

    p_mod = sub.add_parser("modules", help="list registered modules / run the smoke gate")
    p_mod.add_argument("action", choices=["list", "verify"])
    p_mod.set_defaults(func=_cmd_modules)

    p_pack = sub.add_parser("pack", help="convert a loose CSV into an ATOM Dataset Package")
    p_pack.add_argument("csv")
    p_pack.add_argument("--out", "-o", default=".", help="output directory")
    p_pack.add_argument("--name", help="package name (default: CSV stem)")
    p_pack.add_argument("--target", help="target/label column name")
    p_pack.set_defaults(func=_cmd_pack)

    p_fetch = sub.add_parser("fetch", help="fetch kaggle:<slug> and convert to an ADP")
    p_fetch.add_argument("source")
    p_fetch.add_argument("--target", help="target column for the packed ADP")
    p_fetch.add_argument("--file", help="specific CSV inside the dataset (default: largest)")
    p_fetch.add_argument("--name")
    p_fetch.add_argument("--out", "-o", default=".")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_pimg = sub.add_parser("pack-images",
                            help="convert an image folder (class-per-subfolder) into an ADP")
    p_pimg.add_argument("folder")
    p_pimg.add_argument("--out", "-o", default=".")
    p_pimg.add_argument("--name")
    p_pimg.set_defaults(func=_cmd_pack_images)

    args = parser.parse_args(argv)
    return args.func(args)


def _cmd_pack_images(args: argparse.Namespace) -> int:
    from atom.data import pack_images

    root = pack_images(args.folder, args.out, name=args.name)
    print(f"wrote image ADP: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
