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
    print(f"samples    : " + "  ".join(f"{s}={c:,}" for s, c in fp.counts.items()))
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

    p_pack = sub.add_parser("pack", help="convert a loose CSV into an ATOM Dataset Package")
    p_pack.add_argument("csv")
    p_pack.add_argument("--out", "-o", default=".", help="output directory")
    p_pack.add_argument("--name", help="package name (default: CSV stem)")
    p_pack.add_argument("--target", help="target/label column name")
    p_pack.set_defaults(func=_cmd_pack)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
