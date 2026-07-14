# ADR-0003: Standard Dataset Structure — ATOM Dataset Package (ADP)

- Status: **accepted** (2026-07-14)
- Date: 2026-07-10
- Reference sample: `cic-ids-2017-ml-package.zip` (manifest `dms-ml-package-v1`)

## Context

ATOM needs one canonical on-disk dataset structure that ingest can consume
without per-dataset glue code. A sample package (CIC-IDS-2017, produced by
DMS) demonstrates a strong baseline: datasheet README, machine-readable
manifest, immutable `raw/`, ready-to-train `processed/` parquet, split
recorded as data, and a deterministic rebuild script. ATOM must read the
package **both as a `.zip` archive and as an extracted folder**, with
identical semantics.

## Decision

Adopt the DMS package layout as the basis of **ATOM Dataset Package v1**
(`manifest_version: "atom-dataset-v1"`, backward-readable from
`dms-ml-package-v1`) with the extensions below.

### Canonical layout (identical in zip and folder)

```
<dataset-slug>/
├── README.md            # human datasheet
├── manifest.json        # the machine contract — single source of truth
├── raw/                 # exactly as uploaded, immutable, checksummed
├── processed/           # train/val/test, typed, ready for ML
├── splits/split_vN.json # split is data, not code; versioned
└── scripts/build.py     # deterministic raw → processed rebuild
```

### Extensions over dms-ml-package-v1

1. **Column roles.** The sample declares only `id_column`. ADP adds a
   `roles` block so task inference never guesses:

   ```json
   "roles": {
     "id": "sample_id",
     "target": ["Label"],
     "ignore": ["Flow ID", "Source IP", "Destination IP", "Timestamp"],
     "group": "Flow ID",
     "time": "Timestamp"
   }
   ```

   `ignore` marks identifier/leak-prone columns; `group`/`time` enable
   leak-safe (group- and time-aware) splitting and evaluation.

2. **Typed processed data.** The sample stores every parquet value as
   `string` with advisory `inferred_type` (mostly `unknown`, sometimes
   wrong). ADP: `raw/` stays byte-exact, but `processed/` MUST be typed
   parquet; the manifest schema records the canonical dtype per column,
   plus declared missing-value sentinels (e.g. `Infinity`, `NaN` strings).

3. **Enumerated labels.** The sample has a `Label` column yet
   `labels: []` and `label_completeness: 1` — contradictory. ADP requires,
   for every target column: class names, per-class counts, and per-split
   class distribution. Empty `labels` with a declared target is invalid.

4. **Split methods beyond row hash.** Seeded row-hash (the sample's
   method) stays the default, but ADP registers `hash | provided |
   group_hash | time_holdout`. Temporally correlated data (network flows,
   sensor streams) must be able to split by group/time or the locked test
   set silently leaks.

5. **Fingerprint block (optional).** ATOM's profiler may write its
   computed fingerprint back into the package (`fingerprint.json`) so
   re-ingest and meta-KB lookup skip re-profiling; keyed by the checksum
   of `processed/`.

6. **Multi-modal file groups.** `schema.mode` generalizes to per-group
   blocks so one package can carry `tabular` + `image` + `text` parts
   (e.g. flows + pcap snippets), each with its own schema and roles.

### Zip and folder are equally first-class

- The loader resolves a *package locator* that is either a directory or a
  `.zip`; all internal paths are relative to the package root. Semantics
  are identical; API surface exposes no difference.
- **Zip members in `processed/` SHOULD be archived uncompressed
  (`stored`)**: parquet is already snappy-compressed, deflate re-compression
  buys little, and stored members allow random access / memory-mapping of
  parquet row groups directly inside the archive. `raw/` may be deflated.
- Integrity: verify manifest checksums lazily — on first read of each
  member, not by hashing 1.6 GB at open time.

### ADP is mandatory inside ATOM; loose inputs are converted, not rejected

The standard structure is required for everything ATOM processes and for
sharing with lab members. For compatibility, ingest ACCEPTS non-standard
inputs (bare CSV/TSV, parquet, image folders) but immediately converts them
through the **packager**: build `manifest.json` + `roles` (interactively or
inferred + confirm gate), compute checksums, apply a default split, and emit
a valid ADP. The run then proceeds from the ADP — there is no second data
path through the engine.

### Relationship to DMS and package identity

DMS is an independent dataset-sharing system; **there is no direct
integration**. Alignment is by format: ADP stays a strict superset of
`dms-ml-package-v1` so DMS exports remain directly ingestible, and the same
standard serves both systems. Package identity inside ATOM is the
**content hash** of `manifest.json` (which itself pins all file checksums);
`dms_dataset_id`, when present, is carried as pass-through metadata.
If DMS later exposes an API, a fetch-from-DMS packager source can be added
without changing this contract.

## Consequences

- Ingest & Profiler has a single contract to implement; anything not an
  ADP goes through a thin "packager" import step first.
- The `roles` block is what makes automatic task inference sound; it is
  required for supervised packages.
- Keeping `raw/` + split + deterministic build script preserves full
  reproducibility (`processed/` can always be regenerated and verified
  against checksums).
- Typed `processed/` moves parsing cost to package build time, once,
  instead of every training run.
