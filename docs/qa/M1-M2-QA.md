# QA Report — M1 (Skeleton & Data Plane) + M2 (Tabular MVP)

Date: 2026-07-14 · Scope: commits `0c92153` (M1), `4ebd70f` (M2) + QA fixes.
Method: design-conformance review against ADR-0001..0007 + method-taxonomy
v2.1 + roadmap exit criteria, plus dynamic verification on the reference
dataset (CIC-IDS-2017, 623 MB zip) and the synthetic test fixtures.

## 1. Verification evidence

- Test suite: **18/18 green** (contract validation incl. ADR-0005 bars,
  ADP round-trip folder↔zip, tamper detection, task-inference routing,
  end-to-end run with provenance assertions, budget bound respected).
- Real-data run (`atom run cic-ids-2017.zip --target Label --time-budget
  180 --max-rows 60000`): 31 trials, ensemble final, locked-test
  **f1_macro 0.897 / accuracy 0.997** over 13 classes; elapsed 191 s vs
  180 s stated (§3.F1).
- `atom inspect` on the zip: 4 s, no extraction; flags the dataset's known
  defects (no target declared, duplicated column, Infinity values).

## 2. ADR conformance matrix

| Design item | Status | Notes |
|---|---|---|
| ADR-0001 contract (4 methods) | ✅ | `Module` ABC; `hints()` default-implemented; fidelity ladders drive SHA rungs |
| ADR-0001 declaration validation | ✅ | foundation⇒adaptation, anomaly⇒setting, enforced at `register()` (tested) |
| ADR-0001 `report()` progress channel | ⏳ locked default | not needed until iterative trainers (M5+) |
| ADR-0001 serializable run boundary | ⚠️ partial | in-process artifacts hold live sklearn objects; MUST be addressed with M5 venv isolation (subprocess boundary) |
| ADR-0002 decorator + entry-point discovery | ✅ | `discover()` implemented; not yet exercised by a real external package (M5 promotion test will) |
| ADR-0002 module lifecycle field | ⏳ M5 | `experimental/stable/deprecated` not yet on Declaration; all built-ins are de-facto stable |
| ADR-0003 zip+folder identical semantics | ✅ | tested byte-identical fingerprints |
| ADR-0003 lazy per-member checksums | ✅ | tamper test passes |
| ADR-0003 loose CSV → typed ADP with roles+labels | ✅ | `atom pack`; split rule byte-compatible with DMS sample |
| ADR-0003 content-hash package id (+dms passthrough) | ✅ | `sha256(manifest bytes)` |
| ADR-0003 group_hash / time_holdout splits | ⏳ M5 | packager emits `hash` only; TaskSpec already records the policy |
| ADR-0003 zip write with stored `processed/` | ⏳ | packager writes folders only; rule applies when `pack --zip` lands |
| ADR-0003 fingerprint.json write-back | ⏳ M4 | with meta-KB |
| ADR-0004 AMP / ONNX | ⏳ M3 | provenance dir already mirrors AMP layout (`provenance/`, `metrics.json`, `native/`) |
| ADR-0005 nine families + v1 foundation bars | ✅ | enum matches taxonomy; `full-finetune`/`distill` rejected (tested) |
| ADR-0005 per-user, no daemon | ✅ | library + CLI only |
| ADR-0005 privacy-safe fingerprint | ✅ | summary statistics only; no raw values in any persisted artifact |
| ADR-0006 wall-clock primary, optional trials | ✅ | first-bound-wins tested; min-trials honored |
| ADR-0006 estimated end time | ✅ | shown per rung (`~Ns left`) |
| ADR-0006 reserved tail → always a usable artifact | ✅ | finalize keeps ≥1 candidate even past budget |
| ADR-0006 budget honesty | ⚠️ ~6% overrun | see F1; residual = mandatory finalize + locked-test load |
| ADR-0007 promotion pipeline | ⏳ M5 | nothing in code contradicts it |
| Taxonomy anomaly routing rule | ✅ | labels→classification (incl. rare-class); unlabeled→outlier; drift note on time role (tested) |
| Taxonomy diversity constraint | ⏳ | batch round-robins over methods (weak form); `(category, paradigm)` constraint not yet enforced |
| Roadmap M1 exit | ✅ | inspect on zip; CSV→valid ADP round-trip |
| Roadmap M2 exit | ✅ | one command, honest evaluation, stated budget (with F1 residual) |

## 3. Findings (found during QA → fixed)

- **F1 — budget overrun 38%→6%.** Search started trials it couldn't afford
  and finalize refit 5 candidates unbounded. Fixed three ways: cost-model
  admission control (per method×fidelity mean, extrapolated across
  fidelities), budget-bounded finalize (≥1 candidate always), and an
  invest-one rule (first full-fidelity trial always admitted so finalize
  reuses its cached fit). Residual overrun is the mandatory ≥1 finalize
  candidate + test-split load; documented, acceptable for v1.
- **F2 — degenerate ensemble.** `best_trials` required full fidelity, so
  usually 1 candidate survived → no ensemble. Fixed: lower-fidelity
  survivors fill top-K and are re-validated at full fidelity during
  finalize. Ensemble now wins when it should (observed on CIC-IDS).
- **F3 — unresolved class count on `--target` override.** n_classes/
  imbalance stayed unknown → wrong default metric (roc_auc on 13-class
  data). Fixed: resolved from loaded training labels before search;
  roc_auc now only when classes are known to be 2.
- **F4 — invalid UTF-8 in DMS parquet** (cp1252 0x96 in `Label`:
  "Web Attack – Brute Force"). All Arrow column reads now go through a
  tolerant decoder (replace, never fatal). Consider adding an ADP
  validation rule: string columns must be valid UTF-8 at pack time.
- **F5 — lint debris** (unused variable, ambiguous name). Fixed; ruff clean.

## 4. Known limitations (accepted, tracked)

1. **Numeric features only** (M2): string columns dropped with recorded
   reason; a mostly-numeric column polluted by junk strings profiles as
   string and is dropped whole. Categorical encoding lands M5.
2. **No stratified subsampling** at low fidelity: rare classes can vanish
   from a 10% subsample of imbalanced data, making low-rung scores noisy
   (visible as rung-to-rung variance on CIC-IDS). Recommend stratified
   fidelity sampling for classification in M-next.
3. **SMOTE/resampling module absent** (imbalanced-learn optional, not yet
   wrapped); imbalance is currently addressed only by metric choice.
4. **Search strategy hard-coded** (random+SHA in orchestrator); Search
   registry exists but has no strategy modules yet — swap point is
   localized in `search.py` by design.
5. Budget clock includes data loading but admission control governs only
   trials; very slow package IO erodes search time silently.

## 5. Verdict

M1 and M2 exit criteria hold; no unfixed correctness defects known. The
two ⚠️ items (serializable run boundary, budget residual) are the ones to
watch — both have a concrete home (M5 isolation; orchestrator v2).
