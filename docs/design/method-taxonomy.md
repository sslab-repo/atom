# Methods Registry — Algorithm Taxonomy (v2.1)

> Status: **family structure accepted** via ADR-0005 (2026-07-14); the
> category tables remain a living document that grows with new algorithms.
> v2 reduced 13 overlapping families to 9 canonical ones.
> v2.1 adds discipline-aware terminology: every family now has an exact
> definition, a synonym map across disciplines, and — for detection — an
> explicit *setting* axis, because "anomaly detection" alone is ambiguous
> (outlier vs. novelty vs. OOD vs. drift vs. supervised rare-class).

## Structure rules

| Level | Example | Who may change it | Velocity |
|---|---|---|---|
| **Task family** | `classification` | ADR required (core enum) | ~never |
| **Category** | `ensemble-boosting` | open convention — a tag on the declaration | monthly |
| **Algorithm (module)** | `xgboost@2.1` | anyone — drop-in via contract | daily |

Module identity `(family, name, version)`; versions coexist; lifecycle
`experimental → stable → deprecated`. The family axis answers exactly one
question — **what is estimated from data** — never data type (that is the
modality axis) and never purpose (tags) or application domain (metadata).

## The nine families and what they are called elsewhere

Family names are ATOM's canonical vocabulary. Ingest/task-inference and all
documentation use only these; the synonym map exists so modules wrapping
literature from any discipline declare the right family.

| ATOM family | Exact definition (what is estimated) | Known elsewhere as |
|---|---|---|
| `classification` | f: X → discrete label, from labeled data | pattern recognition (engineering), discriminant analysis (statistics), diagnosis (medicine) |
| `regression` | f: X → continuous value, from labeled data | estimation (signal processing), function approximation (ML), forecasting (econometrics; = `temporal` tag), system identification (control), curve fitting |
| `clustering` | a partition/grouping of unlabeled data | unsupervised classification, segmentation (marketing, vision), numerical taxonomy (biology), vector quantization (signal processing) |
| `dimension-reduction` | a lower-dimensional representation or feature subset | feature extraction (engineering), representation learning (deep learning), factor analysis (psychometrics), ordination (ecology), blind source separation (signal processing), manifold learning |
| `anomaly-detection` | a score/decision of non-conformity to normal data | outlier detection (statistics), novelty detection (ML), OOD detection (deep learning), fault detection / condition monitoring (engineering), abnormality detection (medical imaging); intrusion/fraud detection are **application domains**, not tasks |
| `generative` | the data distribution p(x) (or p(x\|c)), sampleable | density estimation (statistics), generative modeling (ML), data synthesis (privacy), simulation modeling (OR) |
| `structured-prediction` | f: X → structured object (boxes, masks, sequences, images) | structured output learning (ML), recognition (vision), sequence transduction (speech/NLP), inverse problems (imaging — restoration) |
| `association-mining` | frequent patterns / co-occurrence rules | market basket analysis (retail), frequent pattern mining (data mining) |
| `preference-learning` | an ordering / relevance function | learning to rank (IR), recommender systems, collaborative filtering, choice modeling (econometrics) |

Cross-check against established tools: MATLAB (Classification, Regression,
Cluster Analysis, Dimensionality Reduction & Feature Extraction, Anomaly
Detection, Probability Distributions), SPSS (Regression, Classify,
Dimension Reduction, Forecasting, Survival), R task views (MachineLearning,
Cluster, TimeSeries, Multivariate, Distributions). Families 1–5 are the
undisputed data-mining textbook core (Han–Kamber–Pei; Tan–Steinbach–Kumar).

### v1 → v2 disposition (for the record)

`estimate`→`regression` · `forecast`→`regression`+`temporal` ·
`reduce`→`dimension-reduction` · `detect`→`anomaly-detection` ·
`denoise`→ split: classical filters → Preprocessing, learned restoration →
`structured-prediction/restoration` · `amplify`→ split: SMOTE/classical
augmentation → Preprocessing, generative upsampling → `generative`+`augment` ·
`synthesize`→`generative` · `vision`/`language`→ modality axis ·
`rank`/`recommend`→`preference-learning` · `associate`→`association-mining`.

---

## Categories per family

### 1. `classification` — supervised, discrete target

| Category | Representative algorithms |
|---|---|
| `linear` | logistic regression, LDA/QDA, SGD classifier |
| `kernel-margin` | SVM (linear/RBF/poly) |
| `probabilistic` | naive Bayes, Bayesian networks |
| `instance-based` | kNN, nearest centroid |
| `tree` | CART, C4.5/C5.0 |
| `ensemble-bagging` | random forest, extra trees |
| `ensemble-boosting` | XGBoost, LightGBM, CatBoost, AdaBoost |
| `neural` | MLP; CNN/ViT (`× image`); transformers (`× text`) |
| `foundation` | TabPFN; VLM zero-shot (`× image`) |
| `rule-based` | RIPPER, decision lists, scoring systems |
| `semi-supervised` | self-training, label propagation, FixMatch-style |

*Terminology note:* **rare-class / imbalanced classification** (labeled
anomalies: labeled intrusion, fraud, defect data) belongs HERE, not in
`anomaly-detection` — see the routing rule below. Imbalance handling =
Preprocessing resampling and/or cost-sensitive microcontrols declared in
`space()`.

### 2. `regression` — supervised, continuous target

| Category | Representative algorithms |
|---|---|
| `linear` | OLS, GLM, ridge, lasso, elastic net |
| `kernel-gp` | SVR, Gaussian process regression |
| `tree-ensemble` | RF/GBM/XGBoost/LightGBM regressors |
| `neural` | MLP, tabular transformers |
| `probabilistic` | quantile regression, NGBoost, conformal intervals |
| `robust` | Huber, RANSAC, Theil–Sen |
| `symbolic` | symbolic regression (PySR-style) |
| `temporal-statistical` | ARIMA/SARIMA, ETS, Theta, Prophet, state-space/Kalman |
| `temporal-neural` | DeepAR, N-BEATS/N-HiTS, TFT, PatchTST; foundation: Chronos, TimesFM |
| `survival` | Cox PH, survival forests |

*Terminology note:* `temporal` covers both **forecasting** (extrapolate a
series into the future) and **time-indexed regression with exogenous
variables**; the fingerprint's `time` role triggers temporal-capable search
and time-aware splits (ADR-0003).

### 3. `clustering` — unsupervised grouping

| Category | Representative algorithms |
|---|---|
| `partitional` | k-means, k-medoids/PAM, mini-batch k-means |
| `hierarchical` | agglomerative (Ward/average), BIRCH |
| `density-based` | DBSCAN, HDBSCAN, OPTICS, mean-shift |
| `model-based` | GMM, Bayesian GMM, Dirichlet process mixtures |
| `graph-spectral` | spectral clustering, Louvain, Leiden |
| `self-organizing` | SOM/Kohonen maps |
| `fuzzy` | fuzzy c-means |
| `deep` | DEC/IDEC, contrastive clustering |

Category names follow the standard survey taxonomy (Jain); `model-based`
is the R mclust term for mixture-model clustering.

### 4. `dimension-reduction` — feature extraction, selection, representation

| Category | Representative algorithms |
|---|---|
| `linear-projection` | PCA, factor analysis, correspondence analysis, NMF, random projection |
| `source-separation` | ICA, blind source separation (signal-processing lineage — kept distinct from PCA on purpose) |
| `supervised-projection` | LDA-as-reducer, PLS |
| `manifold` | MDS, Isomap, LLE, t-SNE, UMAP |
| `autoencoder` | AE, VAE-as-encoder, denoising AE |
| `self-supervised` | SimCLR, BYOL, DINO-style |
| `pretrained-embedding` | CLIP/DINOv2 (`× image`), sentence-transformers (`× text`) |
| `feature-selection` | filter (mutual info, χ², mRMR), wrapper (RFE), embedded (L1, tree importance), Boruta |
| `feature-construction` | interaction/polynomial features, autofeat, deep feature synthesis |

*Terminology note:* statistics says *feature extraction* (new axes) vs.
*feature selection* (subset of original axes) — the category names keep
that distinction because their provenance differs (selected features stay
interpretable; extracted ones don't).

### 5. `anomaly-detection` — non-conformity scoring

The most terminology-fractured family. ATOM disambiguates with a required
**`setting` tag** (following Chandola–Banerjee–Kumar's survey axes:
supervision mode × anomaly type):

| `setting` | Training data | Question answered | Discipline name |
|---|---|---|---|
| `outlier` | unlabeled, contaminated | which points in THIS dataset deviate? | outlier detection (statistics); data cleaning |
| `novelty` | normal-only (one-class) | does a NEW point conform to normal? | novelty detection (ML); fault detection (engineering) |
| `ood` | labeled for another task | is this input outside the training distribution? | out-of-distribution detection (deep learning) |
| `drift` | a stream / time series | did the data-generating process change? | change-point / concept-drift detection |

**Routing rule:** if anomalies are *labeled* and examples of both classes
exist → that is **rare-class `classification`**, not this family. If only
normal data is trustworthy → `novelty`. If nothing is labeled → `outlier`.
Application domains (intrusion, fraud, defect, abnormality in medical
imaging) are metadata, never the task. CIC-IDS-2017 with its labels is
rare-class classification; the same data unlabeled is `outlier`/`drift`.

Every anomaly module also tags the **anomaly type** it handles:
`point | contextual | collective` (Chandola's typology — a contextual
anomaly is normal in value but not in context, e.g. 30 °C in winter;
a collective anomaly is a sequence anomalous only as a whole).

| Category | Representative algorithms | Typical settings |
|---|---|---|
| `statistical-test` | Grubbs, generalized ESD, extreme value theory | outlier |
| `proximity` | kNN-distance, LOF (distance- and density-based) | outlier, novelty |
| `clustering-based` | CBLOF, distance-to-centroid scoring | outlier |
| `probabilistic-density` | GMM likelihood, KDE score, HBOS, ECOD/COPOD | outlier, novelty |
| `one-class` | one-class SVM, SVDD | novelty |
| `isolation-subspace` | isolation forest, extended IF, subspace outliers (high-dim) | outlier |
| `reconstruction` | PCA residual, autoencoder error | outlier, novelty |
| `deep` | Deep SVDD, DAGMM; OOD scores: max-softmax, Mahalanobis, energy | novelty, ood |
| `temporal` | matrix profile, spectral residual, change-point (PELT/BOCPD), drift detectors (ADWIN, DDM) | drift |

### 6. `generative` — density estimation & generative modeling

Purpose tags: `synthesis`, `augment` (minority upsampling), `simulation`.

| Category | Representative algorithms |
|---|---|
| `parametric-density` | KDE, GMM-as-density |
| `copula` | Gaussian/vine copulas |
| `latent-variable` | VAE, TVAE |
| `adversarial` | CTGAN (`× tabular`), StyleGAN (`× image`), TimeGAN (`× timeseries`) |
| `diffusion` | latent diffusion (`× image`), TabDDPM (`× tabular`) |
| `autoregressive` | LLMs (`× text`), PixelCNN-style |
| `flow` | normalizing flows |
| `privacy-preserving` | DP-GAN, DP synthetic tabular |

### 7. `structured-prediction` — supervised, structured output

| Category | Representative algorithms |
|---|---|
| `object-detection` | YOLO family, Faster R-CNN, DETR/DINO (`× image`) |
| `segmentation` | U-Net, DeepLab, Mask R-CNN, SAM (`× image`) |
| `keypoint-pose` | HRNet, RTMPose (`× image`) |
| `sequence-labeling` | HMM, CRF, transformer NER (`× text`) |
| `seq2seq` | translation, summarization, ASR/TTS, OCR |
| `restoration` | DnCNN, Noise2Noise, SwinIR, super-resolution, diffusion restorers — the imaging-science **inverse problems** family (learned denoise/deblur/inpaint) |

### 8. `association-mining` — frequent pattern & rule mining

| Category | Representative algorithms |
|---|---|
| `association-rules` | Apriori, FP-Growth, Eclat |
| `sequential-patterns` | PrefixSpan, SPADE |

### 9. `preference-learning` — ranking & recommendation

| Category | Representative algorithms |
|---|---|
| `learning-to-rank` | LambdaMART, RankNet |
| `collaborative-filtering` | matrix factorization (ALS), item-kNN |
| `neural-recsys` | two-tower, neural CF, sequential recommenders |

### Reserved future family names

`causal`, `control` (RL), `optimize`, `graph`.

---

## Moved to the Preprocessing registry (not learning tasks)

Signal/image filtering (wavelet, Savitzky–Golay, Kalman smoothing, NLM,
BM3D) · resampling/class balance (SMOTE family, ADASYN) · classical
augmentation (flips, RandAugment, mixup/CutMix, jitter, text EDA) · data
repair (label-noise cleaning, outlier *repair*, dedup) · impute, scale,
encode, tokenize.

Rule of thumb: **output is a dataset and fit is trivial → preprocessing;
estimates a model of the data (distribution, boundary, function) → Method.**
Note the pairing: `anomaly-detection` *finds* suspect points (a model);
outlier *repair/removal* is preprocessing that may consume its scores.

## Deep learning & foundation models — the paradigm axis

Deep learning and generative AI are deliberately **not** families: they are
*how*, not *what*. A CNN and logistic regression solve the same task; an LLM
and a copula both estimate a samplable distribution. Architectures churn
daily; the task list doesn't. They enter the taxonomy as:

- **paradigm = `deep`** — architectures trained from scratch (or lightly
  pretrained) on the run's data: CNNs, transformers, GANs, diffusion, deep
  clustering/AD. Present as `neural`/`deep` categories in every family.
- **paradigm = `foundation`** — large pretrained models (LLMs, LVMs:
  CLIP, SAM, DINOv2; TabPFN, Chronos) that ATOM *adapts* rather than
  trains. One backbone typically serves several families through thin
  wrapper modules (CLIP = zero-shot classifier + embedding extractor +
  retrieval), each declaring its family honestly and sharing a cached
  backbone (asset management: open question).

Foundation modules additionally declare an **`adaptation`** tag:

| `adaptation` | Meaning | Cost profile |
|---|---|---|
| `zero-shot` | frozen weights, task via prompt/instruction | cheapest |
| `few-shot` | in-context examples, no weight update | cheap; k = a microcontrol |
| `prompt-tuning` | learned soft prompts / prompt search | moderate |
| `peft` | LoRA / adapters — small trainable deltas | moderate |
| `full-finetune` | all weights updated | expensive |
| `distill` | train a small deployable student from the big model | expensive, but yields an ONNX-friendly artifact |

This maps directly onto the contract: adaptation depth is a natural
**fidelity ladder** for `hints()` (zero-shot → few-shot → peft →
full-finetune), so the multi-fidelity orchestrator can triage cheaply with
zero-shot before spending GPU budget on fine-tuning. Prompts, adapter rank,
and example count are ordinary `space()` microcontrols. For deployability,
large generative backbones are typically `deployable: false` under ADR-0004
rule 6 — `distill` is the sanctioned path to an exportable model.

## Cross-cutting declaration tags

- **modalities** — tabular, image, text, timeseries, audio, video, mixed
- **paradigm** — `classical | deep | foundation`
- **adaptation** (foundation only, required) — `zero-shot | few-shot | prompt-tuning | peft | full-finetune | distill`
- **supervision** — `supervised | unsupervised | self-supervised | semi-supervised`
- **setting** (anomaly-detection only, required) — `outlier | novelty | ood | drift`
- **anomaly type** (anomaly-detection) — `point | contextual | collective`
- **structure/purpose tags** — `temporal`, `augment`, `synthesis`, `simulation`
- **exportable** — ONNX exportability (ADR-0004 rules 6–7)

Diversity constraint operates on `(category, paradigm)`.
