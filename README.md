# Churn Prediction with a Revenue Number Attached

**Predict who leaves. Rank who to contact. Put a currency figure next to the alert.**

This repository is a portfolio-grade, tutorial-style applied ML project: three fully executed Jupyter notebooks that download real public datasets, run classical churn classifiers and Google’s **TabFM** tabular foundation model, and convert predictions into **revenue-at-risk** and **outreach prioritization** views—not just accuracy scores.

| Audience | Start here |
|----------|------------|
| **Portfolio / technical reviewers** | [Architecture & design decisions](#1-for-portfolio-evaluators) · [Real evidence](#real-results-evidence-from-executed-runs) · [Limitations](#honest-limitations) |
| **Hands-on operators** | [Installation](#2-for-hands-on-users) · [Runbook](#runbook) · [Troubleshooting](#troubleshooting) · [Extending](#extending-the-project) |
| **Tutorial learners** | [Concepts](#3-for-tutorial-learners) · [End-to-end flow](#implementation-flow-step-by-step) · [How to read each notebook](#how-to-read-the-notebooks) |

---

## Table of contents

1. [Project overview](#project-overview)
2. [1. For portfolio evaluators](#1-for-portfolio-evaluators)
3. [Real results (evidence from executed runs)](#real-results-evidence-from-executed-runs)
4. [2. For hands-on users](#2-for-hands-on-users)
5. [3. For tutorial learners](#3-for-tutorial-learners)
6. [Repository map](#repository-map)
7. [Licenses & data provenance](#licenses--data-provenance)
8. [Honest limitations](#honest-limitations)

---

## Project overview

### Problem

Customer churn is usually framed as binary classification. Business stakeholders care about three different questions:

1. **Discrimination** — Can the model rank true leavers above stayers under class imbalance?
2. **Exposure** — How much revenue (or historical value) is attached to customers we would flag today?
3. **Action** — If we can only call the top *K*% of the list, how much true-churn value do we cover?

Most tutorial notebooks stop at (1). This project deliberately finishes (2) and (3).

### Deliverables

Three notebooks, same methodological skeleton, different data realities:

| # | Notebook | Domain | Label | Revenue anchor |
|---|----------|--------|-------|----------------|
| 1 | [`notebooks/01_iranian_churn.ipynb`](notebooks/01_iranian_churn.ipynb) | Telecom (UCI 563) | Given (`Churn`) | `Customer Value` |
| 2 | [`notebooks/02_telco_churn.ipynb`](notebooks/02_telco_churn.ipynb) | Telecom (IBM sample) | Given (`Yes`/`No`) | `MonthlyCharges` (+ `TotalCharges` context) |
| 3 | [`notebooks/03_online_retail_ii_churn.ipynb`](notebooks/03_online_retail_ii_churn.ipynb) | Online retail (UCI 502) | **Engineered** (RFM + 90-day inactivity) | Pre-cutoff `Monetary` |

Each notebook, in order:

1. Business framing  
2. Reproducible setup (versions, seed, kernel intent)  
3. Live data acquisition  
4. Thorough EDA with **dataset-specific** interpretation  
5. Feature / label engineering  
6. Stratified train/test split (shared by both modeling parts)  
7. **Part 1** — LazyPredict → top 3 → proper hyperparameter search + full reports  
8. **Part 2** — Google TabFM zero-shot + same report suite  
9. Side-by-side metrics, revenue-at-risk, lift/prioritization curve, manager summary  

Jupytext percent-format `.py` sources sit beside each `.ipynb` for script debugging.

### What this is *not*

| Avoided on purpose | Why |
|--------------------|-----|
| Chat LLMs / Ollama | No conversational or retrieval stage; tabular classification only |
| Unsloth / LoRA / QLoRA | Nothing fine-tunes foundation weights; TabFM `.fit()` stores in-context rows |
| Pre-committed multi-GB CSVs | Data is fetched inside notebooks from official sources |
| Accuracy-only leaderboards | Imbalanced churn; primary ranking uses F1 / PR-AUC |

---

## 1. For portfolio evaluators

### Design thesis

Treat churn as a **decision-support** problem under imbalance, not a pure accuracy contest.

- **Screen widely, then invest compute carefully.** LazyPredict is a shortlist generator with defaults—not the final model.
- **Tune on the metric that matches cost structure.** Hyperparameter search uses `average_precision` (PR-AUC), not accuracy.
- **Compare apples-to-apples.** Classical models and TabFM share the same stratified split indices.
- **Dual preprocessing, one split.** TabFM accepts mixed-type frames natively; sklearn models use train-fitted scaling / one-hot. Indices stay aligned.
- **Revenue is an interpretation layer, not a second loss.** Anchors are summed over predicted-positive customers with explicit non-causal framing.
- **Fail loudly on bad engineered labels.** Online Retail stops if engineered churn is near 0% or 100%.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  Shared uv env (Python 3.13.13) + kernel churn-revenue-project  │
└─────────────────────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Notebook 1  Notebook 2  Notebook 3
 UCI 563     IBM Telco   UCI 502 (RFM label)
    │           │           │
    └───────────┼───────────┘
                ▼
   Live download → EDA → features/labels
                │
       Stratified split (seed=42)
       ┌────────┴────────┐
       ▼                 ▼
  Part 1 (classical)  Part 2 (TabFM)
  LazyPredict shortlist  load HF weights (cuda)
  RandomizedSearchCV     fit = store context
  CM / ROC / PR / F1     same metrics
       └────────┬────────┘
                ▼
   Best Part-1 vs TabFM table
   Revenue-at-risk on flags
   Lift curve (score → contact order)
   Manager summary
```

### Critical architecture decisions

| Decision | Choice | Rationale | Tradeoff |
|----------|--------|-----------|----------|
| Env manager | `uv` only, project-local `.venv` | Reproducible pins via `uv.lock` | Requires `uv` on PATH |
| Python | **3.13.13** | Project standard; TabFM needs ≥3.11 | Some wheels may lag; fallback path is 3.12.10 if needed |
| Model screening | LazyPredict `LazyClassifier` | Cheap multi-model baseline | Defaults under-tuned; must re-implement winners |
| Top-3 ranking | **F1 Score** column from LazyPredict | Better than accuracy under imbalance | Not identical to PR-AUC; we re-tune with AP |
| Hyperparameter search | `RandomizedSearchCV`, 5-fold stratified, score=`average_precision` | PR-AUC aligns with rare-positive ranking | Stochastic grids; seed fixed |
| Foundation model | **TabFM** (PyTorch, GitHub 1.0.1) | Zero-shot tabular ICL; sklearn-compatible API | Non-commercial weights; large download; VRAM |
| TabFM device | `device="cuda"` when available | CPU load of ~6.5GB weights thrashed under memory pressure in testing | GPU strongly recommended |
| TabFM install | Git source, not PyPI 1.0.0 alone | PyPI 1.0.0 looks for `pytorch_model.bin`; HF ships `model.safetensors` | Pins to git commit via uv |
| Retail label | Cutoff 2011-06-01 + 90d inactivity | Defensible, documented assumption | Not CRM ground truth |
| TabFM context cap | Subsample to 3000 if train larger (NB3) | Memory scales with context rows | Disclosed; Part 1 still uses full train |
| Seeds | `RANDOM_STATE = 42` | Reproducible split / search | Single split ≠ nested CV |

### Reproducibility checklist

| Control | Implementation |
|---------|----------------|
| Python pin | `.python-version` → `3.13.13` |
| Lockfile | `uv.lock` |
| Kernel | `churn-revenue-project` → project `.venv` |
| Seed | `42` for NumPy, split, LazyPredict, search |
| Stratify | Always on churn label |
| Preprocess fit | Train only (`ColumnTransformer.fit` on train) |
| Data source | Code-path downloads (URLs / UCI IDs in notebooks) |
| Outputs | Executed `.ipynb` cells with real plots/metrics |
| Version print | Setup cells print Python, sklearn, lazypredict, tabfm, torch, CUDA |

### What technical reviewers should look for

1. **Imbalance discipline** — PR-AUC / churn F1 emphasized over accuracy (especially Iranian 15.71% churn).  
2. **Leakage control** — Retail features use only pre-cutoff activity; post window is label-only.  
3. **Honest dual path** — TabFM vs one-hot sklearn matrices documented, same indices.  
4. **Evidence over narrative** — Tables below are from executed notebooks on this stack, not blog copy.  
5. **License hygiene** — MIT for *this* code; TabFM *weights* separately non-commercial; datasets retain upstream terms.

### Comparison protocol (fairness)

Across both model families in a notebook:

- Same `X_train` / `X_test` row indices  
- Same binary label encoding  
- Same positive class = churn  
- Same report suite: classification report, confusion matrix, ROC, PR, F1/PR-AUC/ROC-AUC  
- Revenue and lift computed on **test** predictions only  

Cross-notebook metrics are **not** comparable as a single leaderboard (different domains, base rates, and label definitions).

---

## Real results (evidence from executed runs)

> Captured from fully executed notebooks on this machine: **Python 3.13.13**, **tabfm 1.0.1**, **torch 2.13.0+cu130**, CUDA available, seed **42**, kernel `churn-revenue-project`.  
> Re-runs with different hardware, package pins, or HF weight revisions may differ slightly.

### Runtime stack

| Component | Observed |
|-----------|----------|
| Python | 3.13.13 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| lazypredict | 0.3.0 |
| tabfm | 1.0.1 (git `google-research/tabfm`) |
| torch | 2.13.0+cu130 · CUDA: True |
| TabFM device | `cuda` |

### Notebook 1 — Iranian Churn (UCI 563)

| Fact | Observed |
|------|----------|
| Shape | 3,150 rows × 13 features + `Churn` |
| Churn rate | **15.71%** (495 / 3,150) |
| Source | `ucimlrepo.fetch_ucirepo(id=563)` · CC BY 4.0 |
| LazyPredict rank metric | F1 Score |
| Top 3 shortlist | LGBMClassifier, ExtraTreesClassifier, RandomForestClassifier |
| TabFM context | Full train **2,520** rows (no subsample) |

**Tuned Part-1 (test):**

| Model | Accuracy | Prec (churn) | Rec (churn) | F1 (churn) | ROC-AUC | PR-AUC |
|-------|----------|--------------|-------------|------------|---------|--------|
| Tuned LGBM | 0.9619 | 0.8947 | 0.8586 | 0.8763 | 0.9914 | **0.9589** |
| Tuned ExtraTrees | 0.9667 | 0.9063 | 0.8788 | **0.8923** | 0.9892 | 0.9549 |
| Tuned RandomForest | 0.9667 | 0.9239 | 0.8586 | 0.8901 | 0.9885 | 0.9464 |

**Best Part-1 vs TabFM (selection by Part-1 PR-AUC → LGBM):**

| Model | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC |
|-------|----------|------|-----|-----|---------|--------|
| Tuned LGBM | 0.9619 | 0.8947 | 0.8586 | 0.8763 | 0.9914 | 0.9589 |
| **TabFM v1.0.0** | **0.9889** | **0.9340** | **1.0000** | **0.9659** | **0.9993** | **0.9963** |

**Revenue-at-risk** (Σ `Customer Value` on test rows with predicted churn):

| Model | Flagged / test n | Revenue-at-risk |
|-------|------------------|-----------------|
| Tuned LGBM | 95 / 630 | **11,640.61** |
| TabFM | 106 / 630 | **12,130.14** |

**Reading:** On this clean, small, fully labeled telecom table, TabFM dominated ranking metrics. High scores also mean the task is relatively separable (e.g. strong signals like `Complains` / `Status` in EDA)—do not over-generalize to messier CRM data.

---

### Notebook 2 — IBM Telco Customer Churn

| Fact | Observed |
|------|----------|
| Shape | 7,043 × 21 raw columns |
| Churn rate (Yes) | **26.54%** |
| Source | IBM GitHub mirror of Cognos sample CSV |
| `TotalCharges` blanks | **11** strings, all `tenure == 0` → coerced to 0 |
| LazyPredict top 3 | LogisticRegression, AdaBoostClassifier, LinearSVC |
| TabFM context | Full train **5,634** rows |

**Tuned Part-1 (test):**

| Model | Accuracy | Prec (churn) | Rec (churn) | F1 (churn) | ROC-AUC | PR-AUC |
|-------|----------|--------------|-------------|------------|---------|--------|
| Tuned AdaBoost | 0.8034 | 0.6580 | 0.5401 | 0.5932 | 0.8426 | **0.6585** |
| Tuned LogisticRegression | 0.7395 | 0.5060 | 0.7861 | **0.6157** | 0.8413 | 0.6339 |
| Tuned LinearSVC | 0.8020 | 0.6480 | 0.5562 | 0.5986 | 0.8397 | 0.6317 |

**Best Part-1 (PR-AUC) vs TabFM:**

| Model | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC |
|-------|----------|------|-----|-----|---------|--------|
| Tuned AdaBoost | 0.8034 | 0.6580 | 0.5401 | **0.5932** | 0.8426 | 0.6585 |
| **TabFM v1.0.0** | **0.8048** | **0.6667** | 0.5294 | 0.5902 | **0.8516** | **0.6633** |

**Revenue-at-risk** (test set):

| Model | Flagged / test n | Σ MonthlyCharges | Σ TotalCharges (context) |
|-------|------------------|------------------|--------------------------|
| Tuned AdaBoost | 307 / 1,409 | **24,278.10** | 307,552.80 |
| TabFM | 297 / 1,409 | **23,125.40** | 246,248.50 |

**Reading:** Classical and TabFM are **close**. TabFM edges PR-AUC / ROC-AUC; AdaBoost is competitive on F1. This is the more realistic “messy categorical telecom” case—gains are incremental, not magical.

---

### Notebook 3 — Online Retail II (engineered churn)

| Fact | Observed |
|------|----------|
| Raw lines (both Excel sheets) | **1,067,371** |
| Clean transactions | ~805k after cancellations / missing IDs / non-positive qty-price |
| Customer table | **4,933** customers |
| `ucimlrepo(id=502)` | **Failed** (not available for Python import) → official UCI zip fallback |
| Cutoff / window | **2011-06-01** + **90 days** post inactivity = churn |
| Engineered churn rate | **67.14%** (majority churn under this definition) |
| LazyPredict top 3 | RandomForest, LGBM, AdaBoost |
| TabFM context | **Explicit subsample 3,946 → 3,000** (stratified) |

**Tuned Part-1 (test):**

| Model | Accuracy | Prec (churn) | Rec (churn) | F1 (churn) | ROC-AUC | PR-AUC |
|-------|----------|--------------|-------------|------------|---------|--------|
| Tuned LGBM | **0.7822** | 0.7902 | **0.9201** | **0.8502** | 0.8330 | **0.9042** |
| Tuned RandomForest | 0.7751 | **0.8472** | 0.8115 | 0.8290 | 0.8318 | 0.9035 |
| Tuned AdaBoost | 0.7720 | 0.7859 | 0.9080 | 0.8425 | 0.8244 | 0.8955 |

**Best Part-1 vs TabFM:**

| Model | Accuracy | Prec | Rec | F1 | ROC-AUC | PR-AUC |
|-------|----------|------|-----|-----|---------|--------|
| Tuned LGBM | **0.7822** | 0.7902 | **0.9201** | **0.8502** | 0.8330 | **0.9042** |
| TabFM v1.0.0 | 0.7751 | 0.7936 | 0.8989 | 0.8430 | **0.8336** | 0.9014 |

**Revenue-at-risk** (Σ pre-cutoff `Monetary` on predicted churners):

| Model | Flagged / test n | Monetary at risk |
|-------|------------------|------------------|
| Tuned LGBM | 772 / 987 | **642,564.45** |
| TabFM | 751 / 987 | **618,967.83** |

**Reading:** Label engineering dominates the story. At 67% base churn, a high recall model flags many customers—revenue-at-risk is large because the positive class is common. Prefer PR-AUC / F1 interpretation over “we will save $X.” Tuned gradient boosting slightly beat TabFM here; TabFM still competitive under a reduced context budget.

### How to interpret “revenue-at-risk” (all notebooks)

```text
revenue_at_risk = sum(anchor_column | model predicts churn on test set)
```

| It **is** | It **is not** |
|-----------|----------------|
| Value historically/currently associated with flagged customers | Guaranteed future lost cash |
| A size estimate of the alert list | Net savings after interventions |
| Useful for prioritization discussions | Causal uplift without experiments |

---

## 2. For hands-on users

### Prerequisites

| Requirement | Notes |
|-------------|--------|
| OS | Linux tested |
| [`uv`](https://github.com/astral-sh/uv) | Package + venv manager (**required**) |
| Disk | Several GB free (TabFM weights ~6GB+ in HF cache) |
| GPU | **Strongly recommended** (NVIDIA + CUDA). CPU possible but slow / memory-heavy |
| Network | First run downloads datasets + HF weights |

### Installation

```bash
cd "/path/to/Churn Prediction with a Revenue Number Attached"

# 1) Exact Python patch preferred by the project
uv python install 3.13.13
uv python pin 3.13.13

# 2) Create env + install lockfile deps (includes tabfm from GitHub)
uv sync

# 3) Register Jupyter kernel → this project's .venv
uv run python -m ipykernel install --user \
  --name churn-revenue-project \
  --display-name "Python (churn-revenue-project)"

# 4) Sanity checks
uv run python -c "import sys,tabfm,torch,lazypredict,sklearn; \
print(sys.version.split()[0], tabfm.__version__, torch.__version__, torch.cuda.is_available())"
```

Expected shape of a healthy install:

```text
3.13.13  1.0.1  2.13.0+cu130  True
```

(`cuda` may be `False` on CPU-only machines; notebooks still run, TabFM will be slower.)

### Runbook

#### A. Interactive (recommended for learning)

```bash
uv run jupyter lab
# or: uv run jupyter notebook
```

1. Open `notebooks/01_iranian_churn.ipynb` first.  
2. Kernel → **Python (churn-revenue-project)** (not a system `python3`).  
3. Run All. Confirm setup cell prints expected versions.  
4. Proceed to notebooks 02 and 03 the same way.

Committed notebooks already contain **executed outputs** from a full run. Re-running regenerates them.

#### B. Headless re-execution (CI-style / verification)

```bash
for n in 01_iranian_churn 02_telco_churn 03_online_retail_ii_churn; do
  uv run jupytext --to ipynb "notebooks/${n}.py" -o "notebooks/${n}.ipynb"
  MPLBACKEND=Agg uv run jupyter nbconvert --to notebook --execute "notebooks/${n}.ipynb" \
    --output "${n}.ipynb" \
    --ExecutePreprocessor.kernel_name=churn-revenue-project \
    --ExecutePreprocessor.timeout=3600
done
```

#### C. Script mode (fastest debugging)

```bash
MPLBACKEND=Agg uv run python notebooks/01_iranian_churn.py
MPLBACKEND=Agg uv run python notebooks/02_telco_churn.py
MPLBACKEND=Agg uv run python notebooks/03_online_retail_ii_churn.py
```

Errors print as normal Python tracebacks (easier than notebook UI while fixing bugs).

### Data acquisition (no manual pre-download required)

| Notebook | Mechanism |
|----------|-----------|
| 01 Iranian | `ucimlrepo.fetch_ucirepo(id=563)` |
| 02 Telco | `pandas.read_csv` from IBM GitHub raw URL |
| 03 Retail | Tries `ucimlrepo` id 502 → on failure downloads [UCI static zip](https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip) and reads both Excel sheets |

Optional: place caches under `data/` (gitignored). Notebooks do not require anything pre-staged there.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError: ... pytorch_model.bin` | TabFM **1.0.0** from PyPI | Use git source (this repo’s `pyproject.toml` already does); `uv sync` |
| `NameError: safetensors` | Missing dependency | `uv add safetensors` (already in project deps) |
| TabFM hang / thrashing, high CPU, low GPU mem | Loading weights on **CPU** under RAM pressure | Load with `device="cuda"` (notebooks do this when CUDA is available) |
| CUDA OOM on TabFM | Large context × model size | Reduce `MAX_TABFM_CONTEXT` in NB3; free other GPU processes |
| Wrong kernel / missing packages in notebook | Jupyter not using project venv | Select `churn-revenue-project`; reinstall kernel command above |
| `DatasetNotFoundError` for UCI 502 | Known: not importable via ucimlrepo API | Expected; NB3 falls back to UCI zip automatically |
| Telco `TotalCharges` type errors on pandas 3 | Assigning int into string dtype | Notebook replaces blanks with `"0"` then `to_numeric` |
| LazyPredict model name not in zoo | Rare model wins shortlist | Falls back to RandomForest with warning—or add a grid |
| Plots blank in headless run | No display | Set `MPLBACKEND=Agg` (as in runbook) |
| First TabFM run very slow | HF weight download | One-time; subsequent runs use `~/.cache/huggingface` |

### Extending the project

| Goal | Where to change |
|------|-----------------|
| New classical model grid | `MODEL_ZOO` dict in each notebook’s Part 1 section |
| Different LazyPredict rank metric | Ranking column selection after `lazy.fit` (try Balanced Accuracy) |
| Alternate retail churn definition | `CUTOFF` / `POST_DAYS` in NB3 **and** re-check class balance gate |
| Larger TabFM context | `MAX_TABFM_CONTEXT` in NB3 (watch VRAM) |
| Cost-sensitive threshold | After `predict_proba`, sweep thresholds for max expected utility (not implemented—natural next step) |
| Fourth dataset | Copy notebook skeleton; keep split + dual Part1/Part2 + revenue close |
| CLI wrapper | Add `src/` entrypoint that shells `uv run python notebooks/...` or parameterizes seeds via env vars |

Suggested production-oriented extensions (not in scope of current notebooks): nested CV, calibration plots, intervention uplift tests, model cards per dataset, MLflow/W&B tracking of grids.

### Operational smoke test (2 minutes after install)

```bash
uv run python -c "
import numpy as np, pandas as pd, torch
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = tabfm_v1_0_0.load(model_type='classification', device=dev)
clf = TabFMClassifier(model=m)
X = pd.DataFrame({'a':[1.,2.,3.,4.], 'b':['x','y','x','y']})
y = np.array([0,1,0,1])
clf.fit(X,y)
print('device', dev, 'preds', clf.predict(X), 'proba', clf.predict_proba(X).shape)
"
```

---

## 3. For tutorial learners

This section teaches **why** the pipeline looks the way it does. You do not need prior foundation-model experience, but basic supervised learning helps.

### Core concepts

#### Churn as a class-imbalanced problem

In real portfolios, leavers are usually the minority (Iranian **15.71%**, Telco **26.54%**). A model that always predicts “stay” can look accurate and still be useless.

| Metric | Friendly meaning | When it misleads |
|--------|------------------|------------------|
| Accuracy | Overall % correct | High when majority class dominates |
| Precision (churn) | Of flagged leavers, how many truly left | Low if you cry wolf |
| Recall (churn) | Of true leavers, how many you caught | Low if you miss leavers |
| F1 (churn) | Harmonic mean of precision & recall | Single number for ranking models |
| ROC-AUC | Ranking quality across thresholds | Can look good while precision is poor at useful recall |
| **PR-AUC** | Ranking quality emphasizing the positive class | Preferred under imbalance for model selection |

**This project:** LazyPredict shortlist ranked by **F1**; tuning optimizes **average precision (PR-AUC)**; final tables report all of the above.

#### Train/test split and leakage

- **Stratified split** keeps churn rate similar in train and test.  
- **Fit on train only** for scalers / one-hot encoders.  
- **No future information** in features—especially critical in notebook 3, where the label uses a future window.

#### LazyPredict → “do it properly”

LazyPredict fits many sklearn-style models with defaults and prints a leaderboard. Defaults are convenient but rarely optimal.

Workflow taught in every notebook:

1. Screen many algorithms (LazyPredict).  
2. Name the top 3 explicitly from *this* run’s board.  
3. Re-implement with `RandomizedSearchCV` + stratified CV.  
4. Produce confusion matrices, ROC, and PR curves—not only a single score.

#### What TabFM is (and is not)

**TabFM** (Google Research tabular foundation model) treats your labeled training rows as **in-context examples**. At inference it predicts new rows in (essentially) a forward pass.

| Idea | TabFM | Classic LightGBM/sklearn |
|------|-------|---------------------------|
| Learning on your data | Context packing (no weight update) | Gradient-based `.fit` updates parameters |
| Hyperparameter search | Not required for base zero-shot use | Usually required for best results |
| Training cost | Memory scales with # context rows | Trees scale with data + depth/leaves |
| Fine-tuning with Unsloth | N/A (not a causal-LM LoRA setup) | N/A |

Notebooks load:

```python
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
model = tabfm_v1_0_0.load(model_type="classification", device="cuda")  # or cpu
clf = TabFMClassifier(model=model)
clf.fit(X_train, y_train)   # stores / prepares context
proba = clf.predict_proba(X_test)
```

**License mental model:** library code ≈ Apache 2.0; **weights** ≈ non-commercial. Fine for learning; review before client work.

#### RFM label engineering (notebook 3 only)

Online Retail II is **line-item transactions**, not a churn table.

1. **Clean:** drop cancellations (`Invoice` starting with `C`), missing customers, non-positive quantity/price.  
2. **Aggregate** to one row per customer before a cutoff date:  
   - **R**ecency — days since last purchase before cutoff  
   - **F**requency — distinct invoices  
   - **M**onetary — sum(Quantity × Price)  
3. **Label:** churn if the customer has **no** purchase in the 90 days *after* the cutoff.

That label is a **modeling assumption**. A different cutoff is a different study—not “the” truth.

#### Revenue-at-risk and lift curves

- **Revenue-at-risk:** sum of a money column over customers the model flags.  
- **Lift / prioritization curve:** sort everyone by predicted churn probability; plot cumulative true-churn value vs number contacted. Steeper than the random diagonal ⇒ useful ranking for limited call centers.

These connect ML scores to **retention operations**, which is the point of the project title.

### Implementation flow (step-by-step)

```text
[0] Environment
    uv sync · pin Python · register kernel · print versions

[1] Acquire data (network)
    UCI / IBM / UCI zip — never invent rows

[2] EDA
    shape · missingness · class balance · univariates
    churn rate by category · correlation · outliers
    Write what *this* dataset showed (not generic text)

[3] Features / labels
    Encode · fix dirty fields (Telco TotalCharges)
    Or engineer RFM + churn (Retail)
    Keep revenue columns available for Part 9

[4] Split
    train_test_split(..., stratify=y, random_state=42)
    Freeze indices for Part 1 and Part 2

[5] Part 1 Classical
    LazyClassifier.fit → leaderboard
    Pick top 3 by F1
    RandomizedSearchCV(scoring='average_precision')
    Full metric bundle per model

[6] Part 2 TabFM
    License note in markdown
    load(device=cuda|cpu) · fit context · predict
    Same metric bundle
    Disclose subsample if used

[7] Decision layer
    Side-by-side table
    Revenue-at-risk for flags
    Lift curve
    Manager paragraph in plain language
```

### How to read the notebooks

| Section | Learner focus |
|---------|----------------|
| Title / framing | What business question is answered? |
| Setup | Can you reproduce the env from printed versions? |
| Acquisition | Where did the data come from? License? |
| EDA | What is the base churn rate? Which features look predictive? |
| Engineering | Any leakage risk? Encoding choices? |
| Part 1 | Why these top 3? Did tuning change the story? |
| Part 2 | How does TabFM compare *on the same test set*? |
| Close | Is revenue framed cautiously? Would you trust this for budget? |

**Suggested study order:** Notebook 1 (clean labels, strong signal) → Notebook 2 (categoricals, dirty fields) → Notebook 3 (you build the label).

### Conceptual FAQ

**Q: Why not only use TabFM?**  
A: Classical models are still strong on many tables (see Telco and Retail). Comparing both is the scientific habit; foundation models are not automatically winners.

**Q: Why F1 for LazyPredict but PR-AUC for tuning?**  
A: LazyPredict exposes F1 on its board (easy shortlist). Tuning uses PR-AUC because it better matches imbalanced ranking. Final tables report both families of metrics.

**Q: Is 100% recall on Iranian TabFM “solved churn”?**  
A: No. Small test set (99 positives), strong features, single split. Treat as evidence on *this* sample, not production certification.

**Q: Can I use this commercially with TabFM?**  
A: Review the **TabFM Non-Commercial License** for weights yourself. This repo’s MIT license covers *our* code, not Google’s weights.

---

## Repository map

```text
.
├── README.md                 # this document
├── LICENSE                   # MIT (project code & notebooks)
├── pyproject.toml            # uv project + dependencies
├── uv.lock                   # locked versions
├── .python-version           # 3.13.13
├── .gitignore
├── data/                     # optional caches (ignored); .gitkeep only
└── notebooks/
    ├── 01_iranian_churn.ipynb / .py
    ├── 02_telco_churn.ipynb / .py
    └── 03_online_retail_ii_churn.ipynb / .py
```

---

## Licenses & data provenance

| Asset | License / terms |
|-------|-----------------|
| This repository (notebooks, scripts, docs) | **MIT** — see [`LICENSE`](LICENSE) |
| TabFM Python package code | Apache 2.0 (upstream) |
| TabFM **weights** (`google/tabfm-1.0.0-pytorch`) | **TabFM Non-Commercial License v1.0** — [model card](https://huggingface.co/google/tabfm-1.0.0-pytorch) |
| UCI Iranian Churn (563) | CC BY 4.0 (via UCI / ucimlrepo) |
| UCI Online Retail II (502) | CC BY 4.0 (UCI static zip) |
| IBM Telco sample | IBM’s public GitHub sample distribution |

Nothing in this README is legal advice.

---

## Honest limitations

1. **Single stratified split** — Final metrics are not nested-CV estimates of generalization error.  
2. **No calibration / threshold optimization for cost** — Default 0.5 decision threshold for class labels; ranking metrics matter more.  
3. **No uplift modeling** — We do not estimate treatment effect of retention offers.  
4. **TabFM context subsample on Retail** — Part 2 may be slightly disadvantaged vs full-train Part 1.  
5. **Engineered retail label** — 67% churn under 90-day inactivity is a definition, not CRM truth; high revenue-at-risk partly reflects base rate.  
6. **Hardware dependence** — Results used CUDA; CPU-only runs may time out or OOM without adjustment.  
7. **Non-commercial TabFM weights** — Blocks some portfolio-to-client copy-paste paths without legal review.  
8. **Package API drift risk** — TabFM is young; pin git source carefully when refreshing deps.  
9. **Cross-dataset leaderboards are invalid** — Different problems; do not average F1 across notebooks.  
10. **Duplicate rows (Iranian EDA noted 300)** — Left as-is for fidelity to published table; a production pipeline might dedupe with domain rules.

---

## Quick start (shortest path)

```bash
uv sync
uv run python -m ipykernel install --user --name churn-revenue-project \
  --display-name "Python (churn-revenue-project)"
uv run jupyter lab
# Open notebooks/01_iranian_churn.ipynb → kernel churn-revenue-project → Run All
```

---

## Contributing & community

- [Contributing guide](CONTRIBUTING.md) — setup for PRs, notebook norms, verification  
- [Code of Conduct](CODE_OF_CONDUCT.md) — Contributor Covenant v2.1  
- [Issue templates](.github/ISSUE_TEMPLATE/) — bug, feature, and question forms on GitHub  

## Acknowledgments

- UCI Machine Learning Repository — Iranian Churn (563), Online Retail II (502)  
- IBM — Telco Customer Churn sample (public GitHub mirror)  
- Google Research — [TabFM](https://github.com/google-research/tabfm) and the [TabFM announcement](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)  

---

*Evidence-backed tutorial project: classical ML + tabular foundation models + revenue-aware decision framing.*
