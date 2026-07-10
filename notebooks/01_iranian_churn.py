# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (churn-revenue-project)
#     language: python
#     name: churn-revenue-project
# ---

# %% [markdown]
# # Iranian Telecom Churn — Production-Grade Pipeline
#
# **Goal.** Predict churn, rank customers for retention, and attach a
# **Customer Value** revenue number — with methods used in production churn
# systems (not tutorial defaults only).
#
# **Production upgrades (vs baseline notebook):**
# 1. **Train / validation / test** stratified splits (tune threshold on val only)
# 2. **Imbalance-aware GBMs** (`class_weight`, `scale_pos_weight`, CatBoost balanced)
# 3. Deeper **RandomizedSearchCV** on PR-AUC
# 4. **Soft-voting ensemble** + optional **stacking**
# 5. **Threshold moving** (maximize F1 on validation — not fixed 0.5)
# 6. **Isotonic calibration** for usable probabilities
# 7. **TabFM.ensemble()** preset (feature crosses, SVD, NNLS, Platt calibration)
# 8. Dummy / majority baseline for honesty
#
# **Sources of practice:** class weights first; threshold moving; calibrate before
# expected-value use; ensemble GBMs; TabFM HF ensemble preset (Google TabFM 1.0.x).

# %% [markdown]
# ## 1. Setup

# %%
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import lazypredict
import tabfm
import torch

from ucimlrepo import fetch_ucirepo
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.base import clone

from churn_revenue.metrics import classification_bundle, evaluate_scores, lift_revenue_curve
from churn_revenue.threshold import tune_threshold_f1, apply_threshold
from churn_revenue.modeling import (
    RANDOM_STATE,
    build_boosting_candidates,
    resolve_model,
    tune_model,
    soft_vote_proba,
    fit_calibrated_isotonic,
    production_classical_stack,
    predict_proba_matrix,
)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):
        print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else obj)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 4.5)

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__, "| sklearn:", sklearn.__version__)
print("lazypredict:", lazypredict.__version__, "| tabfm:", tabfm.__version__)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Seed:", RANDOM_STATE)

# %% [markdown]
# ## 2. Data + EDA (essentials)

# %%
iranian = fetch_ucirepo(id=563)
df = pd.concat([iranian.data.features, iranian.data.targets], axis=1)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
display(df.head(3))

target_col = "Churn"
value_col = "Customer Value"
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])
revenue = df[value_col].copy()
churn_rate = float(y.mean())
print(f"Churn rate: {churn_rate:.2%}  | imbalance: {churn_rate < 0.35}")
print("Missing:", int(df.isna().sum().sum()), "| duplicates:", int(df.duplicated().sum()))

fig, ax = plt.subplots()
y.value_counts().sort_index().plot(kind="bar", color=["#4C78A8", "#E45756"], ax=ax, rot=0)
ax.set_title(f"Churn balance ({churn_rate:.1%})")
plt.tight_layout()
plt.show()

corr = df.select_dtypes("number").corr()[target_col].drop(target_col).sort_values(key=np.abs, ascending=False)
print("Top |corr| with Churn:\n", corr.head(8))

# %% [markdown]
# ## 3. Train / validation / test split
#
# Production pattern: **never** tune thresholds or early-stopping decisions on the
# final test set. We use 60% train / 20% val / 20% test, all stratified.

# %%
X_trainval, X_test, y_trainval, y_test, rev_trainval, rev_test = train_test_split(
    X, y, revenue, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val, rev_train, rev_val = train_test_split(
    X_trainval,
    y_trainval,
    rev_trainval,
    test_size=0.25,  # 0.25 * 0.80 = 0.20 overall
    random_state=RANDOM_STATE,
    stratify=y_trainval,
)
print("Train:", X_train.shape, f"churn={y_train.mean():.2%}")
print("Val:  ", X_val.shape, f"churn={y_val.mean():.2%}")
print("Test: ", X_test.shape, f"churn={y_test.mean():.2%}")

# LazyPredict screening uses train→val (not test)
X_train_lp, X_val_lp = X_train.copy(), X_val.copy()
X_train_tab, X_val_tab, X_test_tab = X_train.copy(), X_val.copy(), X_test.copy()

# %% [markdown]
# ## 4. Part 1 — screen + production classical models

# %%
print("=== Dummy baseline (most frequent) ===")
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
d_proba = predict_proba_matrix(dummy, X_test)
d_pred = dummy.predict(X_test)
dummy_metrics = classification_bundle(y_test, d_pred, d_proba, "Dummy most_frequent")

print("\n=== LazyPredict screen (train→val) ===")
lazy = LazyClassifier(verbose=0, ignore_warnings=True, random_state=RANDOM_STATE)
models_df, _ = lazy.fit(X_train_lp, X_val_lp, y_train, y_val)
display(models_df.head(12))
rank_col = "F1 Score" if "F1 Score" in models_df.columns else models_df.columns[0]
top3 = list(models_df.sort_values(rank_col, ascending=False).head(3).index)
print("LazyPredict top-3 by", rank_col, "→", top3)

# Always train production GBM core + LazyPredict winners
must_train = ["LGBMClassifier", "XGBClassifier", "CatBoostClassifier", "HistGradientBoostingClassifier"]
to_train = list(dict.fromkeys(top3 + must_train))  # unique, preserve order
print("Models to tune:", to_train)

zoo = build_boosting_candidates(y_train.to_numpy())
fitted: dict[str, any] = {}
val_probas: dict[str, np.ndarray] = {}

for raw in to_train:
    name, est, grid = resolve_model(raw, y_train.to_numpy(), zoo)
    if name in fitted:
        continue
    print(f"\n### Tuning {name}")
    best = tune_model(clone(est), grid, X_train, y_train, n_iter=35, n_jobs=2)
    fitted[name] = best
    val_probas[name] = predict_proba_matrix(best, X_val)[:, 1]

# Soft-vote ensemble of all tuned models
print("\n### Soft-vote ensemble")
vote_models = list(fitted.values())
val_vote = soft_vote_proba(vote_models, X_val)[:, 1]
fitted["SoftVote"] = ("soft_vote", vote_models)  # special

# Stacking top 3 by val PR-AUC among single models
from sklearn.metrics import average_precision_score

single_names = [n for n in fitted if n != "SoftVote"]
val_ap = {n: average_precision_score(y_val, val_probas[n]) for n in single_names}
stack_bases = sorted(val_ap, key=val_ap.get, reverse=True)[:3]
print("Stacking bases:", stack_bases, {k: round(val_ap[k], 4) for k in stack_bases})
stack = production_classical_stack([(n, fitted[n]) for n in stack_bases], X_train, y_train)
fitted["Stacking"] = stack
val_probas["Stacking"] = predict_proba_matrix(stack, X_val)[:, 1]
val_probas["SoftVote"] = val_vote

# Thresholds on validation
print("\n=== Validation threshold tuning (max F1) ===")
thresholds: dict[str, float] = {}
val_rows = []
for name, score in val_probas.items():
    t, row = tune_threshold_f1(y_val, score)
    thresholds[name] = t
    row["model"] = name
    row["val_pr_auc"] = float(average_precision_score(y_val, score))
    val_rows.append(row)
    print(f"  {name}: t={t:.3f}  val_F1={row['f1_churn']:.4f}  val_PR-AUC={row['val_pr_auc']:.4f}")

val_df = pd.DataFrame(val_rows).set_index("model").sort_values("f1_churn", ascending=False)
display(val_df)

best_classical_name = val_df["f1_churn"].idxmax()
print("Best classical by val F1@tuned threshold:", best_classical_name)

# Optional calibration of best single model (if not ensemble markers)
calibrated = None
if best_classical_name not in ("SoftVote", "Stacking"):
    print(f"\nIsotonic-calibrating {best_classical_name} via CV...")
    calibrated = fit_calibrated_isotonic(clone(fitted[best_classical_name]), X_train, y_train, cv=3)
    cal_val = predict_proba_matrix(calibrated, X_val)[:, 1]
    t_cal, row_cal = tune_threshold_f1(y_val, cal_val)
    thresholds["Calibrated"] = t_cal
    fitted["Calibrated"] = calibrated
    val_probas["Calibrated"] = cal_val
    print(f"  Calibrated t={t_cal:.3f} val_F1={row_cal['f1_churn']:.4f}")
    if row_cal["f1_churn"] >= val_df.loc[best_classical_name, "f1_churn"] - 1e-6:
        best_classical_name = "Calibrated"
        print("  → using Calibrated as best classical")

# %% [markdown]
# ## 5. Classical models — final test evaluation

# %%
def get_test_proba(name: str) -> np.ndarray:
    obj = fitted[name]
    if name == "SoftVote":
        return soft_vote_proba(obj[1], X_test)[:, 1]
    return predict_proba_matrix(obj, X_test)[:, 1]


part1_metrics = []
test_scores: dict[str, np.ndarray] = {}
for name in [best_classical_name] + [n for n in ["SoftVote", "Stacking", "LGBMClassifier", "XGBClassifier", "CatBoostClassifier"] if n in fitted and n != best_classical_name]:
    if name not in fitted:
        continue
    score = get_test_proba(name)
    t = thresholds.get(name, 0.5)
    pred = apply_threshold(score, t)
    m = classification_bundle(y_test, pred, np.column_stack([1 - score, score]), f"{name} (t={t:.3f})")
    m["threshold"] = t
    part1_metrics.append(m)
    test_scores[name] = score

part1_df = pd.DataFrame(part1_metrics).set_index("model")
display(part1_df.sort_values("f1_churn", ascending=False))

best_part1_row = part1_df["f1_churn"].idxmax()
best_part1_key = best_classical_name if best_classical_name in test_scores else list(test_scores.keys())[0]
# align key to metrics index loosely
for k in test_scores:
    if k in best_part1_row or best_part1_row.startswith(k):
        best_part1_key = k
        break
print("Selected Part-1 for comparison:", best_part1_key, "| table row:", best_part1_row)

# %% [markdown]
# ## 6. Part 2 — TabFM production ensemble
#
# **License:** TabFM weights = Non-Commercial License v1.0; code Apache 2.0.
# Review before commercial use — no legal opinion offered.
#
# We use `TabFMClassifier.ensemble()`: feature crosses, SVD features, NNLS
# blending, Platt calibration (Google’s recommended heavier preset).

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TabFM on {device}...")
base = tabfm_v1_0_0.load(model_type="classification", device=device)

# Context = train only (val used for classical threshold; keep TabFM pure train context)
# Optionally merge train+val for more context (common for ICL) — we use train+val
# for TabFM context to maximize ICL examples while test stays pure holdout.
X_ctx = pd.concat([X_train, X_val], axis=0)
y_ctx = pd.concat([y_train, y_val], axis=0)
print(f"TabFM context rows: {len(X_ctx)} (train+val); test={len(X_test)}")

# n_estimators=16 for speed/VRAM balance on laptop GPU; override ensemble default 32 if needed
tab_clf = TabFMClassifier.ensemble(
    base,
    n_estimators=16 if device == "cuda" else 8,
    random_state=RANDOM_STATE,
    verbose=True,
    batch_size=1,
)
tab_clf.fit(X_ctx, y_ctx.to_numpy())
tab_proba = np.asarray(tab_clf.predict_proba(X_test_tab), dtype=float)
tab_score = tab_proba[:, 1] if tab_proba.ndim == 2 else tab_proba

# Threshold from validation scores of TabFM (need val predict)
tab_val_proba = np.asarray(tab_clf.predict_proba(X_val_tab), dtype=float)
tab_val_score = tab_val_proba[:, 1] if tab_val_proba.ndim == 2 else tab_val_proba
t_tab, tab_val_row = tune_threshold_f1(y_val, tab_val_score)
print(f"TabFM val threshold={t_tab:.3f} F1={tab_val_row['f1_churn']:.4f}")
tab_pred = apply_threshold(tab_score, t_tab)
tab_metrics = classification_bundle(
    y_test, tab_pred, tab_proba, f"TabFM.ensemble (t={t_tab:.3f})"
)
tab_metrics["threshold"] = t_tab

# %% [markdown]
# ## 7. Final comparison + revenue-at-risk

# %%
compare = pd.concat(
    [
        part1_df.loc[[best_part1_row]],
        pd.DataFrame([tab_metrics]).set_index("model"),
        part1_df.loc[[dummy_metrics["model"]]] if dummy_metrics["model"] in part1_df.index else pd.DataFrame([dummy_metrics]).set_index("model"),
    ]
)
# ensure dummy present
if "Dummy most_frequent" not in compare.index:
    compare = pd.concat([compare, pd.DataFrame([dummy_metrics]).set_index("model")])

print("Side-by-side (TEST only):")
display(compare.sort_values("f1_churn", ascending=False))

fig, ax = plt.subplots(figsize=(9, 4.5))
plot_cols = [c for c in ["f1_churn", "pr_auc", "roc_auc", "recall_churn", "precision_churn"] if c in compare.columns]
compare[plot_cols].plot(kind="bar", ax=ax, rot=20)
ax.set_title("Test metrics — production classical vs TabFM ensemble vs baseline")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()

# Revenue
best_score = test_scores[best_part1_key]
best_t = thresholds.get(best_part1_key, 0.5)
best_pred = apply_threshold(best_score, best_t)
rev_test_arr = rev_test.to_numpy()

for label, pred, score in [
    (best_part1_row, best_pred, best_score),
    (f"TabFM.ensemble (t={t_tab:.3f})", tab_pred, tab_score),
]:
    mask = np.asarray(pred).astype(int) == 1
    print(f"\n{label}:")
    print(f"  Flagged: {int(mask.sum())} / {len(pred)}")
    print(f"  Revenue-at-risk (Customer Value): {float(rev_test_arr[mask].sum()):,.2f}")
    print("  Framing: associated value of flagged customers — not proven lost cash.")

lift_revenue_curve(y_test, best_score, rev_test_arr, title=str(best_part1_row), ylabel="Cumulative Customer Value of true churners")
lift_revenue_curve(y_test, tab_score, rev_test_arr, title="TabFM.ensemble", ylabel="Cumulative Customer Value of true churners")

# %% [markdown]
# ## 8. Manager summary
#
# - Iranian churn is **imbalanced (~15.7%)**; we optimize ranking (PR-AUC) and
#   **threshold on validation**, never on test.
# - Production classical stack: imbalance-aware GBMs (LGBM/XGB/CatBoost), soft
#   vote / stacking, optional calibration.
# - **TabFM.ensemble** uses Google’s heavier zero-shot preset (crosses + NNLS +
#   Platt); weights are **non-commercial**.
# - Revenue-at-risk uses **Customer Value** on predicted churners for campaign
#   sizing — not causal savings without uplift tests.
# - Compare tables above for this run’s winner.

# %%
print("Notebook 01 (production) complete.")
print(compare.round(4).to_string())
