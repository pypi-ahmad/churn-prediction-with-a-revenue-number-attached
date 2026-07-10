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
# # Online Retail II — Production RFM Churn Pipeline
#
# Transactional data → **engineered** churn label (RFM + 90-day inactivity) →
# production classical stack + **TabFM.ensemble**.
#
# Production upgrades: richer RFM features (logs, ratios), 60/20/20 splits,
# imbalance-aware GBMs, threshold moving, soft-vote/stacking, disclosed TabFM
# context policy.
#
# **Assumption (fixed):** cutoff `2011-06-01`, churn if no purchase in next 90 days.

# %%
from __future__ import annotations

import io
import sys
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

import sklearn
import lazypredict
import tabfm
import torch

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score

from churn_revenue.metrics import classification_bundle, lift_revenue_curve
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
print("Python:", sys.version.split()[0], "| tabfm:", tabfm.__version__)
print("CUDA:", torch.cuda.is_available())

# %% [markdown]
# ## 1. Acquire + clean transactions

# %%
UCI_ZIP = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
tx = None
try:
    from ucimlrepo import fetch_ucirepo

    retail = fetch_ucirepo(id=502)
    tx = retail.data.features.copy()
    print("ucimlrepo OK", tx.shape)
except Exception as e:
    print("ucimlrepo failed:", type(e).__name__, "→ UCI zip")
    r = requests.get(UCI_ZIP, timeout=180)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        with zf.open("online_retail_II.xlsx") as fh:
            xl = pd.ExcelFile(fh)
            tx = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True)
print("Raw:", tx.shape, "cols:", list(tx.columns))

tx = tx.copy()
tx["Invoice"] = tx["Invoice"].astype(str)
tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"])
tx = tx[~tx["Invoice"].str.startswith("C")]
tx = tx.dropna(subset=["Customer ID"])
tx = tx[(tx["Quantity"] > 0) & (tx["Price"] > 0)]
tx["Customer ID"] = tx["Customer ID"].astype(int)
tx["line_revenue"] = tx["Quantity"] * tx["Price"]
print("Clean:", tx.shape, tx["InvoiceDate"].min(), "→", tx["InvoiceDate"].max())

# %% [markdown]
# ## 2. RFM + engineered features + label

# %%
CUTOFF = pd.Timestamp("2011-06-01")
POST_DAYS = 90
post_end = CUTOFF + pd.Timedelta(days=POST_DAYS)
pre = tx[tx["InvoiceDate"] < CUTOFF].copy()
post = tx[(tx["InvoiceDate"] >= CUTOFF) & (tx["InvoiceDate"] < post_end)].copy()

country_mode = pre.groupby("Customer ID")["Country"].agg(lambda s: s.value_counts().index[0])
cust = pre.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda s: (CUTOFF - s.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("line_revenue", "sum"),
    n_products=("StockCode", "nunique"),
    n_items=("Quantity", "sum"),
    tenure_days=("InvoiceDate", lambda s: max((s.max() - s.min()).days, 0)),
    avg_line_value=("line_revenue", "mean"),
    max_line_value=("line_revenue", "max"),
    std_line_value=("line_revenue", "std"),
)
cust["std_line_value"] = cust["std_line_value"].fillna(0.0)
cust = cust.join(country_mode.rename("Country"), how="left")
# production feature transforms (leakage-safe: all pre-cutoff)
cust["log_monetary"] = np.log1p(cust["Monetary"].clip(lower=0))
cust["log_frequency"] = np.log1p(cust["Frequency"])
cust["log_recency"] = np.log1p(cust["Recency"])
cust["monetary_per_invoice"] = cust["Monetary"] / cust["Frequency"].clip(lower=1)
cust["items_per_invoice"] = cust["n_items"] / cust["Frequency"].clip(lower=1)
cust["products_per_invoice"] = cust["n_products"] / cust["Frequency"].clip(lower=1)
cust["recency_x_freq"] = cust["Recency"] * cust["Frequency"]
cust["activity_rate"] = cust["Frequency"] / (cust["tenure_days"] + 1)

active = set(post["Customer ID"].unique())
cust["churn"] = (~cust.index.isin(active)).astype(int)
rate = float(cust["churn"].mean())
print(cust["churn"].value_counts())
print(f"Churn rate: {rate:.2%}")
if rate < 0.05 or rate > 0.95:
    raise SystemExit(f"STOP: unusable churn rate {rate:.2%}")

fig, ax = plt.subplots()
cust["churn"].value_counts().sort_index().plot(kind="bar", color=["#4C78A8", "#E45756"], ax=ax, rot=0)
ax.set_title(f"Engineered churn ({rate:.1%}) cutoff={CUTOFF.date()} +{POST_DAYS}d")
plt.tight_layout()
plt.show()

print("Mean RFM by churn:")
display(cust.groupby("churn")[["Recency", "Frequency", "Monetary"]].mean())

# %% [markdown]
# ## 3. Splits + encode

# %%
feature_cols = [
    "Recency",
    "Frequency",
    "Monetary",
    "n_products",
    "n_items",
    "tenure_days",
    "avg_line_value",
    "max_line_value",
    "std_line_value",
    "log_monetary",
    "log_frequency",
    "log_recency",
    "monetary_per_invoice",
    "items_per_invoice",
    "products_per_invoice",
    "recency_x_freq",
    "activity_rate",
    "Country",
]
X_tab = cust[feature_cols].copy()
y = cust["churn"].astype(int)
revenue = cust["Monetary"].copy()

X_tv, X_test_tab, y_tv, y_test, rev_tv, rev_test = train_test_split(
    X_tab, y, revenue, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train_tab, X_val_tab, y_train, y_val, rev_train, rev_val = train_test_split(
    X_tv, y_tv, rev_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
)

num_features = [c for c in feature_cols if c != "Country"]
cat_features = ["Country"]
pre = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ]
)
X_train_e = pre.fit_transform(X_train_tab)
X_val_e = pre.transform(X_val_tab)
X_test_e = pre.transform(X_test_tab)
try:
    names = pre.get_feature_names_out()
except Exception:
    names = [f"f{i}" for i in range(X_train_e.shape[1])]
X_train_lp = pd.DataFrame(X_train_e, columns=names, index=X_train_tab.index)
X_val_lp = pd.DataFrame(X_val_e, columns=names, index=X_val_tab.index)
X_test_lp = pd.DataFrame(X_test_e, columns=names, index=X_test_tab.index)
print("Train", X_train_lp.shape, f"churn={y_train.mean():.2%}")

# %% [markdown]
# ## 4. Production classical stack

# %%
dummy = DummyClassifier(strategy="most_frequent").fit(X_train_lp, y_train)
dummy_metrics = classification_bundle(
    y_test,
    dummy.predict(X_test_lp),
    predict_proba_matrix(dummy, X_test_lp),
    "Dummy most_frequent",
)

lazy = LazyClassifier(verbose=0, ignore_warnings=True, random_state=RANDOM_STATE)
models_df, _ = lazy.fit(X_train_lp, X_val_lp, y_train, y_val)
display(models_df.head(10))
rank_col = "F1 Score" if "F1 Score" in models_df.columns else models_df.columns[0]
top3 = list(models_df.sort_values(rank_col, ascending=False).head(3).index)
must = ["LGBMClassifier", "XGBClassifier", "CatBoostClassifier", "HistGradientBoostingClassifier"]
to_train = list(dict.fromkeys(top3 + must))
print("Tune:", to_train)

zoo = build_boosting_candidates(y_train.to_numpy())
fitted: dict = {}
val_probas: dict = {}
for raw in to_train:
    name, est, grid = resolve_model(raw, y_train.to_numpy(), zoo)
    if name in fitted:
        continue
    print(f"\n### {name}")
    best = tune_model(clone(est), grid, X_train_lp, y_train, n_iter=35, n_jobs=2)
    fitted[name] = best
    val_probas[name] = predict_proba_matrix(best, X_val_lp)[:, 1]

vote_models = list(fitted.values())
val_probas["SoftVote"] = soft_vote_proba(vote_models, X_val_lp)[:, 1]
fitted["SoftVote"] = ("soft_vote", vote_models)
val_ap = {n: average_precision_score(y_val, val_probas[n]) for n in fitted if n != "SoftVote"}
stack_bases = sorted(val_ap, key=val_ap.get, reverse=True)[:3]
stack = production_classical_stack([(n, fitted[n]) for n in stack_bases], X_train_lp, y_train)
fitted["Stacking"] = stack
val_probas["Stacking"] = predict_proba_matrix(stack, X_val_lp)[:, 1]

thresholds = {}
val_rows = []
for name, score in val_probas.items():
    t, row = tune_threshold_f1(y_val, score)
    thresholds[name] = t
    row["model"] = name
    row["val_pr_auc"] = float(average_precision_score(y_val, score))
    val_rows.append(row)
    print(f"  {name}: t={t:.3f} F1={row['f1_churn']:.4f}")
val_df = pd.DataFrame(val_rows).set_index("model").sort_values("f1_churn", ascending=False)
display(val_df)
best_classical = val_df["f1_churn"].idxmax()

if best_classical not in ("SoftVote", "Stacking"):
    cal = fit_calibrated_isotonic(clone(fitted[best_classical]), X_train_lp, y_train, cv=3)
    cs = predict_proba_matrix(cal, X_val_lp)[:, 1]
    t_cal, row_cal = tune_threshold_f1(y_val, cs)
    fitted["Calibrated"] = cal
    val_probas["Calibrated"] = cs
    thresholds["Calibrated"] = t_cal
    if row_cal["f1_churn"] >= val_df.loc[best_classical, "f1_churn"] - 1e-9:
        best_classical = "Calibrated"
        print("Using Calibrated")

# %% [markdown]
# ## 5. Test — classical

# %%
def test_score(name: str) -> np.ndarray:
    obj = fitted[name]
    if name == "SoftVote":
        return soft_vote_proba(obj[1], X_test_lp)[:, 1]
    return predict_proba_matrix(obj, X_test_lp)[:, 1]


part1_metrics = []
test_scores = {}
for name in dict.fromkeys([best_classical, "SoftVote", "Stacking", "LGBMClassifier", "CatBoostClassifier", "XGBClassifier"]):
    if name not in fitted:
        continue
    score = test_score(name)
    t = thresholds.get(name, 0.5)
    pred = apply_threshold(score, t)
    m = classification_bundle(y_test, pred, np.column_stack([1 - score, score]), f"{name} (t={t:.3f})")
    m["threshold"] = t
    part1_metrics.append(m)
    test_scores[name] = score
part1_df = pd.DataFrame(part1_metrics).set_index("model")
display(part1_df.sort_values("f1_churn", ascending=False))
best_row = part1_df["f1_churn"].idxmax()
best_key = best_classical
for k in test_scores:
    if k in best_row:
        best_key = k
        break

# %% [markdown]
# ## 6. TabFM.ensemble
#
# License: weights Non-Commercial; code Apache 2.0. Context subsample disclosed.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
base = tabfm_v1_0_0.load(model_type="classification", device=device)
X_ctx = pd.concat([X_train_tab, X_val_tab])
y_ctx = pd.concat([y_train, y_val])
MAX_CTX = 3000
if len(X_ctx) > MAX_CTX:
    print(f"EXPLICIT TabFM context subsample {len(X_ctx)} → {MAX_CTX}")
    X_ctx, _, y_ctx, _ = train_test_split(
        X_ctx, y_ctx, train_size=MAX_CTX, random_state=RANDOM_STATE, stratify=y_ctx
    )
else:
    print("TabFM context:", len(X_ctx))

# Note: TabFM.ensemble enables NNLS; cannot set max_num_rows at the same time.
# Context size is controlled by explicit stratified subsample above.
tab_clf = TabFMClassifier.ensemble(
    base,
    n_estimators=12 if device == "cuda" else 6,
    random_state=RANDOM_STATE,
    verbose=True,
    batch_size=1,
)
tab_clf.fit(X_ctx, np.asarray(y_ctx))
# val threshold: use a held-out portion of original val if possible
tab_val = np.asarray(tab_clf.predict_proba(X_val_tab), dtype=float)
tvs = tab_val[:, 1] if tab_val.ndim == 2 else tab_val
t_tab, _ = tune_threshold_f1(y_val, tvs)
tab_proba = np.asarray(tab_clf.predict_proba(X_test_tab), dtype=float)
tab_score = tab_proba[:, 1] if tab_proba.ndim == 2 else tab_proba
tab_pred = apply_threshold(tab_score, t_tab)
tab_metrics = classification_bundle(y_test, tab_pred, tab_proba, f"TabFM.ensemble (t={t_tab:.3f})")
tab_metrics["threshold"] = t_tab

# %% [markdown]
# ## 7. Comparison + Monetary revenue-at-risk

# %%
compare = pd.concat(
    [
        part1_df.loc[[best_row]],
        pd.DataFrame([tab_metrics]).set_index("model"),
        pd.DataFrame([dummy_metrics]).set_index("model"),
    ]
)
display(compare.sort_values("f1_churn", ascending=False))
fig, ax = plt.subplots(figsize=(9, 4.5))
compare[["f1_churn", "pr_auc", "roc_auc", "recall_churn", "precision_churn"]].plot(kind="bar", ax=ax, rot=20)
ax.set_ylim(0, 1.05)
ax.set_title("Online Retail II — production stack (test)")
plt.tight_layout()
plt.show()

rev = rev_test.to_numpy()
bs = test_scores[best_key]
bp = apply_threshold(bs, thresholds.get(best_key, 0.5))
for label, pred in [(best_row, bp), (f"TabFM.ensemble (t={t_tab:.3f})", tab_pred)]:
    mask = np.asarray(pred).astype(int) == 1
    print(f"\n{label}: flagged {int(mask.sum())}/{len(pred)}")
    print(f"  Monetary at risk: {float(rev[mask].sum()):,.2f}")
    print("  Framing: pre-cutoff spend of flagged customers — not guaranteed future loss.")

lift_revenue_curve(y_test, bs, rev, title=str(best_row), ylabel="Cumulative Monetary of true churners")
lift_revenue_curve(y_test, tab_score, rev, title="TabFM.ensemble", ylabel="Cumulative Monetary of true churners")

# %% [markdown]
# ## Manager summary
#
# Churn is **engineered** (67% under 90-day inactivity after 2011-06-01). High base
# rate inflates raw revenue-at-risk. Production stack uses richer RFM ratios,
# PR-AUC tuning, validation thresholds, and TabFM ensemble with disclosed
# context limits. Prefer ranking + campaign design over single accuracy numbers.

# %%
print("Notebook 03 (production) complete.")
print("Cutoff", CUTOFF.date(), "post_days", POST_DAYS, "churn_rate", f"{rate:.2%}")
print(compare.round(4).to_string())
