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
# # Online Retail II — Engineered Churn + Monetary Revenue
#
# **Why this notebook is different.** The UCI Online Retail II dataset is
# **transactional line items**, not a pre-labeled churn table. We engineer a
# defensible churn label from **RFM** (Recency, Frequency, Monetary), then run
# the same two-part modeling story as notebooks 01–02: LazyPredict top-3
# properly tuned, Google TabFM, and a revenue number attached via `Monetary`.
#
# **Business framing.** For a retailer, "churn" means a once-active buyer who
# goes quiet. Predicting that risk lets retention teams prioritize win-back
# spend against customers whose **historical spend** (Monetary) is large.

# %% [markdown]
# ## 1. Setup

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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.to_string())
        else:
            print(obj)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (8, 4.5)

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("lazypredict:", lazypredict.__version__)
print("tabfm:", tabfm.__version__)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Kernel intent: churn-revenue-project")
print("Seed:", RANDOM_STATE)

# %% [markdown]
# ## 2. Data acquisition — UCI Online Retail II (id 502)
#
# Spec path: `ucimlrepo.fetch_ucirepo(id=502)`. We try that first. On this
# environment, UCI marks the dataset as **not available for Python import**
# via `ucimlrepo`, so we fall back to the official UCI static zip of the same
# dataset (CC BY 4.0) and load both Excel sheets (2009–2010 and 2010–2011).
#
# We do **not** invent a third-party mirror — only the UCI archive file.

# %%
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ucimlrepo_status = None
tx = None

try:
    from ucimlrepo import fetch_ucirepo

    retail = fetch_ucirepo(id=502)
    X = retail.data.features
    y = retail.data.targets
    print("ucimlrepo fetch succeeded")
    print("features shape:", getattr(X, "shape", None), "targets:", type(y), getattr(y, "shape", None))
    tx = X.copy() if y is None or (hasattr(y, "empty") and y.empty) else pd.concat([X, y], axis=1)
    ucimlrepo_status = "ok"
except Exception as e:
    ucimlrepo_status = f"FAILED: {type(e).__name__}: {e}"
    print("ucimlrepo path:", ucimlrepo_status)
    print("Falling back to official UCI zip:", UCI_ZIP_URL)
    resp = requests.get(UCI_ZIP_URL, timeout=180)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        print("Zip contents:", zf.namelist())
        with zf.open("online_retail_II.xlsx") as fh:
            xl = pd.ExcelFile(fh)
            print("Sheets:", xl.sheet_names)
            frames = [xl.parse(s) for s in xl.sheet_names]
            tx = pd.concat(frames, ignore_index=True)

print("Raw transaction shape:", tx.shape)
print("Columns:", list(tx.columns))
print(tx.dtypes)
display(tx.head())
print("InvoiceDate range:", pd.to_datetime(tx["InvoiceDate"]).min(), "→", pd.to_datetime(tx["InvoiceDate"]).max())

# %% [markdown]
# ## 3. Cleaning line items
#
# Retail exports need hygiene before RFM:
# 1. Drop cancelled invoices (`Invoice` starting with `C`)
# 2. Drop missing `Customer ID`
# 3. Drop non-positive `Quantity` or `Price`
# 4. Parse dates

# %%
raw_n = len(tx)
tx = tx.copy()
tx["Invoice"] = tx["Invoice"].astype(str)
tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"])

n_cancel = tx["Invoice"].str.startswith("C").sum()
tx = tx[~tx["Invoice"].str.startswith("C")]
n_no_cust = tx["Customer ID"].isna().sum()
tx = tx.dropna(subset=["Customer ID"])
n_nonpos = ((tx["Quantity"] <= 0) | (tx["Price"] <= 0)).sum()
tx = tx[(tx["Quantity"] > 0) & (tx["Price"] > 0)]
tx["Customer ID"] = tx["Customer ID"].astype(int)
tx["line_revenue"] = tx["Quantity"] * tx["Price"]

print(f"Raw rows: {raw_n:,}")
print(f"Dropped cancellations: {n_cancel:,}")
print(f"Dropped missing Customer ID: {n_no_cust:,}")
print(f"Dropped non-positive qty/price: {n_nonpos:,}")
print(f"Clean rows: {len(tx):,}")
print("Clean date range:", tx["InvoiceDate"].min(), "→", tx["InvoiceDate"].max())
print("Unique customers:", tx["Customer ID"].nunique())

# %% [markdown]
# ## 4. EDA on transactions (before RFM)

# %%
print(tx[["Quantity", "Price", "line_revenue"]].describe().T)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col in zip(axes, ["Quantity", "Price", "line_revenue"]):
    # log1p for heavy tails
    sns.histplot(np.log1p(tx[col].clip(lower=0)), bins=40, ax=ax, color="#4C78A8")
    ax.set_title(f"log1p({col})")
plt.suptitle("Transaction value distributions (log scale)", y=1.02)
plt.tight_layout()
plt.show()

monthly = tx.set_index("InvoiceDate").resample("MS")["line_revenue"].sum()
fig, ax = plt.subplots(figsize=(10, 3.5))
monthly.plot(ax=ax, color="#F58518")
ax.set_title("Monthly revenue (clean transactions)")
ax.set_ylabel("Revenue")
plt.tight_layout()
plt.show()
print("Top countries by rows:")
print(tx["Country"].value_counts().head(10))

# %% [markdown]
# **Interpretation.** Line-item quantities and prices are heavy-tailed (bulk
# orders and occasional high unit prices). Monthly revenue shows seasonality —
# useful context when we pick a cutoff date for the churn label.

# %% [markdown]
# ## 5. Label engineering — RFM + 90-day inactivity
#
# ### Modeling assumption (explicit; not silently retuned)
#
# | Choice | Value |
# |--------|-------|
# | Cutoff date | **2011-06-01** (mid/late in Dec 2009–Dec 2011 window) |
# | Feature window | all purchases with `InvoiceDate < cutoff` |
# | Churn observation window | `[cutoff, cutoff + 90 days)` |
# | Label | `churn = 1` if customer has **zero** invoices in the post window |
#
# Features (Recency / Frequency / Monetary / extras) use **only pre-cutoff**
# activity — no post-cutoff leakage.
#
# If the resulting churn rate were near 0% or 100%, we would stop and propose
# a different cutoff/window rather than quietly shopping definitions.

# %%
CUTOFF = pd.Timestamp("2011-06-01")
POST_DAYS = 90
post_end = CUTOFF + pd.Timedelta(days=POST_DAYS)

pre = tx[tx["InvoiceDate"] < CUTOFF].copy()
post = tx[(tx["InvoiceDate"] >= CUTOFF) & (tx["InvoiceDate"] < post_end)].copy()
print(f"Pre rows: {len(pre):,} | Post rows ({POST_DAYS}d): {len(post):,}")

country_mode = (
    pre.groupby("Customer ID")["Country"]
    .agg(lambda s: s.value_counts().index[0])
    .rename("Country")
)

cust = pre.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda s: (CUTOFF - s.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("line_revenue", "sum"),
    n_products=("StockCode", "nunique"),
    n_items=("Quantity", "sum"),
    tenure_days=("InvoiceDate", lambda s: (s.max() - s.min()).days),
    avg_line_value=("line_revenue", "mean"),
)
cust = cust.join(country_mode, how="left")

active_post = set(post["Customer ID"].unique())
cust["churn"] = (~cust.index.isin(active_post)).astype(int)

churn_rate = float(cust["churn"].mean())
print("Customer table shape:", cust.shape)
print("Churn value counts:\n", cust["churn"].value_counts())
print(f"Churn rate: {churn_rate:.2%}")

if churn_rate < 0.05 or churn_rate > 0.95:
    raise SystemExit(
        f"STOP: churn rate {churn_rate:.2%} is near 0/100% under cutoff={CUTOFF.date()} "
        f"window={POST_DAYS}d. Propose a different cutoff/window before modeling."
    )
if churn_rate < 0.35 or churn_rate > 0.65:
    print(
        "Note: class balance is skewed (not near 0/100, so we proceed). "
        "Imbalance-aware metrics (F1 / PR-AUC) remain required."
    )

# %% [markdown]
# ### Observed label balance (this run)
#
# The printed churn rate above is the **ground truth we engineered** for this
# cutoff/window. A different cutoff would be a legitimate alternative study —
# we keep this one fixed for the rest of the notebook.

# %%
fig, ax = plt.subplots()
cust["churn"].value_counts().sort_index().plot(kind="bar", color=["#4C78A8", "#E45756"], ax=ax, rot=0)
ax.set_title(f"Engineered churn balance (rate={churn_rate:.1%})")
ax.set_xlabel("churn")
ax.set_ylabel("customers")
plt.tight_layout()
plt.show()

# RFM EDA
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
    sns.histplot(cust[col], bins=40, ax=ax, color="#54A24B")
    ax.set_title(col)
plt.suptitle("RFM feature distributions (customer level)", y=1.02)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
    sns.boxplot(data=cust, x="churn", y=col, ax=ax)
    if col == "Monetary":
        ax.set_yscale("log")
    ax.set_title(f"{col} by churn")
plt.tight_layout()
plt.show()

print("Mean RFM by churn:")
display(cust.groupby("churn")[["Recency", "Frequency", "Monetary"]].mean())

corr = cust.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("Customer-level correlation heatmap")
plt.tight_layout()
plt.show()

print("IQR outlier rates on Monetary/Frequency:")
for col in ["Monetary", "Frequency", "Recency"]:
    q1, q3 = cust[col].quantile(0.25), cust[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rate = ((cust[col] < lo) | (cust[col] > hi)).mean()
    print(f"  {col}: {rate:.1%}")

# %% [markdown]
# **Interpretation.** High Recency (long time since last purchase) should
# associate with higher engineered churn; high Frequency/Monetary with
# retention. That is the RFM intuition we are testing — if boxplots disagree
# strongly, the label definition may be misaligned with buyer behavior.

# %% [markdown]
# ## 6. Train/test split (customer level)
#
# Same stratified split for Part 1 and Part 2. `Monetary` is both a model
# feature (pre-cutoff spend) and the Part-9 revenue anchor.

# %%
feature_cols = [
    "Recency",
    "Frequency",
    "Monetary",
    "n_products",
    "n_items",
    "tenure_days",
    "avg_line_value",
    "Country",
]
X_tab = cust[feature_cols].copy()
y = cust["churn"].astype(int)
revenue_anchor = cust["Monetary"].copy()

print(X_tab.dtypes)
print("Shape:", X_tab.shape, "churn rate:", f"{y.mean():.2%}")

X_train_tab, X_test_tab, y_train, y_test, rev_train, rev_test = train_test_split(
    X_tab,
    y,
    revenue_anchor,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)
print("Train:", X_train_tab.shape, "Test:", X_test_tab.shape)
print("Train churn:", f"{y_train.mean():.2%}", "Test churn:", f"{y_test.mean():.2%}")

# Sklearn path: encode Country
X_train_sk = X_train_tab.copy()
X_test_sk = X_test_tab.copy()
num_features = [c for c in feature_cols if c != "Country"]
cat_features = ["Country"]

preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ]
)
X_train_enc = preprocess.fit_transform(X_train_sk)
X_test_enc = preprocess.transform(X_test_sk)
try:
    enc_names = preprocess.get_feature_names_out()
except Exception:
    enc_names = [f"f{i}" for i in range(X_train_enc.shape[1])]
X_train_lp = pd.DataFrame(X_train_enc, columns=enc_names, index=X_train_sk.index)
X_test_lp = pd.DataFrame(X_test_enc, columns=enc_names, index=X_test_sk.index)
print("Encoded shape:", X_train_lp.shape)

# %% [markdown]
# ## 7. Evaluation helpers

# %%
def classification_bundle(y_true, y_pred, y_proba, title: str) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_score = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    metrics = {
        "model": title,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_churn": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_churn": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_churn": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }
    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, digits=3))
    print(
        f"ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
        f"F1(churn)={metrics['f1_churn']:.4f}"
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title(f"Confusion — {title}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[1], name=title)
    axes[1].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axes[2], name=title)
    axes[2].set_title("Precision–Recall")
    plt.tight_layout()
    plt.show()
    return metrics


def lift_revenue_curve(y_true, y_score, revenue, title: str, n_bins: int = 20):
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]
    rev_sorted = np.asarray(revenue, dtype=float)[order]
    true_churn_rev = rev_sorted * (y_sorted == 1)
    cum_rev = np.cumsum(true_churn_rev)
    cum_n = np.arange(1, len(y_sorted) + 1)
    idx = np.linspace(0, len(y_sorted) - 1, num=min(n_bins * 5, len(y_sorted)), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cum_n[idx], cum_rev[idx], label=title, lw=2)
    total = true_churn_rev.sum()
    ax.plot(cum_n[idx], total * (cum_n[idx] / len(y_sorted)), "--", color="gray", label="Random order")
    ax.set_xlabel("Customers contacted (by predicted churn prob)")
    ax.set_ylabel("Cumulative pre-cutoff Monetary of true churners")
    ax.set_title("Win-back prioritization: monetary value vs outreach volume")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return float(total)

# %% [markdown]
# ## 8. Part 1 — LazyPredict → tune top 3
#
# Ranking metric: **F1 Score** (fallback Balanced Accuracy / ROC AUC).

# %%
lazy = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    predictions=False,
    random_state=RANDOM_STATE,
    classifiers="all",
)
models_df, _ = lazy.fit(X_train_lp, X_test_lp, y_train, y_test)
print("Leaderboard columns:", list(models_df.columns))
display(models_df)

rank_col = None
for candidate in ["F1 Score", "F1", "Balanced Accuracy", "ROC AUC", "Accuracy"]:
    if candidate in models_df.columns:
        rank_col = candidate
        break
leaderboard = models_df.sort_values(rank_col, ascending=False)
top3_names = list(leaderboard.head(3).index)
print(f"Ranking metric: {rank_col}")
print("Top 3:", top3_names)
display(leaderboard.head(10))

# %%
MODEL_ZOO = {
    "XGBClassifier": (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=2, tree_method="hist"),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        },
    ),
    "LGBMClassifier": (
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=2),
        {
            "n_estimators": [100, 200, 400],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        },
    ),
    "RandomForestClassifier": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=2, class_weight="balanced"),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ),
    "ExtraTreesClassifier": (
        ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=2, class_weight="balanced"),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ),
    "GradientBoostingClassifier": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 4]},
    ),
    "HistGradientBoostingClassifier": (
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {"max_iter": [100, 200, 300], "learning_rate": [0.05, 0.1], "max_depth": [None, 4, 8]},
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
        {"C": np.logspace(-2, 2, 8)},
    ),
    "KNeighborsClassifier": (
        KNeighborsClassifier(),
        {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]},
    ),
    "SVC": (
        SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    ),
    "AdaBoostClassifier": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.5, 1.0]},
    ),
    "BaggingClassifier": (
        BaggingClassifier(random_state=RANDOM_STATE, n_jobs=2),
        {"n_estimators": [20, 50, 100], "max_samples": [0.5, 0.8, 1.0]},
    ),
    "DecisionTreeClassifier": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 5]},
    ),
    "SGDClassifier": (
        SGDClassifier(loss="log_loss", random_state=RANDOM_STATE, class_weight="balanced"),
        {"alpha": np.logspace(-5, -2, 6)},
    ),
    "LinearDiscriminantAnalysis": (LinearDiscriminantAnalysis(), {"solver": ["svd", "lsqr"]}),
    "QuadraticDiscriminantAnalysis": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.0, 0.1, 0.5]}),
    "GaussianNB": (GaussianNB(), {"var_smoothing": np.logspace(-11, -7, 5)}),
    "RidgeClassifier": (
        RidgeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {"alpha": np.logspace(-2, 2, 8)},
    ),
    "LinearSVC": (
        CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, class_weight="balanced", max_iter=5000)
        ),
        {"estimator__C": [0.1, 1.0, 10.0]},
    ),
    "CalibratedClassifierCV": (
        CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        {},
    ),
}


def resolve_model(name: str):
    if name in MODEL_ZOO:
        return name, MODEL_ZOO[name]
    lower_map = {k.lower(): k for k in MODEL_ZOO}
    if name.lower() in lower_map:
        key = lower_map[name.lower()]
        return key, MODEL_ZOO[key]
    print(f"WARNING: no grid for {name!r}; using RandomForestClassifier")
    return "RandomForestClassifier", MODEL_ZOO["RandomForestClassifier"]


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
part1_metrics = []
part1_models = {}
part1_probas = {}

for raw_name in top3_names:
    zoo_name, (estimator, param_grid) = resolve_model(raw_name)
    print(f"\n### Tuning {raw_name} → {zoo_name}")
    if param_grid:
        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=min(20, max(5, len(param_grid) * 3)),
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=2,
            refit=True,
            verbose=0,
        )
        search.fit(X_train_lp, y_train)
        best = search.best_estimator_
        print("Best params:", search.best_params_)
        print("Best CV AP:", f"{search.best_score_:.4f}")
    else:
        best = estimator.fit(X_train_lp, y_train)

    if hasattr(best, "predict_proba"):
        proba = best.predict_proba(X_test_lp)
        pred = best.predict(X_test_lp)
    elif hasattr(best, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler

        scores = best.decision_function(X_test_lp)
        y_score = MinMaxScaler().fit_transform(np.asarray(scores).reshape(-1, 1)).ravel()
        proba = np.column_stack([1 - y_score, y_score])
        pred = best.predict(X_test_lp)
    else:
        pred = best.predict(X_test_lp)
        proba = np.column_stack([1 - pred, pred])

    m = classification_bundle(y_test, pred, proba, title=f"Tuned {zoo_name}")
    part1_metrics.append(m)
    part1_models[zoo_name] = best
    part1_probas[zoo_name] = proba[:, 1] if proba.ndim == 2 else proba

part1_df = pd.DataFrame(part1_metrics).set_index("model")
display(part1_df.sort_values("pr_auc", ascending=False))

fig, ax = plt.subplots(figsize=(8, 4))
part1_df[["f1_churn", "pr_auc", "roc_auc"]].plot(kind="bar", ax=ax, rot=15)
ax.set_title("Part 1 — tuned top-3 (test)")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

best_part1_name = part1_df["pr_auc"].idxmax()
best_part1_key = list(part1_probas.keys())[list(part1_df.index).index(best_part1_name)]
best_part1_proba = part1_probas[best_part1_key]
best_part1_pred = part1_models[best_part1_key].predict(X_test_lp)
print("Best Part-1 by PR-AUC:", best_part1_name)

# %% [markdown]
# ## 9. Part 2 — Google TabFM
#
# ### License flag
#
# - **Weights:** TabFM Non-Commercial License v1.0
#   (https://huggingface.co/google/tabfm-1.0.0-pytorch)
# - **Code:** Apache 2.0
#
# Review before commercial reuse; no legal opinion offered.
#
# Memory scales with training context size. Customer table is ~5k rows —
# full train context is used unless OOM forces an explicit stratified subsample.

# %%
tabfm_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TabFM on device={tabfm_device}...")
tabfm_base = tabfm_v1_0_0.load(model_type="classification", device=tabfm_device)
tab_clf = TabFMClassifier(model=tabfm_base)

MAX_TABFM_CONTEXT = 3000
n_train = len(X_train_tab)
if n_train > MAX_TABFM_CONTEXT:
    print(
        f"EXPLICIT subsample: TabFM context {n_train} → {MAX_TABFM_CONTEXT} "
        f"(stratified; Part 1 still uses full train)"
    )
    X_ctx, _, y_ctx, _ = train_test_split(
        X_train_tab,
        y_train,
        train_size=MAX_TABFM_CONTEXT,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
else:
    X_ctx, y_ctx = X_train_tab, y_train
    print(f"TabFM context rows: {n_train} (full train; no subsample)")

tab_clf.fit(X_ctx, np.asarray(y_ctx))
tab_pred = np.asarray(tab_clf.predict(X_test_tab)).astype(int)
tab_proba = tab_clf.predict_proba(X_test_tab)
tab_metrics = classification_bundle(y_test, tab_pred, tab_proba, title="TabFM v1.0.0")

# %% [markdown]
# ## 10. Final comparison + Monetary revenue-at-risk
#
# Revenue anchor = pre-cutoff **Monetary** for customers the model flags.
# This is historical spend associated with predicted churners — **not** a
# proven forecast of future lost sales.

# %%
compare = pd.concat(
    [part1_df.loc[[best_part1_name]], pd.DataFrame([tab_metrics]).set_index("model")]
)
print("Side-by-side test metrics:")
display(compare)

fig, ax = plt.subplots(figsize=(8, 4))
compare[["f1_churn", "pr_auc", "roc_auc", "recall_churn", "precision_churn"]].plot(
    kind="bar", ax=ax, rot=15
)
ax.set_title("Best Part-1 vs TabFM")
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

rev = rev_test.to_numpy()
tab_score = tab_proba[:, 1] if np.asarray(tab_proba).ndim == 2 else np.asarray(tab_proba)

for label, pred, proba in [
    (best_part1_name, best_part1_pred, best_part1_proba),
    ("TabFM v1.0.0", tab_pred, tab_score),
]:
    mask = np.asarray(pred).astype(int) == 1
    print(f"\n{label}:")
    print(f"  Flagged: {int(mask.sum())} / {len(pred)}")
    print(f"  Revenue-at-risk (sum Monetary of flagged): {float(rev[mask].sum()):,.2f}")
    print(
        "  Framing: pre-cutoff historical spend associated with predicted churners — "
        "not guaranteed future loss, and not adjusted for win-back success rates."
    )

lift_revenue_curve(y_test, best_part1_proba, rev, title=best_part1_name)
lift_revenue_curve(y_test, tab_score, rev, title="TabFM v1.0.0")

# %% [markdown]
# ## 11. Manager summary
#
# - Online Retail II has **no native churn label**; we defined churn as
#   no purchase in the 90 days after **2011-06-01**, with RFM features built
#   only on earlier history.
# - Class balance is imperfect (see printed rate) but usable; metrics favor
#   F1 / PR-AUC.
# - LazyPredict → **tuned top 3**, then **TabFM** (non-commercial weights) on
#   the same customer split; TabFM context subsample disclosed if used.
# - Revenue attached via **Monetary** on the alert list, plus a prioritization
#   curve for limited win-back budget.
# - Next steps in production: validate the label with business, run uplift
#   tests, and monitor seasonality around the cutoff.

# %%
print("Notebook 03 complete.")
print("Cutoff:", CUTOFF.date(), "post_days:", POST_DAYS, "churn_rate:", f"{churn_rate:.2%}")
print(compare.round(4).to_string())
