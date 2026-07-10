"""Nested / repeated CV reporting for honest uncertainty.

Why: a single holdout can flatter a lucky seed. Repeated stratified folds
give mean ± std of PR-AUC / F1 for portfolio credibility.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from churn_revenue.threshold import tune_threshold_f1, apply_threshold


def repeated_stratified_metrics(
    estimator: Any,
    X: Any,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    tune_threshold: bool = True,
) -> pd.DataFrame:
    """Fit clone(estimator) on each fold train; score on fold test.

    If tune_threshold, split fold-train further 80/20 for threshold selection
    (cheap approximation to nested CV).
    """
    y = np.asarray(y).astype(int)
    rows = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + rep
        )
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            est = clone(estimator)
            if hasattr(X, "iloc"):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
            else:
                X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]

            thr = 0.5
            if tune_threshold and len(tr) > 50:
                # inner split for threshold
                inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
                # use last inner fold as threshold val
                inner_splits = list(inner.split(np.zeros(len(y_tr)), y_tr))
                itr, iva = inner_splits[0]
                if hasattr(X_tr, "iloc"):
                    est.fit(X_tr.iloc[itr], y_tr[itr])
                    p_va = est.predict_proba(X_tr.iloc[iva])[:, 1]
                else:
                    est.fit(X_tr[itr], y_tr[itr])
                    p_va = est.predict_proba(X_tr[iva])[:, 1]
                thr, _ = tune_threshold_f1(y_tr[iva], p_va)
                # refit on full fold train
                est = clone(estimator)
                est.fit(X_tr, y_tr)
            else:
                est.fit(X_tr, y_tr)

            p = est.predict_proba(X_te)[:, 1]
            pred = apply_threshold(p, thr)
            rows.append(
                {
                    "repeat": rep,
                    "fold": fold,
                    "threshold": thr,
                    "pr_auc": float(average_precision_score(y_te, p)),
                    "roc_auc": float(roc_auc_score(y_te, p)),
                    "f1_churn": float(f1_score(y_te, pred, pos_label=1, zero_division=0)),
                }
            )
    return pd.DataFrame(rows)


def summarize_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std table for README."""
    metrics = ["pr_auc", "roc_auc", "f1_churn"]
    out = []
    for m in metrics:
        out.append(
            {
                "metric": m,
                "mean": float(df[m].mean()),
                "std": float(df[m].std(ddof=1)),
                "min": float(df[m].min()),
                "max": float(df[m].max()),
            }
        )
    return pd.DataFrame(out)
