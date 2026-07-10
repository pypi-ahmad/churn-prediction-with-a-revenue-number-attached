"""Hybrid TabFM + GBM stacking.

Why: gradient-boosted trees and tabular foundation models make different
errors. A logistic (or light GBM) meta-learner on out-of-fold probabilities
often beats either alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def oof_predict_proba(
    estimator: Any,
    X: np.ndarray | Any,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Out-of-fold P(y=1) for a sklearn-like classifier."""
    y = np.asarray(y).astype(int)
    n = len(y)
    oof = np.zeros(n, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in skf.split(np.zeros(n), y):
        est = clone(estimator)
        # support DataFrame row indexing
        if hasattr(X, "iloc"):
            est.fit(X.iloc[tr], y[tr])
            p = est.predict_proba(X.iloc[va])
        else:
            est.fit(X[tr], y[tr])
            p = est.predict_proba(X[va])
        oof[va] = p[:, 1] if np.ndim(p) == 2 else p
    return oof


def fit_hybrid_meta(
    p_gbm_oof: np.ndarray,
    p_tabfm_oof: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit logistic meta-learner on stacked OOF probabilities."""
    X_meta = np.column_stack(
        [
            np.asarray(p_gbm_oof, dtype=float),
            np.asarray(p_tabfm_oof, dtype=float),
            np.asarray(p_gbm_oof, dtype=float) * np.asarray(p_tabfm_oof, dtype=float),
        ]
    )
    meta = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    meta.fit(X_meta, np.asarray(y).astype(int))
    return meta


def hybrid_predict_proba(
    meta: LogisticRegression,
    p_gbm: np.ndarray,
    p_tabfm: np.ndarray,
) -> np.ndarray:
    """P(churn) from fitted meta-learner."""
    X_meta = np.column_stack(
        [
            np.asarray(p_gbm, dtype=float),
            np.asarray(p_tabfm, dtype=float),
            np.asarray(p_gbm, dtype=float) * np.asarray(p_tabfm, dtype=float),
        ]
    )
    return meta.predict_proba(X_meta)[:, 1]
