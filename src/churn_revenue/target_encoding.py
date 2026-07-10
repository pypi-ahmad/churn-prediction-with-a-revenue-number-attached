"""Out-of-fold target encoding for high-cardinality categoricals.

Why: one-hot blows up dimensionality; naive target encoding leaks the label
into features. K-fold OOF encoding uses only other folds' means for training
rows, then global means for transform on val/test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Leakage-safe target encoding for categorical columns."""

    def __init__(
        self,
        cols: list[str] | None = None,
        *,
        n_splits: int = 5,
        smoothing: float = 20.0,
        random_state: int = 42,
        min_samples_leaf: int = 5,
    ):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: pd.DataFrame, y: Any = None):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y).astype(float)
        self.cols_ = self.cols or [
            c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"
            or X[c].dtype.name in ("string", "str")
        ]
        # also allow user-specified already-object-like columns by name
        self.cols_ = [c for c in self.cols_ if c in X.columns]
        self.global_mean_ = float(np.mean(y))
        self.maps_: dict[str, dict[Any, float]] = {}
        self.counts_: dict[str, dict[Any, int]] = {}

        for col in self.cols_:
            tmp = pd.DataFrame({"k": X[col].astype(str).fillna("__NA__"), "y": y})
            agg = tmp.groupby("k")["y"].agg(["mean", "count"])
            smooth = (
                (agg["count"] * agg["mean"] + self.smoothing * self.global_mean_)
                / (agg["count"] + self.smoothing)
            )
            # suppress rare levels toward global mean
            rare = agg["count"] < self.min_samples_leaf
            smooth = smooth.mask(rare, self.global_mean_)
            self.maps_[col] = smooth.to_dict()
            self.counts_[col] = agg["count"].to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        out = X.copy()
        for col in self.cols_:
            keys = X[col].astype(str).fillna("__NA__")
            mapped = keys.map(self.maps_[col]).astype(float)
            out[f"{col}__te"] = mapped.fillna(self.global_mean_)
        return out

    def fit_transform_oof(self, X: pd.DataFrame, y: Any) -> pd.DataFrame:
        """Return frame with OOF target encodings for train rows (no leakage)."""
        X = pd.DataFrame(X).copy()
        y = np.asarray(y).astype(int)
        cols = self.cols or [
            c
            for c in X.columns
            if X[c].dtype == object
            or str(X[c].dtype) == "category"
            or X[c].dtype.name in ("string", "str")
        ]
        cols = [c for c in cols if c in X.columns]
        self.cols_ = cols
        self.global_mean_ = float(np.mean(y))
        out = X.copy()
        for col in cols:
            oof = np.full(len(X), np.nan, dtype=float)
            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            keys = X[col].astype(str).fillna("__NA__").to_numpy()
            for tr, va in skf.split(X, y):
                tmp = pd.DataFrame({"k": keys[tr], "y": y[tr]})
                agg = tmp.groupby("k")["y"].agg(["mean", "count"])
                smooth = (
                    (agg["count"] * agg["mean"] + self.smoothing * self.global_mean_)
                    / (agg["count"] + self.smoothing)
                )
                rare = agg["count"] < self.min_samples_leaf
                smooth = smooth.mask(rare, self.global_mean_)
                mp = smooth.to_dict()
                oof[va] = pd.Series(keys[va]).map(mp).astype(float).to_numpy()
            # leftovers → global
            oof = np.where(np.isnan(oof), self.global_mean_, oof)
            out[f"{col}__te"] = oof
        # fit global maps for later transform (val/test)
        self.fit(X, y)
        return out
