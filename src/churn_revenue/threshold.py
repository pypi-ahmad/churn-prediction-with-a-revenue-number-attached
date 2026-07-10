"""Decision threshold selection for imbalanced churn."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def apply_threshold(y_score: Any, threshold: float) -> np.ndarray:
    """Map scores to hard labels."""
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    return (y_score >= threshold).astype(int)


def tune_threshold_f1(
    y_true: Any,
    y_score: Any,
    *,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Pick threshold maximizing F1 on the positive (churn) class.

    Production practice for imbalanced classification: train with log-loss /
    ranking objectives, then *move the threshold* on a validation set instead of
    trusting 0.5 (Machine Learning Mastery; industry churn guides).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]

    if grid is None:
        # denser around extremes for skewed score distributions
        grid = np.unique(
            np.concatenate(
                [
                    np.linspace(0.05, 0.95, 37),
                    np.quantile(y_score, np.linspace(0.05, 0.95, 19)),
                ]
            )
        )

    best_t = 0.5
    best_f1 = -1.0
    best_row: dict[str, float] = {}
    for t in grid:
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
            best_row = {
                "threshold": best_t,
                "f1_churn": best_f1,
                "precision_churn": float(
                    precision_score(y_true, pred, pos_label=1, zero_division=0)
                ),
                "recall_churn": float(
                    recall_score(y_true, pred, pos_label=1, zero_division=0)
                ),
            }
    return best_t, best_row
