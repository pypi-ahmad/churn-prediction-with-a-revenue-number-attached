"""Segment-level error analysis for churn models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score, recall_score


def segment_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    segments: pd.Series,
    *,
    threshold: float = 0.5,
    min_count: int = 30,
) -> pd.DataFrame:
    """PR-AUC / F1 / recall by segment (e.g. Contract, Monetary quartile)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    pred = (s >= threshold).astype(int)
    seg = segments.reset_index(drop=True)
    rows = []
    for name, idx in seg.groupby(seg).groups.items():
        idx = list(idx)
        if len(idx) < min_count:
            continue
        yt, ys, yp = y[idx], s[idx], pred[idx]
        if len(np.unique(yt)) < 2:
            pr = float("nan")
        else:
            pr = float(average_precision_score(yt, ys))
        rows.append(
            {
                "segment": name,
                "n": len(idx),
                "churn_rate": float(yt.mean()),
                "pr_auc": pr,
                "f1_churn": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
                "recall_churn": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)
