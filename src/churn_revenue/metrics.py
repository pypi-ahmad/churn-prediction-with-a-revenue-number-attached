"""Evaluation and business reporting utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def evaluate_scores(
    y_true: Any,
    y_score: Any,
    *,
    threshold: float = 0.5,
    title: str = "model",
) -> dict[str, float | str]:
    """Compute ranking + thresholded metrics for binary churn."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, float | str] = {
        "model": title,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_churn": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_churn": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_churn": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
    # Calibration-sensitive scores (need both classes present)
    if len(np.unique(y_true)) > 1:
        try:
            metrics["brier"] = float(brier_score_loss(y_true, y_score))
            metrics["log_loss"] = float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6)))
        except ValueError:
            metrics["brier"] = float("nan")
            metrics["log_loss"] = float("nan")
    return metrics


def classification_bundle(
    y_true: Any,
    y_pred: Any,
    y_proba: Any,
    title: str,
    *,
    show_plots: bool = True,
) -> dict[str, float | str]:
    """Full report suite: print + optional plots; return metrics dict."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_score = y_proba[:, 1] if y_proba.ndim == 2 else y_proba

    # recover threshold approx from pred if mixed
    metrics = evaluate_scores(y_true, y_score, threshold=0.5, title=title)
    # overwrite thresholded metrics with provided preds (may use tuned threshold)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision_churn"] = float(
        precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    metrics["recall_churn"] = float(
        recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    metrics["f1_churn"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, digits=3))
    print(
        f"ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
        f"F1(churn)={metrics['f1_churn']:.4f} | Brier={metrics.get('brier', float('nan')):.4f}"
    )

    if show_plots:
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


def lift_revenue_curve(
    y_true: Any,
    y_score: Any,
    revenue: Any,
    title: str,
    *,
    ylabel: str = "Cumulative revenue of true churners",
    n_bins: int = 20,
) -> float:
    """Contact prioritization: rank by score, accumulate true-churn revenue."""
    order = np.argsort(-np.asarray(y_score, dtype=float))
    y_sorted = np.asarray(y_true)[order]
    rev_sorted = np.asarray(revenue, dtype=float)[order]
    true_churn_rev = rev_sorted * (y_sorted == 1)
    cum_rev = np.cumsum(true_churn_rev)
    cum_n = np.arange(1, len(y_sorted) + 1)
    idx = np.linspace(0, len(y_sorted) - 1, num=min(n_bins * 5, len(y_sorted)), dtype=int)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cum_n[idx], cum_rev[idx], label=title, lw=2)
    total = float(true_churn_rev.sum())
    ax.plot(
        cum_n[idx],
        total * (cum_n[idx] / len(y_sorted)),
        "--",
        color="gray",
        label="Random order",
    )
    ax.set_xlabel("Customers contacted (ranked by predicted churn prob)")
    ax.set_ylabel(ylabel)
    ax.set_title("Retention prioritization: value captured vs outreach volume")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return total
