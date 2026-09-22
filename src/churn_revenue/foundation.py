"""Small adapters for the foundation-model benchmarks."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churn_revenue.threshold import tune_threshold_f1


def mitra_probabilities(
    train_x: pd.DataFrame, train_y: pd.Series, eval_x: pd.DataFrame, path: Path
) -> np.ndarray:
    """Fit zero-shot Mitra and return probability of churn class 1."""
    from autogluon.tabular import TabularPredictor

    label = "__churn__"
    if len(train_x) > 500:
        train_x, _, train_y, _ = train_test_split(
            train_x, train_y, train_size=500, random_state=42, stratify=train_y
        )
    train_data = train_x.copy()
    train_data[label] = train_y.to_numpy()
    predictor = TabularPredictor(label=label, problem_type="binary", path=str(path), verbosity=1)
    predictor.fit(
        train_data,
        hyperparameters={
            "MITRA": {"hf_model": "autogluon/mitra-classifier-2", "fine_tune": False}
        },
        ag_args_fit={"ag.max_memory_usage_ratio": 1.15},
    )
    probabilities = predictor.predict_proba(eval_x, as_multiclass=True)
    if not isinstance(probabilities, pd.DataFrame):
        raise TypeError("Mitra did not return class probabilities as a DataFrame")
    if 1 not in probabilities.columns:
        raise ValueError("Mitra did not return a churn class 1 probability")
    result = probabilities[1].to_numpy(dtype=float)
    del probabilities
    del predictor
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    return result


def risk_metrics(y_true: Any, risk: Any, threshold: float, model: str) -> dict[str, float | str]:
    """Evaluate an uncalibrated churn-risk ranking without probability claims."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(risk, dtype=float)
    predicted = (scores >= threshold).astype(int)
    return {
        "model": model,
        "threshold": float(threshold),
        "precision_churn": float(precision_score(y, predicted, zero_division=0)),
        "recall_churn": float(recall_score(y, predicted, zero_division=0)),
        "f1_churn": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }


def timesfm_risk(series: list[np.ndarray], *, chunk_size: int = 4) -> np.ndarray:
    """Forecast weekly activity and return higher-is-riskier raw churn scores."""
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    if not torch.cuda.is_available():
        raise RuntimeError("TimesFM benchmark requires CUDA; torch.cuda.is_available() is false")
    model = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path="google/timesfm-3.0-pytorch",
            per_core_batch_size=1,
            device="cuda",
        )
    )
    output: list[float] = []
    for start in range(0, len(series), chunk_size):
        batch = series[start : start + chunk_size]
        try:
            forecasts = list(model.predict_batch(batch, horizon=13, return_quantiles=False))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            forecasts = [next(model.predict_batch([row], horizon=13, return_quantiles=False)) for row in batch]
        for forecast in forecasts:
            values = np.asarray(forecast.forecast, dtype=float)
            output.append(-float(np.clip(values, 0, None).sum()))
    return np.asarray(output)


def tuned_risk_metrics(y_val: Any, risk_val: Any, y_test: Any, risk_test: Any, model: str) -> dict[str, float | str]:
    """Freeze a validation-selected threshold before one test evaluation."""
    threshold, _ = tune_threshold_f1(y_val, risk_val, grid=np.unique(np.asarray(risk_val, dtype=float)))
    return risk_metrics(y_test, risk_test, threshold, model)
