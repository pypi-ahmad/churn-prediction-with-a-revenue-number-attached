"""Production-oriented helpers for churn prediction notebooks."""

from churn_revenue.metrics import classification_bundle, evaluate_scores, lift_revenue_curve
from churn_revenue.threshold import tune_threshold_f1, apply_threshold
from churn_revenue.modeling import (
    RANDOM_STATE,
    build_boosting_candidates,
    tune_model,
    soft_vote_proba,
    fit_calibrated_isotonic,
    production_classical_stack,
)

__all__ = [
    "RANDOM_STATE",
    "classification_bundle",
    "evaluate_scores",
    "lift_revenue_curve",
    "tune_threshold_f1",
    "apply_threshold",
    "build_boosting_candidates",
    "tune_model",
    "soft_vote_proba",
    "fit_calibrated_isotonic",
    "production_classical_stack",
]
