"""Production-oriented helpers for churn prediction notebooks (v2 + v3)."""

from churn_revenue.metrics import classification_bundle, evaluate_scores, lift_revenue_curve
from churn_revenue.threshold import tune_threshold_f1, apply_threshold
from churn_revenue.modeling import (
    RANDOM_STATE,
    build_boosting_candidates,
    tune_model,
    soft_vote_proba,
    fit_calibrated_isotonic,
    production_classical_stack,
    predict_proba_matrix,
)
from churn_revenue.value_policy import (
    expected_contact_value,
    policy_top_k_fraction,
    evaluate_policy,
    sweep_top_k,
    best_top_k_on_validation,
)
from churn_revenue.hybrid import oof_predict_proba, fit_hybrid_meta, hybrid_predict_proba
from churn_revenue.target_encoding import OutOfFoldTargetEncoder
from churn_revenue.nested_cv import repeated_stratified_metrics, summarize_cv
from churn_revenue.segment_report import segment_metrics
from churn_revenue.multiwindow_rfm import (
    build_customer_features,
    hazard_churn_label,
    multi_horizon_labels,
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
    "predict_proba_matrix",
    "expected_contact_value",
    "policy_top_k_fraction",
    "evaluate_policy",
    "sweep_top_k",
    "best_top_k_on_validation",
    "oof_predict_proba",
    "fit_hybrid_meta",
    "hybrid_predict_proba",
    "OutOfFoldTargetEncoder",
    "repeated_stratified_metrics",
    "summarize_cv",
    "segment_metrics",
    "build_customer_features",
    "hazard_churn_label",
    "multi_horizon_labels",
]
