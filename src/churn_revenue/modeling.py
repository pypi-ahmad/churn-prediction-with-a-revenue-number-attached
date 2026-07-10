"""Production classical model training: imbalance-aware GBMs, Optuna, voting."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    HAS_CATBOOST = False

try:
    import optuna
    from optuna.integration import OptunaSearchCV

    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    HAS_OPTUNA = False

RANDOM_STATE = 42


def _pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return float(n_neg / n_pos)


def build_boosting_candidates(y_train: np.ndarray) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Strong tabular defaults with class imbalance handling.

    Research-backed defaults for churn:
    - class_weight / scale_pos_weight first (before SMOTE)
    - gradient boosting (XGBoost, LightGBM, CatBoost, HistGB)
    - later threshold moving on validation, not 0.5
    """
    spw = _pos_weight(y_train)
    zoo: dict[str, tuple[Any, dict[str, Any]]] = {
        "LGBMClassifier": (
            LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=2,
                class_weight="balanced",
            ),
            {
                "n_estimators": [200, 400, 800],
                "num_leaves": [15, 31, 63, 127],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_samples": [10, 20, 40],
                "reg_lambda": [0.0, 0.1, 1.0, 5.0],
            },
        ),
        "XGBClassifier": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="aucpr",
                n_jobs=2,
                tree_method="hist",
                scale_pos_weight=spw,
            ),
            {
                "n_estimators": [200, 400, 800],
                "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "min_child_weight": [1, 3, 5, 10],
                "reg_lambda": [0.5, 1.0, 5.0],
                "gamma": [0.0, 0.1, 0.5],
            },
        ),
        "HistGradientBoostingClassifier": (
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                early_stopping=True,
                validation_fraction=0.1,
            ),
            {
                "max_iter": [200, 400, 600],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [None, 4, 8, 12],
                "min_samples_leaf": [10, 20, 40],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=2,
                class_weight="balanced_subsample",
            ),
            {
                "n_estimators": [200, 400, 800],
                "max_depth": [None, 8, 16, 24],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        ),
        "ExtraTreesClassifier": (
            ExtraTreesClassifier(
                random_state=RANDOM_STATE,
                n_jobs=2,
                class_weight="balanced",
            ),
            {
                "n_estimators": [200, 400, 800],
                "max_depth": [None, 8, 16, 24],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "LogisticRegression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=3000,
                            random_state=RANDOM_STATE,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            {"clf__C": np.logspace(-2, 2, 12)},
        ),
    }
    if HAS_CATBOOST:
        zoo["CatBoostClassifier"] = (
            CatBoostClassifier(
                random_seed=RANDOM_STATE,
                verbose=0,
                loss_function="Logloss",
                eval_metric="PRAUC",
                auto_class_weights="Balanced",
                thread_count=2,
            ),
            {
                "iterations": [300, 600, 1000],
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
            },
        )
    return zoo


def resolve_model(
    name: str,
    y_train: np.ndarray,
    zoo: dict[str, tuple[Any, dict[str, Any]]] | None = None,
) -> tuple[str, Any, dict[str, Any]]:
    """Map LazyPredict name → estimator + search space (exact/case-insensitive)."""
    if zoo is None:
        zoo = build_boosting_candidates(y_train)
    if name in zoo:
        est, grid = zoo[name]
        return name, est, grid
    lower = {k.lower(): k for k in zoo}
    if name.lower() in lower:
        key = lower[name.lower()]
        est, grid = zoo[key]
        return key, est, grid
    # fuzzy safe fallbacks
    for key in zoo:
        if key.lower().replace("classifier", "") in name.lower().replace("classifier", ""):
            if key == "SVC" and "Linear" in name:
                continue
            est, grid = zoo[key]
            return key, est, grid
    print(f"WARNING: no production grid for {name!r}; using LGBMClassifier")
    est, grid = zoo["LGBMClassifier"]
    return "LGBMClassifier", est, grid


def tune_model(
    estimator: Any,
    param_grid: dict[str, Any],
    X_train: Any,
    y_train: Any,
    *,
    n_iter: int = 40,
    cv_splits: int = 5,
    scoring: str = "average_precision",
    n_jobs: int = 2,
) -> Any:
    """RandomizedSearchCV with stratified folds, PR-AUC scoring."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    if not param_grid:
        estimator.fit(X_train, y_train)
        return estimator

    n_iter_eff = min(n_iter, max(8, len(param_grid) * 4))
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_grid,
        n_iter=n_iter_eff,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print(f"Best CV {scoring}: {search.best_score_:.4f}")
    return search.best_estimator_


def soft_vote_proba(models: list[Any], X: Any) -> np.ndarray:
    """Average predicted P(churn=1) across fitted estimators."""
    probs = []
    for m in models:
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)
            probs.append(p[:, 1] if p.ndim == 2 else p)
        elif hasattr(m, "decision_function"):
            from sklearn.preprocessing import MinMaxScaler

            s = np.asarray(m.decision_function(X), dtype=float).reshape(-1, 1)
            probs.append(MinMaxScaler().fit_transform(s).ravel())
        else:
            probs.append(np.asarray(m.predict(X), dtype=float))
    avg = np.mean(np.column_stack(probs), axis=1)
    return np.column_stack([1.0 - avg, avg])


def fit_calibrated_isotonic(
    base_estimator: Any,
    X_train: Any,
    y_train: Any,
    *,
    cv: int = 3,
) -> Any:
    """Isotonic calibration via CV (probabilities usable for expected value)."""
    # clone-ish: CalibratedClassifierCV refits
    cal = CalibratedClassifierCV(base_estimator, method="isotonic", cv=cv)
    cal.fit(X_train, y_train)
    return cal


def production_classical_stack(
    base_estimators: list[tuple[str, Any]],
    X_train: Any,
    y_train: Any,
) -> Any:
    """Stacking with logistic meta-learner (balanced)."""
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        passthrough=False,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=2,
    )
    stack.fit(X_train, y_train)
    return stack


def predict_proba_matrix(model: Any, X: Any) -> np.ndarray:
    """Always return (n, 2) probability matrix."""
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=float)
        if p.ndim == 1:
            return np.column_stack([1 - p, p])
        return p
    if hasattr(model, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler

        s = np.asarray(model.decision_function(X), dtype=float).reshape(-1, 1)
        y = MinMaxScaler().fit_transform(s).ravel()
        return np.column_stack([1 - y, y])
    pred = np.asarray(model.predict(X), dtype=float)
    return np.column_stack([1 - pred, pred])
