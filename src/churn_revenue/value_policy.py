"""Business-aware contact policies: top-K and expected-value thresholds.

Why: maximizing F1 is not the same as maximizing retention ROI.
A production system contacts customers only when expected saved value
exceeds contact cost (or when they fall in a fixed outreach budget).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def expected_contact_value(
    p_churn: np.ndarray,
    customer_value: np.ndarray,
    *,
    p_save: float = 0.25,
    contact_cost: float = 10.0,
) -> np.ndarray:
    """Expected net value of contacting each customer.

    EV_i = p_churn_i * value_i * P(save | contact) - contact_cost

    Assumptions (document in notebooks):
    - p_save is a constant save rate (replace with uplift model later)
    - value is CLV / MonthlyCharges / Monetary available at scoring time
    """
    p = np.asarray(p_churn, dtype=float)
    v = np.asarray(customer_value, dtype=float)
    return p * v * float(p_save) - float(contact_cost)


def policy_contact_if_positive_ev(
    p_churn: np.ndarray,
    customer_value: np.ndarray,
    *,
    p_save: float = 0.25,
    contact_cost: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Contact iff expected value > 0. Returns (mask, ev)."""
    ev = expected_contact_value(
        p_churn, customer_value, p_save=p_save, contact_cost=contact_cost
    )
    return ev > 0, ev


def policy_top_k_fraction(
    scores: np.ndarray,
    *,
    k_fraction: float = 0.10,
) -> np.ndarray:
    """Contact top k% by score (budget-limited call center)."""
    s = np.asarray(scores, dtype=float)
    n = len(s)
    k = max(1, int(np.ceil(n * float(k_fraction))))
    order = np.argsort(-s)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def evaluate_policy(
    y_true: np.ndarray,
    contact_mask: np.ndarray,
    customer_value: np.ndarray,
    *,
    p_save: float = 0.25,
    contact_cost: float = 10.0,
) -> dict[str, float]:
    """Evaluate a binary contact policy on labeled holdout."""
    y = np.asarray(y_true).astype(int)
    m = np.asarray(contact_mask).astype(bool)
    v = np.asarray(customer_value, dtype=float)
    contacted = int(m.sum())
    # true churners among contacted
    true_pos = int(((y == 1) & m).sum())
    # value of true churners we reached
    value_reached = float(v[(y == 1) & m].sum())
    # naive expected savings under constant p_save
    expected_save = value_reached * float(p_save)
    total_cost = contacted * float(contact_cost)
    return {
        "n_contacted": float(contacted),
        "contact_rate": float(contacted / max(len(y), 1)),
        "churners_reached": float(true_pos),
        "recall_at_policy": float(true_pos / max(int((y == 1).sum()), 1)),
        "precision_at_policy": float(true_pos / max(contacted, 1)),
        "value_of_churners_reached": value_reached,
        "expected_saved_value": expected_save,
        "contact_cost_total": total_cost,
        "net_expected_value": expected_save - total_cost,
    }


def sweep_top_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    customer_value: np.ndarray,
    *,
    fractions: list[float] | None = None,
    p_save: float = 0.25,
    contact_cost: float = 10.0,
) -> pd.DataFrame:
    """Sweep outreach budgets; pick fraction maximizing net expected value."""
    if fractions is None:
        fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    rows = []
    for f in fractions:
        mask = policy_top_k_fraction(scores, k_fraction=f)
        row = evaluate_policy(
            y_true, mask, customer_value, p_save=p_save, contact_cost=contact_cost
        )
        row["k_fraction"] = f
        rows.append(row)
    return pd.DataFrame(rows).sort_values("net_expected_value", ascending=False)


def best_top_k_on_validation(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    value_val: np.ndarray,
    *,
    p_save: float = 0.25,
    contact_cost: float = 10.0,
) -> tuple[float, pd.DataFrame]:
    """Choose k% on validation by net EV; return (best_fraction, full_table)."""
    table = sweep_top_k(
        y_val,
        scores_val,
        value_val,
        p_save=p_save,
        contact_cost=contact_cost,
    )
    best_f = float(table.iloc[0]["k_fraction"])
    return best_f, table
