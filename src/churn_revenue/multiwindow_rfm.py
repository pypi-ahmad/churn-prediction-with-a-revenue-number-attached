"""Multi-window RFM + simple discrete-time hazard labels for retail churn.

Why a single Recency/Frequency/Monetary snapshot collapses history.
Multi-window features (30/90/180d) capture trends; hazard-style labels ask
"no purchase in the next H days after cutoff" with explicit horizon H.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_customer_features(
    tx: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    customer_col: str = "Customer ID",
    date_col: str = "InvoiceDate",
    invoice_col: str = "Invoice",
    stock_col: str = "StockCode",
    qty_col: str = "Quantity",
    revenue_col: str = "line_revenue",
    country_col: str = "Country",
    windows_days: tuple[int, ...] = (30, 90, 180),
) -> pd.DataFrame:
    """Aggregate pre-cutoff multi-window RFM + trend features per customer."""
    pre = tx[tx[date_col] < cutoff].copy()
    if pre.empty:
        raise ValueError("No pre-cutoff transactions")

    # overall RFM
    country_mode = (
        pre.groupby(customer_col)[country_col].agg(lambda s: s.value_counts().index[0])
        if country_col in pre.columns
        else None
    )
    base = pre.groupby(customer_col).agg(
        Recency=(date_col, lambda s: (cutoff - s.max()).days),
        Frequency=(invoice_col, "nunique"),
        Monetary=(revenue_col, "sum"),
        n_products=(stock_col, "nunique"),
        n_items=(qty_col, "sum"),
        tenure_days=(date_col, lambda s: max((s.max() - s.min()).days, 0)),
        avg_line=(revenue_col, "mean"),
        std_line=(revenue_col, "std"),
        n_lines=(revenue_col, "count"),
    )
    base["std_line"] = base["std_line"].fillna(0.0)
    if country_mode is not None:
        base = base.join(country_mode.rename("Country"), how="left")

    # interpurchase stats
    def _gap_stats(g: pd.DataFrame) -> pd.Series:
        d = g[date_col].sort_values().drop_duplicates()
        if len(d) < 2:
            return pd.Series({"mean_gap_days": np.nan, "std_gap_days": np.nan})
        gaps = d.diff().dt.days.dropna()
        return pd.Series(
            {"mean_gap_days": float(gaps.mean()), "std_gap_days": float(gaps.std(ddof=0))}
        )

    gaps = pre.groupby(customer_col).apply(_gap_stats, include_groups=False)
    base = base.join(gaps, how="left")
    base["mean_gap_days"] = base["mean_gap_days"].fillna(base["Recency"])
    base["std_gap_days"] = base["std_gap_days"].fillna(0.0)

    # multi-window activity
    for w in windows_days:
        start = cutoff - pd.Timedelta(days=w)
        wtx = pre[pre[date_col] >= start]
        agg = wtx.groupby(customer_col).agg(
            **{
                f"freq_{w}d": (invoice_col, "nunique"),
                f"mon_{w}d": (revenue_col, "sum"),
                f"lines_{w}d": (revenue_col, "count"),
            }
        )
        base = base.join(agg, how="left")
        for c in [f"freq_{w}d", f"mon_{w}d", f"lines_{w}d"]:
            base[c] = base[c].fillna(0.0)

    # trends: short vs long window
    if 30 in windows_days and 180 in windows_days:
        base["freq_trend_30_180"] = base["freq_30d"] / (base["freq_180d"] / 6.0 + 1e-6)
        base["mon_trend_30_180"] = base["mon_30d"] / (base["mon_180d"] / 6.0 + 1e-6)

    base["log_monetary"] = np.log1p(base["Monetary"].clip(lower=0))
    base["log_frequency"] = np.log1p(base["Frequency"])
    base["log_recency"] = np.log1p(base["Recency"])
    base["activity_rate"] = base["Frequency"] / (base["tenure_days"] + 1)
    base["monetary_per_invoice"] = base["Monetary"] / base["Frequency"].clip(lower=1)
    return base


def hazard_churn_label(
    tx: pd.DataFrame,
    customers: pd.Index,
    *,
    cutoff: pd.Timestamp,
    horizon_days: int = 90,
    customer_col: str = "Customer ID",
    date_col: str = "InvoiceDate",
) -> pd.Series:
    """churn=1 if customer has zero purchases in [cutoff, cutoff+horizon)."""
    end = cutoff + pd.Timedelta(days=horizon_days)
    post = tx[(tx[date_col] >= cutoff) & (tx[date_col] < end)]
    active = set(post[customer_col].unique())
    lab = pd.Series(
        [0 if c in active else 1 for c in customers],
        index=customers,
        name="churn",
    )
    return lab.astype(int)


def multi_horizon_labels(
    tx: pd.DataFrame,
    customers: pd.Index,
    *,
    cutoff: pd.Timestamp,
    horizons: tuple[int, ...] = (30, 60, 90, 120),
    customer_col: str = "Customer ID",
    date_col: str = "InvoiceDate",
) -> pd.DataFrame:
    """Several hazard horizons for sensitivity / multi-task style analysis."""
    cols = {}
    for h in horizons:
        cols[f"churn_{h}d"] = hazard_churn_label(
            tx,
            customers,
            cutoff=cutoff,
            horizon_days=h,
            customer_col=customer_col,
            date_col=date_col,
        )
    return pd.DataFrame(cols, index=customers)
