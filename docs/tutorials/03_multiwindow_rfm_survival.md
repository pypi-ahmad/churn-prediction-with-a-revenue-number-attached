# Tutorial: Multi-window RFM and hazard-style churn labels

## The problem with a single RFM snapshot

Classic RFM uses one Recency, one Frequency, and one Monetary value over the full history before a cutoff. That loses changes over time:

- A customer active every week for 2 years who went silent for 20 days  
- vs someone who bought once 20 days ago  

These customers can have similar Recency values but very different risk.

## Multi-window features

For windows \(w \in \{30, 90, 180\}\) days before cutoff, compute:

- frequency of invoices in window  
- monetary sum in window  
- line counts  

**Trend ratios** (example):

\[
\text{freq\_trend} = \frac{\text{freq}_{30d}}{\text{freq}_{180d}/6 + \epsilon}
\]

Values ≪ 1 mean the customer is cooling off relative to their long-run rate — a strong churn signal.

Also useful:

- mean/std of inter-purchase gaps  
- log transforms of monetary/frequency  

All of this is built **only on pre-cutoff data** → no leakage.

## Hazard-style label (discrete time)

Instead of a vague “churned,” define:

> **churn_H = 1** if the customer has **zero** purchases in \([cutoff, cutoff+H)\) days.

v3 reports multiple horizons (30/60/90/120d) for sensitivity; the **primary model label** remains 90d for comparability with older notebooks.

This is a **discrete-time hazard** view: “event = no repurchase within H.”  
A full survival model (Cox, Weibull AFT) is a natural extension; multi-window RFM + hazard labels already capture most of the practical benefit for tree models.

## Why results improve

| Change | Effect |
|--------|--------|
| Multi-window activity | Trees split on “cooling off,” not only lifetime totals |
| Gap statistics | Encodes regularity of buying |
| Multi-horizon tables | Shows label sensitivity (honest science) |

Large differences between the 30d and 120d rates show how much the business definition affects the result.

## Code map

- `src/churn_revenue/multiwindow_rfm.py`  
- `notebooks/v3/03_retail_v3_awesome.py`  
