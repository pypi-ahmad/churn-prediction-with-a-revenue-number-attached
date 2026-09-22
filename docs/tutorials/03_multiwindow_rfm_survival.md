# Tutorial: Multi-window RFM and hazard-style churn labels

## The problem with a single RFM snapshot

Classic RFM uses one Recency, Frequency, and Monetary value from the full history before a cutoff. It does not show how activity changes over time:

- A customer active every week for 2 years who went silent for 20 days  
- vs someone who bought once 20 days ago  

These customers can have similar Recency values and very different risk.

## Multi-window features

For windows \(w \in \{30, 90, 180\}\) days before cutoff, compute:

- frequency of invoices in window  
- monetary sum in window  
- line counts  

Example trend ratio:

\[
\text{freq\_trend} = \frac{\text{freq}_{30d}}{\text{freq}_{180d}/6 + \epsilon}
\]

Values ≪ 1 mean the customer is becoming less active than their long-run rate, which can indicate churn risk.

Also useful:

- mean/std of inter-purchase gaps  
- log transforms of monetary/frequency  

All features use only pre-cutoff data, so they do not leak future information.

## Hazard-style label (discrete time)

Instead of a vague “churned,” define:

> **churn_H = 1** if the customer has **zero** purchases in \([cutoff, cutoff+H)\) days.

v3 reports multiple horizons (30/60/90/120d) for sensitivity; the **primary model label** remains 90d for comparability with older notebooks.

This is a discrete-time hazard view: “event = no repurchase within H.”
A full survival model, such as Cox or Weibull AFT, is a possible extension. Multi-window RFM with hazard labels captures much of the practical benefit for tree models.

## Why results improve

| Change | Effect |
|--------|--------|
| Multi-window activity | Trees can split on declining activity as well as lifetime totals |
| Gap statistics | Encodes regularity of buying |
| Multi-horizon tables | Shows how the label changes with the horizon |

Large differences between the 30d and 120d rates show how much the business definition affects the result.

## Code map

- `src/churn_revenue/multiwindow_rfm.py`  
- `notebooks/v3/03_retail_v3_awesome.py`  
