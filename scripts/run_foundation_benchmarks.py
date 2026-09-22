"""Run zero-shot Mitra churn benchmarks and Retail TimesFM risk backtest."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from churn_revenue.foundation import (
    mitra_probabilities,
    timesfm_risk,
    tuned_risk_metrics,
)
from churn_revenue.metrics import evaluate_scores
from churn_revenue.modeling import RANDOM_STATE
from churn_revenue.multiwindow_rfm import build_customer_features, hazard_churn_label
from churn_revenue.threshold import tune_threshold_f1

TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def split(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv)
    return x_train, x_val, x_test, y_train, y_val, y_test


def iranian() -> tuple[pd.DataFrame, pd.Series]:
    data = fetch_ucirepo(id=563)
    frame = pd.concat([data.data.features, data.data.targets], axis=1)
    return frame.drop(columns="Churn"), frame["Churn"].astype(int)


def telco() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(TELCO_URL)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"].astype(str).str.strip().replace("", "0"))
    frame["tenure_clip"] = frame["tenure"].clip(lower=1)
    frame["avg_charge_per_month"] = frame["TotalCharges"] / frame["tenure_clip"]
    frame["charge_tenure_ratio"] = frame["MonthlyCharges"] / frame["tenure_clip"]
    frame["num_services"] = sum((frame[column] == "Yes").astype(int) for column in ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"])
    frame["contract_x_internet"] = frame["Contract"].astype(str) + "|" + frame["InternetService"].astype(str)
    frame["contract_x_payment"] = frame["Contract"].astype(str) + "|" + frame["PaymentMethod"].astype(str)
    frame["tenure_band"] = pd.cut(frame["tenure"], [-0.1, 12, 24, 48, 100], labels=["0-12", "12-24", "24-48", "48+"]).astype(str)
    columns = [column for column in frame.columns if column not in {"customerID", "Churn"}]
    return frame[columns], (frame["Churn"] == "Yes").astype(int)


def retail_transactions() -> pd.DataFrame:
    response = requests.get(RETAIL_URL, timeout=180)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive, archive.open(
        "online_retail_II.xlsx"
    ) as handle:
        workbook = pd.ExcelFile(handle)
        frame = pd.concat([workbook.parse(sheet) for sheet in workbook.sheet_names], ignore_index=True)
    frame["Invoice"] = frame["Invoice"].astype(str)
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])
    frame = frame[~frame["Invoice"].str.startswith("C")].dropna(subset=["Customer ID"])
    frame = frame[(frame["Quantity"] > 0) & (frame["Price"] > 0)].copy()
    frame["Customer ID"] = frame["Customer ID"].astype(int)
    frame["line_revenue"] = frame["Quantity"] * frame["Price"]
    return frame


def mitra_result(name: str, x: pd.DataFrame, y: pd.Series, output: Path) -> dict[str, float | str]:
    train_x, val_x, test_x, train_y, val_y, test_y = split(x, y)
    val = mitra_probabilities(train_x, train_y, val_x, output / "models" / f"{name}-validation")
    threshold, _ = tune_threshold_f1(val_y, val)
    final = mitra_probabilities(pd.concat([train_x, val_x]), pd.concat([train_y, val_y]), test_x, output / "models" / f"{name}-test")
    result = evaluate_scores(test_y, final, threshold=threshold, title=f"mitra_zero_shot_{name}")
    result.update({"dataset": name, "protocol": "stratified_60_20_20"})
    return result


def weekly_series(tx: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[list[np.ndarray], pd.Series]:
    customer = build_customer_features(tx, cutoff=cutoff).index
    labels = hazard_churn_label(tx, customer, cutoff=cutoff)
    pre = tx[(tx["InvoiceDate"] < cutoff) & tx["Customer ID"].isin(customer)].copy()
    pre["week"] = pre["InvoiceDate"].dt.to_period("W").dt.start_time
    grouped = pre.groupby(["Customer ID", "week"])["Invoice"].nunique()
    rows: list[np.ndarray] = []
    for customer_id in customer:
        values = grouped.loc[customer_id] if customer_id in grouped.index.get_level_values(0) else pd.Series(dtype=float)
        start = values.index.min() if not values.empty else cutoff
        weeks = pd.date_range(start=start, end=cutoff - pd.Timedelta(days=1), freq="W-MON")
        rows.append(values.reindex(weeks, fill_value=0.0).to_numpy(dtype=np.float32))
    return rows, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="run all Mitra datasets and Retail TimesFM")
    parser.add_argument("--output", type=Path, default=Path("artifacts/foundation-benchmarks"))
    args = parser.parse_args()
    if not args.all:
        parser.error("use --all")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; install the locked CUDA Torch environment first.")
    output = args.output
    shutil.rmtree(output / "models", ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    rows = [mitra_result("iranian", *iranian(), output), mitra_result("telco", *telco(), output)]
    tx = retail_transactions()
    cutoff = pd.Timestamp("2011-06-01")
    customers = build_customer_features(tx, cutoff=cutoff)
    rows.append(mitra_result("retail", customers, hazard_churn_label(tx, customers.index, cutoff=cutoff), output))
    val_series, val_y = weekly_series(tx, pd.Timestamp("2011-03-01"))
    test_series, test_y = weekly_series(tx, cutoff)
    val_risk = timesfm_risk(val_series)
    test_risk = timesfm_risk(test_series)
    timesfm = tuned_risk_metrics(val_y, val_risk, test_y, test_risk, "timesfm3_retail_raw_risk")
    timesfm.update({"dataset": "retail", "protocol": "2011-03-01_validation__2011-06-01_test"})
    rows.append(timesfm)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    (output / "run.json").write_text(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "rows": len(rows)}, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
