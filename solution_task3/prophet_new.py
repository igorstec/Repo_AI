"""Prophet model for Task 3 — per-device monthly avg x2 forecast (May-Oct 2025).
Uses t1, x1, temperature as regressors from data3.csv (hourly, with weather).
Extra features:
  - temp_delta_12h: temperature change vs 12 hours ago (per device, hourly -> daily avg)
  - devicetype_mean_x2: mean x2 of the device's deviceType from training (static regressor)
Multiplicative seasonality, linear growth, parallel processing.
"""

import warnings
import pandas as pd
import numpy as np
from prophet import Prophet
from multiprocessing import Pool, cpu_count
import logging
import os

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

DATA_PATH = "data/out/data3.csv"
OUT_PATH = "data/out/submission_prophet_new.csv"

FORECAST_MONTHS = [5, 6, 7, 8, 9, 10]
FORECAST_YEAR = 2025
REGRESSORS = ["t1", "x1", "temperature", "temp_delta_12h", "devicetype_mean_x2"]


def load_daily(path: str):
    usecols = ["deviceId", "timedate", "period", "x2", "t1", "x1", "temperature", "deviceType"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["timedate"] = pd.to_datetime(df["timedate"], utc=True)

    # --- Feature: temp_delta_12h (hourly level, per device) ---
    df = df.sort_values(["deviceId", "timedate"])
    df["temp_delta_12h"] = df.groupby("deviceId")["temperature"].diff(periods=12)
    df["temp_delta_12h"] = df["temp_delta_12h"].fillna(0.0)

    # --- Feature: devicetype_mean_x2 (static, from training data) ---
    train_mask = df["period"] == "train"
    dt_mean_x2 = df.loc[train_mask].groupby("deviceType")["x2"].mean().rename("devicetype_mean_x2")
    df = df.merge(dt_mean_x2, on="deviceType", how="left")
    # For safety, fill any NaN with global train mean
    global_train_mean = df.loc[train_mask, "x2"].mean()
    df["devicetype_mean_x2"] = df["devicetype_mean_x2"].fillna(global_train_mean)

    # --- Aggregate to daily ---
    df["ds"] = df["timedate"].dt.normalize().dt.tz_localize(None)
    agg_cols = ["x2", "t1", "x1", "temperature", "temp_delta_12h", "devicetype_mean_x2"]
    daily = df.groupby(["deviceId", "ds", "period"])[agg_cols].mean().reset_index()
    del df

    train = daily[daily["period"] == "train"].rename(columns={"x2": "y"}).copy()
    future = daily[daily["period"].isin(["valid", "test"])].copy()
    return train, future


def fit_prophet(df_train):
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
    )
    for reg in REGRESSORS:
        m.add_regressor(reg, standardize=True)
    m.fit(df_train)
    return m


def process_device(args):
    dev_id, df_tr, df_fu = args

    fallback = float(df_tr["y"].mean()) if len(df_tr) > 0 else 0.0

    if len(df_tr) < 14 or df_fu.empty:
        return [(dev_id, FORECAST_YEAR, mo, fallback) for mo in FORECAST_MONTHS]

    for reg in REGRESSORS:
        df_fu[reg] = df_fu[reg].fillna(df_tr[reg].mean())

    try:
        m = fit_prophet(df_tr[["ds", "y"] + REGRESSORS])
        forecast = m.predict(df_fu[["ds"] + REGRESSORS])
    except Exception:
        return [(dev_id, FORECAST_YEAR, mo, fallback) for mo in FORECAST_MONTHS]

    forecast["year"] = forecast["ds"].dt.year
    forecast["month"] = forecast["ds"].dt.month
    forecast["yhat"] = forecast["yhat"].clip(lower=0.0)

    monthly = (
        forecast[
            (forecast["year"] == FORECAST_YEAR)
            & (forecast["month"].isin(FORECAST_MONTHS))
        ]
        .groupby(["year", "month"])["yhat"]
        .mean()
    )

    rows = []
    predicted_months = set()
    for (yr, mo), pred in monthly.items():
        rows.append((dev_id, yr, mo, float(pred)))
        predicted_months.add(mo)
    for mo in FORECAST_MONTHS:
        if mo not in predicted_months:
            rows.append((dev_id, FORECAST_YEAR, mo, fallback))

    return rows


def main():
    print("Loading data ...")
    train_daily, future_daily = load_daily(DATA_PATH)
    device_ids = sorted(train_daily["deviceId"].unique())
    print(f"Devices: {len(device_ids)}  |  Train rows: {len(train_daily):,}")
    print(f"Regressors: {REGRESSORS}")

    # Prepare per-device data
    device_args = []
    for dev_id in device_ids:
        df_tr = train_daily[train_daily["deviceId"] == dev_id][["ds", "y"] + REGRESSORS].copy()
        df_fu = future_daily[future_daily["deviceId"] == dev_id][["ds"] + REGRESSORS].copy()
        device_args.append((dev_id, df_tr, df_fu))

    n_workers = min(cpu_count(), 8)
    print(f"Processing with {n_workers} workers...")

    all_rows = []
    with Pool(n_workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_device, device_args)):
            all_rows.extend(rows)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(device_ids)} devices done")

    print(f"All {len(device_ids)} devices done.")

    submission = (
        pd.DataFrame(all_rows, columns=["deviceId", "year", "month", "prediction"])
        .sort_values(["deviceId", "year", "month"])
        .reset_index(drop=True)
    )
    submission["year"] = submission["year"].astype(int)
    submission["month"] = submission["month"].astype(int)

    n_expected = len(device_ids) * len(FORECAST_MONTHS)
    print(f"Rows: {len(submission)} / {n_expected}")

    print(f"\nPrediction stats:\n{submission['prediction'].describe()}")
    print(f"\nMean prediction per month:")
    print(submission.groupby("month")["prediction"].mean())
    print(f"\nZeros (< 0.001) per month:")
    zeros = submission[submission["prediction"] < 0.001].groupby("month").size()
    print(zeros if len(zeros) > 0 else "None!")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
