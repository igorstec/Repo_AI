"""Prophet model for Task 3 — per-device monthly avg x2 forecast (May–Oct 2025)."""

import logging
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

DATA_PATH = "./data2.csv"
DEVICES_PATH = "./devices.csv"
OUT_PATH = "/home/taliyah/ensemble/solves/task3/submission_prophet.csv"

FORECAST_MONTHS = [5, 6, 7, 8, 9, 10]
FORECAST_YEAR = 2025
REGRESSORS = ["t1", "x1", "temperature"]


def load_daily(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = ["deviceId", "timedate", "period", "x2"] + REGRESSORS
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["ds"] = pd.to_datetime(df["timedate"], utc=True).dt.normalize()

    agg_cols = ["x2"] + REGRESSORS
    daily = df.groupby(["deviceId", "ds", "period"])[agg_cols].mean().reset_index()
    del df

    train = daily[daily["period"] == "train"].rename(columns={"x2": "y"}).copy()
    valid = daily[daily["period"] == "valid"].copy()
    future = daily[daily["period"].isin(["valid", "test"])].copy()
    return train, valid, future


def fit_prophet(df_train: pd.DataFrame) -> Prophet:
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


def main():
    print("Loading data ...")
    train_daily, valid_daily, future_daily = load_daily(DATA_PATH)
    device_ids = sorted(train_daily["deviceId"].unique())
    print(f"Devices: {len(device_ids)}  |  Train rows: {len(train_daily):,}")

    all_rows = []
    val_maes = []

    for dev_id in tqdm(device_ids, desc="Prophet"):
        df_tr = train_daily[train_daily["deviceId"] == dev_id][["ds", "y"] + REGRESSORS].copy()
        df_fu = future_daily[future_daily["deviceId"] == dev_id][["ds"] + REGRESSORS].copy()
        df_val = valid_daily[valid_daily["deviceId"] == dev_id].copy()

        fallback = float(df_tr["y"].mean()) if len(df_tr) > 0 else 0.0

        if len(df_tr) < 14 or df_fu.empty:
            for mo in FORECAST_MONTHS:
                all_rows.append((dev_id, FORECAST_YEAR, mo, fallback))
            continue

        for reg in REGRESSORS:
            df_fu[reg] = df_fu[reg].fillna(df_tr[reg].mean())

        try:
            m = fit_prophet(df_tr[["ds", "y"] + REGRESSORS])
            forecast = m.predict(df_fu[["ds"] + REGRESSORS])
        except Exception:
            for mo in FORECAST_MONTHS:
                all_rows.append((dev_id, FORECAST_YEAR, mo, fallback))
            continue

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

        predicted_months = set()
        for (yr, mo), pred in monthly.items():
            all_rows.append((dev_id, yr, mo, float(pred)))
            predicted_months.add(mo)
        for mo in FORECAST_MONTHS:
            if mo not in predicted_months:
                all_rows.append((dev_id, FORECAST_YEAR, mo, fallback))

        # Validation against ground truth (May–Jun)
        if len(df_val) > 0:
            df_val["month"] = df_val["ds"].dt.month
            true_mo = df_val.groupby("month")["x2"].mean()
            val_fc = forecast[
                (forecast["year"] == FORECAST_YEAR) & (forecast["month"].isin([5, 6]))
            ]
            if len(val_fc) > 0:
                pred_mo = val_fc.groupby("month")["yhat"].mean()
                common = pred_mo.index.intersection(true_mo.index)
                if len(common) > 0:
                    val_maes.append((pred_mo[common] - true_mo[common]).abs().mean())

    # Validation report
    if val_maes:
        arr = np.array(val_maes)
        print(f"\nValidation MAE (May–Jun 2025): mean={arr.mean():.5f}  median={np.median(arr):.5f}  p90={np.percentile(arr, 90):.5f}")

    # Build submission
    submission = (
        pd.DataFrame(all_rows, columns=["deviceId", "year", "month", "prediction"])
        .sort_values(["deviceId", "year", "month"])
        .reset_index(drop=True)
    )
    submission["year"] = submission["year"].astype(int)
    submission["month"] = submission["month"].astype(int)

    n_expected = len(device_ids) * len(FORECAST_MONTHS)
    print(f"Rows: {len(submission)} / {n_expected}")

    submission.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
