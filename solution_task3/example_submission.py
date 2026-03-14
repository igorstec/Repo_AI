import os
import csv

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

load_dotenv()

ENDPOINT = "task3"
API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")
DATA_DIR = os.getenv("DATA_DIR", "data")  # path to unpacked task3 dataset

CSV_FILE = "data/out/submission_train3.csv"

FORECAST_MONTHS = [5, 6, 7, 8, 9, 10]
FORECAST_YEAR = 2025


def load_data():
    """Load and merge time-series telemetry with device metadata."""
    data = pd.read_csv(os.path.join(DATA_DIR, "data.csv"), parse_dates=["timedate"])
    devices = pd.read_csv(os.path.join(DATA_DIR, "devices.csv"))
    data = data.merge(devices, on="deviceId", how="left")
    return data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based and aggregated features from raw telemetry."""
    df = df.copy()
    df["year"] = df["timedate"].dt.year
    df["month"] = df["timedate"].dt.month
    df["hour"] = df["timedate"].dt.hour
    df["dayofweek"] = df["timedate"].dt.dayofweek

    # Encode categorical device type
    le = LabelEncoder()
    if "deviceType" in df.columns:
        df["deviceType_enc"] = le.fit_transform(df["deviceType"].astype(str))

    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-min readings to monthly level per device."""
    temp_cols = [f"t{i}" for i in range(1, 14) if f"t{i}" in df.columns]
    agg_dict = {col: "mean" for col in temp_cols}

    if "x1" in df.columns:
        agg_dict["x1"] = "mean"
    if "x2" in df.columns:
        agg_dict["x2"] = "mean"  # target for training months
    if "latitude" in df.columns:
        agg_dict["latitude"] = "first"
    if "longitude" in df.columns:
        agg_dict["longitude"] = "first"
    if "deviceType_enc" in df.columns:
        agg_dict["deviceType_enc"] = "first"
    if "x3" in df.columns:
        agg_dict["x3"] = "first"

    monthly = (
        df.groupby(["deviceId", "year", "month"])
        .agg(agg_dict)
        .reset_index()
    )
    return monthly


def make_month_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month as cyclical sin/cos features."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


FEATURE_COLS = [
    "month_sin", "month_cos",
    "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13",
    "x1", "x3", "latitude", "longitude", "deviceType_enc",
]


def get_available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def train_model(train_monthly: pd.DataFrame):
    """Train a LightGBM model on monthly aggregates."""
    from lightgbm import LGBMRegressor

    train_monthly = make_month_sin_cos(train_monthly)
    feats = get_available_features(train_monthly)

    X = train_monthly[feats].values
    y = train_monthly["x2"].values

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    print(f"Trained on {len(train_monthly)} monthly samples, features: {feats}")
    return model, feats


def generate_submission(model, feats: list[str], monthly: pd.DataFrame) -> list[tuple]:
    """Generate per-device, per-month predictions for May–Oct 2025."""
    device_profiles = (
        monthly.groupby("deviceId")
        .agg({c: "mean" for c in feats if c not in ("month_sin", "month_cos")})
        .reset_index()
    )

    rows = []
    for _, row in device_profiles.iterrows():
        device_id = row["deviceId"]
        for month in FORECAST_MONTHS:
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            feat_row = []
            for f in feats:
                if f == "month_sin":
                    feat_row.append(month_sin)
                elif f == "month_cos":
                    feat_row.append(month_cos)
                else:
                    feat_row.append(row.get(f, 0.0))

            pred = float(model.predict([feat_row])[0])
            pred = float(np.clip(pred, 0.0, 1.0))
            rows.append((device_id, FORECAST_YEAR, month, pred))

    return rows


def save_csv(predictions: list[tuple]):
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["deviceId", "year", "month", "prediction"])
        writer.writerows(predictions)
    print(f"Saved {len(predictions)} predictions to {CSV_FILE}")


def submit():
    if not API_TOKEN:
        raise ValueError("TEAM_TOKEN not provided. Define TEAM_TOKEN in .env")
    if not SERVER_URL:
        raise ValueError("SERVER_URL not defined. Define SERVER_URL in .env")

    headers = {"X-API-Token": API_TOKEN}
    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"csv_file": open(CSV_FILE, "rb")},
        headers=headers,
    )

    try:
        data = response.json()
    except Exception:
        data = response.text

    print("Response:", response.status_code, data)


def main():
    print("Loading data...")
    df = load_data()

    print("Building features...")
    df = build_features(df)

    print("Aggregating to monthly level...")
    monthly = aggregate_monthly(df)

    # Train only on months where x2 is available (training split)
    train_monthly = monthly[monthly["x2"].notna()].copy()

    print("Training model...")
    model, feats = train_model(train_monthly)

    print("Generating predictions...")
    predictions = generate_submission(model, feats, monthly)

    save_csv(predictions)
    submit()


if __name__ == "__main__":
    main()
