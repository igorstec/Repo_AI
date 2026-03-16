"""
Task 3: SUPREMACY LightGBM — per-deviceType + seasonal bias correction.

Key improvements over v2:
1. dev_month_x2: per-device monthly historical mean — CRITICAL for monthly MAE
2. Seasonal bias correction: per-device × month-of-year OOF residuals
3. Lag features: t2_lag_1h, t2_lag_3h, x1_lag_1h, x1_rolling_3h
4. Extended rolling: 48h + 168h temperature windows
5. More per-device stats: device_mean_x1, device_mean_t1, dev_cv_x2, dev_dow_x2
6. LightGBM with MAE objective + num_leaves=255, 3000 trees, lr=0.015
7. Multi-seed ensemble: 3 seeds averaged for final predictions
8. Stronger recency weights: most recent 3 months get 3× weight
9. day_of_year cyclic feature
10. Per-device monthly temperature mean (seasonal context per device)

Target: MAE < 0.0018
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import os
import gc

DATA_PATH = "data/out/data3.csv"
OUT_PATH = "submission_optimized.csv"

FORECAST_MONTHS = [5, 6, 7, 8, 9, 10]
FORECAST_YEAR = 2025

LGBM_SEEDS = [42, 123, 777]  # multi-seed ensemble


def add_rolling_features(df):
    """Add rolling mean/lag features per device."""
    print("  Computing rolling features...")
    grp = df.groupby("deviceId")

    # Core rolling windows: 6h, 24h, 48h
    roll_cols = ["temperature", "t1", "t2", "t10", "x1"]
    windows = [6, 24, 48]
    for col in roll_cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll{w}h"] = grp[col].transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )

    # Weekly temperature rolling (168h)
    df["temperature_roll168h"] = grp["temperature"].transform(
        lambda s: s.rolling(168, min_periods=1).mean()
    )

    # Temperature rolling std (weather variability)
    df["temp_roll_std_24h"] = grp["temperature"].transform(
        lambda s: s.rolling(24, min_periods=1).std()
    ).fillna(0.0)

    # Lag features (thermal inertia)
    df["t2_lag_1h"] = grp["t2"].shift(1).bfill()
    df["t2_lag_3h"] = grp["t2"].shift(3).bfill()
    df["x1_lag_1h"] = grp["x1"].shift(1).bfill()

    # Compressor short rolling
    df["x1_rolling_3h"] = grp["x1"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # Sunshine rolling
    if "sunshine_duration" in df.columns:
        df["sunshine_rolling_6h"] = grp["sunshine_duration"].transform(
            lambda s: s.rolling(6, min_periods=1).mean()
        )
        df["sunshine_rolling_24h"] = grp["sunshine_duration"].transform(
            lambda s: s.rolling(24, min_periods=1).mean()
        )

    # Rain rolling sum 24h
    if "rain" in df.columns:
        df["rain_last_24h"] = grp["rain"].transform(
            lambda s: s.rolling(24, min_periods=1).sum()
        )

    # HEX cross-interaction rolling 6h
    df["hex_cross_rolling_6h"] = grp["hex_cross"].transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )

    # Temperature delta vs 12h ago
    df["temp_delta_12h"] = grp["temperature"].diff(periods=12).fillna(0.0)

    # Delta 5h + rolling mean 5h for top features
    top_feats = ["T_app_diff", "temperature", "delta_T_out", "hex_cross"]
    for feat in top_feats:
        if feat in df.columns:
            df[f"{feat}_delta_5h"] = grp[feat].diff(periods=5).fillna(0.0)
            df[f"{feat}_rmean_5h"] = grp[feat].transform(
                lambda s: s.rolling(5, min_periods=1).mean()
            )

    return df


def main():
    print("=" * 60)
    print("Task 3: SUPREMACY — per-deviceType LGBMs + seasonal bias")
    print("=" * 60)

    # ==========================================
    # 1. Load data
    # ==========================================
    print("\n[1/8] Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["timedate"] = pd.to_datetime(df["timedate"], utc=True)
    df = df.sort_values(["deviceId", "timedate"]).reset_index(drop=True)
    print(f"  Total rows: {len(df):,}")

    # Join devices.csv for lat/lon
    devices_path = os.path.join(os.path.dirname(DATA_PATH), "..", "devices.csv")
    if os.path.exists(devices_path):
        devices = pd.read_csv(devices_path)
        df = df.merge(devices[["deviceId", "latitude", "longitude"]], on="deviceId", how="left")
        df["latitude"] = df["latitude"].fillna(0.0)
        df["longitude"] = df["longitude"].fillna(0.0)
        print("  Merged lat/lon from devices.csv")

    # ==========================================
    # 2. Feature engineering
    # ==========================================
    print("\n[2/8] Engineering features...")

    train_mask = df["period"] == "train"

    # Time features
    df["hour"] = df["timedate"].dt.hour
    df["dayofweek"] = df["timedate"].dt.dayofweek
    df["month"] = df["timedate"].dt.month
    df["day"] = df["timedate"].dt.day
    df["year"] = df["timedate"].dt.year
    df["day_of_year"] = df["timedate"].dt.dayofyear

    # Cyclic encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Binary time flags
    df["is_night"] = ((df["hour"] <= 6) | (df["hour"] >= 22)).astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    # Temperature deltas
    df["delta_T_out"] = df["t1"] - df["t2"]
    df["delta_load_hex"] = df["t5"] - df["t6"]
    df["delta_source_hex"] = df["t3"] - df["t4"]
    df["delta_load_3_4"] = df["t12"] - df["t13"]
    df["delta_weather_internal"] = df["temperature"] - df["t2"]
    df["T_app_diff"] = df["apparent_temperature"] - df["temperature"]

    # Heating/cooling demand
    df["heating_demand"] = (0.7 - df["t2"]).clip(lower=0.0)
    df["cooling_demand"] = (df["t2"] - 0.7).clip(lower=0.0)

    # HEX cross-interaction
    df["hex_cross"] = (df["t3"] - df["t5"]) * (df["t4"] - df["t6"])

    # Non-linear temperature features
    df["temperature_sq"] = df["temperature"] ** 2
    df["temp_warm"] = (df["temperature"] - 10).clip(lower=0.0)
    df["temp_cold"] = (10 - df["temperature"]).clip(lower=0.0)
    df["temp_x_night"] = df["temperature"] * df["is_night"]

    # DeviceType one-hot for interactions
    for dt in [7, 11, 19]:
        df[f"is_dtype_{dt}"] = (df["deviceType"] == dt).astype(np.int8)
        df[f"temp_x_dtype_{dt}"] = df["temperature"] * df[f"is_dtype_{dt}"]

    # Fill NaN in sensor columns
    fill_cols = ["t7", "t8", "t9", "t10", "t11", "t12", "t13", "x3",
                 "cloud_cover", "showers", "relative_humidity_2m", "dew_point_2m",
                 "wind_speed_10m", "weather_code", "surface_pressure",
                 "sunshine_duration", "rain_sum", "rain", "uv_index_max",
                 "precipitation_probability"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Rolling features
    df = add_rolling_features(df)

    # ==========================================
    # Per-device static stats from training
    # ==========================================
    dev_stats = df.loc[train_mask].groupby("deviceId").agg(
        dev_mean_x2=("x2", "mean"),
        dev_std_x2=("x2", "std"),
        dev_median_x2=("x2", "median"),
        dev_min_x2=("x2", "min"),
        dev_max_x2=("x2", "max"),
        dev_mean_x1=("x1", "mean"),
        dev_mean_t1=("t1", "mean"),
        dev_std_t1=("t1", "std"),
    ).reset_index()
    dev_stats["dev_std_x2"] = dev_stats["dev_std_x2"].fillna(0.0)
    dev_stats["dev_cv_x2"] = (
        dev_stats["dev_std_x2"] / (dev_stats["dev_mean_x2"] + 1e-9)
    )
    df = df.merge(dev_stats, on="deviceId", how="left")

    # Per-deviceType stats
    dt_stats = df.loc[train_mask].groupby("deviceType").agg(
        dt_mean_x2=("x2", "mean"), dt_std_x2=("x2", "std")
    ).reset_index()
    df = df.merge(dt_stats, on="deviceType", how="left")

    # *** CRITICAL: Per-device MONTHLY historical profile ***
    # This is the single most important feature for monthly prediction accuracy.
    # It tells the model: "device X consumed Y amount in month M historically."
    dev_month_stats = df.loc[train_mask].groupby(["deviceId", "month"]).agg(
        dev_month_x2=("x2", "mean"),
        dev_month_x2_std=("x2", "std"),
        dev_month_temp_mean=("temperature", "mean"),  # seasonal temp context
    ).reset_index()
    dev_month_stats["dev_month_x2_std"] = dev_month_stats["dev_month_x2_std"].fillna(0.0)
    df = df.merge(dev_month_stats, on=["deviceId", "month"], how="left")
    # Fallback to device mean where no monthly data
    df["dev_month_x2"] = df["dev_month_x2"].fillna(df["dev_mean_x2"])
    df["dev_month_x2_std"] = df["dev_month_x2_std"].fillna(df["dev_std_x2"])
    df["dev_month_temp_mean"] = df["dev_month_temp_mean"].fillna(df["temperature"])

    # Temperature deviation from device's monthly mean (how unusual is today's temp?)
    df["temp_dev_from_monthly"] = df["temperature"] - df["dev_month_temp_mean"]

    # Per-device hourly profile
    dev_hour = df.loc[train_mask].groupby(["deviceId", "hour"])["x2"].mean().rename("dev_hour_x2").reset_index()
    df = df.merge(dev_hour, on=["deviceId", "hour"], how="left")
    df["dev_hour_x2"] = df["dev_hour_x2"].fillna(df["dev_mean_x2"])

    # Per-device day-of-week profile
    dev_dow = df.loc[train_mask].groupby(["deviceId", "dayofweek"])["x2"].mean().rename("dev_dow_x2").reset_index()
    df = df.merge(dev_dow, on=["deviceId", "dayofweek"], how="left")
    df["dev_dow_x2"] = df["dev_dow_x2"].fillna(df["dev_mean_x2"])

    # Per-device temperature sensitivity
    def calc_temp_coefs(group):
        x = group["temperature"].values
        y = group["x2"].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            return pd.Series({"dev_temp_slope": 0.0, "dev_temp_warm_slope": 0.0})
        x, y = x[mask], y[mask]
        slope = np.polyfit(x, y, 1)[0]
        warm_mask = x > 10
        warm_slope = np.polyfit(x[warm_mask], y[warm_mask], 1)[0] if warm_mask.sum() > 30 else slope
        return pd.Series({"dev_temp_slope": slope, "dev_temp_warm_slope": warm_slope})

    print("  Computing per-device temperature response...")
    temp_coefs = df.loc[train_mask].groupby("deviceId").apply(
        calc_temp_coefs, include_groups=False
    ).reset_index()
    df = df.merge(temp_coefs, on="deviceId", how="left")
    df["dev_temp_slope"] = df["dev_temp_slope"].fillna(0.0)
    df["dev_temp_warm_slope"] = df["dev_temp_warm_slope"].fillna(0.0)

    # Fill remaining NaN in device stats
    for col in ["dev_mean_x2", "dev_std_x2", "dev_cv_x2", "dev_median_x2",
                "dev_min_x2", "dev_max_x2", "dev_mean_x1", "dev_mean_t1", "dev_std_t1",
                "dt_mean_x2", "dt_std_x2"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"  Total columns: {len(df.columns)}")

    # ==========================================
    # 3. Define feature columns
    # ==========================================
    feature_cols = [
        # Raw sensors
        "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9",
        "t10", "t11", "t12", "t13", "x1", "x3",
        # Weather
        "temperature", "apparent_temperature", "cloud_cover",
        "wind_speed_10m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "sunshine_duration", "rain_sum", "rain",
        "showers", "weather_code", "uv_index_max", "precipitation_probability",
        # Device
        "deviceType",
        # Time
        "sin_hour", "cos_hour", "sin_month", "cos_month",
        "sin_dow", "cos_dow", "sin_doy", "cos_doy",
        "is_night", "is_weekend",
        "hour", "month", "dayofweek", "day_of_year",
        # Engineered deltas
        "delta_T_out", "delta_load_hex", "delta_source_hex",
        "delta_load_3_4", "delta_weather_internal", "T_app_diff",
        "heating_demand", "cooling_demand", "hex_cross",
        # Non-linear temperature
        "temperature_sq", "temp_warm", "temp_cold", "temp_x_night",
        # DeviceType interactions
        "is_dtype_7", "is_dtype_11", "is_dtype_19",
        "temp_x_dtype_7", "temp_x_dtype_11", "temp_x_dtype_19",
        # Rolling features (6h, 24h, 48h)
        "temperature_roll6h", "temperature_roll24h", "temperature_roll48h",
        "temperature_roll168h",
        "t1_roll6h", "t1_roll24h", "t1_roll48h",
        "t2_roll6h", "t2_roll24h", "t2_roll48h",
        "t10_roll6h", "t10_roll24h", "t10_roll48h",
        "x1_roll6h", "x1_roll24h", "x1_roll48h",
        "temp_roll_std_24h",
        # Lag features
        "t2_lag_1h", "t2_lag_3h", "x1_lag_1h", "x1_rolling_3h",
        # Weather rolling
        "sunshine_rolling_6h", "sunshine_rolling_24h",
        "rain_last_24h", "hex_cross_rolling_6h", "temp_delta_12h",
        # 5h delta/rolling for top features
        "T_app_diff_delta_5h", "T_app_diff_rmean_5h",
        "temperature_delta_5h", "temperature_rmean_5h",
        "delta_T_out_delta_5h", "delta_T_out_rmean_5h",
        "hex_cross_delta_5h", "hex_cross_rmean_5h",
        # *** Per-device MONTHLY profile — most important ***
        "dev_month_x2", "dev_month_x2_std", "dev_month_temp_mean",
        "temp_dev_from_monthly",
        # Per-device static
        "dev_mean_x2", "dev_std_x2", "dev_cv_x2", "dev_median_x2",
        "dev_min_x2", "dev_max_x2",
        "dev_mean_x1", "dev_mean_t1", "dev_std_t1",
        "dt_mean_x2", "dt_std_x2",
        "dev_hour_x2", "dev_dow_x2",
        "dev_temp_slope", "dev_temp_warm_slope",
        # Geolocation
        "latitude", "longitude",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Features: {len(feature_cols)}")

    # ==========================================
    # 4. Sample weights — strong recency bias
    # ==========================================
    # Weight most recent months highest: they're closest to prediction period
    # April 2025 (month 4) = 3.0, March = 2.0, Oct/Nov = 2.0, winter = 0.7
    month_weights = {1: 0.6, 2: 0.6, 3: 1.5, 4: 3.0, 10: 2.0, 11: 1.5, 12: 0.6}
    df["sample_weight"] = df["month"].map(month_weights).fillna(1.0)
    # Also up-weight by year (2025 data > 2024)
    if df["year"].max() >= 2025:
        df.loc[df["year"] == 2025, "sample_weight"] *= 1.5

    # ==========================================
    # 5. Train per-deviceType LGBMs (multi-seed ensemble)
    # ==========================================
    print("\n[3/8] Training per-deviceType LGBMs (3 seeds)...")
    device_types = sorted(df["deviceType"].unique())
    models_by_seed = {seed: {} for seed in LGBM_SEEDS}
    oof_preds = pd.Series(index=df.index, dtype=float)

    predict_mask = df["period"].isin(["valid", "test"])

    for dt in device_types:
        dt_mask = df["deviceType"] == dt
        dt_train = dt_mask & train_mask
        dt_pred = dt_mask & predict_mask

        n_train = dt_train.sum()
        n_pred = dt_pred.sum()
        print(f"\n  DeviceType {dt}: train={n_train:,}, predict={n_pred:,}")

        X_tr = df.loc[dt_train, feature_cols].values.astype(np.float32)
        y_tr = df.loc[dt_train, "x2"].values.astype(np.float32)
        w_tr = df.loc[dt_train, "sample_weight"].values.astype(np.float32)
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)

        # OOF predictions (use first seed for bias estimation)
        oof = np.zeros(n_train, dtype=np.float32)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
            m = LGBMRegressor(
                objective="mae",
                n_estimators=3000,
                learning_rate=0.015,
                num_leaves=255,
                max_depth=10,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.05,
                reg_lambda=0.5,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            m.fit(X_tr[tr_idx], y_tr[tr_idx], sample_weight=w_tr[tr_idx])
            oof[val_idx] = m.predict(X_tr[val_idx])

        oof_preds.loc[dt_train] = oof
        oof_mae = np.abs(y_tr - oof).mean()
        print(f"    OOF MAE (hourly): {oof_mae:.6f}")

        # Train final models — one per seed
        X_p = df.loc[dt_pred, feature_cols].values.astype(np.float32)
        X_p = np.nan_to_num(X_p, nan=0.0, posinf=0.0, neginf=0.0)
        seed_preds = []

        for seed in LGBM_SEEDS:
            model = LGBMRegressor(
                objective="mae",
                n_estimators=3000,
                learning_rate=0.015,
                num_leaves=255,
                max_depth=10,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.05,
                reg_lambda=0.5,
                n_jobs=-1,
                random_state=seed,
                verbose=-1,
            )
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            models_by_seed[seed][dt] = model
            seed_preds.append(model.predict(X_p).clip(0.0))

        # Average across seeds
        df.loc[dt_pred, "x2_pred"] = np.mean(seed_preds, axis=0)

        # Feature importance (first seed)
        imp = pd.Series(
            models_by_seed[LGBM_SEEDS[0]][dt].feature_importances_,
            index=feature_cols
        ).sort_values(ascending=False)
        print(f"    Top 5 features: {list(imp.head(5).index)}")

        del X_tr, y_tr, w_tr, X_p
        gc.collect()

    # ==========================================
    # 6. Seasonal per-device bias correction (device × month)
    # ==========================================
    print("\n[4/8] Computing seasonal per-device bias correction...")
    df.loc[train_mask, "oof_pred"] = oof_preds[train_mask]
    df.loc[train_mask, "oof_residual"] = df.loc[train_mask, "x2"] - df.loc[train_mask, "oof_pred"]

    # Flat device-level bias (fallback)
    dev_bias = df.loc[train_mask].groupby("deviceId")["oof_residual"].mean().rename("dev_bias")
    print(f"  Flat device bias: mean={dev_bias.mean():.6f}, std={dev_bias.std():.6f}")

    # *** Seasonal per-device × month-of-year bias ***
    # This is key: for each device, how much does the model under/over-predict in each month?
    dev_month_bias = (
        df.loc[train_mask]
        .groupby(["deviceId", "month"])["oof_residual"]
        .mean()
        .rename("dev_month_bias")
        .reset_index()
    )
    n_month_corrections = len(dev_month_bias)
    print(f"  Seasonal bias entries: {n_month_corrections} (device×month pairs)")

    # Merge flat bias first (fallback), then seasonal override
    df = df.merge(dev_bias, on="deviceId", how="left")
    df["dev_bias"] = df["dev_bias"].fillna(0.0)

    df = df.merge(dev_month_bias, on=["deviceId", "month"], how="left")
    # Use seasonal bias where available, fall back to flat device bias
    df["dev_month_bias"] = df["dev_month_bias"].fillna(df["dev_bias"])

    # Apply seasonal bias correction to predictions
    df.loc[predict_mask, "x2_pred"] = (
        df.loc[predict_mask, "x2_pred"] + df.loc[predict_mask, "dev_month_bias"]
    ).clip(0.0)

    # ==========================================
    # 7. Aggregate to monthly
    # ==========================================
    print("\n[5/8] Aggregating to monthly predictions...")
    pred_df = df.loc[predict_mask, ["deviceId", "year", "month", "x2_pred"]].copy()

    monthly = (
        pred_df
        .groupby(["deviceId", "year", "month"])["x2_pred"]
        .mean()
        .reset_index()
        .rename(columns={"x2_pred": "prediction"})
    )

    # Filter to forecast months
    monthly = monthly[
        (monthly["year"] == FORECAST_YEAR)
        & (monthly["month"].isin(FORECAST_MONTHS))
    ].reset_index(drop=True)

    # Ensure all devices × months present
    device_ids = sorted(df["deviceId"].unique())
    full_index = pd.DataFrame([
        {"deviceId": d, "year": FORECAST_YEAR, "month": m}
        for d in device_ids for m in FORECAST_MONTHS
    ])
    monthly = full_index.merge(monthly, on=["deviceId", "year", "month"], how="left")

    # Fill missing with device monthly mean, then device mean fallback
    dev_month_map = dev_month_stats.set_index(["deviceId", "month"])["dev_month_x2"]
    dev_means_map = dev_stats.set_index("deviceId")["dev_mean_x2"]

    missing_mask = monthly["prediction"].isna()
    if missing_mask.any():
        print(f"  Filling {missing_mask.sum()} missing predictions")
        for idx in monthly[missing_mask].index:
            did = monthly.loc[idx, "deviceId"]
            mo = monthly.loc[idx, "month"]
            val = dev_month_map.get((did, mo), dev_means_map.get(did, dev_means_map.median()))
            monthly.loc[idx, "prediction"] = val

    monthly["prediction"] = monthly["prediction"].clip(lower=0.0)

    # ==========================================
    # 8. Internal validation (OOF monthly MAE)
    # ==========================================
    print("\n[6/8] Internal validation (OOF monthly MAE)...")
    train_df = df.loc[train_mask, ["deviceId", "year", "month", "x2", "oof_pred", "dev_month_bias"]].copy()
    train_df["oof_corrected"] = (train_df["oof_pred"] + train_df["dev_month_bias"]).clip(0.0)
    train_monthly = train_df.groupby(["deviceId", "year", "month"]).agg(
        actual=("x2", "mean"), predicted=("oof_corrected", "mean")
    ).reset_index()
    oof_monthly_mae = (train_monthly["actual"] - train_monthly["predicted"]).abs().mean()
    print(f"  OOF monthly MAE (all training months): {oof_monthly_mae:.6f}")

    # April only (closest to prediction)
    april = train_monthly[train_monthly["month"] == 4]
    if len(april) > 0:
        april_mae = (april["actual"] - april["predicted"]).abs().mean()
        print(f"  OOF monthly MAE (April only): {april_mae:.6f}")

    # Summer months in training data (if available)
    summer = train_monthly[train_monthly["month"].isin([5, 6, 7, 8, 9, 10])]
    if len(summer) > 0:
        summer_mae = (summer["actual"] - summer["predicted"]).abs().mean()
        print(f"  OOF monthly MAE (summer months, same as forecast): {summer_mae:.6f}")

    # ==========================================
    # 9. Save submission
    # ==========================================
    print("\n[7/8] Saving submission...")
    submission = (
        monthly[["deviceId", "year", "month", "prediction"]]
        .sort_values(["deviceId", "year", "month"])
        .reset_index(drop=True)
    )
    submission["year"] = submission["year"].astype(int)
    submission["month"] = submission["month"].astype(int)

    n_expected = len(device_ids) * len(FORECAST_MONTHS)
    print(f"  Rows: {len(submission)} / {n_expected}")
    print(f"\n  Prediction stats:\n{submission['prediction'].describe()}")
    print(f"\n  Mean prediction per month:")
    print(submission.groupby("month")["prediction"].mean())

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
