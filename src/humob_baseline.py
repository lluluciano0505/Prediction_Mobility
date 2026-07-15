import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_DATE_CANDIDATES = ["date", "day", "dt"]
DEFAULT_ORIGIN_CANDIDATES = ["origin", "o", "origin_id", "from_id"]
DEFAULT_DEST_CANDIDATES = ["destination", "d", "destination_id", "to_id"]
DEFAULT_TARGET_CANDIDATES = ["flow", "trip_count", "count", "y"]


def pick_column(df: pd.DataFrame, explicit_name: str | None, candidates: list[str], label: str) -> str:
    if explicit_name:
        if explicit_name not in df.columns:
            raise ValueError(f"Explicit {label} column '{explicit_name}' not found in data columns: {list(df.columns)}")
        return explicit_name

    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    raise ValueError(
        f"Could not infer {label} column. Provide --{label}-col explicitly. "
        f"Available columns: {list(df.columns)}"
    )


def build_features(df: pd.DataFrame, date_col: str, origin_col: str, dest_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError("Some dates could not be parsed. Check date format in input data.")

    df = df.sort_values([origin_col, dest_col, date_col]).reset_index(drop=True)
    grp = df.groupby([origin_col, dest_col], sort=False)[target_col]

    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = grp.shift(lag)

    rolling = grp.shift(1).rolling(window=7)
    df["lag7_mean"] = rolling.mean().reset_index(level=[0, 1], drop=True)
    df["lag7_std"] = rolling.std().reset_index(level=[0, 1], drop=True)

    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df


def split_train_valid(df: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Hold out the challenge test period as validation for local benchmarking.
    valid_start = pd.Timestamp("2024-02-01")
    valid_end = pd.Timestamp("2024-03-31")

    valid_mask = (df[date_col] >= valid_start) & (df[date_col] <= valid_end)
    valid_df = df.loc[valid_mask].copy()
    train_df = df.loc[~valid_mask].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Train/validation split resulted in empty partition. Check date range and input file.")

    return train_df, valid_df


def run_pipeline(
    data_path: Path,
    out_dir: Path,
    date_col: str | None,
    origin_col: str | None,
    dest_col: str | None,
    target_col: str | None,
) -> None:
    df = pd.read_csv(data_path, sep="\t")

    date_col = pick_column(df, date_col, DEFAULT_DATE_CANDIDATES, "date")
    origin_col = pick_column(df, origin_col, DEFAULT_ORIGIN_CANDIDATES, "origin")
    dest_col = pick_column(df, dest_col, DEFAULT_DEST_CANDIDATES, "dest")
    target_col = pick_column(df, target_col, DEFAULT_TARGET_CANDIDATES, "target")

    featured = build_features(df, date_col, origin_col, dest_col, target_col)

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag7_mean",
        "lag7_std",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    modeled = featured.dropna(subset=feature_cols + [target_col]).copy()
    train_df, valid_df = split_train_valid(modeled, date_col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(float)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col].astype(float)

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    preds = np.clip(preds, 0.0, None)

    mae = float(mean_absolute_error(y_valid, preds))
    rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "mae": mae,
        "rmse": rmse,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_out = valid_df[[date_col, origin_col, dest_col, target_col]].copy()
    pred_out["prediction"] = preds
    pred_out.to_csv(out_dir / "validation_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs in: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HuMob Challenge 2026 baseline pipeline")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to humob2026-dataset.tsv")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for outputs")
    parser.add_argument("--date-col", type=str, default=None, help="Date column name")
    parser.add_argument("--origin-col", type=str, default=None, help="Origin column name")
    parser.add_argument("--dest-col", type=str, default=None, help="Destination column name")
    parser.add_argument("--target-col", type=str, default=None, help="Flow target column name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        out_dir=args.out_dir,
        date_col=args.date_col,
        origin_col=args.origin_col,
        dest_col=args.dest_col,
        target_col=args.target_col,
    )


if __name__ == "__main__":
    main()
