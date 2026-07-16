import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DEFAULT_DATE_CANDIDATES = ["date", "day", "dt"]
DEFAULT_ORIGIN_CANDIDATES = ["origin", "o", "origin_id", "from_id"]
DEFAULT_DEST_CANDIDATES = ["destination", "d", "destination_id", "to_id"]
DEFAULT_TARGET_CANDIDATES = ["flow", "trip_count", "count", "y"]
DEFAULT_NEWS_DATE_CANDIDATES = ["date", "day", "dt", "published_at", "publish_date", "timestamp", "time"]
DEFAULT_NEWS_TEXT_CANDIDATES = ["text", "content", "title", "headline", "news"]
DEFAULT_NEWS_SCORE_CANDIDATES = ["sentiment", "score", "impact", "news_score"]


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
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding helps models capture periodic mobility patterns.
    df["dow_sin"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (df["month"] - 1.0) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (df["month"] - 1.0) / 12.0)
    df["doy_sin"] = np.sin(2.0 * np.pi * (df["day_of_year"] - 1.0) / 366.0)
    df["doy_cos"] = np.cos(2.0 * np.pi * (df["day_of_year"] - 1.0) / 366.0)

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


def load_news_daily_features(
    news_path: Path,
    news_date_col: str | None,
    news_text_col: str | None,
    news_score_col: str | None,
) -> pd.DataFrame:
    if not news_path.exists():
        raise ValueError(f"News file not found: {news_path}")

    # Auto-detect delimiter so CSV/TSV both work.
    news_df = pd.read_csv(news_path, sep=None, engine="python")

    date_col = pick_column(news_df, news_date_col, DEFAULT_NEWS_DATE_CANDIDATES, "news-date")

    text_col: str | None = None
    score_col: str | None = None

    try:
        text_col = pick_column(news_df, news_text_col, DEFAULT_NEWS_TEXT_CANDIDATES, "news-text")
    except ValueError:
        text_col = None

    try:
        score_col = pick_column(news_df, news_score_col, DEFAULT_NEWS_SCORE_CANDIDATES, "news-score")
    except ValueError:
        score_col = None

    if text_col is None and score_col is None:
        raise ValueError(
            "Could not infer news text/score columns. Provide --news-text-col and/or --news-score-col."
        )

    news_df[date_col] = pd.to_datetime(news_df[date_col], errors="coerce")
    if news_df[date_col].isna().all():
        raise ValueError("News date parsing failed for all rows. Check --news-date-col and date format.")

    news_df = news_df.dropna(subset=[date_col]).copy()
    news_df["news_date"] = news_df[date_col].dt.normalize()

    agg_spec: dict[str, str] = {}
    if text_col is not None:
        news_df["_news_text_len"] = news_df[text_col].astype(str).str.len()
        agg_spec["_news_text_len"] = "mean"
    if score_col is not None:
        news_df["_news_score"] = pd.to_numeric(news_df[score_col], errors="coerce")
        agg_spec["_news_score"] = "mean"

    daily = news_df.groupby("news_date", as_index=False).agg(agg_spec) if agg_spec else pd.DataFrame()
    count_df = news_df.groupby("news_date", as_index=False).size().rename(columns={"size": "news_article_count"})

    if daily.empty:
        daily = count_df
    else:
        daily = daily.merge(count_df, on="news_date", how="left")

    rename_map = {
        "_news_text_len": "news_text_len_mean",
        "_news_score": "news_score_mean",
    }
    daily = daily.rename(columns=rename_map)

    # Ensure deterministic order and daily continuity before creating lagged news features.
    daily = daily.sort_values("news_date").reset_index(drop=True)
    full_range = pd.date_range(start=daily["news_date"].min(), end=daily["news_date"].max(), freq="D")
    daily = daily.set_index("news_date").reindex(full_range).rename_axis("news_date").reset_index()

    base_cols = [c for c in ["news_article_count", "news_text_len_mean", "news_score_mean"] if c in daily.columns]
    daily[base_cols] = daily[base_cols].fillna(0.0)

    for col in base_cols:
        daily[f"{col}_lag1"] = daily[col].shift(1)
        daily[f"{col}_lag3_mean"] = daily[col].shift(1).rolling(window=3).mean()
        daily[f"{col}_lag7_mean"] = daily[col].shift(1).rolling(window=7).mean()

    lagged_cols = [c for c in daily.columns if c.endswith("_lag1") or c.endswith("_lag3_mean") or c.endswith("_lag7_mean")]
    daily[lagged_cols] = daily[lagged_cols].fillna(0.0)

    keep_cols = ["news_date"] + lagged_cols
    return daily[keep_cols]


def run_pipeline(
    data_path: Path,
    out_dir: Path,
    model_name: str,
    date_col: str | None,
    origin_col: str | None,
    dest_col: str | None,
    target_col: str | None,
    news_path: Path | None,
    news_date_col: str | None,
    news_text_col: str | None,
    news_score_col: str | None,
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
        "day_of_year",
        "month",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
    ]

    if news_path is not None:
        news_daily = load_news_daily_features(
            news_path=news_path,
            news_date_col=news_date_col,
            news_text_col=news_text_col,
            news_score_col=news_score_col,
        )
        featured["news_date"] = featured[date_col].dt.normalize()
        featured = featured.merge(news_daily, on="news_date", how="left")
        news_feature_cols = [c for c in news_daily.columns if c != "news_date"]
        feature_cols.extend(news_feature_cols)
        featured[news_feature_cols] = featured[news_feature_cols].fillna(0.0)
        featured = featured.drop(columns=["news_date"])

    modeled = featured.dropna(subset=feature_cols + [target_col]).copy()
    train_df, valid_df = split_train_valid(modeled, date_col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(float)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col].astype(float)

    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=450,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
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
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "used_news_features": bool(news_path is not None),
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
    parser.add_argument(
        "--model",
        type=str,
        choices=["hgbt", "random_forest"],
        default="hgbt",
        help="Model type. Default 'hgbt' is usually better for periodic patterns.",
    )
    parser.add_argument("--date-col", type=str, default=None, help="Date column name")
    parser.add_argument("--origin-col", type=str, default=None, help="Origin column name")
    parser.add_argument("--dest-col", type=str, default=None, help="Destination column name")
    parser.add_argument("--target-col", type=str, default=None, help="Flow target column name")
    parser.add_argument("--news-path", type=Path, default=None, help="Optional path to news CSV/TSV")
    parser.add_argument("--news-date-col", type=str, default=None, help="News date column name")
    parser.add_argument("--news-text-col", type=str, default=None, help="News text/title column name")
    parser.add_argument("--news-score-col", type=str, default=None, help="News score/sentiment column name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        out_dir=args.out_dir,
        model_name=args.model,
        date_col=args.date_col,
        origin_col=args.origin_col,
        dest_col=args.dest_col,
        target_col=args.target_col,
        news_path=args.news_path,
        news_date_col=args.news_date_col,
        news_text_col=args.news_text_col,
        news_score_col=args.news_score_col,
    )


if __name__ == "__main__":
    main()
