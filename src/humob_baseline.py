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

    shifted = grp.shift(1)
    rolling_grp = shifted.groupby([df[origin_col], df[dest_col]], sort=False)
    df["lag7_mean"] = rolling_grp.transform(lambda s: s.rolling(window=7).mean())
    df["lag7_std"] = rolling_grp.transform(lambda s: s.rolling(window=7).std())

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
        unique_dates = sorted(pd.to_datetime(df[date_col]).dropna().unique())
        if len(unique_dates) < 10:
            raise ValueError("Train/validation split resulted in empty partition. Need more dated rows.")

        # Fallback for local datasets without Feb-Mar labels: use the last 20% dates as validation.
        cutoff_idx = max(1, int(len(unique_dates) * 0.8))
        cutoff_date = pd.Timestamp(unique_dates[cutoff_idx])
        valid_mask = pd.to_datetime(df[date_col]) >= cutoff_date
        valid_df = df.loc[valid_mask].copy()
        train_df = df.loc[~valid_mask].copy()

        if train_df.empty or valid_df.empty:
            raise ValueError("Fallback split still produced an empty partition.")

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


def load_poi_features(poi_path: Path, poi_cell_col: str | None) -> tuple[pd.DataFrame, str, list[str]]:
    if not poi_path.exists():
        raise ValueError(f"POI file not found: {poi_path}")

    poi_df = pd.read_csv(poi_path, sep=None, engine="python")
    cell_col = pick_column(poi_df, poi_cell_col, ["cell_id", "grid_id", "region_id", "cell"], "poi-cell")

    numeric_cols = [
        c
        for c in poi_df.columns
        if c != cell_col and pd.api.types.is_numeric_dtype(poi_df[c])
    ]
    if not numeric_cols:
        raise ValueError("No numeric POI columns found in POI file.")

    poi_df = poi_df[[cell_col] + numeric_cols].copy()
    poi_df[cell_col] = poi_df[cell_col].astype(str)
    poi_df[numeric_cols] = poi_df[numeric_cols].fillna(0.0)
    return poi_df, cell_col, numeric_cols


def load_satellite_daily_features(
    satellite_path: Path,
    satellite_date_col: str | None,
    satellite_cell_col: str | None,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    if not satellite_path.exists():
        raise ValueError(f"Satellite file not found: {satellite_path}")

    sat_df = pd.read_csv(satellite_path, sep=None, engine="python")
    date_col = pick_column(sat_df, satellite_date_col, ["date", "day", "dt"], "satellite-date")
    cell_col = pick_column(sat_df, satellite_cell_col, ["cell_id", "grid_id", "region_id", "cell"], "satellite-cell")

    sat_df = sat_df.copy()
    sat_df[date_col] = pd.to_datetime(sat_df[date_col], errors="coerce").dt.normalize()
    sat_df[cell_col] = sat_df[cell_col].astype(str)

    numeric_cols = [
        c
        for c in sat_df.columns
        if c not in {date_col, cell_col} and pd.api.types.is_numeric_dtype(sat_df[c])
    ]
    if not numeric_cols:
        raise ValueError("No numeric satellite columns found in satellite file.")

    sat_df = sat_df[[date_col, cell_col] + numeric_cols].copy()
    sat_df[numeric_cols] = sat_df[numeric_cols].fillna(0.0)
    sat_df = sat_df.groupby([date_col, cell_col], as_index=False)[numeric_cols].mean()
    return sat_df, date_col, cell_col, numeric_cols


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
    news_embedding_path: Path | None,
    news_embedding_date_col: str | None,
    news_embedding_cell_col: str | None,
    stage: str,
    poi_path: Path | None,
    poi_cell_col: str | None,
    satellite_path: Path | None,
    satellite_date_col: str | None,
    satellite_cell_col: str | None,
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

    stage_rank = {"baseline": 0, "poi": 1, "satellite": 2, "news": 3}
    current_rank = stage_rank[stage]

    use_poi = current_rank >= stage_rank["poi"]
    use_satellite = current_rank >= stage_rank["satellite"]
    use_news = current_rank >= stage_rank["news"]

    if use_poi and poi_path is None:
        raise ValueError("Stage requires POI features. Provide --poi-path.")
    if use_satellite and satellite_path is None:
        raise ValueError("Stage requires satellite features. Provide --satellite-path.")
    if use_news and news_path is None:
        raise ValueError("Stage 'news' requires --news-path.")

    if use_poi and poi_path is not None:
        poi_df, poi_cell, poi_cols = load_poi_features(poi_path=poi_path, poi_cell_col=poi_cell_col)
        featured[origin_col] = featured[origin_col].astype(str)
        featured[dest_col] = featured[dest_col].astype(str)

        poi_o = poi_df.rename(columns={poi_cell: origin_col, **{c: f"poi_o_{c}" for c in poi_cols}})
        poi_d = poi_df.rename(columns={poi_cell: dest_col, **{c: f"poi_d_{c}" for c in poi_cols}})

        featured = featured.merge(poi_o, on=[origin_col], how="left")
        featured = featured.merge(poi_d, on=[dest_col], how="left")

        poi_feature_cols = [f"poi_o_{c}" for c in poi_cols] + [f"poi_d_{c}" for c in poi_cols]
        featured[poi_feature_cols] = featured[poi_feature_cols].fillna(0.0)
        feature_cols.extend(poi_feature_cols)

    if use_satellite and satellite_path is not None:
        sat_df, sat_date, sat_cell, sat_cols = load_satellite_daily_features(
            satellite_path=satellite_path,
            satellite_date_col=satellite_date_col,
            satellite_cell_col=satellite_cell_col,
        )

        featured["sat_date"] = pd.to_datetime(featured[date_col], errors="coerce").dt.normalize()
        featured[origin_col] = featured[origin_col].astype(str)
        featured[dest_col] = featured[dest_col].astype(str)

        sat_o = sat_df.rename(
            columns={sat_date: "sat_date", sat_cell: origin_col, **{c: f"sat_o_{c}" for c in sat_cols}}
        )
        sat_d = sat_df.rename(
            columns={sat_date: "sat_date", sat_cell: dest_col, **{c: f"sat_d_{c}" for c in sat_cols}}
        )

        featured = featured.merge(sat_o, on=["sat_date", origin_col], how="left")
        featured = featured.merge(sat_d, on=["sat_date", dest_col], how="left")

        sat_feature_cols = [f"sat_o_{c}" for c in sat_cols] + [f"sat_d_{c}" for c in sat_cols]
        featured[sat_feature_cols] = featured[sat_feature_cols].fillna(0.0)
        feature_cols.extend(sat_feature_cols)
        featured = featured.drop(columns=["sat_date"])

    if use_news and news_path is not None:
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

    if use_news and news_embedding_path is not None:
        emb_df = pd.read_csv(news_embedding_path)
        emb_date_col = pick_column(
            emb_df,
            news_embedding_date_col,
            ["news_date", "date", "day", "dt"],
            "news-embedding-date",
        )
        emb_cell_col = pick_column(
            emb_df,
            news_embedding_cell_col,
            ["news_cell", "cell_id", "grid_id", "region_id", "h3", "h3_id"],
            "news-embedding-cell",
        )

        emb_df = emb_df.copy()
        emb_df[emb_date_col] = pd.to_datetime(emb_df[emb_date_col], errors="coerce").dt.normalize()
        emb_df[emb_cell_col] = emb_df[emb_cell_col].astype(str)

        emb_cols = [
            c
            for c in emb_df.columns
            if c not in {emb_date_col, emb_cell_col}
            and pd.api.types.is_numeric_dtype(emb_df[c])
        ]
        if not emb_cols:
            raise ValueError("No numeric embedding columns found in news embedding file.")

        emb_src = emb_df[[emb_date_col, emb_cell_col] + emb_cols].copy()

        featured["news_date"] = pd.to_datetime(featured[date_col], errors="coerce").dt.normalize()
        featured[origin_col] = featured[origin_col].astype(str)
        featured[dest_col] = featured[dest_col].astype(str)

        emb_o = emb_src.rename(
            columns={
                emb_date_col: "news_date",
                emb_cell_col: origin_col,
                **{c: f"news_emb_o_{c}" for c in emb_cols},
            }
        )
        emb_d = emb_src.rename(
            columns={
                emb_date_col: "news_date",
                emb_cell_col: dest_col,
                **{c: f"news_emb_d_{c}" for c in emb_cols},
            }
        )

        featured = featured.merge(emb_o, on=["news_date", origin_col], how="left")
        featured = featured.merge(emb_d, on=["news_date", dest_col], how="left")

        emb_feature_cols = [f"news_emb_o_{c}" for c in emb_cols] + [f"news_emb_d_{c}" for c in emb_cols]
        feature_cols.extend(emb_feature_cols)
        featured[emb_feature_cols] = featured[emb_feature_cols].fillna(0.0)
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
        "stage": stage,
        "mae": mae,
        "rmse": rmse,
        "used_poi_features": bool(use_poi and poi_path is not None),
        "used_satellite_features": bool(use_satellite and satellite_path is not None),
        "used_news_features": bool(use_news and news_path is not None),
        "used_news_embedding_features": bool(use_news and news_embedding_path is not None),
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
    parser.add_argument(
        "--stage",
        type=str,
        choices=["baseline", "poi", "satellite", "news"],
        default="baseline",
        help="Feature stage: baseline -> +POI -> +satellite -> +news",
    )
    parser.add_argument("--poi-path", type=Path, default=None, help="Path to grid-level POI feature table")
    parser.add_argument("--poi-cell-col", type=str, default=None, help="Cell id column in POI file")
    parser.add_argument("--satellite-path", type=Path, default=None, help="Path to date-cell satellite feature table")
    parser.add_argument("--satellite-date-col", type=str, default=None, help="Date column in satellite file")
    parser.add_argument("--satellite-cell-col", type=str, default=None, help="Cell id column in satellite file")
    parser.add_argument("--news-path", type=Path, default=None, help="Optional path to news CSV/TSV")
    parser.add_argument("--news-date-col", type=str, default=None, help="News date column name")
    parser.add_argument("--news-text-col", type=str, default=None, help="News text/title column name")
    parser.add_argument("--news-score-col", type=str, default=None, help="News score/sentiment column name")
    parser.add_argument(
        "--news-embedding-path",
        type=Path,
        default=None,
        help="Optional path to aggregated date-cell embedding CSV",
    )
    parser.add_argument(
        "--news-embedding-date-col",
        type=str,
        default=None,
        help="Date column name in embedding CSV (for example: news_date)",
    )
    parser.add_argument(
        "--news-embedding-cell-col",
        type=str,
        default=None,
        help="Cell column name in embedding CSV (for example: news_cell)",
    )
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
        news_embedding_path=args.news_embedding_path,
        news_embedding_date_col=args.news_embedding_date_col,
        news_embedding_cell_col=args.news_embedding_cell_col,
        stage=args.stage,
        poi_path=args.poi_path,
        poi_cell_col=args.poi_cell_col,
        satellite_path=args.satellite_path,
        satellite_date_col=args.satellite_date_col,
        satellite_cell_col=args.satellite_cell_col,
    )


if __name__ == "__main__":
    main()
