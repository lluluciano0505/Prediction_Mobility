import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


# HuMob grid metadata from official docs
GRID_MIN_LON = 136.029
GRID_MAX_LON = 138.042
GRID_MIN_LAT = 36.203
GRID_MAX_LAT = 37.646
GRID_X_COUNT = 100
GRID_Y_COUNT = 70


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def latlon_to_cell_id(lat: float, lon: float) -> str | None:
    if lon < GRID_MIN_LON or lon > GRID_MAX_LON or lat < GRID_MIN_LAT or lat > GRID_MAX_LAT:
        return None

    x_ratio = (lon - GRID_MIN_LON) / (GRID_MAX_LON - GRID_MIN_LON)
    y_ratio = (lat - GRID_MIN_LAT) / (GRID_MAX_LAT - GRID_MIN_LAT)

    x = clamp(int(x_ratio * GRID_X_COUNT) + 1, 1, GRID_X_COUNT)
    y = clamp(int(y_ratio * GRID_Y_COUNT) + 1, 1, GRID_Y_COUNT)
    return f"{y}_{x}"


def cell_centroid(cell_id: str) -> tuple[float, float]:
    y_str, x_str = cell_id.split("_")
    y = int(y_str)
    x = int(x_str)

    lon_step = (GRID_MAX_LON - GRID_MIN_LON) / GRID_X_COUNT
    lat_step = (GRID_MAX_LAT - GRID_MIN_LAT) / GRID_Y_COUNT

    lon = GRID_MIN_LON + (x - 0.5) * lon_step
    lat = GRID_MIN_LAT + (y - 0.5) * lat_step
    return lat, lon


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    subprocess.run(cmd, check=True)


def collect_poi(args: argparse.Namespace) -> None:
    cmd_args = ["--out-path", str(args.out_path)]
    if args.with_elevation:
        cmd_args.extend(["--with-elevation", "--max-elevation-cells", str(args.max_elevation_cells)])
    run_module("src.collect_grid_static_features", cmd_args)


def collect_news(args: argparse.Namespace) -> None:
    cmd_args = [
        "--query",
        args.query,
        "--max-items",
        str(args.max_items),
        "--days-back",
        str(args.days_back),
        "--out-path",
        str(args.out_path),
    ]
    run_module("src.news_collect_to_cells", cmd_args)


def collect_satellite(args: argparse.Namespace) -> None:
    sep = "\t" if args.in_path.suffix.lower() == ".tsv" else ","
    src_df = pd.read_csv(args.in_path, sep=sep)

    if args.date_col not in src_df.columns:
        raise ValueError(f"Missing required column: {args.date_col}")

    if args.cell_col:
        if args.cell_col not in src_df.columns:
            raise ValueError(f"Missing cell column: {args.cell_col}")
        cell_series = src_df[args.cell_col].astype(str)
    else:
        if not args.lat_col or not args.lon_col:
            raise ValueError("Use --cell-col, or provide both --lat-col and --lon-col")
        if args.lat_col not in src_df.columns or args.lon_col not in src_df.columns:
            raise ValueError("Latitude/longitude columns not found in input")

        cell_series = src_df.apply(
            lambda row: latlon_to_cell_id(lat=float(row[args.lat_col]), lon=float(row[args.lon_col])), axis=1
        )

    if args.feature_cols.strip():
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        missing = [c for c in feature_cols if c not in src_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
    else:
        excluded = {args.date_col, args.cell_col, args.lat_col, args.lon_col}
        feature_cols = [c for c in src_df.columns if c not in excluded and pd.api.types.is_numeric_dtype(src_df[c])]

    if not feature_cols:
        raise ValueError("No usable satellite feature columns found")

    out_df = pd.DataFrame(
        {
            "date": pd.to_datetime(src_df[args.date_col], errors="coerce").dt.date.astype("string"),
            "cell_id": cell_series.astype("string"),
        }
    )
    for col in feature_cols:
        out_df[col] = pd.to_numeric(src_df[col], errors="coerce")

    out_df = out_df.dropna(subset=["date", "cell_id"])
    out_df = out_df[out_df["cell_id"].str.contains("_")]

    grouped = out_df.groupby(["date", "cell_id"], as_index=False)[feature_cols].mean()
    grouped = grouped.sort_values(["date", "cell_id"]).reset_index(drop=True)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.out_path, sep="\t", index=False)

    print(
        json.dumps(
            {
                "mode": "satellite_import",
                "rows_in": int(len(src_df)),
                "rows_out": int(len(grouped)),
                "feature_cols": feature_cols,
                "out_path": str(args.out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def fetch_power_daily(lat: float, lon: float, start: str, end: str, parameters: str) -> dict:
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    query = {
        "parameters": parameters,
        "community": "RE",
        "longitude": f"{lon:.6f}",
        "latitude": f"{lat:.6f}",
        "start": start,
        "end": end,
        "format": "JSON",
    }
    req = Request(f"{base}?{urlencode(query)}", method="GET")
    with urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def fetch_open_meteo_cloud_daily(lat: float, lon: float, start_date: str, end_date: str) -> dict[str, float | None]:
    base = "https://archive-api.open-meteo.com/v1/archive"
    query = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": f"{start_date[0:4]}-{start_date[4:6]}-{start_date[6:8]}",
        "end_date": f"{end_date[0:4]}-{end_date[4:6]}-{end_date[6:8]}",
        "daily": "cloud_cover_mean",
        "timezone": "UTC",
    }
    req = Request(f"{base}?{urlencode(query)}", method="GET")
    with urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    payload = json.loads(body)

    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    clouds = daily.get("cloud_cover_mean", [])
    out: dict[str, float | None] = {}
    for d, c in zip(dates, clouds):
        out[str(d)] = float(c) if c is not None else None
    return out


def resolve_cell_ids(args: argparse.Namespace) -> list[str]:
    if args.cell_ids.strip():
        return [c.strip() for c in args.cell_ids.split(",") if c.strip()]

    if args.cell_source and Path(args.cell_source).exists():
        src_df = pd.read_csv(args.cell_source, sep="\t")
        if "cell_id" not in src_df.columns:
            raise ValueError("cell_source must contain a cell_id column")
        if args.only_nonzero and "poi_count_total" in src_df.columns:
            src_df = src_df[src_df["poi_count_total"] > 0]
        return src_df["cell_id"].astype(str).head(args.max_cells).tolist()

    cell_ids = [f"{y}_{x}" for y in range(1, GRID_Y_COUNT + 1) for x in range(1, GRID_X_COUNT + 1)]
    return cell_ids[: args.max_cells]


def merge_nightlight(out_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.nightlight_path:
        out_df["sat_night_light"] = None
        return out_df

    nightlight_path = Path(args.nightlight_path)
    sep = "\t" if nightlight_path.suffix.lower() == ".tsv" else ","
    nl = pd.read_csv(nightlight_path, sep=sep)

    required_cols = [args.nightlight_cell_col, args.nightlight_value_col]
    for col in required_cols:
        if col not in nl.columns:
            raise ValueError(f"Nightlight file missing column: {col}")

    if args.nightlight_date_col and args.nightlight_date_col in nl.columns:
        nl["_merge_date"] = pd.to_datetime(nl[args.nightlight_date_col], errors="coerce").dt.date.astype("string")
        nl = nl.dropna(subset=["_merge_date", args.nightlight_cell_col])
        nl_small = nl[["_merge_date", args.nightlight_cell_col, args.nightlight_value_col]].copy()
        nl_small = nl_small.rename(
            columns={
                "_merge_date": "date",
                args.nightlight_cell_col: "cell_id",
                args.nightlight_value_col: "sat_night_light",
            }
        )
        merged = out_df.merge(nl_small, on=["date", "cell_id"], how="left")
        return merged

    month_col = args.nightlight_month_col
    if month_col not in nl.columns:
        raise ValueError("Nightlight file must have either date column or month column")

    nl["_month"] = pd.to_datetime(nl[month_col], errors="coerce").dt.to_period("M")
    nl = nl.dropna(subset=["_month", args.nightlight_cell_col])
    out = out_df.copy()
    out["_month"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M")

    nl_small = nl[["_month", args.nightlight_cell_col, args.nightlight_value_col]].copy()
    nl_small = nl_small.rename(
        columns={args.nightlight_cell_col: "cell_id", args.nightlight_value_col: "sat_night_light"}
    )
    merged = out.merge(nl_small, on=["_month", "cell_id"], how="left").drop(columns=["_month"])
    return merged


def collect_satellite_crawl(args: argparse.Namespace) -> None:
    cell_ids = resolve_cell_ids(args)

    if not cell_ids:
        raise ValueError("No cells selected for satellite crawl")

    rows: list[dict] = []
    failed = 0
    for cell in cell_ids:
        try:
            lat, lon = cell_centroid(cell)
            payload = fetch_power_daily(
                lat=lat,
                lon=lon,
                start=args.start_date,
                end=args.end_date,
                parameters=args.parameters,
            )
            param_map = payload.get("properties", {}).get("parameter", {})
            if not param_map:
                continue

            date_keys = set()
            for _, series in param_map.items():
                date_keys.update(series.keys())

            for d in sorted(date_keys):
                out = {
                    "date": f"{d[0:4]}-{d[4:6]}-{d[6:8]}",
                    "cell_id": cell,
                }
                for param_name, series in param_map.items():
                    value = series.get(d)
                    out[f"sat_{param_name.lower()}"] = float(value) if value is not None and value > -900 else None
                rows.append(out)
        except Exception:
            failed += 1

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["date", "cell_id"]).reset_index(drop=True)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_path, sep="\t", index=False)

    print(
        json.dumps(
            {
                "mode": "satellite_crawl",
                "cells_requested": len(cell_ids),
                "cells_failed": failed,
                "rows_out": int(len(out_df)),
                "out_path": str(args.out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def collect_satellite_cloud_nightlight(args: argparse.Namespace) -> None:
    cell_ids = resolve_cell_ids(args)
    if not cell_ids:
        raise ValueError("No cells selected for satellite cloud/nightlight crawl")

    rows: list[dict] = []
    failed = 0
    for cell in cell_ids:
        try:
            lat, lon = cell_centroid(cell)
            power_payload = fetch_power_daily(
                lat=lat,
                lon=lon,
                start=args.start_date,
                end=args.end_date,
                parameters=args.parameters,
            )
            power_map = power_payload.get("properties", {}).get("parameter", {})
            cloud_map = fetch_open_meteo_cloud_daily(
                lat=lat,
                lon=lon,
                start_date=args.start_date,
                end_date=args.end_date,
            )

            date_keys = set(cloud_map.keys())
            for _, series in power_map.items():
                for d in series.keys():
                    date_keys.add(f"{d[0:4]}-{d[4:6]}-{d[6:8]}")

            for date_str in sorted(date_keys):
                out = {
                    "date": date_str,
                    "cell_id": cell,
                    "sat_cloud_cover_mean": cloud_map.get(date_str),
                }
                for param_name, series in power_map.items():
                    key = date_str.replace("-", "")
                    value = series.get(key)
                    out[f"sat_{param_name.lower()}"] = float(value) if value is not None and value > -900 else None
                rows.append(out)
        except Exception:
            failed += 1

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["date", "cell_id"]).reset_index(drop=True)

    out_df = merge_nightlight(out_df, args)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_path, sep="\t", index=False)

    print(
        json.dumps(
            {
                "mode": "satellite_cloud_nightlight",
                "cells_requested": len(cell_ids),
                "cells_failed": failed,
                "rows_out": int(len(out_df)),
                "nightlight_path": args.nightlight_path,
                "out_path": str(args.out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified data collection entrypoint")
    sub = parser.add_subparsers(dest="mode", required=True)

    poi = sub.add_parser("poi", help="Collect POI features with optional elevation")
    poi.add_argument("--out-path", type=Path, default=Path("data/features/grid_static_features.tsv"))
    poi.add_argument("--with-elevation", action="store_true")
    poi.add_argument("--max-elevation-cells", type=int, default=400)
    poi.set_defaults(func=collect_poi)

    news = sub.add_parser("news", help="Collect public news and map to grid cells")
    news.add_argument("--query", type=str, default="能登 OR 石川 地震 OR 交通 OR 観光")
    news.add_argument("--max-items", type=int, default=120)
    news.add_argument("--days-back", type=int, default=365)
    news.add_argument("--out-path", type=Path, default=Path("data/news/news_collected_cells.tsv"))
    news.set_defaults(func=collect_news)

    satellite = sub.add_parser("satellite", help="Transform existing satellite export into date-cell feature table")
    satellite.add_argument("--in-path", type=Path, required=True)
    satellite.add_argument("--out-path", type=Path, default=Path("data/features/satellite_cell_daily.tsv"))
    satellite.add_argument("--date-col", type=str, default="date")
    satellite.add_argument("--cell-col", type=str, default="")
    satellite.add_argument("--lat-col", type=str, default="lat")
    satellite.add_argument("--lon-col", type=str, default="lon")
    satellite.add_argument(
        "--feature-cols",
        type=str,
        default="ndvi,ndwi,ndbi,lst",
        help="Comma-separated satellite features to keep; empty string means infer numeric columns",
    )
    satellite.set_defaults(func=collect_satellite)

    sat_crawl = sub.add_parser("satellite-crawl", help="Crawl daily satellite-derived variables from NASA POWER API")
    sat_crawl.add_argument("--out-path", type=Path, default=Path("data/features/satellite_power_daily.tsv"))
    sat_crawl.add_argument("--start-date", type=str, default="20231101", help="YYYYMMDD")
    sat_crawl.add_argument("--end-date", type=str, default="20241031", help="YYYYMMDD")
    sat_crawl.add_argument(
        "--parameters",
        type=str,
        default="ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR,RH2M,WS2M",
        help="NASA POWER parameter list",
    )
    sat_crawl.add_argument(
        "--cell-source",
        type=str,
        default="data/features/grid_static_features.tsv",
        help="TSV file with cell_id (optional)",
    )
    sat_crawl.add_argument("--only-nonzero", action="store_true", help="Only use nonzero-poi cells from cell-source")
    sat_crawl.add_argument("--max-cells", type=int, default=50, help="Max cells to crawl when cell_ids not specified")
    sat_crawl.add_argument("--cell-ids", type=str, default="", help="Comma-separated cell ids, e.g. 18_32,54_56")
    sat_crawl.set_defaults(func=collect_satellite_crawl)

    sat_plus = sub.add_parser(
        "satellite-cloud-nightlight",
        help="Crawl cloud+satellite daily features and merge nightlight table",
    )
    sat_plus.add_argument("--out-path", type=Path, default=Path("data/features/satellite_cloud_nightlight_daily.tsv"))
    sat_plus.add_argument("--start-date", type=str, default="20231101", help="YYYYMMDD")
    sat_plus.add_argument("--end-date", type=str, default="20241031", help="YYYYMMDD")
    sat_plus.add_argument(
        "--parameters",
        type=str,
        default="ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR,RH2M,WS2M",
        help="NASA POWER parameter list for non-cloud satellite/weather vars",
    )
    sat_plus.add_argument(
        "--cell-source",
        type=str,
        default="data/features/grid_static_features.tsv",
        help="TSV file with cell_id (optional)",
    )
    sat_plus.add_argument("--only-nonzero", action="store_true", help="Only use nonzero-poi cells from cell-source")
    sat_plus.add_argument("--max-cells", type=int, default=50, help="Max cells to crawl when cell_ids not specified")
    sat_plus.add_argument("--cell-ids", type=str, default="", help="Comma-separated cell ids, e.g. 18_32,54_56")
    sat_plus.add_argument("--nightlight-path", type=str, default="", help="Path to nightlight table (CSV/TSV)")
    sat_plus.add_argument("--nightlight-cell-col", type=str, default="cell_id")
    sat_plus.add_argument("--nightlight-date-col", type=str, default="date")
    sat_plus.add_argument("--nightlight-month-col", type=str, default="month")
    sat_plus.add_argument("--nightlight-value-col", type=str, default="night_light")
    sat_plus.set_defaults(func=collect_satellite_cloud_nightlight)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()