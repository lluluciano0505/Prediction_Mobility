import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


# HuMob grid metadata from official docs.
GRID_MIN_LON = 136.029
GRID_MAX_LON = 138.042
GRID_MIN_LAT = 36.203
GRID_MAX_LAT = 37.646
GRID_X_COUNT = 100
GRID_Y_COUNT = 70

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OPENTOPO_URL = "https://api.opentopodata.org/v1/srtm90m"

POI_FEATURE_BUCKETS = [
    "poi_count_total",
    "poi_amenity_count",
    "poi_shop_count",
    "poi_tourism_count",
    "poi_leisure_count",
    "poi_office_count",
    "poi_public_transport_count",
    "poi_amenity_food_count",
    "poi_amenity_health_count",
    "poi_amenity_education_count",
    "poi_amenity_finance_count",
    "poi_amenity_parking_count",
    "poi_amenity_other_count",
    "poi_shop_convenience_count",
    "poi_shop_supermarket_count",
    "poi_shop_fashion_count",
    "poi_shop_car_count",
    "poi_shop_other_count",
    "poi_tourism_hotel_count",
    "poi_tourism_attraction_count",
    "poi_tourism_museum_count",
    "poi_tourism_other_count",
    "poi_leisure_park_count",
    "poi_leisure_sports_count",
    "poi_leisure_other_count",
    "poi_public_transport_bus_count",
    "poi_public_transport_rail_count",
    "poi_public_transport_other_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect grid-level static features from OSM and optional elevation")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/features/grid_static_features.tsv"),
        help="Output TSV path",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=f"{GRID_MIN_LAT},{GRID_MIN_LON},{GRID_MAX_LAT},{GRID_MAX_LON}",
        help="Bounding box as min_lat,min_lon,max_lat,max_lon",
    )
    parser.add_argument("--with-elevation", action="store_true", help="Call OpenTopoData API for elevation")
    parser.add_argument(
        "--max-elevation-cells",
        type=int,
        default=400,
        help="Max number of cell centroids to request elevation for",
    )
    return parser.parse_args()


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def lonlat_to_cell_id(lon: float, lat: float) -> str | None:
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


def http_post(url: str, query_text: str) -> dict:
    req = Request(
        url=url,
        data=query_text.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=UTF-8",
            "Accept": "application/json",
            "User-Agent": "PredictionMobility/1.0",
        },
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def http_get_json(url: str, params: dict[str, str]) -> dict:
    query = urlencode(params)
    req = Request(f"{url}?{query}", method="GET")
    with urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def build_overpass_query(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> str:
    # Collect broad POI classes as static geography proxies.
    tags = ["amenity", "shop", "tourism", "leisure", "office", "public_transport"]
    lines: list[str] = ["[out:json][timeout:180];", "("]
    for t in tags:
        lines.append(f'  node["{t}"]({min_lat},{min_lon},{max_lat},{max_lon});')
        lines.append(f'  way["{t}"]({min_lat},{min_lon},{max_lat},{max_lon});')
        lines.append(f'  relation["{t}"]({min_lat},{min_lon},{max_lat},{max_lon});')
    lines.append(");")
    lines.append("out center tags;")
    return "\n".join(lines)


def classify_poi(tags: dict) -> list[str]:
    if not tags:
        return []

    groups: set[str] = set()

    amenity_value = str(tags.get("amenity", "")).lower()
    if amenity_value:
        groups.add("amenity")
        if amenity_value in {"restaurant", "cafe", "fast_food", "bar", "pub", "food_court"}:
            groups.add("amenity_food")
        elif amenity_value in {"hospital", "clinic", "doctors", "pharmacy", "dentist"}:
            groups.add("amenity_health")
        elif amenity_value in {"school", "university", "college", "kindergarten", "library"}:
            groups.add("amenity_education")
        elif amenity_value in {"bank", "atm"}:
            groups.add("amenity_finance")
        elif amenity_value in {"parking", "parking_entrance", "parking_space"}:
            groups.add("amenity_parking")
        else:
            groups.add("amenity_other")

    shop_value = str(tags.get("shop", "")).lower()
    if shop_value:
        groups.add("shop")
        if shop_value in {"convenience"}:
            groups.add("shop_convenience")
        elif shop_value in {"supermarket", "mall", "department_store"}:
            groups.add("shop_supermarket")
        elif shop_value in {"clothes", "shoes", "jewelry", "bag"}:
            groups.add("shop_fashion")
        elif shop_value in {"car", "car_repair", "car_parts", "motorcycle", "bicycle"}:
            groups.add("shop_car")
        else:
            groups.add("shop_other")

    tourism_value = str(tags.get("tourism", "")).lower()
    if tourism_value:
        groups.add("tourism")
        if tourism_value in {"hotel", "guest_house", "motel", "hostel", "apartment"}:
            groups.add("tourism_hotel")
        elif tourism_value in {"attraction", "viewpoint", "theme_park", "zoo"}:
            groups.add("tourism_attraction")
        elif tourism_value in {"museum", "gallery"}:
            groups.add("tourism_museum")
        else:
            groups.add("tourism_other")

    leisure_value = str(tags.get("leisure", "")).lower()
    if leisure_value:
        groups.add("leisure")
        if leisure_value in {"park", "garden", "nature_reserve"}:
            groups.add("leisure_park")
        elif leisure_value in {"sports_centre", "stadium", "pitch", "fitness_centre", "swimming_pool"}:
            groups.add("leisure_sports")
        else:
            groups.add("leisure_other")

    office_value = str(tags.get("office", "")).lower()
    if office_value:
        groups.add("office")

    public_transport_value = str(tags.get("public_transport", "")).lower()
    railway_value = str(tags.get("railway", "")).lower()
    bus_value = str(tags.get("bus", "")).lower()
    if public_transport_value or railway_value in {"station", "halt", "tram_stop", "subway_entrance"}:
        groups.add("public_transport")
        if public_transport_value in {"station", "platform", "stop_position"} and railway_value in {
            "station",
            "halt",
            "tram_stop",
            "subway_entrance",
        }:
            groups.add("public_transport_rail")
        elif public_transport_value in {"platform", "stop_position"} or bus_value == "yes":
            groups.add("public_transport_bus")
        else:
            groups.add("public_transport_other")

    return sorted(groups)


def collect_osm_cell_features(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> pd.DataFrame:
    query = build_overpass_query(min_lat=min_lat, min_lon=min_lon, max_lat=max_lat, max_lon=max_lon)
    raw: dict | None = None
    last_error: str = ""
    for endpoint in OVERPASS_URLS:
        try:
            raw = http_post(endpoint, query)
            break
        except Exception as exc:
            last_error = str(exc)

    if raw is None:
        raise RuntimeError(f"Failed to query Overpass endpoints. Last error: {last_error}")

    counts: dict[str, dict[str, int]] = {}
    for elem in raw.get("elements", []):
        tags = elem.get("tags", {}) or {}
        cats = classify_poi(tags)
        if not cats:
            continue

        lat = elem.get("lat")
        lon = elem.get("lon")
        center = elem.get("center") if isinstance(elem.get("center"), dict) else None
        if lat is None or lon is None:
            if center is not None:
                lat = center.get("lat")
                lon = center.get("lon")
        if lat is None or lon is None:
            continue

        cell = lonlat_to_cell_id(lon=float(lon), lat=float(lat))
        if cell is None:
            continue

        if cell not in counts:
            counts[cell] = {bucket: 0 for bucket in POI_FEATURE_BUCKETS}

        counts[cell]["poi_count_total"] += 1
        for cat in cats:
            key = f"poi_{cat}_count"
            if key not in counts[cell]:
                counts[cell][key] = 0
            counts[cell][key] += 1

    rows = []
    for y in range(1, GRID_Y_COUNT + 1):
        for x in range(1, GRID_X_COUNT + 1):
            cell = f"{y}_{x}"
            stats = counts.get(
                cell,
                {bucket: 0 for bucket in POI_FEATURE_BUCKETS},
            )
            rows.append({"cell_id": cell, **stats})

    df = pd.DataFrame(rows)
    df["poi_log1p_total"] = df["poi_count_total"].apply(lambda v: math.log1p(v))
    return df


def add_elevation(df: pd.DataFrame, max_cells: int) -> pd.DataFrame:
    target = df[df["poi_count_total"] > 0].copy()
    if target.empty:
        target = df.copy()
    target = target.head(max_cells)

    points = [cell_centroid(c) for c in target["cell_id"]]
    locations = "|".join([f"{lat:.6f},{lon:.6f}" for lat, lon in points])
    resp = http_get_json(OPENTOPO_URL, {"locations": locations})

    results = resp.get("results", [])
    elev_map: dict[str, float] = {}
    for cell, r in zip(target["cell_id"].tolist(), results):
        e = r.get("elevation")
        if e is None:
            continue
        elev_map[cell] = float(e)

    out = df.copy()
    out["elevation_m"] = out["cell_id"].map(elev_map).fillna(0.0)
    return out


def main() -> None:
    args = parse_args()
    min_lat, min_lon, max_lat, max_lon = [float(x) for x in args.bbox.split(",")]

    features = collect_osm_cell_features(min_lat=min_lat, min_lon=min_lon, max_lat=max_lat, max_lon=max_lon)
    if args.with_elevation:
        features = add_elevation(features, max_cells=args.max_elevation_cells)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_path, sep="\t", index=False)

    summary = {
        "rows": int(len(features)),
        "nonzero_poi_cells": int((features["poi_count_total"] > 0).sum()),
        "with_elevation": bool(args.with_elevation),
        "out_path": str(args.out_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
