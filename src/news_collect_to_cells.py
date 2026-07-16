import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import trafilatura


GRID_MIN_LON = 136.029
GRID_MAX_LON = 138.042
GRID_MIN_LAT = 36.203
GRID_MAX_LAT = 37.646
GRID_X_COUNT = 100
GRID_Y_COUNT = 70

# Simple seed gazetteer for Ishikawa/Noto area. Extend as needed.
PLACE_CENTROIDS = {
    "noto": (37.300, 137.150),
    "noto peninsula": (37.300, 137.150),
    "能登": (37.300, 137.150),
    "石川": (36.700, 136.900),
    "ishikawa": (36.700, 136.900),
    "wajima": (37.390, 136.900),
    "輪島": (37.390, 136.900),
    "suzu": (37.450, 137.270),
    "珠洲": (37.450, 137.270),
    "nanao": (37.040, 136.970),
    "七尾": (37.040, 136.970),
    "anamizu": (37.230, 136.900),
    "穴水": (37.230, 136.900),
    "shika": (37.010, 136.780),
    "志賀": (37.010, 136.780),
    "hakui": (36.900, 136.780),
    "羽咋": (36.900, 136.780),
    "kanazawa": (36.560, 136.650),
    "金沢": (36.560, 136.650),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect public news and map to HuMob grid cells")
    parser.add_argument(
        "--query",
        type=str,
        default="能登 OR 石川 地震 OR 交通 OR 観光",
        help="Google News RSS query",
    )
    parser.add_argument("--max-items", type=int, default=300, help="Max RSS items to process")
    parser.add_argument("--days-back", type=int, default=365, help="Keep only items within this many days")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/news/news_collected_cells.tsv"),
        help="Output TSV path",
    )
    return parser.parse_args()


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def latlon_to_cell(lat: float, lon: float) -> str | None:
    if lon < GRID_MIN_LON or lon > GRID_MAX_LON or lat < GRID_MIN_LAT or lat > GRID_MAX_LAT:
        return None

    x_ratio = (lon - GRID_MIN_LON) / (GRID_MAX_LON - GRID_MIN_LON)
    y_ratio = (lat - GRID_MIN_LAT) / (GRID_MAX_LAT - GRID_MIN_LAT)

    x = clamp(int(x_ratio * GRID_X_COUNT) + 1, 1, GRID_X_COUNT)
    y = clamp(int(y_ratio * GRID_Y_COUNT) + 1, 1, GRID_Y_COUNT)
    return f"{y}_{x}"


def extract_place_latlon(text: str) -> tuple[float, float] | None:
    low = text.lower()
    for key, latlon in PLACE_CENTROIDS.items():
        if key in low or key in text:
            return latlon
    return None


def extract_text_from_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return extracted or ""


def parse_published(entry: dict) -> datetime | None:
    if "published_parsed" in entry and entry["published_parsed"] is not None:
        return datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)
    if "updated_parsed" in entry and entry["updated_parsed"] is not None:
        return datetime(*entry["updated_parsed"][:6], tzinfo=timezone.utc)
    return None


def build_feed_url(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"


def main() -> None:
    args = parse_args()
    feed_url = build_feed_url(args.query)
    feed = feedparser.parse(feed_url)

    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    seen_urls: set[str] = set()

    for entry in feed.entries[: args.max_items]:
        url = entry.get("link", "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        published_at = parse_published(entry)
        if published_at is None:
            continue

        age_days = (now - published_at).days
        if age_days > args.days_back:
            continue

        title = (entry.get("title", "") or "").strip()
        source = ""
        if "source" in entry and isinstance(entry["source"], dict):
            source = (entry["source"].get("title", "") or "").strip()

        body = extract_text_from_url(url)
        merged_text = f"{title}\n{body}".strip()

        loc = extract_place_latlon(merged_text)
        if loc is None:
            continue

        cell_id = latlon_to_cell(lat=loc[0], lon=loc[1])
        if cell_id is None:
            continue

        rows.append(
            {
                "date": published_at.date().isoformat(),
                "published_at": published_at.isoformat(),
                "cell_id": cell_id,
                "headline": title,
                "text": body,
                "source": source,
                "url": url,
            }
        )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["date", "cell_id", "published_at"]).drop_duplicates(subset=["url"])

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_path, sep="\t", index=False)

    summary = {
        "feed_url": feed_url,
        "rows_out": int(len(out_df)),
        "out_path": str(args.out_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
