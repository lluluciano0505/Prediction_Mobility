import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


DEFAULT_NEWS_DATE_CANDIDATES = ["date", "day", "dt", "published_at", "publish_date", "timestamp", "time"]
DEFAULT_NEWS_TEXT_CANDIDATES = ["text", "content", "title", "headline", "news"]
DEFAULT_NEWS_CELL_CANDIDATES = ["cell_id", "grid_id", "region_id", "h3", "h3_id", "location_id", "place_id"]


def pick_column(df: pd.DataFrame, explicit_name: str | None, candidates: list[str], label: str) -> str:
    if explicit_name:
        if explicit_name not in df.columns:
            raise ValueError(f"Explicit {label} column '{explicit_name}' not found. Available: {list(df.columns)}")
        return explicit_name

    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    raise ValueError(f"Could not infer {label} column from candidates: {candidates}")


def embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        vectors.extend([item.embedding for item in resp.data])
    return np.array(vectors, dtype=np.float32)


def build_embedding_features(
    news_path: Path,
    out_path: Path,
    model: str,
    base_url: str,
    batch_size: int,
    news_date_col: str | None,
    news_text_col: str | None,
    news_cell_col: str | None,
    sample_rows: int | None,
) -> None:
    api_key = os.getenv("VECTORENGINE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing VECTORENGINE_API_KEY environment variable.")

    news_df = pd.read_csv(news_path, sep=None, engine="python")

    date_col = pick_column(news_df, news_date_col, DEFAULT_NEWS_DATE_CANDIDATES, "news-date")
    text_col = pick_column(news_df, news_text_col, DEFAULT_NEWS_TEXT_CANDIDATES, "news-text")
    cell_col = pick_column(news_df, news_cell_col, DEFAULT_NEWS_CELL_CANDIDATES, "news-cell")

    selected = news_df[[date_col, text_col, cell_col]].copy()
    selected[date_col] = pd.to_datetime(selected[date_col], errors="coerce")
    selected[text_col] = selected[text_col].astype(str)
    selected[cell_col] = selected[cell_col].astype(str)
    selected = selected.dropna(subset=[date_col])
    selected = selected[selected[text_col].str.len() > 0]

    if sample_rows is not None and sample_rows > 0:
        selected = selected.head(sample_rows)

    if selected.empty:
        raise ValueError("No valid news rows after filtering by date/text/cell.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    texts = selected[text_col].tolist()
    vectors = embed_texts(client=client, model=model, texts=texts, batch_size=batch_size)

    emb_cols = [f"emb_{i:04d}" for i in range(vectors.shape[1])]
    emb_df = pd.DataFrame(vectors, columns=emb_cols)

    selected = selected.reset_index(drop=True)
    selected["news_date"] = selected[date_col].dt.normalize()
    selected["news_cell"] = selected[cell_col]

    joined = pd.concat([selected[["news_date", "news_cell"]], emb_df], axis=1)
    agg = joined.groupby(["news_date", "news_cell"], as_index=False).mean(numeric_only=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)

    summary = {
        "rows_input": int(len(selected)),
        "rows_output": int(len(agg)),
        "embedding_dim": int(vectors.shape[1]),
        "model": model,
        "out_path": str(out_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build date-cell news embedding features")
    parser.add_argument("--news-path", type=Path, required=True, help="Path to news CSV/TSV")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/news_embedding_features.csv"),
        help="Output CSV path for aggregated date-cell embeddings",
    )
    parser.add_argument("--model", type=str, default="text-embedding-3-large", help="Embedding model name")
    parser.add_argument("--base-url", type=str, default="https://api.vectorengine.ai/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding request batch size")
    parser.add_argument("--news-date-col", type=str, default=None, help="Optional explicit date column")
    parser.add_argument("--news-text-col", type=str, default=None, help="Optional explicit text column")
    parser.add_argument("--news-cell-col", type=str, default=None, help="Optional explicit cell column")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional cap for quick test runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_embedding_features(
        news_path=args.news_path,
        out_path=args.out_path,
        model=args.model,
        base_url=args.base_url,
        batch_size=args.batch_size,
        news_date_col=args.news_date_col,
        news_text_col=args.news_text_col,
        news_cell_col=args.news_cell_col,
        sample_rows=args.sample_rows,
    )


if __name__ == "__main__":
    main()
