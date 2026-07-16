import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


DEFAULT_DATE_CANDIDATES = ["date", "day", "dt"]
DEFAULT_ORIGIN_CANDIDATES = ["origin", "o", "origin_id", "from_id"]
DEFAULT_DEST_CANDIDATES = ["destination", "d", "destination_id", "to_id"]
DEFAULT_TARGET_CANDIDATES = ["flow", "trip_count", "count", "y"]


def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    raise ValueError(f"Could not infer {label} column from candidates: {candidates}")


def build_text_rows(df: pd.DataFrame, sample_size: int) -> list[str]:
    date_col = pick_column(df, DEFAULT_DATE_CANDIDATES, "date")
    origin_col = pick_column(df, DEFAULT_ORIGIN_CANDIDATES, "origin")
    dest_col = pick_column(df, DEFAULT_DEST_CANDIDATES, "destination")
    target_col = pick_column(df, DEFAULT_TARGET_CANDIDATES, "target")

    sample = df[[date_col, origin_col, dest_col, target_col]].dropna().head(sample_size).copy()
    sample[date_col] = pd.to_datetime(sample[date_col], errors="coerce")
    sample = sample.dropna(subset=[date_col])

    return [
        f"date={row[date_col].date()} origin={row[origin_col]} destination={row[dest_col]} flow={row[target_col]}"
        for _, row in sample.iterrows()
    ]


def run_chat_smoke(client: OpenAI, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Please reply with exactly one word: READY",
            }
        ],
        temperature=0,
    )

    text = (resp.choices[0].message.content or "").strip()
    return {
        "mode": "chat",
        "ok": "READY" in text.upper(),
        "raw_reply": text,
    }


def run_embedding_smoke(client: OpenAI, model: str, rows: list[str]) -> dict:
    if not rows:
        return {"mode": "embedding", "ok": False, "error": "No valid sample rows available."}

    resp = client.embeddings.create(model=model, input=rows)
    vectors = [item.embedding for item in resp.data]

    dim = len(vectors[0]) if vectors else 0
    norms = [float(np.linalg.norm(np.array(v))) for v in vectors] if vectors else []

    return {
        "mode": "embedding",
        "ok": len(vectors) == len(rows) and dim > 0,
        "samples_requested": len(rows),
        "samples_embedded": len(vectors),
        "embedding_dim": dim,
        "mean_vector_norm": float(np.mean(norms)) if norms else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VectorEngine API usability test on HuMob dataset")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to dataset TSV")
    parser.add_argument("--base-url", type=str, default="https://api.vectorengine.ai/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--mode", type=str, choices=["chat", "embedding"], default="embedding", help="Smoke test mode")
    parser.add_argument("--model", type=str, required=True, help="Model name for selected mode")
    parser.add_argument("--sample-size", type=int, default=32, help="Number of dataset rows to test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("VECTORENGINE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing VECTORENGINE_API_KEY environment variable.")

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    if args.mode == "chat":
        result = run_chat_smoke(client, args.model)
    else:
        df = pd.read_csv(args.data_path, sep="\t")
        rows = build_text_rows(df, sample_size=args.sample_size)
        result = run_embedding_smoke(client, args.model, rows)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
