import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HuMob raw 2-column TSV into tabular OD rows")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to raw humob2026-dataset.tsv")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/humob2026-tabular.tsv"),
        help="Path to output tabular TSV",
    )
    return parser.parse_args()


def parse_payload(payload: str) -> dict:
    payload = str(payload).strip()
    if payload.upper() in {"NA", "NAN"} or payload == "":
        return {}
    return ast.literal_eval(payload)


def main() -> None:
    args = parse_args()
    raw_df = pd.read_csv(
        args.input_path,
        sep="\t",
        header=None,
        names=["date", "payload"],
        dtype=str,
        keep_default_na=False,
    )

    rows: list[tuple[str, str, str, float]] = []
    for _, r in raw_df.iterrows():
        day = r["date"]
        od_dict = parse_payload(r["payload"])
        for origin, dest_map in od_dict.items():
            if not isinstance(dest_map, dict):
                continue
            for dest, flow in dest_map.items():
                try:
                    rows.append((day, str(origin), str(dest), float(flow)))
                except (TypeError, ValueError):
                    continue

    out_df = pd.DataFrame(rows, columns=["date", "origin", "destination", "flow"])
    out_df["date"] = pd.to_datetime(out_df["date"], format="%Y%m%d", errors="coerce")
    out_df = out_df.dropna(subset=["date"]).copy()
    out_df = out_df.sort_values(["origin", "destination", "date"]).reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, sep="\t", index=False)

    summary = {
        "input_rows": int(len(raw_df)),
        "output_rows": int(len(out_df)),
        "output_path": str(args.output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
