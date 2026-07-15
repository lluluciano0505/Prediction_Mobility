# Prediction Mobility - HuMob Challenge 2026 Baseline

This repository provides a practical baseline pipeline for the HuMob Challenge 2026 dataset.

## Scope

- Task: Predict aggregated origin-destination mobility flow volumes.
- Data source: HuMob Challenge 2026 dataset on Zenodo.
- Geographic coverage: Noto / Ishikawa study area (about 2 km grids).
- Time setup:
  - Training: 2023-11-01 to 2024-01-31, and 2024-04-01 to 2024-06-30
  - Test: 2024-02-01 to 2024-03-31

## Data Access and Usage Notes

This dataset is restricted and requires access approval via the challenge instructions.
Use of the data must follow the challenge terms and ethical restrictions.

Reference dataset DOI:
- https://doi.org/10.5281/zenodo.20709796

Related citation requirement described by the challenge organizers:
- Yabe et al. (2024), Scientific Data 11(1):397

## Repository Layout

- src/humob_baseline.py: End-to-end baseline pipeline
- requirements.txt: Python dependencies

## Expected Input Format

The script assumes a tab-separated file and attempts to infer key columns. It looks for:

- Date column: one of date, day, dt
- Origin column: one of origin, o, origin_id, from_id
- Destination column: one of destination, d, destination_id, to_id
- Target flow column: one of flow, trip_count, count, y

If your column names differ, pass explicit names through CLI arguments.

## Quick Start

1) Create environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run baseline training and validation

```bash
python -m src.humob_baseline \
  --data-path /path/to/humob2026-dataset.tsv \
  --out-dir outputs
```

3) Optional explicit columns

```bash
python -m src.humob_baseline \
  --data-path /path/to/humob2026-dataset.tsv \
  --date-col date \
  --origin-col origin \
  --dest-col destination \
  --target-col flow \
  --out-dir outputs
```

## Outputs

The pipeline writes:

- outputs/metrics.json: Validation MAE and RMSE
- outputs/validation_predictions.csv: Per-row predictions on validation split

## Baseline Method

- Sort by OD pair and date.
- Build lag features from prior days within each OD pair.
- Add simple calendar features (day of week, month, weekend).
- Train a tree-based regressor (RandomForestRegressor).

This baseline is intentionally simple and is meant as a starting point.
