# Prediction Mobility - HuMob Challenge 2026 Baseline

This repository provides a practical baseline pipeline for the HuMob Challenge 2026 dataset.

Implementation architecture and delivery plan:
- See `ARCHITECTURE.md`

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

Optional news input (for past-news features):

- Provide a second CSV/TSV file with at least:
  - News date column (for example: date, published_at)
  - And one of:
    - Text/title column (for example: text, headline, title)
    - Numeric score column (for example: sentiment, score)

The pipeline aggregates news by day and creates lagged news features (lag-1, lag-3 mean, lag-7 mean), so only past news information is used.

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

Default model is `hgbt` (HistGradientBoostingRegressor), which usually captures periodic patterns better than RandomForest on tabular time features.

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

4) Run with news-enhanced features

```bash
python -m src.humob_baseline \
  --data-path /path/to/humob2026-dataset.tsv \
  --news-path /path/to/news.csv \
  --out-dir outputs
```

5) Run with explicit news columns

```bash
python -m src.humob_baseline \
  --data-path /path/to/humob2026-dataset.tsv \
  --news-path /path/to/news.tsv \
  --news-date-col published_at \
  --news-text-col headline \
  --news-score-col sentiment \
  --out-dir outputs
```

6) Optional model selection

```bash
python -m src.humob_baseline \
  --data-path /path/to/humob2026-dataset.tsv \
  --model hgbt \
  --out-dir outputs
```

```bash
python -m src.humob_baseline \
  --data-path /path/to/humob2026-dataset.tsv \
  --model random_forest \
  --out-dir outputs
```

7) Build date-cell news embedding features (Model C input)

```bash
export VECTORENGINE_API_KEY="<your_key>"
python -m src.news_embedding_features \
  --news-path /path/to/news.tsv \
  --out-path outputs/news_embedding_features.csv \
  --model text-embedding-3-large
```

Expected output columns include:

- `news_date`
- `news_cell`
- `emb_0000 ... emb_N`

8) Train with embedding features merged to origin/destination cells

```bash
python -m src.humob_baseline \
  --data-path /path/to/humob2026-dataset.tsv \
  --news-embedding-path outputs/news_embedding_features.csv \
  --out-dir outputs/model_c
```

9) Collect public news and map to grid cells (crawl pipeline)

```bash
python -m src.news_collect_to_cells \
  --query "能登 OR 石川 地震 OR 交通 OR 観光" \
  --max-items 300 \
  --out-path data/news/news_collected_cells.tsv
```

This produces a training-ready news file with columns:

- `date`
- `cell_id`
- `headline`
- `text`
- `source`
- `url`

10) Build Model C embeddings from crawled news

```bash
export VECTORENGINE_API_KEY="<your_key>"
python -m src.news_embedding_features \
  --news-path data/news/news_collected_cells.tsv \
  --news-date-col date \
  --news-text-col text \
  --news-cell-col cell_id \
  --out-path outputs/news_embedding_features.csv \
  --model text-embedding-3-large
```

11) Staged experiments in your requested order

Model A (baseline only):

```bash
python -m src.humob_baseline \
  --data-path data/humob2026-tabular.tsv \
  --stage baseline \
  --out-dir outputs/model_a_baseline
```

Model B (+POI):

```bash
python -m src.humob_baseline \
  --data-path data/humob2026-tabular.tsv \
  --stage poi \
  --poi-path data/features/grid_static_features.tsv \
  --out-dir outputs/model_b_poi
```

Model C (+satellite on top of POI):

```bash
python -m src.humob_baseline \
  --data-path data/humob2026-tabular.tsv \
  --stage satellite \
  --poi-path data/features/grid_static_features.tsv \
  --satellite-path data/features/satellite_cloud_nightlight_daily.tsv \
  --out-dir outputs/model_c_satellite
```

Model D (+news on top of POI+satellite):

```bash
python -m src.humob_baseline \
  --data-path data/humob2026-tabular.tsv \
  --stage news \
  --poi-path data/features/grid_static_features.tsv \
  --satellite-path data/features/satellite_cloud_nightlight_daily.tsv \
  --news-path data/news/news_collected_cells.tsv \
  --out-dir outputs/model_d_news
```

## Outputs

The pipeline writes:

- outputs/metrics.json: Validation MAE and RMSE
- outputs/validation_predictions.csv: Per-row predictions on validation split

`metrics.json` also reports whether news features were used.

## Baseline Method

- Sort by OD pair and date.
- Build lag features from prior days within each OD pair.
- Add calendar + cyclical periodic features (day/week/month/year seasonality via sin/cos).
- Train a tree-based regressor (default: HistGradientBoostingRegressor, optional: RandomForestRegressor).

This baseline is intentionally simple and is meant as a starting point.

## Geography Embedding Recommendation

For this project, `srai` is the best fit among open-source options.

Background reading (GIS perspective):

- ArcGIS Blog: An introduction to embeddings for GIS analysts
  - https://www.esri.com/arcgis-blog/products/arcgis-pro/geoai/an-introduction-to-embeddings-for-gis-analysts

- Why it fits: `srai` is strong at turning spatial units (for example, grid cells or H3 cells) into vector embeddings, which matches OD flow forecasting well.
- How to use it here: map both origin and destination grid IDs to embeddings, then combine those embeddings with temporal lag features in the prediction model.

Practical integration path:

1) Keep the current lag/time baseline as a stable benchmark.
2) Add origin and destination embedding features.
3) Concatenate embedding features with current lag/calendar features.
4) Retrain and compare MAE/RMSE against the baseline.

Chinese summary:

- srai 适配度最高。它擅长把空间单元（如网格/H3）做成向量表示，和 OD 流量预测任务最接近。
- 可将 origin/destination 的网格 ID 映射成 embedding，再与时间滞后特征一起输入模型。
