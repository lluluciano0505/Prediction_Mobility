# Data Collection Guide (in data/)

This folder now includes a unified collection script:

- `data/collect_data.py`

It supports:

- POI collection (`poi`)
- Elevation collection (`poi --with-elevation`)
- Public news collection (`news`)
- Satellite crawling from NASA POWER API (`satellite-crawl`)
- Cloud + nightlight daily grid table (`satellite-cloud-nightlight`)
- Satellite export transformation (`satellite`)

## 1) POI only

```bash
. .venv/bin/activate
python data/collect_data.py poi \
  --out-path data/features/grid_static_features.tsv
```

POI is now split into coarse + fine-grained categories.

Coarse columns:

- `poi_amenity_count`
- `poi_shop_count`
- `poi_tourism_count`
- `poi_leisure_count`
- `poi_office_count`
- `poi_public_transport_count`

Fine columns:

- `poi_amenity_food_count`, `poi_amenity_health_count`, `poi_amenity_education_count`, `poi_amenity_finance_count`, `poi_amenity_parking_count`, `poi_amenity_other_count`
- `poi_shop_convenience_count`, `poi_shop_supermarket_count`, `poi_shop_fashion_count`, `poi_shop_car_count`, `poi_shop_other_count`
- `poi_tourism_hotel_count`, `poi_tourism_attraction_count`, `poi_tourism_museum_count`, `poi_tourism_other_count`
- `poi_leisure_park_count`, `poi_leisure_sports_count`, `poi_leisure_other_count`
- `poi_public_transport_bus_count`, `poi_public_transport_rail_count`, `poi_public_transport_other_count`

If you generated `grid_static_features.tsv` before this change, rerun `poi` once to refresh columns.

## 2) POI + elevation

```bash
. .venv/bin/activate
python data/collect_data.py poi \
  --with-elevation \
  --max-elevation-cells 1000 \
  --out-path data/features/grid_static_features_with_elev.tsv
```

Notes:

- Elevation uses OpenTopoData SRTM API.
- `--max-elevation-cells` controls API volume.

## 3) Public news collection

```bash
. .venv/bin/activate
python data/collect_data.py news \
  --query "能登 OR 石川 地震 OR 交通 OR 観光" \
  --max-items 200 \
  --days-back 365 \
  --out-path data/news/news_collected_cells.tsv
```

Output schema:

- `date`
- `cell_id`
- `headline`
- `text`
- `source`
- `url`

## 4) Satellite data (import/transform)

Use this mode to convert satellite products (for example, GEE/Sentinel/Landsat exports) into `date x cell_id` features.

### Option A: your file already has `cell_id`

```bash
. .venv/bin/activate
python data/collect_data.py satellite \
  --in-path data/features/satellite_raw.csv \
  --out-path data/features/satellite_cell_daily.tsv \
  --date-col date \
  --cell-col cell_id \
  --feature-cols ndvi,ndwi,ndbi,lst
```

### Option B: your file has `lat/lon` (script maps to HuMob cell)

```bash
. .venv/bin/activate
python data/collect_data.py satellite \
  --in-path data/features/satellite_raw.csv \
  --out-path data/features/satellite_cell_daily.tsv \
  --date-col date \
  --feature-cols ndvi,ndwi,ndbi,lst \
  --lat-col lat \
  --lon-col lon
```

Expected raw input columns (minimum):

- `date`
- one of:
  - `cell_id`, or
  - `lat` and `lon`
- one or more numeric satellite indicators (for example `ndvi`, `ndwi`, `ndbi`, `lst`)

## 5) Satellite data (direct crawl)

Use this mode when you do not already have a satellite export file and want to crawl daily remote-sensing/weather variables directly by grid cell centroid.

```bash
. .venv/bin/activate
python data/collect_data.py satellite-crawl \
  --start-date 20231101 \
  --end-date 20241031 \
  --parameters ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR,RH2M,WS2M \
  --cell-ids 18_32 \
  --out-path data/features/satellite_power_daily.tsv
```

Batch example (from existing grid file):

```bash
. .venv/bin/activate
python data/collect_data.py satellite-crawl \
  --start-date 20231101 \
  --end-date 20241031 \
  --only-nonzero \
  --max-cells 300 \
  --out-path data/features/satellite_power_daily.tsv
```

Output columns are in `date x cell_id` format, with variables prefixed by `sat_`.

## 6) Cloud + nightlight version (grid-level, no pixel detail)

This mode outputs one row per `date x cell_id` and includes:

- `sat_cloud_cover_mean` (daily cloud cover)
- regular satellite/weather vars from NASA POWER (for example `sat_allsky_sfc_sw_dwn`, `sat_t2m`)
- `sat_night_light` merged from your nightlight table

Example:

```bash
. .venv/bin/activate
python data/collect_data.py satellite-cloud-nightlight \
  --start-date 20231101 \
  --end-date 20231107 \
  --cell-ids 18_32 \
  --nightlight-path data/features/nightlight_monthly.tsv \
  --nightlight-cell-col cell_id \
  --nightlight-month-col month \
  --nightlight-value-col night_light \
  --out-path data/features/satellite_cloud_nightlight_daily.tsv
```

Nightlight input table supports either:

- daily join: `cell_id,date,night_light`
- monthly join: `cell_id,month,night_light` (e.g. `2023-11`)

## 7) Quick checks

```bash
. .venv/bin/activate
python data/collect_data.py --help
python data/collect_data.py poi --help
python data/collect_data.py news --help
python data/collect_data.py satellite --help
python data/collect_data.py satellite-crawl --help
python data/collect_data.py satellite-cloud-nightlight --help
```

## 8) Single-cell validation example

Use one grid cell to validate whether your collection output is usable.

Example cell: `18_32`

```bash
. .venv/bin/activate
python - <<'PY'
import pandas as pd

cell = "18_32"
poi = pd.read_csv("data/features/grid_static_features.tsv", sep="\t")
row = poi[poi["cell_id"] == cell]
print(row.to_string(index=False))
PY
```

Example interpretation (current run):

- `poi_count_total=1415`: this cell has dense POI coverage
- `poi_amenity_count=756`: strong daily-life/service intensity
- `poi_tourism_count=331`: strong tourism-related activity
- `poi_public_transport_count=99`: transport connectivity signal exists

Why this validates your pipeline:

- It proves `cell_id` mapping is working.
- It proves category counters are non-zero and interpretable.
- It gives one concrete feature vector you can trace in Model B (origin-side / destination-side merge).
