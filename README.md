# VIX Challenger

Compute a **VIX-style (model-free) 30-day implied volatility index** from **SPY options** (2020–2022 EOD quotes), then compare against **FRED VIXCLS** (SPX-based).

## Quick Start

```bash
# Install dependencies
uv sync

# Download SPY options data from Kaggle
uv run python scripts/download_data.py

# Inspect schema
uv run python scripts/00_inspect_schema.py

# Convert to partitioned parquet
uv run python scripts/01_convert_csv.py

# Compute VIX-like index
uv run python scripts/02_compute_vix_like.py

# Compare to FRED VIXCLS
uv run python scripts/03_compare_to_fred.py
```

## Project Structure

```
vix-challenger/
├── data/
│   ├── raw/              # Downloaded CSV data
│   └── processed/        # Partitioned parquet files
├── src/vix_challenger/   # Core library
│   ├── io/               # Data loading
│   ├── vix/              # VIX computation
│   ├── pipelines/        # Processing pipelines
│   └── viz/              # Visualization
├── scripts/              # CLI entry points
├── notebooks/            # Jupyter notebooks
└── reports/              # Output reports
```

## Methodology

Implements Cboe-style VIX methodology:
1. Select two expirations bracketing 30 days
2. Compute model-free variance from OTM option strips
3. Interpolate to constant 30-day maturity
4. Output: `index = 100 * sqrt(var_30d)`

See `docs/project_scope.md` for full details.

