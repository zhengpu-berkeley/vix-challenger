# VIX Challenger

Compute a **VIX-style (model-free) 30-day implied volatility index** from equity/ETF options, then compare against **FRED VIXCLS** (SPX-based).

Supports multiple tickers: **SPY** (2020-2022), **AAPL** (2016-2020), **TSLA** (2019-2022), **NVDA** (2020-2022).

## Quick Start

```bash
# Install dependencies
uv sync

# Download options data from Kaggle
uv run python scripts/download_data.py

# Convert to partitioned parquet
uv run python scripts/01_convert_csv.py --ticker SPY

# Compute VIX-like index
uv run python scripts/02_compute_vix_like.py --ticker SPY

# Compare to FRED VIXCLS
uv run python scripts/04_compare_to_fred.py --ticker SPY --plot

# Cross-ticker analysis
uv run python scripts/05_cross_ticker_analysis.py --plot
```

## Project Structure

```
vix-challenger/
├── data/
│   ├── raw/              # Downloaded CSV + split metadata JSON
│   └── processed/        # Partitioned parquet files
├── src/vix_challenger/   # Core library
│   ├── io/               # Data loading (options, splits, FRED)
│   ├── vix/              # VIX computation + methodology docs
│   └── viz/              # Visualization
├── scripts/              # CLI entry points
├── tests/                # Unit tests
└── reports/              # Output reports and figures
```

## Results

| Ticker | Correlation with VIXCLS | Mean Bias |
|--------|-------------------------|-----------|
| SPY    | **0.9970**              | -0.42     |
| AAPL   | 0.8731                  | +11.26    |
| TSLA   | 0.6912                  | +48.11    |
| NVDA   | 0.7463                  | +28.42    |

See [`reports/consolidated_summary.md`](reports/consolidated_summary.md) for the full analysis.

## Methodology

Implements Cboe-style VIX methodology with robustness enhancements for single-stock chains:

1. Select two expirations bracketing 30 days
2. Compute forward price via put-call parity (with moneyness guards)
3. Build OTM strip with zero-bid cutoff and tail-strike suppression
4. Calculate model-free variance with split-aware filtering
5. Interpolate to constant 30-day maturity
6. Output: `index = 100 * sqrt(var_30d)`

See [`src/vix_challenger/vix/methodology.md`](src/vix_challenger/vix/methodology.md) for technical details.

