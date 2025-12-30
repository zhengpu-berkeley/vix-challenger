# Multi-Ticker VIX-like Index Analysis

## Executive Summary

This report summarizes the computation of VIX-like (model-free implied volatility) indices for four equity options series: SPY, AAPL, TSLA, and NVDA. The methodology follows the Cboe VIX calculation with 30-day constant maturity interpolation.

## Dataset Summary

| Ticker | Period | Trading Days | Success Rate | Data Source |
|--------|--------|--------------|--------------|-------------|
| SPY | 2020-2022 | 758 | 100% | Kaggle (kylegraupe) |
| AAPL | 2016-2020 | 1,253 | 100% | Kaggle (kylegraupe) |
| TSLA | 2019-2022 | 1,010 | 98% | Kaggle (kylegraupe) |
| NVDA | 2020-2022 | 756 | 100% | Kaggle (kylegraupe) |

## VIX-like Index Statistics

### Full Period Statistics

| Ticker | Mean | Std Dev | Min | Max | Period |
|--------|------|---------|-----|-----|--------|
| SPY | 24.42 | 12.51 | 11.71 | 81.11 | 2020-2022 |
| AAPL | 28.96 | 11.23 | 14.21 | 101.46 | 2016-2020 |
| TSLA | 76.41 | 95.02 | 35.57 | 2754.54* | 2019-2022 |
| NVDA | 53.67 | 18.61 | 31.01 | 333.76 | 2020-2022 |

*TSLA has extreme outliers in early 2019 due to low options liquidity.

### Common Period Statistics (2020)

For the 249 trading days common to all tickers (Jan-Dec 2020):

| Ticker | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| SPY | 28.88 | 12.14 | 11.71 | 81.11 |
| AAPL | 43.53 | 13.23 | 26.21 | 101.46 |
| TSLA | 92.66 | 36.32 | 52.07 | 452.62 |
| NVDA | 53.25 | 12.81 | 31.01 | 107.60 |
| VIXCLS | 29.31 | 12.39 | 12.10 | 82.69 |

## Correlation Analysis

### Cross-Ticker Correlation Matrix (2020 Common Period)

|        | SPY    | AAPL   | TSLA   | NVDA   | VIXCLS |
|--------|--------|--------|--------|--------|--------|
| SPY    | 1.0000 | 0.7782 | 0.4158 | 0.9073 | 0.9987 |
| AAPL   | 0.7782 | 1.0000 | 0.5642 | 0.8007 | 0.7799 |
| TSLA   | 0.4158 | 0.5642 | 1.0000 | 0.5039 | 0.4141 |
| NVDA   | 0.9073 | 0.8007 | 0.5039 | 1.0000 | 0.9063 |
| VIXCLS | 0.9987 | 0.7799 | 0.4141 | 0.9063 | 1.0000 |

### Key Findings

1. **SPY replicates VIXCLS near-perfectly** (r = 0.9987)
   - This validates our VIX methodology

2. **NVDA has high market correlation** (r = 0.91 with VIXCLS)
   - Tech sector volatility closely tracks market volatility
   - Mean bias: +24 points vs VIXCLS

3. **AAPL has moderate market correlation** (r = 0.78 with VIXCLS)
   - Single-stock idiosyncratic risk adds ~15 points to baseline
   
4. **TSLA has low market correlation** (r = 0.41 with VIXCLS)
   - High idiosyncratic "meme stock" volatility
   - Mean IV ~63 points higher than market VIX
   - Extreme spikes during stock splits and earnings

## Comparison to FRED VIXCLS

| Ticker | Correlation | RMSE | Mean Bias | Period |
|--------|-------------|------|-----------|--------|
| SPY | 0.9969 | 0.82 | -0.41 | 2020-2022 |
| AAPL | 0.8598 | 12.55 | +11.32 | 2016-2020 |
| TSLA | 0.1076 | 107.42 | +53.81 | 2019-2022 |
| NVDA | 0.5453 | 31.58 | +28.80 | 2020-2022 |

## Volatility Premium Analysis

Single-stock options command a volatility premium over index options due to:

1. **Idiosyncratic risk**: Company-specific events not diversified away
2. **Higher beta**: Tech stocks amplify market movements
3. **Retail speculation**: Especially for TSLA with high retail interest
4. **Earnings volatility**: Individual stock earnings create uncertainty

### Premium over VIXCLS (2020 Common Period)

| Ticker | Premium | Notes |
|--------|---------|-------|
| SPY | -0.4 pts | Near-perfect replication |
| AAPL | +14.2 pts | Large-cap tech premium |
| NVDA | +23.9 pts | Semiconductor/AI exposure |
| TSLA | +63.4 pts | Meme stock, high retail interest |

## Generated Artifacts

### Data Files

- `data/processed/spy_vix_like.parquet` - SPY VIX-like daily series
- `data/processed/aapl_vix_like.parquet` - AAPL VIX-like daily series
- `data/processed/tsla_vix_like.parquet` - TSLA VIX-like daily series
- `data/processed/nvda_vix_like.parquet` - NVDA VIX-like daily series
- `data/processed/cross_ticker_vix.parquet` - Joined series for common period

### Figures

**Per-Ticker vs VIXCLS:**
- `reports/figures/{ticker}/01_overlay.png` - Time series overlay
- `reports/figures/{ticker}/02_scatter.png` - Scatter with regression
- `reports/figures/{ticker}/03_residuals.png` - Residual analysis
- `reports/figures/{ticker}/04_rolling_correlation.png` - 60-day rolling correlation
- `reports/figures/{ticker}/05_rolling_beta.png` - 60-day rolling beta

**Cross-Ticker Analysis:**
- `reports/figures/cross_ticker/01_correlation_heatmap.png`
- `reports/figures/cross_ticker/02_multi_ticker_overlay.png`
- `reports/figures/cross_ticker/03_spread_*_vs_vixcls.png`

## Methodology Notes

1. **Forward price calculation**: Put-call parity with r=0 assumption
2. **OTM strip construction**: Two consecutive zero-bid cutoff rule
3. **Strike spacing**: Mid-point between adjacent strikes
4. **Interpolation**: Standard Cboe 30-day constant maturity formula
5. **Data source**: All datasets from Kyle Graupe's Kaggle collections

## Conclusions

1. The VIX methodology successfully replicates VIXCLS when applied to SPY options (correlation > 0.99)

2. Single-stock VIX-like indices show expected behavior:
   - Higher baseline volatility than market VIX
   - Lower correlation with market VIX for higher-beta/speculative names
   - TSLA exhibits the highest idiosyncratic volatility

3. The cross-ticker analysis enables volatility relative value comparisons and can inform options trading strategies

---

*Report generated: 2024*

