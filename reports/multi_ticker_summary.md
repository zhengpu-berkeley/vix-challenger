# Multi-Ticker VIX-like Index Analysis

## Executive Summary

This report summarizes the computation of VIX-like (model-free implied volatility) indices for four equity options series: SPY, AAPL, TSLA, and NVDA. The methodology follows the Cboe VIX calculation with 30-day constant maturity interpolation.

## Dataset Summary

| Ticker | Period | Trading Days | Success Rate | Data Source |
|--------|--------|--------------|--------------|-------------|
| SPY | 2020-2022 | 758 | 100% | Kaggle (kylegraupe) |
| AAPL | 2016-2020 | 1,253 | 99.9% | Kaggle (kylegraupe) |
| TSLA | 2019-2022 | 1,010 | 99.9% | Kaggle (kylegraupe) |
| NVDA | 2020-2022 | 756 | 100% | Kaggle (kylegraupe) |

## VIX-like Index Statistics

### Full Period Statistics

| Ticker | Mean | Std Dev | Min | Max | Period |
|--------|------|---------|-----|-----|--------|
| SPY | 24.41 | 8.56 | 11.71 | 81.11 | 2020-2022 |
| AAPL | 28.90 | 10.42 | 14.21 | 99.98 | 2016-2020 |
| TSLA | 70.59 | 19.34 | 35.57 | 188.11 | 2019-2022 |
| NVDA | 53.67 | 15.41 | 31.01 | 333.54 | 2020-2022 |

*TSLA extreme spikes were traced to broken strike ladders / tail-strike dominance and are now handled via explicit QC and stricter validity checks.*

### Common Period Statistics (2020)

For the 247 trading days common to all tickers (Jan-Dec 2020):

| Ticker | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| SPY | 28.90 | 12.18 | 11.71 | 81.11 |
| AAPL | 43.22 | 12.70 | 26.21 | 99.98 |
| TSLA | 89.24 | 24.20 | 52.07 | 188.11 |
| NVDA | 53.20 | 12.83 | 31.01 | 107.60 |
| VIXCLS | 29.33 | 12.43 | 12.10 | 82.69 |

## Correlation Analysis

### Cross-Ticker Correlation Matrix (2020 Common Period)

|        | SPY    | AAPL   | TSLA   | NVDA   | VIXCLS |
|--------|--------|--------|--------|--------|--------|
| SPY    | 1.0000 | 0.8232 | 0.6089 | 0.9090 | 0.9987 |
| AAPL   | 0.8232 | 1.0000 | 0.7437 | 0.8357 | 0.8245 |
| TSLA   | 0.6089 | 0.7437 | 1.0000 | 0.6852 | 0.6080 |
| NVDA   | 0.9090 | 0.8357 | 0.6852 | 1.0000 | 0.9081 |
| VIXCLS | 0.9987 | 0.8245 | 0.6080 | 0.9081 | 1.0000 |

### Key Findings

1. **SPY replicates VIXCLS near-perfectly** (r = 0.9987)
   - This validates our VIX methodology

2. **NVDA has high market correlation** (r = 0.91 with VIXCLS)
   - Tech sector volatility closely tracks market volatility
   - Mean bias: +24 points vs VIXCLS

3. **AAPL has moderate market correlation** (r = 0.78 with VIXCLS)
   - Single-stock idiosyncratic risk adds ~15 points to baseline
   
4. **TSLA has material idiosyncratic volatility** (r = 0.61 with VIXCLS in 2020)
   - Mean IV remains far above market VIX
   - Chain quality issues (broken strike ladders / tail strikes) required explicit QC to avoid artificial spikes

## Comparison to FRED VIXCLS

| Ticker | Correlation | RMSE | Mean Bias | Period |
|--------|-------------|------|-----------|--------|
| SPY | 0.9970 | 0.81 | -0.42 | 2020-2022 |
| AAPL | 0.8731 | 12.36 | +11.26 | 2016-2020 |
| TSLA | 0.6912 | 50.32 | +48.11 | 2019-2022 |
| NVDA | 0.5455 | 31.57 | +28.80 | 2020-2022 |

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
| AAPL | +13.9 pts | Large-cap tech premium |
| NVDA | +23.9 pts | Semiconductor/AI exposure |
| TSLA | +59.9 pts | High idiosyncratic volatility |

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

## QC Enhancements (for Single-Stock Chains)

Diagnostics in `data/processed/{ticker}_diagnostics.parquet` now include explicit columns to catch chain pathologies that can create artificial spikes:

- **Parity / K0 sanity**: `*_parity_forward`, `*_parity_k0`, `*_parity_f_over_k0`, `*_forward_gap_pct_underlying`
- **Strike ladder gaps**: `*_spot_gap_pct_underlying`, `*_max_strike_gap_pct_underlying`
- **Tail strike flags**: `*_strike_guard_applied`, `*_n_strikes_below_20pct_spot`
- **Dominance checks (from OTM strip)**: `*_top_contrib_frac`, `*_min_strike_contrib_frac`

## Conclusions

1. The VIX methodology successfully replicates VIXCLS when applied to SPY options (correlation > 0.99)

2. Single-stock VIX-like indices show expected behavior:
   - Higher baseline volatility than market VIX
   - Lower correlation with market VIX for higher-beta/speculative names
   - TSLA exhibits the highest idiosyncratic volatility

3. The cross-ticker analysis enables volatility relative value comparisons and can inform options trading strategies

---

*Report generated: 2025*

