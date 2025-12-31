# VIX Methodology

This document describes the Cboe VIX-style model-free variance calculation implemented in this project.

## Overview

The VIX index measures the market's expectation of 30-day volatility implied by S&P 500 (SPX) index option prices. Our implementation adapts this methodology to **equity/ETF options** (SPY, AAPL, TSLA, NVDA) using end-of-day option chains.

Because single-stock chains (and some Kaggle datasets) can contain **real-world data issues** that violate the assumptions of the VIX white paper, the implementation includes **robustness filters** and **explicit QC outputs** to prevent artificial spikes caused by:

- Illiquid / placeholder quotes (e.g., **bid=0**, wide ask, midquote looks “valid”)
- Broken strike ladders (missing strikes near spot/forward)
- Corporate actions / nonstandard contracts (e.g., **stock split day** where strike/quote scale mismatches underlying price)

## References

- [Cboe VIX White Paper (2019)](https://www.cboe.com/tradable_products/vix/vix_white_paper/)
- Demeterfi, K., Derman, E., Kamal, M., & Zou, J. (1999). "More Than You Ever Wanted to Know About Volatility Swaps"

## Step 1: Select Expirations

For each trading day, select two option expirations:

- **Near-term (T1)**: Maximum expiry with DTE ≤ 30 days
- **Next-term (T2)**: Minimum expiry with DTE > 30 days

This brackets the 30-day target.

## Step 2: Compute Forward Price

For each expiration, compute the forward price using put-call parity:

1. Find the strike `K*` where the absolute difference between call and put midquotes is minimized:

   ```
   K* = argmin |C_mid(K) - P_mid(K)|
   ```

   **Robustness note (important for equities):** in practice, we do *not* search over all strikes blindly. Some chains contain extreme tail strikes (e.g., 1–10 when spot is in the hundreds) with bid=0 and wide asks; these can artificially minimize \(|C-P|\) and produce absurd forwards. We therefore apply a **moneyness-restricted search** around spot first (with fallbacks).

2. Compute the forward price:

   ```
   F = K* + exp(r*T) * (C_mid(K*) - P_mid(K*))
   ```

   Where:
   - `r` = risk-free rate (we use r=0 for POC)
   - `T` = time to expiration in years (DTE / 365)

3. Determine the at-the-money strike:

   ```
   K0 = max{K : K ≤ F}
   ```

   **Robustness note:** `K0` is computed against the **full strike set** for the expiry (not only the parity-candidate subset). We also enforce **sanity checks** downstream to ensure `K0` is close to spot and close to `F` (see “Implementation Notes” and “Diagnostics”).

## Step 3: Build OTM Strip

Construct the out-of-the-money (OTM) option strip:

| Strike Region | Option Used |
|---------------|-------------|
| K < K0 | Put midquote |
| K = K0 | Average of call and put midquotes |
| K > K0 | Call midquote |

### Zero-Bid Cutoff Rule

Apply the "two consecutive zero-bid" cutoff on each wing:

- **Put wing**: Starting from K0 and moving to lower strikes, exclude all strikes below (and including) the first pair of consecutive strikes with zero bid.
- **Call wing**: Starting from K0 and moving to higher strikes, exclude all strikes above (and including) the first pair of consecutive strikes with zero bid.

This prevents deep OTM options with no liquidity from distorting the variance calculation.

### Additional Robustness Filters (Single-Stock Chains)

The official VIX methodology assumes a clean strike ladder and quotes consistent with no-arbitrage bounds. With single-stock EOD chains, we observed several pathologies that can create **artificial variance spikes** via the \(\Delta K / K^2\) weighting. We apply the following filters before building the final OTM strip:

1. **Strike moneyness guard (tail strike suppression)**
   - Keep strikes within a broad envelope around spot: \([0.20\times S,\; 5.00\times S]\)
   - This prevents extreme tiny strikes (e.g., \(K=7\)) from dominating the variance integral.

2. **No-arbitrage bounds on OTM midquotes**
   - For OTM calls (\(K > K_0\)): enforce \(0 \le C_{mid} \le S\)
   - For OTM puts (\(K < K_0\)): enforce \(0 \le P_{mid} \le K\)

3. **Implied-forward consistency filter (split/nonstandard contract detection)**
   - For rows with both call+put mids, compute implied forward:
     \[
     F_i = K + (C_{mid}(K) - P_{mid}(K))
     \]
   - Keep strikes where \(F_i \in [0.5\times S,\; 1.5\times S]\).
   - This reliably catches corporate-action scale issues (e.g., on a split day, some strikes/quotes may be on a pre-split scale while the reported spot is post-split).

### Stock Split Awareness

The implementation includes explicit handling for stock split events via metadata files stored in `data/raw/{ticker}_stock_splits.json`. This addresses a common issue where options data around split dates may contain stale/unadjusted strikes.

**Metadata Loading:**

Split metadata is loaded from JSON files with the following structure:

```json
{
  "ticker": "NVDA",
  "split_ratio": 4.0,
  "split_date": "2021-07-20",
  "company": "NVIDIA Corporation",
  "source": "NVIDIA news release"
}
```

**Split-Aware Processing:**

For each trading day, the pipeline:

1. **Checks proximity to splits**: Uses `is_near_split(ticker, quote_date, window_days=10)` to detect if the quote date is within 10 days of a known split.

2. **Filters absurd strikes**: Near split dates, strikes with extreme moneyness ratios (K/S > 2.5 or K/S < 0.4) are filtered out. These typically represent unadjusted pre-split strikes.

3. **Records diagnostics**: Split-related fields are added to the `DailyVIXResult`:
   - `near_split`: Boolean indicating proximity to a split
   - `days_to_split`: Signed integer (negative = before split, positive = after)
   - `split_ratio`: The split ratio (e.g., 4.0 for a 4:1 split)
   - `strikes_filtered_by_split`: Count of strikes filtered due to split proximity

**Why This Matters:**

On a split day, the underlying price is typically adjusted immediately, but the options chain may contain a mix of:
- New post-split strikes (correct scale)
- Old pre-split strikes (incorrect scale, appears absurdly OTM)

Without split awareness, these stale strikes can:
- Corrupt the forward price calculation (put-call parity finds spurious \(K^*\))
- Dominate the variance integral (the \(1/K^2\) weighting amplifies small strikes)
- Produce artificial VIX spikes of 200%+ on split days

## Step 4: Compute Strike Spacing

For each strike `K_i` in the OTM strip, compute the strike spacing `ΔK_i`:

- **Interior strikes**: `ΔK_i = (K_{i+1} - K_{i-1}) / 2`
- **First strike**: `ΔK_0 = K_1 - K_0`
- **Last strike**: `ΔK_n = K_n - K_{n-1}`

## Step 5: Variance Formula

Compute the model-free variance for each expiry:

```
σ²(T) = (2/T) × Σ_i [ΔK_i / K_i²] × exp(r×T) × Q(K_i) - (1/T) × (F/K0 - 1)²
```

Where:
- `T` = time to expiry in years
- `K_i` = strike prices in the OTM strip
- `ΔK_i` = strike spacing
- `Q(K_i)` = OTM option midquote
- `F` = forward price
- `K0` = ATM strike
- `r` = risk-free rate

### Term Breakdown

1. **Sum term**: `(2/T) × Σ_i [ΔK_i / K_i²] × Q(K_i)`
   - Each strike contributes based on its price and inverse square of strike
   - This is the "variance swap" replication formula

2. **Adjustment term**: `(1/T) × (F/K0 - 1)²`
   - Corrects for the discrete nature of strikes
   - Small when K0 is close to F

### Variance Validity Checks (Practical)

In theory, the model-free variance should be non-negative. In practice, **negative variance** usually indicates a broken chain (strike ladder gaps, scale mismatch, or bad quotes). We therefore:

- **Reject** expiries where `K0` is far from spot (outside \([0.5S, 1.5S]\)) or where \(|F/K_0 - 1|\) is too large (broken ladder symptom).
- **Do not** take `abs(variance)` when variance is negative. Large negative variance is treated as a **skip** for that day/expiry (otherwise it can create large artificial spikes after interpolation).

## Step 6: 30-Day Interpolation

Interpolate between the two expirations to get constant 30-day variance:

```
σ²_30d = w1 × σ²(T1) × (T1/T30) + w2 × σ²(T2) × (T2/T30)
```

Where:
- `T30 = 30/365` (target time horizon)
- `w1`, `w2` are time-based weights that sum to 1

The final VIX-like index:

```
VIX_like = 100 × √(σ²_30d)
```

## Implementation Notes

### POC Simplifications

1. **Risk-free rate**: Using r=0 (can add Treasury rates later)
2. **Time convention**: Using days/365 (official VIX uses minutes)
3. **Underlying**: SPY ETF vs SPX index options

### Known Differences from Official VIX

1. **SPY vs SPX**: SPY options are on the ETF, not the index
2. **Exercise style**: SPY options are American-style (VIX assumes European)
3. **Quote timing**: EOD quotes vs intraday VIX calculation
4. **Strike availability**: May have different strike granularity

## Diagnostics

Key sanity checks for each computation:

1. **σ² > 0**: Variance must be positive
2. **Implied vol in [5%, 150%]**: Typical range for equity volatility
3. **K0 close to F**: Should differ by < 2%
4. **Strikes on both wings**: Should have both OTM puts and calls

### Explicit QC Outputs

For each ticker, we write a per-day diagnostics parquet:

- `data/processed/{ticker}_diagnostics.parquet`

This includes day-level quote/spread/coverage stats *plus* expiry-level QC fields to debug spikes and skips:

- **Strike ladder / gaps**: `*_spot_gap_pct_underlying`, `*_max_strike_gap_pct_underlying`, `*_forward_gap_pct_underlying`
- **Parity sanity**: `*_parity_forward`, `*_parity_k0`, `*_parity_f_over_k0`, `*_parity_diff`
- **Tail strike flags**: `*_strike_guard_applied`, `*_n_strikes_below_20pct_spot`, `*_n_strikes_above_5x_spot`
- **Dominance checks (from the OTM strip integral)**: `*_top_contrib_frac`, `*_min_strike_contrib_frac`

These QC fields were essential for diagnosing:

- TSLA spikes driven by tiny strikes with bid=0 / wide asks (parity + \(\frac{1}{K^2}\) dominance)
- TSLA days with broken strike ladders (e.g., \(K_0\) collapsing far below spot)
- NVDA split-day scale mismatch (OTM calls priced above spot and implied-forward inconsistencies)

### Split-Related Diagnostics

Additional fields track stock split proximity and filtering:

- `near_split`: Boolean - is this date within 10 days of a known split?
- `days_to_split`: Signed integer - days relative to split (negative = before)
- `split_ratio`: The split ratio if near a split (e.g., 4.0 for 4:1)
- `strikes_filtered_by_split`: Count of strikes removed due to split-related filtering

These fields are populated by loading split metadata from `data/raw/{ticker}_stock_splits.json` files.

