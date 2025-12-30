# VIX Methodology

This document describes the Cboe VIX-style model-free variance calculation implemented in this project.

## Overview

The VIX index measures the market's expectation of 30-day volatility implied by S&P 500 (SPX) index option prices. Our implementation adapts this methodology to SPY (S&P 500 ETF) options.

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

