"""OTM option strip construction with zero-bid cutoff.

Per VIX methodology:
1. For K < K0: use put midquote (OTM put)
2. For K > K0: use call midquote (OTM call)
3. For K = K0: use average of call and put midquotes
4. Apply "two consecutive zero-bid" cutoff on each wing
5. Compute strike spacing delta_K
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from vix_challenger.io.spy_csv import Cols


@dataclass
class OTMStrip:
    """Out-of-the-money option strip for variance computation."""
    
    # Sorted array of strikes in the strip
    strikes: np.ndarray
    
    # OTM option midquotes Q(K) for each strike
    quotes: np.ndarray
    
    # Strike spacing delta_K for each strike
    delta_k: np.ndarray
    
    # ATM strike K0
    k0: float
    
    # Cutoff strike on put side (lowest strike used)
    put_cutoff: float
    
    # Cutoff strike on call side (highest strike used)
    call_cutoff: float
    
    # Number of puts in strip (K < K0)
    n_puts: int
    
    # Number of calls in strip (K > K0)
    n_calls: int
    
    # Whether K0 is included in strip
    has_k0: bool


def build_otm_strip(
    expiry_df: pl.DataFrame,
    k0: float,
    apply_cutoff: bool = True,
) -> OTMStrip:
    """Build OTM option strip for variance computation.
    
    Args:
        expiry_df: DataFrame with options for one expiry.
                   Must have: strike, c_mid, p_mid, c_bid, p_bid
        k0: ATM strike from forward price computation
        apply_cutoff: Whether to apply zero-bid cutoff rule
        
    Returns:
        OTMStrip with strikes, quotes, and delta_K
    """
    # Sort by strike
    df = expiry_df.sort(Cols.STRIKE)

    # -------------------------------------------------------------------------
    # Strike sanity guard (single-stock chains can contain extreme deep-tail
    # strikes with spurious/non-sense quotes that dominate the 1/K^2 weighting).
    #
    # We keep a broad moneyness envelope around spot to avoid tiny strikes like
    # 1, 7, 20 when spot is in the hundreds. These strikes can have bad quotes
    # (bid/ask glitches) and blow up variance.
    #
    # For index-like underlyings (e.g. SPY), this filter usually has no effect
    # because strike ranges are already reasonable.
    # -------------------------------------------------------------------------
    if Cols.UNDERLYING_PRICE in df.columns:
        spot_series = df[Cols.UNDERLYING_PRICE].drop_nulls().drop_nans()
        if len(spot_series) > 0:
            spot = float(spot_series.head(1).item())
            if spot > 0:
                min_moneyness = 0.20
                max_moneyness = 5.00
                strike_min_guard = spot * min_moneyness
                strike_max_guard = spot * max_moneyness

                # Apply only if the chain includes extreme strikes (otherwise no-op)
                if (df[Cols.STRIKE].min() < strike_min_guard) or (df[Cols.STRIKE].max() > strike_max_guard):
                    df = df.filter(
                        (pl.col(Cols.STRIKE) >= strike_min_guard)
                        & (pl.col(Cols.STRIKE) <= strike_max_guard)
                    )
    
    # Build OTM quote column:
    # - K < K0: use P_MID
    # - K > K0: use C_MID  
    # - K = K0: use (C_MID + P_MID) / 2
    df = df.with_columns(
        pl.when(pl.col(Cols.STRIKE) < k0)
        .then(pl.col(Cols.P_MID))
        .when(pl.col(Cols.STRIKE) > k0)
        .then(pl.col(Cols.C_MID))
        .otherwise((pl.col(Cols.C_MID) + pl.col(Cols.P_MID)) / 2)
        .alias("otm_quote"),
        
        # Also track which bid to use for cutoff
        pl.when(pl.col(Cols.STRIKE) < k0)
        .then(pl.col(Cols.P_BID))
        .when(pl.col(Cols.STRIKE) > k0)
        .then(pl.col(Cols.C_BID))
        .otherwise(pl.max_horizontal(pl.col(Cols.C_BID), pl.col(Cols.P_BID)))
        .alias("otm_bid"),
    )
    
    # Filter out rows with null/nan OTM quotes
    df = df.filter(
        pl.col("otm_quote").is_not_null() &
        pl.col("otm_quote").is_not_nan() &
        (pl.col("otm_quote") > 0)
    )
    
    strikes = df[Cols.STRIKE].to_numpy()
    quotes = df["otm_quote"].to_numpy()
    bids = df["otm_bid"].to_numpy()
    
    if len(strikes) == 0:
        raise ValueError("No valid OTM quotes available")
    
    # Find index of K0 (or closest strike to K0)
    k0_idx = np.argmin(np.abs(strikes - k0))
    actual_k0 = strikes[k0_idx]
    
    # Apply two-consecutive-zero-bid cutoff
    if apply_cutoff:
        put_cutoff_idx, call_cutoff_idx = _apply_zero_bid_cutoff(
            strikes, bids, k0_idx
        )
    else:
        put_cutoff_idx = 0
        call_cutoff_idx = len(strikes) - 1
    
    # Slice to valid range
    strikes = strikes[put_cutoff_idx:call_cutoff_idx + 1]
    quotes = quotes[put_cutoff_idx:call_cutoff_idx + 1]
    
    if len(strikes) == 0:
        raise ValueError("No strikes remaining after cutoff")
    
    # Compute delta_K (strike spacing)
    delta_k = _compute_delta_k(strikes)
    
    # Count puts and calls
    n_puts = np.sum(strikes < actual_k0)
    n_calls = np.sum(strikes > actual_k0)
    has_k0 = actual_k0 in strikes
    
    return OTMStrip(
        strikes=strikes,
        quotes=quotes,
        delta_k=delta_k,
        k0=actual_k0,
        put_cutoff=strikes[0],
        call_cutoff=strikes[-1],
        n_puts=int(n_puts),
        n_calls=int(n_calls),
        has_k0=has_k0,
    )


def _apply_zero_bid_cutoff(
    strikes: np.ndarray,
    bids: np.ndarray,
    k0_idx: int,
) -> tuple[int, int]:
    """Apply two-consecutive-zero-bid cutoff rule.
    
    Per VIX methodology:
    - On put side (K < K0): going down from K0, stop at the first strike
      where TWO consecutive strikes have zero bid
    - On call side (K > K0): going up from K0, stop at the first strike
      where TWO consecutive strikes have zero bid
      
    Args:
        strikes: Sorted array of strikes
        bids: OTM bid prices for each strike
        k0_idx: Index of K0 in the strikes array
        
    Returns:
        (put_cutoff_idx, call_cutoff_idx) - indices of valid range
    """
    n = len(strikes)
    
    # Define "zero bid" threshold (using small epsilon for floating point)
    zero_threshold = 1e-6
    is_zero_bid = bids <= zero_threshold
    
    # Put side: search downward from K0
    put_cutoff_idx = 0
    for i in range(k0_idx - 1, 0, -1):
        # Check if strikes[i] and strikes[i-1] both have zero bid
        if is_zero_bid[i] and is_zero_bid[i - 1]:
            # Stop AFTER the first zero (i.e., include strike at i+1)
            put_cutoff_idx = i + 1
            break
    
    # Call side: search upward from K0
    call_cutoff_idx = n - 1
    for i in range(k0_idx + 1, n - 1):
        # Check if strikes[i] and strikes[i+1] both have zero bid
        if is_zero_bid[i] and is_zero_bid[i + 1]:
            # Stop BEFORE the first zero (i.e., include strike at i-1)
            call_cutoff_idx = i - 1
            break
    
    # Ensure we at least include K0
    put_cutoff_idx = min(put_cutoff_idx, k0_idx)
    call_cutoff_idx = max(call_cutoff_idx, k0_idx)
    
    return put_cutoff_idx, call_cutoff_idx


def _compute_delta_k(strikes: np.ndarray) -> np.ndarray:
    """Compute strike spacing delta_K for each strike.
    
    Per VIX methodology:
    - Interior strikes: delta_K = (K_{i+1} - K_{i-1}) / 2
    - First strike: delta_K = K_1 - K_0 (one-sided)
    - Last strike: delta_K = K_n - K_{n-1} (one-sided)
    
    Args:
        strikes: Sorted array of strikes
        
    Returns:
        Array of delta_K values
    """
    n = len(strikes)
    if n == 0:
        return np.array([])
    if n == 1:
        # Single strike - use a nominal spacing (shouldn't happen in practice)
        return np.array([1.0])
    
    delta_k = np.zeros(n)
    
    # First strike (one-sided)
    delta_k[0] = strikes[1] - strikes[0]
    
    # Interior strikes
    for i in range(1, n - 1):
        delta_k[i] = (strikes[i + 1] - strikes[i - 1]) / 2
    
    # Last strike (one-sided)
    delta_k[n - 1] = strikes[n - 1] - strikes[n - 2]
    
    return delta_k

