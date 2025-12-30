"""Model-free variance computation per VIX methodology.

The variance formula (per expiry T):

    sigma2(T) = (2/T) * sum[(delta_K / K^2) * exp(rT) * Q(K)] - (1/T) * (F/K0 - 1)^2

Where:
- T = time to expiry in years (DTE / 365 for POC)
- K = strike prices in the OTM strip
- delta_K = strike spacing
- Q(K) = OTM option midquote
- F = forward price
- K0 = ATM strike
- r = risk-free rate (0 for POC)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from vix_challenger.io.spy_csv import Cols
from vix_challenger.vix.parity import ForwardResult, compute_forward_price
from vix_challenger.vix.strip import OTMStrip, build_otm_strip


@dataclass
class VarianceResult:
    """Result of variance computation for a single expiry."""
    
    # Model-free variance sigma^2
    variance: float
    
    # Implied volatility (annualized): sqrt(variance) * 100
    implied_vol: float
    
    # Forward price F
    forward: float
    
    # ATM strike K0
    k0: float
    
    # Time to expiry in years
    T: float
    
    # Days to expiration
    dte: int
    
    # Sum term: (2/T) * sum[(delta_K / K^2) * Q(K)]
    sum_term: float
    
    # Adjustment term: (1/T) * (F/K0 - 1)^2
    adjustment_term: float
    
    # Per-strike contributions to variance
    strike_contributions: np.ndarray
    
    # Strikes used in computation
    strikes: np.ndarray
    
    # Number of strikes used
    n_strikes: int
    
    # Number of OTM puts (K < K0)
    n_puts: int
    
    # Number of OTM calls (K > K0)
    n_calls: int


def compute_expiry_variance(
    expiry_df: pl.DataFrame,
    r: float = 0.0,
    dte: Optional[int] = None,
    apply_cutoff: bool = True,
) -> VarianceResult:
    """Compute model-free variance for a single expiry.
    
    Args:
        expiry_df: DataFrame with options for ONE expiry
        r: Risk-free rate (default 0 for POC)
        dte: Days to expiration (if None, taken from data)
        apply_cutoff: Whether to apply zero-bid cutoff
        
    Returns:
        VarianceResult with variance and diagnostics
    """
    # Step 1: Compute forward price and K0
    forward_result = compute_forward_price(expiry_df, r=r, dte=dte)
    F = forward_result.forward
    K0 = forward_result.k0
    
    # Get DTE from data if not provided
    if dte is None:
        dte_values = expiry_df[Cols.DTE].unique().to_list()
        dte = dte_values[0]
    
    # Time to expiry in years
    T = dte / 365.0
    
    # Handle edge case: DTE = 0 (expiring today)
    if T <= 0:
        raise ValueError(f"Cannot compute variance for DTE <= 0 (got {dte})")
    
    # Step 2: Build OTM strip
    strip = build_otm_strip(expiry_df, K0, apply_cutoff=apply_cutoff)
    
    # Step 3: Compute variance
    # exp(rT) factor (1.0 when r=0)
    exp_rT = np.exp(r * T)
    
    # Per-strike contributions: (delta_K / K^2) * exp(rT) * Q(K)
    contributions = (strip.delta_k / strip.strikes**2) * exp_rT * strip.quotes
    
    # Sum term: (2/T) * sum of contributions
    sum_term = (2 / T) * np.sum(contributions)
    
    # Adjustment term: (1/T) * (F/K0 - 1)^2
    adjustment_term = (1 / T) * (F / K0 - 1)**2
    
    # Final variance
    variance = sum_term - adjustment_term
    
    # Implied vol (annualized, as percentage)
    implied_vol = np.sqrt(max(variance, 0)) * 100 if variance > 0 else 0.0
    
    return VarianceResult(
        variance=variance,
        implied_vol=implied_vol,
        forward=F,
        k0=K0,
        T=T,
        dte=dte,
        sum_term=sum_term,
        adjustment_term=adjustment_term,
        strike_contributions=contributions,
        strikes=strip.strikes,
        n_strikes=len(strip.strikes),
        n_puts=strip.n_puts,
        n_calls=strip.n_calls,
    )


def compute_variance_for_day(
    day_df: pl.DataFrame,
    expiration,
    r: float = 0.0,
    apply_cutoff: bool = True,
) -> VarianceResult:
    """Convenience function to compute variance for a specific expiry on a day.
    
    Args:
        day_df: DataFrame with all options for one trading day
        expiration: Expiration date to compute variance for
        r: Risk-free rate
        apply_cutoff: Whether to apply zero-bid cutoff
        
    Returns:
        VarianceResult for the specified expiry
    """
    expiry_df = day_df.filter(pl.col(Cols.EXPIRATION) == expiration)
    
    if len(expiry_df) == 0:
        raise ValueError(f"No data for expiration {expiration}")
    
    return compute_expiry_variance(expiry_df, r=r, apply_cutoff=apply_cutoff)


def print_variance_diagnostics(result: VarianceResult, expiration=None):
    """Print diagnostic information for a variance computation."""
    print("=" * 60)
    if expiration:
        print(f"VARIANCE DIAGNOSTICS - Expiry: {expiration}")
    else:
        print("VARIANCE DIAGNOSTICS")
    print("=" * 60)
    
    print(f"\n[Forward & ATM]")
    print(f"  Forward (F):        {result.forward:.4f}")
    print(f"  ATM Strike (K0):    {result.k0:.2f}")
    print(f"  F/K0 - 1:           {(result.forward / result.k0 - 1)*100:.4f}%")
    
    print(f"\n[Time to Expiry]")
    print(f"  DTE:                {result.dte} days")
    print(f"  T (years):          {result.T:.6f}")
    
    print(f"\n[Strike Coverage]")
    print(f"  Total strikes:      {result.n_strikes}")
    print(f"  OTM puts (K < K0):  {result.n_puts}")
    print(f"  OTM calls (K > K0): {result.n_calls}")
    print(f"  Strike range:       [{result.strikes[0]:.2f}, {result.strikes[-1]:.2f}]")
    
    print(f"\n[Variance Computation]")
    print(f"  Sum term:           {result.sum_term:.6f}")
    print(f"  Adjustment term:    {result.adjustment_term:.6f}")
    print(f"  Variance (sigma²):  {result.variance:.6f}")
    print(f"  Implied Vol:        {result.implied_vol:.2f}%")
    
    # Top contributing strikes
    print(f"\n[Top 5 Strike Contributions]")
    top_idx = np.argsort(result.strike_contributions)[::-1][:5]
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. K={result.strikes[idx]:.2f}: {result.strike_contributions[idx]:.6f}")
    
    print("=" * 60)

