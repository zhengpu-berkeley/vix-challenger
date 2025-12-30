"""Forward price computation via put-call parity.

Per VIX methodology:
1. Find K* = strike where |C_mid - P_mid| is minimized
2. Compute F = K* + exp(rT) * (C_mid(K*) - P_mid(K*))
3. K0 = max strike where strike <= F

For POC, we use r=0 so exp(rT) = 1.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from vix_challenger.io.spy_csv import Cols


@dataclass
class ForwardResult:
    """Result of forward price computation."""
    
    # The strike K* where put-call parity difference is minimized
    k_star: float
    
    # Forward price F
    forward: float
    
    # ATM strike K0 = max(strike where strike <= F)
    k0: float
    
    # Call midquote at K*
    c_mid_at_k_star: float
    
    # Put midquote at K*
    p_mid_at_k_star: float
    
    # Absolute parity difference at K*
    parity_diff: float
    
    # Number of strikes with valid C/P pairs
    n_valid_strikes: int


def compute_forward_price(
    expiry_df: pl.DataFrame,
    r: float = 0.0,
    dte: Optional[int] = None,
) -> ForwardResult:
    """Compute forward price from put-call parity for a single expiry.
    
    Args:
        expiry_df: DataFrame with options data for ONE expiry.
                   Must have columns: strike, c_mid, p_mid, dte
        r: Risk-free rate (default 0 for POC)
        dte: Days to expiration. If None, taken from data.
        
    Returns:
        ForwardResult with forward price F, ATM strike K0, and diagnostics
        
    Raises:
        ValueError: If no valid strikes with both call and put midquotes
    """
    # Get DTE from data if not provided
    if dte is None:
        dte_values = expiry_df[Cols.DTE].unique().to_list()
        if len(dte_values) != 1:
            raise ValueError(f"Expected single DTE value, got {dte_values}")
        dte = dte_values[0]
    
    # Time to expiry in years
    T = dte / 365.0
    
    # Filter for strikes with valid call and put midquotes
    valid_df = expiry_df.filter(
        pl.col(Cols.C_MID).is_not_null() & 
        pl.col(Cols.P_MID).is_not_null() &
        pl.col(Cols.C_MID).is_not_nan() &
        pl.col(Cols.P_MID).is_not_nan() &
        (pl.col(Cols.C_MID) > 0) &
        (pl.col(Cols.P_MID) > 0)
    ).sort(Cols.STRIKE)
    
    n_valid = len(valid_df)
    if n_valid == 0:
        raise ValueError("No valid strikes with both call and put midquotes")
    
    # Compute parity difference: |C_mid - P_mid|
    valid_df = valid_df.with_columns(
        (pl.col(Cols.C_MID) - pl.col(Cols.P_MID)).abs().alias("parity_diff")
    )
    
    # Find K* = argmin |C_mid - P_mid|
    min_diff_row = valid_df.filter(
        pl.col("parity_diff") == pl.col("parity_diff").min()
    ).head(1)
    
    k_star = min_diff_row[Cols.STRIKE].item()
    c_mid_k_star = min_diff_row[Cols.C_MID].item()
    p_mid_k_star = min_diff_row[Cols.P_MID].item()
    parity_diff = min_diff_row["parity_diff"].item()
    
    # Compute forward: F = K* + exp(rT) * (C_mid(K*) - P_mid(K*))
    exp_rT = np.exp(r * T) if T > 0 else 1.0
    forward = k_star + exp_rT * (c_mid_k_star - p_mid_k_star)
    
    # K0 = max strike where strike <= F
    strikes = valid_df[Cols.STRIKE].to_numpy()
    strikes_below_f = strikes[strikes <= forward]
    
    if len(strikes_below_f) == 0:
        # F is below all strikes, use the lowest strike
        k0 = strikes[0]
    else:
        k0 = strikes_below_f[-1]
    
    return ForwardResult(
        k_star=k_star,
        forward=forward,
        k0=k0,
        c_mid_at_k_star=c_mid_k_star,
        p_mid_at_k_star=p_mid_k_star,
        parity_diff=parity_diff,
        n_valid_strikes=n_valid,
    )


def get_expiry_data(
    day_df: pl.DataFrame,
    expiration,
) -> pl.DataFrame:
    """Extract options data for a specific expiration from a day's data.
    
    Args:
        day_df: DataFrame with all options for one trading day
        expiration: Expiration date to filter for
        
    Returns:
        DataFrame filtered to the specified expiration
    """
    return day_df.filter(pl.col(Cols.EXPIRATION) == expiration)


def list_expirations(day_df: pl.DataFrame) -> pl.DataFrame:
    """List available expirations for a trading day with summary stats.
    
    Args:
        day_df: DataFrame with all options for one trading day
        
    Returns:
        DataFrame with expiration, dte, and strike count
    """
    return (
        day_df
        .group_by([Cols.EXPIRATION, Cols.DTE])
        .agg(
            pl.col(Cols.STRIKE).n_unique().alias("n_strikes"),
            pl.col(Cols.C_MID).drop_nulls().drop_nans().len().alias("n_calls"),
            pl.col(Cols.P_MID).drop_nulls().drop_nans().len().alias("n_puts"),
        )
        .sort(Cols.EXPIRATION)
    )

