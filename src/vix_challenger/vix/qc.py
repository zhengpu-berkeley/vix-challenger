"""Quality control metrics for VIX computation.

Computes per-day and per-expiry data quality statistics to help
diagnose issues and understand data coverage.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import polars as pl

from vix_challenger.io.spy_csv import Cols


@dataclass
class DayQCMetrics:
    """Data quality metrics for a single trading day."""
    
    quote_date: date
    
    # Overall counts
    n_rows: int
    n_expirations: int
    n_strikes_total: int
    
    # Quote quality
    pct_missing_c_mid: float  # % of rows with null/nan call midquote
    pct_missing_p_mid: float  # % of rows with null/nan put midquote
    pct_missing_both: float   # % of rows missing both C and P
    pct_zero_c_bid: float     # % of rows with zero call bid
    pct_zero_p_bid: float     # % of rows with zero put bid
    
    # Spread quality
    median_c_spread: float    # Median call bid-ask spread
    median_p_spread: float    # Median put bid-ask spread
    median_c_spread_pct: float  # Median call spread as % of mid
    median_p_spread_pct: float  # Median put spread as % of mid
    
    # Strike coverage (relative to underlying)
    min_strike_pct: float     # Min strike / underlying
    max_strike_pct: float     # Max strike / underlying
    
    # Underlying
    underlying_price: Optional[float] = None


def compute_day_qc_metrics(
    day_df: pl.DataFrame,
    quote_date: date,
) -> DayQCMetrics:
    """Compute quality control metrics for a single trading day.
    
    Args:
        day_df: DataFrame with all options for one trading day
        quote_date: The trading date
        
    Returns:
        DayQCMetrics with quality statistics
    """
    n_rows = len(day_df)
    
    if n_rows == 0:
        return DayQCMetrics(
            quote_date=quote_date,
            n_rows=0,
            n_expirations=0,
            n_strikes_total=0,
            pct_missing_c_mid=100.0,
            pct_missing_p_mid=100.0,
            pct_missing_both=100.0,
            pct_zero_c_bid=100.0,
            pct_zero_p_bid=100.0,
            median_c_spread=0.0,
            median_p_spread=0.0,
            median_c_spread_pct=0.0,
            median_p_spread_pct=0.0,
            min_strike_pct=0.0,
            max_strike_pct=0.0,
        )
    
    # Underlying price
    underlying = day_df[Cols.UNDERLYING_PRICE].drop_nulls().drop_nans()
    underlying_price = underlying.head(1).item() if len(underlying) > 0 else None
    
    # Counts
    n_expirations = day_df[Cols.EXPIRATION].n_unique()
    n_strikes_total = day_df[Cols.STRIKE].n_unique()
    
    # Missing midquotes
    c_mid_missing = (
        day_df[Cols.C_MID].is_null() | 
        day_df[Cols.C_MID].is_nan()
    ).sum()
    p_mid_missing = (
        day_df[Cols.P_MID].is_null() | 
        day_df[Cols.P_MID].is_nan()
    ).sum()
    both_missing = (
        (day_df[Cols.C_MID].is_null() | day_df[Cols.C_MID].is_nan()) &
        (day_df[Cols.P_MID].is_null() | day_df[Cols.P_MID].is_nan())
    ).sum()
    
    pct_missing_c_mid = 100.0 * c_mid_missing / n_rows
    pct_missing_p_mid = 100.0 * p_mid_missing / n_rows
    pct_missing_both = 100.0 * both_missing / n_rows
    
    # Zero bids
    c_bid_zero = (day_df[Cols.C_BID] <= 0).sum()
    p_bid_zero = (day_df[Cols.P_BID] <= 0).sum()
    pct_zero_c_bid = 100.0 * c_bid_zero / n_rows
    pct_zero_p_bid = 100.0 * p_bid_zero / n_rows
    
    # Spreads
    c_spread = (day_df[Cols.C_ASK] - day_df[Cols.C_BID]).drop_nulls().drop_nans()
    p_spread = (day_df[Cols.P_ASK] - day_df[Cols.P_BID]).drop_nulls().drop_nans()
    
    median_c_spread = c_spread.median() if len(c_spread) > 0 else 0.0
    median_p_spread = p_spread.median() if len(p_spread) > 0 else 0.0
    
    # Spread as % of mid
    c_spread_pct_raw = (
        (day_df[Cols.C_ASK] - day_df[Cols.C_BID]) / day_df[Cols.C_MID] * 100
    ).drop_nulls().drop_nans()
    # Filter out extreme values (> 1000% spread)
    c_spread_pct = c_spread_pct_raw.filter(c_spread_pct_raw < 1000)
    
    p_spread_pct_raw = (
        (day_df[Cols.P_ASK] - day_df[Cols.P_BID]) / day_df[Cols.P_MID] * 100
    ).drop_nulls().drop_nans()
    p_spread_pct = p_spread_pct_raw.filter(p_spread_pct_raw < 1000)
    
    median_c_spread_pct = c_spread_pct.median() if len(c_spread_pct) > 0 else 0.0
    median_p_spread_pct = p_spread_pct.median() if len(p_spread_pct) > 0 else 0.0
    
    # Strike coverage
    strikes = day_df[Cols.STRIKE].drop_nulls().drop_nans()
    if len(strikes) > 0 and underlying_price and underlying_price > 0:
        min_strike_pct = 100.0 * strikes.min() / underlying_price
        max_strike_pct = 100.0 * strikes.max() / underlying_price
    else:
        min_strike_pct = 0.0
        max_strike_pct = 0.0
    
    return DayQCMetrics(
        quote_date=quote_date,
        n_rows=n_rows,
        n_expirations=n_expirations,
        n_strikes_total=n_strikes_total,
        pct_missing_c_mid=pct_missing_c_mid,
        pct_missing_p_mid=pct_missing_p_mid,
        pct_missing_both=pct_missing_both,
        pct_zero_c_bid=pct_zero_c_bid,
        pct_zero_p_bid=pct_zero_p_bid,
        median_c_spread=median_c_spread,
        median_p_spread=median_p_spread,
        median_c_spread_pct=median_c_spread_pct,
        median_p_spread_pct=median_p_spread_pct,
        min_strike_pct=min_strike_pct,
        max_strike_pct=max_strike_pct,
        underlying_price=underlying_price,
    )


def qc_metrics_to_dict(metrics: DayQCMetrics) -> dict:
    """Convert DayQCMetrics to dictionary for DataFrame creation."""
    return {
        "quote_date": metrics.quote_date,
        "n_rows": metrics.n_rows,
        "n_expirations": metrics.n_expirations,
        "n_strikes_total": metrics.n_strikes_total,
        "pct_missing_c_mid": metrics.pct_missing_c_mid,
        "pct_missing_p_mid": metrics.pct_missing_p_mid,
        "pct_missing_both": metrics.pct_missing_both,
        "pct_zero_c_bid": metrics.pct_zero_c_bid,
        "pct_zero_p_bid": metrics.pct_zero_p_bid,
        "median_c_spread": metrics.median_c_spread,
        "median_p_spread": metrics.median_p_spread,
        "median_c_spread_pct": metrics.median_c_spread_pct,
        "median_p_spread_pct": metrics.median_p_spread_pct,
        "min_strike_pct": metrics.min_strike_pct,
        "max_strike_pct": metrics.max_strike_pct,
        "underlying_price": metrics.underlying_price,
    }


def summarize_skip_reasons(results: list) -> dict:
    """Summarize skip reasons from a list of DailyVIXResult objects.
    
    Args:
        results: List of DailyVIXResult objects
        
    Returns:
        Dictionary mapping skip_reason to count
    """
    from collections import Counter
    
    skip_reasons = [r.skip_reason for r in results if r.skip_reason is not None]
    return dict(Counter(skip_reasons))

