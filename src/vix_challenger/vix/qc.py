"""Quality control metrics for VIX computation.

Computes per-day and per-expiry data quality statistics to help
diagnose issues and understand data coverage.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

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

    # --- VIX computation context (optional) ---
    vix_success: Optional[bool] = None
    vix_skip_reason: Optional[str] = None
    vix_warning: Optional[str] = None

    near_exp: Optional[date] = None
    next_exp: Optional[date] = None
    near_dte: Optional[int] = None
    next_dte: Optional[int] = None

    # --- Expiry-level parity / strike-ladder QC (raw chain) ---
    # Near expiry
    near_n_rows: Optional[int] = None
    near_n_strikes: Optional[int] = None
    near_strike_min_raw: Optional[float] = None
    near_strike_max_raw: Optional[float] = None
    near_min_strike_pct_underlying: Optional[float] = None
    near_max_strike_pct_underlying: Optional[float] = None
    near_max_strike_gap: Optional[float] = None
    near_max_strike_gap_pct_underlying: Optional[float] = None
    near_spot_gap: Optional[float] = None
    near_spot_gap_pct_underlying: Optional[float] = None
    near_n_strikes_below_20pct_spot: Optional[int] = None
    near_n_strikes_above_5x_spot: Optional[int] = None
    near_strike_guard_applied: Optional[bool] = None
    near_parity_k_star: Optional[float] = None
    near_parity_k_star_pct_underlying: Optional[float] = None
    near_parity_diff: Optional[float] = None
    near_parity_n_valid: Optional[int] = None
    near_parity_forward: Optional[float] = None
    near_parity_k0: Optional[float] = None
    near_parity_k0_pct_underlying: Optional[float] = None
    near_parity_f_over_k0: Optional[float] = None
    near_n_strikes_le_forward: Optional[int] = None
    near_forward_gap: Optional[float] = None
    near_forward_gap_pct_underlying: Optional[float] = None
    near_parity_error: Optional[str] = None

    # Next expiry
    next_n_rows: Optional[int] = None
    next_n_strikes: Optional[int] = None
    next_strike_min_raw: Optional[float] = None
    next_strike_max_raw: Optional[float] = None
    next_min_strike_pct_underlying: Optional[float] = None
    next_max_strike_pct_underlying: Optional[float] = None
    next_max_strike_gap: Optional[float] = None
    next_max_strike_gap_pct_underlying: Optional[float] = None
    next_spot_gap: Optional[float] = None
    next_spot_gap_pct_underlying: Optional[float] = None
    next_n_strikes_below_20pct_spot: Optional[int] = None
    next_n_strikes_above_5x_spot: Optional[int] = None
    next_strike_guard_applied: Optional[bool] = None
    next_parity_k_star: Optional[float] = None
    next_parity_k_star_pct_underlying: Optional[float] = None
    next_parity_diff: Optional[float] = None
    next_parity_n_valid: Optional[int] = None
    next_parity_forward: Optional[float] = None
    next_parity_k0: Optional[float] = None
    next_parity_k0_pct_underlying: Optional[float] = None
    next_parity_f_over_k0: Optional[float] = None
    next_n_strikes_le_forward: Optional[int] = None
    next_forward_gap: Optional[float] = None
    next_forward_gap_pct_underlying: Optional[float] = None
    next_parity_error: Optional[str] = None

    # --- Expiry-level OTM strip dominance QC (from computed variance) ---
    near_strip_strike_min: Optional[float] = None
    near_strip_strike_max: Optional[float] = None
    near_top_contrib_strike: Optional[float] = None
    near_top_contrib_frac: Optional[float] = None
    near_min_strike_contrib_frac: Optional[float] = None

    next_strip_strike_min: Optional[float] = None
    next_strip_strike_max: Optional[float] = None
    next_top_contrib_strike: Optional[float] = None
    next_top_contrib_frac: Optional[float] = None
    next_min_strike_contrib_frac: Optional[float] = None


def compute_day_qc_metrics(
    day_df: pl.DataFrame,
    quote_date: date,
    vix_result: Optional[Any] = None,
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
    
    metrics = DayQCMetrics(
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

    # ---------------------------------------------------------------------
    # Expiry-level QC for the selected near/next expiries (if provided).
    # ---------------------------------------------------------------------
    if vix_result is None:
        return metrics

    metrics.vix_success = getattr(vix_result, "success", None)
    metrics.vix_skip_reason = getattr(vix_result, "skip_reason", None)
    metrics.vix_warning = getattr(vix_result, "warning", None)

    metrics.near_exp = getattr(vix_result, "near_exp", None)
    metrics.next_exp = getattr(vix_result, "next_exp", None)
    metrics.near_dte = getattr(vix_result, "near_dte", None)
    metrics.next_dte = getattr(vix_result, "next_dte", None)

    # Copy strip dominance QC computed during variance calculation (if present)
    for field in [
        "near_strip_strike_min",
        "near_strip_strike_max",
        "near_top_contrib_strike",
        "near_top_contrib_frac",
        "near_min_strike_contrib_frac",
        "next_strip_strike_min",
        "next_strip_strike_max",
        "next_top_contrib_strike",
        "next_top_contrib_frac",
        "next_min_strike_contrib_frac",
    ]:
        setattr(metrics, field, getattr(vix_result, field, None))

    def _expiry_qc(prefix: str, expiration) -> None:
        if expiration is None or underlying_price is None or underlying_price <= 0:
            return

        expiry_df = day_df.filter(pl.col(Cols.EXPIRATION) == pl.lit(expiration))
        if len(expiry_df) == 0:
            return

        strikes_s = expiry_df[Cols.STRIKE].drop_nulls().drop_nans().unique()
        if len(strikes_s) == 0:
            return

        strikes = np.sort(strikes_s.to_numpy())
        n_strikes = int(len(strikes))
        strike_min = float(strikes[0])
        strike_max = float(strikes[-1])

        min_pct = 100.0 * strike_min / underlying_price
        max_pct = 100.0 * strike_max / underlying_price

        gaps = np.diff(strikes)
        max_gap = float(np.max(gaps)) if gaps.size > 0 else 0.0
        max_gap_pct = 100.0 * max_gap / underlying_price if underlying_price > 0 else None

        # Gap around spot (nearest strikes below/above spot)
        below_spot = strikes[strikes <= underlying_price]
        above_spot = strikes[strikes > underlying_price]
        spot_below = float(below_spot[-1]) if below_spot.size > 0 else None
        spot_above = float(above_spot[0]) if above_spot.size > 0 else None
        spot_gap = (spot_above - spot_below) if (spot_below is not None and spot_above is not None) else None
        spot_gap_pct = (100.0 * spot_gap / underlying_price) if (spot_gap is not None and underlying_price > 0) else None

        # Tail strike counts relative to spot (matches strip guard thresholds)
        lo_guard = 0.20 * underlying_price
        hi_guard = 5.00 * underlying_price
        n_below = int(np.sum(strikes < lo_guard))
        n_above = int(np.sum(strikes > hi_guard))
        guard_applied = (strike_min < lo_guard) or (strike_max > hi_guard)

        setattr(metrics, f"{prefix}_n_rows", int(len(expiry_df)))
        setattr(metrics, f"{prefix}_n_strikes", n_strikes)
        setattr(metrics, f"{prefix}_strike_min_raw", strike_min)
        setattr(metrics, f"{prefix}_strike_max_raw", strike_max)
        setattr(metrics, f"{prefix}_min_strike_pct_underlying", min_pct)
        setattr(metrics, f"{prefix}_max_strike_pct_underlying", max_pct)
        setattr(metrics, f"{prefix}_max_strike_gap", max_gap)
        setattr(metrics, f"{prefix}_max_strike_gap_pct_underlying", max_gap_pct)
        setattr(metrics, f"{prefix}_spot_gap", spot_gap)
        setattr(metrics, f"{prefix}_spot_gap_pct_underlying", spot_gap_pct)
        setattr(metrics, f"{prefix}_n_strikes_below_20pct_spot", n_below)
        setattr(metrics, f"{prefix}_n_strikes_above_5x_spot", n_above)
        setattr(metrics, f"{prefix}_strike_guard_applied", bool(guard_applied))

        # Put-call parity diagnostics
        try:
            from vix_challenger.vix.parity import compute_forward_price

            fr = compute_forward_price(expiry_df)
            fwd = float(fr.forward)
            k0 = float(fr.k0)
            k_star = float(fr.k_star)

            k_star_pct = 100.0 * k_star / underlying_price if underlying_price > 0 else None
            k0_pct = 100.0 * k0 / underlying_price if underlying_price > 0 else None
            f_over_k0 = (fwd / k0) if k0 > 0 else None

            n_le_f = int(np.sum(strikes <= fwd))

            below_f = strikes[strikes <= fwd]
            above_f = strikes[strikes > fwd]
            f_below = float(below_f[-1]) if below_f.size > 0 else None
            f_above = float(above_f[0]) if above_f.size > 0 else None
            f_gap = (f_above - f_below) if (f_below is not None and f_above is not None) else None
            f_gap_pct = (100.0 * f_gap / underlying_price) if (f_gap is not None and underlying_price > 0) else None

            setattr(metrics, f"{prefix}_parity_k_star", k_star)
            setattr(metrics, f"{prefix}_parity_k_star_pct_underlying", k_star_pct)
            setattr(metrics, f"{prefix}_parity_diff", float(fr.parity_diff))
            setattr(metrics, f"{prefix}_parity_n_valid", int(fr.n_valid_strikes))
            setattr(metrics, f"{prefix}_parity_forward", fwd)
            setattr(metrics, f"{prefix}_parity_k0", k0)
            setattr(metrics, f"{prefix}_parity_k0_pct_underlying", k0_pct)
            setattr(metrics, f"{prefix}_parity_f_over_k0", f_over_k0)
            setattr(metrics, f"{prefix}_n_strikes_le_forward", n_le_f)
            setattr(metrics, f"{prefix}_forward_gap", f_gap)
            setattr(metrics, f"{prefix}_forward_gap_pct_underlying", f_gap_pct)

        except Exception as e:
            setattr(metrics, f"{prefix}_parity_error", str(e))

    _expiry_qc("near", metrics.near_exp)
    _expiry_qc("next", metrics.next_exp)

    return metrics


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

        # VIX context
        "vix_success": metrics.vix_success,
        "vix_skip_reason": metrics.vix_skip_reason,
        "vix_warning": metrics.vix_warning,
        "near_exp": metrics.near_exp,
        "next_exp": metrics.next_exp,
        "near_dte": metrics.near_dte,
        "next_dte": metrics.next_dte,

        # Near expiry raw-chain QC
        "near_n_rows": metrics.near_n_rows,
        "near_n_strikes": metrics.near_n_strikes,
        "near_strike_min_raw": metrics.near_strike_min_raw,
        "near_strike_max_raw": metrics.near_strike_max_raw,
        "near_min_strike_pct_underlying": metrics.near_min_strike_pct_underlying,
        "near_max_strike_pct_underlying": metrics.near_max_strike_pct_underlying,
        "near_max_strike_gap": metrics.near_max_strike_gap,
        "near_max_strike_gap_pct_underlying": metrics.near_max_strike_gap_pct_underlying,
        "near_spot_gap": metrics.near_spot_gap,
        "near_spot_gap_pct_underlying": metrics.near_spot_gap_pct_underlying,
        "near_n_strikes_below_20pct_spot": metrics.near_n_strikes_below_20pct_spot,
        "near_n_strikes_above_5x_spot": metrics.near_n_strikes_above_5x_spot,
        "near_strike_guard_applied": metrics.near_strike_guard_applied,
        "near_parity_k_star": metrics.near_parity_k_star,
        "near_parity_k_star_pct_underlying": metrics.near_parity_k_star_pct_underlying,
        "near_parity_diff": metrics.near_parity_diff,
        "near_parity_n_valid": metrics.near_parity_n_valid,
        "near_parity_forward": metrics.near_parity_forward,
        "near_parity_k0": metrics.near_parity_k0,
        "near_parity_k0_pct_underlying": metrics.near_parity_k0_pct_underlying,
        "near_parity_f_over_k0": metrics.near_parity_f_over_k0,
        "near_n_strikes_le_forward": metrics.near_n_strikes_le_forward,
        "near_forward_gap": metrics.near_forward_gap,
        "near_forward_gap_pct_underlying": metrics.near_forward_gap_pct_underlying,
        "near_parity_error": metrics.near_parity_error,

        # Next expiry raw-chain QC
        "next_n_rows": metrics.next_n_rows,
        "next_n_strikes": metrics.next_n_strikes,
        "next_strike_min_raw": metrics.next_strike_min_raw,
        "next_strike_max_raw": metrics.next_strike_max_raw,
        "next_min_strike_pct_underlying": metrics.next_min_strike_pct_underlying,
        "next_max_strike_pct_underlying": metrics.next_max_strike_pct_underlying,
        "next_max_strike_gap": metrics.next_max_strike_gap,
        "next_max_strike_gap_pct_underlying": metrics.next_max_strike_gap_pct_underlying,
        "next_spot_gap": metrics.next_spot_gap,
        "next_spot_gap_pct_underlying": metrics.next_spot_gap_pct_underlying,
        "next_n_strikes_below_20pct_spot": metrics.next_n_strikes_below_20pct_spot,
        "next_n_strikes_above_5x_spot": metrics.next_n_strikes_above_5x_spot,
        "next_strike_guard_applied": metrics.next_strike_guard_applied,
        "next_parity_k_star": metrics.next_parity_k_star,
        "next_parity_k_star_pct_underlying": metrics.next_parity_k_star_pct_underlying,
        "next_parity_diff": metrics.next_parity_diff,
        "next_parity_n_valid": metrics.next_parity_n_valid,
        "next_parity_forward": metrics.next_parity_forward,
        "next_parity_k0": metrics.next_parity_k0,
        "next_parity_k0_pct_underlying": metrics.next_parity_k0_pct_underlying,
        "next_parity_f_over_k0": metrics.next_parity_f_over_k0,
        "next_n_strikes_le_forward": metrics.next_n_strikes_le_forward,
        "next_forward_gap": metrics.next_forward_gap,
        "next_forward_gap_pct_underlying": metrics.next_forward_gap_pct_underlying,
        "next_parity_error": metrics.next_parity_error,

        # Strip dominance QC (from variance)
        "near_strip_strike_min": metrics.near_strip_strike_min,
        "near_strip_strike_max": metrics.near_strip_strike_max,
        "near_top_contrib_strike": metrics.near_top_contrib_strike,
        "near_top_contrib_frac": metrics.near_top_contrib_frac,
        "near_min_strike_contrib_frac": metrics.near_min_strike_contrib_frac,
        "next_strip_strike_min": metrics.next_strip_strike_min,
        "next_strip_strike_max": metrics.next_strip_strike_max,
        "next_top_contrib_strike": metrics.next_top_contrib_strike,
        "next_top_contrib_frac": metrics.next_top_contrib_frac,
        "next_min_strike_contrib_frac": metrics.next_min_strike_contrib_frac,
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

