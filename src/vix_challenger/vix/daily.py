"""Single-day VIX computation with full diagnostics.

Combines expiry selection, variance computation, and interpolation
into a single function that handles all edge cases gracefully.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import traceback

import numpy as np
import polars as pl

from vix_challenger.io.spy_csv import Cols
from vix_challenger.vix.selection import (
    ExpirySelection,
    SelectionError,
    select_vix_expiries,
)
from vix_challenger.vix.variance import compute_expiry_variance, VarianceResult
from vix_challenger.vix.interpolate import interpolate_and_compute_index


@dataclass
class DailyVIXResult:
    """Result of daily VIX computation with diagnostics."""
    
    # Core result
    quote_date: date
    index: Optional[float] = None  # VIX-like index (100 * sqrt(var_30d))
    var_30d: Optional[float] = None
    
    # Expiry info
    near_exp: Optional[date] = None
    next_exp: Optional[date] = None
    near_dte: Optional[int] = None
    next_dte: Optional[int] = None
    is_standard_bracket: bool = True
    
    # Per-expiry variance
    sigma2_near: Optional[float] = None
    sigma2_next: Optional[float] = None
    iv_near: Optional[float] = None  # Implied vol for near expiry
    iv_next: Optional[float] = None  # Implied vol for next expiry
    
    # Forward prices
    forward_near: Optional[float] = None
    forward_next: Optional[float] = None
    k0_near: Optional[float] = None
    k0_next: Optional[float] = None
    
    # Strike counts
    n_strikes_near: Optional[int] = None
    n_strikes_next: Optional[int] = None

    # --- Expiry-level QC summaries (OTM strip / variance contributions) ---
    # These are intended to help diagnose spikes due to broken strike ladders
    # or tail-strike dominance in the 1/K^2 weighting.
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
    
    # Underlying price
    underlying_price: Optional[float] = None
    
    # Status
    success: bool = False
    skip_reason: Optional[str] = None
    warning: Optional[str] = None
    error_detail: Optional[str] = None


def compute_daily_vix(
    day_df: pl.DataFrame,
    quote_date: date,
    target_dte: int = 30,
    min_dte: int = 1,
    apply_cutoff: bool = True,
    min_strikes: int = 5,
) -> DailyVIXResult:
    """Compute VIX-like index for a single trading day.
    
    Args:
        day_df: DataFrame with all options for one trading day
        quote_date: The trading date
        target_dte: Target days to expiration (default 30)
        min_dte: Minimum DTE for expiry selection (default 1)
        apply_cutoff: Whether to apply zero-bid cutoff
        min_strikes: Minimum strikes required per expiry
        
    Returns:
        DailyVIXResult with index value and diagnostics
    """
    result = DailyVIXResult(quote_date=quote_date)

    def _populate_contrib_qc(prefix: str, var_result: VarianceResult) -> None:
        """Populate strip/contribution QC fields on the result object."""
        if var_result.n_strikes <= 0:
            return
        strikes = var_result.strikes
        contrib = var_result.strike_contributions
        if strikes.size == 0 or contrib.size == 0:
            return
        total = float(np.sum(contrib))
        if total <= 0:
            return

        min_idx = int(np.argmin(strikes))
        top_idx = int(np.argmax(contrib))

        setattr(result, f"{prefix}_strip_strike_min", float(np.min(strikes)))
        setattr(result, f"{prefix}_strip_strike_max", float(np.max(strikes)))
        setattr(result, f"{prefix}_top_contrib_strike", float(strikes[top_idx]))
        setattr(result, f"{prefix}_top_contrib_frac", float(contrib[top_idx]) / total)
        setattr(result, f"{prefix}_min_strike_contrib_frac", float(contrib[min_idx]) / total)
    
    # Get underlying price
    underlying_prices = day_df[Cols.UNDERLYING_PRICE].drop_nulls().drop_nans()
    if len(underlying_prices) > 0:
        result.underlying_price = underlying_prices.head(1).item()
    
    # Step 1: Select expirations
    try:
        selection = select_vix_expiries(day_df, target_dte=target_dte, min_dte=min_dte)
    except SelectionError as e:
        result.skip_reason = e.code
        result.error_detail = e.message
        return result
    
    result.near_exp = selection.near_exp
    result.next_exp = selection.next_exp
    result.near_dte = selection.near_dte
    result.next_dte = selection.next_dte
    result.is_standard_bracket = selection.is_standard_bracket
    
    if selection.note:
        result.warning = selection.note
    
    # Step 2: Compute variance for near-term expiry
    near_df = day_df.filter(pl.col(Cols.EXPIRATION) == selection.near_exp)
    
    try:
        var_near_result = compute_expiry_variance(
            near_df,
            dte=selection.near_dte,
            apply_cutoff=apply_cutoff,
        )
        
        if var_near_result.n_strikes < min_strikes:
            result.skip_reason = "TOO_FEW_STRIKES_NEAR"
            result.error_detail = f"Near expiry has only {var_near_result.n_strikes} strikes (min: {min_strikes})"
            return result
        
        result.sigma2_near = var_near_result.variance
        result.iv_near = var_near_result.implied_vol
        result.forward_near = var_near_result.forward
        result.k0_near = var_near_result.k0
        result.n_strikes_near = var_near_result.n_strikes
        _populate_contrib_qc("near", var_near_result)
        
    except ValueError as e:
        result.skip_reason = "NEAR_VARIANCE_ERROR"
        result.error_detail = str(e)
        return result
    except Exception as e:
        result.skip_reason = "COMPUTATION_ERROR"
        result.error_detail = f"Near expiry: {e}\n{traceback.format_exc()}"
        return result
    
    # Step 3: Compute variance for next-term expiry
    next_df = day_df.filter(pl.col(Cols.EXPIRATION) == selection.next_exp)
    
    try:
        var_next_result = compute_expiry_variance(
            next_df,
            dte=selection.next_dte,
            apply_cutoff=apply_cutoff,
        )
        
        if var_next_result.n_strikes < min_strikes:
            result.skip_reason = "TOO_FEW_STRIKES_NEXT"
            result.error_detail = f"Next expiry has only {var_next_result.n_strikes} strikes (min: {min_strikes})"
            return result
        
        result.sigma2_next = var_next_result.variance
        result.iv_next = var_next_result.implied_vol
        result.forward_next = var_next_result.forward
        result.k0_next = var_next_result.k0
        result.n_strikes_next = var_next_result.n_strikes
        _populate_contrib_qc("next", var_next_result)
        
    except ValueError as e:
        result.skip_reason = "NEXT_VARIANCE_ERROR"
        result.error_detail = str(e)
        return result
    except Exception as e:
        result.skip_reason = "COMPUTATION_ERROR"
        result.error_detail = f"Next expiry: {e}\n{traceback.format_exc()}"
        return result
    
    # Step 4: Handle negative variance
    #
    # Large negative variance indicates a bad chain (e.g., broken strike ladder,
    # forward/K0 mismatch, or bad quotes). Do NOT take abs() here; that can
    # create massive artificial spikes.
    neg_tol = -1e-8
    if result.sigma2_near is not None and result.sigma2_near < 0:
        if result.sigma2_near < neg_tol:
            result.skip_reason = "NEGATIVE_VAR_NEAR"
            result.error_detail = f"Near variance < 0: {result.sigma2_near:.6f}"
            return result
        result.warning = (result.warning or "") + f" NEGATIVE_VAR_NEAR({result.sigma2_near:.6f})"
        result.sigma2_near = 0.0

    if result.sigma2_next is not None and result.sigma2_next < 0:
        if result.sigma2_next < neg_tol:
            result.skip_reason = "NEGATIVE_VAR_NEXT"
            result.error_detail = f"Next variance < 0: {result.sigma2_next:.6f}"
            return result
        result.warning = (result.warning or "") + f" NEGATIVE_VAR_NEXT({result.sigma2_next:.6f})"
        result.sigma2_next = 0.0
    
    # Step 5: Interpolate to 30-day variance
    try:
        var_30d, vix_index = interpolate_and_compute_index(
            var_near=result.sigma2_near,
            dte_near=result.near_dte,
            var_next=result.sigma2_next,
            dte_next=result.next_dte,
            target_dte=target_dte,
        )
        
        result.var_30d = var_30d
        result.index = vix_index
        result.success = True
        
    except Exception as e:
        result.skip_reason = "INTERPOLATION_ERROR"
        result.error_detail = str(e)
        return result
    
    return result


def result_to_dict(result: DailyVIXResult) -> dict:
    """Convert DailyVIXResult to a dictionary for DataFrame creation."""
    return {
        "quote_date": result.quote_date,
        "index": result.index,
        "var_30d": result.var_30d,
        "near_exp": result.near_exp,
        "next_exp": result.next_exp,
        "near_dte": result.near_dte,
        "next_dte": result.next_dte,
        "is_standard_bracket": result.is_standard_bracket,
        "sigma2_near": result.sigma2_near,
        "sigma2_next": result.sigma2_next,
        "iv_near": result.iv_near,
        "iv_next": result.iv_next,
        "forward_near": result.forward_near,
        "forward_next": result.forward_next,
        "k0_near": result.k0_near,
        "k0_next": result.k0_next,
        "n_strikes_near": result.n_strikes_near,
        "n_strikes_next": result.n_strikes_next,
        "near_strip_strike_min": result.near_strip_strike_min,
        "near_strip_strike_max": result.near_strip_strike_max,
        "near_top_contrib_strike": result.near_top_contrib_strike,
        "near_top_contrib_frac": result.near_top_contrib_frac,
        "near_min_strike_contrib_frac": result.near_min_strike_contrib_frac,
        "next_strip_strike_min": result.next_strip_strike_min,
        "next_strip_strike_max": result.next_strip_strike_max,
        "next_top_contrib_strike": result.next_top_contrib_strike,
        "next_top_contrib_frac": result.next_top_contrib_frac,
        "next_min_strike_contrib_frac": result.next_min_strike_contrib_frac,
        "underlying_price": result.underlying_price,
        "success": result.success,
        "skip_reason": result.skip_reason,
        "warning": result.warning,
    }

