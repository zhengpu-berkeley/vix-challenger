"""Expiry selection for VIX computation.

Per VIX methodology, select two expirations bracketing 30 days:
- near_exp: max expiry with DTE <= 30
- next_exp: min expiry with DTE > 30

Handles edge cases where standard bracketing isn't possible.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import polars as pl

from vix_challenger.io.spy_csv import Cols


@dataclass
class ExpirySelection:
    """Result of expiry selection for VIX computation."""
    
    # Near-term expiration (typically DTE <= 30)
    near_exp: date
    near_dte: int
    
    # Next-term expiration (typically DTE > 30)
    next_exp: date
    next_dte: int
    
    # Whether this is a standard bracketing (near <= 30 < next)
    is_standard_bracket: bool
    
    # Warning/note about selection
    note: Optional[str] = None


class SelectionError(Exception):
    """Error during expiry selection."""
    
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def select_vix_expiries(
    day_df: pl.DataFrame,
    target_dte: int = 30,
    min_dte: int = 1,
) -> ExpirySelection:
    """Select near-term and next-term expirations for VIX computation.
    
    Standard VIX uses expirations bracketing 30 days:
    - near_exp: max expiry with DTE <= 30 (and DTE >= min_dte)
    - next_exp: min expiry with DTE > 30
    
    Edge cases:
    - No DTE <= 30: use two nearest above 30
    - No DTE > 30: use two nearest below 30
    - Only one valid expiry: raise SelectionError
    
    Args:
        day_df: DataFrame with options for one trading day
        target_dte: Target days to expiration (default 30)
        min_dte: Minimum DTE to consider (default 1, excludes same-day)
        
    Returns:
        ExpirySelection with near and next expirations
        
    Raises:
        SelectionError: If unable to select two valid expirations
    """
    # Get unique expirations with their DTE
    expirations = (
        day_df
        .select([Cols.EXPIRATION, Cols.DTE])
        .unique()
        .filter(pl.col(Cols.DTE) >= min_dte)  # Exclude same-day and past
        .sort(Cols.DTE)
    )
    
    if len(expirations) == 0:
        raise SelectionError(
            "NO_VALID_EXPIRIES",
            f"No expirations with DTE >= {min_dte}"
        )
    
    if len(expirations) == 1:
        raise SelectionError(
            "SINGLE_EXPIRY",
            f"Only one valid expiry found: DTE={expirations[Cols.DTE].item()}"
        )
    
    # Split into near (DTE <= target) and next (DTE > target)
    near_candidates = expirations.filter(pl.col(Cols.DTE) <= target_dte)
    next_candidates = expirations.filter(pl.col(Cols.DTE) > target_dte)
    
    # Standard case: have both near and next
    if len(near_candidates) > 0 and len(next_candidates) > 0:
        # near = max DTE <= target
        near_row = near_candidates.sort(Cols.DTE, descending=True).head(1)
        near_exp = near_row[Cols.EXPIRATION].item()
        near_dte = near_row[Cols.DTE].item()
        
        # next = min DTE > target
        next_row = next_candidates.sort(Cols.DTE).head(1)
        next_exp = next_row[Cols.EXPIRATION].item()
        next_dte = next_row[Cols.DTE].item()
        
        return ExpirySelection(
            near_exp=near_exp,
            near_dte=near_dte,
            next_exp=next_exp,
            next_dte=next_dte,
            is_standard_bracket=True,
        )
    
    # Edge case: No near-term (all DTE > target)
    if len(near_candidates) == 0:
        # Use two nearest expirations above target
        sorted_exp = expirations.sort(Cols.DTE).head(2)
        
        near_exp = sorted_exp[0, Cols.EXPIRATION]
        near_dte = sorted_exp[0, Cols.DTE]
        next_exp = sorted_exp[1, Cols.EXPIRATION]
        next_dte = sorted_exp[1, Cols.DTE]
        
        return ExpirySelection(
            near_exp=near_exp,
            near_dte=near_dte,
            next_exp=next_exp,
            next_dte=next_dte,
            is_standard_bracket=False,
            note=f"No DTE <= {target_dte}; using two nearest above",
        )
    
    # Edge case: No next-term (all DTE <= target)
    if len(next_candidates) == 0:
        # Use two nearest expirations below target
        sorted_exp = expirations.sort(Cols.DTE, descending=True).head(2)
        
        # Reverse so near < next in DTE
        near_exp = sorted_exp[1, Cols.EXPIRATION]
        near_dte = sorted_exp[1, Cols.DTE]
        next_exp = sorted_exp[0, Cols.EXPIRATION]
        next_dte = sorted_exp[0, Cols.DTE]
        
        return ExpirySelection(
            near_exp=near_exp,
            near_dte=near_dte,
            next_exp=next_exp,
            next_dte=next_dte,
            is_standard_bracket=False,
            note=f"No DTE > {target_dte}; using two nearest below",
        )
    
    # Should never reach here
    raise SelectionError("SELECTION_LOGIC_ERROR", "Unexpected state in expiry selection")


def get_available_expirations(day_df: pl.DataFrame, min_dte: int = 0) -> pl.DataFrame:
    """Get summary of available expirations for a trading day.
    
    Args:
        day_df: DataFrame with options for one trading day
        min_dte: Minimum DTE to include
        
    Returns:
        DataFrame with expiration, dte, n_strikes, n_valid_quotes
    """
    return (
        day_df
        .filter(pl.col(Cols.DTE) >= min_dte)
        .group_by([Cols.EXPIRATION, Cols.DTE])
        .agg([
            pl.col(Cols.STRIKE).n_unique().alias("n_strikes"),
            # Count strikes with valid C and P midquotes
            (
                pl.col(Cols.C_MID).is_not_null() & 
                pl.col(Cols.P_MID).is_not_null() &
                (pl.col(Cols.C_MID) > 0) &
                (pl.col(Cols.P_MID) > 0)
            ).sum().alias("n_valid_pairs"),
        ])
        .sort(Cols.DTE)
    )

