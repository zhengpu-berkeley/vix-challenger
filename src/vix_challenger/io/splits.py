"""
Stock split metadata loader and utilities.

Provides functions to load split data and determine split-related
adjustments for options data processing.
"""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from ..config import RAW_DATA_DIR


@dataclass
class SplitEvent:
    """Represents a single stock split event."""
    
    ticker: str
    split_ratio: float
    split_date: date
    company: str = ""
    announced_date: Optional[date] = None
    source: str = ""
    
    def __post_init__(self):
        # Convert string dates to date objects if needed
        if isinstance(self.split_date, str):
            self.split_date = date.fromisoformat(self.split_date)
        if isinstance(self.announced_date, str):
            self.announced_date = date.fromisoformat(self.announced_date)


def load_splits(ticker: str) -> list[SplitEvent]:
    """
    Load stock split history for a ticker.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA')
        
    Returns
    -------
    list[SplitEvent]
        List of split events, sorted by date (newest first)
    """
    ticker_lower = ticker.lower()
    split_file = RAW_DATA_DIR / f"{ticker_lower}_stock_splits.json"
    
    if not split_file.exists():
        return []
    
    with open(split_file) as f:
        data = json.load(f)
    
    splits = []
    for item in data:
        split = SplitEvent(
            ticker=item.get("ticker", ticker.upper()),
            split_ratio=float(item["split_ratio"]),
            split_date=item["split_date"],
            company=item.get("company", ""),
            announced_date=item.get("announced_date"),
            source=item.get("source", ""),
        )
        splits.append(split)
    
    # Sort by date, newest first
    splits.sort(key=lambda s: s.split_date, reverse=True)
    return splits


def get_splits_in_range(
    ticker: str,
    start_date: date,
    end_date: date,
) -> list[SplitEvent]:
    """
    Get splits that occurred within a date range.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start_date : date
        Start of date range (inclusive)
    end_date : date
        End of date range (inclusive)
        
    Returns
    -------
    list[SplitEvent]
        Splits within the range, sorted newest first
    """
    all_splits = load_splits(ticker)
    return [s for s in all_splits if start_date <= s.split_date <= end_date]


def get_most_recent_split(
    ticker: str,
    as_of_date: date,
) -> Optional[SplitEvent]:
    """
    Get the most recent split on or before a given date.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    as_of_date : date
        Reference date
        
    Returns
    -------
    SplitEvent or None
        Most recent split, or None if no splits before this date
    """
    splits = load_splits(ticker)
    for split in splits:
        if split.split_date <= as_of_date:
            return split
    return None


def is_near_split(
    ticker: str,
    query_date: date,
    window_days: int = 5,
) -> tuple[bool, Optional[SplitEvent]]:
    """
    Check if a date is within a window of any split.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    query_date : date
        Date to check
    window_days : int
        Number of days before/after split to consider "near"
        
    Returns
    -------
    tuple[bool, SplitEvent or None]
        (is_near, nearest_split) - True if within window, and the relevant split
    """
    splits = load_splits(ticker)
    
    for split in splits:
        days_diff = abs((query_date - split.split_date).days)
        if days_diff <= window_days:
            return True, split
    
    return False, None


def get_cumulative_split_factor(
    ticker: str,
    from_date: date,
    to_date: date,
) -> float:
    """
    Calculate the cumulative split adjustment factor between two dates.
    
    This is useful for adjusting historical prices/strikes to a common basis.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    from_date : date
        Start date
    to_date : date
        End date
        
    Returns
    -------
    float
        Cumulative factor. Values > 1 mean the price/strike should be divided
        to convert from from_date basis to to_date basis.
        
    Example
    -------
    If there was a 4:1 split between from_date and to_date,
    returns 4.0 (meaning old prices should be divided by 4).
    """
    splits = load_splits(ticker)
    
    factor = 1.0
    for split in splits:
        if from_date < split.split_date <= to_date:
            factor *= split.split_ratio
    
    return factor


def should_filter_strike(
    strike: float,
    underlying_price: float,
    ticker: str,
    quote_date: date,
    max_otm_ratio: float = 2.5,
) -> bool:
    """
    Determine if a strike should be filtered out due to split-related issues.
    
    After a split, old (unadjusted) strikes may still appear in the data.
    These will be absurdly far OTM and should be excluded.
    
    Parameters
    ----------
    strike : float
        Strike price
    underlying_price : float
        Current underlying price
    ticker : str
        Stock ticker symbol
    quote_date : date
        Quote date
    max_otm_ratio : float
        Maximum allowed K/S ratio for OTM options (default 2.5)
        
    Returns
    -------
    bool
        True if the strike should be filtered out
    """
    near, split = is_near_split(ticker, quote_date, window_days=10)
    
    if not near and split is None:
        # Check if we're after any split (could have stale strikes)
        recent_split = get_most_recent_split(ticker, quote_date)
        if recent_split is None:
            return False  # No splits, no filtering needed
        
        # If we're more than 10 days after the most recent split,
        # still check for absurd strikes
        days_since = (quote_date - recent_split.split_date).days
        if days_since > 30:
            # Far enough from split, use standard filtering
            return False
    
    # Check if strike is absurdly far from underlying
    ratio = strike / underlying_price
    
    # Filter if strike is too far OTM (either direction)
    if ratio > max_otm_ratio:
        return True
    if ratio < 1 / max_otm_ratio:
        return True
    
    return False


def get_split_diagnostics(
    ticker: str,
    quote_date: date,
    strikes: list[float],
    underlying_price: float,
) -> dict:
    """
    Generate diagnostic information about split-related filtering.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    quote_date : date
        Quote date
    strikes : list[float]
        List of strikes in the options chain
    underlying_price : float
        Underlying price
        
    Returns
    -------
    dict
        Diagnostic information including:
        - near_split: bool
        - days_to_split: int or None
        - split_ratio: float or None
        - strikes_filtered: int
        - max_strike_ratio: float
    """
    near, split = is_near_split(ticker, quote_date, window_days=10)
    
    # Count strikes that would be filtered
    filtered_count = sum(
        1 for k in strikes
        if should_filter_strike(k, underlying_price, ticker, quote_date)
    )
    
    # Calculate max strike ratio
    max_ratio = max(k / underlying_price for k in strikes) if strikes else 0
    min_ratio = min(k / underlying_price for k in strikes) if strikes else 0
    
    result = {
        "near_split": near,
        "days_to_split": None,
        "split_ratio": None,
        "strikes_filtered": filtered_count,
        "max_strike_ratio": max_ratio,
        "min_strike_ratio": min_ratio,
    }
    
    if split:
        result["days_to_split"] = (quote_date - split.split_date).days
        result["split_ratio"] = split.split_ratio
    
    return result


if __name__ == "__main__":
    # Quick test
    for ticker in ["AAPL", "TSLA", "NVDA"]:
        splits = load_splits(ticker)
        print(f"\n{ticker} splits:")
        for s in splits:
            print(f"  {s.split_date}: {s.split_ratio}:1")

