"""IO modules for data loading.

- options_loader: Generic options data loading (multi-ticker)
- spy_csv: Backwards-compatible SPY loader
- fred: FRED API client
- splits: Stock split metadata loading
"""

from vix_challenger.io.options_loader import (
    RawCols,
    Cols,
    scan_options_csv,
    load_options_csv,
    scan_ticker_csv,
    load_ticker_csv,
    get_unique_quote_dates,
    load_day_partition,
)

from vix_challenger.io.fred import (
    download_fred_series,
    download_vixcls,
    load_vixcls,
)

from vix_challenger.io.splits import (
    SplitEvent,
    load_splits,
    get_splits_in_range,
    get_most_recent_split,
    is_near_split,
    get_cumulative_split_factor,
    should_filter_strike,
    get_split_diagnostics,
)

__all__ = [
    # Options loader
    "RawCols",
    "Cols",
    "scan_options_csv",
    "load_options_csv",
    "scan_ticker_csv",
    "load_ticker_csv",
    "get_unique_quote_dates",
    "load_day_partition",
    # FRED
    "download_fred_series",
    "download_vixcls",
    "load_vixcls",
    # Splits
    "SplitEvent",
    "load_splits",
    "get_splits_in_range",
    "get_most_recent_split",
    "is_near_split",
    "get_cumulative_split_factor",
    "should_filter_strike",
    "get_split_diagnostics",
]
