"""IO modules for data loading.

- options_loader: Generic options data loading (multi-ticker)
- spy_csv: Backwards-compatible SPY loader
- fred: FRED API client
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
]
