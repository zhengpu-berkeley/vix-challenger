"""SPY options CSV loading - backwards compatibility module.

This module maintains backwards compatibility with existing code.
All functionality has been moved to options_loader.py.
"""

# Re-export everything from options_loader for backwards compatibility
from vix_challenger.io.options_loader import (
    RawCols,
    Cols,
    scan_options_csv as scan_spy_csv,
    load_options_csv as load_spy_csv,
    get_unique_quote_dates,
)

# Also export these for direct imports
__all__ = [
    "RawCols",
    "Cols", 
    "scan_spy_csv",
    "load_spy_csv",
    "get_unique_quote_dates",
]
