"""Generic options data loading for multiple tickers.

The Kaggle datasets have a wide format where each row contains both call and put
data for a given quote_date / expiration / strike combination.

Column names have leading spaces (except the first column).
This schema is consistent across SPY, AAPL, TSLA, NVDA datasets.
"""

from pathlib import Path
from typing import Optional

import polars as pl

from vix_challenger.config import TickerConfig, get_ticker_config


# =============================================================================
# Raw column names (as they appear in CSV, note leading spaces)
# =============================================================================

class RawCols:
    """Raw column names from the CSV files.
    
    These are consistent across Kyle Graupe's Kaggle datasets.
    """
    
    # Date columns
    QUOTE_UNIXTIME = "[QUOTE_UNIXTIME]"
    QUOTE_READTIME = " [QUOTE_READTIME]"
    QUOTE_DATE = " [QUOTE_DATE]"
    QUOTE_TIME_HOURS = " [QUOTE_TIME_HOURS]"
    
    # Expiration
    EXPIRE_DATE = " [EXPIRE_DATE]"
    EXPIRE_UNIX = " [EXPIRE_UNIX]"
    DTE = " [DTE]"
    
    # Underlying
    UNDERLYING_LAST = " [UNDERLYING_LAST]"
    
    # Strike
    STRIKE = " [STRIKE]"
    STRIKE_DISTANCE = " [STRIKE_DISTANCE]"
    STRIKE_DISTANCE_PCT = " [STRIKE_DISTANCE_PCT]"
    
    # Call data
    C_BID = " [C_BID]"
    C_ASK = " [C_ASK]"
    C_LAST = " [C_LAST]"
    C_SIZE = " [C_SIZE]"
    C_VOLUME = " [C_VOLUME]"
    C_DELTA = " [C_DELTA]"
    C_GAMMA = " [C_GAMMA]"
    C_VEGA = " [C_VEGA]"
    C_THETA = " [C_THETA]"
    C_RHO = " [C_RHO]"
    C_IV = " [C_IV]"
    
    # Put data
    P_BID = " [P_BID]"
    P_ASK = " [P_ASK]"
    P_LAST = " [P_LAST]"
    P_SIZE = " [P_SIZE]"
    P_VOLUME = " [P_VOLUME]"
    P_DELTA = " [P_DELTA]"
    P_GAMMA = " [P_GAMMA]"
    P_VEGA = " [P_VEGA]"
    P_THETA = " [P_THETA]"
    P_RHO = " [P_RHO]"
    P_IV = " [P_IV]"


# =============================================================================
# Logical column names (clean names for our processing)
# =============================================================================

class Cols:
    """Clean column names for processed data."""
    
    # Identifier
    TICKER = "ticker"
    
    # Dates
    QUOTE_DATE = "quote_date"
    EXPIRATION = "expiration"
    DTE = "dte"
    
    # Prices
    UNDERLYING_PRICE = "underlying_price"
    STRIKE = "strike"
    
    # Call data
    C_BID = "c_bid"
    C_ASK = "c_ask"
    C_MID = "c_mid"
    C_VOLUME = "c_volume"
    
    # Put data
    P_BID = "p_bid"
    P_ASK = "p_ask"
    P_MID = "p_mid"
    P_VOLUME = "p_volume"


# =============================================================================
# Loading functions
# =============================================================================

def scan_options_csv(
    csv_path: Path,
    ticker: Optional[str] = None,
) -> pl.LazyFrame:
    """Scan options CSV lazily with proper column selection and type casting.
    
    Args:
        csv_path: Path to the raw CSV file
        ticker: Optional ticker symbol to add as column
        
    Returns:
        LazyFrame with cleaned column names and proper types
    """
    lf = pl.scan_csv(csv_path)
    
    # Helper to safely cast string to float (handles empty strings)
    def safe_float(col_name: str, alias: str) -> pl.Expr:
        return (
            pl.col(col_name)
            .str.strip_chars()
            .str.replace("^$", "NaN")  # Replace empty string with NaN
            .cast(pl.Float64, strict=False)
            .alias(alias)
        )
    
    # Select and rename columns, cast types
    columns = [
        # Parse quote_date
        pl.col(RawCols.QUOTE_DATE).str.strip_chars().str.to_date("%Y-%m-%d").alias(Cols.QUOTE_DATE),
        
        # Parse expiration
        pl.col(RawCols.EXPIRE_DATE).str.strip_chars().str.to_date("%Y-%m-%d").alias(Cols.EXPIRATION),
        
        # DTE as integer
        pl.col(RawCols.DTE).str.strip_chars().cast(pl.Float64, strict=False).cast(pl.Int32).alias(Cols.DTE),
        
        # Underlying price
        safe_float(RawCols.UNDERLYING_LAST, Cols.UNDERLYING_PRICE),
        
        # Strike
        safe_float(RawCols.STRIKE, Cols.STRIKE),
        
        # Call bid/ask (may have empty strings)
        safe_float(RawCols.C_BID, Cols.C_BID),
        safe_float(RawCols.C_ASK, Cols.C_ASK),
        
        # Put bid/ask (may have empty strings)
        safe_float(RawCols.P_BID, Cols.P_BID),
        safe_float(RawCols.P_ASK, Cols.P_ASK),
        
        # Volumes (may have empty strings)
        safe_float(RawCols.C_VOLUME, Cols.C_VOLUME),
        safe_float(RawCols.P_VOLUME, Cols.P_VOLUME),
    ]
    
    lf = lf.select(columns)
    
    # Compute midquotes
    lf = lf.with_columns([
        ((pl.col(Cols.C_BID) + pl.col(Cols.C_ASK)) / 2).alias(Cols.C_MID),
        ((pl.col(Cols.P_BID) + pl.col(Cols.P_ASK)) / 2).alias(Cols.P_MID),
    ])
    
    # Add ticker column if provided
    if ticker:
        lf = lf.with_columns(pl.lit(ticker.upper()).alias(Cols.TICKER))
    
    return lf


def load_options_csv(
    csv_path: Path,
    ticker: Optional[str] = None,
) -> pl.DataFrame:
    """Load options CSV into memory.
    
    Args:
        csv_path: Path to the raw CSV file
        ticker: Optional ticker symbol to add as column
        
    Returns:
        DataFrame with cleaned columns and proper types
    """
    return scan_options_csv(csv_path, ticker=ticker).collect()


def scan_ticker_csv(ticker: str) -> pl.LazyFrame:
    """Scan options CSV for a registered ticker.
    
    Args:
        ticker: Ticker symbol (e.g., "SPY", "AAPL")
        
    Returns:
        LazyFrame with options data
    """
    config = get_ticker_config(ticker)
    return scan_options_csv(config.raw_csv_path, ticker=ticker)


def load_ticker_csv(ticker: str) -> pl.DataFrame:
    """Load options CSV for a registered ticker.
    
    Args:
        ticker: Ticker symbol (e.g., "SPY", "AAPL")
        
    Returns:
        DataFrame with options data
    """
    return scan_ticker_csv(ticker).collect()


def get_unique_quote_dates(csv_path: Path) -> list:
    """Get list of unique quote dates in the dataset.
    
    Args:
        csv_path: Path to the raw CSV file
        
    Returns:
        Sorted list of unique quote dates
    """
    dates = (
        pl.scan_csv(csv_path)
        .select(pl.col(RawCols.QUOTE_DATE).str.strip_chars().str.to_date("%Y-%m-%d"))
        .unique()
        .sort(RawCols.QUOTE_DATE)
        .collect()
        .to_series()
        .to_list()
    )
    return dates


def load_day_partition(
    ticker: str,
    quote_date,
) -> pl.DataFrame:
    """Load a single day's partitioned parquet data.
    
    Args:
        ticker: Ticker symbol
        quote_date: Date to load (date object or string)
        
    Returns:
        DataFrame with options data for that day
    """
    config = get_ticker_config(ticker)
    
    if hasattr(quote_date, 'strftime'):
        date_str = quote_date.strftime("%Y-%m-%d")
    else:
        date_str = str(quote_date)
    
    partition_path = config.options_by_date_dir / f"quote_date={date_str}" / "data.parquet"
    
    if not partition_path.exists():
        raise FileNotFoundError(f"No data for {ticker} on {date_str}: {partition_path}")
    
    return pl.read_parquet(partition_path)


# =============================================================================
# Backwards compatibility aliases
# =============================================================================

# These maintain compatibility with existing code that imports from spy_csv
scan_spy_csv = scan_options_csv
load_spy_csv = load_options_csv

