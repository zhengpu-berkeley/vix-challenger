"""SPY options CSV loading and column mappings.

The Kaggle dataset has a wide format where each row contains both call and put
data for a given quote_date / expiration / strike combination.

Column names have leading spaces (except the first column).
"""

import polars as pl
from pathlib import Path


# =============================================================================
# Raw column names (as they appear in CSV, note leading spaces)
# =============================================================================

class RawCols:
    """Raw column names from the CSV file."""
    
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
    
    QUOTE_DATE = "quote_date"
    EXPIRATION = "expiration"
    DTE = "dte"
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
# Columns to keep for VIX computation
# =============================================================================

COLUMNS_TO_LOAD = [
    RawCols.QUOTE_DATE,
    RawCols.EXPIRE_DATE,
    RawCols.DTE,
    RawCols.UNDERLYING_LAST,
    RawCols.STRIKE,
    RawCols.C_BID,
    RawCols.C_ASK,
    RawCols.P_BID,
    RawCols.P_ASK,
    RawCols.C_VOLUME,
    RawCols.P_VOLUME,
]


# =============================================================================
# Loading functions
# =============================================================================

def scan_spy_csv(csv_path: Path) -> pl.LazyFrame:
    """Scan SPY options CSV lazily with proper column selection and type casting.
    
    Args:
        csv_path: Path to the raw CSV file
        
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
    lf = lf.select([
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
    ])
    
    # Compute midquotes
    lf = lf.with_columns([
        ((pl.col(Cols.C_BID) + pl.col(Cols.C_ASK)) / 2).alias(Cols.C_MID),
        ((pl.col(Cols.P_BID) + pl.col(Cols.P_ASK)) / 2).alias(Cols.P_MID),
    ])
    
    return lf


def load_spy_csv(csv_path: Path) -> pl.DataFrame:
    """Load SPY options CSV into memory.
    
    Args:
        csv_path: Path to the raw CSV file
        
    Returns:
        DataFrame with cleaned columns and proper types
    """
    return scan_spy_csv(csv_path).collect()


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

