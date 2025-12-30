"""FRED API client for downloading economic data.

Downloads VIXCLS (VIX close) and other series from the Federal Reserve
Economic Data (FRED) API.
"""

import os
from datetime import date
from typing import Optional

import polars as pl
import requests

from vix_challenger.config import FRED_VIXCLS_PARQUET


# FRED API endpoint
FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"


def get_fred_api_key() -> Optional[str]:
    """Get FRED API key from environment."""
    return os.environ.get("FRED_API_KEY")


def download_fred_series(
    series_id: str,
    start_date: date,
    end_date: date,
    api_key: Optional[str] = None,
) -> pl.DataFrame:
    """Download a data series from FRED.
    
    Args:
        series_id: FRED series ID (e.g., "VIXCLS")
        start_date: Start date for data
        end_date: End date for data
        api_key: FRED API key (if None, tries environment)
        
    Returns:
        DataFrame with columns: date, value
        
    Raises:
        ValueError: If API key not found or request fails
    """
    if api_key is None:
        api_key = get_fred_api_key()
    
    if not api_key:
        raise ValueError(
            "FRED API key not found. Set FRED_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date.isoformat(),
        "observation_end": end_date.isoformat(),
    }
    
    print(f"Downloading {series_id} from FRED...")
    print(f"Date range: {start_date} to {end_date}")
    
    response = requests.get(FRED_API_BASE, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    if "observations" not in data:
        raise ValueError(f"Unexpected FRED response: {data}")
    
    observations = data["observations"]
    print(f"Received {len(observations)} observations")
    
    # Parse to DataFrame
    records = []
    for obs in observations:
        date_str = obs["date"]
        value_str = obs["value"]
        
        # FRED uses "." for missing values
        if value_str == ".":
            continue
        
        try:
            value = float(value_str)
            records.append({
                "date": date.fromisoformat(date_str),
                "value": value,
            })
        except ValueError:
            print(f"  Skipping invalid value: {date_str} = {value_str}")
    
    df = pl.DataFrame(records)
    
    # Rename value column to series_id
    df = df.rename({"value": series_id.lower()})
    
    print(f"Parsed {len(df)} valid observations")
    
    return df


def download_vixcls(
    start_date: date = date(2020, 1, 1),
    end_date: date = date(2022, 12, 31),
    api_key: Optional[str] = None,
    save_path: Optional[str] = None,
) -> pl.DataFrame:
    """Download VIXCLS (VIX Close) from FRED.
    
    Args:
        start_date: Start date (default 2020-01-01)
        end_date: End date (default 2022-12-31)
        api_key: FRED API key
        save_path: If provided, save to parquet
        
    Returns:
        DataFrame with columns: date, vixcls
    """
    df = download_fred_series(
        series_id="VIXCLS",
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
    )
    
    if save_path:
        df.write_parquet(save_path)
        print(f"Saved to: {save_path}")
    
    return df


def load_vixcls(path: str = None) -> pl.DataFrame:
    """Load VIXCLS from parquet file.
    
    Args:
        path: Path to parquet file (default: config path)
        
    Returns:
        DataFrame with VIXCLS data
    """
    if path is None:
        path = FRED_VIXCLS_PARQUET
    
    return pl.read_parquet(path)

