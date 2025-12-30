#!/usr/bin/env python3
"""Download VIXCLS data from FRED.

Downloads the VIX closing values from the Federal Reserve Economic Data (FRED)
API and saves to parquet.

Usage:
    uv run python scripts/03_download_fred.py
    uv run python scripts/03_download_fred.py --start 2020-01-01 --end 2022-12-31
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from vix_challenger.config import FRED_VIXCLS_PARQUET, ensure_directories
from vix_challenger.io.fred import download_vixcls, get_fred_api_key


def main():
    parser = argparse.ArgumentParser(description="Download VIXCLS from FRED")
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2022-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FRED_VIXCLS_PARQUET,
        help="Output parquet path"
    )
    args = parser.parse_args()
    
    ensure_directories()
    
    # Check for API key
    api_key = get_fred_api_key()
    if not api_key:
        print("ERROR: FRED_API_KEY not found in environment")
        print("Please set FRED_API_KEY in your .env file")
        print("\nYou can get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)
    
    print("=" * 60)
    print("FRED VIXCLS DOWNLOAD")
    print("=" * 60)
    
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    
    # Download data
    df = download_vixcls(
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
        save_path=args.output,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Observations: {len(df)}")
    print(f"Date range:   {df['date'].min()} to {df['date'].max()}")
    print(f"VIXCLS range: {df['vixcls'].min():.2f} to {df['vixcls'].max():.2f}")
    print(f"VIXCLS mean:  {df['vixcls'].mean():.2f}")
    print(f"Output:       {args.output}")


if __name__ == "__main__":
    main()

