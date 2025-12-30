#!/usr/bin/env python3
"""Convert options CSV to partitioned parquet files for a ticker.

Partitions by quote_date for efficient daily access:
  data/processed/{ticker}_options_by_date/quote_date=YYYY-MM-DD/*.parquet

Usage:
    uv run python scripts/01_convert_csv.py                    # SPY (default)
    uv run python scripts/01_convert_csv.py --ticker SPY
    uv run python scripts/01_convert_csv.py --ticker AAPL
"""

import argparse
import sys
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from tqdm import tqdm

from vix_challenger.config import (
    get_ticker_config,
    list_tickers,
    ensure_ticker_directories,
)
from vix_challenger.io.options_loader import scan_options_csv, Cols


def convert_csv_to_partitioned_parquet(
    ticker: str,
    overwrite: bool = True,
) -> dict:
    """Convert CSV to partitioned parquet files for a ticker.
    
    Args:
        ticker: Ticker symbol
        overwrite: If True, remove existing output directory first
        
    Returns:
        Dictionary with conversion statistics
    """
    config = get_ticker_config(ticker)
    ensure_ticker_directories(ticker)
    
    csv_path = config.raw_csv_path
    output_dir = config.options_by_date_dir
    
    print(f"Ticker: {ticker}")
    print(f"Input CSV: {csv_path}")
    print(f"Output dir: {output_dir}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}. Run download_data.py --ticker {ticker} first.")
    
    # Remove existing output if overwriting
    if overwrite and output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process the data
    print("\nScanning CSV...")
    lf = scan_options_csv(csv_path, ticker=ticker)
    
    # Get unique dates for progress bar
    print("Getting unique dates...")
    unique_dates = (
        lf.select(pl.col(Cols.QUOTE_DATE))
        .unique()
        .sort(Cols.QUOTE_DATE)
        .collect()
        .to_series()
        .to_list()
    )
    
    print(f"Found {len(unique_dates)} unique trading days")
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    
    # Collect full dataframe
    print("\nLoading full dataset into memory...")
    df = lf.collect()
    print(f"Total rows: {len(df):,}")
    
    # Write partitioned parquet files
    print("\nWriting partitioned parquet files...")
    
    stats = {
        "ticker": ticker,
        "total_dates": len(unique_dates),
        "total_rows": len(df),
        "rows_per_date": {},
    }
    
    for date in tqdm(unique_dates, desc="Partitioning"):
        date_str = date.strftime("%Y-%m-%d")
        partition_dir = output_dir / f"quote_date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter for this date
        date_df = df.filter(pl.col(Cols.QUOTE_DATE) == date)
        
        # Write parquet
        output_file = partition_dir / "data.parquet"
        date_df.write_parquet(output_file)
        
        stats["rows_per_date"][date_str] = len(date_df)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"CONVERSION COMPLETE - {ticker}")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total partitions: {len(unique_dates)}")
    print(f"Total rows: {len(df):,}")
    
    # Verify by reading one partition
    print("\nVerifying by reading a sample partition...")
    sample_date = unique_dates[len(unique_dates) // 2]
    sample_dir = output_dir / f"quote_date={sample_date.strftime('%Y-%m-%d')}"
    sample_df = pl.read_parquet(sample_dir / "data.parquet")
    print(f"Sample date: {sample_date}")
    print(f"Rows: {len(sample_df)}")
    
    # Check read time
    import time
    start = time.time()
    _ = pl.read_parquet(sample_dir / "data.parquet")
    elapsed = time.time() - start
    print(f"Read time: {elapsed*1000:.1f}ms")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert options CSV to partitioned parquet"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help=f"Ticker symbol. Available: {', '.join(list_tickers())}"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite existing output"
    )
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    try:
        stats = convert_csv_to_partitioned_parquet(
            ticker=ticker,
            overwrite=not args.no_overwrite,
        )
        print(f"\n✓ {ticker} conversion successful!")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
