#!/usr/bin/env python3
"""Convert SPY options CSV to partitioned parquet files.

Partitions by quote_date for efficient daily access:
  data/processed/spy_options_by_date/quote_date=YYYY-MM-DD/*.parquet

Usage:
    uv run python scripts/01_convert_csv.py
    uv run python scripts/01_convert_csv.py --csv data/raw/spy_2020_2022.csv --out data/processed/spy_options_by_date
"""

import argparse
import sys
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from tqdm import tqdm

from vix_challenger.config import SPY_RAW_CSV, SPY_OPTIONS_BY_DATE, ensure_directories
from vix_challenger.io.spy_csv import scan_spy_csv, Cols


def convert_csv_to_partitioned_parquet(
    csv_path: Path,
    output_dir: Path,
    overwrite: bool = True,
) -> dict:
    """Convert CSV to partitioned parquet files.
    
    Args:
        csv_path: Path to input CSV
        output_dir: Directory for partitioned output
        overwrite: If True, remove existing output directory first
        
    Returns:
        Dictionary with conversion statistics
    """
    ensure_directories()
    
    print(f"Input CSV: {csv_path}")
    print(f"Output dir: {output_dir}")
    
    # Remove existing output if overwriting
    if overwrite and output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process the data
    print("\nScanning CSV...")
    lf = scan_spy_csv(csv_path)
    
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
        "total_dates": len(unique_dates),
        "total_rows": len(df),
        "rows_per_date": {},
        "errors": [],
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
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total partitions: {len(unique_dates)}")
    print(f"Total rows: {len(df):,}")
    
    # Verify by reading one partition
    print("\nVerifying by reading a sample partition...")
    sample_date = unique_dates[len(unique_dates) // 2]  # Pick a middle date
    sample_dir = output_dir / f"quote_date={sample_date.strftime('%Y-%m-%d')}"
    sample_df = pl.read_parquet(sample_dir / "data.parquet")
    print(f"Sample date: {sample_date}")
    print(f"Rows: {len(sample_df)}")
    print(f"Columns: {sample_df.columns}")
    
    # Show a few rows
    print("\nSample data:")
    print(sample_df.head(3))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert SPY options CSV to partitioned parquet"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SPY_RAW_CSV,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SPY_OPTIONS_BY_DATE,
        help="Output directory for partitioned parquet"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite existing output"
    )
    args = parser.parse_args()
    
    if not args.csv.exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        print("Run scripts/download_data.py first.")
        sys.exit(1)
    
    stats = convert_csv_to_partitioned_parquet(
        csv_path=args.csv,
        output_dir=args.out,
        overwrite=not args.no_overwrite,
    )
    
    # Check acceptance criteria
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA CHECK")
    print("=" * 60)
    
    # 1. Partitions exist
    partitions = list(args.out.glob("quote_date=*"))
    print(f"✓ Partitions created: {len(partitions)}")
    
    # 2. Reading one day is fast
    import time
    sample_partition = partitions[len(partitions) // 2]
    start = time.time()
    _ = pl.read_parquet(sample_partition / "data.parquet")
    elapsed = time.time() - start
    if elapsed < 1.0:
        print(f"✓ Read time: {elapsed*1000:.1f}ms (< 1s)")
    else:
        print(f"✗ Read time: {elapsed:.2f}s (should be < 1s)")


if __name__ == "__main__":
    main()

