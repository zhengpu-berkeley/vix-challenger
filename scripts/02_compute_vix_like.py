#!/usr/bin/env python3
"""Compute VIX-like index for all trading days 2020-2022.

This script:
1. Loads each day's partitioned parquet data
2. Computes the VIX-like index using 30-day interpolation
3. Collects diagnostics and QC metrics
4. Saves results to parquet files

Usage:
    uv run python scripts/02_compute_vix_like.py
    uv run python scripts/02_compute_vix_like.py --limit 50  # First 50 days only
    uv run python scripts/02_compute_vix_like.py --start 2021-01-01 --end 2021-01-31
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from tqdm import tqdm

from vix_challenger.config import (
    SPY_OPTIONS_BY_DATE,
    SPY_VIX_LIKE_PARQUET,
    DIAGNOSTICS_PARQUET,
    ensure_directories,
)
from vix_challenger.vix import (
    compute_daily_vix,
    result_to_dict,
    compute_day_qc_metrics,
    qc_metrics_to_dict,
    summarize_skip_reasons,
)


def get_available_dates(data_dir: Path) -> list[date]:
    """Get sorted list of available trading dates from partitioned data."""
    dates = []
    for partition in data_dir.glob("quote_date=*"):
        date_str = partition.name.split("=")[1]
        dates.append(date.fromisoformat(date_str))
    return sorted(dates)


def load_day_data(data_dir: Path, quote_date: date) -> pl.DataFrame:
    """Load options data for a specific trading day."""
    date_str = quote_date.strftime("%Y-%m-%d")
    partition_path = data_dir / f"quote_date={date_str}" / "data.parquet"
    return pl.read_parquet(partition_path)


def run_pipeline(
    data_dir: Path,
    output_path: Path,
    diagnostics_path: Path,
    start_date: date = None,
    end_date: date = None,
    limit: int = None,
    verbose: bool = False,
) -> dict:
    """Run VIX computation pipeline on all trading days.
    
    Args:
        data_dir: Directory with partitioned parquet data
        output_path: Path to save spy_vix_like.parquet
        diagnostics_path: Path to save diagnostics.parquet
        start_date: Start date filter (inclusive)
        end_date: End date filter (inclusive)
        limit: Maximum number of days to process
        verbose: Print verbose output
        
    Returns:
        Dictionary with pipeline statistics
    """
    ensure_directories()
    
    # Get available dates
    all_dates = get_available_dates(data_dir)
    print(f"Found {len(all_dates)} trading days")
    print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
    
    # Apply filters
    dates = all_dates
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    if limit:
        dates = dates[:limit]
    
    print(f"Processing {len(dates)} days")
    if start_date or end_date:
        print(f"Date filter: {start_date or 'start'} to {end_date or 'end'}")
    
    # Process each day
    results = []
    qc_metrics = []
    
    for quote_date in tqdm(dates, desc="Computing VIX"):
        try:
            # Load data
            day_df = load_day_data(data_dir, quote_date)
            
            # Compute VIX
            result = compute_daily_vix(day_df, quote_date)
            results.append(result)
            
            # Compute QC metrics
            qc = compute_day_qc_metrics(day_df, quote_date)
            qc_metrics.append(qc)
            
            if verbose and not result.success:
                print(f"  {quote_date}: SKIP - {result.skip_reason}")
            
        except Exception as e:
            print(f"ERROR on {quote_date}: {e}")
            # Create a failed result
            from vix_challenger.vix import DailyVIXResult
            result = DailyVIXResult(
                quote_date=quote_date,
                skip_reason="LOAD_ERROR",
                error_detail=str(e),
            )
            results.append(result)
    
    # Convert to DataFrames
    results_df = pl.DataFrame([result_to_dict(r) for r in results])
    qc_df = pl.DataFrame([qc_metrics_to_dict(q) for q in qc_metrics])
    
    # Save results
    print(f"\nSaving results to {output_path}")
    results_df.write_parquet(output_path)
    
    print(f"Saving diagnostics to {diagnostics_path}")
    qc_df.write_parquet(diagnostics_path)
    
    # Compute statistics
    n_success = sum(1 for r in results if r.success)
    n_failed = len(results) - n_success
    success_rate = 100.0 * n_success / len(results) if results else 0
    
    skip_reasons = summarize_skip_reasons(results)
    
    # Get successful results for stats
    successful_results = [r for r in results if r.success]
    
    stats = {
        "total_days": len(dates),
        "successful": n_success,
        "failed": n_failed,
        "success_rate": success_rate,
        "skip_reasons": skip_reasons,
    }
    
    if successful_results:
        indices = [r.index for r in successful_results]
        stats["index_min"] = min(indices)
        stats["index_max"] = max(indices)
        stats["index_mean"] = sum(indices) / len(indices)
    
    return stats


def print_summary(stats: dict):
    """Print pipeline summary statistics."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    print(f"\n[Processing Stats]")
    print(f"  Total days:     {stats['total_days']}")
    print(f"  Successful:     {stats['successful']}")
    print(f"  Failed/Skipped: {stats['failed']}")
    print(f"  Success rate:   {stats['success_rate']:.1f}%")
    
    if stats.get('skip_reasons'):
        print(f"\n[Skip Reasons]")
        for reason, count in sorted(stats['skip_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    
    if stats.get('index_mean'):
        print(f"\n[Index Statistics]")
        print(f"  Min:  {stats['index_min']:.2f}")
        print(f"  Max:  {stats['index_max']:.2f}")
        print(f"  Mean: {stats['index_mean']:.2f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compute VIX-like index for all trading days"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SPY_OPTIONS_BY_DATE,
        help="Directory with partitioned parquet data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SPY_VIX_LIKE_PARQUET,
        help="Output path for VIX results"
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=DIAGNOSTICS_PARQUET,
        help="Output path for diagnostics"
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of days to process"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()
    
    # Parse dates
    start_date = date.fromisoformat(args.start) if args.start else None
    end_date = date.fromisoformat(args.end) if args.end else None
    
    # Run pipeline
    stats = run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        diagnostics_path=args.diagnostics,
        start_date=start_date,
        end_date=end_date,
        limit=args.limit,
        verbose=args.verbose,
    )
    
    print_summary(stats)
    
    # Exit with error if too many failures
    if stats['success_rate'] < 50:
        print("\nWARNING: Success rate below 50%!")
        sys.exit(1)


if __name__ == "__main__":
    main()

