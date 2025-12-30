#!/usr/bin/env python3
"""Validate VIX-like series after computation.

Performs post-computation validation:
- Check for gaps in date series
- Identify outliers
- Spot-check specific dates (COVID spike, calm periods)
- Summary statistics

Usage:
    uv run python scripts/validate_vix_series.py
    uv run python scripts/validate_vix_series.py --input data/processed/spy_vix_like.parquet
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import numpy as np

from vix_challenger.config import SPY_VIX_LIKE_PARQUET, DIAGNOSTICS_PARQUET


# Key dates to spot-check
SPOT_CHECK_DATES = {
    "COVID Crash Peak": date(2020, 3, 16),
    "COVID Recovery": date(2020, 6, 1),
    "Calm Q4 2021": date(2021, 11, 15),
    "2022 Volatility": date(2022, 1, 24),
    "Mid 2022": date(2022, 6, 15),
}

# Expected VIX ranges for validation
EXPECTED_RANGES = {
    "COVID Crash Peak": (50, 100),  # VIX was ~80+
    "COVID Recovery": (20, 40),
    "Calm Q4 2021": (10, 25),
    "2022 Volatility": (20, 50),
    "Mid 2022": (20, 40),
}


def load_vix_series(path: Path) -> pl.DataFrame:
    """Load VIX-like series from parquet."""
    df = pl.read_parquet(path)
    return df.filter(pl.col("success") == True).sort("quote_date")


def check_date_gaps(df: pl.DataFrame) -> list[tuple[date, date, int]]:
    """Find gaps in the date series (excluding weekends).
    
    Returns:
        List of (gap_start, gap_end, gap_days) tuples
    """
    dates = df["quote_date"].to_list()
    gaps = []
    
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        
        # Calculate expected next trading day (skip weekends)
        expected = prev_date + timedelta(days=1)
        while expected.weekday() >= 5:  # Skip Saturday/Sunday
            expected += timedelta(days=1)
        
        # If current date is more than 1 trading day after previous
        if curr_date > expected:
            gap_days = (curr_date - prev_date).days
            # Only report if gap is > 3 days (to account for holidays)
            if gap_days > 3:
                gaps.append((prev_date, curr_date, gap_days))
    
    return gaps


def identify_outliers(df: pl.DataFrame, window: int = 20, threshold: float = 3.0) -> pl.DataFrame:
    """Identify outliers using rolling z-score.
    
    Args:
        df: DataFrame with VIX series
        window: Rolling window size
        threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame with outlier rows
    """
    # Compute rolling mean and std
    df_with_stats = df.with_columns([
        pl.col("index").rolling_mean(window).alias("rolling_mean"),
        pl.col("index").rolling_std(window).alias("rolling_std"),
    ])
    
    # Compute z-score
    df_with_stats = df_with_stats.with_columns([
        ((pl.col("index") - pl.col("rolling_mean")) / pl.col("rolling_std")).alias("z_score")
    ])
    
    # Filter outliers
    outliers = df_with_stats.filter(
        pl.col("z_score").abs() > threshold
    )
    
    return outliers


def spot_check_dates(df: pl.DataFrame, dates_to_check: dict) -> dict:
    """Spot-check VIX values on specific dates.
    
    Args:
        df: DataFrame with VIX series
        dates_to_check: Dict mapping label to date
        
    Returns:
        Dict with results for each date
    """
    results = {}
    
    for label, check_date in dates_to_check.items():
        # Find closest date in series
        row = df.filter(pl.col("quote_date") == check_date)
        
        if len(row) == 0:
            # Try to find closest date within 5 days
            for delta in range(-5, 6):
                nearby_date = check_date + timedelta(days=delta)
                row = df.filter(pl.col("quote_date") == nearby_date)
                if len(row) > 0:
                    break
        
        if len(row) > 0:
            index_val = row["index"].item()
            actual_date = row["quote_date"].item()
            
            # Check if in expected range
            expected = EXPECTED_RANGES.get(label, (0, 200))
            in_range = expected[0] <= index_val <= expected[1]
            
            results[label] = {
                "date": actual_date,
                "index": index_val,
                "expected_range": expected,
                "in_range": in_range,
            }
        else:
            results[label] = {
                "date": check_date,
                "index": None,
                "expected_range": EXPECTED_RANGES.get(label),
                "in_range": False,
                "note": "Date not found",
            }
    
    return results


def compute_summary_stats(df: pl.DataFrame) -> dict:
    """Compute summary statistics for VIX series."""
    index_col = df["index"]
    
    return {
        "count": len(df),
        "date_range": (df["quote_date"].min(), df["quote_date"].max()),
        "index_min": index_col.min(),
        "index_max": index_col.max(),
        "index_mean": index_col.mean(),
        "index_median": index_col.median(),
        "index_std": index_col.std(),
        "index_p10": index_col.quantile(0.10),
        "index_p90": index_col.quantile(0.90),
    }


def validate_series(
    vix_path: Path,
    diagnostics_path: Path = None,
    verbose: bool = True,
) -> dict:
    """Run full validation on VIX series.
    
    Args:
        vix_path: Path to spy_vix_like.parquet
        diagnostics_path: Path to diagnostics.parquet (optional)
        verbose: Print detailed output
        
    Returns:
        Validation results dictionary
    """
    print("=" * 60)
    print("VIX SERIES VALIDATION")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading {vix_path}...")
    df = load_vix_series(vix_path)
    print(f"Loaded {len(df)} successful days")
    
    results = {"valid": True, "issues": []}
    
    # Summary statistics
    print("\n[Summary Statistics]")
    stats = compute_summary_stats(df)
    results["stats"] = stats
    
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"  Count:      {stats['count']} days")
    print(f"  Index min:  {stats['index_min']:.2f}")
    print(f"  Index max:  {stats['index_max']:.2f}")
    print(f"  Index mean: {stats['index_mean']:.2f}")
    print(f"  Index std:  {stats['index_std']:.2f}")
    print(f"  P10-P90:    [{stats['index_p10']:.2f}, {stats['index_p90']:.2f}]")
    
    # Check for negative or NaN values
    print("\n[Data Quality Checks]")
    n_negative = (df["index"] < 0).sum()
    n_null = df["index"].is_null().sum()
    n_nan = df["index"].is_nan().sum()
    
    if n_negative > 0:
        print(f"  ✗ Found {n_negative} negative values")
        results["issues"].append(f"{n_negative} negative values")
        results["valid"] = False
    else:
        print(f"  ✓ No negative values")
    
    if n_null > 0 or n_nan > 0:
        print(f"  ✗ Found {n_null + n_nan} null/NaN values")
        results["issues"].append(f"{n_null + n_nan} null/NaN values")
        results["valid"] = False
    else:
        print(f"  ✓ No null/NaN values")
    
    # Date gaps
    print("\n[Date Gaps]")
    gaps = check_date_gaps(df)
    results["gaps"] = gaps
    
    if gaps:
        print(f"  Found {len(gaps)} significant gaps (> 3 days):")
        for start, end, days in gaps[:5]:  # Show first 5
            print(f"    {start} to {end} ({days} days)")
        if len(gaps) > 5:
            print(f"    ... and {len(gaps) - 5} more")
    else:
        print(f"  ✓ No significant gaps")
    
    # Outliers
    print("\n[Outlier Detection]")
    outliers = identify_outliers(df)
    results["n_outliers"] = len(outliers)
    
    if len(outliers) > 0:
        print(f"  Found {len(outliers)} outliers (z-score > 3):")
        for row in outliers.head(5).iter_rows(named=True):
            print(f"    {row['quote_date']}: {row['index']:.2f} (z={row['z_score']:.2f})")
        if len(outliers) > 5:
            print(f"    ... and {len(outliers) - 5} more")
    else:
        print(f"  ✓ No extreme outliers detected")
    
    # Spot checks
    print("\n[Spot Checks]")
    spot_results = spot_check_dates(df, SPOT_CHECK_DATES)
    results["spot_checks"] = spot_results
    
    all_pass = True
    for label, result in spot_results.items():
        if result.get("index") is not None:
            status = "✓" if result["in_range"] else "⚠"
            if not result["in_range"]:
                all_pass = False
            print(f"  {status} {label} ({result['date']}): {result['index']:.2f} "
                  f"(expected: {result['expected_range']})")
        else:
            print(f"  ? {label}: {result.get('note', 'Not found')}")
    
    if not all_pass:
        results["issues"].append("Some spot checks outside expected range")
    
    # Load diagnostics if available
    if diagnostics_path and diagnostics_path.exists():
        print("\n[Skip Reason Summary]")
        full_df = pl.read_parquet(vix_path)
        skipped = full_df.filter(pl.col("success") == False)
        
        if len(skipped) > 0:
            skip_counts = (
                skipped
                .group_by("skip_reason")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )
            for row in skip_counts.iter_rows(named=True):
                print(f"  {row['skip_reason']}: {row['count']}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if results["valid"] and not results["issues"]:
        print("VALIDATION PASSED ✓")
    else:
        print("VALIDATION COMPLETED WITH ISSUES:")
        for issue in results["issues"]:
            print(f"  - {issue}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate VIX-like series")
    parser.add_argument(
        "--input",
        type=Path,
        default=SPY_VIX_LIKE_PARQUET,
        help="Path to spy_vix_like.parquet"
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=DIAGNOSTICS_PARQUET,
        help="Path to diagnostics.parquet"
    )
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run scripts/02_compute_vix_like.py first.")
        sys.exit(1)
    
    results = validate_series(
        vix_path=args.input,
        diagnostics_path=args.diagnostics,
    )
    
    # Exit with error if validation failed
    if not results["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

