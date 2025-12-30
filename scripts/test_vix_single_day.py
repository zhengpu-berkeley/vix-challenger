#!/usr/bin/env python3
"""Test VIX computation on a single day with diagnostic output.

This script:
1. Loads a sample trading day from partitioned parquet
2. Lists available expirations and picks one with reasonable DTE
3. Computes variance using the VIX methodology
4. Prints detailed diagnostics
5. Optionally plots strike contribution curve

Usage:
    uv run python scripts/test_vix_single_day.py
    uv run python scripts/test_vix_single_day.py --date 2021-07-09
    uv run python scripts/test_vix_single_day.py --date 2020-03-16  # COVID volatility spike
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import numpy as np

from vix_challenger.config import SPY_OPTIONS_BY_DATE
from vix_challenger.io.spy_csv import Cols
from vix_challenger.vix import (
    list_expirations,
    compute_variance_for_day,
    print_variance_diagnostics,
)


def load_day_data(quote_date: date) -> pl.DataFrame:
    """Load options data for a specific trading day."""
    date_str = quote_date.strftime("%Y-%m-%d")
    partition_path = SPY_OPTIONS_BY_DATE / f"quote_date={date_str}" / "data.parquet"
    
    if not partition_path.exists():
        raise FileNotFoundError(f"No data for {date_str}. Path: {partition_path}")
    
    return pl.read_parquet(partition_path)


def pick_expiry_for_vix(day_df: pl.DataFrame, target_dte: int = 30) -> tuple:
    """Pick an expiration with DTE close to target (default 30 days).
    
    Returns:
        (expiration_date, dte)
    """
    expirations = list_expirations(day_df)
    
    # Filter for reasonable DTEs (> 0 to exclude same-day expiry)
    valid = expirations.filter(pl.col(Cols.DTE) > 0)
    
    if len(valid) == 0:
        raise ValueError("No valid expirations found (all DTE <= 0)")
    
    # Find closest to target DTE
    valid = valid.with_columns(
        (pl.col(Cols.DTE) - target_dte).abs().alias("dist_to_target")
    )
    best = valid.sort("dist_to_target").head(1)
    
    expiration = best[Cols.EXPIRATION].item()
    dte = best[Cols.DTE].item()
    
    return expiration, dte


def plot_contribution_curve(result, save_path: Path = None):
    """Plot strike contribution to variance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Bar chart of contributions
    ax1 = axes[0]
    colors = ['tab:red' if k < result.k0 else 'tab:blue' if k > result.k0 else 'tab:green' 
              for k in result.strikes]
    ax1.bar(result.strikes, result.strike_contributions, color=colors, alpha=0.7, width=2)
    ax1.axvline(result.k0, color='black', linestyle='--', label=f'K0={result.k0:.0f}')
    ax1.axvline(result.forward, color='green', linestyle=':', label=f'F={result.forward:.2f}')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Contribution to Variance')
    ax1.set_title(f'Strike Contributions (DTE={result.dte})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Cumulative contribution
    ax2 = axes[1]
    cumsum = np.cumsum(result.strike_contributions)
    ax2.plot(result.strikes, cumsum, 'b-', linewidth=2)
    ax2.axvline(result.k0, color='black', linestyle='--', label=f'K0={result.k0:.0f}')
    ax2.axhline(cumsum[-1], color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Strike')
    ax2.set_ylabel('Cumulative Contribution')
    ax2.set_title(f'Cumulative Variance Contribution\n(Total Sum Term = {result.sum_term:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test VIX computation on a single day")
    parser.add_argument(
        "--date",
        type=str,
        default="2021-07-09",
        help="Quote date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--target-dte",
        type=int,
        default=30,
        help="Target DTE for expiry selection"
    )
    parser.add_argument(
        "--no-cutoff",
        action="store_true",
        help="Disable zero-bid cutoff"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show contribution curve plot"
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Save plot to file"
    )
    args = parser.parse_args()
    
    # Parse date
    quote_date = date.fromisoformat(args.date)
    
    print("=" * 60)
    print(f"VIX SINGLE-DAY TEST - {quote_date}")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data for {quote_date}...")
    try:
        day_df = load_day_data(quote_date)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nAvailable dates:")
        dates = sorted([d.name.split("=")[1] for d in SPY_OPTIONS_BY_DATE.glob("quote_date=*")])
        print(f"  First: {dates[0]}")
        print(f"  Last:  {dates[-1]}")
        print(f"  Total: {len(dates)} days")
        sys.exit(1)
    
    print(f"Loaded {len(day_df):,} option records")
    
    # Show underlying price
    underlying = day_df[Cols.UNDERLYING_PRICE].head(1).item()
    print(f"Underlying (SPY): ${underlying:.2f}")
    
    # List expirations
    print("\n[Available Expirations]")
    expirations = list_expirations(day_df)
    print(expirations)
    
    # Pick expiration
    expiration, dte = pick_expiry_for_vix(day_df, target_dte=args.target_dte)
    print(f"\nSelected expiry: {expiration} (DTE={dte})")
    
    # Compute variance
    print("\n" + "-" * 60)
    print("COMPUTING VARIANCE...")
    print("-" * 60)
    
    try:
        result = compute_variance_for_day(
            day_df,
            expiration,
            apply_cutoff=not args.no_cutoff,
        )
        
        # Print diagnostics
        print_variance_diagnostics(result, expiration)
        
        # Sanity checks
        print("\n[SANITY CHECKS]")
        
        # Check 1: Variance > 0
        if result.variance > 0:
            print(f"  ✓ Variance > 0: {result.variance:.6f}")
        else:
            print(f"  ✗ Variance <= 0: {result.variance:.6f} (PROBLEM!)")
        
        # Check 2: Implied vol in reasonable range (5-150%)
        if 5 <= result.implied_vol <= 150:
            print(f"  ✓ Implied Vol in range [5%, 150%]: {result.implied_vol:.2f}%")
        else:
            print(f"  ⚠ Implied Vol outside typical range: {result.implied_vol:.2f}%")
        
        # Check 3: K0 close to forward
        k0_vs_f_pct = abs(result.k0 - result.forward) / result.forward * 100
        if k0_vs_f_pct < 2:
            print(f"  ✓ K0 close to F: {k0_vs_f_pct:.2f}% difference")
        else:
            print(f"  ⚠ K0 far from F: {k0_vs_f_pct:.2f}% difference")
        
        # Check 4: Strikes on both sides
        if result.n_puts > 0 and result.n_calls > 0:
            print(f"  ✓ Strikes on both wings: {result.n_puts} puts, {result.n_calls} calls")
        else:
            print(f"  ⚠ Missing wing: {result.n_puts} puts, {result.n_calls} calls")
        
        # Plot if requested
        if args.plot or args.save_plot:
            plot_contribution_curve(result, save_path=args.save_plot)
        
    except Exception as e:
        print(f"ERROR computing variance: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()

