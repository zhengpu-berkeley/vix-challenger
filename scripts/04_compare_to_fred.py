#!/usr/bin/env python3
"""Compare SPY VIX-like index to FRED VIXCLS.

Joins the two series, computes validation metrics, and generates
comparison plots.

Usage:
    uv run python scripts/04_compare_to_fred.py
    uv run python scripts/04_compare_to_fred.py --plot  # Also generate plots
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import polars as pl

from vix_challenger.config import (
    SPY_VIX_LIKE_PARQUET,
    FRED_VIXCLS_PARQUET,
    JOINED_SPY_VS_VIX_PARQUET,
    REPORTS_DIR,
    ensure_directories,
)


def load_and_join_series(
    spy_path: Path,
    fred_path: Path,
) -> pl.DataFrame:
    """Load and join SPY VIX-like and FRED VIXCLS series.
    
    Args:
        spy_path: Path to spy_vix_like.parquet
        fred_path: Path to fred_vixcls.parquet
        
    Returns:
        Joined DataFrame with columns: date, spy_vix, vixcls
    """
    # Load SPY VIX-like
    spy_df = pl.read_parquet(spy_path)
    spy_df = spy_df.filter(pl.col("success") == True).select([
        pl.col("quote_date").alias("date"),
        pl.col("index").alias("spy_vix"),
    ])
    
    # Load FRED VIXCLS
    fred_df = pl.read_parquet(fred_path).select([
        pl.col("date"),
        pl.col("vixcls"),
    ])
    
    # Inner join on date
    joined = spy_df.join(fred_df, on="date", how="inner").sort("date")
    
    return joined


def compute_metrics(df: pl.DataFrame) -> dict:
    """Compute comparison metrics between SPY_VIX and VIXCLS.
    
    Args:
        df: DataFrame with spy_vix and vixcls columns
        
    Returns:
        Dictionary of metrics
    """
    spy = df["spy_vix"].to_numpy()
    vix = df["vixcls"].to_numpy()
    
    # Basic metrics
    n = len(df)
    
    # Correlation
    correlation = np.corrcoef(spy, vix)[0, 1]
    
    # Residuals
    residuals = spy - vix
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(residuals))
    
    # Mean Bias
    mean_bias = np.mean(residuals)
    
    # Regression: SPY_VIX = alpha + beta * VIXCLS
    vix_mean = np.mean(vix)
    spy_mean = np.mean(spy)
    beta = np.sum((vix - vix_mean) * (spy - spy_mean)) / np.sum((vix - vix_mean) ** 2)
    alpha = spy_mean - beta * vix_mean
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((spy - spy_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Percentile statistics of residuals
    residuals_p5 = np.percentile(residuals, 5)
    residuals_p95 = np.percentile(residuals, 95)
    
    return {
        "n_observations": n,
        "correlation": correlation,
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "mean_bias": mean_bias,
        "std_residuals": np.std(residuals),
        "residuals_p5": residuals_p5,
        "residuals_p95": residuals_p95,
        "regression_alpha": alpha,
        "regression_beta": beta,
        "spy_vix_mean": spy_mean,
        "spy_vix_std": np.std(spy),
        "vixcls_mean": vix_mean,
        "vixcls_std": np.std(vix),
    }


def compute_rolling_metrics(df: pl.DataFrame, window: int = 60) -> pl.DataFrame:
    """Compute rolling correlation and beta.
    
    Args:
        df: DataFrame with spy_vix and vixcls columns
        window: Rolling window size in days
        
    Returns:
        DataFrame with rolling metrics added
    """
    # Add rolling correlation
    df = df.with_columns([
        pl.struct(["spy_vix", "vixcls"])
        .map_batches(
            lambda s: pl.Series(
                np.array([
                    np.corrcoef(
                        s.struct.field("spy_vix")[max(0, i-window+1):i+1],
                        s.struct.field("vixcls")[max(0, i-window+1):i+1]
                    )[0, 1] if i >= window - 1 else np.nan
                    for i in range(len(s))
                ])
            ),
            return_dtype=pl.Float64
        )
        .alias("rolling_corr"),
    ])
    
    # Add rolling beta (regression coefficient)
    def compute_rolling_beta(spy_vals, vix_vals, window):
        betas = []
        for i in range(len(spy_vals)):
            if i < window - 1:
                betas.append(np.nan)
            else:
                y = spy_vals[i-window+1:i+1]
                x = vix_vals[i-window+1:i+1]
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
                betas.append(beta)
        return betas
    
    spy_vals = df["spy_vix"].to_numpy()
    vix_vals = df["vixcls"].to_numpy()
    rolling_betas = compute_rolling_beta(spy_vals, vix_vals, window)
    
    df = df.with_columns([
        pl.Series("rolling_beta", rolling_betas),
    ])
    
    # Add residuals
    df = df.with_columns([
        (pl.col("spy_vix") - pl.col("vixcls")).alias("residual"),
    ])
    
    return df


def print_metrics_summary(metrics: dict):
    """Print metrics summary to console."""
    print("\n" + "=" * 60)
    print("COMPARISON METRICS: SPY_VIX vs VIXCLS")
    print("=" * 60)
    
    print(f"\n[Sample Size]")
    print(f"  Matched observations: {metrics['n_observations']}")
    
    print(f"\n[Correlation]")
    print(f"  Pearson correlation: {metrics['correlation']:.4f}")
    print(f"  R-squared:           {metrics['r_squared']:.4f}")
    
    print(f"\n[Error Metrics]")
    print(f"  RMSE:                {metrics['rmse']:.4f}")
    print(f"  MAE:                 {metrics['mae']:.4f}")
    print(f"  Mean Bias:           {metrics['mean_bias']:+.4f}")
    print(f"  Std of Residuals:    {metrics['std_residuals']:.4f}")
    print(f"  Residuals P5-P95:    [{metrics['residuals_p5']:.2f}, {metrics['residuals_p95']:.2f}]")
    
    print(f"\n[Regression: SPY_VIX = α + β × VIXCLS]")
    print(f"  Alpha (intercept):   {metrics['regression_alpha']:.4f}")
    print(f"  Beta (slope):        {metrics['regression_beta']:.4f}")
    
    print(f"\n[Descriptive Stats]")
    print(f"  SPY_VIX mean:        {metrics['spy_vix_mean']:.2f}")
    print(f"  SPY_VIX std:         {metrics['spy_vix_std']:.2f}")
    print(f"  VIXCLS mean:         {metrics['vixcls_mean']:.2f}")
    print(f"  VIXCLS std:          {metrics['vixcls_std']:.2f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare SPY VIX-like to FRED VIXCLS")
    parser.add_argument(
        "--spy",
        type=Path,
        default=SPY_VIX_LIKE_PARQUET,
        help="Path to SPY VIX-like parquet"
    )
    parser.add_argument(
        "--fred",
        type=Path,
        default=FRED_VIXCLS_PARQUET,
        help="Path to FRED VIXCLS parquet"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=JOINED_SPY_VS_VIX_PARQUET,
        help="Output path for joined data"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots"
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPORTS_DIR / "figures",
        help="Directory for saving plots"
    )
    args = parser.parse_args()
    
    ensure_directories()
    
    # Check inputs exist
    if not args.spy.exists():
        print(f"ERROR: SPY VIX file not found: {args.spy}")
        print("Run scripts/02_compute_vix_like.py first.")
        sys.exit(1)
    
    if not args.fred.exists():
        print(f"ERROR: FRED VIXCLS file not found: {args.fred}")
        print("Run scripts/03_download_fred.py first.")
        sys.exit(1)
    
    print("=" * 60)
    print("SPY VIX vs FRED VIXCLS COMPARISON")
    print("=" * 60)
    
    # Load and join
    print("\nLoading and joining series...")
    joined_df = load_and_join_series(args.spy, args.fred)
    print(f"Matched {len(joined_df)} observations")
    print(f"Date range: {joined_df['date'].min()} to {joined_df['date'].max()}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(joined_df)
    
    # Add rolling metrics
    print("Computing rolling metrics (60-day window)...")
    joined_df = compute_rolling_metrics(joined_df, window=60)
    
    # Save joined data
    print(f"\nSaving joined data to {args.output}")
    joined_df.write_parquet(args.output)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        
        from vix_challenger.viz.comparison import generate_all_plots
        generate_all_plots(joined_df, metrics, args.figures_dir)
        print(f"Plots saved to: {args.figures_dir}")
    
    # Acceptance check
    print("\n[ACCEPTANCE CHECK]")
    if metrics["correlation"] > 0.90:
        print(f"  ✓ Correlation > 0.90: {metrics['correlation']:.4f}")
    else:
        print(f"  ✗ Correlation <= 0.90: {metrics['correlation']:.4f}")
    
    if len(joined_df) >= 700:
        print(f"  ✓ Observations >= 700: {len(joined_df)}")
    else:
        print(f"  ✗ Observations < 700: {len(joined_df)}")


if __name__ == "__main__":
    main()

