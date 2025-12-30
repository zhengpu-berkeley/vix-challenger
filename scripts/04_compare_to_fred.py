#!/usr/bin/env python3
"""Compare ticker VIX-like index to FRED VIXCLS.

Usage:
    uv run python scripts/04_compare_to_fred.py                    # SPY (default)
    uv run python scripts/04_compare_to_fred.py --ticker SPY
    uv run python scripts/04_compare_to_fred.py --ticker SPY --plot
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import polars as pl

from vix_challenger.config import (
    get_ticker_config,
    list_tickers,
    FRED_VIXCLS_PARQUET,
    REPORTS_DIR,
    ensure_ticker_directories,
)


def load_and_join_series(
    ticker: str,
    fred_path: Path,
) -> pl.DataFrame:
    """Load and join ticker VIX-like and FRED VIXCLS series."""
    config = get_ticker_config(ticker)
    
    # Load ticker VIX-like
    ticker_df = pl.read_parquet(config.vix_like_parquet)
    ticker_df = ticker_df.filter(pl.col("success") == True).select([
        pl.col("quote_date").alias("date"),
        pl.col("index").alias(f"{ticker.lower()}_vix"),
    ])
    
    # Load FRED VIXCLS
    fred_df = pl.read_parquet(fred_path).select([
        pl.col("date"),
        pl.col("vixcls"),
    ])
    
    # Inner join on date
    joined = ticker_df.join(fred_df, on="date", how="inner").sort("date")
    
    return joined


def compute_metrics(df: pl.DataFrame, ticker: str) -> dict:
    """Compute comparison metrics."""
    ticker_col = f"{ticker.lower()}_vix"
    ticker_vals = df[ticker_col].to_numpy()
    vix_vals = df["vixcls"].to_numpy()
    
    n = len(df)
    correlation = np.corrcoef(ticker_vals, vix_vals)[0, 1]
    residuals = ticker_vals - vix_vals
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    mean_bias = np.mean(residuals)
    
    vix_mean = np.mean(vix_vals)
    ticker_mean = np.mean(ticker_vals)
    beta = np.sum((vix_vals - vix_mean) * (ticker_vals - ticker_mean)) / np.sum((vix_vals - vix_mean) ** 2)
    alpha = ticker_mean - beta * vix_mean
    
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ticker_vals - ticker_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        "ticker": ticker,
        "n_observations": n,
        "correlation": correlation,
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "mean_bias": mean_bias,
        "std_residuals": np.std(residuals),
        "regression_alpha": alpha,
        "regression_beta": beta,
        f"{ticker.lower()}_vix_mean": ticker_mean,
        f"{ticker.lower()}_vix_std": np.std(ticker_vals),
        "vixcls_mean": vix_mean,
        "vixcls_std": np.std(vix_vals),
    }


def compute_rolling_metrics(df: pl.DataFrame, ticker: str, window: int = 60) -> pl.DataFrame:
    """Compute rolling correlation and beta."""
    ticker_col = f"{ticker.lower()}_vix"
    
    df = df.with_columns([
        pl.struct([ticker_col, "vixcls"])
        .map_batches(
            lambda s: pl.Series(
                np.array([
                    np.corrcoef(
                        s.struct.field(ticker_col)[max(0, i-window+1):i+1],
                        s.struct.field("vixcls")[max(0, i-window+1):i+1]
                    )[0, 1] if i >= window - 1 else np.nan
                    for i in range(len(s))
                ])
            ),
            return_dtype=pl.Float64
        )
        .alias("rolling_corr"),
    ])
    
    ticker_vals = df[ticker_col].to_numpy()
    vix_vals = df["vixcls"].to_numpy()
    
    def compute_rolling_beta(y_vals, x_vals, window):
        betas = []
        for i in range(len(y_vals)):
            if i < window - 1:
                betas.append(np.nan)
            else:
                y = y_vals[i-window+1:i+1]
                x = x_vals[i-window+1:i+1]
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
                betas.append(beta)
        return betas
    
    rolling_betas = compute_rolling_beta(ticker_vals, vix_vals, window)
    
    df = df.with_columns([
        pl.Series("rolling_beta", rolling_betas),
        (pl.col(ticker_col) - pl.col("vixcls")).alias("residual"),
    ])
    
    return df


def print_metrics_summary(metrics: dict):
    """Print metrics summary."""
    ticker = metrics["ticker"]
    print("\n" + "=" * 60)
    print(f"COMPARISON METRICS: {ticker}_VIX vs VIXCLS")
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
    
    print(f"\n[Regression: {ticker}_VIX = α + β × VIXCLS]")
    print(f"  Alpha (intercept):   {metrics['regression_alpha']:.4f}")
    print(f"  Beta (slope):        {metrics['regression_beta']:.4f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare ticker VIX to FRED VIXCLS")
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help=f"Ticker symbol. Available: {', '.join(list_tickers())}"
    )
    parser.add_argument(
        "--fred",
        type=Path,
        default=FRED_VIXCLS_PARQUET,
        help="Path to FRED VIXCLS parquet"
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    config = get_ticker_config(ticker)
    ensure_ticker_directories(ticker)
    
    if not config.vix_like_parquet.exists():
        print(f"ERROR: VIX results not found: {config.vix_like_parquet}")
        print(f"Run scripts/02_compute_vix_like.py --ticker {ticker} first.")
        sys.exit(1)
    
    if not args.fred.exists():
        print(f"ERROR: FRED data not found: {args.fred}")
        print("Run scripts/03_download_fred.py first.")
        sys.exit(1)
    
    print("=" * 60)
    print(f"{ticker} VIX vs FRED VIXCLS COMPARISON")
    print("=" * 60)
    
    print("\nLoading and joining series...")
    joined_df = load_and_join_series(ticker, args.fred)
    print(f"Matched {len(joined_df)} observations")
    
    print("\nComputing metrics...")
    metrics = compute_metrics(joined_df, ticker)
    
    print("Computing rolling metrics...")
    joined_df = compute_rolling_metrics(joined_df, ticker, window=60)
    
    # Save joined data
    output_path = config.joined_comparison_parquet
    print(f"\nSaving to {output_path}")
    joined_df.write_parquet(output_path)
    
    print_metrics_summary(metrics)
    
    if args.plot:
        print("\nGenerating plots...")
        config.figures_dir.mkdir(parents=True, exist_ok=True)
        from vix_challenger.viz.comparison import generate_all_plots
        generate_all_plots(joined_df, metrics, config.figures_dir)
        print(f"Plots saved to: {config.figures_dir}")
    
    print("\n[ACCEPTANCE CHECK]")
    if metrics["correlation"] > 0.90:
        print(f"  ✓ Correlation > 0.90: {metrics['correlation']:.4f}")
    else:
        print(f"  ✗ Correlation <= 0.90: {metrics['correlation']:.4f}")


if __name__ == "__main__":
    main()
