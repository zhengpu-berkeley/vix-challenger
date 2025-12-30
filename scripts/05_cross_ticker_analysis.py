#!/usr/bin/env python3
"""Cross-ticker VIX correlation analysis.

Analyzes correlations between VIX-like indices computed for different tickers.

Usage:
    uv run python scripts/05_cross_ticker_analysis.py
    uv run python scripts/05_cross_ticker_analysis.py --plot
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from vix_challenger.config import (
    get_ticker_config,
    list_tickers,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    FRED_VIXCLS_PARQUET,
)


def load_all_vix_series() -> dict[str, pl.DataFrame]:
    """Load VIX-like series for all available tickers."""
    series = {}
    
    for ticker in list_tickers():
        config = get_ticker_config(ticker)
        path = config.vix_like_parquet
        
        if path.exists():
            df = pl.read_parquet(path)
            df = df.filter(pl.col("success") == True).select([
                pl.col("quote_date").alias("date"),
                pl.col("index").alias(f"{ticker.lower()}_vix"),
            ])
            series[ticker] = df
            print(f"  {ticker}: {len(df)} observations ({df['date'].min()} to {df['date'].max()})")
        else:
            print(f"  {ticker}: Not found ({path})")
    
    return series


def load_vixcls() -> pl.DataFrame:
    """Load FRED VIXCLS data."""
    if not FRED_VIXCLS_PARQUET.exists():
        print(f"VIXCLS not found: {FRED_VIXCLS_PARQUET}")
        return None
    
    df = pl.read_parquet(FRED_VIXCLS_PARQUET).select([
        pl.col("date"),
        pl.col("vixcls"),
    ])
    print(f"  VIXCLS: {len(df)} observations ({df['date'].min()} to {df['date'].max()})")
    return df


def join_all_series(series: dict[str, pl.DataFrame], vixcls: pl.DataFrame = None) -> pl.DataFrame:
    """Join all series on common dates."""
    if not series:
        return None
    
    # Start with first ticker
    tickers = list(series.keys())
    joined = series[tickers[0]]
    
    # Join remaining tickers
    for ticker in tickers[1:]:
        joined = joined.join(series[ticker], on="date", how="inner")
    
    # Join VIXCLS if available
    if vixcls is not None:
        joined = joined.join(vixcls, on="date", how="inner")
    
    return joined.sort("date")


def compute_correlation_matrix(df: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Compute correlation matrix for specified columns."""
    data = df.select(columns).to_numpy()
    return np.corrcoef(data.T)


def print_correlation_matrix(corr_matrix: np.ndarray, labels: list[str]):
    """Print formatted correlation matrix."""
    n = len(labels)
    
    # Header
    print("\n" + " " * 12 + "".join(f"{lbl:>12}" for lbl in labels))
    
    # Rows
    for i, label in enumerate(labels):
        row = f"{label:<12}"
        for j in range(n):
            row += f"{corr_matrix[i, j]:>12.4f}"
        print(row)


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    labels: list[str],
    save_path: Path = None,
    figsize: tuple = (10, 8),
):
    """Plot correlation matrix as heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom", fontsize=12)
    
    # Ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{corr_matrix[i, j]:.3f}",
                          ha="center", va="center", fontsize=10,
                          color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    
    ax.set_title("VIX-like Index Correlation Matrix", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


def plot_multi_ticker_overlay(
    df: pl.DataFrame,
    columns: list[str],
    save_path: Path = None,
    figsize: tuple = (14, 8),
):
    """Plot all VIX-like series on one chart."""
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
    
    dates = df["date"].to_list()
    
    for i, col in enumerate(columns):
        label = col.replace("_vix", "").upper()
        if col == "vixcls":
            label = "VIXCLS"
        ax.plot(dates, df[col].to_numpy(), label=label, linewidth=1.5,
                alpha=0.8, color=colors[i])
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volatility Index", fontsize=12)
    ax.set_title("Multi-Ticker VIX-like Index Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    ax.set_ylim(0, min(150, df.select(columns).max().to_numpy().max() * 1.1))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


def plot_volatility_spread(
    df: pl.DataFrame,
    ticker1: str,
    ticker2: str,
    save_path: Path = None,
    figsize: tuple = (14, 5),
):
    """Plot volatility spread between two tickers."""
    col1 = f"{ticker1.lower()}_vix"
    col2 = f"{ticker2.lower()}_vix" if ticker2.upper() != "VIXCLS" else "vixcls"
    
    if col1 not in df.columns or col2 not in df.columns:
        print(f"  Skipping spread plot: missing {col1} or {col2}")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dates = df["date"].to_list()
    spread = (df[col1] - df[col2]).to_numpy()
    
    colors = ["#2E86AB" if s >= 0 else "#A23B72" for s in spread]
    ax.bar(dates, spread, color=colors, alpha=0.7, width=1)
    
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(np.mean(spread), color="#F18F01", linewidth=2, linestyle="--",
               label=f"Mean spread: {np.mean(spread):+.2f}")
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"Spread ({ticker1} - {ticker2})", fontsize=12)
    ax.set_title(f"Volatility Spread: {ticker1} vs {ticker2}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Cross-ticker VIX analysis")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CROSS-TICKER VIX ANALYSIS")
    print("=" * 60)
    
    # Load all series
    print("\n[Loading VIX-like series]")
    series = load_all_vix_series()
    vixcls = load_vixcls()
    
    if not series:
        print("No VIX series found!")
        sys.exit(1)
    
    # Join all series
    print("\n[Joining series on common dates]")
    joined = join_all_series(series, vixcls)
    
    print(f"Joined dataset: {len(joined)} common observations")
    print(f"Date range: {joined['date'].min()} to {joined['date'].max()}")
    
    # Identify columns for correlation
    vix_columns = [c for c in joined.columns if c.endswith("_vix") or c == "vixcls"]
    
    print(f"\n[Computing correlations for: {', '.join(vix_columns)}]")
    
    corr_matrix = compute_correlation_matrix(joined, vix_columns)
    
    labels = [c.replace("_vix", "").upper() if c != "vixcls" else "VIXCLS" for c in vix_columns]
    print_correlation_matrix(corr_matrix, labels)
    
    # Summary statistics
    print("\n[Summary Statistics - Common Period]")
    for col in vix_columns:
        label = col.replace("_vix", "").upper() if col != "vixcls" else "VIXCLS"
        vals = joined[col].to_numpy()
        print(f"  {label:8} | Mean: {np.mean(vals):6.2f} | Std: {np.std(vals):6.2f} | "
              f"Min: {np.min(vals):6.2f} | Max: {np.max(vals):6.2f}")
    
    # Save joined data
    output_path = PROCESSED_DATA_DIR / "cross_ticker_vix.parquet"
    joined.write_parquet(output_path)
    print(f"\n[Saved to {output_path}]")
    
    if args.plot:
        print("\n[Generating plots]")
        figures_dir = REPORTS_DIR / "figures" / "cross_ticker"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Correlation heatmap
        plot_correlation_heatmap(
            corr_matrix, labels,
            save_path=figures_dir / "01_correlation_heatmap.png"
        )
        
        # Multi-ticker overlay
        plot_multi_ticker_overlay(
            joined, vix_columns,
            save_path=figures_dir / "02_multi_ticker_overlay.png"
        )
        
        # Spreads vs VIXCLS
        if "vixcls" in joined.columns:
            for ticker in series.keys():
                if ticker != "SPY":  # SPY is very close to VIXCLS
                    plot_volatility_spread(
                        joined, ticker, "VIXCLS",
                        save_path=figures_dir / f"03_spread_{ticker.lower()}_vs_vixcls.png"
                    )
        
        plt.close("all")
        print(f"Plots saved to: {figures_dir}")
    
    print("\n" + "=" * 60)
    print("CROSS-TICKER ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

