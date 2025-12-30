"""Visualization module for ticker VIX vs VIXCLS comparison.

Generates publication-quality plots for comparison reports.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "ticker_vix": "#2E86AB",  # Blue
    "vixcls": "#A23B72",      # Magenta
    "residual": "#F18F01",    # Orange
    "rolling": "#C73E1D",     # Red
}


def _get_ticker_col(df: pl.DataFrame) -> tuple[str, str]:
    """Get the ticker VIX column name and ticker symbol from the dataframe.
    
    Returns:
        Tuple of (column_name, ticker_symbol)
    """
    for col in df.columns:
        if col.endswith("_vix") and col != "vixcls":
            ticker = col.replace("_vix", "").upper()
            return col, ticker
    # Fallback to spy_vix for backwards compatibility
    return "spy_vix", "SPY"


def plot_overlay(
    df: pl.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot overlay of ticker VIX and VIXCLS time series.
    
    Args:
        df: DataFrame with date, {ticker}_vix, vixcls columns
        save_path: If provided, save figure to this path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    ticker_col, ticker = _get_ticker_col(df)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dates = df["date"].to_list()
    ticker_vix = df[ticker_col].to_numpy()
    vixcls = df["vixcls"].to_numpy()
    
    ax.plot(dates, ticker_vix, color=COLORS["ticker_vix"], linewidth=1.5, 
            label=f"{ticker} VIX-like", alpha=0.9)
    ax.plot(dates, vixcls, color=COLORS["vixcls"], linewidth=1.5, 
            label="VIXCLS (FRED)", alpha=0.9)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volatility Index", fontsize=12)
    
    # Get date range
    date_start = dates[0].strftime("%Y") if hasattr(dates[0], 'strftime') else str(dates[0])[:4]
    date_end = dates[-1].strftime("%Y") if hasattr(dates[-1], 'strftime') else str(dates[-1])[:4]
    ax.set_title(f"{ticker}-based VIX vs Official VIXCLS ({date_start}-{date_end})", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    # Add annotation for peak
    peak_idx = np.argmax(vixcls)
    ax.annotate(
        f"Peak\n{vixcls[peak_idx]:.1f}",
        xy=(dates[peak_idx], vixcls[peak_idx]),
        xytext=(dates[peak_idx], vixcls[peak_idx] + 10),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7),
    )
    
    ax.set_ylim(0, max(ticker_vix.max(), vixcls.max()) * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


def plot_scatter(
    df: pl.DataFrame,
    metrics: dict,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Plot scatter of ticker VIX vs VIXCLS with regression line.
    
    Args:
        df: DataFrame with {ticker}_vix, vixcls columns
        metrics: Dictionary with regression_alpha, regression_beta
        save_path: If provided, save figure to this path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    ticker_col, ticker = _get_ticker_col(df)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ticker_vix = df[ticker_col].to_numpy()
    vixcls = df["vixcls"].to_numpy()
    
    # Scatter plot
    ax.scatter(vixcls, ticker_vix, alpha=0.4, s=20, color=COLORS["ticker_vix"], 
               edgecolors="none")
    
    # Regression line
    x_line = np.array([vixcls.min(), vixcls.max()])
    y_line = metrics["regression_alpha"] + metrics["regression_beta"] * x_line
    ax.plot(x_line, y_line, color=COLORS["residual"], linewidth=2, 
            label=f"y = {metrics['regression_alpha']:.2f} + {metrics['regression_beta']:.2f}x")
    
    # 45-degree reference line
    max_val = max(ticker_vix.max(), vixcls.max())
    ax.plot([0, max_val], [0, max_val], color="gray", linestyle="--", 
            linewidth=1, alpha=0.7, label="y = x (perfect match)")
    
    ax.set_xlabel("VIXCLS (Official)", fontsize=12)
    ax.set_ylabel(f"{ticker} VIX-like", fontsize=12)
    ax.set_title(f"Scatter: {ticker} VIX vs VIXCLS\n(r = {metrics['correlation']:.4f})", 
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


def plot_residuals(
    df: pl.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot residuals (ticker_VIX - VIXCLS) over time.
    
    Args:
        df: DataFrame with date, residual columns
        save_path: If provided, save figure to this path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    ticker_col, ticker = _get_ticker_col(df)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    dates = df["date"].to_list()
    residuals = df["residual"].to_numpy()
    
    # Bar chart for residuals
    colors = [COLORS["ticker_vix"] if r >= 0 else COLORS["vixcls"] for r in residuals]
    ax.bar(dates, residuals, color=colors, alpha=0.7, width=1)
    
    # Zero line
    ax.axhline(0, color="black", linewidth=1)
    
    # Mean bias line
    mean_bias = np.mean(residuals)
    ax.axhline(mean_bias, color=COLORS["residual"], linewidth=2, linestyle="--",
               label=f"Mean bias: {mean_bias:+.2f}")
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"Residual ({ticker}_VIX - VIXCLS)", fontsize=12)
    ax.set_title("Residuals Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


def plot_rolling_correlation(
    df: pl.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot 60-day rolling correlation.
    
    Args:
        df: DataFrame with date, rolling_corr columns
        save_path: If provided, save figure to this path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    ticker_col, ticker = _get_ticker_col(df)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out NaN values
    valid_df = df.filter(pl.col("rolling_corr").is_not_null())
    dates = valid_df["date"].to_list()
    rolling_corr = valid_df["rolling_corr"].to_numpy()
    
    ax.plot(dates, rolling_corr, color=COLORS["rolling"], linewidth=1.5)
    ax.fill_between(dates, rolling_corr, alpha=0.3, color=COLORS["rolling"])
    
    # Reference lines
    ax.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.axhline(0.9, color="gray", linewidth=1, linestyle=":", alpha=0.7, 
               label="0.9 threshold")
    
    mean_corr = np.nanmean(rolling_corr)
    ax.axhline(mean_corr, color=COLORS["ticker_vix"], linewidth=2, linestyle="--",
               label=f"Mean: {mean_corr:.4f}")
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rolling 60-Day Correlation", fontsize=12)
    ax.set_title(f"Rolling Correlation: {ticker} VIX vs VIXCLS", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    
    ax.set_ylim(0.8, 1.02)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


def plot_rolling_beta(
    df: pl.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot 60-day rolling regression beta.
    
    Args:
        df: DataFrame with date, rolling_beta columns
        save_path: If provided, save figure to this path
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    ticker_col, ticker = _get_ticker_col(df)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out NaN values
    valid_df = df.filter(pl.col("rolling_beta").is_not_null())
    dates = valid_df["date"].to_list()
    rolling_beta = valid_df["rolling_beta"].to_numpy()
    
    ax.plot(dates, rolling_beta, color=COLORS["vixcls"], linewidth=1.5)
    ax.fill_between(dates, rolling_beta, 1.0, alpha=0.3, color=COLORS["vixcls"])
    
    # Reference line at 1.0 (perfect match)
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="β = 1 (perfect)")
    
    mean_beta = np.nanmean(rolling_beta)
    ax.axhline(mean_beta, color=COLORS["ticker_vix"], linewidth=2, linestyle="--",
               label=f"Mean: {mean_beta:.4f}")
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rolling 60-Day Beta", fontsize=12)
    ax.set_title(f"Rolling Regression Beta: {ticker} VIX vs VIXCLS", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    
    ax.set_ylim(0.8, 1.2)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    
    return fig


def generate_all_plots(
    df: pl.DataFrame,
    metrics: dict,
    output_dir: Path,
) -> dict:
    """Generate all comparison plots.
    
    Args:
        df: Joined DataFrame with all columns
        metrics: Metrics dictionary
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    # 1. Overlay plot
    plots["overlay"] = output_dir / "01_overlay.png"
    plot_overlay(df, save_path=plots["overlay"])
    
    # 2. Scatter plot
    plots["scatter"] = output_dir / "02_scatter.png"
    plot_scatter(df, metrics, save_path=plots["scatter"])
    
    # 3. Residuals plot
    plots["residuals"] = output_dir / "03_residuals.png"
    plot_residuals(df, save_path=plots["residuals"])
    
    # 4. Rolling correlation
    plots["rolling_corr"] = output_dir / "04_rolling_correlation.png"
    plot_rolling_correlation(df, save_path=plots["rolling_corr"])
    
    # 5. Rolling beta
    plots["rolling_beta"] = output_dir / "05_rolling_beta.png"
    plot_rolling_beta(df, save_path=plots["rolling_beta"])
    
    plt.close("all")
    
    return plots
