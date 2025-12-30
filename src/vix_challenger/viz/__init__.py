"""Visualization modules.

- comparison: SPY VIX vs VIXCLS comparison plots
"""

from vix_challenger.viz.comparison import (
    plot_overlay,
    plot_scatter,
    plot_residuals,
    plot_rolling_correlation,
    plot_rolling_beta,
    generate_all_plots,
)

__all__ = [
    "plot_overlay",
    "plot_scatter",
    "plot_residuals",
    "plot_rolling_correlation",
    "plot_rolling_beta",
    "generate_all_plots",
]
