"""VIX computation modules.

Core modules:
- parity: Forward price via put-call parity
- strip: OTM option strip construction
- variance: Model-free variance computation
- selection: Near/next expiry selection
- interpolate: 30-day constant maturity interpolation
- daily: Single-day VIX computation
- qc: Quality control metrics
"""

from vix_challenger.vix.parity import (
    ForwardResult,
    compute_forward_price,
    get_expiry_data,
    list_expirations,
)

from vix_challenger.vix.strip import (
    OTMStrip,
    build_otm_strip,
)

from vix_challenger.vix.variance import (
    VarianceResult,
    compute_expiry_variance,
    compute_variance_for_day,
    print_variance_diagnostics,
)

from vix_challenger.vix.selection import (
    ExpirySelection,
    SelectionError,
    select_vix_expiries,
    get_available_expirations,
)

from vix_challenger.vix.interpolate import (
    interpolate_30d_variance,
    compute_vix_index,
    interpolate_and_compute_index,
)

from vix_challenger.vix.daily import (
    DailyVIXResult,
    compute_daily_vix,
    result_to_dict,
)

from vix_challenger.vix.qc import (
    DayQCMetrics,
    compute_day_qc_metrics,
    qc_metrics_to_dict,
    summarize_skip_reasons,
)

__all__ = [
    # Parity
    "ForwardResult",
    "compute_forward_price",
    "get_expiry_data",
    "list_expirations",
    # Strip
    "OTMStrip",
    "build_otm_strip",
    # Variance
    "VarianceResult",
    "compute_expiry_variance",
    "compute_variance_for_day",
    "print_variance_diagnostics",
    # Selection
    "ExpirySelection",
    "SelectionError",
    "select_vix_expiries",
    "get_available_expirations",
    # Interpolation
    "interpolate_30d_variance",
    "compute_vix_index",
    "interpolate_and_compute_index",
    # Daily
    "DailyVIXResult",
    "compute_daily_vix",
    "result_to_dict",
    # QC
    "DayQCMetrics",
    "compute_day_qc_metrics",
    "qc_metrics_to_dict",
    "summarize_skip_reasons",
]
