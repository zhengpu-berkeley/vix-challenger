"""30-day constant maturity interpolation for VIX computation.

Per Cboe VIX methodology, interpolate between near-term and next-term
variance to produce a constant 30-day variance estimate.

Official VIX uses minutes; for POC we use days.
"""

import numpy as np


# Constants
DAYS_PER_YEAR = 365
TARGET_DTE = 30


def interpolate_30d_variance(
    var_near: float,
    dte_near: int,
    var_next: float,
    dte_next: int,
    target_dte: int = TARGET_DTE,
) -> float:
    """Interpolate to constant 30-day variance.
    
    Uses the VIX interpolation formula (simplified for days instead of minutes):
    
    var_30d = w1 * var_near * (T_near / T_30) + w2 * var_next * (T_next / T_30)
    
    Where weights w1, w2 ensure linear interpolation in time.
    
    Args:
        var_near: Variance of near-term expiry (sigma^2)
        dte_near: Days to expiration for near-term
        var_next: Variance of next-term expiry (sigma^2)
        dte_next: Days to expiration for next-term
        target_dte: Target DTE for interpolation (default 30)
        
    Returns:
        Interpolated 30-day variance
        
    Note:
        If both expirations are on the same side of target (both < 30 or both > 30),
        we extrapolate rather than interpolate.
    """
    # Time in years
    T_near = dte_near / DAYS_PER_YEAR
    T_next = dte_next / DAYS_PER_YEAR
    T_target = target_dte / DAYS_PER_YEAR
    
    # Handle edge case: same DTE (shouldn't happen but be safe)
    if dte_near == dte_next:
        # Just average the variances
        return (var_near + var_next) / 2
    
    # Compute weights for interpolation
    # This is the standard VIX formula structure
    N_near = dte_near
    N_next = dte_next
    N_target = target_dte
    
    # Weight formula: linear interpolation in time
    # w1 = (N_next - N_target) / (N_next - N_near)
    # w2 = (N_target - N_near) / (N_next - N_near)
    
    denom = N_next - N_near
    w_near = (N_next - N_target) / denom
    w_next = (N_target - N_near) / denom
    
    # VIX formula scales each variance by its time ratio
    # Then sums and normalizes to target time
    #
    # var_30d = T_near * var_near * w_near + T_next * var_next * w_next
    # Then multiply by (N_365 / N_target) to annualize
    #
    # Simplified (since var is already annualized variance):
    # var_30d = w_near * var_near + w_next * var_next
    
    # Actually, the Cboe formula is:
    # VIX^2 = 100^2 * { T1*σ1^2 * [(N_T2 - N_30)/(N_T2 - N_T1)]
    #                 + T2*σ2^2 * [(N_30 - N_T1)/(N_T2 - N_T1)] } * (N_365 / N_30)
    #
    # Since our σ^2 is already in per-year units, we do:
    var_30d = (
        T_near * var_near * w_near + 
        T_next * var_next * w_next
    ) * (DAYS_PER_YEAR / target_dte)
    
    return var_30d


def compute_vix_index(var_30d: float) -> float:
    """Convert 30-day variance to VIX-style index.
    
    Args:
        var_30d: 30-day variance (annualized)
        
    Returns:
        VIX-like index: 100 * sqrt(var_30d)
    """
    if var_30d < 0:
        # Negative variance is theoretically impossible but can happen
        # due to numerical issues. Take absolute value with warning.
        var_30d = abs(var_30d)
    
    return 100.0 * np.sqrt(var_30d)


def interpolate_and_compute_index(
    var_near: float,
    dte_near: int,
    var_next: float,
    dte_next: int,
    target_dte: int = TARGET_DTE,
) -> tuple[float, float]:
    """Interpolate variance and compute VIX index in one call.
    
    Args:
        var_near: Near-term variance
        dte_near: Near-term DTE
        var_next: Next-term variance
        dte_next: Next-term DTE
        target_dte: Target DTE (default 30)
        
    Returns:
        (var_30d, vix_index) tuple
    """
    var_30d = interpolate_30d_variance(
        var_near, dte_near,
        var_next, dte_next,
        target_dte,
    )
    vix_index = compute_vix_index(var_30d)
    
    return var_30d, vix_index

