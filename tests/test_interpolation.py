"""Tests for 30-day interpolation."""

import pytest
import numpy as np


class TestInterpolation:
    """Tests for variance interpolation."""
    
    def test_interpolate_basic(self):
        """Test basic interpolation."""
        from vix_challenger.vix.interpolate import interpolate_30d_variance
        
        # Near at 25 days, next at 35 days
        # Should weight towards the closer one
        var_near = 0.04  # 20% vol
        var_next = 0.09  # 30% vol
        
        var_30d = interpolate_30d_variance(
            var_near=var_near, dte_near=25,
            var_next=var_next, dte_next=35,
        )
        
        assert var_30d > 0
        # Result should be between the two inputs (roughly)
    
    def test_interpolate_at_boundary(self):
        """Test interpolation when near=30."""
        from vix_challenger.vix.interpolate import interpolate_30d_variance
        
        var_near = 0.04
        var_next = 0.09
        
        var_30d = interpolate_30d_variance(
            var_near=var_near, dte_near=30,
            var_next=var_next, dte_next=37,
        )
        
        assert var_30d > 0
    
    def test_vix_index_computation(self):
        """Test VIX index from variance."""
        from vix_challenger.vix.interpolate import compute_vix_index
        
        # 16% annual vol -> variance = 0.0256
        variance = 0.0256
        vix = compute_vix_index(variance)
        
        assert abs(vix - 16.0) < 0.1
    
    def test_vix_index_handles_negative(self):
        """Test that negative variance is handled."""
        from vix_challenger.vix.interpolate import compute_vix_index
        
        # Should not crash, should return positive value
        vix = compute_vix_index(-0.01)
        assert vix > 0

