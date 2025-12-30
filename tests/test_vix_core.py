"""Tests for VIX core computation modules."""

import pytest
from datetime import date

import numpy as np
import polars as pl


class TestForwardPriceComputation:
    """Tests for forward price via put-call parity."""
    
    def test_forward_price_computed(self, sample_day_df):
        """Test that forward price can be computed."""
        from vix_challenger.vix.parity import compute_forward_price
        from vix_challenger.io.spy_csv import Cols
        
        # Pick an expiry with DTE around 30
        expirations = sample_day_df.group_by("expiration").agg(
            pl.col("dte").first()
        ).filter(pl.col("dte").is_between(20, 40))
        
        assert len(expirations) > 0, "No expirations with DTE 20-40"
        
        exp = expirations["expiration"].head(1).item()
        exp_df = sample_day_df.filter(pl.col("expiration") == exp)
        
        result = compute_forward_price(exp_df)
        
        assert result.forward > 0
        assert result.k0 > 0
        assert result.k0 <= result.forward  # K0 should be <= F
        assert result.n_valid_strikes >= 10
    
    def test_forward_close_to_underlying(self, sample_day_df):
        """Test that forward is close to underlying price."""
        from vix_challenger.vix.parity import compute_forward_price
        
        # Get underlying price
        underlying = sample_day_df["underlying_price"].head(1).item()
        
        # Pick a short-dated expiry
        expirations = sample_day_df.group_by("expiration").agg(
            pl.col("dte").first()
        ).filter(pl.col("dte").is_between(5, 15))
        
        if len(expirations) == 0:
            pytest.skip("No short-dated expirations")
        
        exp = expirations["expiration"].head(1).item()
        exp_df = sample_day_df.filter(pl.col("expiration") == exp)
        
        result = compute_forward_price(exp_df)
        
        # Forward should be within 2% of underlying for short-dated
        pct_diff = abs(result.forward - underlying) / underlying * 100
        assert pct_diff < 2, f"Forward {result.forward} too far from underlying {underlying}"


class TestOTMStrip:
    """Tests for OTM strip construction."""
    
    def test_otm_strip_built(self, sample_day_df):
        """Test that OTM strip can be built."""
        from vix_challenger.vix.parity import compute_forward_price
        from vix_challenger.vix.strip import build_otm_strip
        
        # Get an expiry
        expirations = sample_day_df.group_by("expiration").agg(
            pl.col("dte").first()
        ).filter(pl.col("dte").is_between(20, 40))
        
        exp = expirations["expiration"].head(1).item()
        exp_df = sample_day_df.filter(pl.col("expiration") == exp)
        
        # Compute forward
        fwd = compute_forward_price(exp_df)
        
        # Build strip
        strip = build_otm_strip(exp_df, fwd.k0)
        
        assert len(strip.strikes) > 0
        assert len(strip.quotes) == len(strip.strikes)
        assert len(strip.delta_k) == len(strip.strikes)
        assert strip.n_puts >= 0
        assert strip.n_calls >= 0
    
    def test_otm_strip_has_both_wings(self, sample_day_df):
        """Test that strip has puts and calls."""
        from vix_challenger.vix.parity import compute_forward_price
        from vix_challenger.vix.strip import build_otm_strip
        
        expirations = sample_day_df.group_by("expiration").agg(
            pl.col("dte").first()
        ).filter(pl.col("dte").is_between(20, 40))
        
        exp = expirations["expiration"].head(1).item()
        exp_df = sample_day_df.filter(pl.col("expiration") == exp)
        
        fwd = compute_forward_price(exp_df)
        strip = build_otm_strip(exp_df, fwd.k0)
        
        assert strip.n_puts > 0, "No OTM puts in strip"
        assert strip.n_calls > 0, "No OTM calls in strip"


class TestVarianceComputation:
    """Tests for variance computation."""
    
    def test_variance_positive(self, sample_day_df):
        """Test that computed variance is positive."""
        from vix_challenger.vix.variance import compute_expiry_variance
        
        expirations = sample_day_df.group_by("expiration").agg(
            pl.col("dte").first()
        ).filter(pl.col("dte").is_between(20, 40))
        
        exp = expirations["expiration"].head(1).item()
        exp_df = sample_day_df.filter(pl.col("expiration") == exp)
        
        result = compute_expiry_variance(exp_df)
        
        assert result.variance > 0
        assert result.implied_vol > 0
    
    def test_implied_vol_reasonable(self, sample_day_df):
        """Test that implied vol is in reasonable range."""
        from vix_challenger.vix.variance import compute_expiry_variance
        
        expirations = sample_day_df.group_by("expiration").agg(
            pl.col("dte").first()
        ).filter(pl.col("dte").is_between(20, 40))
        
        exp = expirations["expiration"].head(1).item()
        exp_df = sample_day_df.filter(pl.col("expiration") == exp)
        
        result = compute_expiry_variance(exp_df)
        
        # Normal market: IV should be 5-50%
        assert 5 < result.implied_vol < 50, f"IV {result.implied_vol}% outside normal range"
    
    def test_covid_high_vol(self, covid_day_df):
        """Test that COVID day shows high volatility."""
        from vix_challenger.vix.variance import compute_expiry_variance
        
        expirations = covid_day_df.group_by("expiration").agg(
            pl.col("dte").first()
        ).filter(pl.col("dte").is_between(20, 40))
        
        if len(expirations) == 0:
            pytest.skip("No suitable expirations on COVID day")
        
        exp = expirations["expiration"].head(1).item()
        exp_df = covid_day_df.filter(pl.col("expiration") == exp)
        
        result = compute_expiry_variance(exp_df)
        
        # COVID peak: IV should be > 50%
        assert result.implied_vol > 50, f"COVID IV {result.implied_vol}% too low"


class TestDailyVIX:
    """Tests for daily VIX computation."""
    
    def test_daily_vix_computes(self, sample_day_df, sample_date):
        """Test that daily VIX can be computed."""
        from vix_challenger.vix.daily import compute_daily_vix
        
        result = compute_daily_vix(sample_day_df, sample_date)
        
        assert result.success
        assert result.index is not None
        assert result.index > 0
    
    def test_daily_vix_has_diagnostics(self, sample_day_df, sample_date):
        """Test that daily VIX returns diagnostics."""
        from vix_challenger.vix.daily import compute_daily_vix
        
        result = compute_daily_vix(sample_day_df, sample_date)
        
        assert result.near_exp is not None
        assert result.next_exp is not None
        assert result.sigma2_near is not None
        assert result.sigma2_next is not None

