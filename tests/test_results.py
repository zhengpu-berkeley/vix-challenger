"""Tests for pipeline results and validation."""

import pytest
import numpy as np


class TestSPYVIXResults:
    """Tests for computed SPY VIX-like results."""
    
    def test_results_exist(self, spy_vix_like_df):
        """Test that results file exists and has data."""
        assert len(spy_vix_like_df) > 0
    
    def test_sufficient_observations(self, spy_vix_like_df):
        """Test that we have enough successful observations."""
        successful = spy_vix_like_df.filter(spy_vix_like_df["success"] == True)
        assert len(successful) >= 700, f"Only {len(successful)} successful days"
    
    def test_no_negative_index(self, spy_vix_like_df):
        """Test that no index values are negative."""
        successful = spy_vix_like_df.filter(spy_vix_like_df["success"] == True)
        negative = successful.filter(successful["index"] < 0)
        assert len(negative) == 0, f"Found {len(negative)} negative index values"
    
    def test_no_null_index(self, spy_vix_like_df):
        """Test that successful rows have non-null index."""
        successful = spy_vix_like_df.filter(spy_vix_like_df["success"] == True)
        null_count = successful["index"].is_null().sum()
        assert null_count == 0, f"Found {null_count} null index values"
    
    def test_index_range_reasonable(self, spy_vix_like_df):
        """Test that index values are in reasonable range."""
        successful = spy_vix_like_df.filter(spy_vix_like_df["success"] == True)
        
        min_idx = successful["index"].min()
        max_idx = successful["index"].max()
        
        assert min_idx > 5, f"Min index {min_idx} too low"
        assert max_idx < 150, f"Max index {max_idx} too high"
    
    def test_covid_spike_captured(self, spy_vix_like_df):
        """Test that COVID spike is captured."""
        from datetime import date
        
        # March 2020
        march_2020 = spy_vix_like_df.filter(
            (spy_vix_like_df["quote_date"] >= date(2020, 3, 1)) &
            (spy_vix_like_df["quote_date"] <= date(2020, 3, 31)) &
            (spy_vix_like_df["success"] == True)
        )
        
        max_march = march_2020["index"].max()
        assert max_march > 60, f"March 2020 max {max_march} should be > 60"


class TestFREDComparison:
    """Tests for FRED VIXCLS comparison."""
    
    def test_fred_data_exists(self, fred_vixcls_df):
        """Test that FRED data exists."""
        assert len(fred_vixcls_df) > 0
    
    def test_fred_data_sufficient(self, fred_vixcls_df):
        """Test that FRED data has enough observations."""
        assert len(fred_vixcls_df) >= 700
    
    def test_joined_data_exists(self, joined_comparison_df):
        """Test that joined comparison data exists."""
        assert len(joined_comparison_df) > 0
    
    def _get_ticker_col(self, df):
        """Get the ticker VIX column from dataframe."""
        for col in df.columns:
            if col.endswith("_vix") and col != "vixcls":
                return col
        return "spy_vix"  # Fallback
    
    def test_correlation_high(self, joined_comparison_df):
        """Test that correlation is high (> 0.90)."""
        ticker_col = self._get_ticker_col(joined_comparison_df)
        ticker_vals = joined_comparison_df[ticker_col].to_numpy()
        vix = joined_comparison_df["vixcls"].to_numpy()
        
        correlation = np.corrcoef(ticker_vals, vix)[0, 1]
        assert correlation > 0.90, f"Correlation {correlation} too low"
    
    def test_rmse_reasonable(self, joined_comparison_df):
        """Test that RMSE is reasonable (< 5)."""
        ticker_col = self._get_ticker_col(joined_comparison_df)
        residuals = (
            joined_comparison_df[ticker_col] - 
            joined_comparison_df["vixcls"]
        ).to_numpy()
        
        rmse = np.sqrt(np.mean(residuals ** 2))
        assert rmse < 5, f"RMSE {rmse} too high"
    
    def test_mean_bias_small(self, joined_comparison_df):
        """Test that mean bias is small (< 2)."""
        ticker_col = self._get_ticker_col(joined_comparison_df)
        residuals = (
            joined_comparison_df[ticker_col] - 
            joined_comparison_df["vixcls"]
        ).to_numpy()
        
        bias = np.mean(residuals)
        assert abs(bias) < 2, f"Mean bias {bias} too large"

