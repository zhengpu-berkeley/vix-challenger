"""Tests for IO modules."""

import pytest
from datetime import date

import polars as pl


class TestOptionsDataLoading:
    """Tests for options data loading."""
    
    def test_sample_day_loads(self, sample_day_df):
        """Test that sample day data loads correctly."""
        assert len(sample_day_df) > 0
        
    def test_sample_day_has_required_columns(self, sample_day_df):
        """Test that required columns exist."""
        required_cols = [
            "quote_date", "expiration", "dte", "underlying_price",
            "strike", "c_bid", "c_ask", "p_bid", "p_ask", "c_mid", "p_mid"
        ]
        for col in required_cols:
            assert col in sample_day_df.columns, f"Missing column: {col}"
    
    def test_sample_day_has_valid_strikes(self, sample_day_df):
        """Test that strikes are positive and reasonable."""
        strikes = sample_day_df["strike"].drop_nulls()
        assert strikes.min() > 0
        assert strikes.max() < 10000  # Reasonable upper bound
        
    def test_sample_day_has_multiple_expirations(self, sample_day_df):
        """Test that multiple expirations are available."""
        n_exp = sample_day_df["expiration"].n_unique()
        assert n_exp >= 5, f"Expected >= 5 expirations, got {n_exp}"
    
    def test_midquotes_computed_correctly(self, sample_day_df):
        """Test that midquotes are average of bid and ask."""
        # Sample a few rows
        sample = sample_day_df.head(100)
        
        expected_c_mid = (sample["c_bid"] + sample["c_ask"]) / 2
        expected_p_mid = (sample["p_bid"] + sample["p_ask"]) / 2
        
        # Check they match (within floating point tolerance)
        c_diff = (sample["c_mid"] - expected_c_mid).abs().max()
        p_diff = (sample["p_mid"] - expected_p_mid).abs().max()
        
        assert c_diff < 0.01, f"Call midquote mismatch: {c_diff}"
        assert p_diff < 0.01, f"Put midquote mismatch: {p_diff}"


class TestDataPartitioning:
    """Tests for partitioned data structure."""
    
    def test_partitions_exist(self, spy_options_dir):
        """Test that partitions exist."""
        partitions = list(spy_options_dir.glob("quote_date=*"))
        assert len(partitions) >= 700, f"Expected >= 700 partitions, got {len(partitions)}"
    
    def test_each_partition_has_data(self, spy_options_dir):
        """Test that each partition has a data file."""
        partitions = list(spy_options_dir.glob("quote_date=*"))[:10]  # Check first 10
        for partition in partitions:
            data_file = partition / "data.parquet"
            assert data_file.exists(), f"Missing data file in {partition}"

