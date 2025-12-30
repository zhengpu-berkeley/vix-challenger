"""Pytest configuration and fixtures."""

import pytest
from datetime import date
from pathlib import Path

import polars as pl

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


@pytest.fixture
def spy_options_dir():
    """Path to SPY partitioned options data."""
    return PROCESSED_DIR / "spy_options_by_date"


@pytest.fixture
def sample_date():
    """A sample date known to have good data."""
    return date(2021, 7, 9)


@pytest.fixture
def covid_peak_date():
    """COVID volatility peak date."""
    return date(2020, 3, 16)


@pytest.fixture
def sample_day_df(spy_options_dir, sample_date):
    """Load a sample day's options data."""
    date_str = sample_date.strftime("%Y-%m-%d")
    path = spy_options_dir / f"quote_date={date_str}" / "data.parquet"
    if not path.exists():
        pytest.skip(f"Sample data not found: {path}")
    return pl.read_parquet(path)


@pytest.fixture
def covid_day_df(spy_options_dir, covid_peak_date):
    """Load COVID peak day options data."""
    date_str = covid_peak_date.strftime("%Y-%m-%d")
    path = spy_options_dir / f"quote_date={date_str}" / "data.parquet"
    if not path.exists():
        pytest.skip(f"COVID data not found: {path}")
    return pl.read_parquet(path)


@pytest.fixture
def spy_vix_like_df():
    """Load computed SPY VIX-like results."""
    path = PROCESSED_DIR / "spy_vix_like.parquet"
    if not path.exists():
        pytest.skip(f"SPY VIX results not found: {path}")
    return pl.read_parquet(path)


@pytest.fixture
def fred_vixcls_df():
    """Load FRED VIXCLS data."""
    path = PROCESSED_DIR / "fred_vixcls.parquet"
    if not path.exists():
        pytest.skip(f"FRED data not found: {path}")
    return pl.read_parquet(path)


@pytest.fixture
def joined_comparison_df():
    """Load joined SPY vs VIXCLS comparison data."""
    path = PROCESSED_DIR / "joined_spy_vs_vix.parquet"
    if not path.exists():
        pytest.skip(f"Joined data not found: {path}")
    return pl.read_parquet(path)

