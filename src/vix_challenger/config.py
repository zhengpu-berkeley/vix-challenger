"""Project configuration and paths.

Supports multi-ticker configuration with a registry system.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================

# Project root (vix-challenger/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
SCHEMA_NOTES_PATH = REPORTS_DIR / "schema_notes.md"

# FRED data (shared across all tickers)
FRED_VIXCLS_PARQUET = PROCESSED_DATA_DIR / "fred_vixcls.parquet"


# =============================================================================
# Ticker Configuration
# =============================================================================

@dataclass
class TickerConfig:
    """Configuration for a single ticker/symbol."""
    
    # Ticker symbol (e.g., "SPY", "AAPL")
    ticker: str
    
    # Kaggle dataset ID
    kaggle_dataset: str
    
    # Raw CSV filename (after download)
    raw_csv_name: str
    
    # Date range (for reference)
    start_year: int
    end_year: int
    
    # Optional: reference VIX series for comparison
    reference_vix: Optional[str] = None
    
    @property
    def raw_csv_path(self) -> Path:
        """Path to raw CSV file."""
        return RAW_DATA_DIR / self.raw_csv_name
    
    @property
    def options_by_date_dir(self) -> Path:
        """Path to partitioned parquet directory."""
        return PROCESSED_DATA_DIR / f"{self.ticker.lower()}_options_by_date"
    
    @property
    def vix_like_parquet(self) -> Path:
        """Path to VIX-like results parquet."""
        return PROCESSED_DATA_DIR / f"{self.ticker.lower()}_vix_like.parquet"
    
    @property
    def diagnostics_parquet(self) -> Path:
        """Path to diagnostics parquet."""
        return PROCESSED_DATA_DIR / f"{self.ticker.lower()}_diagnostics.parquet"
    
    @property
    def joined_comparison_parquet(self) -> Path:
        """Path to joined comparison parquet."""
        return PROCESSED_DATA_DIR / f"joined_{self.ticker.lower()}_vs_vix.parquet"
    
    @property
    def schema_notes_path(self) -> Path:
        """Path to schema notes for this ticker."""
        return REPORTS_DIR / f"schema_notes_{self.ticker.lower()}.md"
    
    @property
    def figures_dir(self) -> Path:
        """Path to figures directory for this ticker."""
        return REPORTS_DIR / "figures" / self.ticker.lower()


# =============================================================================
# Ticker Registry
# =============================================================================

TICKER_REGISTRY: dict[str, TickerConfig] = {
    "SPY": TickerConfig(
        ticker="SPY",
        kaggle_dataset="kylegraupe/spy-daily-eod-options-quotes-2020-2022",
        raw_csv_name="spy_2020_2022.csv",
        start_year=2020,
        end_year=2022,
        reference_vix="VIXCLS",
    ),
    "AAPL": TickerConfig(
        ticker="AAPL",
        kaggle_dataset="kylegraupe/aapl-options-data-2016-2020",
        raw_csv_name="aapl_2016_2020.csv",
        start_year=2016,
        end_year=2020,
        reference_vix=None,  # No direct VIX equivalent
    ),
    "TSLA": TickerConfig(
        ticker="TSLA",
        kaggle_dataset="kylegraupe/tsla-daily-eod-options-quotes-2019-2022",
        raw_csv_name="tsla_2019_2022.csv",
        start_year=2019,
        end_year=2022,
        reference_vix=None,
    ),
    "NVDA": TickerConfig(
        ticker="NVDA",
        kaggle_dataset="kylegraupe/nvda-daily-option-chains-q1-2020-to-q4-2022",
        raw_csv_name="nvda_2020_2022.csv",
        start_year=2020,
        end_year=2022,
        reference_vix=None,
    ),
}


def get_ticker_config(ticker: str) -> TickerConfig:
    """Get configuration for a ticker.
    
    Args:
        ticker: Ticker symbol (case-insensitive)
        
    Returns:
        TickerConfig for the requested ticker
        
    Raises:
        ValueError: If ticker not found in registry
    """
    ticker_upper = ticker.upper()
    if ticker_upper not in TICKER_REGISTRY:
        available = ", ".join(TICKER_REGISTRY.keys())
        raise ValueError(f"Unknown ticker: {ticker}. Available: {available}")
    return TICKER_REGISTRY[ticker_upper]


def list_tickers() -> list[str]:
    """List all available tickers."""
    return list(TICKER_REGISTRY.keys())


# =============================================================================
# Legacy SPY-specific paths (for backwards compatibility)
# =============================================================================

# These map to the SPY ticker config
_spy_config = TICKER_REGISTRY["SPY"]
SPY_RAW_CSV = _spy_config.raw_csv_path
SPY_OPTIONS_BY_DATE = _spy_config.options_by_date_dir
SPY_VIX_LIKE_PARQUET = _spy_config.vix_like_parquet
DIAGNOSTICS_PARQUET = _spy_config.diagnostics_parquet
JOINED_SPY_VS_VIX_PARQUET = _spy_config.joined_comparison_parquet


# =============================================================================
# Directory Management
# =============================================================================

def ensure_directories(ticker: Optional[str] = None):
    """Create all necessary directories.
    
    Args:
        ticker: If provided, also create ticker-specific directories
    """
    # Base directories
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Ticker-specific directories
    if ticker:
        config = get_ticker_config(ticker)
        config.options_by_date_dir.mkdir(parents=True, exist_ok=True)
        config.figures_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Create for all registered tickers
        for config in TICKER_REGISTRY.values():
            config.options_by_date_dir.mkdir(parents=True, exist_ok=True)


def ensure_ticker_directories(ticker: str):
    """Ensure directories exist for a specific ticker."""
    ensure_directories(ticker)
