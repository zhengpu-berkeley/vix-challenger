"""Project configuration and paths."""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root (vix-challenger/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific data paths
SPY_RAW_CSV = RAW_DATA_DIR / "spy_2020_2022.csv"
SPY_OPTIONS_BY_DATE = PROCESSED_DATA_DIR / "spy_options_by_date"
SPY_VIX_LIKE_PARQUET = PROCESSED_DATA_DIR / "spy_vix_like.parquet"
FRED_VIXCLS_PARQUET = PROCESSED_DATA_DIR / "fred_vixcls.parquet"
JOINED_SPY_VS_VIX_PARQUET = PROCESSED_DATA_DIR / "joined_spy_vs_vix.parquet"
DIAGNOSTICS_PARQUET = PROCESSED_DATA_DIR / "diagnostics.parquet"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
SCHEMA_NOTES_PATH = REPORTS_DIR / "schema_notes.md"

# Ensure directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SPY_OPTIONS_BY_DATE]:
        dir_path.mkdir(parents=True, exist_ok=True)

