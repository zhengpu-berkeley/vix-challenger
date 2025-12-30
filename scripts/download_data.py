#!/usr/bin/env python3
"""Download SPY options data from Kaggle."""

import os
import shutil
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vix_challenger.config import RAW_DATA_DIR, SPY_RAW_CSV, ensure_directories


def download_spy_options():
    """Download SPY options data from Kaggle using kagglehub."""
    import kagglehub
    
    ensure_directories()
    
    print("Downloading SPY options data from Kaggle...")
    print("Dataset: kylegraupe/spy-daily-eod-options-quotes-2020-2022")
    
    # Download using kagglehub
    path = kagglehub.dataset_download("kylegraupe/spy-daily-eod-options-quotes-2020-2022")
    print(f"Downloaded to: {path}")
    
    # Find the CSV file(s) in the downloaded directory
    download_path = Path(path)
    csv_files = list(download_path.glob("*.csv"))
    
    if not csv_files:
        # Check subdirectories
        csv_files = list(download_path.rglob("*.csv"))
    
    if not csv_files:
        print("ERROR: No CSV files found in downloaded data!")
        print(f"Contents of {download_path}:")
        for item in download_path.rglob("*"):
            print(f"  {item}")
        return False
    
    print(f"Found CSV files: {csv_files}")
    
    # Copy/link the main CSV to our raw data directory
    # The dataset typically has one large CSV
    main_csv = max(csv_files, key=lambda f: f.stat().st_size)
    print(f"Main CSV file: {main_csv} ({main_csv.stat().st_size / 1e6:.1f} MB)")
    
    # Create symlink or copy
    if SPY_RAW_CSV.exists():
        SPY_RAW_CSV.unlink()
    
    try:
        # Try symlink first (saves disk space)
        SPY_RAW_CSV.symlink_to(main_csv)
        print(f"Created symlink: {SPY_RAW_CSV} -> {main_csv}")
    except OSError:
        # Fall back to copy if symlink fails
        print(f"Copying {main_csv} to {SPY_RAW_CSV}...")
        shutil.copy2(main_csv, SPY_RAW_CSV)
        print(f"Copied to: {SPY_RAW_CSV}")
    
    print("\nDownload complete!")
    print(f"Data available at: {SPY_RAW_CSV}")
    return True


if __name__ == "__main__":
    success = download_spy_options()
    sys.exit(0 if success else 1)

