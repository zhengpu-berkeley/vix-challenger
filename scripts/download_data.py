#!/usr/bin/env python3
"""Download options data from Kaggle for a specified ticker.

Usage:
    uv run python scripts/download_data.py              # Downloads SPY (default)
    uv run python scripts/download_data.py --ticker SPY
    uv run python scripts/download_data.py --ticker AAPL
    uv run python scripts/download_data.py --ticker TSLA
    uv run python scripts/download_data.py --ticker NVDA
    uv run python scripts/download_data.py --all        # Downloads all tickers
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vix_challenger.config import (
    get_ticker_config,
    list_tickers,
    ensure_ticker_directories,
)


def download_ticker_data(ticker: str) -> bool:
    """Download options data for a ticker from Kaggle.
    
    Args:
        ticker: Ticker symbol (e.g., "SPY", "AAPL")
        
    Returns:
        True if successful, False otherwise
    """
    import kagglehub
    
    config = get_ticker_config(ticker)
    ensure_ticker_directories(ticker)
    
    print(f"\n{'='*60}")
    print(f"Downloading {ticker} data from Kaggle")
    print(f"{'='*60}")
    print(f"Dataset: {config.kaggle_dataset}")
    
    # Download using kagglehub
    try:
        path = kagglehub.dataset_download(config.kaggle_dataset)
        print(f"Downloaded to: {path}")
    except Exception as e:
        print(f"ERROR downloading {ticker}: {e}")
        return False
    
    # Find the CSV file(s) in the downloaded directory
    download_path = Path(path)
    csv_files = list(download_path.glob("*.csv"))
    
    if not csv_files:
        csv_files = list(download_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"ERROR: No CSV files found in downloaded data!")
        print(f"Contents of {download_path}:")
        for item in download_path.rglob("*"):
            print(f"  {item}")
        return False
    
    print(f"Found CSV files: {csv_files}")
    
    # Copy/link the main CSV to our raw data directory
    main_csv = max(csv_files, key=lambda f: f.stat().st_size)
    print(f"Main CSV file: {main_csv} ({main_csv.stat().st_size / 1e6:.1f} MB)")
    
    target_path = config.raw_csv_path
    
    if target_path.exists():
        target_path.unlink()
    
    try:
        target_path.symlink_to(main_csv)
        print(f"Created symlink: {target_path} -> {main_csv}")
    except OSError:
        print(f"Copying {main_csv} to {target_path}...")
        shutil.copy2(main_csv, target_path)
        print(f"Copied to: {target_path}")
    
    print(f"\n✓ {ticker} download complete!")
    print(f"Data available at: {target_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download options data from Kaggle")
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help=f"Ticker symbol to download. Available: {', '.join(list_tickers())}"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download data for all registered tickers"
    )
    args = parser.parse_args()
    
    if args.all:
        tickers = list_tickers()
        print(f"Downloading all tickers: {tickers}")
        results = {}
        for ticker in tickers:
            results[ticker] = download_ticker_data(ticker)
        
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        for ticker, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {ticker}")
        
        if not all(results.values()):
            sys.exit(1)
    else:
        success = download_ticker_data(args.ticker.upper())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
