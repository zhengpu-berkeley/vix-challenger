#!/usr/bin/env python3
"""Inspect schema of SPY options CSV to determine column mappings.

Outputs:
- Console: column names, dtypes, sample values, cardinalities
- reports/schema_notes.md: documentation of findings
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from vix_challenger.config import SPY_RAW_CSV, SCHEMA_NOTES_PATH, ensure_directories


def inspect_schema(csv_path: Path) -> dict:
    """Inspect CSV schema and return findings as a dict."""
    print(f"Inspecting: {csv_path}")
    print(f"File size: {csv_path.stat().st_size / 1e9:.2f} GB")
    print()
    
    # Scan CSV lazily to get schema without loading all data
    print("Scanning CSV schema...")
    lf = pl.scan_csv(csv_path)
    schema = lf.collect_schema()
    
    print("\n" + "=" * 60)
    print("COLUMN SCHEMA")
    print("=" * 60)
    for col_name, dtype in schema.items():
        print(f"  {col_name:<25} {dtype}")
    
    # Sample first few rows
    print("\n" + "=" * 60)
    print("SAMPLE DATA (first 5 rows)")
    print("=" * 60)
    sample_df = lf.head(5).collect()
    print(sample_df)
    
    # Get basic stats on a sample
    print("\n" + "=" * 60)
    print("CARDINALITY & VALUE RANGES (from sample of 100k rows)")
    print("=" * 60)
    
    sample_100k = lf.head(100_000).collect()
    
    findings = {
        "columns": list(schema.keys()),
        "dtypes": {k: str(v) for k, v in schema.items()},
        "row_count_sample": len(sample_100k),
    }
    
    # Check for date-like columns
    date_cols = []
    for col in sample_100k.columns:
        sample_vals = sample_100k[col].head(5).to_list()
        print(f"\n{col}:")
        print(f"  Unique count (in 100k sample): {sample_100k[col].n_unique()}")
        print(f"  Sample values: {sample_vals[:3]}")
        
        # Check if it looks like a date
        if any(kw in col.lower() for kw in ["date", "expir", "time"]):
            date_cols.append(col)
            print(f"  [Likely DATE column]")
    
    findings["date_columns"] = date_cols
    
    # Check for call/put indicator
    cp_candidates = [c for c in sample_100k.columns if any(kw in c.lower() for kw in ["type", "cp", "call", "put", "option"])]
    if cp_candidates:
        print(f"\nCall/Put indicator candidates: {cp_candidates}")
        for col in cp_candidates:
            print(f"  {col} unique values: {sample_100k[col].unique().to_list()[:10]}")
    findings["cp_candidates"] = cp_candidates
    
    # Check strike column
    strike_candidates = [c for c in sample_100k.columns if "strike" in c.lower()]
    if strike_candidates:
        print(f"\nStrike candidates: {strike_candidates}")
        for col in strike_candidates:
            vals = sample_100k[col]
            print(f"  {col} min={vals.min()}, max={vals.max()}, sample={vals.head(3).to_list()}")
    findings["strike_candidates"] = strike_candidates
    
    # Check bid/ask
    bid_candidates = [c for c in sample_100k.columns if "bid" in c.lower()]
    ask_candidates = [c for c in sample_100k.columns if "ask" in c.lower()]
    print(f"\nBid columns: {bid_candidates}")
    print(f"Ask columns: {ask_candidates}")
    findings["bid_columns"] = bid_candidates
    findings["ask_columns"] = ask_candidates
    
    # Check for underlying price
    underlying_candidates = [c for c in sample_100k.columns if any(kw in c.lower() for kw in ["underlying", "spot", "stock", "close"])]
    print(f"\nUnderlying price candidates: {underlying_candidates}")
    findings["underlying_candidates"] = underlying_candidates
    
    # Count unique dates and expiries
    print("\n" + "=" * 60)
    print("DATE CARDINALITIES (full scan)")
    print("=" * 60)
    
    # Do a lazy aggregation for unique dates
    if date_cols:
        for col in date_cols[:2]:  # First two date columns
            unique_count = lf.select(pl.col(col).n_unique()).collect().item()
            print(f"  {col}: {unique_count} unique values")
            findings[f"{col}_unique"] = unique_count
    
    # Get total row count
    print("\n" + "=" * 60)
    print("TOTAL ROWS")
    print("=" * 60)
    total_rows = lf.select(pl.len()).collect().item()
    print(f"  Total rows: {total_rows:,}")
    findings["total_rows"] = total_rows
    
    return findings, sample_df


def generate_schema_notes(findings: dict, sample_df: pl.DataFrame, output_path: Path):
    """Generate schema_notes.md from findings."""
    
    lines = [
        "# SPY Options Data Schema Notes",
        "",
        "## Source",
        "- Dataset: `kylegraupe/spy-daily-eod-options-quotes-2020-2022`",
        "- File: `spy_2020_2022.csv`",
        f"- Total rows: {findings.get('total_rows', 'N/A'):,}",
        "",
        "## Columns",
        "",
        "| Column | Dtype | Notes |",
        "|--------|-------|-------|",
    ]
    
    for col, dtype in findings["dtypes"].items():
        notes = ""
        if col in findings.get("date_columns", []):
            notes = "Date column"
        elif col in findings.get("cp_candidates", []):
            notes = "Call/Put indicator"
        elif col in findings.get("strike_candidates", []):
            notes = "Strike price"
        elif col in findings.get("bid_columns", []):
            notes = "Bid price"
        elif col in findings.get("ask_columns", []):
            notes = "Ask price"
        elif col in findings.get("underlying_candidates", []):
            notes = "Underlying/spot price"
        lines.append(f"| `{col}` | {dtype} | {notes} |")
    
    lines.extend([
        "",
        "## Sample Data",
        "",
        "```",
        str(sample_df),
        "```",
        "",
        "## Column Mapping (for spy_csv.py)",
        "",
        "Based on inspection, the following mappings are recommended:",
        "",
        "```python",
        "# Column name mappings to logical names",
        "COLUMN_MAP = {",
    ])
    
    # Generate mapping suggestions
    mappings = {}
    for col in findings["columns"]:
        col_lower = col.lower()
        if "quote" in col_lower and "date" in col_lower:
            mappings["quote_date"] = col
        elif "expir" in col_lower:
            mappings["expiration"] = col
        elif "strike" in col_lower:
            mappings["strike"] = col
        elif col_lower in ["c_p", "cp", "type", "option_type"]:
            mappings["cp"] = col
        elif col_lower == "bid" or (col_lower.startswith("bid") and "size" not in col_lower):
            mappings["bid"] = col
        elif col_lower == "ask" or (col_lower.startswith("ask") and "size" not in col_lower):
            mappings["ask"] = col
        elif any(kw in col_lower for kw in ["underlying", "spot"]) and "bid" not in col_lower and "ask" not in col_lower:
            mappings["underlying_price"] = col
    
    for logical, actual in mappings.items():
        lines.append(f'    "{logical}": "{actual}",')
    
    lines.extend([
        "}",
        "```",
        "",
        "## Open Questions Resolved",
        "",
        "- Strike units: Check sample values above",
        "- Bid/ask: EOD quotes",
        "- Underlying price: Check if present",
        "",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nSchema notes written to: {output_path}")
    
    return mappings


def main():
    parser = argparse.ArgumentParser(description="Inspect SPY options CSV schema")
    parser.add_argument("--csv", type=Path, default=SPY_RAW_CSV, help="Path to CSV file")
    args = parser.parse_args()
    
    ensure_directories()
    
    if not args.csv.exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        print("Run scripts/download_data.py first.")
        sys.exit(1)
    
    findings, sample_df = inspect_schema(args.csv)
    mappings = generate_schema_notes(findings, sample_df, SCHEMA_NOTES_PATH)
    
    print("\n" + "=" * 60)
    print("RECOMMENDED COLUMN MAPPINGS")
    print("=" * 60)
    for logical, actual in mappings.items():
        print(f"  {logical:<20} <- {actual}")


if __name__ == "__main__":
    main()

