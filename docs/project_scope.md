# Project: vix-challenger

Compute a **VIX-style (model-free) 30-day implied volatility index** from **equity/ETF options** (starting with SPY 2020–2022 EOD quotes), then compare/overlay against **FRED VIXCLS** (SPX-based) and other reference series.

> Note: “VIX-like from SPY” will not equal VIX exactly (VIX is computed from SPX index options), but should be strongly correlated and should match stress regimes.

---

## 1) Goals & Non-Goals

### 1.1 Goals
- Implement Cboe-style VIX methodology for a single underlying:
  - per-expiry model-free variance from an OTM option strip
  - constant 30-day maturity interpolation
- Produce daily time series:
  - `SPYVIX_like(t)` for 2020–2022
- Validate / analyze vs:
  - FRED `VIXCLS` (daily close)
- Build intuition:
  - strike-by-strike contribution plots
  - diagnostics on “bad days” (missing strikes, wide spreads, early cutoff)

### 1.2 Non-Goals (initial POC)
- Intraday / minute-level VIX (EOD only)
- Perfect replication of official VIX level (SPY vs SPX mismatch)
- Dividend yield modeling (optional later)
- American exercise modeling (VIX methodology assumes European-ish integration; we follow the published math)

---

## 2) Deliverables

### 2.1 Data artifacts
- `data/processed/spy_options_by_date/quote_date=YYYY-MM-DD/*.parquet`
- `data/processed/spy_vix_like.parquet` (daily index + metadata)
- `data/processed/fred_vixcls.parquet`
- `data/processed/joined_spy_vs_vix.parquet`
- `data/processed/diagnostics.parquet` (per-date QC stats)

### 2.2 Plots
- Overlay: `SPYVIX_like` vs `VIXCLS`
- Scatter: `SPYVIX_like` vs `VIXCLS`
- Residuals over time: `SPYVIX_like - VIXCLS`
- “Contribution curve” plot: strike contribution to variance on selected dates
- Rolling metrics:
  - rolling corr (60d)
  - rolling beta / regression of DIY vs VIXCLS

### 2.3 Report
- `reports/poc_summary.md`:
  - headline stats (corr, RMSE, bias)
  - known sources of discrepancy (SPX vs SPY; EOD convention; strike truncation; cutoff rules)
  - validation on stress windows (e.g., Mar 2020)

---

## 3) Repo Layout

vix-challenger/
├── README.md
├── pyproject.toml
├── .python-version
├── .env.example
├── data/
│   ├── raw/
│   │   └── spy_2020_2022.csv
│   └── processed/
│       ├── spy_options_by_date/
│       ├── spy_vix_like.parquet
│       ├── fred_vixcls.parquet
│       ├── joined_spy_vs_vix.parquet
│       └── diagnostics.parquet
├── src/
│   └── vix_challenger/
│       ├── init.py
│       ├── config.py
│       ├── io/
│       │   ├── spy_csv.py
│       │   └── fred.py
│       ├── vix/
│       │   ├── methodology.md
│       │   ├── selection.py
│       │   ├── parity.py
│       │   ├── strip.py
│       │   ├── variance.py
│       │   ├── interpolate.py
│       │   └── qc.py
│       ├── pipelines/
│       │   ├── convert_csv_to_parquet.py
│       │   ├── compute_spy_vix_like.py
│       │   └── compare_to_fred.py
│       └── viz/
│           ├── overlay.py
│           ├── residuals.py
│           ├── scatter.py
│           └── contributions.py
├── scripts/
│   ├── 00_inspect_schema.py
│   ├── 01_convert_csv.py
│   ├── 02_compute_vix_like.py
│   └── 03_compare_to_fred.py
├── notebooks/
│   └── 01_poc_spy_one_week.ipynb
└── reports/
└── poc_summary.md

`.gitignore` (minimum):

data/raw/*
data/processed/*
.env
*.parquet
*.duckdb

---

## 4) Environment & Dependencies

### 4.1 Python
- Python 3.11 recommended

### 4.2 Core dependencies
- `polars` (scan + partition 1.28GB CSV efficiently)
- `pyarrow` (parquet)
- `numpy`
- `pandas` (light use)
- `matplotlib` (plots)
- `requests` (FRED download)
- `tqdm`

### 4.3 Optional
- `duckdb` (ad hoc queries on parquet)
- `numba` (speed for per-day loops if needed)

---

## 5) Data: Required Columns & Mapping

### 5.1 Required columns (logical)
For each option record:
- `quote_date` (date)
- `expiration` (date)
- `strike` (float)
- `cp` (C/P)
- `bid` (float)
- `ask` (float)

Strongly preferred:
- `underlying_price` (float, EOD close)

Optional:
- `bid_size`, `ask_size`, `volume`, `open_interest`

### 5.2 First task: schema inspection
Run `scripts/00_inspect_schema.py` to:
- print column names + dtypes
- check cardinalities (unique expiries/day; strikes/day)
- confirm units (strike scaling, e.g. 450 vs 450000)

Output:
- `reports/schema_notes.md`
- `src/vix_challenger/io/spy_csv.py` mapping constants

---

## 6) Methodology (VIX-style, per Cboe math)

### 6.1 High-level steps per trade date `t`
1. Choose 2 expirations bracketing 30D:
   - near: max expiry with `DTE <= 30`
   - next: min expiry with `DTE > 30`
2. For each expiry `T`:
   - build per-strike call/put midquotes
   - compute forward `F` via put-call parity at `K*`
   - set `K0 = max strike <= F`
   - construct OTM strip `Q(K)` from midquotes
   - apply “two consecutive zero-bid” cutoff per wing
   - compute model-free variance `sigma2(T)`
3. Interpolate `sigma2(T1)` and `sigma2(T2)` to constant 30D variance
4. Output: `index = 100 * sqrt(var_30d)`

### 6.2 Key definitions
- Midquote: `mid = (bid + ask)/2`
- Time to expiry:
  - POC: `T = DTE_days / 365`
  - Later: minutes convention

### 6.3 Forward price via parity
For strikes where both C and P exist:
- `diff(K) = |C_mid(K) - P_mid(K)|`
- choose `K* = argmin diff(K)`
- `F = K* + exp(rT) * (C_mid(K*) - P_mid(K*))`

Rates:
- POC: `r=0`
- Later: daily Treasury interpolation

### 6.4 Strike spacing
Given sorted strikes `K_i` used in strip:
- interior: `ΔK_i = (K_{i+1} - K_{i-1}) / 2`
- ends: one-sided

### 6.5 Variance formula (per expiry)
Let `Q(K_i)` be OTM option midquote (put for K<K0, call for K>K0, avg at K0):

sigma2(T) =
(2/T) * Σ [ (ΔK_i / K_i^2) * exp(rT) * Q(K_i) ]
	•	(1/T) * (F/K0 - 1)^2

### 6.6 Constant 30D interpolation
Compute `var_30d` as weighted combination of the two terms (weights depend on how close 30D is to each expiry). Use the standard VIX interpolation.

---

## 7) Implementation Plan (Granular Tasks)

### 7.1 Phase A — IO & Partitioning
- [ ] A1: `00_inspect_schema.py` (determine column mapping)
- [ ] A2: `01_convert_csv.py`
  - scan CSV lazily
  - parse dates
  - compute `mid`
  - write parquet partitioned by `quote_date`

Acceptance:
- `data/processed/spy_options_by_date/quote_date=.../` exists
- Reading one day takes < 1s

### 7.2 Phase B — Core VIX Computation (one day, one expiry)
- [ ] B1: Load one date partition
- [ ] B2: Pick one expiry and compute:
  - `calls(K)`, `puts(K)` midquotes
  - `F`, `K0`
  - `Q(K)` strip + cutoff
  - `sigma2(T)`
- [ ] B3: Unit test on a single date:
  - sanity: `sigma2 > 0`
  - reasonable magnitude

Acceptance:
- For a chosen date, printed diagnostics make sense:
  - K0 close to forward
  - includes strikes on both wings
  - cutoff triggers after reasonable strikes

### 7.3 Phase C — Daily Index Computation (2020–2022)
- [ ] C1: Implement near/next expiry selection per date
- [ ] C2: Compute `sigma2_1`, `sigma2_2`
- [ ] C3: Interpolate to 30D index
- [ ] C4: Save results to parquet

Acceptance:
- `spy_vix_like.parquet` has ~500+ rows (trading days)
- no more than a small fraction of skipped days (explain why)

### 7.4 Phase D — FRED Download & Comparison
- [ ] D1: Download `VIXCLS` into parquet
- [ ] D2: Join and compute metrics
- [ ] D3: Plots + diagnostics

Acceptance:
- overlay plot exists
- correlation and residual analysis computed

---

## 8) Diagnostics & QC (must-have)

Per trade date output:
- chosen expiries (`near_exp`, `next_exp`) and DTE
- `F1`, `K0_1`, strike counts used, cutoff strikes
- `F2`, `K0_2`, strike counts used
- `sigma2_1`, `sigma2_2`, `var_30d`, `index`
- data quality stats:
  - % missing C/P pairs at strikes
  - median bid-ask spread
  - min/max strike used relative to spot

Skips:
- record reason codes (e.g., `NO_BRACKETING_EXPIRY`, `TOO_FEW_STRIKES`, `NEGATIVE_VAR`, `MISSING_QUOTES`)

---

## 9) Validation Strategy

### 9.1 Baselines
- Compare DIY index vs FRED `VIXCLS`:
  - corr, RMSE, mean bias
  - rolling corr
  - regression slope

### 9.2 Stress-window spot checks
- Mar 2020: DIY should spike strongly, similar pattern to VIXCLS
- late 2021: calmer regime
- 2022: elevated vol

### 9.3 Interpret discrepancies
Expected sources:
- SPY vs SPX options
- EOD timestamp mismatch
- strike truncation in dataset
- midquote vs official quote conventions
- rates approximation

---

## 10) CLI Usage (target)

1) Inspect schema:

uv run python scripts/00_inspect_schema.py –csv data/raw/spy_2020_2022.csv

2) Convert to parquet:

uv run python scripts/01_convert_csv.py –csv data/raw/spy_2020_2022.csv –out data/processed/spy_options_by_date

3) Compute VIX-like:

uv run python scripts/02_compute_vix_like.py –parquet data/processed/spy_options_by_date –out data/processed/spy_vix_like.parquet

4) Compare to FRED:

uv run python scripts/03_compare_to_fred.py –spy data/processed/spy_vix_like.parquet –out_dir reports/

---

## 11) Open Questions (to resolve during schema inspection)
- Are strikes quoted as dollars or scaled integers?
- Are bid/ask truly EOD NBBO, or vendor mid/mark?
- Does the dataset include `underlying_price`? If not, we can still compute using `F` from parity.
- Are there known missing days / early market close days?

---

## 12) Success Criteria (POC)
- End-to-end pipeline runs locally without manual cleanup
- Produces a daily series for 2020–2022 with low skip rate
- DIY series tracks VIXCLS with high correlation and sensible stress spikes
- Diagnostics explain the biggest residuals