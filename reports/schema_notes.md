# SPY Options Data Schema Notes

## Source
- Dataset: `kylegraupe/spy-daily-eod-options-quotes-2020-2022`
- File: `spy_2020_2022.csv`
- Total rows: 3,589,079
- Unique trading days: 758

## Key Findings

### Data Format
- **Wide format**: Each row contains BOTH call and put data for a given strike/expiry
- **Column names**: Have leading spaces (except first column), e.g., ` [QUOTE_DATE]`
- **Quote time**: All quotes are at 16:00 (4 PM EST / market close)
- **Strike units**: Dollar amounts (e.g., 270, 275, 280) - correct for SPY

### Data Quality
- Greeks included for both calls and puts
- Implied volatility available
- Volume and size data present
- DTE (days to expiration) pre-computed

## Columns

| Column | Dtype | Purpose |
|--------|-------|---------|
| `[QUOTE_UNIXTIME]` | Int64 | Quote date as Unix timestamp |
| ` [QUOTE_READTIME]` | String | Quote datetime "YYYY-MM-DD HH:MM" |
| ` [QUOTE_DATE]` | String | Quote date "YYYY-MM-DD" |
| ` [QUOTE_TIME_HOURS]` | String | Quote time in hours (always 16.0) |
| ` [UNDERLYING_LAST]` | String | SPY closing price |
| ` [EXPIRE_DATE]` | String | Expiration date "YYYY-MM-DD" |
| ` [EXPIRE_UNIX]` | String | Expiration as Unix timestamp |
| ` [DTE]` | String | Days to expiration |
| ` [C_DELTA]` | String | Call delta |
| ` [C_GAMMA]` | String | Call gamma |
| ` [C_VEGA]` | String | Call vega |
| ` [C_THETA]` | String | Call theta |
| ` [C_RHO]` | String | Call rho |
| ` [C_IV]` | String | Call implied volatility |
| ` [C_VOLUME]` | String | Call volume |
| ` [C_LAST]` | String | Call last trade price |
| ` [C_SIZE]` | String | Call bid x ask size |
| ` [C_BID]` | String | Call bid price |
| ` [C_ASK]` | String | Call ask price |
| ` [STRIKE]` | String | Strike price in dollars |
| ` [P_BID]` | String | Put bid price |
| ` [P_ASK]` | String | Put ask price |
| ` [P_SIZE]` | String | Put bid x ask size |
| ` [P_LAST]` | String | Put last trade price |
| ` [P_DELTA]` | String | Put delta |
| ` [P_GAMMA]` | String | Put gamma |
| ` [P_VEGA]` | String | Put vega |
| ` [P_THETA]` | String | Put theta |
| ` [P_RHO]` | String | Put rho |
| ` [P_IV]` | String | Put implied volatility |
| ` [P_VOLUME]` | String | Put volume |
| ` [STRIKE_DISTANCE]` | String | Distance from ATM in dollars |
| ` [STRIKE_DISTANCE_PCT]` | String | Distance from ATM as percentage |

## Column Mapping (for spy_csv.py)

```python
# Raw column names (note leading spaces)
RAW_COLS = {
    "quote_unixtime": "[QUOTE_UNIXTIME]",
    "quote_date": " [QUOTE_DATE]",
    "expire_date": " [EXPIRE_DATE]",
    "dte": " [DTE]",
    "underlying_last": " [UNDERLYING_LAST]",
    "strike": " [STRIKE]",
    "c_bid": " [C_BID]",
    "c_ask": " [C_ASK]",
    "p_bid": " [P_BID]",
    "p_ask": " [P_ASK]",
    "c_volume": " [C_VOLUME]",
    "p_volume": " [P_VOLUME]",
}
```

## Open Questions Resolved

1. **Strike units**: Dollar amounts (270, 275, etc.) - no scaling needed
2. **Bid/ask**: EOD quotes at 4 PM EST
3. **Underlying price**: Present as ` [UNDERLYING_LAST]`
4. **Missing days**: Data covers 758 trading days (2020-2022)
5. **Format**: Wide format - one row per strike/expiry, with both C and P data
