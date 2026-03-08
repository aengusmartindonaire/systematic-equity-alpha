# Data Directory

## Raw Data

**File:** `raw/20251109_Blg_Rsk_Factors.par`

This is a Bloomberg risk factor panel dataset covering US equities from **April 1999 to September 2025** (~780,000 rows). The file is not included in this repository due to licensing restrictions.

### Schema

| Column | Type | Description |
|---|---|---|
| `Date` | int (YYYYMM) | Monthly observation date |
| `Ticker` | string | Bloomberg equity ticker |
| **Core Risk Factors** | | |
| `Sz` | float | Size (market cap rank) |
| `Prof` | float | Profitability |
| `Vol` | float | Volatility |
| `Trd Act` | float | Trading Activity |
| `Lev` | float | Leverage |
| `Mom` | float | Momentum |
| `Val` | float | Value |
| `Gr` | float | Growth |
| `Dvd Yld` | float | Dividend Yield |
| `Earn Var` | float | Earnings Variability |
| **Extra Features** | | |
| `P/S`, `BEst P/S BF12M` | float | Price-to-Sales (trailing and forward) |
| `P/B`, `BEst P/B BF12M` | float | Price-to-Book (trailing and forward) |
| `P/E`, `BEst P/E BF12M` | float | Price-to-Earnings (trailing and forward) |
| `ROE LF` | float | Return on Equity (last fiscal) |
| `Beta:Y-1` | float | 1-year trailing beta |
| `Total Return:Y-1` | float | 1-year trailing total return |
| `Number of Employees:Y` | float | Headcount (annual) |
| `Market Cap` | float | Market capitalization |
| **Classification** | | |
| `GICS_Sector_Name` | string | GICS Sector |
| `GICS_Industry_Name` | string | GICS Industry |
| `GICS_SubInd_Name` | string | GICS Sub-Industry |
| **Target** | | |
| `FwdRet` | float | Forward 1-month return |

### Index Structure

After loading, the data is set to a `MultiIndex(Date, Ticker)` sorted chronologically.

## Processed Data

| File | Created By | Description |
|---|---|---|
| `processed/ingested_raw_check.parquet` | Notebook 01 | Raw data after schema validation |
| `processed/final_model_data.parquet` | Notebook 02 | Feature-engineered dataset ready for modeling |
