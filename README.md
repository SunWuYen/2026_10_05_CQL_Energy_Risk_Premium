# Energy Risk Premium

Research project on risk premia in energy futures (time-series and cross-sectional
signals, including hedging pressure from CFTC/ICE Commitments of Traders data),
portfolio construction and strategy returns.

## Pipeline

| Module | Script | Status | Output |
|---|---|---|---|
| 1. Process data | `src/process_data.py` | done | `data/processed/` |
| 2. Compute returns (risk premia, portfolios, strategies) | `src/energy_risk_premium_cross_sectional_vs_time_series.py`, `src/func_and_class_monthly.py` | in progress | `outputs/` |
| 3. Plots | not started | planned | `outputs/` |

`src/config.py` holds all paths (relative to the project root), so scripts can be run from anywhere.

## Folder layout

```
src/          code
data/raw/     inputs (not tracked): Bloomberg price/expiration workbook, ICE COTHistYYYY.csv
data/processed/  output of module 1 (not tracked)
outputs/      tables and figures (not tracked)
reference/    papers and reference code (not tracked)
docs/         working documents (not tracked)
archive/      old code (not tracked)
```

Data, references and outputs are not in the repository because they are licensed
(Bloomberg, ICE), copyrighted (papers) or regenerable.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data inputs

Place these in `data/raw/`:

- `20260930_energy_futures_price_and_expiration.xlsx`: Bloomberg prices (sheet 0), expiration dates (sheet 1), volume (sheet 2).
- `COTHistYYYY.csv`: ICE Futures Europe COT history from <https://www.ice.com/report/122>. Years are detected automatically from the files present.

CFTC legacy and disaggregated COT data are downloaded at run time from the CFTC public reporting API (Socrata).

## Run

```bash
python src/process_data.py
```

Writes `processed_data.csv`, `cftc_cot.csv` and `cftc_dcot.csv` to `data/processed/`.
