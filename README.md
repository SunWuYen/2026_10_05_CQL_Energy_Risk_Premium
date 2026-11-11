# Energy Risk Premium

Research project on risk premia in energy futures, using signals including
hedging pressure from CFTC/ICE Commitments of Traders data. Current work focuses
on cross-sectional portfolio construction and strategy returns.

## Current scope

The current backtest supports the `csffm` and `cs` cross-sectional portfolio
constructions. Time-series portfolio construction is not being worked on at
this stage; the project description and current backtest outputs should not be
read as indicating that it is implemented.

## Pipeline

| Module | Script | Status | Output |
|---|---|---|---|
| 1. Process data | `src/process_data.py` | done | `data/processed/` |
| 2. Cross-sectional backtest (signals, portfolios, returns, tables) | `src/run_backtest.py` (uses `returns.py`, `signals.py`, `portfolio.py`, `performance.py`) | in progress | `outputs/<style>/` |
| 3. Plots | `src/plots.py` (called from `run_backtest.py`) | basic | `outputs/<style>/figures/` |

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

CFTC legacy and disaggregated COT data are downloaded at run time from the
CFTC public reporting API (Socrata), so processing requires an internet
connection. The API extracts are also saved under `data/processed/`; retain
those files with the processed dataset when you need to reproduce a run, since
the public source can be revised.

## Run

```bash
python src/process_data.py
```

Writes `processed_data.csv`, `cftc_cot.csv` and `cftc_dcot.csv` to `data/processed/`.

```bash
python src/run_backtest.py
python src/run_backtest.py --style csffm
python src/run_backtest.py --style cs
python src/run_backtest.py --style both
```

Reads `data/processed/processed_data.csv` and writes `Table2_PanelA.csv`,
`Table2_PanelB.csv`, `Table3.xlsx`, `Table4.xlsx`, `ret.csv`,
`ret_by_product.csv`, `weight_by_product.csv`, `signal_by_product.csv`,
`normalized_signal_by_product.csv` and figures to `outputs/csffm/` or
`outputs/cs/`. Running without `--style` (or
using `--style both`) runs both constructions; each writes to its own folder.
Use `--style csffm` or `--style cs` to run only one construction.
For CS, a warning reports dates where tied signals cause the long and short
selections to overlap; the existing short assignment takes precedence.
The product-level files have style and date index columns followed by product
columns. The signal files contain same-day raw and normalized signals,
respectively; EWI has no raw signal, so its raw entries are missing. The weight
file contains the prior month-end weights actually applied to each day's
return, with zeros outside the backtest calculation period. Like the product
return file, these files include EWI but not the separately reported CUA/NG pair.
`Table3.xlsx` contains full-sample statistics and separate statistics for
the sample through March 2019 and April 2019 onward, including a separate
Long CUA Short NG portfolio. Its CSFFM weights are +0.5/-0.5; its CS weights
are volatility-scaled and normalized to gross exposure 1. `Table4.xlsx`
reports that portfolio's correlation with EWI for the same sample periods.
Tickers, contract multipliers and the sample period are set in `src/config.py`.
