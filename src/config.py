from pathlib import Path

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Raw input files
PRICE_EXPIRATION_FILE = DATA_RAW / "20260930_energy_futures_price_and_expiration.xlsx"

# Processed files (output of process_data.py)
PROCESSED_DATA_FILE = DATA_PROCESSED / "processed_data.csv"
CFTC_COT_FILE = DATA_PROCESSED / "cftc_cot.csv"
CFTC_DCOT_FILE = DATA_PROCESSED / "cftc_dcot.csv"

# Backtest universe: tickers and their contract multipliers (same order)
LIST_TICKERS = ["CO", "CL", "QS", "HO", "XBW", "NG", "CUA", "BAP", "MFE"]
LIST_MULTIPLES = [1000, 1000, 100, 420, 420, 10000, 42000, 42000, 1000]

PERIOD_START = "2011-01-01"
PERIOD_END = "2026-08-31"

# MFE signals are masked from this date onward (data no longer usable)
MFE_SIGNAL_END = "2022-03-23"

# Signal look-back windows (trading days)
SIGNAL_WINDOW = 252
LIQUIDITY_WINDOW = 42
VALUE_LAG = int(4.5 * 252)

# COT reports are as of Tuesday but released Friday: lag positioning
# signals by 3 trading-day rows to avoid look-ahead
COT_RELEASE_LAG = 3
