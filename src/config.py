from pathlib import Path

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REFERENCE_DIR = PROJECT_ROOT / "reference"

# Raw input files
PRICE_EXPIRATION_FILE = DATA_RAW / "20260930_energy_futures_price_and_expiration.xlsx"

# Processed files (output of process_data.py)
PROCESSED_DATA_FILE = DATA_PROCESSED / "processed_data.csv"
CFTC_COT_FILE = DATA_PROCESSED / "cftc_cot.csv"
CFTC_DCOT_FILE = DATA_PROCESSED / "cftc_dcot.csv"
