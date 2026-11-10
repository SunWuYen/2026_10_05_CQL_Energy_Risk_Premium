import config
import pandas as pd
from pathlib import Path
from sodapy import Socrata

DICT_TICKERS = {
    "CO": "ICE Brent Crude Futures - ICE Futures Europe", "CO1":"06765T",
    "CL": "067651", "CL1": "067411", "CL2": "06765A",
    "QS": "ICE Gasoil Futures - ICE Futures Europe", 
    "HO": "022651", "HU": "111652", "XB": "111659", 
    "NG": "023651", "NG1": "023391", "NG2": "03565B", "NG3": "03565C", "NG4": "023A55", "NG5": "023A56", "NG6": "023392",
    "DL": "025601", "CUA": "025651", "MFE": "024656", "BAP": "06665O"
}

# Position columns (without ticker prefix), in output order.
COT_COLUMNS = [
    "open_interest",
    "comm_long", "comm_short",
    "noncomm_long", "noncomm_short", "noncomm_spread",
    "nonrept_long", "nonrept_short",
]
DCOT_COLUMNS = [
    "prod_merc_long", "prod_merc_short",
    "swap_long", "swap_short", "swap_spread",
    "m_money_long", "m_money_short", "m_money_spread",
    "other_rept_long", "other_rept_short", "other_rept_spread",
]
POSITION_COLUMNS = COT_COLUMNS + DCOT_COLUMNS

# Source column name -> position column (without ticker prefix)
CFTC_COT_RENAME = {
    "open_interest_all": "open_interest",
    "comm_positions_long_all": "comm_long",
    "comm_positions_short_all": "comm_short",
    "noncomm_positions_long_all": "noncomm_long",
    "noncomm_positions_short_all": "noncomm_short",
    "noncomm_postions_spread_all": "noncomm_spread",  # CFTC typo
    "nonrept_positions_long_all": "nonrept_long",
    "nonrept_positions_short_all": "nonrept_short",
}
CFTC_DCOT_RENAME = {
    "prod_merc_positions_long": "prod_merc_long",
    "prod_merc_positions_short": "prod_merc_short",
    "swap_positions_long_all": "swap_long",
    "swap_positions_short_all": "swap_short",
    "swap_positions_spread_all": "swap_spread",
    "swap__positions_short_all": "swap_short",   # Some historical files use double underscores
    "swap__positions_spread_all": "swap_spread", # Some historical files use double underscores
    "m_money_positions_long_all": "m_money_long",
    "m_money_positions_short_all": "m_money_short",
    "m_money_positions_spread": "m_money_spread",
    "other_rept_positions_long": "other_rept_long",
    "other_rept_positions_short": "other_rept_short",
    "other_rept_positions_spread": "other_rept_spread",
}
ICE_RENAME = {
    "Open_Interest_All": "open_interest",
    "Prod_Merc_Positions_Long_All": "prod_merc_long",
    "Prod_Merc_Positions_Short_All": "prod_merc_short",
    "Swap_Positions_Long_All": "swap_long",
    "Swap_Positions_Short_All": "swap_short",
    "Swap_Positions_Spread_All": "swap_spread", 
    "Swap__Positions_Short_All": "swap_short",    # Some historical files use double underscores
    "Swap__Positions_Spread_All": "swap_spread",  # Some historical files use double underscores
    "M_Money_Positions_Long_All": "m_money_long",
    "M_Money_Positions_Short_All": "m_money_short",
    "M_Money_Positions_Spread_All": "m_money_spread",
    "Other_Rept_Positions_Long_All": "other_rept_long",
    "Other_Rept_Positions_Short_All": "other_rept_short",
    "Other_Rept_Positions_Spread_All": "other_rept_spread",
    "NonRept_Positions_Long_All": "nonrept_long",
    "NonRept_Positions_Short_All": "nonrept_short",
}

def _ticker_columns(ticker: str, columns: list[str]) -> list[str]:
    return [f"{ticker}_{col}" for col in columns]

def _ticker_rename(ticker: str, rename: dict[str, str]) -> dict[str, str]:
    return {src: f"{ticker}_{col}" for src, col in rename.items()}

def _clean_date_index(
    df: pd.DataFrame,
    label: str,
    date_format: str | None = None,
) -> pd.DataFrame:
    """Parse the index as dates, drop unparseable dates, sort, dedupe."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, format=date_format, errors="coerce")
    df = df.loc[df.index.notna()]
    return _drop_duplicate_dates(df, label)

def _prepare_cftc_frame(
    df_all: pd.DataFrame,
    contract_market_code: str,
    ticker: str,
    rename: dict[str, str],
    label: str,
) -> pd.DataFrame:
    """Select one contract from a CFTC dataset, index by date, rename columns."""
    df = df_all.loc[
        df_all["cftc_contract_market_code"] == contract_market_code
    ].copy()
    if df.empty:
        return df

    df = df.set_index("report_date_as_yyyy_mm_dd")
    df = _clean_date_index(df, f"{label} {ticker}")
    return df.rename(columns=_ticker_rename(ticker, rename))

def _align_positions(
    df_data: pd.DataFrame,
    df_source: pd.DataFrame,
    columns: list[str],
    start_date: str,
) -> None:
    """Write source columns onto df_data's dates on/after start_date (in place)."""
    aligned = df_source.reindex(df_data.index)
    mask = df_data.index >= pd.Timestamp(start_date)
    df_data.loc[mask, columns] = aligned.loc[mask, columns]

def _drop_duplicate_dates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Sort by date and keep the last row per duplicated date, with a warning."""
    df = df.sort_index()
    duplicated = df.index.duplicated(keep="last")
    if duplicated.any():
        dates = df.index[duplicated].unique()
        print(
            f"Warning: {label}: dropped {duplicated.sum()} duplicate "
            f"date row(s), keeping the last. Dates: "
            f"{[d.strftime('%Y-%m-%d') for d in dates[:5]]}"
            f"{' ...' if len(dates) > 5 else ''}"
        )
    return df.loc[~duplicated].copy()

def ice_cot_dcot(
    df_data: pd.DataFrame,
    ticker: str,
    market_and_exchange_names: str,
    start_year: int = 2011,
    most_recent_year: int | None = None,
    data_dir: str | Path = config.DATA_RAW,
) -> pd.DataFrame:
    """
    Read and process ICE CoT/DCoT historical CSV files.

    Data source:
    https://www.ice.com/report/122

    Parameters
    ----------
    df_data : pd.DataFrame
        Master dataframe to which CFTC CoT/DCoT columns will be added.

    ticker : str
        Ticker used as the prefix for output columns.
        Example: "CO" for ICE Brent.

    market_and_exchange_names : str
        Exact ICE 'Market_and_Exchange_Names' value.

    start_year : int, default 2011
        First year to read.

    most_recent_year : int or None, default None
        Last year to read. If None, the latest COTHistYYYY.csv found
        in data_dir is used.

    data_dir : str or Path, default config.DATA_RAW
        Directory containing COTHistYYYY.csv files.

    Returns
    -------
    pd.DataFrame
        Copy of the master dataframe with ICE CoT/DCoT columns added or
        updated. The input df_data is not modified.
    """

    df_data = df_data.copy()
    frames = []

    available_years = sorted(
        int(path.stem.removeprefix("COTHist"))
        for path in Path(data_dir).glob("COTHist[0-9][0-9][0-9][0-9].csv")
    )
    if not available_years:
        raise FileNotFoundError(
            f"No COTHistYYYY.csv files found in {data_dir}."
        )

    if most_recent_year is None:
        most_recent_year = available_years[-1]

    years = [
        year for year in available_years
        if start_year <= year <= most_recent_year
    ]

    # Read annual ICE CoT files
    for year in years:

        file_path = Path(data_dir) / f"COTHist{year}.csv"

        df_temp = pd.read_csv(
            file_path,
            header=0,
            index_col="As_of_Date_Form_MM/DD/YYYY",
        )

        # Futures only + selected market
        df_temp = df_temp.loc[
            (df_temp["FutOnly_or_Combined"] == "FutOnly")
            & (
                df_temp["Market_and_Exchange_Names"]
                == market_and_exchange_names
            )
        ].copy()

        if df_temp.empty:
            print(
                "Warning: no observations found for "
                f"{market_and_exchange_names} in {year}."
            )
            continue

        frames.append(
            df_temp.rename(columns=_ticker_rename(ticker, ICE_RENAME))
        )

    if not frames:
        raise ValueError(
            f"No data found for '{market_and_exchange_names}' "
            f"between {start_year} and {most_recent_year}."
        )

    df_cot = _clean_date_index(
        pd.concat(frames, axis=0),
        f"ICE CoT {ticker}",
        date_format="%m/%d/%Y",
    )

    # --------------------------------------------------------------
    # Aggregate trader categories
    # --------------------------------------------------------------

    # Commercial = producer/merchant + swap dealers. Swap dealer spread
    # positions are held on both sides, so they enter both long and short
    # (the CFTC legacy "commercial" categories are not available for ICE).
    for side in ("long", "short"):
        df_cot[f"{ticker}_comm_{side}"] = (
            df_cot[f"{ticker}_prod_merc_{side}"]
            + df_cot[f"{ticker}_swap_{side}"]
            + df_cot[f"{ticker}_swap_spread"]
        )

    # Non-commercial
    for side in ("long", "short", "spread"):
        df_cot[f"{ticker}_noncomm_{side}"] = (
            df_cot[f"{ticker}_m_money_{side}"]
            + df_cot[f"{ticker}_other_rept_{side}"]
        )

    output_columns = _ticker_columns(ticker, POSITION_COLUMNS)
    df_data.loc[:, output_columns] = df_cot.reindex(
        df_data.index
    )[output_columns]

    return df_data

def get_cftc_datasets(
    client,
    contract_codes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download CFTC legacy CoT and disaggregated CoT data
    for a selected list of contract market codes.

    Returns
    -------
    df_cot_all : pd.DataFrame
        Legacy CoT dataset.

    df_dcot_all : pd.DataFrame
        Disaggregated CoT dataset.
    """

    codes = ", ".join(f'"{code}"' for code in contract_codes)

    where_clause = (
        f"cftc_contract_market_code in ({codes})"
    )

    # ----------------------------------------------------------
    # Legacy CoT
    # ----------------------------------------------------------
    cot_records = client.get_all(
        dataset_identifier="6dca-aqww",
        where=where_clause,
        order="report_date_as_yyyy_mm_dd",
    )

    df_cot_all = pd.DataFrame.from_records(cot_records)

    # ----------------------------------------------------------
    # Disaggregated CoT
    # ----------------------------------------------------------
    dcot_records = client.get_all(
        dataset_identifier="72hh-3qpy",
        where=where_clause,
        order="report_date_as_yyyy_mm_dd",
    )

    df_dcot_all = pd.DataFrame.from_records(dcot_records)

    return df_cot_all, df_dcot_all

def cftc_cot_dcot(
    df_data: pd.DataFrame,
    df_cot_all: pd.DataFrame,
    df_dcot_all: pd.DataFrame,
    ticker: str,
    cftc_contract_market_code: str,
    start_date: str = "1979-12-01",
) -> pd.DataFrame:
    """
    Process CFTC CoT/DCoT data for one ticker and add it
    to a copy of the master df_data dataframe, which is returned.
    """

    df_data = df_data.copy()
    df_data.index = pd.to_datetime(df_data.index)

    # Legacy CoT
    df_cot = _prepare_cftc_frame(
        df_cot_all, cftc_contract_market_code, ticker,
        CFTC_COT_RENAME, "CFTC CoT",
    )
    if not df_cot.empty:
        _align_positions(
            df_data, df_cot, _ticker_columns(ticker, COT_COLUMNS), start_date
        )

    # Disaggregated CoT
    df_dcot = _prepare_cftc_frame(
        df_dcot_all, cftc_contract_market_code, ticker,
        CFTC_DCOT_RENAME, "CFTC DCoT",
    )
    if not df_dcot.empty:
        _align_positions(
            df_data, df_dcot, _ticker_columns(ticker, DCOT_COLUMNS), start_date
        )

    return df_data

def main() -> pd.DataFrame:

    df_data = pd.read_excel(
        config.PRICE_EXPIRATION_FILE,
        skiprows = [4,5], header = 3, index_col = 0
    )
    df_data.index = pd.to_datetime(df_data.index)
    
    df_volume = pd.read_excel(
        config.PRICE_EXPIRATION_FILE,
        sheet_name = 2, skiprows = [4,5], header = 3, index_col = 0
    )
    df_volume.index = pd.to_datetime(df_volume.index)
    
    for ticker in DICT_TICKERS:
        if ticker == "XB":
            volume_col = "XBW2 Comdty"
            output_col = "XBW volume"
        else:
            volume_col = f"{ticker}2 Comdty"
            output_col = f"{ticker} volume"
        if volume_col in df_volume.columns:
            df_data[output_col] = df_volume[volume_col]
            
    # Retrieve CoT/DCoT data through CFTC API
    client = Socrata('publicreporting.cftc.gov', None)

    # ----------------------------------------------------------
    # Download CFTC data only once
    # ----------------------------------------------------------
    cftc_contract_market_codes = [code for code in DICT_TICKERS.values() if not code.endswith("ICE Futures Europe")]

    df_cot_all, df_dcot_all = get_cftc_datasets(
        client=client,
        contract_codes=cftc_contract_market_codes,
    )
    
    for ticker, element in DICT_TICKERS.items():
        if element.endswith("ICE Futures Europe"):
            df_data = ice_cot_dcot(
                df_data=df_data,
                ticker=ticker,
                market_and_exchange_names = element
            )
        else:
            df_data = cftc_cot_dcot(
                df_data=df_data,
                df_cot_all=df_cot_all,
                df_dcot_all=df_dcot_all,
                ticker=ticker,
                cftc_contract_market_code=element,
            )

    # Convert data type from object to numeric type
    cot_suffixes = tuple(f"_{col}" for col in POSITION_COLUMNS)

    for col in df_data.columns:
        if col.endswith(cot_suffixes):
            df_data[col] = pd.to_numeric(df_data[col], errors="coerce")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_data.to_csv(config.PROCESSED_DATA_FILE)
    df_cot_all.to_csv(config.CFTC_COT_FILE)
    df_dcot_all.to_csv(config.CFTC_DCOT_FILE)

    return df_data

if __name__ == "__main__":
    df_data = main()