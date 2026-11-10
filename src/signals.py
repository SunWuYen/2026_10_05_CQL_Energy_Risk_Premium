import numpy as np
import pandas as pd

import config
from portfolio import StyleBase

WINDOW = config.SIGNAL_WINDOW
LIQUIDITY_WINDOW = config.LIQUIDITY_WINDOW
VALUE_LAG = config.VALUE_LAG
COT_LAG = config.COT_RELEASE_LAG


def _first_valid_position(df_data: pd.DataFrame, columns: list[str]) -> int:
    """Position of the first row where all columns are non-null."""
    valid = df_data[columns].notnull().all(axis=1).values
    return int(valid.argmax()) if valid.any() else len(valid) - 1

def _finalize(
    df: pd.DataFrame,
    df_data: pd.DataFrame,
    ticker: str,
    start_pos: int,
    window: int,
) -> None:
    """Blank the warm-up period of one ticker's signal (in place)."""
    df.loc[df_data.index[:start_pos + window - 1], ticker] = np.nan

def _mask_mfe(df: pd.DataFrame) -> None:
    df.loc[config.MFE_SIGNAL_END:, "MFE"] = np.nan

def _empty_signal(df_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.nan, index=df_data.index, columns=config.LIST_TICKERS
    )

def term_structure(df_data: pd.DataFrame) -> pd.DataFrame:
    df = _empty_signal(df_data)
    for ticker in config.LIST_TICKERS:
        df[ticker] = np.log(
            df_data[ticker + "1 Comdty"] / df_data[ticker + "2 Comdty"]
        ).ffill()
    _mask_mfe(df)
    return df

def hedging_pressure(df_data: pd.DataFrame) -> pd.DataFrame:
    df = _empty_signal(df_data)
    for ticker in config.LIST_TICKERS:
        df[ticker] = (
            (df_data[f"{ticker}_comm_short"] - df_data[f"{ticker}_comm_long"])
            / (df_data[f"{ticker}_comm_short"] + df_data[f"{ticker}_comm_long"])
        ).shift(COT_LAG).ffill()
    _mask_mfe(df)
    return df

def speculative_pressure(df_data: pd.DataFrame) -> pd.DataFrame:
    df = _empty_signal(df_data)
    for ticker in config.LIST_TICKERS:
        df[ticker] = (
            (df_data[f"{ticker}_m_money_long"] - df_data[f"{ticker}_m_money_short"])
            / (df_data[f"{ticker}_m_money_long"] + df_data[f"{ticker}_m_money_short"])
        ).shift(COT_LAG).ffill()
    _mask_mfe(df)
    return df

def momentum(df_data: pd.DataFrame) -> pd.DataFrame:
    df = _empty_signal(df_data)
    for ticker in config.LIST_TICKERS:
        pos = _first_valid_position(df_data, [f"{ticker}_rolling_ret"])
        df[ticker] = (
            df_data[f"{ticker}_rolling_ret"]
            .rolling(window=WINDOW, min_periods=1).sum()
        )
        _finalize(df, df_data, ticker, pos, WINDOW)
    _mask_mfe(df)
    return df

def value(df_data: pd.DataFrame) -> pd.DataFrame:
    df = _empty_signal(df_data)
    for ticker in config.LIST_TICKERS:
        pos = _first_valid_position(df_data, [f"{ticker}1 Comdty"])
        df[ticker] = (
            df_data[f"{ticker}1 Comdty"]
            .rolling(window=WINDOW, min_periods=1).mean()
        )
        _finalize(df, df_data, ticker, pos, WINDOW)
        df[ticker] = np.log(
            df[ticker].shift(VALUE_LAG) / df_data[f"{ticker}1 Comdty"]
        ).ffill()
    _mask_mfe(df)
    return df

def liquidity(df_data: pd.DataFrame) -> pd.DataFrame:
    df = _empty_signal(df_data)
    for product, multiple in zip(config.LIST_TICKERS, config.LIST_MULTIPLES):
        pos = _first_valid_position(
            df_data, [f"{product}2 Comdty", f"{product} volume"]
        )
        # Same-day mask so numerator and denominator cover identical days
        mask = (
            df_data[f"{product}_rolling_ret"].notna()
            & df_data[f"{product}2 Comdty"].notna()
            & df_data[f"{product} volume"].notna()
        )
        dollar_volume = (
            multiple
            * df_data[f"{product}2 Comdty"]
            * df_data[f"{product} volume"]
        ).where(mask)
        abs_ret = df_data[f"{product}_rolling_ret"].abs().where(mask)

        # Amivest = sum(dollar volume) / sum(|return|) over the window;
        # the signal is -Amivest (long the least liquid).
        roll = lambda s: s.rolling(window=LIQUIDITY_WINDOW, min_periods=1).sum()
        amivest = roll(dollar_volume) / roll(abs_ret)
        df[product] = -amivest * 10**-9
        _finalize(df, df_data, product, pos, LIQUIDITY_WINDOW)
    _mask_mfe(df)
    return df

def skewness(df_data: pd.DataFrame) -> pd.DataFrame:
    df = _empty_signal(df_data)
    for ticker in config.LIST_TICKERS:
        pos = _first_valid_position(df_data, [f"{ticker}_rolling_ret"])
        df[ticker] = -(
            df_data[f"{ticker}_rolling_ret"]
            .rolling(window=WINDOW, min_periods=1).skew()
        )
        _finalize(df, df_data, ticker, pos, WINDOW)
    _mask_mfe(df)
    return df

def build_style_signals(
    df_data: pd.DataFrame, style_cls: type[StyleBase]
) -> list[StyleBase]:
    """Style signals in reporting order, wrapped in the given style class"""
    return [
        style_cls("TS", signal=term_structure(df_data)),
        style_cls("HP_COM", signal=hedging_pressure(df_data)),
        style_cls("SP_MM", signal=speculative_pressure(df_data)),
        style_cls("Mom", signal=momentum(df_data)),
        style_cls("Value", signal=value(df_data)),
        style_cls("Liquidity", signal=liquidity(df_data)),
        style_cls("Skewness", signal=skewness(df_data)),
    ]
