"""Backtest of energy risk premia (cross-sectional style portfolios).

Reads data/processed/processed_data.csv (output of process_data.py) and writes
tables, returns and figures to outputs/<style>/.

Usage: python src/run_backtest.py
       python src/run_backtest.py --style csffm
       python src/run_backtest.py --style cs
       python src/run_backtest.py --style both
"""
import argparse
import numpy as np
import pandas as pd

import config
import plots
from performance import summary_statistics
from portfolio import StyleBase, StyleCSFFM, StyleCS, compute_return
from returns import compute_rolling_ret, fillna
from signals import build_style_signals


GROUP_MAP = {
    "Natural Gas": ["NG"],
    "Ethanol": ["CUA"],
    "Propane": ["BAP"],
    "Coal": ["MFE", "BAP"],
    "Petroleum Complex": ["CO", "CL", "QS", "HO", "XBW"]
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_data = pd.read_csv(config.PROCESSED_DATA_FILE, index_col=0)
    df_data.index = pd.to_datetime(df_data.index)
    df_data = df_data.sort_index()

    df_expiration = pd.read_excel(
        config.PRICE_EXPIRATION_FILE, sheet_name=1, header=0
    )
    return df_data, df_expiration


def add_rolling_returns(
    df_data: pd.DataFrame, df_expiration: pd.DataFrame
) -> pd.DataFrame:
    """Add <ticker>_rolling_ret / _rolling_cum_ret / _rolling_vol columns."""
    # Add one business day after the last date
    next_business_date = df_data.index[-1] + pd.offsets.BDay(1)
    df_data.loc[next_business_date] = df_data.iloc[-1].copy(deep=True)

    price_cols = [f"{ticker}1 Comdty" for ticker in config.LIST_TICKERS]
    oi_cols = [
        f"{ticker}_open_interest"
        for ticker in config.LIST_TICKERS if ticker != "XBW"
    ]

    valid_index = (
        df_data[price_cols].notna().any(axis=1)
        | df_data[oi_cols].notna().any(axis=1)
    )
    df_data = df_data.loc[valid_index]

    for ticker in config.LIST_TICKERS:
        idx_start, idx_end = fillna(ticker, df_data)
        print(ticker, idx_start, idx_end)
        compute_rolling_ret(
            ticker, df_data, df_expiration, idx_start, idx_end, False
        )
    return df_data.drop(index=next_business_date, errors="ignore")


def combine_xbw(df_data: pd.DataFrame) -> pd.DataFrame:
    """Merge HU (historical) and XB (recent) positioning into XBW columns."""
    df_data = df_data.copy()

    xb_data = df_data.loc[:, "XB_open_interest":"XB_nonrept_short"]
    hu_data = df_data.loc[:, "HU_open_interest":"HU_nonrept_short"]

    # Observations where both HU and XB are missing stay NaN
    both_missing = xb_data.isna() & hu_data.isna().values

    combined = xb_data.fillna(0.0) + hu_data.fillna(0.0).values
    combined = combined.mask(both_missing)

    df_data.loc[:, "XB_open_interest":"XB_nonrept_short"] = combined

    return df_data.rename(
        columns={
            col: col.replace("XB_", "XBW_", 1)
            for col in df_data.columns if col.startswith("XB_")
        }
    )


def drop_unused_columns(df_data: pd.DataFrame) -> pd.DataFrame:
    prefixes = (
        "CO1_", "CL1_", "CL2_", "HU_",
        "NG1_", "NG2_", "NG3_", "NG4_", "NG5_", "NG6_",
    )
    return df_data.drop(
        columns=[col for col in df_data.columns if col.startswith(prefixes)]
    )


def equal_weight_style(
    list_style: list[StyleBase], df_data: pd.DataFrame
) -> StyleBase:
    """EWI: average of the normalized signals of all style portfolios."""

    if not list_style:
        raise ValueError("Cannot build EWI without any style portfolios.")

    style = type(list_style[0])("EWI")
    style.normalized_signal = pd.DataFrame(
        np.nanmean(
            np.stack([s.normalized_signal.values for s in list_style]),
            axis=0,
        ),
        index=df_data.index,
        columns=config.LIST_TICKERS,
    )
    style.compute_weight()
    return style


def main(style_cls: type[StyleBase] = StyleCSFFM) -> dict:
    style_names = {
        StyleCSFFM: "csffm",
        StyleCS: "cs",
    }
    if style_cls not in style_names:
        raise ValueError(f"Unsupported style class: {style_cls.__name__}")

    style_name = style_names[style_cls]
    output_dir = config.OUTPUTS_DIR / style_name
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_data, df_expiration = load_data()
    df_data = add_rolling_returns(df_data, df_expiration)

    period_start_plus_one = df_data.index[
        df_data.index >= config.PERIOD_START
    ][1]
    
    df_data = drop_unused_columns(combine_xbw(df_data))

    # Table 2 Panel A
    cols = [f"{ticker}_rolling_ret" for ticker in config.LIST_TICKERS]
    table2_panel_a = summary_statistics(
        df_data.loc[
            period_start_plus_one:config.PERIOD_END, cols
        ].rename(
            columns={
                f"{ticker}_rolling_ret": ticker
                for ticker in config.LIST_TICKERS
            }
        )
    )
    table2_panel_a.to_csv(output_dir / "Table2_PanelA.csv")

    # Style signals
    list_style = build_style_signals(df_data, style_cls)

    # Table 2 Panel B
    table2_panel_b = pd.DataFrame(
        {
            s.name: s.signal.loc[
                config.PERIOD_START:config.PERIOD_END, :
            ].mean().values
            for s in list_style
        },
        index=config.LIST_TICKERS,
    ).T
    table2_panel_b.to_csv(output_dir / "Table2_PanelB.csv")

    # Normalize signals, compute weights, add EWI
    for style in list_style:
        style.normalize_signal(df_data=df_data)
        style.compute_weight()
    list_style.append(equal_weight_style(list_style, df_data))

    # Returns and Table 3
    df_ret, df_ret_by_product = compute_return(list_style, df_data)
    march_2019 = pd.Timestamp("2019-03-31")
    table3_periods = {
        "Full Sample": df_ret.loc[period_start_plus_one:config.PERIOD_END],
        "Start-Mar 2019": df_ret.loc[
            period_start_plus_one:march_2019
        ],
        "Mar 2019-End": df_ret.loc[march_2019+pd.Timedelta(days=1):config.PERIOD_END],
    }
    table3_stats = {
        name: summary_statistics(period_returns)
        for name, period_returns in table3_periods.items()
    }
    with pd.ExcelWriter(output_dir / "Table3.xlsx") as writer:
        for sheet_name, statistics in table3_stats.items():
            statistics.to_excel(writer, sheet_name=sheet_name)
    df_ret.to_csv(output_dir / "ret.csv")
    df_ret_by_product.to_csv(output_dir / "ret_by_product.csv")

    # Figures
    plots.plot_cumulative_returns(
        df_ret,
        benchmark="EWI",
        title=f"Cumulative Log Returns Over Time",
        save_path=figures_dir / f"cumulative_returns_{style_name}.png",
        show=False,
    )
    plots.plot_style_decomposition(
        df_ret_by_product,
        title=f"Cumulative Product-level Contributions for EWI",
        save_path=figures_dir / f"ewi_product_contributions_{style_name}.png",
        show=False,
    )
    plots.plot_grouped_contribution(
        df_ret_by_product,
        group_map=GROUP_MAP,
        title=f"Cumulative Grouped Contributions for EWI",
        save_path=figures_dir / f"ewi_grouped_contributions_{style_name}.png",
        show=False,
    )

    return {
        "style": style_name,
        "output_dir": output_dir,
        "df_data": df_data,
        "list_style": list_style,
        "df_ret": df_ret,
        "df_ret_by_product": df_ret_by_product,
        "table3": table3_stats["Full Sample"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run energy risk premium backtests.")
    parser.add_argument(
        "--style",
        choices=("csffm", "cs", "both"),
        default="both",
        help="Portfolio construction style to run (default: both).",
    )
    args = parser.parse_args()

    selected_styles = {
        "csffm": (StyleCSFFM,),
        "cs": (StyleCS,),
        "both": (StyleCSFFM, StyleCS),
    }[args.style]
    results = {
        style.__name__: main(style)
        for style in selected_styles
    }
