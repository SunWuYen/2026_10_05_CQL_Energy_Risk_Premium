import numpy as np
import pandas as pd


def fillna(
    ticker: str,
    df_data: pd.DataFrame
):
    columns = [f"{ticker}1 Comdty", f"{ticker}2 Comdty", f"{ticker}3 Comdty"]

    # Dates where all required generic contracts exist
    valid = df_data[columns].notna().all(axis=1)

    if not valid.any():
        raise ValueError(
            f"No date contains valid data for all {ticker} contracts."
        )

    idx_start = valid[valid].index[0]
    idx_end = valid[valid].index[-1]

    # Forward-fill only inside the valid sample
    df_data.loc[idx_start:idx_end, columns] = (
        df_data.loc[idx_start:idx_end, columns]
        .ffill()
    )

    return idx_start, idx_end

def compute_rolling_ret(
        ticker: str, 
        df_data: pd.DataFrame, 
        df_expiration: pd.DataFrame, 
        idx_start: str, 
        idx_end: str, 
        is_on_prompt_contract: bool,
        delta: float = 60/61,
        annualization_factor: int = 252,
):

    m1_col = f"{ticker}1 Comdty"
    m2_col = f"{ticker}2 Comdty"
    m3_col = f"{ticker}3 Comdty"

    ret_col = f"{ticker}_rolling_ret"
    cum_ret_col = f"{ticker}_rolling_cum_ret"
    vol_col = f"{ticker}_rolling_vol"

    # --------------------------------------------------------
    # Initialize output columns
    # --------------------------------------------------------

    df_data[ret_col] = np.nan
    df_data[cum_ret_col] = np.nan
    df_data[vol_col] = np.nan

    # --------------------------------------------------------
    # Expiration dates
    # --------------------------------------------------------

    expiration_dates = (
        pd.DatetimeIndex(df_expiration[ticker].dropna())
        .normalize()
        .sort_values()
    )

    def _same_month(date_1, date_2):
        return (
            date_1.year == date_2.year
            and date_1.month == date_2.month
        )

    def _month_has_expiry(date):
        return (
            (expiration_dates.year == date.year)
            & (expiration_dates.month == date.month)
        ).any()

    # --------------------------------------------------------
    # Construct rolling-contract
    # --------------------------------------------------------

    # Pre-compute exponential weights
    arr_delta = delta ** np.arange(len(df_data))  

    for imp in range(1, len(df_data.index) - 1):

        previous_date = df_data.index[imp - 1]
        current_date = df_data.index[imp]
        next_date = df_data.index[imp + 1]

        if not (idx_start < current_date <= idx_end):
            continue

        previous_was_expiry = previous_date.normalize() in expiration_dates
        entered_new_month = not _same_month(previous_date, current_date)
        current_is_month_end = not _same_month(current_date, next_date)
        next_has_expiry = _month_has_expiry(next_date)

        # ------------------------------------------------------------
        # 1. Generic-contract relabeling immediately after expiry
        # ------------------------------------------------------------
        if previous_was_expiry:

            if entered_new_month and _month_has_expiry(current_date):
                # The previous expiry was on the final observation of the
                # month, and the newly entered month also has an expiry.
                # The strategy remains on M2:
                # yesterday's M3 becomes today's M2.
                previous_col = m3_col
                current_col = m2_col
                is_on_prompt_contract = False

            else:
                # Standard generic shift:
                # yesterday's M2 becomes today's M1.
                # Normally the strategy is now on M1. However, if today is
                # month-end and next month has an expiry, switch to M2 from
                # the next observation onward.
                previous_col = m2_col
                current_col = m1_col
                is_on_prompt_contract = not (
                    current_is_month_end
                    and next_has_expiry
                )

        # ------------------------------------------------------------
        # 2. Normal daily log returns while holding M1
        # ------------------------------------------------------------
        elif is_on_prompt_contract:
            # Calculate today's log returns using M1, then move to M2 for the next
            # observation if the following month contains an expiry.
            previous_col = m1_col
            current_col = m1_col
            if current_is_month_end and next_has_expiry:
                is_on_prompt_contract = False

        # ------------------------------------------------------------
        # 3. Normal daily log returns while holding M2
        # ------------------------------------------------------------
        else:
            previous_col = m2_col
            current_col = m2_col

        # ----------------------------------------------------
        # Daily log returns
        # ----------------------------------------------------

        df_data.loc[current_date, ret_col] = np.log(
            df_data.loc[current_date, current_col]
            / df_data.loc[previous_date, previous_col]
        )

        # ----------------------------------------------------
        # Exponentially weighted volatility
        # ----------------------------------------------------
        ret_history = (
            df_data.loc[idx_start:current_date, ret_col]
            .dropna()
        )
        length = len(ret_history)

        if length >= 63:
            weights = arr_delta[:length][::-1]
            df_data.loc[current_date, vol_col] = np.sqrt(
                (
                    ret_history.pow(2).values
                    * weights
                ).sum()
                / weights.sum()
                * annualization_factor
            )
    # --------------------------------------------------------
    # Cumulative Returns
    # --------------------------------------------------------

    df_data.loc[idx_start:idx_end, cum_ret_col] = (
        df_data.loc[idx_start:idx_end, ret_col]
        .cumsum()
    )
