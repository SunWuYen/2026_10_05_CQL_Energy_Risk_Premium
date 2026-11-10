# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8


# In[1]:
# --------------------------------------------------------
# Import libraries and Define classess and functions
# --------------------------------------------------------

LIST_TICKERS = ["CO", "CL", "QS", "HO", "XBW", "NG", "CUA", "BAP", "MFE"]
LIST_MULTIPLES = [ 1000, 1000, 100, 420, 420, 10000, 42000, 42000, 1000]
PERIOD_START = "2001-07-01"; PERIOD_END = "2026-08-31"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.formula.api import ols

def fillna(
    ticker: str,
    df_data: pd.DataFrame
):
    columns = [f"{ticker}1 Comdty", f"{ticker}2 Comdty"]

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

    def same_month(date_1, date_2):
        return (
            date_1.year == date_2.year
            and date_1.month == date_2.month
        )

    def month_has_expiry(date):
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
        entered_new_month = not same_month(previous_date, current_date)
        current_is_month_end = not same_month(current_date, next_date)
        next_has_expiry = month_has_expiry(next_date)

        # ------------------------------------------------------------
        # 1. Generic-contract relabeling immediately after expiry
        # ------------------------------------------------------------
        if previous_was_expiry:

            if entered_new_month and month_has_expiry(current_date):
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

def value_at_risk(df: pd.DataFrame, level=0.99, horizon=21) -> pd.Series:
    """
    Compute Cornish-Fisher Value-at-Risk for each column in a DataFrame
    using non-overlapping horizon-day log returns.

    Parameters
    ----------
    df : pd.DataFrame
        Daily log-return series, with assets/portfolios in columns.

    level : float, default 0.99
        Confidence level. For example, 0.99 corresponds
        to the 1% left-tail quantile.

    horizon : int, default 21
        Number of trading days in each non-overlapping return block.

    Returns
    -------
    pd.Series
        Column-wise VaR expressed as a positive loss fraction.
    """

    # Non-overlapping horizon-day log returns
    group = np.arange(len(df)) // horizon

    df_horizon = (
        df.groupby(group)
        .sum(min_count=horizon)
    )

    # Cumulants
    k1 = df_horizon.mean()
    k2 = ((df_horizon - k1)**2).mean()
    k3 = ((df_horizon - k1)**3).mean()
    k4 = ((df_horizon - k1)**4).mean() - 3 * k2**2

    # k5 = ((df_horizon - k1)**5).mean() - 10 * k3 * k2

    s = np.sqrt(k2).replace(0, np.nan)

    # Standardized cumulants
    skew = k3 / s**3
    kurt = k4 / s**4

    # kappa5 = k5 / s**5

    # Left-tail normal quantile
    z = norm.ppf(1 - level)

    # Cornish-Fisher adjusted quantile
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3*z) * kurt / 24
        - (2*z**3 - 5*z) * skew**2 / 36

        # + (z**4 - 6*z**2 + 3) * kappa5 / 120
        # - (z**4 - 5*z**2 + 2) * skew * kurt / 24
        # + (12*z**4 - 53*z**2 + 17) * skew**3 / 324
    )

    return 1 - np.exp(k1 + z_cf * s)

def max_drawdown(df: pd.DataFrame) -> pd.Series:
    """
    Compute maximum drawdown for each column of log returns.

    Parameters:
        df (pd.DataFrame): DataFrame of daily log returns (columns = portfolios).

    Returns:
        pd.Series: Maximum drawdown (as negative percentages) for each portfolio.
    """

    # Convert log returns to cumulative returns
    cum_returns = np.exp(df.cumsum())

    # Compute rolling high-water mark
    hwm = cum_returns.cummax()

    # Compute drawdowns
    drawdowns = cum_returns / hwm - 1

    # Max drawdown is the minimum of the drawdown curve
    return drawdowns.min()

def summary_statistics(
    df,
    annualization_factor=252,
    var_multiple=21,
    hac_maxlags=15,
):
    list_tstats = []
    list_pvalue = []

    for col in df.columns:
        data = df[[col]].dropna()

        reg = ols(
            f"{col} ~ 1",
            data=data,
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_maxlags},
        )

        list_tstats.append(reg.tvalues.iloc[0])
        list_pvalue.append(reg.pvalues.iloc[0])

    arr_var = value_at_risk(
        df=df,
        horizon=var_multiple,
    ).values

    arr_maxdd = max_drawdown(
        df=df,
    ).values

    mean = df.mean()
    std = df.std()

    return pd.DataFrame(
        {
            "Annualized Mean": (
                mean * annualization_factor
            ).values,

            "Annualized StDev": (
                std * annualization_factor**0.5
            ).values,

            "Annualized Sharpe ratio": (
                mean * annualization_factor
                / (std * annualization_factor**0.5)
            ).values,

            "Newey-West t-statistics": list_tstats,
            "Newey-West p-value": list_pvalue,
            "Maximum drawdown": arr_maxdd,
            "99% monthly VaR (Cornish-Fisher)": arr_var,
        },
        index=df.columns,
    ).T

class StyleBase:
    def __init__(self, name=None, signal=None):
        self.name = name
        self.signal = signal
        self.normalized_signal = None
        self.weight = None

    def normalize_signal(self, df_data=None):
        raise NotImplementedError

    def compute_weight(self):
        denominator = self.normalized_signal.abs().sum(axis=1)

        self.weight = self.normalized_signal.div(
            denominator.replace(0, np.nan),
            axis=0,
        ).fillna(0.0)
        
class StyleTSFFM(StyleBase):
    def normalize_signal(self, df_data=None):
        self.normalized_signal = self.signal.copy(deep=True)
        
class StyleTS(StyleBase):
    def normalize_signal(self, df_data: pd.DataFrame):
        df_vol = df_data.loc[
            self.signal.index,
            [f"{ticker}_rolling_vol" for ticker in self.signal.columns]
        ].copy()
        df_vol.columns = self.signal.columns
        df_vol = df_vol.replace(0, np.nan)
        
        self.normalized_signal = np.sign(self.signal).div(df_vol)
        
class StyleCSFFM(StyleBase):
    def normalize_signal(self, df_data=None):
        mean = self.signal.mean(axis=1)
        std = self.signal.std(axis=1)

        self.normalized_signal = (
            self.signal.sub(mean, axis=0)
            .div(std.replace(0, np.nan), axis=0)
        )
        
class StyleCS(StyleBase):
    def normalize_signal(self, df_data: pd.DataFrame):
        normalized = pd.DataFrame(
            np.nan,
            index=self.signal.index,
            columns=self.signal.columns,
        )

        for index, row in self.signal.iterrows():
            valid = row.dropna()

            if valid.empty:
                continue

            n = len(valid)
            n_side = n // 2

            temp = pd.Series(0.0, index=row.index)

            temp.loc[valid.nlargest(n_side).index] = 1.0
            temp.loc[valid.nsmallest(n_side).index] = -1.0

            normalized.loc[index] = temp

        df_vol = df_data.loc[
            self.signal.index,
            [f"{ticker}_rolling_vol" for ticker in self.signal.columns]
        ].copy()
        df_vol.columns = self.signal.columns
        df_vol = df_vol.replace(0, np.nan)

        self.normalized_signal = normalized.div(df_vol)

def compute_return(
    list_style,
    df_data,
):

    cols = [
        f"{product}_rolling_ret"
        for product in LIST_TICKERS
    ]

    period_start_plus_one = df_data.index[
        df_data.index >= PERIOD_START
    ][1]

    list_name = [
        style.name
        for style in list_style
    ]

    month_end_dates = (
        df_data.index
        .to_series()
        .groupby(df_data.index.to_period("M"))
        .max()
    )

    first_rebalance_date = month_end_dates[
        month_end_dates <= PERIOD_START
    ].iloc[-1]

    list_rebalance_date = month_end_dates[
        month_end_dates >= first_rebalance_date
    ].tolist()

    df_ret = pd.DataFrame(
        0.0,
        index=df_data.index,
        columns=list_name,
    )

    index_tuples = [
        (name, date)
        for name in list_name
        for date in df_data.index
    ]

    multi_index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=["Style", "Date"],
    )

    df_ret_by_product = pd.DataFrame(
        0.0,
        index=multi_index,
        columns=LIST_TICKERS,
    )

    n = 0

    for index in df_data.loc[
        period_start_plus_one:PERIOD_END
    ].index:


        log_ret = df_data.loc[index, cols].values

        for style in list_style:

            weight = style.weight.loc[
                list_rebalance_date[n]
            ].values

            product_ret = weight * log_ret

            df_ret.loc[index, style.name] = np.nansum(
                product_ret
            )

            df_ret_by_product.loc[
                (style.name, index)
            ] = product_ret

        if (
            n + 1 < len(list_rebalance_date)
            and index >= list_rebalance_date[n + 1]
        ):
            n += 1

    return df_ret, df_ret_by_product


# In[2]:
# --------------------------------------------------------
# Read data and compute rolling returns
# --------------------------------------------------------


df_data = pd.read_csv(
    f"intermediate_outputs/processed_data.csv", 
    index_col = 0
)
df_data.index = pd.to_datetime(df_data.index)
df_data = df_data.sort_index()

# Add one business day after the last date
next_business_date = df_data.index[-1] + pd.offsets.BDay(1)
df_data.loc[next_business_date] = df_data.iloc[-1].copy(deep=True)

df_expiration = pd.read_excel(
    "data/20260930_energy_futures_price_and_expiration.xlsx", 
    sheet_name = 1, header = 0
)

price_cols   = [f"{ticker}1 Comdty" for ticker in LIST_TICKERS]
oi_cols = [
    f"{ticker}_open_interest" for ticker in LIST_TICKERS if ticker != "XBW"
]

valid_index = (
    df_data[price_cols].notna().any(axis=1) |
    df_data[oi_cols].notna().any(axis=1)
)
df_data = df_data.loc[valid_index]

for ticker in LIST_TICKERS:
    idx_start, idx_end = fillna(ticker,df_data); print(ticker, idx_start, idx_end)
    compute_rolling_ret(ticker, df_data, df_expiration, idx_start, idx_end, False)

df_data = df_data.drop(index=next_business_date, errors="ignore")


# In[3]:
# --------------------------------------------------------
# Table 2 Panel A
# --------------------------------------------------------


cols = [ticker + "_rolling_ret" for ticker in LIST_TICKERS]
period_start_plus_one = df_data.index[df_data.index >= PERIOD_START][1]

table2_panel_a = summary_statistics(
    df_data.loc[
        period_start_plus_one:PERIOD_END,
        cols,
    ].rename(
        columns={
            f"{ticker}_rolling_ret": ticker
            for ticker in LIST_TICKERS
        }
    )
)

table2_panel_a.to_csv(
    "intermediate_outputs/Table2_PanelA.csv"
)


# In[4]:
# --------------------------------------------------------
# Combine data for CO, CL, NG (Optional)
# --------------------------------------------------------

'''
df_data.loc[:,'CO_open_interest':'CO_other_rept_spread'] = (
    df_data.loc[:,'CO_open_interest':'CO_other_rept_spread'] +
    df_data.loc[:,'CO1_open_interest':'CO1_other_rept_spread'].fillna(0.0).values
)

df_data.loc[:,'CL_open_interest':'CL_other_rept_spread'] = (
    df_data.loc[:,'CL_open_interest':'CL_other_rept_spread'] +
    df_data.loc[:,'CL1_open_interest':'CL1_other_rept_spread'].fillna(0.0).values +
    df_data.loc[:,'CL2_open_interest':'CL2_other_rept_spread'].fillna(0.0).values
)

df_data.loc[:,'NG_open_interest':'NG_other_rept_spread'] = (
    df_data.loc[:,'NG_open_interest':'NG_other_rept_spread'] +
    df_data.loc[:,'NG1_open_interest':'NG1_other_rept_spread'].fillna(0.0).values/4 +
    df_data.loc[:,'NG2_open_interest':'NG2_other_rept_spread'].fillna(0.0).values/4 +
    df_data.loc[:,'NG3_open_interest':'NG3_other_rept_spread'].fillna(0.0).values/4 +
    df_data.loc[:,'NG4_open_interest':'NG4_other_rept_spread'].fillna(0.0).values +
    df_data.loc[:,'NG5_open_interest':'NG5_other_rept_spread'].fillna(0.0).values +
    df_data.loc[:,'NG6_open_interest':'NG6_other_rept_spread'].fillna(0.0).values/4
)'''



# In[5]:
# --------------------------------------------------------
# Combine data for XBW and remove some columns
# --------------------------------------------------------


xb_data = df_data.loc[
    :, "XB_open_interest":"XB_nonrept_short"
]

hu_data = df_data.loc[
    :, "HU_open_interest":"HU_nonrept_short"
]

# Identify observations where both HU and XB are missing
both_missing = xb_data.isna() & hu_data.isna().values

# Combine HU historical data and XB recent data
combined = (
    xb_data.fillna(0.0) +
    hu_data.fillna(0.0).values
)

# Keep genuinely missing observations as NaN
combined = combined.mask(both_missing)

df_data.loc[
    :, "XB_open_interest":"XB_nonrept_short"
] = combined

# Rename combined XB columns to XBW
df_data.rename(
    columns={
        col: col.replace("XB_", "XBW_", 1)
        for col in [col for col in df_data.columns if col.startswith("XB_")]
    },
    inplace=True,
)

# Remove columns
df_data.drop(columns=[col for col in df_data.columns if col.startswith("CO1_")], inplace=True)
df_data.drop(columns=[col for col in df_data.columns if col.startswith("CL1_") or col.startswith("CL2_")], inplace=True)
df_data.drop(columns=[col for col in df_data.columns if col.startswith("HU_")], inplace=True)
df_data.drop(
    columns=[
        col for col in df_data.columns
        if (
            col.startswith("NG1_")
            or col.startswith("NG2_")
            or col.startswith("NG3_")
            or col.startswith("NG4_")
            or col.startswith("NG5_")
            or col.startswith("NG6_")
        )
    ],
    inplace=True,
)


# In[6]:
# --------------------------------------------------------
# Term Structure (TS)
# --------------------------------------------------------


list_StyleCSFFM = list()
list_StyleTSFFM = list()
list_StyleCS = list()
list_StyleTS = list()

# ---------- TS1 ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for ticker in LIST_TICKERS:
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{ticker}1 Comdty', f'{ticker}2 Comdty']].notnull().all():
                break

    df[ticker] = np.log(
            df_data[ticker + '1 Comdty']/df_data[ticker + '2 Comdty']
        ).rolling(window=252, min_periods=1).mean()
    
    df.loc[df_data.index[:imp_idx_start+252-1], ticker] = np.nan
    
df.loc['2022-03-23':,'MFE'] = np.nan

list_StyleCSFFM.append(StyleCSFFM(f'TS', signal = df))
#list_StyleTSFFM.append(StyleTSFFM(f'TS', signal = df))
#list_StyleCS.append(StyleCS(f'TS', signal = df))
#list_StyleTS.append(StyleTS(f'TS', signal = df))

# ---------- TS2 ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for ticker in LIST_TICKERS:
    df[ticker] = np.log(
            df_data[ticker + '1 Comdty']/df_data[ticker + '13 Comdty']
        ).ffill()
    
df.loc['2022-03-23':,'MFE'] = np.nan


# In[7]:
# --------------------------------------------------------
# Hedging Pressure (HP) and Speculation Pressure (SP)
# --------------------------------------------------------


# ---------- HP1 ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for ticker in LIST_TICKERS:
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{ticker}_comm_long', f'{ticker}_comm_short']].notnull().all():
            break
    
    temp = (
        (df_data[f'{ticker}_comm_short'] - df_data[f'{ticker}_comm_long']) /
        (df_data[f'{ticker}_comm_short'] + df_data[f'{ticker}_comm_long'])
    )

    df[ticker] = temp.rolling(window=252, min_periods=1).mean()
    df.loc[df_data.index[:imp_idx_start+252-1], ticker] = np.nan

df.loc['2022-03-23':,'MFE'] = np.nan

list_StyleCSFFM.append(StyleCSFFM(f'HP_COM', signal = df))
#list_StyleTSFFM.append(StyleTSFFM(f'HP_COM', signal = df))
#list_StyleCS.append(StyleCS(f'HP_COM', signal = df))
#list_StyleTS.append(StyleTS(f'HP_COM', signal = df))

# ---------- HP2 ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for ticker in LIST_TICKERS:
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{ticker}_comm_long', f'{ticker}_comm_short']].notnull().all():
            break
    
    temp = (
        (df_data[f'{ticker}_comm_short'] - df_data[f'{ticker}_comm_long']) /
        (df_data[f'{ticker}_comm_short'] + df_data[f'{ticker}_comm_long'])
    )

    df[ticker] = (
        temp - temp.rolling(window=252, min_periods=1).mean()
    ) / temp.rolling(window=252, min_periods=1).std()
    df[ticker] = df[ticker].ffill()
    df.loc[df_data.index[:imp_idx_start+252-1], ticker] = np.nan

df.loc['2022-03-23':,'MFE'] = np.nan

# ---------- SP1 ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)
          
for ticker in LIST_TICKERS:
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{ticker}_noncomm_long', f'{ticker}_noncomm_short']].notnull().all():
            break
    
    temp = (
        (df_data[f'{ticker}_noncomm_long'] - df_data[f'{ticker}_noncomm_short']) /
        (df_data[f'{ticker}_noncomm_long'] + df_data[f'{ticker}_noncomm_short'])
    )

    df[ticker] = temp.rolling(window=252, min_periods=1).mean()
    df.loc[df_data.index[:imp_idx_start+252-1], ticker] = np.nan

df.loc['2022-03-23':,'MFE'] = np.nan

list_StyleCSFFM.append(StyleCSFFM(f'SP_NC', signal = df))
#list_StyleTSFFM.append(StyleTSFFM(f'SP_NC', signal = df))
#list_StyleCS.append(StyleCS(f'SP_NC', signal = df))
#list_StyleTS.append(StyleTS(f'SP_NC', signal = df))

# ---------- SP2 ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for ticker in LIST_TICKERS:
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{ticker}_noncomm_long', f'{ticker}_noncomm_short']].notnull().all():
            break
    
    temp = (
        (df_data[f'{ticker}_noncomm_long'] - df_data[f'{ticker}_noncomm_short']) /
        (df_data[f'{ticker}_noncomm_long'] + df_data[f'{ticker}_noncomm_short'])
    )

    df[ticker] = (
        temp - temp.rolling(window=252, min_periods=1).mean()
    ) / temp.rolling(window=252, min_periods=1).std()
    df[ticker] = df[ticker].ffill()
    df.loc[df_data.index[:imp_idx_start+252-1], ticker] = np.nan

df.loc['2022-03-23':,'MFE'] = np.nan


# In[8]:
# --------------------------------------------------------
# Momentum
# --------------------------------------------------------


# ---------- Mom ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for product in LIST_TICKERS:
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{product}_rolling_ret']].notnull().all():
            break
    df[product] = df_data.loc[:, f'{product}_rolling_ret'].rolling(window=252, min_periods=1).sum()
    df.loc[df_data.index[:imp_idx_start+252-1], product] = np.nan
    
df.loc['2022-03-23':,'MFE'] = np.nan

list_StyleCSFFM.append(StyleCSFFM(f'Mom', signal = df))
#list_StyleTSFFM.append(StyleTSFFM(f'Mom', signal = df))
#list_StyleCS.append(StyleCS(f'Mom', signal = df))
#list_StyleTS.append(StyleTS(f'Mom', signal = df))


# In[9]:
# --------------------------------------------------------
# Value, Liquidity and Skewness
# --------------------------------------------------------


# ---------- Value ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for ticker in LIST_TICKERS:
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{ticker}1 Comdty']].notnull().all():
            break
    df[ticker] = df_data.loc[:, f'{ticker}1 Comdty'].rolling(window=252, min_periods=1).mean()
    df.loc[df_data.index[:imp_idx_start+252-1], ticker] = np.nan
    
    df[ticker] = np.log(
        df[ticker].shift(int(4.5*252)) / df_data.loc[:, f'{ticker}1 Comdty']
    ).ffill()
    
df.loc['2022-03-23':,'MFE'] = np.nan

list_StyleCSFFM.append(StyleCSFFM(f'Value', signal = df))
#list_StyleTSFFM.append(StyleTSFFM(f'Value', signal = df))
#list_StyleCS.append(StyleCS(f'Value', signal = df))
#list_StyleTS.append(StyleTS(f'Value', signal = df))


# ---------- Liquidity ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for product, multiple in zip(LIST_TICKERS, LIST_MULTIPLES):
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{product}2 Comdty', f'{product} volume']].notnull().all():
            break
    
    mask = (
        df_data[f"{product}_rolling_ret"].notna()
        & df_data[f"{product}_rolling_ret"].ne(0.0)
        & df_data[f"{product}2 Comdty"].notna()
        & df_data[f"{product} volume"].notna()
    )

    df.loc[mask, product] = (
        multiple * df_data.loc[mask, f'{product}2 Comdty'] * df_data.loc[mask, f'{product} volume'] /
        np.abs(df_data.loc[mask, f'{product}_rolling_ret'])
    )
    
    df[product] = -df[product].rolling(window=42, min_periods=1).mean()*10**-12
    df.loc[df_data.index[:imp_idx_start+42-1], product] = np.nan

df.loc['2022-03-23':,'MFE'] = np.nan

list_StyleCSFFM.append(StyleCSFFM(f'Liquidity', signal = df))
#list_StyleTSFFM.append(StyleTSFFM(f'Liquidity', signal = df))
#list_StyleCS.append(StyleCS(f'Liquidity', signal = df))
#list_StyleTS.append(StyleTS(f'Liquidity', signal = df))

# ---------- Skewness ----------
df = pd.DataFrame(np.nan, index = df_data.index, columns = LIST_TICKERS)

for ticker in LIST_TICKERS:
    for imp_idx_start, idx_start in enumerate(df_data.index):
        if df_data.loc[idx_start, [f'{ticker}_rolling_ret']].notnull().all():
            break
    df[ticker] = -(df_data.loc[:, f'{ticker}_rolling_ret']).rolling(window=252, min_periods=1).skew()
    df.loc[df_data.index[:imp_idx_start+252-1], ticker] = np.nan

df.loc['2022-03-23':,'MFE'] = np.nan

list_StyleCSFFM.append(StyleCSFFM(f'Skewness', signal = df))
#list_StyleTSFFM.append(StyleTSFFM(f'Skewness', signal = df))
#list_StyleCS.append(StyleCS(f'Skewness', signal = df))
#list_StyleTS.append(StyleTS(f'Skewness', signal = df))


# In[10]:
# --------------------------------------------------------
# Table 2 Panel B
# --------------------------------------------------------

pd.DataFrame({
        s.name: s.signal.loc[PERIOD_START:PERIOD_END,:].mean().values for s in list_StyleCSFFM
    },index = LIST_TICKERS).T.to_csv("intermediate_outputs/Table2_PanelB.csv")


# In[11]:
# --------------------------------------------------------
# Normalize signals and compute weights
# Instantiate EWI as the average of all style portfolios
# --------------------------------------------------------


# ---------- StyleCSFFM ----------
for style in list_StyleCSFFM:
    style.normalize_signal(df_data=df_data)
    style.compute_weight()
    
style = StyleCSFFM(f'EWI')
style.normalized_signal = pd.DataFrame( 
        np.nanmean(np.stack([s.normalized_signal.values for s in list_StyleCSFFM]),axis=0),
        index = df_data.index, 
        columns = LIST_TICKERS
)
style.compute_weight()
list_StyleCSFFM.append(style)

'''
# ---------- StyleTSFFM ----------
for style in list_StyleTSFFM:    
    style.normalize_signal(df_data=df_data)
    style.compute_weight()
    
style = StyleTSFFM(f'EWI')
style.normalized_signal = pd.DataFrame( 
        np.nanmean(np.stack([s.normalized_signal.values for s in list_StyleTSFFM]),axis=0),
        index = df_data.index, 
        columns = LIST_TICKERS
)
style.compute_weight()
list_StyleTSFFM.append(style)

# ---------- StyleCS ----------
for style in list_StyleCS:    
    style.normalize_signal(df_data=df_data)
    style.compute_weight()
    
style = StyleCS(f'EWI')
style.normalized_signal = pd.DataFrame( 
        np.nanmean(np.stack([s.normalized_signal.values for s in list_StyleCS]),axis=0),
        index = df_data.index, 
        columns = LIST_TICKERS
)
style.compute_weight()
list_StyleCS.append(style)

# ---------- StyleTS ----------
for style in list_StyleTS:    
    style.normalize_signal(df_data=df_data)
    style.compute_weight()    

style = StyleTS(f'EWI')
style.normalized_signal = pd.DataFrame( 
        np.nanmean(np.stack([s.normalized_signal.values for s in list_StyleTS]),axis=0),
        index = df_data.index, 
        columns = LIST_TICKERS
)
style.compute_weight()
list_StyleTS.append(style)
'''

# In[12]:
# --------------------------------------------------------
# Compute returns and generate Table 3
# --------------------------------------------------------

df_ret1, df_ret1_by_product = compute_return(list_StyleCSFFM, df_data)
table3 = summary_statistics(df_ret1.loc[period_start_plus_one:PERIOD_END])
table3.to_excel('intermediate_outputs/Table3_1.xlsx')
df_ret1.to_csv('intermediate_outputs/ret1.csv')
df_ret1_by_product.to_csv('intermediate_outputs/ret1_by_product.csv')
'''
df_ret2, df_ret2_by_product = compute_return(list_StyleTSFFM, df_data)
table3 = summary_statistics(df_ret2.loc[period_start_plus_one:PERIOD_END])
table3.to_excel('intermediate_outputs/Table3_2.xlsx')

df_ret3, df_ret3_by_product = compute_return(list_StyleCS, df_data)
table3 = summary_statistics(df_ret3.loc[period_start_plus_one:PERIOD_END])
table3.to_excel('intermediate_outputs/Table3_3.xlsx')

df_ret4, df_ret4_by_product = compute_return(list_StyleTS, df_data)
table3 = summary_statistics(df_ret4.loc[period_start_plus_one:PERIOD_END])
table3.to_excel('intermediate_outputs/Table3_4.xlsx')
'''

# In[13]:
# --------------------------------------------------------
# Plot cumulative returns
# --------------------------------------------------------

def plot_cumulative_returns(
    df_ret,
    benchmark="EWI",
    title="Cumulative Log Returns Over Time",
    figsize=(12, 6),
):
    """
    Plot cumulative log returns for all columns in df_ret.
    """
    period_start_plus_one = df_ret.index[df_ret.index >= PERIOD_START][1]
    
    # Cumulative log returns
    df_plot = df_ret.loc[
        period_start_plus_one:PERIOD_END
    ].cumsum()

    # All columns except benchmark
    regular_columns = [
        col for col in df_plot.columns
        if col != benchmark
    ]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    df_plot[regular_columns].plot(
        ax=ax,
        linewidth=1.5,
        alpha=0.7,
    )

    # Highlight benchmark
    if benchmark in df_plot.columns:
        df_plot[benchmark].plot(
            ax=ax,
            linewidth=1.5,
            alpha=0.7,
            color="black",
            label=benchmark,
        )

    # Aesthetics
    ax.set_title(
        title,
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        title="Style Portfolio",
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
    )

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig, ax

plot_cumulative_returns(
    df_ret1,
    benchmark="EWI",
    title="Cumulative Log Returns Over Time - CSFFM",
    figsize=(12, 6),
)
'''
plot_cumulative_returns(
    df_ret2,
    benchmark="EWI",
    title="Cumulative Log Returns Over Time - TSFFM",
    figsize=(12, 6),
)

plot_cumulative_returns(
    df_ret3,
    benchmark="EWI",
    title="Cumulative Log Returns Over Time - CS",
    figsize=(12, 6),
)

plot_cumulative_returns(
    df_ret4,
    benchmark="EWI",
    title="Cumulative Log Returns Over Time - CS",
    figsize=(12, 6),
)
'''


# In[14]:
# --------------------------------------------------------
# Plot style decomposition for EWI
# --------------------------------------------------------

def plot_style_decomposition(
    df_ret_by_product,
    title="Cumulative Product-level Contributions for EWI",
    figsize=(12, 6),
):
    """
    Plot cumulative product-level contributions for one style.
    """
     
    df_plot = (
        df_ret_by_product
        .loc['EWI']
        .loc[PERIOD_START:PERIOD_END]
        .cumsum()
        .ffill()
    )

    fig, ax = plt.subplots(figsize=figsize)

    df_plot.plot(
        ax=ax,
        linewidth=1.5,
        alpha=0.7,
    )

    ax.set_title(
        title,
        fontsize=16,
        weight="bold",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        title="Products",
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
    )

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig, ax


plot_style_decomposition(
    df_ret1_by_product,
    title="Cumulative Product-level Contributions for EWI - CSFFM",
    figsize=(12, 6),
)

'''
plot_style_decomposition(
    df_ret2_by_product,
    style="EWI",
    figsize=(12, 6),
)

plot_style_decomposition(
    df_ret3_by_product,
    style="EWI",
    figsize=(12, 6),
)

plot_style_decomposition(
    df_ret4_by_product,
    style="EWI",
    figsize=(12, 6),
)'''


# In[15]:
# --------------------------------------------------------
# Plot style decomposition for EWI
# --------------------------------------------------------


def plot_grouped_contribution(
    df_ret_by_product,
    group_map,
    title="Cumulative Grouped Contributions for EWI",
    figsize=(12, 6)
):
    """
    Plot cumulative grouped product contributions for one style.
    """

    # Daily product-level contributions
    df_daily_ret = (
        df_ret_by_product
        .loc['EWI']
        .loc[PERIOD_START:PERIOD_END]
    )

    # Group-level daily contributions
    df_group_contrib = pd.DataFrame(
        index=df_daily_ret.index
    )

    for group_name, products in group_map.items():
        available_products = [
            product for product in products
            if product in df_daily_ret.columns
        ]

        df_group_contrib[group_name] = (
            df_daily_ret[available_products]
            .sum(axis=1)
        )

    # Total portfolio contribution
    df_group_contrib["Total"] = (
        df_group_contrib.sum(axis=1)
    )

    # Cumulative contribution
    df_cum_ret = (
        df_group_contrib
        .cumsum()
        .ffill()
    )

    # Plot
    fig, ax = plt.subplots(
        figsize=figsize,
    )

    base = np.zeros(len(df_cum_ret))

    for group_name in group_map:
        top = base + df_cum_ret[group_name].values

        ax.fill_between(
            df_cum_ret.index,
            base,
            top,
            label=group_name,
            alpha=0.6,
        )

        base = top

    # Total portfolio
    df_cum_ret["Total"].plot(
        ax=ax,
        color="black",
        linewidth=2,
        label="EWI",
    )

    ax.set_title(
        title,
        fontsize=16,
        weight="bold",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Log Return", fontsize=12)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        title="Groups",
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
    )

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()

    return fig, ax

group_map = {
    "Natural Gas": ["NG"],
    "Combined Petroleum": ["CO", "CL", "QS", "HO", "XBW"],
    "Ethanol, Propane, and Coal": ["CUA", "BAP", "MFE"],
}

plot_grouped_contribution(
    df_ret1_by_product,
    group_map=group_map,
    title="Cumulative Grouped Contributions for EWI - CSFFM",
)

# %%
