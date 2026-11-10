import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm


def _value_at_risk(df: pd.DataFrame, level=0.99, horizon=21) -> pd.Series:
    """
    Compute Cornish-Fisher Value-at-Risk using daily log returns,
    with cumulants scaled to the specified horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Daily log-return series, with assets/portfolios in columns.

    level : float, default 0.99
        Confidence level (0.99 corresponds to the 1% left tail).

    horizon : int, default 21
        Number of trading days for the VaR horizon.

    Returns
    -------
    pd.Series
        Column-wise horizon-day VaR expressed as a loss fraction.
    """

    # Daily cumulants
    k1 = df.mean()
    k2 = ((df - k1)**2).mean()
    k3 = ((df - k1)**3).mean()
    k4 = ((df - k1)**4).mean() - 3 * k2**2

    # Scale cumulants to the horizon
    k1 *= horizon
    k2 *= horizon
    k3 *= horizon
    k4 *= horizon

    # Horizon volatility
    s = np.sqrt(k2).replace(0, np.nan)

    # Horizon standardized cumulants
    skew = k3 / s**3
    kurt = k4 / s**4

    # Left-tail normal quantile
    z = norm.ppf(1 - level)

    # Cornish-Fisher adjusted quantile
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3*z) * kurt / 24
        - (2*z**3 - 5*z) * skew**2 / 36
    )

    return 1 - np.exp(k1 + z_cf * s)

def _max_drawdown(df: pd.DataFrame) -> pd.Series:
    """
    Compute maximum drawdown for each column of log returns.

    Parameters:
        df (pd.DataFrame): DataFrame of daily log returns.

    Returns:
        pd.Series: Maximum drawdown (as negative percentages) for each column.
    """
    results = {}

    for col in df.columns:
        r = df[col].dropna()

        if r.empty:
            results[col] = np.nan
            continue

        wealth = np.exp(r.cumsum())
        hwm = wealth.cummax().clip(lower=1.0)

        results[col] = (wealth / hwm - 1).min()

    return pd.Series(results, index=df.columns)

def summary_statistics(
    df,
    annualization_factor=252,
    var_horizon=21,
    hac_maxlags=None,
):
    """Summary statistics for daily log returns.

    If hac_maxlags is None, the Newey-West lag is set per column with the
    rule of thumb floor(4 * (T / 100) ** (2 / 9)).
    """
    list_tstats = []
    list_pvalue = []

    for col in df.columns:
        data = df[col].dropna()
        maxlags = (
            hac_maxlags
            if hac_maxlags is not None
            else int(np.floor(4 * (len(data) / 100) ** (2 / 9)))
        )

        reg = sm.OLS(
            data.to_numpy(),
            np.ones(len(data)),
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags},
        )

        list_tstats.append(reg.tvalues[0])
        list_pvalue.append(reg.pvalues[0])

    arr_var = _value_at_risk(
        df=df,
        horizon=var_horizon,
    ).values

    arr_maxdd = _max_drawdown(
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
                mean * np.sqrt(annualization_factor)
                / std.replace(0, np.nan)
            ).values,

            "Newey-West t-statistics": list_tstats,
            "Newey-West p-value": list_pvalue,
            "Maximum drawdown": arr_maxdd,
            "99% monthly VaR (Cornish-Fisher)": arr_var,
        },
        index=df.columns,
    ).T
