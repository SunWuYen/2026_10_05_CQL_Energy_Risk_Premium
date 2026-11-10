import numpy as np
import pandas as pd

from scipy.stats import norm

# Parameter for vol estimation
delta = 60/61


# Forward Fill NA Values
def fillna(product: str, df_data: pd.DataFrame):
    columns = [f'{product}1 Comdty', f'{product}2 Comdty']
    
    # search for the start and end index
    for idx_start in df_data.index:
        if df_data.loc[idx_start, columns].notnull().all():
            break
    for idx_end in df_data.index[::-1]:
        if df_data.loc[idx_end, columns].notnull().all():
            break
            
    # forward fill na values between the start and end index
    df_data.loc[idx_start:idx_end, columns] = (
        df_data.loc[idx_start:idx_end, columns].fillna(method='ffill')
    )
    
    return idx_start, idx_end


# Compute Rolling Return
def compute_rolling_ret(product: str, df_data: pd.DataFrame, idx_start: str, idx_end: str, is_on_prompt_contract: bool):
    df_data[f'{product}_rolling_ret'] = np.nan
    df_data.loc[idx_start, product+'_rolling_ret'] = 0.0
    
    df_data[f'{product}_rolling_vol'] = np.nan
    arr_delta = np.array([delta**n for n in range(2*len(df_data.index))])
    
    for imp in range(len(df_data.index)):
        if idx_start < df_data.index[imp] < idx_end:
            if df_data.loc[df_data.index[imp-1], product+' expiration'] == 1:
                if df_data.index[imp].month == df_data.index[imp-1].month:
                    df_data.loc[df_data.index[imp], product+'_rolling_ret'] = np.log(
                        df_data.loc[df_data.index[imp], product+'1 Comdty'] / df_data.loc[df_data.index[imp-1], product+'2 Comdty']
                    )
                    is_on_prompt_contract = True if df_data.index[imp].month == df_data.index[imp+1].month else False
                else:
                    df_data.loc[df_data.index[imp], product+'_rolling_ret'] = np.log(
                        df_data.loc[df_data.index[imp], product+'2 Comdty'] / df_data.loc[df_data.index[imp-1], product+'3 Comdty']
                    )
                    is_on_prompt_contract = False                                                   
            elif is_on_prompt_contract:
                df_data.loc[df_data.index[imp], product+'_rolling_ret'] = np.log(
                    df_data.loc[df_data.index[imp], product+'1 Comdty'] / df_data.loc[df_data.index[imp-1], product+'1 Comdty']
                )
                if df_data.index[imp].month != df_data.index[imp+1].month:
                    is_on_prompt_contract = False            
            else:
                df_data.loc[df_data.index[imp], product+'_rolling_ret'] = np.log(
                    df_data.loc[df_data.index[imp], product+'2 Comdty'] / df_data.loc[df_data.index[imp-1], product+'2 Comdty']
                )
            
            length = len(df_data.loc[idx_start:df_data.index[imp-1]])
            df_data.loc[df_data.index[imp],f'{product}_rolling_vol'] = (
                (df_data.loc[idx_start:df_data.index[imp-1],f'{product}_rolling_ret'] * 
                 arr_delta[:length][::-1]).sum() / arr_delta[:length].sum()
            )
            df_data.loc[df_data.index[imp],f'{product}_rolling_vol'] = (
                ((df_data.loc[idx_start:df_data.index[imp-1],f'{product}_rolling_ret'] - 
                  df_data.loc[df_data.index[imp],f'{product}_rolling_vol'])**2 * 
                  arr_delta[:length][::-1]).sum()/arr_delta[:length].sum()*252
            )**0.5
    
    df_data[f'{product}_rolling_std_ret'] = np.nan
    df_data[f'{product}_rolling_std_ret'][2:] = 0.4 * (df_data[f'{product}_rolling_ret'][2:] /
                                                       df_data[f'{product}_rolling_vol'][2:])

    
# Cornish-Fisher VaR
# https://www.value-at-risk.net/the-cornish-fisher-expansion/
def value_at_risk(df: pd.DataFrame, level=0.99, multiple=1) -> pd.Series:
    """
    Compute Cornish-Fisher Value-at-Risk for each column in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame of return series (columns = portfolios).
        level (float): Confidence level (e.g., 0.99 for 1% left-tail VaR).
        multiple (int): Scaling factor for time horizon (e.g., 1 = daily, 21 = monthly).

    Returns:
        pd.Series: Column-wise Value-at-Risk values.
    """
    
    k1 = df.mean()
    k2 = ((df - k1)**2).mean()
    k3 = ((df - k1)**3).mean()
    k4 = ((df - k1)**4).mean() - 3 * k2**2
    k5 = ((df - k1)**5).mean() - 10 * k3 * k2

    s = np.sqrt(k2)  # standard deviation

    # Standardize the cumulants
    skew = k3 / s**3
    kurt = k4 / s**4
    kappa5 = k5 / s**5

    z = norm.ppf(1 - level)

    # Cornish-Fisher adjusted quantile
    z_cf = (
        z 
        + (z**2 - 1) * skew / 6 
        + (z**3 - 3*z) * kurt / 24 
        - (2*z**3 - 5*z) * skew**2 / 36 
        #+ (z**4 - 6*z**2 + 3) * kappa5 / 120 
        #- (z**4 - 5*z**2 + 2) * skew * kurt / 24 
        #+ (12*z**4 - 53*z**2 + 17) * skew**3 / 324
    )

    return 1 - np.exp(k1 * multiple + z_cf * s * np.sqrt(multiple))


# Maximum Drawdown
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


class Style_time_series_FFM:
    def __init__(self, name=None, signal=None, normalized_signal=None, weight=None):
        self.name = name
        self.signal = signal
        self.normalized_signal = normalized_signal
        self.weight = weight
        
    def normalize_signal(self):
        self.normalized_signal = self.signal.copy(deep=True)
        
    def compute_weight(self):
        self.weight = self.normalized_signal.copy(deep=True)
        
        for index in self.weight.index:
            if self.weight.loc[index,:].isnull().all():
                continue
            self.weight.loc[index,:] = self.weight.loc[index,:] / self.weight.loc[index,:].abs().sum()
        self.weight.fillna(0.0, inplace=True)


class Style_time_series:
    def __init__(self, name=None, signal=None, normalized_signal=None, weight=None):
        self.name = name
        self.signal = signal
        self.normalized_signal = normalized_signal
        self.weight = weight
        
    def normalize_signal(self):
        self.normalized_signal = self.signal.copy(deep=True)
        self.normalized_signal = np.sign(self.normalized_signal)
        
    def compute_weight(self):
        self.weight = self.normalized_signal.copy(deep=True)
        
        for index in self.weight.index:
            if self.weight.loc[index,:].isnull().all():
                continue
            self.weight.loc[index,:] = self.weight.loc[index,:] / self.weight.loc[index,:].abs().sum()
        self.weight.fillna(0.0, inplace=True)
        
        
class Style_cross_product_FFM:
    def __init__(self, name=None, signal=None, normalized_signal=None, weight=None):
        self.name = name
        self.signal = signal
        self.normalized_signal = normalized_signal
        self.weight = weight
        
    def normalize_signal(self):
        self.normalized_signal = self.signal.copy(deep=True)
        
        for index in self.signal.index:
            if self.signal.loc[index,:].isnull().all():
                continue
            self.normalized_signal.loc[index, :] = (
                self.normalized_signal.loc[index, :] - self.normalized_signal.loc[index, :].mean()
            ) / self.normalized_signal.loc[index, :].std()
      
    def compute_weight(self):
        self.weight = self.normalized_signal.copy(deep=True)
        
        for index in self.weight.index:
            if self.weight.loc[index,:].isnull().all():
                continue
            self.weight.loc[index,:] = self.weight.loc[index,:] / self.weight.loc[index,:].abs().sum()
        self.weight.fillna(0.0, inplace=True)
        
        
class Style_cross_product:
    def __init__(self, name=None, signal=None, normalized_signal=None, weight=None):
        self.name = name
        self.signal = signal
        self.normalized_signal = normalized_signal
        self.weight = weight
        
    def normalize_signal(self):
        self.normalized_signal = self.signal.copy(deep=True)
        
        for index in self.signal.index:
            if self.signal.loc[index,:].isnull().all():
                continue
            length = self.normalized_signal.loc[index,:].count()
            series_temp = pd.Series(0, index=self.normalized_signal.loc[index,:].index)
            series_temp[self.normalized_signal.loc[index,:].nlargest(length//2).index] = 1
            series_temp[self.normalized_signal.loc[index,:].nsmallest(length//2).index] = -1
            self.normalized_signal.loc[index, :] = series_temp
      
    def compute_weight(self):
        self.weight = self.normalized_signal.copy(deep=True)
        
        for index in self.weight.index:
            if self.weight.loc[index,:].isnull().all():
                continue
            self.weight.loc[index,:] = self.weight.loc[index,:] / self.weight.loc[index,:].abs().sum()
        self.weight.fillna(0.0, inplace=True)
        
        
def compute_return_FFM(list_style, df_data, period_start, period_end):
    list_product_ = list_style[0].signal.columns
    cols = [product + '_rolling_ret' for product in list_product_]
    period_start_plus_one = df_data.index[df_data.index >= period_start][1]
    
    list_name = [style.name for style in list_style]
    month_end_dates = df_data.index.to_series().groupby(df_data.index.to_period("M")).max()
    first_rebalance_date = month_end_dates[month_end_dates <= period_start][-1]
    list_rebalance_date = [idx for idx in month_end_dates if idx >= first_rebalance_date]
    
    # Total return per style per date (aggregated)
    df_ret = pd.DataFrame(0.0, index=df_data.index, columns=list_name)

    # Daily return per product per style → MultiIndex DataFrame
    index_tuples = [(name, date) for name in list_name for date in df_data.index]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["Style", "Date"])
    df_ret_by_product = pd.DataFrame(0.0, index=multi_index, columns=list_product_)
    
    n = 0           
    for index in df_ret.loc[list_rebalance_date[0]:list_rebalance_date[-2]].index:
        if df_data.index[df_data.index > period_end][0] > index >= period_start_plus_one:
            print(list_rebalance_date[n], index)
            log_ret = df_data.loc[index, cols].values  # shape: (n_products,)

            for style in list_style:
                weight = style.weight.loc[list_rebalance_date[n], :].values  # shape: (n_products,)
                #print(style.name, weight, log_ret)
                df_ret.loc[index, style.name] = np.nansum(weight * log_ret)
                df_ret_by_product.loc[(style.name, index), :] = weight * log_ret

        # Rebalance date switch
        if index >= list_rebalance_date[n + 1]:
            n += 1
    
    return list_name, df_ret, df_ret_by_product


def compute_return(list_style, df_data, period_start, period_end):
    list_product_ = list_style[0].signal.columns
    cols = [product + '_rolling_std_ret' for product in list_product_]
    period_start_plus_one = df_data.index[df_data.index >= period_start][1]
    
    list_name = [style.name for style in list_style]
    month_end_dates = df_data.index.to_series().groupby(df_data.index.to_period("M")).max()
    first_rebalance_date = month_end_dates[month_end_dates <= period_start][-1]
    list_rebalance_date = [idx for idx in month_end_dates if idx >= first_rebalance_date]
    
    # Total return per style per date (aggregated)
    df_ret = pd.DataFrame(0.0, index=df_data.index, columns=list_name)

    # Daily return per product per style → MultiIndex DataFrame
    index_tuples = [(name, date) for name in list_name for date in df_data.index]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["Style", "Date"])
    df_ret_by_product = pd.DataFrame(0.0, index=multi_index, columns=list_product_)
    
    n = 0           
    for index in df_ret.loc[list_rebalance_date[0]:list_rebalance_date[-2]].index:
        if df_data.index[df_data.index > period_end][0] > index >= period_start_plus_one:
            print(list_rebalance_date[n], index)
            log_ret = df_data.loc[index, cols].values  # shape: (n_products,)

            for style in list_style:
                weight = style.weight.loc[list_rebalance_date[n], :].values  # shape: (n_products,)
                #print(style.name, weight, log_ret)
                df_ret.loc[index, style.name] = np.nansum(weight * log_ret)
                df_ret_by_product.loc[(style.name, index), :] = weight * log_ret

        # Rebalance date switch
        if index >= list_rebalance_date[n + 1]:
            n += 1
    
    return list_name, df_ret, df_ret_by_product
    
    
    
    
    
    
    
  