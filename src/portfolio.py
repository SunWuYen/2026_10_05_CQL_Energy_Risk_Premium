import numpy as np
import pandas as pd

import config


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
        overlap_count = 0
        overlap_examples = []

        for index, row in self.signal.iterrows():
            valid = row.dropna()

            if valid.empty:
                continue

            n = len(valid)
            n_side = n // 2

            temp = pd.Series(0.0, index=row.index)
            long_index = valid.nlargest(n_side).index
            short_index = valid.nsmallest(n_side).index
            overlap = long_index.intersection(short_index, sort=False)
            if not overlap.empty:
                overlap_count += 1
                if len(overlap_examples) < 5:
                    overlap_examples.append(f"{index}: {overlap.tolist()}")

            temp.loc[long_index] = 1.0
            temp.loc[short_index] = -1.0

            normalized.loc[index] = temp

        if overlap_count:
            print(
                f"Warning: StyleCS {self.name!r}: long and short selections "
                f"overlap on {overlap_count} date(s); short assignments "
                f"overwrite long assignments. Examples: {overlap_examples}"
            )

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
        for product in config.LIST_TICKERS
    ]

    period_start_plus_one = df_data.index[
        df_data.index >= config.PERIOD_START
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
        month_end_dates <= config.PERIOD_START
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
        columns=config.LIST_TICKERS,
    )
    df_weight_by_product = df_ret_by_product.copy()

    n = 0

    for index in df_data.loc[
        period_start_plus_one:config.PERIOD_END
    ].index:


        log_ret = df_data.loc[index, cols].values

        for style in list_style:

            weight = style.weight.loc[
                list_rebalance_date[n]
            ].values

            df_weight_by_product.loc[(style.name, index)] = weight
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

    return df_ret, df_ret_by_product, df_weight_by_product
