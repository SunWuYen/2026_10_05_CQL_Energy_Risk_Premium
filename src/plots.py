from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def _finish(fig, save_path: Path | None, show: bool) -> None:
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def plot_cumulative_returns(
    df_ret,
    benchmark="EWI",
    title="Cumulative Log Returns Over Time",
    figsize=(12, 6),
    save_path: Path | None = None,
    show: bool = True,
):
    """
    Plot cumulative log returns for all columns in df_ret.
    """
    period_start_plus_one = df_ret.index[df_ret.index >= config.PERIOD_START][1]
    
    # Cumulative log returns
    df_plot = df_ret.loc[
        period_start_plus_one:config.PERIOD_END
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

    _finish(fig, save_path, show)

    return fig, ax


def plot_style_decomposition(
    df_ret_by_product,
    title="Cumulative Product-level Contributions for EWI",
    figsize=(12, 6),
    save_path: Path | None = None,
    show: bool = True,
):
    """
    Plot cumulative product-level contributions for one style.
    """
     
    df_plot = (
        df_ret_by_product
        .loc['EWI']
        .loc[config.PERIOD_START:config.PERIOD_END]
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

    _finish(fig, save_path, show)

    return fig, ax


def plot_grouped_contribution(
    df_ret_by_product,
    group_map,
    title="Cumulative Grouped Contributions for EWI",
    figsize=(12, 6),
    save_path: Path | None = None,
    show: bool = True,
):
    """
    Plot cumulative grouped product contributions for one style.
    """

    # Daily product-level contributions
    df_daily_ret = (
        df_ret_by_product
        .loc['EWI']
        .loc[config.PERIOD_START:config.PERIOD_END]
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

    _finish(fig, save_path, show)

    return fig, ax
