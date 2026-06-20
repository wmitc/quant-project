"""Cross-sectional momentum strategy.

Signal: the trailing 12-month return skipping the most recent month — the
classic "12-1" momentum, where skipping the last month avoids short-term
reversal. At each rebalance we rank the universe and go long the top quantile /
short the bottom quantile, equal-weighted within each leg and dollar-neutral.
"""
from __future__ import annotations

import pandas as pd


def momentum_signal(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 momentum: return from `lookback` days ago to `skip` days ago.

    Uses only past prices at each date, so there is no look-ahead.
    """
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def month_end_rebalances(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month present in `index`."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().to_numpy())


def momentum_weights(
    prices: pd.DataFrame,
    rebal_dates,
    lookback: int = 252,
    skip: int = 21,
    quantile: float = 0.1,
) -> pd.DataFrame:
    """Long top / short bottom `quantile` by momentum, dollar-neutral.

    Returns a (rebalance dates x assets) frame of target weights. The long leg
    sums to +0.5 and the short leg to -0.5 (gross exposure 1.0, net 0).
    """
    signal = momentum_signal(prices, lookback, skip)
    rows = {}
    for date in rebal_dates:
        s = signal.loc[date].dropna()
        n = int(len(s) * quantile)
        if n < 1:
            continue
        ranked = s.sort_values()
        longs = ranked.index[-n:]
        shorts = ranked.index[:n]
        w = pd.Series(0.0, index=prices.columns)
        w[longs] = 0.5 / n
        w[shorts] = -0.5 / n
        rows[date] = w
    return pd.DataFrame(rows).T
