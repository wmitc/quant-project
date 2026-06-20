"""ETF pairs / statistical-arbitrage mean reversion.

Two related assets usually move together, so a linear combination of them (the
*spread*) is mean-reverting even though each price wanders. We estimate the
hedge ratio and spread statistics on a training window, then trade the z-score
of the spread out-of-sample: enter when it stretches far from its mean, exit
when it reverts, stop out if it keeps diverging (the relationship has broken).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from quantbt.backtest import walk_forward_windows


def hedge_ratio(y, x) -> tuple[float, float]:
    """OLS fit y = beta * x + alpha; return (beta, alpha)."""
    beta, alpha = np.polyfit(np.asarray(x), np.asarray(y), 1)
    return float(beta), float(alpha)


def cointegration_pvalue(y, x) -> float:
    """Engle-Granger cointegration test p-value (low = cointegrated)."""
    return float(coint(y, x)[1])


def half_life(spread) -> float:
    """Mean-reversion half-life (days) from an AR(1) fit of the spread.

    Regress the change in the spread on its lagged level; a negative slope `b`
    implies reversion with half-life -ln(2)/b. Returns inf if not reverting.
    """
    s = pd.Series(spread).dropna()
    lag = s.shift(1)
    delta = s - lag
    df = pd.concat([delta, lag], axis=1).dropna()
    b, _ = np.polyfit(df.iloc[:, 1].to_numpy(), df.iloc[:, 0].to_numpy(), 1)
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / b)


def positions_from_zscore(z, entry=2.0, exit_threshold=0.5, stop=4.0) -> pd.Series:
    """Map a spread z-score series to a position (+1 long / -1 short / 0 flat).

    Long the spread when it is cheap (z < -entry), short it when rich
    (z > entry), close near the mean (|z| < exit_threshold), and stop out if it
    diverges past `stop`.
    """
    pos = 0
    out = []
    for val in z:
        if pos == 0:
            if val > entry:
                pos = -1
            elif val < -entry:
                pos = 1
        elif pos == 1:  # long spread
            if val >= -exit_threshold or val < -stop:
                pos = 0
        elif pos == -1:  # short spread
            if val <= exit_threshold or val > stop:
                pos = 0
        out.append(pos)
    return pd.Series(out, index=pd.Index(z.index))


def pair_weights(
    prices: pd.DataFrame,
    y_col: str,
    x_col: str,
    train: int = 252,
    test: int = 63,
    entry: float = 2.0,
    exit_threshold: float = 0.5,
    stop: float = 4.0,
    coint_threshold: float = 0.05,
) -> pd.DataFrame:
    """Walk-forward daily target weights for one pair (columns y_col, x_col).

    For each window we first test whether the pair is cointegrated *in the train
    block*; if not (p-value above `coint_threshold`), we stand aside that window.
    Otherwise we fit the hedge ratio and spread mean/std on the train block and
    trade the z-score on the (out-of-sample) test block. Weights are scaled so
    the pair's gross exposure is 1 when in a position.
    """
    y, x = prices[y_col], prices[x_col]
    pieces = []
    for tr, te in walk_forward_windows(prices.index, train, test):
        if cointegration_pvalue(y.loc[tr], x.loc[tr]) > coint_threshold:
            continue  # relationship not present in-sample -> don't trade it
        beta, alpha = hedge_ratio(y.loc[tr], x.loc[tr])
        train_spread = y.loc[tr] - (beta * x.loc[tr] + alpha)
        mu, sigma = train_spread.mean(), train_spread.std(ddof=1)
        if sigma == 0:
            continue
        z = (y.loc[te] - (beta * x.loc[te] + alpha) - mu) / sigma
        pos = positions_from_zscore(z, entry, exit_threshold, stop)
        scale = 1.0 / (1.0 + abs(beta))
        w = pd.DataFrame(index=te, columns=[y_col, x_col], dtype=float)
        w[y_col] = pos * scale
        w[x_col] = -pos * beta * scale
        pieces.append(w)
    if not pieces:
        return pd.DataFrame(columns=[y_col, x_col])
    return pd.concat(pieces)
