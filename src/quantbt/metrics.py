"""Performance and risk metrics for backtested return streams.

Every function takes a pandas Series of periodic simple returns (e.g. daily)
unless noted. Annualized figures scale by `periods_per_year` (252 for daily).
The functions are pure (no I/O, no state) so they are easy to unit-test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric (CAGR-style) annualized return."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    growth = (1.0 + returns).prod()
    if growth <= 0:  # wiped out
        return -1.0
    return growth ** (periods_per_year / len(returns)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized standard deviation of returns."""
    returns = returns.dropna()
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe ratio. `rf` is the per-period risk-free rate."""
    returns = returns.dropna()
    excess = returns - rf
    sd = excess.std(ddof=1)
    if sd == 0:
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / sd


def sortino_ratio(
    returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sortino ratio: like Sharpe but penalizes only downside risk."""
    returns = returns.dropna()
    excess = returns - rf
    downside = excess.clip(upper=0.0)
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / downside_dev


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve (e.g. -0.2 == -20%)."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return drawdown.min()


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return float((returns > 0).mean())


def turnover(weights: pd.DataFrame) -> float:
    """Average one-sided turnover per rebalance.

    `weights` has rebalance dates on the rows and assets on the columns.
    Turnover at each step is 0.5 * sum(|w_t - w_{t-1}|); we average across
    steps. A value of 1.0 means the whole book is replaced each rebalance.
    """
    changes = weights.diff().abs().sum(axis=1)
    return float(0.5 * changes.iloc[1:].mean())


def capacity(weights: pd.DataFrame, dollar_adv: pd.Series, participation: float = 0.01) -> float:
    """Rough AUM capacity (USD) before trades exceed `participation` of ADV.

    At each rebalance the strategy must trade |dw_i| * AUM dollars in name i,
    which we cap at `participation` of that name's average daily dollar volume.
    The binding name sets capacity = min_i (participation * adv_i / max|dw_i|).
    """
    changes = weights.diff().abs()
    changes.iloc[0] = weights.iloc[0].abs()
    max_trade = changes.max(axis=0)
    max_trade = max_trade[max_trade > 0]
    adv = dollar_adv.reindex(max_trade.index)
    return float((participation * adv / max_trade).min())


def summarize(
    returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> dict[str, float]:
    """Bundle the headline metrics into one dict for reporting."""
    return {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, rf, periods_per_year),
        "sortino": sortino_ratio(returns, rf, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
    }


def tail_stats(returns: pd.Series, levels=(0.95, 0.99)) -> dict[str, float]:
    """Tail-risk descriptors: skew, excess kurtosis, worst day, and historical
    VaR and CVaR (expected shortfall) at the given confidence levels.

    VaR is the left-tail quantile of returns (negative); CVaR is the average
    return in that tail, so cvar <= var <= 0. These matter most for short-vol
    strategies, where Sharpe alone hides a fat left tail.
    """
    r = returns.dropna()
    out = {
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurt()),
        "worst_day": float(r.min()),
    }
    for level in levels:
        var = r.quantile(1.0 - level)
        out[f"var_{int(level * 100)}"] = float(var)
        out[f"cvar_{int(level * 100)}"] = float(r[r <= var].mean())
    return out
