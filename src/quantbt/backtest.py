"""Walk-forward backtest engine.

Two pieces:

- `walk_forward_windows` rolls (train, test) windows through a date index so a
  strategy can estimate parameters on the past and trade the next block, with
  non-overlapping test windows and no look-ahead.
- `simulate` turns a stream of target weights into an after-cost return series
  and equity curve, charging transaction costs whenever the book is rebalanced.

Simplifying assumption: weights are reset to their target at each rebalance and
held fixed until the next one (intra-period drift is ignored). This keeps the
accounting transparent, which is the point of the project.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantbt import metrics
from quantbt.costs import CostModel


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from a price panel."""
    return prices.sort_index().pct_change()


def walk_forward_windows(index, train: int, test: int, step: int | None = None):
    """Yield (train_index, test_index) tiles rolling through `index`.

    `train`, `test` and `step` are counts of periods (rows). Test windows do not
    overlap (step defaults to `test`), so concatenating them gives one
    continuous out-of-sample track record.
    """
    step = step or test
    n = len(index)
    start = 0
    while start + train + test <= n:
        yield index[start : start + train], index[start + train : start + train + test]
        start += step


@dataclass
class BacktestResult:
    returns: pd.Series        # net (after-cost) daily returns
    gross_returns: pd.Series  # before costs
    equity: pd.Series         # net equity curve, starts near 1.0
    weights: pd.DataFrame     # target weights at each rebalance
    costs: pd.Series          # cost paid at each rebalance (fraction of capital)

    def summary(self, periods_per_year: int = metrics.TRADING_DAYS) -> dict[str, float]:
        out = metrics.summarize(self.returns, periods_per_year=periods_per_year)
        out["turnover"] = metrics.turnover(self.weights)
        out["total_cost"] = float(self.costs.sum())
        return out


def simulate(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """Run a portfolio from target weights and return after-cost performance.

    returns: daily simple returns (dates x assets).
    weights: target weights at rebalance dates (a subset of `returns.index`).
    """
    cost_model = cost_model or CostModel()
    returns = returns.sort_index()
    weights = weights.sort_index().reindex(columns=returns.columns).fillna(0.0)

    # Hold each rebalance's weights from the *next* day onward (no look-ahead):
    # a position decided at today's close earns tomorrow's return.
    daily_w = weights.reindex(returns.index, method="ffill").fillna(0.0)
    held = daily_w.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)

    # Costs on the change in target weights at each rebalance (the first
    # rebalance trades from an empty book).
    changes = weights.diff()
    changes.iloc[0] = weights.iloc[0]
    cost_per_rebal = changes.apply(lambda row: cost_model.cost(row.to_numpy()), axis=1)
    daily_cost = cost_per_rebal.reindex(returns.index).fillna(0.0)

    net = gross - daily_cost
    equity = (1.0 + net).cumprod()

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        equity=equity,
        weights=weights,
        costs=cost_per_rebal,
    )
