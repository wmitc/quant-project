"""Run the cross-sectional momentum backtest end to end.

Downloads (and caches) a fixed universe of liquid US large-caps, runs the
walk-forward momentum backtest after costs, prints the headline metrics and a
cost-sensitivity sweep, and saves a tearsheet plot.

Survivorship-bias note: the universe is a fixed list of names that are large-cap
*today*, so it excludes companies that were delisted over the sample. This
biases the backtest upward; it is disclosed rather than hidden.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantbt import metrics
from quantbt.backtest import daily_returns, simulate
from quantbt.costs import CostModel
from quantbt.data import load_dollar_adv, load_prices
from quantbt.plotting import save_tearsheet
from quantbt.strategies.momentum import momentum_weights, month_end_rebalances

# Fixed universe of liquid US large-caps that have traded since before 2015.
UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX", "KO", "PEP", "INTC",
    "CSCO", "WMT", "MRK", "PFE", "ABT", "TMO", "COST", "MCD", "NKE", "ORCL",
    "ACN", "TXN", "QCOM", "AMD", "IBM", "GE", "CAT", "BA", "MMM", "HON",
    "UPS", "LMT", "RTX", "GS", "MS", "AXP", "C", "WFC", "USB", "BLK",
    "CVX", "XOM", "COP", "SLB", "GILD", "AMGN", "BMY", "CVS", "UNH", "LLY",
    "DHR", "MDT", "UNP", "LOW", "SBUX", "TGT", "GM", "F", "DE", "EMR",
    "DUK", "SO", "NEE", "T", "VZ", "CMCSA", "PM", "MO", "CL", "KMB",
    "GIS", "MDLZ", "SYK", "ISRG", "BKNG", "ADI", "MU", "AMAT", "INTU", "AMT",
]

START, END = "2015-01-01", "2024-12-31"
CACHE = "data/momentum_universe.csv"
RESULTS = "results"


def main():
    prices = load_prices(UNIVERSE, START, END, CACHE)
    print(f"universe: {prices.shape[1]} names, {prices.shape[0]} days "
          f"({prices.index[0].date()} to {prices.index[-1].date()})")

    returns = daily_returns(prices)
    rebalances = month_end_rebalances(prices.index)
    weights = momentum_weights(prices, rebalances, quantile=0.1)  # top/bottom decile

    result = simulate(returns, weights, CostModel())
    summary = pd.Series(result.summary())
    print("\nHeadline metrics (after costs):")
    print(summary.round(3).to_string())

    # Cost-sensitivity sweep: does the edge survive higher costs?
    base = CostModel()
    sweep = {}
    for factor in (0.0, 1.0, 2.0, 3.0):
        res = simulate(returns, weights, base.scaled(factor))
        sweep[f"{factor:g}x costs"] = res.summary()
    sweep = pd.DataFrame(sweep).T[["ann_return", "sharpe", "max_drawdown"]]
    print("\nCost-sensitivity sweep:")
    print(sweep.round(3).to_string())

    # Capacity: AUM at which trades would hit 1% of a name's dollar volume.
    adv = load_dollar_adv(UNIVERSE, START, END, "data/momentum_dollar_adv.csv")
    cap = metrics.capacity(weights, adv, participation=0.01)
    print(f"\nCapacity (1% ADV participation): ${cap / 1e6:,.0f}M")

    # Persist results for the README.
    Path(RESULTS).mkdir(exist_ok=True)
    summary.round(4).to_csv(f"{RESULTS}/momentum_metrics.csv", header=["value"])
    sweep.round(4).to_csv(f"{RESULTS}/momentum_cost_sweep.csv")
    pd.Series({"capacity_usd": cap}).to_csv(f"{RESULTS}/momentum_capacity.csv", header=["value"])
    save_tearsheet(result, f"{RESULTS}/momentum_tearsheet.png")
    print(f"\nsaved metrics + tearsheet to {RESULTS}/")


if __name__ == "__main__":
    main()
