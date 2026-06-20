"""Run the ETF pairs / stat-arb backtest end to end.

Pairs are chosen a priori from economically related ETFs (not data-mined across
all combinations, which would invite multiple-testing bias). For each pair we
report the full-sample cointegration p-value and half-life as diagnostics, then
trade the spread z-score walk-forward (estimate on train, trade on test) and
combine the pairs into one equally-split portfolio.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantbt.backtest import daily_returns, simulate
from quantbt.costs import CostModel
from quantbt.data import load_prices
from quantbt.plotting import save_tearsheet
from quantbt.strategies.pairs import (
    cointegration_pvalue,
    half_life,
    hedge_ratio,
    pair_weights,
)

# Economically motivated pairs (chosen a priori, not screened across all combos).
PAIRS = [
    ("EWA", "EWC"),   # Australia vs Canada (commodity-linked economies)
    ("GDX", "GDXJ"),  # gold miners vs junior gold miners
    ("XLE", "XOP"),   # energy sector vs oil & gas exploration/production
    ("SMH", "SOXX"),  # two semiconductor baskets
    ("SPY", "IVV"),   # two S&P 500 trackers
]
CACHE = "data/etf_pairs.csv"
RESULTS = "results"


def main():
    tickers = sorted({t for pair in PAIRS for t in pair})
    prices = load_prices(tickers, "2015-01-01", "2024-12-31", CACHE)
    returns = daily_returns(prices)

    diagnostics = []
    weights_by_pair = {}
    for y, x in PAIRS:
        beta, alpha = hedge_ratio(prices[y], prices[x])
        spread = prices[y] - (beta * prices[x] + alpha)
        w = pair_weights(prices, y, x)
        weights_by_pair[(y, x)] = w
        standalone = simulate(returns[[y, x]].loc[w.index], w, CostModel())
        diagnostics.append({
            "pair": f"{y}/{x}",
            "coint_pvalue": cointegration_pvalue(prices[y], prices[x]),
            "half_life_days": half_life(spread),
            "sharpe": standalone.summary()["sharpe"],
        })

    diag = pd.DataFrame(diagnostics)
    print("Per-pair diagnostics (full-sample cointegration, walk-forward Sharpe):")
    print(diag.round(3).to_string(index=False))

    # Combine pairs into one portfolio, splitting capital equally across pairs.
    oos = pd.DatetimeIndex(sorted(set().union(*[w.index for w in weights_by_pair.values()])))
    combined = pd.DataFrame(0.0, index=oos, columns=tickers)
    for w in weights_by_pair.values():
        combined.loc[w.index, w.columns] = combined.loc[w.index, w.columns].add(
            w / len(PAIRS), fill_value=0.0
        )

    result = simulate(returns.loc[combined.index], combined, CostModel())
    summary = pd.Series(result.summary())
    print("\nCombined pairs portfolio (after costs):")
    print(summary.round(3).to_string())

    Path(RESULTS).mkdir(exist_ok=True)
    diag.round(4).to_csv(f"{RESULTS}/pairs_diagnostics.csv", index=False)
    summary.round(4).to_csv(f"{RESULTS}/pairs_metrics.csv", header=["value"])
    save_tearsheet(result, f"{RESULTS}/pairs_tearsheet.png")
    print(f"\nsaved metrics + tearsheet to {RESULTS}/")


if __name__ == "__main__":
    main()
