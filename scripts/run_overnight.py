"""Run the overnight-effect backtest on SPY.

Decomposes SPY's total return into its overnight (close -> next open) and
intraday (open -> close) pieces, then evaluates the simple "buy at close, sell
at open" strategy after a per-day round-trip cost. One trade a day, so costs are
survivable — and SPY's real round-trip is well under the 1bp shown here.

Caveats: holding overnight bears gap risk, and you cannot trade exactly at the
official close/open prints — both disclosed, not hidden.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quantbt import metrics
from quantbt.data import load_ohlc
from quantbt.plotting import save_tearsheet
from quantbt.strategies.overnight import intraday_returns, overnight_returns

START, END = "2005-01-01", "2024-12-31"
CACHE = "data/spy_ohlc.csv"
RESULTS = "results"


def main():
    ohlc = load_ohlc("SPY", START, END, CACHE)
    overnight = overnight_returns(ohlc)
    intraday = intraday_returns(ohlc)
    buy_hold = ohlc["Close"].pct_change()

    decomp = pd.DataFrame({
        "buy_hold": metrics.summarize(buy_hold),
        "intraday": metrics.summarize(intraday),
        "overnight": metrics.summarize(overnight),
    }).T
    print("Return decomposition (SPY, gross):")
    print(decomp.round(3).to_string())

    net = (overnight - 1e-4).dropna()  # ~1 round-trip/day
    print("\nOvernight strategy, after ~1bp/day:")
    print(pd.Series(metrics.summarize(net)).round(3).to_string())

    sweep = {f"{b:g}bp": metrics.summarize(overnight - b * 1e-4) for b in (0.0, 0.5, 1.0, 2.0)}
    sweep = pd.DataFrame(sweep).T[["ann_return", "sharpe", "max_drawdown"]]
    print("\nCost-sensitivity sweep (round-trip/day):")
    print(sweep.round(3).to_string())

    Path(RESULTS).mkdir(exist_ok=True)
    decomp.round(4).to_csv(f"{RESULTS}/overnight_decomposition.csv")
    sweep.round(4).to_csv(f"{RESULTS}/overnight_cost_sweep.csv")
    equity = (1 + net).cumprod()
    save_tearsheet(SimpleNamespace(equity=equity, returns=net), f"{RESULTS}/overnight_tearsheet.png")
    print(f"\nsaved metrics + tearsheet to {RESULTS}/")


if __name__ == "__main__":
    main()
