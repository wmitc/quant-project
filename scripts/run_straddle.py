"""Run the delta-hedged short straddle backtest end to end.

Loads SPX (^GSPC) and VIX (^VIX), sells a rolling 1-month ATM straddle priced
off VIX and delta-hedges it daily, then reports after-cost metrics, a
cost-sensitivity sweep over the assumed option bid-ask, and tail-risk stats
(the short-vol left tail is the whole point). Saves a tearsheet and a P&L
decomposition.

Limitation disclosed up front: options are priced synthetically off VIX with a
modelled spread rather than a real historical option chain, so the headline is
optimistic — the cost sweep is there to show how the edge erodes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quantbt import metrics
from quantbt.data import load_prices
from quantbt.plotting import save_tearsheet
from quantbt.strategies.short_vol import short_straddle

START, END = "2010-01-01", "2024-12-31"
CACHE = "data/spx_vix.csv"
RESULTS = "results"


def main():
    px = load_prices(["^GSPC", "^VIX"], START, END, CACHE)
    spx, vix = px["^GSPC"], px["^VIX"]
    print(f"data: {spx.dropna().index[0].date()} to {spx.dropna().index[-1].date()}")

    res = short_straddle(spx, vix)  # headline: 1 vol-pt spread, 0.5bp hedge
    summary = pd.Series(metrics.summarize(res.returns))
    tail = pd.Series(metrics.tail_stats(res.returns))
    print("\nHeadline (after costs, 1 vol-pt option spread):")
    print(summary.round(3).to_string())
    print("\nTail risk:")
    print(tail.round(4).to_string())

    sweep = {}
    for spread in (0.0, 0.5, 1.0, 2.0):
        r = short_straddle(spx, vix, option_spread_vol_pts=spread)
        sweep[f"{spread:g} vol-pt"] = metrics.summarize(r.returns)
    sweep = pd.DataFrame(sweep).T[["ann_return", "sharpe", "max_drawdown"]]
    print("\nCost-sensitivity sweep (option bid-ask in vol points):")
    print(sweep.round(3).to_string())

    Path(RESULTS).mkdir(exist_ok=True)
    summary.round(4).to_csv(f"{RESULTS}/straddle_metrics.csv", header=["value"])
    tail.round(4).to_csv(f"{RESULTS}/straddle_tail.csv", header=["value"])
    sweep.round(4).to_csv(f"{RESULTS}/straddle_cost_sweep.csv")
    save_tearsheet(res, f"{RESULTS}/straddle_tearsheet.png")

    cum = res.pnl[["option", "hedge", "cost"]].cumsum()
    ax = cum.plot(figsize=(10, 5))
    ax.set_title("Short straddle: cumulative P&L decomposition (per notional)")
    ax.set_ylabel("cumulative P&L / notional")
    ax.figure.tight_layout()
    ax.figure.savefig(f"{RESULTS}/straddle_pnl_decomp.png", dpi=120)
    plt.close(ax.figure)
    print(f"\nsaved metrics + plots to {RESULTS}/")


if __name__ == "__main__":
    main()
