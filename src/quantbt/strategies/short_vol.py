"""Delta-hedged short straddle — harvesting the variance risk premium.

We repeatedly sell a ~1-month at-the-money SPX straddle priced off VIX (used as
the implied vol), and delta-hedge it daily against the underlying. Selling
implied while paying out realized captures the variance risk premium; the daily
hedge strips out market direction so the P&L is (mostly) a bet on implied vs
realized vol.

The daily P&L decomposes into:
  - option: mark-to-market of the short straddle (premium decay minus the cost
    of realized moves),
  - hedge:  the delta hedge against the underlying,
  - cost:   option bid-ask paid to open each straddle, plus hedge trading cost.

Everything is expressed as a fraction of the underlying notional, so the result
plugs straight into `quantbt.metrics`. Synthetic pricing off VIX with a modelled
spread is a simplification (no real option chain); it is disclosed, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbt import blackscholes as bs


@dataclass
class StraddleResult:
    returns: pd.Series        # net (after-cost) daily returns, fraction of notional
    gross_returns: pd.Series  # before costs
    equity: pd.Series         # net equity curve
    pnl: pd.DataFrame         # columns: option, hedge, cost, total


def short_straddle(
    spx: pd.Series,
    vix: pd.Series,
    cycle: int = 21,
    r: float = 0.0,
    option_spread_vol_pts: float = 1.0,
    hedge_cost_bps: float = 0.5,
    delta_hedge: bool = True,
) -> StraddleResult:
    """Backtest a rolling, delta-hedged short ATM straddle.

    spx: daily underlying closes. vix: daily implied vol in points (e.g. 18.5).
    A new `cycle`-day straddle is sold each cycle and held to expiry. The
    `option_spread_vol_pts` bid-ask (in vol points) is paid when selling; the
    hedge pays `hedge_cost_bps` on each share traded.
    """
    df = pd.concat([spx.rename("S"), vix.rename("v")], axis=1).dropna()
    S = df["S"].to_numpy(float)
    sig = df["v"].to_numpy(float) / 100.0
    n = len(S)
    option = np.zeros(n)
    hedge = np.zeros(n)
    cost = np.zeros(n)

    i = 0
    while i < n - 1:
        K = S[i]
        T0 = cycle / 252
        v_prev = bs.straddle_price(S[i], K, T0, sig[i], r)
        # pay half the bid-ask (in vol points) on the straddle's vega when selling
        straddle_vega = 2 * bs.vega(S[i], K, T0, sig[i], r)
        cost[i] += 0.5 * option_spread_vol_pts * 0.01 * straddle_vega / S[i]
        shares = bs.straddle_delta(S[i], K, T0, sig[i], r) if delta_hedge else 0.0
        cost[i] += abs(shares) * hedge_cost_bps * 1e-4

        for j in range(i + 1, min(i + cycle + 1, n)):
            T = max((cycle - (j - i)) / 252, 0.0)
            v_now = bs.straddle_price(S[j], K, T, sig[j], r)
            option[j] += -(v_now - v_prev) / S[i]          # short straddle MTM
            hedge[j] += shares * (S[j] - S[j - 1]) / S[i]
            v_prev = v_now
            new_shares = bs.straddle_delta(S[j], K, T, sig[j], r) if (delta_hedge and T > 0) else 0.0
            cost[j] += abs(new_shares - shares) * hedge_cost_bps * 1e-4
            shares = new_shares
        i += cycle

    pnl = pd.DataFrame({"option": option, "hedge": hedge, "cost": -cost}, index=df.index)
    pnl["total"] = pnl["option"] + pnl["hedge"] + pnl["cost"]
    gross = pnl["option"] + pnl["hedge"]
    equity = (1.0 + pnl["total"]).cumprod()
    return StraddleResult(returns=pnl["total"], gross_returns=gross, equity=equity, pnl=pnl)
