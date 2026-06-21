import numpy as np
import pandas as pd

from quantbt.strategies.short_vol import short_straddle


def _path(n, vol=0.01, drift=0.0, seed=0, start=2000.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    rets = rng.normal(drift, vol, n)
    return pd.Series(start * np.cumprod(1 + rets), index=dates)


def _beta(y, x):
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    return np.cov(df["y"], df["x"])[0, 1] / np.var(df["x"])


def test_premium_captured_when_underlying_flat():
    # underlying never moves -> short straddle just collects time decay
    dates = pd.bdate_range("2015-01-01", periods=100)
    spx = pd.Series(2000.0, index=dates)
    vix = pd.Series(20.0, index=dates)
    res = short_straddle(spx, vix, option_spread_vol_pts=0.0, hedge_cost_bps=0.0)
    assert res.gross_returns.sum() > 0


def test_delta_hedge_reduces_directional_exposure():
    spx = _path(400, vol=0.01, drift=0.0005, seed=1)  # trending
    vix = pd.Series(18.0, index=spx.index)
    hedged = short_straddle(spx, vix, delta_hedge=True)
    unhedged = short_straddle(spx, vix, delta_hedge=False)
    spx_ret = spx.pct_change()
    assert abs(_beta(hedged.gross_returns, spx_ret)) < abs(_beta(unhedged.gross_returns, spx_ret))


def test_costs_reduce_pnl_monotonically():
    spx = _path(300, vol=0.01, seed=2)
    vix = pd.Series(18.0, index=spx.index)
    cheap = short_straddle(spx, vix, option_spread_vol_pts=0.5).returns.sum()
    pricey = short_straddle(spx, vix, option_spread_vol_pts=3.0).returns.sum()
    assert pricey < cheap


def test_pnl_components_sum_to_total():
    spx = _path(150, seed=3)
    vix = pd.Series(19.0, index=spx.index)
    p = short_straddle(spx, vix).pnl
    assert np.allclose(p["total"], p["option"] + p["hedge"] + p["cost"])
