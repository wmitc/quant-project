import pandas as pd
import pytest

from quantbt.backtest import daily_returns, simulate, walk_forward_windows
from quantbt.costs import CostModel

NO_COST = CostModel(0.0, 0.0, 0.0)


def _dates(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_daily_returns_basic():
    d = _dates(3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=d)
    r = daily_returns(prices)
    assert r["A"].iloc[1] == pytest.approx(0.10)
    assert r["A"].iloc[2] == pytest.approx(-0.10)


def test_no_lookahead_same_day_return_not_captured():
    d = _dates(3)
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0]}, index=d)
    # decide to hold A at the close of the +10% day
    weights = pd.DataFrame({"A": [1.0]}, index=[d[1]])
    res = simulate(returns, weights, NO_COST)
    assert res.gross_returns.loc[d[1]] == pytest.approx(0.0)
    assert res.gross_returns.loc[d[2]] == pytest.approx(0.0)


def test_weight_earns_next_day_return():
    d = _dates(3)
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0]}, index=d)
    weights = pd.DataFrame({"A": [1.0]}, index=[d[0]])  # set at close of d0
    res = simulate(returns, weights, NO_COST)
    assert res.gross_returns.loc[d[1]] == pytest.approx(0.10)


def test_initial_cost_charged_from_empty_book():
    d = _dates(2)
    returns = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]}, index=d)
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[d[0]])
    res = simulate(returns, weights, CostModel(2.0, 0.5, 1.0))  # 3.5 bps
    assert res.costs.loc[d[0]] == pytest.approx(3.5e-4)


def test_cost_charged_on_rebalance():
    d = _dates(3)
    returns = pd.DataFrame({"A": [0.0] * 3, "B": [0.0] * 3}, index=d)
    weights = pd.DataFrame({"A": [0.5, 0.0], "B": [0.5, 1.0]}, index=[d[0], d[1]])
    res = simulate(returns, weights, CostModel(2.0, 0.5, 1.0))
    # rebalance d1 trades A -0.5 and B +0.5 -> notional 1.0 -> 3.5 bps
    assert res.costs.loc[d[1]] == pytest.approx(3.5e-4)


def test_compounding_matches_constant_returns():
    d = _dates(5)
    returns = pd.DataFrame({"A": [0.01] * 5}, index=d)
    weights = pd.DataFrame({"A": [1.0]}, index=[d[0]])
    res = simulate(returns, weights, NO_COST)
    # held from d1..d4 -> four days of +1% (d0 is not earned)
    assert res.equity.iloc[-1] == pytest.approx(1.01 ** 4)


def test_walk_forward_windows_non_overlapping():
    idx = _dates(10)
    windows = list(walk_forward_windows(idx, train=4, test=2))
    assert len(windows) == 3
    _, test0 = windows[0]
    assert list(test0) == list(idx[4:6])
    all_test = [t for _, te in windows for t in te]
    assert len(all_test) == len(set(all_test))  # test windows don't overlap
