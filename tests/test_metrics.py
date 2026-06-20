import numpy as np
import pandas as pd
import pytest

from quantbt import metrics


def test_max_drawdown_simple():
    # equity: 1 -> 1.1 -> 0.55, so the worst drawdown is 0.55/1.1 - 1 = -0.5
    r = pd.Series([0.1, -0.5])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_hit_rate():
    r = pd.Series([0.1, -0.2, 0.3, 0.0])  # 2 of 4 strictly positive
    assert metrics.hit_rate(r) == pytest.approx(0.5)


def test_annualized_return_doubles_in_one_year():
    # 4 quarterly periods, total growth x2 over exactly 1 year -> +100%
    r = pd.Series([1.0, 0.0, 0.0, 0.0])
    assert metrics.annualized_return(r, periods_per_year=4) == pytest.approx(1.0)


def test_annualized_volatility_matches_manual():
    r = pd.Series([0.01, -0.01, 0.01, -0.01])
    expected = r.std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_volatility(r) == pytest.approx(expected)


def test_sharpe_matches_formula():
    r = pd.Series([0.02, 0.0, 0.02, 0.0])
    expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)


def test_sortino_no_downside_is_nan():
    r = pd.Series([0.01, 0.0, 0.02])  # no negative excess returns
    assert np.isnan(metrics.sortino_ratio(r))


def test_turnover_half_book():
    # rebalance from 50/50 to 0/100: |delta| = 0.5 + 0.5 = 1.0, *0.5 = 0.5
    w = pd.DataFrame([[0.5, 0.5], [0.0, 1.0]], columns=["A", "B"])
    assert metrics.turnover(w) == pytest.approx(0.5)


def test_capacity_binding_name():
    # max |dw| is 0.5 for both names; A has the smaller ADV so it binds
    w = pd.DataFrame([[0.5, 0.5], [0.0, 0.5]], columns=["A", "B"])
    adv = pd.Series({"A": 1e8, "B": 1e9})
    # A: 0.01 * 1e8 / 0.5 = 2e6 ; B: 0.01 * 1e9 / 0.5 = 2e7 -> min = 2e6
    assert metrics.capacity(w, adv, participation=0.01) == pytest.approx(2e6)
