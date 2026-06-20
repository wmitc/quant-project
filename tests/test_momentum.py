import numpy as np
import pandas as pd
import pytest

from quantbt.strategies.momentum import (
    momentum_signal,
    momentum_weights,
    month_end_rebalances,
)


def _bdays(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_momentum_signal_uses_only_past():
    d = _bdays(5)
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=d)
    sig = momentum_signal(prices, lookback=3, skip=1)
    # signal at d3 = price[d2] / price[d0] - 1 = 12/10 - 1
    assert sig["A"].loc[d[3]] == pytest.approx(12 / 10 - 1)
    # earliest dates have no signal (not enough history)
    assert np.isnan(sig["A"].loc[d[0]])


def test_month_end_rebalances_one_per_month():
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    rebal = month_end_rebalances(idx)
    assert len(rebal) == 3  # Jan, Feb, Mar
    assert rebal[0] == idx[idx.month == 1][-1]


def test_momentum_weights_dollar_neutral_and_ranked():
    d = _bdays(300)
    # 10 assets with strictly increasing trend strength
    rates = np.linspace(0.0001, 0.001, 10)
    data = {f"S{i}": 100 * np.cumprod(1 + np.full(300, r)) for i, r in enumerate(rates)}
    prices = pd.DataFrame(data, index=d)

    w = momentum_weights(prices, [d[-1]], lookback=252, skip=21, quantile=0.2)
    row = w.iloc[0]

    assert row.sum() == pytest.approx(0.0)          # dollar-neutral
    assert row[row > 0].sum() == pytest.approx(0.5)  # long leg
    assert row[row < 0].sum() == pytest.approx(-0.5)  # short leg
    # strongest trend is long, weakest is short
    assert row["S9"] > 0
    assert row["S0"] < 0
