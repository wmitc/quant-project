import numpy as np
import pandas as pd
import pytest

from quantbt.strategies.pairs import (
    cointegration_pvalue,
    half_life,
    hedge_ratio,
    positions_from_zscore,
)


def test_hedge_ratio_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.01, size=500)
    beta, alpha = hedge_ratio(y, x)
    assert beta == pytest.approx(2.0, abs=0.01)
    assert alpha == pytest.approx(1.0, abs=0.01)


def test_half_life_positive_for_mean_reverting():
    # AR(1) with phi=0.5 is mean-reverting -> finite, positive half-life
    rng = np.random.default_rng(1)
    s = [0.0]
    for _ in range(2000):
        s.append(0.5 * s[-1] + rng.normal(scale=0.1))
    hl = half_life(pd.Series(s))
    assert np.isfinite(hl)
    assert hl > 0


def test_half_life_infinite_for_random_walk():
    rng = np.random.default_rng(2)
    walk = pd.Series(np.cumsum(rng.normal(size=2000)))
    assert half_life(walk) == float("inf") or half_life(walk) > 100


def test_positions_state_machine():
    z = pd.Series([0.0, 2.5, 1.0, 0.3, -2.5, -0.1])
    pos = positions_from_zscore(z, entry=2.0, exit_threshold=0.5, stop=4.0)
    assert list(pos) == [0, -1, -1, 0, 1, 0]


def test_cointegration_pvalue_detects_relationship():
    rng = np.random.default_rng(3)
    x = pd.Series(np.cumsum(rng.normal(size=600)))
    y = 2.0 * x + rng.normal(scale=0.5, size=600)  # cointegrated with x
    independent = pd.Series(np.cumsum(rng.normal(size=600)))
    assert cointegration_pvalue(y, x) < 0.05
    assert cointegration_pvalue(x, independent) > 0.05
