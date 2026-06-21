import pandas as pd
import pytest

from quantbt import metrics


def test_skew_sign():
    pos = pd.Series([-0.01] * 9 + [0.5])   # one big up move -> positive skew
    neg = pd.Series([0.01] * 9 + [-0.5])   # one big down move -> negative skew
    assert metrics.tail_stats(pos)["skew"] > 0
    assert metrics.tail_stats(neg)["skew"] < 0


def test_worst_day_and_var_ordering():
    r = pd.Series([0.01, -0.02, 0.0, -0.05, 0.03, -0.01, 0.02, 0.01, -0.03, 0.0])
    ts = metrics.tail_stats(r, levels=(0.9,))
    assert ts["worst_day"] == pytest.approx(-0.05)
    # the tail average (CVaR) is at least as bad as the quantile (VaR), both <= 0
    assert ts["cvar_90"] <= ts["var_90"] <= 0
