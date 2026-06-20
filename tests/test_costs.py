import pytest

from quantbt.costs import CostModel


def test_total_bps_sums_components():
    m = CostModel(half_spread_bps=2.0, commission_bps=0.5, slippage_bps=1.0)
    assert m.total_bps == pytest.approx(3.5)


def test_cost_of_trading_full_book():
    # one-sided trade of the whole book (sum|dw| = 1) at 3.5 bps -> 0.00035
    m = CostModel(2.0, 0.5, 1.0)
    assert m.cost([1.0]) == pytest.approx(3.5e-4)


def test_cost_uses_absolute_traded_notional():
    # buying 0.5 and selling 0.5 trades 1.0 of notional
    m = CostModel(2.0, 0.5, 1.0)
    assert m.cost([0.5, -0.5]) == pytest.approx(3.5e-4)


def test_scaled_multiplies_all_components():
    m = CostModel(2.0, 0.5, 1.0).scaled(2.0)
    assert m.total_bps == pytest.approx(7.0)
    assert m.cost([1.0]) == pytest.approx(7.0e-4)
