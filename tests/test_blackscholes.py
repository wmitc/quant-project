import pytest

from quantbt import blackscholes as bs


def test_atm_call_known_value():
    # S=K=100, T=1, sigma=0.2, r=0 -> classic ATM call price ~ 7.9656
    assert bs.price(100, 100, 1.0, 0.2, option="call") == pytest.approx(7.9656, abs=1e-3)


def test_put_call_parity():
    # with r=0: C - P = S - K
    S, K, T, sig = 100, 95, 0.5, 0.25
    c = bs.price(S, K, T, sig, option="call")
    p = bs.price(S, K, T, sig, option="put")
    assert c - p == pytest.approx(S - K)


def test_straddle_price_is_call_plus_put():
    S, K, T, sig = 100, 100, 0.25, 0.2
    expected = bs.price(S, K, T, sig, option="call") + bs.price(S, K, T, sig, option="put")
    assert bs.straddle_price(S, K, T, sig) == pytest.approx(expected)


def test_atm_straddle_delta_near_zero():
    # at the money, net straddle delta is small (calls and puts roughly offset)
    assert abs(bs.straddle_delta(100, 100, 21 / 252, 0.2)) < 0.05


def test_call_put_delta_relationship():
    # delta_call - delta_put = 1 (with r=0)
    S, K, T, sig = 100, 105, 0.3, 0.2
    assert bs.delta(S, K, T, sig, option="call") - bs.delta(S, K, T, sig, option="put") == pytest.approx(1.0)


def test_gamma_and_vega_positive():
    assert bs.gamma(100, 100, 0.25, 0.2) > 0
    assert bs.vega(100, 100, 0.25, 0.2) > 0


def test_long_option_theta_negative():
    # a long option loses value as time passes
    assert bs.theta(100, 100, 0.25, 0.2, option="call") < 0
    assert bs.theta(100, 100, 0.25, 0.2, option="put") < 0


def test_expired_option_is_intrinsic():
    assert bs.price(110, 100, 0.0, 0.2, option="call") == pytest.approx(10.0)
    assert bs.price(90, 100, 0.0, 0.2, option="put") == pytest.approx(10.0)
