import pandas as pd
import pytest

from quantbt.strategies.overnight import intraday_returns, overnight_returns


def _ohlc():
    return pd.DataFrame(
        {"Open": [100.0, 102.0, 101.0], "Close": [101.0, 103.0, 100.0]},
        index=pd.bdate_range("2020-01-01", periods=3),
    )


def test_overnight_return_is_close_to_next_open():
    on = overnight_returns(_ohlc())
    # day 2 overnight = open[1] / close[0] - 1 = 102/101 - 1
    assert on.iloc[1] == pytest.approx(102 / 101 - 1)


def test_intraday_return_is_open_to_close():
    intra = intraday_returns(_ohlc())
    # day 1 intraday = close[0] / open[0] - 1 = 101/100 - 1
    assert intra.iloc[0] == pytest.approx(101 / 100 - 1)


def test_overnight_and_intraday_compound_to_close_to_close():
    ohlc = _ohlc()
    on = overnight_returns(ohlc)
    intra = intraday_returns(ohlc)
    close_to_close = ohlc["Close"] / ohlc["Close"].shift(1) - 1
    combined = (1 + on) * (1 + intra) - 1
    assert combined.iloc[1] == pytest.approx(close_to_close.iloc[1])
