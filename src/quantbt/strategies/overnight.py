"""The overnight effect: buy at the close, sell at the next open.

Historically almost all of a broad index's return has accrued *overnight* (from
one day's close to the next day's open), while the regular session is roughly
flat. This strategy simply holds the overnight move and is flat intraday — one
round-trip per day, so transaction costs stay survivable.

It's deliberately simple: the whole thesis is one sentence, and the interesting
part is the decomposition of total return into its overnight and intraday pieces.
"""
from __future__ import annotations

import pandas as pd


def overnight_returns(ohlc: pd.DataFrame) -> pd.Series:
    """Close-to-next-open returns — the overnight holding period."""
    return ohlc["Open"] / ohlc["Close"].shift(1) - 1.0


def intraday_returns(ohlc: pd.DataFrame) -> pd.Series:
    """Open-to-close returns — the regular trading session."""
    return ohlc["Close"] / ohlc["Open"] - 1.0
