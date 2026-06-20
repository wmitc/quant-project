"""Price data loading and caching.

Downloads daily split- and dividend-adjusted closing prices from Yahoo Finance
and caches them to CSV so backtests are reproducible without re-downloading.

Note on survivorship bias: a fixed list of tickers that exist today excludes
names that were delisted or went bankrupt, which biases backtested returns
upward. We use fixed universes and flag this caveat in the README.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def clean_prices(prices: pd.DataFrame, max_missing: float = 0.1) -> pd.DataFrame:
    """Sort by date and drop tickers with too much missing data.

    We don't forward-fill: inventing prices on non-trading days could leak
    information. Tickers missing more than `max_missing` of their observations
    are dropped as unreliable.
    """
    prices = prices.sort_index()
    keep = prices.columns[prices.isna().mean() <= max_missing]
    return prices[keep]


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_file: str | Path,
    force: bool = False,
) -> pd.DataFrame:
    """Return a (dates x tickers) frame of adjusted close prices.

    Reads from `cache_file` (a CSV) if it exists, otherwise downloads the data,
    saves it, and returns it. Pass force=True to re-download.
    """
    cache_file = Path(cache_file)
    if cache_file.exists() and not force:
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # With multiple tickers yfinance returns a (field, ticker) column MultiIndex.
    prices = raw["Close"].copy()
    if isinstance(prices, pd.Series):  # single ticker
        prices = prices.to_frame(tickers[0])

    prices = clean_prices(prices)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_file)
    return prices
