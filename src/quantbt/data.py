"""Price data load and cache.

Downloads daily split and dividend-adjusted closing prices and caches them to
parquet so that backtests are reproducible.

Note on survivor-ship bias: Some tickets may not exist in certain date ranges
due to events like bankrupties/de-listings or IPOs. In the case of bankrupties,
backtested returns will be biased upwards.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def _cache_path(tickers: list[str], start: str, end: str, cache_dir: Path) -> Path:
    """Map a price request to a stable cache filename."""
    key = "_".join(sorted(tickers)) + f"_{start}_{end}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]

    return cache_dir / f"prices_{digest}.parquet"

def clean_prices(prices: pd.DataFrame, max_missing: float = 0.1) -> pd.DataFrame:
    """Sort by date and drop tickers for which there is 
    a significant amount of missing datea"""
    prices = prices.sort_index()
    keep = prices.columns[prices.isna().mean() <= max_missing]

    return prices[keep]

def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = DATA_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Return a (dates x tickers) frame of adjusted close prices.

    Cached to parquet and keyed by the request; pass force=True to re-download.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(tickers, start, end, cache_dir)

    if path.exists() and not force:
        return pd.read_parquet(path)

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    # With multiple tickers yfinance returns a (field, ticker) column MultiIndex.
    prices = raw["Close"].copy()
    if isinstance(prices, pd.Series):  # single-ticker case
        prices = prices.to_frame(tickers[0])

    prices = clean_prices(prices)
    prices.to_parquet(path)
    return prices