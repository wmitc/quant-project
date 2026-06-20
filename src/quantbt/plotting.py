"""Plots for a backtest result: equity curve, drawdown, and rolling Sharpe."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantbt import metrics


def plot_equity(equity, ax=None, label=None):
    """Equity curve on a log scale (compounding reads better in log)."""
    ax = ax or plt.gca()
    ax.plot(equity.index, equity.values, label=label)
    ax.set_yscale("log")
    ax.set_title("Equity curve (after costs)")
    ax.set_ylabel("Growth of $1")
    return ax


def plot_drawdown(equity, ax=None):
    """Underwater plot: percent below the running high-water mark."""
    ax = ax or plt.gca()
    drawdown = equity / equity.cummax() - 1.0
    ax.fill_between(drawdown.index, drawdown.values, 0.0, color="tab:red", alpha=0.4)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    return ax


def plot_rolling_sharpe(returns, window=252, ax=None):
    """Rolling annualized Sharpe to show stability over time."""
    ax = ax or plt.gca()
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    rolling = np.sqrt(metrics.TRADING_DAYS) * mean / std
    ax.plot(rolling.index, rolling.values, color="tab:green")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling Sharpe ({window}d)")
    ax.set_ylabel("Sharpe")
    return ax


def save_tearsheet(result, path: str | Path):
    """Stack equity, drawdown and rolling Sharpe into one figure and save it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    plot_equity(result.equity, ax=axes[0])
    plot_drawdown(result.equity, ax=axes[1])
    plot_rolling_sharpe(result.returns, ax=axes[2])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
