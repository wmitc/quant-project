"""Black-Scholes pricing and greeks for European options.

Closed-form prices and greeks under the standard Black-Scholes model with a
continuously-compounded rate `r` (default 0, a fine approximation for short-dated
index options) and no dividends. `sigma` is annualized volatility and `T` is the
time to expiry in years.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, sigma, r=0.0):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def price(S, K, T, sigma, r=0.0, option="call") -> float:
    """European option price. `option` is 'call' or 'put'."""
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return float(intrinsic)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    disc = np.exp(-r * T)
    if option == "call":
        return float(S * norm.cdf(d1) - K * disc * norm.cdf(d2))
    return float(K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))


def delta(S, K, T, sigma, r=0.0, option="call") -> float:
    if T <= 0:
        if option == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return float(norm.cdf(d1)) if option == "call" else float(norm.cdf(d1) - 1.0)


def gamma(S, K, T, sigma, r=0.0) -> float:
    """Gamma is the same for calls and puts."""
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def vega(S, K, T, sigma, r=0.0) -> float:
    """Sensitivity to a 1.0 (i.e. 100 vol-point) change in sigma."""
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return float(S * norm.pdf(d1) * np.sqrt(T))


def theta(S, K, T, sigma, r=0.0, option="call") -> float:
    """Per-year theta (divide by 252 for per-trading-day decay)."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    disc = np.exp(-r * T)
    decay = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option == "call":
        return float(decay - r * K * disc * norm.cdf(d2))
    return float(decay + r * K * disc * norm.cdf(-d2))


def straddle_price(S, K, T, sigma, r=0.0) -> float:
    """Price of an at-the-money-ish straddle: a call plus a put at strike K."""
    return price(S, K, T, sigma, r, "call") + price(S, K, T, sigma, r, "put")


def straddle_delta(S, K, T, sigma, r=0.0) -> float:
    """Net delta of a long straddle (= 2*N(d1) - 1)."""
    return delta(S, K, T, sigma, r, "call") + delta(S, K, T, sigma, r, "put")
