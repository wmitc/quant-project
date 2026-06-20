# quantbt — a walk-forward backtester with two strategies

A small, reproducible backtesting framework for systematic equity strategies,
built to measure performance the way a trading desk would: **walk-forward, after
costs, with turnover and capacity** — not just a pretty gross equity curve.

The reusable engine is the point; the two strategies (cross-sectional momentum
and ETF pairs / statistical arbitrage) are plug-ins that exercise it. The honest
finding is as interesting as the numbers: naive versions of both effects deliver
thin edge after realistic costs, and the framework makes that visible.

## Why

It's easy to produce a backtest that looks great on gross returns and falls
apart once you account for transaction costs, look-ahead, and capacity. I wanted
to build the disciplined evaluation machinery first — realistic costs, no
look-ahead, walk-forward out-of-sample testing — and then run two well-known
strategies through it and report whatever I found, good or bad.

## Install & reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q                      # 27 unit tests
python scripts/run_momentum.py # momentum backtest + plots
python scripts/run_pairs.py    # pairs / stat-arb backtest + plots
```

Price data is cached as CSV under `data/` and committed, so the scripts run
offline and reproduce the committed results in `results/`.

For a narrative walkthrough of each strategy (signal → weights → after-cost
backtest, with inline plots), see the notebooks in `notebooks/`.

## Methodology

- **No look-ahead.** Positions decided at a day's close earn the *next* day's
  return; signals only ever use past data. There's an explicit unit test for it.
- **Walk-forward.** Parameters are estimated on a rolling training window and
  traded on the following, non-overlapping out-of-sample window.
- **After costs.** A proportional cost model (half-spread + commission +
  slippage, in bps) is charged on traded notional at every rebalance. Headline
  numbers are net, and a 1×/2×/3× **cost-sensitivity sweep** shows how the edge
  decays.
- **Capacity & turnover.** Reported alongside returns, because an edge that's
  real but un-tradeable at size isn't an edge.

## Results

### 1. Cross-sectional momentum

12-1 momentum (trailing 12-month return skipping the last month) on 90 liquid US
large-caps, 2015–2024. Long top decile / short bottom decile, equal-weighted,
dollar-neutral, monthly rebalance.

| Metric | After costs |
|---|---|
| Annualized return | 0.5% |
| Annualized vol | 12.9% |
| Sharpe | 0.10 |
| Sortino | 0.14 |
| Max drawdown | −33.3% |
| Hit rate | 47.0% |
| Monthly turnover | 27.1% |
| Capacity (1% ADV) | ~$34M |

Cost-sensitivity sweep:

| Costs | Ann. return | Sharpe | Max DD |
|---|---|---|---|
| 0× | 0.7% | 0.12 | −32.9% |
| 1× | 0.5% | 0.10 | −33.3% |
| 2× | 0.3% | 0.09 | −33.7% |
| 3× | 0.1% | 0.07 | −34.0% |

![Momentum tearsheet](results/momentum_tearsheet.png)

**Read:** the edge is thin even gross (Sharpe ~0.12) and barely survives costs.
~90 mega-caps that mostly rose together offer little cross-sectional dispersion
to exploit, and 2015–2024 was a weak stretch for momentum. Capacity is bound by
the least-liquid name in the universe — dropping a few of those would raise it
substantially.

### 2. ETF pairs / statistical arbitrage

Five economically-motivated pairs (chosen a priori, not screened across all
combinations). Each is traded only when it tests as **cointegrated in the
training window** (Engle-Granger, p < 0.05); otherwise the strategy stands aside.
We fit the hedge ratio and spread statistics in-sample and trade the spread
z-score out-of-sample (enter at |z|>2, exit near 0, stop at |z|>4).

Per-pair diagnostics (full-sample cointegration, walk-forward Sharpe):

| Pair | Cointegration p | Half-life (days) | Sharpe |
|---|---|---|---|
| EWA/EWC | 0.019 | 36 | 0.22 |
| GDX/GDXJ | 0.594 | 174 | −2.43 |
| XLE/XOP | 0.955 | 1267 | 0.32 |
| SMH/SOXX | 1.000 | ∞ | 0.46 |
| SPY/IVV | 0.497 | 10 | −1.60 |

Combined, equal-split portfolio (after costs):

| Metric | After costs |
|---|---|
| Annualized return | 0.2% |
| Annualized vol | 1.1% |
| Sharpe | 0.18 |
| Sortino | 0.30 |
| Max drawdown | −3.3% |
| Turnover (daily) | 1.5% |

![Pairs tearsheet](results/pairs_tearsheet.png)

**Read:** only EWA/EWC (Australia vs Canada) is robustly cointegrated. The
cointegration gate keeps the book flat much of the time, producing a low-vol,
low-return hedged portfolio — thin edge after costs. SPY/IVV is a nice cautionary
case: nearly identical trackers whose spread is too tight to trade profitably
once you pay to cross it.

## What it shows (and where it breaks)

- **Momentum.** Decays under costs; suffers sharp reversals ("momentum crashes")
  visible in the drawdown. The fixed, present-day universe also carries
  **survivorship bias**, so even this modest result is optimistic.
- **Pairs.** Cointegration that holds in-sample often weakens out-of-sample;
  gating on it is what separates a real relationship from a spurious one. In
  liquid ETFs most of the relative-value edge appears arbitraged away.
- **Both.** The value isn't a magic Sharpe — it's that the measurement is honest:
  net of costs, free of look-ahead, and sized against capacity.

## Limitations

- Daily close-to-close; weights held fixed between rebalances (no intra-period
  drift modeled).
- Proportional cost model, not a full market-impact model.
- Survivorship bias in the momentum universe (disclosed, not corrected).
- A priori pairs to sidestep multiple-testing; a broad cointegration screen would
  need a correction.

## Layout

```
src/quantbt/      data, metrics, costs, backtest engine, plotting, strategies/
scripts/          run_momentum.py, run_pairs.py
notebooks/        narrative walkthroughs (momentum, pairs)
tests/            27 unit tests (metrics, costs, engine, strategies)
data/             committed price/volume CSVs
results/          committed metrics tables and tearsheets
```
