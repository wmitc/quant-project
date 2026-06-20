"""Simple proportional transaction-cost model.

Costs are charged on the *traded* notional at each rebalance (the change in
weights), expressed in basis points. We bundle three effects:

- half_spread_bps: paying half the bid-ask spread to cross.
- commission_bps:  broker / exchange fees.
- slippage_bps:    market impact / not trading exactly at the mid.

It is intentionally simple. The point is to report returns *after* costs and to
let us stress-test how much of the edge survives when costs are higher.
"""
from __future__ import annotations

import numpy as np


class CostModel:
    """Proportional cost as basis points of traded notional."""

    def __init__(
        self,
        half_spread_bps: float = 2.0,
        commission_bps: float = 0.5,
        slippage_bps: float = 1.0,
    ):
        self.half_spread_bps = half_spread_bps
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    @property
    def total_bps(self) -> float:
        return self.half_spread_bps + self.commission_bps + self.slippage_bps

    def cost(self, weight_changes) -> float:
        """Fraction of capital lost to costs for the given per-asset trades.

        `weight_changes` is the vector of changes in portfolio weights; the
        traded notional is sum(|weight_changes|).
        """
        traded = float(np.abs(np.asarray(weight_changes, dtype=float)).sum())
        return traded * self.total_bps * 1e-4

    def scaled(self, factor: float) -> "CostModel":
        """Return a copy with all cost components multiplied by `factor`.

        Handy for cost-sensitivity sweeps, e.g. model.scaled(2.0) for 2x costs.
        """
        return CostModel(
            self.half_spread_bps * factor,
            self.commission_bps * factor,
            self.slippage_bps * factor,
        )
