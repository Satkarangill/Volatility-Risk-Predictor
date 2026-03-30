"""
Backtesting helpers for module-first strategy execution.

This module currently implements:
- Buy & Hold (equal weight, monthly rebalancing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and sanitize input price matrix."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("`prices` must be a non-empty pandas DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("`prices` must use a DatetimeIndex.")

    clean = prices.sort_index().copy()
    clean = clean.dropna(axis=1, how="all")
    if clean.empty:
        raise ValueError("`prices` has no valid columns after dropping empty assets.")
    return clean


def get_monthly_rebalance_dates(price_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return month-end rebalance dates aligned to available trading days.

    Uses the last available trading day for each month in the provided index.
    """
    if len(price_index) == 0:
        return pd.DatetimeIndex([])

    idx = pd.DatetimeIndex(sorted(price_index.unique()))
    month_ends = idx.to_series().groupby(idx.to_period("M")).max()
    rebalance_dates = pd.DatetimeIndex(month_ends.values)

    # Ensure the very first trading day is included for initial allocation.
    if idx[0] not in rebalance_dates:
        rebalance_dates = pd.DatetimeIndex([idx[0], *rebalance_dates])

    return rebalance_dates.sort_values()


def buy_and_hold_equal_weight(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build equal-weight target weights on monthly rebalance dates.

    Parameters
    ----------
    prices:
        Price DataFrame with DatetimeIndex and one column per asset.

    Returns
    -------
    pd.DataFrame
        Weights indexed by rebalance date, columns = assets.
    """
    clean_prices = _validate_prices(prices)
    rebalance_dates = get_monthly_rebalance_dates(clean_prices.index)

    n_assets = len(clean_prices.columns)
    equal_w = np.full(n_assets, 1.0 / n_assets)
    weights = pd.DataFrame(
        np.tile(equal_w, (len(rebalance_dates), 1)),
        index=rebalance_dates,
        columns=clean_prices.columns,
        dtype=float,
    )
    return weights


@dataclass
class BacktestConfig:
    """Configuration for strategy backtests."""

    transaction_cost: float = 0.0  # cost fraction per 1.0 turnover
    risk_free_rate: float = 0.0
    periods_per_year: int = 252


class Backtester:
    """Minimal backtester for module-first strategy execution."""

    def __init__(self, prices: pd.DataFrame, config: Optional[BacktestConfig] = None):
        self.prices = _validate_prices(prices)
        self.returns = self.prices.pct_change().dropna()
        self.config = config or BacktestConfig()

    @staticmethod
    def _align_weights_to_returns(
        returns: pd.DataFrame, rebalance_weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align sparse rebalance weights to daily returns via forward fill.
        """
        daily_w = rebalance_weights.reindex(returns.index).ffill().fillna(0.0)
        return daily_w

    @staticmethod
    def _calculate_turnover(rebalance_weights: pd.DataFrame) -> pd.Series:
        """Turnover at rebalance dates as sum(abs(delta weights))."""
        turnover = rebalance_weights.diff().abs().sum(axis=1).fillna(0.0)
        return turnover

    def run_buy_and_hold_equal_weight(self) -> Dict[str, object]:
        """
        Run Buy & Hold with equal-weight monthly rebalancing.

        Returns
        -------
        Dict[str, object]
            keys include weights, returns, cumulative_returns, turnover, costs, metrics.
        """
        rebalance_weights = buy_and_hold_equal_weight(self.prices)
        daily_weights = self._align_weights_to_returns(self.returns, rebalance_weights)

        # Strategy gross daily return
        strategy_gross = (self.returns * daily_weights).sum(axis=1)

        # Transaction cost applied only on rebalance dates
        turnover_rebalance = self._calculate_turnover(rebalance_weights)
        turnover_daily = turnover_rebalance.reindex(strategy_gross.index).fillna(0.0)
        daily_cost = turnover_daily * float(self.config.transaction_cost)
        strategy_net = strategy_gross - daily_cost

        cumulative_returns = (1.0 + strategy_net).cumprod()

        # Metrics
        periods = float(self.config.periods_per_year)
        ann_return = (
            (1.0 + strategy_net).prod() ** (periods / max(len(strategy_net), 1)) - 1.0
            if len(strategy_net) > 0
            else 0.0
        )
        ann_vol = float(strategy_net.std() * np.sqrt(periods)) if len(strategy_net) > 1 else 0.0
        sharpe = (
            (ann_return - float(self.config.risk_free_rate)) / ann_vol
            if ann_vol > 0
            else np.nan
        )
        total_return = float(cumulative_returns.iloc[-1] - 1.0) if len(cumulative_returns) else 0.0

        metrics = {
            "total_return": total_return,
            "annualized_return": float(ann_return),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
            "total_turnover": float(turnover_rebalance.sum()),
            "average_rebalance_turnover": float(turnover_rebalance.mean()),
            "total_transaction_cost": float(daily_cost.sum()),
        }

        return {
            "rebalance_weights": rebalance_weights,
            "daily_weights": daily_weights,
            "strategy_returns": strategy_net,
            "cumulative_returns": cumulative_returns,
            "turnover_rebalance": turnover_rebalance,
            "turnover_daily": turnover_daily,
            "transaction_costs_daily": daily_cost,
            "metrics": metrics,
        }
