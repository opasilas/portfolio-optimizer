"""
portfolio_backtester.py
------------------------
Standalone walk-forward portfolio backtester that consumes the covariance
forecasts persisted by `main.py` (`outputs/forecasts.pkl`) and simulates
weekly portfolio rebalancing for each volatility model, under two
objective functions and three transaction-cost regimes, benchmarked
against a static equal-weight baseline.

At each forecast date t, for every model (GARCH, Markov, XGBoost) and
every objective we solve one of:

    Max-Sharpe:  maximise  (w'mu - Rf) / sqrt(w' Sigma_t w)
    Max-Return:  maximise  w'mu

    subject to (both):  sum(w) = 1,  0 <= w_i <= 1

where `mu` is the trailing 52-week historical mean return vector (using
only data strictly before t, so no look-ahead) and `Sigma_t` is that
model's validated one-week-ahead covariance forecast (unused by
Max-Return, kept in the call signature for a uniform objective dispatch).
Portfolio weights depend only on (mu, Sigma), not on transaction costs, so
each (model, objective) weight path is optimised once and then replayed
against three turnover-fee rates -- 0.0%, 0.1% and 0.3% -- to answer the
transaction-cost-sensitivity research question without re-running the
optimiser per fee tier.

Usage
-----
    python portfolio_backtester.py [--forecasts outputs/forecasts.pkl] [--out outputs/portfolio_results.csv]
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

np.random.seed(42)

FORECASTS_PATH = "outputs/forecasts.pkl"
RESULTS_PATH = "outputs/portfolio_results.csv"

RISK_FREE_RATE = 0.0
MU_LOOKBACK_WEEKS = 52  # rolling historical window used to estimate mu

# Transaction-cost sensitivity sweep (SRQ 2): frictionless, default 0.1%,
# and a crypto-realistic 0.3% high-friction tier.
FEE_RATES: Tuple[float, ...] = (0.0, 0.001, 0.003)
DEFAULT_FEE_RATE = 0.001

MODELS: Tuple[str, ...] = ("GARCH", "Markov", "XGBoost")
EQUAL_WEIGHT_STRATEGY = "Equal-Weight"

# Objective name -> strategy-name suffix, e.g. "GARCH" + "Max-Sharpe" -> "GARCH-Sharpe".
OBJECTIVE_SUFFIX: Dict[str, str] = {"Max-Sharpe": "Sharpe", "Max-Return": "Return"}

# BTC-USD -> BTC, SPY -> SPY, ... for the w_{TICKER} output columns.
ASSET_LABELS: Dict[str, str] = {"BTC-USD": "BTC", "SPY": "SPY", "GLD": "GLD", "TLT": "TLT"}


@dataclass
class ForecastBundle:
    """In-memory view of the payload written by main.py to `outputs/forecasts.pkl`."""

    tickers: List[str]
    weekly_returns: pd.DataFrame  # full realised-return history, date-indexed
    forecasts: pd.DataFrame       # columns: date, model, cov_matrix, next_week_returns


def load_forecasts(path: str = FORECASTS_PATH) -> ForecastBundle:
    """Load the covariance-forecast payload persisted by `main.py`."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return ForecastBundle(
        tickers=payload["tickers"],
        weekly_returns=payload["weekly_returns"],
        forecasts=payload["forecasts"],
    )


def rolling_mean_returns(
    weekly_returns: pd.DataFrame, as_of_date: pd.Timestamp, lookback: int = MU_LOOKBACK_WEEKS
) -> Optional[np.ndarray]:
    """
    Trailing `lookback`-week historical mean return vector using only
    observations strictly before `as_of_date` (no look-ahead into the
    week being forecast). Returns None if fewer than `lookback` weeks of
    prior history are available.
    """
    hist = weekly_returns.loc[weekly_returns.index < as_of_date]
    if len(hist) < lookback:
        return None
    return hist.tail(lookback).mean().values


# ---------------------------------------------------------------------------
# Objective functions (SRQ 1: Max-Sharpe vs Max-Return)
# ---------------------------------------------------------------------------

def negative_sharpe_ratio(
    weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float = RISK_FREE_RATE
) -> float:
    """Negative Sharpe ratio of `weights`, for minimisation by scipy.optimize.minimize."""
    port_return = weights @ mu
    port_vol = np.sqrt(weights @ Sigma @ weights)
    if port_vol < 1e-12:
        return 0.0
    return -(port_return - rf) / port_vol


def negative_expected_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Negative expected portfolio return, for minimisation by scipy.optimize.minimize."""
    return -float(weights @ mu)


def _solve_long_only(objective: Callable, args: tuple, n_assets: int) -> np.ndarray:
    """Shared SLSQP solve for sum(w)=1, 0<=w_i<=1; falls back to equal weights on failure."""
    x0 = np.repeat(1.0 / n_assets, n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    result = minimize(
        objective, x0, args=args, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        return x0
    w = np.clip(result.x, 0.0, 1.0)
    return w / w.sum()


def optimise_max_sharpe(mu: np.ndarray, Sigma: np.ndarray, n_assets: int) -> np.ndarray:
    """Solve the long-only, fully-invested maximum-Sharpe-ratio portfolio via SLSQP."""
    return _solve_long_only(negative_sharpe_ratio, (mu, Sigma), n_assets)


def optimise_max_return(mu: np.ndarray, Sigma: np.ndarray, n_assets: int) -> np.ndarray:
    """
    Solve the long-only, fully-invested maximum-expected-return portfolio.
    `Sigma` is accepted (and ignored) only so this shares a call signature
    with `optimise_max_sharpe` for a uniform objective dispatch. Since the
    objective is linear in `w`, the true optimum is a corner solution (all
    weight on the single highest-mu asset); SLSQP is used anyway for
    implementation consistency with the Sharpe objective.
    """
    return _solve_long_only(negative_expected_return, (mu,), n_assets)


OBJECTIVES: Dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "Max-Sharpe": optimise_max_sharpe,
    "Max-Return": optimise_max_return,
}


class PortfolioBacktester:
    """Walk-forward weekly rebalancing simulator across models, objectives and fee tiers."""

    def __init__(
        self,
        bundle: ForecastBundle,
        fee_rates: Tuple[float, ...] = FEE_RATES,
        rf: float = RISK_FREE_RATE,
    ) -> None:
        self.tickers = bundle.tickers
        self.weight_cols = [f"w_{ASSET_LABELS.get(t, t)}" for t in self.tickers]
        self.weekly_returns = bundle.weekly_returns
        self.forecasts = bundle.forecasts
        self.fee_rates = fee_rates
        self.rf = rf

    def _weight_record(self, w: np.ndarray) -> Dict[str, float]:
        return dict(zip(self.weight_cols, w))

    def _model_weight_path(self, model: str, objective_fn: Callable) -> List[dict]:
        """
        Compute the fee-independent (date, weights, gross_return) path for
        one (model, objective) pair. Optimised once; replayed against every
        fee tier by `_apply_fee_rate` below.
        """
        sub = self.forecasts[self.forecasts["model"] == model].sort_values("date")
        n_assets = len(self.tickers)
        path = []
        for _, row in sub.iterrows():
            date = row["date"]
            mu = rolling_mean_returns(self.weekly_returns, date)
            if mu is None:
                continue  # insufficient history to estimate mu yet

            Sigma = np.asarray(row["cov_matrix"])
            r_next = np.asarray(row["next_week_returns"])
            w = objective_fn(mu, Sigma, n_assets)
            path.append(dict(date=date, w=w, gross_return=float(w @ r_next)))
        return path

    def _apply_fee_rate(
        self, weight_path: List[dict], fee_rate: float, strategy: str, objective: str
    ) -> pd.DataFrame:
        """Replay a fee-independent weight path under one turnover-fee rate."""
        n_assets = len(self.tickers)
        prev_w = np.zeros(n_assets)  # first week pays the fee to enter the position from cash
        equity = 1.0
        records = []

        for step in weight_path:
            w = step["w"]
            fee = fee_rate * float(np.sum(np.abs(w - prev_w)))
            net_return = step["gross_return"] - fee
            equity *= 1.0 + net_return

            row = dict(
                date=step["date"], strategy=strategy, objective=objective, fee_rate=fee_rate,
                net_return=net_return, gross_return=step["gross_return"], turnover_fee=fee,
            )
            row.update(self._weight_record(w))
            row["cumulative_equity"] = equity
            records.append(row)
            prev_w = w

        return pd.DataFrame.from_records(records)

    def run_model_objective(self, model: str, objective_name: str) -> pd.DataFrame:
        """Simulate one (model, objective) strategy across every fee tier."""
        weight_path = self._model_weight_path(model, OBJECTIVES[objective_name])
        strategy_name = f"{model}-{OBJECTIVE_SUFFIX[objective_name]}"
        frames = [
            self._apply_fee_rate(weight_path, fee, strategy_name, objective_name)
            for fee in self.fee_rates
        ]
        return pd.concat(frames, ignore_index=True)

    def run_equal_weight(self) -> pd.DataFrame:
        """Static 25%-per-asset baseline, evaluated on the same forecast dates as the models."""
        n_assets = len(self.tickers)
        w = np.repeat(1.0 / n_assets, n_assets)
        weight_path = []
        for date in sorted(self.forecasts["date"].unique()):
            if date not in self.weekly_returns.index:
                continue
            r_next = self.weekly_returns.loc[date, self.tickers].values
            weight_path.append(dict(date=date, w=w, gross_return=float(w @ r_next)))

        frames = [
            self._apply_fee_rate(weight_path, fee, EQUAL_WEIGHT_STRATEGY, EQUAL_WEIGHT_STRATEGY)
            for fee in self.fee_rates
        ]
        return pd.concat(frames, ignore_index=True)

    def run_all(self) -> pd.DataFrame:
        """Run every (model, objective) strategy plus the equal-weight baseline, all fee tiers."""
        frames = [self.run_equal_weight()]
        for model in MODELS:
            for objective_name in OBJECTIVES:
                frames.append(self.run_model_objective(model, objective_name))
        results = pd.concat(frames, ignore_index=True)
        return results.sort_values(["strategy", "fee_rate", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forecasts", default=FORECASTS_PATH)
    parser.add_argument("--out", default=RESULTS_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    bundle = load_forecasts(args.forecasts)
    backtester = PortfolioBacktester(bundle)
    results = backtester.run_all()

    results.to_csv(args.out, index=False)
    n_strategies = results["strategy"].nunique()
    n_fee_tiers = results["fee_rate"].nunique()
    print(f"Wrote {len(results)} rows ({n_strategies} strategies x {n_fee_tiers} fee tiers) to {args.out}")

    default_view = results[np.isclose(results["fee_rate"], DEFAULT_FEE_RATE)]
    summary = default_view.groupby("strategy")["net_return"].agg(["mean", "std", "count"])
    summary["annualised_sharpe"] = (summary["mean"] * 52) / (summary["std"] * np.sqrt(52))
    print(f"\nSummary at the default {DEFAULT_FEE_RATE:.1%} fee tier:")
    print(summary.sort_values("annualised_sharpe", ascending=False).to_string())


if __name__ == "__main__":
    main()
