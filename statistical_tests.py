"""
statistical_tests.py
----------------------
Standalone empirical-evaluation script for the dissertation write-up.
Consumes `outputs/portfolio_results.csv` (from `portfolio_backtester.py`)
and `outputs/forecasts.pkl` (from `main.py`) to produce:

  1. An annualised performance summary + bootstrapped 95% Sharpe CIs for
     the default 0.1%-fee Max-Sharpe strategies (plus Equal-Weight).
  2. SRQ 1 -- Max-Sharpe vs Max-Return objective comparison (at 0.1% fee).
  3. SRQ 2 -- transaction-cost sensitivity across 0.0% / 0.1% / 0.3% fees.
  4. SRQ 3 -- performance during the COVID-19 crash and 2022 rate-hike
     stress windows (at 0.1% fee).
  5. A Markov-chain order-selection diagnostic (1st- vs 2nd-order LRT,
     AIC, BIC) on the K-Means regime sequence.
  6. Publication-ready figures in `figures/`:
       Fig1_Equity_Curves.png, Fig2_Asset_Allocations.png,
       Fig2_Asset_Allocations_GARCH.png, Fig2_Asset_Allocations_Markov.png,
       Fig2_Asset_Allocations_XGBoost.png, Fig3_Drawdowns.png

Usage
-----
    python statistical_tests.py
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

from portfolio_backtester import DEFAULT_FEE_RATE, FEE_RATES

np.random.seed(42)
sns.set_theme(style="whitegrid")

RESULTS_PATH = "outputs/portfolio_results.csv"
FORECASTS_PATH = "outputs/forecasts.pkl"
FIGURES_DIR = "figures"

# Fixed per-asset colors so every figure that plots weights (Fig2 and its
# XGBoost-specific counterpart) uses the same asset->color mapping.
ASSET_COLORS: Dict[str, str] = {
    "w_BTC": "#1f77b4",  # blue
    "w_SPY": "#ff7f0e",  # orange
    "w_GLD": "#2ca02c",  # green
    "w_TLT": "#d62728",  # red / rose
}

WEEKS_PER_YEAR = 52
N_BOOTSTRAP = 1_000
CI_LOW, CI_HIGH = 2.5, 97.5

# SRQ 3 market-stress sub-periods.
STRESS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "COVID-19 Crash (2020-02-01 to 2020-05-31)": ("2020-02-01", "2020-05-31"),
    "2022 Fed Rate Hike Cycle (2022-01-01 to 2022-12-31)": ("2022-01-01", "2022-12-31"),
}


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def annualised_return(net_returns: np.ndarray) -> float:
    """Mean weekly net return, annualised by x52."""
    return float(np.mean(net_returns) * WEEKS_PER_YEAR)


def annualised_volatility(net_returns: np.ndarray) -> float:
    """Weekly net-return standard deviation, annualised by x sqrt(52)."""
    if len(net_returns) < 2:
        return float("nan")
    return float(np.std(net_returns, ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def annualised_sharpe(net_returns: np.ndarray) -> float:
    """Annualised Sharpe ratio = annualised return / annualised volatility."""
    vol = annualised_volatility(net_returns)
    return annualised_return(net_returns) / vol if vol and vol > 0 else float("nan")


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough loss (negative number) of a cumulative-equity path."""
    if len(equity_curve) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Full drawdown path (negative, 0 at new equity highs) of a cumulative-equity series."""
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def total_turnover(grp: pd.DataFrame) -> float:
    """Total L1 turnover implied by a strategy's turnover_fee column at its own fee rate."""
    fee_rate = grp["fee_rate"].iloc[0]
    return float(grp["turnover_fee"].sum() / fee_rate) if fee_rate > 0 else float("nan")


def build_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Annualised performance summary for every (strategy, fee_rate) slice present in `results`."""
    rows = []
    for strategy, grp in results.groupby("strategy"):
        grp = grp.sort_values("date")
        net = grp["net_return"].values
        rows.append(
            dict(
                strategy=strategy,
                annualised_return=annualised_return(net),
                annualised_volatility=annualised_volatility(net),
                annualised_sharpe=annualised_sharpe(net),
                max_drawdown=max_drawdown(grp["cumulative_equity"].values),
                total_turnover=total_turnover(grp),
                n_weeks=len(grp),
            )
        )
    return pd.DataFrame.from_records(rows).set_index("strategy").sort_values(
        "annualised_sharpe", ascending=False
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_sharpe_ci(
    net_returns: np.ndarray, n_iterations: int = N_BOOTSTRAP
) -> Tuple[float, float, float]:
    """
    Bootstrap a 95% confidence interval for the annualised Sharpe ratio via
    resampling with replacement (numpy.random.choice), `n_iterations` times.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    n = len(net_returns)
    boot_sharpes = np.empty(n_iterations)
    for b in range(n_iterations):
        sample = np.random.choice(net_returns, size=n, replace=True)
        boot_sharpes[b] = annualised_sharpe(sample)

    point = annualised_sharpe(net_returns)
    lower, upper = np.nanpercentile(boot_sharpes, [CI_LOW, CI_HIGH])
    return point, float(lower), float(upper)


def build_bootstrap_table(results: pd.DataFrame) -> pd.DataFrame:
    """Bootstrapped Sharpe-ratio 95% CI table for every strategy present in `results`."""
    rows = []
    for strategy, grp in results.groupby("strategy"):
        net = grp.sort_values("date")["net_return"].values
        point, lower, upper = bootstrap_sharpe_ci(net)
        rows.append(dict(strategy=strategy, sharpe=point, ci_lower=lower, ci_upper=upper))
    return pd.DataFrame.from_records(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# SRQ 1: Max-Sharpe vs Max-Return objective comparison
# ---------------------------------------------------------------------------

def build_srq1_table(results: pd.DataFrame, fee_rate: float = DEFAULT_FEE_RATE) -> pd.DataFrame:
    """Performance table for every model x objective strategy at a single fee tier."""
    sub = results[np.isclose(results["fee_rate"], fee_rate) & (results["objective"] != "Equal-Weight")]
    return build_summary_table(sub)


def srq1_verdict(table: pd.DataFrame) -> str:
    """
    Compare average risk/return/drawdown across the Max-Sharpe strategies
    vs the Max-Return strategies and report which objective manages risk
    better in this backtest -- computed from the actual results, not
    assumed a priori.
    """
    sharpe_rows = table[table.index.str.endswith("-Sharpe")]
    return_rows = table[table.index.str.endswith("-Return")]
    if sharpe_rows.empty or return_rows.empty:
        return "Insufficient strategy coverage to compare objectives."

    avg_sharpe_obj = sharpe_rows["annualised_sharpe"].mean()
    avg_sharpe_ret = return_rows["annualised_sharpe"].mean()
    avg_vol_obj = sharpe_rows["annualised_volatility"].mean()
    avg_vol_ret = return_rows["annualised_volatility"].mean()
    avg_dd_obj = sharpe_rows["max_drawdown"].mean()
    avg_dd_ret = return_rows["max_drawdown"].mean()

    lines = [
        f"Average annualised Sharpe:      Max-Sharpe = {avg_sharpe_obj:,.4f}   Max-Return = {avg_sharpe_ret:,.4f}",
        f"Average annualised volatility:  Max-Sharpe = {avg_vol_obj:,.4f}   Max-Return = {avg_vol_ret:,.4f}",
        f"Average max drawdown:           Max-Sharpe = {avg_dd_obj:,.4f}   Max-Return = {avg_dd_ret:,.4f}",
        "",
    ]

    sharpe_wins = avg_sharpe_obj > avg_sharpe_ret
    lower_vol = avg_vol_obj < avg_vol_ret
    shallower_dd = avg_dd_obj > avg_dd_ret  # drawdowns are negative; closer to 0 = shallower

    if sharpe_wins:
        lines.append(
            "-> Max-Sharpe optimisation delivers better risk-adjusted performance than pure "
            "Max-Return optimisation across GARCH/Markov/XGBoost (higher average Sharpe, "
            f"{'lower' if lower_vol else 'higher'} average volatility, "
            f"{'shallower' if shallower_dd else 'deeper'} average drawdowns)."
        )
    else:
        lines.append(
            "-> Max-Return optimisation shows a higher average Sharpe ratio than Max-Sharpe "
            "in this backtest window -- Max-Sharpe still targets risk-adjusted return by "
            "construction, so this likely reflects concentrated corner-solution weights "
            "coinciding with a strongly trending sample rather than a general result."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SRQ 2: transaction-cost sensitivity
# ---------------------------------------------------------------------------

def build_srq2_table(results: pd.DataFrame, fee_rates: Tuple[float, ...] = FEE_RATES) -> pd.DataFrame:
    """
    Annualised Sharpe ratio at each fee tier, per strategy, plus percentage
    decay from the frictionless (0.0%) tier to the highest-friction tier:

        Sharpe Decay (%) = (Sharpe_highest_fee - Sharpe_0.0%) / Sharpe_0.0% * 100
    """
    fee_low, fee_high = min(fee_rates), max(fee_rates)
    rows = []
    for strategy, grp in results.groupby("strategy"):
        sharpe_by_fee = {}
        for fee in fee_rates:
            sub = grp[np.isclose(grp["fee_rate"], fee)].sort_values("date")
            sharpe_by_fee[fee] = annualised_sharpe(sub["net_return"].values) if len(sub) else float("nan")

        s_low, s_high = sharpe_by_fee[fee_low], sharpe_by_fee[fee_high]
        decay = (s_high - s_low) / s_low * 100.0 if s_low and not np.isnan(s_low) else float("nan")

        row = {"strategy": strategy}
        for fee in fee_rates:
            row[f"Sharpe ({fee:.1%})"] = sharpe_by_fee[fee]
        row["Sharpe Decay (%)"] = decay
        rows.append(row)

    return pd.DataFrame.from_records(rows).set_index("strategy").sort_values(
        f"Sharpe ({fee_low:.1%})", ascending=False
    )


# ---------------------------------------------------------------------------
# SRQ 3: market-stress sub-period analysis
# ---------------------------------------------------------------------------

def window_metrics(net_returns: np.ndarray) -> dict:
    """Cumulative return, annualised vol, max drawdown and Sharpe within a single sub-period."""
    if len(net_returns) == 0:
        return dict(cumulative_window_return=float("nan"), annualised_volatility=float("nan"),
                     max_drawdown=float("nan"), sharpe=float("nan"), n_weeks=0)
    equity = np.cumprod(1.0 + net_returns)
    return dict(
        cumulative_window_return=float(equity[-1] - 1.0),
        annualised_volatility=annualised_volatility(net_returns),
        max_drawdown=max_drawdown(equity),
        sharpe=annualised_sharpe(net_returns),
        n_weeks=len(net_returns),
    )


def build_srq3_tables(results: pd.DataFrame, fee_rate: float = DEFAULT_FEE_RATE) -> Dict[str, pd.DataFrame]:
    """One performance table per stress window, all strategies, at a single fee tier."""
    sub_fee = results[np.isclose(results["fee_rate"], fee_rate)]
    tables = {}
    for label, (start, end) in STRESS_WINDOWS.items():
        rows = []
        for strategy, grp in sub_fee.groupby("strategy"):
            window = grp[(grp["date"] >= start) & (grp["date"] <= end)].sort_values("date")
            if window.empty:
                continue
            rows.append(dict(strategy=strategy, **window_metrics(window["net_return"].values)))
        table = pd.DataFrame.from_records(rows)
        tables[label] = (
            table.set_index("strategy").sort_values("sharpe", ascending=False) if not table.empty else table
        )
    return tables


# ---------------------------------------------------------------------------
# Markov chain order selection (1st- vs 2nd-order LRT, on the K-Means
# regime sequence), following the classical Anderson & Goodman (1957)
# likelihood-ratio framework for testing Markov chain order.
# ---------------------------------------------------------------------------

@dataclass
class MarkovOrderTest:
    log_likelihood_order1: float
    log_likelihood_order2: float
    aic_order1: float
    aic_order2: float
    bic_order1: float
    bic_order2: float
    lrt_statistic: float
    lrt_df: int
    lrt_p_value: float


def _fit_multinomial_log_likelihood(counts: np.ndarray) -> Tuple[float, int]:
    """
    MLE log-likelihood of a categorical model fit to a contingency table
    whose last axis is the outcome and all leading axes are the
    conditioning context (e.g. shape (K,) for order-0, (K, K) for order-1,
    (K, K, K) for order-2). Cells with zero count contribute 0 (by the
    0*log(0) = 0 convention).
    """
    context_shape = counts.shape[:-1]
    n_states = counts.shape[-1]
    flat = counts.reshape(-1, n_states)
    row_sums = flat.sum(axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(row_sums > 0, flat / row_sums, 0.0)

    mask = flat > 0
    log_likelihood = float(np.sum(flat[mask] * np.log(probs[mask])))
    n_contexts = int(np.prod(context_shape)) if context_shape else 1
    n_params = n_contexts * (n_states - 1)
    return log_likelihood, n_params


def markov_order_selection(state_sequence: pd.Series, n_states: int = 3) -> MarkovOrderTest:
    """
    Test whether the K-Means regime sequence is better described by a
    1st-order or a 2nd-order Markov chain.

    Both models are fit on the *same* set of state triples (s_t, s_{t+1},
    s_{t+2}) so that the order-1 model is properly nested inside the
    order-2 model: the order-2 transition tensor P(s_{t+2} | s_t, s_{t+1})
    is estimated directly, and the order-1 null model P(s_{t+2} | s_{t+1})
    is obtained by marginalising the same tensor over s_t, giving a valid
    likelihood-ratio test of order-1 (H0) vs order-2 (H1).
    """
    states = state_sequence.values.astype(int)
    n_triples = len(states) - 2

    counts_order2 = np.zeros((n_states, n_states, n_states))
    for t in range(n_triples):
        counts_order2[states[t], states[t + 1], states[t + 2]] += 1
    counts_order1 = counts_order2.sum(axis=0)  # marginalise out s_t

    ll1, p1 = _fit_multinomial_log_likelihood(counts_order1)
    ll2, p2 = _fit_multinomial_log_likelihood(counts_order2)

    n_obs = int(counts_order2.sum())
    aic1, aic2 = -2 * ll1 + 2 * p1, -2 * ll2 + 2 * p2
    bic1, bic2 = -2 * ll1 + p1 * np.log(n_obs), -2 * ll2 + p2 * np.log(n_obs)

    lrt_stat = max(0.0, 2.0 * (ll2 - ll1))
    df = p2 - p1
    p_value = float(chi2.sf(lrt_stat, df))

    return MarkovOrderTest(
        log_likelihood_order1=ll1,
        log_likelihood_order2=ll2,
        aic_order1=aic1,
        aic_order2=aic2,
        bic_order1=bic1,
        bic_order2=bic2,
        lrt_statistic=lrt_stat,
        lrt_df=df,
        lrt_p_value=p_value,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_equity_curves(results: pd.DataFrame, out_path: str) -> None:
    """Fig1: cumulative net-return equity curves for the default-fee benchmark strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy, grp in results.groupby("strategy"):
        grp = grp.sort_values("date")
        ax.plot(grp["date"], grp["cumulative_equity"], label=strategy, linewidth=1.5)
    ax.set_title("Cumulative Net-Return Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Equity (growth of $1)")
    ax.legend(title="Strategy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_asset_allocations(
    results: pd.DataFrame,
    strategy: str,
    weight_cols: List[str],
    out_path: str,
    title: Optional[str] = None,
    dpi: int = 200,
    year_axis: bool = False,
) -> bool:
    """
    Stacked area chart of weight evolution for a single strategy. Colors
    are fixed per asset via `ASSET_COLORS` so this figure and any other
    call of this function stay visually consistent with each other.

    Returns False (and writes nothing) if `strategy` has no rows in
    `results` -- e.g. a stress window or fee slice that doesn't cover it --
    rather than emitting a blank/broken figure.
    """
    grp = results[results["strategy"] == strategy].sort_values("date")
    if grp.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        grp["date"],
        [grp[c] for c in weight_cols],
        labels=[c.replace("w_", "") for c in weight_cols],
        colors=[ASSET_COLORS.get(c) for c in weight_cols],
        alpha=0.85,
    )
    ax.set_title(title or f"Asset Allocation Over Time — {strategy} (best Sharpe)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Weight")
    ax.set_ylim(0.0, 1.0)
    if year_axis:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(title="Asset", loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_drawdowns(results: pd.DataFrame, out_path: str) -> None:
    """Fig3: drawdown paths for the default-fee benchmark strategies, to compare stress behaviour."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy, grp in results.groupby("strategy"):
        grp = grp.sort_values("date")
        dd = drawdown_series(grp.set_index("date")["cumulative_equity"])
        ax.plot(dd.index, dd.values, label=strategy, linewidth=1.5)
    ax.set_title("Portfolio Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown from Peak Equity")
    ax.legend(title="Strategy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--forecasts", default=FORECASTS_PATH)
    parser.add_argument("--figures-dir", default=FIGURES_DIR)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE,
                         help="Reference fee tier used for the default summary, bootstrap CIs, "
                              "SRQ1, SRQ3 and figures (SRQ2 sweeps all fee tiers regardless).")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    results = pd.read_csv(args.results, parse_dates=["date"])
    weight_cols = [c for c in results.columns if c.startswith("w_")]

    # Reference view used by the "legacy" sections (overall summary, bootstrap
    # CIs, figures): the default fee tier, Max-Sharpe strategies + Equal-Weight.
    default_view = results[
        np.isclose(results["fee_rate"], args.fee_rate)
        & results["objective"].isin(["Max-Sharpe", "Equal-Weight"])
    ]

    _print_header(f"ANNUALISED PERFORMANCE SUMMARY (fee = {args.fee_rate:.1%}, Max-Sharpe + Equal-Weight)")
    summary = build_summary_table(default_view)
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    _print_header(f"BOOTSTRAPPED 95% CONFIDENCE INTERVALS FOR SHARPE RATIO ({N_BOOTSTRAP} iterations, "
                  f"fee = {args.fee_rate:.1%}, Max-Sharpe + Equal-Weight)")
    bootstrap_table = build_bootstrap_table(default_view)
    print(bootstrap_table.to_string(float_format=lambda x: f"{x:,.4f}"))

    _print_header("=== SRQ 1: OBJECTIVE FUNCTION COMPARISON (Max-Sharpe vs Max-Return, "
                  f"fee = {args.fee_rate:.1%}) ===")
    srq1_table = build_srq1_table(results, args.fee_rate)
    print(srq1_table.to_string(float_format=lambda x: f"{x:,.4f}"))
    print()
    print(srq1_verdict(srq1_table))

    _print_header("=== SRQ 2: TRANSACTION COST SENSITIVITY (0.0% vs 0.1% vs 0.3%) ===")
    srq2_table = build_srq2_table(results)
    print(srq2_table.to_string(float_format=lambda x: f"{x:,.4f}"))

    _print_header(f"=== SRQ 3: MARKET STRESS SUB-PERIOD ANALYSIS (fee = {args.fee_rate:.1%}) ===")
    srq3_tables = build_srq3_tables(results, args.fee_rate)
    for label, table in srq3_tables.items():
        print(f"\n--- {label} ---")
        if table.empty:
            print("No data available in this window (check --start / --end coverage in main.py).")
        else:
            print(table.to_string(float_format=lambda x: f"{x:,.4f}"))

    _print_header("MARKOV CHAIN ORDER SELECTION (1st- vs 2nd-order, K-Means regime sequence)")
    with open(args.forecasts, "rb") as f:
        forecast_payload = pickle.load(f)
    regime_states = forecast_payload["regime_states"]
    order_test = markov_order_selection(regime_states)
    print(f"Log-likelihood  order-1: {order_test.log_likelihood_order1:,.3f}   "
          f"order-2: {order_test.log_likelihood_order2:,.3f}")
    print(f"AIC             order-1: {order_test.aic_order1:,.3f}   "
          f"order-2: {order_test.aic_order2:,.3f}")
    print(f"BIC             order-1: {order_test.bic_order1:,.3f}   "
          f"order-2: {order_test.bic_order2:,.3f}")
    print(f"LRT statistic: {order_test.lrt_statistic:,.4f}  (df={order_test.lrt_df}, "
          f"p-value={order_test.lrt_p_value:,.4f})")
    if order_test.lrt_p_value < 0.05:
        print("-> Reject 1st-order in favour of 2nd-order at the 5% level.")
    else:
        print("-> Fail to reject 1st-order: a 1st-order chain is a parsimonious fit.")

    _print_header("FIGURES")
    plot_equity_curves(default_view, os.path.join(args.figures_dir, "Fig1_Equity_Curves.png"))
    print(f"Wrote {args.figures_dir}/Fig1_Equity_Curves.png")

    default_summary = build_summary_table(default_view)
    best_strategy = default_summary.index[0]
    plot_asset_allocations(
        default_view, best_strategy, weight_cols, os.path.join(args.figures_dir, "Fig2_Asset_Allocations.png")
    )
    print(f"Wrote {args.figures_dir}/Fig2_Asset_Allocations.png (best strategy: {best_strategy})")

    # Dedicated figure for the ML (XGBoost) strategy: the "best Sharpe" pick
    # above is frequently the static Equal-Weight baseline, which plots as a
    # flat, uninformative band -- this always shows the time-varying weights
    # XGBoost actually produced, for slides/dissertation use regardless of
    # which strategy ranks highest on Sharpe.
    xgb_written = plot_asset_allocations(
        default_view,
        "XGBoost-Sharpe",
        weight_cols,
        os.path.join(args.figures_dir, "Fig2_Asset_Allocations_XGBoost.png"),
        title="Asset Allocation Over Time — XGBoost-Sharpe (ML Dynamic)",
        dpi=300,
        year_axis=True,
    )
    if xgb_written:
        print(f"Wrote {args.figures_dir}/Fig2_Asset_Allocations_XGBoost.png")
    else:
        print("Skipped Fig2_Asset_Allocations_XGBoost.png: no XGBoost-Sharpe rows at the reference fee tier.")

    # Same dedicated treatment for GARCH and Markov, so all three models have
    # a like-for-like allocation figure regardless of which one wins on Sharpe.
    garch_written = plot_asset_allocations(
        default_view,
        "GARCH-Sharpe",
        weight_cols,
        os.path.join(args.figures_dir, "Fig2_Asset_Allocations_GARCH.png"),
        title="Asset Allocation Over Time — GARCH-Sharpe (CCC-GARCH)",
        dpi=300,
        year_axis=True,
    )
    if garch_written:
        print(f"Wrote {args.figures_dir}/Fig2_Asset_Allocations_GARCH.png")
    else:
        print("Skipped Fig2_Asset_Allocations_GARCH.png: no GARCH-Sharpe rows at the reference fee tier.")

    markov_written = plot_asset_allocations(
        default_view,
        "Markov-Sharpe",
        weight_cols,
        os.path.join(args.figures_dir, "Fig2_Asset_Allocations_Markov.png"),
        title="Asset Allocation Over Time — Markov-Sharpe (3-State Regime)",
        dpi=300,
        year_axis=True,
    )
    if markov_written:
        print(f"Wrote {args.figures_dir}/Fig2_Asset_Allocations_Markov.png")
    else:
        print("Skipped Fig2_Asset_Allocations_Markov.png: no Markov-Sharpe rows at the reference fee tier.")

    plot_drawdowns(default_view, os.path.join(args.figures_dir, "Fig3_Drawdowns.png"))
    print(f"Wrote {args.figures_dir}/Fig3_Drawdowns.png")

    _print_header("DONE")
    print(f"Evaluated {results['strategy'].nunique()} strategies across "
          f"{results['fee_rate'].nunique()} fee tiers ({len(results)} total rows).")


if __name__ == "__main__":
    main()
