"""
main.py
-------
End-to-end pipeline:

  1. Download BTC-USD, SPY, GLD, TLT daily prices (yfinance).
  2. Compute weekly log-returns and weekly realised volatility.
  3. Walk forward through the sample; at each origin date, fit
       - GARCH(1,1) per asset -> CCC-GARCH covariance matrix,
       - a 3-state Markov chain regime model -> mixture covariance matrix,
       - XGBoost volatility regressors (rolling ~500-day window) -> D_t R_t D_t
     and forecast the one-week-ahead covariance matrix for each model.
  4. Validate that every forecast is symmetric, PSD and finite, and
     report a pass-rate summary plus a CSV of per-forecast diagnostics.

Usage
-----
    python main.py --start 2016-01-01 --min-train-weeks 104
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd

import data as data_mod
import models_garch as garch_mod
import models_markov as markov_mod
import models_xgboost as xgb_mod
from covariance import validate_covariance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)  # reproducibility for any stochastic step in the walk-forward loop


def run_walk_forward(
    weekly_returns: pd.DataFrame,
    weekly_rv: pd.DataFrame,
    min_train_weeks: int = 104,
    refit_every: int = 1,
    xgb_window_weeks: int = xgb_mod.ROLLING_WINDOW_WEEKS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk forward through the weekly sample. At each step i (once
    `min_train_weeks` of history are available), refit each of the three
    models on data up to and including week i, forecast the covariance
    matrix for week i+1, and validate it.

    `refit_every` controls how often GARCH/Markov are re-estimated
    (in weeks) to trade off fidelity against runtime; XGBoost is always
    re-fit each step since it trains on a fixed-size rolling window.

    Returns
    -------
    (validation_results, forecasts) : Tuple[pd.DataFrame, pd.DataFrame]
        `validation_results` is a tidy long-format frame with one row per
        (forecast_date, model) validation result.

        `forecasts` is a tidy long-format frame with one row per
        (forecast_date, model) forecast that *passed* validation, holding
        columns `date`, `model`, `cov_matrix` (4x4 np.ndarray, asset order
        matching `weekly_returns.columns`) and `next_week_returns` (4,
        np.ndarray of realised log-returns for that forecast date). Invalid
        forecasts are dropped here since they are not usable by a
        downstream portfolio optimiser.
    """
    tickers = list(weekly_returns.columns)
    dates = weekly_returns.index
    records = []
    forecast_records = []

    garch_fits, markov_fit = None, None
    n_steps = len(dates) - min_train_weeks - 1
    logger.info("Walk-forward: %d forecast steps starting from week %d", n_steps, min_train_weeks)

    for i in range(min_train_weeks, len(dates) - 1):
        train_returns = weekly_returns.iloc[: i + 1]
        train_rv = weekly_rv.reindex(train_returns.index).dropna(how="all")
        forecast_date = dates[i + 1]

        refit_now = garch_fits is None or ((i - min_train_weeks) % refit_every == 0)

        # ---- GARCH(1,1) / CCC-GARCH ----
        if refit_now:
            garch_fits = garch_mod.fit_garch_all(train_returns)
        Sigma_garch = garch_mod.ccc_covariance_matrix(garch_fits)

        # ---- 3-state Markov chain ----
        if refit_now:
            markov_fit = markov_mod.fit_markov(train_returns, train_rv)
        Sigma_markov = markov_mod.markov_forecast_covariance(markov_fit, train_rv)

        # ---- XGBoost (rolling ~500-trading-day window) ----
        Sigma_xgb = xgb_mod.xgb_forecast_covariance(train_returns, train_rv, xgb_window_weeks)

        for model_name, Sigma in (
            ("GARCH", Sigma_garch),
            ("Markov", Sigma_markov),
            ("XGBoost", Sigma_xgb),
        ):
            if Sigma is None:
                continue
            v = validate_covariance(Sigma)
            records.append(
                dict(
                    forecast_date=forecast_date,
                    model=model_name,
                    is_valid=v.is_valid,
                    is_symmetric=v.is_symmetric,
                    is_psd=v.is_psd,
                    is_finite=v.is_finite,
                    min_eigenvalue=v.min_eigenvalue,
                    max_asymmetry=v.max_asymmetry,
                )
            )
            if v.is_valid:
                forecast_records.append(
                    dict(
                        date=forecast_date,
                        model=model_name,
                        cov_matrix=Sigma.reindex(index=tickers, columns=tickers).values,
                        next_week_returns=weekly_returns.loc[forecast_date, tickers].values,
                    )
                )

    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(forecast_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=list(data_mod.DEFAULT_TICKERS))
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--min-train-weeks", type=int, default=104,
                         help="Burn-in period (weeks) before the first forecast.")
    parser.add_argument("--refit-every", type=int, default=1,
                         help="Re-fit GARCH/Markov every N weeks (1 = every week).")
    parser.add_argument("--out", default="outputs/covariance_validation_results.csv")
    parser.add_argument("--forecasts-out", default="outputs/forecasts.pkl",
                         help="Pickle path for the persisted covariance-forecast/return series "
                              "consumed by portfolio_backtester.py and statistical_tests.py.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.forecasts_out) or ".", exist_ok=True)

    _, weekly_returns, weekly_rv = data_mod.build_dataset(args.tickers, args.start, args.end)
    logger.info(
        "Sample: %d weekly observations, %s to %s",
        len(weekly_returns), weekly_returns.index.min().date(), weekly_returns.index.max().date(),
    )

    results, forecasts = run_walk_forward(
        weekly_returns,
        weekly_rv,
        min_train_weeks=args.min_train_weeks,
        refit_every=args.refit_every,
    )
    results.to_csv(args.out, index=False)
    logger.info("Wrote %d validation records to %s", len(results), args.out)

    # Full-sample Markov fit, kept separate from the walk-forward loop above.
    # It is not used for forecasting (that stays purely walk-forward / no
    # look-ahead); it exists only to hand statistical_tests.py a single
    # regime-state sequence for the order-selection diagnostic (1st- vs
    # 2nd-order Markov chain LRT) requested in Requirement 3.
    full_sample_markov_fit = markov_mod.fit_markov(weekly_returns, weekly_rv)

    forecast_payload = dict(
        tickers=list(weekly_returns.columns),
        weekly_returns=weekly_returns,
        forecasts=forecasts,
        regime_states=full_sample_markov_fit.state_sequence,
    )
    with open(args.forecasts_out, "wb") as f:
        pickle.dump(forecast_payload, f)
    logger.info(
        "Wrote %d covariance forecasts (%d models) to %s",
        len(forecasts), forecasts["model"].nunique() if not forecasts.empty else 0, args.forecasts_out,
    )

    summary = results.groupby("model")["is_valid"].agg(["mean", "sum", "count"])
    summary.columns = ["pct_valid", "n_valid", "n_total"]
    logger.info("Validation summary:\n%s", summary.to_string())

    if results["is_valid"].all():
        logger.info(
            "PASS: all %d covariance-matrix forecasts across all three models "
            "are symmetric, positive semi-definite and finite.", len(results)
        )
    else:
        failures = results[~results["is_valid"]]
        logger.warning("FAIL: %d invalid covariance matrices detected:\n%s", len(failures), failures.to_string())


if __name__ == "__main__":
    main()
