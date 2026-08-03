"""
data.py
-------
Data acquisition and return/volatility computation for the multi-asset
volatility-informed portfolio optimisation project.

Downloads daily adjusted-close prices via yfinance, resamples to weekly
frequency, and computes weekly log-returns together with weekly realised
volatility (RV) constructed from the underlying daily log-returns.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ("BTC-USD", "SPY", "GLD", "TLT")
TRADING_DAYS_PER_YEAR = 252
WEEKS_PER_YEAR = 52
WEEK_ENDING_RULE = "W-FRI"  # weeks end on Friday (last NYSE trading day of a normal week)


def download_daily_prices(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for `tickers` from Yahoo Finance.

    Notes
    -----
    BTC-USD trades 7 days a week while SPY/GLD/TLT trade on the NYSE/NYSE
    Arca calendar. We deliberately do not force daily-level alignment here;
    alignment happens naturally once we resample to weekly frequency in
    `to_weekly_prices`, which is the frequency the models are estimated on.
    """
    logger.info("Downloading daily prices for %s from %s to %s", tickers, start, end)
    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError(
            "yfinance returned no data. Check ticker symbols, date range, "
            "and that this environment has internet access to Yahoo Finance."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single-ticker download collapses to a flat frame.
        prices = raw[["Close"]]
        prices.columns = list(tickers)

    prices = prices.dropna(how="all").sort_index()
    prices.columns = [str(c) for c in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(f"No price data returned for: {missing}")

    return prices[list(tickers)]


def to_weekly_prices(daily_prices: pd.DataFrame, week_ending: str = WEEK_ENDING_RULE) -> pd.DataFrame:
    """Resample daily close prices to weekly frequency (last obs. in each week)."""
    weekly = daily_prices.resample(week_ending).last()
    return weekly.dropna(how="all")


def daily_log_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: r_d = ln(P_d / P_{d-1}), computed asset-by-asset."""
    return np.log(daily_prices).diff().dropna(how="all")


def weekly_log_returns(weekly_prices: pd.DataFrame) -> pd.DataFrame:
    """Weekly log returns: r_t = ln(P_t / P_{t-1}) on weekly closes."""
    return np.log(weekly_prices).diff().dropna(how="all")


def weekly_realised_volatility(
    daily_prices: pd.DataFrame,
    week_ending: str = WEEK_ENDING_RULE,
    annualise: bool = True,
) -> pd.DataFrame:
    """
    Weekly realised volatility (RV), estimated as the square root of the
    sum of squared daily log returns within each calendar week - a
    standard realised-variance estimator:

        RV_week   = sqrt( sum_{d in week} r_d^2 )
        RV_annual = RV_week * sqrt(WEEKS_PER_YEAR)      (if annualise=True)

    Weeks containing fewer than 2 daily observations (e.g. a single
    trading day around a holiday) are dropped as unreliable.
    """
    r_d = daily_log_returns(daily_prices)
    obs_count = r_d.resample(week_ending).count()
    sum_sq = r_d.pow(2).resample(week_ending).sum()

    rv = np.sqrt(sum_sq)
    rv = rv.where(obs_count >= 2)
    if annualise:
        rv = rv * np.sqrt(WEEKS_PER_YEAR)
    return rv.dropna(how="all")


def build_dataset(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning (weekly_prices, weekly_returns, weekly_rv), aligned."""
    daily_prices = download_daily_prices(tickers, start, end)
    weekly_prices = to_weekly_prices(daily_prices)
    weekly_returns = weekly_log_returns(weekly_prices)
    weekly_rv = weekly_realised_volatility(daily_prices)

    common_idx = weekly_returns.index.intersection(weekly_rv.index)
    weekly_returns = weekly_returns.loc[common_idx]
    weekly_rv = weekly_rv.loc[common_idx]
    weekly_prices = weekly_prices.loc[weekly_prices.index.isin(common_idx.union([weekly_prices.index.min()]))]
    return weekly_prices, weekly_returns, weekly_rv
