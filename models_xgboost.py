"""
models_xgboost.py
------------------
XGBoost volatility regressor, trained independently per asset on a
rolling window (~500 trading days), using lagged own volatility, rolling
return standard deviation, cross-asset lagged volatility, and
day-of-week features (see features.py).

Per-asset next-week volatility forecasts are combined into a covariance
matrix using the empirical correlation matrix of returns over the same
rolling window, i.e. the same "diagonal-vol times correlation" (CCC-style)
construction used for the GARCH model, which keeps the three models
comparable and, since a correlation matrix is itself PSD, guarantees a
PSD covariance forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import features as feat_mod

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

ROLLING_WINDOW_DAYS = 500
TRADING_DAYS_PER_WEEK = 5
# ~500 trading days looks back roughly 100 weekly observations.
ROLLING_WINDOW_WEEKS = max(ROLLING_WINDOW_DAYS // TRADING_DAYS_PER_WEEK, 20)
MIN_TRAIN_OBS = 30


@dataclass
class XgbAssetFit:
    ticker: str
    model: XGBRegressor
    feature_columns: List[str]


def _asset_training_frame(
    weekly_returns: pd.DataFrame, weekly_rv: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    frame = feat_mod.build_feature_frame(weekly_returns, weekly_rv, ticker)
    frame["y"] = frame["y"].shift(-1)  # target: next week's realised volatility
    return frame.dropna()


def fit_xgb_window(
    weekly_returns: pd.DataFrame,
    weekly_rv: pd.DataFrame,
    ticker: str,
    window_weeks: int = ROLLING_WINDOW_WEEKS,
) -> Optional[XgbAssetFit]:
    """
    Fit one XGBoost model for `ticker` using only the trailing
    `window_weeks` of history (a rolling, not expanding, window,
    matching a ~500-trading-day lookback).
    """
    frame = _asset_training_frame(weekly_returns, weekly_rv, ticker).tail(window_weeks)
    if len(frame) < MIN_TRAIN_OBS:
        return None

    feature_cols = [c for c in frame.columns if c != "y"]
    X, y = frame[feature_cols], frame["y"]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    return XgbAssetFit(ticker=ticker, model=model, feature_columns=feature_cols)


def xgb_forecast_vol(
    fit: XgbAssetFit, weekly_returns: pd.DataFrame, weekly_rv: pd.DataFrame
) -> float:
    """Predict next week's realised volatility for `fit.ticker` from the latest features."""
    frame = feat_mod.build_feature_frame(weekly_returns, weekly_rv, fit.ticker)
    x_latest = frame[fit.feature_columns].iloc[[-1]]
    pred = fit.model.predict(x_latest)[0]
    return float(max(pred, 1e-8))  # volatility must be strictly positive


def xgb_forecast_covariance(
    weekly_returns: pd.DataFrame,
    weekly_rv: pd.DataFrame,
    window_weeks: int = ROLLING_WINDOW_WEEKS,
) -> Optional[pd.DataFrame]:
    """
    One-step-ahead covariance forecast: XGBoost-predicted volatilities on
    the diagonal, combined with the rolling-window empirical correlation
    matrix of returns, Sigma_t = D_t R D_t.
    """
    tickers = list(weekly_returns.columns)
    fits: Dict[str, XgbAssetFit] = {}
    for t in tickers:
        f = fit_xgb_window(weekly_returns, weekly_rv, t, window_weeks)
        if f is None:
            return None
        fits[t] = f

    vols = np.array([xgb_forecast_vol(fits[t], weekly_returns, weekly_rv) for t in tickers])
    R = weekly_returns.tail(window_weeks).corr().loc[tickers, tickers].values
    D = np.diag(vols)
    Sigma = D @ R @ D
    return pd.DataFrame(Sigma, index=tickers, columns=tickers)
