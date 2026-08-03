"""
features.py
------------
Feature engineering for the XGBoost weekly volatility-forecasting models.

All features are constructed so that, at forecast origin t, only
information available up to and including week t is used (no look-ahead).
"""

from __future__ import annotations

import pandas as pd

OWN_VOL_LAGS = (1, 2, 4, 8)
ROLLING_STD_WINDOWS = (4, 8, 12)
CROSS_ASSET_LAG = 1


def build_feature_frame(
    weekly_returns: pd.DataFrame,
    weekly_rv: pd.DataFrame,
    target_asset: str,
) -> pd.DataFrame:
    """
    Construct the feature matrix (+ contemporaneous RV as `y`) for one asset.

    Features
    --------
    own_rv_lag_{k}     : target asset's realised volatility, lagged k weeks.
    own_rollstd_{w}     : rolling std-dev of the target asset's weekly log
                           returns over the trailing w weeks, lagged one
                           week so it only uses information up to t-1.
    {other}_rv_lag1     : lag-1 realised volatility of every *other* asset
                           in the universe (cross-asset volatility spillover).
    dow_weekend          : day-of-week (Mon=0 ... Sun=6) of the week-ending
                           date, capturing holiday-shifted week boundaries.

    `y` (realised vol in week t) is returned un-shifted; callers that want
    a genuine one-week-ahead forecasting target should shift it by -1
    themselves (this keeps the function reusable for both training-set
    construction and live feature extraction at the most recent date).
    """
    assets = list(weekly_rv.columns)
    idx = weekly_rv.index
    feat = pd.DataFrame(index=idx)

    for k in OWN_VOL_LAGS:
        feat[f"own_rv_lag_{k}"] = weekly_rv[target_asset].shift(k)

    for w in ROLLING_STD_WINDOWS:
        feat[f"own_rollstd_{w}"] = weekly_returns[target_asset].rolling(w).std().shift(1)

    for other in assets:
        if other == target_asset:
            continue
        feat[f"{other}_rv_lag{CROSS_ASSET_LAG}"] = weekly_rv[other].shift(CROSS_ASSET_LAG)

    feat["dow_weekend"] = idx.dayofweek
    feat["y"] = weekly_rv[target_asset]

    return feat
