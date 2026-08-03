"""
models_garch.py
----------------
Univariate GARCH(1,1) volatility model fitted independently to each asset,
combined into a multivariate covariance forecast via the Constant
Conditional Correlation (CCC-GARCH) construction of Bollerslev (1990):

    Sigma_t = D_t R D_t,      D_t = diag(sigma_1t, ..., sigma_Nt)

where sigma_it is asset i's GARCH(1,1) conditional volatility forecast and
R is the (constant) correlation matrix of the standardised GARCH residuals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

logger = logging.getLogger(__name__)


@dataclass
class GarchAssetFit:
    ticker: str
    result: ARCHModelResult
    scale: float  # rescaling factor applied to returns before fitting


def fit_garch11(returns: pd.Series, scale: float = 100.0) -> GarchAssetFit:
    """
    Fit a GARCH(1,1) model (constant mean, Normal innovations) to one
    asset's weekly log returns. Returns are rescaled by `scale` before
    fitting, as recommended by the `arch` package documentation to aid
    numerical convergence of the optimiser.
    """
    ticker = returns.name
    r = returns.dropna() * scale
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    return GarchAssetFit(ticker=ticker, result=res, scale=scale)


def fit_garch_all(weekly_returns: pd.DataFrame) -> Dict[str, GarchAssetFit]:
    """Fit an independent GARCH(1,1) model for every column (asset)."""
    return {col: fit_garch11(weekly_returns[col]) for col in weekly_returns.columns}


def garch_forecast_vol(fit: GarchAssetFit, horizon: int = 1) -> float:
    """One-step-ahead conditional volatility forecast, on the original return scale."""
    fc = fit.result.forecast(horizon=horizon, reindex=False)
    var_h = fc.variance.values[-1, -1]
    return float(np.sqrt(var_h) / fit.scale)


def standardised_residuals(fit: GarchAssetFit) -> pd.Series:
    """Residuals standardised by their conditional volatility, z_t = eps_t / sigma_t."""
    res = fit.result
    resid = res.resid / res.conditional_volatility
    resid.name = fit.ticker
    return resid


def ccc_correlation_matrix(fits: Dict[str, GarchAssetFit]) -> pd.DataFrame:
    """
    Unconditional (constant) correlation matrix R of the standardised
    GARCH residuals across assets, as used in CCC-GARCH.
    """
    z = pd.concat([standardised_residuals(f) for f in fits.values()], axis=1).dropna()
    return z.corr()


def ccc_covariance_matrix(fits: Dict[str, GarchAssetFit], horizon: int = 1) -> pd.DataFrame:
    """One-step-ahead CCC-GARCH covariance matrix forecast, Sigma_t = D_t R D_t."""
    tickers = list(fits.keys())
    vols = np.array([garch_forecast_vol(fits[t], horizon=horizon) for t in tickers])
    R = ccc_correlation_matrix(fits).loc[tickers, tickers].values
    D = np.diag(vols)
    Sigma = D @ R @ D
    return pd.DataFrame(Sigma, index=tickers, columns=tickers)
