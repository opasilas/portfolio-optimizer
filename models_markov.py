"""
models_markov.py
-----------------
Discrete 3-state Markov-chain regime model for the covariance forecast.

The regime state is defined on an equal-weighted average of the four
assets' z-scored realised volatility (a simple market-wide vol proxy).
States are assigned via K-Means (k=3: Low / Medium / High volatility),
a first-order transition matrix is estimated by counting observed
state-to-state transitions, and each state has its own sample covariance
matrix of asset returns computed from the weeks assigned to that state.

The one-step-ahead covariance forecast is the transition-probability
weighted mixture of the three state covariance matrices:

    Sigma_{t+1|t} = sum_k  P(s_t -> k) * Sigma_k

which is guaranteed to be positive semi-definite because a convex
combination (weights >= 0, sum to 1) of PSD matrices is itself PSD, and
every sample covariance matrix is PSD by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

N_STATES = 3
RANDOM_STATE = 42


@dataclass
class MarkovFit:
    kmeans: KMeans
    state_order: np.ndarray            # raw KMeans label -> ordered 0=Low,1=Med,2=High
    transition_matrix: np.ndarray      # shape (N_STATES, N_STATES), row = from-state
    state_covariances: Dict[int, pd.DataFrame]
    state_sequence: pd.Series
    tickers: List[str]


def _z_score(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std > 0 else s * 0.0


def _regime_indicator(weekly_rv: pd.DataFrame) -> pd.Series:
    """Equal-weighted average of per-asset z-scored realised volatility."""
    z = weekly_rv.apply(_z_score)
    return z.mean(axis=1)


def fit_markov(weekly_returns: pd.DataFrame, weekly_rv: pd.DataFrame) -> MarkovFit:
    tickers = list(weekly_returns.columns)
    indicator = _regime_indicator(weekly_rv).dropna()
    aligned_returns = weekly_returns.loc[weekly_returns.index.intersection(indicator.index)]
    indicator = indicator.loc[aligned_returns.index]

    km = KMeans(n_clusters=N_STATES, n_init=10, random_state=RANDOM_STATE)
    raw_labels = km.fit_predict(indicator.values.reshape(-1, 1))

    # Re-order raw cluster labels by centre so that 0 = Low vol, 2 = High vol.
    centre_order = np.argsort(km.cluster_centers_.flatten())
    state_order = np.zeros(N_STATES, dtype=int)
    for ordered_idx, raw_idx in enumerate(centre_order):
        state_order[raw_idx] = ordered_idx
    ordered_labels = state_order[raw_labels]
    state_seq = pd.Series(ordered_labels, index=indicator.index, name="state")

    transition = np.zeros((N_STATES, N_STATES))
    seq = state_seq.values
    for t in range(len(seq) - 1):
        transition[seq[t], seq[t + 1]] += 1
    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = transition / row_sums

    state_covariances: Dict[int, pd.DataFrame] = {}
    fallback_cov = aligned_returns.cov()
    for s in range(N_STATES):
        r_s = aligned_returns.loc[state_seq[state_seq == s].index]
        state_covariances[s] = r_s.cov() if len(r_s) >= max(len(tickers), 2) else fallback_cov

    return MarkovFit(
        kmeans=km,
        state_order=state_order,
        transition_matrix=transition,
        state_covariances=state_covariances,
        state_sequence=state_seq,
        tickers=tickers,
    )


def current_state(fit: MarkovFit, weekly_rv: pd.DataFrame) -> int:
    """Assign the most recent week in `weekly_rv` to one of the three states."""
    indicator = _regime_indicator(weekly_rv).dropna()
    last_val = float(indicator.iloc[-1])
    raw_label = fit.kmeans.predict(np.array([[last_val]]))[0]
    return int(fit.state_order[raw_label])


def markov_forecast_covariance(fit: MarkovFit, weekly_rv: pd.DataFrame) -> pd.DataFrame:
    """One-step-ahead covariance forecast: transition-weighted mixture of state covariances."""
    s_t = current_state(fit, weekly_rv)
    probs = fit.transition_matrix[s_t]
    Sigma = sum(p * fit.state_covariances[s].values for s, p in enumerate(probs))
    return pd.DataFrame(Sigma, index=fit.tickers, columns=fit.tickers)
