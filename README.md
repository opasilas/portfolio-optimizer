<<<<<<< HEAD
# vol_pipeline

Weekly multi-asset volatility / covariance modelling pipeline for BTC-USD,
SPY, GLD and TLT: GARCH(1,1), a 3-state Markov chain, and an XGBoost
regressor, each producing a one-step-ahead **covariance matrix** forecast,
with validation that every forecast is a proper covariance matrix
(symmetric, positive semi-definite, finite).

This was built to slot straight into a walk-forward Sharpe-ratio backtest
(Σ_t feeding a mean-variance or risk-parity optimiser each week).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m vol_pipeline.main --start 2016-01-01 --min-train-weeks 104 --refit-every 1
```

Key arguments:
- `--tickers` — default `BTC-USD SPY GLD TLT`
- `--start` / `--end` — sample date range passed to yfinance
- `--min-train-weeks` — burn-in before the first forecast (default 104 = 2y)
- `--refit-every` — how often (in weeks) to re-estimate GARCH/Markov; XGBoost
  always re-fits each step since it trains on a fixed rolling window
- `--out` — CSV path for the per-forecast validation log

Output: a CSV with one row per `(forecast_date, model)` containing
`is_valid`, `is_symmetric`, `is_psd`, `is_finite`, `min_eigenvalue`, plus a
console summary of the pass-rate per model.

## Module layout

| File | Contents |
|---|---|
| `data.py` | yfinance download, weekly resampling, weekly log-returns, weekly realised volatility |
| `features.py` | XGBoost feature engineering (lagged vol, rolling std, cross-asset vol, day-of-week) |
| `models_garch.py` | Per-asset GARCH(1,1) + CCC-GARCH covariance construction |
| `models_markov.py` | 3-state K-Means regime classification, transition matrix, mixture covariance |
| `models_xgboost.py` | Rolling-window XGBoost volatility regressors + correlation-scaled covariance |
| `covariance.py` | Symmetry / PSD / finiteness validation utilities |
| `main.py` | Walk-forward orchestration and CLI |

## Methodology notes and design choices

**Weekly log-returns.** `r_t = ln(P_t / P_{t-1})` on Friday-anchored weekly
closes (`W-FRI` resample). BTC-USD (7-day market) and SPY/GLD/TLT (NYSE
calendar) are aligned automatically at the weekly frequency rather than
forced to align daily.

**Weekly realised volatility (RV).** Computed from the underlying *daily*
log-returns within each calendar week, `RV_t = sqrt(sum_{d in week} r_d^2)`,
annualised by `sqrt(52)`. This is the standard realised-variance estimator
and is used both as the Markov regime indicator and as the XGBoost
regression target. Weeks with fewer than 2 daily observations (holiday
weeks) are dropped as unreliable.

**GARCH(1,1) / CCC-GARCH.** Each asset gets an independent constant-mean
GARCH(1,1) with Normal innovations (`arch_model(..., vol="GARCH", p=1, q=1)`,
returns rescaled ×100 for solver stability per the `arch` docs). The
multivariate covariance forecast uses the Constant Conditional Correlation
construction (Bollerslev, 1990): `Sigma_t = D_t R D_t`, with `D_t` the
diagonal of one-step-ahead GARCH volatility forecasts and `R` the
unconditional correlation matrix of the standardised GARCH residuals. This
is a standard, tractable way to get a full covariance matrix out of
univariate GARCH models without estimating a full multivariate GARCH
(BEKK/DCC), which is a reasonable simplification to document as a
limitation in the dissertation if a genuine DCC extension isn't in scope.

**3-state Markov chain.** The regime variable is the equal-weighted average
of each asset's z-scored weekly RV (a simple market-wide volatility proxy,
since transitions should reflect the *joint* volatility environment, not
just one asset). States are assigned by K-Means (k=3), ordered by cluster
centre so state 0/1/2 = Low/Medium/High vol, and a first-order transition
matrix is estimated by counting observed transitions in-sample. Each state
carries its own sample covariance matrix of returns (computed only from
weeks assigned to that state). The forecast is the transition-probability-
weighted mixture of the three state covariances — a convex combination of
PSD matrices, so it is PSD by construction, and it naturally captures
regime persistence/mean-reversion via the transition matrix rather than
just picking the single most likely next state.

**XGBoost.** One `XGBRegressor` per asset, retrained each week on a
**rolling** (not expanding) window of ~500 trading days (`ROLLING_WINDOW_DAYS
= 500`, converted to ~100 weekly observations via
`500 // 5 trading days/week`). Features (see `features.py`):
- `own_rv_lag_{1,2,4,8}` — own lagged realised vol,
- `own_rollstd_{4,8,12}` — rolling std of own weekly returns (lagged 1 week
  to avoid look-ahead),
- `{other}_rv_lag1` — lag-1 realised vol of every other asset in the
  universe (cross-asset volatility spillover),
- `dow_weekend` — day-of-week of the week-ending date (captures
  holiday-shifted week boundaries; SPY/GLD/TLT weeks sometimes end on a day
  other than Friday around US holidays).

Predicted per-asset volatilities are combined into a covariance matrix
using the same `D_t R_t D_t` construction as GARCH, with `R_t` the
empirical correlation matrix of returns over the same rolling window — this
keeps the three covariance forecasts structurally comparable for the
Sharpe-ratio backtest, isolating the effect of *how volatility is
forecast* rather than how correlation is estimated.

**Validation.** `covariance.validate_covariance(Sigma)` checks (1) all
entries finite, (2) `Sigma` symmetric within `1e-6`, (3) smallest eigenvalue
of the symmetrised matrix `>= -1e-8` (PSD). `main.run_walk_forward` runs
this check on every model's forecast at every step and reports a pass-rate
table plus a CSV log — this is the artefact to cite/screenshot in the
dissertation as evidence the Σ inputs to the portfolio optimiser were
well-defined throughout the backtest window.

## Known simplifications to flag in the write-up

- CCC-GARCH assumes a *constant* correlation matrix rather than a full
  DCC/BEKK; reasonable as a baseline, but worth naming explicitly as a
  limitation relative to a true multivariate GARCH.
- The Markov regime variable is a simple equal-weighted average vol proxy,
  not a formally estimated latent factor (e.g. via PCA or a fitted HMM) —
  fine for a transparent baseline, but an HMM would let the transition
  matrix and state assignment be jointly (MLE) estimated instead of
  K-Means + counting.
- XGBoost's cross-asset covariance still borrows the empirical correlation
  matrix (it doesn't jointly model correlation); only the diagonal
  (per-asset vol) comes from the ML model, which is consistent with your
  research question of isolating the *volatility forecasting* effect but
  is worth stating explicitly in the methodology section.
=======
# portfolio-optimizer
>>>>>>> fcff5be6b433b793331d092555ee1583ac85acc0
