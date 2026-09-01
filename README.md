# portfolio-optimizer

A multi-asset **volatility forecasting → portfolio optimisation → backtesting
→ statistical evaluation** pipeline for BTC-USD, SPY, GLD and TLT.

Three independent one-step-ahead **covariance matrix** forecasting models
(GARCH(1,1)/CCC, a 3-state Markov regime model, and rolling-window XGBoost)
are walked forward through history, validated for numerical soundness, and
fed into a weekly portfolio optimiser under **two objective functions**
(Max-Sharpe and Max-Return) and **three transaction-cost regimes**
(0.0% / 0.1% / 0.3%), benchmarked against a static equal-weight baseline —
with bootstrapped confidence intervals, drawdown/turnover diagnostics, a
formal Markov-order model-selection test, and dedicated market-stress
sub-period analysis on top.

The evaluation stage directly answers three secondary research questions:

- **SRQ 1 — Objective comparison:** does Max-Sharpe outperform Max-Return
  across GARCH/Markov/XGBoost?
- **SRQ 2 — Transaction-cost sensitivity:** how much Sharpe ratio decays
  going from 0.0% to 0.3% friction?
- **SRQ 3 — Market-stress performance:** do volatility-informed strategies
  hold up during the COVID-19 crash (2020-02 to 2020-05) and the 2022 Fed
  rate-hike cycle (2022-01 to 2022-12)?

```
data.py ─┬─► features.py ─► models_garch.py ────┐
         │                  models_markov.py ────┼─► main.py ─► outputs/forecasts.pkl
         └─────────────────►models_xgboost.py ───┘        │  (+ covariance_validation_results.csv)
                                                            ▼
                                            portfolio_backtester.py ─► outputs/portfolio_results.csv
                                                            │
                                                            ▼
                                              statistical_tests.py ─► figures/*.png + console tables
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the three stages in sequence — each reads the previous stage's output
from `outputs/`:

```bash
# 1. Forecast weekly covariance matrices for all three models, walk-forward,
#    and persist validated forecasts + realised returns.
python main.py --start 2016-01-01 --min-train-weeks 104 --refit-every 1

# 2. Simulate weekly rebalancing for every (model x objective) combination
#    plus equal-weight, replayed across all three fee tiers.
python portfolio_backtester.py

# 3. Compute performance metrics, bootstrap CIs, SRQ1/SRQ2/SRQ3 tables,
#    the Markov order-selection test, and render the publication figures.
python statistical_tests.py
```

All three scripts seed `numpy`'s global RNG with `42` for reproducibility
and create their own output directories (`outputs/`, `figures/`) on demand.

### Stage 1 — `main.py`

| Argument | Default | Meaning |
|---|---|---|
| `--tickers` | `BTC-USD SPY GLD TLT` | Asset universe |
| `--start` / `--end` | `2016-01-01` / today | Sample date range passed to yfinance |
| `--min-train-weeks` | `104` (2y) | Burn-in before the first forecast |
| `--refit-every` | `1` | Re-estimate GARCH/Markov every N weeks; XGBoost always re-fits (fixed rolling window) |
| `--out` | `outputs/covariance_validation_results.csv` | Per-forecast validation log |
| `--forecasts-out` | `outputs/forecasts.pkl` | Persisted covariance forecasts + returns (consumed by stage 2/3) |

Console output: a pass-rate summary (`is_valid`/`is_symmetric`/`is_psd`/
`is_finite`) per model, plus a PASS/FAIL line for the whole run.

> **SRQ 3 date coverage.** The default `--start 2016-01-01` with no `--end`
> (i.e. through today) already spans both stress windows used in SRQ 3
> (COVID-19 crash, 2020-02 to 2020-05; 2022 Fed rate-hike cycle). If you
> narrow `--start`/`--end`, make sure the range still covers both, or
> `statistical_tests.py` will report "No data available" for the missing
> window instead of failing silently.

### Stage 2 — `portfolio_backtester.py`

| Argument | Default | Meaning |
|---|---|---|
| `--forecasts` | `outputs/forecasts.pkl` | Input from stage 1 |
| `--out` | `outputs/portfolio_results.csv` | Weekly portfolio simulation results |

Every run automatically sweeps **both objectives** (`Max-Sharpe`,
`Max-Return`) for **all three models**, plus `Equal-Weight`, each replayed
across **all three fee tiers** (`0.0%`, `0.1%`, `0.3%`) — 7 strategies × 3
fee tiers = 21 `(strategy, fee_rate)` series per run; no flags needed to
opt into the sweep. Console output: mean/std/annualised Sharpe per
strategy at the default 0.1% fee tier.

### Stage 3 — `statistical_tests.py`

| Argument | Default | Meaning |
|---|---|---|
| `--results` | `outputs/portfolio_results.csv` | Input from stage 2 |
| `--forecasts` | `outputs/forecasts.pkl` | Input from stage 1 (for the Markov regime sequence) |
| `--figures-dir` | `figures/` | Output directory for the three PNG figures |
| `--fee-rate` | `0.001` | Reference fee tier for the default summary, bootstrap CIs, SRQ 1, SRQ 3 and figures (SRQ 2 always sweeps every fee tier) |

Console output: the default performance summary + bootstrap Sharpe CI
table, the **SRQ 1** objective-comparison table (+ a data-driven verdict),
the **SRQ 2** fee-sensitivity matrix, the **SRQ 3** stress-window tables,
and the Markov order-selection (LRT/AIC/BIC) diagnostic.

## Module layout

| File | Contents |
|---|---|
| `data.py` | yfinance download, weekly (`W-FRI`) resampling, weekly log-returns, weekly realised volatility |
| `features.py` | XGBoost feature engineering (lagged vol, rolling std, cross-asset vol, day-of-week) |
| `models_garch.py` | Per-asset GARCH(1,1) + CCC-GARCH covariance construction |
| `models_markov.py` | 3-state K-Means regime classification, transition matrix, mixture covariance |
| `models_xgboost.py` | Rolling-window XGBoost volatility regressors + correlation-scaled covariance |
| `covariance.py` | Symmetry / PSD / finiteness validation utilities |
| `main.py` | Walk-forward orchestration, CLI, and persistence of covariance forecasts to `outputs/forecasts.pkl` |
| `portfolio_backtester.py` | Weekly Max-Sharpe / Max-Return portfolio optimiser (SciPy SLSQP), swept across 3 transaction-cost tiers, equal-weight baseline |
| `statistical_tests.py` | Performance metrics, bootstrap CIs, SRQ1/SRQ2/SRQ3 tables, Markov order-selection LRT, and figure generation |

