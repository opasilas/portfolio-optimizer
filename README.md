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

## Methodology notes and design choices

### Volatility / covariance forecasting (stage 1)

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
table plus a CSV log. Only forecasts that pass validation are persisted to
`outputs/forecasts.pkl` for downstream use — an invalid Σ is not a usable
input to a portfolio optimiser, so it is dropped (and logged) rather than
silently propagated.

**Persistence (`outputs/forecasts.pkl`).** A pickled `dict` with:
- `tickers` — asset order used throughout (`[BTC-USD, SPY, GLD, TLT]`),
- `weekly_returns` — the *full* realised weekly log-return history (including
  the pre-forecast burn-in period), used by the backtester to compute a
  genuine 52-week trailing mean even at the very first forecast date,
- `forecasts` — one row per `(date, model)` with `cov_matrix` (4×4
  `np.ndarray`) and `next_week_returns` (4, `np.ndarray`, realised log-returns
  for that forecast date, same asset order as `tickers`),
- `regime_states` — a full-sample K-Means regime-state sequence (0/1/2) from
  a *separate* one-off `fit_markov` call, used only by the Requirement-3
  Markov order-selection diagnostic in `statistical_tests.py` — it plays no
  role in the walk-forward forecasts themselves, which stay strictly
  no-look-ahead.

### Portfolio optimisation & backtesting (stage 2)

**Objectives (SRQ 1).** At every forecast date `t`, each of GARCH/Markov/
XGBoost is optimised under two objectives:

```
Max-Sharpe:  maximise  (w' mu_t - Rf) / sqrt(w' Sigma_t w)
Max-Return:  maximise  w' mu_t

subject to (both):  sum(w) = 1,   0 <= w_i <= 1
```

via `scipy.optimize.minimize` (SLSQP) on the negative objective, with
`Rf = 0`. `mu_t` is the trailing 52-week **historical** mean return vector,
estimated strictly from weeks before `t` (no look-ahead); `Sigma_t` is that
model's validated covariance forecast (unused by Max-Return — see
limitations below). This gives six model×objective strategies
(`GARCH-Sharpe`, `GARCH-Return`, `Markov-Sharpe`, `Markov-Return`,
`XGBoost-Sharpe`, `XGBoost-Return`), plus a static 25%-per-asset
`Equal-Weight` baseline evaluated on the same forecast dates with no
re-optimisation.

**Transaction-cost sweep (SRQ 2).** Portfolio weights depend only on
`(mu_t, Sigma_t)`, never on the fee rate, so each `(model, objective)`
weight path is optimised **once** and then replayed against three
turnover-fee rates without re-running the optimiser:

```
Fee_t = fee_rate * sum(|w_t - w_{t-1}|),   fee_rate in {0.0%, 0.1%, 0.3%}
Net Return_t = (w_t' r_{t+1}) - Fee_t
```

`0.0%` is the frictionless baseline, `0.1%` the default realistic
assumption, and `0.3%` a crypto-realistic high-friction tier. The very
first rebalance for each strategy is charged against an implicit all-cash
`w_0 = 0` starting position (i.e. paying to enter the initial allocation),
which is a deliberate, documented convention rather than an oversight.

**Output (`outputs/portfolio_results.csv`).** One row per
`(date, strategy, fee_rate)` — 7 strategies × 3 fee tiers per forecast
date: `date`, `strategy` (e.g. `GARCH-Sharpe`), `objective` (`Max-Sharpe` /
`Max-Return` / `Equal-Weight`), `fee_rate` (`0.0` / `0.001` / `0.003`),
`net_return`, `gross_return`, `turnover_fee`, `w_BTC`, `w_SPY`, `w_GLD`,
`w_TLT`, `cumulative_equity` (compounded net return, per fee tier).

### Statistical evaluation (stage 3)

**Reference view.** The default performance summary, bootstrap CIs and all
three figures use a fixed slice of `portfolio_results.csv` — the
`--fee-rate` tier (default 0.1%) restricted to `Max-Sharpe` strategies +
`Equal-Weight` — so they read the same as before the SRQ1/2/3 extension.
The SRQ tables below each define their own slice of the full 21-series
result set.

**Performance summary.** Per strategy: annualised return (`mean × 52`),
annualised volatility (`std × sqrt(52)`), annualised Sharpe (return/vol),
maximum drawdown (peak-to-trough on `cumulative_equity`), and total turnover
(`sum(turnover_fee) / fee_rate`), ranked by Sharpe.

**Bootstrap confidence intervals.** For each strategy, 1,000
resamples-with-replacement of its weekly net returns (`numpy.random.choice`,
seeded) each produce an annualised Sharpe estimate; the 2.5th/97.5th
percentiles of that distribution give the 95% CI. This treats weekly net
returns as i.i.d., which understates uncertainty under volatility clustering
or autocorrelation — worth flagging as a limitation alongside a
block-bootstrap extension.

**SRQ 1 — objective comparison (`build_srq1_table` / `srq1_verdict`).** At
the reference fee tier, the six model×objective strategies are compared on
annualised return/volatility/Sharpe, max drawdown and total turnover.
`srq1_verdict` then compares the *average* Sharpe/volatility/drawdown across
the three `-Sharpe` strategies vs the three `-Return` strategies and reports,
in plain language, which objective actually managed risk better **in that
specific backtest** — it is computed from the results each run, not asserted
in advance, so it can (and, in small/adversarial samples, did during
testing) report Max-Return outperforming Max-Sharpe on realised Sharpe.

**SRQ 2 — transaction-cost sensitivity (`build_srq2_table`).** For every
strategy, annualised Sharpe is computed independently at each fee tier, plus

```
Sharpe Decay (%) = (Sharpe_0.3% - Sharpe_0.0%) / Sharpe_0.0% * 100
```

exactly as specified. Note this is a *signed* ratio: if `Sharpe_0.0%` is
negative, a larger negative `Sharpe_0.3%` produces a positive "decay"
percentage (performance got worse but the formula reads as improvement) —
read the raw Sharpe columns alongside the decay column rather than the
decay column in isolation.

**SRQ 3 — market-stress sub-periods (`build_srq3_tables` /
`window_metrics`).** At the reference fee tier, `portfolio_results.csv` is
sliced to two fixed windows — COVID-19 crash (`2020-02-01` to
`2020-05-31`) and the 2022 Fed rate-hike cycle (`2022-01-01` to
`2022-12-31`) — and for every strategy present in that window, equity is
**rebased to 1.0 at the window's first date** before computing cumulative
window return, annualised volatility, max drawdown and Sharpe, so each
window's numbers describe only that window, not a slice of the full-sample
equity curve. The COVID window is short (~17 weeks), so its annualised
figures carry a wide margin of statistical noise — treat them as
descriptive, not inferential.

**Markov order-selection test.** Verifies that the 3-state K-Means regime
sequence (`regime_states` from stage 1) is adequately described by a
first-order chain rather than requiring a second-order one. Both models are
fit on the *identical* set of state triples `(s_t, s_{t+1}, s_{t+2})`: the
order-2 transition tensor `P(s_{t+2} | s_t, s_{t+1})` is estimated directly,
and the order-1 null model `P(s_{t+2} | s_{t+1})` is obtained by
marginalising that same tensor over `s_t` — so the order-1 model is properly
nested inside order-2, making the likelihood-ratio test valid (Anderson &
Goodman, 1957). Reports log-likelihood, AIC, BIC and the LRT `p`-value for
both orders.

**Figures (`figures/`).** All four use the reference view (default fee
tier, Max-Sharpe + Equal-Weight) so they stay readable rather than
overlaying all 21 series. Weight-evolution charts use a fixed
`ASSET_COLORS` mapping (BTC blue, SPY orange, GLD green, TLT red) so both
allocation figures are visually consistent with each other:
- `Fig1_Equity_Curves.png` — cumulative net-return equity curves, all 4 strategies.
- `Fig2_Asset_Allocations.png` — stacked-area weight evolution for the
  highest-Sharpe strategy (selected automatically from the summary table —
  in practice this is often the static `Equal-Weight` baseline, which plots
  as a flat, uninformative band whenever it wins on Sharpe).
- `Fig2_Asset_Allocations_XGBoost.png` — the same stacked-area chart, but
  always for `XGBoost-Sharpe` specifically, regardless of which strategy
  ranks highest — the dedicated view for demonstrating the ML pipeline's
  dynamic rebalancing behaviour in slides/the dissertation. Skipped (with a
  console message, not a crash) if no `XGBoost-Sharpe` rows exist at the
  reference fee tier. Rendered at `dpi=300` with a year-only x-axis.
- `Fig3_Drawdowns.png` — drawdown-from-peak paths, all 4 strategies, to
  compare behaviour during stress periods.

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
- `mu_t` (expected returns) is a naive 52-week trailing sample mean, not a
  shrinkage estimator (e.g. Black-Litterman, Ledoit-Wolf) — mean-variance
  optimisers are notoriously sensitive to noisy mean estimates, so this is
  a standard, citable limitation of the Sharpe-maximising strategies here.
- The optimiser is long-only with no leverage and a flat 0.1% transaction
  cost irrespective of asset (crypto spreads/fees materially exceed 0.1% in
  practice) — a reasonable simplifying assumption to name explicitly.
- The bootstrap CI assumes i.i.d. weekly returns; a block/stationary
  bootstrap would be more defensible under volatility clustering.
- The Markov order-selection test runs on a single full-sample regime fit
  (for tractability), not refit at every walk-forward step — it verifies
  the *specification* choice (1st- vs 2nd-order) rather than re-testing it
  online during the backtest.
- **Max-Return ignores `Sigma_t` entirely**, so `GARCH-Return`,
  `Markov-Return` and `XGBoost-Return` are mathematically identical — the
  covariance model choice is irrelevant under a pure-return objective. This
  is itself a valid SRQ 1 finding (volatility forecasts only matter when the
  objective actually uses them), not a bug, but worth stating explicitly
  rather than presenting the three `-Return` rows as independent evidence.
- Max-Return is a linear objective, so its true optimum is a corner
  solution (100% into the single highest-`mu` asset, absent ties); it is
  solved via the same SLSQP machinery as Max-Sharpe purely for
  implementation consistency, not because SLSQP is necessary for an LP.
- The fee sweep replays a **fixed** weight path under three fee rates
  rather than re-optimising with turnover cost inside the objective — a
  genuinely fee-aware optimiser would trade off expected turnover against
  expected return/risk at each fee tier, which this backtester does not do
  (weights are identical across fee tiers by construction; only realised
  net returns differ).
- SRQ 2's `Sharpe Decay (%)` is a signed ratio and can read as "improvement"
  when the frictionless Sharpe is itself negative — see the SRQ 2 note
  above.
- SRQ 3's stress-window metrics depend entirely on `outputs/forecasts.pkl`
  actually covering 2020 and 2022; a narrow `--start`/`--end` in stage 1
  will make `statistical_tests.py` report "No data available" for the
  missing window rather than a misleading empty/zero result.
