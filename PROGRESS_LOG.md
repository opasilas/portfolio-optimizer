# Progress Log

This log records weekly development progress on the portfolio-optimizer project, together with a summary of each supervision meeting held with Prof. Peter Tino over the course of the dissertation. Because development was carried out primarily in a local environment with infrequent commits (see Appendix A of the dissertation), this log is the more complete record of how the project actually progressed week to week.

**Contents**
1. [Week 1 — 22 June to 28 June](#week-1--22-june-to-28-june) — scoping candidate project directions
2. [Week 2 — 29 June to 5 July](#week-2--29-june-to-5-july) — options/straddle data ruled out
3. [Week 3 — 6 July to 12 July](#week-3--6-july-to-12-july) — pivot to Sharpe-ratio-maximising portfolio construction
4. [Week 4 — 13 July to 19 July](#week-4--13-july-to-19-july) — data-ingestion layer
5. [Week 5 — 20 July to 26 July](#week-5--20-july-to-26-july) — three-model comparison design settled
6. [Week 6 — 27 July to 2 August](#week-6--27-july-to-2-august) — CCC-GARCH and Markov regime mixture implemented
7. [Week 7 — 3 August to 9 August](#week-7--3-august-to-9-august) — XGBoost forecaster and walk-forward loop
8. [Week 8 — 10 August to 16 August](#week-8--10-august-to-16-august) — optimiser, fee sweep, statistical evaluation
9. [Week 9 — 17 August to 23 August](#week-9--17-august-to-23-august) — full backtest and results restructuring
10. [Week 10 — 24 August to 31 August](#week-10--24-august-to-31-august) — demo, final meeting, submission

---

## Week 1 — 22 June to 28 June

**Supervisor meeting:** Initial meeting, Tue 23-06.
- Introductions and discussion of research ideas.
- I proposed a finance-related project, possibly around credit-worthiness/ratings modelling or algorithmic trading.
- Peter advised further reading before committing to a direction, and stressed choosing a scope that was realistically completable in the time available.
- Volatility prediction (standard deviation of returns) was raised as a candidate technical core.
- Broader framing discussed: portfolio management under high-yield/high-risk assets, combining assets to maximise profit, option pricing, and risk estimation via volatility prediction; possibly formulating an original trading strategy (after reviewing existing ones); the underlying question of making allocation decisions from predictive modelling versus historical data alone.

**What I achieved this week**
- Surveyed several candidate project directions in quantitative finance: credit-scoring/ratings prediction, algorithmic trading strategy design, and volatility-based portfolio construction.
- Read introductory material on portfolio theory and volatility as a risk measure to prepare for the initial meeting.

**Challenges encountered**
- The candidate topics were all plausible but very different in scope and data requirements, making it hard to commit to one without supervisor input.

**How I addressed them**
- Used the initial meeting to present all candidate directions rather than pre-committing, and asked directly for feedback on feasibility given the project timeline.

**Plans for next week**
- Narrow down to a single research direction informed by Peter's feedback.
- Investigate the ML-for-finance angle in more depth, particularly around portfolio optimisation.

---

## Week 2 — 29 June to 5 July

**Supervisor meeting:** Meeting 2, Tue 30-06.
- I proposed exploring machine learning applications in finance, specifically for portfolio optimisation, building on Peter's own prior work in this area.
- Peter explained the specifics of straddle and options pricing and asked me to confirm data availability, since a straddle-based approach requires options price data specifically.
- He raised the question of whether options on crypto assets exist at all, and cautioned that this data might be unavailable or prohibitively expensive to obtain, asking me to check before committing to this direction.

**What I achieved this week**
- Settled on machine learning for portfolio optimisation as the general project direction.
- Investigated options and straddle-pricing data availability for both traditional and crypto assets, including checking commercial and free data providers.

**Challenges encountered**
- Options data, and crypto options data in particular, was either unavailable through free sources or required paid institutional-grade data feeds well outside the project's budget and timeline.

**How I addressed them**
- Documented exactly which providers were checked and why each was unsuitable, so this could be reported back to Peter as a concrete finding rather than a vague "couldn't find it."

**Plans for next week**
- Bring the data-availability findings back to Peter and discuss whether to pursue an alternative data source or pivot the project's asset focus.

---

## Week 3 — 6 July to 12 July

**Supervisor meeting:** Meeting 3, Tue 07-07.
- I explained the project pivot away from options/straddle data, which was unavailable, toward standard exchange-listed asset pricing, with the research focus shifting to portfolio optimisation for conventional asset classes rather than options.
- Peter asked me to still pursue a full portfolio-construction system, but built around stocks rather than options.
- Rather than pushing the models to their theoretical extremes, the focus was set on maximising the Sharpe ratio as the primary optimisation objective.
- Peter approved this direction and asked me to additionally explore risk-adjusted returns versus pure return maximisation, and to use a diversified, multi-asset-class universe rather than a single asset class.

**What I achieved this week**
- Formally repositioned the project around Sharpe-ratio-maximising portfolio construction over standard, liquid asset classes.
- Began scoping a multi-asset universe spanning different risk/liquidity profiles rather than equities alone, in line with Peter's diversification guidance.

**Challenges encountered**
- Needed to decide on a concrete, defensible asset universe quickly to avoid losing further time before implementation could start.

**How I addressed them**
- Selected four assets deliberately spanning distinct risk and macroeconomic sensitivity profiles — Bitcoin (BTC-USD), large-cap US equities (SPY), gold (GLD), and long-dated US Treasuries (TLT) — so the resulting covariance structure under test would be genuinely cross-asset-class.

**Plans for next week**
- Begin implementation: set up the data-ingestion layer for the four-asset universe and establish the weekly return/volatility conventions the rest of the pipeline would build on.

---

## Week 4 — 13 July to 19 July

*(No supervisor meeting this week — independent implementation.)*

**What I achieved this week**
- Started implementation of the data-ingestion stage: pulling raw daily prices for BTC-USD, SPY, GLD and TLT and resampling to a common weekly frequency.
- Implemented the weekly log-return and realised-volatility (RV) proxy calculations that the rest of the pipeline would depend on.

**Challenges encountered**
- Bitcoin trades continuously on a 24/7 basis, while SPY, GLD and TLT follow the NYSE trading calendar, so naive daily alignment either discarded Bitcoin's weekend price action or introduced spurious gaps for the other three instruments.

**How I addressed them**
- Anchored all four series to Friday-ending weekly closes (`W-FRI` resampling) as a common synchronisation point, and computed realised volatility from the underlying intra-week daily returns rather than the weekly close alone, so the dispersion of daily price movements within the week wasn't discarded.

**Plans for next week**
- Bring the working data-ingestion layer to the next supervision meeting.
- Start reading around volatility forecasting methods ahead of committing to specific models, per the project's next phase.

---

## Week 5 — 20 July to 26 July

**Supervisor meeting:** Meeting 4, Tue 21-07.
- Peter asked for something more concrete and requested I explore different methods of volatility forecasting specifically.
- He asked me to read on implied volatility, stochastic volatility models, Black-Scholes, and their respective limitations.
- Peter advised sticking to historical-volatility-based approaches and GARCH as the econometric baseline for covariance forecasting.
- He also asked me to research the best candidate models more broadly, explicitly advising against Hidden Markov Models or complex transformer architectures given the project's data and time constraints.

**What I achieved this week**
- Reviewed implied volatility, stochastic volatility, and Black-Scholes-style approaches and documented why each was a poor fit for this project (data requirements, calibration complexity, or reliance on options markets already ruled out).
- Converged on a three-paradigm comparison design: an econometric GARCH baseline, a simpler discrete-state alternative to a full HMM, and a machine learning challenger.

**Challenges encountered**
- Needed an alternative to a full Hidden Markov Model that still captured regime-switching behaviour without the estimation burden Peter had flagged as impractical for the project's scope.

**How I addressed them**
- Settled on a K-means-based discrete regime classifier (low-vol/medium-vol/high-vol states) with an empirically estimated transition matrix, as a tractable, transparent alternative to a fully MLE-estimated HMM, deliberately trading some statistical elegance for implementation robustness within the timeline.

**Plans for next week**
- Begin implementing the GARCH(1,1)/CCC covariance module as the econometric baseline.
- Prototype the K-means regime-classification approach against the realised-volatility series.

---

## Week 6 — 27 July to 2 August

*(No supervisor meeting this week — independent implementation.)*

**What I achieved this week**
- Implemented the CCC-GARCH(1,1) covariance forecaster: per-asset univariate GARCH(1,1) models combined via Bollerslev's Constant Conditional Correlation decomposition into a full covariance matrix.
- Implemented the 3-state Markov regime mixture: K-means clustering (k=3) on a composite z-scored volatility index, an empirically estimated transition matrix, and a transition-weighted convex-combination covariance forecast.
- Began the numerical validation layer (`covariance.py`) to check every forecasted covariance matrix for finiteness, symmetry, and positive semi-definiteness before it could be used downstream.

**Challenges encountered**
- Early GARCH fits occasionally failed to converge cleanly, and floating-point residuals in the Markov mixture construction occasionally violated strict symmetry at machine precision.

**How I addressed them**
- Rescaled returns by a factor of 100 prior to GARCH fitting for better-conditioned optimisation (standard practice for the underlying estimation library), and built the validation gate to symmetrise matrices within a small numerical tolerance and check the smallest eigenvalue rather than assuming exact textbook properties would hold in floating-point arithmetic.

**Plans for next week**
- Start implementation of the machine learning (XGBoost) covariance forecaster.
- Prepare a working multi-model demonstration for the next supervision meeting.

---

## Week 7 — 3 August to 9 August

**Supervisor meeting:** Meeting 5, Tue 04-08 (at the library).
- Most of the project's core pipeline was in place by this point.
- I presented the revised work, showing the completed data-ingestion stage and the multi-model forecasting pipeline running end to end.
- Peter asked me to extend the Markov model to a second-order specification and formally check whether the additional order was statistically justified, and to include additional research questions to make the evaluation more comprehensive.

**What I achieved this week**
- Implemented the rolling-window XGBoost volatility regressors (one `XGBRegressor` per asset, retrained on a fixed 500-trading-day rolling window), completing the third of the three covariance forecasting paradigms.
- Brought all three models into a single walk-forward orchestration loop with a common `D_t R_t D_t` covariance output contract, so the three paradigms could be swapped in and out of the downstream optimiser interchangeably.

**Challenges encountered**
- Needed to respond to Peter's request for a formal Markov order-selection test without simply asserting the first-order specification was "good enough."

**How I addressed them**
- Designed a nested Likelihood Ratio Test comparing the first-order model against a second-order model fitted on the identical sequence of observed regime-state triples, so the two specifications were properly nested and the test was statistically valid, backed up with AIC/BIC as corroborating evidence. This became SRQ 4 in the final dissertation.

**Plans for next week**
- Implement the Markov order-selection diagnostic.
- Begin work on the portfolio optimisation and backtesting stage (Stage 2), since the forecasting stage was now functionally complete.

---

## Week 8 — 10 August to 16 August

*(No supervisor meeting this week — independent implementation.)*

**What I achieved this week**
- Implemented the constrained SLSQP portfolio optimiser under both Max-Sharpe and Max-Return objectives, with the fully-invested, long-only simplex constraints.
- Implemented the turnover-friction layer and the three-tier transaction-cost sweep (0.0%, 0.1%, 0.3%), designed so a single optimised weight path could be replayed across all three fee tiers without re-optimising.
- Implemented the statistical evaluation stage: the 1,000-resample bootstrap confidence intervals on annualised Sharpe, and the SRQ 1–3 comparison tables.
- Implemented the nested Markov order-selection LRT/AIC/BIC diagnostic agreed at the previous meeting.

**Challenges encountered**
- Under the Max-Return objective, the optimiser collapsed to a 100% single-asset corner solution at every rebalancing step, which initially looked like a bug in the solver.
- Deciding how to fairly compare strategies across fee tiers without conflating the effect of the optimisation objective with the effect of transaction costs took some care.

**How I addressed them**
- Confirmed analytically that the corner-solution behaviour was the mathematically correct outcome of a purely linear objective over a probability simplex (the true optimum of a linear programme is a vertex of the feasible region), not a solver bug — this became a substantive finding (SRQ 1) rather than a defect to fix.
- Separated the optimisation and fee-replay steps structurally, so each strategy's weight path is optimised exactly once and then replayed unmodified against all three fee tiers, isolating the fee-sensitivity comparison (SRQ 2) from the optimisation objective itself.

**Plans for next week**
- Run the full walk-forward backtest end to end and extract the headline results.
- Prepare a clear, results-driven presentation ahead of the pre-demo meeting.

---

## Week 9 — 17 August to 23 August

**Supervisor meeting:** Meeting 6 (pre-demo), Fri 21-08.
- Peter reviewed the demo materials and gave feedback on presenting the results well — specifically, to "tell a story" with the presentation rather than just listing numbers.
- He asked that the research questions be presented explicitly rather than left implicit.
- He asked for a clearer explanation of the Markov regime-switching model in particular, and stressed the importance of being able to explain the reasoning behind every result, not just report it.
- I presented the results and findings organised around the research questions.

**What I achieved this week**
- Completed the full walk-forward backtest (449 weekly steps) across all three covariance models, two objectives, and three fee tiers, and generated the headline performance tables and figures.
- Restructured the results presentation around the Primary RQ and SRQs 1–4 directly, following Peter's feedback.
- Prepared clearer explanatory material on the Markov regime-mixture construction for the demo.

**Challenges encountered**
- The initial results presentation was organised around the models rather than the research questions, which made it harder to follow and, per Peter's feedback, didn't clearly justify why each result mattered.

**How I addressed them**
- Reorganised the entire results narrative section-by-section around the Primary RQ and each SRQ in turn, so every reported number was explicitly tied back to the question it was answering.

**Plans for next week**
- Deliver the demo to Vincent Rahli.
- Begin writing up the full dissertation text from the now-finalised results.

---

## Week 10 — 24 August to 31 August

**Demo day:** Tue 25-08, with Vincent Rahli.
- Presented the complete project and answered questions on the methodology and findings.

**Supervisor meeting:** Meeting 7 (final meeting), Fri 28-08.
- Peter reviewed the full written report.
- I clarified the z-score standardisation used for the K-means/Markov regime discretisation, and the lookback-window dimensions used for the XGBoost feature columns.
- I thanked Peter for his mentorship and support throughout the project; Peter commended my dedication to the work.

**What I achieved this week**
- Delivered the project demo successfully.
- Incorporated Peter's final feedback on the written report, including clarifying the K-means/Markov standardisation and the XGBoost feature lookback windows in the methodology chapter.
- Finalised all five chapters, the front matter, and the appendices, and resolved outstanding citation, cross-referencing, and figure issues ahead of submission.

**Challenges encountered**
- Reconciling every numerical result quoted in the results chapter against the underlying pipeline output, and ensuring every citation and cross-reference in the final document resolved correctly, took longer than expected this close to the deadline.

**How I addressed them**
- Went through the compiled document systematically section by section, checking every citation, figure, and cross-reference against the source pipeline output rather than relying on a single final read-through.

**Plans for next week**
- Final proofread and submission.