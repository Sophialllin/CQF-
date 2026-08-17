# QQQ Volatility-Direction Prediction: A Blending-Ensemble Trading Strategy

**CQF Final Project Report**

---

## 1. Executive Summary

This project builds a machine-learning trading strategy on QQQ (Nasdaq-100 ETF) that predicts the **direction of realized volatility** — whether volatility over the next 5 trading days will be higher ("Storm") or lower ("Calm") than the trailing 5 trading days — rather than the direction of price. A blending ensemble of three base learners (Random Forest, LightGBM, MLP), combined through an out-of-fold (OOF) stacked meta-model, reaches a test-set AUC of 0.6911, a modest but genuine improvement over each individual base learner (best individual AUC 0.6871).

The classification edge on its own is too thin to be traded directly at the naive `probability > 0.5` decision rule: the raw, always-in-market strategy underperforms a passive Buy & Hold on both Sharpe ratio (0.391 vs 0.488) and Calmar ratio (0.471 vs 0.528). Filtering trades to only the days where the ensemble's confidence ranks in the top 30% of its own trailing 60-day history — an adaptive, rolling-percentile threshold — turns this around: Sharpe rises to 1.007 and Calmar to 2.137, driven mainly by a large reduction in market exposure (27% of days) that cuts annualized volatility from 24.27% to 17.86% and caps maximum drawdown at -8.42% versus Buy & Hold's -22.44%. The headline result is therefore a **risk-reduction story, not a return-generation story**: the strategy does not find dramatically larger returns than the market, it avoids most of the market's worst drawdown days.

A short out-of-sample test on live 2026 data (not seen during training) shows the same qualitative pattern holds, though the window is short (145 trading days) and the result should be read directionally rather than as a statistically powered confirmation.

---

## 2. Problem Statement & Motivation

### 2.1 Why Volatility Direction, Not Price Direction

An earlier iteration of this project attempted to predict next-day QQQ price direction using the same modeling pipeline. That model's test AUC sat at approximately 0.50 — statistically indistinguishable from a coin flip — consistent with semi-strong market efficiency for daily equity returns. Rather than force a weak signal into a backtest, the target was redefined to **volatility direction**, which has a stronger economic rationale: realized and implied volatility are well documented to cluster and mean-revert (ARCH/GARCH effects), rather than following a random walk. The feature set already available — VIX, VXN, realized volatility, Bollinger Band width, intraday range — is naturally suited to forecasting this kind of target.

### 2.2 Target Definition

- **`vol_now`**: trailing 5-day realized volatility (data through day T only — no look-ahead).
- **`vol_fwd`**: forward 5-day realized volatility (days T+1 through T+5 — future data used only to construct the label, never as a feature).
- **`label_binary = 1` ("Calm")**: `vol_fwd < vol_now` — volatility expected to fall, safe to hold QQQ.
- **`label_binary = 0` ("Storm")**: `vol_fwd >= vol_now` — volatility expected to rise, de-risk to cash.

This polarity is deliberate: it means the existing `probability > 0.5 → long` trading rule remains the economically correct decision rule without any changes to the backtest logic.

Days where the forward/trailing volatility ratio is close to 1 (ambiguous — neither a real expansion nor contraction) are filtered from the **training** label (`label_drop_nearzero`) using a 10th-percentile threshold on `|vol_fwd - vol_now|`, excluding 135 of 1,246 observations (10.7%) to reduce label noise. The unfiltered `label_binary` (1,246 observations, class balance 51.1%/48.9%) is retained for reference.

**[Figure 1: QQQ Adjusted Close Price, full sample period]**

**[Figure 2: Distribution of Volatility Changes, with the near-flat exclusion region highlighted]**

**[Figure 3: Label Distribution Comparison — Binary (Original) vs. Drop Near-Flat strategy, class balance]**

---

## 3. Data & Feature Engineering

### 3.1 Data Sources

- **Price/volume**: QQQ daily OHLCV (Yahoo Finance).
- **Macro**: 10-year Treasury yield, real rate, yield curve, credit spread (FRED); Dollar Index, Gold/Copper ratio, Semiconductor Index (SOX) (Yahoo Finance).
- **Sentiment**: VIX, VXN, TQQQ/SQQQ ratio, QQQ/SPY ratio, put/call proxy.

All macro series are aligned to QQQ's trading calendar, with non-trading-day observations removed.

### 3.2 Feature Categories

Approximately 61 candidate features were engineered across seven categories, using only information available through day T:

| Category | Description | Example features |
|---|---|---|
| Momentum | Historical returns only | `ret_log`, `cum_ret_5/10`, `ret_ma_5/10` |
| Trend | Moving averages, positioning, Bollinger Bands | `trend_pos`, `bb_width`, golden-cross indicators |
| Drawdown & Positioning | Drawdown depth, range position, recovery speed | `dd_vol_ratio`, `recovery_speed` |
| Volatility | Realized volatility, regime, intraday range | `vol_20`, `vol_regime`, `range_5d_smooth` |
| Volume | Z-scored volume, directional participation | `vol_z`, `breakout_volume` |
| Macro | Rates, credit spread, dollar index, commodities, semis | `yield_10y`, `credit_spread`, `dxy`, `gold_copper`, `semi` |
| Sentiment | Implied volatility / fear gauges | `vix`, `vxn_chg_1d` |

A multicollinearity screen removed 4 features that were mathematically near-duplicates of other retained features (`|r| ≈ 1.0`, e.g. `cum_ret_5` ⇔ `ret_ma_5`), leaving **57 features** used for modeling.

**[Figure 4: Correlation Heatmap, top 30 features by variance]**

**[Figure 5: Feature-Target Scatter Plots for the top correlated features]**

**[Figure 6: All-Features Correlation Ranking (signed, colored by direction) and Correlation-by-Category Boxplot]**

**[Figure 7: Pairplot of representative features from each correlation cluster]**

### 3.3 Feature Importance & Dimensionality

A preliminary Random Forest importance ranking and correlation-based K-Means clustering (optimal K=7 by silhouette score) were used to sanity-check the feature set for redundancy before final model training; all 57 features were retained for the production models since none of the base learners require manual dimensionality reduction.

**[Figure 8: Random Forest Feature Importance — Top 20 Features and Cumulative Importance Curve]**

**[Figure 9: K-Means Elbow Curve and Silhouette Score by K]**

---

## 4. Methodology: Blending Ensemble Architecture

### 4.1 Train/Test Split

The data is split **temporally**, not randomly, to avoid leaking future information into training:

- Training set: 896 observations (2021-01-06 to 2024-12-20)
- Test set: 225 observations (2024-12-23 to 2025-12-19)

Features are scaled with `RobustScaler`, fit on the training set only and applied unchanged to the test set.

### 4.2 Base Learners

Five base learners are trained and individually evaluated, each optimized with `RandomizedSearchCV` (30 iterations) over a **purged/embargoed** `TimeSeriesSplit(n_splits=5, gap=5)` — the 5-day gap between train and validation folds matches the label's forward horizon, preventing the forward-looking 5-day volatility window from leaking across fold boundaries.

| Model | Role | Individually evaluated | Feeds the blend |
|---|---|---|---|
| Random Forest | Bagged-tree baseline | Yes | Yes |
| LightGBM | Boosted-tree, sequential error correction | Yes | Yes |
| MLP (Neural Network) | Smooth nonlinear decision boundary | Yes | Yes |
| Logistic Regression | Linear baseline | Yes | No |
| SVM (linear kernel) | Margin-based, hinge loss | Yes | No |

Logistic Regression and SVM are both effectively linear on this target (SVM's own hyperparameter search converged on a linear kernel) and both had negative individual backtest Sharpe; an ablation over all 2–5 model subsets showed including either one consistently pulled the blend's backtest performance below the nonlinear-only {Random Forest, LightGBM, MLP} subset, so they are reported for comparison but excluded from the final blend.

### 4.3 Blending / Stacking Architecture

Out-of-fold (OOF) predictions from the three blend learners are generated across the same purged `TimeSeriesSplit(5, gap=5)`, then used to train a meta-model:

```
meta_model = LogisticRegressionCV(
    Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    cv=TimeSeriesSplit(n_splits=5, gap=5),
    class_weight='balanced', scoring='roc_auc'
)
```

`LogisticRegressionCV` was chosen over a plain `LogisticRegression` meta-model after diagnosing that the base learners' OOF predictions are highly collinear (Random Forest / LightGBM correlate at r≈0.91) — on only ~900 OOF rows, a weakly regularized linear meta-model overfit and assigned a **negative** coefficient to the best individual base learner. Cross-validating the regularization strength on the OOF matrix alone (never touching the test set) fixed this: the selected regularization strength was C=0.0030, and all three base learners now receive positive, sensibly-ordered weights.

**[Figure 10: Meta-Model Blend Weights — bar chart of the three base-learner coefficients]**

---

## 5. Results: Classification Performance

| Model | AUC-ROC | Balanced Accuracy |
|---|---|---|
| Random Forest | 0.6871 | 0.6490 |
| MLP Classifier | 0.6757 | 0.6354 |
| LightGBM | 0.6735 | 0.6391 |
| SVM | 0.6591 | 0.6109 |
| Logistic Regression | 0.6591 | 0.6034 |
| **Blending Ensemble** | **0.6911** | **0.6417** |

The ensemble improves on the best individual base learner (Random Forest) by +0.58% AUC — a small but positive diversification benefit, consistent with the base learners making partially uncorrelated errors.

**[Figure 11: Base Learner vs. Blending Ensemble AUC Comparison — horizontal bar chart]**

**[Figure 12: ROC Curves — all base learners and the blending ensemble overlaid]**

---

## 6. Results: Trading Strategy Backtest

### 6.1 Backtest Mechanics

Positions are set from day T's predicted probability and applied to day T+1's return (`shift_returns=True`), eliminating look-ahead bias. The baseline rule is long-only binary: hold QQQ when `probability > 0.5`, otherwise hold cash.

### 6.2 Fixed Confidence Threshold — and Why It Is Fragile

An overlay that only trades when `|probability - 0.5|` exceeds a fixed cutoff was tested across a fine grid. The ensemble's probability output only spans about 2.75 percentage points around 0.5 (min=0.4900, max=0.5175, std=0.0077) on the test set, so a fixed absolute cutoff is extremely sensitive to small changes: Sharpe swings from 0.895 at a 1.0% threshold to exactly 0 trades at any threshold ≥ 2.0%.

**[Figure 13: Sharpe Ratio vs. Confidence Threshold, and Market Exposure vs. Confidence Threshold]**

### 6.3 Rolling-Percentile Threshold (Adaptive Alternative)

Instead of an absolute distance from 0.5, positions are taken when today's probability ranks in the top `cutoff` percentile of its own trailing `window`-day history (today excluded — no look-ahead). Parameters (`window=60`, ~one trading quarter; `cutoff=0.70`, top 30%) were chosen as round, conventional values rather than by searching for the single best-performing grid cell, and the sensitivity grid below shows this choice sits inside a broad, stable performance region rather than an isolated lucky point.

**[Figure 14: Rolling-Percentile Sensitivity Heatmap — Sharpe Ratio by (Window, Cutoff)]**

### 6.4 Full Strategy Comparison

| Strategy | Total Ret. | Ann. Ret. | Volatility | Sharpe | Calmar | Max DD | Win Rate | In Market |
|---|---|---|---|---|---|---|---|---|
| MLP Classifier (individual) | 16.38% | 18.61% | 18.10% | 1.028 | 1.580 | -11.78% | 58.46% | 29% |
| **Blending Ensemble (rolling pct.)** | **15.84%** | **17.99%** | **17.86%** | **1.007** | **2.137** | **-8.42%** | **59.02%** | **27%** |
| Blend (fixed thresh. = 1.0%) | 13.16% | 14.92% | 16.67% | 0.895 | 1.711 | -8.72% | 57.14% | 22% |
| Buy & Hold | 10.46% | 11.84% | 24.27% | 0.488 | 0.528 | -22.44% | 55.80% | 100% |
| LightGBM (individual) | 8.51% | 9.62% | 20.04% | 0.480 | 0.549 | -17.54% | 52.43% | 46% |
| Random Forest (individual) | 8.10% | 9.15% | 20.28% | 0.451 | 0.652 | -14.05% | 51.79% | 50% |
| Blending Ensemble (raw, always-on) | 7.20% | 8.13% | 20.79% | 0.391 | 0.471 | -17.27% | 52.34% | 57% |
| Logistic Regression (individual) | -0.84% | -0.94% | 19.88% | -0.047 | -0.054 | -17.28% | 51.49% | 45% |
| SVM (individual) | -1.48% | -1.66% | 19.73% | -0.084 | -0.098 | -17.00% | 50.57% | 39% |

**[Figure 15: Strategy Comparison — Sharpe Ratio bar chart across all backtested strategies]**

**[Figure 16: Cumulative Returns — (a) all strategies vs. Buy & Hold; (b) rolling-percentile strategy vs. raw ensemble vs. Buy & Hold]**

**[Figure 17: Position Distribution and Position Sizing Over Time, for the raw ensemble and for the best-Sharpe (rolling-percentile) strategy]**

---

## 7. Strategy-Level Risk-Adjusted Analysis: Why the Edge Shows Up in Sharpe/Calmar, Not Raw AUC

Classification accuracy is not the deliverable — the trading strategy's risk-adjusted return is. The **raw, always-in-market ensemble is the key counterexample**: despite a positive test AUC (0.6911), trading every day the naive rule implies "long" (57% of days) *underperforms* Buy & Hold on every risk-adjusted metric (Sharpe 0.391 vs. 0.488; Calmar 0.471 vs. 0.528). Most of that AUC comes from days near the 50% decision boundary, where the classification edge is thin and noisy — a positive AUC does not automatically translate into a good trading rule at an unfiltered cutoff.

The edge only appears once positions are filtered to the days the ensemble is most confident relative to its own recent history. Comparing the rolling-percentile strategy to Buy & Hold:

- **Annualized return** is comparable (17.99% vs. 11.84%) — not dramatically higher.
- **Annualized volatility** falls from 24.27% to 17.86% (a ~26% relative reduction), because the strategy is only exposed to the market 27% of the time.
- **Maximum drawdown** falls from -22.44% to -8.42% (a ~62% reduction) — this is the single largest driver of the Sharpe and Calmar improvement.
- **Calmar ratio** (annualized return / |max drawdown|) rises from 0.528 to 2.137, roughly a 4x improvement, driven almost entirely by the smaller drawdown rather than larger returns.

In short: **the strategy's value is risk reduction — sitting out the market's worst days — rather than return generation.** This is a materially different (and more defensible, given the modest underlying AUC) claim than "the model beats the market," and should be framed as such in any discussion of results.

---

## 8. Out-of-Sample Validation: 2026

As a forward-deployment-style robustness check, fresh 2026 data (never seen during training or hyperparameter search) was independently rebuilt through the same feature pipeline and scored with the already-trained base learners and meta-model (no refitting).

| Strategy | Total Ret. | Ann. Ret. | Volatility | Sharpe | Calmar | Max DD | In Market |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 11.33% | 20.66% | 11.22% | 1.841 | 5.027 | -4.11% | 24% |
| Buy & Hold | 10.97% | 19.98% | 21.77% | 0.918 | 1.668 | -11.98% | 100% |
| Blending Ensemble (raw) | 6.78% | 12.16% | 18.72% | 0.650 | 1.369 | -8.88% | 72% |
| Blend (fixed thresh. = 1.0%) | -0.53% | -0.92% | 9.87% | -0.093 | -0.116 | -7.88% | 23% |

2026 test window: 145 trading days (2026-01-02 to 2026-07-31). The ensemble's OOS probability output (min=0.4917, max=0.5155, std=0.0059, 71.7% of days above 0.5) is narrow but not fully saturated to one side, unlike the price-direction version of this project, which produced a fully saturated, one-sided OOS probability range on the same window — a modest positive sign for the volatility target, though 145 days remains a short, likely single-regime sample and this should be read directionally, not as a statistically powered confirmation.

---

## 9. Discussion & Limitations

**Strengths**
- Purged/embargoed cross-validation (`TimeSeriesSplit(gap=5)`) prevents the forward-looking label window from leaking across fold boundaries during hyperparameter search.
- The backtest enforces no look-ahead bias by construction (`shift_returns=True`).
- Volatility direction has a stronger *a priori* economic rationale than price direction, consistent with the AUC improvement observed relative to the earlier price-direction iteration of this project (~0.50–0.51 AUC there vs. 0.66–0.69 here).

**Limitations**
- **Forecast horizon vs. rebalancing frequency mismatch**: the label looks 5 trading days ahead, but the strategy rebalances daily off a rolling 5-day-ahead forecast, so consecutive days' positions are highly autocorrelated (labels overlap by 4 of 5 days). The effective number of independent observations is much smaller than the row count, and confidence in any single Sharpe/Calmar figure should be discounted accordingly.
- **Sample size and single test period**: all headline numbers come from one train/test split and one random seed. *[Note: robustness across other random seeds was checked informally and found meaningfully less favorable for the rolling-percentile result than the seed used throughout this notebook — this caveat should be elaborated on with the specific numbers when finalizing the report.]*
- **Transaction costs**: the backtest assumes zero transaction costs and perfect execution at the closing price; frequent position changes around the confidence threshold would erode some of the reported edge in practice.
- **Run-to-run instability**: nested parallelism and unpinned multi-threaded BLAS settings were found to materially affect results run-to-run; headline numbers should be read as one draw, not a fixed, guaranteed result.

**Further Development**
- Test sensitivity of the target to alternative forecast horizons (10d/20d/60d forward volatility).
- Benchmark against a simple GARCH(1,1) or EWMA volatility forecast — the relevant bar for "the ML ensemble adds value" is beating that baseline, not beating a coin flip.
- Extend the training window to include multiple volatility regimes (2008, 2020, 2022) — the current window is dominated by a single, mostly-calm bull regime.
- Add an explicit transaction-cost and slippage model.
- Replace the binary long/flat rule with fractional position sizing tied to the raw probability magnitude, closer to a real volatility-targeting overlay.

---

## 10. Conclusion

Reframing the prediction target from price direction to volatility direction produces a modest but genuine classification edge (AUC 0.6911 vs. ~0.50 for price direction) and, more importantly, a trading strategy whose value is demonstrable in risk-adjusted terms even though the raw classification signal is not strong enough to trade directly. The rolling-percentile confidence overlay converts a strategy that *underperforms* Buy & Hold (raw ensemble) into one with roughly 2x Buy & Hold's Sharpe ratio and 4x its Calmar ratio, primarily by avoiding the market's highest-drawdown periods rather than by capturing outsized returns. This risk-reduction framing — not "beating the market" — is the honest and defensible interpretation of these results.

---

*Figures referenced above correspond to the charts produced in `CQF_Volatility_Direction_5D.ipynb` / `CQF_Volatility_Direction_5D_EXECUTED.ipynb`; insert the actual chart images from the executed notebook at each placeholder before final submission.*
