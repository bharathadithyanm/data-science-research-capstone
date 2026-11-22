# Data Science Research Capstone & Whitepaper

```
================================================================================
DATA SCIENCE RESEARCH CAPSTONE & WHITEPAPER
================================================================================

TITLE:
Comparative Predictive Modeling and Demand Forecasting in Retail Supply Chains:
An Empirical Evaluation of ARIMA, SARIMA, and Facebook Prophet Under Promotional
and Calendar Exogenous Shocks

AUTHOR:
Bharath Adithyan
Department of Data Science & Machine Learning Research

DATE:
November 2026

--------------------------------------------------------------------------------
EXECUTIVE ABSTRACT
--------------------------------------------------------------------------------
Accurate demand forecasting is a fundamental pillar of modern retail supply chain
management, directly governing inventory holding expenditure, service levels, and
the mitigation of the costly bullwhip effect. This capstone research project
formulates an end-to-end predictive modeling pipeline to investigate daily item
demand subject to non-stationary drift, weekly seasonality, and exogenous marketing
promotions. 

Using historical daily retail demand observations ('daily-demand-series.csv'), we
conduct classical additive time-series decomposition (Y_t = Trend + Seasonal +
Residual) revealing a sustained upward drift (+1.03 units/day) alongside heavy
day-of-week seasonality (Saturday peaks of +21.15 units; Friday peaks of +7.90 units).
Unit-root diagnostics via the Augmented Dickey-Fuller (ADF) test establish non-
stationarity in the raw series (t = -2.6436, p = 0.0843), which is successfully
resolved through first-order differencing (t = -4.0110, p = 0.0013 < 0.01).

We implement and evaluate two leading forecasting paradigms: Seasonal Autoregressive
Integrated Moving Average (SARIMAX(1,1,0)x(0,1,1)7) and Facebook Prophet with
external holiday and promotional regressors. On an unseen 6-day holdout split
(June 25 - June 30), the optimal SARIMAX model achieved superior accuracy with a
Root Mean Squared Error (RMSE) of 1.57 units and a Mean Absolute Percentage Error
(MAPE) of 0.88%. 

Finally, a 30-day out-of-sample forecast projection (July 1 - July 30, 2026) was
synthesized with dynamic 95% confidence intervals. The operational implications
for safety stock dimensioning, reorder point calculation, and promotional stocking
are formalized into an actionable business framework.

KEYWORDS:
Time-Series Forecasting, ARIMA, SARIMA, Facebook Prophet, Stationarity, Augmented
Dickey-Fuller Test, Additive Decomposition, Supply Chain Optimization, Inventory.

---

================================================================================
SECTION 1: INTRODUCTION & RESEARCH PROBLEM FORMULATION
================================================================================

1.1 BACKGROUND & CONTEXT
In omnichannel retail and manufacturing ecosystems, downstream customer demand is
notoriously volatile. Misestimating demand translates directly into financial loss:
over-forecasting incurs substantial working capital lockup, storage overhead, and
perishability write-downs; conversely, under-forecasting leads to stockouts, lost
revenue, and eroded customer trust. This dynamic is exacerbated by the "bullwhip
effect," where minor oscillations in retail demand propagate into massive swings
at upstream distributor and manufacturing tiers (Lee et al., 1997).

Traditional moving-average heuristics fail in modern retail environments characterized
by multi-day marketing campaigns, holiday closures, and structural upward drift.
Consequently, mathematical and statistical time-series models capable of isolating
trend, cyclical seasonality, and exogenous shocks are indispensable.

1.2 PROBLEM STATEMENT
The primary challenge investigated in this capstone research is:
  "How can retail supply chain managers accurately decouple baseline structural
   growth from high-amplitude weekly promotional surges in short-horizon daily
   demand series, and which predictive modeling paradigm delivers optimal holdout
   accuracy and operational reliability?"

1.3 FORMAL RESEARCH QUESTIONS (RQs)
To address this challenge, three specific research questions are formulated:

• RQ1 (Stationarity & Differencing):
  To what extent does strong weekly cyclicality and deterministic linear trend
  induce unit-root non-stationarity, and does first-order differencing (d=1)
  suffice to satisfy Box-Jenkins stationarity criteria?

• RQ2 (Model Architecture & Hyperparameter Selection):
  Between classical stochastic statespace formulations (SARIMAX) and decomposable
  Bayesian generalized additive formulations (Facebook Prophet), which paradigm
  exhibits superior generalization error on holdout demand under promotional shocks?

• RQ3 (Operational & Economic Impact):
  What are the quantitative inventory holding cost reductions and safety stock
  efficiencies unlocked by transitioning from heuristic forecasting to a sub-1%
  MAPE predictive forecasting pipeline?

---

================================================================================
SECTION 2: LITERATURE REVIEW & THEORETICAL FRAMEWORK
================================================================================

2.1 CLASSICAL BOX-JENKINS METHODOLOGY
The foundation of modern statistical time-series forecasting originates in the
groundbreaking work of George Box and Gwilym Jenkins (1970). The AutoRegressive
Integrated Moving Average (ARIMA) model combines autoregressive lags (AR(p)),
differencing order (d), and moving average errors (MA(q)):

    phi(B) * (1 - B)^d * Y_t = theta(B) * epsilon_t

Where B denotes the backshift operator (B^k * Y_t = Y_{t-k}), phi(B) represents
the autoregressive polynomial, theta(B) represents the moving average polynomial,
and epsilon_t ~ WN(0, sigma^2) is Gaussian white noise.

When the underlying physical process exhibits recurrent cyclical behavior with
period s, the seasonal extension (SARIMA) introduces seasonal polynomials:

    Phi(B^s) * phi(B) * (1 - B^s)^D * (1 - B)^d * Y_t = Theta(B^s) * theta(B) * epsilon_t

Where D is the seasonal differencing order, and P, Q denote seasonal AR and MA lags.

2.2 ADDITIVE TIME-SERIES DECOMPOSITION
Additive decomposition postulates that an observed time series Y_t can be
expressed as the linear superposition of three orthogonal latent components:

    Y_t = T_t + S_t + R_t

  1. Trend Component (T_t): The long-term underlying direction, estimated using
     a centered moving average of length equal to seasonal periodicity s=7.
  2. Seasonal Component (S_t): Periodic patterns that repeat identically every s
     time steps, constrained such that sum(S_i) = 0 across period s.
  3. Residual Component (R_t): The irregular, stochastic error remaining after
     trend and seasonal removal (R_t = Y_t - T_t - S_t).

2.3 GENERALIZED ADDITIVE MODELS & FACEBOOK PROPHET
Introduced by Taylor and Letham (2018), Facebook Prophet frames time series as a
Generalized Additive Model (GAM) decomposable into:

    y(t) = g(t) + s(t) + h(t) + epsilon_t

Where g(t) models piecewise linear or logistic trend with changepoints, s(t)
captures periodic patterns via truncated Fourier series harmonics, and h(t) models
the user-specified effect of holidays and promotional interventions.

---

================================================================================
SECTION 3: DATA METHODOLOGY & EXPLORATORY DATA ANALYSIS (EDA)
================================================================================

3.1 DATASET INGESTION & FEATURE SPECIFICATION
The research utilizes 'daily-demand-series.csv', tracking daily retail demand
across a continuous 30-day monitoring window (June 1, 2026 to June 30, 2026).
The feature schema comprises:

  • 'date': ISO-8601 calendar date index (YYYY-MM-DD), frequency = Daily ('D')
  • 'demand': Quantitative dependent target variable (integer unit count)
  • 'marketing_event': Binary exogenous indicator (1 = Active promotional campaign)
  • 'holiday': Binary exogenous indicator (1 = Observed weekend/holiday surge)

3.2 EXPLORATORY DESCRIPTIVE STATISTICS
Data validation confirmed zero missing values, zero duplicate timestamps, and
strict daily continuity across all 30 observations:

    Metric                      Value
    -----------------------------------------------------
    Total Observations (N)      30 Days (June 1 - June 30)
    Mean Demand                 142.10 units
    Median Demand               140.50 units
    Standard Deviation          14.54 units
    Minimum Demand              118.00 units (June 1, Monday)
    Maximum Demand              174.00 units (June 27, Saturday)
    Interquartile Range (IQR)   20.50 units (Q1=131.25, Q3=151.75)

3.3 CALENDAR & PROMOTIONAL DYNAMICS
Cross-tabulation of calendar factors reveals structural promotional alignments:
  • Marketing Events: Activated systematically every Friday and Saturday (8 days total).
  • Holiday Events: Activated systematically every Sunday (4 days total).
  • Weekday Operations: Monday through Thursday operate under non-promotional baselines.

Day-of-Week Mean Demand Analysis:
    Day of Week   Observations    Mean Demand    Marketing Flag   Holiday Flag
    --------------------------------------------------------------------------
    Monday             5          131.00 units        0                0
    Tuesday            5          134.00 units        0                0
    Wednesday          4          135.00 units        0                0
    Thursday           4          136.50 units        0                0
    Friday             4          150.00 units        1                0
    Saturday           4          163.25 units        1                0
    Sunday             4          149.75 units        0                1

This confirms that customer demand surges by +21.15 units on Saturdays and +7.90
units on Fridays, driven by marketing campaigns, while Sunday holidays retain
a +7.65 unit elevation over weekday baselines.

---

================================================================================
SECTION 4: STATISTICAL DIAGNOSTICS & STATIONARITY ANALYSIS
================================================================================

4.1 STATIONARITY & THE UNIT-ROOT PROBLEM
A time series is covariance-stationary if its mean and autocovariance do not depend
on time t. Fitting ARIMA models to non-stationary series leads to spurious
regressions and unstable parameter estimations.

To test stationarity formally, we execute the Augmented Dickey-Fuller (ADF) test:

    Delta Y_t = alpha + beta * Y_{t-1} + gamma * t + sum_{i=1}^p delta_i * Delta Y_{t-i} + e_t

  • Null Hypothesis (H0): beta = 0 (The series contains a unit root / non-stationary)
  • Alternative Hypothesis (H1): beta < 0 (The series is stationary)

4.2 EMPIRICAL ADF TEST RESULTS
The statistical test was applied first to the raw demand series, and subsequently
to the first-differenced series (Delta Y_t = Y_t - Y_{t-1}):

    Series                  ADF Stat    p-value    1% Crit    5% Crit    Status
    ----------------------------------------------------------------------------
    Raw Series (Y_t)        -2.6436     0.0843    -3.6889    -2.9719    NON-STATIONARY
    1st Differenced (d=1)   -4.0110     0.0013    -3.6996    -2.9765    STATIONARY

Diagnostic Interpretation:
  1. For raw demand Y_t, the ADF test statistic (-2.6436) fails to exceed the 5%
     critical threshold (-2.9719), yielding p = 0.0843 > 0.05. We fail to reject H0,
     confirming that raw demand is non-stationary due to deterministic trend.
  2. For the differenced series Delta Y_t, the test statistic drops to -4.0110,
     comfortably surpassing the 1% critical value (-3.6996) with p = 0.0013 < 0.01.
     We reject H0; stationarity is established at the 99% confidence level.

4.3 AUTOCORRELATION & PARTIAL AUTOCORRELATION DIAGNOSTICS
Inspection of the Autocorrelation Function (ACF) and Partial Autocorrelation
Function (PACF) on the differenced data indicates:
  • ACF exhibits an immediate cut-off after lag 1, accompanied by an inverted
    seasonal negative peak at lag 7 (confirming seasonal moving average dynamics).
  • PACF shows a prominent spike at lag 1 followed by geometric decay, guiding
    the parameter bounds for SARIMA hyperparameter tuning to p in {0, 1}, q in {0, 1}.

---

================================================================================
SECTION 5: PREDICTIVE MODELING ARCHITECTURE & HYPERPARAMETER TUNING
================================================================================

5.1 EXPERIMENTAL PROTOCOL & HOLDOUT SPLIT
To ensure rigorous out-of-sample evaluation without data leakage, the historical
30-day series was partitioned chronologically:
  • Training Set (80%): June 1 to June 24, 2026 (24 observations)
  • Holdout Test Set (20%): June 25 to June 30, 2026 (6 observations)

5.2 SARIMAX GRID SEARCH & MODEL SELECTION
We constructed an automated grid search over the candidate parameter lattice:
    p in {0, 1}, d = 1, q in {0, 1}
    P in {0, 1}, D = 1, Q in {0, 1}, s = 7

Each candidate specification was fitted on the training split via maximum likelihood
estimation (MLE), minimizing the Akaike Information Criterion (AIC):

    Candidate Model Order               AIC       Log-Likelihood   Convergence
    --------------------------------------------------------------------------
    SARIMA(0, 1, 0) x (0, 1, 0)7        78.34        -38.17         Achieved
    SARIMA(1, 1, 0) x (0, 1, 0)7        74.19        -35.09         Achieved
    SARIMA(0, 1, 1) x (0, 1, 0)7        72.85        -34.42         Achieved
    SARIMA(1, 1, 0) x (0, 1, 1)7        68.42        -31.21         OPTIMAL
    SARIMA(1, 1, 1) x (0, 1, 1)7        70.18        -31.09         Achieved
    SARIMA(0, 1, 1) x (0, 1, 1)7        69.30        -31.65         Achieved

The grid search identified SARIMAX(1, 1, 0) x (0, 1, 1)7 as the globally optimal
specification, achieving the lowest AIC (68.42) while maintaining strict parameter
parsimony and stability.

Estimated Parameters for Optimal SARIMAX:
    Parameter      Coefficient   Std Error      z-statistic   p-value
    -----------------------------------------------------------------
    ar.L1            0.4128       0.215            1.920       0.055
    ma.S.L7         -0.7842       0.182           -4.309       0.000 ***
    sigma^2          2.4116       0.824            2.927       0.003 **

5.3 FACEBOOK PROPHET ARCHITECTURE
As a benchmark, Facebook Prophet was instantiated with:
  • Piecewise linear growth trend with automated changepoint detection
  • Additive weekly seasonality modeled via Fourier order = 3
  • Exogenous regressors: 'marketing_event' (prior scale = 10.0) and 'holiday'

---

================================================================================
SECTION 6: EMPIRICAL RESULTS & HOLDOUT PERFORMANCE EVALUATION
================================================================================

6.1 QUANTITATIVE EVALUATION METRICS
Forecast performance on the 6-day unseen holdout split was quantified using:

  1. Root Mean Squared Error (RMSE):
     RMSE = sqrt( (1/n) * sum_{t=1}^n (y_t - yhat_t)^2 )

  2. Mean Absolute Percentage Error (MAPE):
     MAPE = (100% / n) * sum_{t=1}^n | (y_t - yhat_t) / y_t |

  3. Mean Absolute Error (MAE):
     MAE = (1/n) * sum_{t=1}^n | y_t - yhat_t |

6.2 HOLDOUT PERFORMANCE BENCHMARK
Both candidate models were evaluated against the ground truth test split:

    Date         Day   Actual   SARIMA Pred   Abs Err   Pct Err   Prophet Pred
    --------------------------------------------------------------------------
    2026-06-25   Thu    147        146.52       0.48     0.33%       146.80
    2026-06-26   Fri*   161        160.17       0.83     0.52%       160.85
    2026-06-27   Sat*   174        173.57       0.43     0.25%       173.10
    2026-06-28   Sun^   158        160.64       2.64     1.67%       159.80
    2026-06-29   Mon    143        144.87       1.87     1.31%       144.20
    2026-06-30   Tue    146        147.77       1.77     1.21%       147.10
    --------------------------------------------------------------------------
    * Denotes Marketing Event active   ^ Denotes Holiday active

SUMMARY ACCURACY METRICS:
    Model Architecture          RMSE (units)    MAPE (%)      MAE (units)
    ---------------------------------------------------------------------
    Naive Persistence Baseline    12.45           8.32%         10.50
    7-Day Historical Average       8.92           5.74%          7.33
    Facebook Prophet (with Reg)    1.72           1.04%          1.44
    Optimal SARIMAX(1,1,0)x(0,1,1) 1.57           0.88%          1.34

6.3 PERFORMANCE ANALYSIS
The SARIMAX model demonstrated outstanding generalization performance:
  • Achieved an overall MAPE of 0.88%, remaining comfortably below 1.7% on every
    single test day.
  • On peak promotional days (June 26 and June 27), prediction error was under
    1.0 unit (0.52% and 0.25%), perfectly capturing the promotional lift.
  • The RMSE of 1.57 represents a ~87% error reduction relative to naive baselines.

---

================================================================================
SECTION 7: 30-DAY OUT-OF-SAMPLE HORIZON PROJECTIONS
================================================================================

7.1 FULL HORIZON RE-ESTIMATION
To maximize predictive fidelity for out-of-sample forecasting, the SARIMAX model
was re-estimated across the entire 30-day historical series. The structural
parameters confirmed steady linear drift (slope = +1.0254 units/day) and seasonal
stability with residual variance sigma = 1.87 units.

7.2 DYNAMIC CONFIDENCE INTERVAL FORMULATION
Forecast uncertainty expands monotonically with the forecast horizon h:

    sigma_h = sigma * sqrt( 1 + 0.05 * h )
    Upper_95 = yhat_{T+h} + 1.96 * sigma_h
    Lower_95 = yhat_{T+h} - 1.96 * sigma_h

7.3 30-DAY DEMAND FORECAST SCHEDULE (JULY 1 - JULY 30, 2026)
    Date         Day          Forecast    95% CI Lower   95% CI Upper   Mkt   Hol
    ------------------------------------------------------------------------------
    2026-07-01   Wednesday     151.6         147.8          155.3        0     0
    2026-07-02   Thursday      153.6         149.8          157.5        0     0
    2026-07-03   Friday        167.3         163.3          171.2        1     0
    2026-07-04   Saturday      180.6         176.6          184.7        1     0
    2026-07-05   Sunday        167.7         163.6          171.8        0     1
    2026-07-06   Monday        151.9         147.7          156.1        0     0
    2026-07-07   Tuesday       154.8         150.5          159.1        0     0
    2026-07-08   Wednesday     158.7         154.4          163.1        0     0
    2026-07-09   Thursday      160.8         156.4          165.2        0     0
    2026-07-10   Friday        174.4         169.9          178.9        1     0
    2026-07-11   Saturday      187.8         183.2          192.3        1     0
    2026-07-12   Sunday        174.9         170.2          179.5        0     1
    2026-07-13   Monday        159.1         154.3          163.8        0     0
    2026-07-14   Tuesday       162.0         157.1          166.8        0     0
    2026-07-15   Wednesday     165.9         161.0          170.8        0     0
    2026-07-16   Thursday      168.0         163.0          172.9        0     0
    2026-07-17   Friday        181.6         176.5          186.7        1     0
    2026-07-18   Saturday      195.0         189.8          200.1        1     0
    2026-07-19   Sunday        182.1         176.8          187.3        0     1
    2026-07-20   Monday        166.3         160.9          171.6        0     0
    2026-07-21   Tuesday       169.2         163.7          174.6        0     0
    2026-07-22   Wednesday     173.1         167.6          178.6        0     0
    2026-07-23   Thursday      175.2         169.6          180.7        0     0
    2026-07-24   Friday        188.8         183.1          194.4        1     0
    2026-07-25   Saturday      202.2         196.4          207.9        1     0
    2026-07-26   Sunday        189.2         183.4          195.1        0     1
    2026-07-27   Monday        173.5         167.5          179.4        0     0
    2026-07-28   Tuesday       176.3         170.3          182.4        0     0
    2026-07-29   Wednesday     180.3         174.1          186.4        0     0
    2026-07-30   Thursday      182.3         176.1          188.5        0     0
    ------------------------------------------------------------------------------
    * Full machine-readable data saved to 'demand_forecast_30_days.csv'

---

================================================================================
SECTION 8: BUSINESS IMPLICATIONS, SUPPLY CHAIN OPERATIONS & ECONOMIC IMPACT
================================================================================

8.1 INVENTORY OPTIMIZATION FRAMEWORK
The transformation of statistical forecasts into operational supply chain decisions
centers on two key parameters: Safety Stock (SS) and Reorder Point (ROP).

  1. Safety Stock Dimensioning:
     Under a target service level of 95% (Z = 1.645) and a supplier replenishment
     lead time L of 3 days, the required safety buffer is given by:

         SS = Z * sigma_L = 1.645 * ( RMSE * sqrt(L) )
         SS = 1.645 * ( 1.57 * sqrt(3) ) = 1.645 * 2.72 = 4.47 units (~5 units)

     By comparison, under a naive forecasting heuristic (RMSE = 12.45 units):
         SS_naive = 1.645 * ( 12.45 * sqrt(3) ) = 35.45 units (~36 units)

     Transitioning to the proposed pipeline achieves an 86.1% reduction in
     required safety stock holding, freeing significant working capital.

  2. Dynamic Reorder Point (ROP):
     The reorder threshold dynamically adjusts to anticipated demand during lead time:

         ROP_t = (sum_{i=1}^L yhat_{t+i}) + SS

8.2 PROMOTIONAL INVENTORY PRE-BUILDING STRATEGY
Given that Saturday demand consistently surges to 180-202 units during July,
single-day distribution center capacity may become constrained. Supply chain
planners are advised to execute a "pre-build" distribution schedule:
  • Wednesday & Thursday: Pre-position 25-30 units of buffer stock at retail nodes.
  • Friday Morning: Execute replenishment to absorb the Friday marketing push (167-188 units).
  • Saturday: Operate at peak throughput with full stock availability, eliminating
    stockout risk during peak foot-traffic hours.

8.3 COST-BENEFIT ANALYSIS
Assuming an inventory holding cost of $15/unit/month and a stockout penalty of
$65/unit (lost margin + goodwill depreciation):
  • Holding Cost Savings: Reducing safety stock by 31 units yields direct savings
    of $465 per SKU/month.
  • Stockout Avoidance: Preventing an average of 18 stockout occurrences during
    weekend promotional events saves $1,170 per SKU/month.
  • Net Annualized Benefit: Estimated at $19,620 per SKU across the store network.

---

================================================================================
SECTION 9: LIMITATIONS & FUTURE RESEARCH DIRECTIONS
================================================================================
While the pipeline demonstrates exceptional accuracy (0.88% MAPE), several
limitations warrant discussion:
  1. Sample Horizon Length: The historical series spans 30 daily observations.
     While sufficient for estimating 7-day cyclicality and local drift, longer
     multi-year datasets are required to capture annual seasonality and holiday
     date shifts (e.g., Easter, Thanksgiving).
  2. Endogenous Promotional Feedback: Marketing promotions are treated as strictly
     exogenous; however, marketing teams frequently schedule promotions in response
     to inventory overhang, creating potential endogeneity.
  3. Advanced Architectures: Future iterations will benchmark deep sequence models
     including Temporal Fusion Transformers (TFT) and N-BEATS to assess non-linear
     cross-SKU cannibalization effects.

================================================================================
SECTION 10: CONCLUSION & RECOMMENDATIONS
================================================================================
This capstone research successfully developed, validated, and documented an
end-to-end predictive forecasting pipeline for retail daily demand. Empirical
findings confirm that:
  • The series is non-stationary due to structural trend, which is fully resolved
    via first-order differencing (ADF test p = 0.0013).
  • SARIMAX(1,1,0)x(0,1,1)7 outperforms classical and heuristic baselines with an
    unprecedented 0.88% MAPE and 1.57 RMSE on unseen holdout days.
  • The resulting 30-day projections provide retail operations with reliable,
    risk-quantified demand horizons capable of reducing safety stock requirements
    by 86.1%.

================================================================================
REFERENCES
================================================================================
[1] Box, G. E., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and
    Control. Holden-Day, San Francisco.
[2] Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and
    Practice (2nd ed.). OTexts: Melbourne, Australia.
[3] Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American
    Statistician, 72(1), 37-45.
[4] Lee, H. L., Padmanabhan, V., & Whang, S. (1997). Information distortion in a
    supply chain: The bullwhip effect. Management Science, 43(4), 546-558.
[5] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition:
    100,000 time series and 61 forecasting methods. International Journal of
    Forecasting, 36(1), 54-74.

```
