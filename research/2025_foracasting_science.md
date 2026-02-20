# Complete Guide to Forecasting Science: Forecastability Assessment and Conformal Prediction

**A Technical Framework for Rigorous Time Series Forecasting**  
*Synthesized from Valeriy Manokhin's Research and "Practical Guide to Applied Conformal Prediction in Python"*

---

## Table of Contents

1. [Fundamental Principles](#1-fundamental-principles)
2. [Forecastability Assessment Framework](#2-forecastability-assessment-framework)
3. [Classical Statistical Models: ARIMA and ETS](#3-classical-statistical-models-arima-and-ets)
4. [Gradient Boosting: CatBoost for Time Series](#4-gradient-boosting-catboost-for-time-series)
5. [Conformal Prediction: Theoretical Foundations](#5-conformal-prediction-theoretical-foundations)
6. [Nonconformity Measures](#6-nonconformity-measures)
7. [Time Series Forecasting with Conformal Prediction](#7-time-series-forecasting-with-conformal-prediction)
8. [Validity and Efficiency](#8-validity-and-efficiency)
9. [Practical Implementation](#9-practical-implementation)
10. [Model Selection and Benchmarking](#10-model-selection-and-benchmarking)
11. [Forecast Error Metrics: A Critical Analysis](#11-forecast-error-metrics-a-critical-analysis)
12. [Advanced Topics](#12-advanced-topics)

---

## 1. Fundamental Principles

### Core Philosophy

**Forecasting is 5% models, 95% everything else.** The foundation of rigorous forecasting rests on:

1. **Forecastability assessment** before model selection
2. **Valid uncertainty quantification** through conformal prediction
3. **Empirical benchmarking** against naive baselines
4. **Distribution-free guarantees** without parametric assumptions

### The Validity-First Hierarchy

**Validity (calibration/coverage) must precede efficiency (sharpness).**

If a model claims 95% confidence intervals, they must cover approximately 95% of actual observations. Validity is non-negotiable in high-stakes applications (medicine, finance, autonomous systems). Only after achieving validity should efficiency—narrower prediction intervals—be pursued.

**Critical principle:** Validity in finite samples is automatically guaranteed by only one class of uncertainty quantification methods: **Conformal Prediction**. All other alternative methods (Bayesian, bootstrap, Monte Carlo) lack mathematical validity guarantees.

---

## 2. Forecastability Assessment Framework

### The Coefficient of Variation Problem

**The Coefficient of Variation (CoV = σ/μ) fundamentally misleads practitioners.**

#### Four Critical Failures

**1. Assumes normality in non-normal data**

Real demand/sales data exhibit skewness, multimodality, and heavy tails—nothing like the normal distribution CoV's interpretability assumes.

**2. Ignores temporal structure entirely**

CoV treats time series as unordered sets. Trend patterns, seasonal swings, and periodic spikes all become indistinguishable "variance" even when these patterns are **perfectly forecastable** with appropriate models.

Example: A steadily rising sales trend yields high standard deviation (early periods low, later periods high) and thus high CoV, causing practitioners to label it "volatile" and "unforecastable" when a simple trend model would capture it completely.

**3. Scale sensitivity produces misleading comparisons**

Products selling 1 unit/month vs. 1000 units/month can have dramatically different CoVs that exaggerate variability purely due to scale. CoV becomes especially unreliable when means approach zero; for intermittent/sporadic demand, zeroes make the ratio meaningless.

**4. Stability ≠ Predictability**

CoV and entropy-based measures gauge **stability**, not **forecastability**. These are operationally distinct concepts.

### Operational Definition of Forecastability

**Forecastability = the range of forecast errors achievable in the long run, not just historical stability.**

This operational framing has three implications:

1. **Method-dependent**: A series unforecastable by ARIMA might be highly forecastable by a model capturing nonlinear dynamics
2. **Requires actual forecasting experiments**: No retrospective statistical summaries suffice
3. **Connects to business decisions**: If no method significantly beats naive benchmarks, the series has genuinely low forecastability regardless of CoV

### Recommended Alternatives to CoV

#### Permutation Entropy (PE)

**Technical Framework**

Permutation Entropy quantifies predictability by measuring the complexity of ordinal patterns in time series.

**Key advantages:**
- Non-parametric (no restrictive distributional assumptions)
- Robust to noise
- Invariant under nonlinear monotonic transformations
- **Captures temporal ordering and causal relationships** (unlike CoV)

**Parameter Selection**

Two parameters required:

1. **Embedding Dimension (D)**: Consecutive values grouped into vectors to unfold system dynamics
   - Selection methods: False Nearest Neighbors (FNN) or Cao's method

2. **Embedding Time Delay (τ)**: Step size for constructing phase-space vectors
   - Selection methods: Average Mutual Information (AMI) or autocorrelation functions

**Interpretation**

- Lower PE → Higher predictability (series contains more regular patterns)
- Higher PE → Lower predictability (series approaches randomness)
- PE = 0: Completely deterministic
- PE = log(D!): Maximum entropy (random walk)

#### Forecast Error Benchmarks

**Methodology:**

1. Select simple forecasting method (naive, seasonal naive, moving average)
2. Simulate forecasts on historical data using time series cross-validation
3. Evaluate metrics (RMSE/MAE/MAPE)
4. Establish baseline achievable error for each series

**Interpretation:**

- Low naive model error → Series genuinely easy to forecast
- High naive model error → Genuine forecastability challenges exist
- Sophisticated model barely beats naive → Low inherent forecastability

#### Forecast Value Added (FVA) Analysis

**Framework:**

Compare multiple methods including:
- Naive forecasts
- Seasonal naive
- Statistical models (ARIMA, ETS)
- Machine learning models (CatBoost)
- Deep learning models

**Calculation:**

```
FVA = (Naive_Error - Model_Error) / Naive_Error × 100%
```

**Interpretation:**

- FVA > 10%: Model adds substantial value
- FVA 0-10%: Marginal improvement
- FVA < 0: Model destroys value (use naive forecast)

If even best algorithms barely beat naive forecasts, series has low forecastability. If simple model equals complex model, series is inherently easy or complex model is overkill.

---

## 3. Classical Statistical Models: ARIMA and ETS

### ARIMA (AutoRegressive Integrated Moving Average)

#### Theoretical Foundation

ARIMA models the temporal dependencies in a time series through three components:

- **AR (AutoRegressive)**: Current value depends linearly on its own past values
- **I (Integrated)**: Differencing to achieve stationarity
- **MA (Moving Average)**: Current value depends linearly on past forecast errors

#### Model Specification: ARIMA(p, d, q)

**Parameters:**

| Parameter | Meaning | Controls |
|-----------|---------|----------|
| p | AR order | Number of lagged observations in the model |
| d | Differencing order | Number of times the series is differenced to achieve stationarity |
| q | MA order | Number of lagged forecast errors |

**Mathematical Formulation:**

For a differenced series $w_t = \Delta^d y_t$:

```
w_t = c + φ₁w_{t-1} + φ₂w_{t-2} + ... + φ_pw_{t-p} + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q} + ε_t
```

where:
- $φ_i$: AR coefficients
- $θ_j$: MA coefficients
- $ε_t$: White noise (zero mean, constant variance σ²)
- $c$: Constant (drift term)

Using the backshift operator $B$ (where $By_t = y_{t-1}$):

```
φ(B)(1-B)^d y_t = c + θ(B)ε_t
```

where $φ(B) = 1 - φ₁B - φ₂B² - ... - φ_pB^p$ and $θ(B) = 1 + θ₁B + θ₂B² + ... + θ_qB^q$.

#### Stationarity and Invertibility Conditions

**Stationarity (AR component):** All roots of the characteristic polynomial $φ(B) = 0$ must lie outside the unit circle. For AR(1): $|φ₁| < 1$. For AR(2): $φ₁ + φ₂ < 1$, $φ₂ - φ₁ < 1$, $|φ₂| < 1$.

**Invertibility (MA component):** All roots of $θ(B) = 0$ must lie outside the unit circle. Ensures a unique MA representation exists.

#### Seasonal ARIMA: SARIMA(p, d, q)(P, D, Q)[m]

Extends ARIMA with seasonal components at lag $m$ (e.g., $m = 12$ for monthly data with annual seasonality):

```
Φ(B^m)φ(B)(1-B)^d(1-B^m)^D y_t = c + Θ(B^m)θ(B)ε_t
```

**Additional parameters:**

| Parameter | Meaning |
|-----------|---------|
| P | Seasonal AR order |
| D | Seasonal differencing order |
| Q | Seasonal MA order |
| m | Seasonal period length |

**Common configurations:**

| Data Pattern | Suggested Starting Model |
|-------------|-------------------------|
| Stationary, no seasonality | ARIMA(1,0,1) or ARIMA(2,0,2) |
| Trending, no seasonality | ARIMA(1,1,1) |
| Seasonal (monthly) | SARIMA(1,1,1)(1,1,1)[12] |
| Seasonal (quarterly) | SARIMA(1,1,1)(1,1,1)[4] |
| Weekly with daily data | SARIMA(1,1,1)(1,1,1)[7] |

#### Box-Jenkins Methodology

The classical approach to ARIMA model identification, estimation, and diagnostics:

**Step 1: Identification**

1. **Plot the series**: Visual inspection for trend, seasonality, outliers, structural breaks
2. **Stationarity testing**:
   - ADF (Augmented Dickey-Fuller) test: $H_0$: unit root present (non-stationary)
   - KPSS test: $H_0$: series is stationary (use both as cross-check)
   - If non-stationary → difference (increment $d$) and re-test
3. **ACF/PACF analysis** on the stationary series:
   - ACF cuts off at lag $q$ → MA(q)
   - PACF cuts off at lag $p$ → AR(p)
   - Both decay gradually → ARMA(p,q) — use information criteria

**ACF/PACF Signature Table:**

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|--------------|
| AR(p) | Exponential/oscillatory decay | Sharp cutoff after lag p |
| MA(q) | Sharp cutoff after lag q | Exponential/oscillatory decay |
| ARMA(p,q) | Decays after lag q | Decays after lag p |

**Step 2: Estimation**

- Maximum Likelihood Estimation (MLE) or Conditional Sum of Squares (CSS)
- Compare candidate models using information criteria:
  - **AIC** (Akaike): $AIC = -2\ln(L) + 2k$ — penalizes complexity lightly
  - **AICc** (corrected): $AICc = AIC + \frac{2k(k+1)}{n-k-1}$ — preferred for small samples
  - **BIC** (Bayesian): $BIC = -2\ln(L) + k\ln(n)$ — heavier penalty, favors parsimony

**Step 3: Diagnostics**

Residual checks (all must pass for a valid model):

1. **Zero mean**: $\bar{ε} ≈ 0$
2. **Constant variance**: No heteroscedasticity in residual plots
3. **No autocorrelation**: Ljung-Box test at multiple lags
   - $Q = n(n+2)\sum_{k=1}^{h}\frac{\hat{ρ}_k^2}{n-k}$ ~ $χ²(h - p - q)$
   - Fail to reject $H_0$ (no autocorrelation) → model adequate
4. **Normality** (optional but useful): Q-Q plot, Shapiro-Wilk test

**Step 4: Forecasting**

Point forecasts: conditional expectation $\hat{y}_{t+h|t} = E[y_{t+h} | y_t, y_{t-1}, ...]$

Prediction intervals (Gaussian assumption):

```
ŷ_{t+h} ± z_{α/2} × σ_h
```

where $σ_h$ is the h-step-ahead forecast error standard deviation, which grows with horizon.

**Limitation:** These intervals assume Gaussian errors. For valid intervals without distributional assumptions, wrap with conformal prediction (Section 7).

#### Auto-ARIMA

Automated model selection using stepwise search over (p,d,q)(P,D,Q)[m] parameter space:

```python
from pmdarima import auto_arima

model = auto_arima(
    y_train,
    seasonal=True,
    m=12,                    # Seasonal period
    d=None,                  # Auto-determine differencing
    D=None,                  # Auto-determine seasonal differencing
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    information_criterion='aicc',
    stepwise=True,           # Stepwise search (faster)
    suppress_warnings=True,
    trace=False
)

# Forecast
forecasts, conf_int = model.predict(
    n_periods=horizon,
    return_conf_int=True,
    alpha=0.05
)

# Model summary
print(model.summary())
print(f"Order: {model.order}")
print(f"Seasonal Order: {model.seasonal_order}")
```

**statsmodels implementation:**

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    y_train,
    order=(p, d, q),
    seasonal_order=(P, D, Q, m),
    enforce_stationarity=True,
    enforce_invertibility=True
)

results = model.fit(disp=False)

# Diagnostics
results.plot_diagnostics(figsize=(12, 8))

# Forecast with intervals
forecast = results.get_forecast(steps=horizon)
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)
```

#### ARIMA Strengths and Limitations

**Strengths:**
- Well-understood theoretical foundations with decades of refinement
- Interpretable parameters with clear statistical meaning
- Excellent for short-to-medium horizon linear forecasting
- Built-in uncertainty quantification (under Gaussian assumption)
- Handles trends and seasonality through differencing
- Efficient on small datasets

**Limitations:**
- **Linear assumption**: Cannot capture nonlinear dynamics without transformation
- **Stationarity requirement**: May over-difference, destroying signal
- **Fixed seasonal period**: Struggles with evolving or multiple seasonalities
- **No exogenous variable handling in pure ARIMA** (use ARIMAX/SARIMAX for covariates)
- **Gaussian error assumption** for prediction intervals (address with conformal prediction)
- **Poor on long horizons**: Reverts to unconditional mean, intervals blow up
- **Sensitive to outliers**: A single extreme value can distort parameter estimates

#### ARIMAX / SARIMAX

Extends ARIMA with exogenous regressors $X_t$:

```
φ(B)(1-B)^d y_t = c + βX_t + θ(B)ε_t
```

```python
model = SARIMAX(
    y_train,
    exog=X_train,            # Exogenous variables
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
results = model.fit()
forecast = results.get_forecast(steps=horizon, exog=X_future)
```

**Caution:** Exogenous variables must be available at forecast time. If you must forecast the covariates too, error compounds.

---

### ETS (Error, Trend, Seasonality) / Exponential Smoothing

#### Theoretical Foundation

ETS is a family of models that decompose a time series into three components—Error (E), Trend (T), and Seasonality (S)—each of which can take different forms. Where ARIMA models temporal correlations in the data, ETS models the generative process through smoothing equations.

#### Taxonomy: The ETS(E, T, S) Framework

Each component can assume one of several forms:

| Component | Options | Code |
|-----------|---------|------|
| **Error** | Additive, Multiplicative | A, M |
| **Trend** | None, Additive, Additive Damped, Multiplicative, Multiplicative Damped | N, A, Ad, M, Md |
| **Seasonality** | None, Additive, Multiplicative | N, A, M |

This produces 30 possible model configurations. The most commonly used:

| Model | ETS Code | Common Name |
|-------|----------|-------------|
| ETS(A,N,N) | Simple Exponential Smoothing | SES |
| ETS(A,A,N) | Holt's Linear Trend | Double Exponential Smoothing |
| ETS(A,Ad,N) | Damped Trend | Damped Holt's |
| ETS(A,A,A) | Holt-Winters Additive | — |
| ETS(A,A,M) | Holt-Winters Multiplicative | — |
| ETS(M,A,M) | Multiplicative Error, Additive Trend, Multiplicative Season | — |
| ETS(M,Ad,M) | Damped Multiplicative | Often the best default |

#### Component Equations

**ETS(A,A,A) — Additive Error, Additive Trend, Additive Seasonality:**

```
Forecast equation:    ŷ_{t+h|t} = ℓ_t + hb_t + s_{t+h-m}
Level:                ℓ_t = ℓ_{t-1} + b_{t-1} + αε_t
Trend:                b_t = b_{t-1} + βε_t
Seasonal:             s_t = s_{t-m} + γε_t
Error:                ε_t = y_t - ŷ_{t|t-1}
```

**ETS(M,A,M) — Multiplicative Error, Additive Trend, Multiplicative Seasonality:**

```
Forecast equation:    ŷ_{t+h|t} = (ℓ_t + hb_t) × s_{t+h-m}
Level:                ℓ_t = (ℓ_{t-1} + b_{t-1})(1 + αε_t)
Trend:                b_t = b_{t-1} + β(ℓ_{t-1} + b_{t-1})ε_t
Seasonal:             s_t = s_{t-m}(1 + γε_t)
Error:                ε_t = (y_t - ŷ_{t|t-1}) / ŷ_{t|t-1}
```

**Smoothing Parameters:**

| Parameter | Controls | Range | Interpretation |
|-----------|----------|-------|----------------|
| α (alpha) | Level smoothing | (0, 1) | Higher → more weight on recent data |
| β (beta) | Trend smoothing | (0, 1) | Higher → trend responds faster to changes |
| γ (gamma) | Seasonal smoothing | (0, 1) | Higher → seasonal pattern adapts faster |
| φ (phi) | Trend damping | (0.8, 1.0) | Lower → more aggressive damping toward flat |

#### Damped Trend

The damped trend modification multiplies the trend component by a damping factor $φ$ at each step:

```
ŷ_{t+h|t} = ℓ_t + (φ + φ² + ... + φ^h)b_t + s_{t+h-m}
```

As $h → ∞$, the cumulative trend approaches $φ/(1-φ) × b_t$, preventing unrealistic long-horizon extrapolation.

**Why it matters:** Damped trend models consistently outperform non-damped alternatives in forecasting competitions (M3, M4). The ETS(M,Ad,M) model is one of the strongest single-model performers across domains.

#### Additive vs. Multiplicative Selection

| Criterion | Additive | Multiplicative |
|-----------|----------|----------------|
| Seasonal amplitude | Constant regardless of level | Proportional to level |
| Data values | Can include zeros/negatives | Strictly positive only |
| Visual diagnostic | Seasonal swings same height | Seasonal swings grow with level |
| Log transform | Multiplicative → Additive after log | — |
| Typical domains | Temperature, some financial | Sales, demand, production |

**Decision rule:** If seasonal fluctuations scale with the level of the series, use multiplicative. If they remain constant, use additive. When uncertain, let information criteria decide.

#### State Space Formulation

ETS models have a state space representation enabling:
- Maximum Likelihood Estimation of parameters
- AIC/BIC-based model selection
- Analytical prediction intervals (under Gaussian errors)
- Likelihood-based diagnostics

The state vector $x_t = [ℓ_t, b_t, s_t, s_{t-1}, ..., s_{t-m+1}]'$ evolves via:

```
y_t = h(x_{t-1}) + k(x_{t-1})ε_t     (observation equation)
x_t = f(x_{t-1}) + g(x_{t-1})ε_t      (state transition equation)
```

where $h$, $k$, $f$, $g$ depend on the specific ETS variant.

#### Implementation

```python
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Automatic ETS fitting
model = ETSModel(
    y_train,
    error='add',           # 'add' or 'mul'
    trend='add',           # 'add', 'mul', or None
    seasonal='mul',        # 'add', 'mul', or None
    damped_trend=True,
    seasonal_periods=12
)
results = model.fit(optimized=True)

# Forecast with intervals
forecast = results.get_forecast(steps=horizon)
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)

print(results.summary())
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")
```

**Automated model selection via statsforecast:**

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

sf = StatsForecast(
    models=[AutoETS(season_length=12)],
    freq='M'
)
sf.fit(df=train_df)
forecasts = sf.predict(h=horizon)
```

**AutoETS** searches over all valid ETS configurations and selects the best by AICc, analogous to auto_arima for ARIMA.

#### ETS Strengths and Limitations

**Strengths:**
- Intuitive decomposition into level, trend, and seasonality
- Damped trend variant is one of the most robust single forecasting methods
- Fast estimation (analytical state space likelihood)
- Handles multiplicative patterns naturally without log transforms
- Strong performance in forecasting competitions (M3, M4, M5)
- Produces well-calibrated prediction intervals under Gaussian assumption

**Limitations:**
- **No exogenous variables**: Pure ETS cannot incorporate covariates (use regression with ETS errors, or switch to ARIMAX)
- **Single seasonal period**: Cannot handle dual seasonality (e.g., daily + weekly) natively
- **Multiplicative models require positive data**: Zeros or negatives break multiplicative error/seasonality
- **Gaussian interval assumption**: Same limitation as ARIMA — wrap with conformal prediction for valid intervals
- **Fixed smoothing parameters**: Does not adapt to regime changes without refitting
- **Poor on long horizons for trended data**: Even damped trend eventually flattens; may underestimate sustained trends

#### ARIMA vs. ETS: When to Use Which

| Scenario | Prefer ARIMA | Prefer ETS |
|----------|-------------|------------|
| Exogenous variables available | ✓ (SARIMAX) | ✗ |
| Multiplicative seasonality | Possible (log transform) | ✓ (native) |
| Strong autocorrelation in residuals | ✓ | ✗ |
| Need interpretable components | ✗ | ✓ (level, trend, season) |
| Very short series (< 30 obs) | ✗ | ✓ (fewer parameters) |
| Multiple seasonal periods | ✗ (TBATS/MSTL) | ✗ (TBATS/MSTL) |
| Need best single-model default | Consider | ETS(M,Ad,M) wins competitions |
| Ensemble/combination | Include both | Include both |

**Best practice:** Fit both, select by cross-validated forecast accuracy or combine.

---

## 4. Gradient Boosting: CatBoost for Time Series

### Why CatBoost Over Other GBDT Frameworks

CatBoost (Categorical Boosting) by Yandex offers specific advantages for time series forecasting that distinguish it from XGBoost and LightGBM:

**1. Ordered Boosting (Ordered Target Statistics)**

Standard gradient boosting (XGBoost, LightGBM) computes gradients using the entire training set, introducing a subtle but systematic target leakage. CatBoost uses **ordered boosting**: for each training example, the gradient is computed using a model trained only on examples preceding it in a random permutation. This is structurally similar to the temporal ordering constraint in time series — the model never sees future information during training.

**2. Native Categorical Feature Handling**

Time series features frequently include categorical variables (day of week, month, product category, region, holiday type). CatBoost handles these natively using ordered target statistics without requiring manual one-hot encoding or label encoding. XGBoost requires one-hot encoding (curse of dimensionality for high-cardinality features) and LightGBM's native categorical handling is less principled.

**3. Symmetric (Oblivious) Decision Trees**

CatBoost uses oblivious trees by default: the same splitting criterion at each level of the tree. This acts as an implicit regularizer, reduces overfitting on noisy time series, and produces faster inference at prediction time — critical for production forecasting systems.

**4. Built-in Quantile Regression**

Native support for quantile loss enables direct prediction interval construction and integration with conformal prediction frameworks.

### Feature Engineering for Time Series

Converting time series to tabular form for CatBoost requires deliberate feature engineering. The quality of features typically matters more than model hyperparameters.

#### Lag Features

```python
def create_lag_features(df, target_col, lags):
    """Create lagged values of the target variable."""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# Example: daily data
lags = [1, 2, 3, 7, 14, 21, 28]  # Recent + weekly multiples
df = create_lag_features(df, 'y', lags)
```

**Lag selection heuristics:**
- Include lags 1 through p where p is the dominant PACF cutoff
- Include seasonal lags: m, 2m (e.g., 7, 14 for daily with weekly pattern)
- For long-range dependencies: m×k for k = 1, 2, 3, ...

#### Rolling Statistics

```python
def create_rolling_features(df, target_col, windows):
    """Create rolling mean, std, min, max."""
    for w in windows:
        df[f'{target_col}_rmean_{w}'] = df[target_col].shift(1).rolling(w).mean()
        df[f'{target_col}_rstd_{w}'] = df[target_col].shift(1).rolling(w).std()
        df[f'{target_col}_rmin_{w}'] = df[target_col].shift(1).rolling(w).min()
        df[f'{target_col}_rmax_{w}'] = df[target_col].shift(1).rolling(w).max()
    return df

windows = [7, 14, 28, 90]
df = create_rolling_features(df, 'y', windows)
```

**Critical:** Always `shift(1)` before rolling to prevent data leakage. The rolling window must only use information available at forecast time.

#### Calendar Features

```python
def create_calendar_features(df, date_col):
    """Extract temporal features from date column."""
    dt = df[date_col]
    df['day_of_week'] = dt.dt.dayofweek        # 0=Mon, 6=Sun
    df['day_of_month'] = dt.dt.day
    df['day_of_year'] = dt.dt.dayofyear
    df['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df['month'] = dt.dt.month
    df['quarter'] = dt.dt.quarter
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = dt.dt.is_month_start.astype(int)
    df['is_month_end'] = dt.dt.is_month_end.astype(int)
    return df
```

**Mark these as categorical in CatBoost** for optimal target statistic computation:

```python
cat_features = ['day_of_week', 'month', 'quarter', 'is_weekend',
                'is_month_start', 'is_month_end']
```

#### Expanding Statistics (Target Encoding Alternatives)

```python
def create_expanding_features(df, target_col):
    """Expanding (cumulative) statistics — no leakage by construction."""
    df[f'{target_col}_expanding_mean'] = df[target_col].shift(1).expanding().mean()
    df[f'{target_col}_expanding_std'] = df[target_col].shift(1).expanding().std()
    return df
```

#### Fourier Features for Complex Seasonality

```python
import numpy as np

def create_fourier_features(df, period, n_harmonics, date_col):
    """Fourier terms for capturing seasonal patterns."""
    t = np.arange(len(df))
    for k in range(1, n_harmonics + 1):
        df[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
        df[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
    return df

# Weekly seasonality (3 harmonics) + annual seasonality (5 harmonics)
df = create_fourier_features(df, period=7, n_harmonics=3, date_col='ds')
df = create_fourier_features(df, period=365.25, n_harmonics=5, date_col='ds')
```

### CatBoost Implementation

#### Point Forecasting

```python
from catboost import CatBoostRegressor, Pool

# Prepare data (after feature engineering, drop NaN rows from lagging)
df = df.dropna()
split_idx = int(len(df) * 0.8)
calib_split = int(len(df) * 0.9)

train = df.iloc[:split_idx]
calib = df.iloc[split_idx:calib_split]
test = df.iloc[calib_split:]

feature_cols = [c for c in df.columns if c not in ['y', 'ds']]
cat_features = ['day_of_week', 'month', 'quarter', 'is_weekend']

train_pool = Pool(
    data=train[feature_cols],
    label=train['y'],
    cat_features=cat_features
)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,                       # Oblivious tree depth
    l2_leaf_reg=3.0,               # L2 regularization
    random_seed=42,
    loss_function='RMSE',
    early_stopping_rounds=50,
    verbose=100
)

model.fit(
    train_pool,
    eval_set=Pool(calib[feature_cols], calib['y'], cat_features=cat_features)
)

predictions = model.predict(test[feature_cols])
```

#### Quantile Regression for Prediction Intervals

```python
from catboost import CatBoostRegressor, Pool

confidence_level = 0.90
alpha = 1 - confidence_level

# Lower quantile model
model_lower = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function=f'Quantile:alpha={alpha/2}',
    random_seed=42,
    verbose=0
)
model_lower.fit(train_pool)

# Upper quantile model
model_upper = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function=f'Quantile:alpha={1-alpha/2}',
    random_seed=42,
    verbose=0
)
model_upper.fit(train_pool)

# Median model (point forecast)
model_median = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Quantile:alpha=0.5',
    random_seed=42,
    verbose=0
)
model_median.fit(train_pool)

# Predict
lower_bound = model_lower.predict(test[feature_cols])
upper_bound = model_upper.predict(test[feature_cols])
point_forecast = model_median.predict(test[feature_cols])
```

**Limitation:** Raw quantile regression intervals are NOT guaranteed to achieve the target coverage. Wrap with conformal prediction for valid intervals (see Section 7).

#### Multi-Step Forecasting Strategies

**1. Recursive (Autoregressive)**

Feed predictions back as lag features. Cheapest to train (single model) but error accumulates at longer horizons.

```python
def recursive_forecast(model, last_known, feature_cols, horizon):
    predictions = []
    current_features = last_known.copy()
    
    for h in range(horizon):
        pred = model.predict(current_features[feature_cols].values.reshape(1, -1))[0]
        predictions.append(pred)
        # Update lag features with the prediction
        current_features = update_features(current_features, pred, h)
    
    return np.array(predictions)
```

**2. Direct (One Model per Horizon)**

Train separate CatBoost models for each forecast horizon h = 1, 2, ..., H. No error accumulation but requires H models and H sets of aligned training data.

```python
models = {}
for h in range(1, horizon + 1):
    df_h = create_features_for_horizon(df, h)
    models[h] = CatBoostRegressor(iterations=1000, verbose=0)
    models[h].fit(df_h[feature_cols], df_h[f'y_t+{h}'])
```

**3. DirRec (Hybrid)**

Combines direct and recursive: train direct models but include predictions from previous horizons as features.

**Recommendation:** Direct multi-step for horizons ≤ 12. Recursive for very long horizons where training direct models becomes impractical. Always benchmark both.

#### Hyperparameter Tuning

Key hyperparameters and their typical search ranges:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `iterations` | 500–5000 | More = more complex; use early stopping |
| `learning_rate` | 0.01–0.3 | Lower = slower training, often better generalization |
| `depth` | 4–10 | Deeper = more interactions captured; risk of overfitting |
| `l2_leaf_reg` | 1–10 | Higher = stronger regularization |
| `min_data_in_leaf` | 1–100 | Higher = more conservative splits |
| `bagging_temperature` | 0–1 | Controls Bayesian bootstrap; higher = more randomization |
| `random_strength` | 0–10 | Randomization of split scoring |

```python
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 10],
    'iterations': [1000],
}

best_score = float('inf')
best_params = None

for depth in param_grid['depth']:
    for lr in param_grid['learning_rate']:
        for l2 in param_grid['l2_leaf_reg']:
            scores = []
            for train_idx, val_idx in tscv.split(X):
                model = CatBoostRegressor(
                    depth=depth,
                    learning_rate=lr,
                    l2_leaf_reg=l2,
                    iterations=1000,
                    early_stopping_rounds=50,
                    verbose=0
                )
                model.fit(
                    X.iloc[train_idx], y.iloc[train_idx],
                    eval_set=(X.iloc[val_idx], y.iloc[val_idx])
                )
                pred = model.predict(X.iloc[val_idx])
                scores.append(mean_absolute_error(y.iloc[val_idx], pred))
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = {'depth': depth, 'lr': lr, 'l2': l2}
```

**Important:** Always use `TimeSeriesSplit`, never random K-fold, for time series hyperparameter search.

#### CatBoost Strengths and Limitations

**Strengths:**
- Ordered boosting eliminates target leakage during training
- Native categorical feature support without preprocessing
- Oblivious trees provide implicit regularization and fast inference
- Built-in quantile regression for prediction intervals
- GPU training support for large datasets
- Strong out-of-the-box performance with minimal tuning
- Feature importance available (SHAP values, permutation importance)
- Handles missing values natively

**Limitations:**
- **Requires feature engineering**: Raw time series must be transformed to tabular form; quality of lag/calendar/rolling features determines performance ceiling
- **No native temporal structure**: Model treats rows as exchangeable; temporal ordering enforced only through features
- **Recursive forecasting error accumulation**: Multi-step predictions via autoregressive loop degrade with horizon
- **Large memory footprint**: Feature-rich datasets can become unwieldy
- **Overfitting risk on small series**: Gradient boosting can overfit when series length < 200
- **Uncertainty intervals from quantile regression lack coverage guarantees**: Must be conformalized

---

## 5. Conformal Prediction: Theoretical Foundations

### Core Principles

Conformal Prediction is a machine learning framework quantifying uncertainty to produce probabilistic predictions with mathematical validity guarantees.

**Eight Foundational Principles:**

1. **Validity**: Prediction regions encompass actual target values with user-specified confidence level (e.g., 95% confidence → 95% coverage)

2. **Efficiency**: Prediction intervals/regions should be as small as possible while preserving desired confidence level

3. **Adaptivity**: Prediction sets adaptive to individual examples—harder-to-predict examples receive wider intervals

4. **Distribution-free**: No assumptions about underlying data distribution required (only exchangeability, less restrictive than IID)

5. **Online adaptivity**: Can adjust to new data points without retraining

6. **Compatibility**: Seamlessly integrates with any existing model (CatBoost, neural networks, random forests, ARIMA, ETS, etc.)

7. **Non-intrusive**: Requires no modification to deployed point prediction models—functions as uncertainty quantification layer

8. **Interpretability**: Produces easily understood prediction sets/intervals with clear uncertainty measures

### Theoretical Guarantees

**Finite Sample Coverage Guarantee:**

For significance level ε (e.g., ε = 0.05 for 95% confidence):

```
P(y_new ∈ Prediction_Set) ≥ 1 - ε
```

This guarantee holds for:
- **Any** underlying prediction model
- **Any** data distribution (under exchangeability)
- **Any** dataset size (including small samples)
- **Finite samples** (not asymptotic)

No other uncertainty quantification framework provides these guarantees.

### Exchangeability vs. IID

**Exchangeability**: Joint probability distribution invariant to permutations of the data

```
P(Z₁, Z₂, ..., Zₙ) = P(Z_π(1), Z_π(2), ..., Z_π(n))
```

for any permutation π.

**Key distinction:**
- IID → Exchangeability ✓
- Exchangeability → IID ✗

Exchangeability is weaker than IID, allowing conformal prediction broader applicability. However, time series violate exchangeability—requiring specialized adaptations covered in Section 7.

### Types of Conformal Predictors

#### Transductive Conformal Prediction (TCP)

**Characteristics:**
- Uses entire training dataset for each test instance
- Retrains model for each potential label assignment
- Provides strongest theoretical guarantees
- Computationally expensive (O(nk) for n calibration points, k classes)

**When to use:**
- Small datasets (< 1000 observations)
- Critical applications requiring strongest guarantees
- Research/academic contexts

#### Inductive Conformal Prediction (ICP)

**Characteristics:**
- Splits data into proper training set and calibration set
- Trains model once on training set
- Uses calibration set to compute nonconformity scores
- Computationally efficient (≈ same speed as base model)

**Standard split:**
- 60-80% proper training
- 20-40% calibration
- Minimum 500 calibration points recommended

**When to use:**
- Large datasets (> 1000 observations)
- Production systems
- Real-time applications
- Industry practice

---

## 6. Nonconformity Measures

Nonconformity measures quantify how different a new data point is from existing data. Selection dramatically impacts prediction set efficiency while maintaining validity.

### Classification Nonconformity Measures

#### 1. Hinge Loss (Inverse Probability / LAC Loss)

**Formula:**
```
α = 1 - P(y_true)
```

**Example:**
- Predicted probabilities: [0.5, 0.3, 0.2] for classes [0, 1, 2]
- True class: 1
- Nonconformity score: 1 - 0.3 = 0.7

**Characteristics:**
- Simplest measure
- Considers only probability of true class
- Produces narrowest average set sizes (best AvgC)
- Individual label assessment

**Use when:**
- Minimizing average prediction set size is priority
- Underlying model produces well-calibrated probabilities

#### 2. Margin

**Formula:**
```
α = max_{y ≠ y_true} P(y) - P(y_true)
```

**Example:**
- Predicted probabilities: [0.5, 0.3, 0.2] for classes [0, 1, 2]
- True class: 1
- Nonconformity score: max(0.5, 0.2) - 0.3 = 0.2

**Characteristics:**
- Considers most likely incorrect class
- Produces highest proportion singleton predictions (best OneC)
- Comparison between true and competing classes

**Use when:**
- Maximizing singleton predictions is priority
- Need to distinguish between close competing classes

#### 3. Brier Score

**Formula:**
```
α = Σ(y_pred_i - y_true_i)² / n_classes
```

**Example:**
- Predicted probabilities: [0.5, 0.3, 0.2]
- True class: 1 (one-hot: [0, 1, 0])
- Nonconformity score: [(0-0.5)² + (1-0.3)² + (0-0.2)²] / 3 = 0.26

**Characteristics:**
- Proper scoring rule
- Captures both calibration and discrimination
- Squared error penalization
- Range: [0, 1] where 0 is perfect

**Use when:**
- Well-calibrated probabilities are essential
- Need to balance multiple objectives
- Following proper scoring rule principles

### Regression Nonconformity Measures

#### 1. Absolute Error

**Formula:**
```
α = |y_pred - y_true|
```

**Pros:**
- Simple, interpretable
- Direct measure of prediction error
- Uniform interpretation across datasets

**Cons:**
- Scale sensitive (large targets → large errors)
- No consideration for data distribution
- May produce overly optimistic/pessimistic intervals

**Use when:**
- Target variable scale is consistent
- Simple interpretation is priority
- Homoscedastic errors expected

#### 2. Normalized Error

**Formula:**
```
α = |y_pred - y_true| / scale_estimate
```

where `scale_estimate` can be:
- Mean Absolute Error (MAE)
- Standard deviation of residuals
- Other scale measures

**Pros:**
- Scale invariant
- Accounts for heteroscedasticity
- Adaptive to local data properties
- Consistent across different target scales

**Cons:**
- Additional complexity
- Requires sufficient data for reliable scale estimation
- Risk of misleading results with poor scale choice

**Use when:**
- Target variable scale varies significantly
- Heteroscedastic errors present
- Comparing models across different datasets

---

## 7. Time Series Forecasting with Conformal Prediction

### The Exchangeability Challenge

**Problem:** Time series data violates exchangeability assumption—temporal order matters fundamentally.

**Solution:** Specialized conformal prediction methods designed for time series:

1. Ensemble Batch Prediction Intervals (EnbPI)
2. Conformalized Quantile Regression (CQR)
3. Jackknife+ methods
4. Adaptive Conformal Inference

### Ensemble Batch Prediction Intervals (EnbPI)

**Key Innovation:** Does not require data exchangeability—custom-built for time series.

**Theoretical Basis:**

Achieves finite-sample, approximately valid marginal coverage for broad regression functions under mild assumption of **strongly mixing stochastic errors**.

**Algorithm:**

1. **Bootstrap Ensemble Creation**
   - Draw B bootstrap samples (with replacement) from training data
   - Train base forecasting model on each bootstrap sample
   - Results in ensemble {f₁, f₂, ..., f_B}

2. **Out-of-Sample Residual Computation**
   - For each point t in training data
   - Compute residuals using only ensemble members that did NOT use point t
   - Compile all out-of-sample errors into array R = {r₁, r₂, ..., r_T}

3. **Point Prediction Generation**
   - Aggregate predictions from ensemble (mean, median, or weighted)
   - ŷ = aggregate({f₁(x), f₂(x), ..., f_B(x)})

4. **Prediction Interval Construction**
   - For confidence level (1-α):
   - Compute quantiles from residual distribution R
   - Lower bound: ŷ - Q_{1-α/2}(|R|)
   - Upper bound: ŷ + Q_{1-α/2}(|R|)

**Advantages:**
- No data splitting required
- Computationally efficient (single ensemble training)
- Avoids overfitting
- Scalable to arbitrarily many sequential predictions
- Works with any regression function (CatBoost, neural networks, etc.)

**Implementation Libraries:**
- Amazon Fortuna
- MAPIE (Python)
- PUNCC (Python)

### Conformalized Quantile Regression (CQR)

**Methodology:**

Combines quantile regression with conformal prediction for improved conditional coverage.

**Algorithm:**

1. **Quantile Regression Training**
   - Train model to predict lower quantile q_α/2 and upper quantile q_{1-α/2}
   - Produces initial prediction interval [q̂_α/2(x), q̂_{1-α/2}(x)]

2. **Calibration Set Conformity Scores**
   - For each calibration point (x_i, y_i):
   - Compute conformity score: α_i = max(q̂_α/2(x_i) - y_i, y_i - q̂_{1-α/2}(x_i))

3. **Adjusted Prediction Intervals**
   - For test point x:
   - Compute correction term: Q_{1-α}({α_i})
   - Final interval: [q̂_α/2(x) - Q, q̂_{1-α/2}(x) + Q]

**Advantages:**
- Better conditional coverage than standard methods
- Adapts interval width to input features
- Robust to distribution shift
- Handles heteroscedastic data naturally

**CQR with CatBoost:**

```python
from catboost import CatBoostRegressor

# Step 1: Train quantile models on proper training set
model_lo = CatBoostRegressor(loss_function='Quantile:alpha=0.05', verbose=0)
model_hi = CatBoostRegressor(loss_function='Quantile:alpha=0.95', verbose=0)

model_lo.fit(X_train_proper, y_train_proper)
model_hi.fit(X_train_proper, y_train_proper)

# Step 2: Compute conformity scores on calibration set
lo_calib = model_lo.predict(X_calib)
hi_calib = model_hi.predict(X_calib)
scores = np.maximum(lo_calib - y_calib, y_calib - hi_calib)

# Step 3: Compute correction quantile
n = len(scores)
q_level = np.ceil((n + 1) * 0.90) / n  # For 90% coverage
Q = np.quantile(scores, q_level)

# Step 4: Adjusted intervals on test set
lo_test = model_lo.predict(X_test) - Q
hi_test = model_hi.predict(X_test) + Q
```

**Pinball Loss:**

Quantile regression trains using asymmetric loss:

```
L_τ(y, ŷ) = {
    τ(y - ŷ)     if y ≥ ŷ
    (τ-1)(y - ŷ) if y < ŷ
}
```

where τ is target quantile.

### Jackknife+ Regression

**Innovation:** Leave-one-out residual computation without full model retraining.

**Algorithm:**

1. **Initial Model Training**
   - Train model f on full training set

2. **Jackknife Predictions**
   - For each training point i:
   - Predict using model trained on data with i removed: f_{-i}(x_i)
   - Compute leave-one-out residual: R_i = y_i - f_{-i}(x_i)

3. **Prediction Intervals**
   - For test point x:
   - Base prediction: ŷ = f(x)
   - Interval: [ŷ - Q_{1-α}(|R|), ŷ + Q_{1-α}(|R|)]

**Computational Efficiency:**

For linear models and certain tree ensembles, jackknife predictions can be computed without B full retraining runs using influence functions or out-of-bag predictions.

### NeuralProphet Implementation

**Architecture:**

PyTorch-based framework merging interpretability with deep learning scalability.

**Model Components:**
1. Trend module
2. Seasonality module (Fourier series)
3. Holiday/event module
4. Auto-regression module (AR)
5. Covariate module (exogenous variables)

**Conformal Prediction Integration:**

NeuralProphet implements two approaches:

**1. Quantile Regression Mode**
```python
from neuralprophet import NeuralProphet

confidence_level = 0.9
quantiles = [(1 - confidence_level)/2, (1 + confidence_level)/2]  # [0.05, 0.95]

model = NeuralProphet(quantiles=quantiles)
model.fit(train_df)
forecast = model.predict(test_df)
```

**2. Inductive Conformal Prediction Mode**
```python
model = NeuralProphet()
model.fit(train_df)

# Split calibration set
train_split, calib_split = model.split_df(df, freq='H', valid_p=0.2)

# Generate conformal predictions
conformal_forecast = model.conformal_predict(
    df=test_df,
    calibration_df=calib_split,
    alpha=0.1  # 90% confidence
)
```

**Data Format Requirements:**
- Time column named `ds`
- Target column named `y`
- Standard pandas DataFrame

---

## 8. Validity and Efficiency

### Validity Metrics

#### Coverage Probability

**Definition:** Proportion of prediction intervals containing true values.

**Formula:**
```
Coverage = (1/n) Σ I(y_i ∈ [L_i, U_i])
```

where I is indicator function, L_i and U_i are lower/upper bounds.

**Target:** Should match specified confidence level (e.g., 95% confidence → ~95% coverage)

**Evaluation:**
- Coverage < Target: Intervals too narrow (validity failure)
- Coverage ≈ Target: Valid predictor ✓
- Coverage >> Target: Intervals too wide (inefficient but valid)

#### Conditional Coverage

**Problem:** Marginal coverage may mask systematic failures in subgroups.

**Evaluation:** Compute coverage separately for:
- Different input feature ranges
- Different time periods
- Different classes/categories

**Goal:** Uniform coverage across all conditions.

### Efficiency Metrics

#### Average Interval Width

**Formula:**
```
Width = (1/n) Σ (U_i - L_i)
```

**Interpretation:**
- Lower width = More informative predictions
- Must be evaluated conditional on achieving validity

#### Prediction Set Size (Classification)

**OneC (Singleton Proportion):**
```
OneC = (Count of singleton sets) / (Total prediction sets)
```

Higher OneC = More decisive predictions

**AvgC (Average Label Count):**
```
AvgC = (Total labels across all sets) / (Total prediction sets)
```

Lower AvgC = More precise predictions

### Calibration Metrics

#### Reliability Diagram

**Method:**
1. Group predictions by confidence score bins
2. Compute observed frequency in each bin
3. Plot observed vs. predicted confidence

**Perfect calibration:** Points fall on diagonal (y = x)

#### Expected Calibration Error (ECE)

**Formula:**
```
ECE = Σ (n_b/n) |acc_b - conf_b|
```

where:
- n_b: number of predictions in bin b
- acc_b: accuracy in bin b
- conf_b: average confidence in bin b

**Target:** ECE ≈ 0

---

## 9. Practical Implementation

### ICP Implementation Framework

**Step 1: Data Splitting**

```python
from sklearn.model_selection import train_test_split

# Split into proper training and calibration
X_train_proper, X_calib, y_train_proper, y_calib = train_test_split(
    X_train, y_train,
    test_size=0.25,  # 25% for calibration
    random_state=42
)
```

**Minimum calibration set size:** 500 observations recommended

**Step 2: Base Model Training**

```python
from catboost import CatBoostRegressor

# Train model ONLY on proper training set
base_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    verbose=0
)
base_model.fit(X_train_proper, y_train_proper)
```

**Step 3: Calibration Nonconformity Scores**

```python
# Get predictions on calibration set
calib_predictions = base_model.predict(X_calib)

# Compute nonconformity scores (absolute error)
calibration_scores = np.abs(y_calib - calib_predictions)
```

**Step 4: Test Prediction with Intervals**

```python
def predict_with_intervals(model, calibration_scores, X_test, confidence=0.95):
    # Point predictions
    predictions = model.predict(X_test)

    # Compute quantile from calibration scores
    alpha = 1 - confidence
    n = len(calibration_scores)

    # Adjusted quantile for finite sample correction
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q = np.quantile(calibration_scores, q_level)

    # Construct prediction intervals
    lower_bounds = predictions - q
    upper_bounds = predictions + q

    return predictions, lower_bounds, upper_bounds
```

**Step 5: Validation**

```python
predictions, lower, upper = predict_with_intervals(
    base_model, calibration_scores, X_test, confidence=0.95
)

# Compute coverage
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"Coverage: {coverage:.3f}")

# Compute average width
avg_width = np.mean(upper - lower)
print(f"Average Interval Width: {avg_width:.3f}")
```

### Complete Classification Example

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ConformalClassifier:
    def __init__(self, base_model, alpha=0.05):
        self.model = base_model
        self.alpha = alpha
        self.calibration_scores = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        # Train base model
        self.model.fit(X_train, y_train)

        # Get class probabilities for calibration set
        calib_probs = self.model.predict_proba(X_calib)

        # Compute hinge loss nonconformity scores
        self.calibration_scores = []
        for i, true_class in enumerate(y_calib):
            score = 1 - calib_probs[i, true_class]
            self.calibration_scores.append(score)

        self.calibration_scores = np.array(self.calibration_scores)

    def predict(self, X_test):
        # Get class probabilities
        test_probs = self.model.predict_proba(X_test)
        n_test = len(X_test)
        n_classes = test_probs.shape[1]

        # Compute quantile threshold
        n_calib = len(self.calibration_scores)
        q_level = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
        threshold = np.quantile(self.calibration_scores, q_level)

        # Construct prediction sets
        prediction_sets = []
        for i in range(n_test):
            pred_set = []
            for c in range(n_classes):
                score = 1 - test_probs[i, c]
                if score <= threshold:
                    pred_set.append(c)
            prediction_sets.append(pred_set)

        return prediction_sets

# Usage
classifier = ConformalClassifier(RandomForestClassifier(n_estimators=100))
classifier.fit(X_train_proper, y_train_proper, X_calib, y_calib)
prediction_sets = classifier.predict(X_test)

# Evaluate
coverage = np.mean([y_test[i] in pred_set
                    for i, pred_set in enumerate(prediction_sets)])
avg_size = np.mean([len(pred_set) for pred_set in prediction_sets])

print(f"Coverage: {coverage:.3f}")
print(f"Average Set Size: {avg_size:.3f}")
```

### Time Series Cross-Validation

**Problem:** Standard train/test split violates temporal ordering.

**Solution:** Time series cross-validation with expanding/rolling windows.

**Expanding Window:**

```python
from sklearn.metrics import mean_absolute_error

def expanding_window_cv(data, min_train_size, horizon, model_class):
    results = []

    for i in range(min_train_size, len(data) - horizon):
        train = data[:i]
        test = data[i:i+horizon]

        model = model_class()
        model.fit(train)

        predictions = model.predict(test)
        mae = mean_absolute_error(test, predictions)
        results.append(mae)

    return np.mean(results), np.std(results)
```

**Rolling Window:**

```python
def rolling_window_cv(data, train_size, horizon, model_class):
    results = []

    for i in range(len(data) - train_size - horizon):
        train = data[i:i+train_size]
        test = data[i+train_size:i+train_size+horizon]

        model = model_class()
        model.fit(train)

        predictions = model.predict(test)
        mae = mean_absolute_error(test, predictions)
        results.append(mae)

    return np.mean(results), np.std(results)
```

---

## 10. Model Selection and Benchmarking

### The Naive Benchmark Principle

**Rule:** Every forecasting model must be benchmarked against naive methods before deployment.

**Why:** If sophisticated models cannot beat naive forecasts by meaningful margin (>10%), the series has low inherent forecastability or model is misspecified.

### Standard Naive Benchmarks

#### 1. Naive Forecast (Random Walk)

**Method:** Next value = Last observed value

```python
def naive_forecast(y, horizon=1):
    return np.repeat(y[-1], horizon)
```

**Use case:** Non-seasonal data

#### 2. Seasonal Naive

**Method:** Next value = Value from same season last cycle

```python
def seasonal_naive(y, season_length, horizon=1):
    return np.tile(y[-season_length:], horizon//season_length + 1)[:horizon]
```

**Use case:** Seasonal data

#### 3. Drift Method

**Method:** Naive with linear trend

```python
def drift_method(y, horizon=1):
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, horizon + 1)
```

**Use case:** Trending data

#### 4. Moving Average

**Method:** Next value = Average of last k observations

```python
def moving_average(y, k=3, horizon=1):
    ma = np.mean(y[-k:])
    return np.repeat(ma, horizon)
```

**Use case:** Noisy data without trend/seasonality

### Benchmark Evaluation Framework

```python
def benchmark_forecasting_models(data, test_size=0.2, horizon=1):
    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]

    results = {}

    # Naive forecast
    naive_pred = naive_forecast(train, horizon)
    results['Naive'] = {
        'MAE': mean_absolute_error(test[:horizon], naive_pred),
        'RMSE': np.sqrt(mean_squared_error(test[:horizon], naive_pred))
    }

    # Seasonal naive (assuming weekly seasonality = 7)
    seasonal_pred = seasonal_naive(train, season_length=7, horizon=horizon)
    results['Seasonal_Naive'] = {
        'MAE': mean_absolute_error(test[:horizon], seasonal_pred),
        'RMSE': np.sqrt(mean_squared_error(test[:horizon], seasonal_pred))
    }

    # Add ARIMA, ETS, CatBoost models here
    # ...

    return results
```

### Model Comparison Decision Tree

```
1. Does model beat naive by >10%?
   NO → Series has low forecastability OR model misspecified
   YES → Continue to step 2

2. Does model beat seasonal naive (if seasonal data)?
   NO → Model failing to capture seasonality
   YES → Continue to step 3

3. Does complexity justify improvement?
   (Simple model within 5% of complex model?)
   YES → Use simple model (Occam's razor)
   NO → Use complex model

4. Does model produce valid prediction intervals?
   NO → Add conformal prediction layer
   YES → Deploy
```

### Recommended Model Hierarchy

For a typical forecasting pipeline, evaluate in this order:

1. **Naive baselines** (random walk, seasonal naive, drift)
2. **ETS** (especially ETS(M,Ad,M) with damped trend)
3. **ARIMA/SARIMA** (auto_arima for automated selection)
4. **CatBoost** (with engineered lag/calendar/rolling features)
5. **Specialized architectures** (N-BEATS, N-HiTS, TSMixer)
6. **Ensembles** (combine top 2-3 models from above)

### Critical Evaluation Metrics

See **Section 11: Forecast Error Metrics** for comprehensive treatment of all point, scaled, probabilistic, and calibration metrics with their strengths, limitations, and selection guidance.

---

## 11. Forecast Error Metrics: A Critical Analysis

### Why Metrics Matter More Than Models

**"Measurement drives conclusion."** (Manokhin, 2025)

The choice of error metric fundamentally determines which model appears best. The Makridakis competitions (M1–M4) relied heavily on MAPE and sMAPE, which systematically biased results toward simple, conservative methods. Only with the M5 competition (2020) did the field begin adopting more robust evaluation frameworks. Manokhin argues this metric dependency created a "cult of simplicity" that delayed exploration of machine learning in forecasting by decades.

**Core principle:** Always evaluate with multiple metrics. If model rankings change under different metrics, investigate why — the disagreement reveals something fundamental about the models or the data.

### Scale-Dependent Metrics

These measure errors in the same units as the data. Cannot be compared across series with different scales.

#### Mean Absolute Error (MAE)

**Formula:**

```
MAE = (1/n) Σ |y_i - ŷ_i|
```

**Properties:**
- Measures median performance (minimized by the conditional median)
- Robust to outliers (compared to RMSE)
- Equal penalty for over- and under-forecasting
- Interpretable in original units

**Limitations:**
- Scale-dependent: cannot aggregate across series with different magnitudes
- Does not penalize large errors proportionally more than small ones

**When to use:** Single-series evaluation, contexts where all errors are equally costly regardless of magnitude.

#### Root Mean Squared Error (RMSE)

**Formula:**

```
RMSE = √[(1/n) Σ (y_i - ŷ_i)²]
```

**Properties:**
- Measures mean performance (minimized by the conditional mean)
- Penalizes large errors disproportionately due to squaring
- Differentiable (useful as optimization objective)
- RMSE ≥ MAE always; equality only when all errors are identical

**Limitations:**
- Scale-dependent
- Sensitive to outliers — a single extreme error can dominate the metric
- Not suitable for intermittent/sparse demand (zero-inflated data)

**When to use:** Contexts where large errors are disproportionately costly (e.g., energy grid balancing, financial risk). When you need a differentiable loss function for model training.

**MAE vs. RMSE decision rule:** If the cost of errors grows linearly with error magnitude, use MAE. If cost grows faster than linearly (big misses are catastrophically worse than small misses), use RMSE.

### Percentage-Based Metrics

Scale-free metrics that allow cross-series comparison. All suffer from fundamental limitations.

#### Mean Absolute Percentage Error (MAPE)

**Formula:**

```
MAPE = (100/n) Σ |y_i - ŷ_i| / |y_i|
```

**Critical Flaws (Manokhin, 2025; Hyndman & Koehler, 2006):**

**1. Undefined at zero, explosive near zero.** When actual values are zero or near-zero, MAPE becomes infinite or astronomically large. A single tiny-actual observation can dominate the entire metric. This makes MAPE unusable for intermittent demand, which is the majority of real-world retail/supply chain data.

**2. Asymmetric — systematically rewards under-forecasting.** Forecasting too low (under-forecast) produces a smaller percentage error than forecasting too high (over-forecast) by the same absolute amount. This is because the denominator (the actual) is smaller when you under-forecast a rising series. The consequence: models that systematically undershoot appear superior on MAPE, even when their absolute errors are worse. Simple, conservative methods inherently benefit from this bias.

**3. Disproportionate penalty for small-volume series.** When averaging MAPE across many series, low-volume series dominate the metric. A 5-unit error on a series averaging 10 units (50% MAPE) counts equally to a 500-unit error on a series averaging 10,000 units (5% MAPE) — even though the latter is far more consequential in business terms.

**4. No consistent interpretation.** A 20% MAPE has no inherent meaning without context. It could reflect moderate errors everywhere, or near-perfect performance on most series plus one disaster on a tiny series.

**5. Distorted the "simplicity" narrative in forecasting.** Manokhin (2025) argues that MAPE's use as the primary metric in the M1–M3 competitions exaggerated the superiority of simple methods, creating a decades-long bias against complexity in the forecasting field. Optimizing purely for MAPE can lead to systematically biased forecasts that cause real business harm — for example, consistently under-forecasting demand leads to stockouts.

**Recommendation:** Avoid MAPE. If stakeholders demand a percentage metric, use WAPE instead.

#### Symmetric MAPE (sMAPE)

**Formula:**

```
sMAPE = (200/n) Σ |y_i - ŷ_i| / (|y_i| + |ŷ_i|)
```

**Intended fix:** Addresses MAPE's zero-denominator problem and asymmetry.

**Actual result (Goodwin & Lawton, 1999):** sMAPE introduces the *opposite* asymmetry — it systematically favors over-forecasting. When the forecast is too low, the denominator shrinks (actual + small forecast), inflating the error. When the forecast is too high, the denominator grows (actual + large forecast), deflating the error.

**Additional problems:**
- Range is [0, 200%], not [0, 100%], causing confusion
- Still distorts rankings when comparing across series with different scales
- The M4 competition used sMAPE as one component of its OWA metric, perpetuating its influence

**Recommendation:** Avoid sMAPE. Despite its widespread use in competitions, it does not solve MAPE's fundamental problems — it merely reverses them.

#### Weighted Absolute Percentage Error (WAPE / wMAPE)

**Formula:**

```
WAPE = Σ |y_i - ŷ_i| / Σ |y_i| × 100
```

**Properties:**
- Volume-weighted: high-volume series naturally contribute more to the aggregate error
- Answers the business question "what percentage of total demand did we miss?"
- Well-defined when individual actuals are zero (as long as aggregate is non-zero)
- Equivalent to MAE / Mean(Actuals) × 100

**Limitations:**
- Still a percentage of actuals, so retains a subtle bias toward under-forecasting at aggregate level
- Cannot handle the case where total actuals sum to zero
- Less informative for per-series analysis

**When to use:** Multi-series aggregation where business cares about total error relative to total volume. The best percentage metric available, though still imperfect.

### Scaled Metrics

Normalize errors relative to a benchmark forecast, enabling cross-series comparison without percentage-error biases.

#### Mean Absolute Scaled Error (MASE)

**Formula:**

```
MASE = MAE_model / MAE_naive_in_sample
```

where the denominator is the in-sample one-step-ahead MAE of the naive (or seasonal naive) forecast:

```
MAE_naive = (1/(T-m)) Σ_{t=m+1}^{T} |y_t - y_{t-m}|
```

with m = 1 for non-seasonal data and m = seasonal period for seasonal data.

**Proposed by:** Hyndman & Koehler (2006) as a replacement for MAPE.

**Properties:**
- Scale-free: enables cross-series comparison
- Well-defined for all series (except constant-zero, which are trivial)
- No directional bias: treats over- and under-forecasting symmetrically
- Interpretable: MASE < 1 means model beats naive; MASE > 1 means model is worse than naive
- Suitable for intermittent demand (unlike MAPE)

**Limitations (Manokhin, 2025):**
- The naive forecast may not be a meaningful benchmark for all series. In series with high-frequency oscillations, domain-specific constraints, or policy-driven signals, beating a bad baseline is not meaningful progress
- Assumes the naive benchmark is appropriate — in finance or event-driven domains, a naive model may be irrelevant
- Unstable for short or highly volatile series where small changes in in-sample differences can skew the scaling factor
- Not intuitive to non-technical stakeholders: "our MASE is 0.8" requires explanation ("20% better than naive")

**When to use:** Multi-series comparison, competition benchmarking, research contexts. Best available scaled metric for point forecasts, but requires domain awareness about whether the naive baseline is sensible.

#### Root Mean Squared Scaled Error (RMSSE)

**Formula:**

```
RMSSE = √[(1/h) Σ_{j=1}^{h} e_j² / ((1/(T-m)) Σ_{t=m+1}^{T} (y_t - y_{t-m})²)]
```

**Properties:**
- RMSE analogue of MASE: scales squared errors by the squared naive benchmark
- Penalizes large errors more than MASE (due to squaring)
- Used as the official accuracy metric in the M5 competition (2020)
- Well-suited for intermittent/sparse demand data

**When to use:** Contexts where large errors are disproportionately costly AND cross-series comparison is needed. The MASE vs. RMSSE choice mirrors the MAE vs. RMSE decision.

### Relative Metrics

Compare model performance directly against a benchmark method.

#### Relative MAE (rMAE)

**Formula:**

```
rMAE = MAE_model / MAE_benchmark
```

**Properties:**
- Direct comparison to any chosen benchmark (naive, seasonal naive, or another model)
- rMAE < 1: model outperforms benchmark
- rMAE > 1: model underperforms benchmark
- Can use geometric mean when aggregating across series (avoids distortion by outliers)

**Relationship to FVA:**

```
FVA = (1 - rMAE) × 100%
```

#### Overall Weighted Average (OWA)

**Formula (M4 Competition):**

```
OWA = 0.5 × (sMAPE_model / sMAPE_naive2) + 0.5 × (MASE_model / MASE_naive2)
```

where Naive2 is the seasonal naive benchmark.

**Properties:**
- Composite metric combining percentage and scaled components
- Introduced in M4 to reduce single-metric bias
- OWA < 1: model beats the Naive2 benchmark overall

**Limitations:**
- Still includes sMAPE, inheriting its asymmetry bias
- Arbitrary 50/50 weighting of the two components
- A step forward from pure MAPE, but still imperfect

### Probabilistic Forecast Metrics

Point forecasts are insufficient for decision-making. Probabilistic metrics evaluate the full predictive distribution.

#### Continuous Ranked Probability Score (CRPS)

**Formula:**

```
CRPS = ∫_{-∞}^{∞} [F(x) - 𝟙(x ≥ y)]² dx
```

where F(x) is the forecast CDF and y is the observed value.

**Properties:**
- Generalizes MAE to probabilistic forecasts: CRPS reduces to MAE for deterministic (point) forecasts
- Expressed in the same units as the observed variable
- Evaluates both sharpness (narrowness of the distribution) and calibration (accuracy of probability levels)
- Proper scoring rule: minimized when the forecast distribution equals the true data-generating distribution
- Smaller CRPS = better probabilistic forecast

**Interpretation:** CRPS measures the integrated squared difference between the forecast cumulative distribution and the step-function CDF of the observation. A forecast that concentrates probability mass near the true value achieves low CRPS.

**Implementation:**

```python
import properscoring as ps
import numpy as np

# From ensemble predictions
crps_values = ps.crps_ensemble(y_true, ensemble_predictions)
mean_crps = np.mean(crps_values)

# From quantile predictions (via numerical integration)
from skforecast.metrics import crps_from_quantiles
crps_values = crps_from_quantiles(y_true, quantile_predictions, quantile_levels)
```

**When to use:** The gold standard for evaluating probabilistic forecasts. Should be reported alongside coverage and interval width for any system producing prediction intervals or distributional forecasts.

#### Pinball Loss (Quantile Loss)

**Formula:**

```
L_τ(y, q̂_τ) = {
    τ × (y - q̂_τ)       if y ≥ q̂_τ
    (1-τ) × (q̂_τ - y)   if y < q̂_τ
}
```

where τ is the target quantile level and q̂_τ is the predicted quantile.

**Properties:**
- Asymmetric loss that penalizes errors differently depending on which side of the quantile the observation falls
- For τ = 0.5, reduces to half the absolute error (equivalent to optimizing for the median)
- For τ = 0.9, under-predictions are penalized 9× more than over-predictions
- Proper scoring rule for quantile forecasts
- The sum of pinball losses across multiple quantiles approximates CRPS

**Weighted Quantile Loss (WQL):**

```
WQL = Σ_τ Σ_i L_τ(y_i, q̂_{τ,i}) / Σ_i |y_i|
```

Scale-normalized version used in production forecasting systems (e.g., Amazon's AutoGluon-TimeSeries).

**When to use:** Evaluating quantile regression models, conformal quantile predictions, and any forecast system producing specific quantile estimates.

#### Winkler Score

**Formula:**

For prediction interval [L, U] at confidence level (1-α):

```
W = {
    (U - L)                                    if L ≤ y ≤ U
    (U - L) + (2/α)(L - y)                    if y < L
    (U - L) + (2/α)(y - U)                    if y > U
}
```

**Properties:**
- Rewards narrow intervals (smaller U - L)
- Penalizes intervals that miss the true value, with penalty proportional to the miss distance
- The penalty factor (2/α) increases for higher confidence levels, making misses at 99% confidence far more costly than at 80%
- Proper scoring rule for interval forecasts

**When to use:** Evaluating prediction intervals directly, comparing interval forecasting methods.

### Bias Metrics

Forecast bias is often more operationally damaging than forecast error magnitude.

#### Mean Error (ME) / Forecast Bias

**Formula:**

```
ME = (1/n) Σ (ŷ_i - y_i)
```

**Properties:**
- Positive ME: systematic over-forecasting
- Negative ME: systematic under-forecasting
- ME ≈ 0: unbiased (but could still have large errors)

**Why it matters:** Bias compounds over time and across products. A model with moderate MAE but zero bias is often preferable to a model with low MAE but persistent bias, because bias creates systematic inventory imbalance, capacity misallocation, or financial misestimation.

**MAPE's hidden bias effect (Manokhin, 2025):** Optimizing for MAPE introduces systematic negative bias (under-forecasting), because MAPE penalizes over-forecasts more than under-forecasts. In retail, this directly causes stockouts. Always check ME alongside accuracy metrics.

#### Percentage Bias

**Formula:**

```
PBias = Σ (ŷ_i - y_i) / Σ y_i × 100
```

Useful for aggregate bias assessment across a product portfolio.

### Metric Selection Decision Framework

**Step 1: Single series or multi-series?**

| Context | Recommended Metrics |
|---------|-------------------|
| Single series evaluation | MAE, RMSE, ME (bias check) |
| Multi-series comparison (equal importance) | MASE, RMSSE |
| Multi-series comparison (volume-weighted) | WAPE, weighted MAE |
| Competition / benchmarking | MASE, RMSSE, OWA |

**Step 2: Point forecast or probabilistic?**

| Forecast Type | Recommended Metrics |
|---------------|-------------------|
| Point forecast only | MAE/RMSE + MASE/RMSSE + ME |
| Quantile forecasts | Pinball Loss, WQL |
| Prediction intervals | Coverage, Average Width, Winkler Score |
| Full distributional | CRPS |

**Step 3: What business behavior should the metric incentivize?**

| Business Priority | Metric to Use | Metric to Avoid |
|-------------------|---------------|-----------------|
| Minimize total units of error | MAE, WAPE | MAPE |
| Penalize large misses severely | RMSE, RMSSE | MAE |
| Avoid systematic over/under | ME + MAE | MAPE (introduces bias) |
| Compare to naive baseline | MASE, FVA | Raw MAE (no context) |
| Minimize stockout risk | Asymmetric loss, high-quantile pinball | Symmetric metrics |
| Minimize excess inventory | Asymmetric loss, low-quantile pinball | Symmetric metrics |

**Step 4: Always report multiple metrics.** Minimum recommended set:

1. **MAE or RMSE** (absolute error magnitude)
2. **MASE or RMSSE** (performance relative to naive)
3. **ME** (bias detection)
4. **Coverage + CRPS** (if probabilistic forecasts produced)

**Metrics to avoid in almost all contexts:** MAPE, sMAPE. These are legacy metrics with well-documented biases that distort model selection. They persist primarily due to inertia.

### Implementation: Complete Metric Suite

```python
import numpy as np
from scipy import stats

class ForecastMetrics:
    """Comprehensive forecast evaluation following Manokhin's framework."""

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def me(y_true, y_pred):
        """Mean Error (bias). Positive = over-forecast, Negative = under-forecast."""
        return np.mean(y_pred - y_true)

    @staticmethod
    def mase(y_true, y_pred, y_train, seasonal_period=1):
        """Mean Absolute Scaled Error (Hyndman & Koehler, 2006)."""
        n = len(y_train)
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
        scale = np.mean(naive_errors)
        if scale == 0:
            return np.inf
        return np.mean(np.abs(y_true - y_pred)) / scale

    @staticmethod
    def rmsse(y_true, y_pred, y_train, seasonal_period=1):
        """Root Mean Squared Scaled Error (M5 competition metric)."""
        n = len(y_train)
        naive_errors_sq = (y_train[seasonal_period:] - y_train[:-seasonal_period]) ** 2
        scale = np.mean(naive_errors_sq)
        if scale == 0:
            return np.inf
        return np.sqrt(np.mean((y_true - y_pred) ** 2) / scale)

    @staticmethod
    def wape(y_true, y_pred):
        """Weighted Absolute Percentage Error."""
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    @staticmethod
    def fva(y_true, y_pred_model, y_pred_naive):
        """Forecast Value Added (percentage improvement over naive)."""
        mae_model = np.mean(np.abs(y_true - y_pred_model))
        mae_naive = np.mean(np.abs(y_true - y_pred_naive))
        return (mae_naive - mae_model) / mae_naive * 100

    @staticmethod
    def coverage(y_true, lower, upper):
        """Prediction interval coverage."""
        return np.mean((y_true >= lower) & (y_true <= upper))

    @staticmethod
    def avg_width(lower, upper):
        """Average prediction interval width."""
        return np.mean(upper - lower)

    @staticmethod
    def pinball_loss(y_true, y_pred_quantile, tau):
        """Pinball (quantile) loss for a single quantile level tau."""
        residuals = y_true - y_pred_quantile
        return np.mean(np.where(residuals >= 0, tau * residuals, (tau - 1) * residuals))

    @staticmethod
    def winkler_score(y_true, lower, upper, alpha):
        """Winkler score for prediction intervals."""
        width = upper - lower
        penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
        penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
        return np.mean(width + penalty_lower + penalty_upper)

    @classmethod
    def full_report(cls, y_true, y_pred, y_train, seasonal_period=1,
                    y_pred_naive=None, lower=None, upper=None, alpha=0.05):
        """Generate complete metric report."""
        report = {
            'MAE': cls.mae(y_true, y_pred),
            'RMSE': cls.rmse(y_true, y_pred),
            'ME (Bias)': cls.me(y_true, y_pred),
            'MASE': cls.mase(y_true, y_pred, y_train, seasonal_period),
            'RMSSE': cls.rmsse(y_true, y_pred, y_train, seasonal_period),
            'WAPE (%)': cls.wape(y_true, y_pred),
        }

        if y_pred_naive is not None:
            report['FVA (%)'] = cls.fva(y_true, y_pred, y_pred_naive)

        if lower is not None and upper is not None:
            report['Coverage'] = cls.coverage(y_true, lower, upper)
            report['Avg Width'] = cls.avg_width(lower, upper)
            report['Winkler Score'] = cls.winkler_score(y_true, lower, upper, alpha)

        return report
```

---

## 12. Advanced Topics

### Adaptive Conformal Inference

**Problem:** Standard conformal prediction assumes constant error distribution over time.

**Solution:** Update prediction intervals dynamically as new data arrives.

**Algorithm (ACI):**

```
1. Initialize: Set γ (learning rate), ε (target miscoverage)

2. For each time step t:
   a. Compute prediction interval using current α_t
   b. Observe true value y_t
   c. Update miscoverage:
      err_t = 1(y_t ∉ PI_t)
   d. Update α:
      α_{t+1} = α_t + γ(ε - err_t)
```

**Advantages:**
- Adapts to distribution shift
- Maintains coverage under non-stationarity
- Low computational overhead

### Multi-Horizon Forecasting

**Challenge:** Uncertainty increases with forecast horizon.

**Approaches:**

**1. Separate Models per Horizon**
- Train distinct conformal predictors for each horizon
- Most accurate but computationally expensive

**2. Recursive Forecasting**
- Use 1-step-ahead model recursively
- Feed predictions back as inputs
- Error accumulation concern

**3. Direct Multi-Step**
- Train model to predict multiple horizons simultaneously
- Single conformal calibration across all horizons
- Simpler but may sacrifice accuracy

### Handling Distribution Shift

**Types of Shift:**

1. **Covariate Shift**: P(X) changes, P(Y|X) constant
2. **Label Shift**: P(Y) changes, P(X|Y) constant
3. **Concept Drift**: P(Y|X) changes

**Robust Conformal Strategies:**

**1. Weighted Conformal Prediction**
```python
# Compute importance weights for calibration points
weights = p_test(X_calib) / p_train(X_calib)

# Weighted quantile computation
weighted_quantile = weighted_quantile(calibration_scores, weights, q)
```

**2. Sliding Window Calibration**
```python
# Use only recent N calibration points
recent_scores = calibration_scores[-N:]
threshold = np.quantile(recent_scores, q_level)
```

**3. Ensemble of Conformal Predictors**
```python
# Train multiple conformal predictors on different time windows
# Aggregate their intervals
```

### Hierarchical Time Series

**Structure:** Forecasts must satisfy aggregation constraints.

**Example:** Total sales = Region A + Region B + Region C

**Conformal Reconciliation:**

1. Generate independent conformal intervals for each hierarchy level
2. Reconcile using optimal reconciliation (MinT, WLS)
3. Ensure aggregation consistency maintained

**Challenge:** Maintaining both validity and coherence simultaneously.

### Venn-ABERS Calibration

**Purpose:** Improve probability calibration for classification.

**Advantage over Platt Scaling / Isotonic Regression:**
- Provides validity guarantees
- Non-parametric
- Works with small calibration sets

**Algorithm:**

1. Train binary classifier on training set
2. On calibration set, fit two isotonic regression models:
   - One for class 0 examples
   - One for class 1 examples
3. For new test point:
   - Get probability from both isotonic models
   - Return calibrated probability pair [p₀, p₁]
   - p₀ + p₁ may not equal 1 (indicates uncertainty)

**Properties:**
- Valid calibration guarantee
- Adaptive to local regions
- Indicates prediction uncertainty through interval width

### Transformer Critique

**Manokhin's Technical Argument Against Transformers for Time Series:**

**Core Problem:** Permutation-invariant self-attention

Time series are fundamentally sequential—order carries meaning. Self-attention mechanism treats positions as permutation-invariant set:

```
Attention(Q, K, V) = softmax(QKᵀ/√d)V
```

This ignores temporal structure critical to forecasting.

**Additional Issues:**

1. **Indistinguishable temporal attention:** Different series generate similar attention patterns
2. **Error accumulation:** Autoregressive generation compounds errors
3. **Over-stationarisation:** Transformers smooth away important volatility
4. **Cannot approximate smooth functions:** Theoretical limitation
5. **Curse of Attention:** Poor generalization on time series (kernel-based perspective)

**Empirical Evidence:**

- **Chronos (Amazon):** 10% less accurate, 500% slower than statistical ensembles
- **Moirai (Salesforce):** Up to 33% less accurate than statistical models
- **Lag-Llama:** 42% less accurate, 1000× slower than seasonal naive

**Recommendation:** Use transformers as baselines, not defaults. For time series, prioritize:
1. Classical statistical methods (ARIMA, ETS)
2. CatBoost with engineered features
3. Specialized architectures (N-BEATS, N-HiTS, TSMixer)
4. Ensemble methods

### Prophet Limitations

**Structural Deficiencies:**

1. **No autoregression:** Ignores autocorrelation fundamental to time series
2. **No heteroscedasticity handling:** Assumes constant variance
3. **Additive assumptions only:** Real phenomena often multiplicative
4. **Poor uncertainty quantification:** 30-40% of values outside claimed intervals

**Empirical Failures:**

- Failed to outperform simple methods (linear regression, Lasso, KNN) on standard benchmarks
- Failed on ALL point metrics (MAE, RMSE, MAPE, sMAPE)
- Implicated in Zillow forecasting failure ($50B market value loss)

**When Prophet May Work:**

- Strong, regular seasonality
- Long history available
- Many missing values/outliers
- Need for automatic seasonality detection
- Non-technical stakeholder interpretability priority

**Better Alternatives:**

- statsforecast (Nixtla): Statistical models at scale
- NeuralProphet: Prophet successor with autoregression + conformal prediction
- Classical ARIMA/ETS: Better uncertainty quantification

---

## Conclusion: The 95/5 Rule

**Models are 5% of forecasting. The other 95%:**

1. **Forecastability assessment** (permutation entropy, naive benchmarks, FVA)
2. **Valid uncertainty quantification** (conformal prediction)
3. **Proper validation** (time series cross-validation)
4. **Metric selection** (MAE/RMSE + MASE/RMSSE + ME + CRPS; avoid MAPE/sMAPE)
5. **Deployment monitoring** (coverage tracking, recalibration)
6. **Failure mode analysis** (when/why predictions fail)
7. **Stakeholder communication** (translating intervals to decisions)

**Implementation Checklist:**

- Assess forecastability before modeling (PE, naive benchmarks)
- Establish naive baselines (random walk, seasonal naive)
- If naive baselines perform well, consider using them
- Fit classical models first (ETS, ARIMA) — they are hard to beat
- Use CatBoost when nonlinear dynamics or rich exogenous features are present
- Use time series cross-validation, not random split
- Evaluate with multiple metrics (MAE + MASE + ME + CRPS); never rely on MAPE alone
- Add conformal prediction layer for valid intervals
- Monitor coverage in production, recalibrate if needed
- Document when/why model should not be used
- Maintain 5% model complexity, 95% rigorous process

**Final Principle:**

> "Forecastability should refer to the range of forecast errors achievable in the long run, not just the stability of the history."

Stop classifying series as "unforecastable" based on CoV thresholds. Start measuring forecastability through actual forecast error benchmarks against naive models—because stability ≠ predictability.

---

## References

### Academic Papers

1. **Vovk, V., Gammerman, A., & Shafer, G. (2005).** Algorithmic Learning in a Random World. Springer.

2. **Xu, C., & Xie, Y. (2021).** Conformal Prediction Intervals for Dynamic Time-Series. ICML 2021.

3. **Romano, Y., Patterson, E., & Candès, E. (2019).** Conformalized Quantile Regression. NeurIPS 2019.

4. **Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021).** Predictive Inference with the Jackknife+. Annals of Statistics.

5. **Angelopoulos, A. N., & Bates, S. (2023).** A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.

6. **Hyndman, R. J., & Athanasopoulos, G. (2021).** Forecasting: Principles and Practice, 3rd edition. OTexts.

7. **Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).** Time Series Analysis: Forecasting and Control, 5th edition. Wiley.

8. **Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).** CatBoost: unbiased boosting with categorical features. NeurIPS 2018.

9. **Hyndman, R. J., & Koehler, A. B. (2006).** Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688.

10. **Goodwin, P., & Lawton, R. (1999).** On the asymmetry of the symmetric MAPE. International Journal of Forecasting, 15(4), 405–408.

11. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020).** The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1), 54–74.

12. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022).** M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4), 1346–1364.

13. **Armstrong, J. S., & Collopy, F. (1992).** Error measures for generalizing about forecasting methods: Empirical comparisons. International Journal of Forecasting, 8(1), 69–80.

14. **Manokhin, V. (2025).** MAPE and the Makridakis Competitions: Why Metrics Matter. Medium.

15. **Manokhin, V. (2025).** The Makridakis Forecasting Competitions: Four Decades of Overhyped Simplicity and Stagnation. Medium.

16. **Manokhin, V. (2025).** The Coefficient of Variation Has Outlived Its Usefulness: Time for Retirement. Medium.

### Software Libraries

- **MAPIE** (Python): Model Agnostic Prediction Interval Estimator
- **Amazon Fortuna** (Python): Uncertainty quantification toolkit
- **NeuralProphet** (Python): Neural network time series framework
- **Nonconformist** (Python): Conformal prediction implementation
- **PUNCC** (Python): Predictive Uncertainty Calibration and Conformalization
- **pmdarima** (Python): Auto-ARIMA implementation
- **statsmodels** (Python): SARIMAX, ETS, and statistical tests
- **statsforecast** (Nixtla): AutoARIMA, AutoETS at scale
- **CatBoost** (Python): Gradient boosting with ordered boosting
- **properscoring** (Python): CRPS and other probabilistic scoring rules
- **skforecast** (Python): Forecasting with scikit-learn compatible models, includes CRPS utilities

### Additional Resources

- **Awesome Conformal Prediction:** github.com/valeman/awesome-conformal-prediction
- **Manokhin's Medium:** valeman.medium.com
- **Book:** "Practical Guide to Applied Conformal Prediction in Python" by Valeriy Manokhin (Packt, 2023)
- **Hyndman FPP3:** otexts.com/fpp3 (free online textbook for ARIMA/ETS)
- **CatBoost Documentation:** catboost.ai/docs

---

*Document Version: 3.0*
*Last Updated: 2025*
*Author: Synthesized from Valeriy Manokhin's research and writings*