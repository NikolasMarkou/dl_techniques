# Complete Guide to Forecasting Science: Forecastability Assessment and Conformal Prediction

**A Technical Framework for Rigorous Time Series Forecasting**  
*Synthesized from Valeriy Manokhin's Research and "Practical Guide to Applied Conformal Prediction in Python"*

---

## Table of Contents

1. [Fundamental Principles](#fundamental-principles)
2. [Forecastability Assessment Framework](#forecastability-assessment-framework)
3. [Conformal Prediction: Theoretical Foundations](#conformal-prediction-theoretical-foundations)
4. [Nonconformity Measures](#nonconformity-measures)
5. [Time Series Forecasting with Conformal Prediction](#time-series-forecasting-with-conformal-prediction)
6. [Validity and Efficiency](#validity-and-efficiency)
7. [Practical Implementation](#practical-implementation)
8. [Model Selection and Benchmarking](#model-selection-and-benchmarking)
9. [Advanced Topics](#advanced-topics)

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
- Machine learning models
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

## 3. Conformal Prediction: Theoretical Foundations

### Core Principles

Conformal Prediction is a machine learning framework quantifying uncertainty to produce probabilistic predictions with mathematical validity guarantees.

**Eight Foundational Principles:**

1. **Validity**: Prediction regions encompass actual target values with user-specified confidence level (e.g., 95% confidence → 95% coverage)

2. **Efficiency**: Prediction intervals/regions should be as small as possible while preserving desired confidence level

3. **Adaptivity**: Prediction sets adaptive to individual examples—harder-to-predict examples receive wider intervals

4. **Distribution-free**: No assumptions about underlying data distribution required (only exchangeability, less restrictive than IID)

5. **Online adaptivity**: Can adjust to new data points without retraining

6. **Compatibility**: Seamlessly integrates with any existing model (XGBoost, neural networks, random forests, etc.)

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

Exchangeability is weaker than IID, allowing conformal prediction broader applicability. However, time series violate exchangeability—requiring specialized adaptations covered in Section 5.

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

## 4. Nonconformity Measures

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

## 5. Time Series Forecasting with Conformal Prediction

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
- Works with any regression function (boosted trees, neural networks, etc.)

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

## 6. Validity and Efficiency

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

## 7. Practical Implementation

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
from sklearn.ensemble import RandomForestRegressor

# Train model ONLY on proper training set
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
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
                # Nonconformity score for this class
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
        # Train on all data up to i
        train = data[:i]
        test = data[i:i+horizon]
        
        # Fit model
        model = model_class()
        model.fit(train)
        
        # Evaluate
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
        # Train on fixed-size window
        train = data[i:i+train_size]
        test = data[i+train_size:i+train_size+horizon]
        
        # Fit model
        model = model_class()
        model.fit(train)
        
        # Evaluate
        predictions = model.predict(test)
        mae = mean_absolute_error(test, predictions)
        results.append(mae)
    
    return np.mean(results), np.std(results)
```

---

## 8. Model Selection and Benchmarking

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
    # Split data
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
    
    # Add your sophisticated model here
    # model_pred = your_model.predict(horizon)
    # results['Your_Model'] = {...}
    
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

### Critical Evaluation Metrics

**Point Forecast Accuracy:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (Symmetric MAPE)

**Probabilistic Forecast Quality:**
- Coverage Probability (validity)
- Average Interval Width (efficiency)
- Continuous Ranked Probability Score (CRPS)
- Pinball Loss (quantile accuracy)

**Calibration:**
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Reliability Diagrams

---

## 9. Advanced Topics

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
4. **Cannot approximate smooth functions:** Theoretical limitation (see: "Transformers are Secretly Implicit Neural Kernels")
5. **Curse of Attention:** Poor generalization on time series (kernel-based perspective)

**Empirical Evidence:**

- **Chronos (Amazon):** 10% less accurate, 500% slower than statistical ensembles
- **Moirai (Salesforce):** Up to 33% less accurate than statistical models
- **Lag-Llama:** 42% less accurate, 1000× slower than seasonal naive

**Recommendation:** Use transformers as baselines, not defaults. For time series, prioritize:
1. Classical statistical methods (ARIMA, ETS)
2. Tree-based models (LightGBM, XGBoost)
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
4. **Metric selection** (appropriate for business context)
5. **Deployment monitoring** (coverage tracking, recalibration)
6. **Failure mode analysis** (when/why predictions fail)
7. **Stakeholder communication** (translating intervals to decisions)

**Implementation Checklist:**

✓ Assess forecastability before modeling (PE, naive benchmarks)  
✓ Establish naive baselines (random walk, seasonal naive)  
✓ If naive baselines perform well, consider using them  
✓ Use time series cross-validation, not random split  
✓ Evaluate coverage probability ≥ evaluate point accuracy  
✓ Add conformal prediction layer for valid intervals  
✓ Monitor coverage in production, recalibrate if needed  
✓ Document when/why model should not be used  
✓ Maintain 5% model complexity, 95% rigorous process

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

### Software Libraries

- **MAPIE** (Python): Model Agnostic Prediction Interval Estimator
- **Amazon Fortuna** (Python): Uncertainty quantification toolkit
- **NeuralProphet** (Python): Neural network time series framework
- **Nonconformist** (Python): Conformal prediction implementation
- **PUNCC** (Python): Predictive Uncertainty Calibration and Conformalization

### Additional Resources

- **Awesome Conformal Prediction:** github.com/valeman/awesome-conformal-prediction
- **Manokhin's Medium:** valeman.medium.com
- **Book:** "Practical Guide to Applied Conformal Prediction in Python" by Valeriy Manokhin (Packt, 2023)

---

*Document Version: 1.0*  
*Last Updated: 2025*  
*Author: Synthesized from Valeriy Manokhin's research and writings*