# Conformal Prediction Guide: Adaptive Prediction Sets (APS)

**Based on:** Romano, Y., Sesia, M., & Candès, E. J. (2020). Classification with valid and adaptive coverage. *NeurIPS 33*, 3581–3591.

---

## 1. The Problem Conformal Prediction Solves

### 1.1 The Core Issue: Uncertainty Quantification

Standard machine learning models output point predictions:
- **Classification:** "This image is a cat" (or softmax probabilities like 0.85 for cat)
- **Regression:** "The house price is $450,000"

**The problem:** These outputs don't come with valid, rigorous uncertainty guarantees.

- A softmax probability of 0.85 does **not** mean "85% chance of being correct" — neural networks are notoriously miscalibrated
- A point estimate of $450K tells you nothing about whether the true value could be $400K or $500K
- Model confidence ≠ actual accuracy

### 1.2 Why Point Estimates and Softmax Probabilities Are Risky

**Point estimates hide uncertainty entirely:**

- A prediction of $\hat{y} = 450K$ could come from a model that's highly certain (true value likely between $445K-455K) or highly uncertain (true value could be $300K-600K)
- Decision-makers using point estimates have no way to assess risk
- High-stakes domains (medicine, finance, autonomous systems) require knowing *how wrong* a prediction could be

**Softmax probabilities are not calibrated probabilities:**

Neural network softmax outputs are often interpreted as confidence, but:

| Softmax Output | What People Think | Reality |
|----------------|-------------------|---------|
| $P(\text{cat}) = 0.95$ | "95% chance it's a cat" | Could be correct only 70% of the time |
| $P(\text{cat}) = 0.60$ | "Uncertain, maybe 60% cat" | Could be correct 90% of the time |

**Empirical evidence of miscalibration:**

- Guo et al. (2017) showed modern neural networks are *overconfident* — they output high probabilities even when wrong
- A model saying "95% confident" might only be right 80% of the time
- Calibration error increases with model depth and capacity

**Concrete risks:**

| Domain | Risk from Uncalibrated Predictions |
|--------|-----------------------------------|
| Medical diagnosis | Model says 99% cancer → patient undergoes unnecessary surgery; actual probability was 60% |
| Autonomous driving | Model says 98% "no pedestrian" → vehicle doesn't brake; model was overconfident |
| Credit scoring | Model says 95% will repay → bank approves loan; true default rate is 20% |
| Criminal justice | Model says 90% recidivism risk → harsher sentence; model systematically wrong for certain groups |

**The fundamental issue:**

Softmax probabilities optimize for *discrimination* (separating classes), not *calibration* (matching true probabilities). Cross-entropy loss encourages the model to push probabilities toward 0 or 1, regardless of true uncertainty.

**What's needed:**

Not just a prediction, but a *set* or *interval* with a **guarantee**: "The true value is in here with probability ≥ 95%." This is exactly what conformal prediction provides.

### 1.3 What We Want

Given a new input $X_{n+1}$, instead of a single prediction, we want:

| Task | Point Prediction | What We Want Instead |
|------|------------------|---------------------|
| Classification | $\hat{y} = 3$ | Set $\{2, 3, 5\}$ guaranteed to contain true label |
| Regression | $\hat{y} = 450K$ | Interval $[420K, 490K]$ guaranteed to contain true value |

**The guarantee:** With probability at least $1 - \alpha$, the true label/value is in the set/interval.

### 1.4 Why Existing Approaches Fail

| Approach | Problem |
|----------|---------|
| Softmax probabilities | Miscalibrated; don't reflect true uncertainty |
| Bayesian posteriors | Require model assumptions; computationally expensive; guarantees are asymptotic |
| Ensemble disagreement | Heuristic; no formal coverage guarantee |
| Dropout uncertainty | Approximation; no finite-sample guarantee |
| Calibration methods (Platt, temperature scaling) | Improve calibration but still no coverage guarantee |

### 1.5 What Conformal Prediction Provides

**Conformal prediction solves this by providing:**

1. **Distribution-free guarantees** — No assumptions on $P_{XY}$ (no normality, no parametric form)
2. **Finite-sample validity** — Works for any sample size $n$, not just asymptotically
3. **Model-agnostic** — Wraps around any black-box predictor (neural nets, random forests, etc.)
4. **Rigorous coverage** — Mathematical guarantee: $P[Y_{n+1} \in \hat{C}(X_{n+1})] \geq 1 - \alpha$

**The only assumption:** Data points are *exchangeable* (i.i.d. is sufficient but not necessary).

### 1.6 The Trade-off

Conformal prediction guarantees **marginal** coverage (averaged over the distribution), not **conditional** coverage (for each specific $x$). The APS method in this guide specifically addresses this limitation by producing prediction sets that *adapt* to local difficulty — smaller sets where the model is confident, larger sets where it's uncertain — while maintaining the marginal guarantee.

---

## 2. Problem Setup (Technical)

**Inputs:**
- Training data: $\{(X_i, Y_i)\}_{i=1}^{n}$ with features $X_i \in \mathbb{R}^p$ and discrete labels $Y_i \in \mathcal{Y} = \{1, 2, \ldots, C\}$
- New test point: $X_{n+1}$
- Target coverage level: $1 - \alpha$ (e.g., 0.90)
- Any black-box classifier $\mathcal{B}$ that outputs probability estimates $\hat{\pi}_y(x)$

**Output:**
- Prediction set $\hat{C}(X_{n+1}) \subseteq \mathcal{Y}$ such that $P[Y_{n+1} \in \hat{C}(X_{n+1})] \geq 1 - \alpha$

---

## 3. Core Method: Generalized Inverse Quantile Conformity Score

## 3.1 Key Definitions

**Generalized conditional quantile function:**
$$L(x; \pi, \tau) = \min\{c \in \{1, \ldots, C\} : \pi_{(1)}(x) + \pi_{(2)}(x) + \ldots + \pi_{(c)}(x) \geq \tau\}$$

where $\pi_{(1)}(x) \geq \pi_{(2)}(x) \geq \ldots \geq \pi_{(C)}(x)$ are ordered class probabilities.

**Set function $S$:**
$$S(x, u; \pi, \tau) = \begin{cases} 
\text{indices of } L-1 \text{ largest } \pi_y(x), & \text{if } u \leq V(x; \pi, \tau) \\
\text{indices of } L \text{ largest } \pi_y(x), & \text{otherwise}
\end{cases}$$

where $V(x; \pi, \tau) = \frac{1}{\pi_{(L)}(x)}\left[\sum_{c=1}^{L} \pi_{(c)}(x) - \tau\right]$ and $u \sim \text{Uniform}(0,1)$.

**Conformity score $E$ (Generalized Inverse Quantile):**
$$E(x, y, u; \hat{\pi}) = \min\{\tau \in [0,1] : y \in S(x, u; \hat{\pi}, \tau)\}$$

This score measures the smallest threshold $\tau$ needed for the prediction set to include the true label.

---

## 4. Implementation Options

### Option A: Split Conformal (SC) — Fastest

```
1. Split data: I₁ (training), I₂ (calibration), equal sizes
2. Train classifier on I₁: π̂ ← B({(Xᵢ, Yᵢ)}ᵢ∈I₁)
3. For each i ∈ I₂:
   - Sample Uᵢ ~ Uniform(0,1)
   - Compute Eᵢ = E(Xᵢ, Yᵢ, Uᵢ; π̂)
4. Compute threshold: τ̂ = Q̂₁₋α({Eᵢ}ᵢ∈I₂)
   (the ⌈(1-α)(1+|I₂|)⌉-th largest value)
5. For new point X_{n+1}:
   - Sample U_{n+1} ~ Uniform(0,1)
   - Return Ĉ(X_{n+1}) = S(X_{n+1}, U_{n+1}; π̂, τ̂)
```

**Coverage guarantee:** $P[Y_{n+1} \in \hat{C}] \geq 1 - \alpha$

### Option B: Cross-Validation+ (CV+) — Better Efficiency

```
1. Split data into K folds: I₁, ..., Iₖ
2. For k = 1 to K:
   - Train π̂ᵏ on all data except Iₖ
3. For each y ∈ Y, include y in prediction set if:
   Σᵢ 𝟙[E(Xᵢ, Yᵢ, Uᵢ; π̂^{k(i)}) < E(X_{n+1}, y, U_{n+1}; π̂^{k(i)})] < (1-α)(n+1)
```

**Coverage guarantee:** $P[Y_{n+1} \in \hat{C}] \geq 1 - 2\alpha$ (use $\alpha/2$ for $1-\alpha$ coverage)

### Option C: Jackknife+ (JK+) — Most Powerful, Slowest

Same as CV+ with K = n (leave-one-out).

---

## 5. Step-by-Step Implementation Plan

### Phase 1: Data Preparation
1. Load and preprocess data
2. Define label space $\mathcal{Y}$
3. Choose calibration method (SC/CV+/JK+) based on:
   - SC: Large datasets, computational constraints
   - CV+/JK+: Small datasets, need tighter sets

### Phase 2: Base Classifier Training
1. Select black-box classifier (Neural Network, Random Forest, SVM, etc.)
2. Train on appropriate data split
3. Extract probability estimates $\hat{\pi}_y(x)$ for all classes
4. Ensure outputs are standardized: $\sum_y \hat{\pi}_y(x) = 1$

### Phase 3: Conformity Score Computation
```python
def compute_E(x, y, u, pi_hat):
    """Compute generalized inverse quantile conformity score."""
    # Sort probabilities in descending order
    sorted_probs = sorted(enumerate(pi_hat(x)), key=lambda p: -p[1])
    
    cumsum = 0
    for idx, (label, prob) in enumerate(sorted_probs):
        cumsum += prob
        if label == y:
            # Compute V for randomization
            L = idx + 1
            V = (cumsum - (1 - tau)) / prob if prob > 0 else 0
            if u <= V:
                return cumsum - prob  # Exclude this label
            else:
                return cumsum  # Include this label
    return 1.0
```

### Phase 4: Calibration
1. Compute scores $E_i$ for all calibration points
2. Find quantile threshold $\hat{\tau}$

### Phase 5: Prediction Set Construction
```python
def predict_set(x_new, pi_hat, tau_hat, u):
    """Construct prediction set for new point."""
    sorted_probs = sorted(enumerate(pi_hat(x_new)), key=lambda p: -p[1])
    
    cumsum = 0
    pred_set = []
    for label, prob in sorted_probs:
        cumsum += prob
        V = (cumsum - tau_hat) / prob if prob > 0 else 0
        if cumsum > tau_hat or (cumsum == tau_hat and u > V):
            pred_set.append(label)
        if cumsum >= tau_hat:
            break
    return pred_set
```

---

## 6. Interpreting Prediction Sets and Intervals

### 6.1 Terminology

These are *prediction* sets/intervals, not confidence intervals. Confidence intervals target parameters; prediction intervals target future observations.

### 6.2 Classification: Prediction Sets

**What you get:** A set $\hat{C}(X_{n+1}) \subseteq \{1, \ldots, C\}$

**Interpretation:**

The guarantee $P[Y_{n+1} \in \hat{C}(X_{n+1})] \geq 1 - \alpha$ means:

- If you repeat the entire procedure (draw training data, calibration data, test point) many times, at least $(1-\alpha) \times 100\%$ of the time the true label will be in the set
- This is a **marginal** guarantee — averaged over all possible $(X, Y)$ pairs from the distribution

**What the set size tells you:**

| Set Size | Interpretation |
|----------|----------------|
| Small (e.g., $\{3\}$) | Model is confident, "easy" region of feature space |
| Large (e.g., $\{1, 3, 5, 7\}$) | Model is uncertain, ambiguous region |
| Full set $\mathcal{Y}$ | Highly uncertain or out-of-distribution |

**Marginal vs Conditional Coverage:**

| Type | Statement | Achievable? |
|------|-----------|-------------|
| Marginal | $P[Y \in \hat{C}(X)] \geq 1-\alpha$ | Yes, always |
| Conditional | $P[Y \in \hat{C}(X) \mid X=x] \geq 1-\alpha$ for all $x$ | No, impossible without assumptions |

The APS method *approximates* conditional coverage but cannot guarantee it. In practice: coverage may be 95% overall but 80% for hard examples and 99% for easy ones.

### 6.3 Regression: Prediction Intervals

**What you get:** An interval $\hat{C}(X_{n+1}) = [\hat{L}(X_{n+1}), \hat{U}(X_{n+1})]$

**Interpretation:**

Same marginal guarantee: $P[Y_{n+1} \in \hat{C}(X_{n+1})] \geq 1 - \alpha$

**What interval width tells you:**
- Narrow interval: Low predictive uncertainty in this region
- Wide interval: High uncertainty (heteroscedasticity, sparse data, etc.)

**Common conformal regression methods:**

| Method | Interval Type | Adapts to Heteroscedasticity? |
|--------|--------------|-------------------------------|
| Split conformal (residuals) | Fixed width | No |
| Conformalized Quantile Regression (CQR) | Variable width | Yes |
| Locally-weighted conformal | Variable width | Yes |

### 6.4 Critical Interpretation Points

1. **Not Bayesian credible intervals** — No posterior distribution is implied. The guarantee is frequentist: coverage holds over repeated sampling.

2. **Distribution-free** — Works for any $P_{XY}$ without parametric assumptions, but only marginal coverage is guaranteed.

3. **Finite-sample valid** — Coverage holds exactly for any $n$, not just asymptotically.

4. **Exchangeability required** — If test distribution differs from calibration distribution (covariate shift, time series), guarantees break.

5. **Set size is not a confidence measure** — A singleton set $\{y\}$ doesn't mean "100% confident it's $y$." It means the model's uncertainty is low *relative to the calibration data*, but the label could still be wrong with probability $\leq \alpha$.

---

## 7. Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Marginal Coverage | $\frac{1}{n_{test}} \sum_i \mathbf{1}[Y_i \in \hat{C}(X_i)]$ | $\geq 1 - \alpha$ |
| Conditional Coverage | $P[Y \in \hat{C}(X) \mid X = x]$ | Approximate $1-\alpha$ for all $x$ |
| Average Set Size | $\mathbb{E}[\|\hat{C}(X)\|]$ | Minimize |
| Size Conditional on Coverage | $\mathbb{E}[\|\hat{C}(X)\| \mid Y \in \hat{C}(X)]$ | Minimize |

---

## 8. Practical Recommendations

1. **Classifier choice:** Neural networks or SVMs typically yield better conditional coverage than Random Forests
2. **Calibration split:** Use at least 500+ calibration points for stable thresholds
3. **Randomization:** Always use randomization (via $U$) for tighter coverage bounds
4. **Label-conditional coverage:** Calibrate $\tau$ separately per class if needed
5. **Validation:** Always verify empirical coverage on held-out test set

---

## 9. Comparison with Alternatives

| Method | Marginal Coverage | Conditional Coverage | Set Size |
|--------|-------------------|---------------------|----------|
| APS (this paper) | Guaranteed | Good approximation | Small |
| Homogeneous Conformal (HCC) | Guaranteed | Poor | Small |
| Conformal Quantile Classification (CQC) | Guaranteed | Mixed | Often large |

---

## 10. Reference Implementation

Official Python package: https://github.com/msesia/arc

Key dependencies:
- NumPy/SciPy for computations
- scikit-learn for base classifiers
- PyTorch/TensorFlow for neural network classifiers