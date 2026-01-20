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

### 3.1 Key Definitions

**Sorted probabilities:**

Let $\hat{\pi}_{(1)}(x) \geq \hat{\pi}_{(2)}(x) \geq \ldots \geq \hat{\pi}_{(C)}(x)$ be the class probabilities sorted in descending order.

**Conformity score $E$ (Adaptive Prediction Sets):**

For a calibration point $(x, y)$ where $y$ has rank $k$ among the sorted probabilities (i.e., $y$ is the $k$-th most likely class):

$$E(x, y, u; \hat{\pi}) = \sum_{j=1}^{k-1} \hat{\pi}_{(j)}(x) + u \cdot \hat{\pi}_{(k)}(x)$$

where $u \sim \text{Uniform}(0,1)$ provides randomization for tighter coverage.

**Intuition:**
- The score measures how much cumulative probability mass you need to accumulate (going from most likely to least likely) before including the true label
- Small score → true label was among the top predictions → model was "confident and correct"
- Large score → true label was far down the ranking → model was "wrong"
- The randomization term $u \cdot \hat{\pi}_{(k)}(x)$ breaks ties and enables exact coverage

**Example:**

If $\hat{\pi}(x) = [0.1, 0.6, 0.3]$ for classes $\{1, 2, 3\}$ and true label $y = 3$:
- Sorted: $\hat{\pi}_{(1)} = 0.6$ (class 2), $\hat{\pi}_{(2)} = 0.3$ (class 3), $\hat{\pi}_{(3)} = 0.1$ (class 1)
- True label $y=3$ has rank $k=2$
- Score: $E = \hat{\pi}_{(1)} + u \cdot \hat{\pi}_{(2)} = 0.6 + u \cdot 0.3$
- If $u = 0.5$: $E = 0.6 + 0.15 = 0.75$

---

## 4. Implementation Options

### Option A: Split Conformal (SC) — Fastest

```
1. Split data: I₁ (training), I₂ (calibration), equal sizes
2. Train classifier on I₁: π̂ ← B({(Xᵢ, Yᵢ)}ᵢ∈I₁)
3. For each i ∈ I₂:
   - Sample Uᵢ ~ Uniform(0,1)
   - Compute Eᵢ = E(Xᵢ, Yᵢ, Uᵢ; π̂)
4. Compute threshold: τ̂ = the ⌈(1-α)(|I₂|+1)⌉-th smallest value in {Eᵢ}
   Example: n=1000 calibration points, α=0.1 → τ̂ = 901st smallest score
5. For new point X_{n+1}:
   - Sample U_{n+1} ~ Uniform(0,1)
   - Include label y in prediction set if its score ≤ τ̂
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

**Coverage guarantee:** Theoretical lower bound is $1 - 2\alpha$, but empirically CV+ typically achieves $1-\alpha$ coverage. The $1-2\alpha$ is a worst-case safety bound; in practice, use target $\alpha$ directly.

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
def compute_E(pi_hat_x, y, u):
    """
    Compute APS conformity score for a single calibration point.
    
    Args:
        pi_hat_x: array of probabilities for all classes for input x
        y: the ground truth label index
        u: a random value from Uniform(0,1)
    
    Returns:
        score: the conformity score E
    """
    # 1. Sort probabilities in descending order, tracking original indices
    indexed_probs = list(enumerate(pi_hat_x))
    sorted_probs = sorted(indexed_probs, key=lambda p: p[1], reverse=True)
    
    # 2. Accumulate probability until we reach the true label
    cumsum = 0.0
    for label_idx, prob in sorted_probs:
        if label_idx == y:
            # True label found at this rank
            # Score = (sum of probs more likely than y) + u * (prob of y)
            return cumsum + u * prob
        cumsum += prob
    
    return 1.0  # Should not reach here if y is valid
```

### Phase 4: Calibration
```python
def calibrate(calibration_scores, alpha):
    """
    Compute the threshold τ̂ from calibration scores.
    
    Args:
        calibration_scores: 1D array/list of E_i scores from calibration set
                           (one score per calibration point)
        alpha: miscoverage level (e.g., 0.1 for 90% coverage)
    
    Returns:
        tau_hat: the calibrated threshold
    """
    n = len(calibration_scores)
    # Compute the (1-α) quantile with finite-sample correction
    q = np.ceil((1 - alpha) * (n + 1)) / n
    tau_hat = np.quantile(calibration_scores, q, method='higher')
    return tau_hat
```

### Phase 5: Prediction Set Construction
```python
def predict_set(pi_hat_x_new, tau_hat, u):
    """
    Construct prediction set for a new point.
    
    Args:
        pi_hat_x_new: array of probabilities for all classes for new input
        tau_hat: calibrated threshold from Phase 4
        u: a random value from Uniform(0,1)
           IMPORTANT: Draw u ONCE per test point and reuse for all candidate labels
    
    Returns:
        pred_set: list of label indices in the prediction set
    """
    indexed_probs = list(enumerate(pi_hat_x_new))
    # Note: Python's sorted() is stable; ties are broken by original index order.
    # This is acceptable for practical use. For strict theoretical compliance,
    # ties could be broken randomly.
    sorted_probs = sorted(indexed_probs, key=lambda p: p[1], reverse=True)
    
    cumsum = 0.0
    pred_set = []
    
    for label_idx, prob in sorted_probs:
        # Compute the score this label would have if it were the true label
        candidate_score = cumsum + u * prob
        
        if candidate_score <= tau_hat:
            # Include this label in the prediction set
            pred_set.append(label_idx)
            cumsum += prob
        else:
            # Once a label's score exceeds tau_hat, stop
            # (all remaining labels are less likely and will also exceed)
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