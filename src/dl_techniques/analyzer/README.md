# Model Analyzer: Complete Usage Guide

A comprehensive, modular analysis toolkit for deep learning models built on Keras and TensorFlow. This module provides multi-dimensional model analysis including weight distributions, calibration metrics, information flow patterns, training dynamics, and advanced spectral analysis with publication-ready visualizations.

## 1. Overview

The Model Analyzer is designed to provide deep insights into your neural network models beyond simple accuracy metrics. It helps answer critical questions about model behavior, training efficiency, and production readiness. By automating complex analyses and generating intuitive visualizations, it streamlines the process of model selection, debugging, and hyperparameter tuning.

### Key Features

-   **Comprehensive Analysis**: Five specialized analysis modules covering weights, calibration, information flow, training dynamics, and spectral properties.
-   **Advanced Spectral Analysis (WeightWatcher/SETOL)**: Assess training quality, complexity, and generalization potential using power-law and concentration analysis. This implementation is highly optimized for speed ($O(N)$ complexity), handling large layers efficiently.
-   **Rich Visualizations**: Publication-ready plots and summary dashboards with consistent styling and color schemes.
-   **Modular & Extensible**: Each analysis is independent. The architecture is designed for adding custom analyzers and visualizers.
-   **Training & Hyperparameter Insights**: Deep analysis of training history, convergence patterns, and a powerful Pareto-front analysis for optimal model selection.
-   **Serializable Results**: Export all raw metrics to a single JSON file for reproducible analysis, reporting, or further programmatic use.
-   **Robust & Efficient**: Handles large datasets through smart sampling, caches intermediate results to avoid re-computation, and includes robust error handling.

### Module Structure

The toolkit is organized into distinct components for analysis, visualization, and configuration.

```
analyzer/
├── analyzers/                          # Core analysis logic components
│   ├── base.py                         # Abstract base analyzer interface
│   ├── weight_analyzer.py              # Weight distribution and basic health analysis
│   ├── calibration_analyzer.py         # Model confidence and calibration metrics
│   ├── information_flow_analyzer.py    # Activation patterns and information flow
│   ├── training_dynamics_analyzer.py   # Training history and convergence analysis
│   └── spectral_analyzer.py            # Spectral analysis of weights (WeightWatcher)
├── visualizers/                        # Visualization generation components
│   ├── base.py                         # Abstract base visualizer interface
│   ├── weight_visualizer.py            # Weight analysis visualizations
│   ├── calibration_visualizer.py       # Calibration and confidence plots
│   ├── information_flow_visualizer.py  # Information flow visualizations
│   ├── training_dynamics_visualizer.py # Training dynamics plots
│   ├── spectral_visualizer.py          # Spectral analysis visualizations
│   └── summary_visualizer.py           # Unified summary dashboard
├── config.py                           # Configuration classes and plotting setup
├── data_types.py                       # Structured data types (DataInput, AnalysisResults)
├── constants.py                        # Analysis constants and thresholds
├── spectral_metrics.py                 # Core spectral metric calculations
├── spectral_utils.py                   # Utilities for spectral analysis
├── utils.py                            # General utility functions and helpers
├── model_analyzer.py                   # Main coordinator class
├── SETOL.md                            # SETOL theory documentation
└── README.md                           # This file
```

## 2. Installation & Quick Start

### Prerequisites

Ensure you have the required libraries installed.

```bash
pip install keras tensorflow matplotlib seaborn scikit-learn numpy scipy pandas tqdm
```

### 5-Minute Setup

Get comprehensive analysis results for multiple models in just a few lines of code.

```python
import keras
import numpy as np
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput

# 1. Prepare your models (dictionary format with descriptive names)
# The keys ('ResNet_v1', 'ConvNext_v2') will be used as labels in all plots.
models = {
    'ResNet_v1': your_resnet_model,
    'ConvNext_v2': your_convnext_model
}

# 2. Prepare your test data
# This is required for calibration and information flow analyses.
x_test, y_test = np.random.rand(100, 32, 32, 3), np.random.randint(0, 10, 100)
test_data = DataInput(x_data=x_test, y_data=y_test)

# 3. Prepare your training histories (optional, but required for training dynamics)
# This should be a dictionary where keys match the model names.
# The value for each key is the `history` object from a Keras `model.fit()` call.
training_histories = {
    'ResNet_v1': history1, # e.g., result from your_resnet_model.fit(...)
    'ConvNext_v2': history2  # e.g., result from your_convnext_model.fit(...)
}

# 4. Configure and run the analysis
config = AnalysisConfig(
    analyze_training_dynamics=True, # Enable training analysis
    analyze_spectral=True           # Enable spectral analysis
)
analyzer = ModelAnalyzer(
    models=models,
    training_history=training_histories, # Pass the histories to the analyzer
    config=config,
    output_dir='analysis_results'
)

# Run analysis. Note: DataInput is optional if only doing static analysis (weights/spectral)
results = analyzer.analyze(test_data)

print("Analysis complete! Check the 'analysis_results' folder for plots and data.")
```

## 3. Preparing Your Inputs: A Detailed Guide

To get the most out of the Model Analyzer, it's important to provide the input data in the correct format. The analyzer is initialized with three main components: `models`, `training_history`, and `config`. The analysis itself is run with a `DataInput` object.

### 3.1 The `models` Dictionary

This is the primary input and is required. It's a Python dictionary that maps a human-readable string name to a compiled Keras model instance.

-   **Structure**: `Dict[str, keras.Model]`
-   **Keys**: The string keys (e.g., `'ResNet_v1'`) are crucial as they are used to label your models in all generated plots, tables, and the final JSON output. Choose descriptive names.
-   **Values**: The values must be instances of `keras.Model`.

```python
# Example `models` dictionary
models = {
    'MyCNN_v1': cnn_model_v1,
    'MyCNN_v2_with_dropout': cnn_model_v2
}
```

### 3.2 The `DataInput` Object

This object wraps your dataset and is passed to the `analyzer.analyze()` method. It is **required** for any analysis that depends on data, such as **calibration** and **information flow**.

-   **Structure**: A `DataInput` named tuple with two fields: `x_data` and `y_data`.
-   **`x_data`**: A NumPy array or a dictionary of arrays (for multi-input models) containing your input features. The shape should be `(n_samples, ...)`.
-   **`y_data`**: A NumPy array containing the corresponding true labels. The analyzer can handle both integer labels (e.g., shape `(10000,)`) and one-hot encoded labels (e.g., shape `(10000, 10)`).

**Note on Sampling:** If your dataset is large, the analyzer will automatically sample `config.n_samples` (default 1000) for expensive computations like Information Flow analysis to prevent memory issues.

```python
import numpy as np
from dl_techniques.analyzer import DataInput

# Example test data
x_test = np.random.rand(500, 64, 64, 3) # 500 samples of 64x64 color images
y_test = np.random.randint(0, 5, 500) # 500 integer labels for 5 classes

# Create the DataInput object
test_data = DataInput(x_data=x_test, y_data=y_test)
```

### 3.3 The `training_history` Dictionary

This input is **optional** but is **required** to enable the `TrainingDynamicsAnalyzer`. It provides the epoch-by-epoch learning history for each model.

-   **Structure**: `Dict[str, Dict[str, List[float]]]`
-   **Keys**: The keys of this dictionary **must match the keys of your `models` dictionary exactly**. This is how the analyzer associates a history with a model.
-   **Values**: The value for each model is the `history` attribute of the History object returned by a Keras `model.fit()` call. This is a dictionary where keys are metric names (e.g., `'loss'`, `'val_accuracy'`) and values are lists of floats, with one entry per epoch.

**Crucial Note on Metric Names**: The analyzer robustly searches for common metric names (e.g., it will find `accuracy`, `acc`, or `categorical_accuracy`). However, for best results, use standard Keras metric names. The required metrics for full analysis are a training loss, a validation loss, a training accuracy, and a validation accuracy.

Here is a complete example of how to generate and structure the `training_history` object:

```python
# Assume you have two models defined: cnn_model_v1, cnn_model_v2
# And training/validation data: (x_train, y_train), (x_val, y_val)

# 1. Train your models and capture the History object
history_v1 = cnn_model_v1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)
history_v2 = cnn_model_v2.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

# 2. The object we need is the `.history` attribute, which is a dictionary
#    For example, history_v1.history looks like this:
#    {
#        'loss': [1.2, 0.8, 0.6, ...],
#        'accuracy': [0.65, 0.72, 0.78, ...],
#        'val_loss': [1.0, 0.7, 0.5, ...],
#        'val_accuracy': [0.68, 0.75, 0.80, ...]
#    }

# 3. Construct the training_history dictionary for the analyzer
#    The keys here MUST match the keys you will use in your `models` dictionary.
training_histories = {
    'MyCNN_v1': history_v1.history,
    'MyCNN_v2_with_dropout': history_v2.history
}

# 4. Now you can initialize the analyzer
analyzer = ModelAnalyzer(
    models={'MyCNN_v1': cnn_model_v1, 'MyCNN_v2_with_dropout': cnn_model_v2},
    training_history=training_histories
)
```

## 4. Analysis Capabilities

The analyzer computes a wide range of metrics across five key areas of model behavior.

### Weight Analysis Metrics

This analysis inspects the internal parameters of the model to diagnose its structural health and complexity.

| Metric                | Description                                                                                             | Interpretation                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **L1/L2/Spectral Norms** | Mathematical norms that measure the aggregate magnitude of weights in a layer.                          | Higher values indicate larger, more complex weights which can lead to instability or overfitting. Consistently low norms might suggest underfitting. |
| **Weight Distribution** | Statistical properties like mean, standard deviation, skewness, and kurtosis of the weight values.    | A distribution centered near zero with moderate variance is often ideal. High skew or kurtosis can signal issues like dying ReLUS or unstable gradients. |
| **Sparsity**          | The fraction of weights in a layer that are very close to zero.                                         | High sparsity can indicate that many neurons are not contributing to the model's predictions (i.e., they are "dead" or under-utilized).            |
| **Health Score**      | A composite score (0-1) derived from norm, sparsity, and distribution health.                           | A single-glance metric for layer health. Higher scores (closer to 1) indicate a healthier, more balanced weight distribution.                 |

### Spectral Analysis Metrics (WeightWatcher/SETOL)

This analysis examines the eigenvalue spectrum of weight matrices to assess training quality and complexity **without requiring test data**. It is based on the theory of **Heavy-Tailed Self-Regularization (HTSR)** and the **Semi-Empirical Theory of Learning (SETOL)**, which posit that Stochastic Gradient Descent (SGD) implicitly regularizes deep neural networks, causing the distribution of eigenvalues of their weight matrices — the Empirical Spectral Density (ESD) — to develop a characteristic heavy-tailed shape. This shape, quantifiable with a power-law, correlates strongly with the model's ability to generalize.

**Optimization Note**: This implementation uses optimized prefix-sum algorithms to fit power laws in $O(N)$ time, making it significantly faster than standard implementations on large layers (e.g., large transformers or convolutional layers).

#### Theoretical Foundation

For a weight matrix $\mathbf{W}$ of dimensions $M \times N$, the analyzer constructs the correlation matrix $\mathbf{X} = \mathbf{W}^\intercal\mathbf{W}$ and examines the distribution of its eigenvalues $\{\lambda_i\}$. This is computed efficiently via SVD: $\lambda_i = \sigma_i^2$ where $\{\sigma_i\}$ are the singular values of $\mathbf{W}$. The tail of this distribution is then fit to a truncated power-law: $P(\lambda) \sim \lambda^{-\alpha}$.

The complete analysis pipeline:

```
Keras Layer
  -> [type inference & weight extraction]
  -> [tensor matricization to 2D]    (e.g., Conv2D: kh*kw*in_c x out_c)
  -> [SVD]
  -> eigenvalues {lambda_i}
      |
      |-- [tail fitting]       -> alpha, xmin, D, sigma, pl_pvalue
      |-- [full spectrum]      -> stable_rank, entropy, gini, dominance
      |-- [eigenvectors]       -> participation_ratio, critical_weights
      |-- [SETOL diagnostics]  -> ERG condition, learning_phase, alpha_hat
      '-- [randomization]      -> rand_distance, ww_softrank
```

#### Complete Metric Reference

The analyzer produces 21 distinct spectral metrics per layer. These are organized below by reliability tier based on their mathematical robustness and sensitivity to assumptions.

##### Tier 1 Metrics: Always Reliable

These metrics require no fitting, have no distributional assumptions, and are always computable. Start here when interpreting results.

| Metric | Formula | Range | Description | Interpretation |
|--------|---------|-------|-------------|----------------|
| **stable_rank** | $\sum\lambda_i / \max(\lambda_i)$ | $[1, \min(M,N)]$ | Effective dimensionality of the weight matrix. More robust than discrete rank. | Compare to layer dimension: `stable_rank / min(M,N)` gives capacity utilization. Near 1.0 = full utilization. Much less than 1.0 = many dormant dimensions, potential bottleneck. Scale-invariant and not sensitive to outliers. **Limitation**: Cannot distinguish between different spectral shapes with the same ratio. |
| **entropy** | $-\sum p_i \ln(p_i) / \ln(\text{rank})$ where $p_i = \lambda_i / \sum\lambda_i$ | $[0, 1]$ | Normalized Shannon entropy of the eigenvalue distribution. Measures how uniformly information is spread. | Near 0 = rank collapse (one/few modes dominate everything) — problematic. In (0.3, 0.7) = healthy balance of dominant and distributed modes. Near 1.0 = nearly uniform spectrum — **ambiguous**: can mean well-distributed OR random/untrained. Cross-reference with alpha to disambiguate. |
| **gini_coefficient** | Gini index of $\|\lambda_i\|$ | $[0, 1]$ | Inequality of the eigenvalue distribution (analogous to wealth inequality in economics). | Near 0 = all eigenvalues approximately equal. Above 0.5 = significant inequality, some modes dominate. Above 0.8 = extreme concentration, potential fragility. **Limitation**: Cannot distinguish between different *types* of inequality (gradual power-law decay vs. isolated rank-1 spike). |

##### Tier 2 Metrics: Reliable with Verification

These metrics are the primary diagnostics but depend on a power-law fitting step. Always verify their reliability conditions before interpreting.

| Metric | Formula | Description | Interpretation |
|--------|---------|-------------|----------------|
| **alpha** ($\alpha$) | MLE: $\alpha = 1 + n \cdot [\sum \ln(x_i/x_{\min})]^{-1}$ | Power-law exponent of the ESD tail. **The primary training quality indicator.** | See the [Alpha Phase Diagram](#alpha-phase-diagram) below for detailed interpretation. |
| **alpha_hat** ($\hat{\alpha}$) | $\alpha \times \log_{10}(\lambda_{\max}/N)$ | Scale-weighted quality metric combining tail shape with weight magnitude, normalized by layer dimension $N$. | Lower = better generalization. Use for **within-model layer ranking** only ("which layer is weakest?"). Can be negative for Conv2D with large receptive fields. Do NOT compare across architectures. Inherits all of alpha's failure modes plus adds dimension sensitivity. |
| **alpha_weighted** | $\alpha \times \log_{10}(\lambda_{\max})$ | Legacy (unnormalized) combined metric. | Prefer `alpha_hat` (the SETOL-correct normalized version). Kept for backward compatibility. |
| **learning_phase** | Categorical from $\alpha$ | SETOL learning phase classification. | One of: `over-regularized`, `ideal`, `good`, `fair`, `under-trained`, `failed`. Derived directly from alpha — inherits its limitations. |
| **dominance_ratio** | $\lambda_{\max} / \sum(\lambda_{\text{rest}})$ | How much the single largest eigenvalue dominates the entire spectrum. | Below 0.1 = no single mode dominates (healthy). Between 0.1 and 1.0 = moderate dominance (typical for trained networks). Above 1.0 = top mode contains more variance than all others combined — red flag for rank-1 spikes. **Limitation**: Very sensitive to a single outlier eigenvalue. |
| **xmin** | Optimal KS-distance threshold | The eigenvalue threshold above which the power-law fit applies. Defines the Effective Correlation Space (ECS). | Internal to alpha fitting, but also gates the ERG condition. If xmin captures very few eigenvalues, both alpha and ERG are unreliable. |
| **D** | KS distance at optimal xmin | Kolmogorov-Smirnov goodness-of-fit distance. | Lower = better fit. No absolute threshold — use pl_pvalue instead for formal testing. |
| **sigma** ($\sigma$) | $(\alpha - 1) / \sqrt{n_{\text{tail}}}$ | Standard error of the alpha estimate (asymptotic). | If $\sigma > \alpha/3$, the confidence interval spans multiple learning phases — treat alpha with caution. **Note**: Underestimates true uncertainty for small samples. |
| **num_pl_spikes** | Count of eigenvalues $\geq x_{\min}$ | Number of eigenvalues in the power-law tail. | Below 50 = alpha estimate has high variance and should be treated as approximate. Above 200 = alpha is reliable. |

**When to NOT trust Tier 2 metrics:**
- `pl_pvalue < 0.1` — the ESD is probably not power-law; alpha is fitting the wrong model
- `sigma > alpha / 3` — confidence interval spans multiple phases
- Layer has < 50 eigenvalues (`num_pl_spikes < 50`) — MLE variance too high
- Truncated SVD was used (layer dimension > 15,000) — incomplete tail

##### Tier 3 Metrics: Contextual / Expensive

These metrics provide deep diagnostics but are either computationally expensive, dependent on other metrics being reliable, or only meaningful in specific contexts.

| Metric | Formula | Description | Interpretation |
|--------|---------|-------------|----------------|
| **pl_pvalue** | Bootstrap KS test (Clauset et al. 2009) | Goodness-of-fit p-value: "Is the power-law hypothesis plausible?" | Above 0.1 = power-law not rejected, alpha is meaningful. Below 0.1 = power-law is a poor fit, **downweight alpha**. Equals -1.0 when test was not run (fit failed). **Cost**: ~100x the cost of fitting alpha. Essential for important decisions, skip for quick surveys. **Resolution**: Default 100 bootstraps gives 0.01 granularity. |
| **participation_ratio** | $(\\sum v_i^2)^2 / \\sum v_i^4$ for top-$k$ eigenvectors | Measures how many neurons contribute to the principal components (Anderson localization). | Near 1 = features localized to single neurons (fragile to pruning). Near $N$ = features spread across all neurons (robust but possibly diffuse). Reports mean PR over top-3 eigenvectors. **Cost**: Requires full SVD to extract eigenvectors — expensive for large layers. |
| **concentration_score** | $\log(1 + \text{Gini} \times \text{dominance} / \text{PR})$ | Composite fragility index combining inequality, dominance, and localization. | Use ONLY for **relative ranking** within a model ("which layer is most fragile?"). No meaningful absolute thresholds. High = concentrated/fragile. The log-transform and three-way composition make raw values unintuitive. **Warning**: Compounds errors from its constituent metrics. |
| **erg_log_det** | $\sum \ln(\tilde{\lambda}_i)$ for ECS eigenvalues | SETOL ERG condition: volume-preservation test from renormalization group theory. | Near 0 = layer at critical point (ideal). Much greater than 0 = ECS eigenvalues too large (overfitting). Much less than 0 = ECS eigenvalues too small (under-utilization). **Only meaningful when alpha is near 2.0.** |
| **erg_delta_lambda_min** | Gap between xmin and ERG boundary | Measures how close the power-law boundary is to the theoretical ERG boundary. | Near 0 = boundaries coincide (ideal). Large gap = ECS definition is inconsistent. |
| **erg_satisfied** | $\|\text{erg\_log\_det}\| < 1.0$ | Boolean: is the ERG condition approximately satisfied? | `True` + alpha near 2.0 = ideal learning (SETOL critical point). The threshold of 1.0 is a practical choice, not derived from theory. |
| **rand_distance** | $\sqrt{\text{JSD}(\text{ESD}, \text{ESD}_{\text{random}})}$ | Jensen-Shannon distance between actual and randomly-permuted weight spectra. | Above 0.3 = significant learned structure (good). Below 0.1 = nearly indistinguishable from random (concerning). **Limitations**: Uses single permutation (no variance estimate) and 100-bin histograms (arbitrary binning). |
| **ww_softrank** | $\max(\lambda_{\text{rand}}) / \max(\lambda_{\text{actual}})$ | Ratio of random spectral norm to actual spectral norm. | Below 1 = learned weights have larger spectral norm than random (normal). Above 1 = weights more regularized than random (unusual). |
| **rank_loss** | Count of near-zero singular values | Number of singular values below machine-epsilon tolerance. | Indicates numerical rank deficiency. High rank_loss = many effectively zero dimensions. |
| **weak_rank_loss** | Count of eigenvalues $< 10^{-6}$ | Similar to rank_loss but with a fixed (looser) threshold. | Captures dimensions that are near-zero but above machine epsilon. |
| **lambda_max** | $\max(\lambda_i)$ | Largest eigenvalue (squared spectral norm). | Raw scale metric — NOT comparable across layers of different dimensions without normalization. The code computes $\mathbf{W}^\intercal\mathbf{W}$ without the $1/N$ normalization factor, so lambda_max scales with $N$. |
| **sv_max / sv_min** | Largest / smallest singular values | Extremes of the singular value spectrum. | Large sv_max/sv_min ratio = ill-conditioned layer. |
| **log_norm / log_spectral_norm** | $\log_{10}(\sum\lambda_i)$ / $\log_{10}(\max\lambda_i)$ | Log-scaled Frobenius and spectral norms. | Useful for comparing magnitude across layers on different scales. |
| **log_alpha_norm** | $\log_{10}(\sum \lambda_i^\alpha)$ | Alpha-weighted log norm. | Emphasizes tail behavior — combines norm with alpha-based weighting. |
| **critical_weight_count** | Count of high-contribution weights | Weights that contribute most to top eigenvectors. | High count = many individual weights are spectrally important — model is sensitive to weight perturbations. |

#### Alpha Phase Diagram

The power-law exponent $\alpha$ is the central metric of the spectral analysis. Its value places each layer into a learning phase based on SETOL theory:

| $\alpha$ Range | Phase | Physical State | Characteristics | Recommended Action |
|:---------------|:------|:---------------|:----------------|:-------------------|
| $< 0$ | **Failed** | N/A | Power-law fit did not converge. | Ignore alpha for this layer. Diagnose using stable_rank and entropy only. |
| $[1.0, 2.0)$ | **Over-regularized** | Glassy meta-stable state | Correlation traps (rank-1 spikes). May result from excessive learning rates, very small batch sizes, or over-training into loss crevices. Undefined or infinite weight variance. | Reduce learning rate. Increase batch size. Consider early stopping. Check for rank-1 spikes via dominance_ratio. |
| $[2.0, 2.5)$ | **Ideal** | Critical phase boundary | ERG condition satisfied ($\Delta\lambda_{\min} \approx 0$). Maximal information compression. Optimal bias-variance tradeoff. This is the SETOL target. | Maintain current training configuration. This is the optimal state. |
| $[2.5, 4.0)$ | **Good / Typical** | Heavy-tailed | Standard working regime for most SOTA models. Well-trained but not at the theoretical optimum. | Healthy. Could improve toward $\alpha = 2$ but not necessary for good performance. |
| $(4.0, 6.0]$ | **Fair** | Moderately heavy-tailed | Working but with room for improvement. Features are being learned but not fully compressed. | Train longer. Reduce regularization. Consider increasing model capacity. |
| $> 6.0$ | **Under-trained** | Random-like (Marchenko-Pastur) | Insufficient correlation learning. Layer behaves nearly as if randomly initialized. | Significantly more training needed. Check if layer is receiving gradients. |

#### Practical Decision Framework

Follow this step-by-step process when reading spectral analysis output:

```
Step 1: Check STATUS column
   |-- "failed" -> skip this layer for alpha-based metrics
   '-- "success" -> proceed

Step 2: Check pl_pvalue (if computed)
   |-- < 0.1 -> alpha is UNRELIABLE for this layer
   |            rely on stable_rank, entropy, gini instead
   |-- >= 0.1 -> alpha is meaningful, proceed
   '-- -1.0 -> test was not run, proceed with caution

Step 3: Read alpha and learning_phase
   '-- This is your primary diagnostic

Step 4: Cross-validate with Tier 1 metrics
   |-- Low entropy + low stable_rank = rank collapse (regardless of alpha)
   '-- High entropy + high stable_rank + high alpha = random/untrained

Step 5: Check concentration metrics for anomalies
   |-- dominance_ratio > 1.0 = rank-1 spike (investigate even if alpha looks OK)
   '-- Low participation_ratio = localized features (fragile to pruning)

Step 6: Use alpha_hat for within-model layer ranking
   '-- Sort layers by alpha_hat to find the weakest links
```

#### Known Blind Spots and Limitations

Understanding what the spectral analyzer **cannot** do is as important as understanding what it can:

1. **Non-power-law spectra**: The analyzer only tests the power-law hypothesis. If the ESD is actually log-normal or stretched exponential (which is common in practice), alpha will still produce a number — but it's fitting the wrong model. The `pl_pvalue` test catches this when enabled, but there is no alternative distribution comparison.

2. **Spatial structure loss**: Conv2D weight tensors are matricized by reshaping `(kh, kw, in_c, out_c)` into `(kh*kw*in_c, out_c)`. This destroys spatial correlations. The spectral analysis captures properties of the linear transformation but misses the spatial structure that makes convolutions effective. Expect ~10-15% mismatch for spatially structured architectures.

3. **Batch normalization fusion**: If BatchNorm parameters are folded into convolutional weights (common in inference-optimized models), the spectral properties change but the analyzer has no way to detect or account for this.

4. **Cross-architecture comparison**: `alpha_hat`, `lambda_max`, and all norm-based metrics have different scales for different layer types and architectures. Only `alpha` itself is somewhat comparable across architectures (it's scale-invariant). Even then, different architectures may have naturally different alpha distributions.

5. **Small layers**: Below ~50 eigenvalues, all fitting-based metrics (alpha, ERG, pl_pvalue) are statistically unreliable. The sigma formula `(alpha-1)/sqrt(n)` is only asymptotically correct and underestimates true uncertainty for small samples.

6. **Truncated SVD for large layers**: When a layer has more than 15,000 dimensions, the analyzer switches to truncated SVD for performance. This computes only the top-k singular values, potentially missing the tail of the distribution and biasing alpha upward.

7. **xmin as causal bottleneck**: The `xmin` threshold (where the power-law fit begins) gates alpha quality, alpha_hat, learning_phase, ERG condition, and all downstream recommendations. If xmin is wrong, the entire diagnostic chain is compromised. The concentration metrics (Gini, dominance, PR) bypass this bottleneck because they use the full spectrum.

8. **Extremely rectangular matrices**: When $M \ll N$ or $N \ll M$ (very high aspect ratio), the Q ratio is extreme and metrics may behave unexpectedly. This can occur with embedding layers or very wide/narrow bottleneck layers.

#### Interpreting the Funnel Diagnostic

When examining spectral metric evolution across layers (from input to output):

- **Healthy training**: Early layers have higher alpha (more random), deep layers have alpha approaching 2.0. The `erg_delta_lambda_min` collapses as alpha approaches 2. This creates a characteristic "funnel" shape in the evolution plots.
- **Problem indicator**: Irregular patterns, sudden jumps, or divergence in the alpha-across-layers plot.
- **Layer-position context**: It is normal for the first and last layers to have somewhat different spectral properties than the middle layers.

### Calibration & Confidence Metrics

This analysis evaluates how well the model's predicted probabilities reflect the true likelihood of outcomes.

| Metric             | Description                                                                                             | Ideal Value     |
| ------------------ | ------------------------------------------------------------------------------------------------------- | --------------- |
| **ECE**            | Expected Calibration Error. Measures the average gap between a model's prediction confidence and its actual accuracy. | 0 (perfect calibration) |
| **Brier Score**    | The mean squared error between predicted probabilities and the one-hot encoded true labels. A measure of both calibration and resolution. | 0 (perfect predictions) |
| **Mean Confidence**| The average of the highest probability assigned by the model for each prediction in the dataset.      | Context-dependent; very high values might indicate overconfidence. |
| **Mean Entropy**   | The average Shannon entropy across all prediction distributions, quantifying the model's overall uncertainty. | Context-dependent; lower values indicate more confident (peaked) predictions. |

### Information Flow Metrics

This analysis tracks how information (activations) propagates through the network, helping to identify bottlenecks and pathologies.

| Metric                  | Description                                                                                                   | Interpretation                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Activation Statistics** | The mean, standard deviation, and sparsity of activations for each layer's output.                            | Helps diagnose vanishing gradients (mean/std near zero), exploding gradients (large values), or dead neurons (high sparsity after ReLU).       |
| **Effective Rank**      | A measure of the dimensionality of the feature space represented by a layer's activations.                    | A higher rank suggests that the layer is learning a diverse and rich set of features. A sudden drop in rank can indicate an information bottleneck. |
| **Positive Ratio**      | The fraction of activations that are positive (typically after a ReLU activation).                            | Values near 0 or 1 indicate that the layer is saturated (either always off or always on), which hinders learning. A balanced ratio is healthier. |
| **Specialization Score**| A composite score (0-1) that combines activation health, balance, and effective rank to measure feature learning quality. | Higher scores suggest a layer is effectively transforming information without losing diversity or becoming saturated.                           |

### Training Dynamics Metrics

This analysis examines the model's learning history to understand its training efficiency, stability, and tendency to overfit.

| Metric                  | Description                                                                                             | Interpretation                                                                                                                            |
| ----------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Epochs to Convergence** | The number of epochs required for the model to reach 95% of its peak validation performance.              | A measure of training speed. Lower is faster and indicates more efficient learning.                                                       |
| **Overfitting Index**   | The average difference between validation loss and training loss during the final third of training.    | A positive value indicates overfitting (model performs better on training data). A negative value indicates underfitting.                  |
| **Training Stability**  | The standard deviation of validation loss over the last several epochs.                                 | A lower value indicates a smooth and stable convergence. High values suggest an unstable training process (e.g., learning rate is too high). |
| **Peak Performance**    | The best validation accuracy or loss achieved during the entire training process, and the epoch it occurred. | Represents the model's maximum potential performance. If it occurs early, it may be a sign of early overfitting.                          |
| **Final Gap**           | The difference between validation and training loss at the very last epoch of training.                 | A snapshot of the model's generalization state at the end of training.                                                                    |

## 5. Usage Patterns & Use Cases

### Pattern 1: Single Model Deep Dive

Use the analyzer to thoroughly debug a single model's performance and behavior.

```python
# Scenario: A model's performance is unexpectedly low.
model = keras.models.load_model('path/to/problem_model.keras')
# history = training_history_for_the_model (see section 3.3 for structure)

config = AnalysisConfig(
    analyze_weights=True,
    analyze_information_flow=True,  # Check for dead neurons/layers
    analyze_calibration=True,
    analyze_training_dynamics=True
)

analyzer = ModelAnalyzer(
    models={'ProblemModel': model},
    training_history={'ProblemModel': history},
    config=config
)
results = analyzer.analyze(test_data)

# Next steps:
# 1. Check `weight_learning_journey.png` for exploding/vanishing weights.
# 2. Check `information_flow_analysis.png` for dead layers (low activation/rank).
# 3. Check `training_dynamics.png` for severe overfitting or unstable training.
```

### Pattern 2: Multi-Model Comparison for Selection

Compare different architectures or training runs to select the best candidate.

```python
# Scenario: Choose the best model from multiple architectures for production.
models = {
    'ResNet50': resnet_model,
    'EfficientNet': efficientnet_model,
    'ConvNext': convnext_model
}

analyzer = ModelAnalyzer(models=models)
results = analyzer.analyze(test_data)

# Next steps:
# 1. Start with `summary_dashboard.png`. The performance table gives a quick overview.
# 2. Check the "Calibration Landscape" plot. Models in the bottom-left (low ECE, low Brier) are best.
# 3. Check `spectral_summary.png` to compare training quality (alpha values) and complexity.
```

### Pattern 3: Hyperparameter Sweep Analysis

Efficiently analyze the results of a hyperparameter sweep to find the optimal configuration.

```python
# Scenario: Find the best learning rate and batch size from a sweep.
models = {
    'lr_0.001_batch_32': model_1,
    'lr_0.01_batch_32': model_2,
    'lr_0.001_batch_64': model_3,
    'lr_0.01_batch_64': model_4
}
# sweep_histories is a dict where keys match model names, and values are Keras history dicts
# See section 3.3 for the required structure
histories = {name: h for name, h in sweep_histories.items()}

config = AnalysisConfig(
    analyze_training_dynamics=True,
    pareto_analysis_threshold=2 # Generate Pareto plot if 2+ models are provided
)

analyzer = ModelAnalyzer(models=models, training_history=histories, config=config)
results = analyzer.analyze(validation_data)

# Generate and save the Pareto analysis plot
pareto_fig = analyzer.create_pareto_analysis()

# Next steps:
# 1. Open `pareto_analysis.png`. Models on the Pareto front offer optimal trade-offs.
# 2. Use the heatmap to compare normalized performance across all models and metrics.
```

### Pattern 4: Data-Free Training Quality Check

Use spectral analysis to quickly assess if a model is over-trained or under-trained without a test set.

```python
# Scenario: Quickly validate a newly trained model's quality.
model = keras.models.load_model('path/to/new_model.keras')

config = AnalysisConfig(
    analyze_spectral=True,          # Enable spectral analysis
    spectral_bootstraps=100,        # Enable pl_pvalue computation (set to 0 to skip)
    spectral_concentration_analysis=True  # Enable Gini/PR/dominance
)
analyzer = ModelAnalyzer(models={'NewModel': model}, config=config)
results = analyzer.analyze() # No data needed for this analysis

# Access spectral results programmatically
df = results.spectral_analysis
print(df[['name', 'alpha', 'sigma', 'pl_pvalue', 'learning_phase',
          'stable_rank', 'entropy', 'concentration_score']])

# Check recommendations
for rec in results.spectral_recommendations.get('NewModel', []):
    print(f"  -> {rec}")

# Next steps:
# 1. Open `spectral_summary.png`. Check if the Mean Alpha is in the 2.0-6.0 range.
# 2. Look at the learning_phase column for per-layer diagnostics.
# 3. Check pl_pvalue: layers with p < 0.1 have unreliable alpha values.
# 4. Cross-validate: low entropy + low stable_rank = rank collapse (bad regardless of alpha).
```

### Pattern 5: Improving Generalization with SVD Smoothing

Create a smoothed version of a model to potentially improve its generalization performance.

```python
# Scenario: A model shows signs of being slightly over-trained.
analyzer = ModelAnalyzer(models={'OverTrainedModel': model})
analyzer.analyze() # Run analysis to get spectral data

# Create a smoothed version of the model
smoothed_model = analyzer.create_smoothed_model(
    model_name='OverTrainedModel',
    method='detX' # 'svd', 'detX', or 'lambda_min'
)

# You can now save and evaluate the smoothed_model
```

### Pattern 6: Identifying Fragile Layers for Pruning/Quantization Safety

Before pruning or quantizing, use concentration metrics to find layers where information is dangerously concentrated.

```python
config = AnalysisConfig(
    analyze_spectral=True,
    spectral_concentration_analysis=True  # Must be enabled for this use case
)
analyzer = ModelAnalyzer(models={'ProductionModel': model}, config=config)
results = analyzer.analyze()

df = results.spectral_analysis

# Find layers that are fragile (concentrated information)
fragile_layers = df.nlargest(5, 'concentration_score')[
    ['name', 'concentration_score', 'dominance_ratio', 'participation_ratio', 'gini_coefficient']
]
print("Most fragile layers (handle with care during pruning/quantization):")
print(fragile_layers)

# Layers with dominance_ratio > 1.0 have rank-1 spikes — very sensitive to perturbation
spike_layers = df[df['dominance_ratio'] > 1.0][['name', 'dominance_ratio', 'alpha']]
if not spike_layers.empty:
    print("\nLayers with rank-1 spikes (do NOT prune these aggressively):")
    print(spike_layers)
```

## 6. Advanced Configuration

The `AnalysisConfig` class provides fine-grained control over the analysis process.

```python
config = AnalysisConfig(
    # === Main Toggles ===
    analyze_weights=True,
    analyze_calibration=True,
    analyze_information_flow=True,
    analyze_training_dynamics=True,
    analyze_spectral=True,

    # === Data Sampling ===
    n_samples=1000,  # Max samples for data-dependent analyses (Info Flow, Calibration)

    # === Weight Analysis ===
    weight_layer_types=['Dense', 'Conv2D'],
    analyze_biases=False,
    compute_weight_pca=True,

    # === Spectral Analysis (WeightWatcher/SETOL) ===
    spectral_min_evals=10,                # Min eigenvalues for a layer to be analyzed
    spectral_max_evals=15000,             # Soft cap to switch to truncated SVD for speed
    spectral_concentration_analysis=True, # Enable concentration metrics (Gini, PR, dominance)
    spectral_randomize=False,             # Compare with randomized weights (slow)
    spectral_bootstraps=100,              # Bootstrap iterations for pl_pvalue (0 to skip)
    spectral_glorot_fix=False,            # Apply Glorot normalization before analysis
    spectral_per_layer_diagnostics=True,  # Generate per-layer power-law fit plots

    # === Calibration Analysis ===
    calibration_bins=15,

    # === Training Dynamics ===
    smooth_training_curves=True,
    smoothing_window=5,

    # === Visualization & Output ===
    plot_style='publication',
    save_plots=True,
    save_format='png',
    dpi=300,

    # === JSON Serialization Options to manage file size
    json_include_per_sample_data=False, # Set to True to include bulky arrays like confidence/entropy for every sample.
    json_include_raw_esds=False,        # Set to True to include raw eigenvalue arrays for every layer.
    
    # === Performance & Limits ===
    max_layers_heatmap=12,
    max_layers_info_flow=8,
    verbose=True,
)
```

## 7. Understanding the Output

After running, the analyzer saves plots and a JSON data file to the output directory.

```
analysis_results/
├── summary_dashboard.png              # START HERE: High-level overview of all models.
├── spectral_summary.png               # Compares models on spectral metrics (training quality).
├── training_dynamics.png              # Training curves, overfitting, and convergence analysis.
├── weight_learning_journey.png        # Weight magnitude evolution and health heatmap.
├── confidence_calibration_analysis.png  # Deep dive into model confidence and calibration.
├── information_flow_analysis.png      # Layer-wise analysis of activations and information.
├── pareto_analysis.png               # (Optional) Hyperparameter optimization insights.
├── spectral_plots/                   # Directory with detailed per-layer power-law plots.
│   └── ModelA_layer_5_dense_powerlaw.png
└── analysis_results.json             # Raw data for all computed metrics.
```

### Key Visualizations Explained

#### 1. Summary Dashboard (`summary_dashboard.png`)

A 2x2 grid providing a holistic view of all models.

-   **Performance Table**: A comprehensive summary of key performance indicators, including training efficiency metrics if history is provided.
-   **Model Similarity**: A 2D PCA plot of weight statistics. Models that are close together have learned similar weight distributions.
-   **Confidence Profiles**: Violin plots showing the distribution of prediction confidence for each model.
-   **Calibration Landscape**: A scatter plot of ECE vs. Brier Score. The goal is to be in the bottom-left quadrant (low error, good calibration).

#### 2. Spectral Analysis Summary (`spectral_summary.png`)

A dashboard for comparing models based on their weight matrix spectral properties.

-   **Alpha Distribution with Phase Backgrounds**: Histogram of alpha values overlaid with color-coded phase regions (pink = over-regularized, green = ideal, yellow = under-trained). A quick visual check of how many layers fall in each phase.
-   **Concentration Score Distribution**: Comparison of information fragility across models.
-   **Alpha per Layer**: Scatter plot showing how alpha evolves across the network depth, with reference lines at the phase boundaries (2.0, 6.0). Look for the "funnel" convergence pattern — healthy models show alpha decreasing toward 2.0 in deeper layers.
-   **Stable Rank per Layer**: Log-scale scatter showing capacity utilization across layers.
-   **Recommendations**: The `analysis_results.json` file contains specific, actionable recommendations based on the spectral analysis for each model (e.g., "Model may be over-trained. Consider early stopping...").

#### 3. Training Dynamics (`training_dynamics.png`)

A deep dive into the learning process.

-   **Loss/Accuracy Curves**: Smoothed training and validation curves for a clear view of the learning trajectory.
-   **Overfitting Analysis**: Plots the gap (validation loss - training loss) over epochs to diagnose overfitting.
-   **Best Epoch Performance**: A scatter plot showing each model's peak validation accuracy versus the epoch it was achieved.
-   **Summary Table**: A detailed table of quantitative training metrics like convergence speed and stability.

#### 4. Weight Learning Journey (`weight_learning_journey.png`)

Assesses the health and evolution of model weights.

-   **Weight Evolution**: Shows how the L2 norm of weights changes across layers, helping detect exploding or vanishing gradients.
-   **Health Heatmap**: A layer-by-layer health score for each model, allowing quick identification of problematic layers.

#### 5. Confidence & Calibration Analysis (`confidence_calibration_analysis.png`)

Evaluates the reliability of model predictions.

-   **Reliability Diagram**: Compares predicted probability to actual accuracy. A perfect model lies on the y=x diagonal.
-   **Confidence Distributions**: Violin plots showing the shape of each model's confidence distribution.
-   **Per-Class ECE**: A bar chart showing calibration error for each class, identifying unreliable classes.
-   **Uncertainty Landscape**: A 2D density plot of confidence vs. entropy, showing the model's uncertainty profile.

#### 6. Information Flow Analysis (`information_flow_analysis.png`)

Diagnoses how information propagates through the network.

-   **Activation Flow Overview**: Tracks activation mean and standard deviation to spot vanishing/exploding signals.
-   **Effective Rank Evolution**: Plots the dimensionality of information at each layer to find bottlenecks.
-   **Activation Health Dashboard**: A heatmap showing issues like dead or saturated neurons.
-   **Layer Specialization Analysis**: Plots a score measuring how well each layer learns diverse features.

#### 7. Pareto Analysis (`pareto_analysis.png`)

(Generated with `create_pareto_analysis()`) A powerful tool for hyperparameter tuning.

-   **Pareto Front Plot**: A scatter plot of Peak Accuracy vs. Overfitting Index. Models on the "Pareto Front" represent the best possible trade-offs.
-   **Normalized Performance Heatmap**: Compares all models across key metrics, making it easy to identify the best configuration based on priorities.

#### 8. Detailed Spectral Plots (`spectral_plots/*.png`)

These plots provide a layer-by-layer deep dive into the power-law fit that is summarized in the main spectral dashboard. Each plot visualizes the Empirical Spectral Density (ESD) of a single layer's weight matrix.

-   **What it shows**: A log-log plot of the eigenvalue ($\lambda$) distribution. A straight line in the tail of this plot is the signature of a power-law.
-   **How to read it**:
    -   The **blue dots** represent the actual binned histogram of the layer's eigenvalues.
    -   The **red line** is the best-fit power-law model, $P(\lambda) \sim \lambda^{-\alpha}$. The steepness of this line's slope is the exponent $\alpha$.
    -   The **vertical dashed line (`xmin`)** marks the beginning of the power-law tail, where the fit is applied. Eigenvalues to the left of xmin are considered "bulk" (noise/memorization). Eigenvalues to the right form the Effective Correlation Space (ECS).
    -   **Interpretation**: A well-trained layer will show the blue dots in the tail (right side of the plot) aligning closely with the red line. A poor fit (dots systematically deviating from the line) suggests the ESD is not truly power-law — check `pl_pvalue` to confirm.

## 8. Troubleshooting

-   **`analysis_results.json` is too large**: By default, the analyzer saves summary statistics to keep the file size small. If you need the raw, per-sample data (e.g., the confidence score for every single prediction), you can enable it in the configuration. Be aware this can increase the JSON file size from kilobytes to many megabytes.
    ```python
    config = AnalysisConfig(
        json_include_per_sample_data=True, # Saves raw confidence/entropy arrays
        json_include_raw_esds=True         # Saves raw eigenvalue arrays
    )
    ```
-   **"No training metrics found"**: The analyzer robustly searches for common metric names (`accuracy`, `val_loss`, etc.). If you use non-standard names in your `history` object, analysis will be limited. Ensure your Keras history keys are standard. **See Section 3.3 for the exact required structure of the `training_history` dictionary.**
-   **Memory Issues (OOM)**: For very large models or datasets, analysis can be memory-intensive. Reduce the sample size and disable the most expensive analyses in `AnalysisConfig`:
    ```python
    config = AnalysisConfig(
        n_samples=200,                  # Reduce from default 1000
        analyze_information_flow=False, # This is the most memory-intensive (captures activations)
        max_layers_heatmap=8            # Limit heatmap size
    )
    ```
-   **Spectral analysis shows alpha = -1 for many layers**: This means the power-law fit failed. Common causes: layers are too small (< 10 eigenvalues), weights are all near-zero (dead layers), or the layer type is not supported. Check the `status` column in the spectral DataFrame.
-   **Alpha values seem unreliable**: Enable the bootstrap goodness-of-fit test (`spectral_bootstraps=100`) and check `pl_pvalue`. Values below 0.1 indicate the power-law is a poor model for that layer's ESD. Also check `sigma` — if `sigma > alpha/3`, the estimate has very wide confidence intervals.
-   **Concentration score is very high for one layer**: This indicates a potential rank-1 spike. Check `dominance_ratio` — if above 1.0, the largest eigenvalue dominates the entire spectrum. This can happen with batch normalization artifacts or correlation traps from training.
-   **Plots look wrong/empty**: Enable verbose logging (`config = AnalysisConfig(verbose=True)`) and check the console output. You can also inspect the `analysis_results.json` file to see what data was successfully computed.
-   **Matplotlib Backend Issues**: If running in a Jupyter Notebook, use `%matplotlib inline` before importing the analyzer. The analyzer uses the non-interactive `Agg` backend by default to ensure saving plots works in headless environments, but `plt.show()` might not work immediately without configuration.

## 9. Extensions

The toolkit is designed to be extensible. You can add your own custom analysis and visualization modules by inheriting from the base classes.

### Creating a Custom Analyzer

Extend the `BaseAnalyzer` class and implement the `analyze` method.

```python
from dl_techniques.analyzer.analyzers.base import BaseAnalyzer
from dl_techniques.analyzer.data_types import AnalysisResults, DataInput

class MyCustomAnalyzer(BaseAnalyzer):
    def requires_data(self) -> bool:
        return True # This analyzer needs data

    def analyze(self, results: AnalysisResults, data: DataInput, cache: dict) -> None:
        # Your custom logic here
        custom_metrics = {}
        for model_name, model in self.models.items():
            # ... compute your metrics ...
            custom_metrics[model_name] = {'my_score': 0.99}

        # Store results in the main results object
        results.custom_metrics = custom_metrics
```

### Creating a Custom Visualizer

Extend the `BaseVisualizer` class and implement the plotting logic.

```python
import matplotlib.pyplot as plt
from dl_techniques.analyzer.visualizers.base import BaseVisualizer

class MyCustomVisualizer(BaseVisualizer):
    def create_visualizations(self) -> None:
        if not hasattr(self.results, 'custom_metrics'):
            return

        fig, ax = plt.subplots()
        # ... your custom plotting logic ...
        # You can access self.results, self.config, and self.model_colors

        if self.config.save_plots:
            self._save_figure(fig, 'my_custom_plot')
        plt.close(fig)
```

## 10. Theoretical Background & References

The analyses performed by this toolkit are grounded in established research from machine learning, statistics, and statistical physics.

### SETOL: Semi-Empirical Theory of Learning

The spectral analysis module is built on SETOL, which bridges theoretical understanding and practical deep learning. Unlike traditional statistical learning theory (too pessimistic) or classical statistical mechanics (doesn't capture learning dynamics), SETOL uses actual trained weights to predict layer quality.

Key theoretical components:
- **Heavy-Tailed Self-Regularization (HTSR)**: SGD implicitly regularizes neural networks, causing weight matrix spectra to develop heavy tails. SETOL explains *why* this happens.
- **Effective Correlation Space (ECS)**: The essential subspace where learning actually happens — eigenvalues above xmin. The bulk below xmin represents noise/memorization.
- **ERG Condition**: The Exact Renormalization Group condition ($\ln\det(\tilde{X}) \approx 0$) identifies layers at the critical point of ideal learning, analogous to phase transitions in statistical mechanics.
- **Student-Teacher Framework**: SETOL extends the classical framework from vectors to matrices, using the Harish-Chandra-Itzykson-Zuber (HCIZ) integral to calculate expected generalization error.

For comprehensive SETOL documentation, see `SETOL.md` in this directory.

### Key References

1.  **Martin, C. H., & Hinrichs, C. (2025). "SETOL: A Semi-Empirical Theory of (Deep) Learning." arXiv:2507.17912.** (The theoretical foundation for alpha, alpha_hat, ERG condition, and learning phase classification).
2.  **Martin, C., & Mahoney, M. W. (2021). "Heavy-Tailed Universals in Deep Neural Networks." arXiv preprint arXiv:2106.07590.** (Foundational work on Heavy-Tailed Self-Regularization and its connection to generalization).
3.  **Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). "Power-law distributions in empirical data." SIAM review, 51(4), 661-703.** (Provides the statistical methodology for fitting power-law distributions and the bootstrap goodness-of-fit test used for pl_pvalue).
4.  **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). "On calibration of modern neural networks." ICML.** (A seminal paper on model calibration, introducing Expected Calibration Error (ECE) as a standard metric).
5.  **Roy, O., & Vetterli, M. (2007). "The effective rank: A measure of effective dimensionality." LATS.** (Introduces the concept of "effective rank" used in the information flow analysis to measure feature dimensionality).
6.  **Marchenko, V. A., & Pastur, L. A. (1967). "Distribution of eigenvalues for some sets of random matrices." Mathematics of the USSR-Sbornik.** (The random matrix theory baseline — under-trained layers follow this distribution).
7.  **Pennington, J., & Worah, P. (2017). "Nonlinear random matrix theory for deep learning." NeurIPS.** (Extends RMT to nonlinear neural network settings).
8.  **Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.** (A comprehensive textbook covering many of the foundational concepts used in the analyzer, such as overfitting, norms, and training dynamics).
