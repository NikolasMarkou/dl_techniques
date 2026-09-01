# Mixture Density Network (MDN)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 Mixture Density Network: a feed-forward feature extractor followed by a head that emits the parameters of a **Gaussian mixture** rather than a point prediction. Use it when the answer to "what is y given x?" is a distribution — possibly with several peaks — and not a number.

---

## 1. Overview: What is MDN and Why It Matters

### What is an MDN?

A standard regression network learns `f(x) -> y` and is trained on squared error, which makes it predict the *conditional mean*. When the true conditional distribution has two modes, the mean sits between them, in a region the data never occupies.

An MDN instead predicts the parameters of a mixture of `num_mixtures` Gaussians, conditioned on `x`. It can represent multi-modality, and it reports its own uncertainty as part of the output.

### Key Innovations

1. **Distribution, not a point.** Output is `(mu, sigma, pi)` per component, so `P(y|x)` is available in full.
2. **Multi-modality.** Several components can cover several plausible answers at once.
3. **Heteroscedastic noise.** Each `sigma` is a function of `x`, so the model learns *where* the data are noisy.
4. **Uncertainty decomposition.** The law of total variance splits the predictive spread into aleatoric (data noise) and epistemic (component disagreement) parts.

### Why MDNs Matter

```
Standard regression:  minimize MSE -> predicts E[y|x].
                      On bimodal data this is the average of two answers,
                      which is not an answer.

MDN:                  minimize -log( sum_i pi_i * N(y | mu_i, sigma_i) ).
                      Keeps both modes, learns their weights, and learns
                      how noisy each region is.
```

---

## 2. The Problem MDN Solves

### The limits of standard regression

Consider `y = x * sin(x)` and then invert it: for one value of `y` there are many valid `x`. A squared-error network is forced to output a single value and picks the mean of them.

```
┌─────────────────────────────────────────────────────────────┐
│  One input, several correct outputs                         │
│                                                             │
│  MSE regression:  predicts the mean of the modes -> lands    │
│                   in a gap where no training point exists.   │
│  Fixed-variance:  reports one uncertainty everywhere, even   │
│                   where the data are clean.                  │
└─────────────────────────────────────────────────────────────┘
```

### How MDNs change the game

The network outputs mixture parameters and is trained on the negative log-likelihood of the observations. Nothing forces the components together, so they spread out to cover the modes, and each one's `sigma` adapts to local noise. Uncertainty is not bolted on afterwards; it is what is being fitted.

---

## 3. How MDN Works: Core Concepts

### The two-part architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  ┌─────────────────┐      ┌───────────────────────────────────┐  │
│  │ Feature Extrctr │      │           MDN Layer               │  │
│  │  (Dense stack)  │───►  │  (three parallel heads)           │  │
│  │                 │      │   mu Head    -> means             │  │
│  │ hidden repr.    │      │   sigma Head -> std devs          │  │
│  │    of input x   │      │   pi Head    -> mixture logits    │  │
│  └─────────────────┘      └─────────────────┬─────────────────┘  │
│                                             ▼                    │
│                                  P(y|x) = Σ pi * N(mu, sigma)    │
└──────────────────────────────────────────────────────────────────┘
```

### The predicted distribution

**P(y | x) = Σᵢ πᵢ(x) · N(y | μᵢ(x), σᵢ(x))**

- `πᵢ(x)` — the **mixing coefficient**, the probability of selecting component i. The `π`s sum to 1 after a softmax.
- `N(y | μᵢ, σᵢ)` — a Gaussian with per-component, input-dependent mean and standard deviation.

### The complete data flow

```
STEP 1 — FEATURE EXTRACTION
Input (B, D_in)
  └─► [Dense -> (BatchNorm) -> Activation -> (Dropout)] x N
        └─► hidden features (B, D_hidden)

STEP 2 — PARAMETER PREDICTION (inside MDNLayer)
hidden features
  ├─► mu path    -> Dense -> (BN) -> Act -> Dense   -> (B, num_mix * D_out)
  ├─► sigma path -> Dense -> (BN) -> Act -> Dense   -> softplus + min_sigma
  └─► pi path    -> Dense -> (BN) -> Act -> Dense   -> (B, num_mix) RAW LOGITS
        └─► concatenate [mu, sigma, pi] -> (B, 2 * num_mix * D_out + num_mix)

STEP 3 — LOSS (training)
  split y_pred -> softmax the pi logits -> density of y_true under each Gaussian
  -> weighted sum -> -log(...) -> mean over batch

STEP 4 — SAMPLING (inference)
  split y_pred -> softmax pi -> draw component i ~ Categorical(pi)
  -> draw y ~ N(mu_i, sigma_i) -> repeat num_samples times
```

The concatenated output width is `2 * num_mixtures * output_dimension + num_mixtures`. For `output_dimension=2, num_mixtures=5` that is `(2 * 2 * 5) + 5 = 25`.

---

## 4. Architecture Deep Dive

### 4.1 Feature extractor (`MDNModel`)

A plain feed-forward stack. Each entry in `hidden_layers` expands to up to four sublayers: `Dense -> [BatchNorm] -> Activation -> [Dropout]`, in that order (BatchNorm before the activation, dropout last). All sublayers are created in `__init__`, not `build()`, so a `.keras` weight restore lands on layers that already exist rather than silently re-initializing them.

### 4.2 MDN output layer (`MDNLayer`)

Three independent paths, one per parameter family. They share only the input tensor, no weights.

```
Input (B, D_hidden)
  ├── mu path ────► Dense(inter) ─► [BN] ─► Act ─► Dense(mu)
  ├── sigma path ─► Dense(inter) ─► [BN] ─► Act ─► Dense(sigma) ─► softplus + min_sigma
  └── pi path ────► Dense(inter) ─► [BN] ─► Act ─► Dense(pi)     ─► raw logits
                                  └─► concatenate ─► (B, total_params)
```

**Why separate paths?** The three quantities answer different questions — location (`mu`), spread (`sigma`), mode importance (`pi`). Separate trunks let each learn its own features, which is more stable than sharing one.

**Why `softplus` for sigma?** Standard deviations must be strictly positive. `softplus(x) = log(1 + exp(x))` maps any real to a positive one, and `min_sigma` is added on top so no component can claim zero variance and blow the log-likelihood up.

**Why no activation on the pi head?** Its outputs are *logits*. Exactly one `softmax` or `log_softmax` is applied downstream, in the loss, in sampling, and in the diagnostics. Adding a softplus here compresses the logits and measurably degrades `pi`; do not re-add one.

**Diversity regularization.** The classic MDN failure is *component collapse*, several components converging on the same mean. Setting `diversity_regularizer_strength > 0` adds a pairwise repulsion penalty on the component means. It is applied through `add_loss` and only when `training is True`.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.time_series.mdn import MDNModel

# 1. A genuinely multi-modal problem: invert y = x * sin(x).
X = np.random.uniform(-10, 10, 2000)
y = X * np.sin(X) + np.random.normal(0, 0.5, 2000)
X_train, y_train = y.reshape(-1, 1).astype("float32"), X.reshape(-1, 1).astype("float32")

# 2. Build.
model = MDNModel(
    hidden_layers=[32, 32],     # feature extractor
    output_dimension=1,         # dimensionality of the target
    num_mixtures=5,             # five Gaussians
)

# 3. compile() installs the mixture NLL for you -- pass no loss.
model.compile(optimizer="adam")
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

# 4. Sample from the learned distribution.
X_test = np.linspace(-10, 10, 200).reshape(-1, 1).astype("float32")
samples = model.sample(X_test, num_samples=100)
print(samples.shape)            # (200, 100, 1) -- (batch, samples, output_dim)
```

To visualize, scatter `X_test` against `samples[:, :, 0]` at low alpha over the training data: the samples trace both branches of the inverse, where an MSE regressor would draw one line through the middle.

---

## 6. Component Reference

### 6.1 `MDNModel`

**Location**: `dl_techniques.models.time_series.mdn.model` (re-exported by `models.time_series.mdn` and by the `time_series` family package).

| Parameter | Default | Description |
|:---|:---|:---|
| `hidden_layers` | required | List of ints; the feature extractor widths. Must be non-empty and positive. |
| `output_dimension` | required | Dimensionality of the target `y`. |
| `num_mixtures` | required | Number of Gaussian components. |
| `hidden_activation` | `'relu'` | Activation for the hidden layers. |
| `kernel_initializer` | `'glorot_uniform'` | Kernel initializer. |
| `kernel_regularizer` | `None` | Kernel regularizer. |
| `use_batch_norm` | `False` | Insert BatchNormalization before each activation. |
| `dropout_rate` | `None` | Dropout in `[0, 1)`, or `None`. |

Key methods: `compile()` (installs `self.mdn_layer.loss_func` automatically), `sample(inputs, num_samples, temperature=1.0, seed=None)`, and `predict_with_uncertainty(inputs, confidence_level=0.95)`, which returns a dict with `point_estimates`, `total_variance`, `aleatoric_variance`, `epistemic_variance`, `lower_bound`, `upper_bound`.

**Import note.** `mdn/__init__.py` exports only `MDNModel`. The factory `create_mdn_model` lives in `.model` and is re-exported by the family package:

```python
from dl_techniques.models.time_series.mdn import MDNModel                  # works
from dl_techniques.models.time_series.mdn.model import create_mdn_model    # works
from dl_techniques.models.time_series import MDNModel, create_mdn_model    # works
from dl_techniques.models.time_series.mdn import create_mdn_model          # ImportError
```

`create_mdn_model(hidden_layers, output_dimension, num_mixtures, input_dimension, ...)` also runs a dummy forward pass, so the model it returns is already BUILT and ready for `.summary()`, `.save()`, or weight transfer.

### 6.2 `MDNLayer`

**Location**: `dl_techniques.layers.statistics.mdn_layer.MDNLayer`. Usable in any Keras model.

```python
from dl_techniques.layers.statistics.mdn_layer import MDNLayer

inputs = keras.Input(shape=(128,))
x = keras.layers.Dense(64, activation="relu")(inputs)
mdn_params = MDNLayer(
    output_dimension=1,
    num_mixtures=3,
    intermediate_units=32,
    diversity_regularizer_strength=0.01,
)(x)
model = keras.Model(inputs, mdn_params)   # output shape (None, 9)
```

| Parameter | Description |
|:---|:---|
| `output_dimension` | Dimensionality of `y`. |
| `num_mixtures` | Number of components. |
| `intermediate_units` | Width of the per-path trunk. |
| `diversity_regularizer_strength` | Repulsion penalty on component means. `0.0` (off) by default; must be non-negative. |
| `min_sigma` | Floor added after the softplus, and re-applied in `loss_func` and sampling. |
| `use_batch_norm` | BatchNormalization inside each path. |
| `use_bias` | Switches off the Dense biases *and* the BatchNormalization `center`, so a bias-free layer stays bias-free. |

Key attributes: `loss_func` (the NLL, directly passable to `compile`), `sample()`, `split_mixture_params()` — which returns `(mu, sigma, pi_logits)` with shapes `(B, num_mix, D_out)`, `(B, num_mix, D_out)`, `(B, num_mix)`.

---

## 7. Configuration & Model Variants

### Choosing `num_mixtures`

| Components | Use case | Risk |
|:---:|:---|:---|
| **1** | Standard regression with heteroscedastic noise. | Cannot model multi-modality at all. |
| **3-5** | Good starting point for most problems. | May be too few for complex conditionals. |
| **5-10** | Complex, clearly multi-modal distributions. | Slower, more prone to overfitting. |
| **10+** | High-dimensional or very complex output spaces. | Component collapse; use the diversity regularizer. |

Start at 3 or 5, sample from the trained model, and increase only if visible modes are missing.

### Feature extractor shape

- **Shallow and wide** (`[512]`) — when the input features need little hierarchical processing.
- **Deep and narrow** (`[64, 64, 64]`) — structured inputs where a feature hierarchy helps.
- **Regularization** — `dropout_rate` plus a `kernel_regularizer` for deep models; `use_batch_norm=True` helps training stability.

---

## 8. Comprehensive Usage Examples

### Example 1: Point estimates from a distribution

The expected value `E[y|x] = Σᵢ πᵢ(x) μᵢ(x)` is the natural single-number summary — remembering that on multi-modal data it is exactly the quantity an MDN exists to avoid committing to.

```python
from dl_techniques.layers.statistics.mdn_layer import get_point_estimate

point_estimates = get_point_estimate(model, X_test, model.mdn_layer)   # (200, 1)
```

### Example 2: Decomposing uncertainty

- **Aleatoric** — noise inherent to the data. More data will not reduce it.
- **Epistemic** — disagreement between components, a proxy for model uncertainty.

```python
from dl_techniques.layers.statistics.mdn_layer import get_uncertainty

total_variance, aleatoric_variance = get_uncertainty(
    model, X_test, model.mdn_layer, point_estimates,
)
epistemic_variance = total_variance - aleatoric_variance
total_std = np.sqrt(total_variance)
```

Plot `point_estimates` with a `±total_std` band, and the aleatoric and epistemic standard deviations underneath: the epistemic term spikes exactly where the inverse problem is genuinely ambiguous.

### Example 3: Prediction intervals

```python
from dl_techniques.layers.statistics.mdn_layer import get_prediction_intervals

lower_bound, upper_bound = get_prediction_intervals(
    point_estimates=point_estimates,
    total_variance=total_variance,
    confidence_level=0.95,
)
```

`model.predict_with_uncertainty(X_test)` returns all of the above in one dict if you would rather not call the three helpers separately.

---

## 9. Advanced Usage Patterns

### Pattern 1: Diagnosing component collapse

`check_component_diversity` returns `mean_component_separation`, `std_component_separation`, `mean_mixture_weights`, `std_mixture_weights` and `mean_sigma_values`.

```python
from dl_techniques.layers.statistics.mdn_layer import check_component_diversity

diagnostics = check_component_diversity(model, X_test, model.mdn_layer)
print(diagnostics["mean_component_separation"])
print(np.round(diagnostics["mean_mixture_weights"], 2))
```

A small `mean_component_separation` and a weight vector concentrated on one or two entries is collapse. `MDNModel` does not expose `diversity_regularizer_strength` in its constructor; set it on the layer before compiling, or build the head yourself with `MDNLayer`:

```python
model.mdn_layer.diversity_regularizer_strength = 0.01
model.compile(optimizer="adam")
```

### Pattern 2: Conditional mode sampling

Sample only from the most likely component when you want confident rather than diverse output.

```python
params = model.predict(X_test)
mu, sigma, pi_logits = model.mdn_layer.split_mixture_params(params)
pi = keras.activations.softmax(pi_logits, axis=-1)

idx = keras.ops.argmax(pi, axis=-1)
selected_mu = keras.ops.take_along_axis(mu, idx[:, None, None], axis=1)
selected_sigma = keras.ops.take_along_axis(sigma, idx[:, None, None], axis=1)
samples = selected_mu + selected_sigma * keras.random.normal(keras.ops.shape(selected_mu))
```

---

## 10. Performance Optimization

- **Mixed precision.** `keras.mixed_precision.set_global_policy('mixed_float16')` before construction. Watch `min_sigma`: a floor that underflows in float16 stops guarding the log-likelihood.
- **XLA.** Wrap sampling in `@tf.function(jit_compile=True)` when drawing many samples; the training step compiles through `model.compile(..., jit_compile=True)`.
- **Sampling cost.** `sample()` runs one forward pass and then draws `num_samples` per row, so raising `num_samples` is cheap relative to raising the batch.

---

## 11. Training and Best Practices

### Reading the loss

The loss is a negative log-likelihood, so it can be negative when the model is confident and correct — lower is always better, and there is no meaningful zero. It is volatile early, while components decide which mode to take.

If it goes to `NaN`, it is almost always `sigma` collapsing. `min_sigma` exists to prevent that, but an aggressive learning rate can still get there.

### Hyperparameter order

1. Start with `num_mixtures=3` and `hidden_layers=[32]`.
2. Train, then *sample and look*. Missing modes mean more mixtures or more capacity.
3. At high `num_mixtures`, run `check_component_diversity`. If components have collapsed, raise `diversity_regularizer_strength`.
4. Tune the learning rate last; `1e-3` with `ReduceLROnPlateau` is a solid default.

---

## 12. Serialization & Deployment

`MDNModel` registers as `dl_techniques.models.mdn.model>MDNModel` through `register_dl_technique` (`dl_techniques.utils.keras_registration`), so `.keras` archives load with no `custom_objects`. The legacy `Custom>MDNModel` alias the helper also binds keeps pre-2026-08-29 archives loading.

```python
model.save("mdn_model.keras")
loaded = keras.models.load_model("mdn_model.keras")
samples = loaded.sample(X_test, num_samples=1)
```

Because `MDNModel.compile()` is overridden, Keras does not restore the compile configuration on load and emits a `UserWarning` saying so. Call `loaded.compile(optimizer=...)` again before resuming training; inference and sampling work without it.

For deployment, remember the output is *not* a prediction — it is the parameter vector you then sample or summarize.

---

## 13. Testing & Validation

`tests/test_models/test_mdn/` is the real suite. A minimal set of shape checks:

```python
import keras
import numpy as np
from dl_techniques.models.time_series.mdn import MDNModel

def test_forward_pass_shape():
    model = MDNModel(hidden_layers=[16], output_dimension=2, num_mixtures=5)
    out = model(keras.random.normal(shape=(10, 4)))
    assert out.shape == (10, 25)          # 2 * 2 * 5 mu/sigma + 5 pi

def test_sample_shape():
    model = MDNModel(hidden_layers=[16], output_dimension=2, num_mixtures=5)
    samples = model.sample(keras.random.normal(shape=(10, 4)), num_samples=20)
    assert samples.shape == (10, 20, 2)   # (batch, num_samples, output_dim)

def test_split_shapes():
    model = MDNModel(hidden_layers=[16], output_dimension=1, num_mixtures=5)
    params = model.predict(np.zeros((7, 4), dtype="float32"))
    mu, sigma, pi_logits = model.mdn_layer.split_mixture_params(params)
    assert mu.shape == (7, 5, 1) and sigma.shape == (7, 5, 1)
    assert pi_logits.shape == (7, 5)
```

---

## 14. Failure Modes

- **Loss is NaN or infinite** — `sigma` collapsed. Lower the learning rate, raise `min_sigma`, or clip gradients with `Adam(clipnorm=1.0)`.
- **Only one mode is learned** — train longer (MDNs converge slowly), add capacity, or raise `num_mixtures`. Restarting from a different seed genuinely helps sometimes.
- **All components identical** — component collapse. Set `diversity_regularizer_strength` to something small like `0.01` (§9).
- **`UserWarning` about `compile()` on load** — expected, because `compile()` is overridden. Re-compile before further training.
- **Treating `model.predict(x)` as a forecast** — it returns the concatenated mixture parameters. Use `sample`, `get_point_estimate`, or `predict_with_uncertainty`.

---

## 15. Technical Details

### The MDN loss

Training minimizes the negative log-likelihood of the data under the mixture:

**Loss = -log P(y | x) = -log( Σᵢ πᵢ(x) · N(y | μᵢ(x), σᵢ(x)) )**

with the Gaussian density

**N(y | μ, σ) = (1 / (σ √(2π))) · exp( -(y - μ)² / (2σ²) )**.

Minimizing this is maximizing the probability that the model would have generated the training data. `loss_func` applies exactly one `log_softmax` to the `pi` logits and works in log-space throughout, which is why the raw-logit convention in §4.2 matters.

### Uncertainty decomposition

From the law of total variance, `Var[Y] = E[Var[Y|X]] + Var[E[Y|X]]`:

- **Point estimate**: `E[y|x] = Σᵢ πᵢ μᵢ`
- **Aleatoric variance**: `E[Var[y|x,θ]] = Σᵢ πᵢ σᵢ²` — the expected within-component variance.
- **Epistemic variance**: `Var[E[y|x,θ]] = Σᵢ πᵢ (μᵢ - E[y|x])²` — the spread of the component means about the overall mean.

The second term is what makes an MDN informative on ambiguous inputs: when two components sit far apart with comparable weight, the epistemic term is large and the point estimate is meaningless, and the model says so.

---

## 16. Citation

```bibtex
@techreport{bishop1994mixture,
  title={Mixture density networks},
  author={Bishop, Christopher M},
  year={1994},
  institution={Aston University}
}
```
