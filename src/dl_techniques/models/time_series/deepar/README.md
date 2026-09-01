# DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **DeepAR**: an autoregressive recurrent network trained across many related time series to produce *probabilistic* forecasts. It supports real-valued data (Gaussian likelihood) and count data (Negative Binomial likelihood), and handles series whose magnitudes differ by orders of magnitude.

---

## 1. Overview: What is DeepAR and Why It Matters

### What is DeepAR?

DeepAR learns a **single global model** from the histories of all series in a dataset, rather than fitting one model per series. At inference it does not emit a point value: it runs Monte-Carlo ancestral sampling to draw complete future trajectories, from which any quantile can be read off.

### Key Innovations

1. **Global learning.** Seasonality and trend patterns shared across items are learned once, so a new item with almost no history still gets a sensible forecast ("cold start").
2. **Scale handling.** A `ScaleLayer` divides each series by its own scale factor ν and multiplies the outputs back, so one set of weights serves an item selling 5 units/day and one selling 5,000.
3. **Ancestral sampling.** Sampling whole paths preserves the correlation between horizon steps. Marginal quantiles cannot do this, and the quantile of a sum is not the sum of the quantiles.
4. **Pluggable likelihood.** Gaussian for continuous targets, Negative Binomial for counts.

### Why DeepAR Matters

```
ARIMA / ETS:   one model per series, no information shared, hard to scale.
Plain LSTM:    predicts a single value, ignores uncertainty, fights input scale.

DeepAR:        RNN + likelihood head. Scale by mean history, emit distribution
               parameters, sample trajectories. Scalable, calibrated, and
               magnitude-agnostic.
```

---

## 2. The Problem DeepAR Solves

### Diverse scales

In retail sales or server load, series differ drastically in magnitude:

```
┌─────────────────────────────────────────────────────────────┐
│  Item A: ~100,000 units/day                                 │
│  Item B: ~5 units/day                                       │
│                                                             │
│  One weight matrix cannot serve both without normalization, │
│  and z-scoring is awkward when the forecast must come back  │
│  in the original domain.                                    │
└─────────────────────────────────────────────────────────────┘
```

DeepAR's answer is a *reversible* scale: divide by ν = mean(history) + ε on the way in, multiply by ν on the way out.

### The need for uncertainty

"Sales will be 50" is less useful than "there is a 90% chance sales land between 40 and 60". Inventory optimization needs the stock-out tail; capacity planning needs the 99th percentile, not the mean.

---

## 3. How DeepAR Works: Core Concepts

DeepAR has two distinct modes: **training** (teacher forcing) and **prediction** (autoregressive sampling). They are separate code paths — `call` is training only.

```
┌──────────────────────────────────────────────────────────────────┐
│  Inputs: [target, covariates]                                    │
│             │                                                    │
│    ┌────────▼────────┐     ┌──────────────────────────────────┐  │
│    │ ScaleLayer (ν)  │◄────┤ compute_scale (mean of history)  │  │
│    └────────┬────────┘     └──────────────────────────────────┘  │
│    ┌────────▼─────────┐                                          │
│    │ Stacked LSTMs    │                                          │
│    └────────┬─────────┘                                          │
│    ┌────────▼─────────┐                                          │
│    │ Likelihood head  │  (Gaussian or Negative Binomial)         │
│    └────────┬─────────┘                                          │
│  ┌──────────▼──────────┐                                         │
│  │ Distribution params │  {mu, sigma} or {mu, alpha}             │
│  └─────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────┘
```

**Training (teacher forcing).** The lagged *true* target `[0, z_1, ..., z_{T-1}]` is scaled, concatenated with the covariates, and run through the LSTM stack. The head emits per-step parameters; the NLL is the loss.

**Prediction (ancestral sampling).** For future steps there is no true `z_{t-1}`, so a value is *drawn* from the current predictive distribution and fed back as the next input. Repeat `num_samples` times to get `num_samples` possible futures.

---

## 4. Architecture Deep Dive

All three blocks live in `dl_techniques/layers/time_series/deepar_blocks.py`.

### 4.1 `ScaleLayer`

One multiplication or one division — the *choice* of scale belongs to the call site.

- Forward: `z_scaled = z / ν`, where `ν = mean(z_history) + scale_epsilon`.
- Inverse: multiply back to the original magnitude.
- Gaussian `mu` **and** `sigma` both use ν. The forward path divides z by ν exactly once, so sigma is a first-moment-scale quantity, not a variance — no square root. Only the negative-binomial shape `alpha` uses `1/√ν`.

### 4.2 `GaussianLikelihoodHead`

Projects the LSTM state to `mu` (affine) and `sigma` (affine + softplus, so positive). The loss is the Gaussian NLL of the observations under those parameters.

### 4.3 `NegativeBinomialLikelihoodHead`

For counts. Projects to mean `mu` and shape `alpha`, a Gamma-Poisson mixture that models overdispersion (variance > mean) — the normal case for sales data.

`DeepARCell` also lives in that module; the model itself uses standard `keras.layers.LSTM` stacks for speed.

---

## 5. Quick Start Guide

```python
import numpy as np
import tensorflow as tf
from dl_techniques.models.time_series.deepar.model import create_deepar
from train.time_series.deepar.train_deepar import DeepARTrainingWrapper

B, T, COND, D, C = 32, 100, 80, 1, 5
t = np.linspace(0, 100, T)
target = (np.sin(t / 10) + np.random.normal(0, 0.1, (B, T, D))).astype("float32")
covariates = np.random.normal(0, 1, (B, T, C)).astype("float32")

# 1. Build. `covariate_dim` only sizes the dummy build pass.
#    `conditioning_length` is important: without it the scale is computed over
#    the whole window, leaking the horizon into the normalizer.
base = create_deepar(
    num_layers=2,
    hidden_dim=40,
    likelihood="gaussian",
    num_samples=100,
    covariate_dim=C,
    conditioning_length=COND,
)

# 2. Train through the wrapper (see §11 — `compile(loss=...)` cannot work here).
model = DeepARTrainingWrapper(base=base)
model.compile(optimizer="adam", loss=None)

ds = tf.data.Dataset.from_tensor_slices(
    {"target": target, "covariates": covariates}
).batch(8)
model.fit(ds, epochs=10)

# 3. Sample futures. Shape: (num_samples, batch, horizon, target_dim)
samples = base.predict({
    "conditioning_target": target[:, :COND, :],
    "full_covariates": covariates,        # must cover history AND horizon
})
print(samples.shape)   # (100, 32, 20, 1)

# 4. Or get the unified Forecast object straight away.
forecast = base.predict_forecast({
    "conditioning_target": target[:, :COND, :],
    "full_covariates": covariates,
})
print(forecast.point.shape)           # (32, 20, 1)
print(forecast.quantile_levels)       # [0.1, 0.5, 0.9]
lo, hi = forecast.interval(0.1, 0.9)
```

---

## 6. Component Reference

| Component | Location | Purpose |
|:---|:---|:---|
| `DeepAR` | `models.time_series.deepar.model` | The `keras.Model`. Also a `ForecastMixin`. |
| `create_deepar` | `models.time_series.deepar.model` | Factory; returns an already-built model. |
| `DeepARTrainingWrapper` | `train.time_series.deepar.train_deepar` | Wraps the NLL via `add_loss` so `fit` works. |
| `ScaleLayer` | `layers.time_series.deepar_blocks` | Reversible per-series normalization. |
| `GaussianLikelihoodHead` | `layers.time_series.deepar_blocks` | Emits `mu`, `sigma`. |
| `NegativeBinomialLikelihoodHead` | `layers.time_series.deepar_blocks` | Emits `mu`, `alpha`. |

The `deepar/` package `__init__.py` is intentionally empty, so import from `.model`. The `time_series` family package does re-export the public names, so `from dl_techniques.models.time_series import DeepAR, create_deepar` also works.

---

## 7. Configuration & Likelihoods

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `num_layers` | int | 3 | Stacked LSTM layers. |
| `hidden_dim` | int | 40 | Units per LSTM layer. |
| `likelihood` | str | `'gaussian'` | `'gaussian'` or `'negative_binomial'`. |
| `target_dim` | int | 1 | Target feature width. |
| `dropout_rate` | float | 0.0 | LSTM output dropout. |
| `recurrent_dropout_rate` | float | 0.0 | Recurrent dropout inside the cells. |
| `num_samples` | int | 100 | Monte-Carlo paths drawn at prediction time. |
| `scale_epsilon` | float | 1.0 | Added to the mean so ν is never 0. |
| `conditioning_length` | int or None | None | Steps used to compute ν. `None` warns and uses the full window. |

### Choosing a likelihood

- **Gaussian** — continuous data: temperature, voltage, large aggregate sales.
- **Negative Binomial** — counts, especially sparse series with many zeros or small integers. It handles overdispersion and zero-inflation that a Gaussian smears over.

---

## 8. Comprehensive Usage Examples

### Example 1: Count data

```python
base = create_deepar(
    num_layers=2,
    hidden_dim=64,
    likelihood="negative_binomial",
    covariate_dim=5,
    conditioning_length=80,
)
model = DeepARTrainingWrapper(base=base)
model.compile(optimizer="adam", loss=None)
model.fit(count_dataset, epochs=5)
```

The wrapper picks the matching NLL from `base.likelihood`; there is nothing else to switch.

### Example 2: The two input dictionaries

Training and prediction take **different keys**. This is the most common source of confusion.

```python
# Training / call:  teacher-forced window
{"target": (B, T, D), "covariates": (B, T, C), "scale": optional (B, 1, D)}

# Prediction:       history + covariates that also cover the horizon
{"conditioning_target": (B, T_cond, D),
 "full_covariates":     (B, T_cond + T_pred, C),
 "scale":               optional (B, 1, D)}
```

The horizon length is inferred as `full_covariates.shape[1] - conditioning_target.shape[1]`.

---

## 9. Advanced Usage Patterns

### Pattern 1: Pre-computed scales

Pass `'scale'` in the input dict to override the computed ν — useful when domain knowledge gives a better normalizer (e.g. a yearly average) than the context window does.

```python
my_scales = np.mean(train_data, axis=1, keepdims=True) + 1.0
model.fit(tf.data.Dataset.from_tensor_slices(
    {"target": train_data, "covariates": cov, "scale": my_scales}).batch(8))
```

### Pattern 2: Correlated targets

`target_dim > 1` predicts several variables per series jointly through one LSTM stack.

```python
base = create_deepar(target_dim=3, hidden_dim=64, covariate_dim=5)
# data shape: (batch, time, 3)
```

---

## 10. Performance Optimization

- **Batch size.** DeepAR learns global patterns, so it likes large batches (128-256). Small batches give noisy gradients.
- **Covariates.** Time features — hour of day, day of week, `is_holiday` — are close to mandatory. Without them the LSTM loses seasonality over long sequences.
- **Sampling cost.** Prediction runs a Python-level autoregressive loop `num_samples` times. Drop `num_samples` to 10-20 during development and raise it for the final run.
- **XLA.** `jit_compile=True` on the training wrapper works well with the LSTM stack.

---

## 11. Training and Best Practices

### `compile(loss=DeepAR.gaussian_loss)` does not work — use the wrapper

`DeepAR.call` returns a *dict* `{'mu', 'sigma', 'target'}`, and the static NLLs read the ground truth from `y_pred['target']`. Keras' `compile(loss=...)` path tries to match the loss against the dataset labels, so it raises `KeyError: "The path: ('mu',) ... can't be found"`. The supported route is `DeepARTrainingWrapper`, which computes the NLL inside `call` and registers it with `add_loss`:

```python
model = DeepARTrainingWrapper(base=base)
model.compile(optimizer="adam", loss=None)   # loss=None is correct
```

Feed it a `tf.data.Dataset` yielding the input dict. A bare dict-of-arrays as `x=` is mis-structured by Keras and fails.

### Other notes

- **Teacher forcing.** Training feeds the true `z_{t-1}`, never the model's own output. This is why training is fast and prediction is slow.
- **`conditioning_length`.** Always set it. Left at `None` the model logs a warning and computes ν over the full teacher-forced window, which leaks the horizon into the normalizer and creates train/serve skew.
- **`scale_epsilon`.** With many zero-runs keep it around 1.0 so ν is not dominated by noise; for data on the order of 0.001, lower it.

---

## 12. Serialization & Deployment

The model registers through `register_dl_technique` (`dl_techniques.utils.keras_registration`) as `dl_techniques.models.deepar.model>DeepAR`. The legacy `Custom>DeepAR` alias that helper also binds keeps pre-migration archives loading.

```python
base.save("deepar_model.keras")
loaded = keras.models.load_model("deepar_model.keras")   # no custom_objects needed
```

The registration covers the model and its blocks, so nothing has to be passed by hand. `DeepARTrainingWrapper` serializes its base via `get_config`/`from_config` and round-trips the same way.

---

## 13. Testing & Validation

`tests/test_models/test_deepar/` covers the `get_config` round trip, the `_forecast` sample-to-quantile contract, and the wrapper's `add_loss` behaviour. A minimal shape check:

```python
import numpy as np
from dl_techniques.models.time_series.deepar.model import DeepAR

model = DeepAR(num_layers=1, hidden_dim=10, likelihood="gaussian",
               num_samples=4, conditioning_length=10)

x = {"target": np.random.normal(size=(4, 20, 1)).astype("float32"),
     "covariates": np.random.normal(size=(4, 20, 5)).astype("float32")}

out = model(x, training=True)                    # training mode -> dict
assert set(out) == {"mu", "sigma", "target"}
assert out["mu"].shape == (4, 20, 1)

samples = model.predict({"conditioning_target": x["target"][:, :10, :],
                         "full_covariates": x["covariates"]})
assert samples.shape == (4, 4, 10, 1)            # (samples, batch, horizon, dim)
```

Note that sampling is reached through `predict` (which routes to `predict_step`), never by passing a flag to `call`.

---

## 14. Failure Modes

- **`KeyError: "The path: ('mu',) ..."`** — you compiled the base model with a loss. Use `DeepARTrainingWrapper` and `loss=None` (§11).
- **Loss is NaN** — exploding gradients or a near-zero ν. Clip (`Adam(clipnorm=1.0)`) and raise `scale_epsilon`.
- **Flat-line predictions** — the LSTM never learned the autoregressive link. Normalize the covariates, check the target is not pure noise, raise `hidden_dim`.
- **Shape error in the LSTM at `fit` time** — a raw dict was passed as `x=`. Wrap the inputs in a `tf.data.Dataset`.
- **Mis-calibrated intervals that scale with the series size** — `conditioning_length` is `None`, so ν is computed over the horizon too.

---

## 15. Technical Details

### Autoregressive recurrent network

The value at time t depends on the previous value, the previous hidden state, and the current covariates:

$$ h_t = \text{LSTM}(h_{t-1}, z_{t-1}, x_t), \qquad P(z_t \mid h_t) = \theta(h_t) $$

### Monte-Carlo sampling

At inference `z_{t-1}` is unknown, so it is drawn:

$$ \tilde{z}_t \sim P\big(z \mid \theta(\tilde{h}_t)\big) $$

and fed back as the input for step t+1. Repeating this N times gives N whole futures. `DeepAR` is the only model on the repo's `Forecast` contract that populates `Forecast.samples`, because it is the only sampler.

---

## 16. Citation

```bibtex
@article{salinas2017deepar,
  title={DeepAR: Probabilistic forecasting with autoregressive recurrent networks},
  author={Salinas, David and Flunkert, Valentin and Gasthaus, Jan and Januschowski, Tim},
  journal={International Journal of Forecasting},
  volume={36},
  number={3},
  pages={1181--1191},
  year={2020},
  publisher={Elsevier}
}
```
