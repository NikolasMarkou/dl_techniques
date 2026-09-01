# TiRex: Time Series Forecasting with Mixed Sequential Architectures

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A **TiRex-inspired** probabilistic forecaster in Keras 3. Its defining feature is the **Mixed Sequential Block**, which lets each layer of the encoder be an LSTM, a Transformer, or a fused LSTM-then-attention hybrid. The model forecasts a set of quantiles rather than a point value.

---

## 1. Overview: What is TiRex and Why It Matters

### What is TiRex?

Rather than committing to one sequence processor, this model stacks configurable blocks. Each block is an LSTM, a Transformer, or a `mixed` block that runs both. The output head emits several quantiles at once, so every forecast comes with its own uncertainty band.

### Key Innovations

1. **Hybrid sequential blocks.** LSTMs bring an ordered, stateful inductive bias for local patterns; attention brings direct long-range comparison. `block_types` lets you interleave them per layer.
2. **Patch-based tokenization.** The series is cut into patches and embedded, shortening the sequence the attention blocks see and letting the model reason over temporal features rather than raw samples.
3. **Probabilistic output.** The head emits `len(quantile_levels)` values per horizon step, trained with the pinball loss.
4. **NaN tolerance.** A missingness mask is concatenated onto the feature axis before patch embedding, so gaps in the input do not have to be imputed first.

### Why TiRex Matters

```
ARIMA / ETS:   good on regular trend + seasonality, weak on non-linear dynamics.
Plain LSTM:    strong local ordering, struggles to recall distant events.
Transformer:   strong long-range content matching, no sequential bias of its own.

TiRex:         LSTM blocks summarize local state, transformer blocks relate those
               summaries across the whole window, and a quantile head reports the
               spread rather than a single number.
```

---

## 2. The Problem TiRex Solves

Real series carry patterns at more than one scale, and the two families of model are each good at one of them:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Local sequential structure                              │
│     how t depends on t-1, t-2 -- an LSTM's home ground.     │
│                                                             │
│  2. Global / long-range structure                           │
│     how today relates to the same holiday last year --      │
│     attention compares any two points directly.             │
└─────────────────────────────────────────────────────────────┘
```

Picking one architecture means conceding the other. This implementation lets you build the stack block by block — `lstm` layers first to summarize local trends, `transformer` layers after to relate those summaries, or `mixed` blocks to do both at every level — and it reports a distribution instead of a point, which is what risk decisions actually need.

---

## 3. How TiRex Works: Core Concepts

```
Input series ──► StandardScaler ──► NaN mask concat ──► PatchEmbedding1D
                                                             │
                                                    ┌────────▼───────────┐
                                                    │ Mixed Sequential   │  x N
                                                    │ Blocks (LSTM / TF) │
                                                    └────────┬───────────┘
                                                    output norm
                                                             │
                                                    mean-pool over patches
                                                             │
                                                    ┌────────▼───────┐
                                                    │  QuantileHead  │
                                                    └────────┬───────┘
                                                  (B, prediction_length, Q)
```

**Step 1 — preprocess and embed.** Z-score normalize, build a mask for missing values, concatenate data and mask (doubling the feature axis to `2F`), then `PatchEmbedding1D` turns the sequence into patch tokens. A `ResidualBlock` projects them to `embed_dim`.

**Step 2 — sequential processing.** N `MixedSequentialBlock`s, each shape-preserving, in the order given by `block_types`.

**Step 3 — forecast.** Output normalization, mean-pooling over the patch axis to a single `(B, 1, embed_dim)` vector, then the quantile head projects to `prediction_length * num_quantiles` and reshapes.

**The output is `(batch, prediction_length, num_quantiles)`** — horizon first, quantiles last. Slice `[..., i]` for the i-th quantile level.

---

## 4. Architecture Deep Dive

### 4.1 `StandardScaler`

Keras-native, invertible z-score normalization. Statistics are computed on the fly per batch, `NaN`s are handled, and `inverse_transform` maps outputs back to the data scale. Per-batch statistics make the model adaptive to non-stationary series in a way a fixed global scaler cannot be.

### 4.2 `PatchEmbedding1D`

A `Conv1D` whose `kernel_size` is `patch_size`; `strides` controls patch overlap (`stride < patch_size` gives overlapping patches). Its input carries the concatenated missingness mask, so the model can learn from where the gaps are.

### 4.3 `MixedSequentialBlock`

Three modes:

| `block_types` entry | Structure |
|:---|:---|
| `'lstm'` | Pre-norm LSTM, then FFN, both residual. |
| `'transformer'` | Pre-norm self-attention, then FFN, both residual. |
| `'mixed'` | Three stages: LSTM -> self-attention -> FFN, pre-norm and residual at each. |

The `mixed` block's premise is that attention operating on LSTM outputs sees a sequence that already carries local, stateful context, which makes the content-based comparison more meaningful.

### 4.4 `QuantileHead`

Takes the pooled feature vector, projects it with a single `Dense` to `num_quantiles * prediction_length`, and reshapes to `(batch, prediction_length, num_quantiles)`. Train it with the pinball loss.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.time_series.tirex.model import create_tirex_by_variant
from dl_techniques.losses.quantile_loss import QuantileLoss

SEQ_LEN, PRED_LEN = 128, 32
QUANTILES = [0.1, 0.5, 0.9]

def generate_data(n, seq_len, pred_len):
    X = np.zeros((n, seq_len, 1), dtype="float32")
    y = np.zeros((n, pred_len), dtype="float32")
    for i in range(n):
        start = np.random.rand() * 10
        t = np.linspace(start, start + seq_len + pred_len, seq_len + pred_len)
        series = np.sin(t)
        X[i, :, 0] = series[:seq_len]
        y[i, :] = series[seq_len:]
    return X, y

X_train, y_train = generate_data(1000, SEQ_LEN, PRED_LEN)
X_val, y_val = generate_data(200, SEQ_LEN, PRED_LEN)

model = create_tirex_by_variant(
    variant="tiny",
    input_length=SEQ_LEN,
    prediction_length=PRED_LEN,
    quantile_levels=QUANTILES,
)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=QuantileLoss(quantiles=QUANTILES, normalize=True),
)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)

# (batch, prediction_length, num_quantiles)
forecasts = model.predict(X_val[:1])
print(forecasts.shape)          # (1, 32, 3)

p10, p50, p90 = forecasts[0, :, 0], forecasts[0, :, 1], forecasts[0, :, 2]
```

To plot, fill between `p10` and `p90` and draw `p50` as the median line.

---

## 6. Component Reference

| Component | Location | Purpose |
|:---|:---|:---|
| `TiRexCore` | `models.time_series.tirex.model` | The main model. Also a `ForecastMixin`. |
| `create_tirex_model` | `models.time_series.tirex.model` | Build a fully custom configuration. |
| `create_tirex_by_variant` | `models.time_series.tirex.model` | Build one of the named variants (recommended). |
| `TiRexExtended` | `models.time_series.tirex.model_extended` | Subclass with a different prediction-head / token graph. |
| `create_tirex_extended` | `models.time_series.tirex.model_extended` | Variant factory for `TiRexExtended`. |
| `StandardScaler`, `PatchEmbedding1D`, `MixedSequentialBlock`, `QuantileHead` | `layers.embedding`, `layers.time_series` | The building blocks. |

The `tirex/` package `__init__.py` is intentionally empty, so import from `.model` / `.model_extended`. The `time_series` family package re-exports the three `TiRexCore` names, so `from dl_techniques.models.time_series import TiRexCore, create_tirex_by_variant` also works. `TiRexExtended` is not on that list — import it from its module.

```python
from dl_techniques.models.time_series.tirex.model import TiRexCore

model = TiRexCore.from_variant(
    "small",
    prediction_length=48,
    block_types=["lstm", "lstm", "mixed", "transformer"],
)
```

---

## 7. Configuration & Model Variants

| Variant | Patch Size | Embed Dim | Blocks | Heads | Dropout | Params @ L=64, H=16 |
|:---|:---:|:---:|:---:|:---:|:---:|---:|
| `tiny` | 8 | 64 | 3 | 4 | 0.10 | 342,608 |
| `small` | 12 | 128 | 6 | 8 | 0.10 | 2,528,272 |
| `medium` | 16 | 256 | 8 | 8 | 0.10 | 13,195,152 |
| `large` | 16 | 512 | 12 | 16 | 0.15 | 77,803,152 |

Parameter counts scale with `prediction_length` and `len(quantile_levels)` through the head, so treat the column as a size ordering, not a fixed figure.

### Customizing `block_types`

- `['lstm'] * N` — a deep residual LSTM stack.
- `['transformer'] * N` — a deep Transformer.
- `['lstm', 'lstm', 'transformer', 'transformer']` — local first, global second.
- `['mixed'] * N` — every block runs the full LSTM -> attention -> FFN pipeline. This is the default when `block_types` is not given.

Pass `num_blocks` to match the length of `block_types` when using `create_tirex_model`.

---

## 8. Comprehensive Usage Examples

### Example 1: Multivariate input

The architecture handles multiple input features, but the **build shape decides the feature width**, and `create_tirex_by_variant` builds at one feature. For `F > 1`, construct with `from_variant` and build explicitly:

```python
model = TiRexCore.from_variant("small", prediction_length=24)
model.build((None, 96, 5))          # 5 input features
out = model(np.random.randn(2, 96, 5).astype("float32"))
print(out.shape)                    # (2, 24, 9)  -- 9 default quantiles
```

Feeding a 5-feature batch to a model built at one feature raises
`Input 0 of layer "patch_embedding" is incompatible ... expected axis -1 ... to have value 2`
(2, because the mask doubles the feature axis).

### Example 2: Missing data

```python
x = np.random.randn(2, 96, 5).astype("float32")
x[0, 10:20, 0] = np.nan
out = model(x)
assert np.isfinite(np.asarray(out)).all()
```

The mask is built and concatenated inside `call`; nothing has to be imputed first.

### Example 3: The unified forecast contract

```python
forecast = model.predict_forecast(np.random.randn(2, 96, 5).astype("float32"))
forecast.point            # (2, 24)      median
forecast.quantiles        # (2, 24, 9)
forecast.quantile_levels  # [0.1, ..., 0.9]
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Tail-risk quantiles

```python
risk_quantiles = [0.01, 0.025, 0.5, 0.975, 0.99]
risk_model = create_tirex_by_variant(
    "medium", input_length=256, prediction_length=30,
    quantile_levels=risk_quantiles,
)
# Output: (B, 30, 5)
```

### Pattern 2: Extracting a subset of quantiles

`predict_quantiles` maps requested probability levels onto the trained head's indices and also returns the median as a point forecast. A requested level that was not trained falls back to the nearest trained one, with a warning.

```python
quantile_preds, point_preds = model.predict_quantiles(
    X_test, quantile_levels=[0.1, 0.5, 0.9], batch_size=64,
)
# quantile_preds: (B, prediction_length, 3)
# point_preds:    (B, prediction_length)
```

---

## 10. Performance Optimization

- **Mixed precision.** `keras.mixed_precision.set_global_policy('mixed_float16')` before construction; worthwhile from `medium` upward.
- **XLA.** The shipped trainer compiles with `jit_compile=True`, so it is a supported path: `model.compile(optimizer=..., loss=QuantileLoss(...), jit_compile=True)`.
- **`patch_size`.** Smaller patches see finer structure but lengthen the token sequence and so cost attention time quadratically. Larger patches are cheaper but smooth over local detail.

---

## 11. Training and Best Practices

### The quantile (pinball) loss

Use the shipped `dl_techniques.losses.quantile_loss.QuantileLoss`, which is vectorized over all quantiles:

```python
from dl_techniques.losses.quantile_loss import QuantileLoss
loss = QuantileLoss(quantiles=model.quantile_levels, normalize=True)
```

`normalize=True` divides each sample's loss by the mean absolute value of its target, so series at different magnitudes contribute equally to the gradient. This is what `train/time_series/tirex/train_tirex.py` does.

The label shape is `(batch, prediction_length)`; the loss broadcasts it against the model's `(batch, prediction_length, num_quantiles)` output.

### Other notes

- **Non-stationarity.** The built-in `StandardScaler` normalizes per batch, which adapts as the mean and variance drift. It does not protect against extreme outliers dominating a batch's statistics.
- **`block_types`.** For strongly seasonal data, leading with `transformer` blocks can help; for noisy trend data, leading with `lstm` blocks usually works better. It is cheap to try both.

---

## 12. Serialization & Deployment

`TiRexCore` registers through `register_dl_technique` (`dl_techniques.utils.keras_registration`) as `dl_techniques.models.tirex.model>TiRexCore`, and `TiRexExtended` as `dl_techniques.models.tirex.model_extended>TiRexExtended`. Archives written before the registration migration still load through the legacy `Custom>ClassName` alias the helper also binds.

```python
model.save("my_tirex_model.keras")
loaded = keras.models.load_model("my_tirex_model.keras")   # no custom_objects
```

Only a *custom* loss you wrote yourself needs `custom_objects` at load time, and only if you are reloading for further training; `QuantileLoss` is registered.

`build()` builds every sub-layer explicitly. That is required, not stylistic: on `.keras` load Keras restores weights before the first `call`, and an unbuilt sub-layer has nowhere to put them, so the first forward pass would silently re-initialize.

---

## 13. Testing & Validation

`tests/test_models/test_tirex/` is the real suite. A minimal shape check:

```python
import numpy as np
from dl_techniques.models.time_series.tirex.model import TiRexCore, create_tirex_by_variant

def test_all_variants_build():
    for variant in TiRexCore.MODEL_VARIANTS:
        create_tirex_by_variant(variant, input_length=64, prediction_length=16)

def test_forward_pass_shape():
    model = create_tirex_by_variant(
        "tiny", input_length=96, prediction_length=24,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    out = model.predict(np.random.rand(4, 96, 1).astype("float32"))
    assert out.shape == (4, 24, 3)      # (batch, horizon, quantiles)
```

---

## 14. Failure Modes

- **`Input 0 of layer "patch_embedding" is incompatible ...`** — the model was built at a different feature width than the batch you fed it. Build explicitly for `F > 1` (§8).
- **Indexing the output as `(batch, quantiles, horizon)`** — it is `(batch, horizon, quantiles)`. Slice the last axis for a quantile level.
- **Quantile crossing** (p90 below p50) — the pinball loss does not forbid it; it is common early in training and with little data. Sort the last axis as a post-process, or use a head that enforces monotonicity.
- **Loss flat** — check the learning rate first, then whether extreme outliers are dominating the per-batch scaler statistics, then whether the variant has enough capacity.

---

## 15. Technical Details

**LSTMs** carry a recurrent inductive bias: the hidden state is a compressed summary of the whole past, updated per step, at `O(L * D^2)`. Precise recall of distant events is where it strains.

**Transformers** are permutation-equivariant up to positional information: no ordering bias of their own, but direct content-based comparison between any two positions, at `O(L^2 * D)`.

The `mixed` block feeds LSTM output into self-attention, so attention operates on a sequence that already carries localized state. Patching keeps the `L^2` term small by shortening the token sequence before any attention runs.

---

## 16. Citation

```bibtex
@article{auer2025tirex,
  title={TiRex: Zero-Shot Forecasting Across Long and Short Horizons with
         Enhanced In-Context Learning},
  author={Auer, Andreas and Podest, Patrick and Klotz, Daniel and
          B{\"o}ck, Sebastian and Klambauer, G{\"u}nter and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2505.23719},
  year={2025}
}
```

This package implements the *architecture*. The published zero-shot results depend on large-scale pre-training over a diverse corpus of series, which is not shipped here: `TiRexCore.from_variant` has no `pretrained` argument, and no trained weights ship with the repository.
