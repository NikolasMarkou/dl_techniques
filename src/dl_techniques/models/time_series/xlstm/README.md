# xLSTM: Extended Long Short-Term Memory

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **xLSTM** (Beck et al., 2024), which revisits the LSTM with exponential gating and a matrix memory. The package ships two models with **distinct contracts**: `xLSTM`, a language model over integer tokens `[B, T]`, and `xLSTMForecaster`, a continuous time-series forecaster over `[B, T, F]`.

---

## 1. What is xLSTM?

LSTMs were foundational to sequence modelling and were then largely displaced by Transformers, for two structural reasons: their memory is a vector of limited capacity, and their recurrence blocks parallel training.

xLSTM attacks both. It introduces two new cell variants and stacks them into a hybrid model:

- **sLSTM** keeps scalar memory but adds exponential gating and a normalizer state, which makes it far better at *revising* what it has stored.
- **mLSTM** replaces the memory vector with a matrix updated by an outer product, which gives it attention-like capacity and, by dropping hidden-to-hidden recurrence, full parallelizability during training.

### Why xLSTM?

| Classical LSTM limitation | xLSTM answer |
|:---|:---|
| Scalar memory of limited capacity | Matrix memory `C_t` in the mLSTM |
| Recurrence blocks parallel training | mLSTM abandons hidden-to-hidden recurrence |
| Hard to scale to long sequences | Stack mLSTM for reach, sLSTM for state tracking |

---

## 2. Key Innovations

### 2.1 sLSTM (Scalar LSTM)

**Exponential gating.** The input gate uses `exp` instead of a sigmoid:

```
i_t = exp(W_i x_t + R_i h_{t-1} + b_i)
```

An unbounded multiplicative gate lets the cell make dramatic revisions to what it stores, where a sigmoid caps the update at 1.

**Normalizer state.** Exponential gating on its own is numerically hopeless, so a second state `n_t` tracks the accumulated gate mass and divides it back out:

```
n_t = f_t * n_{t-1} + i_t
h_t = o_t * (c_t / n_t)
```

The result is stable memory dynamics with an unbounded gate. What sLSTM buys is **memory revision** and **long-horizon state tracking**; it stays recurrent and processes step by step.

### 2.2 mLSTM (Matrix LSTM)

**Matrix memory.** The cell state becomes a matrix `C_t ∈ R^(key_dim × value_dim)` rather than a vector, so it can hold key-value associations the way attention does.

**Covariance update.** The update is an outer product:

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
```

Retrieval is then content-based: query the matrix with `q_t`.

**Full parallelizability.** mLSTM has no hidden-to-hidden recurrence, so an entire sequence can be processed in parallel during training, like a Transformer. That is what makes it efficient on GPUs and TPUs. It is also multi-head.

### 2.3 The hybrid architecture

```
Input → Embedding → [mLSTM blocks] × N → [sLSTM blocks] × M
      → Final normalization → Output head
```

`mlstm_ratio` sets the split: higher means more parallelism and better reach on long sequences, lower means more recurrence and better complex state tracking.

---

## 3. About This Implementation

The code integrates with the `dl_techniques` factories and follows the repo's Keras 3 serialization conventions. It has not been trained or benchmarked here; no performance claim is made for it. No trained weights ship: asking `from_variant` for a pretrained model raises `NotImplementedError` by design.

### 3.1 Components

| Name | Location | Role |
|:---|:---|:---|
| `sLSTMCell` | `layers.time_series.xlstm_blocks` | Recurrent cell with exponential gating. |
| `sLSTMLayer` | `layers.time_series.xlstm_blocks` | RNN wrapper for sequence processing. |
| `mLSTMCell` | `layers.time_series.xlstm_blocks` | Matrix-memory cell. |
| `mLSTMLayer` | `layers.time_series.xlstm_blocks` | RNN wrapper for the matrix cell. |
| `sLSTMBlock` | `layers.time_series.xlstm_blocks` | Residual block, post-normalization. |
| `mLSTMBlock` | `layers.time_series.xlstm_blocks` | Residual block, pre-up-projection. |
| `xLSTM` | `models.time_series.xlstm.model` | Language model over token ids. |
| `xLSTMForecaster` | `models.time_series.xlstm.forecaster` | Continuous forecaster; a `ForecastMixin`. |

### 3.2 Framework integration

- **Normalization factory** `create_normalization_layer()` — see §6.1 for the working values.
- **FFN factory** `create_ffn_layer()` — the factory offers seven types, but the xLSTM blocks can only construct `swiglu`; see §6.2.
- **Serialization** — every class implements `get_config()` and registers with `@register_dl_technique`. The keys are `dl_techniques.models.xlstm.model>xLSTM`, `dl_techniques.models.xlstm.forecaster>xLSTMForecaster`, and `dl_techniques.layers.time_series.xlstm_blocks><ClassName>` for the six cells, layers and blocks. Pre-2026-08-29 archives still load through the legacy `Custom>ClassName` alias the helper also binds.
- **Keras 3 discipline** — layers created in `__init__`, weights in `build()`, proper RNN Cell/Layer infrastructure, `keras.ops` throughout, full type hints, masking and state handling, and validation that raises on bad configurations.

---

## 4. Quick Start

### The language model

```python
import keras
import numpy as np
from dl_techniques.models.time_series.xlstm import xLSTM

model = xLSTM(
    vocab_size=10000,
    embed_dim=512,
    num_layers=12,
    mlstm_ratio=0.5,        # 6 mLSTM blocks, 6 sLSTM blocks
)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

tokens = np.random.randint(0, 10000, (4, 256))
print(model(tokens).shape)     # (4, 256, 10000) -- logits per position

model.fit(train_dataset, validation_data=val_dataset, epochs=10)
model.save("xlstm_model.keras")
loaded = keras.models.load_model("xlstm_model.keras")   # no custom_objects
```

### The forecaster

```python
from dl_techniques.models.time_series.xlstm import create_xlstm_forecaster

model = create_xlstm_forecaster(
    input_length=64, prediction_length=24, num_features=1,
    embed_dim=64, num_layers=2,
)

x = np.random.randn(8, 64, 1).astype("float32")
print(model(x).shape)              # (8, 24, 9) -- 9 default quantile levels
print(list(model.quantile_levels)) # [0.1, 0.2, ..., 0.9]
```

Set `use_quantile_head=False` for a plain point forecast of shape `(B, prediction_length, num_features)`.

### A fuller configuration

```python
model = xLSTM(
    vocab_size=50000, embed_dim=768, num_layers=24,
    mlstm_ratio=0.5, mlstm_num_heads=12, mlstm_expansion_factor=2,
    slstm_forget_gate="exp",                    # 'sigmoid' or 'exp'
    ffn_type="swiglu", ffn_expansion_factor=4,
    normalization_type="rms_norm", normalization_kwargs={"epsilon": 1e-6},
    dropout_rate=0.1, embedding_dropout_rate=0.1,
    kernel_regularizer=keras.regularizers.L2(1e-4),
)
```

---

## 5. Architecture Components

All six building blocks import from `dl_techniques.layers.time_series.xlstm_blocks`.

### 5.1 sLSTM cell and layer

```python
from dl_techniques.layers.time_series.xlstm_blocks import sLSTMCell, sLSTMLayer

cell = sLSTMCell(units=128, forget_gate_activation="exp")   # 'sigmoid' or 'exp'
state = cell.get_initial_state(batch_size=32)               # [h_0, c_0, n_0, m_0]
output, new_state = cell(keras.random.normal((32, 64)), state)

outputs = sLSTMLayer(units=128, return_sequences=True)(inputs)   # (B, T, 128)

layer = sLSTMLayer(units=128, return_sequences=False, return_state=True)
output, h, c, n, m = layer(inputs)                          # five tensors
```

Exponential gating, the normalizer state, stabilization against overflow, and the standard Keras RNN API (masking, stateful mode, `Bidirectional`) all apply.

### 5.2 mLSTM cell and layer

```python
from dl_techniques.layers.time_series.xlstm_blocks import mLSTMCell, mLSTMLayer

cell = mLSTMCell(units=256, num_heads=4)
# key_dim / value_dim default to units // num_heads
rnn = keras.layers.RNN(cell, return_sequences=True)

layer = mLSTMLayer(units=256, num_heads=8, key_dim=64, return_sequences=True)
```

Matrix memory, multi-head processing, the covariance update rule, content-based retrieval, and parallel training.

`units` must be divisible by `num_heads`; the constructor raises otherwise. `key_dim` is free — it sizes the query/key space only. **`value_dim` is not**: it must stay at its `units // num_heads` default, and a mismatch is not caught at construction — it surfaces as an `InvalidArgumentError` inside `mLSTMCell.call` on the first forward pass.

### 5.3 sLSTM block

Residual block with post-normalization, Transformer-style:
`Input → sLSTMLayer → Normalization → FFN → Add(residual)`.

```python
from dl_techniques.layers.time_series.xlstm_blocks import sLSTMBlock

block = sLSTMBlock(
    units=256,
    ffn_type="swiglu",               # the only working value; see 6.2
    ffn_expansion_factor=4,          # default 2
    normalization_type="rms_norm",   # default 'layer_norm'
    normalization_kwargs={"epsilon": 1e-6},
    forget_gate_activation="exp",
    dropout_rate=0.1,
)
```

### 5.4 mLSTM block

Residual block with pre-up-projection, SSM-style:
`Input → Dense up-projection → causal depthwise Conv1D → swish → mLSTMLayer → Normalization → Dense down-projection → Add(residual)`.

```python
from dl_techniques.layers.time_series.xlstm_blocks import mLSTMBlock

block = mLSTMBlock(
    units=256, expansion_factor=2, num_heads=8,
    conv_kernel_size=4,              # default 4
    normalization_type="rms_norm",
)
```

### 5.5 The full models

```
xLSTM                                 xLSTMForecaster
─────                                 ───────────────
Token ids (B, T)                      Series (B, T, F)
   ↓ Embedding + dropout                 ↓ Input projection (+ optional instance norm)
[mLSTM blocks] × num_layers*ratio     [mLSTM blocks] × num_layers*ratio
[sLSTM blocks] × the remainder        [sLSTM blocks] × the remainder
   ↓ Final normalization                 ↓ Final normalization
   ↓ Dense head → (B, T, vocab)          ↓ quantile head → (B, H, Q)
                                            or point head → (B, H, F)
```

`xLSTMForecaster` carries the repo's `ForecastMixin`, so `predict_forecast(x)` returns the unified `Forecast` object and `predict_quantiles(x, quantile_levels=[...])` returns `(quantile_preds, point_preds)` with shapes `(B, H, len(levels))` and `(B, H)`.

---

## 6. Configuration Options

### 6.1 Normalization types

| Type | Notes |
|:---|:---|
| `'layer_norm'` | Default, general purpose. |
| `'rms_norm'` | Cheaper than LayerNorm; the usual choice at scale. |
| `'batch_norm'` | CNN-style; rarely right for sequences. |
| `'band_rms'` | Band-constrained RMS, for stability-critical training. |
| `'adaptive_band_rms'` | Dynamic band control. |
| `'dynamic_tanh'` | Normalization-free Transformer style. |

Pass options through `normalization_kwargs`, e.g. `{'epsilon': 1e-6}`.

### 6.2 FFN types

`sLSTMBlock` builds its FFN through `create_ffn_layer(ffn_type=..., output_dim=units, ffn_expansion_factor=..., dropout_rate=...)`. **In practice only `'swiglu'` is usable**, because it is the only factory entry that accepts `ffn_expansion_factor` in place of an explicit `hidden_dim`:

| Type | Status |
|:---|:---|
| `'swiglu'` | Works. Gated SwiGLU, as in LLaMA. The block default, and the only supported value. |
| `'mlp'`, `'geglu'`, `'glu'`, `'differential'`, `'residual'`, `'swin_mlp'` | Raise `ValueError: Required parameters missing ... ['hidden_dim']` at construction. |

The same applies to `xLSTM(ffn_type=...)`, which forwards straight to `sLSTMBlock`. Leave `ffn_type` at its default unless you are also widening the factory call.

### 6.3 Regularizers and initializers

`kernel_regularizer`, `recurrent_regularizer` and `bias_regularizer` are separately configurable, as are `kernel_initializer` (`'glorot_uniform'`), `recurrent_initializer` (`'orthogonal'`) and `bias_initializer` (`'zeros'`). All round-trip through `get_config`.

---

## 7. Usage Examples

### 7.1 Language modelling

AdamW with weight decay, plus the usual `EarlyStopping` / `ReduceLROnPlateau` / `ModelCheckpoint` trio, is a solid default; see §4 for the construction call.

```python
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[...])
```

### 7.2 Sequence classification from the layers

```python
from dl_techniques.layers.time_series.xlstm_blocks import sLSTMLayer, mLSTMLayer

inputs = keras.Input(shape=(None, 128))
x = mLSTMLayer(256, num_heads=4, return_sequences=True)(inputs)
x = sLSTMLayer(256, return_sequences=True)(x)
x = mLSTMLayer(256, num_heads=4, return_sequences=False)(x)   # pool by not returning sequences
x = keras.layers.Dropout(0.3)(x)
model = keras.Model(inputs, keras.layers.Dense(10, activation="softmax")(x))
```

### 7.3 A custom hybrid from the blocks

Front-load the fast parallel blocks, finish with the recurrent ones.

```python
from dl_techniques.layers.time_series.xlstm_blocks import sLSTMBlock, mLSTMBlock

inputs = keras.Input(shape=(None, 256))
x = inputs
for i in range(4):
    x = mLSTMBlock(256, expansion_factor=2, num_heads=8, name=f"frontend_mlstm_{i}")(x)
for i in range(4):
    x = sLSTMBlock(256, ffn_expansion_factor=4, forget_gate_activation="exp",
                   name=f"backend_slstm_{i}")(x)
model = keras.Model(inputs, keras.layers.Dense(vocab_size)(x))
```

Alternating `mLSTMBlock` and `sLSTMBlock` in a loop gives a multi-scale temporal model instead — fast blocks for quick patterns, recurrent blocks for long dependencies.

### 7.4 Size variants

Both models ship a `MODEL_VARIANTS` table and a `from_variant` constructor.

| Model | Variants |
|:---|:---|
| `xLSTM` | `small` (embed 256, 6 layers, 4 heads), `base` (512, 12, 8), `large` (1024, 24, 16, ffn expansion 4) |
| `xLSTMForecaster` | `tiny` (embed 64, 2 layers, 4 heads), `small` (128, 4, 8) |

```python
model = xLSTM.from_variant("small", vocab_size=50000)
fc = xLSTMForecaster.from_variant("tiny", input_length=64, prediction_length=24)
```

`vocab_size` is required for `xLSTM` (it is not part of the variant dict), and `input_length` / `prediction_length` must be supplied as overrides for the forecaster. Asking either `from_variant` for pretrained weights raises `NotImplementedError`: no trained weights ship.

---

## 8. Serialization

```python
model.save("xlstm_model.keras")
loaded = keras.models.load_model("xlstm_model.keras")   # no custom_objects needed

model.save_weights("xlstm_weights.weights.h5")
fresh = xLSTM(vocab_size=..., embed_dim=..., num_layers=...)
fresh.load_weights("xlstm_weights.weights.h5")

config = model.get_config()          # JSON-serializable
rebuilt = xLSTM.from_config(config)  # architecture only, no weights
```

Every layer registers itself, so a `.keras` archive is self-describing. `save_weights` requires the receiving model to be built at the same architecture; `get_config`/`from_config` carries the architecture but no weights.

---

## 9. Performance Tips

### 9.1 Use mLSTM for long training sequences

mLSTM processes the whole sequence in parallel, so it dominates wall-clock time on long training sequences: `mlstm_ratio=0.75` or higher.

### 9.2 Use sLSTM for state tracking

Autoregressive generation is step-by-step whatever you choose, and sLSTM's explicit recurrence and memory revision are what pay off on tasks that need long-range state. Lower `mlstm_ratio` there.

### 9.3 Balance the block ratio

`0.5` balanced (the default), `0.75` parallelization-heavy for long training sequences, `0.33` recurrence-heavy for complex state tracking.

### 9.4 Use RMS normalization

`normalization_type='rms_norm'` is cheaper than LayerNorm in both time and memory and is the usual choice for large stacks.

### 9.5 Enable mixed precision

```python
keras.mixed_precision.set_global_policy("mixed_float16")

model = xLSTM(vocab_size=50000, embed_dim=512, num_layers=12)
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy")
```

`mixed_float16` is supported. The four stability floors in `xlstm_blocks.py` — the two `log` guards and the two divide guards inside `sLSTMCell.call` and `mLSTMCell.call` — are sized so that they remain strictly greater than zero once materialized in the compute dtype. A bare `1e-8` literal does not: `float16(1e-8)` is exactly `0.0`, which turns the guard into a no-op and produces `NaN` from the divides and an infinite gradient from the logs. None of that is visible in float32, so `tests/test_layers/test_time_series/test_the_xlstm_floors_survive_fp16.py` drives each gate into saturation deliberately.

### 9.6 Batch size

mLSTM-heavy models parallelize better and tolerate larger batches (64+); recurrence-heavy stacks are usually happier around 32.

---

## 10. Testing

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_xlstm/ -q
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_layers/test_time_series/ -q -k xlstm
```

| Suite | Covers |
|:---|:---|
| `tests/test_models/test_xlstm/` | Model construction, config round trips, forecaster contract. |
| `test_xlstm_blocks.py` | Cells, layers and blocks: shapes, state return, serialization. |
| `test_the_xlstm_floors_survive_fp16.py` | The four stability floors under `mixed_float16` (§9.5). |
| `test_multivariate_denorm.py` | Forecaster denormalization on multivariate targets. |

---

## 11. Architecture Details

### 11.1 sLSTM formulation

```
i_t = exp(W_i x_t + R_i h_{t-1} + b_i)        # input gate (exponential)
f_t = σ(W_f x_t + R_f h_{t-1} + b_f)          # forget gate (sigmoid or exp)
o_t = σ(W_o x_t + R_o h_{t-1} + b_o)          # output gate
z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)       # cell input
```

**Stabilization** (paper eqs. 15-17) — the stabilizer state `m_t` keeps the exponentials in range, then the states update and the normalizer divides back out:

```
m_t = max(m_{t-1} + log(f_t), log(i_t))
ĩ_t = exp(log(i_t) - m_t)
f̃_t = exp(log(f_t) + m_{t-1} - m_t)

c_t = f̃_t ⊙ c_{t-1} + ĩ_t ⊙ z_t              # cell state
n_t = f̃_t ⊙ n_{t-1} + ĩ_t                    # normalizer state
h_t = o_t ⊙ (c_t / (n_t + ε))                 # hidden state
```

The `log(f_t)` and the `n_t + ε` divide are two of the four floors §9.5 discusses.

### 11.2 mLSTM formulation

```
q_t = W_q x_t + R_q h_{t-1}                   # query
k_t = W_k x_t + R_k h_{t-1}                   # key
v_t = W_v x_t + R_v h_{t-1}                   # value

i_t = exp(W_i x_t + R_i h_{t-1})              # input gate (exponential)
f_t = σ(W_f x_t + R_f h_{t-1})                # forget gate
o_t = σ(W_o x_t + R_o h_{t-1})                # output gate

C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)     # covariance update
n_t = f_t ⊙ n_{t-1} + i_t ⊙ k_t               # normalizer
h_t = o_t ⊙ (C_t q_t / (n_t^T q_t + ε))       # retrieval
```

### 11.3 Against a standard LSTM

| Feature | Standard LSTM | sLSTM | mLSTM |
|---|---|---|---|
| Memory type | Scalar vector | Scalar vector | Matrix |
| Gating | Sigmoid | Exponential | Exponential + sigmoid |
| Normalizer / stabilization | No | Yes | Yes |
| Parallel training | No | No | Yes |
| Memory capacity | Limited | Enhanced | High |
| Compute cost | Low | Medium | Higher |
| Storage revision | Limited | Excellent | Good |
| Long-range reach | Moderate | Good | Excellent |

---

## 12. Requirements

Python >= 3.11, Keras >= 3.8.0, TensorFlow >= 2.18.0 (or torch, or jax). The implementation depends on the `dl_techniques` normalization factory (`layers.norms`) and FFN factory (`layers.ffn`); it is not standalone.

---

## 13. Citation

```bibtex
@article{beck2024xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and
          Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and
          Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```
