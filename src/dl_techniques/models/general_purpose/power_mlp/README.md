# PowerMLP: A ReLU-k Alternative to Kolmogorov-Arnold Networks

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **PowerMLP**, a dual-branch feedforward network that replaces KAN's B-spline edge functions with `ReLU-k` activations, recovering dense GEMM-shaped compute. This implementation has not been trained or benchmarked here; no accuracy or speed number is claimed for it.

---

## 1. Overview: What is PowerMLP and Why It Matters

Kolmogorov-Arnold Networks move the nonlinearity from the nodes to the edges: each connection carries its own learned univariate function, parameterized as a B-spline over a grid. That is what gives KAN its expressiveness per parameter, and it is also what makes it slow — every forward pass must locate each input in the spline grid and evaluate the basis polynomials there, an irregular, memory-bound computation that does not reduce to a matrix multiply.

PowerMLP starts from the observation that the **shape** KAN buys with splines can be approximated by a fixed nonlinearity of the right *order* combined with a linear map. The layer is two branches summed:

```
y = ReLU_k(W_main @ x + b)  +  W_basis @ swish(x)
```

Both branches are ordinary dense projections with elementwise activations, so the whole model is GEMM-shaped end to end.

---

## 2. The Problem PowerMLP Solves

| | MLP | KAN | PowerMLP |
|---|---|---|---|
| Nonlinearity | Fixed, degree 1 piecewise | Learned spline per edge | Fixed, degree `k` piecewise |
| Curvature per layer | Needs depth to accumulate | Arbitrary, per connection | Degree `k` in one layer |
| Compute shape | Dense GEMM | Irregular basis evaluation | Dense GEMM |
| What is learned | `W` | Spline control points | `W_main`, `W_basis` |

A single `ReLU-k` layer can bend where a ReLU layer would need several to approximate the same curvature. That is the same degree of freedom KAN gets from its splines — but as an elementwise power on an already-projected vector, so it costs a `pow` rather than a grid lookup.

---

## 3. How PowerMLP Works: Core Concepts

### The dual-branch layer

```
                Input (..., input_dim)
                  ╱                ╲
      Main branch                   Basis branch
      Dense(units) → ReLU_k         BasisFunction → Dense(units, use_bias=False)
                  ╲                ╱
                   Element-wise add
                          │
                Output (..., units)
```

**Branch ordering is easy to state backwards.** In the main branch the dense map comes *first* and the power is applied to its output, so `k` acts on learned features, not on raw inputs.

**The basis branch applies `swish(x) = x * sigmoid(x)` to the input and then projects it.** Swish is smooth, non-monotonic and unbounded above, which makes it a complementary shape to `ReLU_k`: it is nonzero for negative inputs, where `ReLU_k` is identically zero and its gradient vanishes. Summing the two means a unit is never fully dead — whatever the main branch gates off, the basis branch still passes a signal and a gradient.

The basis projection is deliberately **bias-free**: both branches carrying a bias would be redundant, and the main branch already has one.

Note what the basis branch is *not*. It is a **stateless** activation followed by a linear map; it adds no learned nonlinearity of its own, unlike a KAN edge function. All the learning in that branch lives in `W_basis`. This is the trade PowerMLP makes, and it is why it is faster rather than merely cheaper.

### The model

```
Input (B, hidden_units[0])
    │
    ├─► PowerMLPLayer(hidden_units[1], k)  ─► [BatchNorm] ─► [Dropout]
    ├─► ...
    ├─► PowerMLPLayer(hidden_units[-2], k) ─► [BatchNorm] ─► [Dropout]
    │
    └─► Dense(hidden_units[-1], output_activation)  ─► (B, hidden_units[-1])
```

**`hidden_units` is read as `[input_dim, hidden_1, ..., hidden_n, output_dim]`.** The first entry describes the expected input width rather than creating a layer, and the last sizes the output. Optional batch normalization and dropout are applied after each hidden layer, in that order.

**The output layer is a plain `Dense`, not a PowerMLP layer.** The final map needs an arbitrary activation — softmax, sigmoid, or none for regression — and `ReLU_k` on the logits would clamp them non-negative and destroy the parameterization every downstream loss expects.

---

## 4. Architecture Deep Dive

### 4.1 `ReLU-k`

Computes `max(0, x) ** k` elementwise, where `k` is a **fixed integer hyperparameter validated as such at construction** — it is not learned. `k=1` is plain ReLU; `k > 1` makes the branch piecewise-polynomial of degree `k` for positive inputs.

The preset variants raise `k` with model size (2 for micro, 3 through base, 4 for the two largest), since higher-degree units are only worth their conditioning cost when there is enough width to use them. Large `k` sharpens the activation's gradient near the origin and grows its outputs fast, which is why batch normalization becomes worth enabling as `k` rises.

### 4.2 `BasisFunction`

A stateless layer computing `x / (1 + exp(-x))` — i.e. swish / SiLU. It has no weights and no training-mode behaviour. It does **not** expand the feature set into a Fourier-style basis; the name describes its role in the paper's decomposition, not a dimensionality change.

### 4.3 `save_model` refuses an unbuilt model

`PowerMLP.__init__` takes no input shape, so a model is only built once something traces it. Calling `model.save_model(path)` before that **raises `ValueError`** rather than writing a file:

```
Cannot save an unbuilt PowerMLP: the archive would contain zero weights.
```

That refusal is deliberate. A silent `self.save()` on an unbuilt model produces a syntactically valid `.keras` archive holding **zero** weights, and `load_model()` hands it back as a zero-weight model with only a `UserWarning` in between. Build first — one `predict`, one `fit` step, or an explicit `model.build(shape)`.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.general_purpose.power_mlp.model import PowerMLP

model = PowerMLP.from_variant(
    "tiny",
    num_classes=10,
    input_dim=784,
    output_activation="softmax",
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

x = np.random.randn(16, 784).astype("float32") * 0.1   # normalize your inputs!
y = np.random.randint(0, 10, 16)

loss, acc = model.train_on_batch(x, y)
print(model.predict(x).shape)      # (16, 10)
print(f"{model.count_params():,}") # 104,874
model.summary()                    # only valid once built
```

**Normalize your inputs.** `x^k` amplifies scale: on raw unit-variance noise at `k=3` the first training loss above lands in the tens of thousands. `StandardScaler`-style zero-mean/unit-variance input is not optional advice here.

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`PowerMLP`** | `...power_mlp.model.PowerMLP` | The `keras.Model`. |
| **`PowerMLP.from_variant`** | — | Build a preset; wraps `hidden_units` as `[input_dim] + preset + [num_classes]`. |
| **`create_power_mlp`** | `...power_mlp.model.create_power_mlp` | Create **and compile** in one call. |
| **`create_power_mlp_regressor`** | — | Pins `loss='mse'`, `metrics=['mae','mse']`, linear output. |
| **`create_power_mlp_binary_classifier`** | — | Pins `loss='binary_crossentropy'`, sigmoid output, accuracy/precision/recall. |
| **`PowerMLPLayer`** | `...layers.ffn.power_mlp_layer.PowerMLPLayer` | The dual-branch layer. |
| **`ReLUK`** | `...layers.activations.relu_k.ReLUK` | `max(0, x)^k`. |
| **`BasisFunction`** | `...layers.activations.basis_function.BasisFunction` | Swish. |

All four model-level names are exported from the package.

Model methods beyond the Keras surface: `save_model(filepath, overwrite=True)` (see §4.3; `save_format` is accepted and ignored — Keras 3 picks the format from the extension) and the classmethod `load_model(filepath)`.

### `create_power_mlp` default-loss derivation

`loss=None` (the default) **derives** a categorical cross-entropy whose `from_logits` matches the model's actual `output_activation`:

| `output_activation` | Derived loss |
|---|---|
| `None` (default, linear) | `CategoricalCrossentropy(from_logits=True)` |
| `'softmax'` / `'sigmoid'` | `CategoricalCrossentropy(from_logits=False)` |

A fixed `loss="categorical_crossentropy"` string would compile with `from_logits=False` against the linear default, feeding unnormalized real values to a cross-entropy that renormalizes by `output / sum(output)` and clips — finite, meaningless, silent. Pass an explicit `loss=` to override.

---

## 7. Configuration & Model Variants

`PowerMLP.MODEL_VARIANTS`:

| Variant | Hidden units | `k` |
|:---|:---|:---:|
| **`micro`** | `[32, 16]` | 2 |
| **`tiny`** | `[64, 32]` | 3 |
| **`small`** | `[128, 64, 32]` | 3 |
| **`base`** | `[256, 128, 64]` | 3 |
| **`large`** | `[512, 256, 128]` | 4 |
| **`xlarge`** | `[1024, 512, 256, 128]` | 4 |

`from_variant(variant, num_classes, input_dim, **kwargs)` builds `hidden_units = [input_dim] + preset + [num_classes]`. Any other constructor argument overrides the preset:

```python
model = PowerMLP.from_variant("large", num_classes=1, input_dim=100,
                              k=5, batch_normalization=True)
```

**`hidden_units` is refused by name in `from_variant`.** It used to be accepted and then silently overwritten by the variant's own list. For a custom architecture use the constructor directly: `PowerMLP(hidden_units=[...], k=...)`.

Constructor arguments: `hidden_units`, `k=3`, `kernel_initializer='he_normal'`, `bias_initializer='zeros'`, `kernel_regularizer=None`, `bias_regularizer=None`, `use_bias=True`, `output_activation=None`, `dropout_rate=0.0`, `batch_normalization=False`, `name='power_mlp'`.

---

## 8. Comprehensive Usage Examples

### Example 1: Flattened-image classification

```python
model = PowerMLP.from_variant(
    "small", num_classes=10, input_dim=3072,      # CIFAR-10 flattened
    output_activation="softmax",
    dropout_rate=0.2, batch_normalization=True,
)
model.compile(optimizer="adamw", loss="categorical_crossentropy",
              metrics=["accuracy"])
# model.fit(x_train_flat, y_train_onehot, ...)
```

### Example 2: Regression

```python
from dl_techniques.models.general_purpose.power_mlp.model import create_power_mlp_regressor

# [input_dim, hidden..., output_dim] — 100 features in, 1 target out.
model = create_power_mlp_regressor(
    hidden_units=[100, 256, 128, 1], k=4,
    learning_rate=1e-3, batch_normalization=True,
)
model.predict(x_val)      # builds it
model.summary()
```

Pre-compiled with `mse` and a linear output, so the `create_power_mlp` derivation does not apply.

### Example 3: Binary classification

```python
from dl_techniques.models.general_purpose.power_mlp.model import create_power_mlp_binary_classifier

model = create_power_mlp_binary_classifier(
    hidden_units=[128, 64, 1], k=3, dropout_rate=0.3,
)
```

The last entry should be `1`; anything else logs a warning rather than raising.

### Example 4: As a CNN classification head

```python
x = keras.layers.Flatten()(cnn_base.output)
head = PowerMLP(hidden_units=[x.shape[-1], 256, 128, 10], output_activation="softmax")
model = keras.Model(inputs=cnn_base.input, outputs=head(x))
```

---

## 9. Advanced Usage Patterns

### Choosing `k`

| `k` | Behaviour |
|:---:|---|
| 1 | Main branch is plain ReLU. Still dual-branch, mildest nonlinearity. Safe starting point. |
| 2–3 | Good defaults. Strong balance of expressive power and stability; `base` uses 3. |
| >= 4 | Sharp, high-degree activations with fast-growing outputs. **Enable `batch_normalization`.** |
| >= 5 | Add `clipnorm` to the optimizer as well. |

### Mixed precision

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model = PowerMLP.from_variant("large", num_classes=10, input_dim=1024)
```

Both branches are standard `Dense` layers and elementwise ops, so there is nothing precision-hostile in the graph.

---

## 10. Training and Best Practices

- **Normalize inputs.** The single most important item on this list (§5).
- **Optimizer**: AdamW; the weight decay is a real regularizer here.
- **Schedule**: cosine decay, optionally with a short linear warmup.
- **Batch norm for `k >= 3`.** It controls the scale of the powered activations.
- **Gradient clipping for `k >= 5`**: `keras.optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0)`.
- **Start from `base`** and move outward.

---

## 11. Serialization & Deployment

```python
model.predict(sample)            # ensure it is built — see §4.3
model.save_model('my_powermlp.keras')
loaded = PowerMLP.load_model('my_powermlp.keras')
```

`save_model` creates the parent directory if needed. `PowerMLP.load_model` supplies the custom objects; a plain `keras.models.load_model` also works via the registry. `build()` materializes the sub-layers by tracing `call()` on symbolic inputs, so an explicit `model.build(shape)` — and the `build_from_config` step of deserialization — leave the model *actually* built rather than merely marked built.

---

## 12. Testing & Validation

```python
import numpy as np, tempfile, os, pytest
from dl_techniques.models.general_purpose.power_mlp.model import PowerMLP

def test_serialization_cycle():
    model = PowerMLP.from_variant("micro", num_classes=5, input_dim=100)
    data = np.random.randn(4, 100).astype("float32")
    original = model(data, training=False)      # also builds it

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.keras")
        model.save_model(path)
        loaded = PowerMLP.load_model(path)

    np.testing.assert_allclose(original, loaded(data, training=False),
                               rtol=1e-6, atol=1e-6)

def test_unbuilt_save_is_refused():
    with pytest.raises(ValueError):
        PowerMLP.from_variant("micro", num_classes=5, input_dim=100).save_model("x.keras")
```

The package suite is at `tests/test_models/test_power_mlp/`.

---

## 13. Troubleshooting

**The loss explodes on the first step.** Unnormalized inputs, almost always (§5). Then: lower `k`, enable `batch_normalization`, lower the learning rate, add `clipnorm`.

**`ValueError: Cannot save an unbuilt PowerMLP`.** Correct behaviour — build the model first (§4.3).

**`ValueError: from_variant(...) cannot honour a caller-supplied hidden_units`.** Use the constructor for a custom architecture (§7).

**`model.summary()` raises.** The model is not built yet. Run one forward pass.

**Cross-entropy loss looks meaningless.** Check `from_logits` against your `output_activation` — or let `create_power_mlp` derive it (§6).

---

## 14. Citation

PowerMLP is positioned against KAN; both should be cited when the comparison is the point.

```bibtex
@article{liu2024kan,
  title={{KAN}: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian
          and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

Components: swish (Ramachandran et al., 2017, [arXiv:1710.05941](https://arxiv.org/abs/1710.05941)) and batch normalization (Ioffe and Szegedy, 2015, [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)).
