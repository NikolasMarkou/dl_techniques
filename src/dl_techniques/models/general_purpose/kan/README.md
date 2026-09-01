# KAN: Kolmogorov-Arnold Networks

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **Kolmogorov-Arnold Network**, which puts the learnable nonlinearity on the *edges* of the graph rather than on its nodes. Each connection carries a B-spline over a knot grid; nodes do nothing but sum.

> **Read §11 before you train anything.** A freshly constructed KAN **cannot be trained as-is at the documented defaults**. `model.update_kan_grids(x_sample)` is part of setup, not tuning, and skipping it fails silently as a flat loss curve.

---

## 1. Overview: What is KAN and Why It Matters

An MLP fixes the activation function — ReLU, GELU, whatever — and learns only the linear maps between layers, so every unit in a layer applies the identical nonlinearity and all adaptation happens in the weights. A KAN inverts that: there are no weight matrices and no fixed activations, only a learnable univariate function on each connection.

The motivating result is the **Kolmogorov-Arnold representation theorem**: any continuous multivariate function can be written as a finite composition of continuous univariate functions and addition,

```
f(x_1, ..., x_n) = Σ_q Φ_q( Σ_p ψ_{q,p}(x_p) )
```

That is an *existence* statement about a two-layer construction, not a recipe for a deep network. Read it as the intuition behind the design rather than a guarantee about it.

What the inversion buys in practice is locality and legibility. Each edge function is shaped independently, so different regions of an input's range can be fitted without the global interference a single parameterized activation would suffer — and after training you can plot `φ_ij` and see what the model does with a feature.

---

## 2. The Problem KAN Solves

| | MLP | KAN |
|---|---|---|
| Where the nonlinearity lives | On nodes, fixed | On edges, learned |
| What is learned | Weight matrices | Spline control points per connection |
| Adapting to an odd function shape | Add width/depth until the fixed activation composes into it | Move control points on one edge |
| Inspecting a learned feature map | Not directly available | Plot `φ_ij(x)` |
| Cost per connection | One multiply | Basis evaluation + `grid_size + spline_order` coefficients |

The trade is explicit: KAN spends parameters and FLOPs per connection to buy shape flexibility and interpretability. It pays off where the underlying relationship is genuinely irregular — symbolic regression, PDE fitting, small scientific datasets — and does not where an MLP already fits.

---

## 3. How KAN Works: Core Concepts

### The model

A stack of `KANLinear` layers driven by a list of per-layer config dicts, with an optional final activation applied as its own `Activation` layer.

```
Input (B, F_in)
   │
   ├─► KANLinear  (layer 0)
   ├─► ...
   ├─► KANLinear  (layer N-1, forced to 'linear')
   │
   └─► Activation (e.g. softmax)  ──►  Output (B, F_out)
```

The last layer's `activation` is popped off and applied separately while the `KANLinear` itself is forced linear. Leaving it in place would apply the transform **twice** — a preset carrying `activation='softmax'` would softmax the edge outputs and then softmax their sum.

### Inside `KANLinear`

For every connection `i -> j`, the edge function is a spline plus a scaled base activation:

```
φ_ij(x_i) = base_scaler_ij * base(x_i)  +  spline_scaler_ij * Σ_k B_k(x_i) * C_ijk
y_j       = Σ_i φ_ij(x_i)
```

- `B_k` are the `grid_size + spline_order` B-spline basis functions over the knot grid, evaluated by the Cox-de Boor recursion.
- `C` is `spline_weight`, shape `(input_features, features, grid_size + spline_order)` — the control points.
- `spline_scaler` and `base_scaler` are per-connection `(input_features, features)` weights, both initialized to ones.
- The node is a plain sum over the input axis. There is **no bias vector**.

**The additive base activation matters more than it looks.** It keeps a well-conditioned gradient path through the edge in regions where the spline coefficients are still near zero, which is what makes the layer trainable from initialization rather than only after the splines have found signal.

### Why a spline, specifically

A B-spline basis function is nonzero only over `spline_order + 1` adjacent knot spans. Moving one control point therefore changes the edge function *only near that knot* — the property that makes an edge locally adjustable and makes `grid_size` a real capacity knob rather than a global smoothing parameter.

---

## 4. Architecture Deep Dive

### 4.1 The grid — the part a caller silently gets wrong

Splines are only defined over their knot range. The default grid is `grid_range=(-2.0, 2.0)`, set at construction from nothing but a guess about input scale. If the data occupies a different range, every edge spends its capacity on the wrong interval and extrapolates outside it.

The failure mode is **silence, not an error**. `KANLinear` sums over the input axis, so activations grow roughly 30x per layer and leave `(-2, 2)` after layer 0; the spline basis is then identically zero, and with `base_scaler` initialized to a constant the whole model collapses to a constant function.

Measured on the documented defaults: the output is exactly `1 / output_features` for every input with `std == 0.0`, and **0 of 12 trainable weights receive a non-zero gradient**. After `update_kan_grids` the same model has 12 of 12 live gradients.

`model.grids_adapted` exposes that state on the object, so it is readable rather than inferred from a flat loss curve. The constructor also logs a warning once.

### 4.2 `update_kan_grids(x_data)`

Runs a forward pass to collect the input distribution each layer actually sees — which is *not* the model input for anything past layer 0 — and quantile-matches each layer's knots to its own distribution. Hidden activations come out through a temporary extraction model built on the symbolic `layer.input` tensors, which is available because `KAN` is a Functional model.

Run it on a representative sample (roughly 100–1000 rows) before training on any new dataset. It sets `grids_adapted`.

### 4.3 The knot grid is a non-trainable weight

`grid` has shape `(grid_size + 2 * spline_order + 1,)` and is created with `trainable=False` and `autocast=False`, so it is saved and restored with the layer but never touched by gradient descent. `autocast=False` is deliberate: the grid is a coordinate table, and under `mixed_float16` the Cox-de Boor recursion would divide knot differences at half precision, where the `1e-7` epsilon default is subnormal.

---

## 5. Quick Start Guide

Learn `y = sin(pi * x1) + x2^2`:

```python
import keras
import numpy as np
from dl_techniques.models.general_purpose.kan.model import create_kan_model

def generate(n):
    X = (np.random.rand(n, 2) * 2 - 1).astype("float32")
    return X, np.sin(np.pi * X[:, 0]) + np.square(X[:, 1])

X_train, y_train = generate(2000)
X_val, y_val = generate(400)

model = create_kan_model(
    variant="micro",
    input_features=2,
    output_features=1,
    output_activation="linear",
)
print(model.grids_adapted)          # False

# NOT optional. See §4.1.
model.update_kan_grids(X_train[:100])
print(model.grids_adapted)          # True

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=20, batch_size=64, verbose=1)
```

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`KAN`** | `...kan.model.KAN` | Functional `keras.Model` stacking `KANLinear` layers. |
| **`create_kan_model`** | `...kan.model.create_kan_model` | Recommended factory; forwards to `from_variant`. |
| **`KAN.from_variant`** | `...kan.model.KAN.from_variant` | Build from a preset by name. |
| **`KAN.from_layer_sizes`** | `...kan.model.KAN.from_layer_sizes` | Build from a flat list of node counts, uniform per-layer config. |
| **`KAN.update_kan_grids`** | — | Re-fit every layer's knots to data. **Required before training.** |
| **`KAN.get_architecture_summary`** | — | Per-layer widths, grids, orders and activations, as a string. |
| **`KANLinear`** | `...layers.ffn.kan_linear.KANLinear` | The edge-function layer. Usable standalone. |

Both `KAN` and `create_kan_model` are exported from the package: `from dl_techniques.models.general_purpose.kan import KAN, create_kan_model`.

### `KANLinear` constructor

`features`, `grid_size=5`, `spline_order=3`, `grid_range=(-2.0, 2.0)`, `activation='swish'`, `base_trainable=True`, `spline_trainable=True`, `kernel_initializer='glorot_uniform'`, `base_scaler_initializer='ones'`, `epsilon=1e-7`.

**The base-activation keyword is `activation`, not `base_activation`.** An unrecognized key raises: `ValueError: Unrecognized keyword arguments passed to KANLinear: {'base_activation': 'gelu'}`.

---

## 7. Configuration & Model Variants

`KAN.VARIANT_CONFIGS` (aliased as `KAN.MODEL_VARIANTS` — the same object under both names):

| Variant | Hidden features | `grid_size` | `spline_order` | `activation` |
|:---|:---|:---:|:---:|:---:|
| **`micro`** | `[16, 8]` | 3 | 3 | `swish` |
| **`small`** | `[64, 32, 16]` | 5 | 3 | `swish` |
| **`medium`** | `[128, 64, 32]` | 7 | 3 | `gelu` |
| **`large`** | `[256, 128, 64, 32]` | 10 | 3 | `gelu` |
| **`xlarge`** | `[512, 256, 128, 64]` | 12 | 3 | `gelu` |

Grid resolution and width move together: a fine grid on a narrow layer tends to overfit, a coarse grid on a wide one wastes it.

`from_variant` expands `hidden_features` into one layer config per width, each inheriting the preset's grid/order/activation, then appends an output layer of `output_features`. When `output_activation` is omitted it defaults to `softmax` for `output_features > 1` and `linear` otherwise. The preset dict is copied first, so the class table is never mutated for later callers.

Override any preset field with `override_config`:

```python
model = create_kan_model(
    variant="medium", input_features=256, output_features=10,
    override_config={"grid_size": 5, "activation": "silu"},
)
```

### Pretrained weights

**None ship with `dl_techniques`.** `pretrained=True` raises `NotImplementedError` rather than warning and handing back a random model. Pass a local checkpoint instead: `pretrained="/path/to/weights.keras"`. `weights_dataset` and `weights_input_features` are only consulted on that path, to detect a head or input-width mismatch and skip the affected layer.

---

## 8. Comprehensive Usage Examples

### Example 1: MNIST

```python
import keras
from dl_techniques.models.general_purpose.kan.model import create_kan_model

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

model = create_kan_model("small", input_features=784, output_features=10,
                         output_activation="softmax")
model.update_kan_grids(x_train[:1000])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
```

### Example 2: A hand-written architecture

```python
from dl_techniques.models.general_purpose.kan.model import KAN

layer_configs = [
    {"features": 64, "grid_size": 8, "spline_order": 3, "activation": "gelu"},
    {"features": 32, "grid_size": 5, "spline_order": 2, "activation": "gelu"},
    # The last entry's `activation` becomes the model's final Activation layer.
    {"features": 10, "grid_size": 5, "activation": "softmax"},
]
model = KAN(layer_configs=layer_configs, input_features=784)
model.update_kan_grids(x_train[:500])
```

Every entry must be a dict with a `'features'` key; the rest are forwarded to `KANLinear` verbatim.

### Example 3: From node counts

```python
model = KAN.from_layer_sizes([784, 64, 32, 10], grid_size=5, activation="gelu")
```

`layer_sizes[0]` is the input width; every later entry is one layer, all sharing the same grid, order and activation.

---

## 9. Advanced Usage Patterns

### Visualizing a learned edge function

```python
import matplotlib.pyplot as plt
import numpy as np
import keras

layer = model.get_layer("kan_layer_0")
i, j = 0, 0                                    # input feature -> output feature

x = np.linspace(*layer.grid_range, 200).astype("float32")
xt = keras.ops.convert_to_tensor(x)

basis = layer._compute_bspline_basis(xt)       # (200, grid_size + spline_order)
spline = keras.ops.einsum('bk,k->b', basis, layer.spline_weight[i, j])
base = layer.base_activation_fn(xt)            # (200,)

phi = layer.base_scaler[i, j] * base + layer.spline_scaler[i, j] * spline

plt.plot(x, phi)
plt.xlabel("x_i"); plt.ylabel("phi_ij(x_i)")
plt.show()
```

A rank-1 sweep gets a rank-2 basis `(N, num_basis)`; the forward pass feeds rank 2 and gets rank 3 `(B, input_features, num_basis)`, contracted with `'...ik,iok->...io'`.

### `KANLinear` as a drop-in for `Dense`

```python
from dl_techniques.layers.ffn.kan_linear import KANLinear

x = keras.layers.Flatten()(conv_features)
x = KANLinear(features=128, grid_size=8, activation="gelu")(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
```

A hybrid model has no `update_kan_grids` method of its own — call `update_grid_from_samples(activations)` on each `KANLinear` with the activations it actually sees.

### Mixed precision and XLA

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model.compile(optimizer="adam", loss="mse", jit_compile=True)
```

The knot grid stays in full precision (`autocast=False`, §4.3).

---

## 10. Serialization & Deployment

`KAN` and `KANLinear` round-trip through the `.keras` format with no `custom_objects`. The knot grids are non-trainable weights, so they are saved and restored with everything else — a reloaded model keeps its adapted grids.

```python
model.save('my_kan.keras')
loaded = keras.models.load_model('my_kan.keras')
```

Note that `grids_adapted` is **not** part of `get_config`: a reloaded model reports `False` even though its grids are restored. Trust the weights, not the flag, after a load.

---

## 11. Training and Best Practices

- **Call `update_kan_grids` first.** Always. It is a precondition, not a tuning step (§4.1).
- **Re-adapt periodically.** Hidden-layer distributions drift as training proceeds:

  ```python
  class KANGridUpdate(keras.callbacks.Callback):
      def __init__(self, data, every=5):
          super().__init__(); self.data, self.every = data, every
      def on_epoch_end(self, epoch, logs=None):
          if (epoch + 1) % self.every == 0:
              self.model.update_kan_grids(self.data)

  model.fit(..., callbacks=[KANGridUpdate(X_train[:500])])
  ```
- **Start with a small `grid_size` (3–5).** It encourages smoother functions and holds the parameter count down; raise it only if you are underfitting.
- **Normalize your inputs.** The default grid assumes roughly `[-2, 2]`.

---

## 12. Testing & Validation

```python
import numpy as np
from dl_techniques.models.general_purpose.kan.model import create_kan_model

def test_forward_pass_shape():
    model = create_kan_model("micro", input_features=32, output_features=5)
    assert model.predict(np.random.rand(4, 32).astype("float32")).shape == (4, 5)

def test_grids_adapted_flag():
    model = create_kan_model("micro", input_features=8, output_features=1)
    assert model.grids_adapted is False
    model.update_kan_grids(np.random.rand(64, 8).astype("float32"))
    assert model.grids_adapted is True
```

The package suite is at `tests/test_models/test_kan/`, where the untrainable-without-grids state is pinned by an `xfail(strict=True)` pair.

---

## 13. Troubleshooting

**The loss curve is flat and predictions are constant.** You did not call `update_kan_grids`. Check `model.grids_adapted` (§4.1).

**`ValueError: Unrecognized keyword arguments passed to KANLinear`.** The base activation keyword is `activation`. `KANLinear` also takes no `kernel_regularizer` — regularize with the optimizer's weight decay instead.

**Training is slower than an MLP.** Expected: the basis evaluation is irregular and memory-bound, and the parameter count scales with `grid_size`. Shrink `grid_size`, narrow the network, or enable XLA. If speed dominates, `models/general_purpose/power_mlp/` trades the splines for `ReLU-k` and recovers dense GEMM-shaped compute.

**Overfitting.** Reduce `grid_size` first — the splines are what fit the noise. Then add dropout between layers.

---

## 14. Citation

```bibtex
@article{liu2024kan,
  title={{KAN}: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian
          and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y
          and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

Background:

- Kolmogorov, 1957. *On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition.* Dokl. Akad. Nauk SSSR 114.
- Girosi and Poggio, 1989. *Representation Properties of Networks: Kolmogorov's Theorem Is Irrelevant.* Neural Computation 1(4). — the standing counter-argument, worth reading beside the paper.
- de Boor, 1978. *A Practical Guide to Splines.* Springer.
