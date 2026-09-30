# HKAN: Hierarchical Kolmogorov-Arnold Network

A Keras 3 model for single-target regression over `(B, n_in)` features. Its weights can be
fitted layer by layer by least squares, with no gradient (the paper's training), and the same
weights can also be trained by a stock optimizer through `compile()` and `fit()`.

The method is from Dudek and Rodak, "HKAN: Hierarchical Kolmogorov-Arnold Network without
Backpropagation", 2025, [arXiv:2501.18199](https://arxiv.org/abs/2501.18199). The code here was
written from the paper's equations. Nothing was copied from the authors' reference code
(`rodakt/hkan`), which carries no licence file.

**Location.** The package is `dl_techniques.models.general_purpose.hkan`, not
`dl_techniques.models.hkan`: this repository does not allow a leaf package directly under
`models/`. The Keras registration keys strip the family directory and are
`dl_techniques.models.hkan.model>HKAN` and `dl_techniques.models.hkan.hkan_layer>HKANLayer`.

---

## 1. The model

A Kolmogorov-Arnold network puts a learned one-dimensional function on every edge. HKAN fixes
the non-linear parameters of those functions (the centers and the slope) and keeps only the
linear ones, so every edge function can be fitted by one linear regression.

For a layer with `n_in` inputs, `n_out` outputs and `m = num_basis` basis functions (`q`
indexes outputs, `p` inputs, `r` basis functions):

```text
block (expanding stage)   phi[q, p](x_p) = sum_r coef[q, p, r] * g(slope * (x_p - centers[q, p, r]))
                                           + block_bias[q, p]
connecting stage          h[q]           = sum_p mix[q, p] * phi[q, p] + bias[q]
```

One `HKANLayer` is one paper layer: the expanding stage followed by the connecting stage. An
`HKAN` is a list of them; the outputs `h` of one layer are the inputs of the next, and the last
layer has width 1, so the model maps `(B, n_in)` to `(B, 1)`.

```text
x (B, n_in) -> HKANLayer 0 (n_in -> hidden_units[0]) -> ... -> HKANLayer L (hidden_units[-1] -> 1) -> y_hat (B, 1)
```

Basis functions `g`, chosen by name: `sigmoid`, `gaussian` (`exp(-z^2)`), `relu`, `tanh`,
`softplus`, `identity`.

`identity` ignores `slope`. It is `g(d) = d` on the unscaled difference `x_p - center`, as in
the paper, where the slope belongs to the non-linear basis functions only. A model-wide
`slope=50` with an identity top layer is therefore the paper's model; multiplying the identity
features by 50 would divide that layer's effective ridge strength by 2500. With more than one
basis function `identity` is degenerate: every feature is `x_p` minus a constant, so
`num_basis` only rescales the ridge strength.

### Weights

Each layer creates its weights in `build()` as stacked tensors. Per layer, in the order
`model.weights` lists them:

| Weight | Shape | Trainable | Present |
|---|---|---|---|
| `coef` | `(n_out, n_in, m)` | yes | always |
| `block_bias` | `(n_out, n_in)` | yes | when `use_block_bias` |
| `mix` | `(n_out, n_in)` | yes | always |
| `bias` | `(n_out,)` | yes | when `use_bias` |
| `centers` | `(n_out, n_in, m)` | no | always |

So a layer has 5 weights with both intercepts on (the default) and 3 with both off; a model
with one hidden layer has 10 or 6.

**The centers are fixed in every training mode.** `centers` is non-trainable and stays
bit-identical through `fit()`. Trainable centers and a trainable slope are out of scope.

---

## 2. Three ways to train

The three examples below run as printed, one after the other.

### 2.1 Closed form (the paper's training)

`fit_closed_form(x, y, chunk_size=None)` trains without gradients. Layer by layer, every block
is a ridge regression of its `m` basis features onto the target `y`, then every output is a
linear regression of its `n_in` block outputs onto the same `y`. The solve runs in float64
numpy whatever the model dtype, and the result is cast into the Keras weights. An unbuilt model
is built from `x` first.

```python
import numpy as np
from dl_techniques.models.general_purpose.hkan import HKAN

rng = np.random.default_rng(0)
x_train = rng.uniform(0.0, 1.0, size=(500, 2))
y_train = np.sin(3.0 * x_train[:, 0]) * x_train[:, 1]
x_test = rng.uniform(0.0, 1.0, size=(200, 2))

model = HKAN(
    hidden_units=(16,),
    num_basis=(10, 5),
    basis=("tanh", "identity"),
    slope=5.0,
    l2_block=0.01,
    seed=0,
)
diagnostics = model.fit_closed_form(x_train, y_train)

print(diagnostics["layer_rmse"])               # list of 2 floats, one per layer
print(diagnostics["block_r2"].shape)           # (16, 2): first-layer blocks
print(diagnostics["importance"].shape)         # (2,): mean of block_r2 over the outputs
print(diagnostics["train_predictions"].shape)  # (500,), float64

y_hat = model.predict(x_test, verbose=0)       # (200, 1)
```

The returned dict has exactly those four keys. `importance` is the paper's equation 14.
`train_predictions` are the float64 fit's own; a forward pass of the model reproduces them only
to the accuracy of the model dtype (section 5).

`y` may have shape `(N,)` or `(N, 1)`. A `y` with more than one column raises `ValueError`:
every block regresses onto one target, so the model is single-target. `chunk_size` is the
number of outputs of a layer solved at once; it bounds memory and does not change the result.
`l2_block == 0` selects minimum-norm least squares, `l2_block > 0` a ridge solve; an intercept
is fitted by centering, so the penalty never touches it.

If the solve of a layer fails, the exception propagates (non-finite features were checked by
running: numpy raises `LinAlgError`; a solution that comes back non-finite raises a
`ValueError` naming the layer). Earlier layers keep their newly fitted weights, and the `coef`,
`block_bias`, `mix` and `bias` of the failing layer and of the later ones are left as they
were. With `centers="data"` the failing layer's centers have already been redrawn at that
point.

Use this when you want the paper's procedure: one pass, no learning rate, no epochs.

### 2.2 Stock `compile()` and `fit()`

`coef`, `block_bias`, `mix` and `bias` are ordinary trainable weights, and the package defines
no `train_step`, `test_step` or `predict_step`. `coef` starts from `random_normal` and `mix`
from the constant `1 / n_in`, so the initial state gives every trainable weight a gradient.
`l2_block` and `l2_mix` have no effect here.

```python
import keras

backprop_model = HKAN(hidden_units=(16,), num_basis=10, basis="tanh", slope=5.0,
                      centers="data", seed=0)
backprop_model.initialize_centers(x_train)
backprop_model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
history = backprop_model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
print(history.history["loss"])
```

`initialize_centers(x)` is needed only with `centers="data"` and no closed-form fit (section 3).

The paper has no gradient-only procedure. This path exists because a configurable
backpropagation option was required; its initial scale is not tuned.

### 2.3 Closed form, then `fit()`

The closed-form fit never optimizes one stage jointly with another. A gradient phase afterwards
trains all stages together, starting from the closed-form solution.

```python
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse")
loss_before = model.evaluate(x_train, y_train, verbose=0)
history = model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
print(loss_before, history.history["loss"])
```

`fit_closed_form` may also be called after `compile()`; the assigned weights are what the next
`evaluate` or `fit` sees.

Whether the gradient phase lowers the error depends on the problem and is not claimed here. In
one CPU run of this example the train loss from `evaluate` was 0.0064 before the gradient phase
and 0.0065 after its three epochs: it did not lower it.

---

## 3. Constructor

```python
HKAN(
    hidden_units=(),          # widths of the hidden layers; () is a one-layer model
    num_basis=10,
    basis="sigmoid",
    slope=1.0,
    centers="random",         # "random", "equally_spaced" or "data"
    l2_block=0.0,
    l2_mix=0.0,
    use_block_bias=True,
    use_bias=True,
    coef_initializer="random_normal",
    mix_initializer=None,     # None selects the constant 1 / n_in
    seed=None,
)
```

`num_basis`, `basis`, `slope`, `centers`, `l2_block` and `l2_mix` take one value for all layers
or a list or tuple with one value per layer, `len(hidden_units) + 1` long. A wrong length, an
unknown basis or centers name, a non-positive width or slope, or a negative ridge strength
raises `ValueError` from the constructor.

**Centers.** `random`: uniform on `[0, 1]`. `equally_spaced`: `linspace(0, 1, num_basis)` for
every block. `data`: for every block `(q, p)`, `num_basis` values sampled with replacement from
column `p` of that layer's own input. The `data` draw happens inside `fit_closed_form`, or in
`initialize_centers(x)` for a gradient-only run; until one of the two runs, the weight holds a
uniform draw as a placeholder. `seed` seeds the centers only. `seed=0` is a seed like any
other; `seed=None` draws from the global numpy generator.

**Factory.** `create_hkan(hidden_units=(), input_dim=None, **kwargs)` forwards to the
constructor and, when `input_dim` is given, builds the model so its weights exist before the
first call.

```python
from dl_techniques.models.general_purpose.hkan import create_hkan

built = create_hkan(hidden_units=(5,), input_dim=3, num_basis=(7, 4), seed=0)
print(len(built.weights), built.count_params())   # 10 296
```

**No variants table, no pretrained weights.** The paper defines no named sizes, so there is no
`MODEL_VARIANTS` table and no `from_variant`. No pretrained weights are distributed and the
constructor has no parameter for them; an unknown keyword raises `ValueError` from Keras.

**Dtype policies.** float32 and float64 are supported. `mixed_float16` is not supported and not
tested: float16 has no solve kernel and a slope of 50 saturates it.

---

## 4. Serialization

`get_config()` carries every constructor argument. Fitted values are not in the config; they
travel in the weights of the `.keras` archive.

```python
import os
import tempfile

with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "hkan.keras")
    model.save(path)
    loaded = keras.models.load_model(path)

assert all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip(model.weights, loaded.weights))
assert np.array_equal(model.predict(x_test, verbose=0), loaded.predict(x_test, verbose=0))
```

---

## 5. Differences from the paper and from the reference code

- **Intercepts are on by default** (`use_block_bias=True`, `use_bias=True`). The paper's
  equations 5, 6 and 13 have none; the authors' tutorial notebook fits them. The paper's form is
  `use_block_bias=False, use_bias=False`.
- **A hidden layer of width 1 is allowed.** The reference code raises on it.
- **Centers are seeded** by the `seed` argument. The reference code uses the global numpy
  generator.
- **`l2_mix`** adds a ridge penalty to the connecting stage. Its default is 0, which is the
  paper's plain least squares.
- **Backprop-only runs with `centers="data"`**: `initialize_centers` draws a deeper layer's
  centers from the model's forward activations at its initial weights. This is a choice made
  here; the paper defines the data draw only inside its layer-by-layer fit.
- **The closed-form training is a method, `fit_closed_form`**, not an override of `fit`, so
  `keras.Model.fit` stays the stock gradient loop.
- **The forward pass is float32 by default**, while the fit is float64 (section 6).
- **The path** is `models/general_purpose/hkan/`, not `models/hkan/` (see the top of this file).

---

## 6. Measured facts

Every number in this section was measured on this code on 2026-09-30. The derivations are in
`plans/plan-2026-09-30T082355-4d999dbc/decisions.md` (D-022, D-023, D-024).

**Agreement with scikit-learn (reproducible from the repo).** Per block and per output, in
float64, the closed-form solve is compared with `sklearn.linear_model.Ridge` for a positive
ridge and `LinearRegression` for zero ridge, for six basis functions, both intercept settings
and distinct and duplicated centers:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_models/test_hkan/test_closed_form.py -q -k sklearn
```

364 passed. Coefficients agree to a relative 1e-9 and values to 1e-9 absolute, except in the
eight `sigmoid` and `softplus` zero-ridge cells, whose block designs have condition numbers
above 1e6 and which are asserted at 1e-6.

**Agreement with the authors' reference code (measured once, not reproducible from the repo).**
The reference code is not in this repository, so the probe ran against a scratch copy of it,
with this model's centers injected. Over 44 cells (N = 400, `n_in = 3`, hidden width 5,
`num_basis = (7, 4)`, slope 5; six basis functions in both layers, five non-linear ones under
an identity top layer, both block-intercept settings, ridge 0 and 0.01), the largest absolute
difference between the two sets of float64 train predictions was 6.6e-10 at zero ridge and
1.5e-11 at positive ridge. The command was
`CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python <scratchpad>/hkan_probe/parity_upstream.py`;
the script and the reference copy lived in a session scratch directory and no committed test
re-runs it.

**Float32 forward pass against the float64 fit.** On the authors' tutorial configuration
(5000 rows of the paper's TF5 function generated here, `hidden_units=(912,)`,
`num_basis=(23, 10)`, `basis=("tanh", "identity")`, `slope=50.0`, `centers=("random", "data")`,
`l2_block=(0.01, 0.1)`, `seed=0`), the float32 Keras forward pass differs from the fit's own
float64 train predictions by at most 5.8e-7 in absolute value on CPU and 5.3e-7 on GPU 1. The
float64 fit itself has a train RMSE of 4.7e-13 on that data. The probe scripts were also
scratch files.

The consequence: the RMSE figures of order 1e-14 to 1e-15 that the paper's tables give for its
TF1 and TF5 functions are not reachable with a float32 forward pass, whose error against the
fit is already about 1e-7. Under the float64 dtype policy the forward pass is float64 as well;
on a small check (50 rows, 3 inputs, hidden width 6, tanh, slope 5, ridge 0.01) it differed from
the fit's train predictions by 7.8e-16 at most.

No performance claim is made for this port beyond the numbers above. The benchmark results in
the paper are the paper's, obtained on the authors' data files, and are not restated here.

---

## 7. Measured runs

The trainer is `src/train/hkan/`. Its measured runs, with their commands, are in
`src/train/hkan/README.md`. No run results are quoted in this file.

---

## 8. Tests

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_models/test_hkan -q
```

640 tests: scikit-learn parity, forward arithmetic against the equations above, the fitted
`.keras` round trip at exact equality, stock `fit` from the initial state, frozen centers,
data-driven centers per layer, seeds, and chunk independence.

---

## 9. Citation

```bibtex
@article{dudek2025hkan,
  title={HKAN: Hierarchical Kolmogorov-Arnold Network without Backpropagation},
  author={Dudek, Grzegorz and Rodak, Tomasz},
  journal={arXiv preprint arXiv:2501.18199},
  year={2025}
}
```
