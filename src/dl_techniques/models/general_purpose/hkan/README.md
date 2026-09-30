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
print(diagnostics["forward_rmse"])             # train RMSE of the model's own forward pass
print(diagnostics["forward_deviation_rms"])    # RMS of (forward pass - train_predictions)

y_hat = model.predict(x_test, verbose=0)       # (200, 1)
```

The returned dict has exactly those six keys. `importance` is the paper's equation 14.

`layer_rmse` and `train_predictions` describe the float64 fit. The model holds that solution
cast to its own dtype, and its forward pass is not always the fit (section 6). So
`fit_closed_form` ends with one batched forward pass of the model on `x` and returns
`forward_rmse` (the model against `y`: the number a later `predict` reproduces) and
`forward_deviation_rms` (the model against `train_predictions`). When the deviation exceeds
1e-3 of the target's standard deviation it logs a warning through the repo logger that states
both RMSEs, the largest weight and the remedies. For a constant target the limit is an absolute
1e-6 instead. It is a warning, not an exception: the float64 diagnostics are still correct. In
this example (`l2_mix` at its default, 0.01) `forward_rmse` and `layer_rmse[-1]` are both
0.08270, 5e-09 apart, the deviation is 1.9e-07 RMS, and no warning is logged.

For a constant target `block_r2` and `importance` are 0.0: there is no variance to explain.
"Constant" means a range of at most four float64 epsilons times the target's largest
magnitude, so an exact constant and a constant with one entry one ulp away count, and a target
that varies at 1e-12 does not. The forward check uses its absolute limit for the same targets.
scikit-learn's `r2_score` returns 1.0 for an exact fit of a constant; every other target follows
`r2_score`.

`y` may have shape `(N,)` or `(N, 1)`. A `y` with more than one column raises `ValueError`:
every block regresses onto one target, so the model is single-target. `chunk_size` is the
number of outputs of a layer solved at once; it bounds memory and does not change the result.
The forward check runs the model in batches sized so that a batch's feature tensor is no
larger than one solve chunk's, so `chunk_size` bounds that pass's memory too.
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
one CPU run of this example `evaluate` gave a train loss of 0.00684 before the gradient phase,
and the three epoch losses `fit` reported were 0.00680, 0.00667 and 0.00666: a change in the
fourth decimal, from one run with an unseeded shuffle.

---

## 3. Constructor

```python
HKAN(
    hidden_units=(),          # widths of the hidden layers; () is a one-layer model
    num_basis=10,
    basis="sigmoid",
    slope=5.0,                # the reference code: 1.0
    centers="random",         # "random", "equally_spaced" or "data"
    l2_block=0.01,            # the reference code: 0.0
    l2_mix=0.01,              # the paper and the reference code: 0.0
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
raises `ValueError` from the constructor. Widths, `num_basis` and `seed` may be numpy integers
(`np.int64(16)`); `bool` and floats are refused. `get_config()` returns plain Python numbers.

**Three defaults are not the reference code's.** `slope=5.0`, `l2_block=0.01` and `l2_mix=0.01`
are chosen so that the default model, in float32, computes the fit it reports. That was measured
up to two hidden layers of 64 and on a few wider and deeper models; at three hidden layers of
128 it no longer holds and `fit_closed_form` warns (section 6 has the envelope). The
reference code defaults to slope 1 and no ridge, and the paper fits the connecting stage by
plain least squares. That behaviour is one call away:

```python
reference = HKAN(hidden_units=(16,), slope=1.0, l2_block=0.0, l2_mix=0.0)
```

With those three values a sigmoid is almost linear over `[0, 1]` and is fitted without a ridge:
the coefficients reach 1e9 to 1e11, the float32 model does not compute the fit, and
`fit_closed_form` warns. `l2_block` and `l2_mix` have no effect on gradient training.

**Input width.** A built model accepts only the width it was built for. Any other width, 1
included, raises `ValueError` in `call`, `predict` and `fit`, also after a `.keras` round trip.

**Centers.** `random`: uniform on `[0, 1]`. `equally_spaced`: `linspace(0, 1, num_basis)` for
every block. `data`: for every block `(q, p)`, `num_basis` values sampled with replacement from
column `p` of that layer's own input. The `data` draw happens inside `fit_closed_form`, or in
`initialize_centers(x)` for a gradient-only run; until one of the two runs, the weight holds a
uniform draw as a placeholder. `seed` seeds the centers only. `seed=0` is a seed like any
other; `seed=None` draws from the global numpy generator. A seeded layer draws from
`np.random.default_rng([seed, layer_index, stream, 0x484B414E])`, with `stream` 0 for the
build-time centers and 1 for the data draw. The constant in the last position keeps every stream
of this package apart from `np.random.default_rng(seed)`, which is what callers usually draw
their data from: numpy ignores trailing zeros of a seed list, so without it the first layer's
centers were the first values of the caller's own generator.

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

**Dtype policies.** float32 and float64 are supported, set globally
(`keras.config.set_dtype_policy`) or by the constructor's `dtype` argument, which every layer
receives. Any other compute dtype raises `ValueError` at build: `mixed_float16`, `float16`,
`mixed_bfloat16` and `bfloat16`. Before the refusal existed, a `mixed_float16` model (tanh,
slope 50, hidden width 5, 96 rows) built, fitted and predicted with no error, and its float16
output was up to 2.3e-02 away from the fit's own train predictions on a fit of RMSE 0.154.
Before the constructor's `dtype` reached the layers, `HKAN(dtype="mixed_float16")` still did,
and `HKAN(dtype="float64")` kept float32 layers.

```python
model64 = HKAN(hidden_units=(16,), slope=1.0, l2_block=0.01, l2_mix=0.0, seed=0, dtype="float64")
diagnostics64 = model64.fit_closed_form(x_train, y_train)
print(model64.hkan_layers[0].compute_dtype)    # float64
print(diagnostics64["forward_deviation_rms"])  # about 4e-12; with dtype="float32", 2.2e-03 and a warning
```

A float64 model saved to `.keras` reloads as float64.

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
- **Defaults of `slope`, `l2_block` and `l2_mix`** are 5.0, 0.01 and 0.01. The reference code
  defaults to slope 1 and no block ridge, and the paper fits the connecting stage by plain least
  squares, with no penalty. In float32 those values give a model that is not its own fit
  (section 6), so they are not the defaults here. The reference behaviour is
  `slope=1.0, l2_block=0.0, l2_mix=0.0`, passed explicitly; reproducing a configuration of the
  paper means passing `l2_mix=0.0` as well as the paper's slope and block ridge.
- **The fit checks the model it leaves behind.** `fit_closed_form` runs the Keras model once
  on the training rows, returns `forward_rmse` and `forward_deviation_rms`, and warns when the
  model is not the fit. The reference code has one float64 model and needs no such check.
- **Importance of a constant target is 0.0**, not the 1.0 that scikit-learn's `r2_score` (which
  the reference code calls) gives for an exact fit of a constant.
- **Input width and dtype are checked.** A wrong input width raises, and so does a float16
  compute dtype.
- **`l2_mix`** adds a ridge penalty to the connecting stage, which neither the paper nor the
  reference code has. Its default is 0.01; `l2_mix=0.0` is the paper's plain least squares.
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
`plans/plan-2026-09-30T082355-4d999dbc/decisions.md` (D-022, D-023, D-024, D-034, D-038, D-039,
D-055).

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
difference between the two sets of float64 train predictions was 7.2e-10 at zero ridge and
2.0e-11 at positive ridge. The command was
`CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python <scratchpad>/hkan_probe/parity_upstream.py`;
the script and the reference copy lived in a session scratch directory and no committed test
re-runs it. Per cell, the largest absolute difference (0.0 is exact agreement):

| basis (hidden, top) | intercept, `l2=0` | intercept, `l2=0.01` | no intercept, `l2=0` | no intercept, `l2=0.01` |
|---|---|---|---|---|
| sigmoid, sigmoid | 4.5e-11 | 7.8e-14 | 2.2e-11 | 3.8e-13 |
| gaussian, gaussian | 0.0 | 1.0e-14 | 0.0 | 3.8e-14 |
| relu, relu | 0.0 | 3.9e-14 | 0.0 | 1.3e-12 |
| tanh, tanh | 0.0 | 4.4e-14 | 0.0 | 2.3e-13 |
| softplus, softplus | 7.0e-10 | 1.7e-12 | 1.2e-10 | 1.6e-12 |
| identity, identity | 0.0 | 1.1e-15 | 0.0 | 1.8e-15 |
| sigmoid, identity | 4.0e-11 | 4.0e-14 | 4.0e-11 | 6.0e-13 |
| gaussian, identity | 0.0 | 1.4e-14 | 0.0 | 2.0e-14 |
| relu, identity | 0.0 | 2.3e-14 | 0.0 | 6.4e-13 |
| tanh, identity | 0.0 | 7.6e-14 | 0.0 | 4.5e-13 |
| softplus, identity | 7.2e-10 | 1.2e-12 | 2.1e-10 | 2.0e-11 |

This is the run on the code as it stands (seed 7, the seed streams described in section 3). A
first run, with the seed streams this package had before, gave 6.6e-10 and 1.5e-11 as the two
worst cells; its table is in `decisions.md` D-022.

**The model's forward pass against the float64 fit.** The fit is float64; the model holds the
solution cast to its dtype. The gap between the two is about the largest fitted weight times the
resolution of the model dtype (6e-8 of a value in float32, 1e-16 in float64), because the
forward pass sums terms of that size which mostly cancel. It is not "float32 accuracy" in
general. Measured on CPU in float32, train rows, as returned by `fit_closed_form`:

| Configuration | Data | Largest weight | Float64 fit RMSE | Float32 model RMSE | Deviation RMS, as a fraction of std(y) |
|---|---|---|---|---|---|
| Authors' tutorial: `hidden_units=(912,)`, `num_basis=(23, 10)`, `basis=("tanh", "identity")`, `slope=50.0`, `centers=("random", "data")`, `l2_block=(0.01, 0.1)`, `l2_mix=0.0` | 5000 rows of the paper's TF5 function, generated here | 13 | 4.0e-13 | 1.3e-07 | 7.6e-07 (largest single difference 5.7e-07 in absolute terms; 6.2e-07 on GPU 1) |
| Constructor defaults, `hidden_units=()` | probe | 2.1 | 8.5e-02 | 8.5e-02 | 5.5e-07 |
| Constructor defaults, `hidden_units=(16,)` | probe | 2.1 | 3.9e-02 | 3.9e-02 | 1.2e-06 |
| Constructor defaults, `hidden_units=(256,)` | probe | 4.3 | 3.0e-02 | 3.0e-02 | 2.9e-06 |
| Constructor defaults, `hidden_units=(64, 64)` | probe | 3.6 | 3.3e-02 | 3.3e-02 | 4.7e-06 |
| The configuration of example 2.1 | probe | 3.5 | 8.3e-02 | 8.3e-02 | 8.3e-07 |
| `hidden_units=(16,)`, `slope=1.0`, `l2_block=0.01`, `l2_mix=0.0` | probe | 4.1e+02 | 3.3e-03 | 3.7e-03 | 7.0e-03 |
| `hidden_units=(256,)`, `slope=1.0`, `l2_block=0.01`, `l2_mix=0.0` | probe | 2.3e+03 | 1.2e-08 | 2.1e-02 | 8.4e-02 |
| The paper's Table V row for TF1: `hidden_units=(932,)`, `basis=("sigmoid", "tanh")`, `slope=(1.0, 33.0)`, `num_basis=(2, 13)`, `centers="data"`, `l2_block=(0.1, 10.0)`, `l2_mix=0.0` | probe | 6.3e+03 | 3.4e-08 | 3.2e-03 | 1.3e-02 |
| The same row | 5000 rows of TF1 as `src/train/hkan/data.py` generated it at commit `2313b45c6` (`seed=0`), before its TF1 scaling changed (D-041) | 9.0e+02 | 5.6e-11 | 4.0e-04 | 1.2e-03 |
| Reference defaults, passed explicitly (`slope=1.0, l2_block=0.0, l2_mix=0.0`), `hidden_units=()` | probe | 4.6e+09 | 8.5e-02 | 2.6e+02 | 1.0e+03 |
| Reference defaults, passed explicitly, `hidden_units=(16,)` | probe | 5.1e+10 | 4.9e-02 | 2.0e+07 | 7.9e+07 |

"probe" is 1000 rows, two inputs uniform on `[0, 1]` from `np.random.default_rng(123)`,
`y = sin(3 x1) x2` (standard deviation 0.25); every model has `seed=0`. The scripts were scratch
files: `<scratchpad>/hkan_probe/float32_probe.py` for the first row (it was run when `l2_mix`
still defaulted to 0.0) and `<scratchpad>/step1_2/bad_cases2.py` for the others.

Two causes were measured, and a ridge on the blocks removes only the first:

- **Block coefficients from a weak or zero `l2_block`.** The last two rows, which are the
  reference code's defaults.
- **Connecting weights from `l2_mix = 0` on a wide layer.** Rows seven to ten. The outputs of a
  wide hidden layer are nearly collinear approximations of the same `y`, so the unpenalized
  connecting regression of the next layer returns large weights whatever `l2_block` is: 4e+02 to
  6e+03 in these rows, and up to 4.5e+08 at width 256 in the search recorded in `decisions.md`
  D-034. The paper's connecting stage is unpenalized, and the hyperparameters of its Table V
  row for TF1 show the effect on the probe data: a float64 fit of 3.4e-08 that the float32 model
  reproduces only to 3.2e-03.

That is why the constructor defaults carry a ridge on both stages and a slope of 5
(`decisions.md` D-039). On the probe data the default model is within 5e-06 of std(y) of its fit
at every width in the table. Over a wider search (6 targets, 5 seeds, widths from no hidden
layer to two hidden layers of 64, 1000 rows each, 180 fits) the worst default model was at
5.5e-04 of std(y), on `y = sin(8 x1) cos(5 x2)` with two hidden layers of 64: under the warning
threshold of 1e-3 by a factor of 1.8, which is not a wide margin. The check in
`fit_closed_form` runs on every fit for that reason.

**Where the defaults were measured, and where they stop.** Deviation as a fraction of std(y),
float32, CPU, constructor defaults, the worst fit of each search:

| Search | Fits | Worst |
|---|---|---|
| 6 targets, 5 seeds, no hidden layer to `(64, 64)`, 1000 rows (D-039) | 180 | 5.5e-04 |
| 2, 5 and 10 inputs, 50, 150 and 1000 rows, 6 targets, no hidden layer to `(64, 64)`, 2 seeds (second review) | 576 | 5.0e-04 |
| `(128, 128)`, `(256, 256)`, `(64, 64, 64)`, `(16, 16, 16, 16)`, `(512,)`, 100 and 1000 rows, 3 targets, 2 seeds (second review) | 60 | 8.5e-04 |

The margin shrinks with depth. On `y = sin(8 x1) cos(5 x2)`, 1000 rows, two inputs drawn from
`np.random.default_rng(200 + seed)`, model seeds 0, 1 and 2:

| `hidden_units` | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `(64, 64, 64)` | 5.0e-04 | 8.5e-04 | 6.6e-04 |
| `(64, 64, 64, 64)` | 6.3e-04 | 1.3e-03 | 8.6e-04 |
| `(128, 128, 128)` | 1.43e-03 | 1.35e-03 | 1.24e-03 |

Every entry above 1e-3 is a fit that warns, as designed: at three hidden layers of 128 the
defaults do not give a float32 model that computes its fit, and the warning names the
remedies. The last table was re-run for this README (`decisions.md` D-055). The second and third
rows of the first table are the second review's; of them only the worst cell (8.5e-04, the
`(64, 64, 64)` seed-1 cell of the last table) was re-run, and it reproduced.

Under the float64 dtype policy (`dtype="float64"` or the global policy) the last column of rows two to ten is at most 1.3e-10 (the Table
V row on the probe data: 2.4e-11; the constructor-default rows: 8.4e-15 at most), and the two
reference-default rows read 1.7e-06 and 2.8e-02 (the first row was not re-run in float64). So
float64 closes the gap for weights up to 6e+03 and does not at 5e+10. `fit_closed_form` measures
the gap at whatever dtype the model has and warns whenever the last column exceeds 1e-3, which
is rows seven to twelve in float32 and the last row in float64.

The consequence for the paper's numbers: the train RMSE figures of order 1e-14 to 1e-15 that
its tables give for TF1 and TF5 are float64 numbers. A float32 model of the tutorial
configuration stops at 1.3e-07, and a float32 model of the Table V row for TF1 at 4.0e-04.

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

759 tests: the constructor defaults, scikit-learn parity, forward arithmetic against the equations above, the fitted
`.keras` round trip at exact equality, stock `fit` from the initial state, frozen centers,
data-driven centers per layer, seeds and seed streams, chunk independence, the forward check of
`fit_closed_form`, its warning (also for a frozen model and a NaN forward pass) and its batch
size, the constant-target importance one ulp from a constant, the input-width and dtype
refusals (global policy and constructor `dtype`), the float64 round trip, and numpy integer
arguments.

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
