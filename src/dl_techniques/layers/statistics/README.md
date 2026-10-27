# Statistics layers (`dl_techniques.layers.statistics`)

Layers that bring probability and statistics into a network: output layers that predict a
distribution instead of a point (mixture density network, conditional normalizing flow),
feature layers fitted by an eigendecomposition (deep kernel PCA, a kernel PCA with a decoder
and a denoiser built on it), a reversible scaler, a moving standard deviation filter, a
residual autocorrelation diagnostic, a per-token support gate, and the split-conformal layers
that turn a model's output into an interval or a set with a coverage guarantee.

Nothing here has been trained or benchmarked in this repository, so this file makes no claim
about accuracy or speed.

## Layout

```
statistics/
├── scaler.py                  UnifiedScaler: RevIN or z-score, invertible
├── mdn_layer.py               MDNLayer + four numpy helper functions
├── normalizing_flow.py        AffineCouplingLayer, NormalizingFlowLayer
├── deep_kernel_pca.py         DeepKernelPCA
├── invertible_kernel_pca.py   InvertibleKernelPCA, InvertibleKernelPCADenoiser
├── local_support_gate.py      LocalSupportGate
├── moving_std.py              MovingStd
├── residual_acf.py            ResidualACFLayer, ACFMonitorCallback
├── conformal/                 split-conformal layers, see conformal/README.md
└── __init__.py                empty: import from the modules
```

Import from the module, not from the package: `statistics/__init__.py` is empty. The
sub-package `conformal/` has a curated `__init__.py` of its own.

## Catalogue

| Module | Classes | What it computes | Imported from outside this package |
|---|---|---|---|
| `scaler.py` | `UnifiedScaler` | Normalizes along any axis. `axis=1` gives Reversible Instance Normalization for time series, `axis=-1` gives per-feature standard scaling. Optional learnable scale and shift, NaN replacement, stored statistics, and `inverse_transform` | none |
| `mdn_layer.py` | `MDNLayer`, `get_point_estimate`, `get_uncertainty`, `get_prediction_intervals`, `check_component_diversity` | A Gaussian mixture over the target: means, standard deviations and mixture weights from three separate heads, concatenated on the last axis. Negative log-likelihood in log space, optional diversity penalty on the means | `models/time_series/mdn/model.py`, `train/time_series/mdn/train_mdn.py`, `train/time_series/mdn/export.py` |
| `normalizing_flow.py` | `AffineCouplingLayer`, `NormalizingFlowLayer` | A conditional Real NVP flow, `p(y given context)`, with an exact log-likelihood loss and a sampler | none |
| `deep_kernel_pca.py` | `DeepKernelPCA` | A stack of kernel-PCA levels, each fitted on the previous level's components. The output is every level's components concatenated | none |
| `invertible_kernel_pca.py` | `InvertibleKernelPCA`, `InvertibleKernelPCADenoiser` | Kernel PCA in a Random Fourier Feature space with a learned linear decoder, and a denoiser that projects through it and back | none |
| `local_support_gate.py` | `LocalSupportGate` | A per-token 0/1 gate from the likelihood ratio of two Gaussian mixtures: one fitted to the current data, one to a generic sample | `layers/adapters/gated_adapter.py` (`GatedAdapter`) |
| `moving_std.py` | `MovingStd` | The local standard deviation of an image over a sliding window, per channel, with pooling ops | none |
| `residual_acf.py` | `ResidualACFLayer`, `ACFMonitorCallback` | The autocorrelation of forecast residuals up to `max_lag`, optionally penalized, with the predictions passed through unchanged | none (`layers/regularization/spatial_smoothness.py` only mentions it in a docstring) |
| `conformal/` | `ConformalLayer`, `ConformalInterval`, `ConformalQuantileHead`, `ConformalPredictionSet`, `attach_conformal_sets` | One frozen calibrated quantile `q_hat`, applied to a point estimate, to learned quantiles, or to classifier logits | `layers/time_series/forecasting_layers.py`, `train/common/conformal.py` |

The last column is a grep over import lines, so it can be re-run:

```bash
grep -rnE "(layers\.statistics|\.\.statistics)[.a-z_]* import" src --include=*.py \
  | grep -v "^src/dl_techniques/layers/statistics/"
```

## Quick start

A reversible scaler. With `axis=1` the statistics are taken over the time axis, per sample and
per feature, and `inverse_transform` puts a prediction back on the original scale:

```python
import numpy as np
import keras
from dl_techniques.layers.statistics.scaler import UnifiedScaler

x = np.random.default_rng(0).normal(5.0, 3.0, size=(4, 24, 3)).astype("float32")
scaler = UnifiedScaler(axis=1)
z = scaler(x)
x_back = scaler.inverse_transform(z)
print(float(np.abs(keras.ops.convert_to_numpy(x_back) - x).max()))
```

A mixture density output. The layer returns one tensor of width `2 * num_mixtures *
output_dimension + num_mixtures`; `loss_func` is the loss to compile with, and the helper
functions turn the trained model's output into numbers:

```python
import numpy as np
import keras
from dl_techniques.layers.statistics.mdn_layer import (
    MDNLayer, get_point_estimate, get_uncertainty, get_prediction_intervals,
)

rng = np.random.default_rng(0)
x = rng.normal(size=(256, 8)).astype("float32")
y = (x[:, :1] + 0.1 * rng.normal(size=(256, 1))).astype("float32")

mdn = MDNLayer(output_dimension=1, num_mixtures=3)
inputs = keras.Input((8,))
model = keras.Model(inputs, mdn(inputs))
model.compile(optimizer="adam", loss=mdn.loss_func)
model.fit(x, y, epochs=1, verbose=0)

point = get_point_estimate(model, x, mdn)
total_variance, aleatoric_variance = get_uncertainty(model, x, mdn, point)
lower, upper = get_prediction_intervals(point, total_variance, confidence_level=0.95)
```

`get_prediction_intervals` is a Gaussian band around the mixture mean. It is not calibrated and
can be wrong on a multi-modal prediction. When you need a coverage guarantee, calibrate one with
the conformal layers instead.

A local standard deviation filter:

```python
import numpy as np
from dl_techniques.layers.statistics.moving_std import MovingStd

image = np.random.default_rng(0).random((2, 16, 16, 3)).astype("float32")
local_std = MovingStd(pool_size=(5, 5))(image)
print(tuple(local_std.shape))
```

For the conformal layers, see [`conformal/README.md`](conformal/README.md).

## Things that surprise people

- **Kernel PCA layers need `adapt(data)` first.** `DeepKernelPCA` and `InvertibleKernelPCA` fit
  by an eager eigendecomposition outside `call`. Without `adapt` they still run, on a random
  projection, and that output means nothing. `DeepKernelPCA.adapt` fits the levels greedily, one
  after another; the forward and backward coupling of the reference paper exists only on the
  unfitted fallback path, and the paper's joint multi-level objective is not solved.
- **`InvertibleKernelPCA.inverse_transform` is a learned decoder, not a pre-image solve.**
  Straight after `adapt` it reconstructs no better than returning zeros, and nothing in this
  package trains it; put it in a loss before relying on it. The class docstring has the
  measurements.
- **`InvertibleKernelPCADenoiser` thresholds only in training.** With `adaptive_components=True`
  the hard threshold on the components runs when `training` is true; at inference they pass
  through.
- **`LocalSupportGate` is inert until fitted.** An unfitted gate reports every token as out of
  support. Fitting is always an explicit call (`fit_pos`, `fit_neg`, `fit`, `observe`) and never a
  side effect of `call`. Both mixtures are non-trainable weights, fitted by EM and then frozen.
- **`ResidualACFLayer` takes `[predictions, targets]` and returns the predictions.** It adds a
  loss only when `regularization_weight` is set and the model is training; the measured ACF is
  kept on the `acf_values` attribute.
- **`UnifiedScaler.inverse_transform` divides by a floored scale.** A learnable scale driven to
  zero would otherwise return inf or NaN, so the divide uses its magnitude floored at `eps`.
- **`NormalizingFlowLayer.call` maps data to the latent.** It runs the coupling layers backwards
  (`y` to `z`) and returns `(z, log_det)` for `loss_func`; `sample` runs them forwards.

## Serialization

Every class registers through `@register_dl_technique("dl_techniques.layers.statistics.<module>")`,
and the conformal classes through `...statistics.conformal.<module>`. `keras.saving.get_registered_name`
gives `dl_techniques.layers.statistics.scaler>UnifiedScaler` for the scaler. The conformal layers were
moved and renamed on 2026-10-27/28; a `.keras` file that recorded their old class names or keys does
not load (`conformal/README.md` lists the old names).

## Tests

`tests/test_layers/test_statistics/`: one file per module, the conformal tests in `test_conformal/`,
and four package-wide guards. `test_the_package_is_v2_compliant.py` walks every module of the package
(sub-package included) against the authoring guide; `test_the_package_meets_the_v2_test_bar.py` holds
the per-layer test bar; `test_the_eigenvalue_divides_survive_fp16.py` and
`test_the_four_carried_defects_are_fixed.py` pin specific fixes. `AGENTS.md` in this directory lists
the rules and what pins each. Documentation claims in this README and `AGENTS.md` are checked by
`test_the_statistics_docs_name_real_things.py`.
