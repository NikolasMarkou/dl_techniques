# Statistics layers (`dl_techniques.layers.statistics`)

Keras 3 layers that bring probability and statistics into a network. They live in `src/dl_techniques/layers/statistics/`, part of the `dl_techniques` deep learning library. The package covers output layers that predict a distribution instead of a point (mixture density network, conditional normalizing flow), feature layers fitted by an eigendecomposition (deep kernel PCA, a kernel PCA with a decoder and a denoiser built on it), a reversible scaler, a moving standard deviation filter, a residual autocorrelation diagnostic, a per-token support gate, a probability-to-log-odds map, and the split-conformal layers in the `conformal/` sub-package that turn a model's output into an interval or a set with a coverage guarantee.

Nothing here has been trained or benchmarked in this repository, so this file makes no claim about accuracy or speed.

## What it is for

A plain network returns one number or one score vector and says nothing about how sure it is. These layers cover the gaps: predicting a whole distribution, scaling data in a way you can undo, finding structure by kernel PCA, flagging tokens a model has not seen enough of, checking whether forecast errors are still correlated, and wrapping a trained model so its output carries a coverage promise.

## Layout

```
statistics/
├── scaler.py                  UnifiedScaler: RevIN or z-score, invertible
├── mdn_layer.py               MDNLayer + six helper functions (four NumPy, two mode finders)
├── normalizing_flow.py        AffineCouplingLayer, NormalizingFlowLayer
├── deep_kernel_pca.py         DeepKernelPCA
├── invertible_kernel_pca.py   InvertibleKernelPCA, InvertibleKernelPCADenoiser
├── local_support_gate.py      LocalSupportGate
├── moving_std.py              MovingStd
├── probability_logits.py      ProbabilityLogits
├── residual_acf.py            ResidualACFLayer, ACFMonitorCallback
├── conformal/                 split-conformal layers (own __init__.py)
└── __init__.py                empty: import from the modules
```

Import from the module, not from the package: `statistics/__init__.py` is empty. The sub-package `conformal/` has a curated `__init__.py` with `__all__`, so import from `dl_techniques.layers.statistics.conformal`.

## Catalogue

| Module | Classes | What it computes | Imported from outside this package |
|---|---|---|---|
| `scaler.py` | `UnifiedScaler` | Normalizes along any axis. `axis=1` gives Reversible Instance Normalization (RevIN) for time series, `axis=-1` gives per-feature standard scaling. Optional learnable scale and shift, NaN replacement, stored statistics, and `inverse_transform` | none |
| `mdn_layer.py` | `MDNLayer`, `get_point_estimate`, `get_uncertainty`, `get_prediction_intervals`, `check_component_diversity`, `get_mixture_modes`, `get_mode_estimate` | A Gaussian mixture over the target: means, standard deviations and mixture weights from three separate heads, concatenated on the last axis. Negative log-likelihood in log space, optional diversity penalty on the means | `models/time_series/mdn/model.py`, `train/time_series/mdn/train_mdn.py`, `train/time_series/mdn/export.py` |
| `normalizing_flow.py` | `AffineCouplingLayer`, `NormalizingFlowLayer` | A conditional Real NVP flow, `p(y given context)`, with an exact log-likelihood loss and a sampler | none |
| `deep_kernel_pca.py` | `DeepKernelPCA` | A stack of kernel-PCA levels, each fitted on the previous level's components. The output is every level's components concatenated | none |
| `invertible_kernel_pca.py` | `InvertibleKernelPCA`, `InvertibleKernelPCADenoiser` | Kernel PCA in a Random Fourier Feature space with a learned linear decoder, and a denoiser that projects through it and back | none |
| `local_support_gate.py` | `LocalSupportGate` | A per-token gate from the likelihood ratio of two Gaussian mixtures: one fitted to the current data, one to a generic sample. `output_mode` picks the binary decision (`'hard'`), its causal moving average (`'smoothed'`) or the raw log ratio (`'scores'`) | `layers/adapters/gated_adapter.py` (`GatedAdapter`) |
| `moving_std.py` | `MovingStd` | The local standard deviation of an image over a sliding window, per channel, with pooling ops | none |
| `probability_logits.py` | `ProbabilityLogits` | The log-odds `log(p) - log(1 - p)` of a tensor of probabilities, clipped away from 0 and 1. Turns a per-class probability that is not a softmax (CapsNet capsule lengths) into scores a softmax can separate; the class order is unchanged | `train/capsnet/train_capsnet.py` |
| `residual_acf.py` | `ResidualACFLayer`, `ACFMonitorCallback` | The autocorrelation of forecast residuals up to `max_lag`, optionally penalized, with the predictions passed through unchanged | none (`layers/regularization/spatial_smoothness.py` only mentions it in a docstring) |
| `conformal/` | `ConformalLayer`, `ConformalInterval`, `ConformalQuantileHead`, `ConformalPredictionSet`, `attach_conformal_sets` | One frozen calibrated quantile `q_hat`, applied to a point estimate, to learned quantiles, or to classifier logits | `layers/time_series/forecasting_layers.py`, `train/common/conformal.py` |

The last column is a grep over import lines, so it can be re-run from the repo root:

```bash
grep -rnE "(layers\.statistics|\.\.statistics)[.a-z_]* import" src --include=*.py \
  | grep -v "^src/dl_techniques/layers/statistics/"
```

## The conformal sub-package

Split conformal prediction adds a coverage promise to any trained model without changing it. You hold out a calibration set, compute one "score" per item (how wrong the model was), and take a finite-sample quantile of those scores. That quantile is `q_hat`. Applied to the output, it gives an interval or a set of labels that contains the truth with probability at least `1 - alpha`, provided calibration and deployment data are exchangeable (same distribution, order irrelevant).

`ConformalLayer` (in `conformal/base.py`) owns the non-trainable scalar weight `conformal_q_hat` and two ways to fill it: `calibrate(scores, alpha=0.1)` sets it to the `ceil((n + 1) * (1 - alpha))`-th smallest score (`+inf` when there are too few scores), and `set_q_hat(value)` assigns a value fitted elsewhere. Both raise `ValueError` on an unbuilt layer, so call the layer once on a sample first. The calibration runs on the host in NumPy; the forward pass only reads `q_hat`, so the calibrated model saves as one `.keras` file.

| Class | Applies `q_hat` to | Output | Score to calibrate with |
|---|---|---|---|
| `ConformalInterval` | a point estimate `mu` | `(mu_c, lower, upper)`, or `(lower, upper)` with `return_mu=False`; `mu_c` is `mu` clipped to `[domain_min, domain_max]`, `lower = mu_c - q_hat`, `upper = mu_c + q_hat` | `abs(truth - mu)` |
| `ConformalQuantileHead` | three learned quantiles per step and feature | `call`: `(batch, forecast_length, output_dim, 3)` as lower, median, upper, with no `q_hat` applied. `predict_intervals`: `(median, lower, upper)` with `lower - q_hat` and `upper + q_hat` | `max(q_low - truth, truth - q_high)` |
| `ConformalPredictionSet` | classifier logits | `(probs, in_set)`; `in_set` is a 0/1 float mask `(batch, num_classes)` of labels whose cumulative sorted probability is at most `q_hat` | `aps_scores(..., randomized=False)` from `dl_techniques.utils.conformal_classification` |

`attach_conformal_sets(model, tau, input_shape)` wraps a trained classifier into a Functional model with outputs `(logits, probs, in_set)`, sharing the classifier's weights, with `tau` assigned as `q_hat`.

The NumPy-side fits live in `src/dl_techniques/utils/conformal_denoiser_intervals.py` (`conformal_quantile`, `calibrate_per_sigma`) and `src/dl_techniques/utils/conformal_classification.py` (`aps_scores`, `calibrate_aps`, `predict_sets`, `evaluate_sets`). The trainer step that calibrates and exports is `src/train/common/conformal.py`.

## Quick start

A reversible scaler. With `axis=1` the statistics are taken over the time axis, per sample and per feature, and `inverse_transform` puts a prediction back on the original scale:

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

A mixture density output. The layer returns one tensor of width `2 * num_mixtures * output_dimension + num_mixtures`; `loss_func` is the loss to compile with, and the helper functions turn the trained model's output into numbers:

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

`get_point_estimate` is the mixture mean, which for a bimodal prediction lands between the modes. `get_mode_estimate(model, x, mdn)` returns, per sample, the mode with the highest density, and `get_mixture_modes(mu, sigma, pi_logits)` is the graph-safe function under it. It starts one fixed-point iteration at every component mean (moved off it by the larger of 1e-3 standard deviations and 1e-5 times the mean, in a different fixed direction for each component and dimension) and returns `(modes [B, M, D], log_density [B, M], is_unique [B, M])`; `is_unique` is False for a start that converged onto an earlier one. The returned points are stationary points of the mixture density reached from each start, not a guarantee of local maxima: the iteration never lowers the density, so a start leaves a minimum or a saddle unless it sits exactly on one (the offset exists for that case, for instance components at -1, 0, 1 with weights 0.45, 0.1, 0.45 and sigma 0.5). No fixed direction avoids every symmetry, so a mixture whose symmetry axis lies exactly along the offset of one start (an X mixture rotated until its axis follows the offset direction of the centre component) can still keep that start on a saddle. `get_mode_estimate` keeps the highest-density point, which is always a true mode, so use it when a true mode is needed. Convergence is linear and slows near the merge point of two components, about two standard deviations apart: two equal unit components 1.99 sigma apart still show two unique modes at +-0.067 after 100 steps (the true mode is 0) and one after 1000. Raise `num_steps` or `merge_tol` when components sit within about 2 sigma of each other. Two overlapping components give one unique mode, two separated components give two:

```python
import numpy as np
from dl_techniques.layers.statistics.mdn_layer import get_mixture_modes

mu = np.array([[[-3.0], [3.0]], [[-0.25], [0.25]]], dtype="float32")
sigma = np.full((2, 2, 1), 0.5, dtype="float32")
modes, log_density, is_unique = get_mixture_modes(mu, sigma, np.zeros((2, 2), "float32"))
print(np.asarray(is_unique).sum(axis=1))
```

`get_prediction_intervals` is a Gaussian band around the mixture mean. It is not calibrated and can be wrong on a multi-modal prediction. When you need a coverage guarantee, calibrate one with the conformal layers instead.

A local standard deviation filter:

```python
import numpy as np
from dl_techniques.layers.statistics.moving_std import MovingStd

image = np.random.default_rng(0).random((2, 16, 16, 3)).astype("float32")
local_std = MovingStd(pool_size=(5, 5))(image)
print(tuple(local_std.shape))
```

A conformal interval around a point estimate. `mu` stands in for your model's output on a held-out set; build the layer with one call, calibrate with `abs(truth - mu)`, then call it again:

```python
import numpy as np
import keras
from dl_techniques.layers.statistics.conformal import ConformalInterval

rng = np.random.default_rng(0)
truth = rng.uniform(size=(2000, 8)).astype("float32")
mu = np.clip(truth + rng.normal(0.0, 0.05, truth.shape), 0.0, 1.0).astype("float32")

layer = ConformalInterval()
layer(mu[:1])                                           # build the layer
layer.calibrate(np.abs(truth - mu).ravel(), alpha=0.1)
mu_c, lower, upper = layer(mu)
print(float(keras.ops.convert_to_numpy(layer.q_hat)))
```

## Things that surprise people

- **Kernel PCA layers need `adapt(data)` first.** `DeepKernelPCA` and `InvertibleKernelPCA` fit by an eager eigendecomposition outside `call`. Without `adapt` they still run, on a random projection, and that output means nothing. `DeepKernelPCA.adapt` fits the levels greedily, one after another; the forward and backward coupling of the reference paper exists only on the unfitted fallback path, and the paper's joint multi-level objective is not solved.
- **`InvertibleKernelPCA.inverse_transform` is a learned decoder, not a pre-image solve.** Straight after `adapt` it reconstructs no better than returning zeros, and nothing in this package trains it; put it in a loss before relying on it. The class docstring has the measurements.
- **`InvertibleKernelPCADenoiser` thresholds only in training.** With `adaptive_components=True` the hard threshold on the components runs when `training` is true; at inference they pass through.
- **`LocalSupportGate` is inert until fitted.** An unfitted gate reports every token as out of support. Fitting is always an explicit call (`fit_pos`, `fit_neg`, `fit`, `observe`) and never a side effect of `call`. Both mixtures are non-trainable weights, fitted by EM (expectation-maximization) and then frozen.
- **`ResidualACFLayer` takes `[predictions, targets]` and returns the predictions.** It adds a loss only when `regularization_weight` is set and the model is training; the measured ACF is kept on the `acf_values` attribute.
- **`UnifiedScaler.inverse_transform` divides by a floored scale.** A learnable scale driven to zero would otherwise return inf or NaN, so the divide uses its magnitude floored at `eps`.
- **`NormalizingFlowLayer.call` maps data to the latent.** It takes `[data, context]`, runs the coupling layers backwards (`y` to `z`) and returns `(z, log_det)` for `loss_func`; `sample` runs them forwards.
- **Conformal layers: build before calibrating.** `calibrate` and `set_q_hat` raise `ValueError` on an unbuilt layer. `q_hat` must be non-negative and not NaN; `+inf` is allowed. `ConformalPredictionSet` stores any value above `2.0` as `2.0`, which already keeps every label.
- **The conformal guarantee is marginal.** It holds on average over the data, not for every input or class, and it weakens if the model was chosen using the calibration set. `ConformalInterval` carries one radius, so it covers one noise regime.
- **`ConformalPredictionSet` is the deterministic APS rule.** It can return an empty set when `q_hat` is below the top probability, and needs rank-2 logits with at least 2 classes.
- **`ConformalLayer` alone is not usable.** It is registered but has no `call`.
- **`ConformalInterval` and `ConformalPredictionSet` force `trainable=False`.** `ConformalQuantileHead` has a trainable Dense projection; only `q_hat` is frozen. Under `mixed_float16` the weight stays in its variable dtype and each layer casts it where it is used.

## Serialization

Every layer class registers through `@register_dl_technique("dl_techniques.layers.statistics.<module>")`, and the conformal classes through `...statistics.conformal.<module>`. `ACFMonitorCallback` is a Keras callback, not a layer, and is not registered. `keras.saving.get_registered_name` gives `dl_techniques.layers.statistics.scaler>UnifiedScaler` for the scaler. The conformal layers were moved and renamed on 2026-10-27/28; a `.keras` file that recorded their old class names or keys does not load. The removed names are `ConformalIntervalLayer`, `ConformalPredictionSetLayer`, the constructor arguments `q_init` and `tau_init`, and the old `calibrate(value)` form; no aliases exist, by decision.

## Tests

`tests/test_layers/test_statistics/`: one file per module, the conformal tests in `test_conformal/` (`test_base_contract.py`, `test_interval.py`, `test_quantile_head.py`, `test_prediction_set.py`), and five package-wide guards. `test_the_package_is_v2_compliant.py` walks every module of the package (sub-package included) against the authoring guide (`research/2026_keras_custom_models_instructions_v2.md`); `test_the_package_meets_the_v2_test_bar.py` holds the per-layer test bar; `test_the_eigenvalue_divides_survive_fp16.py` and `test_the_four_carried_defects_are_fixed.py` pin specific fixes; `test_the_statistics_docs_name_real_things.py` checks the claims in this directory's documentation: paths exist, every class is named, every Python snippet in the README runs. Beside the guards sit a shared instrument, `v2_compliance_oracle.py` (no `test_` prefix, so pytest does not collect it), and `test_the_v2_oracle_can_fail.py`, which proves the oracle rejects a broken subject.

Run only this directory, with the repo `.venv`:

```bash
.venv/bin/python -m pytest tests/test_layers/test_statistics -q
```
