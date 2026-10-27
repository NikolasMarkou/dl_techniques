# statistics

Probability and statistics layers for `dl_techniques`: distribution-valued outputs (MDN, conditional
normalizing flow), eigendecomposition-fitted feature layers (deep kernel PCA, invertible kernel PCA and
its denoiser), a reversible scaler, a moving standard deviation filter, a residual-ACF diagnostic, a
per-token support gate, and the `conformal/` sub-package. The parent `../AGENTS.md` owns the cross-package
rules; `README.md` is the user-facing tour; `conformal/AGENTS.md` covers the sub-package.

**Registration.** Every class registers through
`@register_dl_technique("<defining module's dotted path>")` from `dl_techniques.utils.keras_registration`:

```python
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.layers.statistics.scaler")
class UnifiedScaler(keras.layers.Layer):
    ...
```

`keras.saving.get_registered_name(UnifiedScaler)` is `dl_techniques.layers.statistics.scaler>UnifiedScaler`.
Nothing is stripped under `layers/`, so the conformal classes register under
`dl_techniques.layers.statistics.conformal.<module>`. Never a bare
`@keras.saving.register_keras_serializable()`: its key `Custom>ClassName` ignores `__module__`, so two
same-named classes claim one slot. The helper also binds `Custom>ClassName` as an alias for old archives.

## Layout

`__init__.py` is empty: import from the module. `conformal/__init__.py` is curated (see its AGENTS.md).

| File | Holds |
|---|---|
| `scaler.py` | `UnifiedScaler` (RevIN when `axis=1`, z-score when `axis=-1`; `inverse_transform`, `denormalize`, `get_stats`, `reset_stats`) |
| `mdn_layer.py` | `MDNLayer` and the numpy helpers `get_point_estimate`, `get_uncertainty`, `get_prediction_intervals`, `check_component_diversity` |
| `normalizing_flow.py` | `AffineCouplingLayer`, `NormalizingFlowLayer` (conditional Real NVP; `call` is data to latent, `sample` is latent to data) |
| `deep_kernel_pca.py` | `DeepKernelPCA` (needs `adapt`) |
| `invertible_kernel_pca.py` | `InvertibleKernelPCA`, `InvertibleKernelPCADenoiser` (both need `adapt`) |
| `local_support_gate.py` | `LocalSupportGate` (EM-fitted, frozen; inert until fitted) |
| `moving_std.py` | `MovingStd` |
| `residual_acf.py` | `ResidualACFLayer`, `ACFMonitorCallback` (a `keras.callbacks.Callback`, not a layer) |
| `conformal/` | `ConformalLayer` base and three layers, see `conformal/AGENTS.md` |

The class count lives HERE and nowhere else. Re-derive it rather than trust it:

```bash
grep -hoP '^class \K\w+' src/dl_techniques/layers/statistics/*.py \
  src/dl_techniques/layers/statistics/conformal/*.py | sort
```

That prints 15 names: the 10 layers of the top-level modules, `ACFMonitorCallback` (a callback, not a
layer) and the 4 conformal classes (`ConformalLayer` and three layers). Outside importers are a grep, not a claim:

```bash
grep -rnE "(layers\.statistics|\.\.statistics)[.a-z_]* import" src --include=*.py \
  | grep -v "^src/dl_techniques/layers/statistics/"
```

Today: `MDNLayer` (`models/time_series/mdn`, `train/time_series/mdn`), `LocalSupportGate`
(`layers/adapters/gated_adapter.py`), `ConformalQuantileHead` (`layers/time_series/forecasting_layers.py`) and
the conformal classes (`train/common/conformal.py`). The others have no importer outside tests.

## Conventions

- **Docstrings are Sphinx/reST** (`:param:`, `:type:`, `:raises:`) in every module here. The row for this
  package in `../AGENTS.md` once said Google-style; that was wrong (`grep -c "Args:"` is 0 in all files). Match
  the file you edit; never convert a file wholesale.
- **New code uses `import keras` only**, with `keras.ops.*` qualified at the call site. All eight top-level
  modules still say `from keras import ops` (`grep -n "^from keras" src/dl_techniques/layers/statistics/*.py`);
  do not copy that into new code, and do not rewrite a module just to remove it.
- **Sub-layers are created in `__init__`; weights in `build()`; `super().build(input_shape)` last.**
  Pinned by `test_the_package_is_v2_compliant.py` (V2: no sub-layer created in `build`).
- **`call` is symbolic.** No Python branch on a tensor value, no `.assign` in `call`, no logging, no layer
  construction. `DeepKernelPCA` and `InvertibleKernelPCA` therefore fit in an eager `adapt()` and branch in
  `call` on a Python list length, not on a tensor (`_fit_flag` is written and never read on purpose:
  `load_own_variables` counts weights, so deleting it breaks every saved file).
- **Dtypes come from the tensor, never a literal.** `float16(1e-10)` is exactly 0.0, so a stability floor is
  `stability_floor(self.compute_dtype, 1e-10)` from `dl_techniques.utils.dtype_policy`, and a constant is
  `ops.cast(value, tensor.dtype)`, not a bare `ops.sqrt(2.0)` or `ops.eye(n)`. Pinned by
  `test_the_eigenvalue_divides_survive_fp16.py` and `test_the_four_carried_defects_are_fixed.py`.
- **Initializers are cloned per site.** `MDNLayer` gives each Dense its own `clone_initializer(...)`; passing
  the shared instance made `mdn_mus.kernel == mdn_sigmas.kernel` at init (measured).
- **`MDNLayer`'s pi head emits raw logits.** No softplus or softmax on it: every consumer applies exactly one
  softmax or log_softmax. `seed` must reach both `keras.random` draws in `sample`.
- **A method that bypasses `Layer.__call__` builds first.** `AffineCouplingLayer.forward/inverse` and
  `NormalizingFlowLayer.sample` call `self.build(...)` when unbuilt, otherwise their weights land outside the
  layer's name scope (bare `dense_1/kernel`). Do not add a `call()` to fix it.
- **`ResidualACFLayer` never branches on a symbolic shape**: it slices with Python-int lag offsets and decides
  in-range from the static sequence length, and it logs nothing on the forward path.
- **`InvertibleKernelPCA` has one inverse path, a learned linear decoder.** Do not bring back the removed
  arccos / frequency-Gram pseudo-inverse code; a real pre-image solver is out of scope. Its seed guard is
  `is not None`, never truthiness (`random_seed=0` is falsy).
- **`DeepKernelPCA`'s fitted path runs no inter-level coupling.** Do not re-enable it to match the paper.
- **Validation names the value.** Every `raise ValueError(...)` here is an f-string with the offending value;
  `test_no_value_error_message_is_bare` fails otherwise. Pair each new raise with a test.
- **No caller-dict mutation, JSON-safe `get_config`.** Store specs, not instances.

## Tests

`tests/test_layers/test_statistics/`, one file per module, the conformal ones in `test_conformal/`.
Package-wide guards, run by the same directory:

| Guard | Pins |
|---|---|
| `test_the_package_is_v2_compliant.py` | An AST walk over every source file of the package, sub-package included (`rglob`), against the v2 guide: no sub-layer built in `build`, every `ValueError` names the value, cross-parameter contracts at construction, `training=` forwarded. `MIN_SOURCE_FILES` is its anti-vacuity floor |
| `test_the_package_meets_the_v2_test_bar.py` | §16.3 of the guide for the nine older layers (`SUBJECTS`): value round trip, weights before first call, build parity, dtype arms, XLA vs eager, gradients per variable. Strict `xfail` entries (`XFAILS`) carry the measured reason |
| `v2_compliance_oracle.py` + `test_the_v2_oracle_can_fail.py` | The shared instruments and the proof that each rejects a broken subject. The oracle has no `test_` prefix on purpose |
| `test_the_eigenvalue_divides_survive_fp16.py` | The two whitening divides under `mixed_float16` |
| `test_the_four_carried_defects_are_fixed.py` | The dtype and weight-path fixes of plan-2026-08-30T175846 |
| `test_the_statistics_docs_name_real_things.py` | This file, `README.md` and the conformal pair: paths exist, every class is named, every README snippet runs |

A new top-level layer: add it to `SUBJECTS` in the test bar, to the README catalogue and layout table, and to
the table above. A new conformal layer: `conformal/AGENTS.md`.

## Do not

- Do not convert Sphinx docstrings to Google style, or the reverse.
- Do not register with the bare decorator, and do not rename a registered class without saying which archives
  stop loading.
- Do not move an `adapt()` fit or any `.assign` into `call()`.
- Do not write `from keras import ops` in new code.
- Do not put a measured accuracy or speed number in the README or here: nothing in this package has been
  trained or benchmarked in this repository.
