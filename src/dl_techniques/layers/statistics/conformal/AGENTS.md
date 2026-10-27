# statistics/conformal

Split-conformal layers: one frozen scalar `q_hat`, the finite-sample quantile of held-out scores, applied
to a point estimate, to learned quantiles, or to classifier logits. The parent `../AGENTS.md` owns the
package rules; `README.md` is the user-facing tour. The numpy side of the same math lives in
`src/dl_techniques/utils/conformal_denoiser_intervals.py` (`conformal_quantile`, `calibrate_per_sigma`) and
`src/dl_techniques/utils/conformal_classification.py` (APS scores, sets, coverage); the trainer step is
`src/train/common/conformal.py`.

## Layout

`__init__.py` is curated: it exports `ConformalLayer`, `ConformalInterval`, `ConformalQuantileHead`,
`ConformalPredictionSet` and `attach_conformal_sets`, with `__all__`. Import from the package.

| File | Holds |
|---|---|
| `base.py` | `ConformalLayer`: `FROZEN`, `add_q_hat_weight`, `set_q_hat`, `calibrate`, `_normalize_q_hat`, `get_config` |
| `interval.py` | `ConformalInterval`: clips `mu` to `[domain_min, domain_max]`, returns `(mu_c, lower, upper)` or `(lower, upper)` |
| `quantile_head.py` | `ConformalQuantileHead`: a `Dense` to `(B, Tf, output_dim, 3)` quantiles, `predict_intervals` returns `(median, lower, upper)` |
| `prediction_set.py` | `ConformalPredictionSet` (logits to `(probs, in_set)`) and `attach_conformal_sets(model, q_hat, input_shape)` |

Registration keys follow the defining module: `dl_techniques.layers.statistics.conformal.base`, `.interval`,
`.quantile_head`, `.prediction_set`. `ConformalLayer` is registered too (like `BaseVisionHead`) but has no
`call` and is not a usable layer. `test_base_contract.py` pins the key of each concrete class.

## The base contract

Each rule is pinned by `tests/test_layers/test_statistics/test_conformal/test_base_contract.py` unless it
names another test.

- **One weight, one name.** `conformal_q_hat`: a scalar, `trainable=False`, `dtype=self.variable_dtype`,
  `autocast=False`, created by `add_q_hat_weight()`. A subclass calls it from its own `build()` before
  `super().build(input_shape)`. The old names `conformal_q` and `conformal_tau` do not come back.
- **`autocast=False` means the subclass casts it.** Under `mixed_float16` the variable stays float32, so
  `ConformalInterval` casts it to the output dtype and `ConformalPredictionSet` casts it to float32 for the
  comparison. A float16 `q_hat` would round a calibrated radius (possibly down, which under-covers) and make
  the set layer's comparison dtype-mismatch. Pinned by
  `test_prediction_set.py` (`test_the_threshold_is_not_rounded_to_float16_under_mixed_precision`).
- **`FROZEN`.** A layer whose only weight is `q_hat` sets `FROZEN = True`; the base then forces
  `trainable=False` and drops an inbound `trainable` (Keras round-trips it through the config). The head has a
  trainable `Dense`, so it leaves `FROZEN` false.
- **Validation lives in `_normalize_q_hat`.** Non-negative, not NaN, `ValueError` naming the argument and the
  value. A subclass may override it to bound the value and must call `super()` first
  (`ConformalPredictionSet` caps at `2.0`). `__init__`, `set_q_hat` and `calibrate` all go through it.
- **`calibrate(scores, alpha)` is the exact order statistic** (`conformal_quantile`), not an interpolated
  `np.quantile`. For n = 9 and alpha = 0.1 it is the largest score. It refuses an unbuilt layer, empty scores
  and an alpha outside `(0, 1)`; too few scores give `+inf`, which the set layer stores as `2.0`.
- **`get_config()["q_hat_init"]` is the current value**, so `from_config` keeps a calibration. Subclasses add
  their own keys with `config.update(...)` and never write `q_hat_init` themselves.
- **No second copy.** Validation, dtype handling and the config rule are written once, in the base. A
  subclass that needs a different rule overrides the hook, not the method.

## Per-layer rules

- **`ConformalPredictionSet` equals `predict_sets(logits, q_hat, randomized=False)`** of the numpy utility, to
  float32 rounding at the threshold. Pinned by `test_prediction_set.py` (`test_it_matches_the_numpy_prediction_sets`).
  It runs the deterministic `u = 1` rule on purpose (no random draw at inference); do not make it randomized,
  and do not change the rule on one side only. Ties break by class index, as the numpy rule does. It needs at
  least 2 classes (softmax over one class is always 1) and refuses rank other than 2. The comparison is in
  float32 whatever the policy.
- **`ConformalQuantileHead` keeps its `Dense` in `__init__`** (named `quantile_projection`) and builds it in
  `build()`. `call` never applies `q_hat`; only `predict_intervals` does, through `self(...)` so an unbuilt head
  builds, with `q_hat` cast to the output dtype. The return order is `(median, lower, upper)`: the median
  first. The kernel initializer is stored as a spec (`initializer_spec`), not an instance.
- **`ConformalInterval` clips, then widens.** `domain_min < domain_max` is checked at construction. Per-sigma
  (Mondrian) calibration inside one graph is out of scope: one radius covers one regime.
- **`attach_conformal_sets` calls the model on a symbolic `keras.Input`**, so it works for Functional and
  subclassed models, and refuses a multi-output model. A deep-supervised model must be reduced to its primary
  output first (`utils/deep_supervision.create_inference_model_from_training_model`).

## Adding a conformal layer

1. Subclass `ConformalLayer` in a new module here; set `FROZEN`; call `super().__init__(q_hat_init=..., **kwargs)`.
2. Call `self.add_q_hat_weight()` in `build()` before `super().build(...)`, and cast `self.q_hat` where you use it.
3. Register under `dl_techniques.layers.statistics.conformal.<module>`.
4. Add it to `__init__.py` and `__all__`.
5. Add a `Subject` to `SUBJECTS` in `test_base_contract.py`. The first test there fails until you do: it
   compares `SUBJECTS` with `ConformalLayer.__subclasses__()`.
6. Add a test file for what is specific to the layer, covering the guide's §16.3 arms (dtype policies, XLA vs
   eager, an unknown-batch trace, weights compared before the reloaded model's first call).
7. Add it to this file's table, to `README.md`'s table, and to the catalogue in `../README.md`.

## Do not

- Do not add aliases for the removed names: `ConformalIntervalLayer`, `ConformalPredictionSetLayer`, `q_init`,
  `tau_init`, `self.q`, `self.tau`, `conformal_q`, `conformal_tau`, or the old `calibrate(value)`. Archives
  that used them fail to load, by decision.
- Do not set `autocast=True` on `q_hat`, and do not store it in the compute dtype.
- Do not replace the order statistic with an interpolated quantile.
- Do not quote coverage or set-size numbers in these docs: none has been measured on a trained model.
- Do not use `from keras import ops`; this package is `import keras` only.
