# time_series

Forecasting and sequence layers for `dl_techniques`: N-BEATS / N-BEATSx blocks, xLSTM cells and
blocks, the PRISM wavelet tree, DeepAR heads, quantile heads, a TCN, lag attention, temporal fusion,
the naive-benchmark gate and EMA filters. The parent `../AGENTS.md` owns the cross-package rules;
`README.md` is the user-facing tour with runnable examples.

## Layout

`__init__.py` is curated: 32 names in `__all__`, all classes except the builder
`create_manokhin_compliant_model`. There is **no factory**; import a class by name.

| File | Holds |
|---|---|
| `nbeats_blocks.py` | `NBeatsBlock` (abstract base: dense spine + theta heads), `GenericBlock`, `TrendBlock`, `SeasonalityBlock` |
| `nbeatsx_blocks.py` | `ExogenousBlock` (basis built from covariates, optional TCN encoder) |
| `temporal_convolutional_network.py` | `TemporalBlock`, `TemporalConvNet` (dilation `2**i`, residual inside each block) |
| `xlstm_blocks.py` | `sLSTMCell`, `sLSTMLayer`, `sLSTMBlock`, `mLSTMCell`, `mLSTMLayer`, `mLSTMBlock` |
| `prism_blocks.py` | `FrequencyBandStatistics`, `FrequencyBandRouter`, `PRISMNode`, `PRISMTimeTree`, `PRISMLayer` |
| `deepar_blocks.py` | `ScaleLayer`, private `_LikelihoodHead` base, `GaussianLikelihoodHead`, `NegativeBinomialLikelihoodHead`, `DeepARCell` |
| `quantile_head_fixed_io.py` | `QuantileHead` (fixed horizon, optional input flatten) |
| `quantile_head_variable_io.py` | `QuantileSequenceHead` (per-position quantiles) |
| `mixed_sequential_block.py` | `MixedSequentialBlock` (`lstm` / `transformer` / `mixed`, Pre-LN) |
| `adaptive_lag_attention.py` | `AdaptiveLagAttentionLayer` |
| `temporal_fusion.py` | `TemporalFusionLayer` |
| `forecasting_layers.py` | `NaiveResidual`, `ForecastabilityGate`, `create_manokhin_compliant_model` |
| `ema_layer.py` | `ExponentialMovingAverage`, `EMASlopeFilter` |

Re-derive the class list rather than trust this table:

```bash
for f in src/dl_techniques/layers/time_series/*.py; do
  echo "$(basename $f): $(grep -oP '^class \K\w+' $f | tr '\n' ' ')"; done
```

Consumers outside the package are `src/dl_techniques/models/time_series/*` (adaptive_ema, deepar,
nbeats, prism, timestral, tirex, xlstm). Find them with
`grep -rln "layers.time_series" src --include=*.py | grep -v layers/time_series/`.

## Shared machinery: reuse it, do not re-implement

| Need | Use |
|---|---|
| Dense spine of an N-BEATS block | `NBeatsBlock._forward_dense_stack(inputs, training)`. A subclass that overrides `call` (as `ExogenousBlock` does) calls this, never a copy of the loop |
| Fixed-basis expansion (reshape, matmul, transpose, flatten) | `NBeatsBlock._expand_fixed_basis(theta, basis_matrix, dim, length)` |
| A new two-parameter DeepAR head | Subclass `_LikelihoodHead`, set `SECOND_PARAMETER` and `POSITIVE_MEAN`. Positive parameters go through `_floored_softplus` |
| Non-crossing quantiles | `layers/activations/monotonicity_layer.py::MonotonicityLayer(method="cumulative_softplus", axis=-1)`, as both quantile heads do |
| One initializer handed to several sub-layers | `dl_techniques.initializers.clone.clone_initializer` at every site (see Conventions) |
| A positive floor in a reduced-precision dtype | `stability_floor(self.compute_dtype, value)` from `dl_techniques.utils.dtype_policy` |

## Conventions

- **Docstrings are Sphinx/reST** (`:param:`, `:type:`, `:raises:`). Match the file; never convert one wholesale.
- **Registration** is `@register_dl_technique("dl_techniques.layers.time_series.<module>")`. The private
  `_LikelihoodHead` base is deliberately unregistered: it is never serialized on its own.
- **Imports.** Eleven modules still carry `from keras import ...`; `tests/test_the_keras_submodule_import_population_cannot_grow.py`
  is a ratchet on that population. New code writes `keras.ops.*` and adds no new `from keras import` line.
- **Sub-layers are created in `__init__`, built in `build()`, used in `call`.** A sub-layer that only some
  configurations need is still created unconditionally, and `build()` builds only what `call` runs.
  `TemporalBlock.downsample` is the example: always created, built and used only when
  `input_channels != filters` (`project_residual`). `ForecastabilityGate.complexity_analyzer` was once
  constructed in `build()`; it no longer is. No layer here constructs anything in `build()` or `call`.
- **Every sub-layer gets its own initializer.** One Keras `Initializer` instance replays one sample at
  every site it is handed to. Measured before the fix: N-BEATS `dense2 == dense3 == dense4` and the two
  theta heads were bit-identical at init, as were two `TemporalFusionLayer` head pairs, and
  `ForecastabilityGate`'s gate kernel was its hidden kernel's prefix (r = 1.0). Wrap every
  `self.<x>_initializer` handed to a sub-layer in `clone_initializer`. Guarded by
  `test_the_time_series_kernels_start_independent.py` and the repo ratchet
  `tests/test_the_shared_initializer_population_cannot_grow.py`, which counts a list comprehension of
  sub-layers as a loop-carried site.
  `prism_blocks.py` and `xlstm_blocks.py` pass the raw constructor argument through. That is safe at the
  string defaults, which resolve a fresh instance per site; a caller-supplied instance would alias.
- **Static configuration branches in Python, not in the graph.** `ExponentialMovingAverage.adjust` picks
  the scan step at trace time; do not reintroduce an `ops.where` on a constructor flag.
- **Keep static shapes static.** Flatten with `ops.reshape(x, (-1, x.shape[1] * x.shape[2]))` where
  `build()` already requires both trailing dims (`ForecastabilityGate`, `QuantileHead(flatten_input=True)`).
  Never construct `keras.layers.Flatten` inside `call`.
- **`activity_regularizer` is charged once, by `Layer.__call__`.** `AdaptiveLagAttentionLayer` and
  `TemporalFusionLayer` must not `add_loss` it again (D-007, D-008).

## N-BEATS specifics

- The spine is three lists: `dense_layers` (named `dense1..dense4`), `norm_layers` (empty unless
  `use_normalization`), `dropout_layers` (empty unless `dropout_rate > 0`). Their depth is
  `NBeatsBlock.NUM_DENSE_LAYERS`.
- Flat layout is **time-major**: entry `t * dim + i` is step `t` of channel `i`, for every block.
  `ExogenousBlock` reshapes theta dim-major `(B, dim, C)` for this reason (D-009); the channel-major
  spelling type-checks and silently transposes the stream. Its `input_dim == output_dim == 1` fast path
  is kept because the general path is not bit-identical on CPU (D-010).
- `ExogenousBlock` does **not** take `thetas_dim`; passing it raises `TypeError`. Its theta width is the
  basis channel count (`tcn_filters` with `use_tcn`, else `exogenous_dim`), passed to the parent as
  `thetas_dim` so the inherited heads carry `theta_initializer` and `theta_regularizer`. Before this,
  bare replacement heads made `theta_regularizer` inert on exogenous stacks (0 losses under `L2(1.0)`).
  `models/time_series/nbeats/nbeatsx.py` therefore passes `thetas_dim` only to trend, seasonality and
  generic blocks; an exogenous stack's entry in the model's `thetas_dim` list is ignored.
- `TrendBlock` / `SeasonalityBlock` basis matrices are non-trainable weights materialized by an
  initializer closure, not `add_weight(zeros) + assign` (D-028: `StatelessScope` discards the assign).
- `GenericBlock.basis_initializer=None` resolves to `Orthogonal(gain=0.1)` (D-072).

## Decision anchors

A `# DECISION <plan>/D-NNN` comment marks a choice that was measured. Read it before changing the line
below it, and keep the anchor with the code it guards.

| Anchor | Where | Rule |
|---|---|---|
| D-001 (2026-05-12) | `ema_layer.py` | `adjust=True` divides each step and feeds the divided value back; not pandas `ewm`, no `associative_scan` |
| D-009 (2026-08-31) | `ema_layer.py` | floor via `stability_floor`, never `ops.cast(1e-10, dtype)` (0.0 in float16) |
| D-011 | `mixed_sequential_block.py` | only allowlisted `attention_args` keys (`window_size`) may be dropped for a type that rejects them |
| D-021 | `mixed_sequential_block.py` | `DifferentialFFN` takes `branch_activation`; rename, do not drop |
| D-037 | `deepar_blocks.py` | keep `ops.softplus` plus the `MIN_LIKELIHOOD_PARAM` floor |
| D-004 (2026-08-30) | `forecasting_layers.py` | `ForecastabilityGate` without `forecast_length` warns, never raises |
| D-001/D-014 (xLSTM) | `xlstm_blocks.py` | log-domain stabilizer state `m_t`; floors in the promoted dtype |
| D-001/D-004/D-014/D-050 (PRISM) | `prism_blocks.py` | one segment-geometry helper; static-length branches; finite-only statistic repair |

## Tests

`tests/test_layers/test_time_series/`: one comprehensive suite per module plus sentence-named guards
(`test_the_six_time_series_defects.py`, `test_the_ema_floor_is_not_cast_to_zero.py`,
`test_the_xlstm_floors_survive_fp16.py`, `test_basis_matrices_survive_stateless_build.py`,
`test_the_exogenous_theta_regularizer_is_charged.py`, `test_the_time_series_kernels_start_independent.py`).

Scope a run to this package plus its consumers. Never run the full suite as a regression check:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest -q tests/test_layers/test_time_series/ \
  tests/test_models/test_{adaptive_ema,deepar,nbeats,prism,time_series,timestral,tirex,xlstm}
```

A pure refactor here must be bit-identical. Overwrite the trainable weights from a seed keyed by sub-layer
role, not by variable order or auto-generated name, and compare outputs at `atol=0` before and after.
