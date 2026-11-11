# e-means: Integration Research & Plan

Date: 2026-11-11. Status: plan, no code.
Sources: arXiv 2611.09501 (Bessa, Mazare, Szilvasy, Touvron, Picard, Douze, Jegou); faiss `contrib/e_means/{README.md,base.py,e_means.py,e_means_torch.py,e_means_spherical.py}` (CuTe/Triton/quantizer/distributed/IVF files read for scope only).
Normative guides: `research/2026_keras_custom_models_instructions_v2.md`, `AGENTS.md`, `src/dl_techniques/AGENTS.md`, `src/dl_techniques/layers/AGENTS.md`, `src/dl_techniques/layers/mixtures/AGENTS.md`.
License note: reference code is MIT (Meta). Ported files keep a "Portions Copyright (c) Meta Platforms, Inc. and affiliates" MIT notice. No Claude attribution anywhere.

## 0. Context and scope

e-means is a mini-batch k-means with EMA centroid updates, a cosine step schedule, donor-split revival of dead clusters, a last-epoch closing mean, and two fitted scaling laws that set batch size and initial step from (n, k, T). It reaches Lloyd@25 loss in 1.4 to 2.6 epochs.

`src/dl_techniques/layers/mixtures/` already holds `KMeansLayer` (soft assignment, momentum + repulsion EMA, no schedule, no revival, no closing mean). e-means is a different algorithm, not a patch of it, so it lands as a new mixture layer plus a pure-ops module plus a fitter.

Port scope: the reference is torch/faiss/CuTe/Triton. We port the algorithm and recipe, in `keras.ops` only, TF 2.18 / Keras 3.8. We do not port GPU kernels, quantizers, faiss, torch, distributed, sharded-centroid or IVF code.

Files created on implementation:

| Path | Role |
|---|---|
| `src/dl_techniques/layers/mixtures/emeans_ops.py` | stateless ops: recipe laws, schedule, blockwise assign, batch stats, EMA update, revival, closing mean, normalize |
| `src/dl_techniques/layers/mixtures/emeans.py` | `EMeansLayer` and the `fit_emeans` driver |
| `src/dl_techniques/layers/mixtures/factory.py` | add key `emeans` (registry, `MixtureType` Literal, import) |
| `src/dl_techniques/layers/mixtures/__init__.py` | export `EMeansLayer`, `fit_emeans`, `resolve_emeans_recipe` |
| `src/dl_techniques/layers/mixtures/README.md`, `AGENTS.md`, `layers/AGENTS.md` | update hard-coded counts (4 keys -> 5, 11 exports -> new count) |
| `tests/test_layers/test_mixtures/test_emeans.py`, `test_emeans_ops.py`, `emeans_oracle.py` (no `test_` prefix), single-claim guards | tests |

## 1. What e-means is

Maintain per-cluster EMA count N_j and EMA sum S_j; the centroid is c_j = S_j / N_j.

```
N_j <- (1-a) N_j + a n_j        n_j, s_j = batch count and sum of points assigned to j
S_j <- (1-a) S_j + a s_j
c_j  = S_j / N_j
```

Components beyond plain EMA k-means:

| Component | Rule | Reference |
|---|---|---|
| Step schedule | a(t) = a0 (0.1 + 0.9 * 0.5 (1 + cos(pi t / M))), M = total updates, clamped at t = M | `base._scheduled_alpha` |
| Revival | every 5 updates, clusters with N_j < 0.1 mean(N) are rebuilt by splitting a donor drawn without replacement with prob proportional to N; donor and child get c * (1 +- delta * sign), sign alternates per coordinate, delta = 1e-4; N halved between them | `_choose_splits`, `_apply_splits` |
| Closing mean | last epoch records sums and counts per cluster; at the end each nonempty centroid is set to the exact mean of its last-epoch points, empty ones keep position | `finalize` |
| Batch law | B = C_B k^0.1347 n^0.8765 T^0.8133, nearest power of two, capped (throughput C_B = 0.035134; quality C_B = 0.00439175, about 8x smaller B) | `base._BATCH_LAWS` |
| Step law | a0 = 0.8244 k^-0.4617 B^0.9003 n^-0.4498 T^-0.1057, clamped to [1e-4, 1] | `base._ALPHA_LAW` |
| Auto defaults | quality law when d <= 16, throughput otherwise; revival off when d <= 2 | `_AUTO_QUALITY_MAX_D`, `_NO_REVIVAL_MAX_D` |
| Init | random rows without replacement; N_j = 2^-16 so the init carries negligible weight (paper says 1e-10; code uses 2^-16, we follow the code) | `_INIT_USAGE` |
| Spherical | centroids read out as unit vectors, assign by max dot product, closing mean renormalized; objective sum(2 - 2 sim) | `e_means_spherical.normalize_into` |

Laws were fit on DEEP only (d = 96). Law constants in the shipped code differ from the paper's Table 2 (the code says so). We copy the code constants, and record the discrepancy in the ops module docstring.

Epoch plan (`base._epoch_rows`): floor(T) - 1 full passes, an optional fractional prefix pass, then a full final pass (the closing pass). Fractional T is supported; T < 1 only contributes the prefix.

## 2. Math and ops required (all `keras.ops`)

| Need | Op | Note |
|---|---|---|
| Distances | `pairwise_squared_distance` (`utils/tensors.py`) or the dot form ||c||^2 - 2 x.c | reuse; do not reimplement |
| Nearest centroid without a B x k matrix at large k | python loop over static centroid blocks keeping running (min score, argmin) | k and block size are static, so graph-safe |
| Counts and sums | `keras.ops.segment_sum(data, ids, num_segments=k)` | float32 accumulation |
| EMA update | masked arithmetic on (k,) and (k, d) | `ops.where(n > 0, ..., c)`; no boolean indexing |
| Revival | Gumbel top-k on donors: key = log N - log(-log u), u from `keras.random.SeedGenerator` | fixed shapes: rank unused clusters with `cumsum(unused) - 1`, pair rank r with r-th best donor if r < n_donors, apply with `ops.where` and a scatter via `segment_sum` |
| Revival cadence | compute the pure revival result every step, select with `ops.where((step + 1) % 5 == 0, new, old)` | no `ops.cond` around variable assigns; cost O(k log k + k d) per step, small next to B k d assignment |
| Identical donor and child (zero centroid) | nudge coordinate 0 of the child to the neighbouring float (`_separate_splits`); in ops, add `eps * sign` fallback where donor == child | pin with a test |
| Schedule | `ops.cos` on a float step variable | step is a float32 variable, not a python int |
| Spherical readout | max-abs prescale, then L2 normalize, zero stays zero (`normalize_into`) | same op reused for init and closing |
| Row gather | `ops.take(x, idx, axis=0)` | not `take_along_axis` (BroadcastArgs XLA retrace trap) |

Precision rule from the paper (eq. 9 to 11): the centroid norm and the dot product must come from the same cast operands. If assignment runs in `compute_dtype` (bf16/fp16), cast centroids once and compute norms from that cast tensor; sums, counts, N, S and the objective stay float32 (`autocast=False`).

## 3. Reuse inventory (factory first)

| Need | Existing | Path |
|---|---|---|
| Cluster-axis plumbing, `output_mode` validation, initializer resolving | `BaseMixtureLayer`, `_ClusterAxisMixin`, `resolve_initializer_arg`, `resolve_prototype_initializer` | `layers/mixtures/base.py` |
| `training` as 0/1 factor, graph-safe | `resolve_training_factor` | `utils/tensors.py` |
| Pairwise distances | `pairwise_squared_distance` | `utils/tensors.py` |
| Non-trainable state conventions (`trainable=False, dtype=self.dtype, autocast=False`, in-`call` `.assign`) | `KMeansLayer` | `layers/mixtures/kmeans.py` |
| Stateless-ops + thin-layer split | `mean_shift_ops.py` + `mean_shift.py` | `layers/mixtures/` |
| Positive-int/float validators | `positive_int`, `positive_float` | `utils/validation.py` |
| Registration | `register_dl_technique("dl_techniques.layers.mixtures.emeans")` | `utils/keras_registration.py` |
| Initializer cloning | `clone_initializer` | shared initializer family |
| Logging | `dl_techniques.utils.logger` | no `print` |

Adjacent code, not reused, not duplicated:

- `KMeansLayer`: soft assign, momentum + repulsion. Stays as is. e-means is hard-assign with a scheduled EMA ratio. The docs get a comparison row.
- `VectorQuantizer`, `VectorQuantizerRotationTrick` (generative): codebook for VQ-VAE with commitment loss. Not wrapped by the mixtures factory. e-means could later seed `warm_start_codebook` (today sklearn MiniBatchKMeans); out of scope here, see section 6.

## 4. What must be newly written

### 4.1 `emeans_ops.py` (stateless, no layer state)

- `resolve_emeans_recipe(n, k, d, nepochs=5, batch_law="auto", batch_size=None, alpha0=None, unused_threshold=None)` returns a frozen dict: batch_size, alpha0, steps_per_pass, epoch_rows, epoch_steps, total_steps, batch_law resolved plus reason, unused_threshold resolved. Pure Python, mirrors `base._resolve_recipe`, `_epoch_rows`, `_round_power_of_two`, `_step_law`, `_resolve_batch_law`, `_resolve_unused_threshold`. Note the intentional asymmetry: throughput caps at `round_pow2(n/2)`, quality caps at `n // 2`.
- `scheduled_alpha(alpha0, step, total_steps)`: tensor-safe cosine.
- `assign_blockwise(x, centroids, block, spherical)` returns (score, labels); scores in model units.
- `batch_statistics(x, labels, k)` returns sums (k, d), counts (k,).
- `ema_update(c, N, sums, counts, alpha)`: the ratio form `c += (alpha n / N_new)(s/n - c)` for n > 0, N_new = (1 - alpha) N + alpha n; the alpha == 1 case handled by `weight == 1 -> mean`.
- `revive_unused(c, N, key, threshold, epsilon)` returns (c, N, n_split).
- `closing_mean(pass_sums, pass_counts, c, spherical)`.
- `normalize_rows(x)`.

### 4.2 `EMeansLayer(BaseMixtureLayer)`

Registered key `dl_techniques.layers.mixtures.emeans`. Constructor: `num_clusters, alpha0, total_steps, unused_threshold=None, check_unused_every=5, split_epsilon=1e-4, init_usage=2**-16, spherical=False, assignment_block=4096, closing_mean=True, centroid_initializer, output_mode, seed, **base`. Resolved recipe values (alpha0, total_steps, unused_threshold) are plain constructor args so `get_config` round-trips; `EMeansLayer.from_recipe(recipe, **kw)` builds from `resolve_emeans_recipe`.

Variables (all `trainable=False, dtype=self.dtype, autocast=False`, created in `build()`, none assigned in `build()`):

- `centroids (k, d)`, `usage (k,)` initialized to `init_usage` through an initializer, `step ()` float, `pass_sums (k, d)`, `pass_counts (k,)`, `closing_active ()`, `SeedGenerator` state.
- `pass_sums` and `pass_counts` exist only when `closing_mean=True`.

`call(x, training=None)`:

1. flatten leading axes to (B, d) via the cluster-axis helpers, cast to `variable_dtype`;
2. assign against the current centroids (readout: normalized if spherical); compute output first, then update (same DECISION as `KMeansLayer`, do not reorder);
3. if training: `factor = resolve_training_factor(...)`; stats, closing accumulation gated by `closing_active`, EMA update with `scheduled_alpha(step)`, revival select, `step.assign_add(factor)`;
4. cast output to `compute_dtype`.

Methods: `begin_closing_pass()` (zero accumulators, set flag), `finalize()` (closing mean into centroids, clear flag), `reset_centroids(rows)`, `initialize_from_data(rows, seed)` (random rows, eager, outside `call`).

Outputs (`VALID_OUTPUT_MODES`, declared once on the class, the factory reads it): `assignments` (hard one-hot), `mixture` (quantized rows gathered from centroids), `indices` (int labels). Final set decided after reading `base.py`; it must stay a separate set per class (drift test).

Limitations stated in the docstring (same as `KMeansLayer`): replica-local update, invalid under multi-replica `tf.distribute`; masks ignored; revival uses a layer-owned `SeedGenerator`.

### 4.3 `fit_emeans(x, k, nepochs=5, ...)`

Host-side driver that reproduces `Emeans.train`: resolve recipe, init from random rows (seeded), build the layer from the recipe, per epoch shuffle (`_epoch_seed(seed, epoch)` derivation), batches of recipe size with a retained remainder, `begin_closing_pass()` before the last epoch, per-batch layer call under a `tf.function` wrapper, labels captured in the last epoch, `finalize()`. Returns `(centroids, labels)` plus the recipe and per-epoch objective. This is the single algorithm implementation: the layer is the engine, the driver only loops. Also `max_points_per_centroid` subsample (label -1 for dropped rows) and `assign(x)` for new rows.

### 4.4 Factory, exports, docs

`factory.py`: import, `MixtureType` Literal key `emeans`, registry entry with `class, description, required_params, optional_params, use_case`; `optional_params` mirrors constructor defaults exactly (`test_factory_registry_drift.py`). `_check_registry_literal_consistency` must pass. Update `README.md` (comparison table, e-means section), `AGENTS.md` (counts, key files, invariants), `layers/AGENTS.md` repeats.

## 5. Integration shape (house rules)

- `import keras`, qualify `keras.ops.*`; no raw TF in the forward path; Sphinx/reST docstrings; `from ...utils.logger import logger`; log once in `__init__`.
- No sub-layers. `build()` creates variables, calls `super().build()` last. No `.assign` in `build()` (StatelessScope trap); tables via initializers.
- `get_config` returns every constructor argument including serialized initializers and the seed.
- `# DECISION plan-<id>/D-NNN` comments at: N init 2^-16 over paper 1e-10; code law constants over paper Table 2; compute output before update; revival computed every step and selected by `where`; no spherical readout cache (normalize each call, cheap vs assign). Anchors enter `plans/ANCHORS.md` at CLOSE.
- No legacy-compat shims, no fallback backends. If a configuration cannot run (for example `closing_mean` requested without a final pass), raise in `__init__` or `finalize` with the value named.

## 6. Risks and open items

| Risk | Mitigation |
|---|---|
| Scaling laws fit at d = 96 on GPU-sized n; unvalidated for Keras/TF small-n regime | measured validation phase on synthetic mixtures and a real embedding set; report per dataset, not a mean |
| Revival shape logic (rank pairing, donor == child) is the easiest place to be silently wrong | NumPy oracle port of `_choose_splits/_apply_splits`, compared on identical uniforms; mutation-proven guards |
| `segment_sum` and top-k under XLA | `@tf.function(jit_compile=True)` vs eager test; flat-row `take` instead of `take_along_axis` |
| bf16 assignment ties and norm coherence | norms from cast centroids; test fp32 control vs `mixed_bfloat16`; assert fraction of changed assignments is small and measured |
| fp16 overflow of norms for large-magnitude data | document; finite check in the fitter |
| Symbolic `training` | `resolve_training_factor`; test inside `fit()` |
| Multi-replica `tf.distribute` | documented as invalid (same as `KMeansLayer`); `strategy.reduce` of sums/counts is a later phase, not scoped |
| Overlap confusion with `KMeansLayer`, VQ layers | comparison table in README; no edits to those classes |
| The 1 to 3 epoch claim is measured on H100 for the reference | we claim only epochs-to-Lloyd@25 (hardware independent), never wall-clock |

Open decisions, recommended defaults:

1. `output_mode` set for `EMeansLayer`: recommend `{assignments, mixture, indices}`; settle after reading `base.py`.
2. Fitter location: recommend in `emeans.py`, exported; a `src/train/` script is not warranted for a layer.
3. Out of scope now: distributed centroids/replicas, IVF assignment for very large k, FP8/INT8, slow-EMA out-of-sample finalizer (paper appendix I), k-means++ init (paper G.1: marginal), wiring into `VectorQuantizer*` warm start.

## 7. Implementation phases

| Phase | Work | Gate |
|---|---|---|
| 0 | read `base.py`, `kmeans.py`, `tests/.../test_kmeans.py`, `utils/tensors.py`; settle output modes | notes in plan dir |
| 1 | `emeans_ops.py` + `emeans_oracle.py` (NumPy, brute-force assign, no faiss) + `test_emeans_ops.py`: recipe laws against hand-computed values from the README formulas, `epoch_rows`, schedule endpoints, EMA update vs oracle, revival vs oracle, closing mean | pytest on the new file, RED proofs recorded |
| 2 | `EMeansLayer` + `test_emeans.py` (init, shapes, forward, training moves centroids, eval does not, `.keras` round trip, config, symbolic training, `mixed_float16`/bf16, float64, XLA vs eager, stateless-build guard, dead-cluster revival, d <= 2 revival off) | pytest on the file |
| 3 | `fit_emeans`: closing mean equals mean of last-pass labels exactly; fractional T; `max_points_per_centroid`; seed reproducibility; `assign` | pytest |
| 4 | factory, `__init__`, docs, counts | `pytest tests/test_layers/test_mixtures/ tests/test_layers/test_factory_registry_drift.py -q`; `--collect-only` after the public-surface change; serialization registry and package-API guards |
| 5 | measured validation (GPU 0 only, one job, free memory checked first): synthetic Gaussian mixtures (several n, k, d incl. d <= 16) and one real embedding set; epochs to Lloyd@25 (sklearn/NumPy Lloyd baseline), loss at T in {1, 3, 5}, ablations (no cosine, no revival, no closing mean). Results go into this document as a table with noise across seeds | table, not prose |

Test scope for every phase: only `tests/test_layers/test_mixtures/` plus the drift, serialization-registry and package-API guards. Never the full suite. No test or probe writes to `results/` (use `tmp_path`).

## 8. Verification (when implemented)

- `.venv/bin/python -m pytest tests/test_layers/test_mixtures/ tests/test_layers/test_factory_registry_drift.py -q` (quote passed and collected counts).
- Round trip: build, train a few steps, save `.keras`, load, compare outputs and all non-trainable variables at `atol=1e-6, rtol=0`.
- Oracle parity: one batch update, one revival event, one closing mean match the NumPy oracle to float32 tolerance, measured.
- Integration: `fit_emeans(x, k, nepochs=3)` test loss at or below Lloyd (25 iterations) on the synthetic mixture; tolerance taken from a measured seed spread.

## References

- Bessa et al., e-means: outperforming k-means (in one-tenth the epochs), arXiv 2611.09501.
- faiss `contrib/e_means` (README, base.py, e_means.py, e_means_torch.py, e_means_spherical.py).
- van den Oord et al., Neural discrete representation learning (EMA codebook); Sculley, Web-scale k-means; Lloyd 1982; Arthur and Vassilvitskii 2007.
