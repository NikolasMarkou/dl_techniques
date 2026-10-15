# Statistics Review — `src/dl_techniques/layers/statistics/` (2026-10-15)

Method: static re-read of all 9 source files + consumer call sites, CPU-only (`CUDA_VISIBLE_DEVICES=""`); no GPU used. No source code changed during review. Probe scripts ran eager forward passes, build/call edge cases, and save/load round-trips on CPU. Findings double-checked in a second pass; verdicts below mark confirmations, corrections, and downgrades.

Package: `UnifiedScaler` (`scaler.py`), `MovingStd` (`moving_std.py`), `MDNLayer` + helpers (`mdn_layer.py`), `ResidualACFLayer` + `ACFMonitorCallback` (`residual_acf.py`), `DeepKernelPCA` (`deep_kernel_pca.py`), `InvertibleKernelPCA` + `InvertibleKernelPCADenoiser` (`invertible_kernel_pca.py`), `AffineCouplingLayer` + `NormalizingFlowLayer` (`normalizing_flow.py`), `ConformalIntervalLayer` (`conformal_interval.py`), `LocalSupportGate` (`local_support_gate.py`). `__init__.py` empty (exports-nothing, no `factory.py` — intentional per `layers/AGENTS.md`).

## Executive summary

No forward-pass math bug found in the scaler / moving-std / ACF / flow-likelihood / conformal paths at float32 defaults — the documented numerics (two-pass variance, log-space NLL, `-sum(log s)` sign, single-radius conformal) are correct. All high-value findings are build-vs-call validation gaps, stale/unsafe serialization idioms, unfitted-path crashes, and silent-misconfiguration traps: `num_features` mismatch building then crashing at call, pre-`adapt` DeepKPCA crash when `batch > features`, RBF-only fitted path ignoring `kernel_type`, `projection_dim=None` still random-projecting, weights-only gate restore losing `is_fitted`, missing `q`/domain guards, and half-precision floor inconsistency in the ikPCA inverse. One original "duplicate registration" claim was refuted (keys are `package>ClassName`, all distinct).

## 1. Confirmed bugs / validation gaps

- **ST-1 `UnifiedScaler` `num_features` mismatch builds, crashes at call** (`scaler.py:346-347,468`): `num_features=5` + last-dim 3 accepts build with affine shape `(8,)`, then `call` fails `Incompatible shapes [2,8,3] vs [8]`. No check that explicit `num_features` matches input. Normal axes (`1`, `(1,2)`, `-1`) broadcast fine — the heuristic works, just unchecked.
- **ST-2 `stored_std` is not a pooled std + last-batch-wins** (`scaler.py:458-464`): `stored_mean/std = mean over batch of per-call stats`, verified `1.0 → 10.0` across batches. `mean(std)` understates population std; documented as "refresh every call" so not a lie, but streaming callers expecting a running average get last-batch-only. The `inverse_transform`/`get_stats` trap (MAE ~1.0 unless `axis=0`) is correctly documented (`:510-519`) — keep the docs.
- **ST-3 DeepKPCA pre-`adapt` crash when `batch > features`** (`deep_kernel_pca.py:774`): `projection[:batch,:]` slice requires `feature_dim >= batch` at every level. Verified `batch=5, level-1 dim=4` → `InvalidArgumentError MatMul [5,5]@[4,2]`. Documented constraint (`:274-282`), still a crash, not a fallback.
- **ST-4 DeepKPCA fitted path is RBF-only** (`deep_kernel_pca.py:1066-1084`, `adapt`): `_fitted_transform` calls `_rbf_pairwise_keras` unconditionally; `kernel_types` never consulted post-fit. `polynomial/linear/sigmoid/cosine` silently ignored after `adapt`. Source-search confirmed.
- **ST-5 ikPCA inverse whiten floor mismatch** (`invertible_kernel_pca.py:746` vs `:687-690`): `call` uses `stability_floor(dtype,1e-10)`, inverse uses bare `+ 1e-10`. `float16(1e-10)==0` → divide-by-zero risk under mixed precision. One-line inconsistency.
- **ST-6 ikPCA `compute_output_shape` pre-build `(None, None)`** (`invertible_kernel_pca.py:828-829`): `n_components=None` → returns `(batch, None)`, breaks Functional shape inference before build.
- **ST-7 Gate `projection_dim=None` still random-projects** (`local_support_gate.py:445-451,887`): "disabled" builds `Dense(input_dim→input_dim)` with seeded Gaussian kernel (verified `max|W−I|=1.85`), and `call`/`_project_sample` always apply it. Either bypass or rename.
- **ST-8 Gate `is_fitted` lost on weights-only restore** (`local_support_gate.py:966-1024`): flags live in Python dict, restored only via `from_config`. `get/set_weights` round-trip → `False/False` despite restored weights (verified). `from_config` path fine; weight-only path silently routes everything closed.
- **ST-9 Conformal missing guards** (`conformal_interval.py`): `q_init=-1.0` and `domain_min=1.0 > domain_max=0.0` both accepted and run (inverted clip silently yields `1.0`, verified). No `q>=0` / `domain_min<domain_max` checks. Tuple return is Sequential-incompatible (Functional-only, unstated).
- **ST-10 Shared MDN regularizer instances** (`mdn_layer.py:240-241`): `L2(1e-5)`/`L2(1e-6)` mutable defaults → `a.kernel_regularizer is b.kernel_regularizer == True` (verified). Low severity (no in-tree mutation) but real sharing.
- **ST-11 Flow `sample` seedless** (`normalizing_flow.py:871-876`): no `seed` param unlike `MDNLayer.sample(seed=)` — non-reproducible sampling with no API fix.

## 2. Downgraded / nuanced (first pass overstated)

- **MDN lambda does NOT break `.keras` save** (`mdn_layer.py:326`): standalone inner-`Dense` deserialize raises `TypeError`, but full-model `save/load_model` round-trips cleanly at `min_sigma=0.5` and `5.0` with `max|delta|=0.0` and config preserved. Keep as hygiene (unsafe-lambda warning path + standalone failure → prefer a named function), not a save blocker.
- **Registration "duplicates" refuted**: `normalizing_flow` ×2 and `invertible_kernel_pca` ×2 share package strings but full keys are `package>ClassName` — all four distinct (verified via `get_registered_name`). No collision.
- **MDN `loss_func:580` reshape**: mismatched batch raises loudly (`Incompatible shapes [3,1,2] vs [4,2,2]`), just two ops downstream. "Masks" → "defers/misattributes".
- **Flow `ops.log(2*np.pi)` (`:858`)**: float32 loss is float32/finite — no observed failure. Mixed-precision pin only; hygiene, not bug.
- **`MovingStd.compute_output_shape:296`**: `hasattr(pooler,'_build_input_shape')` is `False` even when built, so it always takes the throwaway-pooler path — which returns the correct shape (verified `(None,16,16,3)`). Brittle to Keras internals, not currently wrong.
- **ResidualACF dynamic-T**: `build([(None,None,2),...])` leaves `_seq_length=None` (verified); that regime falls back to `centered[...,:-lag]` slicing — the exact construct D-003 warns about. Holds only for fully-dynamic T, not the tested static-short-seq case (verified `(2,11,2)` correct with zeros past `T`).
- **Ceiling/MAE numbers** (ikPCA 0.9992-vs-zeros / ~0.76 best-linear; scaler MAE 1.02/1.40) are doc-cited measurements, not re-measured here.

## 3. Consumers (verified)

- **MDN**: `models/time_series/mdn/model.py:209,271,430`, `train/time_series/mdn/train_mdn.py:47`, `export.py:56` — correct wiring (`loss_func` as compile loss, `sample(seed+i)` loop). Gap: `MDNModel.__init__` exposes no `diversity_regularizer_strength` (signature confirmed) — post-hoc `model.mdn_layer` mutation is the only path. Helpers double-`predict`: `get_uncertainty` calls `model.predict` even when caller already predicted for `point_estimates` — wasteful, diverges under dropout/BN.
- **Gate**: `layers/adapters/gated_adapter.py:84,324` — one gate/phase, held open during its own training. Correct composition; gate carries no gradient by design.
- **Conformal**: `utils/conformal_denoiser_intervals.py` fits `q` host-side (`conformal_quantile`/`calibrate_per_sigma`); caller `calibrate(q)` assigns. Single-radius-per-graph by design; per-sigma Mondrian in one graph is out of scope.
- **Unconsumed**: `grep src/train + src/models` for `UnifiedScaler|MovingStd|ResidualACF|DeepKernelPCA|InvertibleKernelPCA|NormalizingFlow|ConformalInterval` → zero hits. Tests-only surface; maintenance without callers.
- **Other confirmed mechanics**: gate `call` hard-casts float32 (`:866,887`), shared `em_step_counter` across pos/neg with pos-only reset (`reset:841`); `ResidualACF.use_absolute_acf` is a loss no-op (abs then square/abs, `:431-439`, documented); `ACFMonitorCallback.batch_count` never resets per epoch; `Denoiser` float `n_components` scales `n_random_features` not variance; `fit_transform` runs `call` twice discarding the first; `ConformalIntervalLayer.get_config:186` intentionally carries live `q` (D-004).

## 4. Test gaps (observed absences)

- Scaler `num_features`-mismatch, pooled-vs-mean std, affine-broadcast matrix; DeepKPCA `batch>features` pre-adapt + non-RBF post-fit equivalence; ikPCA pre-build shape + fp16 whiten round-trip; gate `projection_dim=None` identity-vs-random + weights-only flag restore; conformal negative-`q` / inverted-domain rejection; flow `sample` determinism (no API — needs feature first); MDN standalone inner-Dense deserialize.

## 5. Non-issues (checked, cleared)

- Scaler affine broadcast on normal axes, `inverse_transform` trace-scope `RuntimeError`, NaN-replacement path as documented; `MovingStd` shifted two-pass + `channels_first` shape preserved; MDN log-space NLL finite on zero-sigma caller input (floored, large-but-finite ~1e6), pi-logits single-softmax contract; ACF lag-0=`1.0`/out-of-range=`0.0`, `training is True` gate; flow sign (`-sum(log s)` added) and finite loss; gate unfitted-returns-zeros + smoothed shape; registration keys distinct; MDN full-model save/load at extreme `min_sigma`.

## Suggested order

ST-5 (one-line floor) → ST-1/ST-9 (fail-fast validation) → ST-7/ST-8 (projection bypass + flag persistence) → ST-3/ST-4 (pre-adapt guard message + `kernel_type`-ignored warning) → ST-2/ST-10/ST-11 (stored-stat semantics, regularizer defaults, flow seed) → consumer items (expose diversity knob or document post-hoc path; single-predict uncertainty helper).
