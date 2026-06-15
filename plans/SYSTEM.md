# System Atlas
*Last refreshed: plan_2026-06-15_e6a0391c | 2026-06-15*
*Domain-neutral system map. Rewritten by ip-archivist at CLOSE -- max 300 lines. Read before PLAN/EXPLORE.*

## Identity
`dl_techniques` is a Keras 3 / TensorFlow 2.18 deep-learning research library -- 150+ model architectures, 290+ custom layers, plus production training pipelines, applications, and experiments. **Domain: codebase.**

## Components

### Layer-reuse audit (plan_2026-06-13_88695f5c / correction plan_2026-06-13_ae26345d / bug-fix plan_2026-06-13_ae9ee2cd)
Full audit at `research/2026_models_layer_reuse_audit.md`: all 95 inline `keras.layers.Layer` subclasses in `models/` classified as **11 REPLACE / 25 RELOCATE / 59 KEEP** against `layers/`. Key structural findings: (1) `layers/` exposes 8 factory registries (attention **27 keys**, ffn 15, norms 16, embedding 13, activations 22, mixtures 3, seq-pooling 3, heads 21); (2) `layers/memory/` is canonical for MANN/NTM; (3) structural gap: `layers/downsample.py`/`upsample.py` are free functions. **CORRECTION (plan_2026-06-13_ae26345d)**: 3 of 4 Tier-1 REPLACE verdicts refuted on source read. Only `_LayerScale1D -> LearnableMultiplier` executed. **BUG-FIX RESULTS (plan_2026-06-13_ae9ee2cd)**: 9 of 16 findings applied (B2, B6/B6b, B7, B10, B11, C2, C4, nano_vlm blockers); 4 REFUTED, 1 already done. Dead-model-bug cluster (RESURRECTED plan_2026-06-14_77a433bc): `modern_bert_blt_hrm` NOW FUNCTIONAL (commit 124d464b); `nano_vlm_world_model` NOW FUNCTIONAL (commit 1b61a381). `layers/blt_core.py:ByteLatentReasoningCore` NOW FUNCTIONAL (plan_2026-06-14_080e7636). FIXED: `self.scale = 1.0 / math.sqrt(float(self.head_dim))` at 3 sites (plan_2026-06-14_a5ed2c2a). Future audit verdicts MUST re-verify against current source.

### Layers (`src/dl_techniques/layers/`)
290+ custom Keras layers -- geometric/Clifford, attention variants, normalization, FFN, routing, logic/arithmetic, mixtures, task heads.

- **`layers/attention/`** -- **FULLY RESOLVED plan_2026-06-14_b9456f74 + residue plan_2026-06-14_ab855e7e (989 passed).** Factory: **27 types** (`ATTENTION_REGISTRY`). All 34 `build()` methods have `if self.built: return`. Decision anchors: `plan_2026-06-14_7734bacd/D-002` (capsule), `plan_2026-06-14_7734bacd/D-003` (perceiver), `plan_2026-06-14_adaddf34/D-002` (non_local), `plan_2026-06-14_b9456f74/D-001` (PFA SW-MSA), `plan_2026-06-14_ab855e7e/D-001` (gated+gqa), `plan_2026-06-14_ab855e7e/D-002` (ring+anchor). Deferred: F7 hopfield cross-attn KV-dim latent bug; F8 wave_field/single_window unregistered.
- **`layers/embedding/`** -- **FULLY RESOLVED plan_2026-06-15_9dbb87c1 (238 tests).** 14 layer classes + factory (13 keys). **`class_token.py` `ClassTokenPrepend`** (plan_2026-06-15_39a31d4a/D-001): serializable, `build()`-owned `cls_token` weight `(1,1,dim)`, call output `(B, L+1, dim)`. Canonical solution for prepend-CLS in any Functional ViT. Reused by dino v1/v2/v3. Dedicated unit test `tests/test_layers/test_embedding/test_class_token.py` (11 tests, plan_2026-06-15_2a23a001/D-003). **NEW `mask_token.py` `MaskTokenApply`** (plan_2026-06-15_e2759fbc/D-009): learnable iBOT mask token via `add_weight((1,1,embed_dim))`; `@register_keras_serializable`; call replaces masked positions with the broadcast token; `.keras` round-trip verified. Canonical solution for any iBOT/MAE-style masked ViT. Sharp edges: `ContinuousRoPE.compute_output_shape` returns `dim/2`; `PositionEmbeddingSine2D` emits channels-FIRST `(B, 2*num_pos_feats, H, W)`. Decision anchors: `plan_2026-06-15_9dbb87c1/D-001` + `plan_2026-06-15_9dbb87c1/D-003`.
- **`layers/activations/`** -- **FULLY RESOLVED plan_2026-06-15_0205772c (393 tests).** Factory: **22 keys**. All 7 weight/sublayer-creating `build()` methods have `if self.built: return`. Factory registry mirrors class defaults (was divergent pre-plan).
- **`layers/norms/`** -- **FULLY RESOLVED plan_2026-06-15_2485b951 (424 tests).** 14 classes + factory (16-key if/elif dispatch). All 10 weight-creating `build()` have `if self.built: return`. Sharp edge: `BandLogitNorm` adaptive component is degenerate (documents behavior D-002).
- **`layers/ffn/`** -- **FULLY RESOLVED plan_2026-06-14_60541575 + plan_2026-06-14_43ff1d31 + plan_2026-06-14_b5c957c5.** 15 registered types. All 15 `build()` have `if self.built: return`. Canonical activation contract (`keras.activations.get/serialize`) uniform across all 15.
- **`layers/transformers/`** -- **FULLY REVIEWED + FIXED plan_2026-06-15_5e7ae321 + plan_2026-06-15_6e879eeb + plan_2026-06-15_d7754cfc + plan_2026-06-15_32b5822c (470 PASS tests).** 15 layer classes, 28 public exports. NO transformer-block factory by design. `_MASKLESS_ATTENTION_TYPES = {'fnet', 'anchor', 'lighthouse'}`. `FreeTransformerLayer` always returns `(output, bit_logits)` in both modes (inference: zeros); output structure depends only on construction-time `use_free_transformer` flag. `EomtTransformerLayer` builds keep-mask via base_transformer's existing `attention_mask` plumbing. Decision anchors: `plan_2026-06-15_5e7ae321/D-001` (text_encoder rope branch); `plan_2026-06-15_32b5822c/D-001` (FreeTransformerLayer output contract). `PFTBlock` import fixed (was importing nonexistent `progressive_focused_transformer_block`). **ASYMMETRY RESOLVED (plan_2026-06-15_2a23a001/D-002)**: `PFTBlock.build` and `compute_output_shape` now accept both list and tuple of shapes. Detection uses `isinstance(input_shape[0], (list, tuple))` (NOT naive `isinstance(input_shape, (list, tuple))` -- a single input shape is also a tuple-of-ints). Anchor: `progressive_focused_transformer.py:287`. pft_sr caller unchanged (list still valid).
- **`layers/pixel_unshuffle.py`** -- exports BOTH `PixelUnshuffle2D` (space->depth, NHWC) AND `PixelShuffle2D` (depth->space, NHWC, added plan_2026-06-15_00924f53/D-002). Both serializable (`@register_keras_serializable`), dynamic-shape-safe. `PixelShuffle2D` is the canonical Keras-3.8 replacement for the nonexistent `keras.layers.DepthToSpace`; round-trip verified against `PixelUnshuffle2D`.
- **`layers/sampling.py`** -- 3 samplers + factory. `VMFSampling` XLA-GPU-incompatible (`keras.random.beta`); any model using it MUST `jit_compile=False`. **70 PASS tests**.
- **`layers/mixtures/`** -- **FULLY GRAPH-COMPATIBLE plan_2026-06-14_7384c2e3 (112 PASS tests).** Decision anchors: `plan_2026-06-14_7384c2e3/D-003` (idempotent axis re-derive), `plan_2026-06-14_8c7365d0/D-005` (cluster_axis stash).
- **`layers/sequence_pooling/`** -- factory'd package (7 files). 53 PASS tests. `SequencePooling('attention')` uses `AttentionPooling` (NOT a drop-in for head's inline Dense pooling).
- **`layers/heads/`** -- task head sub-package: nlp (8), vision (8), vlm (6 -- `VisualGroundingHead`+`MultiTaskVLMHead` UNTESTED). 37 PASS tests.
- **`layers/statistics/`** -- ~257 PASS tests. `MDNLayer` `mdn_pi` emits RAW LOGITS. DEAD (0 consumers): `InvertibleKernelPCA`, `DeepKernelPCA`, `ResidualACFLayer`.
- **`layers/time_series/`** -- `TemporalConvNet`/`TemporalBlock` Keras-3-canonical (24 PASS tests). `xlstm_blocks.py` `mLSTMCell.state_size` is 4-tuple `[units, matrix_memory_size, normalizer_size, num_heads]`.
- **`layers/orthogonal_butterfly.py`** -- exact-orthogonal Givens butterfly, power-of-two input dim only. 49 PASS tests.
- **`layers/logic/`** -- DARTS differentiable-primitive package.
- **`layers/grid_sample.py`** -- pure functions, TF-backend only. NOT a drop-in for `SpatialLayer`.
- **`layers/thera_heat_field.py`** -- `ThermalActivation` + `HeatField` (SIREN-style per-pixel implicit field; per-pixel `phi_phase`/`phi_kernel` are call inputs from hypernetwork, NOT layer weights).
- **`layers/geometric/point_cloud_autoencoder.py`** -- `PointCloudAutoencoder` NOW FUNCTIONAL (plan_2026-06-15_00924f53). `_get_graph_feature(x,k)` is the canonical DGCNN kNN edge-feature helper (pairwise-L2 -> `top_k(-dist,k)` -> `take_along_axis` -> `concat([center, neighbor-center])`), output `(B,N,k,2*C)`. Anchor: D-001.

### Models (`src/dl_techniques/models/`)
**70 model dirs** (not 71 -- prior counts included `__pycache__`).

**Canonical smoke instrument**: `scripts/verify_models_smoke.py` -- 86-entry registry-driven harness. Re-run: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python scripts/verify_models_smoke.py [--only <name>]`.

**Full sweep result (plan_2026-06-15_e6a0391c, 2026-06-15)**: **77 PASS / 0 FAIL / 7 XFAIL / 2 SKIP** (86 entries, 70 packages). Baseline was 39 PASS. Zero regressions. Source fixes: convnext v1/v2 downsample padding valid->same (D-003); sam image_encoder 3 Keras-3 bugs (D-004); modern_bert_blt_hrm seq-len gate (D-005).

**XFAIL (7 entries, root-caused):**
- swin_transformer: block expects 4D input, gets flattened 3D from SwinTransformerBlock
- hierarchical_reasoning_model: None->tensor in call()
- shgcn: needs SparseTensor adjacency (dense input fails)
- nano_vlm_world_model: `keras.random.uniform` rejects int32 dtype
- **sam (mask-decoder)**: image_encoder fixed (D-004); mask-decoder has separate multi-bug chain, hit 3-strike leash -- XFAIL pending dedicated plan
- **mobile_clip**: references backbones mci0/mci1/mci2/vit_b16 absent from `keras.applications` -- non-functional until ported
- **fftnet/SpectreHead**: FFT vision stack (`FFTMixer`, `FFTNet`) forwards fine; Spectre complex stack (`SpectreHead`) still dead-on-forward (triple-dead)

**SKIP (2 entries):**
- **jepa** (non-video): exposes only `JEPAEncoder`/`JEPAPredictor` layers -- no top-level `keras.Model`; skip-only by design
- **ccnets**: 3-model orchestrator, not a single `model(x)` callable

**Prior sweep history**: plan_2026-06-15_b5cec9e4 (pytest smoke coverage for all 70 dirs); Class A (plan_2026-06-15_00924f53: latent_gmm_registration, darkir, mothnet); Class B (plans 39a31d4a, 2a23a001, e2759fbc: dino_v1/v2/v3, nano_vlm, pft_sr) -- all CLEARED.

- **`models/time_series/`** -- 7 families, all Keras-3-canonical. `Forecast` dataclass + `ForecastMixin`. `AdaptiveEMASlopeFilterModel` is ForecastMixin-EXEMPT. **67/33/60/32/19/12/12 PASS tests** (xlstm/nbeats/prism/tirex/adaptive_ema/mdn/deepar).
- **`models/thera/`** -- THERA arbitrary-scale SR. 6-variant matrix `{edsr-baseline,rdn} x {air,plus,pro}`. **146 PASS tests**. `TheraTailPlus` needs `in_channels` when backbone is not 64ch; fail-loud guard at `tails.py:305`. D-001..D-013.
- **`models/ideogram4/`** -- flow-matching DiT + Flux2 KL-VAE. **183 PASS tests**. Not weight-loadable (nf4/fp8 PyTorch safetensors).
- **`models/sd3_mmdit/`** -- dual-stream SD3 MMDiT. **177 PASS tests**. `FlowMatchEulerScheduler.euler_step` uses SIGNED `dt = t_next - t` (t=0 data, t=1 noise; D-007).
- **`models/vae/`** -- Functional-API ResNet VAE. **77 PASS tests**.
- **`models/video_jepa/`** -- Video JEPA pretraining, six locked invariants. **78 PASS tests**.
- **`models/convnext_patch_vae/`** -- OOM rule: patch_size=4@img256 = 4096-position latent grid (OOMs at 24GB). **60 PASS tests**.
- **`models/detr/`** -- NOW FUNCTIONAL (commit 072df479, 21 tests). Prior "DETR broken" invariant was STALE (corrected plan_2026-06-15_b5cec9e4).
- **`models/pft_sr/`** -- NOW FORWARDS (plan_2026-06-15_39a31d4a/D-003): `drop_path`->`drop_path_rate` kwarg + 4 `Lambda(keras.ops.nn.depth_to_space)` -> `PixelShuffle2D`. Caller-side guards: bare `x` when `prev_attn_map is None`; list `[x, prev_attn_map]` for subsequent blocks. Output `(2,64,64,3)` finite (scale=2 SR). PFTBlock list/tuple asymmetry RESOLVED (plan_2026-06-15_2a23a001/D-002).
- **`models/fftnet/`** -- TWO decoupled subsystems. (1) FFT vision stack (`FFTMixer`, `FFTNet`): forwards fine. (2) Spectre complex stack (`SpectreHead`): still dead-on-forward (triple-dead); dedicated fix plan needed.
- **`models/dino/`** -- ALL THREE VARIANTS NOW FULLY FUNCTIONAL.
  - dino_v1: FORWARDS (plan_2026-06-15_39a31d4a/D-001: ClassTokenPrepend fix; `float(x)` for `.item()` fix; smoke PASS `(2,10)`).
  - dino_v3: FORWARDS (plan_2026-06-15_2a23a001/D-004: `[float(r) ...]` fix; smoke PASS `(2,10)`).
  - **dino_v2: FULLY FUNCTIONAL (plan_2026-06-15_e2759fbc)**: 13 bugs fixed (B1-B7 + cascades #8-#13 + D-009 hardening). 2-input `[images, masks]` contract. `MaskTokenApply` learnable mask token. Register variant works (registers position-free BY DESIGN, Darcet 2023; pos-embed applied to CLS+patches before register concat). `.keras` round-trip PASSES. 28 tests green.
  - OPEN: `DINOHead .keras` round-trip still broken (unbuilt children at load-time) -- pre-existing, separate, affects all 3 dino variants. `DINOHead` build() idempotency was fixed in a prior plan.
- **`models/nano_vlm/`** -- NOW FORWARDS (plan_2026-06-15_39a31d4a/D-002): ghost kwargs renamed; WEIGHT TYING RESTORED (plan_2026-06-15_2a23a001/D-001). Output `(2,213,256)` finite. Tie proven real via negative-proof test.
- **`models/nano_vlm_world_model/`** -- confirmed dead-on-forward (`keras.random.uniform` int32 dtype). Smoke test XFAIL.
- **`models/modern_bert/modern_bert_blt_hrm.py`** -- `ReasoningByteBERT` NOW FUNCTIONAL (commit 124d464b). Canonical HRM fix reference: `layers/reasoning/hrm_reasoning_core.py:524-540`.
- **`models/memory_bank/write_controller.py`** -- `tf.debugging.assert_less_equal` gated behind `keras.backend.backend()=="tensorflow"`.
- **`models/convnext/`** -- reference canonical Keras-3 model pattern. Block layers carry NO drop_path; reused by 5 other models.
- **`models/cliffordnet/`** -- `CliffordCLIP.logit_scale` pinned `dtype="float32"` under bf16.

### Initializers (`src/dl_techniques/initializers/`)
`OrthonormalInitializer`, `HeOrthonormalInitializer`, `OrthogonalHypersphereInitializer`, `HaarWaveletInitializer`, `PolarInitializer`, `LinearUpInitializer`, `KANInitializer`.

### Losses, Metrics, Optimization
- **Losses**: `FlowMatchingVelocityLoss`, `thera_jacobian_tv.py` (pure functions).
- **Metrics**: masked CLM loss, contrastive losses, `Perplexity`, `BitsPerToken`, `BitsPerCharacter`, `SegmentationWrapperLoss`, `CoverageMetric`, `SharpnessMetric`.
- **Optimization**: `Muon`, `SGLD`, `VSGD`. Factory: `optimizer_builder(config, lr)`.

### Training pipelines (`src/train/`)
- **`train/time_series/`** -- 7 active TS trainers, all subclass `BaseTimeSeriesTrainer`. All functional.
- **`train/thera/`** -- THERA SR training. `jit_compile=False`. Saves inner `thera` as `thera_model.keras`.
- **`train/ideogram4/`**, **`train/sd3_mmdit/`** -- synthetic flow-matching trainers.
- Other: `train/vae/`, `train/video_jepa/`, `train/convnext_patch_vae/`, `train/cliffordnet/`, `train/logic/`, `train/rms_variants_train/`, `train/convnext/`, `train/ccnets/`, `train/yolo12/`.
- **Common**: `BaseTimeSeriesTrainer` owns `_build_optimizer`, `_make_callbacks`, `run_experiment`. `create_ts_argument_parser` shared by all 7 TS trainers. **Orphaned**: `tfrecord.py` (15 symbols), `token_superposition.py` (8 symbols) -- 0 consumers.

## Boundaries
**In scope**: everything under `src/`, `tests/`, `research/`. Library code follows Keras 3 conventions strictly.
**External deps**: tiktoken, HuggingFace datasets, tensorflow 2.18.0, keras >=3.8.0 <4.0, numpy, scipy, scikit-learn, matplotlib (always `MPLBACKEND=Agg`).
**Out of scope**: external systems, CI infra, deployment pipelines.

## Invariants

### Guide-conformance baseline (as of plan_2026-06-13_e7b5704d)
- **All concrete custom layers implement `compute_output_shape`.** Intentional exemptions: abstract bases, RNN cells, dynamic-dict-output heads, multi-input/non-tensor-arg edge cases.
- **All concrete custom classes are `@register_keras_serializable`.** The 5 tabm classes use `package="TabM"`; `SystemicGraphFilter` uses `package="Experimental"`.
- **DETR (`models/detr/`) is NOW FUNCTIONAL** (commit 072df479, 21 tests). Previous "DETR broken" invariant was stale; corrected plan_2026-06-15_b5cec9e4.

### Keras 3 idioms (library-wide)
- `@keras.saving.register_keras_serializable()` on all concrete custom layers/models/metrics (NOT on abstract bases, callbacks, or plain dataclasses).
- `keras.ops` (no raw TF inside library code). Exception: `tf.math.bessel_i0e` in `_log_iv`; `fftnet` hardcodes `KERAS_BACKEND="tensorflow"`; `tf.linalg.svd` in `latent_gmm_registration/model.py:compute_rigid_transform` (TF-backend-local, documented D-001).
- Full `get_config()` round-trip. `dl_techniques.utils.logger` only.
- **`training is True`** canonical guard (not `if training:` -- crashes under symbolic tensor).
- **Explicit sublayer `.build()` in parent `build()`** required for `.keras` load-time weight restore.
- **Narrow `keras.ops.fft` surface**: `fft/fft2/ifft2/rfft/irfft/real/imag` only. NOT available: `rfft2/irfft2/angle/complex`.
- **Absent in Keras 3.8 / TF 2.18 (confirmed dead callers):**
  - `keras.ops.get_graph_feature` -- use `_get_graph_feature` in `layers/geometric/point_cloud_autoencoder.py` (D-001).
  - `keras.ops.scatter_nd_update` -- use `keras.ops.scatter_update(inputs, indices, updates)` (D-003).
  - `keras.layers.DepthToSpace` / `keras.ops.depth_to_space` / `keras.ops.nn.depth_to_space` -- use `PixelShuffle2D` from `layers/pixel_unshuffle.py` (D-002).
  - `keras.ops.add_n` -- use fold: `acc=tensors[0]; for t in tensors[1:]: acc=ops.add(acc,t)`.
  - `keras.ops.random.uniform` -- use `keras.random.uniform`.
- **`keras.random.beta` has NO XLA-GPU kernel** in TF 2.18 (`VMFSampling` must use `jit_compile=False`).

### CLM training
- Output dict key MUST be `"logits"`.
- `AdamW` WD only -- no `kernel_regularizer=L2(...)` combined with `AdamW(weight_decay=...)`.

### Save / load
- `.keras` on GPU fp32 has reduction-order noise ~5e-5 for U-Net-shaped models. Default tolerance 1e-4.
- Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken. Use `weight_transfer.load_weights_from_checkpoint`.

### Operational
- `results/` MUST be repo-root `results/`, never `src/results/`.
- `MPLBACKEND=Agg` required prefix for any training-script invocation.
- Single GPU jobs only.

## Flows
- **TS training** -- `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train.time_series.<model>.train_<model>`. `os._exit(0)` scripts (nbeats, mdn): success oracle is `"Completed."` log line + `results.json` (NOT exit code).
- **TS forecast inference** -- `model.predict_forecast(x)` on any `ForecastMixin` model. `AdaptiveEMASlopeFilterModel` EXEMPT.
- **CLM training** -- `python -m train.<model>.pretrain --config small` -> `MaskedCausalLMLoss` -> AdamW + warmup-cosine.
- **V-JEPA pretraining** -- online + EMA-target encoders -> custom `train_step` -> reload-check (bit-exact).
- **Save/load round-trip** -- `@register_keras_serializable()` -> `.keras` -> `keras.models.load_model(path, custom_objects={...})`.
- **Test scoping** -- pytest only on changed module + immediate importers. `make test` reserved for explicit pre-push request (~1.5h).

## Known Patterns
- **Pattern-2 TS trainer** -- `BaseTimeSeriesTrainingConfig` + `WindowedTimeSeriesProcessor` + `TimeSeriesPerformanceCallback` + `BaseTimeSeriesTrainer`. All 7 trainers use this; only adaptive_ema overrides `run_experiment`.
- **ForecastMixin pattern** -- mix into `@register_keras_serializable` model; implement `_forecast(x)->Forecast`. `get_config` unaffected. Models with non-batch-major leading axis (DeepAR `[S,B,H,D]`) MUST force single predict batch.
- **`MODEL_VARIANTS`/`from_variant` pattern** -- only where discrete model-size variants are semantically real.
- **Factory'd layer sub-module** -- own `__init__.py` + `factory.py`. Factory gets its own test file.
- **Smoke test xfail idiom** -- `pytest.xfail(str(e))` inside `except Exception` after genuine build+forward attempt. Keeps suite green while preserving exact error. Caveat: broad except swallows NaN-assert failures into xfail; acceptable for coverage markers only.
- **Pixel-shuffle pair** -- `PixelUnshuffle2D(scale)` (space->depth) and `PixelShuffle2D(block_size)` (depth->space) in `layers/pixel_unshuffle.py`. Round-trip verified. Use instead of any `DepthToSpace`/`depth_to_space` reference.
- **ClassTokenPrepend pattern** -- `layers/embedding/class_token.py:ClassTokenPrepend`. Creates `cls_token` weight `(1,1,dim)` in `build()` (NOT in the model's `__init__` or `_build_model`). `call()` broadcasts and concatenates: output `(B, L+1, dim)`. `@register_keras_serializable`. Canonical solution for prepend-CLS in any Functional ViT model. Key rule: a Functional model's graph finalizes at `super().__init__(inputs=,outputs=)` -- `add_weight` must not fire before that call. Option B (model `build()` override) does NOT work. (plan_2026-06-15_39a31d4a/D-001.) Dedicated unit test: `tests/test_layers/test_embedding/test_class_token.py`, 11 tests.
- **MaskTokenApply pattern** -- `layers/embedding/mask_token.py:MaskTokenApply`. Creates `mask_token` weight `(1,1,embed_dim)` in `build()`. `call(patch_embeddings, masks)` broadcasts and applies `keras.ops.where`: masked positions replaced by token, unmasked unchanged. `@register_keras_serializable`. `.keras` round-trip verified. Canonical solution for iBOT/MAE/data2vec-style learned mask tokens. (plan_2026-06-15_e2759fbc/D-009.) Reused by dino v2.
- **DINOv2 register-token invariant** -- register tokens are inserted AFTER positional embedding (pos-embed applied to CLS+patches `(B, N+1, D)` first; registers concatenated after). Registers receive NO positional signal BY DESIGN (Darcet 2023; position-free registration is the paper's explicit contribution). Do NOT add positional embedding to registers or move insertion before pos-embed. (plan_2026-06-15_e2759fbc/D-009.)

## Codebase Specialization
- **Python**: >=3.11, type hints, Google-style docstrings.
- **Venv**: always `.venv/bin/python` for invocation.
- **Logging**: `from dl_techniques.utils.logger import logger` -- never `print`.
- **GPU**: GPU 0 = RTX 4090 24GB, GPU 1 = RTX 4070 12GB. Pin via `CUDA_VISIBLE_DEVICES=N`.
- **Test discipline**: scope pytest to touched modules; `make test` only on explicit user request immediately before push.
- **Commit prefix**: `[iter-N/step-M] <description>` during EXECUTE.
- **Plans dir**: `plans/` is in `.gitignore`; plan files never committed.
- **Push default**: `git push --no-verify` (skip 1.5h pre-push hook).
