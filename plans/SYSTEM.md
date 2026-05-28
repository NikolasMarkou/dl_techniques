# System Atlas
*Last refreshed: plan_2026-05-27_c3184aea | 2026-05-27 | 64 plans closed*
*Domain-neutral system map. Rewritten by ip-archivist at CLOSE — max 300 lines. Read before PLAN/EXPLORE.*

## Identity
`dl_techniques` is a Keras 3 / TensorFlow 2.18 deep-learning research library — 150+ model architectures, 290+ custom layers, plus production training pipelines, applications, and experiments. **Domain: codebase.**

## Components

### Layers (`src/dl_techniques/layers/`)
290+ custom Keras layers — geometric/Clifford, attention variants (incl. `WaveFieldAttention`), normalization, FFN, routing, logic/arithmetic.

- **`layers/logic/`** — DARTS-style differentiable-primitive package (4 source files): `LearnableArithmeticOperator`, `LearnableLogicOperator`, `CircuitDepthLayer`, `LearnableNeuralCircuit`. All accept `selection_mode: Literal['global','per_channel'] = 'global'`. **Defaults**: `softplus_temperature=True`, `operation_initializer='zeros'`, `allow_unary_degenerate=False`.

- **`layers/ffn/`** — factory'd FFN sub-package: 14 registered types via `create_ffn_layer(<type>, ...)`.

- **`layers/memory/`** — single canonical home for memory-augmented layers (MANN + NTM families + SOMs). **`factory.create_mann(...)`** routes to a `NeuralTuringMachine`. `MannLayer` / `SOM2dLayer` retained as BC aliases. `AddressingMode` pruned to `{CONTENT, HYBRID}`.

- **`layers/transformers/adaln_zero.AdaLNZeroConditionalBlock`** — factory-configurable via 8 optional ctor kwargs. AdaLN-Zero affine-False norm invariant: D-005 anchor `adaln_zero.py:181`.

### Models (`src/dl_techniques/models/`)
150+ architectures, one subpackage per family.

- **`models/video_jepa/`** — Video JEPA pretraining model with EMA target encoder. `VideoJEPA` composes online + target encoders, `Predictor`, `TubeMaskGenerator`, multi-horizon prediction heads. **Six locked invariants**: (1) tube-mask gated on `training` (D-001 `model.py:401`); (2) gate uses `(training is True)` Python identity (D-003 `model.py:407`); (3) `CausalSelfAttnMLPBlock` guards dropout (D-005 `predictor.py:120`); (4) explicit `self.loss_tracker` + `metrics` + `update_state` (D-005 `model.py:176,552`); (5) multi-horizon advisory warning; (6) `ema_divergence` weight-space L2 metric (D-001 `model.py:207,353`, plan_aebd4cbb). Per-component metrics (`loss`, `next_frame_loss`, `mask_loss`, `sigreg_loss`, `ema_m`, `ema_divergence`) flow through history → CSV → plots. Tests: 78 PASS (~231s scoped). Trainer: `src/train/video_jepa/train_video_jepa.py`. Reload-check at `train_video_jepa.py:515`.

- **`models/convnext_patch_vae/`** — Resolution-agnostic ConvNeXt VAE on per-patch 4D latents. Public surface `{ConvNeXtPatchVAE, ConvNeXtPatchVAEConfig, ConvNeXtPatchEncoder, ConvNeXtPatchDecoder, create_convnext_patch_vae, HierarchicalConvNeXtPatchVAE, HierarchicalConvNeXtPatchVAEConfig, create_hierarchical_convnext_patch_vae}`. **Hierarchical variant** (`model_hierarchical.py`, plan_2026-05-27_dee954c6): sibling two-level model — L1 coarse global encoder (large patch, high latent_dim) + L2 fine encoder (small patch); L2 decoder is conditioned on tile-broadcast `z_l1` (nearest-neighbor UpSampling2D → Concat → 1×1 Conv → ConvNeXtV2Block stack). L1 has no pixel-recon head — conditioning-only. Two staggered `BetaAnnealingCallback` instances driven by the same class via `attr_name` kwarg. Trainer flag `--hierarchical` writes to `results/hierarchical_convnext_patch_vae_*`. Both models expose unified `sample_from(x, temperature)` API — reparameterize from encoder posterior; `temperature=0` deterministic, `temperature=1` matches prior scale. **Learnable conditional prior `p(z_l2|z_l1)`** (plan_c3184aea, `_L2ConditionalPrior` layer with zero-init heads; default `learnable_l2_prior=True`): replaces implicit N(0,I) on L2 latent; KL becomes closed-form Gaussian-Gaussian with `[-10,+10]` log_var clipping on both q and p; `sample(num_samples,...)` now coherent (z_l1 ~ N(0,I) → z_l2 ~ p(z_l2|z_l1) → decode). Zero-init heads make at-step-0 KL bit-exact equal to legacy implicit-prior KL — enables clean checkpoint transfer via `weight_transfer.load_weights_from_checkpoint`. 33 hierarchical tests + 21 single-scale tests scoped at ~116s. **Architecture**: encoder = `Conv2D(stride=patch_size)` stem → LN → N x `ConvNextV2Block` → `Conv2D(1,1,2*latent_dim)` bottleneck → split `(mu, log_var)`. Decoder = `Conv2D(1,1,embed_dim)` proj_in → N x `ConvNextV2Block` → LN → `Conv2DTranspose(kernel=stride=patch_size)` head. Latent shape `(B, Hp, Wp, latent_dim)`. **Loss**: `recon + beta_kl * KL_per_patch + lambda_sigreg * SIGReg`. KL averaged over `(B, Hp, Wp)` (resolution-invariant). SIGReg on `ops.reshape(z, (B, Hp*Wp, latent_dim))` post-reparameterization. **Public surface**: `PRESETS = {tiny: embed=64/depth=2/latent=8, base: 128/4/16, large: 192/6/32}`, `from_variant`, `_download_weights -> NotImplementedError` (D-001 `model.py:470`, plan_8faec5b6), `create_convnext_patch_vae` factory (D-002 `model.py:542`, plan_8faec5b6). **Invariants**: (i) `img_size % patch_size == 0`; (ii) no GlobalAveragePooling2D, no learned absolute PE; (iii) explicit `self.loss_tracker` in `__init__` + `update_state` in `train_step` (D-001 `model.py:124`, plan_fb57d478); (iv) four trackers `{loss, recon_loss, kl_loss, sigreg_loss}`; (v) `jit_compile=False` — XLA tracing fails on `ops.reshape` in `_compute_sigreg`. **Ghosts**: no EMA target, no PE, no temporal axis. **Trainer** (`src/train/convnext_patch_vae/train_convnext_patch_vae.py`, plan_74f0eac9): `@dataclass TrainingConfig`, CIFAR-10 pipeline (MSE default with mean/std norm; BCE with /255 only), `ReconVisualizationCallback`, `model.compile(loss=None, jit_compile=False)`, reload check via deterministic `encode()` mu (NOT stochastic reconstruction), `--smoke` flag with explicit `img_size=32` override (base parser default is 224). Success guard on `val_loss <= success_threshold`. 19 PASS tests (~30s scoped). plans: plan_fb57d478, plan_8faec5b6, plan_74f0eac9.

- **`models/memory_bank/`** — backbone-agnostic memory primitives: `LongTermMemoryBank`, `WorkingMemoryBank`, `MemoryReadController`, `MemoryWriteController`, `PhaseScheduler`, `WaveFieldMemoryLLM`, `MemoryStats`.

- **`models/burst_dp/`** — multi-view reference-conditioned vision model. `fusion_type ∈ {"custom","adaln"}`. Trainer `train_burst_dp.py`: `--dataset {coco,div2k,vggface2}`; 20 `--aux-*` CLI flags. 13 PASS tests.

- **`models/vit/`** — `ViT`, `create_vit`, `create_inference_model_from_training_model`. `ViT.MODEL_VARIANTS = {vit_pico/tiny/small/base/large/huge}`. **Trainer `train_vit.py` is canonical Pattern-4 image reference**: `@dataclass TrainingConfig`, CIFAR + ImageNet builders, AdamW/SGD WD branch, `_assert_train_val_distribution_match` before `model.fit` (D-007), guarded SUCCESS log. ViT-pico CIFAR-10 → `val_acc=0.7836`.

- **`models/cliffordnet/`** — public surface `{CliffordNet, create_cliffordnet, CliffordCLIP, CliffordNetLMRouting, CliffordNetLMUNet, CliffordNetEmbedding}`. `CliffordNetLMUNet` flat-keyed dict output; width-rule "Power-of-2 anchored" D-002 `lmunet.py:415`.

- **`models/{bert, gpt2, tree_transformer, depth_anything, accunet, tiny_recursive_model, prism, lewm}/`** — each: `{Model, create_<model>}` public surface, `from_variant(pretrained=True)` raises `NotImplementedError` from `_download_weights`, narrow `except (IOError, OSError, ValueError)`.

### Losses & Metrics (`src/dl_techniques/{losses,metrics}/`)
Masked CLM loss, contrastive losses, `Perplexity`, `BitsPerToken`, `BitsPerCharacter`. **`SegmentationWrapperLoss`** is the canonical save/load-friendly segmentation loss.

### Training utilities (`src/dl_techniques/training/`)
`token_superposition.py` (TST): `TSTConfig`, `TSTState`, `TSTEmbedding`, `TSTCausalLMLoss`, `TSTPhaseCallback`. Invariant (D-007 `token_superposition.py:691`): TWO named dataset transforms, NOT a single `tf.cond`. 54 PASS tests.

### Callbacks (`src/dl_techniques/callbacks/`)
Keras callbacks; analyzer integration; `TemperatureAnnealingCallback`.

### Utils / Datasets
- `utils/`: `logger`, `weight_transfer`, GPU setup.
- `datasets/vision/coco_burst_dp.py` + `image_folder_burst_dp.py`. 9 PASS tests.

### Training pipelines (`src/train/`)
- **`train/video_jepa/`** — V-JEPA pretrainer; smoke + BDD100K dataset wiring; reload-check at `train_video_jepa.py:515`.
- **`train/convnext_patch_vae/`** — ConvNeXtPatchVAE trainer (plan_74f0eac9); `compile(loss=None, jit_compile=False)`; reload check via `encode()` mu; `--smoke` flag. `callbacks.py` (plan_2026-05-26_d8c33dca): `LatentSpaceCallback` (PCA scatter of mu flattened `(B,Hp*Wp*D)`, colored by per-sample KL) + `LatentInterpolationCallback` (linear mu-space interpolation grid). **Augmentation** (plan_2026-05-26_5abf5af3): `_make_filesystem_decode_fn(img_size, img_channels, augment, augment_color)` factory owns decode+augment for all filesystem datasets — geometric (resize+crop+flip) + photometric (brightness 0.2, contrast 0.8–1.2, saturation 0.8–1.2 RGB only, clip_by_value). CIFAR: brightness 0.1 + contrast 0.9–1.1 only (MSE path is standardized). `TrainingConfig.augment_color: bool = True`; `--no-color-augment` CLI flag.
- **`train/logic/`** — LearnableNeuralCircuit benchmark suite + `multiseed_sweep.py` subprocess driver + `multiseed_stats.py`. 30-test stats harness.
- **`train/rms_variants_train/`** — 8-norm comparison harness; `NORM_VARIANTS` append-only invariant (D-001 `config.py:27`). 637+ PASS tests.

## Boundaries
**In scope**: everything under `src/`, `tests/`, `research/`. Library code follows Keras 3 conventions strictly.
**External deps**: tiktoken (gpt2/cl100k_base), HuggingFace datasets (Wikipedia), tensorflow 2.18.0, keras >=3.8.0 <4.0, numpy, scipy, scikit-learn, matplotlib (always `MPLBACKEND=Agg`).
**Out of scope**: external systems, CI infra, deployment pipelines.

## Invariants

### Keras 3 idioms (library-wide)
- `@keras.saving.register_keras_serializable()` on all custom layers/models.
- `keras.ops` (no raw TF inside library code).
- Full `get_config()` round-trip.
- `dl_techniques.utils.logger` only — no `print`.
- Random ops live under `keras.random.*` — `keras.ops.random.uniform` does NOT exist in 3.8.
- Frozen tensor state in layers MUST be `add_weight(trainable=False, ...)` or numpy on `self`.

### CLM training (library-wide)
- **Output dict key MUST be `"logits"`** — `MaskedCausalLMLoss` and `model.compile(loss={"logits": ...})` key on it.
- **AdamW WD only** — no `kernel_regularizer=L2(...)` combined with `AdamW(weight_decay=...)`.
- **`prepare_dict_keyed_compile(model, output_key="logits")`** is a permanent contract for dict-output trainers.
- **CLM compile-time metric floor**: `metrics={"logits": build_clm_metrics(config.encoding_name)}`.
- **Pattern-3 CLI uniformity** — `--steps-per-epoch`, `--seed`, `--min-article-length`, `--shuffle-shards`, `--resume`.
- **`pad_token_id`** must match tokenizer (tiktoken cl100k_base = 100266); model default 0 is a silent semantic bug.

### Causality (Clifford / time-series)
- `(H=1, W=seq_len)` Clifford blocks must use `avg`/`max` pool only.
- DSv2 nearest-upsample requires `_causal_upsample` right-shift by `s-1`.

### Custom `train_step` / multi-optimizer
- **Two-optimizer differential-LR**: register ONE optimizer with `super().compile(...)`, apply the second manually.
- **`current_phase` / `_global_step`: `add_weight(trainable=False, dtype="float32")`** — int32 caused device-placement failures.
- **Custom `train_step` bypassing `compile(loss=...)` (uses `add_loss`)** MUST explicitly create `self.loss_tracker` in `__init__`, expose via `metrics`, and `update_state(loss)` in `train_step`.
- **`training` flag gates** — use `(training is True)` Python identity check under `@tf.function`.
- **VAE compile pattern**: `model.compile(optimizer=AdamW, loss=None, jit_compile=False)`. ConvNeXtPatchVAE: `jit_compile=False` because XLA tracing fails on `ops.reshape` in `_compute_sigreg`.

### Save / load
- `.keras` save/load on GPU under fp32 has reduction-order noise ~5e-5 for U-Net-shaped models. Default tolerance 1e-4.
- Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken. Use `weight_transfer.load_weights_from_checkpoint`.
- Custom-subclass Model wrapping another `keras.Model`: use `save_own_variables` / `load_own_variables`.

### Operational
- **`results/` MUST be repo-root `results/`, never `src/results/`**.
- **`MPLBACKEND=Agg`** required prefix for any training-script invocation.
- **Single GPU jobs only** — never parallel training.
- **Pin GPU via shell env** — TF initialises at `import tensorflow as tf`. Use `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train.<...>`.
- **All training callbacks that write to `save_dir` MUST `os.makedirs(..., exist_ok=True)` at the top of every save method**.

## Flows
- **CLM training** — `python -m train.<model>.pretrain --config small` → `TrainingConfig` → tiktoken/Wikipedia → packed token shards → `MaskedCausalLMLoss` → AdamW + warmup-cosine → `StepCheckpointCallback` + `GenerationProbeCallback`.
- **V-JEPA pretraining** — `python -m train.video_jepa.train_video_jepa --smoke --dataset bdd100k --videos-root <...>` → online + EMA-target encoders → custom `train_step` → reload-check (bit-exact `max|delta|=0.00e+00`).
- **Image training (canonical: `train/vit/train_vit.py`)** — `@dataclass TrainingConfig` → per-dataset builder → augment-then-normalize → `_assert_train_val_distribution_match` → AdamW/SGD → guarded SUCCESS log.
- **ConvNeXtPatchVAE training** — `python -m train.convnext_patch_vae.train_convnext_patch_vae [--smoke]` → `TrainingConfig` → CIFAR-10 (MSE: mean/std norm; BCE: /255 only) → `compile(loss=None, jit_compile=False)` → `ReconVisualizationCallback` → reload check via `encode()` mu.
- **Save / load round-trip** — `@register_keras_serializable()` decorator → `.keras` archive → `keras.models.load_model(path, custom_objects={...})`.
- **Multi-seed sweep** — subprocess-per-seed driver, glob+merge, pure-stats module (mean/std, bootstrap CI, paired sign-flip permutation B=10000 Phipson-Smyth).
- **Test scoping** — pytest only on changed module + immediate importers. `make test` reserved for explicit pre-push request (~1.5h).
- **Plan close** — orchestrator writes `summary.md`, runs decision-anchor audit, updates `plans/LESSONS.md` (≤200 lines) and `plans/SYSTEM.md` (≤300 lines), then `bootstrap.mjs close`.

## Known Patterns
- **Pattern-3 NLP CLM training script** — ~95% generic `TrainingConfig` + `StepCheckpointCallback` + `GenerationProbeCallback` + AdamW/warmup-cosine + tiktoken/Wikipedia + `MaskedCausalLMLoss`. New CLM = mirror file-by-file.
- **Pattern-4 image trainer (canonical: `train/vit/train_vit.py`)** — `@dataclass TrainingConfig` + per-dataset builders + AdamW/SGD WD branch + `_assert_train_val_distribution_match` + `MetricsVisualizationCallback` + guarded SUCCESS log. New image trainer = mirror file-by-file.
- **Pattern-4 VAE variant (canonical: `train/convnext_patch_vae/train_convnext_patch_vae.py`)** — hybrid Pattern-4 (dataset, argparse, dataclass config, callbacks) + video_jepa compile pattern (`loss=None, jit_compile=False`) + `ReconVisualizationCallback` + reload check via deterministic `encode()` mu + `--smoke` with explicit tiny `img_size` override. Dataset emits `(x, x)` tuples; success guard on `val_loss` not accuracy.
- **EMA target encoder pretraining (canonical: `models/video_jepa/`)** — sibling encoder with `trainable=False`, dummy-batch eager build, custom `train_step` with EMA update, cosine momentum via `add_weight("ema_step")`. Observability: `ema_divergence` tracker. Do NOT use for VAE objectives (reconstruction forbids identity).
- **Two-optimizer differential-LR via custom `train_step`** — register one optimizer with `super().compile(...)`, apply the second manually; variable routing via name-prefix split.
- **Subprocess-per-seed multi-seed sweep** — clean TF/Keras init eliminates cross-seed state contamination.
- **Sibling-stack model addition** — new layer that doesn't fit the shared factory: build self-contained inside the new model package; defer factory registration. Zero blast radius on 30+ unrelated models.
- **Mechanical patch replication across N sibling files** — per-file repetition beats shared-helper extraction for N≤4 if extraction would cross package boundaries.
- **Hard-extraction probe via `LARGE × one_hot(argmax)`** — clean faithfulness check for DARTS-style layers.

## Codebase Specialization
- **Python**: >=3.11, type hints, Google-style docstrings.
- **Venv**: always `.venv/bin/python` for invocation.
- **Logging**: `from dl_techniques.utils.logger import logger` — never `print`.
- **GPU**: GPU 0 = RTX 4090 24GB, GPU 1 = RTX 4070 12GB. Pin via `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train....`.
- **Test discipline**: scope pytest to touched modules; `make test` only on explicit user request immediately before push.
- **Commit prefix**: `[iter-N/step-M] <description>` during EXECUTE; no commit on EXPLORE/PLAN/REFLECT/PIVOT.
- **Plans dir**: `plans/` is in `.gitignore`; plan files never committed.
- **Push default**: `git push --no-verify` (skip 1.5h pre-push hook). User runs full suite separately.
