# System Atlas
*Last refreshed: plan_2026-06-03_5c8c6d19 | 2026-06-03 | 80 plans closed*
*Domain-neutral system map. Rewritten by ip-archivist at CLOSE -- max 300 lines. Read before PLAN/EXPLORE.*

## Identity
`dl_techniques` is a Keras 3 / TensorFlow 2.18 deep-learning research library -- 150+ model architectures, 290+ custom layers, plus production training pipelines, applications, and experiments. **Domain: codebase.**

## Components

### Layers (`src/dl_techniques/layers/`)
290+ custom Keras layers -- geometric/Clifford, attention variants (incl. `WaveFieldAttention`), normalization, FFN, routing, logic/arithmetic.

- **`layers/norms/polar_weight_norm.py`** -- polar weight reparameterization. Module-level `polar_encode` / `polar_decode` / `_next_power_of_two`. `PolarWeightNorm` layer: Dense-style; trainable params `radius (units,)` + `angles (units, d-1)`, `d = next_pow2(fan_in)`; kernel reconstructed each forward via decode + slice + renorm-after-slice; exact per-unit L2 norm = |radius|. Module docstring is RBF-structured (merged from former companion `.md`, which is deleted). `PolarInitializer` lives in `initializers/polar_initializer.py` (cross-referenced, not duplicated). 28 PASS tests.

- **`layers/orthogonal_butterfly.py`** -- `OrthogonalButterfly`: standalone layer, exact-orthogonal Givens butterfly (WtW=I), O(d log d) cost, identity-at-init, invertible (`inverse=True` ctor / `.inverse()` method / `log_det_jacobian==0`). Power-of-two input dim only (no padding). Bare `@register_keras_serializable()` (key tied to `__module__`; `package=` omitted intentionally per RBF template precedent). Module docstring is RBF-structured (merged from former companion `.md`). Not exported from `layers/__init__.py` (intentional). 49 PASS tests.

- **`layers/logic/`** -- DARTS-style differentiable-primitive package (4 source files): `LearnableArithmeticOperator`, `LearnableLogicOperator`, `CircuitDepthLayer`, `LearnableNeuralCircuit`. Defaults: `softplus_temperature=True`, `operation_initializer='zeros'`, `allow_unary_degenerate=False`.

- **`layers/ffn/`** -- factory'd FFN sub-package: 14 registered types via `create_ffn_layer(<type>, ...)`.

- **`layers/memory/`** -- canonical home for memory-augmented layers (MANN + NTM families + SOMs). `factory.create_mann(...)` routes to a `NeuralTuringMachine`. `MannLayer` / `SOM2dLayer` retained as BC aliases. `AddressingMode` pruned to `{CONTENT, HYBRID}`.

- **`layers/transformers/adaln_zero.AdaLNZeroConditionalBlock`** -- factory-configurable via 8 optional ctor kwargs. AdaLN-Zero affine-False norm invariant: D-005 anchor `adaln_zero.py:181`.

### Models (`src/dl_techniques/models/`)
150+ architectures, one subpackage per family.

- **`models/video_jepa/`** -- Video JEPA pretraining model with EMA target encoder. `VideoJEPA` composes online + target encoders, `Predictor`, `TubeMaskGenerator`, multi-horizon prediction heads. Six locked invariants: (1) tube-mask gated on `training` (D-001 `model.py:401`); (2) gate uses `(training is True)` Python identity (D-003 `model.py:407`); (3) `CausalSelfAttnMLPBlock` guards dropout (D-005 `predictor.py:120`); (4) explicit `self.loss_tracker` + `metrics` + `update_state` (D-005 `model.py:176,552`); (5) multi-horizon advisory warning; (6) `ema_divergence` weight-space L2 metric (D-001 `model.py:207,353`). Tests: 78 PASS.

- **`models/convnext/`** -- Public surface: `{ConvNeXtV1, ConvNeXtV2, create_convnext_v1, create_convnext_v2}`. Both models accept `drop_path_rate: float = 0.0` and `stochastic_mode: str = 'depth'` (kwarg plumbed through `get_config()`). `'depth'` (default, behavior-preserving) -> `StochasticDepth`; `'gradient'` (opt-in, forward-identity grad-only) -> `StochasticGradient`; other values raise `ValueError`. D-001 anchor: `convnext_v1.py:310`, `convnext_v2.py:327`. Block layers (`ConvNextV1Block`, `ConvNextV2Block`) carry NO drop_path and are reused by 5 other models (convnext_patch_vae, convnext_patch_vae_v2, convunext, bfconvunext + VAE encoder). Factories thread `**kwargs` to the model constructor. `StochasticDepth`/`StochasticGradient` not exported from `layers/__init__.py` -- import direct from module.

- **`models/convnext_patch_vae/`** -- Resolution-agnostic ConvNeXt VAE on per-patch 4D latents. Public surface: `{ConvNeXtPatchVAE, ConvNeXtPatchVAEConfig, ConvNeXtPatchEncoder, ConvNeXtPatchDecoder, create_convnext_patch_vae}`. Hierarchical variant fully removed. Architecture: encoder = `Conv2D(stride=patch_size)` stem -> LN -> N x `ConvNextV2Block` -> `Conv2D(1,1,2*latent_dim)` bottleneck -> split `(mu, log_var)`. Latent shape `(B, Hp, Wp, latent_dim)`. Loss: `recon + beta_kl * KL_per_patch + lambda_sigreg * SIGReg`. PRESETS = {tiny, base, large}. Invariants: `img_size % patch_size == 0`; no GlobalAveragePooling2D; `jit_compile=False`. Tests: 21 PASS.

- **`models/memory_bank/`** -- `LongTermMemoryBank`, `WorkingMemoryBank`, `MemoryReadController`, `MemoryWriteController`, `PhaseScheduler`, `WaveFieldMemoryLLM`, `MemoryStats`.

- **`models/burst_dp/`** -- multi-view reference-conditioned vision model. `fusion_type in {"custom","adaln"}`. Trainer `train_burst_dp.py`: `--dataset {coco,div2k,vggface2}`; 20 `--aux-*` CLI flags. 13 PASS tests.

- **`models/vit/`** -- `ViT`, `create_vit`, `create_inference_model_from_training_model`. `ViT.MODEL_VARIANTS = {vit_pico/tiny/small/base/large/huge}`. Trainer `train_vit.py` is canonical Pattern-4 image reference.

- **`models/cliffordnet/`** -- public surface `{CliffordNet, create_cliffordnet, CliffordCLIP, CliffordNetLMRouting, CliffordNetLMUNet, CliffordNetEmbedding}`. `CliffordNetLMUNet` flat-keyed dict output; width-rule "Power-of-2 anchored" D-002 `lmunet.py:415`.
  - **`CliffordCLIP`** (`clip.py`): `text_use_global_context: bool = False` ctor kwarg. `logit_scale` pinned `dtype="float32"` under bf16 global policy (`# DECISION plan_2026-05-31_76981d58/D-001` at `clip.py:1044`). `encode_image()` / `encode_text()` return L2-normalized embeddings by default. Wrapper model is `ContrastiveCliffordCLIP` (`self.clip_model`); head LayerScale at `inner.{vision,text}_head_scale.gamma` (absent for `head_kind='plain'` -- always getattr-guard).
  - **SRGP docstring**: `clifford_block.py:65-66` correctly reads `(c-s)%D`; impl `roll(shift=s)` @ line 223 is correct.

- **`models/ccnets/`** -- Causal Cooperative Nets framework + migrated architectures. Framework files unchanged: `base.py`, `orchestrators.py`, `trainer.py`, `losses.py`, `control.py`, `utils.py`. Migrated: `blocks.py` (FiLMLayer/ConvBlock/DenseBlock, depend on GoLU); `architectures/{mnist,cifar100,text}.py` (task Explainer/Reasoner/Producer networks, AR+non-AR text producers, factories `create_{mnist,cifar100,text}_ccnet`, HybridCCNetOrchestrator/TextCCNetOrchestrator/ARTextCCNetOrchestrator). Package `__init__` exports 25 symbols. Invariants: variational Explainer returns `(mu, log_var)`; bias-free `Dense(use_bias=False)` label projection (never Embedding); `package=` qualified serialization keys (`ccnets_cifar100`, `ccnets_text`); `dynamic_weighting` stays `False` (deprecated). Tests: `test_orchestrator.py` + `test_architectures.py` (28). Pre-existing flaky: `test_orchestrator.py::TestTrainStep::test_training_reduces_total_error` (no seed).

- **`models/{bert, gpt2, tree_transformer, depth_anything, accunet, tiny_recursive_model, prism, lewm}/`** -- each: `{Model, create_<model>}` public surface, `from_variant(pretrained=True)` raises `NotImplementedError`.

### Initializers (`src/dl_techniques/initializers/`)
Orthonormal, He-orthonormal, hypersphere, Haar-wavelet, `PolarInitializer` initializers.

### Losses & Metrics (`src/dl_techniques/{losses,metrics}/`)
Masked CLM loss, contrastive losses, `Perplexity`, `BitsPerToken`, `BitsPerCharacter`. **`SegmentationWrapperLoss`** is the canonical save/load-friendly segmentation loss.

### Training utilities (`src/dl_techniques/training/`)
`token_superposition.py` (TST): `TSTConfig`, `TSTState`, `TSTEmbedding`, `TSTCausalLMLoss`, `TSTPhaseCallback`. Invariant (D-007 `token_superposition.py:691`): TWO named dataset transforms, NOT a single `tf.cond`. 54 PASS tests.

### Analyzer (`src/dl_techniques/analyzer/`)
WeightWatcher/SETOL HTSR spectral analysis framework. **Authoritative reference: Charles Martin's WeightWatcher source.** Where WW and SETOL.md conflict on mechanism or terminology, WW wins (user-declared, D-001 bc986e52). Compliance doc: `src/dl_techniques/analyzer/SETOL.md`.

- **`spectral_metrics.py`**: core per-layer computation. Eigenvalues default to σ² (no 1/N); 1/N is an opt-in `normalize` flag. ERG path applies `rescale_eigenvalues` wscale (Σλ→N, SETOL §10.2 sanctioned correction; byte-identical to WW). Key outputs:
  - `alpha`: Clauset MLE power-law exponent (joint xmin/KS fit). Tail `n<20` uses bias-corrected `alpha_bc = 1+(n-1)/s` with penalized xmin `J = D_ks − 0.868/√n` (WW small-N branch, D-008 anchor at :265).
  - `alpha_weighted` (`MetricNames.ALPHA_WEIGHTED`): WW canonical = α·log₁₀(λ_max) on σ² eigenvalues. **Primary cross-architecture quality metric.**
  - `alpha_hat`: alias of `alpha_weighted` (same value; SETOL-paper notation).
  - `alpha_hat_normalized`: /N variant = α·log₁₀(λ_max/N) — non-WW SETOL extra, documented as such.
  - `mp_softrank` (`MetricNames.MP_SOFTRANK`): = λ_plus/λ_max after removing num_spikes outliers (WW R6; `compute_mp_softrank`).
  - `rand_sv_ratio` (`MetricNames.RAND_SV_RATIO`): = max(rand_evals)/max(evals) — randomization diagnostic (distinct from mp_softrank).
  - `erg_delta_lambda_min`: **signed** gap (SETOL §7.3). `abs()` MUST NOT be added. ERG tail boundary computed by reusing `compute_detX_constraint` (descending-product loop; D-004 anchor at :424).
  - MP edge: `σ²(1+1/√Q)²`, `σ²(1−1/√Q)²` where Q=N/M, N=larger dim (WW-exact; D-002 anchor at :543). NOT `(1+√Q)²`.
  - TW threshold: `bulk_max + c_TW·√[(1/√Q)·bulk_max^(2/3)·M^(-2/3)]`; `SPECTRAL_TW_SAFETY_FACTOR=1.0` (WW-exact default; D-003 anchor at :551).
  - `classify_learning_phase`: α<0 "failed"; 0≤α<2 "over-trained"; 2≤α≤6 "good"; α>6 "under-trained" (WW labels; D-009 anchor at ~:458). No "ideal" band. No "over-regularized".
- **`spectral_visualizer.py`**: (α, Δλ_min) funnel plot (WW-neutral phrasing; α=2 over-trained/good boundary); MP bulk envelope overlay on per-layer ESD panel.
- **`constants.py`**: `MetricNames` enum — `ALPHA_WEIGHTED`, `ALPHA_HAT`, `MP_SOFTRANK='mp_softrank'`, `RAND_SV_RATIO='rand_sv_ratio'`, `MATRIX_RANK`, `NORM`, `SPECTRAL_NORM`; `SPECTRAL_DEFAULT_SUMMARY_METRICS` includes `ALPHA_WEIGHTED`+`ALPHA_HAT`+`LOG_SPECTRAL_NORM`; `SPECTRAL_SMALL_N_CUTOFF=20`.
- **`spectral_utils.py`**: NORM layers are explicitly skipped with `logger.debug` ("degenerate ESD; spectral analysis skipped"); D-010 anchor at :162.
- Conv2D layers: reshaped to (H·W·C_in, C_out). BN/Dropout/bias excluded. Correlation-trap randomize protocol preserved.
- **α̂ normalization**: `MetricNames.ALPHA_WEIGHTED` exposes the WW un-normalized convention; `alpha_hat_normalized` (/N) is a documented non-WW extra.

### Callbacks (`src/dl_techniques/callbacks/`)
Keras callbacks; analyzer integration; `TemperatureAnnealingCallback`.

### Utils / Datasets
- `utils/`: `logger`, `weight_transfer`, GPU setup.
- `datasets/vision/coco_burst_dp.py` + `image_folder_burst_dp.py`. 9 PASS tests.

### Applications (`src/applications/`)

- **`applications/bias_free_denoiser/`** -- flat 4-file layout (`__init__.py`, `samplers.py`, `main.py`, `README.md`). Conventions: logger only, type hints, Google docstrings.

- **`applications/anomaly_detection/`** -- `PatchEntropyAnomalyDetector` reusing single-scale `ConvNeXtPatchVAE` encoder-only (`encode()` -> `(mu, log_var)`) for per-patch KL anomaly scoring. Entry points: `from_pretrained(path)`, `preprocess()`, `kl_maps()`, `anomaly_mask()`, `score()`, `overlay()`. GUI: `streamlit_app.py` (isolated). Invariants: `log_var` clipped `[-10, +10]`; inputs `/255.0` in `[0,1]`.

### Training pipelines (`src/train/`)
- **`train/video_jepa/`** -- V-JEPA pretrainer; smoke + BDD100K dataset wiring; reload-check at `train_video_jepa.py:515`.
- **`train/convnext_patch_vae/`** -- ConvNeXtPatchVAE trainer (single-scale only). `compile(loss=None, jit_compile=False)`. Reload check via `encode()` mu. `--smoke` with explicit tiny `img_size=32` override. `TrainingConfig.augment_color: bool = True`.
- **`train/cliffordnet/`** -- CliffordNet + CliffordCLIP trainers.
  - **`train_clip.py`**: `--mixed-bfloat16`, `--probe-every-steps` default=750, `--gamma-probe-every-steps`; `GammaProbeCallback` logs mean `vision_head_scale.gamma` / `text_head_scale.gamma`. `IMAGENET_MEAN/STD` used for the `'imagenet'` normalization branch (`# DECISION plan_2026-06-02_35651564/D-002` at line 88).
  - **`eval_clip_retrieval.py`** (159 LOC) -- COCO zero-shot R@1/5/10 harness.
  - **`filter_cc3m_clipscore.py`** (323 LOC) -- CC3M per-pair CLIP-score caption filter. Full 2.9M pass is user-launched.
- **`train/logic/`** -- LearnableNeuralCircuit benchmark suite + `multiseed_sweep.py` subprocess driver + `multiseed_stats.py`. 30-test stats harness.
- **`train/rms_variants_train/`** -- 8-norm comparison harness; `NORM_VARIANTS` append-only invariant (D-001 `config.py:27`). 637+ PASS tests.
- **`train/convnext/`** -- 3 ConvNeXt trainers + 1 comparison driver:
  - `train_convnext_v1.py`: CLI flags `--stochastic-mode {depth,gradient}` (default `depth`), `--seed`, `--no-epoch-analyzer`. Compile with `SparseCategoricalCrossentropy(from_logits=True)` -- the classifier head is bare Dense (no softmax; emitting logits). `set_seeds(args.seed)` called before `load_dataset`.
  - `train_convnext_v2.py`: mirror of v1 plumbing. Already had `from_logits=True`.
  - `train_convnext_v2_mae.py`: `--stochastic-mode` + `--seed` threaded explicitly through fixed-signature `create_convnext_encoder(stochastic_mode=...)` -> `create_convnext_v2(...)` (no `**kwargs` fallback; param must be named explicitly at every boundary).
  - `run_stochastic_comparison.py`: serial subprocess driver. Launches each mode once, discovers the run dir via snapshot-diff of `results/` (assert exactly-one-new-dir or `SystemExit`). Driver runs **CPU-only** (`CUDA_VISIBLE_DEVICES=''` at module top before TF import) to avoid fragmenting the trainer's XLA allocator. Child subprocess gets GPU via hard-set `env['CUDA_VISIBLE_DEVICES']=str(args.gpu)`. Calls `compare_runs(depth_dir, gradient_dir, labels=('depth','gradient'), ...)` from `train.common.compare_runs`.
  - `README.md`: documents the stochastic_mode knob, driver usage, experiment verdict (depth > gradient on CIFAR-10), and 4 gotchas (from_logits, driver CPU-only, strides coupling, epoch-analyzer cost).
  - **Experiment verdict (CIFAR-10, seed 42)**: `depth` (StochasticDepth) outperforms `gradient` (StochasticGradient) -- +0.6pt at 30ep, +1.26pt at 100ep; gradient mode overfits more. Prefer `stochastic_mode='depth'` (default).

- **`train/ccnets/`** -- Causal Cooperative Nets trainers. 4 `train_<task>.py` (mnist/cifar100/cifar100_hybrid/text_sentiment) + 2 `run_<experiment>.py` (baseline_comparison/latent_sweep). All architecture imported from `dl_techniques.models.ccnets`; thin wrappers with main/argparse/setup_gpu/set_seeds/--smoke. Uses CUSTOM `CCNetTrainer` GradientTape loop (NOT model.fit / create_callbacks) — intrinsic CCNet deviation. Sanctioned data-prep sibling edges: `run_latent_sweep`->`train_mnist` (DataConfig + prepare_mnist_data), `train_cifar100_hybrid`->`train_cifar100` (data/eval/config). `README.md` (not CLAUDE.md). Smoke train: acc=0.9504 on mnist, loss finite/decreasing.

### Training common (`src/train/common/`)
**3-plan consolidation arc (`30721a0f` -> `35651564` -> `cc4d4e14`) is now largely complete.** Remaining duplication is intentional-divergence (7 C1 LR sites) or out-of-scope (F13 bug-fix, risky F6/seed sites).

- **`generation_probe.py`** -- `GenerationProbeCallback(logits_fn, probe_every_steps, prompts, encoding_name, max_tokens, temperature, top_p, repetition_penalty, eot_token_id, pad_token_id, ctx_length, stop_on_eot, save_dir, initial_step, step_counter, seed, gc_on_probe, trigger_requires_positive_step)`. Closure contract: `logits_fn(ctx_ids[1,seq]) -> float32[vocab]` for last real position (UNPADDED input; common class always reads `[0,-1,:]`). `_post_generate_hook` overridable. Replaces 5 per-trainer copies (992 LOC). Re-exported via `nlp.py` + `__init__.py`. NOT `@register_keras_serializable`.
- **`step_checkpoint.py`** -- `StepCheckpointCallback(keras.callbacks.Callback)` superset of 6 former per-trainer copies. Constructor: `(save_dir, save_every_steps, analyze_every_steps, max_checkpoints, model_name, initial_step, log_every_steps, plot_every_steps, step_counter=None, gc_on_save=False, csv_fields=None)`. `_global_step` persists across `fit()` calls (resume support). NOT `@register_keras_serializable`.
- **`seed.py`** -- `set_seeds(seed)`. Sets PYTHONHASHSEED + `random.seed` + `np.random.seed` + `keras.utils.set_random_seed`. **Do NOT use at CLM-resume sites** that carry `data_seed = config.seed + initial_step` after the keras seed line.
- **`config_io.py`** -- `save_config_json(config, results_dir, filename="config.json")`: dataclass -> `dataclasses.asdict`; else `vars(config)`; numpy-safe via `json_numpy_default`. `json_numpy_default(obj)`: `np.floating->float`, `np.integer->int`, `np.ndarray->.tolist()`; raises `TypeError` otherwise.
- **`augment.py`** -- `augment_patch(patch: tf.Tensor) -> tf.Tensor` (flip-lr + flip-ud + rot90), `augment_pair(patch, target) -> (tf.Tensor, tf.Tensor)` (same transforms applied consistently to both). Replaces 7 denoiser copies. `bfunet/train_conditional.py` (class_label variant) stays local.
- **`evaluation.py`** -- `setup_visualization_manager(save_dir, color_scheme) -> VisualizationManager`. Replaces the one live caller (`convnext_v2`); resnet/vit copies were dead code and were deleted.
- **`datasets.py`** -- Three DISTINCT normalization constant pairs:
  - `CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]` / `CIFAR10_STD` (CIFAR-10 channel stats)
  - `IMAGENET_MEAN = [0.485, 0.456, 0.406]` / `IMAGENET_STD = [0.229, 0.224, 0.225]` (ImageNet channel stats)
  - DISTINCT from OpenAI CLIP `IMAGE_MEAN/STD` in `image_text.py` (`[0.48145466,...]`) -- never conflate.
  - `load_dataset()` for CIFAR/ImageNet/MNIST (does `/255` only, NO per-channel normalization).
  - `make_imagenet_filesystem_dataset(data_dir, image_size, batch_size, is_training, augment, augment_color, shuffle_buffer, num_parallel_calls, cache_val, drop_remainder, prefetch_buffer)` -- class-subdir ImageNet `*.JPEG` tf.data builder.
  - `collect_image_paths(directories, extensions, max_files, shuffle_seed, sort) -> List[str]` -- rglob path collector. Does NOT replace early-break-after-N monitor/preview sites (fundamentally different semantics).
- **`callbacks.py`** -- `EpochMetricsPlotCallback(viz_dir, metric_names, every_n=5, write_json=False)`. `create_learning_rate_schedule(..., warmup_steps=0, warmup_start_lr=0.0)`: warmup now wired to `WarmupSchedule` when `warmup_steps>0`; `warmup_steps=0` is a no-op (all pre-plan callers unaffected). 4 canonical sites adopted (tirex, prism, nbeats, adaptive_ema); 7 sites with divergent `alpha`/`warmup_start_lr`/`decay_steps` stay local.
- **`image_text.py`** -- `load_coco2017_local_split`, `load_cc3m_local_split` (with npz tokenization sidecar cache), `make_image_text_tf_dataset`, `tokenize_captions`. `IMAGE_MEAN/STD` = OpenAI CLIP normalization (`[0.48145466,...]`).

## Boundaries
**In scope**: everything under `src/`, `tests/`, `research/`. Library code follows Keras 3 conventions strictly.
**External deps**: tiktoken (gpt2/cl100k_base), HuggingFace datasets (Wikipedia), tensorflow 2.18.0, keras >=3.8.0 <4.0, numpy, scipy, scikit-learn, matplotlib (always `MPLBACKEND=Agg`).
**Out of scope**: external systems, CI infra, deployment pipelines.

## Invariants

### Keras 3 idioms (library-wide)
- `@keras.saving.register_keras_serializable()` on all custom layers/models (NOT on callbacks).
- `keras.ops` (no raw TF inside library code). Random ops live under `keras.random.*`.
- Full `get_config()` round-trip.
- `dl_techniques.utils.logger` only -- no `print`.
- Frozen tensor state in layers MUST be `add_weight(trainable=False, ...)` or numpy on `self`.

### CLM training (library-wide)
- **Output dict key MUST be `"logits"`** -- `MaskedCausalLMLoss` and `model.compile(loss={"logits": ...})` key on it.
- **AdamW WD only** -- no `kernel_regularizer=L2(...)` combined with `AdamW(weight_decay=...)`.
- **`prepare_dict_keyed_compile(model, output_key="logits")`** is a permanent contract for dict-output trainers.
- **Pattern-3 CLI uniformity** -- `--steps-per-epoch`, `--seed`, `--min-article-length`, `--shuffle-shards`, `--resume`.
- **`pad_token_id`** must match tokenizer (tiktoken cl100k_base = 100266); model default 0 is a silent semantic bug.

### Causality (Clifford / time-series)
- `(H=1, W=seq_len)` Clifford blocks must use `avg`/`max` pool only.
- DSv2 nearest-upsample requires `_causal_upsample` right-shift by `s-1`.

### Custom `train_step` / multi-optimizer
- **Two-optimizer differential-LR**: register ONE optimizer with `super().compile(...)`, apply the second manually.
- **`current_phase` / `_global_step`: `add_weight(trainable=False, dtype="float32")`** -- int32 caused device-placement failures.
- **Custom `train_step` bypassing `compile(loss=...)` (uses `add_loss`)** MUST explicitly create `self.loss_tracker` in `__init__`, expose via `metrics`, and `update_state(loss)` in `train_step`.
- **`training` flag gates** -- use `(training is True)` Python identity check under `@tf.function`.
- **VAE compile pattern**: `model.compile(optimizer=AdamW, loss=None, jit_compile=False)`.

### Save / load
- `.keras` save/load on GPU under fp32 has reduction-order noise ~5e-5 for U-Net-shaped models. Default tolerance 1e-4.
- Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken. Use `weight_transfer.load_weights_from_checkpoint`.
- Custom-subclass Model wrapping another `keras.Model`: use `save_own_variables` / `load_own_variables`.

### Operational
- **`results/` MUST be repo-root `results/`, never `src/results/`**.
- **`MPLBACKEND=Agg`** required prefix for any training-script invocation.
- **Single GPU jobs only** -- never parallel training.
- **Pin GPU via shell env** -- Use `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train.<...>`.
- **All training callbacks that write to `save_dir` MUST `os.makedirs(..., exist_ok=True)` at the top of every save method**.

## Flows
- **CLM training** -- `python -m train.<model>.pretrain --config small` -> `TrainingConfig` -> tiktoken/Wikipedia -> packed token shards -> `MaskedCausalLMLoss` -> AdamW + warmup-cosine -> `StepCheckpointCallback` + `GenerationProbeCallback`.
- **V-JEPA pretraining** -- `python -m train.video_jepa.train_video_jepa --smoke --dataset bdd100k --videos-root <...>` -> online + EMA-target encoders -> custom `train_step` -> reload-check (bit-exact).
- **Image training (canonical: `train/vit/train_vit.py`)** -- `@dataclass TrainingConfig` -> `make_imagenet_filesystem_dataset` (or per-dataset builder) -> augment-then-normalize -> `_assert_train_val_distribution_match` -> AdamW/SGD -> `EpochMetricsPlotCallback` -> guarded SUCCESS log.
- **ConvNeXtPatchVAE training** -- `python -m train.convnext_patch_vae.train_convnext_patch_vae [--smoke] [--seed N]` -> `TrainingConfig` -> CIFAR-10 -> `compile(loss=None, jit_compile=False)` -> `ReconVisualizationCallback` -> reload check via `encode()` mu.
- **CliffordCLIP A/B training** -- `CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -m train.cliffordnet.train_clip --head-kind plain|learned_query_residual --mixed-bfloat16 --gamma-probe-every-steps N`. Signal-floor gate: COCO R@1 >= 5% on plain arm.
- **CliffordCLIP zero-shot eval** -- `python -m train.cliffordnet.eval_clip_retrieval --checkpoint <ckpt.keras> --coco-root <...>` -> prints R@1/5/10.
- **CC3M CLIP-score filter** -- `python -m train.cliffordnet.filter_cc3m_clipscore --checkpoint <ckpt.keras> --cc3m-root <...> --out-manifest <filtered.jsonl>` (full pass USER-LAUNCHED).
- **Save / load round-trip** -- `@register_keras_serializable()` decorator -> `.keras` archive -> `keras.models.load_model(path, custom_objects={...})`.
- **Multi-seed sweep** -- subprocess-per-seed driver, glob+merge, pure-stats module.
- **Test scoping** -- pytest only on changed module + immediate importers. `make test` reserved for explicit pre-push request (~1.5h).
- **Plan close** -- orchestrator writes `summary.md`, runs decision-anchor audit, updates `plans/LESSONS.md` (<=200 lines) and `plans/SYSTEM.md` (<=300 lines), then `bootstrap.mjs close`.

## Known Patterns
- **Pattern-3 NLP CLM training script** -- ~95% generic `TrainingConfig` + `StepCheckpointCallback` + `GenerationProbeCallback` (from `train.common`) + AdamW/warmup-cosine + tiktoken/Wikipedia + `MaskedCausalLMLoss`. New CLM = mirror file-by-file.
- **Pattern-4 image trainer (canonical: `train/vit/train_vit.py`)** -- `@dataclass TrainingConfig` + `make_imagenet_filesystem_dataset` (or per-dataset builder) + AdamW/SGD WD branch + `_assert_train_val_distribution_match` + `EpochMetricsPlotCallback` + guarded SUCCESS log. New image trainer = mirror file-by-file.
- **Pattern-4 VAE variant (canonical: `train/convnext_patch_vae/train_convnext_patch_vae.py`)** -- hybrid Pattern-4 + compile(`loss=None, jit_compile=False`) + `ReconVisualizationCallback` + reload check via deterministic `encode()` mu.
- **EMA target encoder pretraining (canonical: `models/video_jepa/`)** -- sibling encoder with `trainable=False`, dummy-batch eager build, custom `train_step` with EMA update, cosine momentum via `add_weight("ema_step")`. Do NOT use for VAE objectives.
- **Two-optimizer differential-LR via custom `train_step`** -- register one optimizer with `super().compile(...)`, apply the second manually; variable routing via name-prefix split.
- **Subprocess-per-seed multi-seed sweep** -- clean TF/Keras init eliminates cross-seed state contamination.
- **Mechanical patch replication across N sibling files** -- per-file repetition beats shared-helper extraction for N<=4 if extraction would cross package boundaries.
- **Eval harness = glue, not build** -- for any model with `_compute_retrieval_metrics` / `encode_*` methods, a zero-shot eval harness is ~argparse + `load_model` + 3 existing function calls; write it as a thin script, not a new class.

## Codebase Specialization
- **Python**: >=3.11, type hints, Google-style docstrings.
- **Venv**: always `.venv/bin/python` for invocation.
- **Logging**: `from dl_techniques.utils.logger import logger` -- never `print`.
- **GPU**: GPU 0 = RTX 4090 24GB, GPU 1 = RTX 4070 12GB. Pin via `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train....`.
- **Test discipline**: scope pytest to touched modules; `make test` only on explicit user request immediately before push.
- **Commit prefix**: `[iter-N/step-M] <description>` during EXECUTE; no commit on EXPLORE/PLAN/REFLECT/PIVOT.
- **Plans dir**: `plans/` is in `.gitignore`; plan files never committed.
- **Push default**: `git push --no-verify` (skip 1.5h pre-push hook). User runs full suite separately.
