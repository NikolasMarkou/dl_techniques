# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed. Read full content below for plan-specific detail.*

### Key Findings
- **TreeTransformer (`models/tree_transformer/`)** is structurally sound — save/load + gradient flow + MLM-wrapper integration all correct. Four real bugs fixed in plan_3c3ed037: B-1 fp16 NaN (dtype-aware mask sentinel `-1e4` under float16, plus fp32 cast on GroupAttention DP log/matmul/exp block); B-3 explicit `attention_mask` honored in dict input; B-4 `load_pretrained_weights` via `weight_transfer.load_weights_from_checkpoint` (Keras 3.8 `by_name=True` broken); B-5 `PRETRAINED_WEIGHTS={}` + `NotImplementedError` (no public checkpoints). Trainer `src/train/tree_transformer/{pretrain,finetune}.py` mirrors `bert/`. Anchor: `model.py:318` D-001. **Trainer config MUST pass `pad_token_id=config.pad_token_id` (tiktoken cl100k_base = 100266) to encoder — model default 0 is silent semantic bug.** Aligned to `bert/`/`resnet/` conventions in plan_0a5779e8: bare `create_tree_transformer(variant, ...)` factory added, `__init__.py` trimmed to 3 names (`TreeTransformer`, `create_tree_transformer`, `create_tree_transformer_with_head`; internal layer classes remain importable from `.model` for `nam/` consumers), and `from_variant(pretrained=True)` now raises `NotImplementedError` loudly instead of silently random-initializing (D-001 anchor at `model.py:1133`, narrowed try/except to `(IOError, OSError, ValueError)`).
- **TinyRecursiveModel (`models/tiny_recursive_model/`)** — save/load clean. B-3 Q-learn lookahead `training=False` + `keras.ops.stop_gradient` on `target_q`; B-5 inference halts on learned signal. `hrm_loss`/`HRMMetrics` API-compatible with TRM output schema. Anchor: `model.py:370` D-001 (plan_e6309bd5).
- **`keras.ops.expand_dims(axis=tuple)` works** on Keras 3.8 / TF 2.18 eager + `@tf.function` (B-1 false-positive in plan_e6309bd5).
- **DepthAnything** is now full-feature — real ViT encoder, DPTDecoder linear default + `upsample_factor`, weight-shared frozen teacher via `clone_model`, semi-sup `train_step` (FAL + L1 pseudo-label stop-gradient), on-step EMA via `TeacherEMACallback`, `from_pretrained_encoder(path)`, `StrongAugmentation` + dynamic cutmix.
- **Keras 3 / TF 2.18 idioms**: `keras.random.*` (NOT `keras.ops.random.*`); `keras.ops.*` for backend-agnostic ops; `@keras.saving.register_keras_serializable()` + `get_config()` round-trip; `dl_techniques.utils.logger` only.
- **Save/load on subclassed Models wrapping inner Models**: weights drop unless outer class overrides `save_own_variables` / `load_own_variables` (D-004).
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken** — use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.
- **AdamW double weight decay footgun**: never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)`.
- **CLM training**: use `train.common.nlp.estimate_clm_steps_per_epoch`; `min_article_length=0` correct for packed pipelines.
- **Two-optimizer differential-LR**: register one with `super().compile`; apply second manually inside `train_step` via name-prefix variable routing (leading-component match).
- **`keras.ops.cond` traces BOTH branches under `tf.function`** — multiply-by-zero for compute-amount differences.
- **Frozen state in layers**: `add_weight(trainable=False)` or numpy-on-self — never plain tensors in `build()` (FuncGraph dead-tensor).
- **BERT (`models/bert/`)** aligned to resnet/tree_transformer template in plan_9357982a: `create_bert` bare-encoder factory added; `__init__.py` trimmed to 3-name surface `{BERT, create_bert, create_bert_with_head}` (drop `create_nlp_head` re-export); `_download_weights` raises `NotImplementedError`, `from_variant` try/except narrowed to `(IOError, OSError, ValueError)` (D-001 anchor at `bert.py:687`); docstring/README path fixed to `dl_techniques.layers.nlp_heads`. 28/28 pytest PASS, 0 fix attempts.
- **AccUNet** requires H,W divisible by 16; validation in `call()` raising `ValueError` (plan_bdb2c84d D-001/D-002).
- **`SegmentationWrapperLoss`** is canonical save/load-friendly segmentation loss; `compile=False` workaround removed (plan_17633038 D-002).

### Key Decisions
- **D-001 plan_3c3ed037 (TreeTransformer bundle)**: 4 model bugs + Pattern-3 trainer in one iteration — 5 new files / +950 LOC at the cost of 2 over file-budget; trainer depends on Step 5 re-exports and Step 2 attention_mask honoring, so splitting would force pinning to broken imports.
- **D-001 plan_e6309bd5 (TRM bundle)**: bug fixes + factory + tests + trainer in one plan — at cost of larger review surface; B-5 testable only with same harness as trainer eval path.
- **Pseudo-label loss**: plain L1 + `stop_gradient`, not `compute_loss` against synthetic mask (plan_54e6e303 D-002).
- **Encoder weight-loading**: keep `--pretrained-encoder-weights` + `--init-from` distinct (plan_54e6e303 D-003).
- **D-004 (save_own_variables override)**: canonical Keras-3 fix when `.keras` round-trip drops sub-Model weights.
- **D-003 (Keras-3 canonical train_step)**: `compute_loss(x,y,y_pred)` adds `self.losses` internally — no manual regularization addition.
- **D-005 (StrongAugmentation graph-mode safety)**: symbolic gate; `keras.random.*` not `keras.ops.random.*`.
- **CLM metrics architecture**: math in `dl_techniques/metrics/`; list in `train/common/nlp/build_clm_metrics()`; fresh instances each call.
- **`current_phase` / `_global_step` counters**: `add_weight(trainable=False, dtype="float32")` — int32 fails CPU/GPU device placement.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-06-04_d4ef81f1
### Index

| # | Finding | File | Covers |
|---|---------|------|--------|
| 1 | VAE model structure, sampler site, KL loss, config, z_log_var shapes | `findings/vae-model-and-loss.md` | model + loss + the 2 blockers |
| 2 | Canonical registry factory pattern (sequence_pooling) + inline-vs-package options | `findings/factory-pattern.md` | factory design |
| 3 | train_vae.py gaps + comparison-driver template + compare_runs signature | `findings/vae-trainer-and-comparison.md` | trainer + comparison |
| 4 | Locked design (resolved with user) | this file, "Design Decision" | scope/architecture |

### Key Constraints

### Hard
- **Factory inline** in `layers/sampling.py` (1 precedent `sparse_autoencoder.py:823`; package promotion would break the 3 existing `from dl_techniques.layers.sampling import Sampling` sites). 2 layer types: `gaussian`->`Sampling`, `hypersphere`->`HypersphereSampling`.
- **z_log_var shape fork**: `Sampling` needs `[B,latent_dim]`; `HypersphereSampling.build` hard-rejects last-dim != 1 (`sampling.py:360-364`). Hypersphere modes must feed `[B,1]`.
- **KL/prior mismatch**: Gaussian KL (`-0.5·Σ(1+logvar−μ²−e^logvar)`, `model.py:786-810`) is the wrong prior for the sphere — faithful mode replaces it.
- **Decoder graph extraction** at `model.py:242-244` uses `self.get_layer("vae_sampling").output` — ALL modes MUST keep the sampler layer name `"vae_sampling"` (output is `[B,latent_dim]` in every mode, so extraction is unaffected).
- **`create_vae` assertion** `model.py:948` `z_log_var.shape==(2,latent_dim)` — must branch for faithful ([B,1]).
- **`CUSTOM_OBJECTS`** `train_vae.py:36` only lists `Sampling` — add `HypersphereSampling` for the hypersphere arms' checkpoint reload.
- **compare_runs** (`train/common/compare_runs.py:198`, `(run_a, run_b, labels, output_dir)`) is 2-arm + needs pandas + `training_log.csv` per run. For 3 arms call it pairwise vs the gaussian baseline.
- **Driver CPU-only**: `CUDA_VISIBLE_DEVICES=''` at module top BEFORE TF import (XLA-allocator crash risk; convnext pattern). Child arm gets GPU via `env['CUDA_VISIBLE_DEVICES']=str(gpu)`. Serial subprocess; NEVER parallel.
- **GPU1 for our runs** (`CUDA_VISIBLE_DEVICES=1`, RTX 4070). Single job at a time.
- **Full A/B run (>2 min) MUST be launched by the MAIN thread (orchestrator), NOT a sub-agent** (LESSONS: `run_in_background` from a sub-agent dies when the sub-agent exits). ip-executor handles only the short --smoke; orchestrator runs the full comparison.

### Soft
- Mirror `sequence_pooling/factory.py` surface: `SamplingType` literal, `SAMPLING_REGISTRY`, `create_sampling_layer(type, name=None, **kwargs)`, `create_sampling_from_config(config)`, `validate_sampling_config(type, **kwargs)`, `get_sampling_info()`.
- train_vae.py has no `--smoke`/`--seed`/`--sampler`/`config.json` today — add all four; `set_seeds(args.seed)`; `save_config_json(args, results_dir)`.
- Monitor metric is `val_total_loss`; dataset MNIST default (28x28x1), 10 epochs / batch 128.

### Design Decision (resolved with user via AskUserQuestion)

**Build + RUN the full A/B now on GPU1** (serial). Three VAE `sampling_type` configs:

1. **`gaussian`** (baseline) — unchanged. `Sampling([mu[B,D], log_var[B,D]])`; Gaussian KL over `[B,D]`.
2. **`hypersphere_controlled`** — isolate the sampler. Encoder UNCHANGED (mu[B,D], log_var[B,D]). Adapter: `rlv = mean(log_var, axis=-1, keepdims=True)` -> `[B,1]`; `z = HypersphereSampling([mu, rlv])`. Loss = SAME Gaussian KL on the original `(mu, log_var[B,D])`. Only the sampling op differs. Output dict `z_log_var` stays `[B,D]`.
3. **`hypersphere_faithful`** — geometrically-honest. Encoder: shared trunk -> `mu=Dense(latent_dim)` (direction) + `radius_log_var=Dense(1)` `[B,1]`. `z = HypersphereSampling([mu, radius_log_var])`. Loss = hyperspherical regularizer (replaces Gaussian KL): radius-variance KL `kl = mean(0.5·(exp(rlv) − rlv − 1))` (rlv clipped [-20,20]); direction has a uniform-sphere prior -> NO direction KL term (documented simplification — radius mean is fixed at 1.0 by the layer; not a full vMF S-VAE). Output dict `z_log_var = radius_log_var [B,1]`.

Factory has **2 layer types** (gaussian, hypersphere); the **3rd config (faithful) is a VAE-level encoder+loss mode**, same `hypersphere` layer. All modes name the sampler `"vae_sampling"`.

**Comparison**: `src/train/vae/run_sampler_comparison.py` (new), mirroring `run_stochastic_comparison.py`; 3 serial arms; `compare_runs` called pairwise vs gaussian baseline (gaussian-vs-controlled, gaussian-vs-faithful). Smoke-verify all 3 arms first, then orchestrator launches the full run on GPU1.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-04_a114f829
### Index

| # | Finding | File | Covers |
|---|---------|------|--------|
| 1 | Existing `Sampling` layer is the structural template | `src/dl_techniques/layers/sampling.py:57-218` (read inline) | existing pattern |
| 2 | `Sampling` usage, exports, and test conventions | `findings/sampling-usage-and-tests.md` | affected files, test pattern |
| 3 | Hypersphere math, reuse targets, canonical keras idioms | `findings/hypersphere-math-and-idioms.md` | constraints, idioms |
| 4 | Design intent resolved by user (AskUserQuestion) | this file, "Design Decision" below | scope/interface |

### Key Constraints

### Hard
- **New class is a sibling in the SAME file** `src/dl_techniques/layers/sampling.py` (file already exists with one class `Sampling`). Do NOT create a new file.
- **5-method Keras-3 pattern mandatory** (`research/2026_keras_custom_models_instructions.md`): `__init__` (store config only), `build` (validate shapes; `super().build()` last), `call` (`keras.ops` only), `compute_output_shape`, `get_config` (`super().get_config()` + all params). `@keras.saving.register_keras_serializable()` mandatory.
- **Random ops under `keras.random.*`** — `keras.ops.random.*` does NOT exist in Keras 3.8. Mirror sibling: `keras.random.normal(shape=..., seed=self.seed)`.
- **`layers/__init__.py` is empty by convention** — `Sampling` is NOT exported; callers import `from dl_techniques.layers.sampling import Sampling`. New class needs no export.
- **Logger only, no print**; Google-style docstrings; type hints.

### Soft
- **Mirror sibling idioms**: `seed: Optional[int]` ctor param, `ops.shape(...)`, accept-but-ignore `training`, `logger.debug` init line, ASCII architecture diagram in class docstring.
- **L2 normalize idiom**: `ops.normalize(x, axis=-1)` is the dominant repo pattern. Verify zero-safety in EXECUTE; fall back to manual `x / maximum(norm, eps)` (polar_initializer.py:78-97 pattern) if `ops.normalize` is not zero-safe.
- **Cite references** like `hypersphere_orthogonal_initializer.py` (Marsaglia 1972; Muller 1959) for the Gaussian-normalize-scale method.
- **Tests**: extend existing `tests/test_layers/test_sampling.py` with a `TestHypersphereSampling` class mirroring `TestSampling` (init / forward / shape / gradient / `get_config`+`from_config` / model save+load with `seed=42` deterministic compare per LESSONS "stochastic by design").

### Ghost (rejected)
- **`keras.random.SeedGenerator`** — explorer flagged int-seed statelessness as a "risk", BUT the sibling `Sampling` deliberately uses raw-int `seed` and the existing `test_model_save_load` relies on it (seed=42 → reproducible). Treat SeedGenerator as a GHOST; mirror the sibling exactly. (LESSONS: "Treat sibling-template invariants as GHOSTS until proven applicable" — here the sibling invariant IS the spec.)

### Design Decision (resolved with user via AskUserQuestion)

New class **`HypersphereSampling`**. Inputs `call([z_mean, z_log_var])`:
- `z_mean`: `[B, D]` — encoder direction (carries information)
- `z_log_var`: `[B, 1]` — per-sample single scalar variance (shell thickness)

Formula:
```
eps = N(0, I)  shape [B, D]
eta = N(0, 1)  shape [B, 1]
u   = normalize(z_mean + eps, axis=-1)     # direction on unit sphere
r   = radius + exp(0.5 * z_log_var) * eta  # radius default 1.0; thin shell
z   = r * u                                # [B, D], ||z|| = |r| ~ radius
```
- Direction from encoder mean + Gaussian noise; magnitude (radius) is a thin Gaussian shell centered at `radius` (ctor float, default 1.0) with per-sample variance from the encoder.
- Always stochastic (mirror sibling; `training` accepted-but-unused). `seed` makes it reproducible for tests.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-03_da3a2bbb
### Index

| # | Finding | File | Key takeaway |
|---|---------|------|--------------|
| F1 | Source file + importers | `findings/source-file-and-importers.md` | `sequence_pooling.py` (935 LOC): 3 classes (`AttentionPooling`, `WeightedPooling`, `SequencePooling`) + 2 type aliases (`PoolingStrategy` 18 strategies, `AggregationMethod`). Bare decorators. Importers: `text_encoder.py:99`, `vision_encoder.py:87` (relative), `tests/test_layers/test_sequence_pooling.py:16` (absolute, 1001 LOC). NOT in `layers/__init__.py` (empty). RST/Sphinx docstrings (non-Google). |
| F2 | attention/ template structure | `findings/attention-template-structure.md` | Canonical factory idiom: `Literal` type alias `<Domain>Type`, `<DOMAIN>_REGISTRY: Dict[str,Dict]` (class/required_params/optional_params/description/use_case[/complexity/paper]), `create_<domain>_layer(type, name=None, **kwargs)`, helpers `validate_*`/`create_*_from_config`/`get_*_info`, explicit `__init__.py` re-exports + string `__all__`. README=catalog, GUIDE=contributor guide. ffn/ cross-validates. |
| F3 | Migration mechanics + serialization | `findings/migration-mechanics.md` | `git mv` into new dir; add `__init__.py` re-exporting 3 classes + `PoolingStrategy` + `AggregationMethod`. No relative imports inside the file. Two callers' `from ..sequence_pooling import ...` resolve UNCHANGED (parent still `layers/`). Test absolute import unchanged. Precedent: `neuro_grid.py`→`memory/` (708e615c). |

### Key Constraints

- **[HARD] Serialization keys are SAFE.** Bare `@keras.saving.register_keras_serializable()` → key `Custom>ClassName`, derived from default `package="Custom"`, **NOT from `__module__`**. Verified by orchestrator directly (see Corrections F1-correction). Moving the file does NOT change keys; existing `.keras` saves load fine as long as the class is imported (registered) before load. Re-export from `__init__.py` guarantees that.
- **[HARD] `PoolingStrategy` and (used) symbols must be re-exported** from `sequence_pooling/__init__.py` or `text_encoder.py`/`vision_encoder.py` break at runtime (they do `from ..sequence_pooling import SequencePooling, PoolingStrategy`).
- **[HARD] `SequencePooling.__init__` instantiates `AttentionPooling`/`WeightedPooling` directly** — if classes are split across files, intra-package imports must be wired correctly.
- **[HARD] Factory idiom is fixed**: `create_sequence_pooling_layer(pooling_type: SequencePoolingType, name=None, **kwargs)`, registry dict, pure registry dispatch (no if/elif), explicit string `__all__` (do NOT replicate ffn's object-based `__all__` bug).
- **[SOFT] No `__all__` in source file** — adding one clarifies public surface.
- **[SOFT] Docstring dialect is RST/Sphinx** (`:param:`/`:type:`) not Google-style repo convention. Converting is optional scope.
- **[SOFT] GUIDE.md** exists only in `attention/` (not ffn/norms/memory). User explicitly requested it ("GUIDE etc etc").
- **[GHOST] "Moving deeper breaks relative imports by one dot"** — does NOT apply here: the source file has zero `from ..`/`from ...` imports, and external callers keep parent `layers/`.

### Corrections

- **[CORRECTED iter-0] F1 serialization-key claim was WRONG.** `findings/source-file-and-importers.md` (Summary + Constraints) and its claim that keys are "derived from `__module__`" / "silently break on move" is FALSE. Orchestrator verified directly via `keras.saving.get_registered_name()`: a bare-decorated class registers as `Custom>ClassName` regardless of `__module__` (confirmed key `Custom>SequencePooling` while `__module__`=`__main__`). `findings/migration-mechanics.md` is the correct account: **no serialization break, no `package=` pin needed.** This de-risks the migration substantially.

## plan_2026-06-03_5c8c6d19
### Index

| # | Finding | File | Key facts |
|---|---------|------|-----------|
| 1 | CCNets train folder current state | `findings/ccnets-current-state.md` | 6 scripts + `CLAUDE.md`, no `README.md`; all 6 violate `train_<model>.py`; no argparse in 4; architectures defined inline; cross-script imports between train files |
| 2 | Canonical train-folder conventions | `findings/train-conventions.md` | `train_<task>.py` naming; `README.md` (not subfolder `CLAUDE.md`); models imported from `dl_techniques.models.*`; `main()`+argparse via `create_base_argument_parser`; `setup_gpu(args.gpu)`/`set_seeds(args.seed)`; 12-point checklist |
| 3 | CCNets model package vs train scripts | `findings/ccnets-models.md` | `models/ccnets/` is framework-only (orchestrator/trainer/config/losses), zero architectures; 17 classes in `mnist.py`, 7 in `cifar100.py`, text classes in `text_sentiment.py`; factories `create_mnist_ccnet`/`create_cifar100_ccnet` live in train scripts |

### Key Constraints

### HARD
- CCNet framework contracts (PRINCIPLES_CCNETS.md P1-P11): three-network design `explainer(x)->(mu,log_var)`, `reasoner(x,e)->y`, `producer(y,e)->x_hat`; differentiable label projection `Dense(use_bias=False)` not `Embedding`; variational Explainer. Must be preserved byte-for-byte across any move.
- `src/train/CLAUDE.md:42` — scripts must be named `train_<model>.py`, never bare nouns (`mnist.py` shadows package names).
- Model architecture classes must NOT live in train scripts — belong in `dl_techniques/models/<pkg>/`.
- `CCNetTrainer` owns a CUSTOM training loop (manual GradientTapes, per-network optimizers, KL annealing) — `model.fit()` is NOT used, so `train.common.create_callbacks()` cannot wrap it directly. This is an intrinsic deviation, not neglect.
- `setup_gpu(args.gpu)` must receive the parsed `--gpu` arg.

### SOFT
- Provide `README.md` (not subfolder `CLAUDE.md`) matching the convex/cliffordnet README structure.
- Provide `main()` + argparse with at least `--gpu/--epochs/--batch-size`.
- Call `set_seeds(args.seed)` consistently (currently only 2 of 6 scripts).
- Use `save_config_json` / `validate_model_loading` post-training where feasible.

### GHOST
- Subfolder-level `CLAUDE.md` — only `src/train/CLAUDE.md` should be the AI-instruction layer; no other train subfolder has its own `CLAUDE.md`. The ccnets `CLAUDE.md` content is mostly findings/results that belong in a README.
- `dynamic_weighting` flag in `CCNetConfig` is deprecated (`base.py:76`), stays `False`.

### Structural insight (drives the plan)

Moving architectures into `dl_techniques/models/ccnets/` SOLVES the cross-script-import fragility: today `cifar100.py`<-`mnist.py`, `cifar100_hybrid.py`<-`cifar100.py`, `baseline_comparison.py`/`latent_sweep.py`<-`mnist.py`. Once classes live in the model package, every train script imports cleanly from `dl_techniques.models.ccnets.*` and renaming becomes safe. Therefore: **architecture migration must precede script renaming.**

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*
