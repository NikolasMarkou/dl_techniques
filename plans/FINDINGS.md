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

## plan_2026-05-25_74f0eac9
### Index

| ID | Topic | File | One-line summary |
|----|-------|------|------------------|
| F1 | Model source deep review | `findings/model-source-review.md` | Architecture, loss, train_step, serialization fully documented. 6 quality issues found: use_v2_block dead code, decoder missing docstrings (2), tf.GradientTape backend coupling (repo-wide), optimizer.apply_gradients older API, SIGReg N<knots advisory-only. |
| F2 | Canonical training pattern | `findings/training-pattern.md` | Trainer must hybrid Pattern-4 (vit/train_vit.py) + video_jepa (compile loss=None + custom train_step). CIFAR-10 pipeline with MSE. Key constraints: jit_compile=False, steps_per_epoch required, monitor=val_loss, success threshold on loss not accuracy. |
| F3 | Tests and ConvNextV2Block interface | `findings/tests-and-blocks.md` | 9 test classes, 277 lines, all pass. ConvNextV2Block has no drop_path_rate (EXPANSION_FACTOR=4 hardcoded). No src/train/convnext_patch_vae/ exists yet. train.common exports setup_gpu, create_callbacks, create_base_argument_parser. |

### Key Constraints

### HARD
- `model.compile(loss=None, jit_compile=False)` — losses exclusively from `add_loss` in `call()`. Passing compile loss causes double-counting.
- `img_size % patch_size == 0` enforced by config `__post_init__` and `encoder.build()`.
- File named `train_convnext_patch_vae.py` (not `train.py` — shadows train package per `src/train/CLAUDE.md`).
- `setup_gpu(args.gpu)` before any TF context init; `MPLBACKEND=Agg` before matplotlib imports.
- Outputs to `results/` at repo root (never `src/results/`).
- All 9 existing tests must continue to pass unchanged.
- `pretrained=True` raises `NotImplementedError` — no public checkpoints.
- No `git push` — user handles pushes.

### SOFT
- Use `TrainingCurvesCallback` from `dl_techniques.callbacks.training_curves` — auto-groups VAE metric names.
- Use `create_callbacks()` from `train.common` for EarlyStopping + ModelCheckpoint + CSVLogger.
- Include post-fit reload check (video_jepa pattern).
- Include `--smoke` flag for fast CPU smoke test.
- Include reconstruction visualization callback (adapted from `train/vae/train_vae.py:VisualizationCallback`).
- Success guard on `val_loss <= threshold` (lower-is-better) not accuracy.
- `recon_loss_type="mse"` default for CIFAR-10 with mean/std normalization.

### GHOST
- `use_v2_block=False` V1 block path — dead code, field exists but never consumed; T4c ablation deferred.
- EMA target encoder — not applicable (VAE recon forbids identity).
- Learned absolute positional embedding — breaks resolution-agnostic property.
- `model.load_weights(path, by_name=True)` — broken in Keras 3.8; not needed here (no pretrained weights).

### Corrections
*None — first iteration.*

### Exploration Confidence
- **scope**: deep — all 5 model source files read via explorer agent; canonical trainers (vit, video_jepa, vae) read; callbacks, common utilities, all 9 tests documented.
- **solutions**: constrained — clear hybrid pattern (Pattern-4 + video_jepa compile); CIFAR-10 primary; reconstruction viz callback available to adapt.
- **risks**: clear — existing test suite verifies no model regressions; smoke flag enables fast CI verification.

## plan_2026-05-25_8faec5b6
### Index

| ID | Topic | File | One-line summary |
|----|-------|------|------------------|
| F1 | Current files vs guide | `findings/current-files-vs-guide.md` | Per-file audit listing 13 items (G1-G13). 11 are PASS or doc-only; 2 are real additive changes: (G3) factory + presets + `from_variant`, (G1/G5) `compute_output_shape` on top-level model. Blast radius confirmed ZERO outside the package + its test. |

### Key Constraints

### HARD
- Keras 3.8 / TF 2.18; `@keras.saving.register_keras_serializable()` on every custom class; `keras.ops` + `keras.random.*`; `dl_techniques.utils.logger` only.
- Golden Rule (guide §1.1): all `add_weight` and sub-layer `.build()` calls in `build()`, never in `__init__`. Sub-layers created in `__init__`.
- Custom `train_step` that bypasses `compile(loss=...)` MUST keep `self.loss_tracker = keras.metrics.Mean(name="loss")` explicit + `update_state(loss)` (SYSTEM.md D-005 contract, plan_ca745a6c).
- All 8 existing test classes in `tests/test_models/test_convnext_patch_vae/test_convnext_patch_vae.py` MUST continue to pass unchanged.
- Save/load round-trip with full `.keras` archive must remain bit-stable to current atol=1e-4 (`TestSaveLoad`).
- SIGReg input contract `(..., N, D)` and resolution-agnostic property (no GAP, no learned absolute PE, no `Dense(latent_dim)` over flattened map) must be preserved.
- Per-patch KL averaging (NOT sum) over `(B, Hp, Wp)` must be preserved (`TestForward.test_per_patch_kl_resolution_invariance`).
- `# DECISION plan_2026-05-25_fb57d478/D-001` (loss_tracker) and `D-002` (SIGReg binding) anchors must remain at impact sites.
- Test runtime: current ~24s scoped; budget <=30s after additions.
- No training runs in this plan.

### SOFT
- Repo convention `models/{bert, resnet, tree_transformer, ...}` requires `create_<model>` factory + `from_variant(pretrained=True) -> NotImplementedError` + narrow `except (IOError, OSError, ValueError)` (SYSTEM.md "Models" section). Add this surface to convnext_patch_vae for parity.
- `PRESETS = {"tiny", "base", "large"}` class attribute on `ConvNeXtPatchVAE` mirroring `ModelWithPresets` (guide §7.2) and `WellStructuredModel` (guide §15.2).
- Anchor new factory decisions with `# DECISION plan_2026-05-25_8faec5b6/D-NNN` at impact sites.
- Google-style docstrings with `Args:` / `Input shape:` / `Output shape:` sections per guide §3.1 (CLAUDE.md mandates Google-style; current files use Sphinx `:param X:` form).
- Keep `__init__(config=...)` ctor as-is for parity with video_jepa template (SYSTEM.md flow). Factory wraps config construction.

### GHOST (do NOT inherit)
- "Expose all public names from `__init__.py`" — invalidated by `src/dl_techniques/models/CLAUDE.md` which mandates empty `__init__.py` for `models/`. Keep current behavior.
- "Rewrite `__init__(config=...)` into flat-kwarg ctor" — video_jepa-template parity + existing tests pin the config-ctor shape. Wrap via factory instead.
- "Add EMA target encoder, positional embedding, temporal axis, mask_token" — invalidated for VAE objective in plan_fb57d478.
- "Validate inputs in `[0,1]` at runtime for BCE branch" — would add graph-mode-unsafe assert; use docstring contract.

### Corrections
*None — first iteration.*

### Exploration Confidence
- **scope**: deep — re-read all 5 source files, the full 3106-line guide, the 226-line test suite, ConvNextV2Block signature, and SYSTEM.md atlas entry for this model. Cross-plan LESSONS.md anchors all 8 D-NNN contracts from plan_fb57d478 + plan_ca745a6c.
- **solutions**: constrained — single concrete delta: factory + presets + `from_variant` + `compute_output_shape` + docstring reformatting. No architectural change.
- **risks**: clear — falsification signals testable on existing scoped pytest in <30s. Only risky surface is `from_config` interaction with `keras.Model` base keys (G6); existing save/load test catches regressions.

### Synthesis

The package is already ~80% guide-compliant — built from plan_fb57d478 with the guide implicitly in mind (compute_output_shape on encoder/decoder, Golden Rule, register_keras_serializable everywhere, explicit loss_tracker, get_config/from_config round-trip, keras.ops only). The remaining 20% is **additive, doc-and-surface polish**: add `create_convnext_patch_vae` factory, presets, `from_variant` classmethod, `compute_output_shape` on top-level model, and convert Sphinx docstrings to Google-style blocks. README gets updated quick-start.

Total: ~+200-300 LOC additive, zero behavior deletions, zero blast radius outside the package and its test. All 8 existing tests stay green; add 2 new factory tests.

## plan_2026-05-25_fb57d478
### Index

| ID | Topic | File | One-line summary |
|----|-------|------|------------------|
| F1 | video_jepa reusable parts | `findings/video-jepa-reuse.md` | Config dataclass + custom `train_step` + `add_loss` per-component + Mean trackers via `metrics` property are direct reuses; tube mask / EMA target / predictor are not. PatchEmbedding2D-+-PE-+-block-stack is the encoder template. |
| F2 | SIGReg shape contract + binding | `findings/sigreg-shape-contract.md` | Input `(..., N, D)`, averages over last-but-one. For patch-VAE bind as `(B, Hp*Wp, latent_dim)` on post-reparam `z` (Option A/C). `num_proj` >= 256, `N` >= 64 for stable estimate. |
| F3 | ConvNeXt assets present | `findings/convnext-assets.md` | `ConvNextV1Block` + `ConvNextV2Block` (with GRN) + standalone `GlobalResponseNormalization` are repo primitives. `ConvNeXtV2` model is image-classifier (global pool) — not reusable; we build a flat single-stage block stack. |
| F4 | VAE infrastructure gap | `findings/vae-infra-inventory.md` | `Sampling` layer works on arbitrary rank >= 2 (per-patch latents OK). Existing `VAE` is image-level w/ global pool — not reusable. `MaskedAutoencoder` is patch-level but deterministic. **Gap: no patch-level continuous-Gaussian VAE.** Build new `models/convnext_patch_vae/`. |
| F5 | Anti-collapse precedent + latent layout | `findings/collapse-precedent.md` | SIGReg is the canonical tool; DINO/VICReg/Barlow not in repo. Recommended layout: 4D latent `(B, Hp, Wp, latent_dim)`, KL averaged per-patch, SIGReg on `(B, Hp*Wp, latent_dim)` view of `z`. GHOSTs: no EMA target needed, no positional embedding needed. |

### Key Constraints

### HARD
- Keras 3.8 / TF 2.18; `@keras.saving.register_keras_serializable()` on every custom class; `keras.ops` only; `keras.random.*` (NOT `keras.ops.random.*`); `dl_techniques.utils.logger` only.
- Golden Rule: all `add_weight` in `build()`. Sub-layers created in `__init__` and explicitly `.build(shape)`-ed in parent `build()`.
- Custom `train_step` bypassing `compile(loss=...)` MUST explicitly create `self.loss_tracker = keras.metrics.Mean(name="loss")` in `__init__` and `update_state(loss)` in `train_step` (plan_ca745a6c D-005 — Keras 3.8 does not auto-create it).
- SIGReg input `(..., N, D)`: `D` last axis must be statically known; averaging over last-but-one (`N`).
- Resolution-agnostic = no `GlobalAveragePooling2D`, no learned absolute positional embedding tied to a grid size, no `Dense(latent_dim)` over flattened spatial map.
- `img_size % patch_size == 0`.
- Single-GPU only (RTX 4090 24GB primary, RTX 4070 12GB secondary). `MPLBACKEND=Agg` for any training script. Out-dir = repo-root `results/`.
- This session: NO training. EXPLORE → PLAN → user approval → STOP at PLAN gate.

### SOFT
- Use `ConvNextV2Block` over V1 (GRN is an additional anti-collapse defense inside the encoder).
- Flat single-stage encoder/decoder at iter-1 (Hp = img_size/patch_size, no spatial downsampling beyond stem); move to hierarchical only if recon at 64-128px is insufficient.
- `beta = 0.5`, `lambda_sigreg = 0.1`, `latent_dim = 16` per-patch, `num_proj = 256` as starting hyperparameters.
- New package `src/dl_techniques/models/convnext_patch_vae/` (zero blast radius on existing models; precedent: sibling-stack additive file).
- Anchor key decisions with `# DECISION plan_2026-05-25_fb57d478/D-NNN` at impact sites.

### GHOST (do NOT inherit)
- **"Need EMA target encoder like video_jepa"** — only required for contrastive/JEPA objectives where identity is the trivial solution. A VAE's reconstruction forbids identity. Don't add EMA — saves ~80 LOC + serialization complexity.
- **"Need learned absolute positional embedding"** — ConvNeXt is translation-equivariant; absolute PE breaks resolution-agnostic property. Drop PE entirely.
- **"Need temporal axis"** — borrowing video_jepa's `(B, T, Hp, Wp, D)` plumbing is overkill; patch-VAE works on `(B, H, W, C)` directly.
- **"SIGReg alone prevents all collapse"** — invalidated in plan_15151c75 (live-target + SIGReg → time-invariance collapse). SIGReg is rank/distribution-fitting, not contrastive. KL is the primary anti-posterior-collapse defense; SIGReg is the secondary per-patch defense. Both needed.

### Corrections
*None — first exploration in this plan.*

### Exploration Confidence
- **scope**: deep — full re-read of video_jepa model.py (673 LOC), config.py (275 LOC), encoder.py (222 LOC), masking.py (158 LOC); full read of sigreg.py (281 LOC), convnext_v1_block (442 LOC), convnext_v2_block (465 LOC), GRN (285 LOC), vae model.py (998 LOC), Sampling layer (218 LOC), MAE model (349 LOC), LeWM model (360 LOC); spot-read of ConvNeXtV2 head + jepa encoder + DINO loss + VQ-VAE head; grep-verified SIGReg consumers (5 files) + anti-collapse precedent absence.
- **solutions**: constrained — single architecture (4D-latent per-patch VAE), single backbone (ConvNeXtV2 block stack, flat), single loss stack (recon + beta*KL + lambda*SIGReg).
- **risks**: clear — falsification signals testable on a single CIFAR-10 smoke (recon MSE, KL/patch, SIGReg loss trajectory, latent variance across patch axis). STOP-IF triggers in plan.

## plan_2026-05-25_853605c1
### Index
| # | Topic | File | Summary |
|---|-------|------|---------|
| 1 | Current state of sigreg.py | `findings/sigreg-current-state.md` | 183-line single class; `add_weight` in `__init__` (Golden Rule violation); `numpy` imported inside `__init__`; otherwise compliant. |
| 2 | Consumers + tests | `findings/consumers-and-tests.md` | Used by `models/lewm/model.py`; 3 tests in `tests/test_regularizers/test_sigreg.py`; ctor signature + get_config keys must stay. |
| 3 | Keras guide patterns | `findings/keras-guide-patterns.md` | §1.1 Golden Rule, §2.1 imports, §3.1 simple-layer template, §8.1 serialization — concrete checklist. |

### Key Constraints
- **HARD**: Constructor signature `(knots, num_proj, seed, name, **kwargs)` unchanged — `models/lewm/model.py` and tests depend on it.
- **HARD**: `get_config()` keys remain `{knots, num_proj, seed}` (+ base) — round-trip test asserts.
- **HARD**: Numeric behavior of `call()` unchanged — buffer values + math identical.
- **HARD**: All 3 tests in `tests/test_regularizers/test_sigreg.py` must continue to pass.
- **SOFT**: Attribute names `.t / .phi / .weights_` stay (internal but stable).
- **SOFT**: Module-level docstring (math + rationale) preserved.
- **GHOST**: None identified.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*
