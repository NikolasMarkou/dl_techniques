# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed. Read full content below for plan-specific detail.*

### Key Findings
- **models-vs-layers audit REPLACE verdicts are hypotheses (plan_2026-06-13_ae26345d)**: 3 of 4 Tier-1 "drop-in" REPLACE verdicts in `research/2026_models_layer_reuse_audit.md` were refuted on implementation-time source read: DINOv2Block (6 structural mismatches with TransformerLayer), ByteTokenizer (different special-token attr names + 4 shared BLT callers), TRMReasoningModule (incompatible positional `call()` signature). Only `_LayerScale1D → LearnableMultiplier` was confirmed and executed. Correction addendum appended at `research/2026_models_layer_reuse_audit.md:464`. **Any future implementation of that audit MUST re-verify each finding before acting.**
- **`LearnableMultiplier` default divergence (plan_2026-06-13_ae26345d)**: `constraint='non_neg'`, `initializer='ones'` are the defaults; `_LayerScale1D` used no constraint and `Constant(1e-5)`. Both MUST be overridden explicitly (`constraint=None, initializer=Constant(1e-5)`) or the swap silently changes gradient dynamics.
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

## plan_2026-06-15_e6a0391c
### Index
| # | Finding | File | Covers |
|---|---------|------|--------|
| F1 | Catalog of all 70 model packages: entry point, minimal-build recipe, forward input shape, from_variant presence | `findings/model-catalog.md` | scope, affected files |
| F2 | Build+forward test conventions: canonical `model(inputs, training=False)` idiom, import paths, GPU1 invocation, per-type input/output handling | `findings/test-conventions.md` | existing patterns, how-to-verify |
| F3 | Tiered compliance checklist (T1 build/forward blockers vs T2 serialization hygiene) + suspect-package list with file:line of risky patterns | `findings/compliance-checklist.md` | compliance criteria, known risks |

### Key derived facts
- 70 packages. ~52 expose `from_variant` or a named factory; 18 direct-constructor; `jepa` has no top-level model (SKIP).
- Confirmed DEAD (XFAIL) per SYSTEM.md: `swin_transformer`, `hierarchical_reasoning_model`, `shgcn`, `nano_vlm_world_model`, `fftnet/SpectreHead`.
- Special-construction: `ccnets` (3-model orchestrator), `vq_vae`/`masked_autoencoder`/`masked_language_model` (need external encoder), `dino_v2` (2-input `[images,masks]`), `lewm`/`nano_vlm`/`clip` (dict input).
- `models/__init__.py` is EMPTY — no blanket import; each model imported by subpath. `tests/conftest.py` puts `src/` on path.
- No existing "run all model smoke tests" harness exists — must be built.

### Key Constraints
- [HARD] Keras 3 / TF 2.18 only. No `rfft2`/`irfft2`/`angle`/`complex` in `keras.ops`; no `DepthToSpace` (use `PixelShuffle2D`); use `keras.random.uniform` not `ops.random`.
- [HARD] `.venv/bin/python` interpreter; `CUDA_VISIBLE_DEVICES=1` for GPU; `PYTHONPATH=src` for standalone scripts.
- [HARD] `@keras.saving.register_keras_serializable()` required on every concrete custom class.
- [HARD] No `.numpy()` / `int(tensor.shape[i])` / Python `if` on tensor values inside `call()`.
- [SOFT] Smallest variant (`tiny`/`micro`/`nano`/`small`) for smoke tests; `training=False` for forward-only.
- [SOFT] PRIMARY bar = build + forward works (Tier-1). Serialization round-trip (Tier-2) is secondary.
- [GHOST] `tf.signal.*` inside `call()` locks to TF backend — but several models (fftnet) already do this and "work" under TF; not a forward blocker on this TF-only setup, only a portability concern.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-06-15_e2759fbc
### Index

| # | Finding | File | Confidence |
|---|---------|------|------------|
| F1 | **7-bug inventory for dino_v2.py** (all source-verified, ordered by where they fire). B1 `Dense` in Lambda (mask token, :617-628)→hoist to `__init__`. B2 CLS Dense-hack (:631-639)→replace with `ClassTokenPrepend`. B3 `Dense` in Lambda (register tokens, :660-677)→hoist. B4 wrong attr `self.pos_embed.pos_embed`→`.pos_embedding` (:694,697). B5 `keras.ops.cond` returns dict-vs-tensor (:800-812)→always return dict (option A). B6 `DINOv2` wrapper `cond` on dict backbone output (:1052-1061)→remove Lambda, slice `["x_norm_clstoken"]`. B7 nested `Lambda`-inside-`Lambda` + Python `assert` on symbolic tensor (:651-734)→flatten to `x = x + self.pos_embed(x)` (fixed-res). ~71 lines total, all mechanical. | findings/dino-v2-bugs.md | HIGH |
| F2 | **ClassTokenPrepend reuse (B2)**: slots in cleanly — `(B,N,D)→(B,N+1,D)`, identical contract to v1/v3; v2 masking does not change sequence length pre-CLS. Wiring: `self.cls_token_layer = ClassTokenPrepend(name="cls_token")` in `__init__`; `x = self.cls_token_layer(x)` after masking, before pos-embed. (Same layer built in plan_2026-06-15_39a31d4a, tested in plan_2026-06-15_2a23a001.) | findings/dino-v2-bugs.md | HIGH |
| F3 | **Input/output contract + smoke.** Forward needs 3 inputs `[images (B,H,W,C) f32, masks (B,num_patches) bool, is_training () bool]` — masks REQUIRED (3-input Functional model), all-False OK for smoke. Output: `DINOv2(include_top=True)`→`(B,num_classes)` logits; backbone alone→dict (5 keys). Smallest smoke: `create_dino_v2('tiny', image_size=28, patch_size=14, num_classes=10)` + `(2,28,28,3)` images + `(2,4)` zero-mask → assert `(2,10)`. Stub to fill: `tests/test_models/test_dino/test_dino_v2.py` (0-byte). Mirror v1/v3 smoke. | findings/dino-v2-bugs.md | HIGH |

### Key Constraints

### HARD
- `keras.layers.Dense` (or any weight-creating layer) must NOT be instantiated inside a `Lambda` body or inside a function traced during functional-graph construction — weights become untracked/uncreatable (B1/B3/B7). Hoist to `__init__` + build, or use a proper sub-layer.
- `keras.ops.cond` branches must return the SAME nested structure (B5/B6). The chosen resolution: backbone always returns the dict (option A) — eliminates the runtime branch (the model is Functional with a fixed `is_training` input anyway).
- Python `assert` on a symbolic tensor is a no-op at trace and never fires at runtime — remove (B7).
- Forward smoke is 3-input (`[images, masks, is_training]`); masks required.
- Do NOT touch dino v1/v3 (already fixed) or DINOHead (`.keras` round-trip break is separate, not forward-blocking). Forward-only smoke.

### SOFT
- B7 fixed-resolution simplification: drop the variable-resolution pos-embed interpolation (nested Lambdas) → `x = x + self.pos_embed(x)`. Variable-res support, if ever needed, should be a proper layer (out of scope; note it).
- Reuse `ClassTokenPrepend` (DRY) for B2 rather than a new CLS mechanism.

### GHOST
- The `is_training` runtime branch (B5/B6) is spurious in a Functional model — option A removes it entirely.

### Difficulty / decomposition
**1 iteration, ~71 lines, all mechanical** (per F1). No decomposition needed; fits one plan. Only judgment call is B5 output contract → option A (always dict) is unambiguous. Risk LOW (v1/v3 show the patterns; ClassTokenPrepend already built+tested). Expect possible first-forward cascade (v2 never run) — budget 1-2 extra, xfail if >2 beyond the 7.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*
- [SUPERSEDES plan_2026-06-15_2a23a001 recon] Prior recon called v2 "LARGE, maybe split" with ~3 bugs; full read finds 7 well-scoped mechanical bugs that fit ONE iteration. B4 (wrong pos-embed attr) and B7 (nested Lambda) were NOT in the prior recon.

## plan_2026-06-15_2a23a001
### Index

| # | Finding | File | Confidence |
|---|---------|------|------------|
| F1 | **nano_vlm tying** (SMALL). Logits at `model.py:540` (call) AND `model.py:595` (generate) via `self.output_projection(x)` — `Dense(vocab, use_bias=False)`, independent kernel `(dim,vocab)`; no tying despite the name. Fix = Option B (no repo tied-Dense exists): under the existing `use_shared_embedding` guard (`_create_output_projection:374-376`), replace the Dense call with `ops.matmul(x, ops.transpose(self.text_component.word_embeddings.embeddings))` at BOTH sites. Dim-match EXACT (`__init__:317-323` enforces vision_dim==text_dim). `output_projection` stays built (serialization) but unused on shared path. KEY RISK: `generate():595` 2nd logit site — easy to miss. | findings/nanovlm-tying.md | HIGH |
| F2 | **PFTBlock asymmetry** (SMALL, 2 lines) + **ClassTokenPrepend test** (SMALL). PFTBlock: `isinstance(input_shape, list)`→`(list, tuple)` at `progressive_focused_transformer.py:287` (build) AND `:553` (compute_output_shape); `:452` (call) already correct; no other list-only checks. pft_sr caller stays (list still valid). ClassTokenPrepend: ctor `initializer="truncated_normal"`; build `cls_token (1,1,dim)`; call→`(B,N+1,dim)`; get_config `{initializer}`. Test template: `tests/test_layers/test_embedding/test_positional_embedding.py`; 5 assertions (shape, token@0-broadcast, tokens 1..N unchanged, get_config round-trip, .keras round-trip). | findings/pftblock-and-classtoken-test.md | HIGH |
| F3 | **dino v3** (SMALL, IN SCOPE) + **dino v2** (LARGE, DEFER). v3 src already clean (double-super removed, ClassTokenPrepend at `dino_v3.py:259`); one latent bug `dino_v3.py:274` `[r.item() ...]`→`[float(r) ...]` (fires only when stochastic_depth_rate>0; fix anyway). Smoke: `create_dino_v3("small", image_size=(32,32), num_classes=10)`→`(2,10)`, mirror v1. v2 has NO super-order bug but 3+ structural bugs (`ops.cond` dict-vs-tensor `:800-812`; Dense-in-Lambda closures `:617-628`,`:660-677`; projection-hack CLS `:631-638`; nested cond `:1052-1058`; 3-input masked forward). DEFER v2 to own plan. | findings/dino-v2v3.md | HIGH |

### Key Constraints

### HARD
- nano_vlm tying must be at CALL time (matmul+transpose) — a Keras layer cannot reassign another layer's weight after build (that broke originally). BOTH logit sites (call:540, generate:595) must be patched.
- All 3 existing flipped smoke tests (nano_vlm/pft_sr/dino_v1) must stay green.
- New ClassTokenPrepend test mirrors `test_positional_embedding.py` structure.
- dino: direct submodule import (no `__init__` exports).

### SOFT
- pft_sr caller left passing a list (still valid post-fix) — minimize churn; the layer fix is the real fix.
- nano_vlm `output_projection` Dense stays created (serialization) even though unused on the shared path.

### GHOST / OUT OF SCOPE
- **dino v2 — DEFERRED to its own plan** (LARGE: 3+ structural bugs, masked 3-input forward). Do NOT attempt here.
- DINOHead `.keras` round-trip break (separate known issue, not forward-blocking).

### Scope decision (this plan = 4 SMALL fixes)
1. nano_vlm call-time weight tying (2 sites). 2. PFTBlock list-vs-tuple (2 lines). 3. ClassTokenPrepend test (new file). 4. dino v3 `.item()`→`float()` + author v3 forward smoke test. DEFERRED: dino v2 (own plan).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-06-15_39a31d4a
### Index

| # | Finding | File | Confidence |
|---|---------|------|------------|
| F1 | **nano_vlm / MultiModalFusion drift** (caller-side, but multi-site). `MultiModalFusion.__init__` (`layers/fusion/multimodal_fusion.py:129`) takes `dim` (NOT `embed_dim`) and `attention_config={'num_heads':N}` (NOT top-level `num_heads`). nano_vlm injects the GHOST kwargs at: factory dicts `model.py:727-730/741-744/755-758` (mini/base/large) + `_validate_and_prepare_configs` `model.py:339,343-344` + construction `model.py:364`. Fix = rename at all sites: `embed_dim->dim`, `num_heads->attention_config={'num_heads':N}` (`dim` auto-defaults via `:332 setdefault`). Reference correct call: `layers/heads/vlm/factory.py:121`. Cascade risk: VisionEncoder `output_mode` kwarg + TextEncoder path (moderate; TextDecoder/VisionEncoder otherwise accept embed_dim/num_heads natively). Smoke: `create_nanovlm(variant="mini", vocab_size=256)`, dict input `{images:(2,224,224,3), text_tokens:(2,16) int32}`. | findings/nanovlm-fusion.md | HIGH |
| F2 | **pft_sr / PFTBlock + depth_to_space** (SMALL, 2 fixes, one file). (1) `model.py:168` `drop_path=`→`drop_path_rate=` (PFTBlock ctor `progressive_focused_transformer.py:154` has `drop_path_rate`). (2) CASCADE confirmed: `model.py:229,246,261,286` use `keras.ops.nn.depth_to_space` (absent in 3.8) inside Lambda → replace with `PixelShuffle2D(block_size=N)` from `layers/pixel_unshuffle.py:203` (built in plan_2026-06-15_00924f53, serializable, graph-safe). Tuple-output wiring already correct (`model.py:340` unpacks). Smoke: `create_pft_sr(scale=2, variant="light")`, input `(2,32,32,3)`. | findings/pftsr-pftblock.md | HIGH |
| F3 | **dino / super().__init__ order** (MEDIUM, RISKIEST — Functional-model construction issue, NOT a simple rename). v1: `DINOv1._build_model()` called at `dino_v1.py:514` BEFORE `super().__init__()` at :518; `add_weight(cls_token)` at :541 fires pre-init → crash. Fix per explorer: move the `add_weight` out of `_build_model` into a `build()` override (keeps functional pattern symbolic) — this is a structural change, not a 1-liner. Same pattern in `DINOv2VisionTransformer` (`dino_v2.py:342`) + `DINOv2` (`:913`). v3 DISTINCT bug: double-`super().__init__` (`dino_v3.py:180` then `:211`) → delete :180. 3 files, 4 classes. Forward-blocking cascades: none beyond ctor (DINOHead `.keras` round-trip break is round-trip-only, not forward). Smoke: `create_dino_v1("small", num_classes=10, patch_size=16, input_shape=(32,32,3))`, input `(2,32,32,3)`. | findings/dino-superinit.md | HIGH |

### Key Constraints

### HARD
- Each family has an existing xfail smoke test to flip to passing: `tests/test_models/test_nanovlm/test_model.py`, `tests/test_models/test_pft_sr/test_smoke.py`, `tests/test_models/test_dino/test_dino_v1.py`. Primary oracle.
- Keras 3 rule: `super().__init__()` must be first; `add_weight` cannot fire before init (the dino crash). For a Functional model (`super().__init__(inputs=, outputs=)`), weights created during graph-build need a different home (build() or pre-super) — the dino fix is structural.
- nano_vlm `MultiModalFusion(**fusion_config)` splat means EVERY source of `fusion_config` keys must be fixed (factory dicts AND the validate/prepare injection) — missing one re-injects the ghost kwarg.
- Reuse `PixelShuffle2D` (`layers/pixel_unshuffle.py`) for pft_sr — do NOT write a new depth_to_space (DRY; it exists and is tested).

### SOFT
- dino v2/v3: same bug class; cheap to include IF the v1 fix generalizes, but adds risk + requires filling v2/v3 test stubs. Treat as optional/time-boxed (like the prior plan's step 3a).
- nano_vlm cascade (VisionEncoder output_mode / TextEncoder) may surface a 2nd bug — handle in-scope minimally (these models were never forward-run).

### GHOST
- nano_vlm's top-level `embed_dim`/`num_heads` to MultiModalFusion: a never-built API the caller was coded against.
- pft_sr `keras.ops.nn.depth_to_space`: same nonexistent-symbol class fixed last plan for darkir.

### Risk ranking (hardest-first for step order)
1. dino (structural Functional-model fix; STOP-IF can't be clean in 2 attempts → xfail-revert dino only, keep the other 2).
2. nano_vlm (multi-site rename + possible cascade).
3. pft_sr (2 small edits, PixelShuffle2D reuse).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*
