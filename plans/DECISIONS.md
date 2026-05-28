# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed from 628 lines on 2026-05-13 (refreshed after plan_2026-05-13_a1c9a52d close — merged layers/ntm/ into layers/memory/, deleted ntm package; no new active constraints introduced, but note new constraint below). Read full content below for details on each plan's decisions.*

### Active Constraints (anchored, do-not-break)
- **3-name encoder public surface** (`<Model>`, `create_<model>`, `create_<model>_with_head`) — locked in tree_transformer/bert/cliffordnet; gpt2 is 2-name (LM head intrinsic); cliffordnet now hosts 4+3 names (multiple model classes).
- **`_download_weights` raises `NotImplementedError`** + **`from_variant` narrow `except (IOError, OSError, ValueError)`** — no silent random-init fallback. Anchored in tree_transformer, bert, gpt2, vit, cliffordnet, cliffordnet/embedding_unet.
- **`pad_token_id=<tokenizer_pad>` must be wired from trainer config to encoder ctor** (silent semantic bug otherwise). tiktoken cl100k_base pad = 100266; gpt2 enc pad differs.
- **Output dict key `"logits"`** + **`prepare_dict_keyed_compile(model, output_key="logits")`** required for every Pattern-3 CLM trainer before `model.compile`.
- **`build_clm_metrics(encoding_name, ignore_index)`** — required metric floor for every CLM trainer (replaces bare `["accuracy"]`).
- **`SegmentationWrapperLoss`** is the canonical save/load-friendly seg loss; no more `compile=False` workarounds in trainers.
- **`save_own_variables`/`load_own_variables`** on outer Model classes wrapping inner Models (DepthAnything pattern) — required for `.keras` round-trip when sub-Model weights would otherwise re-initialize.
- **memory_bank dual-optimizer**: register one optimizer with `super().compile`, apply second manually; prefix split via `name.split('/')[0].startswith(p)` (leading-component, NOT substring).
- **U-Net `.keras` round-trip tolerance is atol=1e-4** (not 1e-5) on fp32 GPU due to reduction-order noise. Applies to lmunet + embedding_unet + AccUNet.
- **`dl_techniques.layers.ntm` no longer exists** — all NTM / MANN / SOM imports go through `dl_techniques.layers.memory` (plan_2026-05-13_a1c9a52d D-002). Top-level (`NTMCell`, `NTMConfig`, `create_ntm`, `MannLayer`, `SOM2dLayer`, `SOMLayer`, `SoftSOMLayer`) and deep-submodule paths both supported.

### Failed Approaches (do NOT retry)
- "Modify `lmunet.py` in place with a `causal` flag" — REJECTED (plan_632605aa D-001); also "modify Clifford block classes with `causal` flag" — REJECTED. Sibling-stack additive file is correct.
- `keras.ops.cond` for runtime branch skipping inside `call()` — both branches trace under TF; use multiply-by-zero (plan_0f39a086 D-003).
- Mocking the database in tests / using `compile=False` to dodge a custom-loss round-trip bug — both are workarounds, not fixes (LESSONS).
- SimCSE / contrastive sentence-pair training as iter-1 for an encoder package — explicit deferral pattern (plan_632605aa D-003; plan_146ae899 — staged plans only).
- LR sweep on "smooth-train + cliff-val + sub-random val" signature — that fingerprint = data-pipeline divergence, NOT hparams (plan_f2d29729 D-006/D-007).

### Decision-Anchor Conventions
- Format: `# DECISION plan_<id>/D-NNN: <one-line>` at point of impact. Block, hash, double-dash variants supported. Unqualified `D-NNN` anchors from old plans are tolerated but WARN; new code MUST use qualified form.
- 5 triggers: failure-driven, non-obvious, rejected-alternative, constraint-workaround, 3-strike.
- Anchor at impact site (not at decision definition). One anchor per impact site, even if shared with sibling decision.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-05-27_75849a91
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-27_75849a91/D-NNN` anchor exists in source)
-->

### D-001 | PLAN | 2026-05-27
**Context**: Existing `convnext_patch_vae_v2` is flat single-scale; the user requested a cliffordnet-block version AND chose a hierarchical encoder/decoder (2-3 stages with channel doubling + spatial halving via `CliffordNetBlockDSv2`). `CliffordNetBlock` is strictly isotropic and has an internal residual via `GatedGeometricResidual`.
**Decision**: Create a NEW sibling model package `cliffordnet_patch_vae_v2/` and a NEW sibling training package `train/cliffordnet_patch_vae_v2/`. Do not refactor v2.
**Trade-off**: Code duplication (decoder, model wrapper, callbacks shim, train script) at the cost of zero risk to the working v2 model and trainer.
**Reasoning**: v2 is freshly committed (commits ea19175b/278f94f3/acac27fb) and the user is currently iterating on it. Hierarchical Clifford encoder is a structurally different model, not a block swap. Refactoring v2 to be block-class-configurable would require also generalising MAE/SIGReg wiring + segmentation upsample factor, polluting v2's API for a single experiment.
**Anchor-Refs**: (no in-code anchors yet — to be added during EXECUTE where applicable)

### D-002 | PLAN | 2026-05-27
**Context**: `CliffordNetBlock` adds its residual INTERNALLY via `GatedGeometricResidual` (clifford_block.py:794). The v2 ConvNeXt encoder loop adds the residual EXTERNALLY (`encoder.py:239-241`).
**Decision**: New encoder/decoder loops must NOT add an outer residual when stacking `CliffordNetBlock`s.
**Trade-off**: Departure from the v2 wiring template at the cost of correctness — adding an outer residual would double-apply the skip and break gradients.
**Reasoning**: Documented as a HARD constraint in findings; flagged as a Pre-Mortem Scenario 3 falsification signal during PLAN.
**Anchor-Refs**: encoder.py (to be created, Step 2 — anchor with `# DECISION plan_2026-05-27_75849a91/D-002` at the block-stack loop site)

### D-003 | PLAN | 2026-05-27
**Context**: v2 `SegmentationHead` hardcodes `UpSampling2D(size=(patch_size, patch_size))`. In the hierarchical model the encoder downsamples by an additional factor of `2^(num_stages-1)` after the stem, so the head input is at `(Hp / 2^(N-1), Wp / 2^(N-1))` and the head needs an upsample of `patch_size * 2^(N-1)` to reach full image resolution.
**Decision**: Introduce a new `CliffordSegmentationHead` with an explicit `upsample_factor: int` arg instead of reusing v2's `SegmentationHead`. `AttentionPoolClassifierHead` is reused unmodified.
**Trade-off**: One new tiny Keras class at the cost of zero edits to v2's heads.py.
**Reasoning**: The v2 seg head is the simpler of the two heads and the upsample factor is the only thing that changes; introducing a parallel class is cheaper than parameterising and re-validating v2.
**Anchor-Refs**: heads.py (Step 4)

### D-004 | EXECUTE-Step-7 | 2026-05-27
**Context**: Step 7 save/load round-trip smoke produced ~1e-4 weight delta on 44 of 143 weights (all inside CliffordNetBlock sub-layers in the encoder/decoder block stack). v2's flat `self.blocks` pattern is bit-exact; standalone CliffordNetBlock save/load is also bit-exact. Isolated repro: storing sub-layers as nested `List[List[Layer]]` (one inner list per stage) breaks Keras's layer tracking during save/load — flat `List[Layer]` works.
**Decision**: Refactor encoder.py and decoder.py to store all blocks in a single flat `self.blocks: List[CliffordNetBlock]`, with a parallel Python `self._stage_starts: List[int]` giving the index in `self.blocks` of the first block of each stage. Iteration in `call` slices `self.blocks[start:end]` per stage.
**Trade-off**: ~10 lines of bookkeeping in `__init__` / `build` / `call` at the cost of correct save/load (the only correctness issue we had).
**Reasoning**: Empirical isolation reproduced the bug with a minimal 4-block, 2-stage list-of-lists CustomLayer (8e-5 delta) vs the same 4 blocks flat (0.0 delta). Confirms Keras's nested-list Layer tracker is the culprit, not anything Clifford-specific. Same pattern that v2 uses.
**Anchor-Refs**: encoder.py (flat `self.blocks`, `self._stage_starts`); decoder.py (same).

## plan_2026-05-27_4a444b14
### D-001 — Chosen approach: design+scaffold in iter-1, no actual training runs
**Decision**: Ship V2 as a complete code package (model + losses + training script + smoke tests) in iter-1. Full training runs are explicit follow-ups, gated on user approval after iter-1.

**Trade-off**: Comprehensive design+scaffold at the cost of leaving actual training (T1a-style ablation runs) unscheduled. The user gets a runnable framework but no validated training results in iter-1.

**Why**: Per LESSONS — "DESIGN+SCAFFOLD plans converge in 1 iteration when paired with an operational follow-up doc" and "Smoke != correctness". A 24h+ ADE20K training run is unjustified before unit tests pass and the user has reviewed the code.

### D-002 — LPIPS uses VGG16 + lazy init, NOT FeatureAlignmentLoss
**Decision**: Author a new `LPIPSLoss` in `losses/lpips_loss.py` rather than reusing the existing `FeatureAlignmentLoss` (margin-cosine for distillation) or extending `VGGLoss` (image_restoration_loss.py).

**Trade-off**: A new loss class at the cost of a small overlap with `VGGLoss`. Gained: layer-weighted per-channel L2-normalized distance (LPIPS-flavored) with explicit per-block weights and input-range awareness — semantically different from MSE-on-VGG-features.

**Why**: LPIPS-flavored perceptual loss is the canonical add for VAE training (SD-VAE, Kandinsky, etc.). `VGGLoss` is closer to "perceptual MSE"; LPIPS uses normalized features + learned (or chosen) per-channel weights. We ship a clean "LPIPS-lite" without the official LPIPS weight download.

### D-003 — SimMIM-style MAE masking (not canonical MAE)
**Decision**: Implement MAE-style pretext as **SimMIM-style** (mask ratio applied post-stem, mask token replaces masked patches, full grid passes through ConvNeXt blocks) NOT canonical MAE (variable-length sequence of visible patches only).

**Trade-off**: Less compute savings than canonical MAE at the cost of preserving ConvNeXt's grid-based assumption. ConvNeXt blocks rely on full spatial grid for 7×7 depthwise convs to operate — canonical MAE would require a major surgery.

**Why**: ConvNeXt-V2's FCMAE paper uses exactly this recipe. Matches our backbone family. Anchored at the mask application site in encoder.py.

### D-004 — V2 is a separate package, V1 untouched
**Decision**: New package `models/convnext_patch_vae_v2/` and `train/convnext_patch_vae_v2/`. V1 (`models/convnext_patch_vae/` and `train/convnext_patch_vae/`) is unchanged.

**Trade-off**: Some code duplication (decoder re-export, viz callback ports) at the cost of preserving V1's test contract and avoiding any risk to V1 consumers.

**Why**: V1 ships a documented contract (32-line README intro, 8-test suite passing). A breaking change to V1's encoder.call signature (returning 3-tuple instead of 2-tuple) would surface in 50+ V1 tests + downstream training scripts. Isolation is cheap (re-exports), regression risk avoided.

### D-005 — iter-1 scope explicitly excludes hierarchical V2, xxl preset, distillation head, GAN loss, CLIP head, DINO loss
**Decision**: iter-1 lands single-scale V2 with VAE+LPIPS+MAE+cls+seg + xl preset. Hierarchical V2 and xxl are explicit follow-ups.

**Trade-off**: Smaller scope at the cost of not delivering the full 5-head + 2-scale + 5-variant matrix in one plan. The plan-time line count is already ~3000 LOC; a hierarchical V2 + xxl would push us through the "STOP IF >4500" trigger.

**Why**: Per LESSONS line-count multiplier (greenfield × multi-head 2.3×), iter-1's scope is already at the upper bound. Decomposing into iter-1 (single-scale, all heads) + iter-2 (hierarchical V2 inheriting iter-1 patterns) is cleaner than one mega-plan with high iteration-5 risk.

### D-006 — Multi-task data: cls only on CIFAR, seg head unit-tested only in iter-1
**Decision**: iter-1's training script wires CIFAR labels for the cls head. The seg head is implemented and unit-tested (synthetic labels) but not wired to real ADE20K seg masks. ADE20K-seg data loader integration is an explicit follow-up.

**Trade-off**: Defers real seg training at the cost of not validating end-to-end on real seg labels.

**Why**: ADE20K seg requires a separate label loader (annotation files, palette decoding, label format). Scope cost vs. benefit: the seg head architecture is the interesting part; the data loader is mechanical and uncoupled from the architecture decisions. Keep iter-1 lean; document the missing piece.

## plan_2026-05-27_84f6180d
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-27_84f6180d/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-27
**Context**: 17 CLAUDE.md files; only `src/train/CLAUDE.md` (451) breaks the 400-line cap. Inflated count claims (150+ models / 290+ layers) appear in 4 root-ward docs while actual counts are ~75 models / 231 layer files. Several subpackage docs are missing recently added modules (notably `sgld_optimizer.py` from commits b23f769e/9342eaec/70deb5e9).
**Decision**: Audit-and-correct in place. Edit the 9 inconsistent CLAUDE.md files; leave the 9 accurate ones (analyzer, callbacks, constraints, initializers, metrics, regularizers, visualization, models/ccnets, train/ccnets) untouched.
**Trade-off**: surgical accuracy edits **at the cost of** not normalizing prose style across all docs.
**Reasoning**: Touching the accurate docs would risk introducing new drift while solving nothing. Scope is bounded by the user's two explicit requirements (consistency + 400-line cap).

### D-002 | PLAN → EXECUTE | 2026-05-27
**Context**: User reviewed plan v1 and selected (a) drop inflated counts entirely instead of replacing with real numbers, (b) skip step 9 (trimming `src/train/CLAUDE.md` from 451 → ≤400).
**Decision**: Execute steps 1-8 only. Replace "150+ models / 290+ layers" wording with neutral phrasing ("a comprehensive set of architectures and custom layers"). Leave `src/train/CLAUDE.md` at 451 lines per explicit user choice.
**Trade-off**: User-respected scope **at the cost of** leaving one file over the 400-line cap. The cap was a user-stated requirement that the user themselves elected to waive for this one file.
**Reasoning**: Dropping numbers prevents future drift from re-counting. The user is the requirement source; explicit deferral on step 9 overrides the global cap.

## plan_2026-05-27_68c7fcd6
### D-001 | EXPLORE → PLAN | 2026-05-27
**Context**: Mirror Muon's pattern; SGLD is much simpler (stateless). Reference PyTorch snippet uses `sqrt(lr)` but canonical SGLD (Welling & Teh 2011) and the article's stated formula specify `sqrt(2·lr)`.
**Decision**: Implement canonical SGLD with `sqrt(2·lr)·noise_scale·ε`. Document deviation from snippet.
**Trade-off**: Mathematical correctness **at the cost of** byte-for-byte equivalence with prompt's snippet.
**Reasoning**: Snippet contradicts its own article; we follow the published formula.

### D-002 | PLAN | 2026-05-27
**Context**: SGLD has no per-variable buffers.
**Decision**: Override `build()` minimally (super + seed generator only).
**Trade-off**: Minimal memory **at the cost of** no preconditioner extension hook.
**Reasoning**: KISS; preconditioned SGLD would be a separate subclass.

### D-003 | PLAN | 2026-05-27
**Context**: `SeedGenerator` not trivially serializable.
**Decision**: Store integer `seed` attribute; build `keras.random.SeedGenerator(self.seed)` lazily in `build()`. Include `seed` in `get_config`.
**Trade-off**: Hyperparameter reproducibility **at the cost of** cross-restore exact sample-stream continuity.
**Reasoning**: Standard Keras pattern (Dropout, RandomFlip). Optimizer state continuity across save is not a guarantee.
