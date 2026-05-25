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

## plan_2026-05-25_74f0eac9
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-25_74f0eac9/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-25
**Context**: Model review found 6 quality issues. 4 are low-effort genuine gaps (decoder docstrings ×2, use_v2_block dead-field comment). 2 are known repo-wide patterns with no urgency (tf.GradientTape backend coupling, optimizer.apply_gradients compat shim).
**Decision**: Fix only the 4 low-effort gaps in this plan. Skip the 2 repo-wide patterns.
**Trade-off**: Targeted small fix **at the cost of** leaving tf.GradientTape backend coupling unflagged in the source code.
**Reasoning**: tf.GradientTape is used identically in video_jepa — fixing it in one file without fixing all siblings would be inconsistent and out of scope. The risk is theoretical (JAX/PyTorch backend migration, not a near-term concern). optimizer.apply_gradients still works in TF 2.18.

### D-002 | EXPLORE → PLAN | 2026-05-25
**Context**: Training script must compile with `loss=None` (losses from `add_loss`). The VAE train_vae.py uses standard `compile(loss=...)` (incompatible). The video_jepa trainer has the right `loss=None` pattern but targets a very different architecture. The ViT trainer has the right CIFAR-10 dataset pipeline.
**Decision**: Hybrid Pattern-4 (dataset, callbacks, argparse, dataclass config) + video_jepa (compile pattern, TrainingCurvesCallback, reload check).
**Trade-off**: Hybrid approach **at the cost of** having no single verbatim reference file to copy — must synthesize from two sources.
**Reasoning**: Both halves are well-established in the repo. The hybrid is necessary because no single trainer covers both an image dataset AND a loss=None compile pattern. Synthesis risk is low (both halves are individually tested patterns).

### D-003 | EXPLORE → PLAN | 2026-05-25
**Context**: CIFAR-10 with `recon_loss_type="mse"` needs mean/std normalization (produces values outside [0,1] — compatible with MSE). With `recon_loss_type="bce"` inputs must stay in [0,1] — incompatible with mean/std normalization. The script must handle both without silently corrupting the BCE loss.
**Decision**: Default `recon_loss_type="mse"` with mean/std normalization. When `recon_loss_type="bce"`, use `/255`-only scaling (no subtract). Add explicit check in dataset builder that selects pipeline branch based on `config.recon_loss_type`.
**Trade-off**: Two dataset pipelines **at the cost of** slightly more code in the dataset builder.
**Reasoning**: Silent BCE corruption is worse than 10 extra LOC. The check is simple and statically deterministic at build time.

### D-004 | EXPLORE → PLAN | 2026-05-25
**Context**: Success guard in ViT trainer checks `val_accuracy >= threshold`. For an unsupervised VAE, there is no accuracy — the guard must use `val_loss`.
**Decision**: Guard on `val_loss <= success_threshold` (lower-is-better). Default `success_threshold=0.02` for CIFAR-10 MSE (empirical heuristic from README T1a/T1b menu; not a validated floor). Document this as advisory.
**Trade-off**: Loss-based guard **at the cost of** no validated floor (CIFAR MSE <0.02 is a heuristic, not a literature bound).
**Reasoning**: The guard is advisory — it logs a WARNING if training appears not to have converged, but does not block use of the model. For a new architecture without published convergence floors, a heuristic default with a `--success-threshold` CLI override is the right tradeoff.

## plan_2026-05-25_8faec5b6
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-25_8faec5b6/D-NNN` anchor exists in source)
-->

### D-001 | PLAN | 2026-05-25
**Context**: Repository convention for `models/{bert, resnet, tree_transformer, cliffordnet, vit, depth_anything, prism, lewm, gpt2}` is that `from_variant(pretrained=True)` must raise `NotImplementedError` from `_download_weights()` (NOT silently random-initialize). Anchored historically in plan_3c3ed037 D-001 (tree_transformer), plan_9357982a D-001 (bert), plan_0a5779e8 D-001 (tree_transformer __init__ alignment). The current `convnext_patch_vae` package lacks this surface entirely.
**Decision**: Add `_download_weights(variant) -> NotImplementedError` raise to `ConvNeXtPatchVAE`, surfaced via `from_variant(variant, pretrained=True)`. Anchor at the raise site with `# DECISION plan_2026-05-25_8faec5b6/D-001`.
**Trade-off**: Loud failure on `pretrained=True` **at the cost of** explicit caller error handling (callers must catch `NotImplementedError` if they ever opt into pretrained).
**Reasoning**: Silent random-init was identified as a footgun in plan_9357982a and locked in across the repo. Convnext_patch_vae cannot ship a divergent surface. Alternatives rejected: (a) silently load a fresh model on `pretrained=True` — invalidated by repo convention; (b) raise `ValueError` — wrong taxonomy (it's a missing implementation, not a bad value); (c) auto-download from a URL — no public checkpoints exist for this VAE.
**Anchor-Refs**: `src/dl_techniques/models/convnext_patch_vae/model.py:470` (`_download_weights` `NotImplementedError` raise site, committed in 3505b9b3).

### D-002 | PLAN | 2026-05-25
**Context**: Guide §11.1 + repo convention require a module-level `create_<model>` factory wrapping `from_variant`. Without it, users must call `ConvNeXtPatchVAE(config=ConvNeXtPatchVAEConfig(...))` directly — a two-step import chain that is inconsistent with the bert/resnet/tree_transformer/cliffordnet surface.
**Decision**: Add `create_convnext_patch_vae(variant="base", *, pretrained=False, **overrides) -> ConvNeXtPatchVAE` as a thin delegation to `ConvNeXtPatchVAE.from_variant`. Anchor at the function definition with `# DECISION plan_2026-05-25_8faec5b6/D-002`.
**Trade-off**: Two parallel entry-points (factory + ctor) **at the cost of** mild API duplication.
**Reasoning**: The duplication is documented and shallow (the factory is one line). Alternatives rejected: (a) replace `__init__(config=...)` with flat kwargs — breaks video_jepa-template parity locked in SYSTEM.md; (b) only expose `from_variant` and drop the bare factory — diverges from bert/resnet template; (c) make the factory a wrapper that also constructs the config from individual kwargs — over-engineering, the config dataclass already handles that via `ConvNeXtPatchVAEConfig(**kwargs)`.
**Anchor-Refs**: `src/dl_techniques/models/convnext_patch_vae/model.py:542` (module-level `create_convnext_patch_vae` factory anchor, committed in 585a0801).

### REFLECT | 2026-05-25
**Context**: All 7 EXECUTE steps completed without leash hits, surprises, or pivots. Final scoped pytest 19/19 PASS in 29.07s. All 12 success criteria verified PASS. Zero scope drift. 4 fb57d478 D-anchors + 2 new 8faec5b6 D-anchors all in place at expected line numbers.
**Decision**: Recommend transition to CLOSE.
**Trade-off**: Closing now ships the guide-compliant surface **at the cost of** deferring training and the additional ablation tests (T1-T4 in the README training menu) to follow-up plans — already documented as out-of-scope in plan.md.
**Reasoning**: Every plan-level invariant held; no findings were contradicted during EXECUTE; complexity budget under target (1/2 new abstractions, 0/3 new files, +306 net LOC inside the LESSONS sibling-class band). Devil's-advocate concern: SC10 came in at 29.07s — close to the 30s budget. Future additions to this test file would push over. Mitigation: 3 of the 4 new TestFactory tests use `img_size=8` or `16` to minimize wallclock; further additions should follow that pattern. Logging this for LESSONS at CLOSE.
**Anchor-Refs**: none new.

## plan_2026-05-25_fb57d478
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-25_fb57d478/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-25
**Context**: A ConvNeXt patch-level VAE with SIGReg anti-collapse is greenfield in this repo. video_jepa offers reusable patterns (config dataclass, custom train_step, per-component Mean trackers, add_loss). SIGReg's `(..., N, D)` contract maps naturally to `(B, Hp*Wp, latent_dim)` — N = patches per image. ConvNeXtV2Block + GRN are repo primitives. `Sampling` layer accepts rank-4 latents.
**Decision**: Build a new package `src/dl_techniques/models/convnext_patch_vae/` with `{config, encoder, decoder, model, __init__, README}`. Flat single-stage block stack (no spatial downsampling beyond the stem). Per-patch 4D latent `(B, Hp, Wp, latent_dim)`. Loss = recon + beta_kl * KL_per_patch + lambda_sigreg * SIGReg on `(B, Hp*Wp, latent_dim)` view of post-reparam `z`. No EMA target, no positional embedding, no temporal axis.
**Trade-off**: A new 8-file package and 3 new abstractions (over default complexity budget) **at the cost of** zero blast radius on existing models. The over-budget cost is the established repo convention for new model families.
**Reasoning**: Existing `vae/` is image-level w/ global pool (not resolution-agnostic); `masked_autoencoder/` is patch-level but deterministic. The patch-VAE niche is empty and must be built. Alternatives rejected: (a) extend `vae/` with a `resolution_agnostic=True` toggle — would double-pay the existing global-pool path's complexity; (b) reuse `video_jepa/`'s `(B, T, Hp, Wp, D)` plumbing with `T=1` — wastes temporal infrastructure and confuses the public API; (c) build under `masked_autoencoder/` — that framework's loss is masked-only recon, incompatible with VAE recon. Flat single-stage at iter-1 (not hierarchical) is the smallest experiment that falsifies the hypothesis.
**Anchor-Refs**: (will be added at EXECUTE time at `src/dl_techniques/models/convnext_patch_vae/model.py` — train_step site and SIGReg reshape site)

### D-002 | PLAN | 2026-05-25
**Context**: Two valid SIGReg bindings exist (F2 Option A "per-image, N=patches" vs Option B "per-patch-position, N=batch"). Plus Option C ("on post-reparam z" vs "on encoder grid").
**Decision**: Bind SIGReg on `ops.reshape(z, (B, Hp*Wp, latent_dim))` — post-reparameterization, per-image patch distribution (Option A applied to Option C).
**Trade-off**: Targets the literal "patch-collapse" concept in one cheap pass **at the cost of** not regularizing per-position statistics (Option B) and not regularizing the pre-reparam encoder grid.
**Reasoning**: (1) Conceptually clean — regularize the same quantity KL targets. (2) Resolution-agnostic — N scales with image size; lambda_sigreg need not be retuned per resolution. (3) Single SIGReg call per forward (cheap). Option B (per-position) discourages positional information which is undesired since we WANT spatial structure. Pre-reparam regularization would conflict with the stochastic latent.
**Anchor-Refs**: (will be added at EXECUTE Step 4 at the reshape site in `model.py`)

### D-003 | PLAN | 2026-05-25
**Context**: Training menu must be honest about which experiment falsifies the hypothesis. User goal explicitly asks "honest opinion on which experiments are worth running first".
**Decision**: Tier 1 = CIFAR-10 SIGReg-ON + SIGReg-OFF paired run (~1 hour total on RTX 4090). Tier 2-3 conditional on Tier 1. Do NOT recommend ImageNet, MNIST, hierarchical encoder, or 256+ resolution at iter-1.
**Trade-off**: Smallest falsifiable experiment first **at the cost of** delayed evidence on scale (T2/T3 cover that downstream).
**Reasoning**: CIFAR-10 32x32 patch=4 gives N=64 (above SIGReg's stability floor) with 1-hour wall-clock per run. A SIGReg-on/off pair is the exact falsification test of the claim "SIGReg on per-patch latents prevents collapse without hurting recon". Anything larger or more elaborate before this pair has run is gold-plating. MNIST is too easy — recon is trivial even with collapsed latents, so it cannot falsify.
**Anchor-Refs**: none (training-menu decision, no code anchor).

## plan_2026-05-25_853605c1
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-25_853605c1/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-25
**Context**: `sigreg.py` violates the §1.1 Golden Rule by calling `add_weight` for 3 non-trainable buffers inside `__init__`, and imports `numpy` inside that method. Behavior is otherwise correct and consumers (`models/lewm/model.py`, 3 tests) depend on a stable public surface.
**Decision**: Pure refactor — move buffer creation + numpy precompute into `build()`, hoist `import numpy as np` to module top, modernize class docstring to Google-style (Args / Input shape / Output shape / Architecture), keep math + signature + `get_config()` identical.
**Trade-off**: Conformance + maintainability **at the cost of** trivial code churn (single file) and slightly later buffer materialization (on first call instead of construction).
**Reasoning**: Smallest change that closes the conformance gap. Alternative (leaving as-is) was rejected because the user explicitly asked for guide-compliance, and the Golden Rule violation is the only material gap. Alternative (also renaming attributes / changing signature) rejected — would force changes in `models/lewm/model.py` and tests for zero correctness gain.
**Anchor-Refs**: none — refactor introduces no decision worth anchoring in source.

### D-002 | REFLECT → CLOSE | 2026-05-25
**Context**: After Step 2, all 6 success criteria PASS (3 pytest + 3 grep). No regressions, no scope drift, no debug artifacts, no simplification blockers.
**Decision**: Recommend CLOSE.
**Trade-off**: Ship the refactor now **at the cost of** not adding a buffer-bit-equality snapshot test (deemed unnecessary — behavioral tests cover what matters).
**Reasoning**: Plan executed exactly as designed. Falsification signals (AttributeError on buffers, test 2 flip, config key mismatch) all silent. The only prediction-accuracy delta is +102 LOC vs net-zero target, driven entirely by docstring expansion (Architecture diagram + Example) — not behavioral complexity. Complexity budget intent honored.
**Devil's-advocate**: One reason this could still be wrong — the layer's buffers now materialize on first `call()` rather than at construction, so any user code that introspects `layer.weights` before a forward pass would now see an empty list. Searched consumer files (`models/lewm/model.py`, the 3 tests): none do this. Safe in this repo; would be a behavior change for any hypothetical external user.
**Anchor-Refs**: none.
