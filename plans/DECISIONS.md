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

## plan_2026-05-26_d7a342f2
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-26_d7a342f2/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-26
**Context**: `TrainingConfig.dataset` is a single string field. ADE20K and COCO both use `_build_filesystem_dataset()` with identical `/255.0` normalization. `tf.data.Dataset.sample_from_datasets` is available in TF 2.18 and used in the codebase. The mixing approach (at file-path level vs batch level) is the main design choice.

**Decision**: Mix at file-path level using `tf.data.Dataset.sample_from_datasets(path_datasets, weights=size_proportional)` before decode/resize/batch, with a new `datasets: List[str]` field on `TrainingConfig`.

**Trade-off**: Finer-grained per-sample interleaving **at the cost of** a new `_build_mixed_filesystem_dataset()` function (~40 lines) and a new `datasets` field on `TrainingConfig`.

**Reasoning**: File-path-level mixing ensures each batch contains samples from both sources proportionally. Batch-level mixing (sample_from_datasets on batched datasets) would mean each step draws entirely from one source — less diverse. Concatenating file lists into a single shuffled pool is even simpler but loses explicit size-proportional control. Alternative "ade20k+coco" composite sentinel rejected: doesn't scale if a third dataset is added later, and requires a new hardcoded branch for each pair. `nargs="+"` `--datasets` flag is additive, doesn't break `--dataset` (singular) for all other use patterns.

## plan_2026-05-26_d8c33dca
### D-001 | EXPLORE → PLAN | 2026-05-26 [REVISED after user correction]
**Context**: The 4D spatial latent `(B, Hp, Wp, latent_dim)` must be reduced to `(B, D)` for PCA. Initial plan proposed mean-pooling over `(Hp, Wp)` for large images. User rejected this: mean-pooling destroys patch structure, which is the entire point of the spatial VAE design. For 128 samples × 16384 dims (256x256 case), sklearn's randomized SVD (Halko et al.) runs in O(N*D*k) ≈ 128*16384*2 ops — fast on CPU.
**Decision**: Always flatten `(B, Hp, Wp, D)` → `(B, Hp*Wp*D)`. Use `PCA(n_components=2, svd_solver='randomized', random_state=42)`.
**Trade-off**: Preserves full patch structure **at the cost of** slightly more memory for large images (128 × 16384 float32 ≈ 8MB — negligible).
**Reasoning**: Mean-pooling loses exactly the information that the spatial VAE is designed to capture. Randomized SVD makes the large-dim case fast without any information loss.

### D-002 | EXPLORE → PLAN | 2026-05-26
**Context**: PCA scatter plots need a color signal that doesn't require class labels (which are not available — all datasets use `(x, x)` self-supervised pairs, not `(x, y)`). Options: (a) color by sample index (meaningless but safe), (b) color by per-sample KL divergence (diagnostic — shows which samples are far from the prior), (c) single color (simplest).
**Decision**: Color by per-sample KL divergence: `-0.5 * mean(1 + log_var - mu^2 - exp(log_var))` averaged over `(Hp, Wp, D)`. The `encode()` call already returns `log_var`, so no extra forward pass needed.
**Trade-off**: Informative diagnostic signal **at the cost of** slightly more computation per callback call (two arrays processed instead of one).
**Reasoning**: KL divergence per sample is the most useful diagnostic for VAE training: it shows which samples the encoder is mapping far from the prior. A scatter colored by KL immediately reveals mode coverage, posterior collapse, and outlier samples. The computation overhead is negligible (numpy ops after the encode pass).

### D-003 | EXPLORE → PLAN | 2026-05-26
**Context**: New callbacks can live either (a) inline in `train_convnext_patch_vae.py` alongside `ReconVisualizationCallback`, or (b) in a new `src/train/convnext_patch_vae/callbacks.py`. The training script is already 991 lines; adding ~200 more lines would make it ~1200 lines.
**Decision**: Create a separate `src/train/convnext_patch_vae/callbacks.py` file.
**Trade-off**: Cleaner module separation **at the cost of** one additional file to maintain and an extra import line in the training script.
**Reasoning**: 1200-line files become hard to navigate. The new callbacks are self-contained (no coupling to training logic beyond receiving val_samples at construction). A separate file follows the depth_visualization callback pattern in `src/dl_techniques/callbacks/`. The import overhead is one line.

### D-004 | EXPLORE → PLAN | 2026-05-26
**Context**: Interpolation can use (a) linear interpolation in mu space, (b) spherical linear interpolation (slerp), or (c) interpolation in the full reparameterized z space (with noise). Standard VAE interpolation practice is mu-space linear for latent exploration.
**Decision**: Use linear interpolation between `mu_A` and `mu_B` (no slerp, no noise).
**Trade-off**: Simple and deterministic **at the cost of** not respecting the geometry of the posterior distribution (slerp would be more geometrically correct for spherical Gaussians).
**Reasoning**: For a spatial VAE with per-patch Gaussians, slerp is nontrivial to apply correctly across `(Hp, Wp, D)` dims simultaneously. Linear interpolation in mu-space is the standard first approach in the VAE literature. The results are interpretable and deterministic. Slerp can be added as a follow-up if desired.

## plan_2026-05-26_b11b0e90
### D-001 | EXPLORE → PLAN | 2026-05-26
**Context**: ADE20K (25,574 train / 2,000 val JPEGs) and COCO 2017 (118,287 train / 5,000 val JPEGs) are both available as raw filesystem directories, not TFDS archives. The canonical TFDS `tfds.load()` path cannot be used directly. Two options: (a) build a custom `tf.data.Dataset.list_files()` pipeline per dataset, or (b) wrap each in a TFDS-compatible builder and register them. The model itself is fully resolution-agnostic; only the trainer's `build_dataset()` needs extension.
**Decision**: Use raw `tf.data.Dataset.list_files()` pipeline with `decode_jpeg → resize_with_crop_or_pad → /255` normalization. One shared helper `_build_filesystem_dataset()` parameterized by glob pattern.
**Trade-off**: Simple and zero-dependency **at the cost of** no TFDS metadata (dataset size must be inferred from `glob` count or hardcoded).
**Reasoning**: TFDS builder registration is heavy infrastructure for two datasets we already have on disk. The list_files approach is what sibling trainers use when not on TFDS. Dataset size is needed only for `steps_per_epoch`, which can be computed from file count.

### D-002 | PLAN | 2026-05-26
**Context**: `img_size=128, patch_size=8` chosen for smoke tests (16×16 grid, 256 patches >> sigreg_knots=17). VRAM estimate: ~2GB at batch=8. `img_size=256, patch_size=8` gives 32×32=1024 patches and ~6GB — safe for production runs on RTX 4090.
**Decision**: Default smoke config: `--image-size 128 --patch-size 8 --preset tiny --batch-size 8`. Recommend production: `--image-size 256 --patch-size 8 --preset base --batch-size 32`.
**Trade-off**: Conservative smoke defaults catch pipeline errors quickly **at the cost of** not validating full-resolution behavior in CI.
**Reasoning**: Smoke is a correctness gate, not a performance gate. Full-resolution training is a user decision.

## plan_2026-05-25_a8325e3f
### D-001 | EXPLORE → PLAN | 2026-05-25
**Context**: Analysis session analysis_2026-05-25_abe75634 identified 7 fixes ordered by priority. Fixes 1+2 (beta annealing + warmup) prevent posterior collapse. Fixes 3+4+5 (skip connections, multi-stage decoder, perceptual loss) improve reconstruction quality after collapse is resolved. Fixes 6+7 (log_var zero-init, config default) are independent housekeeping items.
**Decision**: Implement only the non-architectural fixes in this iteration: config default, encoder log_var zero-init, trainer warmup, trainer beta annealing. Defer skip connections, multi-stage decoder, and perceptual loss to a follow-on plan.
**Trade-off**: Faster delivery + testable collapse fix **at the cost of** leaving architectural quality improvements (skip connections, seam-free decoder) for a separate plan.
**Reasoning**: Architectural changes (skip connections, multi-stage upsampling) require API changes to encoder/decoder/model and substantial test updates. Doing them before validating the collapse fix adds risk. The analysis explicitly says items 1+2 are sufficient to achieve non-collapsed training. Perceptual loss requires adding an external frozen model (VGG/ResNet) with no existing precedent in this repo.

### D-002 | PLAN | 2026-05-25
**Context**: `self._beta_kl` in `model.py:119` is a plain Python `float` assigned at construction. Two options for beta annealing: (a) make it a `tf.Variable`, (b) keep as Python float and mutate from callback.
**Decision**: Keep `self._beta_kl` as a plain Python float. Mutate from `BetaAnnealingCallback.on_epoch_begin`.
**Trade-off**: Simpler callback that avoids `tf.Variable` serialization complexity **at the cost of** not being graph-traceable (beta value is captured at trace time under `@tf.function`).
**Reasoning**: `model.call()` is never `@tf.function`-decorated in this codebase (and `jit_compile=False` is enforced). The loss `self._beta_kl * kl_loss` is evaluated eagerly. Python float mutation from `on_epoch_begin` (before any call in that epoch) is therefore correct. No need for `tf.Variable` overhead.

### D-003 | PLAN | 2026-05-25
**Context**: Encoder bottleneck is a single `Conv2D(2*latent_dim)` layer named `"bottleneck"`. Splitting into `mu_head` + `log_var_head` changes the layer names in the serialized model.
**Decision**: Split `bottleneck` into `mu_head` (Glorot init) + `log_var_head` (zeros init). Update `build()` and `call()` accordingly.
**Trade-off**: Zero-init log_var (correct inductive bias, ~70% KL reduction at step 1) **at the cost of** breaking weight compatibility with existing checkpoints trained with the old `bottleneck` layout.
**Reasoning**: No production checkpoint exists with `bottleneck` — all training runs so far produced collapsed/unusable models (that's what the analysis showed). Breaking checkpoint compatibility with collapsed-model checkpoints is zero cost. The `get_config()` serializes `latent_dim` (not layer names), so fresh save/load round-trips work correctly. Existing tests build fresh models; `TestSaveLoad` will pass with the new architecture.
**Anchor-Refs**: `src/dl_techniques/models/convnext_patch_vae/encoder.py:133-139`
