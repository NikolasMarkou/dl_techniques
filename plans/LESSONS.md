# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines — rewrite, don't append forever.*
*Read before any PLAN state. This is institutional memory.*

## Process

- **PLAN -> PLAN cycles are normal.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE. Three-revision cycles close cleanly when each revision is logged with explicit SUPERSEDES in `decisions.md`.
- **CLOSE-rejection that produces 2-3 new concerns is the normal mode of operation.** Don't pre-emptively trim presentations — present findings honestly.
- **Doc updates belong in the same plan as the code change they describe.** Treating README/CLAUDE.md updates as out-of-scope follow-ups means the plan is "complete" while user-visible behavior is still wrong.
- **Run the existing test suite as the FIRST step of EXPLORE when the goal is "fix issues found in a review".** All-green tests are the cheapest evidence that some review claims are wrong.
- **A reuse-review / recommendation doc is a HYPOTHESIS, not a finding — source-read every claim before PLAN.**
- **Pre-Mortem "STOP IF X" triggers earn their cost when they fire in 1-2 plans out of N.** Resolved in 1 attempt because the fallback was already prepared.
- **Plan-time line-count predictions undershoot.** ~2x for sibling-class additions, 5-10x for dtype-semantics under mixed precision, ~1.7x for multi-layer flag-plumbing, ~2.3x for N CLI flags + validation. Greenfield model packages land in the +1000..+1300 code-only band. **Update**: ×2 also applies to *test runtime* when "new unit tests" are actually integration-shaped (full model build + fit).
- **Pre-existing tests can encode bugs as contracts.** When a fix to a library bug causes adjacent tests to fail, read the failing assertions before assuming regression.
- **Verify "every site does X" with grep BEFORE PLAN, not during EXECUTE.**
- **DESIGN+SCAFFOLD plans converge in 1 iteration when paired with an operational follow-up doc.**
- **Audit-driven full-coverage plans close in one iteration when (a) audit doc carries file:line refs + prescribed fix shape, AND (b) EXPLORE pre-resolves design choices in a "design notes" finding before PLAN.**
- **`run_in_background` from a sub-agent kills the background task when the sub-agent exits.** Any task >2 min OR whose wall-clock depends on a download MUST be user-launched. Never fire-and-forget.
- **Step appended at PLAN->EXECUTE handoff inherits PLAN-time falsification-list gaps.** Any step added at PLAN approval after the falsification list was drafted MUST get its own Pre-Mortem before EXECUTE.
- **Smoke != correctness — smoke tests must assert distributional invariants across tf.data + preprocessing boundaries.**
- **A trainer's unguarded "TRAINING COMPLETED SUCCESSFULLY" log is a known footgun.** Guard on `best_val_acc >= max(2/num_classes, threshold)` AND `early_stop_epoch >= 0.5 * total_epochs`.
- **Two-`.fit()` pattern with `initial_epoch=` is the clean Keras-3 way to express a phase boundary.**
- **Diagnostic-vs-fix framing is a load-bearing decision — log it explicitly in `decisions.md` BEFORE writing any code step.**
- **Treat sibling-template invariants as GHOSTS until proven applicable.** video_jepa's EMA-target / positional-embedding / temporal-axis infrastructure are GHOSTS for VAE reconstruction. Classify constraints HARD/SOFT/GHOST in `findings.md` before PLAN.
- **A "stagnation" observed at small N may be undersampling, not pathology — verify by extending N before designing a fix.**
- **Anchored design choices survive deep-review unchanged when EXPLORE checks LESSONS first.** Re-litigating an anchored choice without new evidence wastes an iteration.
- **Audit-driven plans where bug-fix claims are pre-verified against HEAD save EXECUTE iterations.** Grep + source-read every flagged bug in the EXPLORE phase; if it's already landed in a recent commit, scope it out before PLAN. This plan dropped 6 of 6 flagged bug-fix steps after verification, leaving only the architectural change.
- **Output-dict aliases do NOT fix callsites that call `model.encode()` / `model.decode()` directly.** When adding back-compat shims, grep ALL call patterns — dict reads, encode/decode tuples, attribute accesses — not just the output dict. Forgetting this cost one smoke-run-and-fix cycle.
- **GPU-availability assumptions are environmental, not architectural.** "Batch N fits 24GB on the 4090" can fail because *another process* is using the 4090. Pre-mortem F-style mitigations (drop batch + switch GPU) earn their keep precisely when the assumption is environmental — assumption never gets tested, but the fallback path runs.
- **Multi-head model = test-fixture trap when fixture hardcodes a knob the test wants to override.** A `_make_cfg(**overrides)` helper that passes `recon_loss_type="mse"` as a kwarg AND accepts `**overrides` raises `multiple values for keyword argument` when a test passes `recon_loss_type="bce"`. Fix: merge into a dict first (`base = {...}; base.update(overrides); return Cfg(**base)`). Cost was 1 fix attempt — but the failure mode is silent until a test exercises the override path.
- **Greenfield model+tests+training LOC: +3000 prediction lands +27% high.** 19-file greenfield with multi-head architecture (LPIPS loss + 5 module files + 5 test files + training script + 2 READMEs) landed at +3809. Stay inside the +4500 STOP threshold; multi-head 2.3× multiplier from earlier LESSONS holds at the high end of this band.
- **"Skip the whole callback in mode X" is the wrong default — branch the callback instead.** Hierarchical mode in convnext_patch_vae shipped with all three viz callbacks gated behind `if config.hierarchical: skip`, leaving users with no visual diagnostics. The fix is one `hasattr(config, "fieldname")` branch per callback, ~10 lines each. Default to branching; skipping is appropriate only when the diagnostic itself is meaningless (e.g. `MaskedReconViz` when `mae_mask_ratio == 0`).
- **Parallel `to_<X>_model_config()` methods MUST both consult the model's PRESETS table.** Easy regression: add the second config builder, copy-paste the dataclass-field path, forget the PRESETS-override branch from the first. Symptom is silent: `--variant large` produces a base-sized model. Grep test: every `to_*_model_config` in a trainer should reference `<Model>.PRESETS[self.model_variant]` if its model class has a PRESETS dict.

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite ~1.5h. Scope pytest to changed modules only.
- **`results/` MUST be the repo-root `results/` dir, never `src/results/`.**
- **Use `MPLBACKEND=Agg` for any training-script invocation.** Headless server otherwise crashes.
- **Use `dl_techniques.utils.logger` only — no `print`.** Grep `print(` in modified files.
- **Single GPU jobs only.** Never spawn parallel training runs.
- **Pin GPU via shell env, not `setup_gpu(args.gpu)` alone.** TF initialises at `import tensorflow as tf`. Use `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train.<...>`.
- **`env.setdefault("CUDA_VISIBLE_DEVICES", "0")` is a silent footgun.** Hard-set: `env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)`. Expose `--gpu N` CLI flag as single source of truth.
- **All training callbacks that write to `save_dir` MUST `os.makedirs(..., exist_ok=True)` at the top of every save method**, not just at `__init__`.
- **`create_base_argument_parser` sets `--image-size=224` (ImageNet default).** Smoke tests on small datasets MUST explicitly override `img_size` in the smoke override block.
- **VAE Sampling layers are stochastic by design — reload checks must compare deterministic encoder mu, NOT reconstruction outputs.**
- **Custom `train_step` + multi-task dict labels needs an input-extractor helper.** `model.fit(ds)` where `ds` emits `(dict_input, target)` tuples will pass `dict_input` to `train_step.call` directly — but if the dataset emits `(image_tensor, label_dict)` instead, labels are stranded in the second tuple element and the cls/seg loss path receives `None`. Solution: a small `_extract_call_input(data)` helper that detects `(image, {"label_X":...})` tuples and re-merges into a `{"image": image, "label_X": ...}` dict so `call()` sees them. Applies to ConvNeXtPatchVAEV2 and any future multi-task model that accepts dict inputs (plan_2026-05-27_4a444b14/D-006 era).
- **SimMIM-style masking — NOT canonical MAE — for ConvNeXt-V2 backbones.** Canonical MAE drops masked patches from the encoder input and runs a variable-length transformer over the visible set. ConvNeXt 7×7 depthwise convs require a full `(Hp, Wp)` grid, so we apply mask post-stem (replace patches with a learnable mask token) and run the full grid through the block stack. This is the FCMAE recipe from ConvNeXt-V2. The recon-loss weighting splits cleanly into (visible: scale 1.0) + (masked: scale λ_mae) — masked-only weighting is what ConvNeXt-V2 paper does, but iter-1 ships both so the V1-equivalent path with λ_mae=0 stays valid.
- **LPIPS-flavored loss is the canonical perceptual add for VAE pretraining.** ImageNet-pretrained VGG16 block features + per-channel L2 normalization + per-block weighted L1 distance. Repo already has `VGGLoss` (MSE on un-normalized VGG features); LPIPS uses **channel-normalized** features which is semantically different. We ship both. Lazy-build VGG inside the loss so deserialization stays fast; weights are not saved into the model archive (`from_config` rebuilds on first call).
- **`jit_compile=False` required for ConvNeXtPatchVAE.** XLA tracing fails on `ops.reshape` inside `_compute_sigreg`.
- **`model.compile(loss=None)` when all losses come from `add_loss`.** Confirmed pattern for ConvNeXtPatchVAE. `compile(loss=fn)` causes double-counting.
- **The ViT class defaults to `normalization_position="post"` — wrong for deep (>=12-layer) ViTs.** Deep-ViT consumers MUST pass `normalization_position="pre"`.
- **For packed CLM, use `train.common.nlp.estimate_clm_steps_per_epoch`**, not `num_articles // batch_size`.
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken.** Use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.
- **AdamW double weight decay is a footgun.** Never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)`.
- **`CliffordNetBlock` is dim-preserving.** Hierarchical CliffordNets must use explicit downsamplers.
- **CliffordNet does not tolerate aggressive patchify stems** (stride <= 2 before first block stack).
- **GPU 1 (RTX 4070 12GB) caps CliffordNet training at batch 64 for hierarchical variants.**
- **`.keras` save/load on GPU under fp32 has reduction-order noise ~5e-5 for U-Net-shaped models.** Default tolerance 1e-4.
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`.
- **`DepthwiseConv2D` on TF GPU rejects asymmetric strides.** Substitute `Conv2D(filters=channels, kernel_size=(1,k), strides=(1,s), groups=channels)`.
- **`train.common.gpu.setup_gpu` is shared infrastructure across 50+ trainers — do NOT refactor it for one trainer's observability concern.**
- **`model.save("path.keras")` on a never-called subclass Model silently warns and produces a degraded archive.** Always `model(dummy_input, training=False)` before save in tests.
- **When adding new tests to a file with a wall-clock budget, default `img_size` to `8` or `16`.**
- **For raw-filesystem JPEG datasets, use Python `glob.glob(..., recursive=True)` + `tf.data.Dataset.from_tensor_slices(files)`.** Avoids TF `list_files` `**` glob ambiguity.
- **`--steps-per-epoch` override is the clean way to validate a large-dataset pipeline without running a full epoch.**
- **`ReconVisualizationCallback` denorm guard: check `cifar_mean is not None` before applying MSE denorm.**
- **Spatial VAE latent PCA: always flatten `(B, Hp, Wp, D)` -> `(B, Hp*Wp*D)`, never mean-pool.** Mean-pooling destroys per-patch structure.
- **When class labels are unavailable (self-supervised `(x,x)` pairs), color diagnostic scatter plots by per-sample KL divergence.**
- **Duplicated `tf.data` map closures: extract via a factory `_make_<name>_fn(explicit_args)` returning the closure.** Factory captures explicit args — NOT a `config` object (not TF-graph-serializable). This is the correct Python/TF pattern (plan_2026-05-26_5abf5af3).
- **Photometric augmentation for VAE reconstruction (filesystem, `[0,1]` data):** normalize to `[0,1]` (`/ 255.0`) BEFORE any color ops — `random_saturation` requires HSV conversion on `[0,1]`, and `clip_by_value(0,1)` on `[0,255]` destroys all pixel values above 1 (black reconstructions). Correct order: decode → cast → `/255.0` → geometric aug → color aug → `clip_by_value(0,1)` → return. Ops: `random_brightness(0.2)` + `random_contrast(0.8–1.2)` + `random_saturation(0.8–1.2, img_channels==3 only)`. For CIFAR MSE (standardized, not `[0,1]`): brightness(0.1) + contrast(0.9–1.1) only — saturation requires `[0,1]`.
- **Augmentation for self-supervised `(x,x)` pairs must occur BEFORE `lambda x: (x, x)`.** Both input and target must see the same augmented image. Use a separate `augment_color` bool independent of the geometric `augment_data` gate to allow per-mode ablations.
- **When deferring preprocessing into a `.map()` closure, guard the `augment_data=False` path.** If standardization is moved inside `_augment` for ordering hygiene, the path that skips `_augment` (e.g. `lambda x: x`) will not standardize. Fix: apply standardization at numpy level when `augment_data=False`; defer only when `augment_data=True`.

## Keras 3 idioms / serialization / `training`-flag

- **`training` flag must be threaded into every private method that calls `keras.random.*`.**
- **`bool(training)` vs `(training is True)` diverge under `@tf.function`.** Prefer `is True` identity check.
- **Keras 3.8 does NOT auto-create `self.loss_tracker`.** Custom `train_step` using `add_loss` MUST explicitly create the tracker, expose via `metrics`, and `update_state(loss)` in `train_step`.
- **EMA target encoder: skip entirely for VAE / non-contrastive objectives.**
- **SIGReg prevents rank-collapse, not time-invariance.** VAE binding: reshape `(B, Hp, Wp, D) -> (B, Hp*Wp, D)`. `num_proj >= 256`, `N >= 64` for stable estimate. **`SIGRegLayer` itself is N-agnostic (normalized mean over N — no ×N inside the layer). However, the call site in `model.py` MUST multiply by `ops.cast(Hp * Wp, "float32")` so SIGReg pressure is O(N) and does not weaken 16× per resolution doubling relative to KL.** Without the ×N at the call site, effective SIGReg collapses to near-zero at 256×256 (H21). `lambda_sigreg=0.1` is calibrated to this O(N) scale — re-tune if changing `(img_size, patch_size)` significantly. Applies at EACH scale in `HierarchicalConvNeXtPatchVAE`.
- **Pure-prior sampling is broken in two-encoder hierarchical VAEs unless a learned prior `p(z_l2|z_l1)` is added.** The L2 decoder is trained on `(z_l1, z_l2)` pairs from the *same* image — they are correlated in the data distribution. Independent `N(0,I)` samples violate that, producing global/local mismatch. Coherent samples require either (a) reparameterizing from the encoder posterior around a real anchor (`sample_from(x, temperature)` — one-line, no architecture change), (b) a learnable hierarchical prior `p(z_l2|z_l1)` (NVAE/Ladder-VAE recipe, ~200-300 LOC including tests; landed in plan_c3184aea — `model.sample(num_samples)` now coherent), or (c) a post-hoc aggregate-posterior Gaussian/GMM fit.
- **Zero-init heads on a hierarchical-VAE conditional prior are load-bearing.** Both `mu_head` and `log_var_head` of `p(z_l2|z_l1)` MUST be zero-initialized (kernel + bias) so the prior emits exactly `(0, 0)` at step 0 regardless of input. This makes the conditional KL **bit-exact equal** to the legacy `KL(q || N(0,I))` at step 0 — not just within tolerance — which (a) lets old checkpoints transfer cleanly via `weight_transfer.load_weights_from_checkpoint` (new prior layer at random init, but its heads compute zero), and (b) bootstraps from-scratch training from the same operating point as the old implicit-prior model. Verified: `|delta|=0.0` across 54 scoped tests. Without this recipe the "checkpoint reuse" claim is false at step 1.
- **Closed-form Gaussian-Gaussian KL is float32-stable under `[-10, +10]` log_var clipping on BOTH q and p.** The `(mu_q - mu_p)² · exp(-lv_p)` term is the only failure surface; the clip caps `exp(-lv_p)` at ~22026. Sanity-check the formula by substituting `mu_p=0, lv_p=0` — it must reduce to `-0.5·(1 + lv_q - mu_q² - exp(lv_q))`.
- **Per-patch KL averaging (NOT sum) is what makes the loss magnitude resolution-invariant.**
- **Resolution-agnostic = no GlobalAveragePooling2D, no learned absolute PE, no Dense over flattened spatial map.**
- **`keras.ops.cond` traces BOTH branches under `@tf.function`.** Multiply-by-zero is the simpler-equivalent pattern.
- **`ops.pad` with dynamic paddings does not trace under XLA on TF 2.18.** Substitute `ops.concatenate([x, ops.zeros(...)])`.
- **Bare `@register_keras_serializable()` ties key to `__module__`.** Do NOT relocate without `package="dl_techniques"` arg.
- **Manual `child.build(input_shape)` in `parent.build()` required for Keras 3 model-save round-trip.**
- **Nested `List[List[Layer]]` breaks Keras layer-tracking on save/load.** Storing sub-layers as a list-of-lists (e.g. one inner list per stage of a hierarchical encoder) causes silent weight divergence (~1e-4 to 1e-3 on `mu`) after `model.save` + `load_model`, even when `_layers` count, layer paths, and weight count all match between original and loaded. Isolated repro: same N blocks in a flat `List[Layer]` -> 0.0 delta; in `List[List[Layer]]` -> 8e-5 delta. Fix: store layers as a single flat `self.blocks: List[Layer]` and track stage boundaries with a parallel Python `self._stage_starts: List[int]`. Iterate per-stage via `self.blocks[start:end]` slices. v2 `ConvNeXtPatchVAEV2` uses this pattern; mirror it for any hierarchical model.

## CLM / NLP training

- **Metric math in `dl_techniques.metrics/`; per-trainer metric list in `train.common.nlp`.**
- **`build_clm_metrics()` MUST return fresh metric instances on every call.**
- **Keras 3 dict-keyed metrics silently no-op on subclassed models unless `output_names` is set.**
- **Verification gate for trainer metric wiring MUST include a real `model.fit` call.**
- **`pad_token_id` mismatch is the canonical silent bug for encoder-wrapper integrations.** tiktoken cl100k_base = `100266`.

## Multi-optimizer / layer / model patterns

- **Two-optimizer differential-LR: register ONE optimizer with `super().compile(...)`, apply second manually.**
- **`current_phase` / `_global_step`: `add_weight(trainable=False, dtype="float32")`** — int32 causes device-placement failures.
- **Always include `test_save_load_roundtrip` wrapping the layer inside a `keras.Model`.**
- **Multi-seed sweep: subprocess-per-seed beats in-process import.**

## Docs (CLAUDE.md hygiene)

- **Aspirational count claims rot fast.** "150+ models / 290+ layers / 28+ losses" in lead lines drift the moment anyone adds a dir. Prefer neutral phrasing ("a comprehensive set of architectures") or omit entirely.
- **Subpackage CLAUDE.md docs are append-friction-prone.** Recent additions (e.g. `sgld_optimizer.py`) reach the package README but skip the local CLAUDE.md. Sweep periodically with `ls -la *.py` vs the doc's module list.
- **Pointer to canonical guides must live in every doc a contributor lands in.** `research/2026_keras_custom_models_instructions.md` is required reading for new models/layers — referenced from root, `dl_techniques/`, `models/`, `layers/` CLAUDE.md.

## Misc

- **Periodic training-side-effect callbacks must swallow render failures.** `try/except Exception` + `logger.warning`.
- **tf2onnx 1.16.1 cannot convert `tf.while_loop` from `keras.ops.scan`.**
- **CLI flag uniformity across sibling training scripts is an explicit design property.**
