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

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite ~1.5h. Scope pytest to changed modules only.
- **`results/` MUST be the repo-root `results/` dir, never `src/results/`.**
- **Use `MPLBACKEND=Agg` for any training-script invocation.** Headless server otherwise crashes.
- **Use `dl_techniques.utils.logger` only — no `print`.** Grep `print(` in modified files.
- **Single GPU jobs only.** Never spawn parallel training runs.
- **Pin GPU via shell env, not `setup_gpu(args.gpu)` alone.** TF initialises at `import tensorflow as tf`. Use `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train.<...>`.
- **`env.setdefault("CUDA_VISIBLE_DEVICES", "0")` is a silent footgun.** Hard-set: `env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)`. Expose `--gpu N` CLI flag as single source of truth.
- **All training callbacks that write to `save_dir` MUST `os.makedirs(..., exist_ok=True)` at the top of every save method**, not just at `__init__`.
- **`create_base_argument_parser` sets `--image-size=224` (ImageNet default).** Smoke tests on small datasets (CIFAR-10, synthetic) MUST explicitly override `img_size` in the smoke override block — otherwise the model attempts 224x224 patches which is correct for production but wrong for the tiny smoke config (plan_74f0eac9).
- **VAE Sampling layers are stochastic by design — reload checks must compare deterministic encoder mu, NOT reconstruction outputs.** `model.encode(x)[0]` (mu) is deterministic; `model(x)["reconstruction"]` draws from the posterior on every call. Using reconstruction for reload checks produces spurious `max|delta| >> 1e-4` failures even on a correctly saved/loaded model (plan_74f0eac9).
- **`jit_compile=False` required for ConvNeXtPatchVAE.** XLA tracing fails on `ops.reshape` inside `_compute_sigreg`. Do not attempt `jit_compile=True` without resolving the reshape trace first (plan_74f0eac9).
- **`model.compile(loss=None)` when all losses come from `add_loss`.** Confirmed pattern for ConvNeXtPatchVAE (as for video_jepa). Passing any explicit loss argument to `compile` causes double-counting: the `add_loss` components are already aggregated inside `train_step`; `compile(loss=fn)` would add them again (plan_74f0eac9 D-002).
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
- **`model.save("path.keras")` on a never-called subclass Model silently warns and produces a degraded archive.** Always `model(dummy_input, training=False)` to materialize sub-layer weights before save in tests.
- **When adding new tests to a file with a wall-clock budget, default `img_size` to `8` or `16`.** plan_8faec5b6 added 4 TestFactory tests to a 24s baseline; SC10 landed at 29.07s vs 30s budget. Keep test wall-clock under budget by shrinking spatial dims, not by skipping integration coverage.

## Keras 3 idioms / serialization / `training`-flag

- **`training` flag must be threaded into every private method that calls `keras.random.*`.** Also applies to wrapper sub-layers — inference-time non-determinism is a contract failure.
- **`bool(training)` vs `(training is True)` diverge under `@tf.function`.** Prefer `is True` identity check. Document tensor-True→inference short-circuit at the gate site.
- **Keras 3.8 does NOT auto-create `self.loss_tracker` on `keras.Model.__init__`.** Custom `train_step` using `add_loss` MUST explicitly create the tracker, expose via `metrics`, and `update_state(loss)` in `train_step`. Otherwise `history.history['loss']`, `EarlyStopping`, `ModelCheckpoint`, CSVLogger ALL silently report 0 forever.
- **EMA target encoder: skip entirely for VAE / non-contrastive objectives** — reconstruction forbids identity. Only for contrastive/self-supervised pretraining.
- **Weight-space L2 divergence is the canonical BYOL/MoCo EMA divergence metric.** Cold-start 0 (bitwise); healthy 0.01-0.3; >1.0 sustained = collapse.
- **SIGReg prevents rank-collapse, not time-invariance.** For VAE: KL handles posterior collapse; SIGReg on `(B, Hp*Wp, latent_dim)` view of post-reparam `z` handles per-patch rank-collapse.
- **SIGReg input contract `(..., N, D)`.** VAE binding: reshape `(B, Hp, Wp, latent_dim) -> (B, Hp*Wp, latent_dim)`. `num_proj >= 256`, `N >= 64` for stable estimate.
- **Per-patch KL averaging (NOT sum) is what makes the loss magnitude resolution-invariant.**
- **Resolution-agnostic = no GlobalAveragePooling2D, no learned absolute PE, no Dense(latent_dim) over flattened spatial map.** ConvNeXt is translation-equivariant.
- **`keras.ops.cond` traces BOTH branches under `tf.function` on TF backend Keras 3.** Multiply-by-zero is the simpler-equivalent pattern.
- **`tf.cond` inside `ds.map` has a one-batch lag on captured `tf.Variable` reads.** Use two named transforms with statically-determined ranks.
- **`ops.pad` with dynamic paddings does not trace under XLA on TF 2.18.** Substitute `ops.concatenate([x, ops.zeros(...)])`.
- **Bare `@register_keras_serializable()` ties registered key to `__module__`.** Do NOT relocate classes between modules without `package="dl_techniques"` arg.
- **Manual `child.build(input_shape)` in `parent.build()` is required for Keras 3 model-save round-trip** when children are constructed in `__init__` but never called before `model.save()`.

## CLM training metrics & probes

- **Metric math in `dl_techniques.metrics/`; per-trainer-shape metric list in `train.common.nlp`.**
- **`build_clm_metrics()` MUST return fresh metric instances on every call.**
- **Keras 3 dict-keyed metrics silently no-op on subclassed models unless `output_names` is set.** Call `prepare_dict_keyed_compile(model, output_key="logits")` before `model.compile`.
- **Verification gate for trainer metric wiring MUST include a real `model.fit` call.** grep + py_compile + shape-only unit tests are insufficient.
- **`pad_token_id` mismatch is the canonical silent bug for encoder-wrapper integrations.** tiktoken cl100k_base uses `100266`.

## Multi-optimizer / custom train_step

- **Two-optimizer differential-LR: register only ONE optimizer with `super().compile(optimizer=...)`, apply the second manually in `train_step`.**
- **`current_phase` / `_global_step`: `add_weight(trainable=False, dtype="float32")`** — int32 caused CPU/GPU device-placement failures. Do NOT serialize via `get_config`.

## Layer / model testing patterns

- **Always include a `test_save_load_roundtrip` that wraps the layer inside a `keras.Model`.** Call the model once on a dummy batch BEFORE `model.save()`.
- **For redesigned aux-losses, a non-degeneracy test is stronger than a finiteness test.**
- **"Unit test" that does `build(Model(**cfg))` + `model.fit` for 1-2 steps costs integration-test runtime (~10-15s per test)**, not unit-test runtime.

## DARTS-style differentiable-primitive layers (`layers/logic/`)

- **Default `arithmetic_op_types` (including `power`) is a footgun for stacks depth >= 3 with residuals.** Restrict to `add,max,min` or enable `exponent_clip_mode='smooth'`.
- **A `Dense(channels >= 16)` embed in front of `LearnableNeuralCircuit` is required when feeding raw bit vectors.**
- **`mlp_matched` is a brutally strong baseline on small categorical one-hot inputs.** Use as a floor, not a ceiling.

## Multi-seed sweep / stats / benchmarks

- **`keras.utils.set_random_seed(seed)` is the canonical multi-source RNG primitive.**
- **Multi-seed sweep: subprocess-per-seed beats in-process import.** Clean TF/Keras init eliminates cross-seed state contamination.
- **Headline benchmark claim "circuit beats MLP" can numerically hold yet fail statistical significance at n=5.** Reframe to "interpretable AND competitive within noise" — or boost n to ≥15-20.
- **Saturation is a validity threat in benchmarks.** When all baselines hit 1.000, the "competitive" claim is weak.

## Misc

- **Periodic training-side-effect callbacks must swallow render failures.** `try/except Exception` + `logger.warning`.
- **tf2onnx 1.16.1 cannot convert `tf.while_loop` produced by `keras.ops.scan`.** Workaround: monkeypatch `call` to a Python-unrolled `for t in range(T)`.
- **CLI flag uniformity across sibling training scripts is an explicit design property.**
- **For "delta near zero" metrics, `std > |mean|` is the expected outcome, not an alarm.**
- **At n=5, paired sign-flip permutation (B=10000, Phipson-Smyth +1/+1) is the correct non-parametric inference.**
