# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines -- rewrite, don't append forever.*
*Read before any PLAN state. This is institutional memory.*

## Process

- **PLAN -> PLAN cycles are normal.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE. Three-revision cycles close cleanly when each revision is logged with explicit SUPERSEDES in `decisions.md`.
- **CLOSE-rejection that produces 2-3 new concerns is the normal mode of operation.** Don't pre-emptively trim presentations -- present findings honestly.
- **Doc updates belong in the same plan as the code change they describe.** Treating README/CLAUDE.md updates as out-of-scope follow-ups means the plan is "complete" while user-visible behavior is still wrong.
- **Run the existing test suite as the FIRST step of EXPLORE when the goal is "fix issues found in a review".** All-green tests are the cheapest evidence that some review claims are wrong.
- **A reuse-review / recommendation doc is a HYPOTHESIS, not a finding -- source-read every claim before PLAN.**
- **Theory-spec-vs-code compliance audits: parallel EXPLORE produces HYPOTHESES; orchestrator must source-read the subtlest claims before PLAN.** Here the "ERG det=1" discrepancy was downgraded to a GHOST after reading `rescale_eigenvalues` -- trace-normalization IS the SETOL §10.2 sanctioned wscale step; only the `abs()` on Δλ_min was the real ERG bug.
- **Pre-Mortem "STOP IF X" triggers earn their cost when they fire in 1-2 plans out of N.**
- **Plan-time line-count predictions undershoot.** ~2x for sibling-class additions, 5-10x for dtype-semantics under mixed precision, ~1.7x for multi-layer flag-plumbing, ~2.3x for N CLI flags + validation. Greenfield model packages land in the +1000..+1300 code-only band.
- **Pre-existing tests can encode bugs as contracts.** When a fix to a library bug causes adjacent tests to fail, read the failing assertions before assuming regression. Deliberate contract rewrites (not regressions) must be flagged explicitly in decisions.md and the plan step.
- **Verify "every site does X" with grep BEFORE PLAN, not during EXECUTE.**
- **DESIGN+SCAFFOLD plans converge in 1 iteration when paired with an operational follow-up doc.**
- **Audit-driven full-coverage plans close in one iteration when (a) audit doc carries file:line refs + prescribed fix shape, AND (b) EXPLORE pre-resolves all design choices in a "design notes" finding before PLAN.**
- **`run_in_background` from a sub-agent kills the background task when the sub-agent exits.** Any task >2 min OR whose wall-clock depends on a download MUST be user-launched.
- **Step appended at PLAN->EXECUTE handoff inherits PLAN-time falsification-list gaps.** Any step added at PLAN approval after the falsification list was drafted MUST get its own Pre-Mortem before EXECUTE.
- **Smoke != correctness -- smoke tests must assert distributional invariants across tf.data + preprocessing boundaries.**
- **A trainer's unguarded "TRAINING COMPLETED SUCCESSFULLY" log is a known footgun.** Guard on `best_val_acc >= max(2/num_classes, threshold)` AND `early_stop_epoch >= 0.5 * total_epochs`.
- **Two-`.fit()` pattern with `initial_epoch=` is the clean Keras-3 way to express a phase boundary.**
- **Diagnostic-vs-fix framing is a load-bearing decision -- log it explicitly in `decisions.md` BEFORE writing any code step.**
- **Treat sibling-template invariants as GHOSTS until proven applicable.**
- **A "stagnation" observed at small N may be undersampling, not pathology -- verify by extending N before designing a fix.**
- **Anchored design choices survive deep-review unchanged when EXPLORE checks LESSONS first.**
- **Output-dict aliases do NOT fix callsites that call `model.encode()` / `model.decode()` directly.** When adding back-compat shims, grep ALL call patterns.
- **Multi-head model = test-fixture trap when fixture hardcodes a knob the test wants to override.**
- **`validate-plan.mjs` is repo-wide.** It ERRORs on legacy bare `D-NNN` anchors from ALL prior plans, not just the active one. A clean active plan can show 50 inherited ERRORs. Triage by file: errors in untouched files are pre-existing debt, addressed via `bootstrap.mjs retire <plan-id>` or anchor re-qualification as a SEPARATE plan. **Distinguish introduced-vs-inherited debt** -- the gate before CLOSE is zero ERRORs INTRODUCED by the current plan, not zero repo-wide.
- **In-code anchor IDs MUST be numeric D-NNN matching decisions.md.** Finding-report IDs (D-A, D-E style) are not valid anchor IDs -- validate-plan.mjs emits ERROR [anchor-orphan] for them and blocks CLOSE. When an executor writes finding-IDs as anchors, add the corresponding D-NNN entries to decisions.md and repoint the anchors before REFLECT.
- **When extracting a copy-pasted class, identify the MOST divergent copy first.** Build the union API to absorb it (step 1); less-divergent copies then fall out as constructor-default variations.
- **Adopt-existing of a dead local copy is behavior-neutral AND fixes latent bugs for free.**
- **Per-symbol body-usage grep is the source of truth for "is this still used?", not plan notes.**
- **"N verbatim-identical sites" findings are routinely over-optimistic; re-verify each at EXECUTE.** A dedup that silently changes a hyperparameter is a behavior-changing bug, not a refactor -- leave divergent sites local.
- **Dead code discovered mid-refactor: delete it, do not add an unused import to "adopt".**
- **Independently verify any blocking-collision claim before planning around it.** An explorer can fabricate a "destination files already exist" finding; direct `ls`/`diff`/`git ls-files` is the authoritative check. Do NOT plan around an unverified collision.
- **`git mv <dir> <newdir>` (whole-directory form) is the clean one-command history-preserving move** for a doc folder when all files move together. Prefer it over per-file `git mv` when the entire directory relocates intact.
- **Moving a module one package level DEEPER breaks upward relative imports by exactly one dot.** `from ..x import y` (sibling-of-parent) must become `from ...x` after `layers/foo.py` -> `layers/memory/foo.py`. Verify with a `python -c "import <new.path>"` smoke test before committing.
- **Running the experiment is part of verifying it.** Static verification (AST, --help, grep, scope guard) passes while the trained model is broken. A from_logits/head-activation mismatch only shows when you actually train: init loss ~9.5 instead of ln(10)=2.30, val_acc pinned at random.
- **A metric naming convention choice (e.g. which of two spec-sanctioned normalizations a key exposes) is a project-intent decision requiring the user.** SETOL §10.2 sanctions both un-normalized σ² and /N; which one `MetricNames.ALPHA_HAT` exposes is not a correctness bug -- ask the user before implementing.
- **Source-compare against a reference implementation, not only against a spec doc.** Two pre-existing bugs (R1: MP edge factor `(1+√Q)²` vs `(1+1/√Q)²` — factor-4 error for Q=4; R2: `searchsorted` on a non-monotonic `cumsum(log(...))` array → undefined result) were invisible to a spec-only review and were only caught by reading WW's `calc_lambda_plus/minus` and `detX_constraint` source verbatim.
- **decisions.md `## D-NNN` headers MUST be exactly 3 segments: `## D-NNN | <context> | YYYY-MM-DD`.** A 4th `| <title>` segment causes the anchor validator's header regex to match zero entries → every qualified anchor in source reports "anchor-unknown-plan" (all anchors appear broken). Silent footgun; introduced by the plan-writer, fixed at REFLECT.
- **A plan can legitimately REVERSE a closed prior plan's decisions** when a new authoritative source contradicts them. Log each reversal explicitly with `REVERSES prior D-X` in decisions.md. The test suite must deliberately rewrite the old contracts; document the rewrite as a contract-reversal (not a regression).
- **WeightWatcher canonical facts (authoritative reference for the `analyzer/` subsystem):** `alpha_weighted` is the canonical metric name (NOT `alpha_hat`; WW has no `/N` variant). MP edge = `σ²(1+1/√Q)²` (Q=N/M, N=larger dim). TW threshold = `bulk_max + √[(1/√Q)·bulk_max^(2/3)·M^(-2/3)]`. ERG tail boundary = WW's descending-product loop `prod(evals[idx:])<1` (already in `compute_detX_constraint`; reuse it). WW phases: over-trained(<2)/good([2,6])/under-trained(>6), no "ideal" band. `rescale_eigenvalues` (Σλ→N trace-norm) is the §10.2 wscale step — det=1 is checked on the rescaled evals, NOT targeted by the rescale.
- **In-code anchors for regions rewritten by the current plan must be re-anchored to the current plan.** If a step replaces code carrying a prior-plan anchor (e.g., `9e82787d/D-007`), replace the anchor too with the current plan's `bc986e52/D-NNN`. Do not leave a stale prior-plan anchor on new code.
- **Architecture-first migration ordering**: when migrating model classes OUT of train scripts into a model package, migrate into the library BEFORE renaming/deleting the train scripts that import them. Renaming first leaves broken cross-imports; migrating first makes renames safe.
- **Factory decoupling pattern**: a factory that took a train-side `ExperimentConfig` MUST be decoupled to `ModelConfig` + explicit training kwargs (defaults identical to the old `TrainingConfig`) when migrating into a model package — otherwise it creates a forbidden `models -> train` import edge. This pattern recurs every time architecture leaves a train script.
- **Sanctioned vs forbidden train-to-train edges**: architecture/model class imports across train scripts are forbidden; data-prep, eval helpers, and training-config sibling imports are acceptable when that code is genuinely training-side (matches the convnext `run_stochastic_comparison.py` pattern). Distinguish the two before planning.
- **Custom training loop deviation**: `train.common.create_callbacks()` cannot wrap a non-`model.fit()` loop. "Consolidate to convention" for such folders covers naming/docs/architecture-placement/CLI parity ONLY — not callback standardization.
- **A 1000+-line train script imported by sibling scripts is an architecture smell.** Architecture is living in the wrong layer; the fix is migration into the model package, not reorganization within the train folder.

## Consolidation / refactoring safety

- **Re-verify a prior plan's deferred-cluster findings against HEAD before planning -- classifications go stale.**
- **Early-break-after-N scan sites are NOT replaceable by collect-all-then-cap.** Sites that `break` after 8-10 files have fundamentally different semantics from full-walk-then-cap. Leave them local.
- **A "consolidation" that changes normalization constants or tokenizer is a behavior-changing bug-fix, not a dedup.**
- **Three coexisting normalization constant pairs (CLIP / CIFAR10 / ImageNet) must have distinct names + an explicit distinct-from comment.**
- **For padded-vs-unpadded forward-pass divergence: lift the model forward VERBATIM into a per-caller closure.** Do NOT force one position convention into shared code.
- **A test-first token-identity harness makes a previously-"too-risky-to-dedup" callback safely extractable.** Assert token-sequence IDENTITY, not just "tokens were produced".

## git safety in EXECUTE

- **`git add -A` is a contamination footgun.** It sweeps ALL pre-existing untracked/staged files into the commit. Instruct executors to `git add <explicit paths>` ONLY. Verify with `git diff --cached --name-only` before every commit.
- **Contamination is remediable post-hoc on a local feature branch.** `git reset --mixed <base>` + selective re-stage + squash commit restores a clean diff.
- **A contaminated refactor branch is a real user problem** even if all success criteria pass, because the user pushes and reviews the diff.

## Deletion / refactoring safety

- **A whole-file Write can SILENTLY no-op when the file was modified since last read.** Mandatory gate: a repo-wide grep success-criterion run immediately after every large rewrite.
- **Before rewriting a to-be-edited file, restore from VCS.** `git checkout main -- <file>` gives a clean known baseline.
- **Removing a model variant is importer-first, delete-last.**
- **Before removing a model variant class, grep for EXACT-symbol importers** across `src/applications`, `src/train`, and `tests`.
- **Don't assert an asset is missing without globbing the actual directory.**

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite ~1.5h. Scope pytest to changed modules only.
- **`results/` MUST be the repo-root `results/` dir, never `src/results/`.**
- **Use `MPLBACKEND=Agg` for any training-script invocation.**
- **Use `dl_techniques.utils.logger` only -- no `print`.**
- **Single GPU jobs only.** Never spawn parallel training runs.
- **Pin GPU via shell env, not `setup_gpu(args.gpu)` alone.**
- **`env.setdefault("CUDA_VISIBLE_DEVICES", "0")` is a silent footgun.** Hard-set: `env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)`.
- **An experiment driver that imports TF must not share the trainer's GPU.** Force it CPU-only (`CUDA_VISIBLE_DEVICES=''` at module top before the TF-triggering import) or its GPU context fragments the trainer's XLA allocator -> SIGABRT (`Check failed: h != kInvalidChunkHandle`). The driver process needs no GPU (orchestration + pandas/matplotlib only).
- **ConvNeXt V1/V2 heads emit raw logits (bare Dense, no softmax).** Trainers must compile with `from_logits=True`. v1 had `from_logits=False` (latent bug); v2 was correct. A model can pass build/serialization/smoke tests and still be mis-compiled by its trainer.
- **ConvNeXt 4-stage variants couple the downsample stride to the stem `--strides`.** On 32x32 CIFAR inputs, use `--strides 2` or the spatial dims collapse to negative values and crash (stride-4 stem + 4x downsamples on 32x32 -> 0 or negative).
- **Per-epoch WeightWatcher/ModelAnalyzer is ~80s/epoch** on a 28M model. Make it throttleable for sweeps (`--no-epoch-analyzer`; `include_analyzer=False` propagated to callback); end-of-training analysis is unaffected.
- **All training callbacks that write to `save_dir` MUST `os.makedirs(..., exist_ok=True)` at the top of every save method**.
- **`create_base_argument_parser` sets `--image-size=224` (ImageNet default); convnext_v2 + power_mlp must pass explicit `dataset_choices`.**
- **VAE Sampling layers are stochastic by design -- reload checks must compare deterministic encoder mu, NOT reconstruction outputs.**
- **`jit_compile=False` required for ConvNeXtPatchVAE.**
- **`model.compile(loss=None)` when all losses come from `add_loss`.**
- **The ViT class defaults to `normalization_position="post"` -- wrong for deep (>=12-layer) ViTs.**
- **For packed CLM, use `train.common.nlp.estimate_clm_steps_per_epoch`**.
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken.**
- **AdamW double weight decay is a footgun.**
- **`CliffordNetBlock` is dim-preserving.** Hierarchical CliffordNets must use explicit downsamplers.
- **GPU 1 (RTX 4070 12GB) caps CliffordNet training at batch 64 for hierarchical variants.**
- **`.keras` save/load on GPU under fp32 has reduction-order noise ~5e-5 for U-Net-shaped models.** Default tolerance 1e-4.
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`.
- **`DepthwiseConv2D` on TF GPU rejects asymmetric strides.**
- **`train.common.gpu.setup_gpu` is shared infrastructure across 50+ trainers -- do NOT refactor it for one trainer's observability concern.**
- **`model.save("path.keras")` on a never-called subclass Model silently warns.** Always `model(dummy_input, training=False)` before save in tests.
- **When adding new tests to a file with a wall-clock budget, default `img_size` to `8` or `16`.**
- **`--steps-per-epoch` override is the clean way to validate a large-dataset pipeline without running a full epoch.**
- **`steps_per_epoch` miscounts silently corrupt the cosine-LR horizon.**
- **Periodic training-side-effect callbacks must surface failures loudly.**
- **`Dropout(noise_shape=(None,1,1,1))` is semantically equivalent to `StochasticDepth` (per-sample Bernoulli + `/keep_prob` rescale).** `StochasticGradient` is NOT drop_path: it is forward-identity and only stochastically `stop_gradient`s the path. These are distinct regularizers. Confirmed: `depth` (StochasticDepth) outperforms `gradient` (StochasticGradient) as a regularizer on CIFAR-10 (gap widens with longer runs).
- **`StochasticDepth` and `StochasticGradient` are NOT exported from `layers/__init__.py` (empty).** Import directly: `from dl_techniques.layers.stochastic_depth import StochasticDepth`.

## Keras 3 idioms / serialization / `training`-flag

- **`training` flag must be threaded into every private method that calls `keras.random.*`.**
- **`bool(training)` vs `(training is True)` diverge under `@tf.function`.** Prefer `is True` identity check.
- **Keras 3.8 does NOT auto-create `self.loss_tracker`.**
- **EMA target encoder: skip entirely for VAE / non-contrastive objectives.**
- **SIGReg prevents rank-collapse, not time-invariance.**
- **Closed-form Gaussian-Gaussian KL is float32-stable under `[-10, +10]` log_var clipping on BOTH q and p.**
- **Per-patch KL averaging (NOT sum) makes loss magnitude resolution-invariant.**
- **`keras.ops.cond` traces BOTH branches under `@tf.function`.** Multiply-by-zero is the simpler-equivalent pattern.
- **`ops.pad` with dynamic paddings does not trace under XLA on TF 2.18.**
- **Bare `@register_keras_serializable()` ties key to `__module__`.** Do NOT relocate without `package="dl_techniques"` arg.
- **Manual `child.build(input_shape)` in `parent.build()` required for Keras 3 model-save round-trip.**
- **Nested `List[List[Layer]]` breaks Keras layer-tracking on save/load.**
- **Callbacks are NOT `@register_keras_serializable`.**

## CLM / NLP training

- **Metric math in `dl_techniques.metrics/`; per-trainer metric list in `train.common.nlp`.**
- **`build_clm_metrics()` MUST return fresh metric instances on every call.**
- **Keras 3 dict-keyed metrics silently no-op on subclassed models unless `output_names` is set.**
- **`pad_token_id` mismatch is the canonical silent bug for encoder-wrapper integrations.**
- **CLM-resume seed sites use `data_seed = config.seed + initial_step` AFTER the keras seed line.** Do NOT migrate these to `set_seeds`.

## Multi-optimizer / layer / model patterns

- **Two-optimizer differential-LR: register ONE optimizer with `super().compile(...)`, apply second manually.**
- **`current_phase` / `_global_step`: `add_weight(trainable=False, dtype="float32")`** -- int32 causes device-placement failures.
- **Always include `test_save_load_roundtrip` wrapping the layer inside a `keras.Model`.**
- **Multi-seed sweep: subprocess-per-seed beats in-process import.**

## Docs (CLAUDE.md hygiene)

- **Aspirational count claims rot fast.** Prefer neutral phrasing or omit entirely.
- **Subpackage CLAUDE.md docs are append-friction-prone.** Sweep periodically with `ls -la *.py` vs the doc's module list.
- **Pointer to canonical guides must live in every doc a contributor lands in.**
- **"Match quality like template X" is ambiguous when the template uses a different docstring dialect than the canonical instruction doc.** The instruction doc is the higher authority; "match quality" means structural completeness and polish, not dialect/idiom copy.
- **An explorer-flagged "violation" may be a GHOST.** Cross-check against the SOFT/optional list AND the named quality template before planning a fix.
- **"Refine code to pass instructions" may be VERIFY-ONLY.** When EXPLORE finds the code already satisfies all MUST rules, the real work is the docstring merge and cosmetic polish -- not a logic rewrite. Confirm compliance BEFORE planning structural changes.

## Companion .md -> module docstring merges (established pattern, 2 plans)

- **grep ALL reference sites before deleting the `.md`.** Pointer sites can live inside OTHER files' docstrings. Run `grep -rn <filename.md> src/` and fix every hit.
- **Match the target file's OWN section/divider convention.** Do NOT blind-copy the precedent file's exact divider width.
- **Two-commit pattern**: commit 1 = merge docstring + add/adjust dividers (`.py` only); commit 2 = delete `.md` + fix all pointer sites.
- **For sibling classes documented in the `.md` but living in a different file**: cross-reference, do NOT duplicate their docs.

## GPU / environment

- **GPU0 (RTX 4090) is frequently saturated by the user's training jobs; pin ad-hoc test/eval runs to GPU1 (RTX 4070) with `CUDA_VISIBLE_DEVICES=1`.**
- **CC3M is on a spinning HDD (data0_4tb); raw-stream training is GPU-starved at ~4s/step.** Fix via RAM cache, not dataset copies to `data_fast` (repo SSD -- never stage datasets there).

## Misc

- **tf2onnx 1.16.1 cannot convert `tf.while_loop` from `keras.ops.scan`.**
- **CLI flag uniformity across sibling training scripts is an explicit design property.**
