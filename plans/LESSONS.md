# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines -- rewrite, don't append forever.*
*Read before any PLAN state. This is institutional memory.*

## Process

- **PLAN -> PLAN cycles are normal.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE. Three-revision cycles close cleanly when each revision is logged with explicit SUPERSEDES in `decisions.md`.
- **CLOSE-rejection that produces 2-3 new concerns is the normal mode of operation.** Don't pre-emptively trim presentations -- present findings honestly.
- **Doc updates belong in the same plan as the code change they describe.** Treating README/CLAUDE.md updates as out-of-scope follow-ups means the plan is "complete" while user-visible behavior is still wrong.
- **Run the existing test suite as the FIRST step of EXPLORE when the goal is "fix issues found in a review".** All-green tests are the cheapest evidence that some review claims are wrong.
- **A reuse-review / recommendation doc is a HYPOTHESIS, not a finding -- source-read every claim before PLAN.**
- **Pre-Mortem "STOP IF X" triggers earn their cost when they fire in 1-2 plans out of N.**
- **Plan-time line-count predictions undershoot.** ~2x for sibling-class additions, 5-10x for dtype-semantics under mixed precision, ~1.7x for multi-layer flag-plumbing, ~2.3x for N CLI flags + validation. Greenfield model packages land in the +1000..+1300 code-only band.
- **Pre-existing tests can encode bugs as contracts.** When a fix to a library bug causes adjacent tests to fail, read the failing assertions before assuming regression.
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
- **When extracting a copy-pasted class, identify the MOST divergent copy first.** Build the union API to absorb it (step 1); less-divergent copies then fall out as constructor-default variations.
- **Adopt-existing of a dead local copy is behavior-neutral AND fixes latent bugs for free.**
- **Per-symbol body-usage grep is the source of truth for "is this still used?", not plan notes.**
- **"N verbatim-identical sites" findings are routinely over-optimistic; re-verify each at EXECUTE.** A dedup that silently changes a hyperparameter is a behavior-changing bug, not a refactor -- leave divergent sites local.
- **Dead code discovered mid-refactor: delete it, do not add an unused import to "adopt".**
- **Independently verify any blocking-collision claim before planning around it.** An explorer can fabricate a "destination files already exist" finding; direct `ls`/`diff`/`git ls-files` is the authoritative check. Do NOT plan around an unverified collision.
- **`git mv <dir> <newdir>` (whole-directory form) is the clean one-command history-preserving move** for a doc folder when all files move together. Prefer it over per-file `git mv` when the entire directory relocates intact.
- **Moving a module one package level DEEPER breaks upward relative imports by exactly one dot.** `from ..x import y` (sibling-of-parent) must become `from ...x` after `layers/foo.py` → `layers/memory/foo.py`. This is the ONLY correctness-critical edit in such a move; importers and tests using absolute paths are unaffected. Verify with a `python -c "import <new.path>"` smoke test before committing.

## Consolidation / refactoring safety

- **Re-verify a prior plan's deferred-cluster findings against HEAD before planning -- classifications go stale.**
- **Early-break-after-N scan sites are NOT replaceable by collect-all-then-cap.** Sites that `break` after 8-10 files (monitor/preview datasets) have fundamentally different semantics from full-walk-then-cap. Leave them local.
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

## CLIP / CliffordCLIP specifics

- **CliffordCLIP zero-shot eval is glue, not a build.**
- **`logit_scale` (CLIP learnable temperature) MUST be `add_weight(dtype='float32')`.**
- **CC3M loader `load_cc3m_local_split` supports an npz tokenization sidecar cache.**
- **`ContrastiveCliffordCLIP` wraps the inner model as `self.clip_model`.** Callbacks reach head LayerScale via `inner.{vision,text}_head_scale.gamma` and MUST `getattr`-guard.
- **Eval-harness preprocessing MUST match training exactly.**

## Anomaly detection / inference reuse

- **VAE per-patch KL is a clean encoder-only reuse path for anomaly detection.**
- **When a model has a learnable conditional prior, the faithful anomaly score is the conditional KL, not KL-vs-N(0,I).**
- **Resolution-agnostic patch models should pad to a multiple of the patch size, not square-resize.**
- **Keep GUI dependencies out of the importable core module.**
- **Use `mu_l1` (not sampled `z_l1`) as the `l2_prior` input at inference for reproducible, deterministic heatmaps.**

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite ~1.5h. Scope pytest to changed modules only.
- **`results/` MUST be the repo-root `results/` dir, never `src/results/`.**
- **Use `MPLBACKEND=Agg` for any training-script invocation.**
- **Use `dl_techniques.utils.logger` only -- no `print`.**
- **Single GPU jobs only.** Never spawn parallel training runs.
- **Pin GPU via shell env, not `setup_gpu(args.gpu)` alone.**
- **`env.setdefault("CUDA_VISIBLE_DEVICES", "0")` is a silent footgun.** Hard-set: `env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)`.
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
- **`Dropout(noise_shape=(None,1,1,1))` is semantically equivalent to `StochasticDepth` (per-sample Bernoulli + `/keep_prob` rescale).** `StochasticGradient` is NOT drop_path: it is forward-identity and only stochastically `stop_gradient`s the path. These are distinct regularizers. Confirmed by plan_2026-06-03_943569ad.
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
- **"Match quality like template X" is ambiguous when the template uses a different docstring dialect than the canonical instruction doc.** The instruction doc is the higher authority; "match quality" means structural completeness and polish, not dialect/idiom copy. (e.g., template uses Sphinx `:param:` but instructions mandate Google style -- keep Google style.)
- **An explorer-flagged "violation" may be a GHOST.** Cross-check against the SOFT/optional list AND the named quality template before planning a fix. A bare flag without that cross-check is unverified. (e.g., bare `@register_keras_serializable()` looks wrong but is correct when the template also omits `package=`.)
- **"Refine code to pass instructions" may be VERIFY-ONLY.** When EXPLORE finds the code already satisfies all MUST rules, the real work is the docstring merge and cosmetic polish -- not a logic rewrite. Confirm compliance BEFORE planning structural changes.

## Companion .md -> module docstring merges (established pattern, 2 plans)

The `orthogonal_butterfly` and `polar_weight_norm` plans both merged a companion `.md` into the module docstring and deleted the `.md`. Lessons consolidated:

- **grep ALL reference sites before deleting the `.md`.** Pointer sites can live inside OTHER files' docstrings (e.g. `orthogonal_butterfly.py` cross-referenced `norms/polar_weight_norm.md`), not just CLAUDE.md/README. Run `grep -rn <filename.md> src/` and fix every hit. The pointer count is often higher than expected (butterfly=1 site; polar=4 sites).
- **Match the target file's OWN section/divider convention.** Do NOT blind-copy the precedent file's exact divider width. If the target already uses 75-hyphen banner dividers, match them; do NOT import a second divider width. Drop redundant dividers the file already has (e.g. banner already present -> no additional `# ---` needed before the class).
- **Two-commit pattern**: commit 1 = merge docstring + add/adjust dividers (`.py` only); commit 2 = delete `.md` + fix all pointer sites. This keeps pointer fixes atomic with the deletion.
- **For sibling classes documented in the `.md` but living in a different file**: cross-reference, do NOT duplicate their docs. Single-source-of-truth prevents doc drift. (DRY: aspirational/duplicated docs rot.)

## GPU / environment

- **GPU0 (RTX 4090) is frequently saturated by the user's training jobs; pin ad-hoc test/eval runs to GPU1 (RTX 4070) with `CUDA_VISIBLE_DEVICES=1`.**
- **CC3M is on a spinning HDD (data0_4tb); raw-stream training is GPU-starved at ~4s/step.** Fix via RAM cache, not dataset copies to `data_fast` (repo SSD -- never stage datasets there).

## Misc

- **tf2onnx 1.16.1 cannot convert `tf.while_loop` from `keras.ops.scan`.**
- **CLI flag uniformity across sibling training scripts is an explicit design property.**
