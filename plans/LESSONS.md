# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines -- rewrite, don't append forever.*
*Read before any PLAN/EXPLORE state. This is institutional memory.*

## Process

- **PLAN -> PLAN cycles are normal.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE.
- **CLOSE-rejection that produces 2-3 new concerns is the normal mode of operation.** Present findings honestly.
- **Doc updates belong in the same plan as the code change they describe.**
- **Run the existing test suite as the FIRST step of EXPLORE when the goal is "fix issues found in a review".**
- **A reuse-review / recommendation doc is a HYPOTHESIS, not a finding -- source-read every claim before PLAN.**
- **An inventory finding about a layer's `call` signature is a HYPOTHESIS; verify by reading the actual signature before relying on it.**
- **Pre-Mortem "STOP IF X" triggers earn their cost when they fire in 1-2 plans out of N.**
- **Plan-time line-count predictions undershoot.** ~2x for sibling-class additions, 5-10x for dtype-semantics under mixed precision, ~1.7x for multi-layer flag-plumbing. Greenfield model packages land in the +1000..+1300 band. A new contract + metrics + test files budget for +1000..+1400. Additive else-branches + docstrings overshoot ~2x. A new test file for a shared layer undershoots ~3x.
- **Pre-existing tests can encode bugs as contracts.** When a fix causes adjacent tests to fail, read the failing assertions before assuming regression.
- **Verify "every site does X" with grep BEFORE PLAN, not during EXECUTE.**
- **`run_in_background` from a sub-agent kills the background task when the sub-agent exits.** Any task >2 min OR whose wall-clock depends on a download MUST be orchestrator-launched.
- **For `os._exit(0)` scripts the success oracle is `"Completed."` log line + `results.json` presence, NOT exit code.**
- **Smoke != correctness -- smoke tests must assert distributional invariants across tf.data + preprocessing boundaries.**
- **An end-to-end stub `run_experiment` catches latent bugs that shape-only harnesses miss.**
- **Adversarial review (iter>=2) catches opt-in path regressions the stub tests miss.** Always exercise opt-in paths.
- **Diagnostic-vs-fix framing is a load-bearing decision -- log it explicitly in `decisions.md` BEFORE writing any code step.**
- **`validate-plan.mjs` is repo-wide.** Gate before CLOSE is zero ERRORs INTRODUCED by the current plan. Pre-existing orphan/unknown-plan errors from trimmed prior plans are NOT CLOSE blockers -- triage by plan ID.
- **In-code anchor IDs MUST be numeric D-NNN matching decisions.md.** Finding-report IDs (D-A, D-E style) are not valid anchor IDs.
- **decisions.md `## D-NNN` headers MUST be exactly 3 segments: `## D-NNN | <context> | YYYY-MM-DD`.**
- **A plan can legitimately REVERSE a closed prior plan's decisions** when a new authoritative source contradicts them.
- **Architecture-first migration ordering**: migrate into the library BEFORE renaming/deleting the train scripts that import them.
- **git archaeology (`git show <sha>:<path>`) is the cheapest scaffold source when a deleted implementation is the closest ancestor.**
- **First-ever forward-pass tests on never-tested layers surface CASCADING latent bugs.** Budget explicitly.
- **`--visualize_every_n_epochs 1` in smoke runs is required to exercise the prediction-plot savefig path.** Default cadence 5 skips plots in a 3-epoch run.
- **Explorer gap-map claims ("get_config drops param X") MUST be source-verified against the actual `__init__` signature.** Reading 20 lines is 1 tool call; implementing a spurious fix costs a full step.
- **A grep-gate scoped to a package dir catches stale README/doc residue too.**
- **Adversarial explorer CRITICALs MUST be verified against the passing test suite BEFORE planning.** False positives waste fix-work on non-bugs.
- **REFLECT regression runs MUST cover test files that NO EXECUTE step touched.** Invisible regressions only surface in files no step ran.
- **A late fail-loud guard that demands a constructor arg is only safe if ALL call sites (factory + tests + manual construction) plumb that arg.**
- **Check the repo before building "new" layers from a research doc.** Existing implementations often already cover the paper's structure.
- **A guide-conformance pass (explicit build(), compute_output_shape, from_config) is behavior-preserving.** The existing .keras round-trip suite is the regression gate. Any numeric delta from a conformance edit is a BUG -- revert.
- **When extending a config-driven block with new factory types, prefer additive `elif` branches over a dispatch-table rewrite when the dominant risk is regression.**
- **Additive sibling-class optimizer additions (new file + enum member + elif branch + constants section) land at ~2x predicted LOC due to docstrings; this is expected and not a complexity problem.**
- **Earned-abstraction rule for optimizer base classes: do not introduce a shared base until >= 2 concrete call sites need it.**
- **Maskless attention layers (fnet/anchor/lighthouse) need a `call()` branch that omits `attention_mask`.** Use a class-level `_MASKLESS_ATTENTION_TYPES` set as the contract surface.

## Layer-reuse audit methodology (plan_2026-06-13_88695f5c)

- **Build the "replace-with" inventory FIRST.** For a models/-vs-layers/ audit, cataloguing all factory registries and canonical primitive homes before classifying a single inline layer is the load-bearing prerequisite. Without it, REPLACE verdicts have no target and RELOCATE absences cannot be grep-confirmed.
- **Name-match false positives are the dominant risk.** The both-sides-read rule (read BOTH the inline layer's `call`/`build` AND the candidate's `call`/`get_config`) reliably catches them. Confirmed catches: `Gemma3TransformerBlock` dual-post-norm is not expressible via `TransformerLayer` config; `GroupAttention` tree_transformer != `group_query` factory key; `_LearnedQueryPool1D` != `SequencePooling('attention')`.
- **Sub-agent prose summaries routinely miscount verdict tallies.** The committed table rows are authoritative. Always recount verdict totals from the per-class rows at synthesis, not from the explorer's prose summary header.
- **`layers/downsample.py` / `upsample.py` are free functions, not Layer classes.** This is a real structural gap that blocks otherwise-valid REPLACE verdicts for inline `Downsample`/`Upsample` Layer subclasses. Note the gap explicitly in the report row; do not issue a REPLACE verdict pointing at a function.
- **Family-batched decomposition keeps the analyst in one architectural context per step.** For a broad cross-package audit, one batch per coherent model family (transformer, vision/conv, VLM, SSL, super-res) is preferable to per-verdict-type or flat-file batching, which forces context-switching across unrelated architectures within a single step.
- **Documentation-deliverable plans invert the net-LOC sign expectation.** Net-positive line count is correct and expected; the usual net-zero/net-negative default does NOT apply. Do not flag a positive line count at REFLECT for a doc-only plan.
- **RELOCATE absence must be grep-confirmed, not just inventory-lookup-absent.** A "no equivalent exists in layers/" verdict requires a positive grep result (only the named near-miss exists, not the target) to guard against an incomplete inventory yielding a false RELOCATE.

## Porting PyTorch DiT / VAE to Keras 3

- **Reuse first, build only the absent pieces.**
- **Functional builders (`upsample()`, graph builders) cannot be sub-layers of a subclassed Model.** Wrap in a thin `keras.layers.Layer`.
- **PyTorch dynamic scatter -> static one-hot select for XLA-safe mRoPE.**
- **Boolean SDPA keep-mask -> additive finite (-1e9) mask.**
- **The existing `TimestepEmbedding` stores sinusoidal freqs as a plain Python tensor -- breaks `.keras` round-trip.** Fix: `add_weight(trainable=False)`.
- **Abstract the giant text encoder as a precomputed input.** An 8B HuggingFace/PyTorch model cannot run in Keras.
- **"Faithful" for a PyTorch port means component-level numpy-reference parity, NOT end-to-end model output parity.**
- **Porting a model whose premise is loading pretrained checkpoints yields architecture-faithful-but-untrained components.** Set that expectation at PLAN.
- **`ops.tril` on a dynamic-shaped tensor routes through a `tf.cond` that raises `pred must not be a Python bool` during symbolic `.keras` save/load trace.** Use an arange index-comparison (`row >= col`).
- **`ScalarSinusoidalEmbedding` squeezes a trailing size-1 axis.** Pass timestep reshaped to `(B, 1)`.
- **T5 attention deliberately omits `1/sqrt(head_dim)` scaling.** Adding the scale is the single most common T5-port bug.
- **For a fixed 2D sin-cos positional embedding, implement it as a non-trainable weight (`add_weight(trainable=False)`), not via `PositionEmbeddingSine2D`.**
- **SD3 / rectified-flow sign convention: t=0 is clean data, t=1 is pure noise; `euler_step` uses signed `dt = t_next - t`.** Do NOT use `abs(dt)`.

## Keras initializers

- **Keras custom initializers are shape-driven (`__call__(shape, dtype)`) -- do NOT bake `n_in`/`n_out` into the constructor.**
- **For KAN initializers: `N` (spline basis count) MUST match `KANLinear.num_basis_fns = grid_size + spline_order` (NOT the paper's `G+k+1`).**
- **Per-call seed reproducibility requires `np.random.default_rng(seed)` inside `__call__`, NOT `keras.random.SeedGenerator`.**
- **`polar_initializer.py` is the structural template** for new initializers.
- **`FFN_REGISTRY['kan']['optional_params']` must list every new `KANLinear` ctor param.**
- **A string initializer alias that is NOT a registered Keras alias crashes in `__init__` via eager `keras.initializers.get(str)`.**
- **`git mv` + bare `@register_keras_serializable()` = serialization-safe relocation.**
- **A factory is the user-requested deliverable's contract surface -- give it its own test file.**
- **dl_techniques embedding/attention factories are construction-only (REGISTRY-dict + Literal).** Register only construction-clean layers, or leave as direct-import and document why.

## tf.data / data-pipeline determinism

- **For tf.data pipeline determinism, a trace-time Python bool `training` flag with a plain if/else keeps the training path byte-identical.**
- **Validation that reuses the training augmentation recipe gives a noisy early-stopping signal.**
- **`drop_remainder=True` on a non-repeating val pipeline silently yields zero batches when N_val < batch_size.**

## Keras 3 build-ordering (CONFIRMED FOUR-STRIKE defect class)

- **An explicit `build()` that only calls `super().build()` leaves sublayers unbuilt at Keras `.keras` load-time weight-restore.**
- **The durable fix for the lazy-sublayer build-ordering defect is to give the SHARED layer a real `build()` that explicitly builds its sublayers.**
- **A wrapper layer with no `build()` makes every parent `child.build()` call a silent no-op.**
- **A `.keras` round-trip test is the detection oracle AND the removal-safety oracle.**
- **`keras.layers.Identity()` is a serializable drop-in for `Lambda(lambda x: x)`.**
- **Manual `child.build(input_shape)` in `parent.build()` required for Keras 3 model-save round-trip.**
- **`ResnetBlock.nin_shortcut` (and similar structural conditionals) are CORRECTLY kept conditional.** The guide's "always create" rule targets RUNTIME config toggles, NOT structural channel-mismatch branches.

## Guide-conformance and repo-wide mechanical fixes

- **Before a repo-wide conformance fix, AST-scan to QUANTIFY then TRIAGE.** A large fraction of mechanically-flagged violations are legitimate exemptions.
- **`compute_output_shape` is a pure addition** -- never called in forward/train, only during functional-API shape inference. It cannot change numerics.
- **`register_keras_serializable` keys by bare class name.** Two classes with the same name registered under the same package collide at import with `ValueError`. Use `package=` to disambiguate.
- **Parallel ip-executors over DISJOINT file sets + CPU py_compile, with GPU tests run SERIALLY afterward,** respects the no-parallel-GPU rule while still parallelizing mechanical edits.
- **A module that is UNIMPORTABLE cannot be registered.** Fix import-breaking bugs before adding `@register_keras_serializable`.
- **A wrong `compute_output_shape` is worse than none.** Each added method must be verified against a real forward pass before commit.
- **Moving sublayers from `build()` to `__init__()` exposes latent attribute-name bugs hidden by construction-time deferral.**
- **Verify actual attribute/param names in source before writing fix code.**

## Multi-step forecasting

- **A model that already consumes a temporal window can be made multi-step purely by widening its OUTPUT dimension.**
- **The shared-pi diagonal-joint modeling concession is acceptable for univariate windowed forecasting (F=1) and MUST be documented in-code.**

## train/common and TS trainer infra

- **A viz-data-prep failure must NEVER abort a training run.** Wrap viz prep in a non-fatal try/except in callback `__init__`.
- **`Forecast` dataclass + `ForecastMixin` is a clean way to normalize point-vs-probabilistic model outputs WITHOUT a base-class rewrite.**
- **An anonymous `keras.Model(inputs, outputs, name="x")` functional wrapper silently breaks an `isinstance(mixin)` gate.**
- **`compile(loss=static_loss)` is broken for dict-output models.** Fix: `add_loss(...)` inside `call` + `compile(loss=None)`.
- **Deleting the non-conforming outlier beats perpetual special-casing.**

## Keras 3 ops and layer-call contracts

- **`keras.ops.normalize(x, axis=-1)` is the canonical L2-normalize idiom.** The `ops.nn.*` namespace is UNRELIABLE.
- **`.assign()` reachable from `call` is a recurring bug class** for DEAD or no-op in-call updates.
- **`create_attention_layer("multi_head")` is self-attention only.** Cross-attention requires `"multi_head_cross"`.
- **A Keras layer's `build(input_shape)` receives a LIST via the Keras functional API, a tuple via direct call, a dict for multi-input.**
- **`keras.ops.shape()` returns a TUPLE of scalar tensors on the TF backend.** Use `tf.shape(x)` + `tf.concat` for dynamic reshape targets.

## Numerical / ML algorithm stability

- **Modified-Bessel ratio `I_nu/I_{nu-1}` MUST use the continued-fraction / downward (Miller) recurrence.**
- **A vMF sphere VAE's free-learned kappa posterior-collapses to ~0 unless prevented.** Fix: high kappa init + KL warmup.
- **Early-stopping on `val_total_loss` is confounded by KL warmup.** Monitor `val_reconstruction_loss`.
- **mLSTM/matrix-memory cells REQUIRE the log-domain max-stabilizer m_t, same as sLSTM.**
- **Eager correctness and `model.fit()` correctness are distinct failure classes.** Bisect with a manual single-step `GradientTape` loop BEFORE blaming jit/XLA.

## XLA / GPU kernel compatibility

- **`keras.random.beta` lowers to `StatelessRandomGammaV3`, which has NO XLA-GPU kernel in TF 2.18.** Override `Model.compile()` to force `jit_compile=False`.
- **Always run the scoped suite on GPU -- XLA-GPU-only crashes pass silently on CPU.**

## Keras 3 / TF: implicit fields + arbitrary-scale SR (THERA)

- **Nested `List[List[Layer]]` silently breaks `.keras` reload.** Fix: flat single-level attribute list.
- **Per-pixel spatially-varying MLP ports to Keras as a batched einsum** with per-pixel weights passed as `call` INPUTS.
- **Moving sub-layer creation `build()` -> `__init__()` must preserve sub-layer order/names** or `.keras` round-trip breaks.
- **Keras 3.8 does NOT auto-create a loss tracker for a custom `train_step`.** Create explicit trackers and expose via the `metrics` property.

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite ~1.5h. Scope pytest to changed modules only.
- **`results/` MUST be the repo-root `results/` dir, never `src/results/`.**
- **Use `MPLBACKEND=Agg` for any training-script invocation.**
- **Use `dl_techniques.utils.logger` only -- no `print`.**
- **Single GPU jobs only.** Never spawn parallel training runs.
- **ConvNeXt V1/V2 heads emit raw logits.** Trainers must compile with `from_logits=True`.
- **VAE Sampling layers are stochastic by design -- reload checks must compare deterministic encoder mu.**
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`.
- **`DepthwiseConv2D` on TF GPU rejects asymmetric strides.**
- **DETR (`models/detr/`) was fixed in plan_2026-06-13_28f0b453.** Now fully tested (21 tests). `DetrDecoderLayer` deleted; use `TransformerDecoderLayer(use_causal_mask=False)` for detection decoders.

## git safety in EXECUTE

- **`git add -A` is a contamination footgun.** Use `git add <explicit paths>` ONLY.
- **Sub-agents that `git add` (even when told "no commit") cause the orchestrator's sequential commits to sweep files across step boundaries.**

## Deletion / refactoring safety

- **A whole-file Write can SILENTLY no-op when the file was modified since last read.** Gate: repo-wide grep success-criterion immediately after every large rewrite.
- **Before rewriting a to-be-edited file, restore from VCS.**
- **Removing a model variant is importer-first, delete-last.**

## Multi-optimizer / layer / model patterns

- **Two-optimizer differential-LR: register ONE optimizer with `super().compile(...)`, apply second manually.**
- **`add_weight(trainable=False, dtype="float32")`** for `current_phase` / `_global_step` -- int32 causes device-placement failures.
- **Always include `test_save_load_roundtrip` wrapping the layer inside a `keras.Model`.**

## GPU / environment

- **GPU0 (RTX 4090) is frequently saturated by the user's training jobs; pin ad-hoc test/eval runs to GPU1 (RTX 4070) with `CUDA_VISIBLE_DEVICES=1`.**
- **CC3M is on a spinning HDD (data0_4tb); raw-stream training is GPU-starved at ~4s/step.** Fix via RAM cache.
