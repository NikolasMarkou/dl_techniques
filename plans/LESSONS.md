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
- **Explorer audit claims are HYPOTHESES.** Source-verify every explorer claim before planning. A CONFIRMED explorer finding can still be REFUTED on direct source read (confirmed twice: DiffFFN regularizer, GatedMLP docstring math).
- **Pre-Mortem "STOP IF X" triggers earn their cost when they fire in 1-2 plans out of N.**
- **Plan-time line-count predictions undershoot.** ~2x for sibling-class additions, 5-10x for dtype-semantics under mixed precision, ~1.7x for multi-layer flag-plumbing. A new test file for a shared layer undershoots ~3x.
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
- **First-ever forward-pass tests on never-tested layers surface CASCADING latent bugs.** Budget explicitly. In confirmed instances (modern_bert_blt_hrm, ByteLatentReasoningCore, attention T1 layers, OrthoGLUFFN) a single reported bug masked real defects. Assume multi-bug chains in never-executed code.
- **`--visualize_every_n_epochs 1` in smoke runs is required to exercise the prediction-plot savefig path.** Default cadence 5 skips plots in a 3-epoch run.
- **Explorer gap-map claims ("get_config drops param X") MUST be source-verified against the actual `__init__` signature.**
- **A grep-gate scoped to a package dir catches stale README/doc residue too.**
- **Adversarial explorer CRITICALs MUST be verified against the passing test suite BEFORE planning.**
- **REFLECT regression runs MUST cover test files that NO EXECUTE step touched.**
- **A late fail-loud guard that demands a constructor arg is only safe if ALL call sites (factory + tests + manual construction) plumb that arg.**
- **Fail-loud guards require a grep-verified zero-caller check first.** log-warn+fallback is the safe choice when the caller population is non-zero.
- **Check the repo before building "new" layers from a research doc.**
- **A guide-conformance pass (explicit build(), compute_output_shape, from_config) is behavior-preserving.** Any numeric delta is a BUG -- revert.
- **When extending a config-driven block with new factory types, prefer additive `elif` branches over a dispatch-table rewrite.**
- **Earned-abstraction rule for optimizer base classes: do not introduce a shared base until >= 2 concrete call sites need it.**
- **Maskless attention layers (fnet/anchor/lighthouse) need a `call()` branch that omits `attention_mask`.** Use a class-level `_MASKLESS_ATTENTION_TYPES` set as the contract surface.
- **Tiered execution with a hard autonomy leash is load-bearing for bug-fix plans that touch untested models.**
- **Static audits miss runtime bugs.** For any untested layer, a forward + `.keras` round-trip smoke surfaces far more than static analysis.
- **`keras.ops.fft` surface is narrow.** Available: `fft/fft2/ifft2/rfft/irfft/real/imag`. NOT available: `rfft2/irfft2/angle/complex`.
- **MISATTRIBUTION risk: a prior plan's reported bug LOCATION can be wrong.** Verify by checking whether OTHER callers of the shared code work correctly.
- **Audit-doc location claims disagree with reproduction? The traceback wins.**
- **`keras.ops.stop_gradient` is a FUNCTION, not a context manager.** Use `z = ops.stop_gradient(x)`.
- **A missing `return config` in `get_config` returns None silently.** Every new-layer test MUST assert `get_config`/`from_config` directly.
- **Compute STATIC scalars (e.g. attention `scale = 1/sqrt(head_dim)`) with stdlib `math.*`, NOT `keras.ops.*`.**
- **Before adding a new precomputed scale attr, check whether the class already holds one.** In attention layers `self.scale` / `self._scale` are common; reusing them at the new call site is DRY and avoids divergence (plan_2026-06-14_33b77a7a: 4 of 6 sites were reuse, not new attrs).
- **Precompute timing is dictated by when the dimension resolves.** Init-static dims (`head_dim`, `nb_features`, `key_value_channels`) go in `__init__`; build-time dims (`actual_key_dim` resolved from `input_shape`) MUST go in `build()` -- placing them in `__init__` reads `None` and crashes.
- **0-ULP `np.float32` equality is stronger evidence than a tolerance test for a scale-constant swap.** `atol=1e-5` allows drift; byte-identity rules out float32 rounding entirely and is the right gate for behavior-preserving precompute conversions.
- **`LocalEncoder` (blt_blocks) already pools internally -- callers must NOT re-pool.**
- **Keras sublayer-list accumulators in `build()` are a double-build trap.** Any `build()` that appends to `self.some_list` MUST reset the accumulator at the TOP of `build()`.
- **When `get_config` serializes a dataclass via `to_dict()`, add a `from_config` class method that rebuilds the dataclass.**
- **Anticipated multi-bug chains sometimes do NOT materialize.** Fix-on-demand is correct. Do NOT pre-emptively edit unconfirmed bugs.
- **Surface EXECUTE-discovered out-of-scope bugs -- don't silently scope-creep.** Defer to a dedicated follow-up plan.
- **Deferring out-of-scope bugs keeps both the parent plan and the fix plan clean and verifiable.**
- **Surface+defer with xfail-gating is the correct call for multi-bug chains that exceed the 10-line leash.** A focused follow-up plan resolves them cleanly.
- **Ephemeral manual verification is not durable.** When a reviewer flags a blind spot, commit the regression test immediately.
- **"Fully resolved" from a prior plan is scoped to that plan's findings, not exhaustive.** A fresh adversarial pass with per-claim source verification can still find real residue.
- **When fixing a defect class, grep ALL files in the package.** A sweep that fixes "the known instances" can miss siblings.
- **Factory `optional_params` completeness needs a param-passthrough test, not just construct-smoke.** Construct-smoke succeeds even when kwargs are silently dropped. Assert the param actually lands on the instance with a **non-default value**.

## Graph-safe attention patterns (plan_2026-06-14_7734bacd)

- **`range(ops.shape(x)[N])` is graph-broken** under `@tf.function`/jit. A test using a concrete shape passes; only a `TensorSpec([None,...])` symbolic trace reproduces the crash.
- **Static `.shape[N]` + fail-loud `ValueError` on `None`** is the graph-safe fix for any Python loop or Python-int branch over a sequence-length dim.
- **Keras functional API uses `compute_output_shape` and does NOT execute `call()` symbolically.** A `keras.Input(shape=(None,...))` test will NOT trigger a fail-loud guard inside `call()`. Only a `tf.function` + `TensorSpec([None,...])` trace runs `call()` with a dynamic dim.
- **Dense-with-input-shape-dependent-units stays None-sentinel-in-build + idempotency guard.** Fix is `if self.built: return` as the FIRST line of `build()`.
- **Keras tuple->list shape serialization breaks `isinstance(input_shape, list)` multi-input disambiguation.** Fix: check whether elements are list/tuple (list-of-shapes) vs int/None (single serialized shape).

## Attention-specific fixes (plan_2026-06-14_b9456f74)

- **Performer causal FAVOR+ prefix-sum**: `kv_outer = einsum('bhnf,bhnd->bhnfd',k,v)` -> `kv_cumsum = cumsum(kv_outer,axis=2)` -> `out = einsum('bhnf,bhnfd->bhnd',q,kv_cumsum) / expand_dims(z_causal,-1)`. NO `expand_dims` on v or q -- spurious rank-5 expansion is the bug.
- **Ring/blockwise differentiable assembly**: replace `ops.slice_update` (lowers to `XlaDynamicUpdateSlice`, no eager-TF gradient) with Python list-append + `ops.concatenate(axis=seq_dim)`. Forward numerics byte-identical; gradient flows correctly.
- **Swin SW-MSA mask MUST be parametrized by actual static H,W.** Hardcoded 2xws grid is silently wrong for any feature map != 2xws.
- **`if self.built: return` guard sweep across 25 attention files was mechanical and zero-regression.** Confirmed again for 10 ffn/ files (plan_2026-06-14_60541575) -- the sweep is always safe in Keras 3 when layers are single-build today.

## Attention residue cleanup (plan_2026-06-14_ab855e7e)

- **self.scale = 1.0/math.sqrt(float(head_dim)) in `__init__`** is the D-002 scale contract. Prior plans fixed cross/diff; gated/gqa/mobile_mqa were missed. After any D-002 fix sweep, grep the whole package for remaining `ops.sqrt(ops.cast(` scale sites.
- **Factory `optional_params` is the kwarg filter.** Only keys in required_params + optional_params reach the class constructor. Missing keys silently drop real class args.
- **mobile_mqa silently ignores `attention_mask`** due to downsampling making general masking non-trivial. Contract: document with IGNORED note in docstring + README caveat table.
- **Deferred (no caller): hopfield cross-attn KV-dim latent bug.** Self-attention path is correct. Fix only when a cross-attention caller exists.

## Attention layer invariants (F18/F19 lessons -- plan_2026-06-14_adaddf34)

- **Q@Kt REQUIRES Q and K to share the contraction (channel) dim.** A layer that reduces only K/V while leaving Q at full channels is mathematically broken.
- **Keras .keras round-trip serializes tuple ctor args and reloads them as list/TrackedList.** Normalize ALL size-like constructor args with `(x, x) if isinstance(x, int) else tuple(x)`. This recurred in ffn/OrthoGLUFFN (plan_2026-06-14_60541575 D-002) -- use `isinstance(x, (tuple, list))` for guards, then `tuple(x)` to normalize.

## Keras 3 build-ordering (CONFIRMED FOUR-STRIKE defect class)

- **An explicit `build()` that only calls `super().build()` leaves sublayers unbuilt at Keras `.keras` load-time weight-restore.**
- **The durable fix for the lazy-sublayer build-ordering defect is to give the SHARED layer a real `build()` that explicitly builds its sublayers.**
- **A `.keras` round-trip test is the detection oracle AND the removal-safety oracle.**
- **`keras.layers.Identity()` is a serializable drop-in for `Lambda(lambda x: x)`.**
- **Manual `child.build(input_shape)` in `parent.build()` required for Keras 3 model-save round-trip.**
- **`ResnetBlock.nin_shortcut` (and similar structural conditionals) are CORRECTLY kept conditional.**
- **Moving sublayers from `build()` to `__init__()` exposes latent attribute-name bugs hidden by construction-time deferral.**
- **Keras locks the sublayer tracker after first `build()`, so build-idempotency bugs only bite via `from_config`/functional-reuse.**

### Canonical Keras-3 lifecycle-conformance fix pattern

- **`None`-sentinel in `__init__`** when filter/unit counts depend on build-time input shape.
- **`if self.built: return` MUST be the FIRST line of `build()`** -- explicit child `.build()` calls are NOT self-guarded by Keras.
- **`while True` + `if <traced_tensor> < eps:` is graph-broken** under `@tf.function`. Replace with bounded `for range(n)`.
- **Factory registration of construction-clean layers is additive + low-risk.**

## Keras 3 ops and layer-call contracts

- **`keras.ops.normalize(x, axis=-1)` is the canonical L2-normalize idiom.**
- **`.assign()` reachable from `call` is a recurring bug class** for DEAD or no-op in-call updates.
- **`create_attention_layer("multi_head")` is self-attention only.** Cross-attention requires `"multi_head_cross"`.
- **`keras.ops.shape()` returns a TUPLE of scalar tensors on the TF backend.** Use `tf.shape(x)` + `tf.concat` for dynamic reshape targets.

## Porting PyTorch DiT / VAE to Keras 3

- **Reuse first, build only the absent pieces.**
- **Functional builders cannot be sub-layers of a subclassed Model.** Wrap in a thin `keras.layers.Layer`.
- **PyTorch dynamic scatter -> static one-hot select for XLA-safe mRoPE.**
- **Boolean SDPA keep-mask -> additive finite (-1e9) mask.**
- **The existing `TimestepEmbedding` stores sinusoidal freqs as a plain Python tensor -- breaks `.keras` round-trip.** Fix: `add_weight(trainable=False)`.
- **`ops.tril` on a dynamic-shaped tensor routes through a `tf.cond` that raises `pred must not be a Python bool` during symbolic `.keras` save/load trace.** Use an arange index-comparison (`row >= col`).
- **T5 attention deliberately omits `1/sqrt(head_dim)` scaling.**
- **SD3 / rectified-flow sign convention: t=0 is clean data, t=1 is pure noise; `euler_step` uses signed `dt = t_next - t`.**

## Keras initializers

- **Keras custom initializers are shape-driven (`__call__(shape, dtype)`) -- do NOT bake `n_in`/`n_out` into the constructor.**
- **Per-call seed reproducibility requires `np.random.default_rng(seed)` inside `__call__`, NOT `keras.random.SeedGenerator`.**
- **`polar_initializer.py` is the structural template** for new initializers.
- **A string initializer alias that is NOT a registered Keras alias crashes in `__init__` via eager `keras.initializers.get(str)`.**

## Guide-conformance and repo-wide mechanical fixes

- **Before a repo-wide conformance fix, AST-scan to QUANTIFY then TRIAGE.**
- **`compute_output_shape` is a pure addition** -- never called in forward/train, only during functional-API shape inference.
- **`register_keras_serializable` keys by bare class name.** Two classes with the same name collide at import. Use `package=` to disambiguate.
- **A wrong `compute_output_shape` is worse than none.**

## Numerical / ML algorithm stability

- **Modified-Bessel ratio `I_nu/I_{nu-1}` MUST use the continued-fraction / downward (Miller) recurrence.**
- **A vMF sphere VAE's free-learned kappa posterior-collapses to ~0 unless prevented.** Fix: high kappa init + KL warmup.
- **Early-stopping on `val_total_loss` is confounded by KL warmup.** Monitor `val_reconstruction_loss`.
- **mLSTM/matrix-memory cells REQUIRE the log-domain max-stabilizer m_t, same as sLSTM.**
- **Eager correctness and `model.fit()` correctness are distinct failure classes.**
- **Seed BOTH weights AND inputs for any margin-comparison test.** An unseeded weight+input combination magnifies float32 variation under GPU memory pressure, causing non-deterministic failures at tight tolerances. Use `atol=1e-5` for eager-vs-graph comparisons (not 1e-6); a real graph-unroll defect diverges by orders of magnitude, so 1e-5 still locks the structural invariant.

## XLA / GPU kernel compatibility

- **`keras.random.beta` lowers to `StatelessRandomGammaV3`, which has NO XLA-GPU kernel in TF 2.18.** Override `Model.compile()` to force `jit_compile=False`.
- **Always run the scoped suite on GPU -- XLA-GPU-only crashes pass silently on CPU.**
- **`ops.slice_update` lowers to `XlaDynamicUpdateSlice`, which has no registered eager-TF gradient.** Backprop through blockwise output placement requires list-append + `ops.concatenate`.

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
- **DETR (`models/detr/`) was fixed in plan_2026-06-13_28f0b453.** Now fully tested (21 tests).

## git safety in EXECUTE

- **`git add -A` is a contamination footgun.** Use `git add <explicit paths>` ONLY.
- **Sub-agents that `git add` (even when told "no commit") cause the orchestrator's sequential commits to sweep files across step boundaries.**

## Deletion / refactoring safety

- **A whole-file Write can SILENTLY no-op when the file was modified since last read.** Gate: repo-wide grep success-criterion immediately after every large rewrite.
- **Before rewriting a to-be-edited file, restore from VCS.**
- **Removing a model variant is importer-first, delete-last.**

## tf.data / data-pipeline determinism

- **For tf.data pipeline determinism, a trace-time Python bool `training` flag with a plain if/else keeps the training path byte-identical.**
- **Validation that reuses the training augmentation recipe gives a noisy early-stopping signal.**
- **`drop_remainder=True` on a non-repeating val pipeline silently yields zero batches when N_val < batch_size.**

## Multi-optimizer / layer / model patterns

- **Two-optimizer differential-LR: register ONE optimizer with `super().compile(...)`, apply second manually.**
- **`add_weight(trainable=False, dtype="float32")`** for `current_phase` / `_global_step` -- int32 causes device-placement failures.
- **Always include `test_save_load_roundtrip` wrapping the layer inside a `keras.Model`.**

## GPU / environment

- **GPU0 (RTX 4090) is frequently saturated by the user's training jobs; pin ad-hoc test/eval runs to GPU1 (RTX 4070) with `CUDA_VISIBLE_DEVICES=1`.**
- **CC3M is on a spinning HDD (data0_4tb); raw-stream training is GPU-starved at ~4s/step.** Fix via RAM cache.
