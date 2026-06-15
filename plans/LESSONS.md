# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines -- rewrite, don't append forever.*
*Read before any PLAN/EXPLORE state. This is institutional memory.*

## Process

- **PLAN -> PLAN cycles are normal.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE.
- **CLOSE-rejection that produces 2-3 new concerns is the normal mode of operation.** Present findings honestly.
- **Doc updates belong in the same plan as the code change they describe.**
- **Run the existing test suite as the FIRST step of EXPLORE when the goal is "fix issues found in a review".**
- **A reuse-review / recommendation doc is a HYPOTHESIS, not a finding -- source-read every claim before PLAN.**
- **Explorer audit claims are HYPOTHESES.** Source-verify every explorer claim before planning. A CONFIRMED explorer finding can still be REFUTED on direct source read (bert_embeddings:222 refuted; activation plan: `len(inputs.shape)`, `keras.activations.elu`, `keras.backend.epsilon()` all flagged "graph-unsafe" -- all are graph-safe on source read). Do not skip orchestrator verification.
- **`len(tensor.shape)` returns the STATIC RANK as a Python int -- graph-safe.** Do not flag it as eager/graph-unsafe. `keras.activations.*` functions are graph-traceable; `keras.backend.epsilon()` is a Python float. Recurring explorer over-flagging pattern.
- **Pre-Mortem "STOP IF X" triggers earn their cost when they fire in 1-2 plans out of N.**
- **Plan-time line-count predictions undershoot.** ~2x for sibling-class additions, 5-10x for dtype-semantics under mixed precision, ~1.7x for multi-layer flag-plumbing.
- **Pre-existing tests can encode bugs as contracts.** When a fix causes adjacent tests to fail, read the failing assertions before assuming regression.
- **Verify "every site does X" with grep BEFORE PLAN, not during EXECUTE.**
- **`run_in_background` from a sub-agent kills the background task when the sub-agent exits.** Tasks >2 min MUST be orchestrator-launched.
- **For `os._exit(0)` scripts the success oracle is the completion log line + results file, NOT exit code.**
- **Smoke != correctness -- smoke tests must assert distributional invariants across tf.data + preprocessing boundaries.**
- **Adversarial review (iter>=2) catches opt-in path regressions the stub tests miss.** Always exercise opt-in paths.
- **Diagnostic-vs-fix framing is a load-bearing decision -- log it explicitly in `decisions.md` BEFORE writing any code step.**
- **`validate-plan.mjs` is repo-wide.** Gate before CLOSE is zero ERRORs INTRODUCED by the current plan. Pre-existing orphan/unknown-plan errors from prior plans are NOT CLOSE blockers -- triage by plan ID.
- **In-code anchor IDs MUST be numeric D-NNN matching decisions.md.** Finding-report IDs (D-A, D-E style) are not valid anchor IDs.
- **decisions.md `## D-NNN` headers MUST be exactly 3 segments: `## D-NNN | <context> | YYYY-MM-DD`.**
- **A plan can legitimately REVERSE a closed prior plan's decisions** when a new authoritative source contradicts them.
- **Architecture-first migration ordering**: migrate into the library BEFORE renaming/deleting the train scripts that import them.
- **First-ever forward-pass tests on never-tested layers surface CASCADING latent bugs.** Budget explicitly. Assume multi-bug chains in never-executed code.
- **Explorer gap-map claims MUST be source-verified against the actual `__init__` signature.**
- **REFLECT regression runs MUST cover test files that NO EXECUTE step touched.**
- **Fail-loud guards require a grep-verified zero-caller check first.** log-warn+fallback is the safe choice when the caller population is non-zero.
- **Surface EXECUTE-discovered out-of-scope bugs -- don't silently scope-creep.** Defer to a dedicated follow-up plan.
- **Ephemeral manual verification is not durable.** When a reviewer flags a blind spot, commit the regression test immediately.
- **"Fully resolved" from a prior plan is scoped to that plan's findings, not exhaustive.** A fresh adversarial pass can still find real residue.
- **When fixing a defect class, grep ALL files in the package.** A sweep that fixes "the known instances" can miss siblings.
- **Factory `optional_params` completeness needs a param-passthrough test, not just construct-smoke.**
- **Factory registries can silently diverge from class `__init__` defaults** (override a constraint with None, flip a bool). Diff every registry `optional_params` value against the class signature -- class is source-of-truth. (plan_2026-06-15_0205772c: `thresh_max` `trainable_slope` True vs False; `differentiable_step` `shift_constraint` None vs `ValueRangeConstraint(-1,+1)`.)
- **Anticipated multi-bug chains sometimes do NOT materialize.** Fix-on-demand is correct. Do NOT pre-emptively edit unconfirmed bugs.

## Serialization / __init__ package hygiene

- **`__all__` MUST be `List[str]`, not a list of class objects.** Class objects in `__all__` break star-import and static-analysis tooling silently. Check this first when auditing a package `__init__`. (plan_2026-06-15_2485b951: norms `__init__.py` had class objects; 5 classes also simply missing.)
- **`x or DEFAULT` for serializable objects (regularizers, initializers) skips `keras.<x>.get()`.** A serialized dict from `from_config` is truthy and stored verbatim, breaking re-serialization. Use `keras.regularizers.get(x) or DEFAULT` (or the equivalent `.initializers.get`). (plan_2026-06-15_2485b951 B3: `band_rms`/`zero_centered_band_rms_norm`; contrast with `band_initializer` which correctly used `.get()`.)
- **`build()` MUST NOT mutate ctor attrs.** Normalizing `self.axis` in `build()` (e.g. negative → positive) makes `get_config` return build-state-dependent values, breaking deserialization contract. Store the normalized form in `self._attr`; keep the ctor attr verbatim throughout. (plan_2026-06-15_2485b951 B4: DynamicTanh; fix: `self._norm_axis`.)

## Math correctness / "operates as advertised"

- **A layer can advertise "adaptive/learnable" behavior that is mathematically degenerate.** Verify the math, not just the docstring. When the real fix is a redesign and is out of scope, document the limitation honestly and apply only contract fixes. (plan_2026-06-15_2485b951 B2: BandLogitNorm LayerNorm over `[...,1]` → scale ≡ constant.)
- **A layer advertising a mathematical property (monotonicity) may not actually guarantee it.** Verify the math; correctness fix bounds the perturbation relative to guaranteed spacing. (plan_2026-06-15_0205772c: `MonotonicityLayer._sigmoid` flexibility=0.5 could produce inversions.)
- **Modified-Bessel ratio `I_nu/I_{nu-1}` MUST use the continued-fraction / downward (Miller) recurrence.**
- **A vMF sphere VAE's free-learned kappa posterior-collapses to ~0 unless prevented.**
- **mLSTM/matrix-memory cells REQUIRE the log-domain max-stabilizer m_t.**
- **Seed BOTH weights AND inputs for any margin-comparison test.**

## Graph-safety patterns

- **Graph-trace tests catch padding-path eager breaks** that reshape-only code review misses. Always include a `@tf.function` trace test with `TensorSpec([None,...])` when fixing graph-safety issues.
- **Dynamic reshape idiom**: `ops.reshape(t, (*ops.shape(x)[:-1], -1))` is graph-safe with one `-1` inferred trailing dim. Avoid `list(tensor.shape)` in shape tuples -- materializes None at trace time.
- **`range(ops.shape(x)[N])` is graph-broken** under `@tf.function`/jit. Use static `.shape[N]` + fail-loud `ValueError` on None.
- **`keras.ops.shape(x)[i]` is a DYNAMIC scalar tensor (not Python `None`) on the TF backend.** So `ops.broadcast_to(t, (ops.shape(x)[0], N))` is graph-safe and traces fine in the functional API — a recurring explorer "EAGER BREAK" false-positive (refuted by repro in MoE gating `top_k==num_experts` path). Distinguish from `list(tensor.shape)[i]` which CAN be `None`.
- **Before adding an "eager-only" guard, check whether host materialization is actually necessary.** `Variable.assign` and `keras.ops.linspace(tensor, tensor, n)` are graph-safe; only `float(convert_to_numpy(...))` + Python-attr mutation force eager.
- **When removing an eager-only debug warning for graph safety**: keep the ctor arg accepted + stored + serialized (back-compat for saved configs); update docstring; add an in-code NOTE comment.

## Graph-safe training-gate patterns

- **`training is True` is NOT universally applicable.** For layers that must support symbolic-training custom loops: use the masked-factor pattern (`resolve_training_factor` in `utils/tensors.py`).
- **`keras.ops.cond` is awkward for side-effects** (`add_loss` / `.assign`): tensor-masking (`cast(training) * delta/loss`) is cleaner.
- **`if training:` crashes under symbolic tensor.** `if training is not False:` fires under `training=None` (wrong). `if training is True:` is the canonical guard.
- **`model.fit()` passes `training` as python `bool True` in Keras 3.8 / TF 2.18.** Symbolic-tensor case only arises in hand-rolled `@tf.function` custom loops.

## Graph-safe attention patterns

- **Keras functional API uses `compute_output_shape` and does NOT execute `call()` symbolically.**
- **Dense-with-input-shape-dependent-units stays None-sentinel-in-build + idempotency guard.**
- **Keras tuple->list shape serialization breaks `isinstance(input_shape, list)` multi-input disambiguation.** Fix: check whether elements are list/tuple (list-of-shapes) vs int/None (single serialized shape).
- **Performer causal FAVOR+ prefix-sum**: `kv_outer = einsum('bhnf,bhnd->bhnfd',k,v)` -> `kv_cumsum = cumsum(kv_outer,axis=2)` -> `out = einsum('bhnf,bhnfd->bhnd',q,kv_cumsum) / expand_dims(z_causal,-1)`.
- **Ring/blockwise differentiable assembly**: replace `ops.slice_update` with Python list-append + `ops.concatenate(axis=seq_dim)`.

## Activation / serialization (ffn package)

- **`keras.activations.serialize(keras.activations.get('swish')) == 'silu'`** -- alias canonicalization. Tests asserting a literal activation token after `get()` + serialization MUST assert functional equivalence, not the literal.
- **A `lambda` activation cannot round-trip through `.keras` for ANY layer.**
- **All 15 `layers/ffn/` layers use the canonical activation contract**: `keras.activations.get()` in `__init__`, `keras.activations.serialize()` in `get_config`.

## Keras 3 build-ordering

- **An explicit `build()` that only calls `super().build()` leaves sublayers unbuilt at `.keras` load-time weight-restore.**
- **Canonical Keras-3 lifecycle**: `None`-sentinel in `__init__` for build-time dims; `if self.built: return` MUST be the FIRST line of `build()`; explicit sublayer `.build()` in parent `build()`; `super().build()` LAST.
- **Keras sublayer-list accumulators in `build()` are a double-build trap.** Any `build()` that appends to `self.some_list` MUST reset the accumulator at the TOP (or guard via `if self.built: return` first).
- **`if self.built: return` guard sweep is mechanical and zero-regression** (confirmed: 25 attention files, 14 embedding files, 15 FFN files, 7 activation files, 10 norms files, 5 MoE classes -- 6 sub-package sweeps).
- **Missing `if self.built: return` does NOT break `.keras` save/load roundtrip** -- Keras's `__call__`-driven build path already no-ops on a built layer (verified Δ=0). The guard's real payoff is tolerating an EXPLICIT second `layer.build(shape)` (which otherwise raises `ValueError: cannot add state ... already built`) and matching the canonical pattern. Don't report a missing guard as a serialization bug.
- **Manual `child.build(input_shape)` in `parent.build()` required for Keras 3 model-save round-trip.**
- **`keras.layers.Identity()` is a serializable drop-in for `Lambda(lambda x: x)`.**

## Keras 3 ops and layer-call contracts

- **`keras.ops.normalize(x, axis=-1)` is the canonical L2-normalize idiom.**
- **`keras.ops.shape()` returns a TUPLE of scalar tensors on the TF backend.**
- **`keras.ops.fft` surface is narrow.** Available: `fft/fft2/ifft2/rfft/irfft/real/imag`. NOT available: `rfft2/irfft2/angle/complex`.
- **`keras.ops.stop_gradient` is a FUNCTION, not a context manager.**
- **A missing `return config` in `get_config` returns None silently.** Every new-layer test MUST assert `get_config`/`from_config` directly.
- **`compute_output_shape` is a silent functional-API bug surface — test it against ACTUAL `call` output.** keepdims reductions return `[...,1]` (not the reduced shape, not the full input shape); multi-output layers must declare a tuple. Tests that only exercise `call()` pass while shape metadata is wrong. (norms `DMLPlus(model_type='center')`: declared `(B,C)`, actual `(B,1)`.)
- **Config-factory `validate` and `create` must be cross-checked per-key against the REAL ctor signature.** Whitelist drift is recurring: a first sweep fixed some keys, a second pass found more (`dynamic_tanh` missing 5 ctor params; `max_band_width` missing its `<1` upper bound while classes enforce `0<x<1`). When fixing a factory whitelist, enumerate EVERY key and BOTH bounds, and add a `validate`==`create` agreement test.
- **Compute STATIC scalars (e.g. attention `scale = 1/sqrt(head_dim)`) with stdlib `math.*`, NOT `keras.ops.*`.**
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`.
- **`while True` + `if <traced_tensor> < eps:` is graph-broken.** Replace with bounded `for range(n)`.
- **`register_keras_serializable` keys by bare class name.** Use `package=` to disambiguate collisions. NEVER change an existing `package=` string -- it changes the registration KEY and breaks deserialization of already-saved `.keras` models.
- **When `get_config` serializes a dataclass via `to_dict()`, add a `from_config` class method that rebuilds the dataclass.**

## Embedding sub-package specifics

- **`HierarchicalCodebookEmbedding` is direct-import-only.** Non-standard ctor; not factory-registered.
- **`PositionEmbeddingSine2D` emits channels-FIRST `(B, 2*num_pos_feats, H, W)`.** Consumers expecting channels-last must transpose.
- **`ModernBertEmbeddings` has NO ctor defaults** -- all 7 args required. Factory supplies the 4 optional-context defaults.
- **`ContinuousRoPE.compute_output_shape` returns dim/2** (phase width), NOT dim.

## Numerical / ML algorithm stability

- **Seed BOTH weights AND inputs for any margin-comparison test.** Use `atol=1e-5` for eager-vs-graph comparisons.

## XLA / GPU kernel compatibility

- **`keras.random.beta` lowers to `StatelessRandomGammaV3`, which has NO XLA-GPU kernel in TF 2.18.**
- **`ops.slice_update` lowers to `XlaDynamicUpdateSlice`, which has no registered eager-TF gradient.**
- **`DepthwiseConv2D` on TF GPU rejects asymmetric strides.**
- **Always run the scoped suite on GPU.**

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite ~1.5h. Scope pytest to changed modules only.
- **`results/` MUST be the repo-root `results/` dir, never `src/results/`.**
- **Use `MPLBACKEND=Agg` for any training-script invocation.**
- **Use `dl_techniques.utils.logger` only -- no `print`.**
- **Single GPU jobs only.** Never spawn parallel training runs.
- **ConvNeXt V1/V2 heads emit raw logits.** Trainers must compile with `from_logits=True`.

## git safety in EXECUTE

- **`git add -A` is a contamination footgun.** Use `git add <explicit paths>` ONLY.
- **Sub-agents that `git add` cause the orchestrator's sequential commits to sweep files across step boundaries.**

## Deletion / refactoring safety

- **A whole-file Write can SILENTLY no-op when the file was modified since last read.**
- **Before rewriting a to-be-edited file, restore from VCS.**
- **Removing a model variant is importer-first, delete-last.**

## tf.data / data-pipeline determinism

- **For tf.data pipeline determinism, a trace-time Python bool `training` flag with a plain if/else keeps the training path byte-identical.**
- **Validation that reuses the training augmentation recipe gives a noisy early-stopping signal.**
- **`drop_remainder=True` on a non-repeating val pipeline silently yields zero batches when N_val < batch_size.**

## Multi-optimizer / layer / model patterns

- **Two-optimizer differential-LR: register ONE optimizer with `super().compile(...)`, apply second manually.**
- **Always include `test_save_load_roundtrip` wrapping the layer inside a `keras.Model`.**

## GPU / environment

- **GPU0 (RTX 4090) frequently saturated; pin ad-hoc runs to GPU1 (RTX 4070) with `CUDA_VISIBLE_DEVICES=1`.**
- **CC3M on spinning HDD -- raw-stream training GPU-starved at ~4s/step; fix via RAM cache.**
