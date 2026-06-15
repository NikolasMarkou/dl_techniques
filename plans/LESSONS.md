# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines -- rewrite, don't append forever.*
*Read before any PLAN/EXPLORE state. This is institutional memory.*

## Process

- **PLAN -> PLAN cycles are normal.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE.
- **CLOSE-rejection that produces 2-3 new concerns is the normal mode of operation.** Present findings honestly.
- **Doc updates belong in the same plan as the code change they describe.**
- **Run the existing test suite as the FIRST step of EXPLORE when the goal is "fix issues found in a review".**
- **A reuse-review / recommendation doc is a HYPOTHESIS, not a finding -- source-read every claim before PLAN.**
- **Explorer audit claims are HYPOTHESES.** Source-verify every explorer claim before planning. A CONFIRMED explorer finding can still be REFUTED on direct source read.
- **Entrypoint maps produced by explorers are HYPOTHESES.** At least 3 family signatures were stale in plan_2026-06-15_b5cec9e4: fftnet is a vision model not an LM; `create_blt_model` real sig is `(variant, vocab_size, max_sequence_length, ...)`; `create_tabm_mini` requires `cat_cardinalities` as a positional argument. Executors MUST source-read constructor signatures before constructing.
- **`len(tensor.shape)` returns the STATIC RANK as a Python int -- graph-safe.** Do not flag it as eager/graph-unsafe.
- **Pre-Mortem "STOP IF X" triggers earn their cost when they fire in 1-2 plans out of N.**
- **Plan-time line-count predictions undershoot.** ~2x for sibling-class additions, 5-10x for dtype-semantics under mixed precision, ~1.7x for multi-layer flag-plumbing.
- **Pre-existing tests can encode bugs as contracts.** Triage as pre-existing before assuming regression.
- **Verify "every site does X" with grep BEFORE PLAN, not during EXECUTE.**
- **`run_in_background` from a sub-agent kills the background task when the sub-agent exits.** Tasks >2 min MUST be orchestrator-launched.
- **Smoke != correctness -- smoke tests must assert distributional invariants across tf.data + preprocessing boundaries.**
- **Adversarial review (iter>=2) catches opt-in path regressions the stub tests miss.** Always exercise opt-in paths.
- **Diagnostic-vs-fix framing is a load-bearing decision -- log it explicitly in `decisions.md` BEFORE writing any code step.**
- **`validate-plan.mjs` is repo-wide.** Gate before CLOSE is zero ERRORs INTRODUCED by the current plan. Pre-existing orphan/unknown-plan errors from prior plans are NOT CLOSE blockers.
- **In-code anchor IDs MUST be numeric D-NNN matching decisions.md.**
- **decisions.md `## D-NNN` headers MUST be exactly 3 segments: `## D-NNN | <context> | YYYY-MM-DD`.**
- **Architecture-first migration ordering**: migrate into the library BEFORE renaming/deleting the train scripts that import them.
- **First-ever forward-pass tests on never-tested layers surface CASCADING latent bugs.** Budget explicitly. Assume multi-bug chains in never-executed code. In plan_2026-06-15_b5cec9e4 sweep, 10 of 20 gap families broke on first real forward. plan_2026-06-15_e2759fbc: dino v2 surfaced 13 bugs total (7 planned + 6 cascade). Never estimate one bug per family; budget at least two.
- **Explorer gap-map claims MUST be source-verified against the actual `__init__` signature.**
- **REFLECT regression runs MUST cover test files that NO EXECUTE step touched.**
- **Fail-loud guards require a grep-verified zero-caller check first.**
- **Surface EXECUTE-discovered out-of-scope bugs -- don't silently scope-creep.** Defer to a dedicated follow-up plan.
- **Ephemeral manual verification is not durable.** Commit the regression test immediately when a reviewer flags a blind spot.
- **When fixing a defect class, grep ALL files in the package.** A sweep that fixes "the known instances" can miss siblings.
- **Factory `optional_params` completeness needs a param-passthrough test, not just construct-smoke.**
- **Factory registries can silently diverge from class `__init__` defaults.** Diff every registry value against the class signature.
- **Anticipated multi-bug chains sometimes do NOT materialize.** Fix-on-demand is correct. Do NOT pre-emptively edit unconfirmed bugs.
- **Verify the call signature before declaring a multi-input layer broken.** `L([q,kv])` vs `L(q, kv)` looks like a bug but is just the wrong convention.
- **A subclass inherits attributes from its base -- grep the base class before reporting an "undefined attribute".**
- **A layer's OUTPUT STRUCTURE must depend ONLY on construction-time config, never on `training`.** `compute_output_shape` is training-agnostic. Emit a semantically correct placeholder for training-only auxiliary outputs at inference (e.g. zeros for bit_logits = uniform prior). (plan_2026-06-15_32b5822c D-001.)
- **xfail-with-captured-error pattern**: `pytest.xfail(str(e))` inside `except Exception` after a genuine build+forward attempt is the correct smoke idiom for report-only sweeps. Broad `except` also converts NaN-assertion failures into xfail. Acceptable for coverage sweep markers; not for correctness tests.
- **Actual model dir count is 70** (not 71 -- the extra was `__pycache__`). Always verify with `find ... -mindepth 1 -maxdepth 1 -type d | grep -v __pycache__`.
- **For math-heavy fixes (Kabsch/SVD, matrix decompositions), add explicit correctness checks at REFLECT** -- not just finiteness.
- **Semantic fix-forwards are adversarial-review targets.** Dropping a broken weight-tie makes a test green but silently removes intended weight-sharing. Must be surfaced in decisions.md and code comments -- not silently accepted. (plan_2026-06-15_39a31d4a D-002 cascade.)
- **When a never-run model is fixed, remove the xfail safety-net as soon as it passes.** Leaving `try/except pytest.xfail` silently swallows future regressions. (plan_2026-06-15_2a23a001 step 5.)
- **Verify adversarial reviewer CRITICALs empirically before patching.** The register-token CRITICAL in plan_2026-06-15_e2759fbc was a design-misread: DINOv2-w-registers keeps registers position-free by design (Darcet 2023). A blind fix would have corrupted the architecture. Build a probe, run it, observe the actual crash (or non-crash) before writing any fix.
- **Cascade budget overrides must be logged immediately in decisions.md, not absorbed silently.** When the declared STOP-IF fires (plan_2026-06-15_e2759fbc D-007: 3rd extra cascade bug), create the D-NNN entry BEFORE continuing. That entry is the only durable record of why the plan pushed through. An override without a log entry is silent scope creep.
- **Fix-forward to non-crash is not the same as correct.** The adversarial review in plan_2026-06-15_e2759fbc caught a degenerate constant-zero mask token (zero-init Dense-on-ones) + a fragile call() override + dead is_training input that all passed the forward smoke. The principled fixes (MaskTokenApply learnable weight, remove dead input) also fixed .keras serialization as a side effect -- non-crash is necessary but not sufficient.
- **The Dense-on-ones-projection is a recurring antipattern.** CLS in dino v1/v2 (plan_2026-06-15_39a31d4a, plan_2026-06-15_e2759fbc D-002) and mask token in dino v2 (D-001, later D-009) both used it. The canonical fix is a small weight-owning layer: ClassTokenPrepend for CLS, MaskTokenApply for iBOT mask. Both live in `layers/embedding/` and are reusable.
- **Cascade budgets are guards, not laws.** A never-run model can surface many bugs. With explicit user authorization + monotonically non-thrashing progress (each bug is a distinct area, 3-strike rule has NOT fired), pushing through is valid. The key contract: each override is LOGGED (D-007/D-008), and the HARD STOP rule changes from "bug count" to "architectural redesign needed OR 3-strike thrash."

## Keras-3 Functional-model construction rules

- **`add_weight` must NOT fire before `super().__init__(inputs=, outputs=)`** in a Functional model. Keras layer machinery is not yet initialized. Reusable fix: create a dedicated sub-layer whose `build()` owns the weight; instantiate the sub-layer inside the model's graph-build function (which runs AFTER `super().__init__`). See `ClassTokenPrepend` in `layers/embedding/class_token.py` (plan_2026-06-15_39a31d4a/D-001).
- **Option B (model `build()` override) does NOT work for Functional models.** The graph finalizes at `__init__`; `build()` never re-runs the symbolic part.
- **A `**kwargs` splat means EVERY source of a ghost kwarg must be fixed.** Missing one source re-injects the broken kwarg. Grep all injection points before PLAN. (plan_2026-06-15_39a31d4a D-002: 5 sources for nano_vlm fusion config.)
- **`PFTBlock.build` and `compute_output_shape` now accept both list and tuple of shapes** (plan_2026-06-15_2a23a001/D-002). Detection must check `isinstance(input_shape[0], (list, tuple))` -- NOT `isinstance(input_shape, (list, tuple))` because a single input SHAPE is a tuple-of-ints. The naive 2-token widen is a latent bug.

## Weight-tying

- **Weight-tying in Keras 3 MUST be at CALL TIME.** Use `ops.matmul(x, ops.transpose(emb))` at the logit site(s). NEVER reassign another layer's weight post-build -- that is illegal under Keras 3 ("cannot add state to an already-built Layer") and was the exact failure in plan_2026-06-15_39a31d4a/D-002.
- **The tying test MUST use a negative-proof.** Zero or perturb the unused Dense kernel; assert logits are unchanged. Finiteness + shape alone is a fake-tie hazard -- the matmul path could silently fall through to the Dense.
- **Keep the unused Dense built.** It is needed for `use_shared_embedding=False`, `get_config()`, and `.keras` save/load of the unshared path.

## Attention and transformer patterns

- **`TransformerLayer.call` ALREADY forwards `attention_mask` to its multi_head attention sublayer.** Masked-attention features MUST reuse this plumbing.
- **`multi_head_cross` is the factory-registered cross-attention type.** Use `encoder_attention_type='multi_head_cross'` and pass `kv_input=...` for real encoder cross-attention.
- **When building an attention keep-mask, keep a query->query block all-ones.** If any attention row is fully masked, softmax over all -inf yields NaN.
- **`create_attention_layer` FILTERS unknown kwargs to the constructor signature.** Unknown kwargs are silently dropped.
- **First-ever tests on never-tested layers must assert BEHAVIORAL properties, not just shapes.**
- **`_MASKLESS_ATTENTION_TYPES = {'fnet', 'anchor', 'lighthouse'}`** -- these three skip the `attention_mask` forward path.

## Graph-safety / Keras 3 call patterns

- **An eager `if not <traced_tensor>:` branch on a probabilistic gate breaks under `@tf.function`/`model.fit`.** Replace with arithmetic blending.
- **`if training:` crashes under symbolic tensor.** Canonical guard: `if training is True:`.
- **`training is True` is NOT universally applicable.** For symbolic-training custom loops: use the masked-factor pattern.
- **`keras.ops.cond` is awkward for side-effects**: tensor-masking is cleaner.
- **Graph-trace tests catch padding-path eager breaks** that reshape-only code review misses.
- **Dynamic reshape idiom**: `ops.reshape(t, (*ops.shape(x)[:-1], -1))` is graph-safe with one `-1` inferred trailing dim.
- **`range(ops.shape(x)[N])` is graph-broken** under `@tf.function`/jit. Use static `.shape[N]` + fail-loud `ValueError` on None.
- **`keras.ops.shape(x)[i]` is a DYNAMIC scalar tensor** on the TF backend -- graph-safe. Distinguish from `list(tensor.shape)[i]` which CAN be `None`.
- **`while True` + `if <traced_tensor> < eps:` is graph-broken.** Replace with bounded `for range(n)`.
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`.

## Keras 3 build-ordering

- **Canonical Keras-3 lifecycle**: `None`-sentinel in `__init__` for build-time dims; `if self.built: return` MUST be the FIRST line of `build()`; explicit sublayer `.build()` in parent `build()`; `super().build()` LAST.
- **`if self.built: return` guard sweep is mechanical and zero-regression.**
- **A weight-bearing sublayer built with a None last-dim Dense crashes build().** If never called in call(), it is DEAD AND build-breaking; remove it entirely.
- **Missing `if self.built: return` does NOT break `.keras` save/load roundtrip.**
- **Keras sublayer-list accumulators in `build()` are a double-build trap.** Reset the accumulator at the TOP of build(), or guard via `if self.built: return`.
- **Manual `child.build(input_shape)` in `parent.build()` required for Keras 3 model-save round-trip.**
- **`keras.layers.Identity()` is a serializable drop-in for `Lambda(lambda x: x)`.**

## Serialization / __init__ package hygiene

- **`__all__` MUST be `List[str]`, not a list of class objects.**
- **`x or DEFAULT` for serializable objects skips `keras.<x>.get()`.** Use `keras.regularizers.get(x) or DEFAULT`.
- **`build()` MUST NOT mutate ctor attrs.** Store normalized form in `self._attr`; keep the ctor attr verbatim.
- **`register_keras_serializable` keys by bare class name.** NEVER change an existing `package=` string -- it breaks deserialization of saved models.
- **A missing `return config` in `get_config` returns None silently.** Every new-layer test MUST assert `get_config`/`from_config`.

## Activation / serialization contract

- **`keras.activations.serialize(keras.activations.get('swish')) == 'silu'`** -- alias canonicalization.
- **A `lambda` activation cannot round-trip through `.keras` for ANY layer.**
- **Canonical activation pattern**: `keras.activations.get(activation)` in `__init__`, `keras.activations.serialize(self.activation)` in `get_config`.

## Math correctness

- **`__init__` that eagerly calls a `_create_*` helper which can `return None` silently produces a broken model.** Make it fail loud.
- **Modified-Bessel ratio `I_nu/I_{nu-1}` MUST use the continued-fraction / downward (Miller) recurrence.**
- **mLSTM/matrix-memory cells REQUIRE the log-domain max-stabilizer m_t.**
- **Compute STATIC scalars (e.g. attention `scale = 1/sqrt(head_dim)`) with stdlib `math.*`, NOT `keras.ops.*`.**
- **`compute_output_shape` is a silent functional-API bug surface -- test it against ACTUAL `call` output.**
- **`tf.linalg.svd` returns `(s, u, v)` with `H = u @ diag(s) @ v^T`.** Kabsch rotation: `R = V @ U^T`. Det-correction: `diag([1,1,det(R)])`. Do NOT unpack as `(U, s, Vt)` or `(U, _, Vt)`.

## Keras 3 ops and layer-call contracts

- **`keras.ops.fft` surface is narrow.** Available: `fft/fft2/ifft2/rfft/irfft/real/imag`. NOT available: `rfft2/irfft2/angle/complex`.
- **`keras.ops.stop_gradient` is a FUNCTION, not a context manager.**
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`.
- **`keras.ops.get_graph_feature` does NOT exist** in Keras 3.8. Use `_get_graph_feature` in `layers/geometric/point_cloud_autoencoder.py` (plan_2026-06-15_00924f53/D-001).
- **`keras.ops.scatter_nd_update` does NOT exist** in Keras 3.8. Use `keras.ops.scatter_update(inputs, indices, updates)` (plan_2026-06-15_00924f53/D-003).
- **`keras.layers.DepthToSpace` / `keras.ops.depth_to_space` / `keras.ops.nn.depth_to_space` do NOT exist** in Keras 3.8. Use `PixelShuffle2D` from `layers/pixel_unshuffle.py` (plan_2026-06-15_00924f53/D-002; also plan_2026-06-15_39a31d4a/D-003).
- **`keras.ops.add_n` does NOT exist** in Keras 3.8. Use a fold: `acc = tensors[0]; for t in tensors[1:]: acc = ops.add(acc, t)`.
- **Keras tensor `.item()` does NOT exist** -- use `float(r)`. NumPy-only; crashes on any non-NumPy tensor. (plan_2026-06-15_39a31d4a dino_v1; plan_2026-06-15_2a23a001/D-004 dino_v3.)

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
- **DETR is functional** (commit 072df479, 21 tests; old SYSTEM.md "DETR broken" invariant was stale as of plan_2026-06-15_b5cec9e4).
- **Model dir count is 70** (not 71; __pycache__ inflates naive directory listings).
- **Canonical model smoke harness**: `scripts/verify_models_smoke.py` (86-entry registry) is the single instrument for build+forward compliance across all 70 model packages. Re-run: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python scripts/verify_models_smoke.py [--only <name>]`.
- **Harness-recipe-error triage**: when a first-pass sweep shows many FAILs, inspect the traceback frame origin before touching model source. Harness lambda frame = recipe error (fix the registry); model source frame = model bug. In the plan_2026-06-15_e6a0391c sweep, 27/40 FAILs were recipe errors (wrong variant, wrong image size, wrong input format).
- **Keras-3 ops gaps (confirmed in sweep)**: `keras.ops` has no `gather` -- use `ops.take(x, idx, axis=0)`; `ops.arange` returns int32, cannot multiply by Python float without explicit cast to float32; convnext `padding="valid"` with kernel==stride collapses small inputs to 0x0 spatial map -> NaN -- use `"same"` (safe when kernel==stride; output shape unchanged for stride-divisible inputs).
- **SAM multi-bug chain**: image_encoder has 3 Keras-3 bugs (all fixed, D-004); mask_decoder has a separate multi-bug chain -- hit the 3-strike leash. XFAIL until a dedicated fix plan.
- **mobile_clip is non-functional**: references backbones mci0/mci1/mci2/vit_b16 absent from keras.applications. Non-functional until those backbones are ported.

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
