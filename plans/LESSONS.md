# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines — rewrite, don't append forever.*
*Read before any PLAN state. This is institutional memory.*

## Process

- **"Remove X" can mean two different things.** Distinguish (a) the *capability* is unwanted from (b) the *naming or layering* is confusing. EXPLORE should classify each piece before PLAN drafts deletion steps.
- **Doc updates belong in the same plan as the code change they describe.** Treating README/CLAUDE.md updates as out-of-scope follow-ups means the plan is technically "complete" while user-visible behavior is still wrong.
- **Static grep gates can substitute for runtime probes only for purely mechanical surface changes.** Renames with no internal recursion and no out-of-file callers are safe to verify with `grep -n`; anything that touches behavior needs a runtime smoke test.
- **PLAN -> PLAN cycles are normal.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE.
- **Mechanical pattern-replication across N files closes in a single iteration when EXPLORE per-file diffs the precedent's structural assumptions BEFORE PLAN.** Resist extracting a shared helper for <=4 sites if extraction crosses package boundaries.
- **Pure additive layer/class work fits a single iteration** — non-destructive change + tests scoped to one file.
- **Sibling-stack model additions: prefer parallel package over factory retrofit when the new component has a different mask shape or unique hyperparameters.** Trade ~150 LOC duplication for zero blast radius on 30+ unrelated models.
- **Orchestrator-crash mid-EXECUTE recovery: trust git log + filesystem, not stale state.md.** Enumerate `git log` since baseline, diff vs plan.md, re-run pytest, write a "Resume Notes" section, proceed to REFLECT. Do NOT re-execute completed steps.
- **Plan-time line-count predictions underestimate by ~2x for sibling-class additions; by 5-10x for dtype-semantics plans under mixed precision OR plans spanning >3 consumer scripts; by ~1.7x for multi-layer flag-plumbing plans.** Predict accordingly.
- **Pre-existing tests can encode bugs as contracts.** When a fix to a library bug causes adjacent tests to fail, read the failing assertions before assuming regression — they may be locking in buggy behavior. Update asserts and anchor the new contract with `# DECISION`.
- **Empirical confirmation of a finding does not separate bug from contract.** Grep the test suite for tests that *lock in* current behavior before classifying.
- **Run the existing test suite as the FIRST step of EXPLORE when the goal is "fix issues found in a review".** All-green tests are the cheapest evidence that some review claims are wrong.
- **Internal-consistency arguments are not falsification.** "X is causal because Y" needs a unit test on X *alone* before it gates an end-to-end design.
- **Verify reviewer claims empirically before applying fixes from a "deep review".** Reviews contain false positives. For each "critical" claim about API behavior, run a 3-line empirical check first. Lock empirically-correct behavior in a test. Verify BOTH directions (plan_e6309bd5).
- **A correctly-flagged bug can mask a worse adjacent bug.** When a reviewer flags one mixed-precision/dtype/serialization issue, sweep the whole category before merging the narrow fix (plan_3c3ed037).
- **Verify "every site does X" with grep BEFORE PLAN, not during EXECUTE.** Textual claims that look like inventory facts have a way of being half-true.
- **A bug finding can be semantically real but practically unreachable.** If the constructor validates the precondition away (e.g. `num_arith >= 1`), the math-bug path is dead. Verify the precondition is actually reachable before claiming observability. Defensive widening is still worth doing when free — it future-proofs the guard (plan_2026-05-13_e33114da B3 / D-004).
- **"Same direction, different math" naming is a footgun.** `gate_entropy_coefficient` minimises L2 of probs, not entropy; both are convex peakiness measures with the same uniform optimum. When renaming a config field, match the new name to the actual math of the implementation, or correct the rationale comment rather than ping-ponging the name (plan_2026-05-13_e33114da D-007).
- **Keras 3 list-of-shapes detection has two idioms.** `hasattr(x, '__iter__')` vs `not isinstance(x, (int, type(None)))` produce the same result on common inputs but diverge on edge cases. Pick one per module and apply it consistently; drift between `build()` and `compute_output_shape()` is a latent footgun (plan_2026-05-13_e33114da B6).
- **`training` flag must be threaded into every private method that calls `keras.random.*`.** The Keras layer base only forwards `training` to `call()`; private helpers get no automatic forwarding. Gumbel noise injected at inference (`model.predict()`) is non-deterministic and violates standard Keras convention (plan_2026-05-13_e33114da B2 / D-003).
- **Per-trainer "1-line bind" of a callback hook is actually a 4-line delta** (`probe_cb = X(...); probe_cb._hook = fn; callbacks.append(probe_cb)`).
- **Greenfield model packages with explicit blueprint converge in a single iteration when EXPLORE pins each algorithmic shape to a concrete Keras 3 idiom BEFORE PLAN** (plan_146ae899).
- **Audit-driven full-coverage plans converge in one iteration when (a) the audit doc carries file:line references and prescribed fix shape, AND (b) EXPLORE pre-resolves design choices for non-trivial fixes in a "design notes" finding before PLAN** (plan_0f39a086).
- **Falsification triggers must pair the symptom with a triage check.** plan_f2d29729 Scenario B fired and the plan-prescribed action was an LR sweep — wrong for the actual data-pipeline bug.
- **REV-1 (plan revision after PC-PLAN approval) is the right protocol move when user expands scope mid-plan with Revert-First constraint** (plan_44694bc9).
- **Bundling related EXECUTE steps into one commit is sound when one verification command validates all of them and they share a single file** (plan_54e6e303).
- **Reference-doc / benchmark-snapshot plans close single-pass via one research agent when EXPLORE pre-enumerates the coverage inventory** (plan_16ac1621).
- **Line-count caps on greenfield reference docs are sanity guards, not quality gates.** "Exceeds upper bound" should trigger a density check before being treated as bloat.
- **LOC overshoot from docstrings + per-flag argparse `help=` text is acceptable when no new abstractions are introduced beyond planned.** Track abstractions and blast-radius instead of raw LOC at REFLECT.
- **When the EXECUTE pipeline reports unexpected test failures, baseline-diff BEFORE calling regression** (plan_0090b0b8).
- **Smoke != correctness — smoke tests must assert distributional invariants across tf.data + preprocessing boundaries.** Canonical: `_assert_train_val_distribution_match(train_ds, val_ds, mean_tol=0.5, std_ratio_tol=0.5)` called immediately before `model.fit` (plan_f2d29729 D-007).
- **"Smooth-train + cliff-val + sub-random val" is the fingerprint of a train/val preprocessing divergence** — fix data pipeline first; sweep only on a clean baseline (plan_f2d29729 D-006/D-007).
- **A trainer's unguarded "TRAINING COMPLETED SUCCESSFULLY" log is a known footgun.** Guard on `best_val_acc >= max(2/num_classes, threshold)` AND `early_stop_epoch >= 0.5 * total_epochs` (plan_f2d29729 D-007).
- **Pure-additive integration-alignment plans on a self-contained layer package close in 1 iteration / 0 fix attempts** when EXPLORE pre-counts sibling factory'd packages, non-test consumers, and `register_keras_serializable` decorator shape (plan_2aaad563).
- **A strict-rank `build()` precondition tighter than what `call()` requires is a ghost constraint.** Confirm by reading `call()` end-to-end before proposing the relax.
- **Unary-input degeneracy in DARTS-style layers is a documented footgun, not a bug to fix in code.** Document in README Limitations and move on.
- **Double-checking a "deep review" against empirical probes is mandatory and routinely catches false positives.** Enumerate every claim with a one-line empirical test, run it, recalibrate the review before PLAN (plan_e52a5ac8).
- **Signal-preservation tests on chained unary inputs in DARTS-style layers are meaningless** — unary-input degeneracy dominates. Test on BINARY inputs (plan_e52a5ac8).
- **`compute_output_shape` on shape-preserving layers decays silently** — list-deserialized single shape vs binary inputs list diverge; existing tests rarely exercise the list-form-single-shape path.
- **"Manual `child.build(input_shape)` in parent.build()" is required for Keras 3 model-save round-trip.** When children are constructed in `__init__` but never built before `model.save()`, the saved weights file lists variables for unbuilt children and `load_model` raises. Keras 3 docs encourage lazy `__call__` build for forward pass, but serialization requires explicit pre-build (plan_a2b0f17b D-003).
- **`tf.GradientTape` returns `None` for `tf.Variable(np.ones(...))` in some Keras 3 environments.** In gradient-flow tests use the explicit-array literal form (plan_a2b0f17b).
- **The smooth-divide formulation `x·y/(y²+ε²)` has gradient `1/ε²` at y=0, NOT `1/ε`.** Document the correct mechanism (plan_a2b0f17b).
- **Real restriction of complex power for negative bases: `cos(πy)·|x|^y`, NOT `sign(x)·|x|^y`** (plan_a2b0f17b D-001).
- **Pre-mortem with explicit STOP-IF triggers pays off.** When NaN risk is predicted (scenario A+B), the first-sign catch and recovery fit within a 2-attempt leash budget. Write the pre-mortem; it shapes the recovery action (plan_2026-05-13_d256b568).
- **Default `arithmetic_op_types` (including `power`) is a footgun for stacks depth >= 3 with residuals.** Learned exponents amplify magnitudes; residuals compound the effect across depths. Restrict to `add,max,min` or enable `exponent_clip_mode='smooth'` before stacking deep (plan_2026-05-13_d256b568).
- **A `Dense(channels >= 16)` embed in front of `LearnableNeuralCircuit` is required when feeding raw bit vectors.** Without it, sigmoid in depth-0 saturates immediately and the circuit gets no gradient signal (plan_2026-05-13_d256b568).
- **`to_symbolic()` is a genuine interpretability readout, not decoration.** XOR emerged as depth-1 dominant op for parity after training — the symbolic walker reflects what the network actually learned. Empirical validation is possible and should be done (plan_2026-05-13_d256b568).
- **Anchored design choices survive deep-review unchanged when EXPLORE checks LESSONS first.** Re-litigating an anchored choice without new evidence wastes an iteration (plan_3a2f1d23 vs L38+L44).
- **DARTS-style per-channel selection is near-zero-blast-radius when defaulted off.** Default `'global'` keeps bit-exact back-compat (plan_3a2f1d23 C3).
- **Phased risk-ordered EXECUTE (A: LOW correctness, B: MED features, C: HIGH arch, D: NET-NEW) closes 16+ steps single-pass when EXPLORE empirically pre-verified every critical claim** (plan_3a2f1d23).
- **Hard-extraction via `LARGE × one_hot(argmax)` on `operation_weights` is a clean faithfulness probe.** Snapshot + mutate + eval + restore — bit-exact reversibility verified by post-restore diff < 1e-6. The pattern bypasses callback ordering issues and is simpler than a Keras callback approach (plan_2026-05-13_25774a34 D-002).
- **`LearnableNeuralCircuit.to_symbolic` is empirically faithful at K=6.** Hard-extracted models match soft accuracy on every scalar task (Δ=0.000) and within 0.007 on the multi-output task. This validates the interpretability claim with a real number (plan_2026-05-13_25774a34).
- **A ~600-LOC single-file benchmark runner is viable when (a) components share one CSV schema, (b) only one entry point exists, (c) tests cover sub-functions individually.** The coupling is real but not painful at this scale (plan_2026-05-13_25774a34 D-001).
- **Wall-clock predictions on tiny boolean tasks are systematically too pessimistic by 5-8×.** Early-stop at val_accuracy=1.000 in 20-30 epochs kills runs fast; predicted 15-22 min, actual 3m 38s (plan_2026-05-13_25774a34).
- **Saturation is a validity threat in benchmarks.** When all baselines hit 1.000, the "competitive" claim is weak. Comparisons need tasks where models score 0.85-0.95 to discriminate. Design harder tasks or use bigger K when circuit vs MLP comparisons matter (plan_2026-05-13_25774a34).
- **Pre-mortem scenarios covering the honest-negative case absorb it gracefully.** shift_xor circuit (0.964) < mlp_large (0.992) — flagged in plan, reported honestly in report.md. No scope creep (plan_2026-05-13_25774a34).
- **Pre-Mortem falsification triggers route fix-attempt-2 to a rule refinement, not a plan abort.** plan_798d3a60 Scenario 4 ("circuit_attributions ratio > 2x on parity_k6") fired at step-5 fix-attempt-1 (gradient×input gave 2.85x ratio because random Dense-embedding weights don't marginalize at single samples). Fix-attempt-2 swapped to integrated gradients (Sundararajan et al. 2017) inside the leash budget; oracle then passed at < 2.0. Falsification was correct (rule was flawed) — the resolution was a documented in-leash refinement, anchored as D-002.
- **Hard-extraction Δ at non-saturation is essentially zero on `LearnableNeuralCircuit` band-entry checkpoints across MNIST, CIFAR-10, mux_11bit, parity_k8, random_dnf_8input_4term — confirmed Δ ≤ 0.016 in every case** (plan_2026-05-13_798d3a60). The circuit operates in essentially-discrete mode at partial convergence; hardening the inner ops loses no accuracy. This is the strong-positive Pre-Mortem Scenario 3 outcome, not a failure to discriminate. Future plans probing the soft↔hard gap should design tasks/training that DELIBERATELY keep the circuit in a mixed regime if they want a non-zero Δ to compare.
- **Conv-stem + LearnableNeuralCircuit on CIFAR-10 enters val_acc=0.5 in 1 epoch on a 4090.** Plan-time wall-clock prediction of 6h was 240× too pessimistic for the band-entry checkpoint workflow. When the target is band-checkpoint only (not full saturation), budget per-task at minutes, not hours.
- **Verbose third-party logging from SHAP and LIME floods training script logs.** SHAP's `_kernel.py` INFO messages emit one block per `shap_values` call (potentially thousands). Either configure the third-party logger to WARNING, or accept the log noise — but at REFLECT, grep only your own trainer's log prefixes to find real signals.
- **The user-supplied "wall-clock budget cap" override pattern works.** When the orchestrator's autonomous user-pre-approved EXECUTE loop hits an unexpectedly-slow attribution sweep, the right move is to kill, reduce hyperparameters (num_attr_samples 24→8, lime_num_samples 5000→200, shap_nsamples 256→32) and re-launch — not to extend the timeout indefinitely. The reduced run still answered the headline question (3/3 tasks faithfulness metrics in 6 minutes) within plan SCs (plan_2026-05-13_798d3a60).
- **Load-bearing scorers converge in 1 iteration when a deterministic self-roundtrip test gates Step-3 BEFORE any training.** The 28-test harness for `rule_recovery.py` caught zero bugs at gate time — but its EXISTENCE eliminated the failure mode that would have wrecked the experiment. Pattern: "load-bearing scientific component" → "self-roundtrip + analytically-derived count tests" → "all green" before model training opens (plan_2026-05-14_e26eede2 step 3).
- **The Monks UCI test set IS the full 432-config categorical enumeration.** Canonical Thrun split convention. Useful structural fact for any future Monks-style benchmark (plan_2026-05-14_e26eede2 step 4).
- **`mlp_matched` (Dense(h)→Dense(h)→sigmoid with h sized to circuit param count) is a brutally strong baseline on small categorical one-hot inputs.** ~3K params yields 0.999/0.998/0.966 on Monks-1/2/3 — within noise of LearnableNeuralCircuit. Use mlp_matched as a *floor*, not a ceiling, when claiming inductive-bias wins on tiny tabular data (plan_2026-05-14_e26eede2).
- **Hard-extraction faithfulness at convergence (Δ_hard_soft ≈ 0) holds on REAL UCI data, not just synthetic boolean tasks.** Monks Δ = -0.001 / -0.002 / +0.000 across 3 problems. The L64 result transfers cleanly from synthetic to UCI (plan_2026-05-14_e26eede2).
- **A ">5pt margin on >=2/3 tasks" comparative criterion can FAIL while the underlying capability claim PASSES.** Frame interpretability-positive plans around the *capability* axis (does the model recover the truth?) rather than the comparative axis (does the model win the benchmark?) — they answer different questions. Honest-negative on the comparative + positive on the capability is a publishable framing (plan_2026-05-14_e26eede2 D-004).
- **`keras.utils.set_random_seed(seed)` is the canonical multi-source RNG primitive.** Seeds Python `random` + NumPy + TF + Keras backend simultaneously. If a training script already calls this in `main()` before any dataset/model construction, a multi-seed sweep needs ZERO library/script edits — just a subprocess driver. Verify the call exists at source level (grep) before assuming you need to add seed-plumbing code (plan_2026-05-14_9c6387a3 F1).
- **Multi-seed sweep design: subprocess-per-seed beats in-process import.** Clean TF/Keras init eliminates cross-seed state contamination; ~3-5s startup × N seeds is negligible against full training time. Pattern: one orchestrator script that runs `subprocess.run([..., '--seed', S, '--out-dir', f'.../seed_{S}/'], timeout=cap_s)` per seed, then globs the per-seed CSVs and concats with an injected `seed` column. Pure-function stats module (mean/std, bootstrap CI, paired permutation) handles aggregation; unit-test the stats module against known oracle cases (plan_2026-05-14_9c6387a3 D-001).
- **At n=5, the paired sign-flip permutation test is the correct non-parametric inference for headline gap comparisons.** Only 2^5 = 32 unique sign patterns; B=10000 Monte Carlo sample with Phipson-Smyth +1/+1 correction. Worst-case p-resolution ~0.03 (effectively complete enumeration). Welch t requires normality which 5 samples cannot validate. CRITICAL: handle degenerate all-zero-diffs case explicitly (return p=1.0) — saturation is real in these benchmarks (plan_2026-05-14_9c6387a3 D-002, D-004).
- **For "delta near zero" metrics (faithfulness Δ_hard), `std > |mean|` is the expected outcome, not an alarm.** When the true mean IS approximately zero, the std/|mean| ratio is structurally large. Report as `0 ± std` or `[ci_lo, ci_hi]` around zero, NOT as a bare mean. The high-variance flag is still useful to catch the failure mode where mean is non-zero but std swamps it (e.g. LIME stability on parity_k8: -0.032 ± 0.297 is a genuine "this metric is uninformative on this task") (plan_2026-05-14_9c6387a3).
- **Headline benchmark claim "circuit beats MLP" can numerically hold (~1.4 pp gap) yet fail statistical significance at n=5.** When std-of-paired-diff exceeds half the effect size, n=5 cannot discriminate. To rescue such a claim, you need either (a) n>=15-20 seeds, OR (b) a harder benchmark where the effect exceeds 5-10 pp. Reframe to "interpretable AND competitive within noise" rather than "interpretable AND superior" (plan_2026-05-14_9c6387a3).
- **Pure-stats modules unit-test trivially when written as functions over numpy arrays with explicit `rng: np.random.Generator` injection.** 30 tests / 229 LOC pinned: known-value oracle cases (`mean_std([1..5])=3, sqrt(2.5)`), determinism (same seed -> same result), zero-variance degenerate cases, two-sided symmetry (`paired(a,b) p == paired(b,a) p`), NaN tolerance, arg validation. All pass in 0.2s on the new test file alone (plan_2026-05-14_9c6387a3 step-2).

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite is ~1.5h and is the pre-push hook. Scope pytest to changed modules only.
- **Use `MPLBACKEND=Agg` for any training-script invocation.** Headless servers crash on the default matplotlib backend.
- **Use `dl_techniques.utils.logger` only — no `print` calls.** Grep `print(` in modified files.
- **Single GPU jobs only.** Never spawn parallel training runs.
- **Pin GPU via shell env, not `setup_gpu(args.gpu)` alone.** TF initialises on `import tensorflow as tf`. Use `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train.<...>`.
- **GPU-OOM during EXPLORE/REFLECT verification: fall back to CPU with `CUDA_VISIBLE_DEVICES=""`.**
- **CLI flag uniformity across sibling training scripts is an explicit design property.** Once one consumer adopts a flag, all siblings should.
- **For packed CLM, use `train.common.nlp.estimate_clm_steps_per_epoch(...)`, not `num_articles // batch_size`.**
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken.** Use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.
- **AdamW double weight decay is a footgun.** Never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)`.
- **`CliffordNetBlock` is dim-preserving.** Hierarchical CliffordNets must use explicit downsamplers.
- **CliffordNet does not tolerate aggressive patchify stems.** patch=4 stems collapse spatial structure. Stem stride <= 2 before first block stack.
- **GPU 1 (RTX 4070 12GB) caps CliffordNet training at batch 64 for hierarchical variants.**
- **`DepthwiseConv2D` on TF GPU rejects asymmetric strides.** Substitute `Conv2D(filters=channels, kernel_size=(1,k), strides=(1,s), groups=channels)`.
- **For causal `(H=1, W=seq_len)` Clifford blocks, pool surface MUST be restricted to `avg`/`max` only.**
- **Pool -> nearest-upsample round trip is NOT causal at fine resolution.** Mitigate via `_causal_upsample(x, s)` right-shift by `s-1`.
- **`.keras` save/load on GPU under fp32 has reduction-order noise ~5e-5 for U-Net-shaped models.** Default tolerance 1e-4.
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`.

## CLM training metrics & probes

- **Metric *math* belongs in `dl_techniques.metrics/`; per-trainer-shape *metric list* belongs in `train.common.nlp`.**
- **`build_clm_metrics()` MUST return fresh metric instances on every call.**
- **For new CLM metrics, set `dtype="float32"` explicitly on `add_weight` accumulators.** Forward-compat with `mixed_float16`.
- **Probe-time aggregate metrics ride on `_post_generate_hook`, not a new probe subclass.**
- **Keras 3 dict-keyed metrics silently no-op on subclassed models unless `output_names` is set.** Call `prepare_dict_keyed_compile(model, output_key="logits")` before `model.compile`.
- **Verification gate for trainer metric wiring MUST include a real `model.fit` call.** grep + py_compile + shape-only unit tests are insufficient.
- **`keras.backend.count_params` was removed in Keras 3.** Use `int(np.prod(w.shape))`.
- **`pad_token_id` mismatch is the canonical silent bug for encoder-wrapper integrations.** Model defaults to `0`; tokenizer (tiktoken cl100k_base) uses `100266`.

## Multi-optimizer / custom train_step

- **Two-optimizer differential-LR pattern: register only ONE optimizer with `super().compile(optimizer=...)`, apply the second manually inside `train_step`.**
- **`current_phase` and `_global_step` counters: `add_weight(trainable=False, dtype="float32")`** — int32 caused CPU/GPU device-placement failures. Do NOT serialize via `get_config`.
- **Carry-based ACT-loop trainers use per-step `optimizer.apply_gradients` inside the unroll**, NOT per-unroll.

## Keras serialization & ops.* tracing

- **`ops.pad` with dynamic paddings does not trace under XLA on TF 2.18.** Substitute `ops.concatenate([x, ops.zeros(...)])`.
- **`ops.one_hot(num_classes=M)` and `ops.top_k(k=K)` need static Python ints.** Bake on `self` at `__init__`.
- **Aux-loss `add_loss` calls must be gated by `training=True`.**
- **`keras.ops.cond` traces BOTH branches under `tf.function` on TF backend Keras 3.** Multiply-by-zero is the simpler-equivalent pattern.
- **`keras.random.shuffle(keras.ops.arange(N))[:k]` is the backend-agnostic replacement for `tf.random.shuffle`.**
- **For Keras-2 -> Keras-3 `train_step` rewrites, `compute_loss(x, y, y_pred)` adds `self.losses` internally** — the manual `if self.losses: loss = loss + ops.sum(self.losses)` block is redundant.
- **`.keras` save/load on a custom-subclass Model that wraps another `keras.Model`: use `save_own_variables`/`load_own_variables` overrides delegating into numbered subkeys.**
- **A bug masked by a crash earlier in the call path becomes visible the moment the crash is fixed.** Audit the same block for graph-mode bugs sitting behind a fixed crash.
- **`compile=False` on `load_model(...)` is a workaround, not a fix, for `WrappedLoss.get_config()` round-trip bugs.**
- **Bare `@register_keras_serializable()` ties registered key to `__module__`.** Do NOT relocate classes between modules.

## Keras 3 layer / model composition gotchas

- **Frozen tensor state inside a layer MUST live in `add_weight(trainable=False, ...)` OR stored as numpy on `self` — never as a plain tensor created in `build()`.**
- **`compute_output_spec` declares symbolic output dtype independently from `call()`'s runtime return.**
- **Probability-distribution layers under fp16 mixed precision: clip+sigmoid in fp16 underflows.** Cast logits to fp32 BEFORE sigmoid+clip.
- **`add_weight` variables behave differently under AMP than activations.** Variables stay fp32 but get autocast on read.
- **Custom Keras `Initializer` subclass beats split-weight refactors when test count is locked.**
- **For symmetric U-Net spatial-divisibility: `padding='same'` on encoder pool alone is insufficient.** Fix via `Cropping2D`/`Resizing` per skip OR explicit `ValueError` at model boundary.

## Layer / model testing patterns

- **Test class structure for new layers: mirror existing sibling classes in the same file.**
- **`test_residual_identity_at_init`** (gamma~0 -> output~input) is the cheapest sanity check that residual wiring is correct.
- **Always include a `test_save_load_roundtrip` that wraps the layer inside a `keras.Model`.**
- **Causality test recipe for `(H=1, W=seq_len)` models:** (1) perturb-last, (2) perturb-middle depth>=2, (3) non-multiple `seq_len % total_stride`, (4) `use_global_context=True`. Tolerance `< 1e-5`.
- **For multi-flag plumbing, verify the default-off path is bit-exactly preserved** by running ALL pre-existing tests unchanged.
- **For redesigned aux-losses (e.g. InfoNCE), a non-degeneracy test is stronger than a finiteness test.**
- **Sigmoid-based "soft signals" do NOT form a soft partition.** Per-channel `[0, 1]` is the genuine contract.
- **tf2onnx 1.16.1 cannot convert `tf.while_loop` produced by `keras.ops.scan`.** Workaround: monkeypatch `call` to a Python-unrolled `for t in range(T)` body — input length must be static (plan_5f0e087c D-004).
- **Compare ONNX vs Keras through the SAME unrolled path or you measure float32 chaos, not export fidelity.**
- **TF backend of `keras.ops.scan` is stricter than other backends.**
- **"Bit-equivalent" claims must distinguish division-bearing branches from division-free.** Chained divisions diverge ~1e-5 at T~512.
- **TimeSeriesGenerator returns `(T, 1)` arrays.** Squeeze trailing singleton before passing to `tf.data.Dataset.from_generator`.
- **Review-driven cleanup plans close single-pass when structural refactors are reframed as ADDITIVE (alias + factory) rather than replacement** (plan_8c1dc6fd).
- **Matryoshka head extension: post-`head_norm` pivot tensor as single source; `primary head name stays "logits"`; smaller widths become `f"logits_w{w}"`** (plan_13c70aed D-002).
