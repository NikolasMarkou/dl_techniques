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
- **Verify reviewer claims empirically before applying fixes from a "deep review".** Reviews contain false positives. For each "critical" claim about API behavior, run a 3-line empirical check first. Lock empirically-correct behavior in a test. Verify BOTH directions — false-positives drop AND false-negatives confirm (plan_e6309bd5).
- **A correctly-flagged bug can mask a worse adjacent bug.** When a reviewer flags one mixed-precision/dtype/serialization issue, sweep the whole category before merging the narrow fix (plan_3c3ed037).
- **Verify "every site does X" with grep BEFORE PLAN, not during EXECUTE.** Textual claims that look like inventory facts have a way of being half-true.
- **Per-trainer "1-line bind" of a callback hook is actually a 4-line delta** (`probe_cb = X(...); probe_cb._hook = fn; callbacks.append(probe_cb)`).
- **Greenfield model packages with explicit blueprint converge in a single iteration when EXPLORE pins each algorithmic shape to a concrete Keras 3 idiom BEFORE PLAN** (plan_146ae899).
- **Audit-driven full-coverage plans converge in one iteration when (a) the audit doc carries file:line references and prescribed fix shape, AND (b) EXPLORE pre-resolves design choices for non-trivial fixes in a "design notes" finding before PLAN** (plan_0f39a086).
- **Falsification triggers must pair the symptom with a triage check.** plan_f2d29729 Scenario B fired ("val_acc<0.30 @ epoch 10") and the plan-prescribed action was an LR sweep — wrong for the actual data-pipeline bug. Pair each trigger with quick diagnostics.
- **REV-1 (plan revision after PC-PLAN approval) is the right protocol move when user expands scope mid-plan with Revert-First constraint.** Re-emit PC-PLAN with bounded steps (plan_44694bc9).
- **Bundling related EXECUTE steps into one commit is sound when one verification command validates all of them and they share a single file** (plan_54e6e303).
- **Reference-doc / benchmark-snapshot plans close single-pass via one research agent when EXPLORE pre-enumerates the coverage inventory.** Recipe: (a) findings file lists every entry; (b) one agent briefed with the inventory + sibling template; (c) REFLECT runs `wc -l` + `grep -c '^## '` + style checks + 5 spot-checks (plan_16ac1621).
- **Line-count caps on greenfield reference docs are sanity guards, not quality gates.** "Exceeds upper bound" should trigger a density check (lines per entry) before being treated as bloat.
- **LOC overshoot from docstrings + per-flag argparse `help=` text is acceptable when no new abstractions are introduced beyond planned.** Track abstractions and blast-radius instead of raw LOC at REFLECT.
- **When the EXECUTE pipeline reports unexpected test failures, baseline-diff BEFORE calling regression.** `git diff <cp-000-hash>..HEAD -- <failing-paths>` empty -> pre-existing -> log in summary.md "Out of Scope", do not block CLOSE (plan_0090b0b8).
- **Smoke != correctness — smoke tests must assert distributional invariants across tf.data + preprocessing boundaries.** Canonical: `_assert_train_val_distribution_match(train_ds, val_ds, mean_tol=0.5, std_ratio_tol=0.5)` called immediately before `model.fit`. Reference: `src/train/vit/train_vit.py` (plan_f2d29729 D-007).
- **"Smooth-train + cliff-val + sub-random val" is the fingerprint of a train/val preprocessing divergence — never bundle hparam sweeps with data-pipeline fixes.** Fix data pipeline first; sweep only on a clean baseline (plan_f2d29729 D-006/D-007).
- **A trainer's unguarded "TRAINING COMPLETED SUCCESSFULLY" log is a known footgun.** Guard on `best_val_acc >= max(2/num_classes, threshold)` AND `early_stop_epoch >= 0.5 * total_epochs`. Existing resnet trainer at `train_resnet.py:579` still unguarded — backport opportunistically (plan_f2d29729 D-007).
- **Pure-additive integration-alignment plans (`factory.py` + populated `__init__.py` + `README.md` + small ghost-constraint relax) on a self-contained sibling-pattern-misaligned layer package close in 1 iteration / 0 fix attempts** when EXPLORE pre-counts (a) sibling factory'd packages for the precedent, (b) every non-test consumer of the target package, (c) `register_keras_serializable` decorator shape (bare vs `package=`). Recipe: factory mirroring nearest sibling -> populated `__init__` with `__all__` -> ghost-constraint relax in-place (preserve `__module__` keys) -> cosmetic fixes -> README (Math/Classes/Factory/Integration/Limitations/Examples) -> factory tests (smoke + registry contents + round-trip via factory) -> rank-relax tests for both rank-2 and rank-3 -> scoped pytest. plan_2aaad563.
- **A strict-rank `build()` precondition tighter than what `call()` requires is a ghost constraint.** Confirm by reading `call()` end-to-end (no rank-dependent slicing, all ops driven by `ops.shape(inputs)` or `expand_dims`) before proposing the relax. Pre-existing tests that assert on the strict-rank error message must be updated as part of the same plan, not deferred — Failure Modes row should pre-flag this.
- **Unary-input degeneracy in DARTS-style "softmax over primitives" layers** (subtract->0, divide->1, binary-logic gates degenerate when caller passes a single tensor as both operands) is a documented footgun, not a bug to fix in code. Document in README Limitations and move on.

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite is ~1.5h and is the pre-push hook. Scope pytest to changed modules only.
- **Use `MPLBACKEND=Agg` for any training-script invocation.** Headless servers crash on the default matplotlib backend.
- **Use `dl_techniques.utils.logger` only — no `print` calls.** Grep `print(` in modified files.
- **Single GPU jobs only.** Never spawn parallel training runs.
- **Pin GPU via shell env, not `setup_gpu(args.gpu)` alone.** TF initialises on `import tensorflow as tf`. Use `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train.<...>`.
- **GPU-OOM during EXPLORE/REFLECT verification: fall back to CPU with `CUDA_VISIBLE_DEVICES=""`.**
- **CLI flag uniformity across sibling training scripts is an explicit design property.** Once one consumer adopts a flag, all siblings should.
- **For packed CLM, `steps_per_epoch = num_articles // batch_size` is wrong.** Use `train.common.nlp.estimate_clm_steps_per_epoch(...)`.
- **For Wikipedia CLM, the default `min_article_length=0` is correct** (packed pipeline concatenates every token).
- **Smoke recipe for `train_clip.py`-class scripts:** `--synthetic --max-train-samples 64 --max-val-samples 32 --epochs 1 --batch-size 8`.
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken.** Use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.
- **AdamW double weight decay is a footgun.** Never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)` — pick one. Branch shape: AdamW -> optimizer-builder WD only, `kernel_regularizer=None`; SGD -> no opt WD, `kernel_regularizer=L2(wd)`. Exclude `logit_scale` from AdamW WD in CLIP-style training.
- **`CliffordNetBlock` is dim-preserving.** Hierarchical CliffordNets must use explicit downsamplers (`CliffordNetBlockDS` vision, `CausalCliffordNetBlockDSv2` language causal).
- **CliffordNet does not tolerate aggressive patchify stems.** patch=4 stems collapse spatial structure (~6 pp val_acc loss on CIFAR-100). Stem stride <= 2 before first block stack.
- **GPU 1 (RTX 4070 12GB) caps CliffordNet training at batch 64 for hierarchical variants.** Halve batch when switching from GPU 0.
- **For shape-preserving Clifford block variants: pool BEFORE the stream split.** Geometric product is element-wise on `(z_det, z_ctx)`.
- **`DepthwiseConv2D` on TF GPU rejects asymmetric strides.** Substitute `Conv2D(filters=channels, kernel_size=(1,k), strides=(1,s), groups=channels)`.
- **For causal `(H=1, W=seq_len)` Clifford blocks, pool surface MUST be restricted to `avg`/`max` only.** `padding="same"` IS causal.
- **Pool -> nearest-upsample round trip is NOT causal at fine resolution.** Mitigate via `_causal_upsample(x, s)` right-shift by `s-1`.
- **`.keras` save/load on GPU under fp32 has reduction-order noise ~5e-5 for U-Net-shaped models.** Default tolerance 1e-4, not 1e-5.
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`.
- **`DepthAnything` is full-feature** — real ViT encoder, DPTDecoder, weight-shared frozen teacher, semi-supervised train_step, on-step EMA, `from_pretrained_encoder`. Save/load via `save_own_variables`/`load_own_variables` override on the outer class.

## CLM training metrics & probes

- **Metric *math* belongs in `dl_techniques.metrics/`; per-trainer-shape *metric list* belongs in `train.common.nlp`.** Math: `BitsPerToken` / `BitsPerCharacter` in `metrics/llm_metrics.py`. List: `build_clm_metrics(encoding_name, ignore_index, chars_per_token=None)` in `train/common/nlp.py`.
- **`build_clm_metrics()` MUST return fresh metric instances on every call.** Test: `a = build(); b = build(); assert a[i] is not b[i]`.
- **For new CLM metrics, set `dtype="float32"` explicitly on `add_weight` accumulators.** Forward-compat with `mixed_float16`.
- **Probe-time aggregate metrics ride on `_post_generate_hook`, not a new probe subclass.** Free function bound by `probe_cb._post_generate_hook = fn`. Make schema-tolerant.
- **Keras 3 dict-keyed metrics silently no-op on subclassed models unless `output_names` is set.** Call `prepare_dict_keyed_compile(model, output_key="logits")` before `model.compile`.
- **Verification gate for trainer metric wiring MUST include a real `model.fit` call.** grep + py_compile + shape-only unit tests are insufficient.
- **`keras.backend.count_params` was removed in Keras 3.** Use `int(np.prod(w.shape))`.
- **`pad_token_id` mismatch is the canonical silent bug for encoder-wrapper integrations.** Model defaults to `0`; tokenizer (tiktoken cl100k_base) uses `100266`. Trainer MUST pass `pad_token_id=config.pad_token_id`.

## Multi-optimizer / custom train_step

- **Two-optimizer differential-LR pattern: register only ONE optimizer with `super().compile(optimizer=...)`, apply the second manually inside `train_step`.** Variable routing via name-prefix split. Use leading-component prefix match `name.split('/')[0].startswith(p)`, NOT substring.
- **Custom `train_step` is compatible with `prepare_dict_keyed_compile` on subclassed `keras.Model`.** Inside `train_step`: extract `y_pred["logits"]` if dict.
- **`current_phase` and `_global_step` counters: `add_weight(trainable=False, dtype="float32")`** — int32 caused CPU/GPU device-placement failures. Variable IS the state — do NOT serialize via `get_config`.
- **Carry-based ACT-loop trainers (HRM, TRM) use per-step `optimizer.apply_gradients` inside the unroll**, NOT per-unroll, NOT `model.fit`. `stop_gradient` on inner carry severs cross-step gradient flow -> per-step is equivalent and cheaper.

## Keras serialization & ops.* tracing

- **`ops.pad` with dynamic paddings does not trace under XLA on TF 2.18.** Substitute `ops.concatenate([x, ops.zeros(...)])`. Pad length must be a static int from `__init__`.
- **`ops.one_hot(num_classes=M)` and `ops.top_k(k=K)` need static Python ints.** Bake on `self` at `__init__`.
- **Aux-loss `add_loss` calls must be gated by `training=True`.** Further-gate on `if training and any(self.enable_*)`.
- **`keras.ops.cond` traces BOTH branches under `tf.function` on TF backend Keras 3.** Multiply-by-zero is the simpler-equivalent pattern.
- **`keras.random.shuffle(keras.ops.arange(N))[:k]` is the backend-agnostic replacement for `tf.random.shuffle`.** No `import tensorflow as _tf` for randomness in library code.
- **For Keras-2 -> Keras-3 `train_step` rewrites, the canonical pattern is *shorter*.** `compute_loss(x, y, y_pred)` adds `self.losses` internally — the manual `if self.losses: loss = loss + ops.sum(self.losses)` block is redundant.
- **`.keras` save/load on a custom-subclass Model that wraps another `keras.Model`: use `save_own_variables`/`load_own_variables` overrides on the outer class** delegating into numbered subkeys. Diagnostic: dump `[(w.path, hash(w.numpy().tobytes())) for w in model.weights]` before/after.
- **A bug masked by a crash earlier in the call path becomes visible the moment the crash is fixed.** Audit the same block for graph-mode bugs sitting behind a fixed crash.
- **`compile=False` on `load_model(...)` is a workaround, not a fix, for `WrappedLoss.get_config()` round-trip bugs.** Treat a `compile=False` comment in a trainer as a high-signal pointer to a custom-loss serialization bug. Canonical fix shape: `dl_techniques.losses.segmentation_wrapper_loss.SegmentationWrapperLoss`.
- **One-loss-per-module convention.** Sibling-loss survey before PLAN: file `<topic>_loss.py`, class `<Topic>Loss`, bare `@register_keras_serializable()`, test file mirrors module name.
- **Bare `@register_keras_serializable()` ties registered key to `__module__`.** Do NOT relocate classes between modules — would invalidate existing `.keras` archives. Verify zero in-repo `.keras` fixtures before allowing relocation.

## Keras 3 layer / model composition gotchas

- **Frozen tensor state inside a layer composed into a `keras.Model` MUST live in `add_weight(trainable=False, ...)` OR be stored as numpy on `self` and converted inside `call()` — never as a plain tensor created in `build()`.** Keras `compute_output_spec` runs `build()` in a transient `FuncGraph`; tensors made there get captured and dereferenced as dead tensors at runtime. Always include a "wrap in `keras.Model` + save round-trip" test.
- **`compute_output_spec` declares symbolic output dtype independently from `call()`'s runtime return.** Override when a layer must return a different dtype than `compute_dtype`.
- **Probability-distribution layers under fp16 mixed precision: clip+sigmoid in fp16 underflows.** Cast logits to fp32 BEFORE sigmoid+clip.
- **`add_weight` variables behave differently under AMP than activations.** Variables stay fp32 but get autocast on read. Add `@pytest.mark.parametrize("policy", ["mixed_float16","mixed_bfloat16"])` end-to-end tests.
- **Custom Keras `Initializer` subclass beats split-weight refactors when test count is locked.** Replacing `np.random` in `build()` via two `add_weight` calls breaks tests asserting `len(layer.trainable_variables) == N`. A `class _MyInit(keras.initializers.Initializer)` keeps topology unchanged AND is serializable.
- **For symmetric U-Net spatial-divisibility: `padding='same'` on encoder pool alone is insufficient** — `Conv2DTranspose(strides=2, padding='same')` always emits exactly `2*H_in`; encoder ceil-divides odd dims, decoder cannot un-ceil. Fix via `Cropping2D`/`Resizing` per skip OR explicit `ValueError` at model boundary. Spatial validation in `call()`, NOT `build()`.

## Layer / model testing patterns

- **Test class structure for new layers: mirror existing sibling classes in the same file.**
- **`test_residual_identity_at_init`** (gamma~0 -> output~input) is the cheapest sanity check that residual wiring is correct.
- **For pool-choice variants (avg vs max), verify they produce different outputs** with `layer_scale_init` near zero.
- **Always include a `test_save_load_roundtrip` that wraps the layer inside a `keras.Model`** and round-trips through `.keras` save/load.
- **Causality test recipe for `(H=1, W=seq_len)` models:** (1) perturb-last, (2) perturb-middle depth>=2, (3) non-multiple `seq_len % total_stride`, (4) `use_global_context=True` if applicable. Tolerance `< 1e-5`.
- **For new metric classes (BPT/BPC/etc.), test against a reference implementation by identity** — e.g. `BitsPerToken == log2(Perplexity)` to within 1e-5.
- **For multi-flag plumbing, verify the default-off path is bit-exactly preserved** by running ALL pre-existing tests unchanged.
- **For redesigned aux-losses (e.g. InfoNCE), a non-degeneracy test is stronger than a finiteness test.** Assert `|cos(grad, anchor)| < threshold` to catch gradient-direction collapse.
- **Recipe-replication plans (sibling N of "tree_transformer iter-1 recipe": narrow `from_variant` except, `_download_weights` raises `NotImplementedError`, package `__init__` trim, 3 lock-in tests) close in 1 iteration / 0 fix attempts when EXPLORE produces F-001 callers audit + F-002 template-mapping.** Public-surface count tracks actual sibling factories: 4 (cliffordnet), 3 (bert/tree_transformer with_head analog), 2 (gpt2 — LM head intrinsic).
- **File-split refactors (relocate sub-layer classes from `model.py` to `components.py`) close single-pass when EXPLORE produces (a) callers audit of deep-import sites and (b) confirms `@register_keras_serializable` is bare AND no public `.keras` archives exist** (so `__module__`-key drift on registered classes is safe). Recipe: create `components.py` verbatim -> delete bodies + re-export from `.model` -> importlib smoke -> 3 lock-in tests (plan_46ecfa0b). **Extension — "move into existing factory'd package + register":** pure relocation + factory wiring across N<=7 consumers closes single-pass with consumers-enumerated grep + bare `@register_keras_serializable()` + zero in-repo `.keras` fixtures (plan_03176394). **Extension — "merge sibling packages atomically (no shim)":** package-merge via `git mv` + N-caller rewrite + delete-old-dir, single-pass when EXPLORE (a) repo-greps every `layers.<oldpkg>` caller, (b) confirms bare decorators, (c) confirms zero in-repo `.keras` fixtures (plan_a1c9a52d).
- **Bundle `code-review + README + trainer scaffolding` as one additive plan only when review is informational (no model.py edits).** Surface findings as README Limitations + trainer docstrings + REFLECT chat — never in-plan code fixes. Decouples review's blast radius (zero — prose) from code-fix risk (plan_94b9fab5, plan_ebb5fac5).
- **Code-fix companion plan to a prior informational review closes in single iteration when EXPLORE explicitly enumerates ghost-constraints from the review** (plan_86f14c6e).
- **Pattern-3 NLP MLM trainer for a new encoder closes in single iteration / 0 fix attempts when EXPLORE pins three shapes BEFORE PLAN: (1) data pipeline parent (HF-streaming vs `load_wikipedia_train_val`); (2) LR-schedule shape (epoch-shaped vs step-shaped `WarmupSchedule(CosineDecay)`); (3) two-stage launch — synchronous ~5-min `--smoke` loss-decline gate THEN backgrounded real run.** Trainer is structurally a hybrid of two parents. `jit_compile=False` default for new architectures. Plan can CLOSE while real run is alive in background — SC scope is launch-success (plan_6a2cd5b3).
- **Bidirectional sibling of an existing causal U-Net LM closes single iteration when the layer library already ships non-causal block siblings AND EXPLORE end-to-end-reads the causal precedent file** (plan_632605aa).
- **In-scope regression catches that introduce a new test or expand an existing lock-in test should be tracked as numbered sub-steps (e.g. step-5b) in progress.md + state.md, NOT folded into the next step's commit.** Preserves auditability and keeps "0 fix attempts" accurate.
- **Sigmoid-based "soft signals" do NOT form a soft partition.** Per-channel `[0, 1]` is the genuine contract; hard partition (`above + below + between == 1`) is recovered only at inference. Drop sum-<=1 assertions in tests.
- **tf2onnx 1.16.1 cannot convert `tf.while_loop` produced by `keras.ops.scan`.** Workaround: wrap `model.export(format="onnx")` AND the verifier's Keras forward pass in a `contextmanager` that monkeypatches `call` to a Python-unrolled `for t in range(T)` body — input length must be static. Pattern: `src/train/adaptive_ema/export.py:_ema_unrolled_for_export` (plan_5f0e087c D-004).
- **Compare ONNX vs Keras through the SAME unrolled path or you measure float32 chaos, not export fidelity.** Apply the unroll patch on both sides; `rtol=atol=1e-4` achievable.
- **TF backend of `keras.ops.scan` is stricter than other backends.** (1) Per-iteration output `y` must match the carry in shape/dtype; (2) tuple carries are rejected. Compose state into a single flat tensor or precompute per-step constants via `xs` lists.
- **"Bit-equivalent" claims must distinguish division-bearing branches from division-free.** `keras.ops.scan` matches an unrolled Python `for` loop bit-exactly for pure mul/add; chained divisions diverge ~1e-5 at T~512. Set tolerances per branch.
- **TimeSeriesGenerator returns `(T, 1)` arrays.** Squeeze trailing singleton at the target-builder boundary before passing to `tf.data.Dataset.from_generator` with 1D `output_signature`.
- **Review-driven cleanup plans (bugs + structural refactors in one pass) close single-pass when EXPLORE verifies every flagged line empirically AND structural refactors are reframed as ADDITIVE (alias + factory) rather than replacement** (plan_8c1dc6fd).
- **Matryoshka head extension to an existing dict-output CLM closes single iteration when EXPLORE resolves three shapes upfront: post-`head_norm` pivot tensor as single source for all width-sliced heads; `prepare_dict_keyed_compile` extended via `output_keys: Optional[List[str]]` rather than a new abstraction; primary head name stays `"logits"` (SYSTEM.md invariant), smaller widths become `f"logits_w{w}"`, embeddings `f"embedding_w{w}"`.** Width-rule "Power-of-2 anchored, base preserved" is the right default for non-power-of-2 `base_channels` (plan_13c70aed D-002).
