# Lessons Learned
*Cross-plan lessons. Updated and consolidated on close. Max 200 lines — rewrite, don't append forever.*
*Read before any PLAN state. This is institutional memory.*

## Process

- **"Remove X" can mean two different things.** When a user asks to remove a multi-stage / multi-mode surface, distinguish between (a) the *capability* is unwanted and (b) the *naming or layering* is confusing. EXPLORE should classify each piece before PLAN drafts deletion steps.
- **Doc updates belong in the same plan as the code change they describe.** Treating README/CLAUDE.md updates as out-of-scope follow-ups means the plan is technically "complete" while the user-visible behavior is still wrong. Include doc steps from the start.
- **Static grep gates can substitute for runtime probes — but only for purely mechanical surface changes.** Renames with no internal recursion and no out-of-file callers are safe to verify with `grep -n`. Anything that touches behavior needs a runtime smoke test.
- **PLAN -> PLAN cycles are normal.** Better to revise the written plan once after user feedback than to silently re-interpret the goal during EXECUTE.
- **Mechanical pattern-replication across N files closes in a single iteration when EXPLORE per-file diffs the precedent's structural assumptions BEFORE PLAN.** Resist extracting a shared helper for ≤4 sites if extraction crosses package boundaries.
- **Pure additive layer/class work fits a single iteration.** EXPLORE → PLAN → EXECUTE → REFLECT closes in one session when the change is non-destructive and tests are scoped to one file.
- **Sibling-stack model additions: prefer parallel package over factory retrofit when the new component has a different mask shape or unique hyperparameters.** Trade ~150 LOC duplication for zero blast radius on 30+ unrelated models.
- **Orchestrator-crash mid-EXECUTE recovery: trust git log + filesystem, not stale state.md.** Enumerate `git log` since baseline, diff vs plan.md "Files To Modify", re-run pytest, write a "Resume Notes" section in state.md, proceed to REFLECT. Do NOT re-execute completed steps.
- **Plan-time line-count predictions for sibling-class additions underestimate by ~2x.** "Mirror the parent class" tends to count algorithmic body only, missing validation, docstring, `get_config`, and `compute_output_shape`.
- **Plan-time LOC predictions underestimate by 5-10x for plans that touch dtype semantics under mixed precision OR span >3 consumer scripts.** Bug fixes in dtype semantics pull in symbolic-graph plumbing changes (e.g. `compute_output_spec` overrides) that aren't visible at EXPLORE time.
- **Plan-time LOC predictions underestimate by ~1.7x for opportunity-driven additive plans where one feature plumbs a new flag across 4+ layers** (e.g. multi_head_keys across LongTermMemoryBank/WorkingMemoryBank/both controllers/model). Multi-layer plumbing is the dominant factor; predict accordingly.
- **Pre-existing tests can encode bugs as contracts.** When a fix to a library bug causes adjacent tests to fail, read the failing assertions before assuming regression — they may be locking in buggy behavior. Update asserts and anchor the new contract with `# DECISION`.
- **Empirical confirmation of a finding does not separate bug from contract.** A "deep review" agent can confirm a behavior occurs without checking whether that behavior is *enforced by an existing test as a deliberate contract*. Grep the test suite for tests that *lock in* the current behavior.
- **Run the existing test suite as the FIRST step of EXPLORE when the goal is "fix issues found in a review".** All-green tests are the cheapest evidence that some review claims are wrong.
- **Internal-consistency arguments are not falsification.** A claim of the form "X is causal because Y" must have a unit test on X *alone* before it gates an end-to-end design. Add a micro-test on the falsifiable building block as part of the plan.
- **Verify reviewer claims empirically before applying fixes from a "deep review".** Reviews contain false positives. For each "critical" claim about API behavior, run a 3-line empirical check first. Lock empirically-correct behavior in a test.
- **A correctly-flagged bug can mask a worse adjacent bug.** When a reviewer flags one mixed-precision/dtype/serialization issue, sweep the whole category before merging the narrow fix.
- **A correctly-scoped baseline-test failure can be fixed for free inside an unrelated step** when the planned change naturally absorbs the failure (plan_0f39a086: 2 baseline `int32`-on-CPU device-placement failures landed alongside the B5 dict-keyed metric fix when `current_phase` switched to `float32`).
- **Verify "every site does X" with grep BEFORE PLAN, not during EXECUTE.** Textual claims in findings that look like inventory facts have a way of being half-true. EXPLORE-time `grep -rn 'X' src/...` is cheap insurance.
- **Per-trainer "1-line bind" of a callback hook is actually a 4-line delta.** Pattern `callbacks.append(GenerationProbeCallback(...))` must become `probe_cb = GenerationProbeCallback(...); probe_cb._hook = fn; callbacks.append(probe_cb)`.
- **Greenfield model packages with explicit blueprint converge in a single iteration when EXPLORE pins each algorithmic shape to a concrete Keras 3 idiom BEFORE PLAN.** plan_146ae899 closed in 1 iteration / 8 steps because finding F-004 mapped each blueprint section to canonical idioms in the existing codebase. Lesson: produce a "blueprint decomposition" finding that names the file-level precedent for each non-trivial idiom.
- **Audit-driven full-coverage plans converge in one iteration when (a) the audit doc carries file:line references and prescribed fix shape, AND (b) EXPLORE pre-resolves design choices for the non-trivial fixes in a "design notes" finding before PLAN.** plan_0f39a086 fixed 26 audit items in 17 steps / 0 fix attempts because F-003 pre-resolved B2/R1/O4/O7/D2.
- **Falsification signals earn their keep when the riskiest step has a pre-flagged STOP-IF trigger that is checked empirically inside the step.** plan_0f39a086 step 6 (R1) tested `keras.ops.cond` branch-tracing behavior at the start of the step; the trigger fired and the step pivoted to "documented design limitation" without wasted code.
- **For Keras-2 → Keras-3 `train_step` rewrites, the canonical pattern is *shorter* than the original.** `compute_loss(x, y, y_pred)` adds `self.losses` (regularization) internally, so the manual `if self.losses: loss = loss + ops.sum(self.losses)` block is redundant. Deletion-bound LOC must account for this — keeping the redundant block to satisfy a too-tight bound would *double-count* regularization. Pattern source: `dl_techniques/models/masked_language_model/mlm.py:309-343`. plan_2026-05-10_44694bc9 D-004.
- **REV-1 (plan revision after PC-PLAN approval) is the right protocol move when the user expands scope mid-plan with Revert-First constraint.** Re-emitting PC-PLAN with the new steps documented and bounded (e.g. ≤10 LOC patch + smoke verify) is cleaner than silent in-execute scope creep. plan_2026-05-10_44694bc9.
- **A 1-step CPU `model.fit` is the right verification for API-level (`AttributeError`) bugs in `train_step`.** Probe model output shape and match the target — don't assume placeholder encoder/decoder pairs are same-resolution. `CUDA_VISIBLE_DEVICES=""` keeps the smoke off-GPU and serial-safe. plan_2026-05-10_44694bc9 SC-14.
- **When `.keras` save/load on a custom-subclass Model that wraps another `keras.Model` (e.g. ViT) drops a subset of inner weights despite get_config round-trip, the canonical Keras-3 fix is `save_own_variables` / `load_own_variables` overrides on the OUTER class** that delegate persistence of named sub-Models into numbered subkeys of the store. Two cheaper attempts to try first: (1) lazy-build the inner Model in `build()` so it registers after the outer; (2) force-build via dummy `keras.Input` in `__init__`. If both fail, the override is the bounded (≤30 LOC) fix — no need to refactor to Functional. plan_2026-05-10_bd098beb D-004.
- **Diagnostic recipe for sub-Model round-trip mismatches**: dump `[(w.path, hash(w.numpy().tobytes())) for w in model.weights]` before save and after load; the diff identifies exactly which paths are losing values. Equivalent dump on the inner Model alone separates "the inner saves cleanly when standalone" (path-mapping bug, fixable via override) from "the inner can't save at all" (deeper topology bug). plan_2026-05-10_bd098beb D-004.
- **A bug masked by a crash earlier in the call path becomes visible the moment the crash is fixed.** When fixing an API-drift crash (e.g. `keras.ops.random` → `keras.random`), audit the same code block for graph-mode bugs that were sitting behind the crash — `if not symbolic_tensor:` Python-bool branches, dtype mismatches, etc. plan_2026-05-10_bd098beb step 1b.
- **Autonomy-leash hits driven by architectural mismatch can resolve in 1 fix attempt after PIVOT — but only when the user supplies the fix order.** plan_2026-05-10_bd098beb hit the leash on Step 3 SC-6; user-directed PIVOT (D-004) prescribed fix order A→B→C with explicit bounds; fix-A (≤30 LOC override) was sufficient. The leash exists to surface architectural calls; user-provided ordering shrinks them back to bounded edits.

## Codebase-specific (dl_techniques)

- **Do not run `make test` as a regression check.** Full suite is ~1.5h and is the pre-push hook. Scope pytest to changed modules only.
- **Use `MPLBACKEND=Agg` for any training-script invocation.** Headless servers crash on the default matplotlib backend.
- **Use `dl_techniques.utils.logger` only — no `print` calls in library or training code.** Hard convention; grep for `print(` in modified files.
- **Single GPU jobs only.** Never spawn parallel training runs (memory + driver contention).
- **Pin GPU via shell env, not `setup_gpu(args.gpu)` alone.** TF initialises on `import tensorflow as tf`, before `main()` runs `setup_gpu()`. Use `CUDA_VISIBLE_DEVICES=N MPLBACKEND=Agg python -m train.<...>`.
- **GPU-OOM during EXPLORE/REFLECT verification: fall back to CPU with `CUDA_VISIBLE_DEVICES=""`.** Metric ops on `keras.ops.*` are backend-agnostic.
- **CLI flag uniformity across sibling training scripts is an explicit design property.** Once one consumer in a script family adopts a flag, all siblings should.
- **For packed CLM, `steps_per_epoch = num_articles // batch_size` is wrong.** Use `train.common.nlp.estimate_clm_steps_per_epoch(...)`.
- **For Wikipedia CLM, the default `min_article_length=0` is correct.** Packed pipeline concatenates every token regardless of source article length.
- **Smoke recipe for `train_clip.py`-class scripts:** `--synthetic --max-train-samples 64 --max-val-samples 32 --epochs 1 --batch-size 8`.
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken.** Use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.
- **AdamW double weight decay is a footgun.** Never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)` — pick one. Exclude `logit_scale` from AdamW WD in CLIP-style training.
- **`CliffordNetBlock` is dim-preserving.** Hierarchical CliffordNets must use explicit downsamplers (`CliffordNetBlockDS` vision, `CausalCliffordNetBlockDSv2` language causal).
- **CliffordNet does not tolerate aggressive patchify stems.** patch=4 stems collapse spatial structure before any CliffordNetBlock runs (~6 pp val_acc loss on CIFAR-100). Keep stem stride ≤ 2 before the first block stack.
- **GPU 1 (RTX 4070 12GB) caps CliffordNet training at batch 64 for hierarchical variants.** Halve batch when switching from GPU 0.
- **For shape-preserving Clifford block variants: pool BEFORE the stream split.** The geometric product is element-wise on `(z_det, z_ctx)`; their spatial+channel dims must match.
- **`DepthwiseConv2D` on TF GPU rejects asymmetric strides.** Substitute `Conv2D(filters=channels, kernel_size=(1,k), strides=(1,s), groups=channels)`.
- **For causal `(H=1, W=seq_len)` Clifford blocks, pool surface MUST be restricted to `avg`/`max` only.** `padding="same"` IS causal.
- **Pool → nearest-upsample round trip is NOT causal at fine resolution.** Mitigate via `_causal_upsample(x, s)` right-shift by `s-1`.
- **`.keras` save/load on GPU under fp32 has reduction-order noise ~5e-5 for U-Net-shaped models.** Default tolerance for U-Net-shaped layer tests should be 1e-4, not 1e-5.
- **`keras.ops.random.uniform` does NOT exist in Keras 3.8.** Random ops live under `keras.random.*`, not `keras.ops.random.*`. Several library layers (e.g. `dl_techniques.layers.strong_augmentation.StrongAugmentation._apply_color_jitter`) carry this latent Keras-2 → Keras-3 drift and crash on first training-mode forward pass. plan_2026-05-10_44694bc9 D-005.
- **`DepthAnything` (`models/depth_anything/`) now has a real ViT encoder behind `encoder_kind='real'` (default), DPTDecoder linear default + `upsample_factor`, weight-shared frozen teacher via `clone_model`, and a semi-supervised `train_step` path behind `enable_semi_supervised=True`.** Save/load round-trip works via `save_own_variables`/`load_own_variables` override (D-004 anchor at `model.py:608`). DINOv2 weight loading + pseudo-label-depth-on-unlabeled-stream remain deferred to future plans (require pretrained weights and a different data path). plan_2026-05-10_bd098beb closed all 14 README Known Issues + D-005.

## CLM training metrics & probes

- **Metric *math* belongs in `dl_techniques.metrics/`; per-trainer-shape *metric list* belongs in `train.common.nlp`.** Math: `BitsPerToken` / `BitsPerCharacter` in `dl_techniques/metrics/llm_metrics.py`. List: `build_clm_metrics(encoding_name, ignore_index, chars_per_token=None)` in `train/common/nlp.py` with per-encoding `_CHARS_PER_TOKEN_DEFAULTS`.
- **`build_clm_metrics()` MUST return fresh metric instances on every call.** Test: `a = build(); b = build(); assert a[i] is not b[i]`.
- **For new CLM metrics, set `dtype="float32"` explicitly on `add_weight` accumulators.** Forward-compat with `mixed_float16`.
- **Probe-time aggregate metrics ride on `_post_generate_hook`, not a new probe subclass.** Free function bound by `probe_cb._post_generate_hook = fn` is cleaner. Make schema-tolerant.
- **Keras 3 dict-keyed metrics silently no-op on subclassed models unless `output_names` is set.** Call `prepare_dict_keyed_compile(model, output_key="logits")` before `model.compile`. Place inside `compile_model(...)` so it runs on fresh + resumed instances.
- **Verification gate for trainer metric wiring MUST include a real `model.fit` call.** grep + py_compile + shape-only unit tests are insufficient — they pass while dict-keyed metrics silently no-op.
- **`keras.backend.count_params` was removed in Keras 3.** Use `int(np.prod(w.shape))`.

## Multi-optimizer / custom train_step

- **Two-optimizer differential-LR pattern: register only ONE optimizer with `super().compile(optimizer=...)`, apply the second manually inside `train_step`.** Variable routing via name-prefix split (`memory_*` / `gate_*` → memory optimizer; everything else → backbone). Test invariant: `len(memory_vars)>0`, `len(backbone_vars)>0`, union equals `model.trainable_variables` exactly. First repo precedent: plan_146ae899 (`WaveFieldMemoryLLM`).
- **Use leading-component prefix match `name.split('/')[0].startswith(p)`, not substring `if "memory_" in name`.** Substring match accepts unrelated names (e.g. `memory_efficient_attention`). Leading-component is safe when layer names already carry the prefix — verified across all 11 memory weights in plan_0f39a086 (R3+R4 / D-004).
- **Custom `train_step` is compatible with `prepare_dict_keyed_compile` on subclassed `keras.Model`.** Order: `prepare_dict_keyed_compile(...)` → `super().compile(optimizer=backbone_opt, loss=..., metrics={"logits": build_clm_metrics(...)})`. Inside `train_step`: when reading non-loss metric `y_pred`, extract `y_pred["logits"]` if dict (plan_0f39a086 B5).
- **`current_phase` and `_global_step` counters: use `add_weight(trainable=False, dtype="float32")` when the train_step path mixes them with float ops on GPU.** Originally `int32` caused device-placement failures (CPU int32 vs GPU train_step). Float32 is dtype-uniform and survives `.keras` save/load. Do NOT serialize via `get_config` — the variable IS the state (plan_0f39a086 step 4).

## Keras serialization & ops.* tracing

- **`ops.pad` with dynamic paddings does not trace under XLA on TF 2.18.** Substitute `ops.concatenate([x, ops.zeros((B, pad_len, D), dtype=x.dtype)], axis=1)`. Pad length must be a *static* int from `__init__` constants.
- **`ops.one_hot(num_classes=M)` and `ops.top_k(k=K)` need static Python ints.** Bake on `self` at `__init__` (e.g. `self.M_static = int(S_lt + max_seq_len)`).
- **Aux-loss `add_loss` calls must be gated by `training=True`.** Pattern: `if not training: return X` early-returns the no-aux path. Further-gate the block on `if training and any(self.enable_*)` to skip the bool ops when no aux losses are enabled (plan_0f39a086 D4).
- **`keras.ops.cond` traces BOTH branches under `tf.function` on TF backend Keras 3.** Eval-time "skip" branches do not actually skip the kernel launch; multiply-by-zero is the simpler-equivalent pattern when the branches differ only in compute amount, not in side effects (plan_0f39a086 R1 / D-003).
- **`keras.random.shuffle(keras.ops.arange(N))[:k]` is the backend-agnostic replacement for `tf.random.shuffle`.** No `import tensorflow as _tf` for randomness anywhere in library code (plan_0f39a086 B4).

## Keras 3 layer / model composition gotchas

- **Frozen state inside a layer composed into a `keras.Model` MUST live in `add_weight(trainable=False, ...)` OR be stored as numpy on `self` and converted inside `call()` — never as a plain tensor created in `build()`.** Keras `compute_output_spec` runs `build()` in a transient scratch `FuncGraph`; tensors made there get captured and dereferenced as dead tensors at runtime.
- **Symptom-only manifestation:** invisible standalone; tests that wrap in `keras.Model` fail. Always include a "wrap in `keras.Model` + save round-trip" test for any new layer with frozen state.
- **`compute_output_spec` declares symbolic output dtype independently from `call()`'s runtime return.** Override when you need a layer to return a different dtype than `compute_dtype`.
- **Probability-distribution layers under fp16 mixed precision:** clip+sigmoid in fp16 underflows for vocabularies with leaves around the fp16 normal range. Cast logits to fp32 BEFORE sigmoid+clip.
- **`add_weight` variables behave differently under AMP than activations.** Variables stay fp32 but get autocast to compute_dtype when read inside `call()`. Resolution: explicit `ops.cast` at boundaries; add `@pytest.mark.parametrize("policy", ["mixed_float16","mixed_bfloat16"])` end-to-end test.
- **Custom Keras `Initializer` subclass beats split-weight refactors when test count is locked.** Replacing `np.random` inside `build()` via two `add_weight` calls (Identity + RandomNormal) breaks tests that assert `len(layer.trainable_variables) == N`. A `class _MyInit(keras.initializers.Initializer)` keeps topology unchanged AND is fully serializable.

## Layer / model testing patterns

- **Test class structure for new layers: mirror existing sibling classes in the same file.** `tests/test_layers/test_geometric/test_clifford_block.py` is a strong reference.
- **`test_residual_identity_at_init`** (gamma~0 → output~input) is the cheapest sanity check that residual wiring is correct.
- **For pool-choice variants (avg vs max), verify they produce different outputs** with `layer_scale_init` near zero so the residual dominates.
- **Always include a `test_save_load_roundtrip` that wraps the layer inside a `keras.Model`** and round-trips through `.keras` save/load.
- **For causal models with downsample/upsample, write a unit micro-test on each spatial-mixing helper in isolation, in addition to the end-to-end causality test.**
- **Causality test recipe for `(H=1, W=seq_len)` models:** (1) perturb-last-position, (2) perturb-middle-position with depth≥2, (3) non-multiple `seq_len % total_stride`, (4) `use_global_context=True` if applicable. Tolerance on earlier-position diff: `< 1e-5`.
- **For new metric classes (BPT/BPC/etc.), test against a reference implementation by identity** — e.g. `BitsPerToken == log2(Perplexity)` to within 1e-5 on the same synthetic logits/labels.
- **For multi-flag plumbing (e.g. `multi_head_keys`), verify the default-off path is bit-exactly preserved by running ALL pre-existing tests unchanged after the plumbing lands.** plan_0f39a086 step 14 (O4) used the 70 prior tests as the equivalence harness.
- **For redesigned aux-losses (e.g. InfoNCE), a non-degeneracy test is stronger than a finiteness test.** Assert `|cos(grad, anchor)| < threshold` to catch the failure mode where the gradient direction collapses onto the anchor itself (plan_0f39a086 B2 / D-001).
