# Consolidated Decisions
*Cross-plan decision archive. Entries merged from per-plan decisions.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed from 628 lines on 2026-05-13 (refreshed after plan_2026-05-13_a1c9a52d close — merged layers/ntm/ into layers/memory/, deleted ntm package; no new active constraints introduced, but note new constraint below). Read full content below for details on each plan's decisions.*

### Active Constraints (anchored, do-not-break)
- **3-name encoder public surface** (`<Model>`, `create_<model>`, `create_<model>_with_head`) — locked in tree_transformer/bert/cliffordnet; gpt2 is 2-name (LM head intrinsic); cliffordnet now hosts 4+3 names (multiple model classes).
- **`_download_weights` raises `NotImplementedError`** + **`from_variant` narrow `except (IOError, OSError, ValueError)`** — no silent random-init fallback. Anchored in tree_transformer, bert, gpt2, vit, cliffordnet, cliffordnet/embedding_unet.
- **`pad_token_id=<tokenizer_pad>` must be wired from trainer config to encoder ctor** (silent semantic bug otherwise). tiktoken cl100k_base pad = 100266; gpt2 enc pad differs.
- **Output dict key `"logits"`** + **`prepare_dict_keyed_compile(model, output_key="logits")`** required for every Pattern-3 CLM trainer before `model.compile`.
- **`build_clm_metrics(encoding_name, ignore_index)`** — required metric floor for every CLM trainer (replaces bare `["accuracy"]`).
- **`SegmentationWrapperLoss`** is the canonical save/load-friendly seg loss; no more `compile=False` workarounds in trainers.
- **`save_own_variables`/`load_own_variables`** on outer Model classes wrapping inner Models (DepthAnything pattern) — required for `.keras` round-trip when sub-Model weights would otherwise re-initialize.
- **memory_bank dual-optimizer**: register one optimizer with `super().compile`, apply second manually; prefix split via `name.split('/')[0].startswith(p)` (leading-component, NOT substring).
- **U-Net `.keras` round-trip tolerance is atol=1e-4** (not 1e-5) on fp32 GPU due to reduction-order noise. Applies to lmunet + embedding_unet + AccUNet.
- **`dl_techniques.layers.ntm` no longer exists** — all NTM / MANN / SOM imports go through `dl_techniques.layers.memory` (plan_2026-05-13_a1c9a52d D-002). Top-level (`NTMCell`, `NTMConfig`, `create_ntm`, `MannLayer`, `SOM2dLayer`, `SOMLayer`, `SoftSOMLayer`) and deep-submodule paths both supported.

### Failed Approaches (do NOT retry)
- "Modify `lmunet.py` in place with a `causal` flag" — REJECTED (plan_632605aa D-001); also "modify Clifford block classes with `causal` flag" — REJECTED. Sibling-stack additive file is correct.
- `keras.ops.cond` for runtime branch skipping inside `call()` — both branches trace under TF; use multiply-by-zero (plan_0f39a086 D-003).
- Mocking the database in tests / using `compile=False` to dodge a custom-loss round-trip bug — both are workarounds, not fixes (LESSONS).
- SimCSE / contrastive sentence-pair training as iter-1 for an encoder package — explicit deferral pattern (plan_632605aa D-003; plan_146ae899 — staged plans only).
- LR sweep on "smooth-train + cliff-val + sub-random val" signature — that fingerprint = data-pipeline divergence, NOT hparams (plan_f2d29729 D-006/D-007).

### Decision-Anchor Conventions
- Format: `# DECISION plan_<id>/D-NNN: <one-line>` at point of impact. Block, hash, double-dash variants supported. Unqualified `D-NNN` anchors from old plans are tolerated but WARN; new code MUST use qualified form.
- 5 triggers: failure-driven, non-obvious, rejected-alternative, constraint-workaround, 3-strike.
- Anchor at impact site (not at decision definition). One anchor per impact site, even if shared with sibling decision.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-05-14_9c6387a3
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-14_9c6387a3/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-14
**Context**: EXPLORE found that all three training scripts (`train_e1_image.py`, `train_e3_faithfulness.py`, `train_e5_clevr_hans.py`) already accept `--seed` and call `keras.utils.set_random_seed(args.seed)` in `main()` before any dataset shuffling or model construction. This is the canonical Keras 3 multi-source seed primitive (seeds Python `random`, `np.random`, `tf.random.set_seed`, and Keras backend RNG simultaneously).
**Decision**: Drive each script via subprocess from a new `multiseed_sweep.py`; do not edit library or training scripts.
**Trade-off**: Process-startup overhead per seed (~3-5s for TF init) **at the cost of** preserving the FROZEN invariant on training scripts and avoiding any cross-seed state leakage (every TF session is fresh).
**Reasoning**: Subprocess driver is preferred (per goal text). Each seed gets a clean TF/Keras initialization, eliminating cross-seed state contamination. The ~3-5s overhead × 5 seeds × 4 experiments = ~80s total — negligible against ~90min total run. Alternative (in-process import + `keras.backend.clear_session()` between runs) is brittle and conflates state.
**Anchor-Refs**: none (no source edits).

### D-002 | PLAN | 2026-05-14
**Context**: E5 specifies a paired permutation test for circuit-vs-MLP shortcut-gap difference. With n=5 paired samples, parametric tests (Welch t) are statistically dubious. Permutation tests are exact and non-parametric in small n.
**Decision**: Use paired sign-flip permutation test (B=10000, two-sided) on per-seed `(circuit_shortcut_gap - mlp_shortcut_gap)` differences, with a deterministic RNG (`np.random.default_rng(20260514)`).
**Trade-off**: Conservatism on small-n (worst-case p-value resolution = 1/2^5 = 0.03125 if all sign flips enumerated) **at the cost of** zero distributional assumptions.
**Reasoning**: At n=5, only 2^5=32 unique sign-flip patterns exist. Sampling B=10000 is essentially complete enumeration (with replacement). p-value resolution is exact for the chosen sample. Welch t would require normality assumption that 5 paired samples cannot validate; permutation is the textbook choice (Efron 1979, Phipson & Smyth 2010).
**Anchor-Refs**: none (new module).

### D-003 | PLAN | 2026-05-14
**Context**: E3 attribution sweep uses LIME and SHAP. Default hyperparams (`--lime-num-samples 2000 --shap-nsamples 128 --num-attr-samples 32`) make a single E3 run take ~30 min. With 5 seeds, this is 2.5h on E3 alone — still inside the 4h leash but using most of it for marginal statistical benefit.
**Decision**: Pass reduced hyperparams `--num-attr-samples 8 --lime-num-samples 200 --shap-nsamples 32` to the E3 subprocess invocations (matching the LESSONS L67 reduced run that completed E3 in 6 min total).
**Trade-off**: Lower attribution-stability resolution per seed (LIME/SHAP variance is dominated by `lime_num_samples`) **at the cost of** total wall-clock budget headroom.
**Reasoning**: The headline metrics for the multi-seed sweep are mean and std across seeds, not within-seed attribution stability. The seed-to-seed variance from `keras.utils.set_random_seed` is much larger than the LIME/SHAP sampling variance at these hyperparams. The prior 6-min E3 run produced usable circuit_suff_auc / lime_suff_auc / shap_suff_auc numbers; we replicate that recipe.
**Anchor-Refs**: none.

### D-004 | PLAN | 2026-05-14
**Context**: Some metrics may saturate (E5 oracle val_acc=1.000 across all seeds; E3 parity_k8 hard_acc near 1.000) yielding std=0. Bootstrap CI on zero-variance data has width 0; permutation test on identical paired data has p=1.0.
**Decision**: Stats module handles zero-variance cases explicitly: `bootstrap_ci` returns `(value, value)` when all data identical; `paired_permutation_test` returns `(0.0, 1.0)` when all paired diffs are zero. Both behaviors are unit-tested.
**Trade-off**: Slight branching in pure functions **at the cost of** robust degenerate-case behavior (no NaN propagation, no divide-by-zero).
**Reasoning**: Saturation is an expected real outcome of these benchmarks (LESSONS L61). Refusing to handle it cleanly is a footgun. The unit tests pin the contract.
**Anchor-Refs**: none.

## plan_2026-05-14_c95e848c
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-14_c95e848c/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-14
**Context**: E5 (CLEVR-Hans3) needs a vision pipeline with a `LearnableNeuralCircuit` head. Dataset is MIT-licensed and headless-downloadable (~2.4 GB). `keras.applications.ResNet50` is the only live pretrained Keras CNN; in-house ResNet18 has placeholder weights. NS-CL paper-reproduction is 2-3 days — out of 16h budget.
**Decision**: Three-way comparison: ResNet50-frozen+circuit vs ResNet50-frozen+MLP (param-matched) vs perfect-perception oracle (scene-graph JSON → circuit). Wall-clock leashes: download 2h, per-model 6h, total 16h. Honest-negative branch if download fails.
**Trade-off**: Substitute ResNet-18 with ResNet50 **at the cost of** strictly param-matched comparison to the analysis summary's original spec.
**Reasoning**: ResNet50 is the only live pretrained Keras CNN. NS-CL replaced with oracle because (i) it isolates the reasoning-head question from perception quality, (ii) ships within budget. Follows `train_e1_image.py` (LESSONS L51 circuit defaults).
**Anchor-Refs**: pending — add `# DECISION plan_2026-05-14_c95e848c/D-001` to `src/train/logic/train_e5_clevr_hans.py` at model-factory site during EXECUTE.

## plan_2026-05-14_e26eede2
### D-001 | EXPLORE → PLAN | 2026-05-14
**Context**: `to_symbolic()` returns a multi-line human-readable string, not an executable AST (F1). Parsing it back into a categorical-domain predicate is brittle and crosses an unnecessary abstraction layer.
**Decision**: Score rule recovery by evaluating the **hard-extracted Keras model** directly on the enumerated 432-config one-hot encoded categorical domain. Compare predictions vs published-rule ground truth on those 432 configs.
**Trade-off**: Conceptual clarity + bit-exact correctness **at the cost of** not producing a SymPy/z3-style symbolic-equivalence certificate (the score is empirical, not algebraic). Acceptable because Monks attribute domain is exhaustively enumerable (432 configs) so empirical = algebraic here.
**Reasoning**: Truth-table equivalence on a finite domain IS algebraic equivalence. Rejected: (a) parsing to_symbolic (brittle, ambiguous), (b) z3/sympy SMT (overkill for 432 rows, install cost).

### D-002 | EXECUTE/step-1 smoke | 2026-05-14
**Context**: OpenML returns combined (124 train + 432 test concat, but ORDER not contractually guaranteed); UCI direct-download (`https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-N.{train,test}`) returns Thrun's canonical splits in standard format `class a1..a6 id`.
**Decision**: Use UCI direct-download as PRIMARY path. `openml` install is kept as a soft dep but the loader does not depend on it. UCI files are cached on first download under `~/.cache/dl_techniques/monks/`.
**Trade-off**: Loader is simpler + matches canonical splits exactly **at the cost of** depending on the UCI mirror staying up. Acceptable: UCI is the canonical source; files are tiny (~5KB each) so caching is trivial.
**Reasoning**: Smoke-test in step 1 showed all 6 UCI files download in ~2s with no auth/throttling; OpenML combined-row encoding requires inferring split order (non-canonical).

### D-003 | PLAN | 2026-05-14
**Context**: User-supplied "FULL AUTONOMY" override: skip PC-PLAN/PC-REFLECT user gates, do not pause on AskUserQuestion, run honest-negative CLOSE if criteria fail unrecoverably.
**Decision**: Proceed directly from PLAN to EXECUTE without user approval. Log every conservative-default decision here. On Pre-Mortem trigger or autonomy-leash hit, follow the plan-specified action (REFLECT and honest-negative CLOSE) without pausing.
**Trade-off**: Speed + the user's stated minimum-interaction preference **at the cost of** giving up the gate-check protection that would normally catch a malformed plan before EXECUTE.
**Reasoning**: User explicit override. Plan is grounded in 3 deep findings + frozen-helper inventory; PC-PLAN gate would be a no-op here.

### D-004 | REFLECT | 2026-05-14
**Context**: All 8 SC PASS. The >5pt-on-≥2-of-3 paper-hook claim from `analyses/analysis_2026-05-13_9c535f78` §E4 is REFUTED on Monks (mlp_matched ties or slightly beats circuit on all three tasks; xgboost is the weak baseline). The rule-recovery capability claim is CONFIRMED (circuit hits 0.995 / 1.000 / 0.965). The MUX learning curve shows circuit/MLP/XGBoost converge together — no unique low-data inductive bias for circuit.
**Decision**: Honest-negative CLOSE on the comparative claim, honest-positive CLOSE on the rule-recovery claim. Do NOT re-tune the circuit to chase the >5pt criterion; that would be p-hacking against the falsification signal that just fired (Pre-Mortem Scenario B / D variants).
**Trade-off**: Scientific integrity + the explicit honest-negative protocol from this plan + LESSONS L62/L67 (honest-negative precedent) **at the cost of** a less-glamorous paper-hook framing.
**Reasoning**: User said "honest opinion ... no need to celebrate trivial results." Two negatives + one positive is more useful than tweaking until a different verdict appears.

<!-- Schema example — DO NOT REMOVE. Real entries follow this shape.
     See references/file-formats.md "Entry Schema by Type" for required fields per entry type.
     In-code anchors carry the plan-id prefix: `# DECISION plan_2026-05-14_e26eede2/D-NNN` (see references/decision-anchoring.md).

### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-14_e26eede2/D-NNN` anchor exists in source)
-->

## plan_2026-05-13_798d3a60
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_798d3a60/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-14
**Context**: EXPLORE produced 4 findings (F1–F4) confirming: (a) `LearnableNeuralCircuit` already accepts rank-4 input (no library change for E1, ghost constraint), (b) `extract_hard_inplace`/`_iter_inner_ops`/`roundtrip_check`/CSV writers are reusable from `train_benchmark.py`, (c) `lime` + `shap` are NOT installed in `.venv` (HARD BLOCKER, first plan step), (d) `circuit_depth=2` + `arithmetic_op_types=['add','max','min']` + `apply_sigmoid_per_depth='first_only'` are NaN-safe pins (LESSONS L51). E1 = Conv-stem + circuit + GlobalPool + Dense on MNIST/CIFAR-10. E3 = 3 new boolean tasks (`mux_11bit`, `parity_k8`, `random_dnf_8input_4term`) with `StopOnAccuracyBand` partial-convergence + circuit/LIME/SHAP attribution faithfulness comparison. Headline metric is hard-extraction Δ at non-saturation.
**Decision**: Ship 2 new sibling training scripts (`train_e1_image.py`, `train_e3_faithfulness.py`) plus 2 small helper modules (`callbacks_band.py`, `attributions.py`) under `src/train/logic/`. Library code and `train_benchmark.py` are FROZEN.
**Trade-off**: ~1960 LOC of new code (4 source + 4 tests) **at the cost of** some duplication between E1 and E3 trainers (band-checkpoint helper, hard-extraction wrapper) that could in principle live in a shared util module.
**Reasoning**: N=2 sibling files is below the LESSONS L11 coupling-crossover (N≥4). Pre-extracting a shared util would inflate abstraction count (5/4 budget) and create a 3-file change boundary for any future tweak. If REFLECT shows >40 LOC of true duplication, factor then. Alternative considered & rejected: extend `train_benchmark.py` in place — rejected because it's anchored as the boolean benchmark (plan_25774a34 D-001 coherent unit) and would break SC7 (existing 254 tests + schema). Alternative considered & rejected: extend the circuit layer — rejected because rank-4 already works (ghost constraint, F1 + plan_2aaad563 precedent).
**Anchor-Refs**: pending — will add `# DECISION plan_2026-05-13_798d3a60/D-001` at top of `src/train/logic/train_e1_image.py` and `src/train/logic/train_e3_faithfulness.py` once those files are created in Steps 6 and 10.

### D-002 | EXECUTE step-5 fix-attempt-2 | 2026-05-14
**Context**: Step 5 SC6 test failed on gradient×input attribution: per-bit normalized mass on a trained parity_k6 model had a max/min ratio of 2.85x (>2x falsification threshold from Pre-Mortem Scenario 4). The random Dense embedding's per-bit weights don't marginalize when grad×input is evaluated on a single sample, even averaged over inputs.
**Decision**: Replace `circuit_attributions` with integrated gradients (Sundararajan et al. 2017) on a baseline=0.5 → x path with n_steps=32. Test now passes with ratio < 2.0.
**Trade-off**: Modest 32× compute overhead per attribution **at the cost of** symmetry-preserving, completeness-axiom-satisfying attributions that align with the SC6 oracle.
**Reasoning**: Integrated gradients is the standard differentiable-saliency method and is mathematically guaranteed to preserve task symmetries. The compute overhead is acceptable: ~32 forward+backward passes per sample, dwarfed by LIME (5000 samples) and SHAP (256 samples). The plan's original wording ("combination probabilities × involved input indices") was design speculation; the integrated-gradient implementation is the same "circuit-native differentiable" idea executed correctly. Pre-Mortem Scenario 4 is therefore not a falsification of the experiment design — it correctly flagged a flawed first attempt at the rule.
**Anchor-Refs**: `src/train/logic/attributions.py:81-86` (in-code `# DECISION plan_2026-05-13_798d3a60/D-002` block immediately above `circuit_attributions`).

### REFLECT Verdict | 2026-05-14
**Phase 1 Gate-In**: read state.md, plan.md, progress.md, verification.md, findings.md, checkpoints/, decisions.md, changelog.md.

**Phase 2 Evaluate**:
- All 8 success criteria PASS (see verification.md).
- 231 pre-existing layer logic tests still pass (no regression).
- 53 new train logic tests pass.
- No scope drift: exactly the 8 planned files created; one in-scope D-002 anchor block added in attributions.py during REFLECT to address validate-plan anchor-refs-stale warning.
- Changelog scan: one MED-radius edit (attributions.py at step-5) anchored to D-002 with clear reason; no thin reasons.
- Validate plan script: 28 ERRORs are all orphan anchors from OTHER plans (pre-existing codebase debt), 1 transition-format WARN is cosmetic — none introduced by this plan.
- 6 Simplification Checks: no blockers (no wrapper cascades, no copy-paste, no exception swallowing, no type escapes, no adapters, no "temporary" workarounds introduced).
- Headline result: **hard-extraction Δ FAITHFUL on every band-entry checkpoint across all 5 task/dataset configurations** (MNIST, CIFAR-10, mux_11bit, parity_k8, random_dnf). Pre-Mortem Scenario 3 STRONG-POSITIVE outcome confirmed.
- D-002 (Pre-Mortem Scenario 4 fired at step-5) was correctly resolved within autonomy leash by switching to integrated gradients — not a plan failure, just a rule refinement.

**Root-cause analysis**: not applicable; all criteria pass on first iteration.

**Recommendation**: CLOSE. Per the user's pre-approval (sleeping), proceed directly to CLOSE without blocking.

## plan_2026-05-13_25774a34
### D-001 | EXPLORE to PLAN | 2026-05-14
**Context**: User asked for "something more elaborate / train and test it". Previous plan (`plan_2026-05-13_d256b568`) trivially proved the layer trains on K=4 parity. The natural escalation is a multi-task benchmark + comparison + faithfulness study.

**Decision**: One new file `src/train/logic/train_benchmark.py` containing 4 tasks × 3 models grid plus an in-place hard-extraction faithfulness test, dumping CSV + markdown report.

**Trade-off**: A single fat file (~500 LOC) **at the cost of** test/refactor coupling — if any one component (task generator, model factory, extraction) breaks, the whole file is touched. Accepted because: (a) keeps within the 2/3 files budget; (b) entire benchmark is a coherent unit; (c) follows the precedent set in `train_boolean_circuit.py` (also a single file).

**Reasoning**:
- Rejected: split into `tasks.py` + `models.py` + `extraction.py` + `runner.py` — would be 4 files, well over budget; the coupling is real but not painful at this scale.
- Rejected: extend `train_boolean_circuit.py` in place — that file is the "single-task convenience" entry per prior plan; keeping benchmark vs single-task separate preserves both use cases.
- Rejected: a Keras `Callback`-based extraction probe — too clever; in-place weight swap + restore is simpler and bypasses callback ordering issues.

**Anchor-Refs**: D-002 will anchor at the extract_hard_inplace function in S1; D-003 if any test-locked-in invariant emerges during S2.

## plan_2026-05-13_d256b568
### D-001 | EXPLORE to PLAN | 2026-05-14
**Context**: User asked to "create a model and a train file so we can see for real if this thing works". The "thing" is `LearnableNeuralCircuit` from the prior plan. Done well, the validation is an empirical proof that the layer's symbolic readout reflects what it actually learned.

**Decision**: Single-file train script `src/train/logic/train_boolean_circuit.py`. Task = synthetic boolean function recovery (parity, majority, random k-DNF). Architecture = Dense embed → LearnableNeuralCircuit → Dense head. Eval = held-out accuracy + `to_symbolic()` printout + save/load round-trip.

**Trade-off**: Synthetic data **at the cost of** less "realism" than image / NLP. Acceptable because: (a) the goal is to validate the layer, not to set a benchmark; (b) ground-truth boolean functions give an interpretable target the symbolic readout can be checked against; (c) tiny fast-iterating models.

**Reasoning**:
- Rejected: vision classifier (would confound circuit-vs-CNN); precedent at `src/train/latent_reasoning_vision/circuit.py` is 1200 LOC of vision pipeline noise.
- Rejected: NLP classification (token embeddings, batching complexity not needed).
- Rejected: only majority (linearly separable; doesn't prove non-linear capacity).
- Picked: parity + majority + random-DNF as a small task triad — each tests a different property.

**Anchor-Refs**: any DECISIONS anchored in code will be added in S2/S3.

## plan_2026-05-13_e33114da
### D-001 | EXPLORE to PLAN | 2026-05-13
**Context**: Prior epistemic-deconstructor review (session `analysis_2026-05-13_62e26431`) produced a punch-list: 6 BUGs, 4 GAPs, ~8 DESIGN, 3 DOC. User asked for double-check + implement-everything. Each finding re-verified by re-reading source and (for B1) numerical check.

**Decision**: Implement everything in one plan with 17 granular steps; commit per step. No new abstractions beyond `inner_*_kwargs` dict params (1/2 budget). All math/API changes default to legacy behavior; new flags opt-in.

**Trade-off**: 17 commits and ~200 net LOC **at the cost of** churn in `neural_circuit.py` and `logic_operators.py`. Acceptable because every fix is independently small and reversible by reverting that one commit.

**Reasoning**:
- Rejected "single mega-commit": too hard to bisect if a regression surfaces later.
- Rejected "fix bugs only, skip design": user explicitly said "implement everything, MAXIMUM EFFORT".
- Rejected "rename `entropy_coefficient` / `gate_entropy_coefficient`": the names were chosen deliberately last iteration; better to fix wording in comments/docs than ping-pong on names.

**Anchor-Refs**: anchors will be added in S2 (Hamacher), S3 (Gumbel training), S4 (risky_stack), S8 (inner_kwargs forwarding rule), S11 (vectorized diversity).

### D-002 | S2 B1 Hamacher boundary | 2026-05-13
**Context**: `_hamacher_or(1,1)` returned 0 (cliff discontinuity from `max(denom, 1e-9)`); mathematical limit is 1. `_hamacher_and(0,0)` returned 0 via additive eps — coincidentally correct but with different eps strategy.

**Decision**: Unified both Hamacher t-norms with `ops.where(denom < eps, fill, num/denom_safe)`. Fill = 0 for AND (limit at (0,0)) and 1 for OR (limit at (1,1)). Eps `1e-7` set as class constant `_HAMACHER_SINGULAR_EPS`.

**Trade-off**: One extra `where` op per Hamacher call **at the cost of** preventing silently wrong outputs at sigmoid-saturated corners (fp16 or upstream clipped to [0,1]).

**Anchor-Refs**: `src/dl_techniques/layers/logic/logic_operators.py:442`

### D-003 | S3 B2 Gumbel inference | 2026-05-13
**Context**: `_operation_probs` had `deterministic: bool = False` but `call()` never passed `training`. `model.predict()` re-sampled Gumbel noise on every call → non-deterministic inference.

**Decision**: Add `training: Optional[bool] = None` to `_operation_probs`. Gumbel noise injected only when `training is True`. Treat `None` and `False` as inference (skip noise). Match standard Keras dropout/BN convention.

**Trade-off**: Slightly stricter behavior than DARTS literature (some implementations sample at inference) **at the cost of** matching Keras-user expectations that `model.predict()` is deterministic. Users who explicitly want stochastic inference can call `layer(x, training=True)`.

**Anchor-Refs**: `src/dl_techniques/layers/logic/logic_operators.py:506`, `src/dl_techniques/layers/logic/arithmetic_operators.py:470`

### D-004 | S4 B3 widened risky_stack | 2026-05-13
**Context**: B3 finding posited a residual-leak case with `num_arith=0` that turned out to be impossible because the constructor validates `num_arith >= 1`. Original `risky_stack` condition was effectively always-true for valid constructions. False positive partially.

**Decision**: Widen condition to `(num_arith > 0 OR use_residual)` regardless — defensive future-proofing, and clearer semantic statement of *why* force-clip is needed (out-of-[0,1] inputs from either arithmetic experts or residual). If the validation is ever relaxed, the guard still works.

**Trade-off**: Slightly more verbose condition + warning message **at the cost of** zero functional regression (default config still warns; logic is just better-stated).

**Anchor-Refs**: `src/dl_techniques/layers/logic/neural_circuit.py:636`

### D-005 | S5 B4 per-channel load balance | 2026-05-13
**Context**: `_maybe_load_balance_loss` was called with `mean(combination_probs, axis=0)` for per-channel mode, then L2'd. Channel-wise peakiness averaged out → invisible to regularizer.

**Decision**: Inline per-row L2 (`sum(beta^2, axis=-1)`) then mean. Unified handling for both `(N,)` global and `(C, N)` per-channel shapes.

**Trade-off**: One extra `mean` call per forward pass when per-channel **at the cost of** properly penalizing channel-wise peakiness as intended.

**Anchor-Refs**: `src/dl_techniques/layers/logic/neural_circuit.py:357`

### D-006 | S6 S8 G1 inner_kwargs collision rule | 2026-05-13
**Context**: `inner_logic_kwargs` and `inner_arithmetic_kwargs` allow forwarding arbitrary config into inner ops. Some keys (`operation_types`, `apply_sigmoid`, `selection_mode`, `force_clip_when_no_sigmoid`, `allow_unary_degenerate`, `name`) are wrapper-controlled — overriding them through the dict would silently violate wrapper invariants.

**Decision**: Wrapper-owned keys always win. User-passed values for those keys are stripped from the dict and a `UserWarning` is emitted naming the collided keys.

**Trade-off**: Slightly opinionated (could just raise) **at the cost of** zero breakage for users who copy-paste a kwargs dict from one context to another. Warning makes the override visible without blocking.

**Anchor-Refs**: `src/dl_techniques/layers/logic/neural_circuit.py:220`

### D-007 | S15 H6 rationale comment correction | 2026-05-13
**Context**: The plan_2026-05-13_3a2f1d23/D-002 rationale claimed the H6 rename `load_balance_coefficient → gate_entropy_coefficient` reflected the loss being "Shazeer gate-entropy regularizer, not load-balance". Math review: implementation is `coef * N * mean(sum(beta^2))` — L2 of probs, not entropy. Both are convex measures of peakiness with the same uniform optimum, but they're not the same function.

**Decision**: Correct the rationale comment in place; preserve the name (renaming again would be churn for users with saved configs). Document the actual semantics in README.

**Trade-off**: Misleading name persists **at the cost of** zero migration churn.

## plan_2026-05-13_3a2f1d23
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-05-13_3a2f1d23/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-05-13
**Context**: Reviewer surfaced 5 critical + 10 high + 10 medium claims against logic/. Empirical verification of all 5 critical: C1 (Gumbel form) confirmed differential statistical effect; C2 (smooth-divide dipole) is the LESSONS L44 / D-001-anchored design choice — reviewer's math is right but classification as bug is wrong; C3 (1-D selection) confirmed by code inspection; C4 (sigmoid-stacking unsoundness) confirmed by tracing call graph; C5 (to_symbolic gumbel leak) confirmed.
**Decision**: Phase changes by risk into A (LOW-risk correctness fixes + default flips), B (MED-risk feature additions: alias, force-clip), C (HIGH-risk arch: per-channel selection mode), D (NET-NEW features: t-norms, walker, callback, diversity). Present two scope options to user (Quick Wins = Phase A only; Full Scope = all). Reject reviewer H9 (remove explicit child.build) — LESSONS L42 empirically reversed this earlier.
**Trade-off**: comprehensive implementation **at the cost of** ~1100 LOC, 16 commits, and a non-trivial architectural addition (per-channel mode) that doubles the weight count for selection in the new mode.
**Reasoning**: User asked MAXIMUM EFFORT. Phasing by risk lets the user revert mid-plan if Phase C/D goes sideways without losing the correctness wins. LESSONS L42 takes precedence over reviewer H9 because we have a documented empirical reversal anchor.
**Anchor-Refs**: `findings/critical-claims-verification.md`, `findings/scope-and-risk-assessment.md`, `src/dl_techniques/layers/logic/arithmetic_operators.py:442`, `src/dl_techniques/layers/logic/logic_operators.py:467`

### D-002 | EXECUTE step-7 | 2026-05-13
**Context**: H6 review claim — `load_balance_coefficient` is a misnomer; the loss it gates is the Shazeer (2017) gate-entropy regularizer (mean over batch of -p·log(p)), not a load-balance loss. External callers (only `src/train/latent_reasoning_vision/circuit.py`) currently pass `load_balance_coefficient`.
**Decision**: Add `gate_entropy_coefficient` as the canonical kwarg on `CircuitDepthLayer` + `LearnableNeuralCircuit`. Keep `load_balance_coefficient` as a deprecated alias that emits a one-time `DeprecationWarning` and resolves to the same internal `_gate_entropy_coef`. `get_config()` emits the new name.
**Trade-off**: API clarity **at the cost of** carrying a deprecated alias indefinitely (cannot remove without breaking external consumer).
**Reasoning**: Renaming silently would break the external consumer. Adding the canonical name with an alias preserves bit-exact behavior for existing callers while letting new code use the correct term. Aligns with library convention that misnamed APIs get aliased, not replaced.
**Anchor-Refs**: `src/dl_techniques/layers/logic/neural_circuit.py:45`
