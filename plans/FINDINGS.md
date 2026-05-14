# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

<!-- COMPRESSED-SUMMARY -->
## Summary (compressed)
*Auto-compressed. Read full content below for plan-specific detail.*

### Key Findings
- **TreeTransformer (`models/tree_transformer/`)** is structurally sound — save/load + gradient flow + MLM-wrapper integration all correct. Four real bugs fixed in plan_3c3ed037: B-1 fp16 NaN (dtype-aware mask sentinel `-1e4` under float16, plus fp32 cast on GroupAttention DP log/matmul/exp block); B-3 explicit `attention_mask` honored in dict input; B-4 `load_pretrained_weights` via `weight_transfer.load_weights_from_checkpoint` (Keras 3.8 `by_name=True` broken); B-5 `PRETRAINED_WEIGHTS={}` + `NotImplementedError` (no public checkpoints). Trainer `src/train/tree_transformer/{pretrain,finetune}.py` mirrors `bert/`. Anchor: `model.py:318` D-001. **Trainer config MUST pass `pad_token_id=config.pad_token_id` (tiktoken cl100k_base = 100266) to encoder — model default 0 is silent semantic bug.** Aligned to `bert/`/`resnet/` conventions in plan_0a5779e8: bare `create_tree_transformer(variant, ...)` factory added, `__init__.py` trimmed to 3 names (`TreeTransformer`, `create_tree_transformer`, `create_tree_transformer_with_head`; internal layer classes remain importable from `.model` for `nam/` consumers), and `from_variant(pretrained=True)` now raises `NotImplementedError` loudly instead of silently random-initializing (D-001 anchor at `model.py:1133`, narrowed try/except to `(IOError, OSError, ValueError)`).
- **TinyRecursiveModel (`models/tiny_recursive_model/`)** — save/load clean. B-3 Q-learn lookahead `training=False` + `keras.ops.stop_gradient` on `target_q`; B-5 inference halts on learned signal. `hrm_loss`/`HRMMetrics` API-compatible with TRM output schema. Anchor: `model.py:370` D-001 (plan_e6309bd5).
- **`keras.ops.expand_dims(axis=tuple)` works** on Keras 3.8 / TF 2.18 eager + `@tf.function` (B-1 false-positive in plan_e6309bd5).
- **DepthAnything** is now full-feature — real ViT encoder, DPTDecoder linear default + `upsample_factor`, weight-shared frozen teacher via `clone_model`, semi-sup `train_step` (FAL + L1 pseudo-label stop-gradient), on-step EMA via `TeacherEMACallback`, `from_pretrained_encoder(path)`, `StrongAugmentation` + dynamic cutmix.
- **Keras 3 / TF 2.18 idioms**: `keras.random.*` (NOT `keras.ops.random.*`); `keras.ops.*` for backend-agnostic ops; `@keras.saving.register_keras_serializable()` + `get_config()` round-trip; `dl_techniques.utils.logger` only.
- **Save/load on subclassed Models wrapping inner Models**: weights drop unless outer class overrides `save_own_variables` / `load_own_variables` (D-004).
- **Keras 3.8 `model.load_weights(path.keras, by_name=True)` is broken** — use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`.
- **AdamW double weight decay footgun**: never combine `AdamW(weight_decay=...)` with `kernel_regularizer=L2(weight_decay)`.
- **CLM training**: use `train.common.nlp.estimate_clm_steps_per_epoch`; `min_article_length=0` correct for packed pipelines.
- **Two-optimizer differential-LR**: register one with `super().compile`; apply second manually inside `train_step` via name-prefix variable routing (leading-component match).
- **`keras.ops.cond` traces BOTH branches under `tf.function`** — multiply-by-zero for compute-amount differences.
- **Frozen state in layers**: `add_weight(trainable=False)` or numpy-on-self — never plain tensors in `build()` (FuncGraph dead-tensor).
- **BERT (`models/bert/`)** aligned to resnet/tree_transformer template in plan_9357982a: `create_bert` bare-encoder factory added; `__init__.py` trimmed to 3-name surface `{BERT, create_bert, create_bert_with_head}` (drop `create_nlp_head` re-export); `_download_weights` raises `NotImplementedError`, `from_variant` try/except narrowed to `(IOError, OSError, ValueError)` (D-001 anchor at `bert.py:687`); docstring/README path fixed to `dl_techniques.layers.nlp_heads`. 28/28 pytest PASS, 0 fix attempts.
- **AccUNet** requires H,W divisible by 16; validation in `call()` raising `ValueError` (plan_bdb2c84d D-001/D-002).
- **`SegmentationWrapperLoss`** is canonical save/load-friendly segmentation loss; `compile=False` workaround removed (plan_17633038 D-002).

### Key Decisions
- **D-001 plan_3c3ed037 (TreeTransformer bundle)**: 4 model bugs + Pattern-3 trainer in one iteration — 5 new files / +950 LOC at the cost of 2 over file-budget; trainer depends on Step 5 re-exports and Step 2 attention_mask honoring, so splitting would force pinning to broken imports.
- **D-001 plan_e6309bd5 (TRM bundle)**: bug fixes + factory + tests + trainer in one plan — at cost of larger review surface; B-5 testable only with same harness as trainer eval path.
- **Pseudo-label loss**: plain L1 + `stop_gradient`, not `compute_loss` against synthetic mask (plan_54e6e303 D-002).
- **Encoder weight-loading**: keep `--pretrained-encoder-weights` + `--init-from` distinct (plan_54e6e303 D-003).
- **D-004 (save_own_variables override)**: canonical Keras-3 fix when `.keras` round-trip drops sub-Model weights.
- **D-003 (Keras-3 canonical train_step)**: `compute_loss(x,y,y_pred)` adds `self.losses` internally — no manual regularization addition.
- **D-005 (StrongAugmentation graph-mode safety)**: symbolic gate; `keras.random.*` not `keras.ops.random.*`.
- **CLM metrics architecture**: math in `dl_techniques/metrics/`; list in `train/common/nlp/build_clm_metrics()`; fresh instances each call.
- **`current_phase` / `_global_step` counters**: `add_weight(trainable=False, dtype="float32")` — int32 fails CPU/GPU device placement.
<!-- /COMPRESSED-SUMMARY -->

## plan_2026-05-14_9c6387a3
### Index

| # | Topic | File | Confidence |
|---|-------|------|------------|
| F1 | **Seed propagation already correct** — all three scripts call `keras.utils.set_random_seed(args.seed)` in `main()`. No library/script edits required. | `findings/seed-propagation-audit.md` | High (source-verified, line-cited) |
| F2 | **CSV schemas + aggregation surface** — E1/E3 write `benchmark_results.csv`, E5 writes `results.csv`. Aggregation keys: E1 `(dataset, model)`, E3 `(task,)`, E5 `(model,)`. No `seed` column — inject at aggregation. | `findings/csv-schemas-and-aggregation.md` | High (verified against actual CSVs) |
| F3 | **Wall-clock budget** — total estimated 5-seed run is ~90 min, well under 12h global cap. Per-experiment leashes generous. | `findings/wallclock-and-orchestration.md` | High (extrapolated from existing wall_s numbers) |

### Key Constraints

### HARD (non-negotiable)
- **All three scripts already accept `--seed` and call `keras.utils.set_random_seed(args.seed)`.** Library code (`src/dl_techniques/layers/logic/`) and existing training scripts stay FROZEN. **No script edits.**
- **`MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0`** prefix on every subprocess (LESSONS).
- **Serial GPU only** — never parallel (user memory, LESSONS L80).
- **CIFAR-10 band override required**: `--band-low 0.50 --band-high 0.80` (default is MNIST-tuned 0.70/0.95).
- **E5 dataset pre-check**: must verify `data/clevr_hans3/` exists before launching E5; pass `--skip-download`.
- **Hard wall-clock leashes** (from goal): E1 6h, E3 4h, E5 2h, global 12h. Realistic estimate ~1.5h total, so massive headroom.
- **2-attempt autonomy leash per step**. Honest-negative on partial seed failures.
- **No `make test`** (1.5h pre-push hook). Scope pytest to `tests/test_train/test_logic/test_multiseed_stats.py`.
- **n=1 regression check** (REFLECT): existing seed=42 runs (`logic_e1_mnist_20260514_012632`, `logic_e1_cifar10_20260514_013356`, `logic_e3_20260514_013937`, `logic_e5_run`) define the prior n=1 values that must fall within new mean ± 2*std.
- **E3 attribution-sweep hyperparams reduced** (LESSONS L67): `--num-attr-samples 8 --lime-num-samples 200 --shap-nsamples 32` to keep per-seed runtime at ~6 min.
- **High-variance retract/qualify rule**: any metric where std > |mean| flagged at REFLECT.

### SOFT
- **Two new files** (`multiseed_sweep.py`, `multiseed_stats.py`) + one test file. Coupling at N=2 — do not pre-extract shared utilities (LESSONS L11).
- **Subprocess driver**, not in-process import — gives each seed a clean TF/Keras initialization, matches the goal's "runs the underlying scripts via subprocess or imports their main()" preference (a).
- **pandas + numpy + scipy for stats** — pandas already in deps; scipy used elsewhere for similar tests.
- **Permutation test (E5)** — paired by seed index between `resnet50_circuit` and `resnet50_mlp` on `shortcut_gap`. B=10000, two-sided. Independent fixed RNG (e.g. `np.random.default_rng(20260514)`) so the p-value is reproducible.
- **Bootstrap CI (B=2000)** for the headline gap metric per (model, dataset). Same fixed RNG.

### GHOST (constraints that no longer apply)
- **"Maybe we need to add `set_global_seed`"** — superseded. The scripts already have correct propagation. Adding a shim would be dead code.
- **"E3 has 3 model configs"** (goal wording) — actually 3 tasks × 1 model (circuit), with 3 attribution methods as columns. Aggregation key is `(task,)`. Resolved at finding F2.
- **"Determinism flags for cuDNN"** — explicitly NOT required: the whole point of n=5 with mean±std is to characterize stochastic variation; forcing bit-exactness would slow runs without changing the statistics.

### Exploration Confidence
- **Scope: deep.** All three scripts source-verified, line-cited. All four prior n=1 CSVs read for schema confirmation. Wall-clock predictions derived from actual `wall_s` columns.
- **Solutions: constrained.** Aggregator pattern is fully specified (subprocess + glob + pandas group-by). Stats module is pure-function and easy to unit-test.
- **Risks: clear.** Three real risks: (a) E5 dataset absence — pre-check + honest negative, (b) attribution-sweep slowness on E3 — pre-reduce hyperparams per LESSONS L67, (c) saturation/zero-variance metrics — stats module must handle degenerate cases (all-same data → CI width 0, mean = data, std = 0).

### Synthesis paragraph

No script edits are needed — every training script already wires `--seed` through `keras.utils.set_random_seed`, which is the canonical Keras 3 multi-source seed primitive (Python random + NumPy + TF + Keras backend). The plan reduces to two new files: a subprocess-driven sweep orchestrator (`multiseed_sweep.py`) and a pure-stats module (`multiseed_stats.py`) with unit tests. CSV schemas are stable across runs and have stable aggregation keys per experiment (E1: `(dataset, model)`, E3: `(task,)`, E5: `(model,)`). Realistic wall-clock for 5 seeds × 3 experiments is ~90 min — orders of magnitude inside the 12h global cap. Regression check at REFLECT compares each prior seed=42 value against the new n=5 mean ± 2*std, with high-variance (std > |mean|) flagged for retraction. E5's circuit-vs-MLP shortcut_gap is the only test that needs a real significance test (paired permutation, B=10000, two-sided, deterministic RNG).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-14_c95e848c
### Index

| Topic | File | One-line takeaway |
|-------|------|-------------------|
| CLEVR-Hans3 dataset reality | `findings/clevr_hans3_dataset.md` | MIT-licensed, 2.4GB headless download verified live; "clean" = official test split, "confounded" = val split; 26GB free disk is tight but feasible; perfect-perception oracle (scene-graph JSON) replaces infeasible NS-CL baseline. |
| Pretrained backbone options | `findings/keras_backbone_options.md` | In-house ResNet18 has no pretrained weights (placeholder URL); `keras.applications.ResNet50(weights="imagenet")` is the only live ImageNet backbone. Use ResNet50 at 128x128, frozen, GAP-then-Dense(64)-embed -> circuit. |
| Repo precedent for image-circuit pipeline | `findings/repo_precedent_image_circuit.md` | Pattern-1 trainer template + `latent_reasoning_vision` + `train_e1_image.py` are the precedents; new `train_e5_clevr_hans.py` will define the keras.applications+circuit pattern. circuit depth=2, arith=['add','max','min'] per LESSONS L51. |

### Key Constraints

### HARD (non-negotiable)
- **Disk: 26 GB free on `/media/arxwn/data_fast` (89% full).** Dataset zip 2.4 GB + extracted ~5-8 GB. Must delete zip after extract.
- **GPU 0 (RTX 4090, 24GB) ONLY** for image-resolution training. GPU 1 OK for oracle baseline.
- **Library frozen**: `src/dl_techniques/layers/logic/` and existing `src/train/logic/*.py`.
- **Single GPU jobs**, `CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg`.
- **Hard wall-clock leashes** (orchestrator override): download 2h, per-model 6h, total 16h.
- **No `keras.applications.ResNet18`** — use ResNet50 (real ImageNet weights) instead of in-house ResNet18 (placeholder URL).
- **Circuit defaults** (LESSONS L51): `circuit_depth=2`, `arithmetic_op_types=['add','max','min']`, `apply_sigmoid_per_depth='first_only'`.
- **AdamW WD only** (LESSONS L72). **`dl_techniques.utils.logger`** — no `print`.

### SOFT
- ResNet50 at 128x128 input frozen. Fallback 96x96 if OOM.
- `Dense(64)` embed before circuit (LESSONS L52 family).
- `tf.data` with `.cache().shuffle().batch().prefetch(AUTOTUNE)`.

### GHOST
- "ResNet-18 from torchvision" — no Keras-native pretrained ResNet18 exists; ResNet50 substitutes cleanly (already pre-flagged in goal).
- "NS-CL reproduction" — out of budget; perfect-perception oracle better isolates the reasoning-head question.

### Exploration confidence
- Scope: **deep** (3 indexed findings cover all 5 KEY UNKNOWNS in the goal).
- Solutions: **adequate** (3-way comparison; graceful fallback to 2-way on dataset failure).
- Risks: **clear** (disk pressure, dataset flakiness, ResNet50 substitution, NS-CL skipped — all mitigated).

### Synthesis paragraph

E5 is runnable within the 16h cap. Download is verified live (HTTP 200, range requests work, zip-magic confirmed). The "ResNet-18" specification in the analysis summary is suggestive; `keras.applications.ResNet50` is the only Keras-native pretrained CNN in this stack, used frozen at 128x128. Shortcut-gap metric: `val_acc - test_acc` — val keeps the training-distribution confounders, test is the clean (non-confounded) split per README. NS-CL is replaced by a **perfect-perception oracle** trained directly on the scene-graph JSON attributes, isolating the reasoning-head question from perception. Plan ships `clevr_hans_data.py` (download + loader + symbolic encoder), `train_e5_clevr_hans.py` (3 configs), unit tests, and an honest-negative branch that ships code + tests even if download fails.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong.*

## plan_2026-05-14_e26eede2
### Index

| # | Topic | File | Confidence |
|---|-------|------|------------|
| F1 | `to_symbolic` output format — multi-line string, NOT executable AST. Score by evaluating hard-extracted Keras model on enumerated 432-config Monks domain. | `findings/f1-to-symbolic-format.md` | High (source-read) |
| F2 | OpenML Monks loader — `openml` package NOT installed (Step 1 must install). 17-bit one-hot encoding (3+3+2+3+4+2). 432-config canonical enumeration. UCI direct-download fallback. | `findings/f2-openml-monks-loader.md` | High (verified) |
| F3 | Reusable helpers from FROZEN `train_benchmark.py` / `train_e3_faithfulness.py` — `build_circuit`, `build_mlp`, `find_mlp_hidden_for_param_budget`, `extract_hard_inplace`, `restore_soft_weights`, `roundtrip_check`, `gen_mux_11bit`. xgboost NOT installed. Wall-clock budget ~20 min. | `findings/f3-reusable-helpers.md` | High |

### Key Constraints

### HARD
- **`openml` + `xgboost` NOT installed.** Step 1 of EXECUTE: `.venv/bin/pip install openml xgboost`. Verified by `ModuleNotFoundError`.
- **`circuit_depth=2`, `arithmetic_op_types=['add','max','min']`, `apply_sigmoid_per_depth='first_only'`** (LESSONS L51/L60). Already baked into `build_circuit` factory in FROZEN `train_benchmark.py`.
- **Library code (`src/dl_techniques/layers/logic/*`) is FROZEN.** All work is training-side.
- **`train_benchmark.py`, `train_e1_image.py`, `train_e3_faithfulness.py` are FROZEN.** New scripts must IMPORT helpers, not modify the frozen scripts.
- **`.venv/bin/python`, `MPLBACKEND=Agg`, `CUDA_VISIBLE_DEVICES=0`, single GPU serial.** No `make test`.
- **254 existing tests stay green.** New tests in new files only (`tests/test_train/test_logic/test_rule_recovery.py`, `test_e4_monks.py`).
- **Rule-recovery scorer is the LOAD-BEARING component.** Must unit-test against round-trips of the published Monks rules themselves before claiming any rule-recovery result.
- **Wall-clock leashes**: each Monks training run <30 min; total MUX learning curve <1 h. Two consecutive overruns → kill and proceed honest-negative.
- **FULL AUTONOMY**: no PC-PLAN / PC-REFLECT user gates; conservative-default decisions logged to `decisions.md`; honest-negative CLOSE allowed.

### SOFT
- Prefer truth-table equivalence (enumerate 432 configs) over z3/BDD — Monks domain is tiny, this is the cheapest defensible approach (per F1).
- Use OpenML by package; UCI direct-download is fallback if `openml` install or fetch flakes.
- 3 random seeds per (task, model) to get a meaningful mean ± std (Monks-1/2/3 has 124-169 train samples — high variance is expected).
- Architecture A only: circuit on raw one-hot input via `build_circuit(num_bits=17, num_outputs=1)`. No tweaking of architectures unless a falsification signal fires.
- One new `rule_recovery.py` module + two new train scripts + tests. No shared util extraction (N≤4 sibling files, LESSONS L11).

### GHOST
- The temptation to parse `to_symbolic` into an executable AST. F1 establishes the scorer evaluates the hard-extracted Keras model directly; `to_symbolic` is diagnostic-only.
- "Need an extension to circuit for categorical inputs" — false; raw one-hot fed through `Dense(channels, relu)` embed works the same way as for the synthetic boolean tasks.

### Exploration Confidence
- Scope: **deep** (source-verified to_symbolic format, reusable helpers, install status; published Monks rules locked).
- Solutions: **constrained** (rule-recovery scorer methodology pinned; baseline triad locked).
- Risks: **clear** (load-bearing scorer is the only real risk → mitigated by self-round-trip unit tests; install-time deps verified missing → Step 1 installs).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-13_798d3a60
### Index

| # | Topic | File | Confidence |
|---|-------|------|------------|
| F1 | Current logic layer state — shapes, ops, `to_symbolic`, `extract_hard_inplace`, depth ceiling, rank assumptions | `findings/f1-logic-layer-state.md` | High (verified by source reading + LESSONS cross-check) |
| F2 | Existing benchmark machinery — what's reusable from `train_benchmark.py`/tests for E1/E3 | `findings/f2-benchmark-machinery.md` | High |
| F3 | Image-input pathway options for E1 (MNIST / CIFAR-10) | `findings/f3-image-pathway.md` | High (precedent at `train/latent_reasoning_vision/circuit.py:121-129`) |
| F4 | E3 specifics — partial-convergence training, LIME/SHAP, attribution metrics, **lime+shap NOT installed** | `findings/f4-e3-specifics.md` | High (empirically verified pip status) |

### Key Constraints

### HARD
- **LIME and SHAP NOT installed in `.venv`** — confirmed `ModuleNotFoundError`. Step 1 of plan must `pip install lime shap` before any E3 work.
- **GPU**: GPU 1 (RTX 4070 12GB) preferred; GPU 0 (RTX 4090) available; serial only — never parallel jobs (memory + LESSONS).
- **`.venv` Python + `MPLBACKEND=Agg`** prefix for every training invocation (headless server).
- **`circuit_depth >= 3` + default arithmetic ops → NaN** (LESSONS L51, plan_d256b568). Pin `circuit_depth=2` and `arithmetic_op_types=['add','max','min']` for E1 + E3.
- **`LearnableNeuralCircuit` accepts rank-4 already** — no library code change for E1. Library code is FROZEN for this plan; all work is training-side.
- **Existing 254 PASS tests in `tests/test_layers/test_logic/` + `tests/test_train/test_logic/` MUST stay green.** Do not modify `train_benchmark.py` schema — append new train scripts as siblings.
- **No `git push --no-verify` without explicit user direction** — user pushes themselves (LESSONS, user memory).
- **Hard-extraction Δ at non-saturation (val_acc ∈ [0.7, 0.95]) is the HEADLINE METRIC** (pinned by user: value prop = differentiable rule extraction). All E1+E3 success criteria must be framed around this number.
- **Do NOT run `make test`** (1.5h pre-push hook). Scope pytest to `tests/test_train/test_logic/` only.
- **MNIST + CIFAR-10 training runs are long (1h + 6h)** — mark plan steps "USER RUNS" and proceed past them with the most recent saved checkpoint.

### SOFT
- **Ship 2 new training files** (`train_e1_image.py`, `train_e3_faithfulness.py`) + 2 new test files mirroring `tests/test_train/test_logic/` pattern. Share helpers via import from existing `train_benchmark.py`. Do NOT extract a shared util module — N=2 sibling files, the coupling crossover is at N>=4 (LESSONS L11).
- Use `train.common.load_dataset` for MNIST/CIFAR-10 (Pattern-1 trainer template).
- Architecture A (Conv-stem circuit) for E1; defer architecture B (patch-flatten) unless A fails.
- Honest negative reporting expected — record bands that yield no data, tasks that saturate, etc. (LESSONS — plan_25774a34 honest-negative precedent).

### GHOST
- Reviewer claim "extending `LearnableNeuralCircuit` to spatial inputs" in summary §E1 — **already supported, NOT a code change**. The rank-4 path was opened by plan_2aaad563. Don't waste budget on it.
- "Need extension of the circuit for E1" — same as above. Architectural delta is ZERO; only training-side code (Conv stem builder) is new.

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*

## plan_2026-05-13_25774a34
### Index
- F1: Circuit is ~3× smaller than the obvious MLP (401 vs 1313 params at default K=6, depth=2). Need 2 MLP baselines for honest comparison. (`findings/f1-param-budget.md`)
- F2: `selection_mode='per_channel'` + multi-output head works end-to-end on rank-2 input (verified by 2-epoch smoke). Enables the bitwise-shift-XOR task as a unique exerciser of the per-channel feature. (`findings/f2-per-channel-multi-output.md`)
- F3: Task design — pick tasks that span (a) linearly inseparable (parity), (b) linearly separable (majority), (c) conditional / non-uniform (multiplexer), (d) multi-output per-channel (shift_xor). Multiplexer is famously hard for small MLPs. (`findings/f3-tasks.md`)
- F4: Hard-extraction methodology — freeze each inner op to its argmax (set `operation_weights = LARGE * one_hot(argmax)`) and re-evaluate. Accuracy delta = how much the soft mixture matters. (`findings/f4-hard-extraction.md`)

### Key Constraints
- **HARD**: `.venv` Python; GPU 1 (RTX 4070) free; no parallel GPU jobs.
- **HARD**: depth >= 3 with default arith ops → NaN (LESSONS, prior plan d256b568). Stay at depth=2 or restrict to `add,max,min`.
- **HARD**: Dense embed (channels >= 16) required for raw bit-vector inputs.
- **HARD**: existing test suite (241/241 from prior plans) must remain green.
- **HARD**: report negative results HONESTLY if MLP beats circuit. "Did it work" matters more than "did it win".
- **SOFT**: prefer single new file under `src/train/logic/` (`train_benchmark.py`). The existing `train_boolean_circuit.py` stays as single-task convenience entry.
- **SOFT**: wall-clock target < 20 min for full benchmark.
- **SOFT**: write a markdown report alongside the CSV — humans want narrative.

### Corrections
*(none yet)*

## plan_2026-05-13_d256b568
### Index
- F1: Functional smoke test confirms LearnableNeuralCircuit composes as a mid-network block on rank-2 input. Trains under standard `model.fit`, `to_symbolic` reports per-depth dominant ops. (`findings/f1-functional-smoke.md`)
- F2: Existing `src/train/latent_reasoning_vision/circuit.py` is the closest precedent — vision classifier built around LearnableNeuralCircuit. We want something simpler (purely synthetic, no images) so the validation is unambiguous. (`findings/f2-existing-precedent.md`)
- F3: Best demonstration task = N-bit boolean function recovery. Parity is the gold-standard hardest-easy task — linearly inseparable, needs XOR. If `to_symbolic` returns "xor" after training, that's a slam-dunk proof. (`findings/f3-task-choice.md`)

### Key Constraints
- **HARD**: Use `.venv` Python. Use `MPLBACKEND=Agg` if any plotting. GPU 1 (RTX 4070, 12GB) is fine — small models.
- **HARD**: No parallel GPU jobs (memory).
- **HARD**: Don't claim success on the trivial task only — confirm with verification (held-out test set + per-task accuracy + symbolic readout that matches ground truth).
- **HARD**: Save+load round-trip must work for the trained model (regression check for the prior plan's serialization work).
- **SOFT**: Single-file script preferred over splitting model/train across files — keeps blast radius small. Follow the train Pattern-2 (synthetic data, local argparse) since we're not using `load_dataset()`.
- **SOFT**: Don't pull `make test` (1.5h pre-push); run only logic + new train test.

### Corrections
[CORRECTED iter-1, S5] **F3 default op set is unstable for stacked circuits.** First parity attempt (channels=32, depth=3, default arith ops including `power` + `divide`) blew up to NaN by epoch ~12 — loss went to nan, accuracy stuck at 0.5. Diagnosis: `power` with learned exponents can produce arbitrarily large magnitudes; the residual connection then compounds these through depth. Recovery (attempt 1/2): restrict arithmetic ops to `add,max,min` (bounded), reduce circuit_depth from 3 to 2, drop LR to 1e-3. Result: test acc = 1.000, exact enumeration = 1.000, XOR emerges as the dominant depth-1 logic op (62% combined or+xor). **Lesson for future tasks**: default `arithmetic_op_types` is a footgun for deep stacks. The README should recommend `['add', 'max', 'min']` (or at least flag `power` as numerically aggressive) when using depth >= 3. (This is a follow-up doc opportunity, not in scope for this plan.)

## plan_2026-05-13_e33114da
### Source
The detailed review is in `analyses/analysis_2026-05-13_62e26431/summary.md` (the epistemic-deconstructor session that preceded this plan). Re-verified each finding before listing here.

### Index
- [F1: Hamacher OR boundary bug](plan_2026-05-13_e33114da/findings/f1-hamacher-or-boundary.md) — confirmed numerically; `_hamacher_or(1.0, 1.0) → 0` (correct: 1)
- [F2: Gumbel softmax leaks into inference](plan_2026-05-13_e33114da/findings/f2-gumbel-inference-leak.md) — confirmed via grep; `_operation_probs` ignores `training` flag
- [F3: risky_stack guard misses residual-only case](plan_2026-05-13_e33114da/findings/f3-risky-stack-residual.md) — confirmed by tracing `+X` residual through stack
- [F4: Per-channel load-balance averages-then-L2](plan_2026-05-13_e33114da/findings/f4-percent-channel-loadbalance.md) — confirmed by re-reading neural_circuit.py:366-369
- [F5: `diversity_coefficient` unreachable through wrapper/factory](plan_2026-05-13_e33114da/findings/f5-diversity-unreachable.md) — confirmed; not in factory registry, not in `LearnableNeuralCircuit.__init__`
- [F6: Inconsistent shape detection in arithmetic_operators.py](plan_2026-05-13_e33114da/findings/f6-shape-detection.md) — confirmed; `build()` vs `compute_output_shape()` use different heuristics
- [F7: Inner-op knobs not forwarded by circuit wrappers](plan_2026-05-13_e33114da/findings/f7-knob-forwarding.md) — confirmed; ~10 params unreachable through circuit
- [F8: Smaller items](plan_2026-05-13_e33114da/findings/f8-smaller-items.md) — G2-G4, D1-D9, DOC

### Key Constraints
- **HARD**: Keras 3 / TF 2.18; round-trip serialization through `.keras` archives must continue working. Existing tests must pass.
- **HARD**: prior comments reference plan IDs (`plan_2026-05-13_a2b0f17b`, `plan_2026-05-13_3a2f1d23`). Anchors must coexist; do not rewrite history.
- **HARD**: every new flag must default to the OLD behavior — implementation MUST be backwards-compatible for existing saved models.
- **SOFT**: prefer minimal additive changes (no breaking API). Where a fix changes math (e.g. Hamacher boundary), gate behind an opt-out flag if the change is observable.
- **GHOST**: assumption that "names = math" was carried by the H6 rename (`load_balance → gate_entropy`) — the math stayed L2, the name claims entropy. Don't compound by renaming again; just document.

### Corrections
[CORRECTED iter-1, S4] **F3 / B3 risky_stack residual leak: partial false positive.** The constructor validates `num_arithmetic_ops_per_depth > 0` (neural_circuit.py:503-504), so the "pure logic stack with residual" scenario I posited cannot occur with current validation. The original `risky_stack` condition `num_arith > 0` was effectively always-true for any valid construction, so the practical bug was masked by the validation. Widening the condition (S4) is kept as defensive future-proofing — if `num_arith >= 0` is ever allowed, the condition still catches the residual case. The widened wording is also more semantically honest about why force-clip is needed. Cost: 0 (extra OR term in a constructor-time condition).

## plan_2026-05-13_3a2f1d23
### Index

| # | Topic | File | Confidence |
|---|-------|------|------------|
| F-001 | C1..C5 + H6/H9 empirical verification | `findings/critical-claims-verification.md` | High (empirical) |
| F-002 | Scope, risk, package definitions | `findings/scope-and-risk-assessment.md` | High |

### Key Constraints

**HARD**
- Keras 3 / TF 2.18, `keras.ops` only.
- `@keras.saving.register_keras_serializable()` on every layer.
- Full `get_config()` round-trip.
- Tests scoped to `tests/test_layers/test_logic/` ONLY — full suite is 1.5h.
- User pushes themselves (no `git push`).
- Bare decorator → DO NOT relocate classes between modules.
- LESSONS L42: explicit `child.build(input_shape)` in parent's `build()` IS REQUIRED for Keras 3 serialization — rejects reviewer's H9.
- LESSONS L44: `_safe_divide` smooth mode dipole behavior IS the documented design (D-001 anchor of plan_a2b0f17b).

**SOFT**
- Python 3.11+, Google-style docstrings, `dl_techniques.utils.logger`.
- Sole external consumer `src/train/latent_reasoning_vision/circuit.py` uses only stable kwargs — backward-compatible changes are safe.
- Tests that lock current defaults: only `test_unary_input_allowed_when_default` (logic, line 555) breaks if `allow_unary_degenerate` default flips.

**GHOST**
- Reviewer's H9 ("explicit child.build is cargo-cult") — already empirically reversed (LESSONS L42 / plan_a2b0f17b D-003).
- Reviewer's C2 "non-monotone bug" classification — dipole IS by design (LESSONS L44 / D-001). Math claim is correct but it's not a bug.

### Exploration Confidence
- Scope: **deep** (every critical claim empirically verified; full test inventory; consumer audit done).
- Solutions: **constrained** (per-finding fix shape known; phased plan groups by risk).
- Risks: **clear** (default flips identify the one breaking test; C3 is the only HIGH-risk arch change).

### Corrections
*Append [CORRECTED iter-N] entries here when earlier findings prove wrong. Reference the original finding file and what changed.*
