# Production Map — `dl_techniques` Layer Production-Quality Roadmap

**Mission.** Bring every custom Keras layer in `src/dl_techniques/layers/` up to the project's
production-quality standard, one self-contained, context-cleared session ("round") at a time. The
single source of truth for "production-quality" is the canonical exemplar
`src/dl_techniques/layers/convnext_v1_block.py` plus the authoritative spec
`research/2026_keras_custom_models_instructions.md` (Pattern 2 begins at line 381). The mechanical
compliance baseline is produced by the read-only AST scanner `scripts/audit_layers.py`.

**Baseline snapshot (commit `2d96078a`, live tree walk).** 245 source files under
`src/dl_techniques/layers/` (excluding `__init__.py` and `__pycache__`):
**179 PASS / 38 FAIL / 28 N/A** (0 PARSE-ERROR). 217 files contain at least one concrete
`keras.layers.Layer` subclass; 305 concrete layers graded. Dominant mechanical gap:
`super().build()` not last in `build()` (`super_build_last` 224/260 = 86.2%); 7 layers across 4
files carry raw `tf.*` in the forward path; `compute_output_shape` 297/305 = 97.4%; decorator,
`get_config`, and no-`print` are all at 100% over concrete layers.

> **This is a LIVING document.** Each round edits §4 in place — flipping `[ ]` → `[x]`, updating the
> per-round status, and bumping the progress tally. §1–§3 and §5 are stable reference. The handover
> prompt to drive a fresh round is in §6.

---

## §1 Mission & Levels

The effort is staged in three levels. Only **Level 1 is ACTIVE**. Levels 2 and 3 are **FUTURE** —
not yet started, listed here only to record scope and prevent re-litigation later.

### Level 1 — Layers (ACTIVE)
Make all **245** layer source files under `src/dl_techniques/layers/` production-quality, graded
against the §2 rubric, with a real test under `tests/test_layers/` for each concrete layer. This is
the entire active scope of this roadmap; §4 is its worklist.

### Level 2 — Models (ACTIVE — see PART II below)
Level 1 is COMPLETE (245/245). The same audit+fix discipline now applies to model definitions under
`src/dl_techniques/models/` (**70 model directories, 183 source files**). Models compose layers, so
they inherit the same serialization / `get_config` / `build` discipline plus model-level concerns
(factory functions, end-to-end `.keras` round-trip, weight handling). The Level-2 worklist,
rubric, per-round procedure, and handover prompt are in **PART II — LEVEL 2: MODELS** at the end of
this file (sections §L2-1…§L2-7). The canonical Level-2 exemplar is
`src/dl_techniques/models/bert/bert.py`. Level-1 sections §1–§7 below are frozen (DONE) and are not
touched by Level-2 rounds.

### Level 3 — Losses / Metrics / Optimizers / Regularizers / Initializers / Constraints (FUTURE, not started)
The remaining custom-component families (`losses/`, `metrics/`, `optimizers/`, `regularizers/`,
`initializers/`, `constraints/`) get their own audit pass last. These have a different rubric shape
(no `build`/`call` lifecycle, but `get_config`/registration and numerical-correctness tests still
apply). **Not in scope for any round here.**

---

## §2 The Production-Quality Rubric

Distilled from `src/dl_techniques/layers/convnext_v1_block.py` and
`research/2026_keras_custom_models_instructions.md` (Pattern 2, line 381). Each item is tagged
**[HARD]** (mandatory — a miss is a non-compliant file), **[SOFT]** (canonical style; fix when
reasonable), or **[GHOST]** (NOT a requirement — see the boxed note).

### [HARD] items — a file FAILS the rubric if any concrete layer misses one

| # | Item | What it means | Canonical / spec source |
|---|------|---------------|-------------------------|
| H1 | `register_keras_serializable` | `@keras.saving.register_keras_serializable()` decorator on every concrete layer class. | `convnext_v1_block.py:53`; spec §2.2 |
| H2 | Sublayers created in `__init__` | All sub-layers instantiated in `__init__` (never in `build`); `add_weight` NEVER in `__init__`. | `convnext_v1_block.py:180-289`; spec §1.1 |
| H3 | All `__init__` args stored | Every constructor parameter saved as `self.*` so it survives the `get_config` round-trip. | `convnext_v1_block.py:170-178`; spec §1.3 |
| H4 | Input validation | Invalid parameter values raise `ValueError` in `__init__`. | `convnext_v1_block.py:152-167`; spec §3.1/§5.1 |
| H5 | Explicit sublayer `build` | `build()` calls `.build(shape)` on each sub-layer in computational order with correct propagated/intermediate shapes. | `convnext_v1_block.py:291-338`; spec §1.1 |
| H6 | `super().build()` LAST | `super().build(input_shape)` is the **last statement** of `build()` (structural last-statement check, not line position). | `convnext_v1_block.py:338`; spec §1.1/§3.1 |
| H7 | `compute_output_shape` | Implemented on every layer, using **stored config** (e.g. `self.filters`) not weight shapes; works before build. | `convnext_v1_block.py:377-398`; spec §1.1/§3.4 |
| H8 | Full `get_config` | Calls `super().get_config()`, merges, and returns **every** `__init__` arg; serializes complex objects via `keras.regularizers.serialize` / `keras.initializers.serialize` / `keras.activations.serialize`. | `convnext_v1_block.py:408-419`; spec §8.1 |
| H9 | `from_config` deserialization | When config carries regularizer/initializer/activation objects, `from_config` deserializes them before `cls(**config)`. | `convnext_v1_block.py:422-440`; spec §8.1 |
| H10 | Graph-safe `call()` | Only `keras.ops` in the forward path — no raw `tf.*` tensor ops, no `list()`/`tuple()`/`.numpy()`/`int()` casts on symbolic tensors, no Python `if` on tensor values (use `ops.where`/`ops.cond`). | `convnext_v1_block.py:340-375`; spec §4.1-§4.3 |
| H11 | `training` forwarded | `call()` accepts `training` and forwards it to every sublayer that uses it. | `convnext_v1_block.py:340-375`; spec §2.3 |
| H12 | Type hints | Every method signature has parameter + return-type annotations. | `convnext_v1_block.py:137-149,291,340,377,400`; spec §2.3 |
| H13 | Structured docstrings | Class/method docstrings document params (Google `Args:`/`Returns:`/`Raises:` or RST `:param:`); a file with NO structured parameter docs is non-compliant. | spec §3.1; `convnext_v1_block.py` class docstring |
| H14 | Logger, not `print` | No `print()` in library code — use `from dl_techniques.utils.logger import logger`. | `src/dl_techniques/CLAUDE.md`; spec §2.1/§9.2 |

### [SOFT] items — canonical style; apply when reasonable, don't block a round on them

- **S1** Module-level docstring with title, paper reference, Key-Features / Architecture sections and
  an ASCII computation-flow diagram. (`convnext_v1_block.py:1-35`)
- **S2** Class-level `ALL_CAPS` constants for magic numbers (epsilon, stddev, expansion factors).
  (`convnext_v1_block.py:126-135`)
- **S3** Named sublayers via `name=` on every sublayer instantiation. (`convnext_v1_block.py:194…286`)
- **S4** `copy.deepcopy()` of shared mutable config (regularizers) when fanning out to several
  sublayers. (`convnext_v1_block.py:193,214`)
- **S5** Identity / passthrough fallback (`keras.layers.Lambda(lambda x: x)`) for optional sublayers
  so every attribute always exists as a Keras-tracked layer regardless of config — every conditional
  sublayer must assign the same attribute in its `else` branch. (`convnext_v1_block.py:251-289`)
- **S6** Separator comments around import blocks / at file end; inline comments on each sublayer
  construction block; explicit named intermediate shapes in `build()`;
  `isinstance(input_shape, list)` handling in `compute_output_shape` for multi-input variants;
  `config_copy = config.copy()` before mutating in `from_config`. (`convnext_v1_block.py` throughout)

### GHOST — do NOT grade on this

> **GHOST: `if self.built: return` at the top of `build()`.**
> **NEITHER** the canonical `convnext_v1_block.py` (whose `build()` spans lines 291-338 and does
> **not** contain it) **NOR** `research/2026_keras_custom_models_instructions.md` (Pattern 2, line
> 381) uses this guard. Its **ABSENCE is NOT a production-quality defect.** Do **NOT** pass/fail any
> file on it; do not add it as round work. The scanner `scripts/audit_layers.py` deliberately ignores
> it. (SYSTEM.md's "87-file `if self.built` gap" — actual live count 109 — is a stale prior-plan
> convention, not the user's stated standard. See decision D-003.) Adding the guard is harmless but is
> never required and must never inflate a round.

### Scanner-mechanical vs human-judged

`scripts/audit_layers.py` mechanically (via Python `ast`) checks the **mechanical HARD items**:
H1 (`register_decorator`), H7 (`compute_output_shape` present), H6 (`super_build_last`),
H8 (`get_config` present), H10 (`forward_raw_tf` — raw `tf.` call inside `call()`), and H14
(`print_call`). It also reports `build_present`, `sublayer_build` (info), and a best-effort
`type_hints` flag. Everything else is **human-judged** during the round: H2/H3/H4/H5/H9/H11
correctness, H12/H13 quality, and ALL SOFT items. The scanner is an **aid, not an oracle** — it can
mislabel an aliased import or a metaclass ABC; the round author grades each file against the FULL
rubric by reading it.

---

## §3 How a Round Works

A "round" is one self-contained, context-cleared working session that clears a single batch from §4.
The standing user decisions baked into this procedure:

- **FULL re-audit.** Do NOT trust any prior "FULLY RESOLVED" / "DONE" status from SYSTEM.md or older
  plans. Every file is graded fresh against the §2 rubric, including subpackages that were previously
  marked resolved (they are the LAST rounds precisely so they still get a real re-read).
- **Audit AND fix in the SAME round.** A round both grades and repairs its batch; it does not defer
  fixes to a later pass.
- **One round per cleared-context session.** Pick the next PENDING round, do it end-to-end, commit,
  stop. Don't chain rounds in one context.
- **Push is the user's job.** Commit locally only; never push.

Per-round procedure:

1. **Read this file.** Find the next round in §4 whose status is PENDING (`[ ]` rows). That round's
   file list is the batch.
2. **Mechanical audit.** Run the scanner over just the batch:
   ```
   .venv/bin/python scripts/audit_layers.py --path <space-separated batch files>
   ```
   or, for a whole subpackage batch:
   ```
   .venv/bin/python scripts/audit_layers.py --subpackage <name>
   ```
   (Add `--json /tmp/round.json` for a machine-readable report.)
3. **Grade + fix each file.** For EVERY file in the batch: open it, grade against the FULL §2 rubric.
   The scanner covers the mechanical HARD items; SOFT items and docstring/type-hint quality are
   human-judged. Fix **all [HARD] gaps** and **reasonable [SOFT] gaps**. Match the canonical
   `convnext_v1_block.py` style. Do NOT add `if self.built: return` (GHOST — §2).
4. **Tests.** Ensure a test exists under `tests/test_layers/…` mirroring the source path. If missing
   or thin, add/repair one covering: construction (incl. `ValueError` paths), forward pass,
   `.keras` serialization round-trip (save → load → identical output), and `compute_output_shape`
   vs the actual `call()` output shape.
5. **Run scoped tests** (GPU1, serial — repo convention; never run GPU jobs in parallel):
   ```
   CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_layers/<batch tests> -x
   ```
6. **Re-run the scanner** on the batch to confirm every file now reports PASS (or a documented,
   accepted N/A — see §5 for accepted raw-`tf` exceptions).
7. **Update §4 in place.** Flip the batch's `[ ]` → `[x]`, set the round's status to DONE, and bump
   the top-of-§4 progress tally ("X / 245").
8. **Commit** (locally; do not push):
   ```
   git commit -m "[production-map/round-N] <short batch description>"
   ```

---

## §4 Batched Worklist

**Progress: 245 / 245 files production-verified**  (Level 1 COMPLETE; `canny.py` + `complex_layers.py` carry documented accepted-exception headers for forward-path raw-`tf`, see §5)

Status legend: `[ ]` PENDING · `[~]` IN-PROGRESS · `[x]` DONE.
`verdict` is the current `scripts/audit_layers.py` mechanical result at baseline `2d96078a`
(re-run the scanner during the round — files change). `gap-hint` is the scanner's failing HARD
item(s) for FAIL files, `N/A` for non-Layer modules (Enums / dataclasses / pure-fn / factory-only —
still human-confirmed during the round), or `rubric-verify` for current-PASS files that still need a
human SOFT/docstring/type-hint review per §2/§3.

Priority order: **Rounds 1-5** = NO-TESTS + untested files (highest risk: zero coverage). **Rounds
6-12** = NEEDS-AUDIT subpackages. **Rounds 13-31** = full re-audit of previously "resolved"
subpackages and tested root files. 31 rounds total; 245 file rows.

<!-- WORKLIST -->
### Round 1 — physics/ (NO test dir) + tokenizers/bpe.py  (3 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `physics/approximate_lagrange_layer.py` | PASS | done: +module docstring (S1), +`training` fwd (H11); build-time input-dependent output proj accepted |
| `[x]` | `physics/lagrange_layer.py` | FAIL→ACCEPTED | done: H10 accepted-exception documented in header (`tf.GradientTape`/`tf.linalg.pinv`, no `keras.ops` eq — §5); fixed real batch_jacobian forward bug; +`training` fwd |
| `[x]` | `tokenizers/bpe.py` | PASS | done: +`max_length` validation (H4, BPETokenizer); +input validation (H4) & explicit `build()` (H5, TokenEmbedding) |

### Round 2 — graphs/ (4 of 5 untested) + heads/vlm/  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `graphs/entity_graph_refinement.py` | PASS | done: H6 fixed (moved log before super().build()); existing test covers round-trip |
| `[x]` | `graphs/fermi_diract_decoder.py` | PASS | done: clean; +new test |
| `[x]` | `graphs/graph_neural_network.py` | PASS | done: deserialize regularizers in __init__ (H8/H9); +new test |
| `[x]` | `graphs/relational_graph_transformer_blocks.py` | PASS | done: regularizer deserialize (H8/H9) + RELGTTokenEncoder H5 (build type/hop embeddings); +new test |
| `[x]` | `graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py` | PASS | done: raw-tf re-check resolved — tf.constant in build()→keras.ops, dropped tf import; +new test |
| `[x]` | `heads/vlm/factory.py` | PASS | done: H7 compute_output_shape on 4 heads + H8/H9 task_config enum serialize/from_config across all 6 heads; +tests |
| `[x]` | `heads/vlm/task_types.py` | N/A | N/A confirmed (VLMTaskType Enum + VLMTaskConfig dataclass; no concrete layer) |

### Round 3 — untested root-level files (1/3)  (11 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `anchor_generator.py` | PASS | done: clean; +new test |
| `[x]` | `bitlinear_layer.py` | PASS | done: H6 (super().build last) + H2 (input_norm to __init__) + fixed real per-channel-rescale forward bug (units!=input_dim); +test |
| `[x]` | `blt_blocks.py` | PASS | done: H6 ×4 (EntropyModel/LocalEncoder/GlobalTransformer/LocalDecoder super().build last); +test (7 layers) |
| `[x]` | `clahe.py` | PASS | done: raw-tf re-check — tf.histogram_fixed_width is forward-path accepted-exception (documented header); removed @tf.function (broken graph trace) → eager; +test |
| `[x]` | `conditional_output_layer.py` | PASS | done: clean (stateless); +test |
| `[x]` | `conv2d_builder.py` | N/A | N/A confirmed (ConvType Enum + factory fns) |
| `[x]` | `depthwise_separable_block.py` | PASS | done: H6 (logger before super().build()) + H8/H9 regularizer deserialize; +test |
| `[x]` | `downsample.py` | N/A | N/A confirmed (pure-function module) |
| `[x]` | `eomt_mask.py` | PASS | done: H8/H9 regularizers+constraints deserialize in __init__; +test |
| `[x]` | `fft_layers.py` | PASS | done: +module docstring (S1); +test (FFT+IFFT) |
| `[x]` | `film.py` | PASS | done: clean (input-dependent projections built in build()); +test |

### Round 4 — untested root-level files (2/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `fractal_block.py` | PASS | done: H6 (logger before super().build()); +test |
| `[x]` | `hierarchical_mlp_stem.py` | PASS | done: clean (existing `if self.built` is GHOST — left as-is); +test |
| `[x]` | `inverted_residual_block.py` | PASS* | done: concrete layer (scanner N/A mislabel — subclasses UIB); fixed broken from_config (forced name= collision) + S1 module docstring; +test |
| `[x]` | `io_preparation.py` | PASS | done: clean (4 preprocessing layers); +test |
| `[x]` | `modality_projection.py` | PASS | done: H5 (explicit sublayer build, was lazy) + H6 (super().build() last); +test |
| `[x]` | `mothnet_blocks.py` | PASS | done: H8/H9 regularizer deserialize (AntennalLobe + HebbianReadout); RBM-style clean; +test |
| `[x]` | `one_hot_encoding.py` | PASS | done: H4 cardinalities validation; +test |
| `[x]` | `patch_merging.py` | PASS | done: H5 (added explicit build() — sublayers were lazy-only); +test |
| `[x]` | `random_fourier_features.py` | PASS | done: H8/H9 regularizer+constraint deserialize; fixed list/tuple build-shape concat bug on reload; +test |
| `[x]` | `restricted_boltzmann_machine.py` | PASS | done: clean (regularizer via get; "tf.GradientTape" is a comment only — code uses ops); +test |

### Round 5 — untested root-level files (3/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `router.py` | PASS | done: DEEP gap resolved — only raw-tf was `tf.minimum`→`ops.minimum` (clean migration, not deep); dropped tf import; +test |
| `[x]` | `selective_gradient_mask.py` | PASS | done: clean (stateless STE); +test |
| `[x]` | `sparse_autoencoder.py` | PASS | done: clean (uses .get + from_config); +test (5 variants) |
| `[x]` | `spatial_layer.py` | PASS | done: fixed broken forward — `ops.image.resize(image=...)`→`images=` (wrong kwarg); +test |
| `[x]` | `standard_blocks.py` | PASS | done: H8/H9 regularizer+initializer+constraint deserialize across all 5 layers; +test |
| `[x]` | `stochastic_gradient.py` | PASS | done: clean; +test |
| `[x]` | `strong_augmentation.py` | PASS | done: H4 validation + H12 compute_output_shape type hints; +test |
| `[x]` | `tabm_blocks.py` | PASS | done: fixed MLPBlock/TabMBackbone H2+H5 (sublayers created in build & not built → broken weight restore); +test (5 layers) |
| `[x]` | `universal_inverted_bottleneck.py` | PASS | done: clean; +test |
| `[x]` | `upsample.py` | N/A | N/A confirmed (pure-function module) |

### Round 6 — memory/ (NEEDS-AUDIT)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `memory/baseline_ntm.py` | PASS | done: scanner-clean + no raw-tf/reg/print; comprehensive existing test passes |
| `[x]` | `memory/factory.py` | N/A | N/A confirmed (create_mann / create_som_2d functions only) |
| `[x]` | `memory/mann.py` | PASS | done: clean rubric; +new test (LSTM path full; GRU forward xfail — keras GRU return_state drops batch dim upstream) |
| `[x]` | `memory/neuro_grid.py` | PASS | done: scanner-clean + no raw-tf/reg/print; comprehensive existing test passes |
| `[x]` | `memory/ntm_interface.py` | N/A | N/A confirmed (Enums + dataclasses + ABC interfaces w/ @abstractmethod) |
| `[x]` | `memory/som_2d_layer.py` | N/A* | concrete SOMLayer subclass (scanner N/A mislabel); inherits fixed build; comprehensive existing test passes |
| `[x]` | `memory/som_nd_layer.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `memory/som_nd_soft_layer.py` | PASS | done: H6 (logger before super().build()); existing test passes |

### Round 7 — moe/ + fusion/ (NEEDS-AUDIT)  (6 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `moe/config.py` | N/A | N/A confirmed (ExpertConfig/GatingConfig/MoEConfig dataclasses) |
| `[x]` | `moe/experts.py` | PASS* | concrete FFNExpert (scanner N/A mislabel — subclasses BaseExpert ABC); H6-compliant, clean; +direct test |
| `[x]` | `moe/gating.py` | PASS* | 3 concrete gating layers (scanner N/A mislabel — subclass BaseGating ABC); H6-compliant, init via .get; +direct test (incl. CosineGating serialization gap) |
| `[x]` | `moe/integration.py` | N/A | N/A confirmed (MoETrainingConfig dataclass + MoEOptimizerBuilder helper) |
| `[x]` | `moe/layer.py` | PASS | done: H6 (logger before super().build()); comprehensive existing test passes |
| `[x]` | `fusion/multimodal_fusion.py` | PASS | done: clean (init via .get, super().build() last, no raw-tf); existing test passes |

### Round 8 — statistics/ (NEEDS-AUDIT; 3 dead-code candidates)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `statistics/deep_kernel_pca.py` | PASS | KEPT (user decision; 0 consumers): H8/H9 regularizer deserialize ×3; existing test passes |
| `[x]` | `statistics/invertible_kernel_pca.py` | PASS | KEPT (user decision; 0 consumers): H8/H9 regularizer deserialize ×2; existing test passes |
| `[x]` | `statistics/mdn_layer.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `statistics/moving_std.py` | PASS | done: clean; existing test passes |
| `[x]` | `statistics/normalizing_flow.py` | PASS | done: clean; existing test passes |
| `[x]` | `statistics/residual_acf.py` | PASS | KEPT (user decision; 0 consumers): clean; existing test passes |
| `[x]` | `statistics/scaler.py` | PASS | done: clean; existing test passes |

> **Round 8 dead-code decision (§5):** user chose KEEP for all 3 zero-consumer candidates
> (`deep_kernel_pca`, `invertible_kernel_pca`, `residual_acf`). Denominator stays 245.

### Round 9 — time_series/ (1/2, NEEDS-AUDIT)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `time_series/adaptive_lag_attention.py` | PASS | done: H6 (logger before super().build()); +test |
| `[x]` | `time_series/deepar_blocks.py` | PASS | done: H7 compute_output_shape on DeepARCell (RNN cell); +test (4 layers) |
| `[x]` | `time_series/ema_layer.py` | PASS | done: clean; existing test passes |
| `[x]` | `time_series/forecasting_layers.py` | PASS | done: H8/H9 kernel_regularizer deserialize ×2 (NaiveResidual/ConformalQuantileHead); +test (3 layers) |
| `[x]` | `time_series/mixed_sequential_block.py` | PASS | done: clean; existing test passes |
| `[x]` | `time_series/nbeats_blocks.py` | PASS* | concrete Generic/Trend/Seasonality blocks (scanner N/A mislabel — subclass NBeatsBlock); super().build() last, clean; existing test passes |
| `[x]` | `time_series/nbeatsx_blocks.py` | PASS* | concrete ExogenousBlock (scanner N/A mislabel); H6 (super().build() was not last — encoder built after) + S1 module docstring; +test |

### Round 10 — time_series/ (2/2, NEEDS-AUDIT)  (6 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `time_series/prism_blocks.py` | PASS | done: clean (5 PRISM layers); +test (needs seq_len>=32 for wavelet bands) |
| `[x]` | `time_series/quantile_head_fixed_io.py` | PASS | done: clean; existing test passes |
| `[x]` | `time_series/quantile_head_variable_io.py` | PASS | done: clean; existing test passes |
| `[x]` | `time_series/temporal_convolutional_network.py` | PASS | done: clean (str-passthrough initializer, self-consistent); existing test passes |
| `[x]` | `time_series/temporal_fusion.py` | PASS | done: H6 (logger before super().build()); +test |
| `[x]` | `time_series/xlstm_blocks.py` | PASS | done: H7 compute_output_shape on sLSTMCell/mLSTMCell + H8/H9 regularizer deserialize (6 classes); existing test passes |

### Round 11 — logic/ + reasoning/ + geometric/ (NEEDS-AUDIT)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `logic/arithmetic_operators.py` | PASS | done: clean (no raw-tf/reg/print); existing test passes |
| `[x]` | `logic/factory.py` | N/A | N/A confirmed (create_logic_layer / create_logic_from_config functions only — no inline layers) |
| `[x]` | `logic/logic_operators.py` | PASS | done: clean; existing test passes |
| `[x]` | `logic/neural_circuit.py` | PASS | done: clean; existing test passes |
| `[x]` | `reasoning/hrm_reasoning_core.py` | PASS | done: clean; existing test passes |
| `[x]` | `reasoning/hrm_reasoning_module.py` | PASS | done: clean; existing test passes |
| `[x]` | `reasoning/hrm_sparse_puzzle_embedding.py` | PASS | done: clean; existing test passes |
| `[x]` | `geometric/clifford_block.py` | PASS | done: clean (7 layers, all direct Layer subclasses); existing tests pass |
| `[x]` | `geometric/point_cloud_autoencoder.py` | PASS | done: clean; existing test passes |
| `[x]` | `geometric/supernode_pooling.py` | PASS | done: clean; existing test passes |

### Round 12 — heads/ remaining (nlp / vision / root, NEEDS-AUDIT)  (6 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `heads/factory.py` | N/A | N/A confirmed (thin create_head dispatch facade; no layer) |
| `[x]` | `heads/nlp/factory.py` | PASS | done: rubric-verified clean (compute_output_shape+from_config+task_config serialize on all heads; no raw-tf/print) |
| `[x]` | `heads/nlp/task_types.py` | N/A | N/A confirmed (NLPTaskType Enum + NLPTaskConfig dataclass) |
| `[x]` | `heads/task_types.py` | N/A | N/A confirmed (aggregator re-exporting domain enums/configs) |
| `[x]` | `heads/vision/factory.py` | PASS | done: H7 compute_output_shape on all 8 heads + H5 explicit sublayer build (compute_output_shape bypasses call-trace → lazy builds dropped on reload) + H6 super().build() last; +tests |
| `[x]` | `heads/vision/task_types.py` | N/A | N/A confirmed (VisionTaskType Enum + TaskConfiguration/helpers) |

### Round 13 — attention/ re-audit (1/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `attention/anchor_attention.py` | PASS | done: rubric-verified clean (H3/H5/H8/H9/H11 all good) |
| `[x]` | `attention/attention_routing_capsule.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `attention/capsule_routing_attention.py` | PASS | done: clean (activity_regularizer round-trips via base Layer.get_config) |
| `[x]` | `attention/channel_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/convolutional_block_attention.py` | PASS | done: rubric-verified clean (CBAM) |
| `[x]` | `attention/differential_attention.py` | PASS | done: clean (activity_regularizer via base) |
| `[x]` | `attention/factory.py` | N/A | N/A confirmed (create_attention_layer registry/factory fns) |
| `[x]` | `attention/fnet_fourier_transform.py` | PASS | done: rubric-verified clean (parameter-free DFT layer) |
| `[x]` | `attention/gated_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/group_query_attention.py` | PASS | done: H12 (_apply_mask type hints); existing test passes |

### Round 14 — attention/ re-audit (2/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `attention/hopfield_attention.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `attention/ideogram4_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/lighthouse_attention.py` | PASS | done: clean (un-forwarded sublayers are training-invariant RMSNorm/softmax) |
| `[x]` | `attention/mmdit_joint_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/mobile_mqa.py` | PASS* | done: concrete MobileMQA (subclass GQA; scanner N/A mislabel); H6 (super().build() was not last — lambda/downsample built after); H7 via inherited compute_output_shape; existing test passes |
| `[x]` | `attention/multi_head_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/multi_head_cross_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/multi_head_latent_attention.py` | PASS | done: clean (un-forwarded norms training-invariant; dropout gets training) |
| `[x]` | `attention/non_local_attention.py` | PASS | done: rubric-verified clean (H3/H5/H8/H9/H11 good) |
| `[x]` | `attention/perceiver_attention.py` | PASS | done: rubric-verified clean |

### Round 15 — attention/ re-audit (3/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `attention/performer_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/progressive_focused_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/ring_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/rpc_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/shared_weights_cross_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/single_window_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/spatial_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/tripse_attention.py` | PASS | done: H12 (_SEWeights build/call/compute_output_shape/get_config type hints); existing test passes |
| `[x]` | `attention/wave_field_attention.py` | PASS | done: rubric-verified clean |
| `[x]` | `attention/window_attention.py` | PASS | done: H9 self-consistency (resolve initializers/regularizers via keras.*.get in __init__); existing test passes |

### Round 16 — embedding/ re-audit (1/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `embedding/albert_factorized_embedding.py` | PASS | done: rubric-verified clean |
| `[x]` | `embedding/bert_embeddings.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `embedding/class_token.py` | PASS | done: clean (str-typed initializer round-trips verbatim by design) |
| `[x]` | `embedding/continuous_rope_embedding.py` | PASS | done: rubric-verified clean (parameter-only RoPE) |
| `[x]` | `embedding/continuous_sin_cos_embedding.py` | PASS | done: rubric-verified clean |
| `[x]` | `embedding/dual_rotary_position_embedding.py` | PASS | done: rubric-verified clean |
| `[x]` | `embedding/factory.py` | N/A | N/A confirmed (create_embedding_layer registry/factory fns) |
| `[x]` | `embedding/hierarchical_codebook_embedding.py` | PASS | done: rubric-verified clean (H5/H9/H11 good) |

### Round 17 — embedding/ re-audit (2/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `embedding/mask_token.py` | PASS | done: rubric-verified clean |
| `[x]` | `embedding/modern_bert_embeddings.py` | PASS | done: rubric-verified clean |
| `[x]` | `embedding/multi_axis_rope.py` | PASS | done: rubric-verified clean (parameter-only RoPE) |
| `[x]` | `embedding/patch_embedding.py` | PASS | done: H6 ×2 (logger before super().build() on PatchEmbedding2D + 1D); existing test passes |
| `[x]` | `embedding/positional_embedding.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `embedding/positional_embedding_sine_2d.py` | PASS | done: H12 (__init__ -> None return hint); existing test passes |
| `[x]` | `embedding/rotary_position_embedding.py` | PASS | done: rubric-verified clean |
| `[x]` | `embedding/scalar_sinusoidal_embedding.py` | PASS | done: rubric-verified clean |

### Round 18 — activations/ re-audit (1/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `activations/adaptive_softmax.py` | PASS | done: rubric-verified clean |
| `[x]` | `activations/basis_function.py` | PASS | done: rubric-verified clean (stateless) |
| `[x]` | `activations/differentiable_step.py` | PASS | done: rubric-verified clean (init/reg/constraint via keras.*.get) |
| `[x]` | `activations/expanded_activations.py` | PASS | done: rubric-verified clean |
| `[x]` | `activations/factory.py` | N/A | N/A confirmed (create_activation_layer registry/factory fns; no inline layers) |
| `[x]` | `activations/golu.py` | PASS | done: rubric-verified clean; +new test |
| `[x]` | `activations/hard_sigmoid.py` | PASS | done: H5 (explicit build of wrapped ReLU6 sublayer); +new test |
| `[x]` | `activations/hard_swish.py` | PASS | done: H5 (explicit build of wrapped ReLU6 sublayer); +new test |

### Round 19 — activations/ re-audit (2/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `activations/mish.py` | PASS | done: rubric-verified clean (Mish + SaturatedMish) |
| `[x]` | `activations/monotonicity_layer.py` | PASS | done: rubric-verified clean |
| `[x]` | `activations/probability_output.py` | PASS | done: rubric-verified clean (strategy_layer built; type_config serialize/from_config) |
| `[x]` | `activations/relu_k.py` | PASS | done: rubric-verified clean (H4 type+value validation); +new test |
| `[x]` | `activations/routing_probabilities.py` | PASS | done: H6 (super().build() moved to end; M7 early-build was traceback-only; routing save/load FuncGraph tests pass) |
| `[x]` | `activations/sparsemax.py` | PASS | done: rubric-verified clean (H4 axis validation); +new test |
| `[x]` | `activations/squash.py` | PASS | done: H4 (axis/epsilon validation added); +new test |
| `[x]` | `activations/thresh_max.py` | PASS | done: rubric-verified clean (init/reg/constraint via keras.*.get) |

### Round 20 — ffn/ re-audit (1/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `ffn/counting_ffn.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/diff_ffn.py` | PASS | done: clean (Dense sublayers training-invariant; real dropout gets training) |
| `[x]` | `ffn/factory.py` | N/A | N/A confirmed (create_ffn_layer registry/factory fns) |
| `[x]` | `ffn/gated_mlp.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/geglu_ffn.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/gelu_mlp_ffn.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/glu_ffn.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/kan_linear.py` | PASS | done: §5 raw-tf re-check RESOLVED — false positive (only docstring "tf.function" prose; call()/build()/helpers are keras.ops only). H10 clean |

### Round 21 — ffn/ re-audit (2/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `ffn/logic_ffn.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `ffn/mlp.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/orthoglu_ffn.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/power_mlp_layer.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `ffn/residual_block.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/swiglu_ffn.py` | PASS | done: rubric-verified clean |
| `[x]` | `ffn/swin_mlp.py` | PASS | done: clean (fc2 deferred to build — runtime input-dim dependency when output_dim=None; guarded+explicitly built, round-trips) |
| `[x]` | `ffn/tversky_projection.py` | PASS | done: rubric-verified clean (add_weight-only; init via keras.*.get) |

### Round 22 — norms/ re-audit (1/2)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `norms/adaptive_band_rms.py` | PASS | done: rubric-verified clean (dense_layer built; band init/reg via keras.*.get) |
| `[x]` | `norms/band_logit_norm.py` | PASS | done: rubric-verified clean (LayerNorm sublayer built; training forwarded) |
| `[x]` | `norms/band_rms.py` | PASS | done: rubric-verified clean |
| `[x]` | `norms/dynamic_tanh.py` | PASS | done: rubric-verified clean (init/reg/constraint via keras.*.get) |
| `[x]` | `norms/factory.py` | N/A | N/A confirmed (create_normalization_layer registry/factory fns) |
| `[x]` | `norms/global_response_norm.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `norms/logit_norm.py` | PASS | done: rubric-verified clean |

### Round 23 — norms/ re-audit (2/2)  (6 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `norms/max_logit_norm.py` | PASS | done: rubric-verified clean (3 classes: MaxLogitNorm/DecoupledMaxLogit/DMLPlus) |
| `[x]` | `norms/polar_weight_norm.py` | PASS | done: §5 raw-tf re-check RESOLVED — false positive (zero tf tokens; forward path keras.ops only; build() numpy is static seed-encoding). H10 clean |
| `[x]` | `norms/rms_norm.py` | PASS | done: rubric-verified clean |
| `[x]` | `norms/zero_centered_adaptive_band_rms_norm.py` | PASS | done: rubric-verified clean (dense_layer built; training forwarded) |
| `[x]` | `norms/zero_centered_band_rms_norm.py` | PASS | done: rubric-verified clean |
| `[x]` | `norms/zero_centered_rms_norm.py` | PASS | done: rubric-verified clean |

### Round 24 — transformers/ re-audit (1/2)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `transformers/adaln_zero.py` | PASS | done: rubric-verified clean |
| `[x]` | `transformers/eomt_transformer.py` | PASS | done: rubric-verified clean |
| `[x]` | `transformers/free_transformer.py` | PASS | done: rubric-verified clean |
| `[x]` | `transformers/ideogram4_block.py` | PASS | done: rubric-verified clean (2 classes) |
| `[x]` | `transformers/perceiver_transformer.py` | PASS | done: fixed real serialization bug (build() misread reloaded single shape as 2 inputs — disambiguate by element type); +new test |
| `[x]` | `transformers/progressive_focused_transformer.py` | PASS | done: fixed real serialization bug (build() `(None,)+x_shape[1:]` failed on reloaded list shape — coerce to tuple); +new test |
| `[x]` | `transformers/sd3_adaln.py` | PASS | done: rubric-verified clean |

### Round 25 — transformers/ re-audit (2/2)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `transformers/swin_conv_block.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `transformers/swin_transformer_block.py` | PASS | done: H6 (logger before super().build()); existing test passes |
| `[x]` | `transformers/text_decoder.py` | PASS | done: H11 (forward training to embed_norm/final_norm, matching text_encoder); existing test passes |
| `[x]` | `transformers/text_encoder.py` | PASS | done: rubric-verified clean (list/dict input disambiguated) |
| `[x]` | `transformers/transformer.py` | PASS | done: rubric-verified clean (moe_config serialized; all conditionals guarded) |
| `[x]` | `transformers/transformer_decoder.py` | PASS | done: rubric-verified clean (cross-attn built with [dec,enc]; tuple-cast shapes) |
| `[x]` | `transformers/vision_encoder.py` | PASS | done: rubric-verified clean |

### Round 26 — mixtures/ + sequence_pooling/ re-audit  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `mixtures/factory.py` | N/A | N/A confirmed (MixtureType + create_mixture_layer/from_config fns) |
| `[x]` | `mixtures/gmm.py` | PASS | done: rubric-verified clean (init/reg serialize+from_config; orthonormal-string branch handled) |
| `[x]` | `mixtures/kmeans.py` | PASS | done: rubric-verified clean |
| `[x]` | `mixtures/radial_basis_function.py` | PASS | done: rubric-verified clean (init/constraint/2 regularizers serialized) |
| `[x]` | `sequence_pooling/attention_pooling.py` | PASS | done: H6 (super().build() moved last; sublayer build + add_weight were after it) |
| `[x]` | `sequence_pooling/factory.py` | N/A | N/A confirmed (create_sequence_pooling factory fns) |
| `[x]` | `sequence_pooling/sequence_pooling.py` | PASS | done: H6 (super().build() moved last) |
| `[x]` | `sequence_pooling/weighted_pooling.py` | PASS | done: H6 (super().build() moved last) |

### Round 27 — tested root-level files re-audit (1/5)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `bias_free_conv1d.py` | PASS | done: clean (shortcut_conv is build-time-dim dependent — accepted, built in build(); conv training-invariant) |
| `[x]` | `bias_free_conv2d.py` | PASS | done: H3+H8 fixed real bug — `use_batch_norm` was dropped (not stored/not in get_config → silent default on reload); +existing test passes |
| `[x]` | `blt_core.py` | PASS | done: H12 type hints on build/call/compute_output_shape/get_config/empty_carry/reset_carry/_create_reasoning_embeddings; initializers round-trip via child layers; existing test round-trips |
| `[x]` | `blur_pool.py` | PASS | done: H6 (logger before super().build()) |
| `[x]` | `canny.py` | FAIL→ACCEPTED | done: H10 accepted-exception documented in header (`tf.nn.dilation2d` directional NMS + hysteresis dilation, `tf.while_loop` hysteresis fixed-point; keras.ops has NO morphological-dilation primitive — §5); user-approved accept. Existing test passes |
| `[x]` | `capsules.py` | PASS | done: H6 ×3 (PrimaryCapsule/RoutingCapsule/CapsuleBlock — logger before super().build()) |
| `[x]` | `complex_layers.py` | FAIL→ACCEPTED | done: H10 accepted-exception documented in header (`tf.complex`/`complex64` dtype + `tf.math.real`/`imag` across layers; keras.ops has real/imag but NO complex constructor/dtype — §5); user-approved accept. Existing test passes |
| `[x]` | `convnext_v1_block.py` | PASS | done: CANONICAL EXEMPLAR re-confirmed fully compliant (H3/H5/H8/H9/H10/H11/H12/H13) |
| `[x]` | `convnext_v2_block.py` | PASS | done: rubric-verified clean |
| `[x]` | `convolutional_kan.py` | PASS | done: H6 (logger before super().build()) |

### Round 28 — tested root-level files re-audit (2/5)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `dynamic_conv2d.py` | PASS | done: rubric-verified clean (input-dim-dependent conv/attention sublayers built explicitly in build(); H8/H9 via keras.*.get; existing test passes) |
| `[x]` | `fnet_encoder_block.py` | PASS | done: rubric-verified clean (factory norm/ffn built in build(); from_config trivial; existing test passes) |
| `[x]` | `gated_delta_net.py` | PASS | done: rubric-verified clean (ops.while_loop delta rule; H8/H9 via initializers.get; existing test test_gated_deltanet.py passes) |
| `[x]` | `gaussian_filter.py` | PASS | done: rubric-verified clean (add_weight kernel in build; `if self.built` is GHOST — left as-is; existing test passes) |
| `[x]` | `gaussian_pyramid.py` | PASS | done: rubric-verified clean (sublayers in __init__, built in build; existing test passes) |
| `[x]` | `global_sum_pool_2d.py` | PASS | done: rubric-verified clean (stateless pooling; existing test passes) |
| `[x]` | `grid_sample.py` | N/A | N/A confirmed (pure-function module; TF-native by documented decision D-003) |
| `[x]` | `haar_wavelet_decomposition.py` | PASS | done: rubric-verified clean (1D/2D/3D DWT, ops-only; existing test passes) |
| `[x]` | `hanc_block.py` | PASS | done: rubric-verified clean (sublayers in __init__, built in build, training fwd; existing test passes) |
| `[x]` | `hanc_layer.py` | PASS | done: rubric-verified clean (ops.image.resize, training fwd to BN; existing test passes) |

### Round 29 — tested root-level files re-audit (3/5)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `laplacian_filter.py` | PASS | done: rubric-verified clean (DoG/LoG/kernel; GaussianFilter sublayer built; existing test passes) |
| `[x]` | `layer_scale.py` | PASS | done: rubric-verified clean (LearnableMultiplier; add_weight in build, super().build last; existing test passes) |
| `[x]` | `mobile_one_block.py` | PASS | done: rubric-verified clean (Conv-BN branches built explicitly; H8/H9 serialize; existing test passes) |
| `[x]` | `mps_layer.py` | PASS | done: rubric-verified clean (tensor-train; add_weight in build; existing test passes) |
| `[x]` | `multi_level_feature_compilation.py` | PASS | done: rubric-verified clean (MLFCLayer 4-input; ops.image.resize; test_mlfc_layers.py passes) |
| `[x]` | `orthoblock.py` | PASS | done: rubric-verified clean (Dense+RMS+gate pipeline; sublayers built; existing test passes) |
| `[x]` | `orthogonal_butterfly.py` | PASS | done: rubric-verified clean (Givens butterfly; add_weight in build; existing test passes) |
| `[x]` | `pixel_shuffle.py` | PASS | done: rubric-verified clean (ViT token space-to-depth; existing test passes) |
| `[x]` | `pixel_unshuffle.py` | PASS | done: H6 fixed (logger.debug moved before super().build() in PixelUnshuffle2D); PixelShuffle2D clean; test passes |
| `[x]` | `repmixer_block.py` | PASS | done: rubric-verified clean (RepMixerBlock + ConvolutionalStem; sublayers built; existing test passes) |

### Round 30 — tested root-level files re-audit (4/5)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `res_path.py` | PASS | done: rubric-verified clean (residual SE blocks; sublayers built; existing test passes) |
| `[x]` | `rigid_simplex_layer.py` | PASS | done: rubric-verified clean (frozen ETF + rotation; add_loss; from_config deserializes init; existing test passes) |
| `[x]` | `sampling.py` | PASS | done: §5 raw-tf RESOLVED — `tf.math.bessel_i0e` is sole accepted-exception (D-001; no keras.ops Bessel) and lives in `_log_iv`→`vmf_kl_divergence`, OFF all 3 call() forward paths; existing test passes |
| `[x]` | `shearlet_transform.py` | PASS | done: rubric-verified clean (NumPy filter gen static in build off forward path; call uses ops.fft2 only; existing test passes) |
| `[x]` | `squeeze_excitation.py` | PASS | done: rubric-verified clean (2D/3D/4D SE; sublayers built; existing test passes) |
| `[x]` | `stochastic_depth.py` | PASS | done: rubric-verified clean (stateless drop-path; keras.random; existing test passes) |
| `[x]` | `thera_heat_field.py` | PASS | done: rubric-verified clean (ThermalActivation + multi-input HeatField; einsum; existing test passes) |
| `[x]` | `vector_quantizer.py` | PASS | done: rubric-verified clean (STE VQ + EMA; add_weight in build; existing test passes) |
| `[x]` | `vector_quantizer_rotation_trick.py` | PASS | done: rubric-verified clean (rotation-trick VQ; k-means warm-start is opt-in eager, default forward pure ops; existing test passes) |
| `[x]` | `yolo12_blocks.py` | PASS | done: rubric-verified clean (6 classes ConvBlock/AreaAttention/AttentionBlock/Bottleneck/C3k2/A2C2f; sublayers built; existing test passes) |

### Round 31 — tested root-level files re-audit (5/5)  (1 file)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `yolo12_heads.py` | PASS | done: rubric-verified clean (3 multi-input heads Detection/Segmentation/Classification; input-dependent Sequential branches built explicitly in build(); H6/H7/H8/H11 good; existing test passes) |
<!-- /WORKLIST -->

---

## §5 Known DEEP-gap + Dead-code

These files need extra care and may warrant slower handling within their assigned round.

### Forward-path raw `tf.*` — reconcile of two measurements

Two independent measurements disagree on which files carry raw `tf.*`, and the difference is the whole
point of using an AST scanner over grep:

- **Scanner (AST, forward-path-only — authoritative for pass/fail):** flags raw `tf.*` **inside the
  `call()` method** of a concrete layer in **4 files** (7 layers total):
  `canny.py` (Round 27), `complex_layers.py` (Round 27), `physics/lagrange_layer.py` (Round 1),
  `router.py` (Round 5). These are the genuine H10 (graph-safe `call`) violations.
- **Findings grep (broader — `tf.` anywhere in the file body, non-import lines):** the
  `findings/compliance-gaps.md` "11-file" list additionally named
  `sampling.py` (Round 30), `clahe.py` (Round 3), `norms/polar_weight_norm.py` (Round 23),
  `graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py` (Round 2),
  `ffn/kan_linear.py` (Round 20), plus `grid_sample.py` and `conv2d_builder.py`. The last two are
  intentionally **not** Layer subclasses (pure-function module / Enum — scanner reports N/A; not
  defects). The other five show their `tf.*` **outside** the `call()` forward path (e.g. in `build`,
  helper methods, or static setup), so the scanner does NOT flag them as H10 violations — but they are
  marked `rubric-verify` with a re-check note in §4 because a human should confirm the `tf.*` is truly
  off the graph-traced forward path (and migrate to `keras.ops` where trivially possible).

**Reconciled DEEP-gap (true forward-path) list — 4 files / 7 layers. ALL RESOLVED:**
three accepted as documented backend-specific exceptions (user-approved), one migrated.

| File | Round | Status |
|------|-------|--------|
| `physics/lagrange_layer.py` | 1 | **ACCEPTED EXCEPTION (documented).** Uses `tf.GradientTape` (and `tf.linalg.pinv`) for physics gradient computation; `tf.GradientTape` has **no `keras.ops` equivalent** (backend-agnostic AD would need e.g. JAX `jax.grad`). Exception documented in the file header; do NOT force a rewrite. The gradient-free `ApproximatedLNNLayer` is the portable alternative. |
| `complex_layers.py` | 27 | **ACCEPTED EXCEPTION (documented, user-approved).** `tf.complex`/`complex64` dtype + `tf.math.real`/`imag` across the layers; `keras.ops` has `real`/`imag` but **no complex-tensor constructor and no complex dtype**. Full fix needs a split-real/imag re-architecture changing the layers' complex-tensor contract. Exception documented in the file header; do NOT force a rewrite. |
| `canny.py` | 27 | **ACCEPTED EXCEPTION (documented, user-approved).** `tf.nn.dilation2d` (directional NMS + hysteresis dilation) + `tf.while_loop` (hysteresis fixed-point); `keras.ops` has **no morphological-dilation primitive**. Exception documented in the file header; do NOT force a rewrite. |
| `router.py` | 5 | **RESOLVED (migrated).** Only raw-tf was `tf.minimum`→`ops.minimum`; clean keras.ops migration, `tf` import dropped. |

**`rubric-verify` raw-`tf`-adjacent (confirm off-forward-path during the round):**
`sampling.py` (R30), `clahe.py` (R3), `norms/polar_weight_norm.py` (R23),
`graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py` (R2), `ffn/kan_linear.py` (R20).

### Dead-code keep-vs-delete candidates

SYSTEM.md notes the following `statistics/` classes have **0 consumers** in the codebase. They fall in
**Round 8** and require a keep-vs-delete decision before (or instead of) being brought to standard —
do not invest in full production-hardening + tests until the user confirms keep:

- `statistics/invertible_kernel_pca.py` — `InvertibleKernelPCA` (0 consumers)
- `statistics/deep_kernel_pca.py` — `DeepKernelPCA` (0 consumers)
- `statistics/residual_acf.py` — `ResidualACFLayer` (0 consumers)

If the decision is DELETE, remove the file + any `__init__.py` export + any orphan test, and decrement
the §4 tally denominator accordingly (note the change at the top of §4). If KEEP, harden + test like
any other layer in Round 8.

---

## §6 Handover Prompt

This is the self-contained instruction block to drive ONE round in a fresh, context-cleared session.
Copy the fenced block below **verbatim** into a new agent session that has zero memory of how this
roadmap was built. It depends on nothing but the files in this repository.

```text
You are picking up an ongoing, multi-session effort to make every custom Keras layer in
`src/dl_techniques/layers/` production-quality. You have NO memory of prior sessions; everything you
need is in the repository. Work exactly ONE round, then stop.

STEP 0 — Read the roadmap.
Read `roadmap/production_map.md` IN FULL. It is the single source of truth. Specifically internalize:
  - §2 The Production-Quality Rubric — the [HARD] items (H1-H14) you must enforce, the [SOFT] items you
    apply when reasonable, and the GHOST item (`if self.built: return`) you must NOT grade on or add.
  - §3 How a Round Works — the per-round procedure and the standing user decisions (FULL re-audit; audit
    AND fix in the same round; one round per session; the user pushes, you never push).
  - §4 Batched Worklist — the ordered rounds, each a small batch of files with a checkbox, a baseline
    scanner verdict, and a gap-hint.
  - §5 Known DEEP-gap + Dead-code — the files that need extra care (forward-path raw `tf.*`, dead-code
    keep-vs-delete candidates, and the `physics/lagrange_layer.py` accepted-exception candidate).

STEP 1 — Pick the next round.
In §4, find the LOWEST-numbered round whose file rows are still `[ ]` PENDING (an incomplete round).
That batch is your work for this session. Announce out loud which round number you are picking up and
list its files before doing anything else.

STEP 2 — Mechanical audit with the scanner.
Run the read-only AST scanner over the round's files to get the CURRENT mechanical verdicts (the §4
verdicts are a stale baseline — files may have changed):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_layers.py --path <file1> <file2> ...
or, for a batch that is a whole subpackage:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_layers.py --subpackage <name>
(Add `--json /tmp/round.json` for a machine-readable report.) The scanner is an AID, not an oracle:
it covers the mechanical HARD items only and can mislabel aliased imports / metaclass ABCs.

STEP 3 — Grade + fix each file against the FULL rubric.
For EVERY file in the batch: open it and grade it against the FULL §2 rubric by reading it (not just the
scanner output). The canonical, gold-standard exemplar to match is
`src/dl_techniques/layers/convnext_v1_block.py`; the authoritative spec is
`research/2026_keras_custom_models_instructions.md` (Pattern 2 begins at line 381).
  - Fix ALL [HARD] gaps (H1-H14) and all reasonable [SOFT] gaps to match the canonical style.
  - Do NOT add `if self.built: return` to `build()` — it is GHOST (§2). Its absence is not a defect and
    must never be treated as round work.
  - If a file has a DEEP forward-path raw-`tf.` gap (§5) that cannot be cleanly migrated to `keras.ops`,
    do NOT force a broken rewrite. Make at most 2 fix attempts; if it still won't come clean, mark that
    file `[~]` in §4 with a one-line note (e.g. "raw-tf in call(); keras.ops has no equivalent — see §5")
    and surface it in your final report rather than committing a broken layer. `physics/lagrange_layer.py`
    in particular is an ACCEPTED-EXCEPTION candidate (`tf.GradientTape` has no `keras.ops` equivalent):
    document the exception in the file header and accept it; do not rewrite it.

STEP 4 — Ensure/repair a test.
Each concrete layer needs a real test under `tests/test_layers/...` mirroring the source path. If missing
or thin, add or repair one that covers, at minimum:
  - construction (including the `ValueError` input-validation paths),
  - a forward pass,
  - a full serialization round-trip: build a tiny model using the layer, save it, reload it with
    `keras.models.load_model(path, custom_objects={...})`, and assert identical output before/after,
  - `compute_output_shape(...)` agreement with the actual `call()` output shape.

STEP 5 — Run scoped tests (GPU1 only, never parallel).
Run pytest scoped to JUST this batch's tests. Use GPU1 and serial execution (repo convention — never run
GPU jobs in parallel):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_layers/<batch tests> -x
All selected tests must pass before you check anything off.

STEP 6 — Re-run the scanner on the batch.
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_layers.py --path <batch files>
Confirm every file now reports PASS — OR a documented, accepted exception per §5 (e.g.
`physics/lagrange_layer.py`'s `tf.GradientTape`). An unexplained FAIL means the round is not done.

STEP 7 — Update §4 in place.
For each completed file, flip its row's `[ ]` to `[x]` (use `[~]` for a deferred DEEP-gap file with its
one-line note). Then update the tally line at the top of §4 — `**Progress: X / 245 files
production-verified**` — incrementing X by the number of files you finished this session. Edit only the
rows for THIS round; do not touch other rounds.

STEP 8 — Commit locally (do NOT push).
Stage ONLY the specific files you edited (the layer files, their tests, and `roadmap/production_map.md`).
Do NOT use `git add -A`. Then commit with a message of the form:
    git commit -m "[production-map/round-N] <short batch description>"
Do NOT push — the user pushes themselves.

STEP 9 — Report and STOP.
Report: which round you completed, a per-file before/after verdict, which tests you ran and their result,
the new "X / 245" tally, and which round is next (the next PENDING round in §4). Then STOP. ONE round per
session is the intended cadence — do not chain into the next round. Let the user clear context.
```

---

## §7 Provenance

This roadmap (the scanner `scripts/audit_layers.py`, the rubric, the 245-file worklist, and the §6
handover prompt) was authored as durable, standalone infrastructure. It is intentionally self-contained:
every future round is driven solely by this document plus the scanner — no external session state or
planning scaffold is required, and none is referenced here.

---
---

# PART II — LEVEL 2: MODELS

> **This is the ACTIVE worklist.** Level 1 (PART I, §1–§7 above) is COMPLETE and frozen. All current
> work is Level 2: bringing every `keras.Model` subclass under `src/dl_techniques/models/` to
> production quality. The single source of truth for "production-quality model" is the canonical
> exemplar `src/dl_techniques/models/bert/bert.py` plus the authoritative spec
> `research/2026_keras_custom_models_instructions.md` (model/composite-model sections: §5 line 863, §6
> line 1000, §7 line 1175, §8 line 1378, §9 line 1527, §11 line 1954, gold example §15.2 line 2778).
> Level-2 sections are numbered **§L2-1 … §L2-7** to avoid collision with PART I's §1–§7. This is a
> LIVING document: each round edits §L2-4 in place (`[ ]` → `[x]`) and bumps the §L2-4 tally.

## §L2-1 Mission & Scope

**Mission.** Make all **183 model source files** across **70 model directories** under
`src/dl_techniques/models/` production-quality, one self-contained, context-cleared session ("round")
at a time, graded against the §L2-2 rubric, verified by two instruments (§L2-3), with a real test
under `tests/test_models/` for each model.

**Baseline snapshot (commit `7b1e28af`, live tree walk + `audit_layers.py --models`).** 183 source
files (excluding `__init__.py` and `__pycache__`) across 70 directories (`jepa/` is an empty package —
only `__init__.py` — and is EXCLUDED from the denominator): **127 PASS / 16 FAIL / 40 N/A** (0
PARSE-ERROR). 99 concrete `keras.layers.Layer` subclasses + 120 concrete `keras.Model` subclasses
graded (219 total). Dominant mechanical gap: `super().build()` not last (`super_build_last` 131/143 =
91.6% — 12 of the 16 FAILs); 2 layers carry raw `tf.*` in the forward path; `compute_output_shape`
96/99 over LAYER subclasses (3 FAILs, all embedded layers). register-decorator, `get_config`, and
no-`print` are 100%.

**Two-instrument verification.** Unlike Level 1 (one AST scanner), Level 2 uses BOTH:
1. **Mechanical AST** — `scripts/audit_layers.py --models` (the `--models` flag grades `keras.Model`
   subclasses; without it the scanner is Level-1 layers-only).
2. **Runtime smoke** — `scripts/verify_models_smoke.py` (87-entry registry harness: build + forward +
   `.keras` reload). Baseline **85 PASS / 0 FAIL / 2 SKIP** (`jepa`, `ccnets` — see §L2-5).

> **LIVING document.** Each round edits §L2-4 in place — flipping `[ ]` → `[x]`, updating the per-round
> status, and bumping the §L2-4 progress tally. §L2-1…§L2-3 and §L2-5 are stable reference. The
> handover prompt to drive a fresh round is §L2-6.

---

## §L2-2 The Model Production-Quality Rubric

Distilled from `src/dl_techniques/models/bert/bert.py` and the spec's model sections. The PART I
layer rubric (H1–H14, §2) is the foundation; this section states how each item maps to a
`keras.Model` subclass and adds the model-only items (M-items). **[HARD]** = a miss is a
non-compliant file; **[SOFT]** = canonical style, apply when reasonable; **[GHOST]** = NOT graded.

### H1–H14 carry-over to models

| # | Item | Status for `keras.Model` | Notes |
|---|------|--------------------------|-------|
| H1 | `@keras.saving.register_keras_serializable()` | **[HARD]** | Required on every concrete model class. `bert.py:78`. |
| H2 | Sublayers created in `__init__` | **[HARD]** → sharpened to **M4** | See M4 (always-create across config flags). `add_weight` never in `__init__`. |
| H3 | All `__init__` args stored as `self.*` | **[HARD]** | Survives `get_config`. `bert.py:346-366`. |
| H4 | Input validation (`ValueError`) | **[HARD]** | bert uses a `_validate_config()` helper. `bert.py:376-425`. |
| H5 | Explicit sublayer `build()` | **[CONDITIONAL]** | Only when the model OVERRIDES `build()` (custom shape-dependent sublayers). Composite models that create sublayers in `__init__` and let Keras auto-build on first call do NOT need a `build()` override (bert has none). When a `build()` override IS present, it must build sublayers in order. |
| H6 | `super().build()` LAST | **[HARD when `build()` exists]** | If the model/embedded-layer defines `build()`, `super().build()` must be the structural last statement. 12 of 16 FAILs are this. |
| H7 | `compute_output_shape` | **[SOFT for models] / [HARD for embedded layers]** | Keras infers model output shape from `call()`; the spec's model examples omit it. But any `keras.layers.Layer` subclass living inside a model dir is graded by the LAYER rule (HARD). 3 FAILs are embedded layers. |
| H8 | Full `get_config` | **[HARD]** | `super().get_config()` + every `__init__` arg; serialize regularizers/initializers/activations when present. `bert.py:748-778`. |
| H9 | `from_config` deserialization | **[HARD]** | Deserialize objects when config carries them; `return cls(**config)` suffices when config is plain scalars (bert). `bert.py:780-789`. |
| H10 | Graph-safe `call()` | **[HARD]** | Only `keras.ops` in the forward path (accepted exceptions in §L2-5). |
| H11 | `training` forwarded | **[HARD]** | `call(..., training=None)` propagated to every sublayer that uses it. `bert.py:465-524`. |
| H12 | Type hints | **[HARD]** | Every method signature annotated. |
| H13 | Structured docstrings | **[HARD]** | bert uses RST `:param:`/`:type:`/`:raises:`/`:ivar:` + architecture diagram. |
| H14 | Logger, not `print` | **[HARD]** | `from dl_techniques.utils.logger import logger`. |

### Model addendum (M-items)

| # | Item | Status | What it means | Source |
|---|------|--------|---------------|--------|
| M1 | Variant configs | **[SOFT]** | `MODEL_VARIANTS` class dict + `from_variant(variant, pretrained=, **kwargs)` classmethod — ONLY where discrete size tiers (tiny/small/base/large) are semantically real. Not every model has tiers. | `bert.py:207-236, 621-746`; spec §7.2/§11.1 |
| M2 | `.keras` round-trip | **[HARD]** | Full `model.save(path)` → `keras.models.load_model(path, custom_objects=...)` → **numerically identical outputs** (`atol=1e-6`; GPU fp32 reduction noise → use `1e-4`, SYSTEM invariant). This is STRONGER than a `from_config` round-trip and is the load-bearing model test. | spec §8.2:1462-1523 |
| M3 | Pretrained weight handling | **[SOFT]** | If a model loads real checkpoints, use `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint(target, ckpt_path, ...)`, NOT `load_weights(by_name=True)` on a `.keras` file (broken in Keras 3.8+). Build-before-load. See weight-handling policy below. | `weight_transfer.py`; spec Pitfall 9 |
| M4 | Always-create sublayers | **[HARD]** | All sublayers instantiated unconditionally in `__init__` regardless of config flags (`include_top`, `enable_aux`, …); flags only gate usage in `call()`. Keeps weight names stable across configurations so `.keras`/checkpoint load works. | spec §9.1 (Pitfall 1) |
| M5 | Module-level factory | **[SOFT]** | `create_<model>(...)` thin wrapper (and `create_<model>_with_head(...)` where heads compose). | `bert.py:829-1011` |
| M6 | Stable explicit layer names | **[SOFT]** | `name=f'...'` on sublayers, especially in list-of-layers loops (weight-name stability). | `bert.py:438,461` |

### Weight-handling policy (decision D-003 of the authoring plan)

- `model.load_weights(path.keras, by_name=True)` **raises** in Keras 3.8 for `.keras` files — it is a
  **latent anti-pattern**. The canonical replacement is
  `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint(...)` (layer-by-layer name match,
  returns a `TransferReport`). Three model files carry this latent bug live and SHOULD migrate (SOFT,
  in their owning round): `cliffordnet/model.py` (~line 413), `bias_free_denoisers/bfunet.py`
  (~line 515), `convnext/convnext_v2.py` (~line 400). These never load a real checkpoint today, so the
  bug is dormant — migration is SOFT, not a round-blocker.
- The exemplar **bert itself** uses `load_weights(by_name=True)` in `load_pretrained_weights`
  (`bert.py:574-578`); it is never exercised (`pretrained=True` → `NotImplementedError`). This
  exemplar quirk is **[GHOST]** — do NOT propagate it to other models, and do NOT rip it out of bert
  as round work.

### GHOST — do NOT grade or add as round work

- **`train_step` override** — the spec has no guidance; not a rubric item.
- **`if self.built: return` at the top of `build()`** — carried from PART I §2 (GHOST). Absence is not
  a defect; presence is harmless but never required.
- **bert's own `load_weights(by_name=True)`** — see weight-handling policy above.

### Scanner-mechanical vs human-judged (models)

`scripts/audit_layers.py --models` mechanically checks: H1 (register), H8 (`get_config` present), H6
(`super_build_last`, when `build()` exists), H10 (`forward_raw_tf` in `call()`), H14 (`print`).
`compute_output_shape` is INFO-only for models (HARD for embedded layers). Everything else —
H3/H4/H9/H11 correctness, H12/H13 quality, M1-M6, and the runtime M2 `.keras` round-trip — is
human-judged + smoke-verified during the round. The scanner is an **aid, not an oracle**.

---

## §L2-3 How a Level-2 Round Works

A "round" is one self-contained, context-cleared session that clears a single batch from §L2-4.
Standing user decisions (same as PART I):

- **FULL re-audit.** Grade every file fresh against §L2-2 — do not trust the stale §L2-4 baseline verdict.
- **Audit AND fix in the SAME round.** Grade and repair the batch; don't defer.
- **One round per cleared-context session.** Pick the next PENDING round, do it end-to-end, commit, stop.
- **Push is the user's job.** Commit locally only; never push.

Per-round procedure:

1. **Read this file** (PART II). Find the lowest-numbered §L2-4 round with `[ ]` rows. That's the batch.
2. **Mechanical audit** over the batch:
   ```
   CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_layers.py --models --path <file1> <file2> ...
   ```
   (`--path` accepts a directory too; add `--json /tmp/round.json` for machine-readable output.)
3. **Grade + fix each file** against the FULL §L2-2 rubric by reading it. Match the canonical
   `models/bert/bert.py` and the spec. Fix all [HARD] gaps and reasonable [SOFT] gaps. Do NOT add
   `if self.built: return` (GHOST). For an H7 FAIL, check whether the class is a `keras.Model` (SOFT —
   skip) or an embedded `keras.layers.Layer` (HARD — add `compute_output_shape`).
4. **Tests.** Ensure a real test under `tests/test_models/test_<name>/` mirroring the model. If missing
   or thin, add/repair one covering: construction (incl. `ValueError` paths), forward pass, **M2 full
   `.keras` save→load→identical-output round-trip**, and (where applicable) variants/factory.
5. **Runtime smoke** for each model in the batch:
   ```
   CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python scripts/verify_models_smoke.py --only <name>
   ```
6. **Scoped pytest** (GPU1, serial — repo convention; never run GPU jobs in parallel):
   ```
   CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_<name>/ -x
   ```
7. **Two-instrument pass.** Re-run the scanner AND the smoke harness; every file must report PASS (or
   a documented accepted exception — §L2-5; or a documented smoke SKIP — `jepa`, `ccnets`).
8. **Update §L2-4 in place.** Flip the batch's `[ ]` → `[x]`, set the round status to DONE, bump the
   §L2-4 tally ("X / 183").
9. **Commit** (locally; do not push):
   ```
   git commit -m "[production-map/L2-round-N] <short batch description>"
   ```

---

## §L2-4 Batched Worklist

**Progress: 183 / 183 files production-verified**  (COMPLETE — all 25 rounds DONE and the post-round-25 `nano_vlm_world_model/model.py` deep M2 gap is now RESOLVED. fftnet counted: documented accepted raw-tf exception, M2 + tests complete.)

> **NOTE (Round 1):** `scripts/verify_models_smoke.py` was removed at HEAD (commit `79bebe5d`,
> authored after the PART II roadmap). STEP 5/7 runtime smoke is therefore unavailable; the pytest
> suites (build + forward + `.keras` reload) provide equivalent runtime coverage and were used instead.

Status legend: `[ ]` PENDING · `[~]` IN-PROGRESS/DEFERRED · `[x]` DONE.
`verdict` is the `scripts/audit_layers.py --models` mechanical result at baseline `7b1e28af`
(re-run the scanner during the round — files change). `gap-hint` is the failing item for FAIL files,
`N/A — <kind>` for non-graded modules (dataclass-config / pure-functions / non-layer — still
human-confirmed), or `rubric-verify` for current-PASS files needing a human rubric/M2/doc review.
Rounds are directory-cohesive (whole directories per round; model files interdepend). 25 rounds.

<!-- L2-WORKLIST -->
### L2-Round 1 — accunet, bert, bias_free_denoisers  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `accunet/model.py` | PASS | rubric-verified; 70 accunet+bert tests pass |
| `[x]` | `bert/bert.py` | PASS | CANONICAL EXEMPLAR re-confirmed (no changes) |
| `[x]` | `bias_free_denoisers/bfcnn.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `bias_free_denoisers/bfconvunext.py` | PASS | H6 fixed (super().build last); fixed latent bug (kernel_initializer passed to ConvNextBlock); H5 explicit sublayer build for .keras round-trip; new test added |
| `[x]` | `bias_free_denoisers/bfunet.py` | N/A | N/A — pure-functions; weight-handling SOFT migrated to load_weights_from_checkpoint |
| `[x]` | `bias_free_denoisers/bfunet_conditional.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `bias_free_denoisers/bfunet_conditional_unified.py` | PASS | H5 explicit sublayer build in both injection layers (fixes .keras round-trip); M2 round-trip test added |

### L2-Round 2 — byte_latent_transformer, capsnet, cbam  (4 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `byte_latent_transformer/model.py` | PASS | H6 fixed (logger before super().build); H4 added (_validate_config); M2 fixed: get_build_config/build_from_config + explicit MHA build in blt_blocks PatchPooling/LocalDecoder (was dropping 16 attn weights on reload); new test_model.py |
| `[x]` | `capsnet/model.py` | PASS | rubric-verified; round-trip test passes |
| `[x]` | `capsnet/model_v2.py` | PASS | rubric-verified; round-trip test passes |
| `[x]` | `cbam/model.py` | PASS | rubric-verified; round-trip test passes |

### L2-Round 3 — ccnets  (9 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `ccnets/architectures/cifar100.py` | PASS | rubric-verified; added reasoner+producer M2 round-trip tests |
| `[x]` | `ccnets/architectures/mnist.py` | PASS | rubric-verified; added producer M2 round-trip test |
| `[x]` | `ccnets/architectures/text.py` | PASS | rubric-verified; added reasoner+producer+AR-producer M2 round-trip tests |
| `[x]` | `ccnets/base.py` | N/A | N/A — non-layer (confirmed; exercised by orchestrator tests) |
| `[x]` | `ccnets/control.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `ccnets/losses.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `ccnets/orchestrators.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `ccnets/trainer.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `ccnets/utils.py` | N/A | N/A — non-layer (ccnets = smoke SKIP, §L2-5) |

### L2-Round 4 — cliffordnet  (10 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `cliffordnet/clip.py` | PASS | rubric-verified |
| `[x]` | `cliffordnet/conditional_denoiser.py` | PASS | rubric-verified; added M2 round-trip test (was untested) |
| `[x]` | `cliffordnet/confidence_denoiser.py` | PASS | rubric-verified; added M2 round-trip test (was untested) |
| `[x]` | `cliffordnet/denoiser.py` | PASS | H6 fixed (dummy-forward before super().build()); added M2 round-trip test (was untested) |
| `[x]` | `cliffordnet/embedding_unet.py` | PASS | rubric-verified |
| `[x]` | `cliffordnet/lm.py` | PASS | rubric-verified |
| `[x]` | `cliffordnet/lm_routing.py` | PASS | rubric-verified |
| `[x]` | `cliffordnet/lmunet.py` | PASS | rubric-verified |
| `[x]` | `cliffordnet/model.py` | PASS | H6 fixed — dummy-forward CLEAN reorder before super().build() (NON-TRIVIAL resolved, no interaction); weight-handling migrated to load_weights_from_checkpoint |
| `[x]` | `cliffordnet/unet.py` | PASS | H7: added _DetectionHeadBlock.compute_output_shape (delegates to YOLOv12DetectionHead) |

### L2-Round 5 — clip, convnext, convnext_patch_vae  (8 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `clip/model.py` | PASS | rubric-verified; round-trip test passes |
| `[x]` | `convnext/convnext_v1.py` | PASS | H6 fixed (dummy-forward before super().build()); save/load test passes |
| `[x]` | `convnext/convnext_v2.py` | PASS | H6 fixed (same reorder); weight-handling migrated to load_weights_from_checkpoint |
| `[x]` | `convnext_patch_vae/config.py` | N/A | N/A — dataclass-config (confirmed) |
| `[x]` | `convnext_patch_vae/decoder.py` | PASS | rubric-verified |
| `[x]` | `convnext_patch_vae/encoder.py` | PASS | rubric-verified |
| `[x]` | `convnext_patch_vae/model.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `convnext_patch_vae/model_hierarchical.py` | PASS | rubric-verified; round-trip tests pass |

### L2-Round 6 — convunext, coshnet, darkir, depth_anything, detr  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `convunext/model.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `coshnet/model.py` | PASS | rubric-verified; replaced xfail-smoke with real test (construction+ValueError+forward+M2 round-trip) |
| `[x]` | `darkir/model.py` | PASS | rubric-verified; added test_model.py (forward + side-loss variant + M2 round-trip) |
| `[x]` | `depth_anything/components.py` | PASS | rubric-verified |
| `[x]` | `depth_anything/model.py` | PASS | rubric-verified (uses weight_transfer — canonical M3) |
| `[x]` | `depth_anything/teacher_ema.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `detr/model.py` | PASS | rubric-verified; round-trip tests pass |

### L2-Round 7 — dino, distilbert, fastvlm, fftnet  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `dino/dino_v1.py` | PASS | H6: DINOHead super().build() last + explicit sublayer build (fixes DINOHead round-trip); ALSO fixed DINOv1 model round-trip (patch_size/image_size deserialized as list broke `//`); added test_model_v1.py |
| `[x]` | `dino/dino_v2.py` | PASS | rubric-verified (2-input [images, masks] contract) |
| `[x]` | `dino/dino_v3.py` | PASS | rubric-verified |
| `[x]` | `distilbert/model.py` | PASS | rubric-verified; added test_model.py with M2 round-trip (was smoke-only) |
| `[x]` | `fastvlm/components.py` | PASS | rubric-verified |
| `[x]` | `fastvlm/model.py` | PASS | rubric-verified (image-only despite VLM name) |
| `[~]` | `fftnet/model.py` | FAIL→ACCEPTED | FFTMixer raw-tf FFT is a DOCUMENTED accepted exception (header note added; keras.ops has no ifft, §L2-5). Also fixed FFTNet M2 round-trip (explicit sublayer build) + added test_model.py. Scanner FAIL is the accepted H10 exception. |

### L2-Round 8 — fnet, fractalnet, gemma, gpt2, hierarchical_reasoning_model  (6 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `fnet/model.py` | PASS | rubric-verified; round-trip test passes |
| `[x]` | `fractalnet/model.py` | PASS | rubric-verified; round-trip test passes |
| `[x]` | `gemma/components.py` | PASS | rubric-verified |
| `[x]` | `gemma/gemma3.py` | PASS | rubric-verified; round-trip test passes |
| `[x]` | `gpt2/gpt2.py` | PASS | rubric-verified; added test_round_trip.py (M2 was untested; round-trip clean) |
| `[x]` | `hierarchical_reasoning_model/model.py` | PASS | rubric-verified (build() override SAM D-008); added test_model.py with M2 round-trip (was smoke-only; round-trip clean) |

### L2-Round 9 — ideogram4  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `ideogram4/config.py` | N/A | N/A — dataclass-config (confirmed: no Layer/Model subclass) |
| `[x]` | `ideogram4/constants.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `ideogram4/latent_norm.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `ideogram4/pipeline.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `ideogram4/scheduler.py` | N/A | N/A — dataclass-config (confirmed) |
| `[x]` | `ideogram4/transformer.py` | PASS | rubric-verified; existing test has real M2 deterministic-velocity round-trip |
| `[x]` | `ideogram4/vae.py` | PASS | rubric-verified; existing test has real M2 deterministic-mu round-trip |

### L2-Round 10 — kan, latent_gmm_registration, lewm  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `kan/model.py` | PASS | rubric-verified; weight-handling migrated load_pretrained_weights → load_weights_from_checkpoint (M3) |
| `[x]` | `latent_gmm_registration/model.py` | PASS | rubric-verified; documented tf.linalg.svd accepted-exception in header (§L2-5); added M2 round-trip test |
| `[x]` | `lewm/config.py` | N/A | N/A — dataclass-config (confirmed) |
| `[x]` | `lewm/embedder.py` | PASS | rubric-verified (ActionEmbedder, H7 present) |
| `[x]` | `lewm/model.py` | PASS | rubric-verified; M2 round-trip covered by test_lewm.py |
| `[x]` | `lewm/predictor.py` | PASS | rubric-verified (ARPredictor, H7 present) |
| `[x]` | `lewm/projector.py` | PASS | rubric-verified (MLPProjector, H7 present); fixed pre-existing stale test_rollout_shape (S==1 contract) |

### L2-Round 11 — mamba, masked_autoencoder  (8 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `mamba/components.py` | PASS | rubric-verified |
| `[x]` | `mamba/components_v2.py` | PASS | rubric-verified |
| `[x]` | `mamba/mamba_v1.py` | PASS | rubric-verified; M3 OK — uses BARE load_weights(path) (not by_name=True), the valid whole-model restore; no migration needed |
| `[x]` | `mamba/mamba_v2.py` | PASS | rubric-verified |
| `[x]` | `masked_autoencoder/conv_decoder.py` | PASS | rubric-verified (embedded Layer, H7 present); covered via MAE round-trip |
| `[x]` | `masked_autoencoder/mae.py` | PASS | rubric-verified; FIXED M2 round-trip (input_shape deserialized as list broke `(None,)+input_shape`); added test_model.py |
| `[x]` | `masked_autoencoder/patch_masking.py` | PASS | rubric-verified (embedded Layer, H7 present) |
| `[x]` | `masked_autoencoder/utils.py` | N/A | N/A — pure-functions (confirmed) |

### L2-Round 12 — masked_language_model, memory_bank  (9 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `masked_language_model/clm.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `masked_language_model/mlm.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `masked_language_model/utils.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `memory_bank/memory_banks.py` | PASS | rubric-verified (embedded Layer; exercised via wave_field round-trip) |
| `[x]` | `memory_bank/memory_stats.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `memory_bank/phase_scheduler.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `memory_bank/read_controller.py` | PASS | rubric-verified (embedded Layer) |
| `[x]` | `memory_bank/wave_field_memory_llm.py` | PASS | rubric-verified; 8 .keras round-trip tests pass |
| `[x]` | `memory_bank/write_controller.py` | PASS | H10 FIXED — removed the tf.debugging raw-tf debug-guard; replaced with keras.ops `ops.maximum(max_seq_len-t, 0)` clamp (graph-safe, no exception needed) |

### L2-Round 13 — mini_vec2vec, mobile_clip, mobilenet  (9 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `mini_vec2vec/example_alignment.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `mini_vec2vec/model.py` | PASS | rubric-verified; added test_model.py with M2 round-trip (was smoke-only; round-trip clean) |
| `[x]` | `mobile_clip/components.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `mobile_clip/mobile_clip_v1.py` | PASS | rubric-verified (CNN substitutes via _BACKBONE_ALIASES); round-trip tests pass |
| `[x]` | `mobile_clip/mobile_clip_v2.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `mobilenet/mobilenet_v1.py` | PASS | rubric-verified; 6 round-trip tests pass |
| `[x]` | `mobilenet/mobilenet_v2.py` | PASS | rubric-verified; 6 round-trip tests pass |
| `[x]` | `mobilenet/mobilenet_v3.py` | PASS | rubric-verified; 6 round-trip tests pass |
| `[x]` | `mobilenet/mobilenet_v4.py` | PASS | rubric-verified; 6 round-trip tests pass |

### L2-Round 14 — modern_bert, mothnet, nam  (8 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `modern_bert/components.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `modern_bert/modern_bert.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `modern_bert/modern_bert_blt.py` | PASS | rubric-verified; added test_modern_bert_blt.py with M2 round-trip (small config; round-trip clean 0.0) |
| `[x]` | `mothnet/model.py` | PASS | rubric-verified; added test_model.py (forward + ValueError + M2 round-trip; was smoke-only) |
| `[x]` | `nam/cell.py` | PASS | rubric-verified (embedded; config round-trip + bit-exact weight transfer tested) |
| `[x]` | `nam/config.py` | N/A | N/A — dataclass-config (confirmed) |
| `[x]` | `nam/model.py` | PASS | rubric-verified; NTM-style stateful (call(carry,batch)) — M2 covered by bit-exact weight round-trip + config round-trip (full .keras not well-defined for carry-based forward) |
| `[x]` | `nam/tokenizer.py` | N/A | N/A — non-layer (confirmed) |

### L2-Round 15 — nano_vlm, nano_vlm_world_model, ntm  (7 files)  — DONE (7/7; nano_vlm_world_model/model.py M2 resolved in the post-round-25 deep-gap effort)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `nano_vlm/model.py` | PASS | H6 fixed (logger before super().build()); added test_model.py (factory + forward + M2 round-trip; clean 0.0) |
| `[x]` | `nano_vlm_world_model/denoisers.py` | PASS | rubric-verified (embedded denoiser Layers: TimestepEmbedding/Conditional/Vision/Text/Joint) |
| `[x]` | `nano_vlm_world_model/model.py` | PASS | M2 RESOLVED (post-round-25 dedicated effort): root cause was lazy first-call build of the denoisers' nested MultiHeadAttention + Sequential sub-layers (dropped ~600 weights on reload). Fix: added explicit `build()` to ConditionalDenoiser/VisionDenoiser/TextDenoiser/JointDenoiser (MHA built via `build(query_shape=,value_shape=)`), and the model's `build()` now builds all denoisers + head + adds get_build_config/build_from_config. Verified by test_round_trip.py: weight-count preserved + every component (encoders + all denoisers, all 3 modes) bit-identical after reload (stochastic top-level forward → component-level determinism is the M2 proof). |
| `[x]` | `nano_vlm_world_model/scheduler.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `nano_vlm_world_model/train.py` | N/A | N/A — non-layer (confirmed) |
| `[x]` | `ntm/model.py` | PASS | rubric-verified; existing test_model.py has 4 .keras round-trips |
| `[x]` | `ntm/model_multitask.py` | PASS | rubric-verified; added test_model_multitask.py (forward + ValueError + M2 round-trip; clean 0.0) |

### L2-Round 16 — pft_sr, power_mlp, power_sampling  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `pft_sr/model.py` | PASS | FIXED M2 round-trip (sublayers created in build() built lazily → 120 unbuilt on reload; added concrete dummy-forward in build()). Added test_model.py. |
| `[x]` | `power_mlp/model.py` | PASS | rubric-verified; added test dir + test_model.py (forward + from_variant + ValueError + M2 round-trip; clean 0.0) |
| `[x]` | `power_sampling/config.py` | N/A | N/A — dataclass-config (confirmed) |
| `[x]` | `power_sampling/forward.py` | N/A | N/A — non-layer (pure-Python inference engine, no keras.Model) |
| `[x]` | `power_sampling/ops.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `power_sampling/protocols.py` | N/A | N/A — non-layer (typing.Protocol) |
| `[x]` | `power_sampling/sampler.py` | N/A | N/A — non-layer (confirmed) |

### L2-Round 17 — pw_fnet  (1 file)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `pw_fnet/model.py` | PASS | H6 fixed in both embedded Downsample + Upsample layers (explicit inner-conv build before super().build()); existing test_model.py (10 round-trips, 46 tests) passes |

### L2-Round 18 — qwen  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `qwen/components.py` | PASS | rubric-verified; 53 tests incl. round-trips |
| `[x]` | `qwen/qwen3.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `qwen/qwen3_embeddings.py` | PASS | was DEAD-ON-FORWARD (both Qwen3Embedding/Reranker models): last-token pooling fed a 2-D index to ops.take_along_axis on a 3-D tensor → fixed to broadcast index to (B,1,D)/(B,1,V). Added test_qwen3_embeddings.py (forward + M2). Was untested. |
| `[x]` | `qwen/qwen3_mega.py` | PASS | rubric-verified; added test_qwen3_mega.py (forward + M2 round-trip; clean 0.0). Was untested. |
| `[x]` | `qwen/qwen3_next.py` | PASS | rubric-verified; round-trip tests pass |
| `[x]` | `qwen/qwen3_omni.py` | N/A | N/A — pure-functions (confirmed) |
| `[x]` | `qwen/qwen3_som.py` | PASS | rubric-verified; added test_qwen3_som.py (gen + cls forward + M2; clean 0.0). Was untested. |

### L2-Round 19 — relgt, resnet, sam  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `relgt/model.py` | PASS | rubric-verified; added test_model.py. Forward is STOCHASTIC at inference (local-neighborhood sampling) → M2 verified by weight-preservation (all 49 weights identical across round-trip), not output-identity |
| `[x]` | `resnet/model.py` | PASS | rubric-verified; added test_round_trip.py (M2 output-identity; clean 0.0) |
| `[x]` | `sam/image_encoder.py` | PASS | rubric-verified (PatchEmbedding2D flatten=False) |
| `[x]` | `sam/mask_decoder.py` | PASS | rubric-verified; sam suite 35 tests pass incl. round-trip (the §L2-5 mask-drift no longer fails) |
| `[x]` | `sam/model.py` | PASS | rubric-verified |
| `[x]` | `sam/prompt_encoder.py` | PASS | H7 FIXED — added PositionEmbeddingRandom.compute_output_shape (call(size)→(2*num_pos_feats, h, w)) |
| `[x]` | `sam/transformer.py` | PASS | rubric-verified |

### L2-Round 20 — scunet  (1 file)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `scunet/model.py` | PASS | done: H4 added (`_validate_config` — in_nc/config len-7+positive/dim even-positive/head_dim/window_size/sd_rate∈[0,1]/input_resolution) + S1 module docstring; added 10 ValueError tests; 63 tests pass incl. M2 round-trips |

### L2-Round 21 — sd3_mmdit  (7 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `sd3_mmdit/blocks.py` | PASS | rubric-verified exemplary (MMDiTBlock/MMDiTFinalLayer: H4/H5/H6/H7/H8/H11 all good); 2 .keras round-trips pass |
| `[x]` | `sd3_mmdit/config.py` | N/A | N/A confirmed (SD3MMDiTConfig frozen dataclass) |
| `[x]` | `sd3_mmdit/pipeline.py` | N/A | N/A confirmed (SD3Pipeline plain-Python inference object) |
| `[x]` | `sd3_mmdit/scheduler.py` | N/A | N/A confirmed (FlowMatchEulerScheduler frozen dataclass) |
| `[x]` | `sd3_mmdit/text_encoders.py` | PASS | done: H4 added to T5Encoder (embed_dim%num_heads — for parity with CLIPTextEncoder; head_dim split needs it); +2 ValueError tests (CLIP+T5); 4 .keras round-trips pass |
| `[x]` | `sd3_mmdit/transformer.py` | PASS | rubric-verified exemplary (SD3MMDiT: H4 TypeError, build-override super().build() last, H7/H9, M1/M5 create_sd3_mmdit factory); 2 .keras round-trips pass |
| `[x]` | `sd3_mmdit/vae.py` | N/A | N/A confirmed (SD3VAE plain-Python wrapper, not a keras.Model) |

### L2-Round 22 — shgcn, som, squeezenet, swin_transformer, tabm, thera  (11 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `shgcn/model.py` | PASS | rubric-verified (3 models; H4 propagates via SHGCNModel backbone; output_activation→string round-trip); added test_round_trip.py (M2 for model/classifier/link-predictor + ValueError paths). Was smoke-only |
| `[x]` | `som/model.py` | PASS | done: H4 added neighborhood_function membership check; host-side train/viz raw-numpy is OFF the call() forward path (H10 clean); existing test_model.py round-trip passes |
| `[x]` | `squeezenet/squeezenet_v1.py` | PASS | rubric-verified (functional model; FireModule embedded layer fully compliant; get_config returns init-args by-design for functional-reconstruct, name cosmetic-drop only); existing round-trip passes |
| `[x]` | `squeezenet/squeezenet_v2.py` | PASS | rubric-verified (functional model; SimplifiedFireModule compliant; 3D path uses keras.Sequential — round-trips via functional graph); existing round-trip passes |
| `[x]` | `swin_transformer/model.py` | PASS | rubric-verified (functional model, extensive H4 validation, H9 deserializes init/reg); added test_round_trip.py (M2 output-identity at 224px + ValueError paths). Was smoke-only |
| `[x]` | `tabm/model.py` | PASS | done: H4 asserts→ValueError (asserts stripped under -O); H10 `//` is tensor floordiv (graph-safe, not a Python cast); added test_round_trip.py (M2 + 6 ValueError paths). Was smoke-only |
| `[x]` | `thera/edsr_backbone.py` | PASS | rubric-verified exemplary (EDSRResidualBlock/EDSRBackbone: H4/H5/H6/H7/H8/H9/H11); existing round-trip passes |
| `[x]` | `thera/hypernetwork.py` | PASS | done: H10 migrated 2 plain-forward-path raw-tf spots (tf.shape/tf.concat → ops.shape tuple + ops.reshape/broadcast_to); decode_with_jac tf.GradientTape.batch_jacobian documented as accepted exception in header (§L2-5; no keras.ops AD); existing round-trip passes |
| `[x]` | `thera/model.py` | PASS | rubric-verified exemplary (Thera: H4/H5/H6/H8/H9/H11, M1 from_variant + M5 build_thera factory); existing round-trip passes |
| `[x]` | `thera/rdn_backbone.py` | PASS | rubric-verified exemplary (RDBConv/RDB/RDNBackbone all compliant); existing round-trip passes |
| `[x]` | `thera/tails.py` | PASS | rubric-verified exemplary (TheraTailAir/_Projection/TheraTailPlus/TheraTailPro + build_thera_tail factory); existing round-trip passes |

### L2-Round 23 — time_series  (11 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `time_series/adaptive_ema/model.py` | PASS | rubric-verified clean (H4/H8/H9/H11 good; training-branch + static-shape checks off-graph); existing round-trip passes |
| `[x]` | `time_series/deepar/model.py` | PASS | rubric-verified (likelihood head is pure Dense → no training to forward; _prediction_mode eager range is predict_step-only, off call()); existing round-trip passes |
| `[x]` | `time_series/forecast.py` | N/A | N/A confirmed (Forecast dataclass + ForecastMixin helper) |
| `[x]` | `time_series/mdn/model.py` | PASS | done: H6 fixed (logger.info moved before super().build()); H8/H9 serialize+deserialize init/reg; existing round-trip passes |
| `[x]` | `time_series/nbeats/nbeats.py` | PASS | rubric-verified clean (NBeatsNet; H4 via _validate_configuration; H9 deserializes init/reg); existing round-trip passes |
| `[x]` | `time_series/nbeats/nbeatsx.py` | PASS | done: H12 type hints (__init__/_create_block_stacks/build/call); existing round-trip passes |
| `[x]` | `time_series/prism/model.py` | PASS | rubric-verified clean (PRISMModel; H5/H6 build last, H8/H9 init/reg, M1+M5); existing round-trip passes |
| `[x]` | `time_series/tirex/model.py` | PASS | rubric-verified clean (TiRexCore; H5/H6 build last, M1 from_variant + M5 factory); existing round-trip passes |
| `[x]` | `time_series/tirex/model_extended.py` | PASS* | concrete TiRexExtended(TiRexCore) — scanner N/A MISLABEL (subclasses TiRexCore not keras.Model directly). H6 (keras.Model.build last), H7, inherited H8/H9; done: H12 call type hints; added test_model_extended.py (forward + factory + M2 round-trip; clean) |
| `[x]` | `time_series/xlstm/forecaster.py` | PASS | done: H9 from_config deserializes 6 init/reg (get_config serialized them; was cls(**config)); existing 2 round-trips pass |
| `[x]` | `time_series/xlstm/model.py` | PASS | done: H9 from_config deserializes 6 init/reg; existing round-trip passes |

### L2-Round 24 — tiny_recursive_model, tree_transformer, vae, video_jepa  (10 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `tiny_recursive_model/components.py` | PASS | done: H7 added TRMInner.compute_output_shape (carry/logits/q-head structure from config); H4 added to TRMReasoningModule + TRMInner (hidden_size%num_heads, positivity, layer counts); existing test_trm.py round-trip passes |
| `[x]` | `tiny_recursive_model/model.py` | PASS | rubric-verified clean (TRM; H4 present, M5 create_trm; H10 branches on Python flags only); existing round-trip passes |
| `[x]` | `tree_transformer/components.py` | PASS | done: H12 type hints on 3 compute_output_shape; H11 GroupAttention forwards training to norm; H4 added to TreeTransformerBlock; existing round-trip passes |
| `[x]` | `tree_transformer/model.py` | PASS | done: H12 type hints (compute_output_shape + _validate_config params); M3 correct (load_weights_from_checkpoint); M1+M5 present; existing round-trip passes |
| `[x]` | `vae/model.py` | PASS | rubric-verified clean (functional VAE; H4/H8/H9 + M1 from_variant + M5 factory; train_step tf.GradientTape is GHOST, not graded); existing round-trip passes |
| `[x]` | `video_jepa/config.py` | N/A | N/A confirmed (dataclass-config, self-validating __post_init__) |
| `[x]` | `video_jepa/encoder.py` | PASS | rubric-verified clean (VideoJEPACliffordEncoder; H5/H6/H7/H11; conditional Dropout has no weights — M4 low-severity ok) |
| `[x]` | `video_jepa/masking.py` | PASS | rubric-verified clean (TubeMaskGenerator; stateless, static-int branch off-graph) |
| `[x]` | `video_jepa/model.py` | PASS | done: H11 fixed — encode_frames now forwards training to online encoder (Clifford BatchNorm was stuck in inference mode during training); train_step GradientTape GHOST; existing round-trip passes |
| `[x]` | `video_jepa/predictor.py` | PASS | rubric-verified clean (CausalSelfAttnMLPBlock/VideoJEPAPredictor; H5/H6/H7/H11; conditional Dropout M4 low-severity ok) |

### L2-Round 25 — vit, vit_hmlp, vit_siglip, vq_vae, vq_vae_rotation, wave_field_llm, yolo12  (8 files)  — DONE

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[x]` | `vit/model.py` | PASS | rubric-verified clean (subclassed ViT; H4/H5/H6/H8/H11 good; M3 load_weights_from_checkpoint); existing round-trip passes |
| `[x]` | `vit_hmlp/model.py` | PASS | done: H6 fixed (logger.info moved before super().build()); existing round-trip passes |
| `[x]` | `vit_siglip/model.py` | PASS | rubric-verified clean (self.norm has None-default at L375 → M4 ok; static-int reshape off-graph); existing round-trip passes |
| `[x]` | `vq_vae/model.py` | PASS | rubric-verified clean (call() pure ops; train_step GradientTape GHOST); existing round-trip passes |
| `[x]` | `vq_vae_rotation/model.py` | PASS | rubric-verified clean (class docstring satisfies H13 file-level; H8/H9 serialize encoder/decoder/init); existing round-trip passes |
| `[x]` | `wave_field_llm/wave_field_llm.py` | PASS | done: WaveFieldDecoderBlock H4 (embed_dim%num_heads/positivity) + H5 (explicit build of all 6 sublayers, was attention-only); WaveFieldLLM clean (M1 from_variant); existing round-trip passes |
| `[x]` | `yolo12/feature_extractor.py` | PASS | done: H12 type hints (__init__/build); FIXED M2 round-trip — lazy first-call sublayer build dropped 76% of weights on reload; added concrete dummy-forward in build() (refactored call→_forward) to materialize weights; added test_round_trip.py |
| `[x]` | `yolo12/multitask.py` | PASS | done: H3 mutable-default lists→None sentinels; H4 (scale membership + class/reg_max positivity); H12 (__init__ -> None, typed kwargs); M2 round-trip fixed via FE materialization; added test_round_trip.py. Was smoke-only |
<!-- /L2-WORKLIST -->

> **Coverage invariant.** The 25 rounds above cover all **183** source files across all **70**
> non-empty model directories exactly once (`jepa/` is empty and excluded). Verified by a live tree
> walk at authoring time (`Glob src/dl_techniques/models/**/*.py` minus `__init__.py`/`__pycache__`
> = 183; union of round rows = 183; 70 distinct dirs).

---

## §L2-5 Known deep-gap + accepted exceptions

### Forward-path raw `tf.*` (accepted exceptions — do NOT force a `keras.ops` rewrite)

| File | Round | Status |
|------|-------|--------|
| `fftnet/model.py` (FFTMixer) | 7 | **ACCEPTED-EXCEPTION.** `fftnet` hardcodes `KERAS_BACKEND="tensorflow"`; uses `tf.signal.rfft/ifft` + `tf.complex`. `keras.ops` has `fft/fft2/ifft2/rfft/irfft/real/imag` but **NO `rfft2/irfft2/angle/complex`**. The transpose-to-last + `tf.complex(re,im)` idiom is the only option. Document in the file header (mirror PART I §5 lagrange/canny pattern) and accept. |
| `latent_gmm_registration/model.py` | 10 | **ACCEPTED-EXCEPTION.** `tf.linalg.svd` — no `keras.ops` equivalent. Document + accept. |
| `thera/hypernetwork.py` (TheraHypernetwork) | 22 | **ACCEPTED-EXCEPTION (documented, header).** `decode_with_jac` uses `tf.GradientTape.batch_jacobian` for the exact per-pixel spatial Jacobian (step-9 TV loss); `keras.ops` has no backend-agnostic AD/batched-Jacobian (mirrors `physics/lagrange_layer`). Training-only loss path, NOT the inference forward path, never graph-traced by `.keras`. The plain forward path (get_phi_at_coords/_compute_rel_and_phi) was MIGRATED off raw-tf to keras.ops in Round 22. |
| `memory_bank/write_controller.py` (MemoryWriteController) | 12 | **VERIFY IN-ROUND (HYPOTHESIS).** The flagged `tf.*` is a debug-guard `tf.debugging.assert_less_equal` behind a backend check, possibly not in the graph forward path. Confirm whether it's inside `call()`; if a debug-only assertion, replace with a `keras.ops` check or accept-and-document. 2-attempt leash; do not force a broken rewrite. |

(Plus inherited layer-level accepted exceptions consumed by models: `tf.math.bessel_i0e` in
`layers/sampling.py`; the PART I §5 `physics/lagrange_layer.py`, `canny.py`, `complex_layers.py`.)

### Non-trivial mechanical FAILs (verify in-round; HYPOTHESES from the authoring audit)

- **`cliffordnet/model.py` (CliffordNet) H6** — Round 4. `super().build()` is placed FIRST, then a
  symbolic dummy-forward block runs. Moving `super().build()` to be structurally last may interact
  with the dummy forward. Investigate; if a clean reorder isn't possible in 2 attempts, mark `[~]`
  with a one-line note and surface it. (HYPOTHESIS — re-verify the actual `build()` body.)
- The other 11 `super_build_last` FAILs are expected TRIVIAL (move `super().build()` / a trailing
  `logger`/extra statement so `super().build()` is last) — but grade each fresh.
- The 3 `compute_output_shape` FAILs (`cliffordnet/unet.py` _DetectionHeadBlock,
  `sam/prompt_encoder.py` PositionEmbeddingRandom, `tiny_recursive_model/components.py` TRMInner) are
  **embedded `keras.layers.Layer` subclasses** → H7 is HARD for them: derive the shape formula from
  `call()`.

### Pre-existing model issues (NOT introduced by Level-2; fix opportunistically, don't block a round)

- **`sam/`** — `test_output_consistency_after_loading` mask drift 28-32% at atol=1e-5 (pre-existing,
  mask-decoder `use_causal_mask` Keras-3 incompat; image-encoder path already fixed). Round 19.
- **`dino/dino_v1.py`** — `DINOHead` `.keras` round-trip broken (pre-existing); likely a side-effect
  of the H6 FAIL — the Round 7 H6 fix should also resolve the round-trip. Verify M2 after fixing.
- **`modern_bert/modern_bert_blt.py`** — OOM in ngram-hash Embedding at seq_len 16384 (resource
  scaling, not a correctness bug); smoke with a small window/config. Round 14.

### Smoke-harness SKIPs (documented; not a per-round blocker)

- **`jepa`** — empty package (encoder/predictor live in `video_jepa`); no top-level `keras.Model`;
  EXCLUDED from the 183-file denominator entirely.
- **`ccnets`** — 3-model orchestrator (Round 3), not a single `model(x)` callable → no smoke entry.
  Its concrete classes still get the mechanical + pytest treatment; the smoke step is a documented
  SKIP for this package.

### No layer-extraction mandate

`research/2026_models_layer_reuse_audit.md` catalogs 102 inline `keras.layers.Layer` subclasses in
`models/`; per institutional memory **0 are safe drop-in replaceable as-is** and audit verdicts are
HYPOTHESES. Level-2 rounds bring inline layers up to the §L2-2 rubric IN PLACE; they do NOT mandate
extracting them into `layers/`. Opportunistic, source-verified reuse only.

---

## §L2-6 Handover Prompt

Copy the fenced block below **verbatim** into a new, context-cleared agent session to drive ONE
Level-2 round. It depends on nothing but the files in this repository.

```text
You are picking up an ongoing, multi-session effort to make every Keras model in
`src/dl_techniques/models/` production-quality (Level 2). You have NO memory of prior sessions;
everything you need is in the repository. Work exactly ONE round, then stop.

STEP 0 — Read the roadmap.
Read `roadmap/production_map.md` — specifically PART II (sections §L2-1…§L2-7). Internalize:
  - §L2-2 The Model Production-Quality Rubric — the H1-H14 carry-over (H5 conditional; H6 HARD when a
    build() override exists; H7 SOFT for keras.Model but HARD for embedded keras.layers.Layer
    subclasses) and the model addendum M1-M6 (M2 .keras round-trip HARD, M4 always-create sublayers
    HARD; M1 variants / M3 weight-handling / M5 factory / M6 names SOFT). Do NOT grade or add
    `if self.built: return` or `train_step` (GHOST). Honor the weight-handling policy.
  - §L2-3 How a Level-2 Round Works — the per-round procedure and standing decisions (FULL re-audit;
    audit AND fix in the same round; one round per session; commit locally, the USER pushes).
  - §L2-4 Batched Worklist — the ordered rounds (checkboxes + baseline verdict + gap-hint).
  - §L2-5 Known deep-gap + accepted exceptions — accepted raw-tf (fftnet, latent_gmm), the non-trivial
    cliffordnet H6, the memory_bank H10 to verify, pre-existing issues (sam, dino), smoke SKIPs.

STEP 1 — Pick the next round.
In §L2-4 find the LOWEST-numbered round whose rows are still `[ ]` PENDING. Announce the round number
and list its files before doing anything else.

STEP 2 — Mechanical audit (the §L2-4 verdicts are a STALE baseline — re-run):
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_layers.py --models --path <file1> <file2> ...
(`--path` accepts a directory; add `--json /tmp/round.json` for machine output.) The scanner is an
AID, not an oracle — it can mislabel aliased imports / metaclass ABCs / a model vs an embedded layer.

STEP 3 — Grade + fix each file against the FULL §L2-2 rubric by READING it.
The canonical exemplar is `src/dl_techniques/models/bert/bert.py`; the spec is
`research/2026_keras_custom_models_instructions.md` (model sections §5/§6/§7/§8/§9/§11; gold example
§15.2). Fix all [HARD] gaps and reasonable [SOFT] gaps to match bert.
  - For an H7 FAIL: if the class is a `keras.Model` subclass, H7 is SOFT (skip); if it is an embedded
    `keras.layers.Layer` subclass, H7 is HARD — add `compute_output_shape` from stored config.
  - For an H6 FAIL: make `super().build()` the structural last statement (move any trailing
    logger/extra statements before it). Do NOT add `if self.built: return`.
  - For a forward-path raw-`tf.` gap that cannot cleanly migrate to `keras.ops` (§L2-5): do NOT force
    a broken rewrite. Make at most 2 fix attempts; if it won't come clean, document the exception in
    the file header (mirror PART I §5) and accept it, or mark the row `[~]` with a one-line note and
    surface it in your final report.
  - Weight handling: if the file loads real checkpoints via `load_weights(by_name=True)` on a `.keras`
    file, migrate to `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint` (SOFT). Do NOT
    change bert's own (GHOST).

STEP 4 — Ensure/repair a test under `tests/test_models/test_<name>/` mirroring the model. Cover at
minimum: construction (incl. ValueError paths), a forward pass, and the M2 full `.keras` round-trip —
`model.save(path)` then `keras.models.load_model(path, custom_objects={...})`, assert identical output
before/after (atol 1e-4 on GPU fp32). Add variant/factory coverage where the model has them.

STEP 5 — Runtime smoke for each model in the batch (skip only the documented SKIPs jepa/ccnets):
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python scripts/verify_models_smoke.py --only <name>

STEP 6 — Scoped pytest (GPU1, serial — repo convention; never run GPU jobs in parallel):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_<name>/ -x
All selected tests must pass before you check anything off.

STEP 7 — Two-instrument re-check on the batch:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_layers.py --models --path <batch files>
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python scripts/verify_models_smoke.py --only <each model>
Confirm every file reports PASS (or a documented accepted exception per §L2-5, or a documented smoke
SKIP). An unexplained FAIL means the round is not done.

STEP 8 — Update §L2-4 in place. Flip each completed row `[ ]` → `[x]` (use `[~]` for a deferred deep
gap with a one-line note). Then update the tally line — `**Progress: X / 183 files
production-verified**` — incrementing X by the files you finished. Edit only THIS round's rows.

STEP 9 — Commit locally (do NOT push). Stage ONLY the files you edited (model files, their tests, and
`roadmap/production_map.md`). Do NOT use `git add -A`. Commit message:
    git commit -m "[production-map/L2-round-N] <short batch description>"
Do NOT push — the user pushes themselves.

STEP 10 — Report and STOP. Report: which round you completed, a per-file before/after verdict, the
tests + smoke you ran and their results, the new "X / 183" tally, and which round is next. Then STOP.
ONE round per session — do not chain. Let the user clear context.
```

---

## §L2-7 Provenance

PART II (the `--models` extension of `scripts/audit_layers.py`, the model rubric, the 183-file
worklist, and the §L2-6 handover prompt) was authored as durable, standalone infrastructure, mirroring
the proven PART I shape. It is self-contained: every Level-2 round is driven solely by this document
plus the two instruments (`scripts/audit_layers.py --models`, `scripts/verify_models_smoke.py`) — no
external session state or planning scaffold is required. Canonical exemplar:
`src/dl_techniques/models/bert/bert.py`. Spec: `research/2026_keras_custom_models_instructions.md`.
Baseline commit: `7b1e28af` (127 PASS / 16 FAIL / 40 N/A over 183 files / 70 dirs).
