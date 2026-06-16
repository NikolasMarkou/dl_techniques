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

### Level 2 — Models (FUTURE, not started)
After Level 1 completes, the same audit+fix discipline applies to model definitions under
`src/dl_techniques/models/` (~70 model directories). Models compose layers, so they inherit the same
serialization / `get_config` / `build` discipline plus model-level concerns (factory functions,
end-to-end `.keras` round-trip, training-loop correctness). **No round in this document touches
models.** A separate worklist will be authored when Level 1 is done.

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

**Progress: 85 / 245 files production-verified**

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
| `[ ]` | `heads/factory.py` | N/A | N/A (factory-only) |
| `[ ]` | `heads/nlp/factory.py` | PASS | rubric-verify |
| `[ ]` | `heads/nlp/task_types.py` | N/A | N/A (Enum / task-type config) |
| `[ ]` | `heads/task_types.py` | N/A | N/A (Enum / task-type config) |
| `[ ]` | `heads/vision/factory.py` | FAIL | compute_output_shape |
| `[ ]` | `heads/vision/task_types.py` | N/A | N/A (Enum / task-type config) |

### Round 13 — attention/ re-audit (1/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `attention/anchor_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/attention_routing_capsule.py` | FAIL | super_build_last |
| `[ ]` | `attention/capsule_routing_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/channel_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/convolutional_block_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/differential_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/factory.py` | N/A | N/A (factory-only) |
| `[ ]` | `attention/fnet_fourier_transform.py` | PASS | rubric-verify |
| `[ ]` | `attention/gated_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/group_query_attention.py` | PASS | rubric-verify |

### Round 14 — attention/ re-audit (2/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `attention/hopfield_attention.py` | FAIL | super_build_last |
| `[ ]` | `attention/ideogram4_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/lighthouse_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/mmdit_joint_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/mobile_mqa.py` | N/A | N/A (scanner — human re-check) |
| `[ ]` | `attention/multi_head_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/multi_head_cross_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/multi_head_latent_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/non_local_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/perceiver_attention.py` | PASS | rubric-verify |

### Round 15 — attention/ re-audit (3/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `attention/performer_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/progressive_focused_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/ring_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/rpc_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/shared_weights_cross_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/single_window_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/spatial_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/tripse_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/wave_field_attention.py` | PASS | rubric-verify |
| `[ ]` | `attention/window_attention.py` | PASS | rubric-verify |

### Round 16 — embedding/ re-audit (1/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `embedding/albert_factorized_embedding.py` | PASS | rubric-verify |
| `[ ]` | `embedding/bert_embeddings.py` | FAIL | super_build_last |
| `[ ]` | `embedding/class_token.py` | PASS | rubric-verify |
| `[ ]` | `embedding/continuous_rope_embedding.py` | PASS | rubric-verify |
| `[ ]` | `embedding/continuous_sin_cos_embedding.py` | PASS | rubric-verify |
| `[ ]` | `embedding/dual_rotary_position_embedding.py` | PASS | rubric-verify |
| `[ ]` | `embedding/factory.py` | N/A | N/A (factory-only) |
| `[ ]` | `embedding/hierarchical_codebook_embedding.py` | PASS | rubric-verify |

### Round 17 — embedding/ re-audit (2/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `embedding/mask_token.py` | PASS | rubric-verify |
| `[ ]` | `embedding/modern_bert_embeddings.py` | PASS | rubric-verify |
| `[ ]` | `embedding/multi_axis_rope.py` | PASS | rubric-verify |
| `[ ]` | `embedding/patch_embedding.py` | FAIL | super_build_last |
| `[ ]` | `embedding/positional_embedding.py` | FAIL | super_build_last |
| `[ ]` | `embedding/positional_embedding_sine_2d.py` | PASS | rubric-verify |
| `[ ]` | `embedding/rotary_position_embedding.py` | PASS | rubric-verify |
| `[ ]` | `embedding/scalar_sinusoidal_embedding.py` | PASS | rubric-verify |

### Round 18 — activations/ re-audit (1/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `activations/adaptive_softmax.py` | PASS | rubric-verify |
| `[ ]` | `activations/basis_function.py` | PASS | rubric-verify |
| `[ ]` | `activations/differentiable_step.py` | PASS | rubric-verify |
| `[ ]` | `activations/expanded_activations.py` | PASS | rubric-verify |
| `[ ]` | `activations/factory.py` | N/A | N/A (factory-only — human re-check for inline layers) |
| `[ ]` | `activations/golu.py` | PASS | rubric-verify |
| `[ ]` | `activations/hard_sigmoid.py` | PASS | rubric-verify |
| `[ ]` | `activations/hard_swish.py` | PASS | rubric-verify |

### Round 19 — activations/ re-audit (2/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `activations/mish.py` | PASS | rubric-verify |
| `[ ]` | `activations/monotonicity_layer.py` | PASS | rubric-verify |
| `[ ]` | `activations/probability_output.py` | PASS | rubric-verify |
| `[ ]` | `activations/relu_k.py` | PASS | rubric-verify |
| `[ ]` | `activations/routing_probabilities.py` | FAIL | super_build_last |
| `[ ]` | `activations/sparsemax.py` | PASS | rubric-verify |
| `[ ]` | `activations/squash.py` | PASS | rubric-verify |
| `[ ]` | `activations/thresh_max.py` | PASS | rubric-verify |

### Round 20 — ffn/ re-audit (1/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `ffn/counting_ffn.py` | PASS | rubric-verify |
| `[ ]` | `ffn/diff_ffn.py` | PASS | rubric-verify |
| `[ ]` | `ffn/factory.py` | N/A | N/A (factory-only) |
| `[ ]` | `ffn/gated_mlp.py` | PASS | rubric-verify |
| `[ ]` | `ffn/geglu_ffn.py` | PASS | rubric-verify |
| `[ ]` | `ffn/gelu_mlp_ffn.py` | PASS | rubric-verify |
| `[ ]` | `ffn/glu_ffn.py` | PASS | rubric-verify |
| `[ ]` | `ffn/kan_linear.py` | PASS | rubric-verify (findings grep flagged raw-tf outside call — re-check, see §5) |

### Round 21 — ffn/ re-audit (2/2)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `ffn/logic_ffn.py` | FAIL | super_build_last |
| `[ ]` | `ffn/mlp.py` | PASS | rubric-verify |
| `[ ]` | `ffn/orthoglu_ffn.py` | PASS | rubric-verify |
| `[ ]` | `ffn/power_mlp_layer.py` | FAIL | super_build_last |
| `[ ]` | `ffn/residual_block.py` | PASS | rubric-verify |
| `[ ]` | `ffn/swiglu_ffn.py` | PASS | rubric-verify |
| `[ ]` | `ffn/swin_mlp.py` | PASS | rubric-verify |
| `[ ]` | `ffn/tversky_projection.py` | PASS | rubric-verify |

### Round 22 — norms/ re-audit (1/2)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `norms/adaptive_band_rms.py` | PASS | rubric-verify |
| `[ ]` | `norms/band_logit_norm.py` | PASS | rubric-verify |
| `[ ]` | `norms/band_rms.py` | PASS | rubric-verify |
| `[ ]` | `norms/dynamic_tanh.py` | PASS | rubric-verify |
| `[ ]` | `norms/factory.py` | N/A | N/A (factory-only) |
| `[ ]` | `norms/global_response_norm.py` | FAIL | super_build_last |
| `[ ]` | `norms/logit_norm.py` | PASS | rubric-verify |

### Round 23 — norms/ re-audit (2/2)  (6 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `norms/max_logit_norm.py` | PASS | rubric-verify |
| `[ ]` | `norms/polar_weight_norm.py` | PASS | rubric-verify (findings grep flagged raw-tf outside call — re-check, see §5) |
| `[ ]` | `norms/rms_norm.py` | PASS | rubric-verify |
| `[ ]` | `norms/zero_centered_adaptive_band_rms_norm.py` | PASS | rubric-verify |
| `[ ]` | `norms/zero_centered_band_rms_norm.py` | PASS | rubric-verify |
| `[ ]` | `norms/zero_centered_rms_norm.py` | PASS | rubric-verify |

### Round 24 — transformers/ re-audit (1/2)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `transformers/adaln_zero.py` | PASS | rubric-verify |
| `[ ]` | `transformers/eomt_transformer.py` | PASS | rubric-verify |
| `[ ]` | `transformers/free_transformer.py` | PASS | rubric-verify |
| `[ ]` | `transformers/ideogram4_block.py` | PASS | rubric-verify |
| `[ ]` | `transformers/perceiver_transformer.py` | PASS | rubric-verify |
| `[ ]` | `transformers/progressive_focused_transformer.py` | PASS | rubric-verify |
| `[ ]` | `transformers/sd3_adaln.py` | PASS | rubric-verify |

### Round 25 — transformers/ re-audit (2/2)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `transformers/swin_conv_block.py` | FAIL | super_build_last |
| `[ ]` | `transformers/swin_transformer_block.py` | FAIL | super_build_last |
| `[ ]` | `transformers/text_decoder.py` | PASS | rubric-verify |
| `[ ]` | `transformers/text_encoder.py` | PASS | rubric-verify |
| `[ ]` | `transformers/transformer.py` | PASS | rubric-verify |
| `[ ]` | `transformers/transformer_decoder.py` | PASS | rubric-verify |
| `[ ]` | `transformers/vision_encoder.py` | PASS | rubric-verify |

### Round 26 — mixtures/ + sequence_pooling/ re-audit  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `mixtures/factory.py` | N/A | N/A (factory-only) |
| `[ ]` | `mixtures/gmm.py` | PASS | rubric-verify |
| `[ ]` | `mixtures/kmeans.py` | PASS | rubric-verify |
| `[ ]` | `mixtures/radial_basis_function.py` | PASS | rubric-verify |
| `[ ]` | `sequence_pooling/attention_pooling.py` | FAIL | super_build_last |
| `[ ]` | `sequence_pooling/factory.py` | N/A | N/A (factory-only) |
| `[ ]` | `sequence_pooling/sequence_pooling.py` | FAIL | super_build_last |
| `[ ]` | `sequence_pooling/weighted_pooling.py` | FAIL | super_build_last |

### Round 27 — tested root-level files re-audit (1/5)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `bias_free_conv1d.py` | PASS | rubric-verify |
| `[ ]` | `bias_free_conv2d.py` | PASS | rubric-verify |
| `[ ]` | `blt_core.py` | PASS | rubric-verify |
| `[ ]` | `blur_pool.py` | FAIL | super_build_last |
| `[ ]` | `canny.py` | FAIL | forward-raw-tf (DEEP gap — see §5) |
| `[ ]` | `capsules.py` | FAIL | super_build_last |
| `[ ]` | `complex_layers.py` | FAIL | compute_output_shape + forward-raw-tf (DEEP gap — see §5) |
| `[ ]` | `convnext_v1_block.py` | PASS | CANONICAL EXEMPLAR — confirm still full-PASS |
| `[ ]` | `convnext_v2_block.py` | PASS | rubric-verify |
| `[ ]` | `convolutional_kan.py` | FAIL | super_build_last |

### Round 28 — tested root-level files re-audit (2/5)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `dynamic_conv2d.py` | PASS | rubric-verify |
| `[ ]` | `fnet_encoder_block.py` | PASS | rubric-verify |
| `[ ]` | `gated_delta_net.py` | PASS | rubric-verify |
| `[ ]` | `gaussian_filter.py` | PASS | rubric-verify |
| `[ ]` | `gaussian_pyramid.py` | PASS | rubric-verify |
| `[ ]` | `global_sum_pool_2d.py` | PASS | rubric-verify |
| `[ ]` | `grid_sample.py` | N/A | N/A (pure-function module, by design) |
| `[ ]` | `haar_wavelet_decomposition.py` | PASS | rubric-verify |
| `[ ]` | `hanc_block.py` | PASS | rubric-verify |
| `[ ]` | `hanc_layer.py` | PASS | rubric-verify |

### Round 29 — tested root-level files re-audit (3/5)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `laplacian_filter.py` | PASS | rubric-verify |
| `[ ]` | `layer_scale.py` | PASS | rubric-verify |
| `[ ]` | `mobile_one_block.py` | PASS | rubric-verify |
| `[ ]` | `mps_layer.py` | PASS | rubric-verify |
| `[ ]` | `multi_level_feature_compilation.py` | PASS | rubric-verify |
| `[ ]` | `orthoblock.py` | PASS | rubric-verify |
| `[ ]` | `orthogonal_butterfly.py` | PASS | rubric-verify |
| `[ ]` | `pixel_shuffle.py` | PASS | rubric-verify |
| `[ ]` | `pixel_unshuffle.py` | FAIL | super_build_last |
| `[ ]` | `repmixer_block.py` | PASS | rubric-verify |

### Round 30 — tested root-level files re-audit (4/5)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `res_path.py` | PASS | rubric-verify |
| `[ ]` | `rigid_simplex_layer.py` | PASS | rubric-verify |
| `[ ]` | `sampling.py` | PASS | rubric-verify (findings grep flagged raw-tf outside call — re-check, see §5) |
| `[ ]` | `shearlet_transform.py` | PASS | rubric-verify |
| `[ ]` | `squeeze_excitation.py` | PASS | rubric-verify |
| `[ ]` | `stochastic_depth.py` | PASS | rubric-verify |
| `[ ]` | `thera_heat_field.py` | PASS | rubric-verify |
| `[ ]` | `vector_quantizer.py` | PASS | rubric-verify |
| `[ ]` | `vector_quantizer_rotation_trick.py` | PASS | rubric-verify |
| `[ ]` | `yolo12_blocks.py` | PASS | rubric-verify |

### Round 31 — tested root-level files re-audit (5/5)  (1 file)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `yolo12_heads.py` | PASS | rubric-verify |
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

**Reconciled DEEP-gap (true forward-path) list — 4 files / 7 layers:**

| File | Round | Note |
|------|-------|------|
| `physics/lagrange_layer.py` | 1 | **ACCEPTED-EXCEPTION candidate.** Uses `tf.GradientTape` (and `tf.linalg.pinv`) for physics gradient computation; `tf.GradientTape` has **no `keras.ops` equivalent**. A backend-agnostic rewrite would require a different architecture (e.g. JAX `jax.grad`). **Document the exception in the file header and accept it — do NOT force a rewrite.** This file may pass the round with a documented N/A on H10 rather than a fix. |
| `complex_layers.py` | 27 | `tf.complex`, `tf.math.real/imag`, complex `dtype` args. `keras.ops.real`/`imag` exist but complex-dtype handling is backend-specific; partial fix possible, full fix may need backend-conditional code. Also FAILs `compute_output_shape`. Real work, not a quick fix. |
| `canny.py` | 27 | `tf.*` in edge-detection ops; migrate to `keras.ops` where possible. |
| `router.py` | 5 | `tf.*` in routing ops; migrate to `keras.ops` where possible. |

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
