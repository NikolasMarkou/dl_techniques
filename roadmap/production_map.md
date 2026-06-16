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

**Progress: 0 / 245 files production-verified**

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
| `[ ]` | `physics/approximate_lagrange_layer.py` | PASS | rubric-verify |
| `[ ]` | `physics/lagrange_layer.py` | FAIL | forward-raw-tf (ACCEPTED-EXCEPTION candidate — see §5) |
| `[ ]` | `tokenizers/bpe.py` | PASS | rubric-verify |

### Round 2 — graphs/ (4 of 5 untested) + heads/vlm/  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `graphs/entity_graph_refinement.py` | FAIL | super_build_last |
| `[ ]` | `graphs/fermi_diract_decoder.py` | PASS | rubric-verify |
| `[ ]` | `graphs/graph_neural_network.py` | PASS | rubric-verify |
| `[ ]` | `graphs/relational_graph_transformer_blocks.py` | PASS | rubric-verify |
| `[ ]` | `graphs/simplified_hyperbolic_graph_convolutional_neural_layer.py` | PASS | rubric-verify (findings grep flagged raw-tf outside call — re-check, see §5) |
| `[ ]` | `heads/vlm/factory.py` | FAIL | compute_output_shape |
| `[ ]` | `heads/vlm/task_types.py` | N/A | N/A (no concrete layer) |

### Round 3 — untested root-level files (1/3)  (11 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `anchor_generator.py` | PASS | rubric-verify |
| `[ ]` | `bitlinear_layer.py` | FAIL | super_build_last |
| `[ ]` | `blt_blocks.py` | FAIL | super_build_last |
| `[ ]` | `clahe.py` | PASS | rubric-verify (findings grep flagged raw-tf outside call — re-check, see §5) |
| `[ ]` | `conditional_output_layer.py` | PASS | rubric-verify |
| `[ ]` | `conv2d_builder.py` | N/A | N/A (ConvType Enum only) |
| `[ ]` | `depthwise_separable_block.py` | FAIL | super_build_last |
| `[ ]` | `downsample.py` | N/A | N/A (no concrete layer) |
| `[ ]` | `eomt_mask.py` | PASS | rubric-verify |
| `[ ]` | `fft_layers.py` | PASS | rubric-verify |
| `[ ]` | `film.py` | PASS | rubric-verify |

### Round 4 — untested root-level files (2/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `fractal_block.py` | FAIL | super_build_last |
| `[ ]` | `hierarchical_mlp_stem.py` | PASS | rubric-verify |
| `[ ]` | `inverted_residual_block.py` | N/A | N/A (no concrete layer) |
| `[ ]` | `io_preparation.py` | PASS | rubric-verify |
| `[ ]` | `modality_projection.py` | FAIL | super_build_last |
| `[ ]` | `mothnet_blocks.py` | PASS | rubric-verify |
| `[ ]` | `one_hot_encoding.py` | PASS | rubric-verify |
| `[ ]` | `patch_merging.py` | PASS | rubric-verify |
| `[ ]` | `random_fourier_features.py` | PASS | rubric-verify |
| `[ ]` | `restricted_boltzmann_machine.py` | PASS | rubric-verify |

### Round 5 — untested root-level files (3/3)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `router.py` | FAIL | forward-raw-tf (DEEP gap — see §5) |
| `[ ]` | `selective_gradient_mask.py` | PASS | rubric-verify |
| `[ ]` | `sparse_autoencoder.py` | PASS | rubric-verify |
| `[ ]` | `spatial_layer.py` | PASS | rubric-verify |
| `[ ]` | `standard_blocks.py` | PASS | rubric-verify |
| `[ ]` | `stochastic_gradient.py` | PASS | rubric-verify |
| `[ ]` | `strong_augmentation.py` | PASS | rubric-verify |
| `[ ]` | `tabm_blocks.py` | PASS | rubric-verify |
| `[ ]` | `universal_inverted_bottleneck.py` | PASS | rubric-verify |
| `[ ]` | `upsample.py` | N/A | N/A (no concrete layer) |

### Round 6 — memory/ (NEEDS-AUDIT)  (8 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `memory/baseline_ntm.py` | PASS | rubric-verify |
| `[ ]` | `memory/factory.py` | N/A | N/A (factory-only) |
| `[ ]` | `memory/mann.py` | PASS | rubric-verify |
| `[ ]` | `memory/neuro_grid.py` | PASS | rubric-verify |
| `[ ]` | `memory/ntm_interface.py` | N/A | N/A (ABC / interface) |
| `[ ]` | `memory/som_2d_layer.py` | N/A | N/A (no concrete layer per scanner — human re-check) |
| `[ ]` | `memory/som_nd_layer.py` | FAIL | super_build_last |
| `[ ]` | `memory/som_nd_soft_layer.py` | FAIL | super_build_last |

### Round 7 — moe/ + fusion/ (NEEDS-AUDIT)  (6 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `moe/config.py` | N/A | N/A (dataclass config) |
| `[ ]` | `moe/experts.py` | N/A | N/A (scanner — human re-check) |
| `[ ]` | `moe/gating.py` | N/A | N/A (scanner — human re-check) |
| `[ ]` | `moe/integration.py` | N/A | N/A (scanner — human re-check) |
| `[ ]` | `moe/layer.py` | FAIL | super_build_last |
| `[ ]` | `fusion/multimodal_fusion.py` | PASS | rubric-verify |

### Round 8 — statistics/ (NEEDS-AUDIT; 3 dead-code candidates)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `statistics/deep_kernel_pca.py` | PASS | rubric-verify (DEAD-CODE candidate — keep/delete, see §5) |
| `[ ]` | `statistics/invertible_kernel_pca.py` | PASS | rubric-verify (DEAD-CODE candidate — keep/delete, see §5) |
| `[ ]` | `statistics/mdn_layer.py` | FAIL | super_build_last |
| `[ ]` | `statistics/moving_std.py` | PASS | rubric-verify |
| `[ ]` | `statistics/normalizing_flow.py` | PASS | rubric-verify |
| `[ ]` | `statistics/residual_acf.py` | PASS | rubric-verify (DEAD-CODE candidate `ResidualACFLayer` — keep/delete, see §5) |
| `[ ]` | `statistics/scaler.py` | PASS | rubric-verify |

### Round 9 — time_series/ (1/2, NEEDS-AUDIT)  (7 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `time_series/adaptive_lag_attention.py` | FAIL | super_build_last |
| `[ ]` | `time_series/deepar_blocks.py` | FAIL | compute_output_shape |
| `[ ]` | `time_series/ema_layer.py` | PASS | rubric-verify |
| `[ ]` | `time_series/forecasting_layers.py` | PASS | rubric-verify |
| `[ ]` | `time_series/mixed_sequential_block.py` | PASS | rubric-verify |
| `[ ]` | `time_series/nbeats_blocks.py` | N/A | N/A (scanner — human re-check) |
| `[ ]` | `time_series/nbeatsx_blocks.py` | N/A | N/A (scanner — human re-check) |

### Round 10 — time_series/ (2/2, NEEDS-AUDIT)  (6 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `time_series/prism_blocks.py` | PASS | rubric-verify |
| `[ ]` | `time_series/quantile_head_fixed_io.py` | PASS | rubric-verify |
| `[ ]` | `time_series/quantile_head_variable_io.py` | PASS | rubric-verify |
| `[ ]` | `time_series/temporal_convolutional_network.py` | PASS | rubric-verify |
| `[ ]` | `time_series/temporal_fusion.py` | FAIL | super_build_last |
| `[ ]` | `time_series/xlstm_blocks.py` | FAIL | compute_output_shape |

### Round 11 — logic/ + reasoning/ + geometric/ (NEEDS-AUDIT)  (10 files)

| done | file | verdict | gap-hint |
|------|------|---------|----------|
| `[ ]` | `logic/arithmetic_operators.py` | PASS | rubric-verify |
| `[ ]` | `logic/factory.py` | N/A | N/A (factory-only — human re-check for inline layers) |
| `[ ]` | `logic/logic_operators.py` | PASS | rubric-verify |
| `[ ]` | `logic/neural_circuit.py` | PASS | rubric-verify |
| `[ ]` | `reasoning/hrm_reasoning_core.py` | PASS | rubric-verify |
| `[ ]` | `reasoning/hrm_reasoning_module.py` | PASS | rubric-verify |
| `[ ]` | `reasoning/hrm_sparse_puzzle_embedding.py` | PASS | rubric-verify |
| `[ ]` | `geometric/clifford_block.py` | PASS | rubric-verify |
| `[ ]` | `geometric/point_cloud_autoencoder.py` | PASS | rubric-verify |
| `[ ]` | `geometric/supernode_pooling.py` | PASS | rubric-verify |

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

*(authored in step 3)*
