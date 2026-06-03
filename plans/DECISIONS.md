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

## plan_2026-06-03_943569ad
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-03_943569ad/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-03
**Context**: ConvNeXt V1/V2 implement drop_path with a hand-rolled `keras.layers.Dropout(noise_shape=(None,1,1,1))` per block, bypassing the repo's dedicated `StochasticDepth` layer. EXPLORE confirmed `StochasticDepth` is a semantic drop-in (per-sample mask + `/keep_prob` rescale) while `StochasticGradient` is a different, forward-identity grad-only regularizer. The user requested BOTH be available.
**Decision**: Add a `stochastic_mode: str = 'depth'` ctor kwarg to both models; `'depth'` -> `StochasticDepth`, `'gradient'` -> `StochasticGradient`; keep `drop_path_rate` and all surrounding schedule/guard/storage/call logic unchanged; plumb `stochastic_mode` through `get_config()` and validate against `{'depth','gradient'}`.
**Trade-off**: A configurable, idiom-aligned, opt-in `'gradient'` regularizer **at the cost of** one extra serialized config key + a 2-site duplicated swap (v1/v2) carried as intentional duplication rather than a shared helper.
**Reasoning**: Default `'depth'` is behavior-preserving (in distribution), so training scripts and existing tests are untouched. Exposing `'gradient'` (zero production usage, unvalidated under save/load on non-TF backends) is acceptable because it is opt-in and its blast radius is confined to that mode. Rejected alternatives: (a) silent unconditional swap to `StochasticDepth` (loses the requested gradient option); (b) renaming `drop_path_rate` (breaks serialization + 3 training scripts + tests — HARD constraint); (c) extracting a shared v1/v2 helper (earned-abstraction rule: only 2 end-of-line call sites, no payoff).
**Anchor-Refs**: `src/dl_techniques/models/convnext/convnext_v1.py:310` (anchor), `:151,184,318-322,629` (kwarg/store/swap/get_config), `src/dl_techniques/models/convnext/convnext_v2.py:327` (anchor), `:164,319-334,682`

### D-002 | REFLECT iter-1 | 2026-06-03
**Context**: All 3 steps shipped (commits e8656c4b, 706af37f, 013218cb). REFLECT ran the full SC battery.
**Outcome**: 6/6 success criteria PASS. Scoped suite **138 passed** (84.69s). A2 confirmed (factories thread `**kwargs`, no factory edits); F4 cleared (`_build_stage` runs after `self.stochastic_mode` assignment in both files); A3 confirmed (only model-level Dropout was the drop_path one; remaining grep hits are comments). 0 errors introduced (both D-001 anchors plan-qualified); the 50 repo-wide validate-plan errors are inherited debt in unrelated files (gpt2/nam/nlp/convnext_patch_vae_v2).
**Simplification checks**: no blockers — 0 files added, 0 abstractions, intentional 2-site duplication. Diff clean (no debug/TODO/dead code).
**Not verified (accepted)**: bit-exact `'depth'`-vs-old-`Dropout` numerics (out of scope, F2); `'gradient'` training convergence (opt-in, build/forward + round-trip only).
**Recommendation**: → CLOSE.

## plan_2026-06-02_da7698bc
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-02_da7698bc/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-02
**Context**: The `.md` documents BOTH `PolarWeightNorm` (this file) and `PolarInitializer`, which lives in a separate file (`initializers/polar_initializer.py`) that already carries its own rich module docstring and is pointed to from `initializers/README.md`.
**Decision**: Merge into `polar_weight_norm.py` ONLY the content pertaining to this module (PolarWeightNorm + polar_encode/decode + shared idea/provenance). For `PolarInitializer`, include a one-line cross-reference, NOT a duplicate of its args table/example.
**Trade-off**: Single-source-of-truth / no doc drift **at the cost of** the merged docstring not being a 1:1 superset of the `.md` (PolarInitializer's usage example moves out of view of this file's reader).
**Reasoning**: DRY (LESSONS: aspirational/duplicated docs rot). Mirrors `orthogonal_butterfly.py` precedent, which merged only its own layer's content. Duplicating PolarInitializer's full docs into a norms-layer file would create two divergent sources for one class. Alternative (full duplication) rejected; alternative (move PolarInitializer docs into its own file's docstring) is out of scope — that file already has a rich docstring. **Surfaced for user approval at PC-PLAN.**

### D-002 | PLAN | 2026-06-02
**Context**: "Refine code to pass the 2026 instructions" — finding 2 verified the code already satisfies all 15 MUST rules (M1-M15).
**Decision**: Scope the code refinement to cosmetic/dialect polish (RBF-style `# ---` dividers; drop Sphinx `:func:`/`:class:` roles from the module docstring) + the docstring merge. No logic/structure edit to `__init__`/`build`/`call`/`get_config`.
**Trade-off**: Minimal, behavior-preserving diff **at the cost of** not "rewriting to look maximally like RBF" (e.g. NOT converting the already-Google class docstring to RBF's Sphinx dialect — that would be a regression per LESSONS line 144).
**Reasoning**: Instruction doc is the authority over template dialect. The code already exceeds RBF on compliance (RBF lacks `logger`). Net risk minimized; existing test stays green.

### D-003 | REFLECT (iter 1) | 2026-06-03
**Context**: Executor added a literal `# ---` divider per plan text, but the file already had a titled banner divider (`# PolarWeightNorm layer`, 75-hyphen) immediately before the class, and an inconsistent bare `# ---` at EOF.
**Decision**: Dropped the redundant pre-class `# ---` (banner already satisfies "divider before class") and made the EOF divider a 75-hyphen line matching the file's OWN existing convention — not butterfly's 69-hyphen literal.
**Trade-off**: Internal file consistency **at the cost of** not byte-matching the butterfly precedent's exact divider width.
**Reasoning**: Quality-match means consistency, not blind copy. The file's pre-existing 75-hyphen banners are the local convention; matching them reads cleaner than importing a second divider width. Folded into the step-1 commit via `--amend` (local, unpushed).

### REFLECT Simplification Checks (6) — iter 1
1. **Reuse before write**: yes — mirrored butterfly precedent; reused existing banner divider instead of adding one.
2. **Net-negative LOC**: yes — -67 net (deleted 129-line `.md`).
3. **Essential vs accidental complexity**: no accidental complexity added; docstring is essential documentation.
4. **Junior-dev test**: a junior reading `polar_weight_norm.py` now has the full design self-contained; no external `.md` hunt.
5. **No forbidden patterns**: no wrappers/toggles/copy-paste/adapters introduced.
6. **Abstraction count**: 0 new abstractions, 0 new files. Budget respected.
**Result**: no simplification blockers. All 5 criteria PASS. Recommend CLOSE.

## plan_2026-06-02_2a0b8192
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-02_2a0b8192/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-03
**Context**: The layer is already instruction-compliant on every HARD requirement; the only deltas vs the RBF template are a thinner module docstring and the absence of `# ---` class rules. The `.md` companion (87 lines) is a richer reference than the current 24-line module docstring.
**Decision**: Merge the `.md` content into a richer top-of-file module docstring, add the two RBF `# ---` rules, delete the `.md`, fix the CLAUDE.md pointer; make ZERO logic changes.
**Trade-off**: A single canonical doc location (the module docstring) + RBF structural parity **at the cost of** deleting the standalone browsable `.md` (surfaced as Assumption A1 — confirm at approval; user may want it kept).
**Reasoning**: "Merge into the docstring" implies consuming the source; keeping both duplicates content and rots. Behavior-preserving so the existing 207-line test is a sufficient safety net. Alternative (keep `.md` + duplicate in docstring) rejected as drift-prone.

### D-002 | EXPLORE → PLAN | 2026-06-03
**Context**: Explorer-1 flagged the bare `@keras.saving.register_keras_serializable()` (no `package=`) as a violation; cross-check against the instructions ([SOFT] §8.1) and the named template RBF (which omits it) reclassified it as a GHOST.
**Decision**: Keep the decorator BARE — no `package=` arg.
**Trade-off**: Instruction + template compliance and an unchanged serialization key **at the cost of** deviating from the broader-repo `package="dl_techniques"` habit seen elsewhere.
**Reasoning**: `package=` is optional per instructions; RBF (the explicit quality bar for this task) omits it; the layer is unused so there is nothing to gain, and adding it would change the registered key. LESSONS: bare decorator ties key to `__module__` — do not perturb without a move (and there is no move here).

### D-003 | EXPLORE → PLAN | 2026-06-03
**Context**: "Match RBF quality" is ambiguous — RBF uses Sphinx `:param:` docstrings, but the repo instructions mandate Google style.
**Decision**: Keep Google-style docstrings everywhere; interpret "match quality" as STRUCTURAL polish (module-docstring richness, `# ---` rules), not docstring-dialect parity.
**Trade-off**: Instruction compliance (Google `Args:`/`Input shape:`) **at the cost of** literal dialect-match with RBF's `:param:` style.
**Reasoning**: Converting to Sphinx would VIOLATE `research/2026_keras_custom_models_instructions.md` §3.1, which is the higher authority for "passes the instructions". The 3 existing `logger.debug` calls are likewise kept (instruction-permitted; removing is needless churn) even though RBF has no logger.

## plan_2026-06-02_e3da3ff9
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-02_e3da3ff9/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-02
**Context**: `neuro_grid.py` lives at `layers/` root, not exported anywhere, with two upward relative imports (`..regularizers`, `..initializers`). Target `layers/memory/` exists with SOM/MANN/NTM siblings, all exported from `memory/__init__.py`. User confirmed: (a) export NeuroGrid from memory pkg, (b) relocate ALL memory-package tests into a new `tests/test_layers/test_memory/` subdir.
**Decision**: git-mv the module + fix `..`→`...`, add NeuroGrid to `memory/__init__.py`, update the 2 importers, and git-mv all memory tests (4 files + `test_ntm/` dir) into a new `test_memory/` with an empty `__init__.py`.
**Trade-off**: Wider blast radius (test reorg) **at the cost of** a single atomic, convention-consistent move — vs. minimal "move neuro_grid only," which would leave memory tests scattered.
**Reasoning**: Use `git mv` to preserve history. Tests use absolute imports → no edits needed beyond neuro_grid's own. Exclude `test_hierarchical_memory_system.py` (tests `layers.experimental`, not memory pkg). Relative-import dot-count is the only correctness-critical edit; everything else mechanical.
**Anchor-Refs**: none (mechanical move, no non-obvious code-level constraint).

### D-002 | scope expansion | 2026-06-02
**Context**: After initial plan scoping, user added "move the other memory layers test into the test_memory subdir."
**Decision**: Expand test moves from {test_neuro_grid} to all memory-package tests: + test_som_2d_layer, test_som_nd_layer, test_som_nd_soft_layer, test_ntm/.
**Trade-off**: 4 extra git-mv ops **at the cost of** leaving the test tree half-migrated.
**Reasoning**: `test_hierarchical_memory_system.py` excluded (F8). No test_mann/factory tests exist.
**Anchor-Refs**: none.

### D-003 | REFLECT | 2026-06-02
**Context**: All 8 steps executed; 7/7 success criteria PASS; 189 memory-pkg tests pass; no regressions. Simplification Checks: no bloat (net = git-mv renames + ~14 line edits + 1 required test `__init__.py`); essential-not-accidental complexity; passes junior-dev readability. `validate-plan.mjs` reports 41 ERRORs but ALL are pre-existing orphan/unknown-plan decision anchors in unrelated files (cliffordnet, lighthouse_attention, gpt2, nam, rms_variants, lewm) from older sliding-window-trimmed plans — confirmed none of this plan's touched files appear in the error set.
**Decision**: Recommend CLOSE. Pre-existing orphan-anchor ERRORs are out of scope for a file-move task and must not be "fixed" here (would be massive scope creep; the correct remedy is `bootstrap.mjs retire <plan-id>` per-plan, separately).
**Trade-off**: Leaving 41 unrelated validator ERRORs **at the cost of** a clean validator run — accepted because addressing them is unrelated tech debt that predates this plan.
**Reasoning**: Scope drift = +1 file (layers/CLAUDE.md doc) justified by F5. Changelog WARNs are cosmetic (combined-step notation). No simplification blockers.
**Anchor-Refs**: none.
