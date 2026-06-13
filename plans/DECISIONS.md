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

## plan_2026-06-13_88695f5c
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-13_88695f5c/D-NNN` anchor exists in source)
-->

### D-001 | Deliverable is a report-only audit, decomposed by model family | 2026-06-13
**Context**: The goal is to classify ~94 inline `keras.layers.Layer` subclasses across `models/` as REPLACE / RELOCATE / KEEP against the `layers/` catalog. Two structural choices: (A) classify-and-report only, leaving all refactoring to a later user-directed plan; (B) classify and immediately refactor (replace/relocate) in the same plan. Decomposition choice: (i) one batch per coherent model family so an analyst keeps context, vs (ii) one batch per verdict-type or per-file flat sweep.
**Decision**: Report-only (A) + family-batched decomposition (i): 7 sequential steps, each source-classifies one coherent family into a per-batch findings file and appends a section to a single growing report; synthesis step fills summary tables + bug appendix + recommendations.
**Trade-off**: A reviewable, low-risk, source-grounded hypothesis document the user can act on selectively **at the cost of** delivering zero immediate code improvement and a positive net line count (a doc, not a refactor).
**Reasoning**: LESSONS.md is explicit that a reuse-review is a HYPOTHESIS to be source-verified, not an executable change set; conflating audit with refactor would risk shipping name-match false-positive REPLACEs as real edits (the exact trap the rubric probe caught: `_LearnedQueryPool1D`, `_ClassificationHeadBlock`). Family-batching keeps the analyst in one architectural context per step, minimizing cross-batch dependency (steps depend on each other only for file-append ordering). Rejected: (B) combined audit+refactor — too much blast radius, violates the user-confirmed report-only HARD constraint; (ii) verdict-type or flat-file batching — forces context-switching across unrelated architectures within a batch, raising miscount and false-positive risk.
**Anchor-Refs**: none (documentation deliverable; no in-source anchors).

### D-002 | REFLECT evaluation — recommend CLOSE | 2026-06-13
**Context**: All 7 steps executed; 95/95 inline layer classes classified across 6 family sections; synthesis tables + 11-bug appendix + 4-tier recommendations written. REFLECT ran 6 success criteria + plan hygiene.
**Decision**: Recommend CLOSE. 6/6 criteria PASS; no regressions; no scope drift; no simplification blockers.
**Trade-off**: Accepting the report as complete on a representative both-sides spot-check (not an exhaustive re-read of all 36 REPLACE+RELOCATE rows) **at the cost of** a small residual risk that a mid-confidence RELOCATE "absence" verdict could be wrong if an equivalent exists under an unsearched name — recorded in verification.md Not Verified.
**Reasoning**: The 47 validator ERRORs are 100% pre-existing orphan/unknown-plan DECISION anchors in src/ from OTHER plans (bdd100k_video, routing_probabilities, lighthouse_attention, etc.); this plan touched zero src/ files and added zero anchors, so it introduced zero ERRORs — consistent with the plan's verification strategy. Two name-match false positives that the source-read rule caught during execution (Gemma3 dual-post-norm not expressible via TransformerLayer; GroupAttention != group_query) confirm the rubric's anti-false-positive discipline held. WARNs (changelog approx-LOC format, classification findings using table layout instead of the EXPLORE findings template) are cosmetic and do not affect the deliverable.
**Devil's-advocate (one reason this could still be wrong)**: "Drop-in" REPLACE claims are structural, not runtime-proven — e.g. `TRMReasoningModule`→`HierarchicalReasoningModule` was verified to share the stack-of-TransformerLayers + input-injection shape, but exact param-name/serialization-key parity was not executed. Mitigation: every drop-in is a recommendation for a future implementation plan, not an applied edit; rows note known shims (e.g. `_LayerScale1D` checkpoint-shim for `Custom>_LayerScale1D` saves).
**Anchor-Refs**: none.

## plan_2026-06-13_5b933e7f
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-13_5b933e7f/D-NNN` anchor exists in source)
-->

### D-001 | Additive sibling-class pattern (Muon template) vs redesigned base class | 2026-06-13

**Context**: VSGD is a fourth custom optimizer in the optimization package. Two structural options exist: (A) create VSGD as a standalone `keras.optimizers.Optimizer` subclass parallel to Muon and SGLD, adding it to the existing `elif` chain in `optimizer_builder`; (B) refactor the package to introduce a shared base class or mixin for the three custom optimizers before adding VSGD. Option B is tempting since Muon, SGLD, and VSGD all share the pattern of manual weight decay and per-variable state, but no concrete second call site that would reuse a shared abstraction currently exists beyond the three classes themselves.

**Decision**: Use the additive sibling-class pattern (Option A): `vsgd_optimizer.py` mirrors `muon_optimizer.py` structurally, and VSGD is wired via an `elif` branch in `optimizer_builder`.

**Trade-off**: Zero regression risk and minimal diff surface **at the cost of** three classes that share structural boilerplate without a common abstraction (earned-abstraction rule not yet satisfied: would need >= 2 concrete call sites to justify a shared base).

**Reasoning**: LESSONS.md mandates "prefer additive `elif` branches over a dispatch-table rewrite when the dominant risk is regression." The earned-abstraction rule (`complexity-control.md`) forbids a new abstraction until >= 2 concrete call sites exist that need it; a shared optimizer base class would currently have zero callers beyond the three classes being aligned. Rejected: (A) shared `VariationalOptimizerBase` mixin -- single-use abstraction at this point, charged as a Complexity-Budget item with no payoff; (B) dispatch-table rewrite of `optimizer_builder` -- higher regression risk for no functional gain.

## plan_2026-06-13_250487cb
### D-001 | Approach: mechanical conformance pass, no redesign | 2026-06-13

**Context**: EXPLORE identified four Keras guide violations across three files (`eomt_transformer.py`, `free_transformer.py`, `progressive_focused_transformer.py`) and two missing export gaps in `__init__.py`. The violations are: (1) Python int step counter mutated in `call()`, (2) raw `tf.*` ops in a layer class, (3)+(4) sublayer creation deferred to `build()`. All are mechanical fixes — no algorithm changes, no API redesign, no new abstractions needed.

**Decision**: Fix each violation in-place with the minimum mechanical change: `add_weight` for the counter, `keras.ops` inline replacement for `tf.*`, attribute assignment move for sublayer deferral. Export the fixed classes. Do not redesign layer hierarchies.

**Trade-off**: Maximum behavior preservation **at the cost of** not consolidating any common patterns across the three affected files. A deeper refactor (e.g., a shared `TrainingStepMixin`) would be over-engineered for a single-use counter.

**Reasoning**: The goal is a conformance pass, explicitly behavior-preserving per LESSONS.md ("A guide-conformance pass is behavior-preserving. Any numeric delta from a conformance edit is a BUG — revert."). Single-use abstractions are excluded by the earned-abstraction rule (`complexity-control.md`). The three violations are independent; fixing them independently minimizes blast radius.

---

### D-002 | False positive: TextDecoder.get_config() finding | 2026-06-13

**Context**: The explorer agent's transformer-layer-files finding (F1) stated: "`TextDecoder.get_config()` omits `embedding_type`, `positional_type`, `attention_type`, `normalization_type`, `normalization_position`, `ffn_type` — deserialization reverts to defaults silently." This was listed as a HARD violation.

**Decision**: No fix applied. The finding is a false positive.

**Trade-off**: Trusting source verification **at the cost of** the risk of a real omission going unfixed. The risk is low because the source was read directly (lines 452-473) and all 6 params confirmed present in the config dict.

**Reasoning**: LESSONS.md mandates: "Explorer gap-map claims ('get_config drops param X') MUST be source-verified against the actual `__init__` signature." and "Adversarial explorer CRITICALs MUST be verified against the passing test suite BEFORE planning. False positives waste fix-work on non-bugs." Source-read of `text_decoder.py:452-473` confirmed all six params are present. No fix was planned.

---

### D-003 | `BinaryMapper.pow2`: inline computation vs. non-trainable weight | 2026-06-13

**Context**: `BinaryMapper` stores `self.pow2 = tf.constant([2**i ...])` in `__init__`. Two options: (a) move to `build()` as `add_weight(trainable=False)`, or (b) compute inline in `call()` as a local variable.

**Decision**: Compute inline in `call()` as a local `ops.array(...)`.

**Trade-off**: Zero state (no weight, no build step) **at the cost of** a trivial per-call list comprehension over `num_latent_bits` elements (typically small, e.g., 8-16).

**Reasoning**: `pow2` is a pure function of `num_latent_bits` (a constructor param). It has no learned value, no need for persistence, and no need for serialization. `add_weight` is semantically wrong for a constant; it adds unnecessary state. Inline computation is the simplest correct solution. The overhead is negligible for typical `num_latent_bits` values.

---

### D-004 | `sd3_adaln.py` and `ideogram4_block.py`: keep direct-import-only | 2026-06-13

**Context**: Factory-wiring finding (F3) flagged `sd3_adaln.py` and `ideogram4_block.py` as absent from `__init__.py`. SYSTEM.md confirms this is intentional ("Direct-import only").

**Decision**: Do not add these files to `__init__.py`.

**Trade-off**: Clean package boundary **at the cost of** slightly reduced discoverability for SD3/Ideogram4 users.

**Reasoning**: SYSTEM.md is an authoritative structural prior. The absence is documented and intentional. Adding these exports would contradict the system atlas without new information justifying the change.

## plan_2026-06-13_28f0b453
### D-001 | EXPLORE -> PLAN | YYYY-MM-DD
**Context**: <one-paragraph background -- what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-13_28f0b453/D-NNN` anchor exists in source)
-->

### D-001 | Full refactor (delete DetrDecoderLayer, use TransformerDecoderLayer) vs minimal patch | 2026-06-13
**Context**: `DetrDecoderLayer` reimplements ~150 lines of cross-attention logic that `TransformerDecoderLayer` already provides correctly. Two fix options exist: (A) minimal patch -- fix `attention_type` string only, keep `DetrDecoderLayer`; (B) full refactor -- delete `DetrDecoderLayer` entirely and replace with `TransformerDecoderLayer(use_causal_mask=False)`. Option A leaves 5 bugs unresolved (BUG-3, BUG-4, BUG-5, BUG-6, BUG-8) and keeps a bespoke class that bypasses the attention factory, violating repo conventions.
**Decision**: Full refactor (Option B) -- delete `DetrDecoderLayer` and use `TransformerDecoderLayer` as the canonical decoder block.
**Trade-off**: Eliminating ~150 lines of bespoke decoder code and all `DetrDecoderLayer` bugs **at the cost of** a larger diff in a single step, requiring careful decoder call-site rewrite in `DetrTransformer.call()`.
**Reasoning**: Option B is strictly better: fewer lines, reuses the canonical factory-driven component, resolves all decoder-related bugs in one change. The positional encoding injection pattern (`tgt + query_embed_expanded` before each layer call) is already how the encoder handles pos_embed and is supported by `TransformerDecoderLayer`'s design. The complexity cost of the larger diff is offset by the elimination of bespoke code.

### D-002 | Attention masking deferred (pass None) vs fix mask shape | 2026-06-13
**Context**: The padding mask is `(B, H*W)` (flat boolean). `TransformerLayer` expects a broadcastable attention mask, typically `(B, 1, T, T)` or `(B, T, T)`. Expanding and tiling the flat mask to the correct attention mask shape requires non-trivial reshape/broadcast logic that is not part of the DETR paper's primary mechanism. This is BUG-12 from findings.
**Decision**: Pass `mask_flat=None` to `DetrTransformer` in the initial fix, disabling attention masking entirely. Document as a known limitation.
**Trade-off**: Unblocking construction and forward pass correctness **at the cost of** not masking padded positions in the encoder attention, which slightly degrades detection on heavily-padded images.
**Reasoning**: Masking is optional in `TransformerLayer`; removing it avoids a silent incorrect masking (a flat boolean mask used as an attention mask produces garbage attention weights). Proper mask support requires a follow-up plan to define the correct attention mask shape and expand logic. The initial fix prioritises correctness of what IS implemented over completeness.
