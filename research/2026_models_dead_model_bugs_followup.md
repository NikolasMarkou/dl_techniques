# Follow-up: Dead-on-forward bugs in untested model files
*Date: 2026-06-14. Origin: plan `plan_2026-06-13_ae9ee2cd` (bug-fix + cleanup implementation of `2026_models_layer_reuse_audit.md`).*

While implementing the audit's bug appendix, re-verifying each finding against current source surfaced a cluster of **pre-existing, dead-on-forward bugs** the static audit never caught — because it never executed the models. Three model files (`darkir`, `modern_bert_blt_hrm`, `nano_vlm_world_model`) could not complete a forward pass. `darkir/FreMLP` was fixed in that plan; the rest are captured here for a dedicated follow-up. None of these models have tests, which is why the bugs survived.

## Scope note
The implementation plan had a HARD constraint: **no edits under `src/dl_techniques/layers/`**. Two of the deferred items require exactly such edits, which is why they were deferred rather than fixed.

---

## 1. modern_bert_blt_hrm — `ReasoningByteBERT` dead on forward (two bugs)

**Status:** B3/B4/B5 (audit appendix #3/#4/#5) are authored and **known-good** (py_compile clean; `HashNGramEmbedding` standalone forward passes; the HRM-convention fix makes the model build and forward ~200 lines deep). Reverted, NOT committed, because a forward smoke / `.keras` round-trip (the success criterion) cannot pass until the two bugs below are fixed.

### 1a. `ReasoningByteCore` ↔ `HierarchicalReasoningModule` call/build convention (models/, FIXABLE — fix is known-good)
- File: `src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py`, class `ReasoningByteCore`.
- `HierarchicalReasoningModule.build` asserts `isinstance(input_shape, list) and len==2`; `.call` asserts `inputs` is a 2-element list `[hidden_states, input_injection]`. (Ground truth: `layers/reasoning/hrm_reasoning_core.py:393-394,528-540`.)
- Current (wrong): build (~685-686) passes a bare shape tuple; 4 call sites (~778/782/785/786) pass two positional tensors, e.g. `self.l_reasoning(z_l, z_h + patch_representations, training=...)`.
- Fix (6 sites, semantics-preserving — HRM adds the injection internally):
  - build: `.build([reasoning_input_shape, reasoning_input_shape])` (same shape twice) for both `h_reasoning` and `l_reasoning`.
  - call: wrap the two positional args in a list, e.g. `self.l_reasoning([z_l, z_h + patch_representations], training=...)`, `self.h_reasoning([z_h, z_l], training=...)`. Keep `training=`.

### 1b. `LocalEncoder` / `PatchPooling` shape contract (layers/ — REQUIRES lifting the no-layers constraint)
- File: `src/dl_techniques/layers/blt_blocks.py:861` (`PatchPooling._attention_pooling`; `_average_pooling` likely shares the flaw).
- Symptom: `InvalidArgumentError: required broadcastable shapes` in `ops.where(mask_expanded, byte_hiddens, ...)` — `byte_hiddens=(B, max_patches, D)` but `patch_ids=(B, seq_len)`.
- Root cause: `LocalEncoder` returns a tensor **already pooled to `max_patches` length**, but `PatchPooling` assumes its input is the **un-pooled per-byte** sequence and re-masks it with per-byte `patch_ids` (length `seq_len`). The two lengths can never broadcast.
- Reproduced on pristine layer code (independent of B3/B4/B5). Affects every BLT caller of this path. Fixing it is its own mini-investigation (which side owns pooling?) and touches a shared, untested `layers/` file with 4+ BLT callers (`blt_core`, `byte_latent_transformer`, `modern_bert_blt_hrm`, `train_blt`) — impact-assess before editing.

### Recommended follow-up sequence
1. Fix 1b (`blt_blocks.py` pooling contract) with a forward smoke for at least one BLT caller.
2. Apply 1a (HRM convention, known-good).
3. Re-apply the B3/B4/B5 diff (vectorize ngram via `ops.take`; delete `if self.built: return` guards; sublayers `build`→`__init__` in `ReasoningByteEmbeddings`/`ReasoningByteCore`/`ReasoningByteBERT`, preserving order/names).
4. Verify: `HashNGramEmbedding` standalone forward; `ReasoningByteBERT` instantiate + forward + `.keras` round-trip. Consider adding the first test for this model.

---

## 2. nano_vlm_world_model — full-model forward dead (one bug remaining)

**Status:** import-path + scheduler-dtype bugs FIXED and committed (`1d15636e`); B7 (scheduler→plain class) committed (`29367bc3`) and verified at the scheduler level. Full-model forward / `.keras` round-trip still blocked by the bug below.

- File: `src/dl_techniques/models/nano_vlm_world_model/model.py` (+ `create_vision_encoder` defaults).
- Symptom: `InvalidArgumentError: Ranks of all input tensors should match: [2,8,384] vs [2,384]` at `ConditionalDenoiser.call` (`denoisers.py:221`, `ops.concatenate([x, c], axis=1)`).
- Root cause: `create_vision_encoder` defaults to `output_mode='cls'` → returns rank-2 pooled `[B, embed_dim]`; the model's `vision_config` never overrides it (unlike `text_config`, which yields rank-3). `ConditionalDenoiser` expects a rank-3 condition `[B, cond_seq_len, dim]`.
- Fix options: set `output_mode` to a sequence-returning mode in the model's `vision_config`, OR make `ConditionalDenoiser` handle a rank-2 condition. >10 lines, untested; pick based on intended world-model semantics.
- After fixing: run the full B7 oracle (`create_score_based_nanovlm(variant='mini', mode='image_to_text', vocab_size=32000)` → forward → `.keras` round-trip → assert `DiffusionScheduler` not in `model.layers` → `.step(t>0)` finite).

---

## 3. Repo-wide: 31 pre-existing orphan `# DECISION` anchors (housekeeping)

`validate-plan.mjs` reports 31 orphan / unknown-plan anchor ERRORs across the repo (e.g. `layers/activations/routing_probabilities.py`, `layers/attention/lighthouse_attention.py`, `layers/geometric/clifford_block.py`, `losses/*`, `models/cliffordnet/lmunet.py`, `models/depth_anything/model.py`, `models/lewm/*`, many `src/train/*`). These are `# DECISION plan_xxxx/D-NNN` anchors from old plans whose `decisions.md` sections were sliding-window-trimmed — none introduced by this plan; none in files this plan touched.

- Remedy per the iterative-planner protocol: `node <skill>/scripts/bootstrap.mjs retire <plan-id>` for each obsolete plan-id (marks anchors `[STALE]`, downgrades ERROR→WARN), OR remove the stale anchors from source.
- This is cross-plan debt; address in a dedicated housekeeping pass, not mixed into feature work.

---

## Meta-lesson
For any **untested** model in this repo, a static audit is necessary but far from sufficient: a forward + `.keras` round-trip smoke at the smallest config surfaces dead-on-forward bugs (wrong ops API, wrong call conventions, broken imports, dtype promotion, shape-contract mismatches) that no static read will catch. Several models here have multi-bug chains spanning `models/` and `layers/` and have clearly never been executed. Prioritize adding a minimal smoke test per model over trusting static analysis.
