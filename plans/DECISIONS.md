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

## plan_2026-06-15_e6a0391c
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-15_e6a0391c/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-15
**Context**: 70 model packages, no existing "run all" harness, `models/__init__.py` empty so no blanket import surfaces errors. Each model has a distinct minimal-build recipe and forward-input contract (catalogued in F1). Several are known-dead, several need special construction.
**Decision**: Build a single data-driven smoke harness (`scripts/verify_models_smoke.py`) holding a per-package registry of build/input recipes with RUN/XFAIL/SKIP modes; use it as the verification instrument, then fix genuine Tier-1 forward blockers surgically.
**Trade-off**: A central hand-maintained registry (must transcribe 70 recipes accurately) **at the cost of** not relying on the heterogeneous, incomplete per-model pytest suites (which lack a uniform forward contract and a single entry point).
**Reasoning**: Alternatives rejected — (1) running the full pytest suite: 1.5h, doesn't isolate build+forward, no single matrix; (2) blanket `import models` + reflection: impossible, `__init__` is empty and constructors need varied args. A registry is the only way to get a uniform PASS/FAIL/XFAIL matrix across all 70. Recipe errors are expected and corrected in Step 2 triage (distinguished from real model bugs by traceback frame origin).
**Anchor-Refs**: none (harness is new instrument code, no in-source DECISION anchor needed)

### D-002 | PLAN | 2026-06-15
**Context**: User directive: "WORK AUTONOMOUSLY, DONT ASK ME QUESTIONS, MAXIMUM EFFORT, MAXIMUM 3 ITERATIONS, LETS GO".
**Decision**: Treat the directive as durable pre-authorization: present PC-PLAN and PC-REFLECT contracts for transparency but proceed through EXECUTE without blocking for approval; cap at 3 iterations.
**Trade-off**: Autonomous momentum **at the cost of** the protocol's normal user-approval gates at PLAN→EXECUTE and REFLECT→CLOSE.
**Reasoning**: The user explicitly waived questioning. Contracts are still emitted as a visible audit trail. Autonomy leash, 10-line rule, and 3-strike still apply internally to bound risk.

### D-003 | EXECUTE step 3 | 2026-06-15
**Context**: convnext_v1/v2 produced non-finite (NaN) output at 32x32/16x16 (sizes their OWN test suite uses). Root cause: kernel==stride==4 downsample with `padding="valid"` collapses spatial map to 0x0 (32→8→2→0), then GlobalAveragePooling over empty tensor = NaN.
**Decision**: Change the downsample Conv2D `padding="valid"` → `"same"` in both convnext files.
**Trade-off**: Robustness to small inputs **at the cost of** a slightly different intermediate spatial size at non-stride-divisible inputs (e.g. 224: stage dims 3→4); final output shape and all weight shapes unchanged.
**Reasoning**: kernel==stride means valid==same when divisible — the only behavioral change is preventing 0-collapse, which was already broken. Chose model-fix over harness-larger-input because convnext's variants + tests advertise 32/16 support. Squeezenet (also non-finite) was instead fixed harness-side (its tests use 64px; pool_size=3≠stride=2 would change its published architecture).
**Anchor-Refs**: `src/dl_techniques/models/convnext/convnext_v1.py:~276`, `src/dl_techniques/models/convnext/convnext_v2.py:~292`

### D-004 | EXECUTE step 3 | 2026-06-15
**Context**: SAM forward crashed with `Cannot convert 1.0 to EagerTensor of dtype int32` in `WindowedAttentionWithRelPos._get_rel_pos` — `ops.arange(q_size)` (int32) multiplied by a Python float ratio.
**Decision**: Cast the arange coords to float32 before the float multiply; use `float(k_size-1)` for the scalar term.
**Trade-off**: One explicit cast **at the cost of** nothing — q_size/k_size are Python ints (typed `int`), so the cast is unconditionally safe.
**Reasoning**: Minimal 3-line dtype fix; coords are re-cast to int32 at the gather (line ~397) so downstream is unchanged.
**Anchor-Refs**: `src/dl_techniques/models/sam/image_encoder.py:~395`

### D-005 | EXECUTE step 3 | 2026-06-15
**Context**: `ReasoningByteEmbeddings.call` (modern_bert_blt_hrm) added hash n-gram embeddings padded by `puzzle_emb_len` whenever `puzzle_emb_len>0`, but puzzle tokens are prepended to `embeddings` only when `puzzle_ids is not None`. With default config (puzzle_emb_len=1) and no puzzle_ids, hash_embeds had seq N+1 vs embeddings N → AddV2 shape error.
**Decision**: Gate the hash padding on `puzzle_emb_len > 0 and puzzle_ids is not None` (mirror the prepend condition).
**Trade-off**: Correct no-puzzle path **at the cost of** nothing — when puzzle_ids is supplied, behavior is unchanged.
**Reasoning**: Matches the exact condition that prepends puzzle tokens (line ~461); minimal one-clause change.
**Anchor-Refs**: `src/dl_techniques/models/modern_bert/modern_bert_blt_hrm.py:~483`

## plan_2026-06-15_e2759fbc
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-15_e2759fbc/D-NNN` anchor exists in source)
-->

### D-000 | PLAN | 2026-06-15
**Context**: dino v2 (`dino_v2.py`) is the last open dino backlog row — a 7-bug chain (Dense-in-Lambda x3, projection-hack CLS, wrong pos-embed attr, ops.cond dict-vs-tensor, wrapper-cond, nested-Lambda+symbolic-assert) blocking construction/forward. v2 has never run end-to-end. The plan-level decision is the DECOMPOSITION and the OUTPUT-CONTRACT resolution; per-bug anchored decisions (D-001..D-006) are logged below as they are applied at EXECUTE.
**Decision**: Fix all 7 in one file via a bottom-up sub-area decomposition (token-prep construction first → output contract → test → aggregate), reuse `ClassTokenPrepend` for the CLS, and resolve the `ops.cond` dict-vs-tensor mismatch with OPTION A (backbone always returns the 5-key dict; the `DINOv2` wrapper slices `["x_norm_clstoken"]`).
**Trade-off**: A fixed, always-dict backbone output contract + fixed-resolution pos-embed **at the cost of** the (spurious) runtime `is_training` branch AND variable-resolution pos-embed interpolation (both dropped as out-of-scope; var-res would need a proper interpolation layer).
**Reasoning**: The model is Functional with a fixed `is_training` Input, so the runtime branch never earned its keep; an always-dict contract is the canonical "output structure depends only on construction-time config" resolution (cf. plan_2026-06-15_32b5822c D-001). Reusing `ClassTokenPrepend` (built+tested in plan_2026-06-15_39a31d4a/2a23a001) avoids a single-use abstraction (earned-abstraction rule). The 7 bugs share one file and one construction path, so a same-file sub-area split (not a multi-file decomposition) minimizes cross-step deps; B5 gates B6, hence the bottom-up order. Rejected: keeping the runtime branch with matched-structure (re-introduces the training-dependent output-structure anti-pattern); a new interpolation layer (out of scope, single-use).
**Anchor-Refs**: (per-bug anchors D-001..D-006 in `src/dl_techniques/models/dino/dino_v2.py`, to be placed at EXECUTE)

<!-- D-001..D-006 are placed in source at EXECUTE; mirror the schema above. Planned mapping:
  D-001 | B1 mask-token Dense hoist (no weight-creating layer inside a Lambda; hoist to __init__)
  D-002 | B2 CLS via ClassTokenPrepend (not a Dense-on-ones hack)
  D-003 | B3 register-token Dense hoist + Concatenate (not Lambda)
  D-004 | B4/B7 flatten pos-embed: x + self.pos_embed(x); no nested Lambda, no symbolic-tensor assert; correct attr .pos_embedding; var-res out of scope
  D-005 | B5 always-return-dict (ops.cond branches must match structure; runtime branch spurious in Functional model)
  D-006 | B6 wrapper slices backbone_output["x_norm_clstoken"] (no Lambda/cond on the now-always-dict output)
-->

### D-001 | EXECUTE iter-1/step-1 | 2026-06-15
**Context**: B1 — `dino_v2.py` instantiated `layers.Dense(name='mask_token_projection')` INSIDE the `apply_masks` Lambda body, a weight-creating layer created during functional-graph trace ⇒ untracked/uncreatable weights.
**Decision**: Hoist the mask-token Dense to `self.mask_token_projection` in `__init__`; compute `mask_tokens` OUTSIDE the Lambda and pass it in as a 3rd Lambda input; the Lambda body is a pure `keras.ops.where`.
**Trade-off**: One extra graph edge (mask_tokens precomputed) **at the cost of** nothing material.
**Reasoning**: The hoisted Dense is a plain `self.x =` assignment (NOT `add_weight`), safe pre-super (lazy build). CASCADE FIX (#8, in-scope): the original used `keras.ops.ones((batch_size, ...))` with `batch_size = ops.shape(x)[0]` — symbolic `None` at trace time; safe inside a Lambda (lazy) but FAILS an eager `ops.ones` once hoisted. Fixed by building mask_tokens with batch dim **1** and broadcasting inside `keras.ops.where` (cond `(B,N,1)`, x `(1,N,D)`, y `(B,N,D)` → `(B,N,D)`). Do NOT reintroduce a Dense in the Lambda or a symbolic batch in `ops.ones`.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:538`, `src/dl_techniques/models/dino/dino_v2.py:645`

### D-002 | EXECUTE iter-1/step-1 | 2026-06-15
**Context**: B2 — CLS token was a `Dense(name='cls_token_projection')` on a `ones((B,1,1))` tensor (a projection hack, not a real learnable CLS, and the same Dense-in-trace footgun).
**Decision**: Replace with `ClassTokenPrepend` (reused; built in plan_2026-06-15_39a31d4a/D-001): `self.cls_token_layer = ClassTokenPrepend(name="cls_token")` in `__init__`; `x = self.cls_token_layer(x)` after masking, before pos-embed. Delete the old cls Dense + `Concatenate(add_cls_token)`.
**Trade-off**: Reuse a canonical layer **at the cost of** zero (DRY win; v1/v3 already use it).
**Reasoning**: `ClassTokenPrepend` owns its `(1,1,dim)` weight in `build()` (lazy), safe to assign pre-super; `(B,N,D)→(B,N+1,D)` matches the pos-embed sizing (`num_patches+1`). Do NOT use a Dense-on-ones hack.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:549`, `src/dl_techniques/models/dino/dino_v2.py:661`

### D-003 | EXECUTE iter-1/step-1 | 2026-06-15
**Context**: B3 — register-token `Dense(name='register_token_projection')` was created inside the build path + inserted via a `Lambda(insert_register_tokens)`.
**Decision**: Hoist the projection to `self.register_token_projection` in `__init__` (guarded `if num_register_tokens > 0`); compute `reg_tokens` outside any Lambda; insert via `Concatenate([cls, reg, rest])`.
**Trade-off**: A pure-op broadcast Lambda for the batch dim **at the cost of** nothing (cold path; `num_register_tokens=0` on the smoke).
**Reasoning**: Same no-weight-in-Lambda rule as D-001. CASCADE FIX (#8 again): reg_tokens built on `ones((1,R,1))` then broadcast to the runtime batch via a pure-op (no-weight) Lambda so `Concatenate` sees a matching batch dim. Do NOT use an in-Lambda Dense or a symbolic-batch `ops.ones`.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:555`, `src/dl_techniques/models/dino/dino_v2.py:686`

### D-004 | EXECUTE iter-1/step-1 | 2026-06-15
**Context**: B4/B7 — `add_pos_embed` Lambda called `_get_interpolated_pos_embed`, which (a) created ANOTHER Lambda inside (illegal layer-creation-in-trace), (b) read the WRONG attribute `self.pos_embed.pos_embed` (the real attr is `.pos_embedding`, verified in `positional_embedding.py:202-243`), and (c) ran a Python `assert N==M*M` on a symbolic tensor (a trace-time no-op).
**Decision**: Flatten to `x = self.pos_embed(x)` (the `PositionalEmbedding.call` slices its `(1,max_seq_len,dim)` weight to the seq length). Delete `_get_interpolated_pos_embed` and both nested Lambdas; drop the symbolic assert. Variable-resolution interpolation is OUT OF SCOPE (would need a proper interpolation layer).
**Trade-off**: Fixed-resolution pos-embed **at the cost of** variable-res interpolation (dropped, out of scope).
**Reasoning**: pos-embed is sized `num_patches + num_tokens` (CLS-inclusive), so `self.pos_embed(x)` on the `(B,N+1,D)` post-CLS tensor is length-consistent (A3 confirmed; NO 8th size bug). Do NOT reintroduce the interpolation method, a nested Lambda, the `.pos_embed` attr, or a symbolic assert.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:676`

### D-CASCADE-ATTN | EXECUTE iter-1/step-1 | 2026-06-15
**Context**: CASCADE bug #9 (in-scope, distinct from the 7) surfaced at backbone construction: `DINOv2Block`/`DINOv2VisionTransformer` defaulted `attention_type='multi_head_attention'`, but the attention factory has no such key — the correct registered key is `'multi_head'` (v1 already uses `'multi_head'`).
**Decision**: Change both `attention_type` defaults to `'multi_head'`; fix the DINOv2Block param-mapping branch to key on `'multi_head'` and pass `dim` + `use_bias` (the `multi_head` factory requires `dim`, accepts `use_bias`).
**Trade-off**: Mirror v1's working key **at the cost of** none.
**Reasoning**: Mirrors dino_v1.py:459 (`attention_type="multi_head"`). This is the 2nd extra cascade bug beyond the 7 (#8 batch-None ones in D-001/D-003, #9 here) — at the F-CASCADE-STOP budget ceiling (>2 would trigger xfail-revert) but NOT over it. Both are mechanical, in-scope, single-file.
**Anchor-Refs**: (no in-source anchor — mechanical default/key fix, not a non-obvious approach choice)

### D-005 | EXECUTE iter-1/step-2 | 2026-06-15
**Context**: B5 — `_build_final_processing.split_outputs` returned a 5-key dict in the `training_output` branch but a bare CLS tensor in `inference_output`, joined by `keras.ops.cond` — mismatched nested structure AND a training-dependent output structure. Step-1's probe error at this Lambda was `NotImplementedError: could not infer the shape of the Lambda's output`.
**Decision**: OPTION A — ALWAYS return the 5-key dict; delete the `keras.ops.cond`/`training_output`/`inference_output` branch. To resolve the shape-inference failure, produce each output key with its OWN per-key `layers.Lambda` returning a single tensor (`slice_cls_token`, `slice_reg_tokens`, `slice_patch_tokens`), then assemble a Python `dict` of those KerasTensors — instead of one dict-returning Lambda (which Keras cannot always shape-infer). Empty register-token case uses a zero-width slice `t[:, 0:0]` (shape-inferable KerasTensor) replacing the old `tf.zeros((shape[0],0,D))`.
**Trade-off**: A few extra named slice Lambdas in the graph **at the cost of** nothing material; gains per-key shape inference and a fixed (training-independent) output contract.
**Reasoning**: Functional model with a fixed `is_training` Input ⇒ the runtime branch was spurious. Per-key single-tensor Lambdas are individually shape-inferable, sidestepping the dict-Lambda inference gap. `is_training` Input stays wired (3-input contract) but unused. Do NOT reintroduce ops.cond, a bare-tensor inference branch, a dict-returning single Lambda, or `tf.zeros` for empty regtokens.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:756`

### D-006 | EXECUTE iter-1/step-2 | 2026-06-15
**Context**: B6 — the `DINOv2._build_model` wrapper ran `extract_cls_features` as a `layers.Lambda` doing `keras.ops.cond(is_training, lambda: out["x_norm_clstoken"], lambda: out)` — a mismatched-structure cond whose inference branch returned the whole dict, not the CLS tensor.
**Decision**: Delete `extract_cls_features` + its Lambda; slice `features = backbone_output["x_norm_clstoken"]` directly on the (now-always-dict, per D-005) functional output. Dict subscription on a Functional output is legal and shape-inferable.
**Trade-off**: Direct dict-slice **at the cost of** nothing (removes dead branch).
**Reasoning**: With D-005 the backbone always emits the 5-key dict, so the cond is both wrong and unnecessary. Do NOT wrap the slice in a Lambda or cond.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:1032`

### D-CASCADE-BACKBONE-NAME | EXECUTE iter-1/step-2 | 2026-06-15 | OVER-BUDGET — NOT FIXED
**Context**: CASCADE bug #10 (genuinely new, independent of B5/B6) surfaced when the FULL wrapper forward was first exercised (`create_dino_v2` → `DINOv2.from_variant` → `_build_model`). `DINOv2VisionTransformer.__init__` HARDCODES `name=f'dinov2_vit_{embed_dim}d_{depth}l'` in its `super().__init__(...)`, while the wrapper `_build_model` constructs the backbone with `name='dinov2_backbone'` (lands in `**kwargs`). Result: `TypeError: Model.__init__() got multiple values for keyword argument 'name'`. This path had NEVER been exercised (step-1 probed the backbone directly, no `name=`), so #10 was masked until now. It is independent of B5/B6 (backbone-only construction still succeeds post-B5: BACKBONE_OK).
**Decision**: STOP — do NOT fix. This is the 3rd EXTRA cascade bug beyond the 7 (after #8 batch-None ones, #9 attention_type key), which is `>2 EXTRA` ⇒ over the F-CASCADE-STOP budget. Per orchestrator instruction and plan Pre-Mortem, surface and let the orchestrator decide (likely: a trivial 1-line fix — drop `name='dinov2_backbone'` from the wrapper call OR let the backbone honor a passed `name` — then xfail-or-graduate decision). Trivial fix is `out[:, ...]`-style mechanical, but the budget gate is the orchestrator's call, not the executor's.
**Trade-off**: Surface + defer the decision **at the cost of** not landing FORWARD_OK in this step.
**Reasoning**: Budget rule is explicit (`>2 EXTRA → xfail-revert + surface`); executor must not unilaterally spend over-budget. B5/B6 edits are correct and left in the working tree (uncommitted); backbone constructs; only #10 blocks the full forward.
**Anchor-Refs**: (no in-source anchor — surfaced, not fixed; fix location is the wrapper `name='dinov2_backbone'` at `_build_model` and/or the backbone `super().__init__` hardcoded name)

### D-007 | EXECUTE step-2 | 2026-06-15
**Title**: CASCADE BUDGET OVERRIDE
**Context**: Step 1 surfaced 2 cascade bugs beyond the planned 7 (#8 batch-None ones broadcast — a direct consequence of the B1/B3 Dense-hoist; #9 wrong attention_type factory key — pre-existing independent). Step 2 applied B5/B6 cleanly (backbone verified) but the first-ever wrapper→backbone forward exposed #10 (backbone hardcodes `name=` while the wrapper also passes `name='dinov2_backbone'` → duplicate-kwarg TypeError). #10 is the 3rd "extra" bug, tripping the plan's F-CASCADE-STOP (">2 extra → xfail/graduate").
**Decision**: OVERRIDE the STOP-IF and authorize the 1-line #10 fix to complete step 2, rather than xfail/graduate.
**Trade-off**: Finishing v2 in this plan (MAXIMUM EFFORT directive) **at the cost of** exceeding the pre-declared cascade budget — accepted because the override is explicit + logged here, not silent scope creep.
**Reasoning**: (1) Progress is strictly monotonic — each bug is a distinct area (mask→CLS→pos→blocks→output→backbone-name); the 3-strike "same area 3×" thrash signal has NOT fired. (2) Every fix is mechanical/single-site, none architectural. (3) #8 is arguably a B1/B3 consequence, so truly-independent extras = 2 (#9,#10) = AT budget under a strict reading. (4) F-CONSUMER confirmed: no external consumer of the backbone/x_norm_clstoken, so B5 always-dict is safe. **HARD GUARD**: if the post-#10 forward surfaces ANY further independent bug, STOP and present to the user — do not continue overriding.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:577-583` (backbone `super().__init__` name dedup — `name = kwargs.pop('name', f'dinov2_vit_...')`).

### D-008 | EXECUTE step-2 | 2026-06-15
**Title**: USER-AUTHORIZED UNBOUNDED CASCADE
**Context**: Budget hard-stop (D-007) escalated #11 (wrapper input_masks graph cycle) to the user. 10 bugs already fixed; backbone works, only the DINOv2 classifier wrapper remains broken (#11 + likely more).
**Decision**: User selected "Push through (MAXIMUM EFFORT)". Authorize fixing the wrapper cascade (#11 and any subsequent mechanical wiring bugs) until DINOv2 forwards (2,10), re-probing after each fix.
**Trade-off**: Completing v2 fully **at the cost of** an unbounded cascade count — accepted by explicit user directive.
**Reasoning**: User is engaged and chose this with full knowledge of the cascade pattern. NEW GUARD (replaces D-007): continue fixing distinct, mechanical sequential bugs; STOP and report ONLY IF (a) a fix requires a genuine architectural redesign (not a mechanical wiring fix), OR (b) the 3-strike rule fires (same area/bug breaks 3× = thrash). Distinct sequential mechanical bugs are now in-scope.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:1006` (#11 wrapper Input unique names), `src/dl_techniques/models/dino/dino_v2.py:797` (#12 masks identity-Lambda passthrough — break input-is-output cycle), `src/dl_techniques/models/dino/dino_v2.py:1033` (#13 DINOv2.call rank-0 is_training coercion).

### Wrapper cascade fixed this session (step-2 continuation, D-008 push-through):
- **#11** wrapper `keras.Input` names (`input_images/input_masks/is_training`) collided with the nested backbone's identically-named inputs. Fix: prefix wrapper inputs with `dinov2_` (`dinov2_input_images/dinov2_input_masks/dinov2_is_training`). (Hardening — isolated but did not by itself break the cycle.)
- **#12** the backbone's output dict echoed the RAW `masks` input KerasTensor as its `"masks"` output → input-is-output aliasing → `_build_map` "Tensor input_masks ... is part of a cycle" when the wrapper nests the backbone. Fix: route through `layers.Lambda(lambda t: t, name='masks_passthrough')(masks)` so the echoed mask is a distinct node. (This was the actual cycle root cause.)
- **#13** wrapper's `dinov2_is_training` is a `keras.Input(shape=())` → Functional graph expects rank-1 `(batch,)` and rejects the conventional rank-0 `np.array(False)` ("Expected shape (None,), but input has incompatible shape ()"). Fix: `DINOv2.call` override reshapes a rank-0 `is_training` to `(-1,)` before delegating to `super().call`; value is spurious/unused (always-dict, D-005). Probe now yields `FORWARD_OK (2,10) finite True`.

### D-009 | REFLECT → EXECUTE (continuation) | 2026-06-15
**Title**: USER OPTION-3 — fix all 3 review findings (MAXIMUM EFFORT)
**Context**: Adversarial review (findings/review-iter-1.md) returned NEEDS_WORK: CRITICAL register-token path (untested; pos-embed sized N+1 while seq becomes 1+R+N), WARNING degenerate mask token (zero-init Dense-on-ones = constant-zero, not a learnable token — the hack D-002 rejected for CLS), WARNING dead is_training input + fragile DINOv2.call override (#13). v2 forward IS verified real for tiny/0-register/inference (input+mask-sensitive, sane logits).
**Decision**: User chose Option 3 — fix all three before closing. Extension steps 5-8 (iter-1 continuation, REFLECT→EXECUTE, iter stays 1): S5 learnable mask token (add_weight) + remove dead is_training input & call() override (→2-input contract); S6 register-token path — EMPIRICALLY probe a register-enabled variant, fix ONLY a genuine crash/shape bug (registers are position-free BY DESIGN in DINOv2-w-registers — do NOT blindly add pos-embed to them), add register-variant smoke; S7 mixed-mask test + .keras round-trip (xfail/skip with reason if DINOHead round-trip is the known-broken blocker); S8 aggregate+regression+diff+report.
**Trade-off**: Full correctness + hardening **at the cost of** more cascade risk + scope — accepted by explicit user MAXIMUM EFFORT directive.
**Reasoning**: Goal "forward inference" already met for tiny; this raises it to large/giant correctness + non-degenerate masking + serialization robustness. GUARD: per-fix 2-attempt leash; if any fix needs architectural redesign, STOP and report. The register "CRITICAL" must be verified empirically (reviewer may have mis-assumed registers need pos-embed).
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v2.py:542` (S5A learnable MaskTokenApply weight, was zero Dense-on-ones), `src/dl_techniques/models/dino/dino_v2.py:657` (S5A mask-token wiring in token-prep), `src/dl_techniques/models/dino/dino_v2.py:574` (S5B backbone 2-input contract — is_training Input removed), `src/dl_techniques/models/dino/dino_v2.py:1009` (S5B wrapper 2-input contract — is_training Input + DINOv2.call override removed). New layer: `src/dl_techniques/layers/embedding/mask_token.py` (MaskTokenApply). [S6 register path / S7-S8 anchors to be appended at those steps.]
**Anchor-Refs (S6 register path, appended EXECUTE step-6)**: `src/dl_techniques/models/dino/dino_v2.py:696` (register tokens are position-free BY DESIGN — Darcet 2023; pos_embed sized N+1 applied pre-concat, registers concatenated after with no positional signal). S6 finding: the reviewer's CRITICAL register-token claim was a DESIGN-MISREAD, not a bug — a 'tiny'+4-register model forwards finite (2,10) and is input-sensitive with NO src fix. Added `tests/test_models/test_dino/test_dino_v2.py::test_register_tokens_forward`. NO pos-embed enlargement / no insertion-reorder applied (both would WRONGLY position the registers).

## plan_2026-06-15_2a23a001
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-15_2a23a001/D-NNN` anchor exists in source)
-->

### D-001 | nano_vlm shared-embedding weight tie at call time | 2026-06-15
**Context**: `use_shared_embedding=True` (the default for every variant) has silently done nothing since plan_2026-06-15_39a31d4a/D-002 dropped the post-build kernel reassignment (which was both illegal under Keras 3 — "cannot add state to an already-built Layer" — and transposed-shape-wrong). `output_projection` is now an independent `Dense(vocab, use_bias=False)` with kernel `(dim, vocab)`. Logits are produced at TWO sites: `call()` (~`model.py:540`) and `generate()` (~`model.py:595`). `word_embeddings.embeddings` is `(vocab, dim)`; the `vision_dim == text_dim` invariant is enforced at `model.py:317-323`.
**Decision**: Re-tie at CALL TIME — under the same guard as `_create_output_projection` (`use_shared_embedding and text_component_type=='decoder' and hasattr(text_component,'word_embeddings')`), produce logits via `keras.ops.matmul(x, keras.ops.transpose(self.text_component.word_embeddings.embeddings))` at BOTH `call()` and `generate()`, instead of `self.output_projection(x)`. Keep `output_projection` built (serialization) but unused on the shared path.
**Trade-off**: Real input/output weight sharing (the intended architecture) **at the cost of** carrying a built-but-unused `output_projection` Dense on the shared path (dead weights), plus the inline matmul duplicated at two call sites rather than a single reusable layer.
**Reasoning**: Findings F1 confirm no tied-Dense layer exists in the repo; the inline matmul is the minimal correct fix (SMALL, zero new classes, no serialization edge-cases). A `TiedDense` abstraction would be a single-use abstraction with cross-save weight-reference hazards (Option A, rejected) — earned-abstraction rule says do not build it for <2 call sites of a shared mechanism. NEVER reassign weights post-build (that is the exact failure that was reverted). The `output_projection` must stay built so `use_shared_embedding=False` and `get_config`/save-load keep working.
**Anchor-Refs**: `src/dl_techniques/models/nano_vlm/model.py:540-552` (call site + D-001 anchor), `src/dl_techniques/models/nano_vlm/model.py:606-616` (generate site)

### D-002 | PFTBlock build/compute_output_shape accept tuple, not just list | 2026-06-15
**Context**: `PFTBlock.call` (`progressive_focused_transformer.py:452`) already accepts `isinstance(inputs, (list, tuple))`, but `build` (`:287`) and `compute_output_shape` (`:553`) accept `isinstance(input_shape, list)` only — a tuple of shapes is mis-read as a single shape, yielding `Invalid dtype: tuple` at norm build. The pft_sr caller works around this by passing a list `[x, prev_attn_map]`. SYSTEM.md documents the `:287` follow-up but NOT `:553`. This asymmetry was surfaced as a non-blocking concern in plan_2026-06-15_39a31d4a's adversarial review.
**Decision**: Change `isinstance(input_shape, list)` → `isinstance(input_shape, (list, tuple))` at BOTH `:287` (build) and `:553` (compute_output_shape). Leave the pft_sr caller's list-passing unchanged (list still satisfies the widened check).
**Trade-off**: A symmetric, caller-agnostic layer contract **at the cost of** leaving a now-partially-stale explanatory comment in the pft_sr caller (acceptable; minimizes churn).
**Reasoning**: The change strictly WIDENS accepted types (tuple now allowed alongside list) — backward-compatible, zero-regression for existing list callers and for the single-input bare-tensor path (`TestPFTBlockConformance`). Reverting the pft_sr caller would add churn for no correctness gain. `:553` must be fixed in the same commit or `compute_output_shape` still rejects tuples (functional-API / serialization surface).
**Anchor-Refs**: `src/dl_techniques/layers/transformers/progressive_focused_transformer.py:287` (build)

### D-003 | ClassTokenPrepend gets a dedicated unit test | 2026-06-15
**Context**: `ClassTokenPrepend` (`layers/embedding/class_token.py`, added plan_2026-06-15_39a31d4a/D-001) is the canonical prepend-CLS solution reused by dino v1 + v3, but has NO dedicated unit test — covered only transitively via the dino_v1 smoke. Flagged as hygiene debt in that plan's adversarial review (concern 3) and in SYSTEM.md. Template exists: `tests/test_layers/test_embedding/test_positional_embedding.py`.
**Decision**: Add `tests/test_layers/test_embedding/test_class_token.py` mirroring the positional-embedding test structure, with 5 behavioral assertions: (a) output shape `(B,N+1,dim)`; (b) `output[:,0,:]` equals broadcast of `cls_token[0,0,:]`; (c) `output[:,1:,:]` equals input exactly; (d) `get_config` round-trip preserves `initializer` and is numerically identical after weight copy; (e) `.keras` save/load round-trip bit-identical. Process/test-only — in-code anchor optional.
**Trade-off**: Durable behavioral coverage of a reused layer **at the cost of** one added test file (a Complexity-Budget file charge, but a test, not production surface).
**Reasoning**: LESSONS: "first-ever tests on never-tested layers must assert BEHAVIORAL properties, not just shapes" and "always include a save/load round-trip wrapping the layer in a keras.Model." Assertions (b)/(c) pin the prepend semantics (not a hollow no-op); (e) pins serialization. Mirroring the existing template minimizes design risk.
**Anchor-Refs**: (none — test-only decision; anchor optional)

### D-004 | dino v3 .item() -> float() + author forward smoke | 2026-06-15
**Context**: `dino_v3.py:274` uses `[r.item() for r in ops.linspace(...)]` — `.item()` is NumPy-only and crashes on a Keras tensor; it fires only when `stochastic_depth_rate > 0` (default 0.0, so default construction passes, but e.g. the "giant" variant sets 0.4). Identical to the v1 bug already fixed. v3 src is otherwise structurally clean (double-super removed, `ClassTokenPrepend` at `:259`). Its test stub `tests/test_models/test_dino/test_dino_v3.py` is a 0-byte file. v3 has never been forward-run end-to-end → cascade risk.
**Decision**: Change `[r.item() for r in ...]` → `[float(r) for r in ...]` at `dino_v3.py:274`, and fill the v3 test stub with a forward-only smoke mirroring `test_dino_v1.py`: `create_dino_v3("small", image_size=(32,32), num_classes=10)`, `model(np.random.rand(2,32,32,3).astype("float32"), training=False)`, assert finite `(2,10)`. EXPECT a cascade (first-ever forward); handle minimally in-scope; if >2 bugs deep, xfail the v3 test with the captured error and surface — do NOT block the other 3 items.
**Trade-off**: A forward-validated, regression-guarded dino v3 (and the proactive `.item()` fix even at default rate) **at the cost of** accepting a documented xfail if the never-run path cascades beyond a 2-bug budget.
**Reasoning**: LESSONS: budget ≥2 bugs per never-executed family; xfail-with-captured-error is the correct smoke idiom for an unresolvable cascade. dino v2 is explicitly DEFERRED (LARGE: cond-dict-vs-tensor, Dense-in-Lambda, 3-input masked forward) to its own plan — out of scope here.
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v3.py:274` (anchor), `src/dl_techniques/models/dino/dino_v3.py:276` (float(r) fix)

## plan_2026-06-15_39a31d4a
### D-001 | EXPLORE → PLAN | YYYY-MM-DD
**Context**: <one-paragraph background — what was discovered in EXPLORE>
**Decision**: <chosen approach in one sentence>
**Trade-off**: <X> **at the cost of** <Y>
**Reasoning**: <why this trade-off is acceptable; what alternatives were rejected>
**Anchor-Refs**: `path/to/file.ext:LL`, `other/file.ext:LL-MM`  (required when a matching `# DECISION plan_2026-06-15_39a31d4a/D-NNN` anchor exists in source)
-->

### D-001 | EXPLORE → PLAN | 2026-06-15
**Context**: `DINOv1.__init__` (`models/dino/dino_v1.py`) runs `outputs = self._build_model(inputs)` at :515 BEFORE `super().__init__(inputs=, outputs=)` at :518, and `_build_model` calls `self.add_weight(cls_token)` at :541 — `add_weight` requires the Keras layer machinery that only `super().__init__` sets up, so construction crashes. The same root cause exists in `DINOv2VisionTransformer` (:342) and `DINOv2` (:913). `dino_v3.py` has a DISTINCT bug: `DINOv3.__init__` calls `super().__init__(**kwargs)` (subclassed) at :180 then `super().__init__(inputs=, outputs=, ...)` (functional) at :211 — a double-init crash.
**Decision**: Restructure `DINOv1.__init__`/`_build_model` so the `cls_token` `add_weight` runs AFTER `super().__init__` (preferred: move it out of `_build_model` into a `build()` override, keeping the functional graph symbolic; fallback: split graph-build vs weight-create around the super call). For v3, delete the first `super().__init__(**kwargs)` at :180 and keep the functional one at :211. v2 is OPTIONAL (same restructure, only if v1 generalizes trivially or budget allows). Assert forward-only (NOT `.keras` round-trip — DINOHead round-trip is a separate known break, out of scope).
**Trade-off**: A structural change to a Functional-API model (touching construction ordering) **at the cost of** a higher-risk edit than a rename, bounded by a 2-attempt leash and a dino-only xfail-revert STOP-IF (2/3 fixed is an acceptable, surfaced partial).
**Reasoning**: Moving `add_weight` after super is the minimal correct fix; a full subclassed-model rewrite is LARGER and out of scope. Alternative rejected: patching by reordering statements alone may leave the functional graph referencing an uncreated weight (F-DINO-1) — hence the `build()`/post-super split. The STOP-IF prevents dino's risk from blocking the 2 low-risk fixes. v3's double-super is unambiguous (delete :180).
**Anchor-Refs**: `src/dl_techniques/models/dino/dino_v1.py:538` (cls_token via ClassTokenPrepend sub-layer), `src/dl_techniques/models/dino/dino_v3.py:181` (deleted bare super()), `src/dl_techniques/models/dino/dino_v3.py:254` (cls_token via ClassTokenPrepend sub-layer), `src/dl_techniques/layers/embedding/class_token.py:43` (new reusable ClassTokenPrepend — build()-owned add_weight)
**Update (2026-06-15, EXECUTE step-1)**: Used Option A (reusable `ClassTokenPrepend` sub-layer) — chosen over a `build()` override on the model (F-DINO-1 risk: Functional model finalizes its graph at __init__, build() never re-runs the symbolic part). The sub-layer is shared by v1 AND v3 (both had add_weight-before-super; the v3 add_weight at old :252 was an UNDOCUMENTED cascade beyond the plan's "just delete :180" — reported). Second cascade: dino_v1.py `dpr = [x.item() ...]` (now :559) crashed (`EagerTensor has no .item()` under TF backend) — fixed to `float(x)` (in-scope, trivial). dino_v1 smoke PASSES, logits (2,10).

### D-002 | EXPLORE → PLAN | 2026-06-15
**Context**: `nano_vlm/model.py` constructs `MultiModalFusion(**self.fusion_config)` at :364. `MultiModalFusion.__init__` (`layers/fusion/multimodal_fusion.py:129`) actually takes `dim` (default 768) and `attention_config={'num_heads':N}` — NOT a top-level `embed_dim` or `num_heads`. nano_vlm injects the GHOST kwargs at 5 sources: the 3 factory `fusion_config` dicts (:727-730, :741-744, :755-758) and `_validate_and_prepare_configs` (:339 sets `embed_dim`, :343-344 sets `num_heads`). Because of the `**` splat, missing any one source re-injects the ghost kwarg.
**Decision**: Rename at ALL 5 sources: `embed_dim`->`dim`, and `num_heads`->nested `attention_config={'num_heads':N}` (the `_validate` cross-attention default uses `setdefault('attention_config', {})['num_heads']=...`). Reference correct form: `layers/heads/vlm/factory.py:121`. Handle any forward cascade (VisionEncoder `output_mode` / TextEncoder) minimally in-scope; STOP-IF >2 bugs deep → xfail-revert nano_vlm only.
**Trade-off**: A multi-site caller-side rename (5 sources) **at the cost of** touching the model at several lines (vs. a single construction point), accepted because the `**` splat makes a partial rename silently re-inject the ghost.
**Reasoning**: The drift is a never-built API the caller was coded against (transformers/fusion refactor). Fixing the callee is wrong (it has live correct callers via the factory). Grep-confirmed 5 sources in findings; LESSONS "verify every site with grep before PLAN" honored.
**Anchor-Refs**: `src/dl_techniques/models/nano_vlm/model.py:366-370` (MultiModalFusion construction; D-002 anchor placed here)
**Update (2026-06-15, EXECUTE step-2)**: Renamed all 5 sources (3 factory dicts + 2 `_validate_and_prepare_configs` injections). Cascade fired exactly ONCE (1 deep, within the ≤2 STOP-IF budget), NOT the predicted VisionEncoder/TextEncoder path: `build()` did `self.output_projection.kernel = self.text_component.word_embeddings.embeddings` (weight-tie) AFTER the Dense was built — Keras 3 forbids post-build state reassignment, and the embeddings (vocab,dim) are the transpose of the Dense kernel (dim,vocab), so the tie was both illegal AND shape-wrong (dead-on-forward). SMALLEST in-scope fix: dropped the broken tie (output_projection keeps its own built kernel; weight-tying lost — re-add via a proper tied-Dense if memory-sharing later needed). nano_vlm smoke PASSES, logits (2, 213, 256) = (batch, vision_seq 197 + text_seq 16, vocab 256), finite. NOTE annotated at the old tie site (~:462).

### D-003 | EXPLORE → PLAN | 2026-06-15
**Context**: `pft_sr/model.py` has two construction-drift bugs: (a) :168 passes `drop_path=self.dpr[block_idx]` to `PFTBlock`, whose ctor (`progressive_focused_transformer.py:154`) only accepts `drop_path_rate`; (b) :229/:246/:261/:286 call `keras.ops.nn.depth_to_space(x, N)` inside `Lambda`, but that symbol does not exist in Keras 3.8 (same class as the darkir `DepthToSpace` break fixed in precedent plan_2026-06-15_00924f53/D-002).
**Decision**: (a) rename :168 `drop_path`->`drop_path_rate`. (b) import `PixelShuffle2D` from `layers/pixel_unshuffle.py:203` and replace each `Lambda(depth_to_space)` with `PixelShuffle2D(block_size=N)`, matching each site's factor (`self.scale` at :229/:286, literal `2` at :246/:261). The `nearest+conv` upsampler's separate `keras.ops.log2`-in-`range()` latent bug is OUT of scope (`variant="light"` defaults to `pixelshuffle`).
**Trade-off**: REUSE the existing serializable `PixelShuffle2D` **at the cost of** a new import dependency in pft_sr (vs. an inline Lambda), accepted because the Lambda symbol is dead and `PixelShuffle2D` is graph-safe, serializable, and round-trip-tested (no new abstraction — DRY).
**Reasoning**: Writing a new depth_to_space would duplicate `PixelShuffle2D` (LESSONS/SYSTEM.md: use it instead of any DepthToSpace reference). The `drop_path_rate` rename is unambiguous from the ctor signature.
**Anchor-Refs**: `src/dl_techniques/models/pft_sr/model.py:169` (drop_path_rate), `src/dl_techniques/models/pft_sr/model.py:228-235` (PixelShuffle2D swap; D-003 anchor at upsampler fix point), `src/dl_techniques/models/pft_sr/model.py:335-346` (D-003 anchor at call-loop cascade fix: list-not-tuple + None-guard)
**Update (2026-06-15, EXECUTE step-3)**: drop_path->drop_path_rate (now :169) + all 4 depth_to_space Lambdas -> PixelShuffle2D (block_size = scale/2/2/scale, verified; no shape-error at build so F-PFT-FACTOR did NOT fire). Cascade fired TWICE (within the F-PFT budget, both in pft_sr/model.py's forward loop, NOT in the shared PFTBlock layer): (1) `block((x, None), ...)` on the first block crashed Keras' `__call__` shape machinery (`optree get_shapes_dict` -> `'NoneType' has no attribute 'shape'`) -> guard: call bare `x` when prev_attn_map is None (PFTBlock.call handles non-tuple). (2) passing a TUPLE `(x, prev_attn_map)` to subsequent blocks hit `Invalid dtype: tuple` in PFTBlock.build:304 — build only special-cases `isinstance(input_shape, list)`, and a tuple is mis-read as a single shape -> pass a LIST `[x, prev_attn_map]` instead (keeps the fix in pft_sr/model.py; PFTBlock layer untouched). Smoke PASSES, output (2,64,64,3) finite. NOTE: the PFTBlock.build tuple/list asymmetry is a latent layer bug — surfaced for REFLECT but left unfixed (out of scope; the list call-convention sidesteps it).
