# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

## plan_2026-05-05_0eac2c81
### Index

- [F-001] Existing test suite passes (115/115). `tests/test_layers/test_geometric/test_clifford_block.py` covers all five layer classes.
- [F-002] Empirical verification of Phase-5 review findings — corrections below.
- [F-003] Call sites of these layers — `models/cliffordnet/{denoiser, clip, conditional_denoiser, confidence_denoiser}.py`. Any signature change must be checked against them.
- [F-004] Existing test pattern (per LESSONS.md): pytest fixtures + class-per-layer + init/build/shape/cli_modes/numerical_stability/serialization/save_load/gradient_flow/stacking. New tests should mirror this.

### Key Constraints

- **Hard:** No public-API breaking changes — these layers are used by 4 cliffordnet models.
- **Hard:** Existing 115 tests must continue to pass.
- **Hard:** Keras 3.8 + TF 2.18 idioms (no print, use `dl_techniques.utils.logger`).
- **Soft:** Match existing class-by-class test layout.

### Empirical Re-verification of Phase-5 Findings

| ID | Claim | Verdict | Evidence |
|----|-------|---------|----------|
| B1/B5 | Multi-input `build` crashes Functional API in `SparseRollingGeometricProduct` / `GatedGeometricResidual` | **REFUTED** | `test_model_save_load` (line 186-200) wraps SRGP in Functional API with two inputs, saves+loads, and passes. Keras 3 resolves the multi-input case by using the first input shape for build. |
| B7 | `CliffordNetBlock` silently fails when input D ≠ self.channels | **CONFIRMED** | Empirical: `CliffordNetBlock(channels=16)(D=8 input)` → cryptic `InvalidArgumentError: required broadcastable shapes` at residual. Should raise at build with a clear message. |
| B16 | `CliffordNetBlockDSv2 ctx_mode='pyramid_diff'` shape mismatch on odd `H/s` | **CONFIRMED** | Empirical: `H=10, strides=2` (so H/s=5, odd) → broadcast error in `z_ctx - z_lo_up`. `H=8` works. |
| B17 | `_make_pool_v2` silently returns `Identity` for `pixel_unshuffle`/`resnetd`/`blur` at `strides=1` | **CONFIRMED** | Empirical: all three return Identity at strides=1. User intent silently dropped. |
| B11 | `CausalCliffordNetBlock` causality is unverified by tests | **PARTIALLY CONFIRMED** | Empirical leak test passes (modifying position 9 changes only output 9, not 0..8). Causality currently holds — but no regression test guards it. Add test. |
| B4 | SRGP accepts `s=0` and negative shifts | **CONFIRMED** | Empirical: `shifts=[0,1]` accepted, `0` kept (wedge term identically zero). |
| B8 | `CliffordNetBlock(channels=1, use_global_context=True)` crashes | **CONFIRMED** | Empirical: `ValueError: All provided shifts [1] are >= channels (1)` from filter inside SRGP. Need clearer validation at outer block. |
| B9 | `(input_shape[2] or 0) + 2` corrupts dynamic seq_len | **REFUTED in practice** | Empirical: `keras.Input(shape=(1, None, 8))` builds and forwards correctly. The `or 0` produces W=2 in build_config but DepthwiseConv2D ignores spatial dims at build-time. Cosmetic-only. Demote to LATENT, optional fix. |
| B3 | `SparseRollingGeometricProduct.get_config` comment claims to store unfiltered shifts but stores filtered list | **CONFIRMED** | Direct inspection: line 252 comment vs line 253 code. |
| B13 | `CliffordNetBlockDS` conv-bias comment claims to "restore the affine degree of freedom" | **CONFIRMED** | A bias only restores shift, not scale. Comment misleads. |
| P1 | SRGP `roll(z_det, s)` is computed for `cli_mode='inner'` even though only wedge uses it | **CONFIRMED** | Direct read of `call`: `z_det_s` is computed unconditionally. |
| P2 | `keras.ops.broadcast_to(c_glo, shape(z_det))` materializes (B,H,W,D) before subtraction | **CONFIRMED** | Read of `CliffordNetBlock.call` lines 733-734. Subtraction would broadcast natively. |
| L1 | `_proj_input_dim` is dead state in SRGP | **CONFIRMED** | Set in `__init__`, used only inside `__init__` to build proj. Could be removed; or used as build-time assertion. |
| L5 | DSv2 `skip_pool="pixel_unshuffle"` is not a true identity skip | **CONFIRMED, doc only** | PixelUnshuffle includes a learned 1×1. Not a bug; document. |
| B12 | Cumsum precision under fp16 | **THEORETICAL** | No current evidence of failure (we don't run fp16 here). Defer. |
| Other | B6, L2, L3, L4, L6, X1, X2, X3, B14, B15, B18 | **DEFER** | Mostly defensive / design choices. Cost > benefit at this iteration. |

### Confirmed Real Bugs to Fix

1. **B16** — `CliffordNetBlockDSv2` pyramid_diff: crop or dynamic-resize `z_lo_up` to match `z_ctx`.
2. **B7** — `CliffordNetBlock.build`: validate `input_shape[-1] == self.channels`, raise `ValueError`.
3. **B17** — `_make_pool_v2`: raise `ValueError` (or warn) when `kind ∈ {pixel_unshuffle, resnetd, blur, gaussian_dw}` and `strides=1` — these kinds only make sense with downsampling.
4. **B4** — `SparseRollingGeometricProduct`: validate `s >= 1` for every shift; reject `0` and negatives.
5. **B8** — `CliffordNetBlock` (and DS variants): when `use_global_context=True`, require `channels >= 2`.
6. **B3** — Fix comment in SRGP `get_config` to match what the code actually does.
7. **B13** — Fix comment in `CliffordNetBlockDS` about conv-bias / affine.
8. **P1** — Guard `z_det_s` computation behind `cli_mode in ("wedge", "full")`.
9. **P2** — Drop `broadcast_to` in CliffordNetBlock global branch (and DS / DSv2 mirrors).
10. **L1** — Remove dead `_proj_input_dim` attribute (or add explicit build assertion).
11. **B11 (test only)** — Add `test_causal_block_no_future_leak` regression test.

### Out of scope (this plan)

B1, B5, B6, B9 (cosmetic), B12, B14, B15, B18, L2, L3, L4, L6, X1, X2, X3.

### Files to modify

- `src/dl_techniques/layers/geometric/clifford_block.py` — primary surgery.
- `tests/test_layers/test_geometric/test_clifford_block.py` — add regression tests for each fix.
- (No model changes — all four cliffordnet models pass channels==D, never trip B7/B8.)

## plan_2026-05-05_60c5be7d
### Index
- [code-and-line-refs.md](plan_2026-05-05_60c5be7d/findings/code-and-line-refs.md) — Verified line numbers for every B/H/M finding, current weight count in tests, fp16 test coverage audit (none exists), and D-004 risk assessment for H-5/M-1.

### Key Constraints

### Hard
- Existing tests at `test_routing_probabilities.py`, `test_routing_probabilities_hierarchical.py`, `test_cliffordnet_lm_routing.py` MUST still pass after fixes.
- D-004 (LESSONS.md L31): plain tensors in `build()` captured by `compute_output_spec` FuncGraph cause "out of scope" errors. Cosine basis is stored as `add_weight(trainable=False)` for this reason. User instruction: do not change cosine basis storage unless verified safe in Keras 3.8+; if uncertain, leave it.
- No `print` statements; use `dl_techniques.utils.logger`.
- Keras 3 / TF 2.18 backend, mixed precision matters.
- `.venv` for all runs. Scope pytest to changed modules.
- No parallel GPU jobs. User pushes commits.

### Soft
- Conservative: prefer smaller change. Decline M-1 (cosine basis) unless explicitly verified.
- H-4 (epsilon=0 warning) is low priority — info-level log only when `epsilon == 0 and mode == 'trainable'`.

### Ghost constraints
- "test asserts `non_trainable_weights == 3` is a load-bearing API contract" — false. Test-as-truth (LESSONS L12). Update when H-5 lands.
- "set_weights ordering in trainable mode is part of public API" — false. Internal tests only.

### Exploration Confidence
- Problem scope: deep — every line ref verified, every external user audited, every test impact mapped.
- Solution space: constrained — fixes are small, localized to `routing_probabilities.py` + 2 test files.
- Risk visibility: clear — main risk is H-5 + cliffordnet save/load (mitigated by running that test on the H-5 step). M-1 explicitly opted out per user guidance.

Ready to transition to PLAN.

### Critical Findings (inline summary)

- B-1 fp16 sigmoid clip underflow: L658-661 sigmoid+clip on fp16 dtype; L669 cast to fp32 too late. Doc L60-65 wrong.
- B-2 Final fp16 cast: L717 `cast(final_probs, inputs.dtype)`. Vocab~50K leaf ~1.5e-5 < fp16 normal (6.1e-5).
- B-3 compute_output_shape: L753-757 uses cached `_normalized_axis`; wrong if called with different-rank shape.
- H-1 _RENORM_TINY comment: L308-313 false fp16 claim. Divide actually fp32 (after L669).
- H-2 bool output_dim: L369, L377-378. `isinstance(True, int)` is True. Add `not isinstance(output_dim, bool)`.
- H-3 deterministic use_bias: L430-431 warns only on `not use_bias`. Should also warn when `use_bias=True` AND `mode=='deterministic'`.
- H-4 epsilon=0 + trainable: L349-358 accepted. Info-level log only.
- H-5 mask weights recomputable: L590-605. ~8*(padded-1) bytes per checkpoint per layer (~512KB at vocab 65536).
- M-1 cosine basis: L555-564. DECLINED — D-004 anchors as add_weight.

### Test impact

- `test_routing_probabilities.py` L142, L547: `non_trainable_weights == 3` → after H-5: `== 1`.
- `test_routing_probabilities_hierarchical.py` L160-162, L200-202, L221-223: `set_weights([kernel, bias, mask_mul, mask_add])` → `[kernel, bias]`.
- No fp16 test anywhere — must ADD for B-1/B-2.
- `test_cliffordnet_lm_routing.py` save/load — must keep working post-H-5.

### D-004 FuncGraph residual risk for H-5

H-5 plan: store `np.ndarray` on `self._mask_mul_np` (numpy, not tensor), then `ops.convert_to_tensor(...)` in call(). Conversion lives in call's trace, unlike D-004's failure mode (tensor stored on self). Should be safe; verified by cliffordnet save/load test on the H-5 commit.

## plan_2026-05-04_1b2810b6
### Index
- [lm-structure.md](plan_2026-05-04_1b2810b6/findings/lm-structure.md) — `CliffordNetLM` forward path, vocab projection wiring, head naming, dict-output `{"logits": ...}` contract, serialization, no deep supervision.
- [loss-compatibility.md](plan_2026-05-04_1b2810b6/findings/loss-compatibility.md) — `MaskedCausalLMLoss` and `FocalCausalLMLoss` both accept `from_logits=False` and work correctly on probabilities. No new loss class needed. Numerical stability bounded by layer's epsilon clip.
- [routing-cost-and-modes.md](plan_2026-05-04_1b2810b6/findings/routing-cost-and-modes.md) — at vocab=50261 padded=65536 / 16 decisions; trainable mode 2K-6K params (~3000x fewer than Dense); deterministic mode is too tight (16 cosine projections to discriminate 50K tokens). Memory peak ~25% higher than Dense head. No existing test covers vocab-scale or 3D `(B, L, V)` routing calls.

### Key Constraints

### Hard
- Output dict key must remain `"logits"` — `train.fit` data wrapper does `(x, y) -> (x, {"logits": y})` and compile uses `loss={"logits": ...}, metrics={"logits": ["accuracy"]}`. Renaming would force changes to the dataset wrapper and loss spec.
- Loss must use `from_logits=False` — RoutingProbabilitiesLayer outputs probabilities in `[eps, 1-eps]` summing to 1. Passing them through `from_logits=True` (default) would softmax already-softmaxed values → wrong loss.
- Causality preserved automatically — routing acts per-token along channels axis, no inter-token mixing.
- Keras 3 conventions: `@keras.saving.register_keras_serializable()`, full `get_config()` round-trip, `dl_techniques.utils.logger` (no print), `MPLBACKEND=Agg` for training, single GPU only.
- Import path: `dl_techniques.layers.activations.routing_probabilities.RoutingProbabilitiesLayer`.
- `load_model_from_checkpoint` `custom_objects` must include `RoutingProbabilitiesLayer` for resume to work.

### Soft
- Routing mode default: `"trainable"` (user's stated preference). Expose `"deterministic"` as opt-in for ablation.
- Naming: model class `CliffordNetLMRouting`; file `lm_routing.py`. Train script `train_cliffordnet_nlp_routing.py`. Sibling files; no edits to `lm.py` or `train_cliffordnet_nlp.py`.
- Head layer naming: keep `head_norm`, `head_dropout`; name routing layer `output_routing` for symmetry with existing `output_proj`.
- Variants: reuse the existing `MODEL_VARIANTS` dict 1:1 (nano/mini/base/large/xl).

### Ghost constraints (none)
- "Must define a new NLL loss for probabilities" — not needed; both existing CLM losses support `from_logits=False`.
- "Must rename output dict key to 'probs'" — not needed; loss is agnostic to key name.

### Exploration Confidence
- **Problem scope**: deep — every line of the swap point (lm.py 257-265, train script 553-566) is mapped; layer behavior at axis/sum level understood; loss compatibility verified.
- **Solution space**: constrained — single swap point in the model, single config flag in the train script. Mode flag is the one open knob, exposed as CLI arg.
- **Risk visibility**: clear — risks are (1) loss collapse / NaN (mitigated by layer epsilon clip), (2) memory bump on `(B*L, 65536)` intermediate (~25% over Dense), (3) expressive bottleneck (16 decisions → 50K classes at info-theoretic floor).

Ready to transition to PLAN.

## plan_2026-05-04_38e259bf
### Index
- [layer-internals.md](plan_2026-05-04_38e259bf/findings/layer-internals.md) — public APIs, internals, and behavioral differences between `RoutingProbabilitiesLayer` and `HierarchicalRoutingLayer`.
- [usage-sites.md](plan_2026-05-04_38e259bf/findings/usage-sites.md) — every src/tests/docs reference, with line numbers, that must be updated.
- [api-design.md](plan_2026-05-04_38e259bf/findings/api-design.md) — chosen merged API: `mode` flag (`"deterministic"` default, `"trainable"` opt-in), validation rules, build/call/get_config plan.

### Key Constraints
- **Hard**: Keras 3 serialization round-trip must work for both modes (`@register_keras_serializable`, full `get_config`).
- **Hard**: Existing call sites in `multi_head_cross_attention.py` and `single_window_attention.py` instantiate `RoutingProbabilitiesLayer()` with default args — must remain valid (deterministic default preserves this).
- **Hard**: `inaturalist.py:146` has a stale import `dl_techniques.layers.hierarchical_routing` that does not exist; must be fixed when migrating that file.
- **Soft**: Test file `test_routing_probabilities_hierarchical.py` should be renamed/merged into `test_routing_probabilities.py` or kept separate using the new API. Decision: keep it as a separate file with updated imports/classname/kwargs (smallest delta, preserves pytest organization).
- **Ghost constraint check**: No backwards-compatibility alias is required — task explicitly says delete the hierarchical file.

## plan_2026-04-29_b6dbc601
### Index
- `findings/existing-block.md` — current CliffordNetBlock structure, context stream, forward path, project conventions
- `findings/variant-design.md` — proposed CliffordNetBlockDS API, forward path, edge cases, validation tests
- `findings/test-conventions.md` — test file structure and patterns to follow

### Key Constraints
- HARD: detail and context streams must share spatial+channel shape (element-wise geometric product).
- HARD: residual `x_skip + h_mix` requires matching spatial dims; when downsampling, x_skip must be pooled.
- HARD: channels are preserved through the block (no channel projection inside).
- SOFT: kernel_size default 7 (per goal); strides default 1 (preserves dim-preserving default behaviour).
- SOFT: skip_pool default "avg" (matches existing patterns for downsamplers per LESSONS.md L23).
- GHOST: docstring says "effective 7x7 RF" for two stacked 3x3 — actually 5x5. New variant truly uses 7x7.

### Exploration confidence
- Problem scope: deep — exact lines and shapes traced
- Solution space: constrained — shape invariants pin the design (pool x_norm before stream split)
- Risk visibility: clear — identified residual-shape and stream-shape invariants up front

## plan_2026-04-24_cf1a9ab7
### Index
- `findings/cliffordnet-architecture.md` — current `CliffordNet` model + `CliffordNetBlock` API (block is dim-preserving — must be wrapped, not modified, for hierarchical use). Files: `src/dl_techniques/models/cliffordnet/model.py`, `src/dl_techniques/layers/geometric/clifford_block.py`.
- `findings/training-infra.md` — existing CIFAR training recipe (`src/train/cliffordnet/train_cliffordnet.py`), `train.common` API, augmentation pipeline, runtime budget (~45-75 min/run @ 100 epochs nano-scale on RTX 4090). 5 variants serial = ~4-6h total.
- `findings/downsampling-design-space.md` — definition of the 5 variants (V1..V5) plus V0 baseline. Covers user's three axes: channel expansion strategy, downsampling block, stride configuration. Reuses `dl_techniques.layers.patch_merging.PatchMerging`.

### Key Constraints

### Hard constraints
- **Single GPU only.** Never run training jobs in parallel (driver/memory contention). 5 runs sequential.
- **Python conventions.** Keras 3, `@keras.saving.register_keras_serializable()`, `get_config`, `dl_techniques.utils.logger` (no `print`), `MPLBACKEND=Agg`, run via `.venv/bin/python -m train.cliffordnet.train_downsampling_techniques`.
- **`CliffordNetBlock` is dim-preserving.** Hierarchical variants must add inter-stage downsamplers; cannot make the block change channels itself.
- **Compute budget user-visible.** Estimated ~5h serial training must be surfaced before EXECUTE so user can approve.

### Soft constraints / decisions
- Hold base channels=128 (where applicable), shifts=[1,2], and total block budget ≈12 across variants → fair architectural comparison.
- Match existing recipe (AdamW WD=0.1, cosine + 5-epoch warmup, full augmentation pipeline) to make variant comparison clean.
- Shorten epochs to 100 (vs paper's 200) to fit in one session; user can rerun the winner at 200 later.

### Ghost constraints (none found)
- Considered: "must reuse `train_cliffordnet.py`". Not a real constraint — that script already takes a `variant` flag but is hardwired to `CliffordNet.from_variant`. Cleaner to write a new sibling script that imports its augmentation helpers.

## plan_2026-04-24_e4c8ebab
### Index
1. **findings/vision-tower.md** — `CliffordCLIP._build_vision_tower` is isotropic (single `vision_channels`/`vision_depth`/`vision_shifts`); blocks are shape-preserving (`x_out = x_prev + H_mix`). Default nano (D=128, depth=12, image=112, patch=4) holds 28×28×128 maps for 12 blocks → ~500-600 MB activations at batch=32. `large` at 224 → ~12 GB. Reusable hierarchical primitives in repo: `layers/patch_merging.py:PatchMerging` (used by `models/swin_transformer/model.py`) and `layers/downsample.py`. Clifford constraints: `shifts < channels` per stage; downsample lives BETWEEN blocks.
2. **findings/text-tower.md** — Text tower also isotropic; reshapes to `(B, 1, L, D_t)` for causal `DepthwiseConv2D(1, 3)`. Memory at L=77 modest (~50 MB nano) but linear in context_length. Causality is binding: stride-2 causal `DepthwiseConv2D(1, 2, strides=(1, 2))` is safe; `PatchMerging` 2D is wrong shape but a tiny `CausalSeqMerging` is straightforward. Pad mask must be downsampled in lockstep.
3. **findings/clip-wiring-and-pretrain.md** — Output contract is `(B, embed_dim)` after L2 — invisible to contrastive loss after refactor. Two pretrain wrappers in `train_clip.py` walk `vision_blocks`/`text_blocks` directly: `_VisionClassifier` (CIFAR-100, easy to adapt via new `_apply_vision_body()` helper) and `_TextLMWrapper` (per-token CLM — **breaks** if text downsamples). Three strategies presented: (1) vision-only hierarchical, (2) both towers + LM pretrain bypass, (3) stem-only stride.

### Key Constraints

### Hard
- `encode_image` / `encode_text` must remain `(B, embed_dim)` after L2 — contrastive loss math depends on it.
- `_TextLMWrapper` (CLM pretrain) requires per-token logits at original `context_length` — text downsampling either breaks this or requires a bypass code path.
- `CliffordNetBlock` / `CausalCliffordNetBlock` residuals require shape-preserving body — stride lives between blocks, not inside.
- `SparseRollingGeometricProduct` requires `shift < channels` per layer — per-stage shift lists must size to per-stage channel count.
- `get_config()` round-trip serialisation test (`test_from_variant_serialization_round_trip`) must keep passing — new config fields must be added.
- Causal text tower must remain causal end-to-end; only causal-friendly downsamplers allowed.
- No `print` (use `dl_techniques.utils.logger`); no parallel GPU jobs; never run `make test` (1.5h).

### Soft
- Variant ladder (`nano`/`nano_g`/`mini`/`small`/`base`/`large`) keys should remain importable via `from_variant`.
- Existing tests assert `vision_depth == 12` etc. — keep `vision_depth` as int (sum of per-stage depths) or update assert.
- Pretrain wrappers should keep working unchanged (the user's actual production loop relies on them).

### Ghost-constraint check
- The **isotropic shape was inherited from `CliffordNet` / `CliffordNetLM` standalone models** (verified by `test_nano_matches_cliffordnet_nano_depth_and_shifts` which asserts `vision_depth == 12, vision_shifts == [1, 2]` exactly). The CLIP-specific need is "two towers cheap to evaluate"; CliffordNet symmetry is a **soft** constraint (code reuse / pretrain compat), not hard. Worth surfacing in PLAN: do we keep CliffordCLIP variant names matching CliffordNet's, or accept a structural divergence?

### Exploration Confidence
- **Problem scope**: deep — exact line numbers for tower construction, forward path, pretrain wrappers, and tests are all known.
- **Solution space**: constrained — three concrete strategies named, each with known costs.
- **Risk visibility**: clear — pretrain wrapper coupling is the main subtle risk; causality on text is named; serialization round-trip test enforces config completeness.

Ready to transition to PLAN.

## plan_2026-04-24_1c5ae010
### Index
1. findings/staging-structure.md — current 4-stage layout, helpers, CLI flags, what "big patch" means.
2. findings/dataset-logging-gaps.md — what is and isn't logged today; proposed summary block.
3. findings/callers-and-impact.md — no callers or tests depend on staging; README has stale flags (out of scope).

### Key Constraints
- Hard: use dl_techniques.utils.logger only (no print).
- Hard: no parallel GPU; smoke-test only via --synthetic --max-train-samples 64.
- Hard: do NOT run `make test`. No tests cover train_clip.py.
- Soft: keep file name and Keras 3 conventions.
- Follow-up (out of scope): src/train/cliffordnet/README.md references removed CLI flags.
- Ghost constraint: curriculum was added for memory-constrained higher-res training; in practice users always run single-stage with `--stage2-epochs 0`. Removing matches actual usage.

### Exploration Confidence
- Scope: deep (file fully read; staging surface mapped).
- Solution space: constrained (direct removal + flag rename).
- Risk visibility: clear (no callers/tests depend on removed surface).
