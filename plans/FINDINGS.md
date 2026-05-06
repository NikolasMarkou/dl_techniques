# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

## plan_2026-05-06_82749628
### Index
- [causal-blocks-api.md](plan_2026-05-06_82749628/findings/causal-blocks-api.md) — CausalCliffordNetBlock (dim-preserving) and CausalCliffordNetBlockDSv2 (causal stride downsampler). Shapes, allowed kwargs, ctx_mode/pool restrictions.
- [upsampling-causality.md](plan_2026-05-06_82749628/findings/upsampling-causality.md) — keras.layers.UpSampling2D(size=(1, s), interpolation="nearest") is causal-safe. Bilinear / transposed-conv leak future. Tail right-pad + crop in call() solves the seq_len % total_stride != 0 case.
- [lm-and-train-mirror.md](plan_2026-05-06_82749628/findings/lm-and-train-mirror.md) — lm.py contract (variants ladder, from_variant, get_config, tie_word_embeddings, {"logits": ...} dict). Train script callbacks are model-agnostic — only model class + custom_objects + dataset/results-dir prefix change.

### Key Constraints

### Hard
- 4D layout (B, 1, seq_len, channels) end-to-end through encoder/decoder.
- Strict causality: changing input position k must leave outputs at all positions < k byte-identical (within fp tolerance).
- DSv2 ctx_mode restricted to {diff, abs} (no pyramid_diff); pool kinds restricted to {avg, max} (LESSONS L33).
- Upsample must be nearest along W only — no bilinear, no transposed conv, no sub-pixel.
- Output shape (B, seq_len, vocab_size) — must preserve full input length even when seq_len % total_stride != 0. Use right-pad + crop inside call().
- Output dict key MUST remain "logits" — MaskedCausalLMLoss and the train script assume this.
- Keras 3 conventions: @keras.saving.register_keras_serializable(), keras.ops, dl_techniques.utils.logger, full get_config() round-trip, no print.
- tie_word_embeddings flag — same default as lm.py (True). Keep output_bias add_weight when tied.
- Skip-connection fusion at SAME resolution as encoder skip — explicit Concatenate(axis=-1) followed by 1x1 Conv2D projection back to channels.
- Test scope: pytest only on the new model file (LESSONS L20: never make test).
- AdamW WD only — no kernel_regularizer=L2.
- Use .venv/bin/python. Commit per step. User pushes themselves.

### Soft
- Mirror lm.py and train_cliffordnet_nlp.py line-by-line where it doesn't conflict with U-Net structure.
- Variant ladder names: nano, mini, base, large, xl (1:1 with lm.py).
- Class name: CliffordNetLMUNet. File path: src/dl_techniques/models/cliffordnet/lmunet.py. Train script: src/train/cliffordnet/train_cliffordnet_nlp_unet.py.
- Default U-Net topology per variant: 3 stages (encoder + bottleneck + decoder), strides [2, 2], mirroring decoder. For deeper variants, allow 4 stages (strides [2, 2, 2]).
- Default upsampler: keras.layers.UpSampling2D(size=(1, s), interpolation="nearest") + Concatenate(axis=-1) + 1x1 Conv2D.

### Ghost constraints (none found)
- "U-Net needs different output dict key" — false; loss + train wrapper + probe pivot on "logits".
- "Need a custom causal upsampler layer" — false; UpSampling2D nearest is sufficient.
- "Need to clone all callbacks" — false; they're model-agnostic.

### Exploration Confidence
- Problem scope: deep — exact line ranges of both causal blocks read; lm.py and train script read in full; vision unet skim confirms encoder/decoder fusion pattern; _make_causal_pool / padding="same" causal status grounded in LESSONS L33.
- Solution space: constrained — block APIs and lm.py/train-script contracts pin the design. Only knobs are stages/strides/blocks-per-stage and skip-fusion (concat-1x1).
- Risk visibility: clear — main risk is a subtle causality leak in upsample/concat/skip path; mitigated by the existing test_causality_* pattern from test_cliffordnet_lm.py (perturb position k, assert all positions < k byte-identical).

Ready for PLAN.

## plan_2026-05-06_13a2df9e
### Index
- [F-001 scope-and-callers.md](plan_2026-05-06_13a2df9e/findings/scope-and-callers.md) — pure-additive sibling class, no callers affected, layers `__init__.py` is empty so no export plumbing.
- [F-002 causality-mechanics.md](plan_2026-05-06_13a2df9e/findings/causality-mechanics.md) — how `CausalCliffordNetBlock` enforces causality + which DSv2 ops are/are not causal under `(H=1, W=seq_len)` layout. Conclusion: avg/max with `padding="same"` are causal; `blur`, `gaussian_dw`, `pyramid_diff`, `pixel_unshuffle`, `resnetd` are not.
- [F-003 dsv2-merge-points.md](plan_2026-05-06_13a2df9e/findings/dsv2-merge-points.md) — exact API delta, build-validation, and call-path changes vs DSv2; padding arithmetic for arbitrary `kernel_size`; tests to mirror.

### Key Constraints

### Hard
- Same file: `src/dl_techniques/layers/geometric/clifford_block.py`. No new modules.
- `H=1, W=seq_len` layout (matches `CausalCliffordNetBlock`).
- Strict causality along W: future positions must not influence past outputs (must be regression-tested).
- Existing 117+ tests in `test_clifford_block.py` must continue to pass — purely additive change.
- Keras 3 / TF 2.18 idioms (`@keras.saving.register_keras_serializable()`, `keras.ops`, `dl_techniques.utils.logger`, no `print`).
- No public-API breakage to existing classes.

### Soft
- Mirror existing class structure (init/build/call/get_config + the helper functions style).
- Match test layout (per LESSONS.md) — class per layer, fixtures, save/load round-trip, gradient flow, causality regression.

### Ghost constraints (not present)
- Layers `__init__.py` is empty (per `layers/CLAUDE.md`) — no export plumbing needed.

### Exploration Confidence
- Problem scope: **deep** — exact line ranges and semantics of both parents read; constraints classified.
- Solution space: **constrained** — combine two known patterns; only thing genuinely new is the narrower pool-kind surface and reasoning about which pools are causal.
- Risk visibility: **clear** — main risk is a subtle pool-future-leak; mitigated by restricting to avg/max only and writing a perturb-future-position regression test.

Ready for PLAN.

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
