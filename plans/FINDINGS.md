# Consolidated Findings
*Cross-plan findings archive. Entries merged from per-plan findings.md on close. Newest first.*

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
