# Findings
*Summary and index of all findings. Detailed files go in findings/ directory.*

*Cross-plan context: see plans/FINDINGS.md, plans/DECISIONS.md, and plans/LESSONS.md*

## User-Provided Context (seed)

### Target
Drone video streaming (mixed altitude), ~30 FPS RTX 4070 12GB. Patch-level anomaly/surprise
detection + SSL backbone. Telemetry (IMU Δr/p/y, GPS velocity, altitude) via AdaLN-zero.

### LeWM assets (plan_2026-04-21_8416bc0b)
- `src/dl_techniques/models/lewm/` — global-CLS JEPA.
- `src/dl_techniques/layers/adaln_zero.py` — AdaLN-zero (identity-at-init).
- `src/dl_techniques/regularizers/sigreg.py` — SIGReg (no EMA needed).

### Clifford primitives (`src/dl_techniques/layers/geometric/clifford_block.py`)
- `SparseRollingGeometricProduct`, `GatedGeometricResidual` (LayerScale).
- `CliffordNetBlock` — 4D `(B,H,W,D)` dual-stream.
- `CausalCliffordNetBlock` — 4D `(B,1,T,D)` left-padded causal.

### Open design decisions (must present in PLAN)
D-001 encoder (ViT vs Clifford vs hybrid); D-002 predictor 5D handling (factorized/flatten/3D);
D-003 target (tube-mask / next-frame / both); D-004 conditioning (where, per-frame vs global);
D-005 SIGReg placement (per-patch vs pooled); D-006 positional; D-007 streaming contract.

### Scope
IN: `models/video_jepa/`, synthetic drone dataset, training smoke test, tests
(shape/causality/serialization/AdaLN-id/SIGReg-finite), streaming API.
OUT: real drone data, surprise viz, downstream heads, full-scale training.

### Ops constraints
`.venv/bin/python`, `CUDA_VISIBLE_DEVICES=1`, `MPLBACKEND=Agg`, never parallel GPU.
Commit per step. Verify hardest case (causality + SIGReg stability) first.

## Index

1. **[clifford-primitives.md](findings/clifford-primitives.md)** — Full contracts of
   `SparseRollingGeometricProduct`, `GatedGeometricResidual`, `CliffordNetBlock`
   (4D `(B,H,W,D)`), `CausalCliffordNetBlock` (4D `(B,1,T,D)` with left-pad & causal
   cumsum mean). Factorized 5D handling is the clean path. BatchNorm-batch-of-1
   risk noted. Identity-at-init possible via LayerScale γ=1e-5.
2. **[lewm-reusable-assets.md](findings/lewm-reusable-assets.md)** —
   `AdaLNZeroConditionalBlock`: `inputs=[x,c]` (B,T,D)(B,T,D), zero-init ⇒
   identity-at-init. `SIGRegLayer`: `(..., N, D)` averages over axis=-3, 3 placement
   options for 5D latents. `LeWM.rollout` is the exact streaming precedent.
   `encode_pixels` pattern (reshape B*T) lifts to patch-level with `pooling=None`.
3. **[positional-and-infrastructure.md](findings/positional-and-infrastructure.md)** —
   `PatchEmbedding2D` (Conv2D) + `PositionEmbeddingSine2D` (channels-first!) +
   `ContinuousSinCosEmbed` for telemetry. Training template copied from
   `src/train/lewm/train_lewm.py`. `JEPAMaskingStrategy` exists for images but not
   temporal tubes — small extension if D-003 = tube masking.

## Key Constraints

### Hard
- **Clifford blocks BatchNormalization** inside context stream — unsafe at batch-of-1.
  Smoke tests must keep B ≥ 2; fallback is swap BN→LayerNorm (structural change,
  noted as risk).
- **`CausalCliffordNetBlock` requires H=1** input. Any 5D handling must reshape
  `(B, H_p, W_p, T, D)` → `(B*H_p*W_p, 1, T, D)` for the temporal pass.
- **`AdaLNZeroConditionalBlock` expects `inputs=[x,c]`** (list, length 2). Keras
  serialization of dict inputs has trapped prior plans — use list.
- **`SIGRegLayer` input convention `(..., N, D)`** — averaging over axis=-3.
  Upstream LeWM passes `(T, B, D)`. For 5D latents we must choose a reshape and
  document it (D-005).
- **Keras 3.8 serialization rules** (LESSONS): every custom class
  `@keras.saving.register_keras_serializable()` + complete `get_config()`,
  sublayers as explicit attrs (not dict).
- **GPU policy**: `CUDA_VISIBLE_DEVICES=1` (RTX 4070 12 GB), never parallel, `MPLBACKEND=Agg`.
- **Causality test = hardest case**: perturbation at temporal position k must not
  alter outputs at positions < k. This invariant MUST be tested first.

### Soft
- ViT-tiny (192d) is the default carryover from LeWM — but a hybrid encoder
  (PatchEmbed + 2-4 Clifford blocks) is parameter-cheap and better for drone
  small-object sensitivity.
- Smoke-scale defaults mirror LeWM: `num_proj=64`, small depth, B=2, short T.
- Commit cadence `[iter-N/step-M] desc`; user pushes.

### Ghost / deferred
- Real drone dataset — not in scope, synthetic only.
- Downstream fine-tuning heads (tracking, detection) — follow-up plan.
- Patch-level surprise visualization — follow-up.
- Scaling beyond smoke — follow-up.

### Key reusable components
- `src/dl_techniques/layers/geometric/clifford_block.py` — `CliffordNetBlock`,
  `CausalCliffordNetBlock`, `SparseRollingGeometricProduct`, `GatedGeometricResidual`.
- `src/dl_techniques/layers/adaln_zero.py` — `AdaLNZeroConditionalBlock`.
- `src/dl_techniques/regularizers/sigreg.py` — `SIGRegLayer`.
- `src/dl_techniques/layers/embedding/patch_embedding.py` — `PatchEmbedding2D`.
- `src/dl_techniques/layers/embedding/positional_embedding_sine_2d.py` — fixed 2D PE.
- `src/dl_techniques/layers/embedding/continuous_sin_cos_embedding.py` — telemetry PE.
- `src/dl_techniques/models/lewm/model.py` — `encode_pixels`, `rollout` patterns.
- `src/dl_techniques/models/vit/model.py` — fallback encoder (include_top=False, pooling=None).
- `src/train/lewm/train_lewm.py` + `train.common` — training script template.
- `src/dl_techniques/datasets/pusht_hdf5.py:synthetic_lewm_dataset` — synthetic
  data generator precedent.

### Exploration Confidence
- **Problem scope**: deep — drone streaming target + patch-level prediction +
  telemetry + Clifford primitives are all concrete with code references.
- **Solution space**: open — seven genuine design decisions (D-001..D-007) each
  with defensible alternatives.
- **Risk visibility**: clear — causality (test first), BN-batch-of-1, serialization
  round-trip, and AdaLN identity-at-init are all testable invariants.
