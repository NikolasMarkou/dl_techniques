# Video-JEPA-Clifford — Training

Self-supervised video world model for drone footage, built on CliffordNet primitives and the JEPA (Joint Embedding Predictive Architecture) objective.

**Model**: `src/dl_techniques/models/video_jepa/`
**Dataset**: `src/dl_techniques/datasets/synthetic_drone.py`
**Trainer (this directory)**: `train_video_jepa.py`
**Tests**: `tests/test_models/test_video_jepa/` (44 tests)

---

## What this trains

A patch-level latent video model that:

- **Encodes** frames with a hybrid `PatchEmbedding2D` + 2–4 `CliffordNetBlock` stack — keeps the 2D patch grid intact for geometric-product interactions.
- **Predicts** future patch latents via a factorized spatial / causal-temporal Clifford predictor (`CliffordNetBlock` per frame, `CausalCliffordNetBlock` per spatial position, alternating).
- **Conditions** the predictor on telemetry (IMU / GPS / altitude) through per-frame AdaLN-zero — separates egomotion from world dynamics.
- **Regularizes** latents with SIGReg (Sketch Isotropic Gaussian Regularizer) — collapse prevention without EMA or stop-gradient.
- **(iter-2)** Adds V-JEPA-style tube-masked latent prediction as a second training target. Single training run, three additive losses.

Streaming inference is supported via a rolling buffer of the last K patch grids (O(1) amortized per frame).

## Loss

```
total = λ_next · L_next_frame + λ_mask · L_mask + λ_sigreg · L_sigreg
```

Defaults: `λ_next=1.0`, `λ_mask=1.0`, `λ_sigreg=0.09`.

- `L_next_frame` — MSE between predicted patch latents for frame `t+1` and the encoder's output on the actual `t+1` frame.
- `L_mask` — MSE on masked patch positions between predictor output and the same encoder's output at the masked slots (latent-space masking — pixel-space is structurally incompatible with the Clifford encoder's intact-grid requirement).
- `L_sigreg` — Epps-Pulley Gaussianity statistic on `(B·T, N, D)` reshape, averaged over projections.

## Quick start

```bash
# Synthetic-data smoke test on GPU 1 (RTX 4070) — what the iter-1/2 verification runs
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.video_jepa.train_video_jepa \
    --synthetic --batch-size 2 --epochs 2 --steps-per-epoch 4 \
    --img-size 56 --patch-size 14 --seq-len 4

# With tube masking enabled (iter-2 default)
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.video_jepa.train_video_jepa \
    --synthetic --batch-size 2 --epochs 2 --steps-per-epoch 4 \
    --mask-prediction --mask-ratio 0.6 --lambda-mask 1.0 --lambda-next-frame 1.0
```

Artifacts land in `results/video_jepa_<timestamp>/` — per-epoch CSV log (total + 3 component losses), `last.keras`, `final_model.keras`.

Batch size must be ≥ 2 — `CliffordNetBlock` uses BatchNormalization in its context stream and is unsafe at batch-of-1.

## Verified behavior

44/44 tests passing. Verification order is hardest-first:

1. **Causality** — perturb frame `k ∈ [1, T-1]`; frames `< k` are identical to baseline within `1e-5`. Future-frame diff `> 1e-6` confirms the predictor isn't collapsing.
2. **SIGReg finiteness** at init and after one fit step.
3. **AdaLN-zero identity-at-init** — `max|Δ| < 1e-6` over 10 random conditioning pairs.
4. **Serialization round-trip** — all components + full model within `atol=1e-5`, including post-training reload.
5. **Shapes** across the predictor pipeline.
6. **Streaming O(1)** — buffer capped at K, shape-tested over K+2 frames (wall-clock ratio is CPU-eager noisy, structural invariant is the real guarantee).
7. **Smoke training** monotonically decreasing total loss. Per-component is noisier — see below.

Smoke runs (16 gradient steps total) are architectural correctness checks, not convergence proofs.

## Known issues / caveats

- **Loss-scale asymmetry in dual-loss mode.** At init, `L_next_frame` is ~1000× smaller than `L_mask`. With equal `λ=1.0`, predictor gradients are dominated by the mask term. In the iter-2 smoke run `L_next_frame` rose `7e-4 → 1.6e-3` while `L_mask` and total both decreased. This is below the statistical resolution of a 16-step run and was accepted as a smoke-regime artifact, not a design flaw — but it motivates loss-weight rebalancing before any serious training.
- **No EMA target encoder.** Targets come from the same live encoder (following LeWM's design choice). SIGReg is the sole collapse-prevention mechanism. Works at smoke scale on a single GPU; scaling story is unvalidated.
- **Pixel-space masking is blocked.** D-009 is a hard constraint, not a preference — a CliffordNet encoder requires an intact 2D patch grid. Encode-then-mask is the only viable path under the current encoder choice.
- **Small-magnitude loss monotonicity.** 2-epoch / 16-step smoke runs are too short to assert monotonicity on `L_next_frame`. Use absolute-bound finite-ness guards in tests, not strict monotone decrease.
- **Synthetic data only.** `synthetic_drone.py` generates moving-shapes video + plausible telemetry; real drone HDF5 hookup is iter-3+ work.
- **Patch grid must be ≥ 2 on each axis** for the spatial Clifford block's depthwise convolutions. Smoke uses `img_size=56, patch_size=14` → 4×4 patches.

## Future work

Queued in `plans/DECISIONS.md` as FW-001 … FW-004 from the closed plan (`plan_2026-04-21_421088a1`):

- **FW-001** — **λ rebalance**. Try `λ_next_frame ≈ 2.0` or use learned loss balancing (e.g. uncertainty weighting) to stop the mask term from starving the next-frame term.
- **FW-002** — **Longer training** (5+ epochs, real dataset) before declaring any monotonicity properties.
- **FW-003** — **EMA target encoder** variant as an ablation against live targets + SIGReg.
- **FW-004** — **Predict-the-mean shortcut probe.** Verify `L_mask` is decreasing via per-position semantic recovery and not via a trivial latent-mean predictor.

Additional directions (not yet formally queued):

- Real drone HDF5 loader + telemetry sync harness (replaces the synthetic generator).
- Per-patch surprise maps for online anomaly detection during streaming.
- Altitude-aware conditioning ablation — does AdaLN on `altitude_bin` alone beat full-telemetry conditioning, or does IMU matter?
- Downstream heads on the frozen encoder — tracking, detection, change-point segmentation.
- Asymmetric context/target views (context = low-res, target = hi-res).
- Mixed-precision (bf16) and multi-GPU training.
- Ablations: SIGReg on/off, Clifford-depth sweep (2 vs 4 blocks), spatial-vs-spatiotemporal tube masks.

## References

- **Closed plan**: `plans/plan_2026-04-21_421088a1/` — `summary.md`, `decisions.md` (D-001 … D-012, FW-001 … FW-004), `verification.md`.
- **Model package**: `src/dl_techniques/models/video_jepa/CLAUDE.md` (if present) or module docstrings.
- **Upstream LeWM port**: `src/train/lewm/` — single-CLS JEPA, same SIGReg + AdaLN primitives, simpler predictor. The video-JEPA model generalizes it to patch level.
- **Clifford primitives**: `src/dl_techniques/layers/geometric/clifford_block.py` — `CliffordNetBlock`, `CausalCliffordNetBlock`, `SparseRollingGeometricProduct`, `GatedGeometricResidual`.
- **Cross-plan lessons**: `plans/LESSONS.md`.
