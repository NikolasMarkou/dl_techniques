# 2026 — Multi-View Reference-Conditioned Vision (`BurstDP`)

Author: Nikolas Markou
Date: 2026-05-19
Status: v1 model + training pipeline scaffolded and smoke-tested.
        End-to-end training runs are pending compute time.

[STATE: Plan executed | Tier: STANDARD | Lead H1=0.70 (architecture choice),
H3=0.30 (attention collapse), H5=0.40 (sim-real gap) | Confidence: Med]

---

## 1. Problem Statement

Conventional vision models discard a strong prior available in real-world
capture: a single physical scene yields *a sequence of correlated views*
under varying nuisance parameters. Drone bursts, automotive sensors and
multi-exposure stills generate one frame we want to commit to (the
**reference**) plus 1-5 nearby observations (**auxiliary views**) sampled
microseconds apart, under different angles, motion blur realisations, and
noise. None of this complementary information is used by single-image
denoising / segmentation / depth models.

We design a model that:

- consumes one reference + a *variable-size* set of 1-5 auxiliary views;
- jointly produces three dense outputs about the reference:
  1. **fidelity reconstruction** (denoise / deblur / restore)
  2. **per-pixel semantic segmentation**
  3. **monocular depth (with multi-view priors)**
- handles permutation-invariance and variable N at runtime by construction
  (a flexible deployment surface, not just a benchmark trick).

The auxiliary set is never assumed clean — the model must extract usable
signal from corrupted neighbours, the realistic case for sensor bursts.

---

## 2. Related Work (selected; not exhaustive)

| Area | Representative | What they share with BurstDP | Delta |
|------|----------------|------------------------------|-------|
| Burst super-resolution | Bhat et al. (Deep Burst SR / BIPNet), Lecouat et al. | multi-frame fusion for fidelity | single-task, often fixed N, no semantics |
| Multi-frame denoising | KPN, NAFNet-burst | same | same |
| Reference-based SR | DCSR, MASA-SR | privileged anchor + reference; cross-attention | exactly 1 reference image; no segmentation/depth |
| Video object segmentation | XMem, STM | reference frame + temporal context for seg | temporal ordering assumed; not used for fidelity/depth |
| Multi-view stereo / MVS | MVSNet, NerF-style | parallax for depth from many views | depth-only; requires known/calibrated poses |
| Self-supervised multi-view | JEPA, V-JEPA, DINO, MAE | shared encoder over multiple views | embedding-only; no dense heads |
| Monocular-with-priors depth | DepthAnything, MiDaS | dense depth from a single view | single image; no aux conditioning |

**Gap BurstDP occupies**: none of these jointly do reconstruction +
segmentation + depth with a variable-N, permutation-invariant auxiliary set
over a privileged anchor.

---

## 3. Hypotheses (priors set at protocol Phase 0; updated as evidence arrives)

| ID | Statement | Prior | Status post-scaffold |
|----|-----------|-------|----------------------|
| H1 | Asymmetric ref-as-query cross-attention beats early channel-concatenation for fidelity, because the problem is genuinely asymmetric. | 0.70 | unchanged (no training evidence yet) |
| H2 | Aux views help **depth > segmentation > fidelity** (parallax is the strongest geometric signal). | 0.55 | unchanged |
| H3 (adversarial) | The model degenerates to ignoring aux tokens (attention collapse). | 0.30 | unchanged; attention-entropy logging is the planned probe |
| H4 (`[H_S_prime]`) | Pseudo-depth teacher (DepthAnything) is the binding constraint: depth quality saturates at teacher quality. | 0.35 | dormant in v1 (no depth) |
| H5 | Synthetic aux is too easy — the model learns to invert known distortions, not perform real multi-view fusion. Held-out real bursts collapse. | 0.40 | unchanged; planned OOD probe on BDD100K |

These priors are tracked in `analyses/<session>/hypotheses.json`.

---

## 4. Architecture

```
ref  (B, H, W, 3) ─┐
                    ├─► shared ViT encoder ─► ref tokens (B, T, D)
aux  (B, N, H, W, 3) ─►                       aux tokens (B, N, T, D)
                                              + aux_mask (B, N)
                          │
                          ▼
        BurstFusionBlock × M
          pre-norm self-attn(ref)
          pre-norm cross-attn(query=ref, kv=flat(aux), mask=expanded_aux_mask)
            └ output gated to zero for samples with no valid aux (N=0)
          pre-norm FFN
                          │
                          ▼
            reshape tokens to (B, h, w, D)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
     recon head        seg head        depth head*
   (DPT, sigmoid)   (DPT, logits)   (DPT, linear, opt-in)
   (B,H,W,3)        (B,H,W,81)        (B,H,W,1)
```

`*` depth head built only when `BurstDPConfig.enable_depth=True` (v1 default
is `False`; no DepthAnything teacher on disk yet).

### Key design choices

| Choice | Rationale |
|--------|-----------|
| Shared ViT encoder for ref + aux | Aux views are the same modality as the ref. Independent encoders waste parameters and break feature-space alignment for cross-attention. |
| Ref as query / aux as key-value | The task is asymmetric — outputs are about the reference, not a symmetric set summary. |
| Flatten aux tokens to `(B, N*T, D)` + 1D padding mask | Cross-attention over a flat key sequence is order-invariant by construction; the mask handles variable N. Simpler than padding to N_max with special tokens. |
| N=0 gate on cross-attention output | Avoids NaN from all-masked softmax; degrades gracefully to single-image inference. |
| Three DPT-style heads share fused tokens | Forces the fusion stack to be useful — without shared decoding the losses cannot punish/reward fusion. |
| Variable N at train time (`sample_n_per_sample=True`, N ∈ [1, 5]) | The model sees the full deployment-time distribution every epoch. |

Files:
- `src/dl_techniques/models/burst_dp/config.py`
- `src/dl_techniques/models/burst_dp/fusion.py`
- `src/dl_techniques/models/burst_dp/heads.py`
- `src/dl_techniques/models/burst_dp/model.py`

Preset sizes (initial run targets `burst_dp_small`):

| Preset | Encoder | Fusion blocks | Approx params |
|--------|---------|---------------|---------------|
| pico  | ViT-Pico (192d, 6L)   | 2 | ~6 M  |
| tiny  | ViT-Tiny (192d, 12L)  | 3 | ~12 M |
| small | ViT-Small (384d, 12L) | 4 | ~50 M |
| base  | ViT-Base (768d, 12L)  | 6 | ~140 M |

---

## 5. Data Pipeline

Underlying source: `COCO2017MultiTaskLoader` (already in the repo) produces
`(image, {classification, segmentation})` from a local COCO checkout.
New wrapper `COCO2017BurstDPLoader` (`src/dl_techniques/datasets/vision/coco_burst_dp.py`)
generates, per sample:

- `ref`: anchor image with **moderate** distortion (model input)
- `aux[N]`: 1..N_max distinct corruptions of the same anchor with **strong**
  distortion + always-on geometric warp
- `aux_mask[N_max]`: 1.0 for real views, 0.0 for padding slots
- `recon`: the **clean** anchor (reconstruction target)
- `segmentation`: 81-way class indices (unchanged from base loader)
- `depth` + `has_depth`: zeros + 0 in v1 (no teacher); future v2 will read
  pseudo-depth `.npy` files cached from DepthAnything

### Distortion families

(implemented from PIL + numpy primitives in `coco_burst_dp.py`)

| Family | Params (anchor → aux) |
|--------|------------------------|
| Gaussian noise | σ ∈ [5,30]/255 → σ ∈ [5,40]/255 |
| Brightness / contrast jitter | ±0.05 / ±0.10 → ±0.12 / ±0.20 |
| Gaussian blur | σ ∈ [0, 1.2] → σ ∈ [0, 2.5] |
| Motion blur | p=0.2, len 3-11 → p=0.4, len 3-21 (uniform-angle) |
| Occlusion (mean-grey boxes) | p=0.1, ≤5% area, 1-2 boxes → p=0.3, ≤15% area, 1-3 boxes |
| Affine warp (rotation + translate + scale) | **disabled** for anchor → ±10°, ±8% translate, ±5% scale for aux |

The anchor never receives a geometric warp — its label coordinates are
defined in anchor pixels and the dense heads only predict in this frame.
Aux warps are stochastic and never inverted; the model fuses them via
attention without an explicit alignment module.

---

## 6. Losses and Training Protocol

| Head | Loss | Initial weight |
|------|------|----------------|
| Reconstruction | Charbonnier (√(diff² + ε²)) | 1.0 |
| Segmentation | Sparse softmax cross-entropy from logits, 81 classes | 1.0 |
| Depth (v2) | `AffineInvariantLoss` (already in repo), gated by `has_depth` | 0.5 |

Optimizer: AdamW (lr 3e-4, wd 0.05). Schedule: 1-epoch warmup + cosine
decay over the remaining epochs. Mixed precision (fp16) on RTX 4090.

Initial training resolution: **256² (v1 smoke), 384² once converged**
(per user decision). Batch size 4 at 256² fits comfortably with mixed
precision; ~2 at 384².

Verified end-to-end via `tests/test_models/test_burst_dp/test_burst_dp_smoke.py`
(9 tests, 19.9 s on CPU):

- model builds at pico preset
- variable N (N=0, N=k<N_max, N=N_max) produces finite outputs (no NaN)
- depth head can be turned on/off independently
- one training step changes every trainable variable (no zero-gradient
  components)
- `model.save()` / `keras.models.load_model()` round-trip preserves shapes
- `BurstDPConfig.to_dict()` / `from_dict()` round-trip is the identity
- `BurstFusionBlock` produces finite outputs even with all-zero mask

---

## 7. Experiment / Verification Plan

| Test | Method | Hypothesis | Pass criterion |
|------|--------|-----------|----------------|
| N-sweep | Eval on COCO val with N ∈ {0,1,2,3,4,5} from the same anchor pool | H1, H2 | Metric monotone-non-decreasing in N up to ≥3; N=5 > N=0 by >5% on at least one head |
| Single-task ablation | Train recon-only / seg-only baselines on same compute | H1 | Multi-task ≥ single-task on ≥2 of 2 heads (v1); no head regresses >2% |
| Attention entropy | Log per-head, per-block mean cross-attention entropy on val every 5 epochs | H3 | Mean entropy stays above 50% of theoretical max; falling below = collapse, claim fails |
| Distortion-severity sweep | Eval at light / moderate / heavy aux corruption | H5 | Gain over single-image holds at moderate severity; report honestly at heavy |
| N=0 fallback at deploy | Run inference with empty aux | constraint | No NaN/Inf; metrics ≈ single-image baseline ±1% (already verified at scaffold level via test_variable_n_aux_no_nan) |
| OOD sanity | 100 BDD100K triplets as natural aux | H5 | Numbers reported with explicit "real bursts" caveat; recon/seg do not blow up |
| Posterior update | After each result, single hypothesis update with capped LR (≤3 in early phases) | All | Updates logged in `analyses/.../hypotheses.json` |

Stopping criterion: either H1 confirmed (>0.90) + H3 refuted (<0.20),
or H3 confirmed (>0.90) and the team pivots to investigating the collapse.

---

## 8. Decisions (X at the cost of Y)

| Decision | Trade-off |
|----------|-----------|
| Shared encoder for ref + aux | Parameter sharing + feature-space alignment, **at the cost of** decoupled aux feature spaces (mitigated by fusion-stack capacity). |
| Synthetic aux generation from COCO | Reproducibility and free labels, **at the cost of** sim-real gap on real bursts (H5 stays alive). |
| No depth supervision in v1 | Ships now without a DepthAnything pseudo-depth pipeline, **at the cost of** the depth head being inactive until v2 (architecture is already wired and gated). |
| COCO-only, single dataset | Simpler bring-up, **at the cost of** generalisation evidence; planned OOD eval on BDD100K. |
| Heads use one shared 1×1 projection + DPT decoder per task | Compact head footprint, **at the cost of** the heads competing for the same fused representation (multi-task interference if loss weights are mis-tuned). |
| Permutation invariance via mask + flatten, no positional embedding on aux | Honest variable-N semantics, **at the cost of** ignoring any structure that real bursts have (e.g. temporal ordering); if temporal cues matter later, a small per-view embedding can be added. |

---

## 9. Lessons / Open Risks

- **Attention collapse (H3)** is the most insidious failure mode: the model
  could nominally "work" while ignoring aux tokens, with the gains
  attributable to encoder capacity alone. Per-block cross-attention entropy
  is the planned diagnostic; if it drifts toward uniform-or-spike, fusion
  is decorative.
- **Distribution gap (H5)**: synthetic aux is well-defined and reproducible,
  but trivially predictable in the sense that the model can invert the
  known warp + noise process. Real bursts have unknown geometric jitter +
  sensor-specific noise that the model never sees. Without a real-burst
  test set, claimed gains overstate field performance.
- **Multi-task balance**: three losses share a single backbone; the seg
  head's softmax dominates gradients early in training. Need to monitor
  per-head loss trajectories and re-balance weights once the warm-up
  epoch is complete.
- **Depth teacher ceiling (H4, dormant)**: when v2 adds pseudo-depth, the
  depth metrics are bounded by DepthAnything quality on COCO scenes (which
  are not its training distribution). Treat depth numbers as relative,
  not absolute.

---

## 10. Files Produced

| Path | Purpose |
|------|---------|
| `src/dl_techniques/models/burst_dp/` | Model package (config + fusion + heads + model + README) |
| `src/dl_techniques/datasets/vision/coco_burst_dp.py` | COCO multi-view loader with synthetic aux generation |
| `src/train/burst_dp/train_burst_dp.py` | Training entry point + README |
| `tests/test_models/test_burst_dp/test_burst_dp_smoke.py` | 9 smoke tests (model build, variable N, gradient flow, save/load) |
| `analyses/<session>/` | Epistemic-deconstructor session tree (this doc + state files) |
| `research/new/2026_ref_image.md` | This document |

---

## 11. Execution Status

- [x] Model + loader + train script scaffolded
- [x] 9/9 smoke tests pass (CPU, 19.9 s)
- [ ] 1-epoch sanity training on RTX 4090 (GPU 0, batch 2, image 256) — pending user authorisation; serial GPU policy means user owns long runs
- [ ] N-sweep evaluation
- [ ] Attention-entropy diagnostics
- [ ] OOD sanity on BDD100K

[STATE: Phase 5 partial | Tier: STANDARD | Hypotheses pending evidence
update from training | Confidence: Med — architecture verified, claims
about multi-view utility are not yet tested]
