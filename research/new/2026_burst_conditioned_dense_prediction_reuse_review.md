# BurstDP — Codebase Reuse Review

Date: 2026-05-19
Companion to: `2026_burst_conditioned_dense_prediction.md`
Scope: Existing `dl_techniques` components evaluated for reuse in BurstDP's three
substitutable surfaces — **(a) shared encoder, (b) ref→aux fusion block, (c)
heads / decoder side** — plus training infrastructure.

---

## TL;DR — three adoptions, three close-calls, three skips

### Adopt now (low risk, real benefit)

| Component | Replaces / augments | Why |
|-----------|--------------------|-----|
| **`PerceiverTransformerLayer`** (`layers/transformers/perceiver_transformer.py`) | The custom `BurstFusionBlock` cross-attention | Same asymmetric ref-as-query / aux-as-KV semantics we already implement, but battle-tested, serializable, and ~100 lines less custom code. **Capability-equivalent, not an upgrade** — adopt for hygiene, not for new behaviour. |
| **`ConvNextV2Block`** as a hierarchical encoder option (`layers/convnext_v2_block.py`) | Plain ViT encoder, as an alternative preset | DPT decoders consume multi-scale features. A 4-stage ConvNeXt-V2 encoder emits `[C, 2C, 4C, 8C]` skip features that the recon + seg + depth heads can fuse properly. Today we throw away half the value of DPT by feeding it a flat token stream. |
| **video_jepa training conventions** (`train/video_jepa/train_video_jepa.py`) | Our current `train_burst_dp.py` skeleton | Three concrete reusable patterns: per-head `keras.metrics.Mean` trackers, callback ordering (`TerminateOnNaN` first, then checkpoint, then visualisation), and the `tf.data.Dataset.from_generator(...) + steps_per_epoch` pattern that avoids the "dataset exhausted after epoch 1" bug we are currently exposed to via PyDataset. |

### Close calls (worth a follow-up pass, not blocking)

| Component | What it would buy us | Why deferred |
|-----------|----------------------|--------------|
| `AdaLNZeroConditionalBlock` (`layers/transformers/adaln_zero.py`) | Cheap per-head aux conditioning: modulate decoder features by a pooled aux summary instead of (or alongside) full cross-attention. Identity-at-init means it cannot hurt convergence. | Only useful if we observe attention-collapse (H3) or want a lightweight head variant. Add in v2. |
| `AccUNet` HANC blocks (`models/accunet/`) / `SCUNet` decoder (`models/scunet/`) | Better dense-prediction decoders than DPT for segmentation (HANC) and reconstruction (SCUNet has the no-bias, restoration-tuned upsample path). | DPT decoder is fine for v1; revisit when we have eval numbers to justify the swap. |
| `LatentMaskOverlayCallback` / `PatchPredictionErrorCallback` patterns (in video_jepa training) | Per-head qualitative visualisation during training (recon grid, seg overlay, depth heatmap). | Polish, not correctness. Add when we run the first real training. |

### Skip (audited, wrong fit)

| Component | Why it doesn't fit |
|-----------|--------------------|
| `CliffordNetBlock` (`layers/geometric/clifford_block.py`) | The block's "equivariance" is to **channel-shift symmetries** in the feature dimension, NOT to spatial 2D rotations/translations. Our aux views are warped by image-space SE(2); Clifford adds ~600K params per block and addresses the wrong symmetry. If aux alignment becomes a measured problem, prefer a 10K-param positional-bias / spatial-transformer route over Clifford. |
| `EoMTTransformer` (`layers/transformers/eomt_transformer.py`) | "Masked attention" is a soft modulation, not hard masking on scores; it also requires ground-truth attention masks (e.g. from optical flow). We don't have them. |
| `FreeTransformer`, `TextEncoder`, `TextDecoder`, `ProgressiveFocusedTransformer` | Wrong domain (LM with VAE bottleneck, NLP, SR-specific). `ProgressiveFocusedTransformer` requires `prev_attn_map` plumbing across blocks that doesn't fit our decoder pipeline cleanly. |

---

## Detailed assessment by surface

### (a) Shared encoder

Current: `dl_techniques.models.vit.ViT` (`include_top=False, pooling=None`) returning `(B, 1+T, D)`.

| Candidate | Score | Notes |
|-----------|-------|-------|
| Plain `ViT` (current) | baseline | Flat token output; DPT heads must upsample 16x from a single scale. Loses the "U-Net-shaped" benefit DPT was designed for. |
| `VisionEncoder` (`layers/transformers/vision_encoder.py`) | STRONG | Drop-in cleaner replacement for `ViT` — configurable patch embedding (linear / SigLIP / conv stem) and pooling. Same flat-token output, so no architectural benefit over ViT, but better hygiene. |
| Hierarchical `ConvNextV2Block` stack | **STRONG** | Build a 4-stage encoder (`ConvNeXt-Pico/Tiny` style) that emits `[H/4, H/8, H/16, H/32]` feature maps. Cross-attention fusion happens at the deepest stage; DPT decoder gets real skip features. Recommended as a new `burst_dp_convnext_small` preset *in addition to* the ViT one, so we can A/B them. |
| Full `ConvUNeXt` model (`models/convunext/`) | POSSIBLE | "U-Net + ConvNeXt V2 + deep supervision" complete model — could *replace* both encoder AND DPT decoder. Trade-off: loses the global receptive field of attention, which segmentation likes. Worth running as a third preset. |
| `SwinTransformerBlock` / `SwinConvBlock` | WEAK | Local-window attention is a poor fit for aux fusion (views are *not* spatially aligned with the reference). Keep on the shelf. |
| `HANCBlock` (`layers/hanc_block.py`) | POSSIBLE | Hierarchical pooling approximates self-attention without quadratic cost; useful inside the encoder if we go conv-only. |

**Recommendation**: in v1.1, add a `burst_dp_convnext_small` preset that uses a 4-stage ConvNeXt-V2 encoder with cross-attention fusion at the bottleneck and the existing DPT decoder reading 4-scale skips. Keep the ViT preset as the control.

### (b) Ref→aux fusion block (the architectural core of BurstDP)

Current: `BurstFusionBlock` — pre-norm self-attn on ref → pre-norm masked cross-attn(ref→aux, flat KV with broadcast mask) → FFN.

| Candidate | Score | Notes |
|-----------|-------|-------|
| Current `BurstFusionBlock` | baseline | Correct semantics, NaN-safe at N=0 via explicit gate. Confirmed by smoke tests. |
| `PerceiverTransformerLayer` (`perceiver_transformer.py`) | **STRONG (hygiene)** | Same shape contract (`query_input, kv_input`), same asymmetric pattern, less custom code. **Not a behavioural upgrade.** Adoption is purely about code surface area. Must verify it handles our `(B, N*T)` padding mask cleanly — agent flagged that mask propagation is internal and needs a probe test. |
| `TransformerLayer` (`transformers/transformer.py`) | WEAK | Generic block, designed for self-attention; would have to add manual cross-attn wiring. Net negative. |
| `AdaLNZeroConditionalBlock` (`adaln_zero.py`) | POSSIBLE-ALT | Replace cross-attention with cheap modulation by `c = avg_pool(aux_tokens)`. Identity-at-init → safe to slot in. Fundamentally a lower-capacity fusion (sees aux only via a pooled summary), so it's a different design point, not a drop-in. Good candidate for an ablation. |
| `EoMTTransformer` | WEAK | Needs supervisory masks we don't have. |
| `CliffordNetBlock` | SKIP | See above — wrong symmetry. |

**Recommendation**: leave the current `BurstFusionBlock` in place for v1. In v1.1, ship a `fusion_type` config knob with three values: `custom` (current), `perceiver` (use `PerceiverTransformerLayer`), `adaln` (use `AdaLNZeroConditionalBlock` with pooled aux). Wire all three through the same `BurstDPConfig.fusion_type` field and ablate.

### (c) Heads / decoder side

Current: three `DPTDecoder` heads sharing a 1x1-projected fused-ref feature map.

| Candidate | Score | Notes |
|-----------|-------|-------|
| Current `DPTDecoder` (`models/depth_anything/components.py`) | baseline | Works, simple. Underutilised because we feed it a single-scale feature map (no skip features). |
| AccUNet decoder pattern (`models/accunet/model.py`, lines 251-442) | STRONG (for seg/depth) | 4-level decoder with `Conv2DTranspose(2x)` + HANC blocks + ResPath + MLFC cross-level fusion. Designed for dense prediction. Worth a head-swap A/B for seg + depth specifically. |
| SCUNet decoder pattern (`models/scunet/model.py`, lines 213-336) | STRONG (for recon) | `Conv2DTranspose(2x) + Swin blocks + skip residual addition`, no-bias by design. Restoration-tuned. Worth using *only* for the recon head. |
| `AdaLNZeroConditionalBlock` inside heads | POSSIBLE | Modulate per-head features by a global aux summary — cheap "task-aware refinement". |
| `ConvNextV2Block` for head refinement | STRONG (cheap) | Drop-in residual refinement block between DPT stages; minimal cost, GRN improves feature diversity for detail-heavy outputs (depth edges, fine seg boundaries). |
| `SwinConvBlock` for head refinement | POSSIBLE | Hybrid conv+attn; only if we want both local texture and global structure in the heads. |
| `PixelShuffle` (`layers/pixel_shuffle.py`) | POSSIBLE | Alternative to bilinear upsample in DPT; reduces grid artefacts. Cosmetic, low priority. |
| `convolutional_kan` | WEAK | Experimental, much slower than standard conv. Park. |
| `mobile_one_block`, `repmixer_block` | WEAK | Lightweight blocks aimed at inference latency, not the limiting factor here. |

**Recommendation**: ship per-head DPT for v1. In v2, optionally replace the recon head with the SCUNet decoder pattern (no-bias is right for denoising) and the seg/depth heads with AccUNet-style HANC decoders that consume real skip features from a hierarchical encoder.

### Training infrastructure (video_jepa)

| Pattern | Score | Action |
|---------|-------|--------|
| Per-head `keras.metrics.Mean` loss trackers exposed via `Model.metrics` (video_jepa `model.py` lines 225-266) | STRONG | Adopt. We currently emit one scalar `val_loss`; replace with `val_recon_loss`, `val_seg_loss`, optionally `val_depth_loss`. Needed for monitoring multi-task balance + attention-collapse diagnosis. |
| Callback ordering: `TerminateOnNaN` → `CSVLogger` → `ModelCheckpoint(last + best)` → visualisation callbacks (`_build_callbacks` lines 107-161) | STRONG | Adopt. Our current order misses `TerminateOnNaN`, which is a real risk for the masked-softmax path when training goes off the rails. |
| `tf.data.Dataset.from_generator(...) + steps_per_epoch` (README.md lines 197-206) | STRONG | Adopt. Our current `pyds_to_tfds()` finite generator silently exhausts after one epoch — a real bug if the user runs >1 epoch. Switch to an indefinite generator + explicit `steps_per_epoch=len(train_ds)`. |
| `LatentMaskOverlayCallback` / `PatchPredictionErrorCallback` (caches an eval batch, wraps figure save in try/except + gc.collect) | POSSIBLE | Adopt the *pattern* — write `BurstReconCallback`, `BurstSegCallback`, `BurstDepthCallback` along the same lines. |
| Shared-encoder flattening `(B,T,H,W,C) → (B*T,H,W,C)` (`model.py` lines 89-104, 164) | already used | We already do the same trick in `BurstDP.call`. No change. |
| `TubeMaskGenerator` (`masking.py`) | SKIP-as-is, ADAPT-pattern | Video-JEPA masks are *spatial* (per-pixel) and broadcast across time; BurstDP masks are *per-view*. The argsort-of-uniform-noise idiom (lines 127-143) generalises to "sample K view indices per sample" — worth lifting if we want stochastic view dropout during training. |

---

## Concrete next-step plan (proposed)

Three independent, low-risk patches we can land in any order:

1. **Training-script upgrade** (no model change):
   - Add `TerminateOnNaN` as the first callback.
   - Switch to `from_generator + steps_per_epoch`.
   - Expose per-head loss metrics (`recon_loss`, `seg_loss`).
   - Mirror `video_jepa`'s callback ordering.
2. **Fusion type knob**:
   - Add `fusion_type: Literal["custom", "perceiver", "adaln"]` to `BurstDPConfig`.
   - Wire `PerceiverTransformerLayer` and `AdaLNZeroConditionalBlock` as drop-in alternatives.
   - Extend the smoke test to forward-pass all three under N=0/k/N_max.
3. **ConvNeXt encoder preset**:
   - Add a hierarchical `ConvNextV2Block` encoder option that emits 4-scale features.
   - New preset `burst_dp_convnext_small`; existing ViT presets unchanged.
   - DPT decoders read 4-scale skip features properly.

Each is independent. (1) gives us real-run safety. (2) gives us the ablation lever for H1/H3 (cross-attention vs. modulation vs. baseline). (3) addresses the "DPT is starved of multi-scale features" structural weakness.

Explicit non-goals: Clifford geometric block (wrong symmetry, ~600K params/block for a problem better solved by a 10K-param positional bias or a spatial transformer), EoMT masked attention (needs supervisory masks), and any of the LM-side transformer variants (`FreeTransformer`, `TextEncoder`, `TextDecoder`).

---

## File paths referenced

```
src/dl_techniques/layers/transformers/perceiver_transformer.py
src/dl_techniques/layers/transformers/adaln_zero.py
src/dl_techniques/layers/transformers/vision_encoder.py
src/dl_techniques/layers/transformers/swin_transformer_block.py
src/dl_techniques/layers/transformers/swin_conv_block.py
src/dl_techniques/layers/convnext_v1_block.py
src/dl_techniques/layers/convnext_v2_block.py
src/dl_techniques/layers/hanc_block.py
src/dl_techniques/layers/geometric/clifford_block.py    # SKIP
src/dl_techniques/models/convunext/model.py
src/dl_techniques/models/accunet/model.py
src/dl_techniques/models/scunet/model.py
src/dl_techniques/models/depth_anything/components.py    # current DPTDecoder
src/dl_techniques/models/video_jepa/{model.py, encoder.py, masking.py}
src/train/video_jepa/train_video_jepa.py
```

[STATE: reuse review complete | adoptions: 3 strong + 3 close-calls + 3 skips |
next step: land patch (1) training-script upgrade as a no-architecture-change PR]
