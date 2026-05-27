# CliffordNetPatchVAE v2

Hierarchical CliffordNet-block patch-level VAE. Sibling of
`convnext_patch_vae_v2`: same loss surface (recon / MAE / KL / SIGReg /
LPIPS / classification / segmentation), same custom `train_step`, same
public API — the only structural change is the backbone, swapped from a
**flat** ConvNeXt block stack to a **hierarchical** CliffordNet stack
with stride-2 downsampling between stages.

## Architecture

```
Input (B, H, W, C)
   |
   v   stem: Conv2D(stage_dims[0], k=patch_size, s=patch_size)
   v   stem_norm: LayerNormalization
   v
   v   [optional MAE mask at patch grid (Hp, Wp)]
   v
   v   for i in 0..N-1:
   v       stage_depths[i]  x  CliffordNetBlock(channels=stage_dims[i],
   v                                            shifts=stage_shifts[i])
   v       if i < N-1:
   v           CliffordNetBlockDSv2(channels=stage_dims[i],
   v                                out_channels=stage_dims[i+1],
   v                                strides=2, stream_pool=downsample_kind)
   v
   |- pre_bottleneck tap: (B, Hb, Wb, stage_dims[-1])    -> cls / seg heads
   |
   |- mu_head      : Conv2D(latent_dim, 1)               TruncatedNormal init
   |- log_var_head : Conv2D(latent_dim, 1)               zeros init (D-003 v1)

Sampling([mu, log_var]) -> z (B, Hb, Wb, latent_dim)

Decoder:
   z -> proj_in (1x1, stage_dims[-1])
        for k in 0..N-1 (deepest first):
            stage_depths[N-1-k] x CliffordNetBlock(channels=stage_dims[N-1-k])
            if k < N-1:
                UpSampling2D(2x bilinear) + 1x1 Conv(stage_dims[N-2-k]) + LN
   pre_head_norm + Conv2DTranspose(img_channels, k=patch_size, s=patch_size)
   -> logits (B, H, W, img_channels)
recon = sigmoid(logits) for BCE, logits for MSE
```

`Hb = H / patch_size / 2**(N-1)` is the **bottleneck per-side** size.
For CIFAR (`img_size=32, patch_size=4`) at N=2 stages, `Hp=8, Hb=4`.
For ImageNet-style 224 inputs at N=3 stages, `Hp=56, Hb=14`.

## Why hierarchical?

CliffordNet blocks are isotropic: input channels == output channels,
spatial dims preserved. Downsampling lives in `CliffordNetBlockDSv2`.
Mirroring `convnext_patch_vae_v2`'s flat single-scale design would
under-use this — hierarchical lets us match ConvNeXt's stage pattern
(96 -> 192 -> ... with spatial halving) without needing a global
attention mechanism.

## Presets

| Variant | `stage_dims`         | `stage_depths`  | `stage_shifts`            | `latent_dim` | Target img |
|---------|----------------------|-----------------|---------------------------|--------------|------------|
| tiny    | `[64, 128]`          | `[2, 2]`        | `[[1,2], [1,2,4]]`        | 8            | 32 (CIFAR) |
| base    | `[96, 192]`          | `[2, 4]`        | `[[1,2], [1,2,4]]`        | 16           | 32 (CIFAR) |
| large   | `[128, 256, 384]`    | `[2, 4, 6]`     | `[[1,2],[1,2,4],[1,2,4,8]]` | 32         | 224        |
| xl      | `[192, 384, 768]`    | `[2, 4, 8]`     | `[[1,2],[1,2,4],[1,2,4,8]]` | 64         | 224        |

Approximate parameter counts (default options, no heads):

| Variant | Params |
|---------|--------|
| tiny    | 1.0M   |
| base    | 4.3M   |
| large   | 33.0M  |
| xl      | 156.2M |

## Quick start

```python
from dl_techniques.models.cliffordnet_patch_vae_v2 import (
    CliffordNetPatchVAEV2Config,
    CliffordNetPatchVAEV2,
    create_cliffordnet_patch_vae_v2,
)

# Convenience factory:
model = create_cliffordnet_patch_vae_v2(
    "base",
    img_size=32,
    mae_mask_ratio=0.5,
    lambda_lpips=0.1,
    use_classification_head=True,
    num_classes_cls=10,
)

# Or explicit:
cfg = CliffordNetPatchVAEV2Config(
    img_size=32,
    stage_dims=[96, 192],
    stage_depths=[2, 4],
    stage_shifts=[[1, 2], [1, 2, 4]],
    mae_mask_ratio=0.5,
)
model = CliffordNetPatchVAEV2(config=cfg)
```

Training is handled by the sibling package
`src/train/cliffordnet_patch_vae_v2/`, which mirrors
`src/train/convnext_patch_vae_v2/` 1:1 with adjusted CLI flags
(`--stage-dims`, `--stage-depths`, `--stage-shifts`, `--downsample-kind`,
`--cli-mode`, `--ctx-mode`, etc.).

## Anchored decisions

See `plans/plan_2026-05-27_75849a91/decisions.md`:

- **D-001** New sibling packages, no edits to v2.
- **D-002** CliffordNetBlock owns its residual via `GatedGeometricResidual`.
  Encoder/decoder block loops must NOT add an outer `residual + x` skip
  (would double-apply). Anchored at the block-stack loops.
- **D-003** New `CliffordSegmentationHead` with explicit
  `upsample_factor = patch_size * 2**(num_stages - 1)`. Cls head is
  re-exported from v2 unchanged.
- **D-004** Sub-layer storage is a flat `self.blocks: List[Layer]` with a
  parallel `self._stage_starts: List[int]`. Nested list-of-lists breaks
  Keras layer tracking during save/load (44/143 weights silently
  diverged on round-trip).
