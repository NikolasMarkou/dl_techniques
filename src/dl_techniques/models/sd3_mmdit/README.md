# SD3 MMDiT — dual-stream rectified-flow text-to-image transformer

This package had **no `README.md`** until 2026-08-18. It has always had
[`PORT_NOTES.md`](PORT_NOTES.md), which is the authority on *what did and did not
survive the port from PyTorch*; this file is the orientation layer and does not
restate it.

## What it is

A Keras 3 port of the MiniDiffusion SD3-style **MMDiT** (Multimodal Diffusion
Transformer): a dual-stream rectified-flow diffusion transformer over a 16-channel
spatial VAE latent, conditioned on pooled CLIP + OpenCLIP vectors and a T5 token
sequence.

**Architecture-faithful, train-from-scratch, and carrying NO pretrained weights.**
The PyTorch original is organized around weight loaders reading `.pth` checkpoints;
there is no equivalent here and no `pretrained=` path anywhere in the package —
by design, not by omission.

## Public API

```python
from dl_techniques.models.sd3_mmdit import (
    SD3MMDiT, create_sd3_mmdit,      # the diffusion transformer
    create_sd3_vae,                  # 16-channel VAE wrapper
    CLIPTextEncoder, T5Encoder,      # from-scratch text towers
    create_sd3_pipeline,             # all five components, dims pre-matched
)
```

Block and scheduler internals stay behind their submodules:

```python
from dl_techniques.models.sd3_mmdit.blocks import MMDiTBlock, MMDiTFinalLayer
```

## Presets

`config.py::PRESETS` has exactly two keys:

- **`tiny`** — smoke-trainable on a 12 GB GPU, and the one to use for anything
  interactive. `embedding_size=192`, `depth=4`, `dual_attention_layers=(0,)`, VAE
  `ch=32`.
- **`full`** — the SD3-medium-ish scale (`embedding_size=1536`, `num_heads=24`,
  `depth=24`, `dual_attention_layers=range(13)`, VAE `ch=128`). **Defined but never
  run locally**; the comment above it in `config.py` says so.

`create_sd3_pipeline(variant="tiny")` builds transformer + VAE + three text encoders
with widths chosen so the dimension contract holds (for `tiny`: T5 512, CLIP 128,
OpenCLIP 128, since `128 + 128 = pooled_projection_dim = 256`). The encoders are
deliberately shallow — the pipeline adds no trained weights and exists to exercise
the integration.

## Relationship to `ideogram4/`

The two packages share five module names and are **not** duplicates left un-merged:
`sd3_mmdit/config.py` imports `AutoEncoderParams` from `ideogram4.config`, and the
VAE is the reused `ideogram4.vae.AutoEncoder` at `z_channels=16` rather than a
second implementation. The one genuine duplication is `_validate_vae_groupnorm`
(15 lines, no divergence). One real asymmetry: `ideogram4` implements
classifier-free guidance and `sd3_mmdit` does not.

## Loss

Rectified-flow velocity MSE:
`dl_techniques.losses.flow_matching_velocity_loss.FlowMatchingVelocityLoss`. The
logit-normal timestep weighting lives in the trainer, not in the loss — see
`PORT_NOTES.md` §2.
