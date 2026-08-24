"""SD3 MMDiT dual-stream text-to-image — public API re-exports.

Hosts the SD3-style MMDiT block, the diffusion transformer, a rectified-flow
scheduler, a 16-channel VAE wrapper, from-scratch CLIP/OpenCLIP/T5 text encoders
and an inference pipeline.

The top-level models, text encoders and builders are re-exported here. The block
and scheduler internals stay behind their submodules, e.g.::

    from dl_techniques.models.sd3_mmdit.blocks import MMDiTBlock, MMDiTFinalLayer
"""
from .pipeline import create_sd3_pipeline
from .text_encoders import CLIPTextEncoder, T5Encoder
from .transformer import SD3MMDiT, create_sd3_mmdit
from .vae import create_sd3_vae

__all__ = [
    "CLIPTextEncoder",
    "SD3MMDiT",
    "T5Encoder",
    "create_sd3_mmdit",
    "create_sd3_pipeline",
    "create_sd3_vae",
]
