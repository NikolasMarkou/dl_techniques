"""SD3 MMDiT (Stable Diffusion 3) dual-stream text-to-image package.

This package hosts the SD3-style MMDiT transformer block, the diffusion
transformer model, a rectified-flow scheduler, a 16-channel VAE wrapper,
from-scratch CLIP/OpenCLIP/T5 text encoders, and an inference pipeline.

Per house convention this ``__init__`` exports nothing; import directly from
the submodules, e.g.::

    from dl_techniques.models.sd3_mmdit.blocks import MMDiTBlock, MMDiTFinalLayer
"""
