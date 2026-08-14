"""Rotation-trick VQ-VAE — public API re-exports.

There is no ``MODEL_VARIANTS`` table and none was invented: Fifty et al. publish
a quantizer, not a backbone family, and this package's auto-built convolutional
encoder/decoder is parameterized continuously by ``hidden_channels``,
``downsample_factor`` and ``num_res_blocks`` with no published named scales.
``create_vq_vae_rotation`` therefore constructs the class directly rather than
delegating to a ``from_variant``.
"""
from dl_techniques.models.vq_vae_rotation.model import (
    VQVAERotationTrick,
    create_vq_vae_rotation,
)

__all__ = [
    "VQVAERotationTrick",
    "create_vq_vae_rotation",
]
