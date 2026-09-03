"""Rotation-trick VQ-VAE public API.

Re-exports the model class and the factory function. Fifty et al. publish a
quantizer, not a backbone family, so this package has no ``MODEL_VARIANTS``
table — its convolutional encoder and decoder scale continuously through
``hidden_channels``, ``downsample_factor`` and ``num_res_blocks``, and
``create_vq_vae_rotation`` constructs the class directly rather than
delegating to a ``from_variant``.
"""
from dl_techniques.models.vision.vq_vae_rotation.model import (
    VQVAERotationTrick,
    create_vq_vae_rotation,
)

__all__ = [
    "VQVAERotationTrick",
    "create_vq_vae_rotation",
]
