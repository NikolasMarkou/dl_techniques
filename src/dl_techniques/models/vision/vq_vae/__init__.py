"""VQ-VAE public API.

Re-exports the model class and the factory function. VQ-VAE wraps a
quantization scheme around an encoder and decoder supplied by the caller, so it
has no backbone or ``MODEL_VARIANTS`` table of its own — ``create_vq_vae`` takes
``encoder`` and ``decoder`` as required arguments instead.
"""
from dl_techniques.models.vision.vq_vae.model import VQVAEModel, create_vq_vae

__all__ = [
    "VQVAEModel",
    "create_vq_vae",
]
