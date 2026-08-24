"""VQ-VAE — public API re-exports.

There is no ``MODEL_VARIANTS`` table and none was invented: VQ-VAE is a
quantization scheme wrapped around an arbitrary autoencoder, and this package
takes the encoder and decoder as constructor arguments rather than defining a
backbone of its own. With no architecture to scale there are no named scales to
enumerate, so ``create_vq_vae`` constructs the class directly and keeps
``encoder``/``decoder`` required.
"""
from dl_techniques.models.vq_vae.model import VQVAEModel, create_vq_vae

__all__ = [
    "VQVAEModel",
    "create_vq_vae",
]
