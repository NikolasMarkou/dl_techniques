"""Ideogram4 text-to-image flow-matching DiT, ported to Keras 3.

Exports the flow-matching DiT transformer and the Flux2 KL-VAE. The package
also holds the logit-normal plus Euler scheduler and sampler, the velocity
loss, an inference pipeline, and training code, imported from their own
submodules. Conditioning is a precomputed `llm_features` call input rather
than a live Qwen3-VL model, since Qwen3-VL has no Keras implementation.

The transformer and autoencoder plus their builders are re-exported here.
Configuration, constants and latent normalization stay behind their submodules::

    from dl_techniques.models.vision_language.ideogram4.config import get_ideogram4_config
"""
from .transformer import Ideogram4Transformer, create_ideogram4_transformer
from .vae import AutoEncoder, create_ideogram4_autoencoder

__all__ = [
    "AutoEncoder",
    "Ideogram4Transformer",
    "create_ideogram4_autoencoder",
    "create_ideogram4_transformer",
]
