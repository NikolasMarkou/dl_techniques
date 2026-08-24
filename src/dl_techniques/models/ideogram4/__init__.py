"""Ideogram4 text-to-image flow-matching DiT (Keras 3 port).

A self-contained Keras 3 reimplementation of the Ideogram4 neural core: the
flow-matching DiT transformer, the Flux2 KL-VAE, the logit-normal + Euler
scheduler/sampler, the velocity loss, an inference pipeline and training code.
Conditioning is abstracted as a precomputed `llm_features` call input — there is
no Qwen3-VL in Keras — which is the port's main deliberate divergence. See
`plans/plan_2026-06-12_59a18a10/` for the design rationale and the
"what doesn't fit / skipped / changed" report.

The transformer and autoencoder plus their builders are re-exported here.
Configuration, constants and latent normalization stay behind their submodules::

    from dl_techniques.models.ideogram4.config import get_ideogram4_config
"""
from .transformer import Ideogram4Transformer, create_ideogram4_transformer
from .vae import AutoEncoder, create_ideogram4_autoencoder

__all__ = [
    "AutoEncoder",
    "Ideogram4Transformer",
    "create_ideogram4_autoencoder",
    "create_ideogram4_transformer",
]
