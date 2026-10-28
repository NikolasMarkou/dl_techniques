"""Variational Autoencoder — public API re-exports.

`sampling_type` selects the latent geometry: `gaussian`, `hypersphere`, or
`vmf` (a true von Mises-Fisher spherical VAE with the closed-form
vMF-to-uniform-sphere KL).
"""
from .model import VAE, compile_vae, create_vae

__all__ = [
    "VAE",
    "compile_vae",
    "create_vae",
]
