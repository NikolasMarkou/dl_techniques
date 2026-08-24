"""Latent-GMM point cloud registration — public API re-exports.

There is no ``MODEL_VARIANTS`` table here and none was invented: the paper
describes one network, and the only scale knobs (``num_gaussians``,
``k_neighbors``) are plain constructor arguments with no published named
configurations to enumerate. ``create_latent_gmm_registration`` therefore
constructs the class directly instead of delegating to a ``from_variant``.

The two module-level helpers are exported as well because they are the
differentiable GMM/Procrustes primitives the model is built from and are
useful (and separately tested) on their own.
"""
from dl_techniques.models.latent_gmm_registration.model import (
    LatentGMMRegistration,
    create_latent_gmm_registration,
    compute_gmm_params,
    compute_rigid_transform,
)

__all__ = [
    "LatentGMMRegistration",
    "create_latent_gmm_registration",
    "compute_gmm_params",
    "compute_rigid_transform",
]
