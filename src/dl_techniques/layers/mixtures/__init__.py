"""Differentiable clustering / mixture layers.

This sub-package groups the soft, end-to-end differentiable clustering and
mixture layers behind a single import surface, mirroring the `attention/`,
`norms/`, and `ffn/` sub-packages (flat re-exports + a config-driven
`factory.py`).

Layers
------
- :class:`RBFLayer` — Radial Basis Function layer with learnable centers and an
  inter-center repulsion mechanism.
- :class:`KMeansLayer` — Differentiable soft K-means assignment layer with
  online centroid updates.
- :class:`GMMLayer` — Differentiable Gaussian Mixture layer producing soft
  component responsibilities with an isometric regularizer.

Factory
-------
- :func:`create_mixture_layer` — build any mixture layer by string type.
- :func:`create_mixture_from_config` — build from a config dict (with a ``type`` key).
- :func:`get_mixture_info` — registry metadata for all mixture types.
- :func:`validate_mixture_config` — validate a mixture configuration.
- :data:`MixtureType` — ``Literal`` of supported type strings.
- :data:`MIXTURE_REGISTRY` — type-string to class/metadata mapping.

Example
-------
>>> from dl_techniques.layers.mixtures import create_mixture_layer
>>> layer = create_mixture_layer('gmm', n_components=8)
"""

from .radial_basis_function import RBFLayer
from .kmeans import KMeansLayer
from .gmm import GMMLayer
from .factory import (
    MixtureType,
    MIXTURE_REGISTRY,
    create_mixture_layer,
    create_mixture_from_config,
    get_mixture_info,
    validate_mixture_config,
)

__all__ = [
    "RBFLayer",
    "KMeansLayer",
    "GMMLayer",
    "MixtureType",
    "MIXTURE_REGISTRY",
    "create_mixture_layer",
    "create_mixture_from_config",
    "get_mixture_info",
    "validate_mixture_config",
]
