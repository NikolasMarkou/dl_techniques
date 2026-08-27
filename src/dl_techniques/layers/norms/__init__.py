"""Normalization layers for ``dl_techniques``.

This package collects the library's normalization layers and the factory that
builds them from a string key. Importing from here gives the public surface;
the modules behind it are implementation detail.

What is exported
----------------

* **RMS family** - :class:`RMSNorm`, :class:`ZeroCenteredRMSNorm`,
  :class:`BandRMS`, :class:`ZeroCenteredBandRMSNorm`,
  :class:`AdaptiveBandRMS`, :class:`ZeroCenteredAdaptiveBandRMS`.
* **Logit family** - :class:`LogitNorm`, :class:`BandLogitNorm`,
  :class:`MaxLogitNorm`, :class:`DecoupledMaxLogit`, :class:`DMLPlus`.
* **Others** - :class:`BiasFreeBatchNorm`, :class:`GlobalResponseNormalization`,
  :class:`DynamicTanh`, :class:`EnergyLayerNorm`, :class:`PolarWeightNorm`,
  plus the module-level helpers :func:`polar_encode` and :func:`polar_decode`.
* **Factory** - :func:`create_normalization_layer`,
  :func:`create_normalization_from_config`, :func:`get_normalization_info`,
  :func:`validate_normalization_config` and the ``NormalizationType`` alias.

``__all__`` names 16 classes, 2 module-level functions, 4 factory functions and
1 type alias: 23 names in total.

Choosing a layer
----------------

Prefer the factory when the layer type comes from a config::

    from dl_techniques.layers.norms import create_normalization_layer

    layer = create_normalization_layer("rms_norm", axis=-1)

``NormalizationType`` lists the 18 accepted string keys. Import a class directly
when the type is fixed at authoring time.

Note
----

Most of these layers preserve the input shape. Three factory keys do not.
``decoupled_max_logit``, ``dml_plus_center`` and ``dml_plus_focal`` all reduce
the feature axis away; measured on a ``(3, 5, 8)`` input at the default
``axis=-1`` they give ``(3, 5)``. Two of the three also return a tuple rather
than one tensor: ``decoupled_max_logit`` a 3-tuple of ``(3, 5)`` tensors, and
``dml_plus_center`` a 2-tuple shaped ``(3, 5)`` and ``(3, 5, 1)``.
``max_logit_norm`` IS shape-preserving; ``(3, 5, 8)`` stays ``(3, 5, 8)`` and it
returns a single tensor. Read the class docstring before dropping any of them
into a shape-sensitive position.
"""

from .rms_norm import RMSNorm
from .bias_free_batch_norm import BiasFreeBatchNorm
from .zero_centered_rms_norm import ZeroCenteredRMSNorm
from .band_rms import BandRMS
from .adaptive_band_rms import AdaptiveBandRMS
from .band_logit_norm import BandLogitNorm
from .zero_centered_band_rms_norm import ZeroCenteredBandRMSNorm
from .zero_centered_adaptive_band_rms_norm import ZeroCenteredAdaptiveBandRMS
from .logit_norm import LogitNorm
from .max_logit_norm import MaxLogitNorm, DecoupledMaxLogit, DMLPlus
from .global_response_norm import GlobalResponseNormalization
from .dynamic_tanh import DynamicTanh
from .energy_layer_norm import EnergyLayerNorm
from .polar_weight_norm import PolarWeightNorm, polar_encode, polar_decode
from .factory import (
    create_normalization_layer,
    create_normalization_from_config,
    get_normalization_info,
    validate_normalization_config,
    NormalizationType,
)

__all__ = [
    "RMSNorm",
    "BiasFreeBatchNorm",
    "ZeroCenteredRMSNorm",
    "BandRMS",
    "AdaptiveBandRMS",
    "BandLogitNorm",
    "ZeroCenteredBandRMSNorm",
    "ZeroCenteredAdaptiveBandRMS",
    "LogitNorm",
    "MaxLogitNorm",
    "DecoupledMaxLogit",
    "DMLPlus",
    "GlobalResponseNormalization",
    "DynamicTanh",
    "EnergyLayerNorm",
    "PolarWeightNorm",
    "polar_encode",
    "polar_decode",
    "create_normalization_layer",
    "create_normalization_from_config",
    "get_normalization_info",
    "validate_normalization_config",
    "NormalizationType",
]
