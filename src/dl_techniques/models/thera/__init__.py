"""THERA: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields.

Public API for the THERA model port. The primary entry points are the
:class:`Thera` model and the :func:`build_thera` factory; the backbones, tails,
and hypernetwork are re-exported for convenience (e.g. building a custom-sized
``Thera`` directly for tests).
"""

from dl_techniques.models.thera.model import (
    Thera,
    build_thera,
    DEFAULT_COMPONENTS_INIT_SCALE,
)
from dl_techniques.models.thera.edsr_backbone import EDSRBackbone
from dl_techniques.models.thera.rdn_backbone import RDNBackbone
from dl_techniques.models.thera.tails import build_thera_tail
from dl_techniques.models.thera.hypernetwork import TheraHypernetwork

__all__ = [
    "Thera",
    "build_thera",
    "DEFAULT_COMPONENTS_INIT_SCALE",
    "EDSRBackbone",
    "RDNBackbone",
    "build_thera_tail",
    "TheraHypernetwork",
]
