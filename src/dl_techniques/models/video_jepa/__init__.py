"""Video-JEPA-Clifford model package.

A patch-based video JEPA backbone using CliffordNet primitives for video
streaming. Core components:

- :class:`VideoJEPAConfig` — dataclass config (:mod:`.config`).
- :class:`VideoJEPACliffordEncoder` — hybrid PatchEmbedding + Clifford blocks
  (:mod:`.encoder`).
- :class:`VideoJEPAPredictor` — factorized spatial/causal-temporal predictor
  (pixels-only, iter-3 / D-013) (:mod:`.predictor`).
- :class:`VideoJEPA` — top-level model with streaming inference API
  (:mod:`.model`).
- :func:`create_video_jepa` — module-level convenience factory.

There is no ``MODEL_VARIANTS`` table and none was invented: this port ships one
``VideoJEPAConfig`` and is retuned field by field (embed_dim, depths, prediction
horizons) rather than by selecting a named scale, and no scale family is
published for it. ``create_video_jepa`` therefore builds a config and
constructs the class rather than delegating to a ``from_variant``.
"""
from dl_techniques.models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.video_jepa.encoder import VideoJEPACliffordEncoder
from dl_techniques.models.video_jepa.predictor import VideoJEPAPredictor
from dl_techniques.models.video_jepa.masking import TubeMaskGenerator
from dl_techniques.models.video_jepa.model import VideoJEPA, create_video_jepa

__all__ = [
    "VideoJEPA",
    "VideoJEPAConfig",
    "create_video_jepa",
    "VideoJEPACliffordEncoder",
    "VideoJEPAPredictor",
    "TubeMaskGenerator",
]
