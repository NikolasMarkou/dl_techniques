"""LeWM (Latent-energy / JEPA-style world model) — public API re-exports.

There is no ``MODEL_VARIANTS`` table and none was invented: LeWM ships a single
configuration (``LeWMConfig``, mirroring the upstream YAML defaults) and is
retuned field by field rather than by selecting a named scale. The one scale
knob, the vision encoder size, lives in ``LeWMConfig.encoder_scale`` and is
forwarded to ``ViT``. ``create_lewm`` therefore builds a config and constructs
the class rather than delegating to a ``from_variant``.
"""
from dl_techniques.models.lewm.config import LeWMConfig
from dl_techniques.models.lewm.embedder import ActionEmbedder
from dl_techniques.models.lewm.projector import MLPProjector
from dl_techniques.models.lewm.predictor import ARPredictor
from dl_techniques.models.lewm.model import LeWM, create_lewm

__all__ = [
    "LeWM",
    "LeWMConfig",
    "create_lewm",
    "ActionEmbedder",
    "MLPProjector",
    "ARPredictor",
]
