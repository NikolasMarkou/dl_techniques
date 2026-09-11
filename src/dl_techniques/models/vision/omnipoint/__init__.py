"""OmniPoint public API.

Re-exports the model class and the factory function. `heads.py`'s
`RayDistanceHead`/`MaskHead`/`MetricScaleHead` are internal composition
building blocks, imported directly from `dl_techniques.models.vision.omnipoint.heads`
by anything that needs them standalone (e.g. tests).
"""

from .model import (
    OmniPoint,
    create_omnipoint,
    MODEL_VARIANTS,
)

__all__ = [
    "OmniPoint",
    "create_omnipoint",
    "MODEL_VARIANTS",
]
