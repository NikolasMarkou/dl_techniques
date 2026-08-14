"""Segment Anything — public API re-exports.

Three generations as nested subpackages, each a full architecture with its own
`MODEL_VARIANTS` and `from_variant`: `SAM1/` (image encoder + prompt encoder +
mask decoder), `SAM2/` (adds the memory bank and attention for video), and
`SAM3/` (adds text-conditioned open-vocabulary segmentation). Nothing is shared
between them beyond the repo's layer library, and none deprecates the others.

Only the top-level models and training models are re-exported here. The internal
components — encoders, necks, decoders, memory modules — stay behind their
submodules, since all three generations define same-named parts that must not
collide at package level.
"""
from .SAM1.image_encoder import ImageEncoderViT
from .SAM1.model import SAM
from .SAM1.training_model import SAMTrainingModel
from .SAM2.model import SAM2, create_sam2
from .SAM2.training_model import SAM2TrainingModel
from .SAM3.sam3_image import Sam3Image
from .SAM3.training_model import Sam3TrainingModel

__all__ = [
    "ImageEncoderViT",
    "SAM",
    "SAM2",
    "SAM2TrainingModel",
    "SAMTrainingModel",
    "Sam3Image",
    "Sam3TrainingModel",
    "create_sam2",
]
