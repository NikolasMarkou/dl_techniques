"""
SAM 3 (Segment Anything with Concepts): the phase-1 text-prompted image path.

Nine independently constructible, serializable layer classes plus two
``keras.Model``s: the ``Sam3Image`` assembly, wiring six of the nine
together, and the ``Sam3TrainingModel`` wrapper. No pretrained weights ship
here and no released SAM 3 checkpoint has been loaded, so this package makes
no accuracy claim. There is no class table and no softmax over categories;
the text prompt is the class. ``from_variant('sam3')`` is the released
configuration; ``'small'`` and ``'tiny'`` are this repository's own
development geometries, and ``'tiny'`` uses ``drop_path_rate=0.0`` where
``'sam3'`` uses the reference's ``0.1``.

Pass ``training=False`` explicitly for the ``sam3`` variant: the shared
``StochasticDepth`` layer only short-circuits when ``training`` is exactly
``False``, so a plain ``model(inputs)`` call, which passes ``training=None``,
silently drops paths.

Out of scope in this phase: the vision-language fusion encoder the reference
runs between neck and decoder, the exemplar and geometry prompt path
(``grid_sample``/``roi_align`` primitives ``keras.ops`` lacks), DAC-DETR
query doubling, ``Sam3TriViTDetNeck``, the video and tracking path, and the
loss and matcher, which live in the losses package.

References:
    - Ravi et al., 2025. SAM 3: Segment Anything with Concepts.

Example:

.. code-block:: python

    from dl_techniques.models.vision_language.sam.sam3 import Sam3Image
    model = Sam3Image.from_variant("tiny")
    outputs = model({"image": images, "token_ids": ids}, training=False)
"""

from .decoder import Sam3DecoderLayer, Sam3TransformerDecoder
from .maskformer_segmentation import Sam3SegmentationHead
from .model_misc import Sam3DotProductScoring
from .necks import Sam3DualViTDetNeck
from .query_selection import Sam3EncoderQuerySelection
from .sam3_image import Sam3Image
from .text_encoder_ve import Sam3TextEncoder
from .training_model import (
    Sam3TrainingModel, compile_sam3_trainer, pack_predictions, pack_targets)
from .vitdet import Sam3ViTDetBackbone, Sam3ViTDetBlock

__all__ = [
    "Sam3DecoderLayer",
    "Sam3DotProductScoring",
    "Sam3DualViTDetNeck",
    "Sam3EncoderQuerySelection",
    "Sam3Image",
    "Sam3SegmentationHead",
    "Sam3TextEncoder",
    "Sam3TrainingModel",
    "Sam3TransformerDecoder",
    "Sam3ViTDetBackbone",
    "Sam3ViTDetBlock",
    "compile_sam3_trainer",
    "pack_predictions",
    "pack_targets",
]
