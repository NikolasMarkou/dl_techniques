"""
SAM 3 (Segment Anything with Concepts): the phase-1 text-prompted image path.
=============================================================================

Nine independently constructible, serializable layer classes plus two
``keras.Model``s -- the ``Sam3Image`` assembly, wiring six of the nine together,
and the ``Sam3TrainingModel`` wrapper. No pretrained weights ship here, and this
package makes NO learnability, quality or accuracy claim: nothing here has been
trained to any quality and no released SAM 3 checkpoint has ever been loaded.

Based on:
---------
- Ravi, N. et al. (2025). "SAM 3: Segment Anything with Concepts."

Key Features:
------------
- Open vocabulary: no class table, no softmax over categories; prompt = class.
- ``from_variant('sam3')`` is the released configuration; ``'small'`` and
  ``'tiny'`` are this repository's own development geometries.
- ``Sam3TrainingModel`` emits ONE packed supervision tensor for one joint loss.
- ``Sam3EncoderQuerySelection`` is this package's OWN opt-in proposal head, not
  a reference component: default OFF, behaviourally inert when off.

Architecture Overview:
---------------------
1. **Sam3ViTDetBackbone** -- plain-ViT trunk, ONE feature map out.
2. **Sam3DualViTDetNeck** -- SimpleFPN, four scales, two weight sets.
3. **Sam3TextEncoder** -- CLIP text tower, per-token memory.
4. **Sam3TransformerDecoder** -- DETR decoder, boxRPB and presence token.
5. **Sam3DotProductScoring** -- open-vocabulary per-query class logits.
6. **Sam3SegmentationHead** -- MaskFormer head, one mask per query.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM3 import Sam3Image
model = Sam3Image.from_variant("tiny")
outputs = model({"image": images, "token_ids": ids}, training=False)
```

Measured caveats:
----------------
- **Pass ``training=False`` for the ``sam3`` variant**: it carries the
  reference's ``drop_path_rate=0.1`` and the shared ``StochasticDepth``
  short-circuits on ``training is False`` ONLY, so the ``training=None`` a plain
  ``model(inputs)`` passes down DROPS PATHS (D-123). ``tiny`` uses 0.0.
- The three variants' measured parameter geometries have ONE home, ``README.md``
  section 5, which also names the test pinning each. They are deliberately not
  restated here: a count restated in two places is a hand-maintained lockstep
  invariant, i.e. a latent defect.
- The segmentation head has no presence mechanism, so exactly ONE presence
  signal exists here: the decoder's own presence token.
- Out of scope in phase 1, named rather than left to be rediscovered: the
  vision-language EARLY-FUSION ENCODER the reference runs between neck and
  decoder (the largest single structural divergence here); the exemplar /
  geometry prompt path, needing ``grid_sample`` / ``roi_align`` primitives
  ``keras.ops`` lacks; DAC-DETR query doubling, gated on ``self.training``
  upstream and so provably inert at inference; ``Sam3TriViTDetNeck``; the video
  / tracking path; and the loss and matcher, which live in the losses package.
- Nothing here may route mask supervision through SAM 1's mask-loss class or the
  shared segmentation focal-loss class it calls, whose probability clip has an
  exactly-zero derivative outside its range. Those two names are deliberately
  NOT spelled: the gate is a grep over every file in this package, and a prose
  mention erodes the instrument as it did four times for the ``tensorflow``
  purity grep.
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
