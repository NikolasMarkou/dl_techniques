"""
SAM 3 (Segment Anything with Concepts) -- phase 1: the text-prompt-only image path.

**Phase boundary.** The package is complete for phase 1: the ViTDet image trunk,
the dual SimpleFPN neck, the CLIP text tower, the DETR decoder, the
open-vocabulary dot-product scorer, the MaskFormer segmentation head, and the
top-level ``Sam3Image`` assembly that wires all six together and -- when
``supervise_joint_box_scores=True``, which is **OFF by default** and which
``from_variant`` never sets -- applies the presence x localization fusion.

Start at :class:`Sam3Image`; ``Sam3Image.from_variant('sam3')`` builds the
released configuration and ``from_variant('tiny')`` a small development one.

**Pass ``training=False`` when you run the ``sam3`` variant for inference.**
The released configuration carries the reference's ``drop_path_rate=0.1``, and
this repository's shared ``StochasticDepth`` short-circuits on ``training is
False`` ONLY -- so the ``training=None`` that a plain ``model(inputs)`` passes
down DROPS PATHS and makes the shipped variant non-deterministic. Use
``model(inputs, training=False)`` (or ``model.predict``). The ``tiny`` variant
sets the rate to 0.0 and is unaffected.
The eight component classes are exported too, because each is independently
constructible, serializable and testable -- which is the property that let this
package be built and gated one leaf at a time.

The segmentation head has **no presence mechanism**: the shipped reference
configuration disables it there and drives presence from the decoder's own
presence token, so only ONE presence signal exists in this package.

**Deliberately out of scope in phase 1**, and named here rather than left to be
rediscovered as a gap:

- the exemplar / geometry prompt path (points and boxes), which needs bilinear
  ``grid_sample`` and ``roi_align`` primitives that ``keras.ops`` does not have;
- DAC-DETR query doubling, which the reference gates on ``self.training`` and is
  therefore provably inert at inference;
- ``Sam3TriViTDetNeck`` (the SAM 3.1 three-way neck);
- the video / tracking path;
- the vision-language EARLY-FUSION ENCODER that the reference runs between the
  neck and the decoder. Phase 1 feeds the neck's image memory and the text
  tower's prompt straight into the decoder; this is the largest single
  structural divergence in the package and it is named, not hidden;
- the loss and the matcher, which live in the losses package, not here. What
  this package now ships of the training path is :class:`Sam3TrainingModel`,
  a wrapper that emits ONE packed supervision tensor so a single joint loss can
  split it (the layout itself is defined by that loss module, not by this one),
  plus its single ``compile`` site. Nothing in this package may route mask
  supervision through SAM 1's mask-loss module or the shared segmentation focal
  loss it calls, whose
  probability clip has an exactly-zero derivative outside its range. (The two
  module names are deliberately NOT spelled here: the close-out gate for that
  constraint is a grep, and a prose mention erodes the instrument -- the same
  failure mode already measured twice in this repo for the ``tensorflow`` purity
  grep.)
"""

from .decoder import Sam3DecoderLayer, Sam3TransformerDecoder
from .maskformer_segmentation import Sam3SegmentationHead
from .model_misc import Sam3DotProductScoring
from .necks import Sam3DualViTDetNeck
from .sam3_image import Sam3Image
from .text_encoder_ve import Sam3TextEncoder
from .training_model import (
    Sam3TrainingModel, compile_sam3_trainer, pack_predictions, pack_targets)
from .vitdet import Sam3ViTDetBackbone, Sam3ViTDetBlock

__all__ = [
    "Sam3DecoderLayer",
    "Sam3DotProductScoring",
    "Sam3DualViTDetNeck",
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
