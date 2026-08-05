"""
SAM 3 (Segment Anything with Concepts) -- phase 1: the text-prompt-only image path.

**Phase boundary.** This package is being built leaf-first. It currently exposes
the ViTDet image trunk only; the neck, text tower, DETR decoder, open-vocabulary
scorer, MaskFormer segmentation head and the top-level ``Sam3Image`` assembly land
in later steps of the same iteration, and the curated export surface is finalized
with them.

**Deliberately out of scope in phase 1**, and named here rather than left to be
rediscovered as a gap:

- the exemplar / geometry prompt path (points and boxes), which needs bilinear
  ``grid_sample`` and ``roi_align`` primitives that ``keras.ops`` does not have;
- DAC-DETR query doubling, which the reference gates on ``self.training`` and is
  therefore provably inert at inference;
- ``Sam3TriViTDetNeck`` (the SAM 3.1 three-way neck);
- the video / tracking path;
- the whole training path -- no loss, matcher, or trainer is defined here, and in
  particular nothing in this package may route mask supervision through SAM 1's
  mask-loss module or the shared segmentation focal loss it calls, whose
  probability clip has an exactly-zero derivative outside its range. (The two
  module names are deliberately NOT spelled here: the close-out gate for that
  constraint is a grep, and a prose mention erodes the instrument -- the same
  failure mode already measured twice in this repo for the ``tensorflow`` purity
  grep.)
"""

from .vitdet import Sam3ViTDetBackbone, Sam3ViTDetBlock

__all__ = [
    "Sam3ViTDetBackbone",
    "Sam3ViTDetBlock",
]
