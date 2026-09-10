"""DocScanner document-image rectification — public API re-exports.

DocScanner unwarps a photographed page in two stages that are trained
INDEPENDENTLY (paper §4.3), not end to end:

* **Localization.** A pruned U2NET-P predicts a document/background confidence
  map at 288x288. Thresholded at 0.5, it multiplicatively masks the background
  out of the input.
* **Rectification.** A RAFT-lineage iterative refiner: a stride-8 feature
  encoder, then 12 GRU update steps that each emit a residual to a
  1/8-resolution coordinate field, convex-upsampled 8x into a full-resolution
  BACKWARD map in pixel units. Sampling the original image through that map
  produces the rectified page.

There is no correlation volume anywhere in this architecture, despite the RAFT
lineage: the update block's ``corr`` input is a bilinear resample of the
encoder's own feature map at the current coordinates. Do not add one.

The public surface is finalized by the step that lands the model classes; until
then this module deliberately re-exports NOTHING, so that no half-built name can
be imported from the package root and depended on. ``components.py`` carries the
port's constants and its single ``_VARIANT_SPEC`` width table, and is imported
directly by this package's own modules and tests.

Two constraints a caller will need before using this package:

* **Input height and width must be divisible by 8**, the rectifier's internal
  stride. The model raises a ``ValueError`` naming the offending size rather
  than padding internally, matching the sibling ``doc_res`` contract: padding is
  a property of the inference shim, which is the only place that can also remove
  it from the prediction.
* **``pretrained=True`` will raise ``NotImplementedError``.** The upstream
  release ships no checkpoint in the reference checkout, and separately this
  port's resampling and channel conventions differ from the PyTorch original, so
  a transferred checkpoint would be silently wrong rather than merely untested.

References:
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2.
    - Teed & Deng, 2020. RAFT: Recurrent All-Pairs Field Transforms for Optical
      Flow. ECCV 2020. (https://arxiv.org/abs/2003.12039) -- the source of the
      convex upsample and the separable ConvGRU update block.
    - Qin et al., 2020. U2-Net: Going Deeper with Nested U-Structure for
      Salient Object Detection. Pattern Recognition.
      (https://arxiv.org/abs/2005.09007) -- the localization backbone.
"""

__all__: list = []
