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

The public surface grows one class at a time as the port lands. Right now it is
the RECTIFIER only: :class:`~.model.DocScannerRectifier` and its factory
:func:`~.model.create_doc_scanner_rectifier`. The segmenter and the composite
``DocScanner`` are not here yet and must not be imported from the package root
until they are -- a half-built name exported early is a name something starts
depending on.

The rectifier is exported NOW rather than at the end for a concrete reason:
``tests/test_models/test_package_api_contract.py`` walks each model package's
own namespace to find every ``pretrained``-taking entry point and CALL it, and a
factory that is not exported is a factory that guard cannot reach. Leaving it
unexported would have parked two live ``pretrained=True`` raise paths outside
behavioural coverage until the last step of the port.

``components.py`` and ``warp.py`` carry the port's constants, its single
``_VARIANT_SPEC`` width table and its sampling convention. They stay unexported
on purpose: they are the port's internals, imported directly by this package's
own modules and tests.

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

from .model import DocScannerRectifier, create_doc_scanner_rectifier

__all__: list = [
    "DocScannerRectifier",
    "create_doc_scanner_rectifier",
]
