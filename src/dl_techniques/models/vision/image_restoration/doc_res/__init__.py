"""DocRes document-image restoration — public API re-exports.

DocRes is one Restormer backbone serving five document tasks (dewarping,
deshadowing, appearance enhancement, deblurring, binarization). Nothing in the
network is task-conditioned: the task is carried entirely by the 3 classical-CV
"DTSPrompt" channels the caller stacks onto the RGB image, giving the
6-channel input this model requires. See ``README.md`` for the input contract,
the task table and the constraints below.

Two names are exported and no more:

* :class:`DocRes` — the ``keras.Model`` subclass. Its variant table is a CLASS
  attribute (``DocRes.MODEL_VARIANTS``), not a module-level constant, so there
  is nothing separate to re-export; reach it through the class.
* :func:`create_doc_res` — the module-level factory, which delegates to
  ``DocRes.from_variant``.

``RestormerTransformerBlock`` / ``RestormerDownsample`` / ``RestormerUpsample``
are deliberately NOT re-exported. They are assembly parts of this one backbone,
not a public surface, and the only consumers outside ``model.py`` are this
package's own tests, which import them from ``.components`` directly. (Contrast
``pw_fnet``, which does export its block because the shared FFN-factory suite
builds it standalone; nothing outside this package builds a Restormer block.)

Two constraints a caller must know before using either name:

* **Input height and width must be divisible by 8** — the model raises a
  ``ValueError`` naming the offending size rather than padding internally.
  Padding belongs to the inference shim, which is the only place that can also
  remove it from the prediction.
* **Upstream PyTorch DocRes weights cannot be loaded into this port.** This
  repo's pixel-shuffle layers use a different channel-block ordering from
  ``torch.nn.PixelShuffle`` (measured; see the D-007 anchors in
  ``components.py``), so a transferred checkpoint would be silently wrong
  rather than merely untested. ``pretrained=True`` raises
  ``NotImplementedError``; train from scratch.
"""
from dl_techniques.models.vision.image_restoration.doc_res.model import (
    DocRes,
    create_doc_res,
)

__all__ = [
    "DocRes",
    "create_doc_res",
]
