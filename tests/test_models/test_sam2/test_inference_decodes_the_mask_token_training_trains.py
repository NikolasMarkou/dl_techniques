"""A SAM 2 built here decodes, at inference, the mask token its training trains.

``SAM2TrainingModel`` decodes every frame with ``multimask_output=False``, so
only the single-mask token (index 0) and its hypernetwork get a gradient
(plan-2026-08-04T044628-4c240b4c D-023); the three multimask tokens keep their
initialisation. ``SAM2MaskDecoder``'s ``dynamic_multimask_via_stability``
(upstream's inference rule, written for checkpoints whose multimask tokens were
trained) replaces an unstable token-0 mask at inference by the multimask token
with the highest predicted IoU, so a model trained here answered with an
untrained token whenever its mask was unstable. Measured on the 30-epoch SAM 2
smoke model (plan-2026-10-14T062616-81938e81 step 12, D-245): validation mask
loss 1.712 in inference mode against 1.042 in training mode on the same 16
clips, and 1.043 in inference mode with the fallback off. ``SAM2.from_variant``
now builds the decoder with the fallback off. The mechanism itself stays, and
``test_mask_decoder.py`` still tests it on a decoder built with it on.
"""

import keras
import numpy as np
from unittest import mock

from dl_techniques.models.vision_language.sam.sam2.mask_decoder import SAM2MaskDecoder
from dl_techniques.models.vision_language.sam.sam2.model import create_sam2
from dl_techniques.models.vision_language.sam.sam2.training_model import SAM2_LOW_RES_LOGITS

from .test_training_model import clip_inputs, trainer


def _constant_stability(value):
    def stability(self, mask_logits):
        return keras.ops.full(keras.ops.shape(mask_logits)[:2], value)
    return stability


def test_the_builder_turns_the_stability_fallback_off():
    assert create_sam2("tiny").mask_decoder.dynamic_multimask_via_stability is False


def test_the_inference_mask_does_not_depend_on_the_stability_score():
    """Every mask forced unstable, then forced stable: with the fallback on, the
    unstable call answers with an untrained multimask token and the two differ."""
    model = trainer(object_score=5.0)
    x = clip_inputs(model)
    outputs = {}
    for name, value in (("unstable", 0.0), ("stable", 1.0)):
        with mock.patch.object(SAM2MaskDecoder, "_get_stability_scores", _constant_stability(value)):
            outputs[name] = np.asarray(model(x, training=False)[SAM2_LOW_RES_LOGITS])
    np.testing.assert_array_equal(outputs["unstable"], outputs["stable"])
