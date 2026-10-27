"""Every VLM head takes a caller's ``name``, and a width error names the head.

Measured at cfa8ba4bf (F-8 NOTEs on the heads review):

* ``VisualGroundingHead(task_config=..., name="custom")`` and the same on the
  other three heads raised ``TypeError: got multiple values for keyword
  argument 'name'``: each ``__init__`` called ``super().__init__(name=...,
  **kwargs)``, and each ``from_config`` dropped the stored name, so a head
  could never carry its own. The NLP heads use ``kwargs.setdefault``.
* ``VisualGroundingHead`` and ``ImageTextMatchingHead`` fed a stream of the
  wrong width raised the fusion layer's "Modality 0 dimension 16 doesn't match
  expected dim 32", naming neither head. Captioning and VQA name themselves.

The default name stays ``<task name>_head``, so weight paths do not move.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vlm import (
    ImageCaptioningHead, ImageTextMatchingHead, VQAHead, VisualGroundingHead,
    VLMTaskConfig, VLMTaskType,
)

DIM = 16

HEADS = [
    (ImageCaptioningHead,
     dict(task_type=VLMTaskType.IMAGE_CAPTIONING, vocab_size=11), dict(num_heads=4)),
    (VQAHead, dict(task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING, num_classes=7), {}),
    (VisualGroundingHead, dict(task_type=VLMTaskType.VISUAL_GROUNDING), {}),
    (ImageTextMatchingHead, dict(task_type=VLMTaskType.IMAGE_TEXT_MATCHING), {}),
]
IDS = [cls.__name__ for cls, _, _ in HEADS]


def _config(fields):
    return VLMTaskConfig(name="task", hidden_size=DIM, **fields)


@pytest.mark.parametrize("cls,fields,kwargs", HEADS, ids=IDS)
def test_a_custom_name_is_kept_and_survives_a_round_trip(cls, fields, kwargs):
    head = cls(task_config=_config(fields), name="custom", **kwargs)
    assert head.name == "custom"
    assert cls.from_config(head.get_config()).name == "custom"
    revived = keras.layers.deserialize(keras.layers.serialize(head))
    assert type(revived) is cls and revived.name == "custom"


@pytest.mark.parametrize("cls,fields,kwargs", HEADS, ids=IDS)
def test_the_default_name_is_still_the_task_name(cls, fields, kwargs):
    # Liveness: the default must not move, or every saved weight path would.
    head = cls(task_config=_config(fields), **kwargs)
    assert head.name == "task_head"
    assert cls.from_config(head.get_config()).name == "task_head"


@pytest.mark.parametrize("cls,fields,kwargs", HEADS[2:], ids=IDS[2:])
@pytest.mark.parametrize("narrow", ["vision_features", "text_features"])
def test_a_width_mismatch_names_the_head(cls, fields, kwargs, narrow):
    head = cls(task_config=_config(fields))
    inputs = {
        "vision_features": np.zeros((2, 4, DIM), "float32"),
        "text_features": np.zeros((2, 5, DIM), "float32"),
    }
    inputs[narrow] = np.zeros((2, inputs[narrow].shape[1], DIM // 2), "float32")
    with pytest.raises(ValueError, match=cls.__name__):
        head(inputs)
