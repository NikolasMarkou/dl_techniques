"""Tests for the merged ``heads.vlm`` sub-package.

The old ``vlm_heads`` package had ZERO test coverage and a broken (empty)
``__init__.py``; the merge populated ``heads/vlm/__init__.py``. This file is
net-new safety: it asserts import + Keras registration of all 6 VLM classes and
constructs two heads via ``create_vlm_head`` (reusing the Step-3 working
``VLMTaskConfig`` + ``vision_dim``/``text_dim``/``num_layers``/``num_heads``
arg shape).

DELIBERATELY WEAKENED (documented): a full forward pass + ``.keras`` round-trip
of the VLM heads is NOT asserted here. Exercising ``ImageCaptioningHead.call``
and ``ImageTextMatchingHead.call`` surfaces TWO PRE-EXISTING latent defects in
the merged-but-untouched VLM code (out of scope for this structural-merge plan):

  1. ``ImageTextMatchingHead.call`` calls ``ops.l2_normalize`` (factory.py:690),
     which does not exist on this Keras version's ``keras.ops`` namespace
     -> ``AttributeError``.
  2. ``ImageCaptioningHead.call`` builds a ``(S, S)`` causal mask that does not
     broadcast against the ``MultiHeadAttention``/``MultiHeadCrossAttention``
     mask expectation -> ``InvalidArgumentError`` (broadcastable shapes).

Both are forward-pass bugs in code merely relocated by this plan (Step 3, dot-fix
only), not regressions introduced by the merge, and fixing them is a separate
concern. The safety the merge itself needs — that the new ``heads.vlm`` import
surface resolves, the classes register, and the factory dispatches to the right
class — is fully covered below.
"""

import pytest
import keras

from dl_techniques.layers.heads.vlm import (
    VLMTaskType,
    VLMTaskConfig,
    create_vlm_head,
    BaseVLMHead,
    ImageCaptioningHead,
    VQAHead,
    VisualGroundingHead,
    ImageTextMatchingHead,
    MultiTaskVLMHead,
)

DIM = 32


# ---------------------------------------------------------------------
# Import + registration of all 6 VLM classes
# ---------------------------------------------------------------------

class TestVLMRegistration:

    @pytest.mark.parametrize("name", [
        "BaseVLMHead",
        "ImageCaptioningHead",
        "VQAHead",
        "VisualGroundingHead",
        "ImageTextMatchingHead",
        "MultiTaskVLMHead",
    ])
    def test_class_registered(self, name) -> None:
        assert keras.saving.get_registered_object(f"Custom>{name}") is not None


# ---------------------------------------------------------------------
# Factory dispatch + construction smoke
# ---------------------------------------------------------------------

class TestVLMFactoryConstruction:

    def test_image_captioning_head_constructs(self) -> None:
        cfg = VLMTaskConfig(
            name="cap",
            task_type=VLMTaskType.IMAGE_CAPTIONING,
            vocab_size=50,
            hidden_size=DIM,
        )
        head = create_vlm_head(
            cfg, vision_dim=DIM, text_dim=DIM, num_layers=1, num_heads=4
        )
        assert isinstance(head, ImageCaptioningHead)
        assert isinstance(head, keras.layers.Layer)
        assert head.num_layers == 1
        assert head.num_heads == 4

    def test_image_text_matching_head_constructs(self) -> None:
        cfg = VLMTaskConfig(
            name="itm",
            task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
            hidden_size=DIM,
        )
        head = create_vlm_head(cfg, vision_dim=DIM, text_dim=DIM)
        assert isinstance(head, ImageTextMatchingHead)
        assert isinstance(head, keras.layers.Layer)

    def test_factory_from_dict_config(self) -> None:
        head = create_vlm_head(
            {
                "name": "itm",
                "task_type": VLMTaskType.IMAGE_TEXT_MATCHING,
                "hidden_size": DIM,
            },
            vision_dim=DIM,
            text_dim=DIM,
        )
        assert isinstance(head, ImageTextMatchingHead)
