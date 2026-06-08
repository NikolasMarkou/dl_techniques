"""Tests for the top-level ``create_head`` dispatch facade (D-004).

The facade is the user-requested deliverable contract surface; per LESSONS the
factory gets its own test file. It is a *thin dispatcher* over the three
per-domain single-head factories with no signature unification — each domain
keeps its native calling convention, forwarded verbatim.
"""

import pytest
import keras

from dl_techniques.layers.heads import create_head
from dl_techniques.layers.heads.nlp import (
    NLPTaskType, NLPTaskConfig, TextClassificationHead,
)
from dl_techniques.layers.heads.vision import VisionTaskType, ClassificationHead
from dl_techniques.layers.heads.vlm import (
    VLMTaskType, VLMTaskConfig, ImageTextMatchingHead,
)


class TestCreateHeadDispatch:

    def test_nlp_dispatch(self) -> None:
        head = create_head(
            "nlp",
            task_config=NLPTaskConfig(
                name="cls",
                task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=3,
            ),
            input_dim=8,
        )
        assert isinstance(head, TextClassificationHead)
        assert isinstance(head, keras.layers.Layer)

    def test_vision_dispatch(self) -> None:
        head = create_head(
            "vision", VisionTaskType.CLASSIFICATION, num_classes=10, hidden_dim=16
        )
        assert isinstance(head, ClassificationHead)
        assert isinstance(head, keras.layers.Layer)

    def test_vlm_dispatch(self) -> None:
        head = create_head(
            "vlm",
            task_config=VLMTaskConfig(
                name="itm",
                task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
                hidden_size=32,
            ),
            vision_dim=32,
            text_dim=32,
        )
        assert isinstance(head, ImageTextMatchingHead)
        assert isinstance(head, keras.layers.Layer)

    def test_bogus_domain_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown head domain"):
            create_head("bogus", num_classes=3)
