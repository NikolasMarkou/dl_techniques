"""Tests for the merged ``heads.vlm`` sub-package.

The old ``vlm_heads`` package had ZERO test coverage and a broken (empty)
``__init__.py``; the merge populated ``heads/vlm/__init__.py``. This file asserts
import + Keras registration of all 6 VLM classes, constructs heads via
``create_vlm_head``, and (as of the bugfix plan) exercises the full forward pass
+ a ``.keras`` round-trip for the three previously-broken heads.

The three forward-pass-fixed heads are locked here:

  * ``ImageTextMatchingHead`` — Bug 1 (``ops.l2_normalize`` -> ``ops.normalize``)
    + the fine-grained-fusion 3-D blocker (D-001). Its fusion path requires
    ``vision_dim == text_dim == hidden_size``, so the tests feed ``DIM`` features
    with ``hidden_size=DIM``.
  * ``ImageCaptioningHead`` — Bug 2 (causal mask shape/polarity) + the
    cross-attention type/kwarg blocker. Returns ``logits`` of ``(B, S, vocab)``.
  * ``VQAHead`` — Bug 4 (shared cross-attention type/kwarg). Its DEFAULT
    ``pooling_strategy="attention"`` reuses one cross-attention layer in both
    directions, so the tests use ``vision_dim == text_dim``.
"""

import os
import tempfile

import numpy as np
import pytest
import keras
from keras import ops

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
B, S = 3, 7
VOCAB = 50
NUM_HEADS = 4
NUM_CLASSES = 11


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
            vocab_size=VOCAB,
            hidden_size=DIM,
        )
        head = create_vlm_head(
            cfg, vision_dim=DIM, text_dim=DIM, num_layers=1, num_heads=NUM_HEADS
        )
        assert isinstance(head, ImageCaptioningHead)
        assert isinstance(head, keras.layers.Layer)
        assert head.num_layers == 1
        assert head.num_heads == NUM_HEADS

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


# ---------------------------------------------------------------------
# Forward-pass builders + dummy inputs
# ---------------------------------------------------------------------

def _captioning_head() -> ImageCaptioningHead:
    cfg = VLMTaskConfig(
        name="cap",
        task_type=VLMTaskType.IMAGE_CAPTIONING,
        vocab_size=VOCAB,
        hidden_size=DIM,
    )
    return create_vlm_head(
        cfg, vision_dim=DIM, text_dim=DIM, num_layers=1, num_heads=NUM_HEADS
    )


def _vqa_head() -> VQAHead:
    cfg = VLMTaskConfig(
        name="vqa",
        task_type=VLMTaskType.VISUAL_QUESTION_ANSWERING,
        hidden_size=DIM,
        num_classes=NUM_CLASSES,
    )
    # DEFAULT pooling_strategy="attention"; equal dims for the shared cross-attn.
    return create_vlm_head(cfg, vision_dim=DIM, text_dim=DIM)


def _itm_head() -> ImageTextMatchingHead:
    cfg = VLMTaskConfig(
        name="itm",
        task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
        hidden_size=DIM,  # D == hidden_size required by the fusion path.
    )
    return create_vlm_head(cfg, vision_dim=DIM, text_dim=DIM)


@pytest.fixture
def vision_feats() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal((B, S, DIM)).astype("float32")


@pytest.fixture
def text_feats() -> np.ndarray:
    rng = np.random.default_rng(8)
    return rng.standard_normal((B, S, DIM)).astype("float32")


# ---------------------------------------------------------------------
# SC3 / SC4 / SC1 — forward pass on the three fixed heads
# ---------------------------------------------------------------------

class TestVLMForwardPass:

    def test_image_captioning_forward(self, vision_feats, text_feats) -> None:
        head = _captioning_head()
        out = head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(text_feats),
        })
        assert "logits" in out
        assert tuple(out["logits"].shape) == (B, S, VOCAB)

    def test_vqa_forward_default_attention_pooling(
        self, vision_feats, text_feats
    ) -> None:
        head = _vqa_head()
        out = head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "question_features": ops.convert_to_tensor(text_feats),
        })
        assert "answer_logits" in out
        assert tuple(out["answer_logits"].shape) == (B, NUM_CLASSES)

    def test_image_text_matching_forward(self, vision_feats, text_feats) -> None:
        head = _itm_head()
        out = head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(text_feats),
        })
        # Full output dict contract (SC1).
        assert set(out.keys()) == {
            "similarity_matrix",
            "logits",
            "match_score",
            "vision_embeddings",
            "text_embeddings",
        }
        assert tuple(out["similarity_matrix"].shape) == (B, B)
        assert tuple(out["logits"].shape) == (B, B)
        assert tuple(out["match_score"].shape) == (B,)
        assert np.all(np.isfinite(ops.convert_to_numpy(out["similarity_matrix"])))

    def test_image_captioning_causal_property(
        self, vision_feats, text_feats
    ) -> None:
        """SC3 (optional): perturbing the LAST text position must not change the
        logits at earlier positions (the causal mask blocks future leakage)."""
        head = _captioning_head()
        base = {
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(text_feats),
        }
        logits0 = ops.convert_to_numpy(head(base)["logits"])

        perturbed = text_feats.copy()
        perturbed[:, S - 1, :] += 5.0  # disturb only the final position's input
        logits1 = ops.convert_to_numpy(head({
            "vision_features": ops.convert_to_tensor(vision_feats),
            "text_features": ops.convert_to_tensor(perturbed),
        })["logits"])

        # Positions 0..S-2 must be unchanged; only the last position may differ.
        np.testing.assert_allclose(
            logits0[:, : S - 1, :], logits1[:, : S - 1, :], atol=1e-5
        )


# ---------------------------------------------------------------------
# SC5 — .keras save/load round-trip of ImageCaptioningHead
# ---------------------------------------------------------------------

class TestVLMRoundtrip:

    def test_image_captioning_roundtrip(self, vision_feats, text_feats) -> None:
        vf = ops.convert_to_tensor(vision_feats)
        tf = ops.convert_to_tensor(text_feats)

        @keras.saving.register_keras_serializable()
        class _CapWrapper(keras.Model):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.head = _captioning_head()

            def call(self, inputs, training=None):
                return self.head(inputs, training=training)

        model = _CapWrapper()
        inputs = {"vision_features": vf, "text_features": tf}
        y0 = model(inputs)  # build before save (LESSONS)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cap_head.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(inputs)
        assert tuple(y1["logits"].shape) == tuple(y0["logits"].shape) == (B, S, VOCAB)
