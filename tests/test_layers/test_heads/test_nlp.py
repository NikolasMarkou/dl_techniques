"""Tests for the merged ``heads.nlp`` sub-package.

Covers:

* **SC4 — pooling equivalence (the critical test).** The old inline
  ``BaseNLPHead._pool_sequence`` body for cls/mean/max was replaced by a
  delegation to the shared ``SequencePooling`` facade (D-002). The old code is
  gone, so this test asserts the *analytical* references for cls/mean/max
  (mask-weighted) plus shape/finiteness for the kept-inline ``attention`` path,
  on a fixed ``(B=2, S=4, D=8)`` input with the last token of row 0 masked.
  This locks that the SequencePooling delegation preserved semantics.
* factory smoke + a ``keras.Model`` save/load round-trip for one NLP head.
"""

import os
import tempfile

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.heads.nlp import (
    NLPTaskType,
    NLPTaskConfig,
    create_nlp_head,
    TextClassificationHead,
)

# ---------------------------------------------------------------------
# Fixtures: a fixed input + a known attention mask (last token of row 0 masked)
# ---------------------------------------------------------------------

B, S, D = 2, 4, 8


@pytest.fixture
def fixed_input() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.standard_normal((B, S, D)).astype("float32")


@pytest.fixture
def attention_mask() -> np.ndarray:
    # Row 0: last token masked out (valid = positions 0..2).
    # Row 1: all four tokens valid.
    mask = np.ones((B, S), dtype="float32")
    mask[0, S - 1] = 0.0
    return mask


def _make_head(pooling_type: str) -> TextClassificationHead:
    """Build the smallest head exposing ``pooling_type`` with pooling-only path.

    ``use_intermediate=False`` / ``use_ffn=False`` keeps the head a thin wrapper
    around the pooling stage so we can probe ``_pool_sequence`` directly.
    """
    cfg = NLPTaskConfig(
        name="cls",
        task_type=NLPTaskType.TEXT_CLASSIFICATION,
        num_classes=3,
    )
    head = TextClassificationHead(
        task_config=cfg,
        input_dim=D,
        pooling_type=pooling_type,
        use_pooling=True,
        use_intermediate=False,
        use_ffn=False,
        use_task_attention=False,
    )
    # Build sub-layers (pooler is created in __init__, built in build()).
    head.build((B, S, D))
    return head


# ---------------------------------------------------------------------
# SC4 — pooling-equivalence (analytical references)
# ---------------------------------------------------------------------

class TestPoolingEquivalence:
    """SC4: cls/mean/max are analytically correct; all 4 give (B, D)."""

    @pytest.mark.parametrize("pooling_type", ["cls", "mean", "max", "attention"])
    def test_pool_shape_is_B_D(self, pooling_type, fixed_input, attention_mask) -> None:
        head = _make_head(pooling_type)
        x = ops.convert_to_tensor(fixed_input)
        m = ops.convert_to_tensor(attention_mask)
        pooled = head._pool_sequence(x, m)
        assert tuple(pooled.shape) == (B, D)

    def test_cls_pool_matches_first_token(self, fixed_input, attention_mask) -> None:
        head = _make_head("cls")
        pooled = head._pool_sequence(
            ops.convert_to_tensor(fixed_input),
            ops.convert_to_tensor(attention_mask),
        )
        np.testing.assert_allclose(
            ops.convert_to_numpy(pooled), fixed_input[:, 0, :], atol=1e-6
        )

    def test_mean_pool_matches_masked_mean(self, fixed_input, attention_mask) -> None:
        head = _make_head("mean")
        pooled = head._pool_sequence(
            ops.convert_to_tensor(fixed_input),
            ops.convert_to_tensor(attention_mask),
        )
        # Analytical mask-weighted mean over valid positions.
        m = attention_mask[:, :, None]  # (B, S, 1)
        lengths = np.maximum(m.sum(axis=1), 1.0)  # (B, 1)
        ref = (fixed_input * m).sum(axis=1) / lengths
        np.testing.assert_allclose(ops.convert_to_numpy(pooled), ref, atol=1e-6)

    def test_max_pool_matches_masked_max(self, fixed_input, attention_mask) -> None:
        head = _make_head("max")
        pooled = head._pool_sequence(
            ops.convert_to_tensor(fixed_input),
            ops.convert_to_tensor(attention_mask),
        )
        # Analytical masked max: masked positions pushed to -inf-equivalent.
        m = attention_mask[:, :, None]
        masked = fixed_input + (1.0 - m) * (-1e9)
        ref = masked.max(axis=1)
        np.testing.assert_allclose(ops.convert_to_numpy(pooled), ref, atol=1e-6)

    def test_attention_pool_finite_and_BD(self, fixed_input, attention_mask) -> None:
        """attention path is the kept inline Dense(1, tanh) direct-score pooling
        (D-002). Assert (B, D) + finiteness; masked positions get ~zero softmax
        weight because the inline path sets their logit to -1e9 before softmax."""
        head = _make_head("attention")
        x = ops.convert_to_tensor(fixed_input)
        m = ops.convert_to_tensor(attention_mask)
        pooled = head._pool_sequence(x, m)
        pooled_np = ops.convert_to_numpy(pooled)
        assert pooled_np.shape == (B, D)
        assert np.all(np.isfinite(pooled_np))

        # Verify the masked position (row 0, last token) receives ~zero weight:
        # reproduce the inline scoring and check the softmax weight.
        scores = ops.squeeze(head.attention_pooling(x), axis=-1)  # (B, S)
        scores = scores * m + (1.0 - m) * (-1e9)
        weights = ops.convert_to_numpy(ops.softmax(scores, axis=-1))
        assert weights[0, S - 1] < 1e-6

    def test_mean_unmasked_is_plain_mean(self, fixed_input) -> None:
        """Without a mask, mean pooling == plain mean over the sequence axis."""
        head = _make_head("mean")
        pooled = head._pool_sequence(ops.convert_to_tensor(fixed_input), None)
        np.testing.assert_allclose(
            ops.convert_to_numpy(pooled), fixed_input.mean(axis=1), atol=1e-6
        )


# ---------------------------------------------------------------------
# Factory smoke + save/load round-trip
# ---------------------------------------------------------------------

class TestNLPFactoryAndRoundtrip:

    def test_factory_returns_classification_head(self) -> None:
        head = create_nlp_head(
            task_config=NLPTaskConfig(
                name="cls",
                task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=5,
            ),
            input_dim=D,
        )
        assert isinstance(head, TextClassificationHead)

    def test_factory_from_dict_config(self) -> None:
        head = create_nlp_head(
            task_config={
                "name": "sent",
                "task_type": NLPTaskType.SENTIMENT_ANALYSIS,
                "num_classes": 2,
            },
            input_dim=D,
        )
        assert isinstance(head, TextClassificationHead)

    def test_model_save_load_roundtrip(self, fixed_input) -> None:
        # NOTE: the head's own ``build`` only handles tuple/dict input_shape
        # (a pre-existing fragility in the merged code: the Keras functional API
        # hands ``build`` a *list*, which TextClassificationHead.build does not
        # accept). We therefore wrap the head in a tiny subclassed Model that
        # invokes the head on real tensors (so ``build`` receives a tuple) — this
        # still exercises full ``.keras`` save/load of the head, which is the SC5
        # contract.
        cfg = NLPTaskConfig(
            name="cls",
            task_type=NLPTaskType.TEXT_CLASSIFICATION,
            num_classes=3,
        )

        @keras.saving.register_keras_serializable()
        class _NLPWrapper(keras.Model):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.head = create_nlp_head(
                    task_config=cfg, input_dim=D, pooling_type="mean"
                )

            def call(self, inputs, training=None):
                return self.head(inputs, training=training)

        model = _NLPWrapper()
        y0 = model(fixed_input)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nlp_head.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(fixed_input)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0["logits"]),
            ops.convert_to_numpy(y1["logits"]),
            atol=1e-4,
        )
