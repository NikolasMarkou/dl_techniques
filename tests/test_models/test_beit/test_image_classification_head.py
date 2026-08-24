"""``BeitForImageClassification`` -- the classification head over the backbone.

Moved verbatim from ``test_model.py`` (class ``TestBeitForImageClassification``,
section 8) during the step-8 decomposition of plan-2026-08-24T074054-247151fd.

Mirror image of ``test_masked_image_modeling_head.py``: one head end to end, its
output shape, its round trip, and ``test_mean_pooling_excludes_the_cls_token``, which
pins the ``use_mean_pooling`` fork -- the one place where pooling over the whole
sequence instead of the patch tokens is a silent accuracy loss, not an error.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.vision.beit import (
    BACKBONE_NAME,
    BeitForImageClassification,
)
from tests.test_models.test_beit.beit_test_geometry import (
    IMG,
    EPS,
    _tiny,
    _images,
    _classifier,
)

class TestBeitForImageClassification:

    @pytest.mark.parametrize("variant", ['tiny', 'small'])
    def test_output_shape(self, variant):
        model = _classifier(variant)
        out = model(_images(), training=False)
        assert tuple(out.shape) == (2, 7)
        assert model.compute_output_shape((2,) + IMG) == (2, 7)

    def test_output_is_logits_not_probabilities(self):
        model = _classifier()
        model.build((None,) + IMG)
        # Same reasoning as the MIM head: `head_norm` is zero-mean over the feature
        # axis, so a constant kernel produces ~0.0 and the [0,1] check becomes a coin
        # flip. Zero the kernel and pin the bias instead.
        pinned = np.linspace(-5.0, 5.0, 7).astype('float32')
        model.head_classifier.set_weights([
            np.zeros_like(ops.convert_to_numpy(model.head_classifier.kernel)),
            pinned,
        ])
        out = ops.convert_to_numpy(model(_images(), training=False))
        np.testing.assert_allclose(
            out, np.broadcast_to(pinned, out.shape), atol=1e-5, rtol=0
        )
        assert out.min() < 0.0 and out.max() > 1.0, "head does not emit logits"
        assert model.head_classifier.activation is keras.activations.linear
        for layer in model._flatten_layers(include_self=False):
            assert not isinstance(layer, keras.layers.Softmax), layer.name
            act = getattr(layer, 'activation', None)
            assert act is not keras.activations.softmax, layer.name
        assert not np.allclose(out.sum(axis=-1), 1.0, atol=1e-3)

    def test_mean_pooling_excludes_the_cls_token(self):
        """A-7 / BEiT's own convention: pool the PATCH tokens only.

        Why this can fail if the implementation is wrong: pooling over all N+1 tokens
        is a one-character change (`exclude_positions=[]`) that changes nothing
        observable in a shape or a loss curve. Here the cls-token weight is perturbed
        and the pooled representation must NOT move.
        """
        model = _classifier(dropout_rate=0.0)
        model.build((None,) + IMG)
        assert model.head_pool.exclude_positions == [0]
        assert model.head_pool.strategy == ['mean']

        x = _images()
        # Pool the trunk output directly: perturbing position 0 must be invisible.
        tokens = ops.convert_to_numpy(model.backbone(x, training=False))
        pooled_a = ops.convert_to_numpy(model.head_pool(tokens))
        tokens_b = tokens.copy()
        tokens_b[:, 0, :] += 100.0
        pooled_b = ops.convert_to_numpy(model.head_pool(tokens_b))
        np.testing.assert_allclose(pooled_a, pooled_b, atol=1e-5, rtol=0)
        # Control: perturbing a PATCH position must move it.
        tokens_c = tokens.copy()
        tokens_c[:, 1, :] += 100.0
        assert not np.allclose(
            pooled_a, ops.convert_to_numpy(model.head_pool(tokens_c)), atol=1e-3
        )

    def test_cls_pooling_mode_has_no_head_norm(self):
        """D-007's other branch: use_mean_pooling=False -> the trunk norms, not us."""
        model = _classifier(use_mean_pooling=False)
        model.build((None,) + IMG)
        assert model.head_pool is None
        assert model.head_norm is None
        assert model.backbone.final_norm is not None
        assert tuple(model(_images(), training=False).shape) == (2, 7)
        names = {l.name for l in model.layers}
        assert "head_norm" not in names
        assert "head_pool" not in names

    def test_head_layers_all_carry_the_head_prefix(self):
        model = _classifier()
        model.build((None,) + IMG)
        head_names = {l.name for l in model.layers} - {BACKBONE_NAME}
        assert head_names == {
            "head_pool", "head_norm", "head_dropout", "head_classifier"
        }
        assert all(n.startswith("head_") for n in head_names)

    def test_head_norm_uses_the_backbone_epsilon(self):
        assert _classifier().head_norm.epsilon == EPS

    def test_gradients_reach_everything_except_the_dead_mask_token(self):
        """The mask token is the ONE weight with no gradient here -- by design.

        The classifier never calls ``MaskTokenApply``, so its mask token receives no
        gradient; it exists only to keep this trunk weight-identical to the MIM trunk.
        Asserting it is the SOLE exception turns that contract into a positive claim
        rather than a blanket exemption: any OTHER unreachable weight still fails.
        """
        model = _classifier(drop_path_rate=0.0)
        model.build((None,) + IMG)
        x = tf.constant(_images())
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, training=True)))
        grads = tape.gradient(loss, model.trainable_variables)
        dead = [v.path for g, v in zip(grads, model.trainable_variables) if g is None]
        assert len(dead) == 1, f"unexpected dead weights: {dead}"
        assert dead[0].endswith("mask_token/mask_token"), dead

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(num_classes=0), "num_classes must be a positive integer"),
            (dict(num_classes=3, dropout_rate=1.5), r"dropout_rate must be in \[0, 1\]"),
        ],
    )
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BeitForImageClassification(backbone=_tiny(), **kwargs)

    def test_keras_roundtrip_preserves_values(self):
        model = _classifier()
        model.build((None,) + IMG)
        x = _images()
        before = ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_classifier.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            after = ops.convert_to_numpy(restored(x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)
        assert restored.num_classes == 7
        assert restored.backbone.name == BACKBONE_NAME
