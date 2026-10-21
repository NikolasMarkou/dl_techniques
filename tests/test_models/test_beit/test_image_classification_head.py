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

def _head(model):
    """The classification head, whichever of the two the model built.

    ``use_mean_pooling=True`` gets ``_MeanPoolingHead`` (SequencePooling ->
    norm -> dropout -> Dense); the CLS-token path gets the shared
    ``ClassificationHead``. Both live one level down now, so nothing is
    reachable as ``model.head_classifier`` any more.
    """
    return model.classification_head


def _logits(model, x):
    """The model's own output: the head's logits, one tensor, no softmax."""
    out = model(x, training=False)
    return keras.ops.convert_to_numpy(out)


class TestBeitForImageClassification:

    @pytest.mark.parametrize("variant", ['tiny', 'small'])
    def test_output_shape(self, variant):
        model = _classifier(variant)
        out = model(_images(), training=False)
        # The model unwraps the head's 'logits' and returns that one tensor.
        assert not isinstance(out, dict)
        assert out.shape == (2, 7)
        assert np.all(np.isfinite(out))
        assert model.compute_output_shape((2,) + IMG) == (2, 7)

    def test_output_is_logits_not_probabilities(self):
        model = _classifier()
        model.build((None,) + IMG)
        head = _head(model)
        classifier = head.classifier
        # Same reasoning as the MIM head: `head_norm` is zero-mean over the feature
        # axis, so a constant kernel produces ~0.0 and the [0,1] check becomes a coin
        # flip. Zero the kernel and pin the bias instead.
        pinned = np.linspace(-5.0, 5.0, 7).astype('float32')
        classifier.set_weights([
            np.zeros_like(keras.ops.convert_to_numpy(classifier.kernel)),
            pinned,
        ])
        out = _logits(model, _images())
        np.testing.assert_allclose(
            out, np.broadcast_to(pinned, out.shape), atol=1e-5, rtol=0
        )
        assert out.min() < 0.0 and out.max() > 1.0, "head does not emit logits"
        assert classifier.activation is keras.activations.linear
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
        pool = _head(model).pool
        assert pool.exclude_positions == [0]
        assert pool.strategy == ['mean']

        x = _images()
        # Pool the trunk output directly: perturbing position 0 must be invisible.
        tokens = keras.ops.convert_to_numpy(model.backbone(x, training=False))
        pooled_a = keras.ops.convert_to_numpy(pool(tokens))
        tokens_b = tokens.copy()
        tokens_b[:, 0, :] += 100.0
        pooled_b = keras.ops.convert_to_numpy(pool(tokens_b))
        np.testing.assert_allclose(pooled_a, pooled_b, atol=1e-5, rtol=0)
        # Control: perturbing a PATCH position must move it.
        tokens_c = tokens.copy()
        tokens_c[:, 1, :] += 100.0
        assert not np.allclose(
            pooled_a, keras.ops.convert_to_numpy(pool(tokens_c)), atol=1e-3
        )

    def test_cls_pooling_mode_has_no_head_norm(self):
        """D-007's other branch: use_mean_pooling=False -> the trunk norms, not us.

        The CLS branch is BEiT's reference head (unilm ``x[:, 0]`` after the
        trunk's ``self.norm``, then ``self.head``): the shared
        ``ClassificationHead`` at ``pooling_type='cls_token'`` with no hidden
        block and no norm of any kind -- it builds no ``norm`` attribute since
        plan-2026-10-14T062616-81938e81 step 1, and no ``pooled_norm`` -- so the
        head owns exactly the classifier's kernel and bias.
        """
        model = _classifier(use_mean_pooling=False)
        model.build((None,) + IMG)
        head = _head(model)
        # The CLS branch builds the shared ClassificationHead, which pools
        # position 0 directly and has no SequencePooling of its own.
        assert not hasattr(head, "pool")
        assert not hasattr(head, "norm"), "the head builds a norm of its own"
        assert not hasattr(head, "pooled_norm"), "the head norms the cls state"
        assert head.dense_blocks == []
        assert {w.path.split("/")[-2] for w in head.weights} == {"classifier"}
        assert model.backbone.final_norm is not None
        out = model(_images(), training=False)
        assert out.shape == (2, 7)
        names = {w.path.split("/", 1)[-1] for w in model.weights}
        assert not any("head_norm" in n for n in names), names
        assert not any("head_pool" in n for n in names), names

    def test_head_layers_all_carry_the_head_prefix(self):
        model = _classifier()
        model.build((None,) + IMG)
        head = _head(model)
        # The head's own layers keep the `head_` prefix; they now live under the
        # head rather than flat on the model. `build()` is what creates them --
        # `_MeanPoolingHead` builds its sub-layers there, so an unbuilt head has
        # no `.layers` at all.
        # `_MeanPoolingHead` builds its sub-layers in `build()`, so it has no
        # `.layers` to enumerate -- Keras never tracked them as children. Assert
        # on the four it owns, by name.
        owned = [head.pool, head.norm, head.dropout, head.classifier]
        assert {l.name for l in owned} == {
            "head_pool", "head_norm", "head_dropout", "head_classifier"
        }
        assert all(l.name.startswith("head_") for l in owned)
        # And the head itself is the only non-backbone layer on the model.
        assert {l.name for l in model.layers} == {BACKBONE_NAME, head.name}

    def test_head_norm_uses_the_backbone_epsilon(self):
        model = _classifier()
        model.build((None,) + IMG)
        assert _head(model).norm.epsilon == EPS

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
            out = model(x, training=True)
            probs = out
            loss = tf.reduce_mean(tf.square(probs))
        grads = tape.gradient(loss, model.trainable_variables)
        dead = [v.path for g, v in zip(grads, model.trainable_variables) if g is None]
        assert len(dead) == 1, f"unexpected dead weights: {dead}"
        assert dead[0].endswith("mask_token/mask_token"), dead

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (dict(num_classes=0), "num_classes must be a positive integer"),
            (dict(num_classes=3, dropout_rate=1.5), r"dropout_rate must be in \[0, 1\)"),
            (dict(num_classes=3, dropout_rate=1.0), r"dropout_rate must be in \[0, 1\)"),
        ],
    )
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BeitForImageClassification(backbone=_tiny(), **kwargs)

    def test_keras_roundtrip_preserves_values(self):
        model = _classifier()
        model.build((None,) + IMG)
        x = _images()
        before = model(x, training=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_classifier.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            after = restored(x, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(before), keras.ops.convert_to_numpy(after),
            atol=1e-6, rtol=0,
        )
        assert restored.num_classes == 7
        assert restored.backbone.name == BACKBONE_NAME
