"""``BeitForMaskedImageModeling`` -- the MIM head over the backbone.

Moved verbatim from ``test_model.py`` (class ``TestBeitForMaskedImageModeling``,
section 7) during the step-8 decomposition of plan-2026-08-24T074054-247151fd.

This file owns one head end to end: its vocabulary, its output shape, its round trip,
and -- the highest-value assertion in the package --
``test_the_head_reads_the_patch_tokens_not_a_shifted_window``, which pins WHICH tokens
the prediction head consumes. An off-by-one slice there leaves every shape correct and
every other test green while the model predicts the wrong patch.
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
    DEFAULT_VOCAB_SIZE,
    BeitForMaskedImageModeling,
)
from tests.test_models.test_beit.beit_test_geometry import (
    IMG,
    NUM_PATCHES,
    SEQ_LEN,
    EPS,
    VOCAB,
    _tiny,
    _images,
    _mask,
    _mim,
)


class TestBeitForMaskedImageModeling:

    def test_default_vocab_size_is_the_dalle_codebook(self):
        assert DEFAULT_VOCAB_SIZE == 8192
        assert _mim().backbone.name == BACKBONE_NAME

    @pytest.mark.parametrize("variant", ['tiny', 'small'])
    def test_output_shape_excludes_the_cls_position(self, variant):
        """(B, N, vocab) -- NOT (B, N+1, vocab).

        Why this can fail if the implementation is wrong: forgetting the ``[:, 1:, :]``
        slice yields ``N + 1`` logits, which still trains against an ``(B, N)`` target
        only if the loss silently broadcasts -- otherwise it puts every target off by
        one patch. Either way there is no architectural error to see.
        """
        model = _mim(variant)
        out = model((_images(), _mask()), training=False)
        assert tuple(out.shape) == (2, NUM_PATCHES, VOCAB)
        assert model.compute_output_shape(
            [(2,) + IMG, (2, NUM_PATCHES)]
        ) == (2, NUM_PATCHES, VOCAB)

    def test_the_head_reads_the_patch_tokens_not_a_shifted_window(self):
        """Pin the cls slice by IDENTITY, not by shape.

        WHY THIS TEST EXISTS (adversarial review, iteration 1): replacing
        ``tokens[:, 1:, :]`` with ``tokens[:, :-1, :]`` in
        ``BeitForMaskedImageModeling.call`` — the exact off-by-one this package's own
        README §14 Issue 2 describes as producing "a finite, plausible loss curve and
        no error" — left 91/91 model tests GREEN. The neighbouring
        ``test_output_shape_excludes_the_cls_position`` cannot see it: the mutated
        code emits ``(B, N, vocab)`` too. Under that mutation every MIM target is
        attributed to the wrong patch and output index ``i`` is patch ``i - 1``, with
        the cls token standing in for the last patch.

        The assertion is therefore an EQUALITY against the head applied to the
        correct slice, plus a control proving the two slices are distinguishable at
        this geometry. Note the tempting alternative — "position 0's logits must not
        depend on the cls token" — is a FALSE invariant: the cls token reaches every
        patch token through self-attention, so it legitimately influences all of
        them. Only the slice boundary itself can be pinned.

        This is also the test for the ``D-012`` anchor at the slice site in
        ``models/vision/beit/model.py``; an anchor with no test that can fail is a comment.
        """
        model = _mim(drop_path_rate=0.0)
        model.build((None,) + IMG)
        x = _images()

        tokens = ops.convert_to_numpy(model.backbone(x, training=False))
        assert tokens.shape == (2, SEQ_LEN, model.hidden_size)

        def _head(slice_: np.ndarray) -> np.ndarray:
            normed = model.decoder_norm(slice_, training=False)
            return ops.convert_to_numpy(model.decoder_head(normed))

        expected = _head(tokens[:, 1:, :])      # cls dropped — the shipped slice
        shifted = _head(tokens[:, :-1, :])      # last patch dropped — the mutant

        # Setup assertion (stays GREEN under the mutation): the two candidate slices
        # really do produce different logits here, so matching one is informative.
        assert not np.allclose(expected, shifted, atol=1e-3), (
            "the two slices are indistinguishable at this geometry — the guard "
            "would pass either way"
        )

        out = ops.convert_to_numpy(model(x, training=False))
        # IDENTITY ASSERTION — RED when the slice is `[:, :-1, :]`.
        np.testing.assert_allclose(
            out, expected, atol=1e-5, rtol=0,
            err_msg=(
                "the MIM head is not reading tokens[:, 1:, :]: output index i must "
                "be the projection of backbone token i + 1 (patch i). Any other "
                "slice still yields (B, N, vocab) and a plausible loss curve while "
                "attributing every code-id target to the wrong patch."
            ),
        )

    def test_forward_without_a_mask_is_accepted(self):
        out = _mim()(_images(), training=False)
        assert tuple(out.shape) == (2, NUM_PATCHES, VOCAB)

    def test_output_is_logits_not_probabilities(self):
        """BOTH halves: a value outside [0, 1] is reachable AND no softmax exists."""
        model = _mim()
        model.build((None,) + IMG)
        # Zero the kernel and PIN the bias, so the head's output is exactly the bias.
        # A constant kernel would NOT work: `decoder_norm`'s output is zero-mean over
        # the feature axis, so a constant kernel maps every token to ~0.0 -- inside
        # [0, 1] up to float noise, which is a coin flip, not a test.
        pinned = np.linspace(-5.0, 5.0, VOCAB).astype('float32')
        model.decoder_head.set_weights([
            np.zeros_like(ops.convert_to_numpy(model.decoder_head.kernel)),
            pinned,
        ])
        out = ops.convert_to_numpy(model(_images(), training=False))
        np.testing.assert_allclose(
            out, np.broadcast_to(pinned, out.shape), atol=1e-5, rtol=0
        )
        assert out.min() < 0.0 and out.max() > 1.0, "head does not emit logits"
        # And structurally: nothing in the head applies a softmax.
        assert model.decoder_head.activation is keras.activations.linear
        for layer in model._flatten_layers(include_self=False):
            assert not isinstance(layer, keras.layers.Softmax), layer.name
            act = getattr(layer, 'activation', None)
            assert act is not keras.activations.softmax, layer.name
        # Nor does the output already sum to 1 over the vocab axis.
        probs = ops.convert_to_numpy(
            keras.activations.softmax(model(_images(), training=False))
        )
        assert not np.allclose(out.sum(axis=-1), 1.0, atol=1e-3)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
        assert not np.allclose(out, probs, atol=1e-3), (
            "the head output is already a probability distribution"
        )

    def test_head_layers_all_carry_the_decoder_prefix(self):
        model = _mim()
        model.build((None,) + IMG)
        head_names = {l.name for l in model.layers} - {BACKBONE_NAME}
        assert head_names == {"decoder_norm", "decoder_head"}
        assert all(n.startswith("decoder_") for n in head_names)

    def test_decoder_norm_uses_the_backbone_epsilon(self):
        model = _mim()
        assert model.decoder_norm.epsilon == EPS

    def test_gradients_reach_the_head_and_the_trunk(self):
        model = _mim(drop_path_rate=0.0)
        model.build((None,) + IMG)
        x, m = tf.constant(_images()), tf.constant(_mask())
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model((x, m), training=True)))
        grads = tape.gradient(loss, model.trainable_variables)
        dead = [v.path for g, v in zip(grads, model.trainable_variables) if g is None]
        assert dead == [], f"no gradient reached: {dead}"

    def test_invalid_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
            BeitForMaskedImageModeling(backbone=_tiny(), vocab_size=0)

    def test_a_non_backbone_is_refused(self):
        with pytest.raises(TypeError, match="backbone must be a BeitModel"):
            BeitForMaskedImageModeling(backbone="not a model")

    def test_keras_roundtrip_preserves_values(self):
        model = _mim()
        model.build((None,) + IMG)
        x, m = _images(), _mask()
        before = ops.convert_to_numpy(model((x, m), training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "beit_mim.keras")
            model.save(path)
            restored = keras.models.load_model(path)
            after = ops.convert_to_numpy(restored((x, m), training=False))
        np.testing.assert_allclose(before, after, atol=1e-6, rtol=0)
        assert restored.vocab_size == VOCAB
        assert restored.backbone.name == BACKBONE_NAME
