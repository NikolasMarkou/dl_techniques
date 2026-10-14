"""
Test suite for FFTNet (adaptive spectral filtering vision foundation model).

Covers construction (from_variant + create_fftnet), a forward pass, and the M2
full .keras save -> load -> identical-output round-trip. The FFTMixer uses a
documented raw-tf FFT path (accepted §L2-5 exception); the model still serializes
and round-trips cleanly. call() takes (B, H, W, 3) and returns a dict.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.vision.fftnet.model import FFTNet, create_fftnet


def _model():
    return create_fftnet(variant="tiny", image_size=32, patch_size=16)


def _images(batch=2):
    return np.random.default_rng(0).random((batch, 32, 32, 3)).astype("float32")


class TestFFTNet:

    def test_forward_dict(self):
        out = _model()(_images(), training=False)
        assert {"last_hidden_state", "cls_token", "patch_features"} <= set(out)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out["last_hidden_state"])))

    def test_mixer_actually_mixes_tokens(self):
        """FFTMixer must transform the TOKEN axis, not the feature axis.

        RED-proof: ``tf.signal.fft`` transforms the INNERMOST axis, so calling
        it on (B, N, D) transformed D. Every token was then processed
        independently and the layer was exactly permutation-EQUIVARIANT — the
        one property a token mixer must not have. The adaptive filter's
        ``mean(x, axis=1)`` is itself permutation-invariant, so it does not
        rescue the property either.

        Asserting the output shape, finiteness, or that the output merely
        differs from the input would all be VACUOUS here — all three held with
        the FFT on the wrong axis.

        The permutation must be neither a cyclic shift nor a reversal. A DFT is
        equivariant to both of those (they map to multiplication by a
        unit-modulus phase and to conjugation, and modReLU rescales by a
        function of the magnitude alone, which both preserve), so a reversal
        probe fails identically with and without the fix — it is a false guard,
        and the first version of this test was exactly that.
        """
        from dl_techniques.models.vision.fftnet.model import FFTMixer

        rng = np.random.default_rng(0)
        x = rng.random((2, 8, 16)).astype("float32")

        # A scrambled permutation: not the identity, not a reversal, not any
        # cyclic shift.
        perm = np.array([3, 1, 7, 0, 5, 2, 6, 4])
        assert not np.array_equal(perm, np.arange(8))
        assert not np.array_equal(perm, np.arange(8)[::-1])
        assert all(not np.array_equal(perm, np.roll(np.arange(8), s))
                   for s in range(8)), "permutation must not be a cyclic shift"
        inverse = np.argsort(perm)

        mixer = FFTMixer(embed_dim=16, mlp_hidden_dim=32)
        mixer.build((None, 8, 16))

        y = keras.ops.convert_to_numpy(mixer(x, training=False))
        # Permute the tokens, push through, then undo the permutation.
        y_perm = keras.ops.convert_to_numpy(
            mixer(x[:, perm, :].copy(), training=False))[:, inverse, :]

        assert not np.allclose(y, y_perm, atol=1e-5), (
            "FFTMixer is permutation-equivariant over tokens, so it performs no "
            "token mixing — the FFT is being applied to the feature axis")

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            FFTNet.from_variant("nonexistent")

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        before = keras.ops.convert_to_numpy(model(x, training=False)["last_hidden_state"])

        path = os.path.join(str(tmp_path), "fftnet.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["last_hidden_state"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="FFTNet differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------
# Gradient flow (plan-2026-08-19-a616f581 step 10)
# ---------------------------------------------------------------------

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


class TestFFTNetGradientFlow:
    """Every trainable weight must be on the backward graph.

    FFTNet's mixer runs a documented raw-``tf.signal`` FFT path (an accepted
    L2-5 exception). A complex-valued detour is a classic place for a gradient to
    be silently dropped -- and this suite's own history shows the forward-side
    version of that failure (the FFT was on the feature axis, and every shape and
    finiteness test passed). The per-weight claim is the backward-side analogue.
    """

    def test_gradients_reach_every_trainable_weight(self):
        model = _model()
        x = _images()
        model(x, training=False)  # a subclassed model is unbuilt until first call

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0
        assert max(v for v in report.values() if v is not None) > 0.0


# ---------------------------------------------------------------------
# The task models emit logits (plan-2026-10-14T062616-81938e81 step 6)
# ---------------------------------------------------------------------

from dl_techniques.models.vision.fftnet.model import (  # noqa: E402
    create_fftnet_classifier,
    create_fftnet_with_head,
)
from tests.numerics import reassociation_atol  # noqa: E402


class TestTheTaskModelsEmitLogits:
    """``create_fftnet_with_head`` classification returned ``{"logits": ...}``
    and segmentation a softmax map before step 6. Both are now one tensor of
    logits; the census row in ``test_a_vision_classifier_emits_logits.py``
    covers the classifier's fit step."""

    def test_the_classifier_returns_bare_logits_and_round_trips(self, tmp_path):
        keras.utils.set_random_seed(0)
        model = create_fftnet_classifier("tiny", num_classes=5, image_size=32,
                                         patch_size=16)
        x = _images(3)
        out = model(x, training=False)
        assert not isinstance(out, dict), type(out)
        logits = np.asarray(out)
        assert logits.shape == (3, 5)
        # A softmax row sums to 1 within this float32 rounding bound.
        atol = reassociation_atol([5], 1, 1.0)
        assert (logits < 0).any() and np.abs(logits.sum(axis=-1) - 1.0).max() > atol

        path = os.path.join(tmp_path, "fftnet_classifier.keras")
        model.save(path)
        # Default safe_mode: an identity Lambda in the head made this raise.
        loaded = keras.models.load_model(path)
        np.testing.assert_array_equal(np.asarray(loaded(x, training=False)), logits)

    def test_the_segmenter_returns_per_pixel_logits(self):
        keras.utils.set_random_seed(0)
        model = create_fftnet_with_head("tiny", task_type="segmentation",
                                        num_classes=4, image_size=32,
                                        patch_size=16)
        out = np.asarray(model(_images(2), training=False)).astype(np.float64)
        assert out.shape == (2, 32, 32, 4)
        assert (out < 0).any(), "no negative value: the map looks softmaxed"
        assert np.abs(out.sum(axis=-1) - 1.0).max() > reassociation_atol([4], 1, 1.0), (
            "every pixel's channels sum to 1: the map is normalized")
