"""
Smoke + serialization tests for VQVAERotationTrick.

Covers:
- Init + forward shape on (B, 32, 32, 3)
- One-step fit produces finite loss
- .keras save/load round-trip atol=1e-5
- norm_type knob exercised: layer_norm + rms_norm
- One case per gradient_mode (rotation / reflection / no_grad_scale / ste)
- create_normalization_layer is actually used inside auto encoder/decoder
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.vq_vae_rotation.model import VQVAERotationTrick


@pytest.fixture
def sample_images():
    keras.utils.set_random_seed(0)
    return ops.cast(keras.random.uniform((4, 32, 32, 3), 0.0, 1.0), "float32")


class TestIndicesRoundTripMatchesCall:
    """``decode_from_indices(encode_to_indices(x))`` must equal ``model(x)``.

    Also the reachability statement: ``distance_mode`` is a public constructor
    argument on ``VQVAERotationTrick`` (and on ``create_vq_vae_rotation``), defaulting
    to ``'euclidean'``, so ``'cosine'`` is opt-in but reachable from the model — not
    layer-only. That is what makes the asymmetry a model defect and not just a layer
    one: a trained index prior could not reproduce the decoder's training-time input.

    RED before the fix (plan step 19): the cosine arm's two paths disagreed by the
    ``||x|| / ||e_k||`` factor ``_lookup`` applied and ``quantize_from_indices``
    omitted. The euclidean arm is the control and passed both ways.
    """

    @pytest.mark.parametrize("distance_mode", ["euclidean", "cosine"])
    def test_decode_from_indices_matches_call(self, sample_images, distance_mode):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
            gradient_mode="ste", distance_mode=distance_mode,
        )
        direct = ops.convert_to_numpy(m(sample_images, training=False))
        via_indices = ops.convert_to_numpy(
            m.decode_from_indices(m.encode_to_indices(sample_images))
        )
        np.testing.assert_allclose(direct, via_indices, atol=1e-5, rtol=1e-5)


class TestForward:
    def test_forward_shape(self, sample_images):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
        )
        out = m(sample_images)
        assert out.shape == sample_images.shape


class TestTraining:
    def test_one_step_fit(self, sample_images):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
        )
        m.compile(optimizer="adam")
        hist = m.fit(sample_images, sample_images, epochs=2, batch_size=2, verbose=0)
        losses = hist.history["loss"]
        assert all(np.isfinite(l) for l in losses)


class TestNormTypes:
    @pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm"])
    def test_norm_type_build(self, sample_images, norm_type):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type=norm_type,
        )
        out = m(sample_images)
        assert out.shape == sample_images.shape


class TestGradientModes:
    @pytest.mark.parametrize(
        "mode", ["rotation", "reflection", "no_grad_scale", "ste"]
    )
    def test_grad_mode_forward(self, sample_images, mode):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            gradient_mode=mode,
            input_shape=(32, 32, 3), hidden_channels=16,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
        )
        out = m(sample_images)
        assert out.shape == sample_images.shape


class TestSerialization:
    def test_save_load_round_trip(self, sample_images):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="layer_norm",
        )
        orig = m(sample_images, training=False)
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "vqvae_rot.keras")
            m.save(fp)
            loaded = keras.models.load_model(fp)
            new = loaded(sample_images, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(orig),
                ops.convert_to_numpy(new),
                atol=1e-5, rtol=1e-5,
            )

    def test_save_load_rms_norm(self, sample_images):
        m = VQVAERotationTrick(
            num_embeddings=32, embedding_dim=16,
            input_shape=(32, 32, 3), hidden_channels=32,
            downsample_factor=4, num_res_blocks=1, norm_type="rms_norm",
        )
        orig = m(sample_images, training=False)
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "vqvae_rot_rms.keras")
            m.save(fp)
            loaded = keras.models.load_model(fp)
            new = loaded(sample_images, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(orig),
                ops.convert_to_numpy(new),
                atol=1e-5, rtol=1e-5,
            )


class TestFactoryIntegration:
    def test_norm_factory_in_source(self):
        """SC8: create_normalization_layer must be referenced in the model module."""
        import inspect
        from dl_techniques.models.vq_vae_rotation import model as model_module
        src = inspect.getsource(model_module)
        assert "create_normalization_layer" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestRotationTrickForwardIdentity:
    """Every gradient_mode must emit the CODEBOOK vector in the forward pass."""

    @pytest.mark.parametrize(
        "mode", ["rotation", "reflection", "no_grad_scale", "ste"])
    def test_forward_equals_the_indexed_codebook_vector(self, mode):
        """call() must agree with encode -> quantize_from_indices -> decode.

        RED-proof for `reflection`: it reflected about the hyperplane with
        normal (u + v), which maps u -> -v, so its forward output was exactly
        -q. A single Householder reflection cannot map u -> +v about that
        normal; the two-reflection `rotation` path corrects for it and
        `reflection` had no second term.

        The existing suite parametrizes all four modes and asserts only the
        output SHAPE, which is invariant under a sign flip -- that is why this
        survived.
        """
        from dl_techniques.layers.vector_quantizer_rotation_trick import (
            VectorQuantizerRotationTrick,
        )

        keras.utils.set_random_seed(0)
        vq = VectorQuantizerRotationTrick(
            num_embeddings=16, embedding_dim=8, gradient_mode=mode)
        vq.build((None, 8))

        z = np.random.default_rng(0).normal(size=(32, 8)).astype("float32")

        out = keras.ops.convert_to_numpy(vq(z, training=False))
        idx = vq.get_codebook_indices(z)
        ref = keras.ops.convert_to_numpy(vq.quantize_from_indices(idx))

        # Sign-discriminating and scale-free: compare the distance to +ref
        # against the distance to -ref. This cannot be satisfied by loosening a
        # tolerance, and it is exactly the axis the defect lived on.
        d_plus = float(np.abs(out - ref).max())
        d_minus = float(np.abs(out + ref).max())
        assert d_plus < 0.1 * d_minus, (
            f"gradient_mode={mode!r}: call() is closer to -q ({d_minus:.3e}) "
            f"than it should be relative to +q ({d_plus:.3e}) -- the forward "
            f"pass is not emitting the codebook vector")

        # The residual floor is the rotation-trick arithmetic in float32 with
        # the eps-regularised norms; it is ~5.3e-05 for all three rotation
        # modes and ~3e-08 for the exact 'ste' path.
        np.testing.assert_allclose(
            out, ref, atol=1e-4,
            err_msg=(f"gradient_mode={mode!r}: call() does not emit the "
                     f"codebook vector, so it disagrees with the index path"))
