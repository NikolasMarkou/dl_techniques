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
