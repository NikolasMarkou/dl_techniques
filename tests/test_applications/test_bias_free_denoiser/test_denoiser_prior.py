"""Fast CPU unit tests for the ``DenoiserPrior`` core wrapper.

Uses a TINY in-memory bias-free Keras model (no checkpoint, no GPU) to verify the
wrapper's mechanics: ``residual`` / ``denoise`` symbolic identities, ``ingest`` /
``denorm`` domain math, and the non-overlapping ``tile`` / ``untile`` round-trip.
The real 22 MB checkpoint is exercised separately (step 6 / drift-guard script),
never in this fast suite.
"""

import keras
import numpy as np
import pytest

from applications.bias_free_denoiser.denoiser_prior import DenoiserPrior


def _tiny_bias_free_model() -> keras.Model:
    """A minimal bias-free (no-bias) conv model on ``(None, None, 3)``.

    Not an identity — a single random-but-fixed 1x1 conv with ``use_bias=False`` so
    it is degree-1 homogeneous, enough to test ``residual = model(y) - y`` exactly.
    """
    inputs = keras.Input(shape=(None, None, 3))
    outputs = keras.layers.Conv2D(
        3, kernel_size=1, use_bias=False,
        kernel_initializer=keras.initializers.GlorotUniform(seed=0),
    )(inputs)
    return keras.Model(inputs, outputs, name="tiny_bias_free")


@pytest.fixture
def prior() -> DenoiserPrior:
    return DenoiserPrior(_tiny_bias_free_model())


class TestDomainHelpers:
    def test_ingest_uint8_maps_to_domain(self):
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)  # [1,1,3]
        out = DenoiserPrior.ingest(img)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, np.array([[[-0.5, 128 / 255 - 0.5, 0.5]]]),
                                   atol=1e-6)

    def test_ingest_maps_0_255_extremes(self):
        img = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        img[..., 0] = 255
        out = DenoiserPrior.ingest(img)
        assert float(out.min()) == pytest.approx(-0.5)
        assert float(out.max()) == pytest.approx(0.5)

    def test_ingest_float_0_1(self):
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = DenoiserPrior.ingest(img)
        np.testing.assert_allclose(out, np.array([[[-0.5, 0.0, 0.5]]]), atol=1e-6)

    def test_ingest_already_normalized_passthrough(self):
        img = np.array([[[-0.4, 0.0, 0.3]]], dtype=np.float32)
        out = DenoiserPrior.ingest(img)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_denorm_inverts_ingest_for_uint8(self):
        img = np.random.randint(0, 256, size=(1, 8, 8, 3)).astype(np.uint8)
        recovered = DenoiserPrior.denorm(DenoiserPrior.ingest(img))
        np.testing.assert_allclose(recovered, img.astype(np.float32) / 255.0, atol=1e-6)

    def test_denorm_maps_to_unit_interval(self):
        x = np.array([[[-0.5, 0.0, 0.5]]], dtype=np.float32)
        np.testing.assert_allclose(DenoiserPrior.denorm(x),
                                   np.array([[[0.0, 0.5, 1.0]]]), atol=1e-6)


class TestCoreMethods:
    def test_domain_attributes(self, prior):
        assert prior.domain_center == 0.0
        assert prior.domain_halfwidth == 0.5
        assert prior.model is not None

    def test_denoise_shape(self, prior):
        y = np.random.uniform(-0.5, 0.5, size=(2, 16, 16, 3)).astype(np.float32)
        d = keras.ops.convert_to_numpy(prior.denoise(y))
        assert d.shape == y.shape

    def test_residual_equals_model_minus_input(self, prior):
        y = np.random.uniform(-0.5, 0.5, size=(1, 16, 16, 3)).astype(np.float32)
        f = keras.ops.convert_to_numpy(prior.residual(y))
        d = keras.ops.convert_to_numpy(prior.model(y, training=False))
        np.testing.assert_allclose(f, d - y, atol=1e-6)
        assert f.shape == y.shape


class TestTiling:
    def test_tile_untile_roundtrip_512(self):
        img = np.random.uniform(-0.5, 0.5, size=(1, 512, 512, 3)).astype(np.float32)
        tiles, meta = DenoiserPrior.tile(img, tile_size=256)
        # 512 / 256 = 2 per axis -> 4 non-overlapping blocks.
        assert tiles.shape == (4, 256, 256, 3)
        assert (meta["nh"], meta["nw"]) == (2, 2)
        recon = DenoiserPrior.untile(tiles, meta)
        assert recon.shape == img.shape
        np.testing.assert_array_equal(recon, img)

    def test_tile_untile_roundtrip_with_padding(self):
        # 300 is not a multiple of 256 -> padded to 512, cropped back exactly.
        img = np.random.uniform(-0.5, 0.5, size=(2, 300, 256, 3)).astype(np.float32)
        tiles, meta = DenoiserPrior.tile(img, tile_size=256)
        assert tiles.shape == (2 * 2 * 1, 256, 256, 3)
        recon = DenoiserPrior.untile(tiles, meta)
        assert recon.shape == img.shape
        np.testing.assert_array_equal(recon, img)
