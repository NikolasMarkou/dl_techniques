"""Tests for the Flux2 KL-VAE (step 7).

Uses the TINY ``AutoEncoderParams`` preset (ch=32, ch_mult=(1,2), z=8,
out_ch=3) with a small 32x32 spatial input to stay fast. Covers:

- Encoder latent spatial / channel shapes (downsample once -> 16x16, 2*z).
- GroupNorm divisibility (all channels /32) + the deliberate bad-params guard.
- AutoEncoder encode (mu/logvar shapes), decode (back to input res), call.
- Decoder-only inference path on an arbitrary latent.
- Finite gradients through encode -> sample -> decode.
- ``.keras`` round-trip comparing DETERMINISTIC encoder mu (Sampling is
  stochastic, so we never compare sampled recon -- per LESSONS VAE rule).
- AttnBlock self-test: residual, shape-preserving.

GPU: scope to GPU1 (the runner passes ``CUDA_VISIBLE_DEVICES=1``).
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.models.ideogram4.config import (
    AutoEncoderParams,
    get_ideogram4_config,
)
from dl_techniques.models.ideogram4.vae import (
    ResnetBlock,
    AttnBlock,
    Downsample,
    Upsample,
    Encoder,
    Decoder,
    AutoEncoder,
    create_ideogram4_autoencoder,
)


# ---------------------------------------------------------------------
# fixtures / constants
# ---------------------------------------------------------------------

BATCH = 2
SPATIAL = 32  # input edge


@pytest.fixture(scope="module")
def tiny_ae_params() -> AutoEncoderParams:
    _, ae = get_ideogram4_config("tiny")
    return ae


@pytest.fixture(scope="module")
def tiny_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((BATCH, SPATIAL, SPATIAL, 3)).astype("float32")


def _latent_spatial(ae: AutoEncoderParams, edge: int) -> int:
    """Spatial edge after (len(ch_mult) - 1) stride-2 downsamples (ceil/2)."""
    s = edge
    for _ in range(len(ae.ch_mult) - 1):
        s = (s + 1) // 2
    return s


# ---------------------------------------------------------------------
# sub-layer unit tests
# ---------------------------------------------------------------------


class TestResnetBlock:
    def test_identity_skip_shape(self):
        blk = ResnetBlock(in_channels=32, out_channels=32)
        x = keras.ops.convert_to_tensor(
            np.random.randn(2, 8, 8, 32).astype("float32")
        )
        y = blk(x)
        assert tuple(y.shape) == (2, 8, 8, 32)
        assert blk.nin_shortcut is None

    def test_projection_skip_channel_change(self):
        blk = ResnetBlock(in_channels=32, out_channels=64)
        x = keras.ops.convert_to_tensor(
            np.random.randn(2, 8, 8, 32).astype("float32")
        )
        y = blk(x)
        assert tuple(y.shape) == (2, 8, 8, 64)
        assert blk.nin_shortcut is not None
        assert blk.compute_output_shape((2, 8, 8, 32)) == (2, 8, 8, 64)

    def test_bad_channels_raise(self):
        with pytest.raises(ValueError):
            ResnetBlock(in_channels=30, out_channels=32)
        with pytest.raises(ValueError):
            ResnetBlock(in_channels=32, out_channels=30)


class TestAttnBlock:
    def test_shape_and_residual(self):
        blk = AttnBlock(channels=32)
        x = keras.ops.convert_to_tensor(
            np.random.randn(2, 4, 4, 32).astype("float32")
        )
        y = blk(x)
        # Shape preserved.
        assert tuple(y.shape) == (2, 4, 4, 32)
        # Residual: output differs from input (proj_out is non-trivially init'd)
        # but is the same shape. We perturb proj_out by running once; identity
        # would require proj_out==0, which default init does not give.
        assert not np.allclose(
            keras.ops.convert_to_numpy(y),
            keras.ops.convert_to_numpy(x),
            atol=1e-6,
        )

    def test_bad_channels_raise(self):
        with pytest.raises(ValueError):
            AttnBlock(channels=30)


class TestDownsample:
    def test_halves_spatial_asym_pad(self):
        ds = Downsample(channels=16)
        x = keras.ops.convert_to_tensor(
            np.random.randn(2, 32, 32, 8).astype("float32")
        )
        y = ds(x)
        # 32 -> (32 + 1 - 3)//2 + 1 = 16.
        assert tuple(y.shape) == (2, 16, 16, 16)
        assert ds.compute_output_shape((2, 32, 32, 8)) == (2, 16, 16, 16)

    def test_odd_spatial(self):
        ds = Downsample(channels=16)
        x = keras.ops.convert_to_tensor(
            np.random.randn(1, 15, 15, 8).astype("float32")
        )
        y = ds(x)
        # 15 -> (15 + 1 - 3)//2 + 1 = 7.
        assert tuple(y.shape) == (1, 7, 7, 16)


class TestUpsample:
    def test_doubles_spatial(self):
        us = Upsample(channels=16)
        x = keras.ops.convert_to_tensor(
            np.random.randn(2, 8, 8, 32).astype("float32")
        )
        y = us(x)
        assert tuple(y.shape) == (2, 16, 16, 16)
        assert us.compute_output_shape((2, 8, 8, 32)) == (2, 16, 16, 16)


# ---------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------


class TestEncoder:
    def test_latent_shape(self, tiny_ae_params, tiny_image):
        ae = tiny_ae_params
        enc = Encoder(
            resolution=ae.resolution,
            in_channels=ae.in_channels,
            ch=ae.ch,
            ch_mult=ae.ch_mult,
            num_res_blocks=ae.num_res_blocks,
            z_channels=ae.z_channels,
        )
        out = enc(keras.ops.convert_to_tensor(tiny_image))
        lat = _latent_spatial(ae, SPATIAL)  # 32 -> 16 (one downsample)
        assert tuple(out.shape) == (BATCH, lat, lat, 2 * ae.z_channels)
        assert enc.compute_output_shape((BATCH, SPATIAL, SPATIAL, 3)) == (
            BATCH,
            lat,
            lat,
            2 * ae.z_channels,
        )

    def test_all_channels_div32(self, tiny_ae_params):
        # ch=32, ch_mult=(1,2) -> stages 32, 64, both /32.
        ae = tiny_ae_params
        assert ae.ch % 32 == 0
        for m in ae.ch_mult:
            assert (ae.ch * m) % 32 == 0


class TestDecoder:
    def test_reconstruct_shape(self, tiny_ae_params):
        ae = tiny_ae_params
        dec = Decoder(
            resolution=ae.resolution,
            ch=ae.ch,
            out_channels=ae.out_ch,
            ch_mult=ae.ch_mult,
            num_res_blocks=ae.num_res_blocks,
            z_channels=ae.z_channels,
        )
        lat = _latent_spatial(ae, SPATIAL)  # 16
        z = keras.ops.convert_to_tensor(
            np.random.randn(BATCH, lat, lat, ae.z_channels).astype("float32")
        )
        out = dec(z)
        assert tuple(out.shape) == (BATCH, SPATIAL, SPATIAL, ae.out_ch)
        assert dec.compute_output_shape((BATCH, lat, lat, ae.z_channels)) == (
            BATCH,
            SPATIAL,
            SPATIAL,
            ae.out_ch,
        )


# ---------------------------------------------------------------------
# AutoEncoder
# ---------------------------------------------------------------------


class TestAutoEncoder:
    def test_encode_shapes(self, tiny_ae_params, tiny_image):
        ae_model = AutoEncoder(params=tiny_ae_params)
        z_mean, z_log_var = ae_model.encode(
            keras.ops.convert_to_tensor(tiny_image)
        )
        lat = _latent_spatial(tiny_ae_params, SPATIAL)
        assert tuple(z_mean.shape) == (BATCH, lat, lat, tiny_ae_params.z_channels)
        assert tuple(z_log_var.shape) == (
            BATCH,
            lat,
            lat,
            tiny_ae_params.z_channels,
        )

    def test_decode_shape(self, tiny_ae_params):
        ae_model = AutoEncoder(params=tiny_ae_params)
        lat = _latent_spatial(tiny_ae_params, SPATIAL)
        z = keras.ops.convert_to_tensor(
            np.random.randn(
                BATCH, lat, lat, tiny_ae_params.z_channels
            ).astype("float32")
        )
        img = ae_model.decode(z)
        assert tuple(img.shape) == (BATCH, SPATIAL, SPATIAL, tiny_ae_params.out_ch)

    def test_call_reconstruction_shape(self, tiny_ae_params, tiny_image):
        ae_model = AutoEncoder(params=tiny_ae_params)
        recon = ae_model(keras.ops.convert_to_tensor(tiny_image))
        assert tuple(recon.shape) == (
            BATCH,
            SPATIAL,
            SPATIAL,
            tiny_ae_params.out_ch,
        )

    def test_decoder_only_inference_path(self, tiny_ae_params):
        # The pipeline only uses decode at inference: arbitrary latent decodes.
        ae_model = AutoEncoder(params=tiny_ae_params)
        lat = _latent_spatial(tiny_ae_params, SPATIAL)
        z = keras.ops.convert_to_tensor(
            np.random.randn(
                1, lat, lat, tiny_ae_params.z_channels
            ).astype("float32")
        )
        img = ae_model.decode(z)
        assert tuple(img.shape) == (1, SPATIAL, SPATIAL, tiny_ae_params.out_ch)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(img)))

    def test_factory(self, tiny_image):
        ae_model = create_ideogram4_autoencoder("tiny", sampling_seed=7)
        recon = ae_model(keras.ops.convert_to_tensor(tiny_image))
        assert tuple(recon.shape) == (BATCH, SPATIAL, SPATIAL, 3)
        assert ae_model.sampling_seed == 7


class TestGroupNormGuard:
    def test_bad_ch_fails_at_build(self):
        # ch=30 not /32 -> ResnetBlock ctor inside Encoder raises.
        bad = AutoEncoderParams(
            resolution=32,
            in_channels=3,
            ch=30,
            out_ch=3,
            ch_mult=(1, 2),
            num_res_blocks=1,
            z_channels=8,
        )
        with pytest.raises(ValueError):
            AutoEncoder(params=bad)


class TestGradients:
    def test_finite_grads_encode_sample_decode(self, tiny_ae_params, tiny_image):
        import tensorflow as tf

        ae_model = create_ideogram4_autoencoder("tiny", sampling_seed=0)
        x = keras.ops.convert_to_tensor(tiny_image)
        # Build weights.
        _ = ae_model(x)

        with tf.GradientTape() as tape:
            recon = ae_model(x, training=True)
            loss = keras.ops.mean(keras.ops.square(recon))
        grads = tape.gradient(loss, ae_model.trainable_variables)

        assert len(grads) > 0
        non_none = [g for g in grads if g is not None]
        assert len(non_none) > 0
        for g in non_none:
            arr = keras.ops.convert_to_numpy(g)
            assert np.all(np.isfinite(arr))


class TestSerialization:
    def test_keras_round_trip_deterministic_mu(
        self, tmp_path, tiny_ae_params, tiny_image
    ):
        ae_model = create_ideogram4_autoencoder("tiny", sampling_seed=123)
        x = keras.ops.convert_to_tensor(tiny_image)
        # Build all weights via the full __call__ so save() persists them
        # (subclassed Model: encode() alone does not mark the model built).
        _ = ae_model(x)
        # Deterministic encoder mu (NOT the stochastic sampled recon).
        mu_before, _ = ae_model.encode(x)
        mu_before = keras.ops.convert_to_numpy(mu_before)

        path = os.path.join(tmp_path, "ae.keras")
        ae_model.save(path)
        reloaded = keras.models.load_model(path)

        mu_after, _ = reloaded.encode(x)
        mu_after = keras.ops.convert_to_numpy(mu_after)

        np.testing.assert_allclose(mu_before, mu_after, atol=1e-5)

    def test_get_config_round_trip(self, tiny_ae_params):
        ae_model = AutoEncoder(params=tiny_ae_params, sampling_seed=5)
        cfg = ae_model.get_config()
        rebuilt = AutoEncoder.from_config(cfg)
        assert rebuilt.params.to_dict() == tiny_ae_params.to_dict()
        assert rebuilt.sampling_seed == 5
