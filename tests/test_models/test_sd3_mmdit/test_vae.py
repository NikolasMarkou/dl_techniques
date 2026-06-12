"""Unit tests for the SD3 16-channel KL-VAE reuse wrapper.

Verifies that the ideogram4 ``AutoEncoder`` reused with ``z_channels=16``:

1. builds via ``create_sd3_vae("tiny")``;
2. encodes a ``(B, 32, 32, 3)`` image to ``(z_mean, z_log_var)`` with 16 latent
   channels and a 2x spatial downsample (-> ``(B, 16, 16, 16)``);
3. samples + decodes back to the input spatial dims ``(B, 32, 32, 3)``;
4. has inverse SD3 scalar latent-norm helpers applying ``(z - 0.0) * 0.13025``;
5. produces a DETERMINISTIC ``z_mean`` that survives a ``.keras`` save/load of
   the AutoEncoder (mu is compared because ``sample`` is stochastic).
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.ideogram4.vae import AutoEncoder
from dl_techniques.models.sd3_mmdit.vae import (
    SD3_SCALING_FACTOR,
    SD3_SHIFT_FACTOR,
    create_sd3_vae,
    normalize_latent,
    denormalize_latent,
    SD3VAE,
)

# Tiny preset: ch=32, ch_mult=(1, 2) -> 2x downsample; z_channels=16.
_IMG_HW = 32
_LATENT_HW = 16
_Z_CHANNELS = 16


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, _IMG_HW, _IMG_HW, 3)).astype(np.float32)


# ---------------------------------------------------------------------
# (1) construction
# ---------------------------------------------------------------------


def test_create_sd3_vae_builds_autoencoder():
    vae = create_sd3_vae("tiny")
    assert isinstance(vae, AutoEncoder)
    assert vae.z_channels == _Z_CHANNELS
    assert vae.params.z_channels == _Z_CHANNELS


def test_create_sd3_vae_explicit_params_overrides_variant():
    from dl_techniques.models.sd3_mmdit.config import get_sd3_config

    _, params = get_sd3_config("tiny")
    vae = create_sd3_vae(params=params)
    assert isinstance(vae, AutoEncoder)
    assert vae.z_channels == _Z_CHANNELS


# ---------------------------------------------------------------------
# (2) encode -> 16-channel latent with 2x spatial downsample
# ---------------------------------------------------------------------


def test_encode_latent_shape_and_channels(image):
    vae = create_sd3_vae("tiny")
    z_mean, z_log_var = vae.encode(image)
    z_mean = keras.ops.convert_to_numpy(z_mean)
    z_log_var = keras.ops.convert_to_numpy(z_log_var)

    assert z_mean.shape == (2, _LATENT_HW, _LATENT_HW, _Z_CHANNELS)
    assert z_log_var.shape == (2, _LATENT_HW, _LATENT_HW, _Z_CHANNELS)
    # Assumption-1 verification: z_channels == 16 latent channel dim.
    assert z_mean.shape[-1] == _Z_CHANNELS


# ---------------------------------------------------------------------
# (3) sample -> decode -> reconstruction at input spatial dims
# ---------------------------------------------------------------------


def test_sample_decode_reconstruction_shape(image):
    vae = create_sd3_vae("tiny", sampling_seed=0)
    z_mean, z_log_var = vae.encode(image)
    z = vae.sample(z_mean, z_log_var)
    z_np = keras.ops.convert_to_numpy(z)
    assert z_np.shape == (2, _LATENT_HW, _LATENT_HW, _Z_CHANNELS)

    recon = vae.decode(z)
    recon = keras.ops.convert_to_numpy(recon)
    # Decoder upsamples 2x back to the input spatial dims; out_ch = 3.
    assert recon.shape == (2, _IMG_HW, _IMG_HW, 3)


# ---------------------------------------------------------------------
# (4) SD3 scalar latent-norm: inverse + known-value
# ---------------------------------------------------------------------


def test_normalize_denormalize_are_inverses():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((2, _LATENT_HW, _LATENT_HW, _Z_CHANNELS)).astype(
        np.float32
    )
    z_norm = normalize_latent(z)
    z_back = denormalize_latent(z_norm)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(z_back), z, atol=1e-5
    )


def test_normalize_applies_sd3_scalar():
    assert SD3_SHIFT_FACTOR == 0.0
    assert SD3_SCALING_FACTOR == pytest.approx(0.13025)

    rng = np.random.default_rng(2)
    z = rng.standard_normal((1, 4, 4, _Z_CHANNELS)).astype(np.float32)
    z_norm = keras.ops.convert_to_numpy(normalize_latent(z))
    expected = (z - SD3_SHIFT_FACTOR) * SD3_SCALING_FACTOR
    np.testing.assert_allclose(z_norm, expected, atol=1e-6)


def test_sd3vae_wrapper_roundtrip_latent_norm(image):
    wrapper = SD3VAE.from_variant("tiny", sampling_seed=0)
    assert isinstance(wrapper.autoencoder, AutoEncoder)
    z_norm = wrapper.encode_to_latent(image)
    z_norm_np = keras.ops.convert_to_numpy(z_norm)
    assert z_norm_np.shape == (2, _LATENT_HW, _LATENT_HW, _Z_CHANNELS)

    recon = wrapper.decode_from_latent(z_norm)
    assert keras.ops.convert_to_numpy(recon).shape == (2, _IMG_HW, _IMG_HW, 3)


# ---------------------------------------------------------------------
# (5) deterministic z_mean survives a .keras save/load
# ---------------------------------------------------------------------


def test_encode_mu_survives_keras_roundtrip(image):
    """DETERMINISTIC z_mean survives a full ``.keras`` save/load.

    ``encode``/``sample``/``decode`` build sub-layers lazily, but the top-level
    ``AutoEncoder`` (a ``keras.Model`` whose ``call`` is encode->sample->decode)
    is only marked built once ``call`` has run. We run a full forward
    (``vae(image)``) before saving so the model's weights are materialized and
    persisted -- otherwise Keras saves an unbuilt model and the reload
    re-initializes random weights (the SD3-side verification of assumption 1's
    serialization path).
    """
    vae = create_sd3_vae("tiny", sampling_seed=0)
    # Full forward marks the top-level Model built so weights are saved.
    _ = vae(image)
    z_mean_before, _ = vae.encode(image)
    z_mean_before = keras.ops.convert_to_numpy(z_mean_before)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sd3_vae.keras")
        vae.save(path)
        reloaded = keras.models.load_model(path)

    z_mean_after, _ = reloaded.encode(image)
    z_mean_after = keras.ops.convert_to_numpy(z_mean_after)

    assert z_mean_after.shape == z_mean_before.shape
    np.testing.assert_allclose(z_mean_after, z_mean_before, atol=1e-5)


def test_encode_mu_is_deterministic_across_calls(image):
    """z_mean is deterministic (Sampling is stochastic; encode/mu is not)."""
    vae = create_sd3_vae("tiny", sampling_seed=0)
    mu_a, _ = vae.encode(image)
    mu_b, _ = vae.encode(image)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(mu_a),
        keras.ops.convert_to_numpy(mu_b),
        atol=1e-6,
    )
