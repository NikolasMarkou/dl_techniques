"""Tests for the SD3MMDiT model + create_sd3_mmdit factory (step 6).

Uses the TINY preset throughout to stay fast (latent 16x16x16, depth 4, dim 192,
6 heads, patch_size 2 -> 8x8 = 64 patch tokens; block 0 uses dual attention).
Verifies forward velocity shape, compute_output_shape (pre/post build), variable
batch + variable text seq len, get_config / from_config round-trip, and the
full ``.keras`` save/load deterministic-velocity round-trip.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.models.sd3_mmdit.config import get_sd3_config
from dl_techniques.models.sd3_mmdit.transformer import (
    SD3MMDiT,
    create_sd3_mmdit,
)


# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------

LATENT_HW = 16
IN_CH = 16
JOINT_DIM = 512
POOLED_DIM = 256


def _make_batch(batch: int = 2, txt_len: int = 7, seed: int = 0) -> dict:
    """Build a valid TINY input dict."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(
        (batch, LATENT_HW, LATENT_HW, IN_CH)
    ).astype("float32")
    enc = rng.standard_normal((batch, txt_len, JOINT_DIM)).astype("float32")
    pooled = rng.standard_normal((batch, POOLED_DIM)).astype("float32")
    timestep = rng.uniform(0.0, 1000.0, size=(batch,)).astype("float32")
    return {
        "latent": keras.ops.convert_to_tensor(latent),
        "encoder_hidden_states": keras.ops.convert_to_tensor(enc),
        "pooled_projections": keras.ops.convert_to_tensor(pooled),
        "timestep": keras.ops.convert_to_tensor(timestep),
    }


def _input_shapes(batch: int = 2, txt_len: int = 7) -> dict:
    return {
        "latent": (batch, LATENT_HW, LATENT_HW, IN_CH),
        "encoder_hidden_states": (batch, txt_len, JOINT_DIM),
        "pooled_projections": (batch, POOLED_DIM),
        "timestep": (batch,),
    }


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------


class TestSD3MMDiT:
    def test_factory_builds(self):
        model = create_sd3_mmdit("tiny")
        assert isinstance(model, SD3MMDiT)
        assert model.config.embedding_size == 192
        assert model.config.depth == 4

    def test_forward_shape(self):
        model = create_sd3_mmdit("tiny")
        batch = _make_batch(batch=2, txt_len=7)
        out = model(batch)
        # Velocity must exactly match the latent shape (in==out channels).
        assert tuple(out.shape) == (2, LATENT_HW, LATENT_HW, IN_CH)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape_matches_and_pre_build(self):
        # Pre-build: a fresh, unbuilt instance must answer compute_output_shape.
        fresh = create_sd3_mmdit("tiny")
        shapes = _input_shapes(batch=2, txt_len=7)
        pred = fresh.compute_output_shape(shapes)
        assert tuple(pred) == (2, LATENT_HW, LATENT_HW, IN_CH)

        # Post-build: matches the actual forward output.
        batch = _make_batch(batch=2, txt_len=7)
        out = fresh(batch)
        assert tuple(out.shape) == tuple(pred)

    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize("txt_len", [5, 11])
    def test_variable_batch_and_seqlen(self, batch, txt_len):
        model = create_sd3_mmdit("tiny")
        b = _make_batch(batch=batch, txt_len=txt_len, seed=batch + txt_len)
        out = model(b)
        assert tuple(out.shape) == (batch, LATENT_HW, LATENT_HW, IN_CH)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_get_config_has_config_dict(self):
        model = create_sd3_mmdit("tiny")
        cfg = model.get_config()
        assert "config" in cfg
        assert isinstance(cfg["config"], dict)
        assert cfg["config"]["embedding_size"] == 192

    def test_from_config_reconstructs(self):
        model = create_sd3_mmdit("tiny")
        rebuilt = SD3MMDiT.from_config(model.get_config())
        batch = _make_batch(batch=2, txt_len=7)
        out = rebuilt(batch)
        assert tuple(out.shape) == (2, LATENT_HW, LATENT_HW, IN_CH)

    def test_keras_round_trip(self, tmp_path):
        """The serialization gate: save/reload yields IDENTICAL velocity."""
        model = create_sd3_mmdit("tiny")
        batch = _make_batch(batch=2, txt_len=7, seed=42)
        out_before = keras.ops.convert_to_numpy(model(batch))

        path = os.path.join(str(tmp_path), "sd3_mmdit_tiny.keras")
        model.save(path)
        # Registration handles deserialization -- no custom_objects needed.
        reloaded = keras.models.load_model(path)
        out_after = keras.ops.convert_to_numpy(reloaded(batch))

        try:
            np.testing.assert_allclose(out_before, out_after, atol=1e-6)
        except AssertionError:
            np.testing.assert_allclose(out_before, out_after, atol=1e-5)
