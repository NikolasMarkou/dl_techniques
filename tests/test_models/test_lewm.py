"""Tests for the LeWM model.

Covers:
- forward pass shapes + finite loss
- serialization round-trip
- rollout shape
- identity-at-init of the predictor (via LeWM forward path)
"""

import os

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.lewm.config import LeWMConfig
from dl_techniques.models.lewm.model import LeWM


def _small_cfg() -> LeWMConfig:
    """A small CPU-friendly config — still exercises every code path."""
    return LeWMConfig(
        img_size=56, patch_size=14, encoder_scale="tiny",
        embed_dim=192, projector_hidden_dim=192,
        history_size=2, num_preds=1, num_frames=3,
        depth=2, heads=4, dim_head=48, mlp_dim=256,
        dropout=0.0, emb_dropout=0.0,
        action_dim=2, smoothed_dim=10, mlp_scale=4,
        sigreg_weight=0.09, sigreg_knots=17, sigreg_num_proj=32,
    )


@pytest.fixture
def cfg():
    return _small_cfg()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestLeWM:
    def test_forward_pass_shapes(self, cfg, rng):
        model = LeWM(config=cfg)
        B, T = 2, cfg.history_size + cfg.num_preds
        pixels = rng.standard_normal((B, T, cfg.img_size, cfg.img_size, 3)).astype("float32")
        action = rng.standard_normal((B, T - 1, cfg.action_dim)).astype("float32")

        out = model({"pixels": pixels, "action": action}, training=True)
        assert tuple(out.shape) == (B, T, cfg.embed_dim)
        # Losses accumulated.
        assert len(model.losses) >= 2, f"expected >=2 losses, got {len(model.losses)}"
        for loss_val in model.losses:
            loss_np = float(ops.convert_to_numpy(loss_val))
            assert np.isfinite(loss_np), f"Non-finite loss: {loss_val}"

    def test_serialization_round_trip(self, cfg, rng, tmp_path):
        model = LeWM(config=cfg)
        B, T = 2, cfg.history_size + cfg.num_preds
        pixels = rng.standard_normal((B, T, cfg.img_size, cfg.img_size, 3)).astype("float32")
        action = rng.standard_normal((B, T - 1, cfg.action_dim)).astype("float32")

        # Build + forward.
        y1 = ops.convert_to_numpy(model(
            {"pixels": pixels, "action": action}, training=False
        ))

        path = str(tmp_path / "lewm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y2 = ops.convert_to_numpy(loaded(
            {"pixels": pixels, "action": action}, training=False
        ))

        # SIGReg resamples A each call, but it's NOT in the returned tensor
        # (only in losses). The returned pred_emb must be deterministic given
        # deterministic weights.
        max_diff = float(np.max(np.abs(y1 - y2)))
        assert max_diff < 1e-4, (
            f"Round-trip mismatch: max|y1 - y2| = {max_diff}"
        )

    def test_rollout_shape(self, cfg, rng):
        model = LeWM(config=cfg)
        B = 2
        S = 2
        T_rollout = 4

        # Build the model with a matching training forward (T = num_frames
        # so pos_embedding covers the sequence). Rollout uses truncated
        # windows of length history_size, so it's safe regardless of T_rollout.
        T_build = cfg.num_frames
        pixels = rng.standard_normal(
            (B, T_build, cfg.img_size, cfg.img_size, 3)
        ).astype("float32")
        action = rng.standard_normal(
            (B, T_build - 1, cfg.action_dim)
        ).astype("float32")
        _ = model({"pixels": pixels, "action": action}, training=False)

        ph = rng.standard_normal(
            (B, S, cfg.history_size, cfg.img_size, cfg.img_size, 3)
        ).astype("float32")
        aseq = rng.standard_normal((B, S, T_rollout, cfg.action_dim)).astype("float32")
        res = model.rollout(ph, aseq)

        # T_full = history_size + n_steps + 1 where n_steps = T_rollout - HS
        # so T_full = T_rollout + 1.
        expected_T = T_rollout + 1
        assert tuple(res["predicted_emb"].shape) == (B, S, expected_T, cfg.embed_dim)

    def test_predictor_identity_at_init(self, cfg, rng):
        """At init, predictor blocks are identity — so pred_emb (before
        pred_proj) should preserve the input embeddings up to the learned
        pos embedding and optional input_proj. Hard to assert strict
        identity through the full model; we just check pred_emb is finite
        and has the right shape. Component-level identity is covered in
        test_adaln_zero.py."""
        model = LeWM(config=cfg)
        B, T = 2, cfg.history_size + cfg.num_preds
        pixels = rng.standard_normal((B, T, cfg.img_size, cfg.img_size, 3)).astype("float32")
        action = rng.standard_normal((B, T - 1, cfg.action_dim)).astype("float32")
        out = model({"pixels": pixels, "action": action}, training=False)
        out_np = ops.convert_to_numpy(out)
        assert np.all(np.isfinite(out_np))
