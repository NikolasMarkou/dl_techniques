"""RED proof for C-20: ``VideoJEPA.stream_step`` must emit the quantity the
training loss supervises.

The training forward (``model.py``, L1) supervises
``pred_heads[h_idx](predictor(z)[:, :-h])`` against ``z_target[:, h:]``. Before
this suite, ``stream_step`` returned ``predictor(buf)[:, -1]`` raw — a tensor
that receives no z-space supervision at all when ``mask_prediction_enabled`` is
off, and only a *same-timestep* reconstruction signal when it is on. These
tests pin the composition, not the shape: the existing
``test_stream_step_shape`` / ``test_stream_step_timing`` pair passes either way.
"""

from __future__ import annotations

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.video_jepa.model import VideoJEPA


def _cfg(**overrides) -> VideoJEPAConfig:
    defaults = dict(
        img_size=32, img_channels=3, patch_size=8, embed_dim=32,
        num_frames=4, history_size_k=4,
        encoder_clifford_depth=1, encoder_shifts=(1, 2),
        predictor_depth=1, predictor_num_heads=2, predictor_dim_head=16,
        predictor_mlp_dim=64, predictor_shifts=(1, 2),
        sigreg_knots=17, sigreg_num_proj=8, sigreg_weight=0.09,
        dropout_rate=0.0,
    )
    defaults.update(overrides)
    return VideoJEPAConfig(**defaults)


def _built_model(cfg: VideoJEPAConfig) -> VideoJEPA:
    model = VideoJEPA(config=cfg)
    pixels = np.random.rand(
        2, cfg.num_frames, cfg.img_size, cfg.img_size, cfg.img_channels,
    ).astype("float32")
    _ = model({"pixels": pixels}, training=False)
    return model


def _stream(model: VideoJEPA, cfg: VideoJEPAConfig, steps: int, **kwargs):
    model.stream_reset(B=2)
    out = None
    rng = np.random.default_rng(0)
    for _ in range(steps):
        frame = rng.random(
            (2, cfg.img_size, cfg.img_size, cfg.img_channels)
        ).astype("float32")
        out = model.stream_step(frame, **kwargs)
    return out


class TestStreamStepRoutesThroughTheTrainedHead:
    def test_head_is_not_an_identity_at_init(self) -> None:
        """Anti-vacuity: the composition assertion below is only meaningful
        if ``pred_heads[0]`` actually moves its input."""
        cfg = _cfg()
        model = _built_model(cfg)
        x = np.random.rand(2, 5, cfg.embed_dim).astype("float32")
        y = np.asarray(model.pred_heads[0](x))
        delta = float(np.max(np.abs(y - x)))
        assert delta > 1e-2, (
            f"pred_head_h1 is (near-)identity at init (max|Wx - x| = {delta}); "
            "the routing assertion would be satisfied by construction."
        )

    def test_stream_output_equals_the_training_path_forecast(self) -> None:
        """``stream_step`` == ``pred_heads[h_idx](predictor(buf))[:, -1]``."""
        cfg = _cfg()
        model = _built_model(cfg)
        out = np.asarray(_stream(model, cfg, cfg.history_size_k + 2))

        buf = model._stream_buf
        raw = model.predictor(buf, training=False)
        expected = np.asarray(model.pred_heads[0](raw[:, -1]))
        np.testing.assert_allclose(out, expected, atol=1e-6, rtol=1e-6)

        raw_last = np.asarray(raw[:, -1])
        assert float(np.max(np.abs(out - raw_last))) > 1e-3, (
            "stream_step returned the raw predictor output — the pre-fix "
            "behaviour (C-20)."
        )

    def test_multi_horizon_selects_the_matching_head(self) -> None:
        cfg = _cfg(predict_horizons=(1, 2))
        model = _built_model(cfg)

        out_default = np.asarray(_stream(model, cfg, cfg.history_size_k + 1))
        buf = model._stream_buf
        raw_last = model.predictor(buf, training=False)[:, -1]

        # Default resolves to min(predict_horizons) == 1 -> head index 0.
        np.testing.assert_allclose(
            out_default,
            np.asarray(model.pred_heads[0](raw_last)),
            atol=1e-6, rtol=1e-6,
        )

        out_h2 = np.asarray(model.stream_step(
            np.random.rand(
                2, cfg.img_size, cfg.img_size, cfg.img_channels,
            ).astype("float32"),
            horizon=2,
        ))
        buf2 = model._stream_buf
        raw_last2 = model.predictor(buf2, training=False)[:, -1]
        np.testing.assert_allclose(
            out_h2,
            np.asarray(model.pred_heads[1](raw_last2)),
            atol=1e-6, rtol=1e-6,
        )
        # The two heads are independently initialized, so h=1 and h=2 must not
        # coincide on the same buffer.
        assert float(np.max(np.abs(
            np.asarray(model.pred_heads[0](raw_last2)) - out_h2
        ))) > 1e-3

    def test_unconfigured_horizon_raises_by_name(self) -> None:
        cfg = _cfg(predict_horizons=(1,))
        model = _built_model(cfg)
        model.stream_reset(B=2)
        frame = np.random.rand(
            2, cfg.img_size, cfg.img_size, cfg.img_channels,
        ).astype("float32")
        with pytest.raises(ValueError, match="no trained prediction head"):
            model.stream_step(frame, horizon=3)
