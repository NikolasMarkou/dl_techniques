"""Smoke tests for ``train.ideogram4.train_ideogram4``.

Covers:
- Synthetic dataset element specs (shapes/dtypes of the six transformer inputs
  + the image-position velocity target).
- Velocity target identity ``v = x1 - x0`` at image positions (verified by
  reconstructing x0/x1 from a fixed-seed batch).
- End-to-end in-process ``fit`` on a few synthetic steps producing a finite
  (non-NaN) loss -- proves the training path wires up and runs.

Runtime target: a few seconds on GPU1. Uses the tiny preset with very small
dims (batch=2, grid 2x2, T=4, 2 steps).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import math  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from train.ideogram4.train_ideogram4 import (  # noqa: E402
    TrainingConfig,
    Ideogram4FlowTrainer,
    build_trainer,
    build_packed_indices,
    make_synthetic_dataset,
    sample_flow_batch,
)
from dl_techniques.models.ideogram4.constants import (  # noqa: E402
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)


def _tiny_config(**overrides) -> TrainingConfig:
    base = dict(
        variant="tiny",
        batch_size=2,
        steps_per_epoch=2,
        epochs=1,
        num_text_tokens=4,
        grid_h=2,
        grid_w=2,
        learning_rate=1e-3,
        seed=123,
    )
    base.update(overrides)
    return TrainingConfig(**base)


class TestPackedIndices:
    def test_layout(self):
        T, gh, gw = 4, 2, 2
        pos, seg, ind, num_image, seq_len = build_packed_indices(T, gh, gw)
        assert num_image == gh * gw == 4
        assert seq_len == T + num_image == 8
        assert pos.shape == (1, seq_len, 3)
        assert seg.shape == (1, seq_len)
        assert ind.shape == (1, seq_len)
        # Text block then image block.
        assert np.all(ind[0, :T] == LLM_TOKEN_INDICATOR)
        assert np.all(ind[0, T:] == OUTPUT_IMAGE_INDICATOR)
        # Single segment.
        assert np.all(seg == 1)


class TestSyntheticDataset:
    def test_element_specs(self):
        config = _tiny_config()
        _, in_channels, llm_dim = build_trainer(config)

        ds = make_synthetic_dataset(config, in_channels, llm_dim)
        inputs, target = next(iter(ds.take(1)))

        B = config.batch_size
        T = config.num_text_tokens
        N = config.grid_h * config.grid_w
        L = T + N

        assert inputs["x"].shape == (B, L, in_channels)
        assert inputs["llm_features"].shape == (B, L, llm_dim)
        assert inputs["t"].shape == (B,)
        assert inputs["position_ids"].shape == (B, L, 3)
        assert inputs["segment_ids"].shape == (B, L)
        assert inputs["indicator"].shape == (B, L)

        assert inputs["x"].dtype == tf.float32
        assert inputs["llm_features"].dtype == tf.float32
        assert inputs["position_ids"].dtype == tf.int32
        assert inputs["segment_ids"].dtype == tf.int32
        assert inputs["indicator"].dtype == tf.int32

        # Target is velocity at image positions only.
        assert target.shape == (B, N, in_channels)

        # Text positions of x are zeroed (only image tokens carry the noised
        # latent); image positions of llm_features are zeroed.
        x_np = inputs["x"].numpy()
        llm_np = inputs["llm_features"].numpy()
        assert np.allclose(x_np[:, :T, :], 0.0)
        assert np.allclose(llm_np[:, T:, :], 0.0)

    def test_velocity_target_is_x1_minus_x0(self):
        """The velocity target equals x1 - x0 and x_t is the rectified path.

        Verified against the canonical ``sample_flow_batch`` generator that the
        dataset reuses, so the A8 identity is checked at its source rather than
        via fragile graph-RNG replay.
        """
        B, L, C = 2, 8, 32
        x_t, v, tau, x0, x1 = sample_flow_batch(B, L, C)

        np.testing.assert_allclose(
            v.numpy(), (x1 - x0).numpy(), atol=1e-6, rtol=1e-6
        )
        expected_xt = ((1.0 - tau) * x0 + tau * x1).numpy()
        np.testing.assert_allclose(x_t.numpy(), expected_xt, atol=1e-6, rtol=1e-6)


class TestTrainerForwardAndFit:
    def test_forward_image_slice_shape(self):
        config = _tiny_config()
        trainer, in_channels, llm_dim = build_trainer(config)
        ds = make_synthetic_dataset(config, in_channels, llm_dim)
        inputs, _ = next(iter(ds.take(1)))
        out = trainer(inputs, training=False)
        B = config.batch_size
        N = config.grid_h * config.grid_w
        assert out.shape == (B, N, in_channels)
        # Velocity head is float32 even though it is sliced post-call.
        assert out.dtype == tf.float32

    def test_in_process_fit_finite_loss(self):
        """A 2-step fit produces a finite (non-NaN) loss -- the smoke gate."""
        config = _tiny_config(steps_per_epoch=2, epochs=1)
        trainer, in_channels, llm_dim = build_trainer(config)
        ds = make_synthetic_dataset(config, in_channels, llm_dim)

        trainer.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss=__import__(
                "dl_techniques.losses", fromlist=["FlowMatchingVelocityLoss"]
            ).FlowMatchingVelocityLoss(),
        )

        history = trainer.fit(
            ds,
            epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch,
            verbose=0,
        )

        losses = history.history["loss"]
        assert len(losses) == config.epochs
        for lv in losses:
            assert math.isfinite(lv), f"non-finite loss: {lv}"
            assert lv >= 0.0
