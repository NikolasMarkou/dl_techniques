"""Tests for the SD3FlowTrainer nested ``.keras`` save/load round-trip (step 12).

The trainer wraps :class:`SD3MMDiT` and serializes it via
``keras.saving.serialize_keras_object`` in ``get_config`` (and deserializes in
``from_config``). The step-11 report noted an unbuilt-save warning, so this test
RUNS ONE fit step first to force the model to build before saving. It then
reloads (registration should make ``custom_objects`` unnecessary) and asserts:

  (a) reload succeeds and is an :class:`SD3FlowTrainer`,
  (b) the reloaded trainer's wrapped ``transformer`` matches the original on the
      same dict input at ``atol=1e-5`` (the predict path),
  (c) ``get_config`` round-trips the nested transformer + scheduler params.

Uses the TINY preset throughout and a 1-step fit to stay fast.
"""

import os

import keras
import numpy as np
import pytest

from train.sd3_mmdit.train_sd3_mmdit import (
    TrainingConfig,
    SD3FlowTrainer,
    build_trainer,
    make_synthetic_dataset,
)
from dl_techniques.models.sd3_mmdit.config import get_sd3_config


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _tiny_config() -> TrainingConfig:
    """A minimal TINY smoke config: 1 epoch, 1 step, tiny batch."""
    return TrainingConfig(
        variant="tiny",
        batch_size=2,
        steps_per_epoch=1,
        epochs=1,
        learning_rate=1e-3,
        num_text_tokens=7,
        seed=123,
    )


def _build_and_fit_one_step(config: TrainingConfig):
    """Build the trainer and run exactly one fit step so it is BUILT.

    Returns ``(trainer, sd3_config)``.
    """
    trainer, sd3_config = build_trainer(config)
    trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate)
    )
    dataset = make_synthetic_dataset(config, sd3_config)
    trainer.fit(dataset, epochs=1, steps_per_epoch=1, verbose=0)

    # train_step calls self.transformer(...) directly, so the WRAPPER's own
    # call() path is never traced and trainer.built stays False (the step-11
    # "unbuilt-save" warning). Invoke call() once on a real dict input to flip
    # built=True so save() does not warn and the wrapper graph is fully traced.
    probe = _make_transformer_input(sd3_config, config, seed=0)
    _ = trainer(probe, training=False)
    return trainer, sd3_config


def _make_transformer_input(sd3_config, config: TrainingConfig, seed: int = 7):
    """Build a valid TINY transformer input dict (the predict path)."""
    rng = np.random.default_rng(seed)
    B = config.batch_size
    S = sd3_config.sample_size
    in_ch = sd3_config.in_channels
    L = config.num_text_tokens
    joint_dim = sd3_config.joint_attention_dim
    pooled_dim = sd3_config.pooled_projection_dim
    return {
        "latent": keras.ops.convert_to_tensor(
            rng.standard_normal((B, S, S, in_ch)).astype("float32")
        ),
        "encoder_hidden_states": keras.ops.convert_to_tensor(
            rng.standard_normal((B, L, joint_dim)).astype("float32")
        ),
        "pooled_projections": keras.ops.convert_to_tensor(
            rng.standard_normal((B, pooled_dim)).astype("float32")
        ),
        "timestep": keras.ops.convert_to_tensor(
            rng.uniform(0.0, 1000.0, size=(B,)).astype("float32")
        ),
    }


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------


class TestSD3FlowTrainerRoundTrip:
    def test_get_config_round_trips_nested_params(self):
        """get_config -> from_config preserves scheduler params + nested model."""
        config = _tiny_config()
        trainer, _sd3_config = build_trainer(config)

        cfg = trainer.get_config()
        # Scheduler scalars survive verbatim.
        assert cfg["shift"] == trainer.shift
        assert cfg["logit_mean"] == trainer.logit_mean
        assert cfg["logit_std"] == trainer.logit_std
        # The nested transformer is serialized as a dict (serialize_keras_object).
        assert isinstance(cfg["transformer"], dict)

        rebuilt = SD3FlowTrainer.from_config(cfg)
        assert isinstance(rebuilt, SD3FlowTrainer)
        assert rebuilt.shift == trainer.shift
        assert rebuilt.logit_mean == trainer.logit_mean
        assert rebuilt.logit_std == trainer.logit_std
        # Nested transformer config (embedding_size / depth) round-trips.
        assert (
            rebuilt.transformer.config.embedding_size
            == trainer.transformer.config.embedding_size
        )
        assert rebuilt.transformer.config.depth == trainer.transformer.config.depth

    def test_keras_save_load_round_trip(self, tmp_path):
        """Build via one fit step, save nested .keras, reload, match velocity.

        The step-11 unbuilt-save warning is avoided by running one fit step so
        every sub-layer is built before ``save``.
        """
        config = _tiny_config()
        trainer, sd3_config = _build_and_fit_one_step(config)

        # Deterministic predict-path velocity BEFORE save (use the transformer
        # directly so this is the inference path, not train_step).
        batch = _make_transformer_input(sd3_config, config, seed=7)
        out_before = keras.ops.convert_to_numpy(
            trainer.transformer(batch, training=False)
        )

        path = os.path.join(str(tmp_path), "trainer.keras")
        trainer.save(path)

        # Registration (@register_keras_serializable on SD3FlowTrainer + the
        # transformer/layers) should make custom_objects unnecessary.
        reloaded = keras.models.load_model(path)
        assert isinstance(reloaded, SD3FlowTrainer)

        out_after = keras.ops.convert_to_numpy(
            reloaded.transformer(batch, training=False)
        )

        # (b) reloaded transformer matches the original on the predict path.
        np.testing.assert_allclose(out_before, out_after, atol=1e-5)

        # (c) nested scheduler params survived the full .keras round-trip.
        assert reloaded.shift == trainer.shift
        assert reloaded.logit_mean == trainer.logit_mean
        assert reloaded.logit_std == trainer.logit_std
        assert (
            reloaded.transformer.config.embedding_size
            == trainer.transformer.config.embedding_size
        )
        assert reloaded.transformer.config.depth == trainer.transformer.config.depth
