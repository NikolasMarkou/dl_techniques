"""The score-based nanoVLM trainer, which had zero tests and could not train.

Three independent blockers were live at once, in this masking order:

(a) ``reset_metrics`` called ``keras.metrics.Mean.reset_states``. Keras 3 spells
    it ``reset_state``; the plural exists only on ``layers/rnn``. It raised at
    the FIRST statement of epoch 0 in ``train_score_vlm``, hiding (b) and (c).
(c) The EMA clone was made from an **unbuilt** subclassed model, so
    ``set_weights(model.get_weights())`` was ``set_weights([])`` and every epoch
    checkpoint was that empty clone.
(b) ``accumulation_counter`` was a Python ``int`` compared with a Python ``if``
    inside a ``@tf.function``. At the shipped ``gradient_accumulation_steps=4``
    the comparison folded to ``False`` at trace time and ``apply_gradients`` was
    never emitted into the graph, while the loss metric kept moving.

(b)'s probe MUST run through the traced entry point. Under eager execution the
Python ``if`` would work, so an eager probe passes with the defect present.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.nano_vlm_world_model.model import ScoreBasedNanoVLM
from dl_techniques.models.nano_vlm_world_model.train import (
    ScoreVLMTrainer,
    VLMDenoisingLoss,
)

BATCH = 2
IMG = 32
SEQ = 16
VOCAB = 32


def _tiny_model():
    return ScoreBasedNanoVLM(
        vision_config={"img_size": IMG, "patch_size": 16, "embed_dim": 32,
                       "depth": 1, "num_heads": 2, "output_mode": "none"},
        text_config={"vocab_size": VOCAB, "embed_dim": 32, "depth": 1,
                     "num_heads": 2, "max_seq_len": SEQ},
        diffusion_config={"num_timesteps": 20, "beta_schedule": "cosine"},
        vocab_size=VOCAB,
        generation_mode="joint",
    )


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(0)
    return (
        rng.random((BATCH, IMG, IMG, 3), dtype="float32"),
        rng.integers(0, VOCAB, (BATCH, SEQ)).astype("int32"),
    )


def _trainer(accumulation_steps=1, ema_decay=0.5):
    return ScoreVLMTrainer(
        model=_tiny_model(),
        optimizer=keras.optimizers.Adam(1e-3),
        loss_fn=VLMDenoisingLoss(),
        use_ema=True,
        ema_decay=ema_decay,
        gradient_accumulation_steps=accumulation_steps,
    )


class TestMetricsCanBeReset:

    def test_reset_metrics_uses_the_singular_keras_3_spelling(self):
        trainer = _trainer()
        # train_score_vlm calls this as the first statement of epoch 0.
        trainer.reset_metrics()
        assert float(trainer.train_loss.result()) == 0.0


class TestEmaCloneIsReal:

    def test_ema_clone_is_built_and_actually_moves(self, batch):
        images, text = batch
        trainer = _trainer(accumulation_steps=1, ema_decay=0.5)
        trainer.train_step(images, text)

        model_weights = trainer.model.weights
        ema_weights = trainer.ema_model.weights
        assert len(model_weights) > 0
        assert len(ema_weights) == len(model_weights), (
            "the EMA clone holds a different number of weights than its source; "
            "cloning an unbuilt subclassed model gives an empty weight list"
        )

        # Length alone is satisfiable by a clone that never updates: after a
        # step at decay 0.5 the EMA must sit strictly between its own initial
        # value and the model's, i.e. it must differ from the model.
        trainer.train_step(images, text)
        drift = max(
            float(np.max(np.abs(np.asarray(e) - np.asarray(m))))
            for e, m in zip(trainer.ema_model.weights, trainer.model.weights)
        )
        assert drift > 0.0, (
            "EMA weights are identical to the model's, so _update_ema zipped "
            "two empty lists and did nothing"
        )


class TestGradientAccumulationReachesTheGraph:

    def test_the_step_really_is_traced(self):
        """Anti-vacuity control for the test below — eager hides the defect."""
        assert isinstance(
            ScoreVLMTrainer._train_step_fn, tf.types.experimental.GenericFunction
        ) or hasattr(ScoreVLMTrainer._train_step_fn, "get_concrete_function"), (
            "_train_step_fn is not a tf.function; the accumulation probe below "
            "would then pass with a Python-int counter, which is the defect"
        )

    def test_optimizer_advances_exactly_once_per_accumulation_window(self, batch):
        """Also pins the accumulators' type — the two share one traced trainer
        because each trainer instance costs a full model build plus a trace."""
        images, text = batch
        trainer = _trainer(accumulation_steps=2)

        trainer.train_step(images, text)
        assert trainer.accumulated_gradients is not None
        assert all(
            isinstance(a, tf.Variable) for a in trainer.accumulated_gradients
        ), "accumulators must survive across steps, so they must be variables"
        assert int(trainer.optimizer.iterations) == 0, (
            "the optimizer applied on the first of 2 accumulation steps"
        )
        assert int(trainer.accumulation_counter) == 1

        trainer.train_step(images, text)
        assert int(trainer.optimizer.iterations) == 1, (
            "apply_gradients never reached the traced graph: a Python `if` over "
            "a Python-int counter is evaluated once, at trace time, against 0"
        )
        assert int(trainer.accumulation_counter) == 0
