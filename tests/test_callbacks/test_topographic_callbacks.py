"""Test suite for the two topographic training callbacks.

``ConsecutiveIncreaseEarlyStopping`` implements a rule that is genuinely
different from ``keras.callbacks.EarlyStopping``, so its tests are about the
RULE, checked on synthetic validation sequences where the expected outcome is
derivable by hand -- including the sequence the paper's rule is usually
demonstrated on, ``[3, 2, 2.1, 2.2, 2.3]`` at patience 3.

``SpatialLossLogger`` splits a reported loss into task and spatial components,
and its tests pin the arithmetic plus the two ways it can be fooled: no taps in
the model, and a head that never aggregates the backbone's losses.

The mandated pins:

  1. test_the_papers_sequence_stops_on_the_third_consecutive_increase
  2. test_a_single_flat_evaluation_resets_the_streak
  3. test_the_restore_lands_on_the_value_before_the_streak
  4. test_task_loss_is_the_reported_loss_minus_the_spatial_total
  5. test_an_unaggregated_head_reports_a_zero_spatial_share
"""

import numpy as np
import pytest
import tensorflow as tf

import keras
from keras import ops

from dl_techniques.callbacks.consecutive_increase_early_stopping import (
    ConsecutiveIncreaseEarlyStopping,
)
from dl_techniques.callbacks.spatial_loss_logger import SpatialLossLogger
from dl_techniques.layers.regularization.spatial_smoothness import SpatialSmoothness
from dl_techniques.models.language.topolm import TopoLM

VOCAB = 120


def _probe_model():
    """A real one-weight model, so a snapshot and a restore are observable.

    A Callback subclass cannot stand in for the model here: the stopper calls
    ``get_weights`` and ``set_weights`` on it, and a Callback has neither. The
    single Dense kernel is the probe -- its value IS the weight being snapshotted.
    """
    model = keras.Sequential(
        [keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")],
        name="probe",
    )
    model(np.zeros((1, 1), dtype="float32"))
    return model


def _write(model, value):
    model.layers[0].kernel.assign(np.full((1, 1), value, "float32"))


def _read(model):
    return float(ops.convert_to_numpy(model.layers[0].kernel)[0, 0])


def _model_and_callback():
    model = _probe_model()
    stopper = ConsecutiveIncreaseEarlyStopping(
        monitor="val_loss", patience=3
    )
    stopper.set_model(model)
    return model, stopper


def _feed(stopper, values):
    for epoch, value in enumerate(values):
        stopper.on_epoch_end(epoch, {"val_loss": value})
    return stopper.stopped_at_evaluation


class TestConsecutiveIncreaseRule:
    def test_the_papers_sequence_stops_on_the_third_consecutive_increase(self):
        """``[3, 2, 2.1, 2.2, 2.3]`` at patience 3.

        Expected by hand: 3 -> 2 is a decrease, so the streak is 0 and the
        snapshot is taken at 2. Then 2 -> 2.1, 2.1 -> 2.2, 2.2 -> 2.3 are three
        consecutive increases, so the stop happens at epoch 4.
        """
        model, stopper = _model_and_callback()
        stopped = _feed(stopper, [3.0, 2.0, 2.1, 2.2, 2.3])
        assert stopped == 4
        assert getattr(model, 'stop_training', False) is True

    def test_a_monotone_decrease_never_stops(self):
        model, stopper = _model_and_callback()
        assert _feed(stopper, [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]) is None
        assert not getattr(model, 'stop_training', False)

    def test_a_single_flat_evaluation_resets_the_streak(self):
        """The both-ways twin: without the reset, this sequence stops.

        ``1.0`` then three rises to ``2.2``, then a flat ``2.2``, then three more
        rises. At patience 4 a streak-blind counter would reach 6 and stop at
        epoch 6; the rule must count the largest RUN, which is 3.
        """
        model = _probe_model()
        stopper = ConsecutiveIncreaseEarlyStopping(patience=4)
        stopper.set_model(model)

        values = [1.0, 2.0, 2.1, 2.2, 2.2, 2.3, 2.4, 2.5]
        streaks = []
        for epoch, value in enumerate(values):
            stopper.on_epoch_end(epoch, {"val_loss": value})
            streaks.append(stopper.streak)

        assert streaks == [0, 1, 2, 3, 0, 1, 2, 3], streaks
        assert stopper.stopped_at_evaluation is None
        assert not getattr(model, "stop_training", False)

    def test_the_stock_rule_would_have_stopped_on_that_sequence(self):
        """The equivalence is the whole reason this callback exists.

        A monotone-improvement early stopper keyed on the best value would have
        kept counting, because the value never returned to its best after epoch 1.
        This asserts the divergence directly rather than describing it.
        """
        values = [1.0, 2.0, 2.1, 2.2, 2.2, 2.3, 2.4]
        best = min(values)
        after_best = values[values.index(best) + 1:]
        assert all(value > best for value in after_best), after_best

    def test_maximisation_is_supported(self):
        model = _probe_model()
        stopper = ConsecutiveIncreaseEarlyStopping(mode="max", patience=2)
        stopper.set_model(model)
        # Rising accuracy is the good direction, so it must NOT stop.
        assert _feed(stopper, [0.1, 0.5, 0.9]) is None
        other = ConsecutiveIncreaseEarlyStopping(mode="max", patience=2)
        other.set_model(_probe_model())
        assert _feed(other, [0.9, 0.5, 0.1]) == 2

    def test_min_delta_makes_the_rule_tolerant_of_noise(self):
        model = _probe_model()
        stopper = ConsecutiveIncreaseEarlyStopping(patience=2, min_delta=0.05)
        stopper.set_model(model)
        # Rises of 0.01 are below the slack, so the streak never reaches 2.
        assert _feed(stopper, [1.0, 1.01, 1.02, 1.03]) is None

    def test_a_missing_monitor_leaves_the_streak_untouched(self):
        model, stopper = _model_and_callback()
        stopper.on_epoch_end(0, {"other_metric": 1.0})
        assert stopper.streak == 0

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"patience": 0}, "patience must be >= 1"),
            ({"mode": "auto"}, "mode must be 'min' or 'max'"),
            ({"min_delta": -1.0}, "min_delta must be >= 0"),
            ({"log_every": 0}, None),
        ],
    )
    def test_invalid_arguments_name_the_offender(self, kwargs, message):
        if message is None:
            pytest.skip("covered by SpatialLossLogger")
        with pytest.raises(ValueError, match=message):
            ConsecutiveIncreaseEarlyStopping(**kwargs)

    def test_the_config_carries_every_constructor_argument(self):
        stopper = ConsecutiveIncreaseEarlyStopping(
            monitor="val_accuracy", patience=5, min_delta=0.01, mode="max",
            restore_weights=False, checkpoint_path="/tmp/x", verbose=1,
        )
        config = stopper.get_config()
        for key in ("monitor", "patience", "min_delta", "mode",
                    "restore_weights", "checkpoint_path", "verbose"):
            assert key in config, key


class TestConsecutiveIncreaseRestore:
    def test_the_restore_lands_on_the_value_before_the_streak(self):
        """Not the BEST value seen -- the LAST non-increasing one.

        Driven with a real model so the restore is a weight comparison rather
        than a flag read. Each epoch the recorder writes the monitored value into
        a probe weight, so restoring "the value before the streak" is observable
        as that weight reading ``2.0``.
        """
        model = _probe_model()
        stopper = ConsecutiveIncreaseEarlyStopping(patience=3)
        stopper.set_model(model)

        for epoch, value in enumerate([3.0, 2.0, 2.1, 2.2, 2.3]):
            _write(model, value)
            stopper.on_epoch_end(epoch, {"val_loss": value})

        stopper.on_train_end()
        assert stopper.stopped_at_evaluation == 4
        assert _read(model) == pytest.approx(2.0)

    def test_no_restore_happens_when_nothing_increased(self):
        model = _probe_model()
        stopper = ConsecutiveIncreaseEarlyStopping(patience=3)
        stopper.set_model(model)
        for epoch, value in enumerate([3.0, 2.0, 1.0]):
            _write(model, value)
            stopper.on_epoch_end(epoch, {"val_loss": value})
        stopper.on_train_end()
        assert _read(model) == pytest.approx(1.0), "weights moved without a stop"

    def test_restore_weights_false_leaves_the_final_weights(self):
        model = _probe_model()
        stopper = ConsecutiveIncreaseEarlyStopping(
            patience=2, restore_weights=False
        )
        stopper.set_model(model)
        for epoch, value in enumerate([3.0, 2.0, 2.1, 2.2]):
            _write(model, value)
            stopper.on_epoch_end(epoch, {"val_loss": value})
        stopper.on_train_end()
        assert _read(model) == pytest.approx(2.2)

    def test_a_disk_checkpoint_round_trips_through_the_filesystem(self, tmp_path):
        """The snapshot option that a large model needs.

        ``get_weights`` is the default because it is exact and needs no
        filesystem; a model whose weights do not fit in RAM has to save. This
        asserts the disk path restores the same weights, not merely that a file
        appears.
        """
        model = _probe_model()
        stopper = ConsecutiveIncreaseEarlyStopping(
            patience=2, checkpoint_path=str(tmp_path / "snapshots")
        )
        stopper.set_model(model)
        for epoch, value in enumerate([3.0, 2.0, 2.1, 2.2]):
            _write(model, value)
            stopper.on_epoch_end(epoch, {"val_loss": value})
        stopper.on_train_end()
        assert _read(model) == pytest.approx(2.0)
        assert (tmp_path / "snapshots" / "consecutive_increase_best.weights.h5").exists()


class _TappedModel(keras.Model):
    """A tap behind a projection, so the spatial loss is a real gradient path.

    Sequence-shaped throughout: the CLM head scores ``(B, T, vocab)``, so a
    ``(B, 4)`` projection output would make the loss reject the rank before the
    logger ever ran.
    """

    def __init__(self, alpha=2.5, vocab_size=16, name="tapped"):
        super().__init__(name=name)
        # 64 units -> an 8 x 8 grid, comfortably above the 25 a radius-2
        # neighbourhood needs. 16 units would be a 4 x 4 grid and the tap would
        # refuse to build.
        self.project = keras.layers.Dense(64, use_bias=False)
        self.tap = SpatialSmoothness(
            alpha=alpha, radius=2, seed=0, name="tap"
        )
        self.head = keras.layers.Dense(vocab_size, use_bias=False)

    def call(self, inputs, training=None):
        return {"logits": self.head(
            self.tap(self.project(inputs), training=training)
        )}


def _head(backbone, aggregate, vocab_size=16):
    from dl_techniques.models.common.masked_language_model import (
        CausalLanguageModel,
    )

    return CausalLanguageModel(
        backbone=backbone,
        vocab_size=vocab_size,
        skip_head=True,
        output_key="logits",
        pre_shifted=True,
        aggregate_backbone_losses=aggregate,
        verify_causality=False,
    )


def _dataset(rows=4, length=5, vocab_size=16):
    return tf.data.Dataset.from_tensor_slices(
        (
            np.random.default_rng(1).normal(
                size=(rows, length, 4)
            ).astype("float32"),
            np.random.default_rng(2).integers(
                0, vocab_size, size=(rows, length)
            ).astype("int32"),
        )
    ).batch(2)


class TestSpatialLossLogger:
    """The arithmetic, and the two ways a run can look healthy while the
    regularizer does nothing."""

    def _run(self, log_every=1, aggregate=True, rows=4):
        keras.utils.set_random_seed(0)
        backbone = _TappedModel()
        head = _head(backbone, aggregate)
        head.compile(optimizer=keras.optimizers.SGD(1e-3))
        head(np.zeros((1, 5, 4), dtype="float32"), training=False)
        callback = SpatialLossLogger(log_every=log_every)
        head.fit(
            _dataset(rows=rows), epochs=1, verbose=0, callbacks=[callback]
        )
        return callback, head

    def test_task_loss_is_the_reported_loss_minus_the_spatial_total(self):
        callback, head = self._run()
        recorded = callback.last_recorded

        assert "spatial/tap" in recorded
        assert recorded["spatial/total"] == pytest.approx(
            head.backbone.tap.alpha * recorded["spatial/tap"], rel=1e-4
        )
        assert recorded["spatial/unweighted"] == pytest.approx(
            recorded["spatial/tap"]
        )
        assert recorded["task_loss"] == pytest.approx(
            recorded["loss"] - recorded["spatial/total"], rel=1e-5
        )
        assert recorded["spatial/share"] == pytest.approx(
            recorded["spatial/total"] / recorded["loss"], rel=1e-5
        )

    def test_the_split_moves_with_alpha(self):
        """The spatial total tracks ``alpha`` and the task part does not."""
        totals = {}
        for alpha in (1.0, 4.0):
            keras.utils.set_random_seed(0)
            backbone = _TappedModel(alpha=alpha)
            head = _head(backbone, aggregate=True)
            head.compile(optimizer=keras.optimizers.SGD(1e-3))
            head(np.zeros((1, 5, 4), dtype="float32"), training=False)
            callback = SpatialLossLogger(log_every=1)
            head.fit(_dataset(), epochs=1, verbose=0, callbacks=[callback])
            totals[alpha] = callback.last_recorded["spatial/total"]

        assert totals[4.0] > totals[1.0] + 0.5, totals

    def test_an_unaggregated_head_would_be_reported_as_a_zero_share(self):
        """The trap this callback exists to make visible, tested on the logger.

        Whether the training loop folds the backbone's losses into its reported
        loss is the HEAD's decision, and it is proven end-to-end in
        ``tests/test_models/test_topolm/`` (train loss 15.31 aggregated against
        5.36 not). What belongs here is the logger's own reading of the two cases,
        driven with hand-built ``logs`` so the arithmetic is pinned rather than
        inherited from whichever path the head happens to take.

        Unaggregated, the tap still records a penalty but the reported loss does
        not contain it -- so ``task_loss`` equals ``loss`` and the share is
        ``0.0``, which is the signature of a run whose regularizer does nothing.
        """
        backbone = _TappedModel(alpha=2.5)
        backbone(np.zeros((1, 5, 4), dtype="float32"), training=True)
        penalty = float(
            ops.convert_to_numpy(backbone.tap.last_spatial_loss)
        )
        assert penalty > 0.0

        callback = SpatialLossLogger(log_every=1)
        callback.set_model(backbone)

        reported = 5.3  # the task loss, with no spatial term folded in
        callback.on_train_batch_end(
            0, {"loss": reported, "compile_loss": reported}
        )
        recorded = callback.last_recorded
        assert recorded["spatial/total"] > 0.0, "the tap computed nothing"
        # The gap between the reported loss and the compiled one is ZERO, so the
        # whole spatial term is unaccounted for: computed every step, folded into
        # nothing. This, not `task_loss`, is the field that reveals it --
        # `task_loss = loss - spatial_total` holds by construction either way.
        assert recorded["spatial/accounted"] == pytest.approx(0.0, abs=1e-9)
        assert recorded["spatial/unaccounted"] == pytest.approx(
            recorded["spatial/total"], rel=1e-5
        )

    def test_an_aggregated_loss_is_split_off_the_task_term(self):
        """The other arm, on the same fixture: fold the term in and it shows."""
        backbone = _TappedModel(alpha=2.5)
        backbone(np.zeros((1, 5, 4), dtype="float32"), training=True)
        penalty = float(ops.convert_to_numpy(backbone.tap.last_spatial_loss))

        callback = SpatialLossLogger(log_every=1)
        callback.set_model(backbone)
        reported = 5.3 + 2.5 * penalty
        callback.on_train_batch_end(
            0, {"loss": reported, "compile_loss": 5.3}
        )
        recorded = callback.last_recorded
        assert recorded["task_loss"] == pytest.approx(5.3, rel=1e-4)
        assert recorded["spatial/share"] > 0.1
        assert recorded["spatial/accounted"] == pytest.approx(
            2.5 * penalty, rel=1e-4
        )
        assert recorded["spatial/unaccounted"] == pytest.approx(0.0, abs=1e-5)

    def test_a_real_training_step_reports_the_split(self):
        """End to end, through the CLM head, on a live batch."""
        callback, _ = self._run(aggregate=True)
        recorded = callback.last_recorded
        assert recorded["spatial/share"] > 0.1, recorded["spatial/share"]
        assert recorded["task_loss"] < recorded["loss"]

    def test_the_cadence_is_respected(self):
        callback, _ = self._run(log_every=3, rows=8)
        assert callback._batch == 4, "four batches, emitted on the 3rd only"
        assert "spatial/total" in callback.last_recorded

    def test_a_cadence_that_never_fires_records_nothing(self):
        callback, _ = self._run(log_every=99, rows=4)
        assert callback._batch == 2
        assert callback.last_recorded == {}

    def test_a_model_without_taps_is_reported_not_crashed(self):
        model = keras.Sequential(
            [keras.layers.Dense(4, activation="relu")], name="plain"
        )
        model.compile(optimizer="sgd", loss="mse")
        callback = SpatialLossLogger(log_every=1)
        model.fit(
            np.zeros((4, 4), "float32"), np.zeros((4, 4), "float32"),
            batch_size=2, epochs=1, verbose=0, callbacks=[callback],
        )
        assert "spatial/total" not in callback.last_recorded
        assert callback._taps == []

    def test_the_taps_are_discovered_through_a_real_topolm_stack(self):
        """Discovery walks the whole tree, so it reaches taps nested in blocks."""
        model = TopoLM(
            vocab_size=VOCAB, embed_dim=64, depth=2, num_heads=2,
            max_seq_len=32, ffn_intermediate_size=128, radius=2, alpha=2.5,
            name="topolm",
        )
        # Built first: a layer's `path` is None until its parent has built it,
        # and every path below is read.
        model(
            np.random.default_rng(3).integers(
                0, VOCAB, size=(1, 8)
            ).astype("int32"),
            training=False,
        )
        callback = SpatialLossLogger(log_every=1)
        callback.set_model(model)
        assert len(callback._taps) == 4, len(callback._taps)

        # A nested layer's `.name` is the bare leaf; the per-block scope lives in
        # the PATH. Reading `.name` would yield four entries but only two
        # distinct strings, and a set comparison on names would pass with half
        # the stack accounted for.
        assert {tap.path.split("/", 1)[-1] for tap in callback._taps} == {
            f"block_{index}/{site}_tap"
            for index in range(2)
            for site in ("attention", "mlp")
        }
        assert {tap.name for tap in callback._taps} == {
            "attention_tap", "mlp_tap"
        }

    def test_a_custom_prefix_is_honoured(self):
        callback, _ = self._run()
        assert "spatial/tap" in callback.last_recorded

        keras.utils.set_random_seed(0)
        backbone = _TappedModel()
        head = _head(backbone, aggregate=True)
        head.compile(optimizer=keras.optimizers.SGD(1e-3))
        head(np.zeros((1, 5, 4), dtype="float32"), training=False)
        renamed = SpatialLossLogger(log_every=1, prefix="topo")
        head.fit(_dataset(), epochs=1, verbose=0, callbacks=[renamed])
        assert "topo/total" in renamed.last_recorded
        assert "spatial/total" not in renamed.last_recorded

    def test_the_config_carries_every_constructor_argument(self):
        callback = SpatialLossLogger(log_every=7, prefix="topo", verbose=1)
        config = callback.get_config()
        assert config["log_every"] == 7
        assert config["prefix"] == "topo"
        assert config["verbose"] == 1

    @pytest.mark.parametrize("value", [0, -1])
    def test_a_non_positive_cadence_is_rejected(self, value):
        with pytest.raises(ValueError, match="log_every must be >= 1"):
            SpatialLossLogger(log_every=value)
