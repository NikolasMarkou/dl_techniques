"""A callback instance reused for a second ``fit`` starts that run from fresh state.

Plan plan-2026-11-11T131349-13b28b37 D-021: ``on_train_begin`` resets the state a
run accumulates, per class. The proof for each class is the same shape: one
instance driven through two runs ends the second run in the state a FRESH
instance reaches after that run alone.

The depth-curve and JEPA resets are owned by the visualization steps and are not
checked here.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import numpy as np
import pytest

from dl_techniques.callbacks.consecutive_increase_early_stopping import (
    ConsecutiveIncreaseEarlyStopping,
)
from dl_techniques.callbacks.convunext_bottleneck_monitor import (
    ConvUnextBottleneckMonitorCallback,
)
from dl_techniques.callbacks.self_iterate_pool import SelfIteratePoolCallback
from dl_techniques.callbacks.spatial_loss_logger import SpatialLossLogger
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback
from dl_techniques.layers.regularization.spatial_smoothness import SpatialSmoothness


def _tiny_model() -> keras.Model:
    model = keras.Sequential(
        [keras.layers.Input(shape=(2,)), keras.layers.Dense(1)]
    )
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    return model


def _fit(model: keras.Model, callback, epochs: int) -> None:
    model.fit(
        np.ones((4, 2), "float32"),
        np.zeros((4, 1), "float32"),
        batch_size=2,
        epochs=epochs,
        verbose=0,
        callbacks=[callback],
    )


# ---------------------------------------------------------------------------
# ConvUnextBottleneckMonitorCallback
# ---------------------------------------------------------------------------
class _ScaledBottleneck:
    """Stub full model whose bottleneck is ``scale`` everywhere."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, batch, training=False):
        bottleneck = np.full((2, 4, 4, 8), self.scale, "float32")
        return [np.zeros((2, 4, 4, 3), "float32"), bottleneck]


def _monitor(tmp_path, scale: float, name: str):
    stub = _ScaledBottleneck(scale)
    callback = ConvUnextBottleneckMonitorCallback(
        full_model=stub,
        val_batch=np.zeros((2, 4, 4, 3), "float32"),
        output_dir=tmp_path / name,
        monitor_freq=1,
    )
    return stub, callback


def test_the_monitor_second_run_records_its_own_statistics(tmp_path):
    stub, reused = _monitor(tmp_path, 1.0, "reused")
    model = _tiny_model()
    _fit(model, reused, epochs=2)
    stub.scale = 2.0
    _fit(model, reused, epochs=3)

    _, fresh = _monitor(tmp_path, 2.0, "fresh")
    _fit(_tiny_model(), fresh, epochs=3)

    assert reused.epochs_seen == fresh.epochs_seen == [1, 2, 3]
    assert reused.history == fresh.history
    # The second run's value, not the first run's, sits at epoch 1.
    assert reused.history["mean_activation"][0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# TrainingCurvesCallback
# ---------------------------------------------------------------------------
def test_training_curves_second_run_starts_its_history_over(tmp_path):
    reused = TrainingCurvesCallback(output_dir=str(tmp_path / "reused"))
    model = _tiny_model()
    _fit(model, reused, epochs=2)
    _fit(model, reused, epochs=3)

    fresh = TrainingCurvesCallback(output_dir=str(tmp_path / "fresh"))
    _fit(_tiny_model(), fresh, epochs=3)

    assert reused._epochs == fresh._epochs == [1, 2, 3]
    assert len(reused._history) == len(fresh._history) == 3


# ---------------------------------------------------------------------------
# SelfIteratePoolCallback
# ---------------------------------------------------------------------------
def _pool_model(channels: int = 3) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4, 4, channels)),
            keras.layers.Conv2D(
                channels,
                1,
                use_bias=False,
                kernel_initializer=keras.initializers.Constant(0.4 / channels),
            ),
        ]
    )
    # A zero learning rate keeps the weights, so two runs see the same model.
    model.compile(optimizer=keras.optimizers.SGD(0.0), loss="mse")
    return model


def _pool(n: int = 8):
    rng = np.random.default_rng(0)
    clean = rng.uniform(-1, 1, size=(n, 4, 4, 3)).astype("float32")
    current = np.clip(
        clean + 0.1 * rng.normal(size=clean.shape).astype("float32"), -1, 1
    )
    return clean, current


def _pool_callback(clean, current):
    return SelfIteratePoolCallback(
        clean_pool=clean,
        current_input=current,
        get_sigma=lambda: 0.1,
        regen_freq=1,
        mix_ratio=0.5,
        seed=5,
    )


def _pool_fit(model, callback, current, epochs):
    model.fit(
        current, current, batch_size=4, epochs=epochs, verbose=0,
        shuffle=False, callbacks=[callback],
    )


def test_the_pool_second_run_equals_a_fresh_instance(tmp_path):
    clean, start = _pool()

    current_a = start.copy()
    reused = _pool_callback(clean, current_a)
    model = _pool_model()
    _pool_fit(model, reused, current_a, epochs=2)
    current_a[...] = start  # the trainer hands every run its own pool
    _pool_fit(model, reused, current_a, epochs=1)

    current_b = start.copy()
    fresh = _pool_callback(clean, current_b)
    _pool_fit(_pool_model(), fresh, current_b, epochs=1)

    assert reused.epochs_seen == fresh.epochs_seen == 1
    assert reused.history == fresh.history
    # The RNG restarts, so the slot permutation and fresh noise repeat.
    np.testing.assert_array_equal(current_a, current_b)


def test_the_pool_reset_does_not_touch_the_global_keras_seed_generator():
    keras.random.normal((1,))  # make sure the global generator exists
    generator = keras.src.backend.common.global_state.get_global_attribute(
        "global_seed_generator"
    )
    before = np.array(generator.state.numpy())

    clean, current = _pool()
    callback = _pool_callback(clean, current)
    callback.on_train_begin()

    np.testing.assert_array_equal(np.array(generator.state.numpy()), before)


def test_the_pool_residual_is_measured_before_the_writes():
    """mix 1.0: every slot is overwritten, so a residual taken afterwards is 0."""
    clean, current = _pool()
    model = _pool_model()
    callback = SelfIteratePoolCallback(
        clean_pool=clean,
        current_input=current,
        get_sigma=lambda: 0.1,
        mix_ratio=1.0,
        seed=3,
    )
    callback.set_model(model)
    old = current.copy()
    expected = float(np.mean(np.abs(np.asarray(model.predict(old, verbose=0)) - old)))
    assert expected > 0.0

    callback.on_epoch_end(0)

    assert callback.history["mean_residual"][0] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# ConsecutiveIncreaseEarlyStopping
# ---------------------------------------------------------------------------
def _drive(stopper, values):
    for epoch, value in enumerate(values):
        stopper.on_epoch_end(epoch, {"val_loss": value})


def test_early_stopping_second_run_does_not_inherit_the_first_runs_streak():
    model = _tiny_model()
    reused = ConsecutiveIncreaseEarlyStopping(patience=3)
    reused.set_model(model)
    reused.on_train_begin()
    _drive(reused, [1.0, 1.1, 1.2, 1.3])
    assert model.stop_training and reused.streak == 3

    model.stop_training = False  # what fit() does at the start of a run
    reused.on_train_begin()
    _drive(reused, [5.0, 4.0])

    fresh_model = _tiny_model()
    fresh = ConsecutiveIncreaseEarlyStopping(patience=3)
    fresh.set_model(fresh_model)
    fresh.on_train_begin()
    _drive(fresh, [5.0, 4.0])

    assert not model.stop_training
    assert reused.streak == fresh.streak == 0
    assert reused._previous == fresh._previous == 4.0
    assert reused.stopped_at_evaluation is None


def test_early_stopping_does_not_restore_a_previous_runs_snapshot():
    model = _tiny_model()
    stopper = ConsecutiveIncreaseEarlyStopping(patience=3)
    stopper.set_model(model)
    stopper.on_train_begin()
    _drive(stopper, [1.0, 1.1, 1.2, 1.3])
    stopper.on_train_end()

    stopper.on_train_begin()
    assert stopper.stopped_at_evaluation is None
    assert stopper._best_snapshot is None
    before = model.get_weights()
    stopper.on_train_end()
    for a, b in zip(before, model.get_weights()):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# SpatialLossLogger
# ---------------------------------------------------------------------------
class _TappedModel(keras.Model):
    def __init__(self):
        super().__init__(name="tapped")
        self.project = keras.layers.Dense(64, use_bias=False)
        self.tap = SpatialSmoothness(alpha=1.0, radius=2, seed=0, name="tap")
        self.head = keras.layers.Dense(1, use_bias=False)

    def call(self, inputs, training=None):
        return self.head(self.tap(self.project(inputs), training=training))


def _tapped_fit(model, callback):
    model.fit(
        np.ones((4, 4), "float32"),
        np.zeros((4, 1), "float32"),
        batch_size=2,
        epochs=1,
        verbose=0,
        callbacks=[callback],
    )


def test_spatial_logger_cadence_restarts_with_each_run():
    """2 batches per fit and ``log_every=3``: a fresh run never logs."""
    keras.utils.set_random_seed(0)
    model = _TappedModel()
    model.compile(optimizer=keras.optimizers.SGD(0.0), loss="mse")
    reused = SpatialLossLogger(log_every=3)
    _tapped_fit(model, reused)
    _tapped_fit(model, reused)

    fresh_model = _TappedModel()
    fresh_model.compile(optimizer=keras.optimizers.SGD(0.0), loss="mse")
    fresh = SpatialLossLogger(log_every=3)
    _tapped_fit(fresh_model, fresh)

    assert fresh.last_recorded == {}
    assert reused.last_recorded == {}
    assert reused._batch == fresh._batch == 2


def test_spatial_logger_forgets_the_last_runs_recorded_keys():
    keras.utils.set_random_seed(0)
    model = _TappedModel()
    model.compile(optimizer=keras.optimizers.SGD(0.0), loss="mse")
    callback = SpatialLossLogger(log_every=1)
    _tapped_fit(model, callback)
    assert callback.last_recorded

    callback.on_train_begin()

    assert callback.last_recorded == {}
    assert callback._batch == 0
