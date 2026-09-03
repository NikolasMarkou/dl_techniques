"""Tests for ``EMAShadowCallback`` (checkpoint-only EMA-shadow driver)."""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.callbacks.ema_shadow_callback import (
    EMAShadowCallback,
    cosine_ema_schedule,
)
from dl_techniques.utils import logger as logger_module


class _ShadowModel(keras.Model):
    """Test-only model: two `Dense` "live" layers plus a matching set of
    non-trainable "shadow" variables that `update_ema_shadow` maintains via
    `.assign()`. Mirrors the scope boundary in the Step 5 prompt -- this is
    NOT `LeVJEPATrainingModel` (which does not exist yet).

    Shadow-initialization ownership lives HERE (the model), not in
    `EMAShadowCallback`: `update_ema_shadow`'s first call copies the live
    weights into the (lazily-built) shadow variables; every later call runs
    the real EMA blend. This matches the docstring's stated split of
    responsibility.
    """

    def __init__(self, **kwargs):
        # Keras derives a default scope name from the class name when none
        # is given; a leading underscore (this class is test-file-private)
        # is not a valid root scope name, so pin an explicit one.
        kwargs.setdefault("name", "shadow_model")
        super().__init__(**kwargs)
        self.dense1 = keras.layers.Dense(6, name="dense1")
        self.dense2 = keras.layers.Dense(3, name="dense2")
        self._shadow_vars = None
        self._shadow_initialized = False
        self.update_calls = []

    def build(self, input_shape):
        # Shadow variables MUST be created here (or in __init__), not lazily
        # inside `update_ema_shadow` at call time: Keras 3 forbids
        # `add_weight` once a layer/model's `built` flag is True, and
        # `update_ema_shadow` fires from `on_train_batch_end`, i.e. strictly
        # AFTER the model's first forward pass has already set `built=True`.
        self.dense1.build(input_shape)
        hidden_shape = tuple(input_shape[:-1]) + (self.dense1.units,)
        self.dense2.build(hidden_shape)
        self._shadow_vars = [
            self.add_weight(
                shape=w.shape,
                initializer="zeros",
                trainable=False,
                name=f"shadow_{i}",
            )
            for i, w in enumerate(self.trainable_weights)
        ]
        super().build(input_shape)

    def call(self, inputs, training=None):
        return self.dense2(self.dense1(inputs))

    def update_ema_shadow(self, decay: float) -> None:
        self.update_calls.append(decay)
        if not self._shadow_initialized:
            for shadow, live in zip(self._shadow_vars, self.trainable_weights):
                shadow.assign(live)
            self._shadow_initialized = True
            return
        for shadow, live in zip(self._shadow_vars, self.trainable_weights):
            shadow.assign(decay * shadow + (1.0 - decay) * live)


class _NoShadowModel(keras.Model):
    """A model deliberately missing `update_ema_shadow`."""

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "no_shadow_model")
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(3, name="dense")

    def call(self, inputs, training=None):
        return self.dense(inputs)


def _fit(model, callback, steps=1, batch_size=4):
    x = np.random.randn(batch_size * steps, 5).astype("float32")
    y = np.random.randn(batch_size * steps, 3).astype("float32")
    model.compile(optimizer="sgd", loss="mse")
    model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0, callbacks=[callback])


class TestEMAShadowCallbackVacuityCheck:
    def test_model_is_unbuilt_at_train_begin(self):
        """Proves this suite WOULD catch an on_train_begin-fires-before-build
        defect: probe the live model's weight count at on_train_begin and
        confirm it is exactly 0 (unbuilt), per plans/LESSONS.md's pinned
        gotcha -- BEFORE checking any post-first-batch behavior below."""
        model = _ShadowModel()
        assert len(model.weights) == 0  # not yet built at all

        captured = {}

        class _ProbeCallback(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                captured["weight_count"] = len(self.model.weights)

        model.compile(optimizer="sgd", loss="mse")
        x = np.random.randn(4, 5).astype("float32")
        y = np.random.randn(4, 3).astype("float32")
        model.fit(x, y, epochs=1, batch_size=4, verbose=0, callbacks=[_ProbeCallback()])

        assert captured["weight_count"] == 0


class TestEMAShadowCallbackUpdateFires:
    def test_shadow_moves_while_live_weights_are_a_normal_sgd_step(self):
        model = _ShadowModel()
        callback = EMAShadowCallback(decay=0.5, update_every=1)
        _fit(model, callback, steps=2, batch_size=2)

        assert model._shadow_vars is not None
        assert len(model.update_calls) == 2  # both steps qualify (update_every=1)

        shadow_final = [np.array(w) for w in model._shadow_vars]
        live_final = [np.array(w) for w in model.trainable_weights]

        # Shadow is a blend of the post-step-0 snapshot and the post-step-1
        # live weights, so it must differ from BOTH the all-zero init and
        # the exact final live weights -- proving the EMA path (not a mere
        # copy) actually ran.
        assert any(np.any(s != 0.0) for s in shadow_final)
        differs_from_live = any(
            not np.allclose(s, l, atol=1e-9, rtol=0)
            for s, l in zip(shadow_final, live_final)
        )
        assert differs_from_live, "shadow collapsed onto live weights (not an EMA)"

    def test_shadow_variables_are_non_trainable_and_ungradiented(self):
        model = _ShadowModel()
        callback = EMAShadowCallback(decay=0.9, update_every=1)
        _fit(model, callback, steps=1, batch_size=4)

        assert model._shadow_vars is not None
        for shadow_var in model._shadow_vars:
            assert shadow_var.trainable is False

        shadow_tf_vars = [
            tf.Variable(np.array(v), trainable=False) for v in model._shadow_vars
        ]
        x = tf.constant(np.random.randn(2, 5).astype("float32"))
        with tf.GradientTape() as tape:
            tape.watch(shadow_tf_vars)
            y_pred = model(x, training=False)
            loss = tf.reduce_mean(tf.square(y_pred))
        grads = tape.gradient(loss, shadow_tf_vars)
        assert all(g is None for g in grads)


class TestEMAShadowCallbackSkipSemantics:
    def test_update_every_skips_non_qualifying_steps(self):
        model = _ShadowModel()
        callback = EMAShadowCallback(decay=0.9, update_every=3)
        _fit(model, callback, steps=2, batch_size=2)

        # current_step in {0, 1}; only step 0 satisfies (0 % 3 == 0).
        assert len(model.update_calls) == 1

    def test_start_step_gates_early_updates(self):
        model = _ShadowModel()
        callback = EMAShadowCallback(decay=0.9, update_every=1, start_step=5)
        _fit(model, callback, steps=3, batch_size=2)

        assert model.update_calls == []


class TestEMAShadowCallbackMissingMethod:
    def test_missing_update_ema_shadow_disables_with_one_warning(self, monkeypatch):
        model = _NoShadowModel()
        callback = EMAShadowCallback(decay=0.9, update_every=1)

        warnings = []
        monkeypatch.setattr(
            logger_module.logger, "warning", lambda msg: warnings.append(msg)
        )

        # Must not raise despite the model lacking update_ema_shadow.
        _fit(model, callback, steps=2, batch_size=2)

        assert len(warnings) == 1
        assert "update_ema_shadow" in warnings[0]


class TestEMAShadowCallbackDecaySchedule:
    def test_flat_decay_is_constant(self):
        callback = EMAShadowCallback(decay=0.7, update_every=1)
        assert callback.decay_schedule(0) == pytest.approx(0.7)
        assert callback.decay_schedule(10) == pytest.approx(0.7)

    def test_schedule_callable_is_used_directly(self):
        schedule = cosine_ema_schedule(0.5, 0.99, total_steps=10)
        callback = EMAShadowCallback(decay=schedule, update_every=1)
        assert callback.decay_schedule is schedule

    def test_invalid_update_every_raises(self):
        with pytest.raises(ValueError, match="update_every"):
            EMAShadowCallback(decay=0.9, update_every=0)

    def test_invalid_start_step_raises(self):
        with pytest.raises(ValueError, match="start_step"):
            EMAShadowCallback(decay=0.9, start_step=-1)
