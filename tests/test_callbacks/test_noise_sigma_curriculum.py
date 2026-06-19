"""Tests for NoiseSigmaCurriculumCallback (plan_2026-06-19_ed071c02)."""

import math

import pytest
import keras
import tensorflow as tf

from dl_techniques.callbacks.noise_sigma_curriculum import (
    NoiseSigmaCurriculumCallback,
)


class TestNoiseSigmaCurriculumCallback:
    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule"):
            NoiseSigmaCurriculumCallback(schedule="bogus")

    def test_invalid_total_epochs_raises(self):
        with pytest.raises(ValueError, match="total_epochs"):
            NoiseSigmaCurriculumCallback(total_epochs=0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            NoiseSigmaCurriculumCallback(sigma_max_start=-0.1)

    def test_exp_requires_positive(self):
        with pytest.raises(ValueError, match="exp"):
            NoiseSigmaCurriculumCallback(
                schedule="exp", sigma_max_start=0.0, sigma_max_end=0.5
            )

    def test_linear_widens_start_mid_end(self):
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_start=0.05, sigma_max_end=0.5, total_epochs=10, schedule="linear"
        )
        assert math.isclose(cb.sigma_max_at(0), 0.05, abs_tol=1e-6)
        assert math.isclose(cb.sigma_max_at(9), 0.5, abs_tol=1e-6)
        # Middle (5/9 of the way): 0.05 + 0.45 * 5/9 = 0.30.
        assert math.isclose(cb.sigma_max_at(5), 0.30, abs_tol=1e-6)
        # Monotonic widening.
        vals = [cb.sigma_max_at(e) for e in range(10)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_cosine_endpoints(self):
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_start=0.05, sigma_max_end=0.5, total_epochs=10, schedule="cosine"
        )
        assert math.isclose(cb.sigma_max_at(0), 0.05, abs_tol=1e-6)
        assert math.isclose(cb.sigma_max_at(9), 0.5, abs_tol=1e-6)
        # cosine is monotonic start->end and stays within [start, end].
        vals = [cb.sigma_max_at(e) for e in range(10)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))
        assert all(0.05 - 1e-9 <= v <= 0.5 + 1e-9 for v in vals)

    def test_exp_endpoints(self):
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_start=0.05, sigma_max_end=0.8, total_epochs=5, schedule="exp"
        )
        assert math.isclose(cb.sigma_max_at(0), 0.05, abs_tol=1e-6)
        assert math.isclose(cb.sigma_max_at(4), 0.8, abs_tol=1e-6)

    def test_single_epoch_returns_end(self):
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_start=0.05, sigma_max_end=0.5, total_epochs=1
        )
        assert math.isclose(cb.sigma_max_at(0), 0.5, abs_tol=1e-6)

    def test_on_epoch_begin_assigns_variable(self):
        var = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_var=var, sigma_max_start=0.05, sigma_max_end=0.5,
            total_epochs=10, schedule="linear",
        )
        cb.on_epoch_begin(0)
        assert math.isclose(float(var), 0.05, abs_tol=1e-6)
        cb.on_epoch_begin(5)
        assert math.isclose(float(var), 0.30, abs_tol=1e-6)
        cb.on_epoch_begin(9)
        assert math.isclose(float(var), 0.5, abs_tol=1e-6)

    def test_on_epoch_begin_no_variable_is_noop(self):
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_start=0.05, sigma_max_end=0.5, total_epochs=10
        )
        # Must not raise when no variable is attached.
        cb.on_epoch_begin(3)

    def test_optional_sigma_min_var_assigned(self):
        smax = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        smin = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_var=smax, sigma_max_start=0.1, sigma_max_end=0.5,
            total_epochs=4, schedule="linear",
            sigma_min_var=smin, sigma_min_start=0.0, sigma_min_end=0.1,
        )
        cb.on_epoch_begin(3)
        assert math.isclose(float(smax), 0.5, abs_tol=1e-6)
        assert math.isclose(float(smin), 0.1, abs_tol=1e-6)

    def test_get_config_round_trip(self):
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_var=tf.Variable(0.0),  # variable must NOT appear in config
            sigma_max_start=0.05, sigma_max_end=0.5, total_epochs=30,
            schedule="cosine", sigma_min_start=0.0, sigma_min_end=0.02,
        )
        cfg = cb.get_config()
        assert "sigma_max_var" not in cfg
        assert "sigma_min_var" not in cfg
        rebuilt = NoiseSigmaCurriculumCallback.from_config(cfg)
        assert rebuilt.sigma_max_start == 0.05
        assert rebuilt.sigma_max_end == 0.5
        assert rebuilt.total_epochs == 30
        assert rebuilt.schedule == "cosine"
        assert rebuilt.sigma_min_end == 0.02
        # Reconstructed callback has no variable -> on_epoch_begin is a safe no-op.
        rebuilt.on_epoch_begin(0)

    def test_keras_serializable_registered(self):
        cb = NoiseSigmaCurriculumCallback(total_epochs=5)
        cfg = keras.saving.serialize_keras_object(cb)
        restored = keras.saving.deserialize_keras_object(cfg)
        assert isinstance(restored, NoiseSigmaCurriculumCallback)
        assert restored.total_epochs == 5
