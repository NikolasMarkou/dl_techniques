"""Unit tests for the SD3 rectified-flow Euler scheduler.

Covers the forward interpolation (``add_noise``), the velocity target, the
sign/dt convention of the reverse Euler step (the load-bearing correctness
check), host-side logit-normal time sampling, the Eq.(19) loss weighting, and
the descending inference time grid.
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.models.sd3_mmdit.scheduler import FlowMatchEulerScheduler


@pytest.fixture
def sched() -> FlowMatchEulerScheduler:
    return FlowMatchEulerScheduler()


@pytest.fixture
def x0_noise():
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((2, 4, 4, 16)).astype(np.float32)
    noise = rng.standard_normal((2, 4, 4, 16)).astype(np.float32)
    return x0, noise


class TestAddNoise:
    def test_t0_is_x0(self, sched, x0_noise):
        x0, noise = x0_noise
        xt = keras.ops.convert_to_numpy(sched.add_noise(x0, noise, 0.0))
        np.testing.assert_allclose(xt, x0, atol=1e-6)

    def test_t1_is_noise(self, sched, x0_noise):
        x0, noise = x0_noise
        xt = keras.ops.convert_to_numpy(sched.add_noise(x0, noise, 1.0))
        np.testing.assert_allclose(xt, noise, atol=1e-6)

    def test_t_half_is_midpoint(self, sched, x0_noise):
        x0, noise = x0_noise
        xt = keras.ops.convert_to_numpy(sched.add_noise(x0, noise, 0.5))
        np.testing.assert_allclose(xt, 0.5 * (x0 + noise), atol=1e-6)

    def test_broadcast_per_sample_t(self, sched, x0_noise):
        x0, noise = x0_noise
        t = np.array([0.0, 1.0], dtype=np.float32).reshape(2, 1, 1, 1)
        xt = keras.ops.convert_to_numpy(sched.add_noise(x0, noise, t))
        np.testing.assert_allclose(xt[0], x0[0], atol=1e-6)
        np.testing.assert_allclose(xt[1], noise[1], atol=1e-6)


class TestVelocityTarget:
    def test_velocity_is_noise_minus_x0(self, sched, x0_noise):
        x0, noise = x0_noise
        v = keras.ops.convert_to_numpy(sched.velocity_target(x0, noise))
        np.testing.assert_allclose(v, noise - x0, atol=1e-6)


class TestEulerConsistency:
    def test_one_step_true_velocity_recovers_x0(self, sched, x0_noise):
        # The load-bearing sign/dt check: x1 = pure noise; stepping t=1 -> t=0
        # with the TRUE velocity recovers x0 exactly (straight-line path).
        x0, noise = x0_noise
        x1 = noise  # add_noise(x0, noise, t=1) == noise
        v = sched.velocity_target(x0, noise)
        x_next = keras.ops.convert_to_numpy(
            sched.euler_step(x1, v, t=1.0, t_next=0.0)
        )
        np.testing.assert_allclose(x_next, x0, atol=1e-5)

    def test_two_half_steps_recover_x0(self, sched, x0_noise):
        # Straight-line: any subdivision of the integration is exact.
        x0, noise = x0_noise
        v = sched.velocity_target(x0, noise)
        x1 = noise
        x_mid = sched.euler_step(x1, v, t=1.0, t_next=0.5)
        x_end = keras.ops.convert_to_numpy(
            sched.euler_step(x_mid, v, t=0.5, t_next=0.0)
        )
        np.testing.assert_allclose(x_end, x0, atol=1e-5)


class TestLogitNormalSampling:
    def test_shape_dtype_range(self, sched):
        t = sched.sample_logit_normal_t(1000, seed=0)
        assert t.shape == (1000,)
        assert t.dtype == np.float32
        assert np.all(t > 0.0)
        assert np.all(t < 1.0)

    def test_mean_loosely_central(self, sched):
        t = sched.sample_logit_normal_t(5000, seed=0)
        # Loose bound: shift=3 skews the mean upward but it stays mid-range.
        assert 0.3 < float(t.mean()) < 0.8

    def test_seed_reproducible(self, sched):
        a = sched.sample_logit_normal_t(64, seed=7)
        b = sched.sample_logit_normal_t(64, seed=7)
        np.testing.assert_array_equal(a, b)


class TestLogitNormalWeight:
    def test_positive_finite(self, sched):
        t = np.linspace(0.01, 0.99, 99).astype(np.float32)
        w = sched.logit_normal_weight(t)
        assert w.dtype == np.float32
        assert np.all(np.isfinite(w))
        assert np.all(w > 0.0)

    def test_matches_direct_formula(self, sched):
        t = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64)
        mean = sched.logit_mean
        std = sched.logit_std
        logit_t = np.log(t / (1.0 - t))
        term1 = t * (1.0 - t) * std * math.sqrt(2.0 * math.pi)
        term2 = np.exp((logit_t - mean) ** 2 / (2.0 * std ** 2))
        expected = (term1 * term2).astype(np.float32)
        got = sched.logit_normal_weight(t)
        np.testing.assert_allclose(got, expected, atol=1e-5)


class TestTimesteps:
    def test_descending_endpoints_length(self, sched):
        ts = sched.timesteps(50)
        assert ts.shape == (51,)  # num_inference_steps + 1 (appended 0.0)
        assert ts.dtype == np.float32
        # Strictly descending.
        assert np.all(np.diff(ts) < 0.0)
        # Starts near 1, ends exactly at 0.
        assert ts[0] == pytest.approx(1.0, abs=1e-3) or ts[0] < 1.0 + 1e-6
        assert ts[0] > 0.9
        assert ts[-1] == 0.0
