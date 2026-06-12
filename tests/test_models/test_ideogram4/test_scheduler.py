"""Tests for the Ideogram4 logit-normal schedule + Euler sampler parameters."""

import math

import numpy as np
import pytest
from scipy.special import ndtri, expit

from dl_techniques.models.ideogram4.scheduler import (
    LogitNormalSchedule,
    get_schedule_for_resolution,
    make_step_intervals,
    SamplerParameters,
    PRESETS,
)


def _numpy_ref(t, mean, std, logsnr_min=-15.0, logsnr_max=18.0):
    """Direct NumPy reference of the PyTorch LogitNormalSchedule.__call__."""
    t_arr = np.asarray(t, dtype=np.float64)
    y = mean + std * ndtri(t_arr)
    t_ = 1.0 - expit(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
    t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
    return np.clip(t_, t_min, t_max).astype(np.float32)


class TestLogitNormalSchedule:
    """LogitNormalSchedule warp + clamp."""

    @pytest.mark.parametrize("mean,std", [(0.0, 1.0), (1.0, 1.5), (-0.5, 1.75)])
    @pytest.mark.parametrize("t", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_scalar_matches_numpy_ref(self, mean, std, t):
        sched = LogitNormalSchedule(mean=mean, std=std)
        out = sched(t)
        ref = _numpy_ref(t, mean, std)
        assert isinstance(out, float)
        np.testing.assert_allclose(out, float(ref), atol=1e-6)

    def test_array_matches_numpy_ref(self):
        mean, std = 1.0, 1.5
        t = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95], dtype=np.float64)
        sched = LogitNormalSchedule(mean=mean, std=std)
        out = sched(t)
        ref = _numpy_ref(t, mean, std)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, ref, atol=1e-6)

    def test_clamp_lower_bound_applied(self):
        # A large positive mean pushes expit(y) -> 1, so 1 - expit -> 0,
        # which must be clamped UP to t_min.
        logsnr_max = 18.0
        t_min = 1.0 / (1.0 + math.exp(0.5 * logsnr_max))
        sched = LogitNormalSchedule(mean=50.0, std=1.0)
        out = sched(0.5)
        # Without clamp the raw value would be ~0; assert it sits at t_min.
        np.testing.assert_allclose(out, t_min, atol=1e-6)
        assert out >= t_min - 1e-7

    def test_clamp_upper_bound_applied(self):
        # A large negative mean pushes expit(y) -> 0, so 1 - expit -> 1,
        # which must be clamped DOWN to t_max.
        logsnr_min = -15.0
        t_max = 1.0 / (1.0 + math.exp(0.5 * logsnr_min))
        sched = LogitNormalSchedule(mean=-50.0, std=1.0)
        out = sched(0.5)
        np.testing.assert_allclose(out, t_max, atol=1e-6)
        assert out <= t_max + 1e-7

    def test_output_in_clamp_bounds(self):
        sched = LogitNormalSchedule(mean=0.0, std=1.0)
        t = np.linspace(0.01, 0.99, 50)
        out = sched(t)
        t_min = 1.0 / (1.0 + math.exp(0.5 * 18.0))
        t_max = 1.0 / (1.0 + math.exp(0.5 * -15.0))
        assert np.all(out >= t_min - 1e-6)
        assert np.all(out <= t_max + 1e-6)

    def test_frozen_dataclass(self):
        sched = LogitNormalSchedule(mean=0.0, std=1.0)
        with pytest.raises(Exception):
            sched.mean = 1.0  # frozen -> FrozenInstanceError


class TestGetScheduleForResolution:
    """Resolution-aware mean shift."""

    def test_known_resolution_no_shift(self):
        sched = get_schedule_for_resolution((512, 512))
        # log(1) == 0 -> mean stays at known_mean.
        np.testing.assert_allclose(sched.mean, 1.0, atol=1e-7)

    def test_double_resolution_shift(self):
        # (1024, 1024) vs (512, 512): pixel ratio = 4 -> mean = 1 + 0.5*log(4).
        sched = get_schedule_for_resolution((1024, 1024))
        expected = 1.0 + 0.5 * math.log(4.0)
        np.testing.assert_allclose(sched.mean, expected, atol=1e-7)

    def test_custom_known_mean_and_std(self):
        sched = get_schedule_for_resolution(
            (1024, 1024), known_resolution=(512, 512), known_mean=0.5, std=1.75
        )
        expected_mean = 0.5 + 0.5 * math.log(4.0)
        np.testing.assert_allclose(sched.mean, expected_mean, atol=1e-7)
        np.testing.assert_allclose(sched.std, 1.75, atol=1e-7)

    def test_returns_schedule_instance(self):
        sched = get_schedule_for_resolution((768, 768))
        assert isinstance(sched, LogitNormalSchedule)


class TestMakeStepIntervals:
    """Linear step intervals."""

    @pytest.mark.parametrize("n", [1, 12, 20, 48])
    def test_length_is_n_plus_one(self, n):
        intervals = make_step_intervals(n)
        assert intervals.shape == (n + 1,)

    @pytest.mark.parametrize("n", [12, 20, 48])
    def test_endpoints(self, n):
        intervals = make_step_intervals(n)
        np.testing.assert_allclose(intervals[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(intervals[-1], 1.0, atol=1e-7)

    @pytest.mark.parametrize("n", [12, 20, 48])
    def test_strictly_increasing(self, n):
        intervals = make_step_intervals(n)
        assert np.all(np.diff(intervals) > 0)

    def test_dtype_float32(self):
        assert make_step_intervals(20).dtype == np.float32


class TestSamplerParameters:
    """SamplerParameters validation + preset registry."""

    def test_valid_construction(self):
        sp = SamplerParameters(
            num_steps=4, guidance_schedule=(3.0, 7.0, 7.0, 7.0), mu=0.0
        )
        assert sp.num_steps == 4
        assert sp.std == 1.0  # default

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="guidance_schedule has length"):
            SamplerParameters(
                num_steps=4, guidance_schedule=(3.0, 7.0), mu=0.0
            )

    def test_length_too_long_raises(self):
        with pytest.raises(ValueError):
            SamplerParameters(
                num_steps=2, guidance_schedule=(3.0, 7.0, 7.0), mu=0.0
            )

    @pytest.mark.parametrize("name", ["V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"])
    def test_preset_present_and_consistent(self, name):
        sp = PRESETS[name]
        assert isinstance(sp, SamplerParameters)
        assert len(sp.guidance_schedule) == sp.num_steps

    def test_preset_keys(self):
        assert set(PRESETS.keys()) == {
            "V4_QUALITY_48",
            "V4_DEFAULT_20",
            "V4_TURBO_12",
        }

    def test_v4_default_20_fields(self):
        sp = PRESETS["V4_DEFAULT_20"]
        assert sp.num_steps == 20
        assert sp.mu == 0.0
        assert sp.std == 1.75
        assert sp.guidance_schedule == (3.0,) * 2 + (7.0,) * 18
        # loop-index order: index 0 is the LAST (polish) step at gw=3.
        assert sp.guidance_schedule[0] == 3.0
        assert sp.guidance_schedule[-1] == 7.0

    def test_v4_turbo_12_fields(self):
        sp = PRESETS["V4_TURBO_12"]
        assert sp.num_steps == 12
        assert sp.mu == 0.5
        assert sp.std == 1.75
        assert sp.guidance_schedule == (3.0,) * 1 + (7.0,) * 11

    def test_v4_quality_48_fields(self):
        sp = PRESETS["V4_QUALITY_48"]
        assert sp.num_steps == 48
        assert sp.mu == 0.0
        assert sp.std == 1.5
        assert sp.guidance_schedule == (3.0,) * 3 + (7.0,) * 45
