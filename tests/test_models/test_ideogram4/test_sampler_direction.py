"""Direction pins for the Ideogram4 Euler sampling loop (C-26 / D-002).

These tests pin the SIGN and the ENDPOINTS of the reverse-time integration in
``Ideogram4Pipeline.__call__``, which nothing in the package tested before:
``test_pipeline.py`` asserts shape / finiteness / range / seed-determinism, all
of which a sampler integrating data -> noise satisfies just as well, and
``test_scheduler.py::TestMakeStepIntervals::test_strictly_increasing`` pins the
UNIFORM grid ``make_step_intervals`` returns, not the schedule VALUES the loop
evaluates.

The oracle is the TRAINING convention, taken from
``src/train/ideogram4/train_ideogram4.py``, never read back out of the
pipeline::

    x_t = (1 - tau) * x0 + tau * x1,   x1 ~ N(0, I),   v = x1 - x0

so ``t = 0`` is clean data, ``t = 1`` is pure noise, and the velocity points
data -> noise. Reverse sampling therefore starts at the NOISE end of the time
grid (which is what the pipeline seeds ``z`` with) and must integrate with a
NEGATIVE ``dt`` down to the data end.

Two independent properties are pinned separately, because a stub velocity that
ignores ``t`` cannot see the second and a timestep recorder cannot see the
first:

* the integration direction -- one Euler pass fed the TRUE velocity recovers
  ``x0`` (``TestTrueVelocityRecoversX0``);
* the timesteps the network is EVALUATED at -- strictly descending from the
  noise end (``TestEvaluatedTimestepsDescend``).

Run on CPU (``CUDA_VISIBLE_DEVICES=""``): the ``atol=1e-6`` recovery bound is
below the GPU's own run-to-run disagreement (~5e-6) and would be meaningless
there.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.ideogram4.pipeline import Ideogram4Pipeline
from dl_techniques.models.ideogram4.scheduler import (
    LogitNormalSchedule,
    PRESETS,
    make_step_intervals,
)

# Tiny geometry, matching test_pipeline.py: pixels_per_token_edge = 4, so an
# 8x8 image is a 2x2 token grid (4 image tokens). NUM_TEXT != NUM_IMAGE is
# load-bearing -- the stubs tell the conditional (full-sequence) branch from the
# unconditional (image-only) branch by sequence length.
BATCH = 1
NUM_TEXT = 3
HEIGHT = 8
WIDTH = 8
NUM_IMAGE = 4

# Wide log-SNR bounds so the clamp is inert: t_max == 1.0 and t_min ~ 2e-22 in
# float32, i.e. the grid spans the FULL [noise, data] interval and a one-step
# integration is exact rather than short by the clamp margin (~7e-4 with the
# shipped -15/+18 bounds).
UNCLAMPED_SCHEDULE = LogitNormalSchedule(
    mean=0.0, std=1.0, logsnr_min=-100.0, logsnr_max=100.0
)


@pytest.fixture(scope="module")
def pipeline():
    return Ideogram4Pipeline.from_config("tiny", seed=0)


@pytest.fixture
def llm_features(pipeline):
    rng = np.random.default_rng(123)
    return rng.standard_normal(
        (BATCH, NUM_TEXT, pipeline.config.llm_features_dim)
    ).astype("float32")


class _Stub:
    """Base transformer stub: records the image-branch call, returns a velocity.

    The pipeline calls the transformer twice per Euler step -- once on the
    packed ``(B, T + num_image, C)`` sequence and once on the image-only
    ``(B, num_image, C)`` unconditional branch -- with the same ``t``, so only
    the image-only call is recorded (one entry per loop iteration). Both
    branches return the SAME per-image-token velocity, which makes the CFG
    blend ``gw*pos + (1-gw)*neg`` collapse to that velocity for any ``gw``.
    """

    def __init__(self):
        self.times = []
        self.latents = []

    def _velocity(self, x_img):
        raise NotImplementedError

    def __call__(self, inputs):
        x = inputs["x"]
        seq_len = int(keras.ops.shape(x)[1])
        # Last NUM_IMAGE rows are the image tokens on both branches (the packed
        # branch zero-pads the text rows of `z`).
        x_img = x[:, seq_len - NUM_IMAGE:, :]
        if seq_len == NUM_IMAGE:
            self.times.append(
                float(keras.ops.convert_to_numpy(inputs["t"])[0, 0])
            )
            self.latents.append(keras.ops.convert_to_numpy(x_img).copy())
        v_img = self._velocity(x_img)
        if seq_len == NUM_IMAGE:
            return v_img
        pad = keras.ops.zeros(
            (int(keras.ops.shape(x)[0]), seq_len - NUM_IMAGE, int(keras.ops.shape(x)[2])),
            dtype="float32",
        )
        return keras.ops.concatenate([pad, v_img], axis=1)


class _ZeroVelocity(_Stub):
    """Velocity 0: `z` never moves, so only the timesteps are observable."""

    def _velocity(self, x_img):
        return keras.ops.zeros_like(x_img)


class _UnitVelocity(_Stub):
    """Velocity 1: each Euler update adds exactly `dt`, making `dt` readable."""

    def _velocity(self, x_img):
        return keras.ops.ones_like(x_img)


class _TrueVelocity(_Stub):
    """Velocity `x1 - x0` for a fixed target `x0` and the sampler's own `x1`.

    ``x1`` is the pure-noise latent the pipeline seeds internally; it is read
    off the FIRST call (the pipeline has not integrated anything yet at that
    point), so the stub does not need to reproduce the internal RNG draw.
    """

    def __init__(self, x0):
        super().__init__()
        self.x0 = x0
        self.x1 = None

    def _velocity(self, x_img):
        if self.x1 is None:
            self.x1 = keras.ops.convert_to_numpy(x_img).copy()
        return keras.ops.convert_to_tensor(self.x1 - self.x0)


def _run(pipeline, monkeypatch, llm_features, stub, **kwargs):
    """Drive the real Euler loop with `stub`, returning the FINAL latent `z`.

    ``_decode`` is replaced by an identity so the VAE is bypassed and the loop's
    own output is observed directly (the decoded image cannot distinguish a
    latent from its mirror image).
    """
    monkeypatch.setattr(pipeline, "transformer", stub)
    monkeypatch.setattr(pipeline, "_decode", lambda z, grid_h, grid_w: z)
    out = pipeline(
        llm_features=llm_features,
        height=HEIGHT,
        width=WIDTH,
        **kwargs,
    )
    return keras.ops.convert_to_numpy(out)


class TestTrueVelocityRecoversX0:
    """Integrating the TRUE velocity must land on the clean data `x0`.

    Ported from ``test_sd3_mmdit/test_scheduler.py::
    test_one_step_true_velocity_recovers_x0``. On a straight-line rectified-flow
    path the velocity is constant, so starting at ``x1`` (the noise end, where
    the pipeline's initial `z` lives) and integrating down to ``t = 0`` gives
    ``x1 + (x1 - x0) * (0 - 1) == x0`` exactly, for ANY subdivision of the
    interval. Integrating the other way lands on ``2*x1 - x0``.
    """

    @pytest.mark.parametrize("num_steps", [1, 4])
    def test_true_velocity_recovers_x0(
        self, pipeline, monkeypatch, llm_features, num_steps
    ):
        rng = np.random.default_rng(7)
        x0 = rng.standard_normal(
            (BATCH, NUM_IMAGE, pipeline.config.in_channels)
        ).astype("float32")
        stub = _TrueVelocity(x0)

        z_final = _run(
            pipeline,
            monkeypatch,
            llm_features,
            stub,
            num_steps=num_steps,
            guidance_scale=7.0,
            seed=0,
            schedule=UNCLAMPED_SCHEDULE,
        )

        # Anti-vacuity: the noise the sampler started from is nowhere near x0,
        # so landing on x0 is a real statement about the integration.
        assert np.abs(stub.x1 - x0).max() > 0.5

        # atol is float32 ULP-bound, not slack: the measured CPU residual is
        # 1.19e-06 at these sample magnitudes (|x| up to ~4, so one ULP is
        # ~4.8e-07 and the update costs two roundings). The sd3_mmdit sibling
        # uses 1e-05 for the same reason. The defect this pins moves the result
        # by ~1.0e+01 (measured against the pre-fix loop), i.e. six orders of
        # magnitude above the bound.
        np.testing.assert_allclose(z_final, x0, atol=5e-6)


class TestEvaluatedTimestepsDescend:
    """The `t` the transformer is evaluated at must run noise -> data.

    The shipped presets are driven at their own ``mu``/``std``/``num_steps``/
    ``guidance_schedule`` but through an explicitly constructed schedule rather
    than ``get_schedule_for_resolution``: at the 8x8 test resolution the
    resolution shift (``mean += 0.5*log(pixels/512^2)``, i.e. -4.16 here) pushes
    the whole grid against the upper clamp, which would make any statement about
    the grid's lower end a statement about the test's image size instead.
    """

    def _grid_extremes(self, schedule, num_steps):
        grid = np.asarray(schedule(make_step_intervals(num_steps).astype(np.float64)))
        return float(grid.max()), float(grid.min())

    @pytest.mark.parametrize("preset_name", sorted(PRESETS))
    def test_timesteps_strictly_descend_from_the_noise_end(
        self, pipeline, monkeypatch, llm_features, preset_name
    ):
        preset = PRESETS[preset_name]
        schedule = LogitNormalSchedule(mean=preset.mu, std=preset.std)
        t_hi, t_lo = self._grid_extremes(schedule, preset.num_steps)
        stub = _ZeroVelocity()

        _run(
            pipeline,
            monkeypatch,
            llm_features,
            stub,
            num_steps=preset.num_steps,
            guidance_schedule=preset.guidance_schedule,
            seed=0,
            schedule=schedule,
        )

        times = np.asarray(stub.times)
        assert times.shape == (preset.num_steps,)
        # Starts at the noise end -- the endpoint the initial `z` (pure
        # keras.random.normal) actually corresponds to.
        assert times[0] == pytest.approx(t_hi, abs=1e-6)
        assert np.all(np.diff(times) < 0.0), f"t not strictly descending: {times}"
        # ...and finishes in the data quarter of the schedule's range.
        assert times[-1] < t_lo + 0.25 * (t_hi - t_lo)

    @pytest.mark.parametrize("preset_name", sorted(PRESETS))
    def test_every_euler_step_size_is_negative(
        self, pipeline, monkeypatch, llm_features, preset_name
    ):
        preset = PRESETS[preset_name]
        schedule = LogitNormalSchedule(mean=preset.mu, std=preset.std)
        t_hi, t_lo = self._grid_extremes(schedule, preset.num_steps)
        stub = _UnitVelocity()

        z_final = _run(
            pipeline,
            monkeypatch,
            llm_features,
            stub,
            num_steps=preset.num_steps,
            guidance_schedule=preset.guidance_schedule,
            seed=0,
            schedule=schedule,
        )

        # With v == 1 each update is `z += dt`, so the per-step dt is readable
        # from the latents the loop handed the stub, plus the final latent.
        traj = np.stack(
            [lat.mean() for lat in stub.latents] + [z_final.mean()]
        )
        dts = np.diff(traj)
        assert dts.shape == (preset.num_steps,)
        assert np.all(dts < 0.0), f"non-negative Euler step(s): {dts}"
        # The loop traverses the WHOLE schedule range, downward.
        assert dts.sum() == pytest.approx(t_lo - t_hi, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
