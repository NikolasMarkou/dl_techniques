"""
RED proofs for ``precision_arm_oracle`` -- each part must reject its own defect.

An fp16 arm that cannot fail is worse than no arm, because it converts an open
row into a closed one. Every assertion in ``assert_precision_arm`` is proved
here against a model that carries exactly the defect that assertion names, and
against a healthy twin that must stay green.

The defects modelled here are the four real shapes this plan measured in the
tree, not invented ones:

* ``_RaisingFFTModel``      -- ``fft2`` under ``mixed_float16``: the literal
  ``pw_fnet`` / ``darkir`` failure (``TypeError: the real and imag components
  have incorrect types: float16 float16``).
* ``_Float32LeakModel``     -- returns float32 under ``mixed_float16``: the
  shape a forward-only arm cannot see.
* ``_VoidGuardModel``       -- divides by ``x + 1e-9``, which is *exactly* 0.0
  in float16: the ``nam`` / ``superpoint`` family.
* ``_DeadBackwardModel``    -- forward is green, backward reaches no variable.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import ops

from .precision_arm_oracle import (
    assert_precision_arm,
    flatten_tensors,
    precision_policy,
    run_forward,
)


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------
class _HealthyModel(keras.Model):
    """A model with nothing wrong with it. The anti-vacuity twin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(4)

    def call(self, inputs, training=None):
        return self.dense(inputs)


class _RaisingFFTModel(keras.Model):
    """``fft2`` with no float32 island -- raises under ``mixed_float16``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(4)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        real, _ = ops.fft2((ops.expand_dims(x, 1), ops.zeros_like(ops.expand_dims(x, 1))))
        return ops.squeeze(real, 1)


class _Float32LeakModel(keras.Model):
    """Green forward, but hands the caller float32 under ``mixed_float16``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(4)

    def call(self, inputs, training=None):
        return ops.cast(self.dense(inputs), "float32")


class _VoidGuardModel(keras.Model):
    """``+ 1e-9`` is exactly 0.0 in float16, so a zero row divides by zero."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # A zeros kernel makes the numerator and the guarded denominator both
        # exactly zero, which is what an all-pad row does in ``nam``.
        self.dense = keras.layers.Dense(4, kernel_initializer="zeros",
                                        bias_initializer="zeros")

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        return x / (ops.sum(ops.abs(x), axis=-1, keepdims=True) + 1e-9)


class _DeadBackwardModel(keras.Model):
    """Finite float16 forward whose output is disconnected from its variable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(4)

    def call(self, inputs, training=None):
        y = self.dense(inputs)
        return ops.stop_gradient(y)


def _inputs():
    return np.random.RandomState(0).randn(2, 6).astype("float32")


# ---------------------------------------------------------------------------
# The healthy twin -- if this is not green, every RED below proves nothing
# ---------------------------------------------------------------------------
def test_the_oracle_accepts_a_healthy_model():
    reports = assert_precision_arm(
        build=_HealthyModel, make_inputs=_inputs, rtol_against_float32=1e-2
    )
    assert reports["mixed_float16"]["dtypes"] == ["float16"]
    assert reports["float32"]["dtypes"] == ["float32"]
    assert reports["backward_mixed_float16"]["n_none"] == 0
    assert reports["backward_mixed_float16"]["grad_norm_sum"] > 0.0


# ---------------------------------------------------------------------------
# Part 1 -- the forward must run at all
# ---------------------------------------------------------------------------
def test_part_1_rejects_a_model_that_raises_under_mixed_float16():
    with pytest.raises(Exception) as excinfo:
        assert_precision_arm(build=_RaisingFFTModel, make_inputs=_inputs)
    # It must be the model raising, not the oracle mis-handling it.
    assert not isinstance(excinfo.value, AssertionError)
    assert "float16" in str(excinfo.value)


def test_part_1_the_same_model_is_green_at_float32():
    """The control that makes the RED above a *precision* finding."""
    report = run_forward(_RaisingFFTModel, _inputs, "float32")
    assert report["dtypes"] == ["float32"]
    assert sum(report["n_nan"]) == 0


# ---------------------------------------------------------------------------
# Part 2 -- the compute dtype must reach the output
# ---------------------------------------------------------------------------
def test_part_2_rejects_a_float32_output_under_mixed_float16():
    with pytest.raises(AssertionError, match="silently opted its consumer out"):
        assert_precision_arm(build=_Float32LeakModel, make_inputs=_inputs)


def test_part_2_is_the_only_part_that_sees_the_float32_leak():
    """
    The leak model is finite, has a live backward and matches float32 exactly.
    Parts 1, 3 and 4 all pass over it -- which is why part 2 is not optional.
    """
    reports = assert_precision_arm(
        build=_Float32LeakModel,
        make_inputs=_inputs,
        expected_compute_dtype=None,
        rtol_against_float32=1e-2,
    )
    assert reports["mixed_float16"]["dtypes"] == ["float32"]
    assert reports["backward_mixed_float16"]["n_nonfinite"] == 0


# ---------------------------------------------------------------------------
# Part 3 -- finiteness
# ---------------------------------------------------------------------------
def test_float16_makes_the_1e_9_guard_exactly_zero():
    """The premise the void-guard subject rests on, measured rather than assumed."""
    assert np.float16(1e-9) == np.float16(0.0)
    assert np.float32(1e-9) != np.float32(0.0)


def test_part_3_rejects_a_guard_that_is_void_in_float16():
    with pytest.raises(AssertionError, match="mixed_float16 forward produced NaN"):
        assert_precision_arm(build=_VoidGuardModel, make_inputs=_inputs,
                             check_backward=False)


def test_part_3_the_void_guard_model_is_finite_at_float32():
    report = run_forward(_VoidGuardModel, _inputs, "float32")
    assert sum(report["n_nan"]) == 0, (
        "the float32 control must be clean, otherwise the NaN above is a "
        "property of the model rather than of the precision"
    )


# ---------------------------------------------------------------------------
# Part 4 -- the backward pass
# ---------------------------------------------------------------------------
def test_part_4_rejects_a_dead_backward_pass():
    with pytest.raises(AssertionError, match="None|reached no variable"):
        assert_precision_arm(build=_DeadBackwardModel, make_inputs=_inputs)


def test_part_4_the_dead_backward_model_passes_every_forward_part():
    """A forward-only arm calls the dead-backward model healthy."""
    reports = assert_precision_arm(
        build=_DeadBackwardModel,
        make_inputs=_inputs,
        check_backward=False,
        rtol_against_float32=1e-2,
    )
    assert reports["mixed_float16"]["dtypes"] == ["float16"]


# ---------------------------------------------------------------------------
# The self-calibrating cross-arm tolerance (D-055)
# ---------------------------------------------------------------------------
class _WrongUnderFloat16Model(keras.Model):
    """Finite float16 output, correct dtype, live backward -- wrong VALUE."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(4)

    def call(self, inputs, training=None):
        y = self.dense(inputs)
        if keras.mixed_precision.global_policy().name == "mixed_float16":
            y = y * 100.0
        return y


def test_the_cross_arm_tolerance_still_rejects_a_wrong_float16_value():
    """
    The D-055 control widens the tolerance by the model's own build spread. On
    a reproducible model that spread is 0.0, so the check must be exactly as
    strict as the flat rtol it replaced.
    """
    with pytest.raises(AssertionError, match="the arms disagree by"):
        assert_precision_arm(
            build=_WrongUnderFloat16Model,
            make_inputs=_inputs,
            rtol_against_float32=1e-2,
        )


def test_a_reproducible_model_has_a_float32_build_spread_of_exactly_zero():
    """The denominator of the control above -- stated, not assumed."""
    reports = assert_precision_arm(
        build=_HealthyModel, make_inputs=_inputs, rtol_against_float32=1e-2
    )
    assert reports["float32_build_spread"] == [0.0]


# ---------------------------------------------------------------------------
# The vacuity control itself
# ---------------------------------------------------------------------------
def test_a_model_built_outside_the_policy_is_rejected_not_passed():
    """
    The commonest way one of these arms silently becomes a float32 test: the
    model is constructed at module import, or in a fixture, and only the CALL
    happens inside the policy. The oracle must refuse that, not bless it.
    """
    prebuilt = _HealthyModel()
    prebuilt(_inputs())
    with pytest.raises(AssertionError, match="wearing an fp16 name"):
        assert_precision_arm(build=lambda: prebuilt, make_inputs=_inputs)


def test_the_policy_context_restores_even_when_the_body_raises():
    before = keras.mixed_precision.global_policy().name
    with pytest.raises(RuntimeError):
        with precision_policy("mixed_float16"):
            assert keras.mixed_precision.global_policy().name == "mixed_float16"
            raise RuntimeError("boom")
    assert keras.mixed_precision.global_policy().name == before


def test_flatten_tensors_handles_dicts_lists_and_none_values():
    x = ops.zeros((1, 2))
    assert len(flatten_tensors(x)) == 1
    assert len(flatten_tensors([x, x])) == 2
    assert len(flatten_tensors({"a": x, "b": None, "c": x})) == 2
