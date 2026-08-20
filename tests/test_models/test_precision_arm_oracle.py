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


# ---------------------------------------------------------------------------
# RED proofs for the step-18.1 additions (D-058, D-059, D-065 and `call_fn`).
#
# Each addition RELAXED or REDIRECTED an assertion, which is exactly the kind
# of change that quietly turns an instrument into a rubber stamp. Every one of
# them is therefore proved to still reject the defect it was carved around.
# ---------------------------------------------------------------------------
class _IntAndFloatModel(keras.Model):
    """Returns a float activation AND an integer mask -- the ``BERT`` shape."""

    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(3)

    def call(self, x, training=None):
        return {"logits": self.dense(x), "mask": ops.cast(x[:, :1] > 0, "int32")}


class _IntOnlyModel(keras.Model):
    """Returns ONLY integers -- part 2 must call itself vacuous, not pass."""

    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(3)

    def call(self, x, training=None):
        return ops.cast(self.dense(x) > 0, "int32")


class _IntAndLeakingFloatModel(keras.Model):
    """An integer mask beside a float32 LEAK -- the exemption must not hide it."""

    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(3)

    def call(self, x, training=None):
        return {
            "leak": ops.cast(self.dense(x), "float32"),
            "mask": ops.cast(x[:, :1] > 0, "int32"),
        }


class _UniformSoftmaxModel(keras.Model):
    """A zero-initialised softmax head: uniform output, the D-059 saddle."""

    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(
            4, kernel_initializer="zeros", bias_initializer="zeros",
            activation="softmax",
        )

    def call(self, x, training=None):
        return self.dense(x)


class _TwoArgumentModel(keras.Model):
    """``call(a, b, training)`` -- the ``TRM`` shape ``default_call`` cannot invoke."""

    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(3)

    def call(self, a, b, training=None):
        return self.dense(a) + self.dense(b)


class _TrainingModeOnlyFiniteModel(keras.Model):
    """Finite in training mode, overflowing at inference -- the ``yolo12`` shape."""

    def __init__(self):
        super().__init__()
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, x, training=None):
        # Two x300 amplifications with a BatchNorm after each -- the yolo12
        # shape in miniature. At INFERENCE an untrained BN is a near-identity,
        # so the product reaches 300 * 300 = 9e4 and overflows float16 (max
        # 65504) while float32 stays finite. In TRAINING mode each BN divides
        # by the BATCH statistics, so nothing ever exceeds O(1) in either
        # dtype. No single step overflows on its own; the compounding does.
        h = self.bn1(x * 300.0, training=training)
        return self.bn2(h * 300.0, training=training)


def test_part2_accepts_an_integer_output_beside_a_float16_one():
    """D-058: an int32 mask is not a failed compute dtype."""
    reports = assert_precision_arm(
        build=_IntAndFloatModel, make_inputs=_inputs, check_backward=True)
    assert reports["mixed_float16"]["dtypes"] == ["float16", "int32"]


def test_part2_still_rejects_a_float32_leak_beside_an_integer_output():
    """D-058 must not become "skip part 2 whenever an int is present"."""
    with pytest.raises(AssertionError, match="opted its consumer out"):
        assert_precision_arm(build=_IntAndLeakingFloatModel,
                             make_inputs=_inputs)


def test_part2_calls_itself_vacuous_on_an_all_integer_output():
    """The float-only rule must not degenerate into ``all([]) is True``."""
    with pytest.raises(AssertionError, match="no floating-point tensor"):
        assert_precision_arm(build=_IntOnlyModel, make_inputs=_inputs,
                             check_backward=False)


def test_the_dtype_exemption_is_scoped_to_the_named_index():
    """``dtype_exempt_outputs`` must exempt output 0 only when asked for 0."""
    # Index 0 is the leak; exempting it leaves nothing to judge -> vacuous.
    with pytest.raises(AssertionError, match="no floating-point tensor"):
        assert_precision_arm(build=_IntAndLeakingFloatModel,
                             make_inputs=_inputs, dtype_exempt_outputs=[0])
    # Exempting the WRONG index (the int mask) leaves the leak convicted.
    with pytest.raises(AssertionError, match="opted its consumer out"):
        assert_precision_arm(build=_IntAndLeakingFloatModel,
                             make_inputs=_inputs, dtype_exempt_outputs=[1])


def test_the_ramp_loss_is_what_makes_the_uniform_softmax_arm_non_vacuous():
    """D-059: the RED proof for the loss, not for a model.

    With a plain ``mean(square(.))`` the gradient of this model is EXACTLY
    zero by symmetry -- which is what made ``mobilenet`` and ``squeezenet``
    look dead. With the ramp it is not. Both halves are asserted so a revert
    to the symmetric loss fails here rather than silently returning the plan
    to a false finding.
    """
    import tensorflow as tf
    from .precision_arm_oracle import _asymmetric_loss

    with precision_policy("mixed_float16"):
        keras.utils.set_random_seed(0)
        model = _UniformSoftmaxModel()
        x = _inputs()
        norms = {}
        for label, lossfn in (
            ("symmetric", lambda t: ops.mean(ops.square(ops.cast(t, "float32")))),
            ("ramp", _asymmetric_loss),
        ):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                loss = lossfn(out)
            grads = tape.gradient(loss, model.trainable_variables)
            norms[label] = sum(
                float(ops.convert_to_numpy(
                    ops.sqrt(ops.sum(ops.square(ops.cast(g, "float32"))))))
                for g in grads if g is not None
            )
    assert norms["symmetric"] == 0.0, (
        "the symmetric loss no longer sits on the saddle this control pins; "
        f"measured {norms['symmetric']!r}"
    )
    assert norms["ramp"] > 0.0, (
        "the ramp loss reaches no variable -- D-059's fix is dead")

    # And the full arm passes on this model only because of the ramp.
    assert_precision_arm(build=_UniformSoftmaxModel, make_inputs=_inputs)


def test_call_fn_invokes_a_model_that_default_call_cannot():
    """The ``TRM`` shape: two positional arguments."""
    with pytest.raises(TypeError):
        assert_precision_arm(build=_TwoArgumentModel, make_inputs=_inputs)

    def two_arg_call(model, inputs, training):
        return model(inputs, inputs, training=training)

    reports = assert_precision_arm(build=_TwoArgumentModel,
                                   make_inputs=_inputs, call_fn=two_arg_call)
    assert reports["mixed_float16"]["dtypes"] == ["float16"]


def test_forward_training_moves_the_arm_off_the_untrained_inference_path():
    """D-065: ``forward_training`` must fix the yolo12 shape and nothing else."""
    with pytest.raises(AssertionError, match="forward produced (NaN|Inf)"):
        assert_precision_arm(build=_TrainingModeOnlyFiniteModel,
                             make_inputs=_inputs, check_backward=False)
    assert_precision_arm(build=_TrainingModeOnlyFiniteModel,
                         make_inputs=_inputs, check_backward=False,
                         forward_training=True)
    # It must NOT rescue a real dtype defect: the float32 control runs in the
    # same mode, so `_Float32LeakModel` is still convicted at `True`.
    with pytest.raises(AssertionError, match="opted its consumer out"):
        assert_precision_arm(build=_Float32LeakModel, make_inputs=_inputs,
                             forward_training=True)


# ===========================================================================
# RED proofs for the R-143 graph/XLA arm and the R-142 float64 arm.
#
# Both arms live in the same oracle and reuse the same helpers, so their
# proofs live here rather than in a third file. Each defect modelled below is a
# shape this step actually measured in the tree, not an invented one:
#
#   _EagerBranchModel     -- computes something DIFFERENT under tf.function.
#                            The whole point of R-143 ("an eager-only fix is
#                            not a fix"); this is the failure the 51-subject
#                            sweep would have to be able to see to mean
#                            anything.
#   _NonDeterministic     -- a stochastic forward. `vae`, `sd3_mmdit` and
#                            `relgt` all measure an eager-vs-XLA delta inside
#                            their own eager-vs-eager spread, and all three
#                            would convict without the control.
#   _DriftingIntModel     -- an exact-integer output that changes under XLA.
#                            R-143 asks for np.array_equal on these, not a
#                            tolerance.
# ===========================================================================

from .precision_arm_oracle import (  # noqa: E402
    assert_float64_arm,
    assert_xla_equivalence,
    float64_scope,
)


class _EagerBranchModel(keras.Model):
    """Returns a DIFFERENT function when traced. The R-143 defect, distilled."""

    def __init__(self):
        # An explicit name is load-bearing: Keras derives
        # "_eagerbranchmodel" from the leading underscore and TF rejects it as a
        # graph root scope, so the traced arm would raise ValueError before it
        # ever measured anything.
        super().__init__(name="eager_branch_model")
        self.d = keras.layers.Dense(3, kernel_initializer="ones",
                                    bias_initializer="zeros")

    def call(self, inputs, training=False):
        import tensorflow as tf
        y = self.d(inputs)
        return ops.where(
            ops.convert_to_tensor(tf.executing_eagerly()), y, y * 2.0)


class _DriftingIntModel(keras.Model):
    """An int output that is off by one under tracing."""

    def __init__(self):
        # An explicit name is load-bearing: Keras derives
        # "_driftingintmodel" from the leading underscore and TF rejects it as a
        # graph root scope, so the traced arm would raise ValueError before it
        # ever measured anything.
        super().__init__(name="drifting_int_model")
        self.d = keras.layers.Dense(3)

    def call(self, inputs, training=False):
        import tensorflow as tf
        y = self.d(inputs)
        bump = ops.cast(ops.logical_not(
            ops.convert_to_tensor(tf.executing_eagerly())), "int32")
        return [y, ops.zeros((1, 3), "int32") + bump]


class _NonDeterministic(keras.Model):
    """A stochastic forward at ``training=False`` -- the `vae` / `relgt` shape."""

    def __init__(self):
        # An explicit name is load-bearing: Keras derives
        # "_nondeterministic" from the leading underscore and TF rejects it as a
        # graph root scope, so the traced arm would raise ValueError before it
        # ever measured anything.
        super().__init__(name="non_deterministic_model")
        self.d = keras.layers.Dense(3)

    def call(self, inputs, training=False):
        return self.d(inputs) + keras.random.normal(ops.shape(inputs)[:1] + (3,))


def _x4():
    return np.random.RandomState(0).randn(2, 4).astype("float32")


def test_the_xla_arm_rejects_a_model_that_computes_a_different_graph():
    with pytest.raises(AssertionError, match="eager-vs-XLA max"):
        assert_xla_equivalence(build=_EagerBranchModel, make_inputs=_x4)


def test_the_xla_arm_accepts_the_healthy_twin():
    """Same shape, same size, no branch -- so the failure above is the BRANCH."""
    report = assert_xla_equivalence(
        build=lambda: keras.Sequential([keras.layers.Input((4,)),
                                        keras.layers.Dense(3)]),
        make_inputs=_x4)
    assert report["eager_spread"] == [0.0]


def test_the_xla_arm_requires_exact_equality_on_an_integer_output():
    """An off-by-one int output must fail, and it must fail on the INT rule --
    a tolerance would have swallowed a delta of 1 on a tensor of zeros."""
    with pytest.raises(AssertionError, match="must be\n?\\s*bit-identical|bit-identical"):
        assert_xla_equivalence(build=_DriftingIntModel, make_inputs=_x4)


def test_the_eager_spread_control_keeps_a_stochastic_model_from_convicting():
    """The reading that reversed three models.

    Both halves are asserted: the spread must be non-zero (otherwise this
    subject is not stochastic and proves nothing about the control), and the
    arm must pass. Delete the ``3 * spread`` term and this test fails.
    """
    report = assert_xla_equivalence(build=_NonDeterministic, make_inputs=_x4,
                                    rtol=1e-6)
    assert report["eager_spread"][0] > 0.0, (
        "this subject is deterministic, so it cannot demonstrate the control")


class _UntraceableModel(keras.Model):
    """Raises under ANY tracing -- the literal ``SAM.postprocess_masks`` shape.

    ``len()`` on a symbolic tensor is exactly what ``ops.image.resize`` does to
    ``SAM``'s tensor ``original_size`` (``SAM/SAM1/model.py:634``), and it is
    why that model fails identically at ``jit_compile=False`` and ``True``.
    """

    def __init__(self):
        super().__init__(name="untraceable_model")
        self.d = keras.layers.Dense(3)

    def call(self, inputs, training=False):
        _ = len(inputs)
        return self.d(inputs)


def test_the_raises_arm_will_not_accept_a_different_failure():
    """``expect='raises'`` keyed on the WRONG reason must still fail.

    Without the ``expect_reason`` check, a documented opt-out would be
    satisfied by any exception at all -- including one the test itself
    introduced -- and would go on passing after the real blocker was fixed.
    This subject raises, but on ``len is not well defined``, not on the
    ``ResizeBicubic`` the caller documented.
    """
    with pytest.raises(AssertionError, match="not for the documented reason"):
        assert_xla_equivalence(build=_UntraceableModel, make_inputs=_x4,
                               expect="raises",
                               expect_reason="ResizeBicubic")


def test_the_raises_arm_accepts_the_reason_it_was_given():
    """The GREEN twin of the test above -- same model, correct reason."""
    report = assert_xla_equivalence(build=_UntraceableModel, make_inputs=_x4,
                                    expect="raises",
                                    expect_reason="len is not well defined")
    assert report["exception"] == "TypeError"


def test_the_raises_arm_fails_when_the_model_actually_compiles():
    with pytest.raises(AssertionError, match="it just\n?\\s*compiled|just compiled"):
        assert_xla_equivalence(
            build=lambda: keras.Sequential([keras.layers.Input((4,)),
                                            keras.layers.Dense(3)]),
            make_inputs=_x4, expect="raises", expect_reason="ResizeBicubic")


def test_the_raises_arm_refuses_to_run_without_a_reason():
    with pytest.raises(AssertionError, match="would pass on ANY"):
        assert_xla_equivalence(build=_EagerBranchModel, make_inputs=_x4,
                               expect="raises")


class _PinnedFloat32(keras.Model):
    """Casts its output to float32 -- the R-142 defect the arm exists to see."""

    def __init__(self):
        super().__init__()
        self.d = keras.layers.Dense(3)

    def call(self, inputs, training=False):
        return ops.cast(self.d(inputs), "float32")


def test_the_float64_arm_rejects_a_pinned_float32_output():
    with pytest.raises(AssertionError, match="float outputs are"):
        assert_float64_arm(build=_PinnedFloat32, make_inputs=_x4)


def test_the_float64_arm_accepts_the_healthy_twin_and_casts_its_input():
    report = assert_float64_arm(
        build=lambda: keras.Sequential([keras.layers.Input((4,)),
                                        keras.layers.Dense(3)]),
        make_inputs=_x4)
    assert report["input_dtypes"] == ["float64"], (
        "the arm did not cast its float input, so its inputs[0].dtype "
        "assertion was asserting the caller's own choice")
    assert report["dtypes"] == ["float64"]


def test_the_float64_arm_rejects_a_model_built_outside_the_scope():
    """The most common way one of these arms silently becomes a float32 test."""
    prebuilt = keras.Sequential([keras.layers.Input((4,)),
                                 keras.layers.Dense(3)])
    with pytest.raises(AssertionError, match="dtype_policy"):
        assert_float64_arm(build=lambda: prebuilt, make_inputs=_x4)


def test_the_float64_scope_restores_both_switches_even_when_the_body_raises():
    before = (keras.backend.floatx(),
              keras.mixed_precision.global_policy().name)
    with pytest.raises(RuntimeError):
        with float64_scope():
            raise RuntimeError("boom")
    assert (keras.backend.floatx(),
            keras.mixed_precision.global_policy().name) == before
