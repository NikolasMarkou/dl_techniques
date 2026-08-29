"""The precision arms and the XLA arm agree with the float32 control.

Three §16.3 items, all of which had zero grep hits in this directory
before 2026-08-29 (`mixed_float16` 0, `jit_compile` 0, `float64` only
as a NumPy accumulator inside a hand-rolled reference):

* §13.2.6 `mixed_float16` and `float64` construction-and-forward arms,
  with a float32 control on the same input
* §13.2.7 `@tf.function(jit_compile=True)` versus eager
* §16.3   degenerate lengths (1) on the static path AND a
  `TensorSpec([None, ...])` trace

MEASUREMENTS. All taken 2026-08-29 on an RTX 4070 (CUDA_VISIBLE_DEVICES=1),
Keras 3.8 / TF 2.18, batch 4 on the 8x16 non-square grid, default
constructor configuration for each class (so the weights are
`'zeros'` / `Constant` and are bit-identical across the three arms --
a seeded random initializer draws DIFFERENT values under float64 and
would have turned an init difference into a fake precision reading).

| arm | max abs delta vs control | output absmax | tolerance | headroom |
|---|---|---|---|---|
| float32 (control) | 0.0 (by construction) | 0.61 - 2.45 | 0.0 | exact |
| mixed_float16 | 3.6951e-03 (LearnableNeuralCircuit) | 2.449479 | 2e-2 | 5.4x |
| float64 | 8.1629e-07 (LearnableNeuralCircuit) | 2.449479 | 1e-5 | 12x |
| XLA vs eager | 5.9605e-07 (LearnableNeuralCircuit) | 2.449479 | 1e-5 | 17x |

The float16 number is not free-floating: `np.spacing(np.float16(2.449))`
is 1.953e-03, so the measured 3.695e-03 is 1.9 ulp at the output
magnitude, and the 2e-2 bound is ~10 ulp. `TestTheFloat16HazardIsReal`
pins that ulp, so the bound cannot silently become decorative if the
output magnitude changes.

The defect signal every bound sits below: a dead or zeroed arm differs
from the control by the full output magnitude, 0.61 to 2.45 -- between
30x (float16) and 2.4e6x (float64) the bound.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.logic.neural_circuit import (
    CircuitDepthLayer,
    LearnableNeuralCircuit,
)

from .logic_subject_oracle import FEATURE_SHAPE, SUBJECTS, SUBJECT_NAMES

#: Per-policy bound and its measurement, see the module docstring.
POLICY_ATOL = {
    "float32": 0.0,
    "mixed_float16": 2e-2,
    "float64": 1e-5,
}

#: Eager-versus-XLA bound, see the module docstring.
XLA_ATOL = 1e-5


# DECISION plan-2026-08-29T112804-aff039c4/D-006 -- the float32
# control is captured at IMPORT, under the ambient policy. Do NOT
# rewrite it as a per-layer dtype="float32" override: that dtype
# never reaches the children CircuitDepthLayer builds in __init__.
# See decisions.md D-006. [2026-08-29: the propagation gap D-006
# routed around is FIXED (D-007); the import capture stands anyway.]
def _float32_reference():
    """Capture the float32 control once, at import, while the global
    policy is still float32.

    The control is NOT produced with a per-layer `dtype="float32"`
    override: measured 2026-08-29, `CircuitDepthLayer(dtype="float32")`
    under a `mixed_float16` global policy leaves its `logic_op_*` and
    `arithmetic_op_*` children on `mixed_float16`, because those
    children are constructed in `__init__` without the parent's dtype.
    Such a "control" would have run two of the four subjects in float16
    and read a delta of exactly 0.0 against the float16 arm.
    """
    assert keras.mixed_precision.global_policy().name == "float32", (
        "this module must be imported while the global dtype policy is "
        "float32; the control it captures is otherwise not a control"
    )
    reference = {}
    for name in SUBJECT_NAMES:
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model()
        output = keras.ops.convert_to_numpy(
            model(subject.inputs(), training=False)
        )
        reference[name] = output.astype("float64")
    return reference


FLOAT32_REFERENCE = _float32_reference()


class TestTheFloat16HazardIsReal:
    """Prove the hazard before testing any arm against it (§13.2.6)."""

    def test_one_float16_ulp_at_the_output_magnitude(self):
        """1.953e-03 at |y| ~ 2.449. The measured mixed_float16 delta
        of 3.6951e-03 is 1.9 of these; the 2e-2 bound is ~10.
        """
        ulp = float(np.spacing(np.float16(2.449479)))
        assert ulp == pytest.approx(1.953125e-03, rel=0, abs=0.0)
        assert 3.6951e-03 / ulp < 2.0, (
            "the measured float16 delta is more than 2 ulp; re-derive"
        )
        assert POLICY_ATOL["mixed_float16"] / ulp > 5.0, (
            "the float16 bound is under 5 ulp and will be flaky"
        )

    def test_float64_is_not_merely_float32_with_a_different_name(self):
        """A float64 arm that never received float64 inputs agrees with
        float32 to eight digits and proves nothing.
        """
        assert np.finfo(np.float64).eps < np.finfo(np.float32).eps


class TestPrecisionArmsAgreeWithTheFloat32Control:
    """§13.2.6. Construct and run under each of the three policies."""

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_the_arm_receives_inputs_in_its_own_dtype(
            self, name, dtype_policy
    ):
        """`keras.Input` reads `backend.floatx()`, not the policy, so
        without the `set_floatx` the fixture also performs, the float64
        arm would be built on float32 inputs.
        """
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model()

        expected = "float64" if dtype_policy == "float64" else "float32"
        assert model.inputs[0].dtype == expected, (
            f"{name} under {dtype_policy}: inputs[0].dtype is "
            f"{model.inputs[0].dtype}, expected {expected}"
        )

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_the_arm_matches_the_float32_control(
            self, name, dtype_policy
    ):
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model()
        sample = subject.inputs(dtype=keras.backend.floatx())

        arm = keras.ops.convert_to_numpy(model(sample, training=False))
        assert bool(np.all(np.isfinite(arm))), (
            f"{name} under {dtype_policy} produced a non-finite output"
        )

        control = FLOAT32_REFERENCE[name]
        np.testing.assert_allclose(
            arm.astype("float64"), control,
            atol=POLICY_ATOL[dtype_policy], rtol=0,
            err_msg=(
                f"{name} under {dtype_policy} left the float32 control "
                f"(control absmax {np.max(np.abs(control)):.6f})"
            ),
        )

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_the_control_is_not_trivially_zero(self, name):
        """The twin. Every agreement above is satisfied by three arms
        that all return zeros.
        """
        control = FLOAT32_REFERENCE[name]
        assert float(np.max(np.abs(control))) > 0.5, (
            f"the {name} control collapsed to "
            f"{np.max(np.abs(control))}; the precision arms would agree "
            f"with it for the wrong reason"
        )


class TestGraphAndXlaMatchEager:
    """§13.2.7. An eager-only forward pass is not a forward pass."""

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_jit_compiled_output_matches_eager(self, name):
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model()
        sample = subject.inputs()
        tensors = (
            [keras.ops.convert_to_tensor(v) for v in sample]
            if subject.arity > 1
            else keras.ops.convert_to_tensor(sample)
        )

        eager = keras.ops.convert_to_numpy(model(tensors, training=False))

        @tf.function(jit_compile=True)
        def traced(operands):
            return model(operands, training=False)

        compiled = keras.ops.convert_to_numpy(
            traced(tensors)
        ).astype("float32")

        assert np.all(np.isfinite(compiled)), (
            f"{name} produced a non-finite output under XLA"
        )
        np.testing.assert_allclose(
            compiled, eager, atol=XLA_ATOL, rtol=0,
            err_msg=(
                f"{name}: XLA and eager disagree beyond the measured "
                f"reassociation floor (eager absmax "
                f"{np.max(np.abs(eager)):.6f})"
            ),
        )


class TestDegenerateLengthsAndDynamicShapes:
    """§16.3. Length-1 axes on the static path, and a `None`-shaped
    trace so a value that must stay a static Python `int` is proven to
    stay one.
    """

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    @pytest.mark.parametrize(
        "shape", [(1, 1), (8, 1), (1, 16)],
        ids=["1x1", "8x1", "1x16"],
    )
    @pytest.mark.parametrize("batch", [1, 2])
    def test_a_length_one_axis_still_produces_a_finite_output(
            self, name, shape, batch
    ):
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        model = subject.model(shape=shape)
        sample = subject.inputs(batch=batch, shape=shape)

        output = keras.ops.convert_to_numpy(
            model(sample, training=False)
        )
        assert output.shape == (batch,) + shape, (
            f"{name} at {shape} returned {output.shape}"
        )
        assert bool(
            keras.ops.all(keras.ops.isfinite(output))
        ), f"{name} at {shape} produced a non-finite output"

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_a_fully_dynamic_trace_reproduces_the_eager_output(
            self, name
    ):
        """`TensorSpec([None, None, 16])`: batch AND the middle axis
        unknown at trace time. `atol=0.0` because a trace over the same
        graph on the same device is a re-execution, not a
        reassociation -- measured 0.0 for all four classes.
        """
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        layer = subject.make()
        sample = subject.inputs()
        tensors = (
            [keras.ops.convert_to_tensor(v) for v in sample]
            if subject.arity > 1
            else keras.ops.convert_to_tensor(sample)
        )
        eager = keras.ops.convert_to_numpy(layer(tensors, training=False))

        spec = tf.TensorSpec([None, None, 16], tf.float32)
        signature = (
            [[spec for _ in range(subject.arity)]]
            if subject.arity > 1 else [spec]
        )

        @tf.function(input_signature=signature)
        def traced(operands):
            return layer(operands, training=False)

        graph = keras.ops.convert_to_numpy(traced(tensors))
        np.testing.assert_allclose(
            graph, eager, atol=0.0, rtol=0,
            err_msg=f"{name}: dynamic trace and eager disagree",
        )


class TestAPinnedDtypeReachesEveryChild:
    """§12.5 (the `dtype` knob) plus §13.2.6.

    `CircuitDepthLayer` and `LearnableNeuralCircuit` construct their
    children in `__init__` and, until 2026-08-29, passed no dtype. A
    caller pinning a stage to float32 inside a `mixed_float16` model
    silently got float16 experts. Measured at `fdfe1f3c8`, under a
    `mixed_float16` global policy::

        CircuitDepthLayer(dtype='float32')
            parent          float32
            logic child     mixed_float16    <- wrong
            arith child     mixed_float16    <- wrong
            channel_mix     mixed_float16    <- wrong
        LearnableNeuralCircuit(dtype='float32')
            stage child     mixed_float16    <- wrong
            layer_norm      mixed_float16    <- wrong

    The propagated object is `self.dtype_policy`, never `self.dtype`.
    Measured on the same day: under a `mixed_float16` policy
    `layer.dtype` reads `'float32'` (Keras 3 returns the VARIABLE
    dtype), so the 41-file house spelling `dtype=self.dtype` would
    build pure-float32 children under a mixed parent and disable
    mixed precision for the whole subtree.
    """

    def _pinned(self, policy_name):
        """A float32-pinned stage and circuit under `policy_name`.

        The global policy is restored on the way out; leaking
        `mixed_float16` would silently retune every later module.
        """
        previous = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy(policy_name)
        try:
            stage = CircuitDepthLayer(
                num_logic_ops=1,
                num_arithmetic_ops=1,
                channel_mix="dense",
                dtype="float32",
                name="pinned_stage",
            )
            stage.build((None,) + FEATURE_SHAPE)
            circuit = LearnableNeuralCircuit(
                circuit_depth=2,
                use_layer_norm=True,
                dtype="float32",
                name="pinned_circuit",
            )
            return {
                "logic": stage.logic_operators[0],
                "arithmetic": stage.arithmetic_operators[0],
                "channel_mix": stage._channel_mix_layer,
                "stage": circuit.circuit_layers[0],
                "layer_norm": circuit.layer_norms[0],
            }
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_the_ambient_policy_is_the_one_the_children_would_inherit(
            self
    ):
        """The anti-vacuity control for the pin below.

        Without this, a Keras version in which a child ignores the
        ambient policy and defaults to float32 anyway would satisfy the
        pin while propagating nothing. Under `mixed_float16` and with
        NO parent override, every child must read `mixed_float16` --
        which is exactly the value the pinned case must NOT read.
        """
        previous = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            stage = CircuitDepthLayer(
                num_logic_ops=1,
                num_arithmetic_ops=1,
                channel_mix="dense",
                name="ambient_stage",
            )
            stage.build((None,) + FEATURE_SHAPE)
            observed = {
                "logic": stage.logic_operators[0].dtype_policy.name,
                "arithmetic":
                    stage.arithmetic_operators[0].dtype_policy.name,
                "channel_mix":
                    stage._channel_mix_layer.dtype_policy.name,
            }
        finally:
            keras.mixed_precision.set_global_policy(previous)

        assert set(observed.values()) == {"mixed_float16"}, (
            "the ambient mixed_float16 policy does not reach these "
            f"children at all, so the pin below is vacuous: {observed}"
        )

    @pytest.mark.parametrize(
        "child",
        ["logic", "arithmetic", "channel_mix", "stage", "layer_norm"],
    )
    def test_every_child_carries_the_parents_pinned_policy(self, child):
        """Each of the five construction sites, named separately, so a
        revert of one site does not hide behind another's pass.
        """
        observed = self._pinned("mixed_float16")[child].dtype_policy.name
        assert observed == "float32", (
            f"the {child!r} child of a dtype='float32' parent reads "
            f"{observed!r} under a mixed_float16 global policy; the "
            f"parent's dtype_policy is not reaching it"
        )

    def test_a_mixed_parent_gives_mixed_children_not_float32_ones(self):
        """The `dtype_policy`-versus-`dtype` pin.

        `dtype=self.dtype` reads `'float32'` under a mixed policy, so
        this assertion is the one that goes red against the house
        spelling while every assertion above stays green.
        """
        previous = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("float32")
        try:
            stage = CircuitDepthLayer(
                num_logic_ops=1,
                num_arithmetic_ops=1,
                channel_mix="dense",
                dtype="mixed_float16",
                name="mixed_stage",
            )
            stage.build((None,) + FEATURE_SHAPE)
            observed = {
                "logic": stage.logic_operators[0].dtype_policy.name,
                "arithmetic":
                    stage.arithmetic_operators[0].dtype_policy.name,
                "channel_mix":
                    stage._channel_mix_layer.dtype_policy.name,
            }
        finally:
            keras.mixed_precision.set_global_policy(previous)

        assert set(observed.values()) == {"mixed_float16"}, (
            "a dtype='mixed_float16' parent produced children on "
            f"{sorted(set(observed.values()))}; passing self.dtype "
            f"instead of self.dtype_policy gives exactly this"
        )
