"""Single-claim guard: `RoutingProbabilitiesLayer` runs its tree at a never-narrow dtype FLOOR.

The claim under test, guide rule L-30: `call()` widens the decision logits (and
the three tree operands that share their arithmetic) to `max(incoming_dtype,
float32)`. Never to an absolute `float32`, and never to nothing.

That single sentence has TWO independent failure directions, so this module
carries TWO independent guards. Each one is RED against the mutation its own
docstring names and GREEN against the other's -- they are not two spellings of
one assertion:

===========================  ==========================================  =========
mutation applied to `call()`  effect                                      RED guard
===========================  ==========================================  =========
`tree_dtype = "float32"`      floor becomes an absolute target; a float64  A
(the pre-fix form)            policy is silently narrowed to float32
                              precision and buys nothing

`tree_dtype = incoming_dtype` widening removed; under fp16 the upper clip   B
(no floor at all)             `1 - 1e-7` rounds to exactly 1.0, so a
                              saturated sigmoid leaves
                              `p_go_left = 1 - 1.0 = 0.0` and a whole
                              subtree is zeroed
===========================  ==========================================  =========

**Why a separate file rather than `test_routing_probabilities.py`.** Both guards
mutate PROCESS-GLOBAL state -- the mixed-precision policy, and for guard A
`keras.backend.floatx()` as well. `tests/test_layers/conftest.py` says in its own
docstring why that restore belongs in exactly one narrow place; its `dtype_policy`
fixture deliberately does NOT touch `floatx`, which guard A requires (without it a
"float64" arm silently agrees with float32 to eight digits and the guard cannot
fail). Keeping these two in their own module keeps the extra `floatx` teardown
away from the several hundred default-policy parametrizations next door.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.activations.routing_probabilities import (
    RoutingProbabilitiesLayer,
)

# --------------------------------------------------------------------------- #
# Shared fixture geometry.
#
# `output_dim=8` is `2 ** 3`, so `padded_output_dim == output_dim`: every leaf of
# the depth-3 tree is a real class, the structural masks are all-ones/all-zeros,
# and the reference below is the plain product form with no mask bookkeeping.
# That is what makes the reference derivable from the class docstring's
# "Architecture Overview" alone rather than from the implementation.
# --------------------------------------------------------------------------- #
INPUT_DIM = 16
OUTPUT_DIM = 8
NUM_DECISIONS = 3
BATCH = 64

# Guard A's tolerance, derived rather than tuned.
#
# The reference and the layer perform the same ~10 float64 roundings per leaf
# (one matmul of length 16, a sigmoid, a clip, 3 multiplications, a sum of 8
# terms and a divide) on values bounded by 1. Each contributes at most one
# float64 ulp, 2.220e-16, so agreement no worse than ~1e-15 is expected;
# 1e-12 leaves three orders of headroom for a backend sigmoid that differs from
# NumPy's by a few ulp. The number that matters is the GAP to the failure it
# must catch: a tree that ran in float32 disagrees with a float64 reference by
# ~1e-07, five orders ABOVE this bound. Any value in [1e-14, 1e-09] would grade
# these two states identically; this is not a knife-edge threshold.
FLOAT64_AGREEMENT_ATOL = 1e-12

# Guard A's second, coarser criterion, quoted from the plan's SC1: the sum-to-one
# residual under a float64 policy. Pre-fix measurement 8.381903e-08 (which is
# float32's floor, the whole point), post-fix 2.220446e-16.
FLOAT64_ROWSUM_ATOL = 1e-13


@pytest.fixture
def float64_policy():
    """Force a genuine float64 policy for one test, then ALWAYS restore both globals.

    Sets `floatx` IN ADDITION to the policy. The policy alone leaves
    `keras.Input`/`convert_to_tensor` producing float32, so the arm agrees with
    float32 to eight digits and a float64 precision guard becomes unfailable.

    :yield: the literal string ``'float64'``.
    :rtype: str
    """
    previous_policy = keras.mixed_precision.global_policy().name
    previous_floatx = keras.backend.floatx()
    keras.backend.set_floatx("float64")
    keras.mixed_precision.set_global_policy("float64")
    try:
        yield "float64"
    finally:
        keras.mixed_precision.set_global_policy(previous_policy)
        keras.backend.set_floatx(previous_floatx)


def _stable_sigmoid(z):
    """Overflow-free logistic sigmoid on float64 input.

    The naive ``1/(1+exp(-z))`` overflows in ``exp`` at ``z`` around -710, which
    guard B reaches on purpose (it feeds a deliberately saturating input). The
    branch below evaluates the algebraically identical form whose exponent
    argument is always non-positive.

    :param z: real-valued array.
    :return: elementwise sigmoid, float64.
    :rtype: numpy.ndarray
    """
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _reference_leaf_probabilities(x, kernel, bias, epsilon, normalize=True):
    """Hand-derived float64 reference for the routing tree. No layer internals used.

    Straight from the class docstring's Architecture Overview: project, sigmoid,
    clip, then walk the binary tree. `keras.ops.stack([left, right], axis=2)`
    followed by a reshape means level 0 contributes the MOST significant bit of
    the leaf index, so leaf ``j`` takes the right branch at level ``i`` exactly
    when bit ``NUM_DECISIONS-1-i`` of ``j`` is set.

    :param x: inputs, shape ``(batch, input_dim)``, float64.
    :param kernel: projection matrix, shape ``(input_dim, num_decisions)``.
    :param bias: bias of shape ``(num_decisions,)``, or ``None``.
    :param epsilon: the sigmoid clip half-width.
    :param normalize: whether to divide by the row sum, as the layer does.
    :return: leaf probabilities, shape ``(batch, 2 ** num_decisions)``, float64.
    :rtype: numpy.ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    z = x @ kernel
    if bias is not None:
        z = z + np.asarray(bias, dtype=np.float64)

    p_right = _stable_sigmoid(z)
    p_right = np.clip(p_right, epsilon, 1.0 - epsilon)

    depth = kernel.shape[1]
    leaves = np.ones((x.shape[0], 2 ** depth), dtype=np.float64)
    for j in range(2 ** depth):
        for i in range(depth):
            goes_right = bool((j >> (depth - 1 - i)) & 1)
            leaves[:, j] *= p_right[:, i] if goes_right else 1.0 - p_right[:, i]

    if normalize:
        leaves = leaves / np.maximum(leaves.sum(axis=1, keepdims=True), 1e-7)
    return leaves


class TestTheDtypeFloorNeverNarrows:
    """The two directions of the L-30 floor, one guard each."""

    def test_float64_policy_actually_buys_float64_precision(self, float64_policy):
        """GUARD A. RED when the floor is replaced by an absolute `tree_dtype = "float32"`.

        This is the exact pre-fix form. It is invisible to every other test in
        this directory: the output dtype is still float64, nothing raises, and
        the sum-to-one invariant still holds to float32's ~1e-07 -- which is
        precisely the precision a float64 caller asked NOT to be given.
        """
        keras.utils.set_random_seed(0)
        x_np = np.random.default_rng(0).standard_normal(
            (BATCH, INPUT_DIM)
        ).astype("float64")
        x = keras.ops.convert_to_tensor(x_np)

        # Premise: without `set_floatx` this reads float32 and the whole guard
        # degenerates into a float32-vs-float32 comparison that cannot fail.
        assert keras.backend.standardize_dtype(x.dtype) == "float64", (
            "premise violated: the realised input dtype is not float64, so this "
            "guard is measuring the float32 path and cannot detect narrowing"
        )

        layer = RoutingProbabilitiesLayer(output_dim=OUTPUT_DIM, mode="trainable")
        y = layer(x)
        y_np = np.asarray(keras.ops.convert_to_numpy(y))

        assert keras.backend.standardize_dtype(y.dtype) == "float64"
        assert layer.num_decisions == NUM_DECISIONS, (
            f"geometry premise moved: num_decisions={layer.num_decisions}, the "
            f"reference below is written for {NUM_DECISIONS}"
        )
        assert layer.output_dim == layer.padded_output_dim, (
            "geometry premise moved: output_dim is no longer a power of two, so "
            "the mask-free reference below no longer describes this tree"
        )

        expected = _reference_leaf_probabilities(
            x_np,
            keras.ops.convert_to_numpy(layer.kernel),
            None if layer.bias is None else keras.ops.convert_to_numpy(layer.bias),
            layer.epsilon,
            normalize=layer.normalize,
        )
        max_dev = float(np.max(np.abs(y_np - expected)))
        assert max_dev < FLOAT64_AGREEMENT_ATOL, (
            f"the tree did NOT run in float64: max deviation from an independent "
            f"float64 reference is {max_dev:.6e}, above {FLOAT64_AGREEMENT_ATOL:.1e}. "
            f"A deviation near 1e-07 means the decision logits, the masks or the "
            f"root mass were cast to an absolute float32 instead of the "
            f"max(incoming, float32) floor -- the L-30 defect."
        )

        rowsum_err = float(np.max(np.abs(y_np.sum(axis=-1) - 1.0)))
        assert rowsum_err < FLOAT64_ROWSUM_ATOL, (
            f"max|rowsum-1| is {rowsum_err:.6e} under a float64 policy, above "
            f"{FLOAT64_ROWSUM_ATOL:.1e}. The pre-fix reading was 8.381903e-08, "
            f"float32's floor: the float64 policy bought nothing."
        )

    def test_fp16_widening_still_protects_the_saturated_subtree(
        self, mixed_float16_policy
    ):
        """GUARD B. RED when the widening is deleted, i.e. `tree_dtype = incoming_dtype`.

        The hazard is exact and measured: `np.float16(1 - 1e-7)` IS `1.0`, so in
        half precision the upper clip is a no-op. A saturated sigmoid then gives
        `p_go_right = 1.0` and `p_go_left = 1.0 - 1.0 = 0.0`, and every leaf under
        that node collects exactly zero mass.

        This guard exists so the float64 fix cannot be "achieved" by simply
        deleting the cast.
        """
        keras.utils.set_random_seed(0)
        # Scale 1e3 so the projection saturates the sigmoid. Well inside fp16's
        # 65504 ceiling -- an input that overflows is itself non-finite and
        # produces a false all-NaN reading (a near-miss recorded in this plan's
        # findings), which is why the finiteness assert below is on the INPUT.
        x_np = (
            np.random.default_rng(0).standard_normal((BATCH, INPUT_DIM)) * 1e3
        ).astype("float16")
        assert np.all(np.isfinite(x_np)), (
            "premise violated: the fp16 INPUT is already non-finite, so any "
            "NaN in the output would be garbage-in-garbage-out, not a defect"
        )
        x = keras.ops.convert_to_tensor(x_np)
        assert keras.backend.standardize_dtype(x.dtype) == "float16"

        layer = RoutingProbabilitiesLayer(output_dim=OUTPUT_DIM, mode="trainable")
        y = layer(x)
        y_np = np.asarray(keras.ops.convert_to_numpy(y))

        # NON-VACUITY PREMISE. Without this the guard would pass for free on an
        # unsaturated input, in both directions. Computed from the layer's
        # weights in NumPy, independently of whatever dtype `call()` chose: at
        # least one decision must round to EXACTLY 1.0 in half precision, which
        # is what makes `1 - p` exactly zero when the tree runs in fp16.
        z = (
            np.asarray(x_np, dtype=np.float32)
            @ np.asarray(keras.ops.convert_to_numpy(layer.kernel), dtype=np.float32)
        )
        if layer.bias is not None:
            z = z + np.asarray(keras.ops.convert_to_numpy(layer.bias), dtype=np.float32)
        p_fp16 = np.float16(_stable_sigmoid(z))
        n_saturated = int(np.sum(p_fp16 == np.float16(1.0)))
        assert n_saturated > 0, (
            "premise violated: no decision saturates to exactly 1.0 in fp16 on "
            "this fixture, so a deleted widening could not zero any subtree and "
            "this guard would pass vacuously"
        )

        n_zero = int(np.sum(y_np == 0.0))
        assert n_zero == 0, (
            f"{n_zero} of {y_np.size} leaf-mass entries are EXACTLY 0.0 with "
            f"{n_saturated} fp16-saturated decisions present. The tree ran in "
            f"float16: the upper clip 1 - {layer.epsilon} rounds to exactly 1.0 "
            f"there, so p_go_left became exactly 0 and a whole subtree was "
            f"zeroed. The max(incoming, float32) widening is missing."
        )
        assert np.all(np.isfinite(y_np)), "fp16 path produced non-finite output"

        rowsum_err = float(np.max(np.abs(y_np.astype(np.float64).sum(axis=-1) - 1.0)))
        assert rowsum_err < 1e-05, (
            f"max|rowsum-1| is {rowsum_err:.6e} under mixed_float16; the tree "
            f"accumulation is no longer running at the float32 floor"
        )
