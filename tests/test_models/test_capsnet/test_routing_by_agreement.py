"""CapsNet's central claim: dynamic ROUTING-BY-AGREEMENT, not an unweighted mean.

Why this file exists
--------------------
`RoutingCapsule.call` runs the Sabour/Frosst/Hinton (2017) recursion: couplings
`c = softmax(b, axis=parents)`, parent input `s_j = sum_i c_ij u_hat_ij`, output
`v_j = squash(s_j)`, then `b_ij += <u_hat_ij, v_j>`. Setting
`routing_iterations=1` leaves `b` at zero, so `c` is uniform and the layer
degenerates to an UNWEIGHTED MEAN of the predictions -- the whole algorithm
removed, with the same weights, the same shapes and the same output ranks.

Before this file, that substitution passed all 60 tests in
`tests/test_models/test_capsnet/` and all of `tests/test_layers/test_capsules.py`
(which contains no numeric comparison at all). `routing_iterations` was checked
only by a stored-attribute echo (`assert model.routing_iterations == n`).

The instrument
--------------
The predictions `u_hat` are set DIRECTLY rather than left to random `W`: with
every input capsule fed the unit vector `e_0`, `u_hat[i, j] = W[0, i, j, :, 0]`,
so assigning `W` chooses each prediction exactly. Two parents are then built:

* parent 0 -- all four input capsules predict the SAME vector (total agreement)
* parent 1 -- the four input capsules predict four MUTUALLY ORTHOGONAL vectors
  (total disagreement), with identical prediction norms

Agreement must win the couplings. MEASURED 2026-08-18 (deterministic, no RNG in
the result -- `W` is fully assigned):

    iterations  |v_agree|  |v_scatter|   ratio
             1   0.941176     0.800000     1.18   <- unweighted mean
             2   0.972757     0.506085     1.92
             3   0.982091     0.081250    12.09
             5   0.984563     0.000047    21123
            10   0.984615     0.000000     6.5e15

The r=1 row IS the dead-component injection: it is what the layer computes when
routing is removed, and every assertion below is checked against it in the same
test, so "the guard would have passed anyway" is not available.

Random `W` would NOT support this claim: at random initialization the sharpening
is real but tiny and not even monotone (measured max/min output-length ratio
over seeds 7/11/42 at r=1 vs r=10: 9.70->11.09, 1.289->1.283, 5.86->6.09 -- the
middle one moves the WRONG way). That is why the predictions are constructed.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.capsules import RoutingCapsule
from dl_techniques.models.capsnet.model import CapsNet

from ..knob_sensitivity_oracle import assert_value_knob_changes_output


D = 4
NUM_INPUT_CAPSULES = 4
NUM_PARENTS = 2
#: Every input capsule is the unit vector along axis 0, so `u_hat` is a slice of `W`.
INPUTS = np.zeros((1, NUM_INPUT_CAPSULES, D), dtype="float32")
INPUTS[0, :, 0] = 1.0


def _parent_lengths(routing_iterations: int) -> np.ndarray:
    """Run the layer with hand-chosen predictions; return `|v_j|` per parent."""
    keras.utils.set_random_seed(0)
    layer = RoutingCapsule(
        num_capsules=NUM_PARENTS,
        dim_capsules=D,
        routing_iterations=routing_iterations,
        use_bias=False,
    )
    tensor = keras.ops.convert_to_tensor(INPUTS)
    layer(tensor)  # build, so `W` exists

    agree = np.zeros(D, dtype="float32")
    agree[0] = 1.0
    scatter = np.eye(D, dtype="float32")

    # W: (1, num_input_capsules, num_parents, dim_out, dim_in)
    weights = np.zeros(tuple(layer.W.shape), dtype="float32")
    for i in range(NUM_INPUT_CAPSULES):
        weights[0, i, 0, :, 0] = agree * 2.0
        weights[0, i, 1, :, 0] = scatter[i] * 2.0  # same norm, no two alike
    layer.W.assign(keras.ops.convert_to_tensor(weights))

    out = np.asarray(keras.ops.convert_to_numpy(layer(tensor)))
    return np.linalg.norm(out[0], axis=-1)


class TestRoutingByAgreement:
    def test_agreement_starves_the_disagreeing_parent(self):
        """The claim itself, measured against the routing-removed baseline."""
        uniform = _parent_lengths(1)  # c is uniform: the unweighted mean
        routed = _parent_lengths(3)

        uniform_ratio = float(uniform[0] / uniform[1])
        routed_ratio = float(routed[0] / routed[1])

        # Control: without routing the two parents are nearly equal -- the
        # unweighted mean cannot tell agreement from disagreement. Measured
        # 1.1765; the bound leaves room without admitting the routed value.
        assert uniform_ratio < 1.5, (
            f"the 1-iteration baseline already separates the parents "
            f"({uniform_ratio:.3f}); the injection is not a control"
        )
        # The claim: routing does tell them apart, by an order of magnitude.
        # Measured 12.09 at 3 iterations.
        assert routed_ratio > 5.0 * uniform_ratio, (
            f"routing did not concentrate on the agreeing parent: "
            f"|v_agree|/|v_scatter| is {routed_ratio:.3f} with 3 iterations vs "
            f"{uniform_ratio:.3f} with routing disabled. An unweighted mean "
            f"scores {uniform_ratio:.3f}."
        )

    def test_the_disagreeing_parent_is_suppressed_monotonically(self):
        """More iterations must suppress the scattered parent further."""
        lengths = {r: _parent_lengths(r) for r in (1, 2, 3, 5)}
        scatter = [float(lengths[r][1]) for r in (1, 2, 3, 5)]
        # Measured: 0.800000, 0.506085, 0.081250, 0.000047
        assert scatter == sorted(scatter, reverse=True) and scatter[0] > scatter[-1], (
            f"|v_scatter| is not decreasing with routing iterations: {scatter}"
        )
        assert scatter[-1] < 0.01 * scatter[0], (
            f"5 routing iterations left the disagreeing parent at "
            f"{scatter[-1]:.6f}, only {scatter[-1] / scatter[0]:.3f} of its "
            f"unrouted length {scatter[0]:.6f}"
        )

    def test_the_agreeing_parent_is_not_suppressed(self):
        """Anti-vacuity: routing must not simply shrink everything."""
        agree = [float(_parent_lengths(r)[0]) for r in (1, 3, 5)]
        # Measured: 0.941176, 0.982091, 0.984563 -- it GROWS.
        assert agree[-1] > agree[0], (
            f"|v_agree| fell from {agree[0]:.6f} to {agree[-1]:.6f}: routing is "
            "attenuating both parents, not selecting between them"
        )


class TestRoutingIterationsReachTheModel:
    """`assert model.routing_iterations == n` was the only model-level check.

    Routing iterations add no parameters, so two CapsNets built under one seed
    hold bit-identical weights and any output difference is attributable to the
    routing loop alone -- the VALUE-knob case of the shared oracle.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "MEASURED 2026-08-18: at random initialization CapsNet's output is "
            "effectively INVARIANT to routing_iterations -- 1 vs 3 iterations, "
            "identical seed and bit-identical weights, move `length` by "
            "max|delta| = 2.65e-07 (the oracle's floor is 1e-05). This is not "
            "the routing loop being skipped: the layer-level tests above prove "
            "the recursion works and separates agreement from disagreement by "
            "12x. It is the statistics of 512 untrained input capsules -- the "
            "agreement terms are near-orthogonal random vectors whose sum is "
            "~sqrt(512) smaller than a coherent one, so `b` barely moves. The "
            "knob is therefore unobservable at the ONLY point a test like this "
            "can cheaply look. Pinned rather than softened; see decisions.md "
            "D-042."
        ),
    )
    def test_routing_iterations_change_the_capsule_lengths(self):
        images = np.random.default_rng(0).random((2, 28, 28, 1)).astype("float32")
        builders = {
            r: (
                lambda r=r: CapsNet(
                    num_classes=10,
                    input_shape=(28, 28, 1),
                    routing_iterations=r,
                    reconstruction=False,
                )
            )
            for r in (1, 3)
        }
        deltas = assert_value_knob_changes_output(
            builders,
            images,
            knob="routing_iterations",
            extract=lambda out: out["length"],
        )
        assert deltas
