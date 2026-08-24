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

At the MODEL level the observable is different, and the second class in this
file explains why: inside a real `CapsNet` the routing recursion demonstrably
runs, but the squash in the small-`|s|` regime crushes its effect on the OUTPUT
below any assertable floor, at every capsule count down to N=1. So the
model-level test pins the coupling coefficients `c` -- what routing actually
computes -- rather than the output they barely move.
"""

import keras
import numpy as np

from dl_techniques.layers.capsules import RoutingCapsule
from dl_techniques.models.capsnet.model import CapsNet


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


#: `RoutingCapsule` softmaxes over `axis=2` = the PARENT axis, so a uniform
#: coupling is `1 / num_capsules`, NOT `1 / num_input_capsules`.
NUM_MODEL_PARENTS = 10
MODEL_IMAGES = np.random.default_rng(0).random((2, 28, 28, 1)).astype("float32")
_COUPLING_CACHE: dict = {}


def _model_couplings(routing_iterations: int) -> list:
    """Every coupling tensor `c` computed inside a real `CapsNet` forward.

    Captured by spying on `keras.activations.softmax` for the duration of the
    forward pass, filtered to rank-5 calls on `axis=2` -- which
    `RoutingCapsule.call`'s single softmax is the only producer of anywhere in
    the CapsNet path. Deliberately does NOT modify `capsules.py`: making `c` an
    output or an attribute of a layer shared by three model packages is a far
    larger change than this observation warrants.

    Returns one `np.ndarray` of shape `(batch, num_input_capsules,
    num_capsules, 1, 1)` per routing iteration, in iteration order.
    """
    if routing_iterations in _COUPLING_CACHE:
        return _COUPLING_CACHE[routing_iterations]

    recorded = []
    real_softmax = keras.activations.softmax

    def spy(x, axis=-1):
        out = real_softmax(x, axis=axis)
        if axis == 2 and len(x.shape) == 5:
            recorded.append(np.asarray(keras.ops.convert_to_numpy(out)))
        return out

    keras.utils.set_random_seed(1234)
    model = CapsNet(
        num_classes=NUM_MODEL_PARENTS,
        input_shape=(28, 28, 1),
        routing_iterations=routing_iterations,
        reconstruction=False,
    )
    model.build((None, 28, 28, 1))

    keras.activations.softmax = spy
    try:
        model(MODEL_IMAGES, training=False)
    finally:
        keras.activations.softmax = real_softmax

    _COUPLING_CACHE[routing_iterations] = recorded
    return recorded


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
    """The model-level question: does the routing recursion run inside `CapsNet`?

    This class used to assert that `routing_iterations` moves the model's
    OUTPUT, pinned `xfail(strict=True)` on the claim that the knob is
    unobservable because 512 near-orthogonal untrained capsules average the
    agreement away. A sweep of 11 capsule counts x 4 seeds (44 cells, all with
    weights asserted bit-identical across the 1-vs-3 arms) settled both halves
    of that and neither survived:

    * The output delta DOES rise as the capsule count falls -- median
      3.15e-07 at N=512 to 4.73e-06 at N=1, a 15x monotone rise, and the N=1
      MINIMUM over seeds (3.92e-06) already exceeds the N=512 MAXIMUM
      (3.63e-07) tenfold against a per-cell seed spread of only 2-3x.
    * It nevertheless never crosses the shared oracle's 1e-05 floor, at any
      capsule count or seed, INCLUDING N=1 -- the hard floor of the axis,
      where there is no averaging left to blame. So no honest model-level
      output assertion exists here, at any config, and the old test asserted a
      model defect that does not exist.

    The `sqrt(N)` averaging law is real as a scaling law (`max|c-0.1|` rises
    22.8x from N=512 to N=1 against `sqrt(512) = 22.6`) but it is NOT the
    binding suppressor. That is the SQUASH in the small-`|s|` regime: measured
    `mean|s| ~= 0.034` into the squash and `mean|v| ~= 0.0012 ~= |s|^2` out of
    it, so `|v|` is crushed quadratically, `agreement = <u_hat, v>` is O(1e-6),
    `b` barely leaves zero and `c` barely leaves uniform. The capsule lengths
    the old test read are 0.0008-0.0020 -- three orders of magnitude below the
    [0.1, 0.9] band CapsNet's own margin loss assumes. Feeding bigger inputs
    does not lift it (x1 -> x300 moves `mean|s|` only 0.0317 -> 0.0388 and
    saturates by x10): `BatchNormalization` normalises the conv stack and
    `PrimaryCapsule`'s own squash caps `|u| < 1`.

    So the observable this class pins is the COUPLING COEFFICIENT `c` itself --
    what routing actually computes -- rather than its (mechanically negligible)
    effect on the output. See decisions.md D-003 and D-008.
    """

    def test_the_routing_loop_runs_once_per_configured_iteration(self):
        """The loop is not collapsed, unrolled away or skipped in the model path.

        `capsules.py`'s single `keras.activations.softmax(b, axis=2)` is the
        only softmax over a rank-5 tensor on axis 2 anywhere in the CapsNet
        forward, so counting those calls counts routing iterations exactly,
        without touching `capsules.py`. MEASURED 2026-08-18:
        `routing_iterations` 1/2/3/5 -> **1/2/3/5** calls, each `c` of shape
        `(batch, 512, 10, 1, 1)`.
        """
        for r in (1, 2, 3, 5):
            couplings = _model_couplings(r)
            assert len(couplings) == r, (
                f"CapsNet(routing_iterations={r}) ran the routing softmax "
                f"{len(couplings)} times, not {r}: the loop is not executing "
                "as configured"
            )
            assert couplings[0].shape[2] == NUM_MODEL_PARENTS

    def test_couplings_start_uniform_and_depart_monotonically(self):
        """`c` is bit-uniform at iteration 1, then departs, more each iteration.

        `b` is initialised to zeros and softmax is over `axis=2` = the PARENT
        axis, so uniform is `1/num_capsules = 1/10` -- NOT
        `1/num_input_capsules`. That makes iteration 1 an exact,
        assumption-free reference point: any departure at all is the recursion
        feeding `<u_hat, v>` back into `b`.

        MEASURED 2026-08-18 at the default 28x28 config (N=512 input capsules,
        seed 1234), `max|c - 0.1|` per iteration:

            iteration  1            2            3            4            5
                       0.0 (exact)  2.376735e-06 4.760921e-06 7.130206e-06 9.514391e-06

        RE-MEASURED 2026-08-22 (D-035), same config, identical in 3 of 3 runs:

            iteration  1            2            3            4            5
                       7.450581e-09 2.369285e-06 4.738569e-06 7.137656e-06 9.506941e-06
                       = 1 ulp      = 318 ulp    = 636 ulp    = 958 ulp    = 1276 ulp

        and at the N=1 floor of the capsule-count axis: 5.43e-05 then
        1.09e-04. The magnitudes are tiny for the squash reason in the class
        docstring, but the SHAPE is exact and reproducible: zero, then strictly
        increasing.

        RED-PROOF (2026-08-18, both injections run): discarding the feedback
        (`softmax(zeros_like(b))`, i.e. routing removed but the loop left in
        place) gives 0.0 at EVERY iteration and fails the monotonicity
        assertion; forcing the layer's own loop to a single pass regardless of
        the configured count gives 1 call at every `routing_iterations` and
        fails the count assertion above. Shipped as written, both pass.
        """
        couplings = _model_couplings(5)
        departures = [
            float(np.max(np.abs(c - 1.0 / NUM_MODEL_PARENTS))) for c in couplings
        ]

        # DECISION plan-2026-08-22T035419-a11304c8/D-035
        # `departures[0] == 0.0` was UNSATISFIABLE and was RED 12 of 12 solo
        # runs at baseline with `assert 7.450580596923828e-09 == 0.0`. Do NOT
        # restore it, and do NOT replace it with a hand-picked `abs=1e-8`.
        # 1/10 is not representable in binary: `float32(0.1)` is
        # 0.100000001490116119384765625, the softmax denominator rounds one ulp
        # the other way, and the reference `1.0 / NUM_MODEL_PARENTS` is a
        # float64 -- so `max|c - 0.1|` is 7.450581e-09 = EXACTLY ONE
        # `np.spacing(np.float32(0.1))`, and no correct implementation can make
        # it zero. LESSONS: a tolerance below the output dtype's representable
        # resolution is not strict, it is broken.
        # The uniformity claim is kept EXACT by asserting it where it IS exact:
        # every element of `c` at iteration 1 is BIT-IDENTICAL (measured: a
        # single unique float32 value, 0.09999999), which is what "b starts at
        # zeros so c must be uniform" actually says and which no tolerance can
        # launder. The magnitude arm below is then the ulp bound. Its 4-ulp
        # headroom is 79x below iteration 2's departure (318 ulp), so it cannot
        # swallow the effect this test exists to see.
        first = couplings[0]
        assert np.unique(first).size == 1, (
            f"`b` starts at zeros, so `c` must be BIT-uniform at iteration 1; "
            f"measured {np.unique(first).size} distinct coupling values, "
            f"spread {float(first.max() - first.min()):.6e}"
        )
        uniform_ulp = float(np.spacing(np.float32(1.0 / NUM_MODEL_PARENTS)))
        assert departures[0] <= 4 * uniform_ulp, (
            f"`c` at iteration 1 sits {departures[0] / uniform_ulp:.2f} ulp from "
            f"uniform (1/{NUM_MODEL_PARENTS}), over the 4-ulp float32 rounding "
            f"budget; measured a departure of {departures[0]:.6e}"
        )
        assert departures[1] > 0.0, (
            "`c` never left uniform after the first agreement update: the "
            "routing feedback is not reaching `b`"
        )
        assert departures == sorted(departures), (
            f"`c`'s departure from uniform is not monotone in the iteration "
            f"count: {['%.6e' % d for d in departures]}"
        )
        assert departures[-1] > departures[1], (
            f"`c` stopped moving after the first update "
            f"({departures[1]:.6e} -> {departures[-1]:.6e}): the recursion is "
            "running but not accumulating"
        )
