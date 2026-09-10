"""RED proof for the transcribed H-Net NumPy oracle.

This module's job is NOT to show that `hnet_reference_numpy.py` runs. It is to
show that the oracle **can fail** -- because an oracle nothing has ever proven
wrong is a decoration, and an oracle written by the same hand in the same
session as the code it grades is a second copy of the same misunderstanding
unless it is anchored to something outside both.

Three mechanisms, applied to every one of the four reference functions:

1. **A hand-computed pin.** One value per function, derived on paper from a
   4-token, 2-channel example, with the arithmetic written out in the test
   docstring digit by digit. A number obtained by running the oracle cannot
   validate the oracle; these numbers were not obtained that way.
2. **A "these differ" twin for every "these agree" assertion** (house rule).
   Each pin is paired with a deliberately wrong input -- a swapped gate, a
   rolled sequence, a mirrored shift, a displaced boundary -- and the test
   asserts the oracle reports the difference. Without the twin, a pin that
   passes proves only that the oracle is deterministic.
3. **Evidence integrity.** The vendored `_reference/*.py` sources are checked
   byte-for-byte against the upstream hashes, and the "pytest must not collect
   the oracle" rule is asserted rather than assumed.

Tolerances: every comparison passes ``rtol=0`` and an ``atol`` derived by
`hnet_reference_numpy.hand_pin_atol` from an operation count spelled out at the
call site. `assert_allclose`'s default ``rtol=1e-7`` would otherwise contribute
~1e-7 of silent tolerance -- eight orders of magnitude above this float64 path's
noise -- and turn every bound here into a decoration.

Pure NumPy: no Keras, no TensorFlow, no GPU.
"""

import hashlib
import pathlib

import numpy as np
import pytest

from .hnet_reference_numpy import (
    P_CLAMP_MAX,
    chunk_reference,
    dechunk_reference,
    hand_pin_atol,
    ratio_loss_reference,
    routing_reference,
)

_HERE = pathlib.Path(__file__).parent
_REFERENCE_DIR = _HERE / "_reference"

# ---------------------------------------------------------------------------
# The one shared 4-token, 2-channel example. Every hand pin below is derived
# from these numbers, and they were chosen so that every intermediate is an
# exact small rational: the (3, 4) / (4, 3) / (-3, 4) triples all have norm 5.
# ---------------------------------------------------------------------------

#: h0 = [1, 0], h1 = [3, 4], h2 = [4, 3], h3 = [-3, 4]
HIDDEN = np.array([[[1.0, 0.0], [3.0, 4.0], [4.0, 3.0], [-3.0, 4.0]]])

#: A separate, deliberately trivial payload for the chunk/dechunk pins, so the
#: gather order is readable straight off the numbers.
PAYLOAD = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

_D = HIDDEN.shape[-1]

# Operation counts on the longest dependency chain, enumerated rather than
# guessed, and fed to `hand_pin_atol`. Each is an upper bound on the rounded
# float64 operations separating the printed hand value from the oracle's value.
#
# routing, per position: two length-D normalizations (D mults + (D-1) adds +
# 1 sqrt + D divides each), one length-D dot (D mults + (D-1) adds), then
# (1 - c), /2 and the clamp compare.
_ROUTING_OPS = 2 * (_D + (_D - 1) + 1 + _D) + (2 * _D - 1) + 3
# dechunk: 4 rounded ops per recurrence step (p*x, 1-p, (1-p)*carry, the sum)
# over M = 3 inner steps, plus the clamp.
_DECHUNK_OPS = 4 * 3 + 1
# ratio loss: two means over 4 elements (3 adds + 1 divide each), then
# 1-tr, 1-ap, two products, one sum, *N, /(N-1).
_RATIO_OPS = 2 * (3 + 1) + 7


# ===========================================================================
# routing_reference
# ===========================================================================


def test_routing_matches_the_hand_computed_pin():
    """Pin `routing_reference` to arithmetic done on paper.

    Hidden states (identity q/k projections, so this is raw adjacent cosine
    similarity -- `dc.py:53-59`)::

        h0 = [ 1, 0]   |h0| = 1
        h1 = [ 3, 4]   |h1| = 5
        h2 = [ 4, 3]   |h2| = 5
        h3 = [-3, 4]   |h3| = 5

    Cosines (`dc.py:86-90`), each pairing h_t with h_{t+1}::

        cos(h0, h1) = (1*3 + 0*4) / (1 * 5) = 3/5      = 0.6
        cos(h1, h2) = (3*4 + 4*3) / (5 * 5) = 24/25    = 0.96
        cos(h2, h3) = (4*-3 + 3*4) / (5 * 5) = 0/25    = 0.0

    Boundary probabilities, p = (1 - cos)/2 (`dc.py:92`), left-padded with 1.0
    at position 0 (`dc.py:95-96`)::

        p0 = 1.0   (forced)
        p1 = (1 - 0.60) / 2 = 0.40 / 2 = 0.20
        p2 = (1 - 0.96) / 2 = 0.04 / 2 = 0.02
        p3 = (1 - 0.00) / 2 = 1.00 / 2 = 0.50

    2-class stack [1-p, p] (`dc.py:102`)::

        [[0.00, 1.00], [0.80, 0.20], [0.98, 0.02], [0.50, 0.50]]

    argmax (`dc.py:104-106`), first-maximum on the tie at position 3::

        idx  = [1, 0, 0, 0]  ->  boundary_mask = [True, False, False, False]

    selected_probs = the winning class (`dc.py:130-132`)::

        [1.00, 0.80, 0.98, 0.50]
    """
    prob, mask, selected = routing_reference(HIDDEN)

    expected_prob = np.array(
        [[[0.00, 1.00], [0.80, 0.20], [0.98, 0.02], [0.50, 0.50]]]
    )
    expected_selected = np.array([[[1.00], [0.80], [0.98], [0.50]]])
    atol = hand_pin_atol(_ROUTING_OPS, scale=1.0)

    np.testing.assert_allclose(prob, expected_prob, rtol=0, atol=atol)
    np.testing.assert_allclose(selected, expected_selected, rtol=0, atol=atol)
    np.testing.assert_array_equal(mask, np.array([[True, False, False, False]]))


def test_routing_pin_twin_a_reversed_sequence_does_not_match_the_pin():
    """The "these differ" twin for the routing pin.

    Reversing the token order leaves the multiset of hidden states, the norms
    and the cosine magnitudes intact, so a routing implementation that ignored
    the ORDER of its pair -- or that read cos(h_{t+1}, h_{t+2}) instead of
    cos(h_t, h_{t+1}) -- would still hit the pin above. It must not.

    Reversed order is h3, h2, h1, h0, giving cos(h3, h2) = 0, cos(h2, h1) = 0.96,
    cos(h1, h0) = 0.6, i.e. p = [1.0, 0.5, 0.02, 0.2] -- the same three values in
    a different place. Position 1 alone moves by |0.5 - 0.2| = 0.3.
    """
    prob_forward, _, _ = routing_reference(HIDDEN)
    prob_reversed, _, _ = routing_reference(HIDDEN[:, ::-1, :])

    delta = float(np.max(np.abs(prob_forward - prob_reversed)))
    assert delta > 0.29, f"reversal must move boundary_prob, got max|delta|={delta}"

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            prob_reversed,
            np.array([[[0.00, 1.00], [0.80, 0.20], [0.98, 0.02], [0.50, 0.50]]]),
            rtol=0,
            atol=hand_pin_atol(_ROUTING_OPS, scale=1.0),
        )


def test_routing_shift_direction_is_pinned_by_a_delta_impulse():
    """`p_{t+1}` is `cos(h_t, h_{t+1})`, not `cos(h_{t+1}, h_{t+2})`.

    A mirrored or off-by-one shift is exactly the defect class a shape test
    cannot see. Perturb ONE token and assert which probabilities move: changing
    `h_2` may only move `p_2` (through `cos(h_1, h_2)`) and `p_3` (through
    `cos(h_2, h_3)`) -- never `p_0` (forced 1.0) and never `p_1`.
    """
    perturbed = HIDDEN.copy()
    perturbed[0, 2] = np.array([1.0, 7.0])

    base, _, _ = routing_reference(HIDDEN)
    moved, _, _ = routing_reference(perturbed)
    per_position = np.max(np.abs(base - moved), axis=-1)[0]

    assert per_position[0] == 0.0, "position 0 is forced to 1.0 and cannot move"
    assert per_position[1] == 0.0, "p_1 depends on (h_0, h_1) only"
    assert per_position[2] > 1e-3, f"p_2 must move, got {per_position[2]}"
    assert per_position[3] > 1e-3, f"p_3 must move, got {per_position[3]}"


def test_routing_applies_q_to_the_EARLIER_token_and_k_to_the_LATER_one():
    """The orientation pin -- and the reason the delta-impulse probe above is
    NOT enough on its own.

    MEASURED during this step, by injecting the mirrored defect into the oracle
    and re-running the suite: swapping the two slices at `dc.py:88-89`, so that
    `q` sees `h_{t+1}` and `k` sees `h_t`, passed every other test in this
    module. It has to. With the reference's IDENTITY initialisation
    (`dc.py:55-59`) the pair is symmetric -- ``cos(h_t, h_{t+1}) ==
    cos(h_{t+1}, h_t)`` -- so under the default projections the mirrored
    implementation is not merely close to the correct one, it is bit-identical
    to it. No amount of identity-projection testing can see this defect.

    It becomes visible the moment `q_proj` and `k_proj` differ, which is the
    state every trained model is in after step 0. So the orientation is pinned
    here with a DISTINCT `q_weight`, by hand:

    `q_weight = R` = the 90-degree rotation [[0, -1], [1, 0]]. `nn.Linear`
    computes ``x @ W.T``, and for a row vector ``x @ R.T == R x``, i.e. the
    ordinary rotation of the column vector::

        q(h0) = R [ 1, 0] = [ 0,  1]
        q(h1) = R [ 3, 4] = [-4,  3]
        q(h2) = R [ 4, 3] = [-3,  4]
        q(h3) = R [-3, 4] = [-4, -3]

    `k_weight = I`, so `k(h_t) = h_t`. Cosines, q on the EARLIER token::

        cos(q(h0), h1) = ([0,1].[3,4]) / (1*5)   =   4/5  =  0.80
        cos(q(h1), h2) = ([-4,3].[4,3]) / (5*5)  =  -7/25 = -0.28
        cos(q(h2), h3) = ([-3,4].[-3,4]) / (5*5) =  25/25 =  1.00

    p = (1 - cos)/2, left-padded with 1.0::

        p = [1.00, 0.10, 0.64, 0.00]

    argmax at p > 0.5 gives boundary_mask = [True, False, True, False].

    The mirrored defect (q on the LATER token) would instead give
    cos(q(h1), h0) = -4/5, cos(q(h2), h1) = 7/25 and cos(q(h3), h2) = -1, i.e.
    p = [1.00, 0.90, 0.36, 1.00] -- different at every position, which is what
    makes this the twin the identity-init tests cannot supply.

    NOTE, recorded because it is the evidence this pin is load-bearing: the
    first draft of this docstring transposed the rotation (it applied ``R.T``
    where the code applies ``R``) and therefore printed the two vectors the
    other way round. The pin FAILED against the oracle, the paper arithmetic was
    redone, and the paper was the thing that was wrong. A pin that has never
    disagreed with the code it grades has not yet been shown to be a pin.
    """
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    identity = np.eye(_D)

    prob, mask, _ = routing_reference(HIDDEN, q_weight=rotation, k_weight=identity)

    atol = hand_pin_atol(_ROUTING_OPS, scale=1.0)
    np.testing.assert_allclose(
        prob[0, :, 1], np.array([1.00, 0.10, 0.64, 0.00]), rtol=0, atol=atol
    )
    np.testing.assert_array_equal(mask, np.array([[True, False, True, False]]))

    # The mirrored answer, stated explicitly so this test fails LOUDLY rather
    # than ambiguously if the slices are ever swapped.
    assert not np.allclose(
        prob[0, :, 1], np.array([1.00, 0.90, 0.36, 1.00]), rtol=0, atol=atol
    )


def test_routing_identity_init_makes_the_orientation_defect_invisible():
    """Why the test above needs a non-identity weight -- asserted, not asserted-in-prose.

    Under the reference's identity init the q/k pair is symmetric, so a mirrored
    implementation is bit-identical to the correct one. This pins that fact: the
    cosine of the reversed pair equals the cosine of the forward pair to 0 ulp
    when both projections are the identity, and does NOT when `q_weight` is the
    rotation used above.
    """
    identity = np.eye(_D)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])

    def cos_of(a, b, q_w, k_w):
        pair = np.array([[a, b]])
        prob, _, _ = routing_reference(pair, q_weight=q_w, k_weight=k_w)
        return float(prob[0, 1, 1])

    h1, h2 = HIDDEN[0, 1], HIDDEN[0, 2]

    assert cos_of(h1, h2, identity, identity) == cos_of(h2, h1, identity, identity)
    assert cos_of(h1, h2, rotation, identity) != cos_of(h2, h1, rotation, identity)


def test_routing_threshold_is_strictly_above_half_because_argmax_breaks_ties_low():
    """`p == 0.5` is NOT a boundary: `argmax([0.5, 0.5])` returns index 0.

    `dc.py:104-106` selects `argmax(dim=-1) == 1`, and both `torch.argmax` and
    `np.argmax` return the FIRST maximal index. The realised predicate is
    therefore ``p > 0.5``, not ``p >= 0.5``. This is reachable exactly rather
    than theoretically: any two orthogonal adjacent hidden states give
    ``cos = 0`` and hence ``p = 0.5`` bit-exactly -- positions 2 and 3 of the
    shared example do it.

    Three-armed, in the output dtype: just below, exactly at, and just above.
    """
    eps = np.finfo(np.float64).eps

    def mask_for(p_value):
        # cos = 1 - 2p, realised by rotating a unit vector to that cosine.
        cos = 1.0 - 2.0 * p_value
        sin = np.sqrt(max(0.0, 1.0 - cos * cos))
        pair = np.array([[[1.0, 0.0], [cos, sin]]])
        _, mask, _ = routing_reference(pair)
        return bool(mask[0, 1])

    assert mask_for(0.5 - eps) is False
    assert mask_for(0.5) is False, "the tie must resolve to NOT-a-boundary"
    assert mask_for(0.5 + 4.0 * eps) is True, "strictly above 0.5 must fire"

    # And the tie really is exercised by the shared example, bit-exactly.
    prob, mask, _ = routing_reference(HIDDEN)
    assert prob[0, 3, 1] == 0.5
    assert bool(mask[0, 3]) is False


def test_routing_mask_forbids_a_boundary_and_the_twin_shows_it_would_otherwise_fire():
    """`boundary_mask &= mask` (`dc.py:107-109`) with its anti-vacuity twin.

    Position 0 is forced to `p = 1.0`, so it is a boundary in every unmasked
    run. Marking it invalid must clear it -- and the twin (the same input with
    no mask) must show the bit was set in the first place, otherwise the
    assertion measures nothing.
    """
    invalidate_first = np.array([[False, True, True, True]])

    _, masked, _ = routing_reference(HIDDEN, mask=invalidate_first)
    _, unmasked, _ = routing_reference(HIDDEN, mask=None)

    assert bool(unmasked[0, 0]) is True, "twin: unmasked position 0 IS a boundary"
    assert bool(masked[0, 0]) is False, "a masked-out position cannot be selected"


def test_routing_default_projection_is_the_identity_and_a_rotation_moves_it():
    """The default is `torch.eye` (`dc.py:55-59`) -- with a twin that it matters.

    Passing an explicit identity must reproduce the default bit-for-bit; passing
    a 90-degree rotation as `q_weight` must not. Without the second arm, the
    first proves only that `None` and `eye` take the same code path.
    """
    identity = np.eye(_D)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])

    default_prob, _, _ = routing_reference(HIDDEN)
    explicit_prob, _, _ = routing_reference(HIDDEN, q_weight=identity, k_weight=identity)
    rotated_prob, _, _ = routing_reference(HIDDEN, q_weight=rotation, k_weight=identity)

    np.testing.assert_allclose(default_prob, explicit_prob, rtol=0, atol=0.0)
    delta = float(np.max(np.abs(default_prob - rotated_prob)))
    assert delta > 1e-3, f"a rotated q_proj must move the output, got {delta}"


# ===========================================================================
# chunk_reference
# ===========================================================================


def test_chunk_matches_the_hand_computed_pin():
    """Pin `chunk_reference` to arithmetic done on paper.

    Payload rows (2 channels each) and the boundary mask::

        x0 = [1, 2]   boundary
        x1 = [3, 4]   not
        x2 = [5, 6]   boundary
        x3 = [7, 8]   not

    Sort key (`dc.py:188-190`), L = 4::

        token_idx = arange(4) + (~mask) * 4
                  = [0, 1, 2, 3] + [0, 4, 0, 4]
                  = [0, 5, 2, 7]

    argsort (`dc.py:191`) of [0, 5, 2, 7] is [0, 2, 1, 3]: the two boundary
    positions first, in position order, then the two dropped ones.

    Gather the first `max_chunks = 3` columns (`dc.py:193-199`)::

        rows 0, 2, 1  ->  [[1, 2], [5, 6], [3, 4]]

    The third column is x1, a NON-boundary row -- padding garbage, exactly as
    upstream produces it -- and `next_mask` (`dc.py:201-204`) marks it invalid::

        num_tokens = 2,  arange(3) < 2  ->  [True, True, False]

    Everything here is a permutation of exactly-representable inputs, so the
    comparison is bit-exact: atol = 0.
    """
    boundary_mask = np.array([[True, False, True, False]])

    chunked, next_mask = chunk_reference(PAYLOAD, boundary_mask, max_chunks=3)

    np.testing.assert_allclose(
        chunked,
        np.array([[[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]]),
        rtol=0,
        atol=0.0,
    )
    np.testing.assert_array_equal(next_mask, np.array([[True, True, False]]))


def test_chunk_pin_twin_moving_one_boundary_changes_the_gather():
    """The "these differ" twin for the chunk pin.

    Same payload, boundary moved from position 2 to position 3. The kept COUNT
    is unchanged, so any check on shapes or on `num_tokens` alone still passes;
    the gathered CONTENT must change, from [x0, x2, x1] to [x0, x3, x1].
    """
    moved = np.array([[True, False, False, True]])

    chunked, next_mask = chunk_reference(PAYLOAD, moved, max_chunks=3)

    np.testing.assert_array_equal(next_mask, np.array([[True, True, False]]))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            chunked,
            np.array([[[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]]),
            rtol=0,
            atol=0.0,
        )
    np.testing.assert_allclose(
        chunked, np.array([[[1.0, 2.0], [7.0, 8.0], [3.0, 4.0]]]), rtol=0, atol=0.0
    )


def test_chunk_with_max_chunks_none_reproduces_the_reference_width():
    """`max_chunks=None` is the reference's own `max(boundary_mask.sum(-1))`.

    The D-007 fixed cap must be a parameter, not a rewrite: with the cap off,
    `chunk_reference` is `dc.py:181-205` verbatim, width 2 here. The twin is the
    capped call, which must produce a DIFFERENT width from the same input.
    """
    boundary_mask = np.array([[True, False, True, False]])

    reference_width, reference_mask = chunk_reference(PAYLOAD, boundary_mask)
    capped, _ = chunk_reference(PAYLOAD, boundary_mask, max_chunks=3)

    assert reference_width.shape == (1, 2, 2)
    assert capped.shape == (1, 3, 2)
    np.testing.assert_array_equal(reference_mask, np.array([[True, True]]))
    np.testing.assert_allclose(
        reference_width, capped[:, :2, :], rtol=0, atol=0.0
    ), "the cap must only truncate; the kept prefix is identical"


def test_chunk_truncation_drops_the_LAST_boundary_not_the_least_likely():
    """Position-order truncation (D-007 / S5), pinned where it is falsifiable.

    Three boundaries, `max_chunks = 2`. Magnitude-order truncation would keep
    whichever boundaries had the highest probability; position-order truncation
    keeps the first two and drops the third, whatever their probabilities. The
    oracle is the reference's stable partition, so it has no access to
    probabilities at all -- and that is precisely the property being pinned,
    because a later reimplementation CAN reach for them.

    token_idx = [0, 1, 6, 3] -> argsort [0, 1, 3, 2] -> first 2 = [x0, x1].
    `next_mask = arange(2) < 3` = [True, True]: both columns are real chunks and
    the third boundary (x3) is simply lost -- D-007's named failure mode, made
    visible rather than assumed rare.
    """
    boundary_mask = np.array([[True, True, False, True]])

    chunked, next_mask = chunk_reference(PAYLOAD, boundary_mask, max_chunks=2)

    np.testing.assert_allclose(
        chunked, np.array([[[1.0, 2.0], [3.0, 4.0]]]), rtol=0, atol=0.0
    )
    np.testing.assert_array_equal(next_mask, np.array([[True, True]]))
    # Twin: x3 IS reachable when the cap admits it, so its absence above is the
    # cap's doing and not a gather that can never see position 3.
    wide, _ = chunk_reference(PAYLOAD, boundary_mask, max_chunks=3)
    np.testing.assert_allclose(wide[0, 2], np.array([7.0, 8.0]), rtol=0, atol=0.0)


def test_chunk_rows_are_independent_of_one_another():
    """Row `i`'s output may not depend on row `j`'s bytes (D-007 invariant 2).

    Hold row 0 fixed, randomise row 1 across 32 draws under a FIXED width, and
    assert row 0's output is bit-identical every time. The twin asserts row 1's
    output did move, so the `atol=0` arm is not passing because nothing changed.
    """
    rng = np.random.default_rng(20260909)
    row0 = PAYLOAD[0]
    mask0 = np.array([True, False, True, False])

    reference = None
    row1_outputs = []
    for _ in range(32):
        row1 = rng.normal(size=row0.shape)
        mask1 = rng.random(size=4) < 0.5
        mask1[0] = True
        batch = np.stack([row0, row1])
        masks = np.stack([mask0, mask1])
        chunked, _ = chunk_reference(batch, masks, max_chunks=3)
        if reference is None:
            reference = chunked[0].copy()
        np.testing.assert_allclose(chunked[0], reference, rtol=0, atol=0.0)
        row1_outputs.append(chunked[1].copy())

    spread = float(np.max(np.abs(np.array(row1_outputs) - row1_outputs[0])))
    assert spread > 0.0, "twin: row 1's output must actually vary across draws"


# ===========================================================================
# dechunk_reference
# ===========================================================================

#: The dechunk pin's inputs, shared with its twins.
INNER = np.array([[[2.0, 0.0], [0.0, 4.0], [9.0, 9.0]]])
DECHUNK_P = np.array([[1.0, 0.3, 0.25, 0.9]])
DECHUNK_MASK = np.array([[True, False, True, False]])


def test_dechunk_matches_the_hand_computed_pin():
    """Pin `dechunk_reference` to arithmetic done on paper.

    Inner (chunk-resolution) values, M = 3::

        z0 = [2, 0]   z1 = [0, 4]   z2 = [9, 9]

    Full-resolution probabilities and boundaries::

        p_full = [1.0, 0.3, 0.25, 0.9]      mask = [T, F, T, F]

    Clamp to [1e-4, 1 - 1e-4] (`dc.py:256`): only p_0 moves, 1.0 -> 0.9999.

    Gather p into boundary order (`dc.py:265-273`), same partition as ChunkLayer
    (token_idx = [0, 5, 2, 7], argsort = [0, 2, 1, 3]), keeping the first M = 3::

        p = [p_0, p_2, p_1] = [0.9999, 0.25, 0.3]

    (The third entry is a NON-boundary probability riding along in the padding
    column -- upstream's behaviour, kept.)

    EMA recurrence h_t = p_t z_t + (1 - p_t) h_{t-1}, h_{-1} = 0 (`dc.py:333`)::

        h0 = 0.9999*[2, 0] + 0.0001*[0, 0]
           = [1.9998, 0]
        h1 = 0.25*[0, 4] + 0.75*[1.9998, 0]
           = [0 + 1.49985, 1 + 0]
           = [1.49985, 1.0]
        h2 = 0.3*[9, 9] + 0.7*[1.49985, 1.0]
           = [2.7 + 1.049895, 2.7 + 0.7]
           = [3.749895, 3.4]

    Scatter back (`dc.py:302-308`)::

        plug_back_idx = cumsum([1, 0, 1, 0]) - 1 = [0, 0, 1, 1]
        out = [h0, h0, h1, h1]

    h2 is never plugged back: it is the padding chunk. That is correct, and it
    is why the pin covers it only through the recurrence, not the output.
    """
    out = dechunk_reference(INNER, DECHUNK_P, DECHUNK_MASK)

    expected = np.array(
        [
            [
                [1.9998, 0.0],
                [1.9998, 0.0],
                [1.49985, 1.0],
                [1.49985, 1.0],
            ]
        ]
    )
    atol = hand_pin_atol(_DECHUNK_OPS, scale=float(np.max(np.abs(expected))))
    np.testing.assert_allclose(out, expected, rtol=0, atol=atol)


def test_dechunk_pin_twin_swapping_the_gate_breaks_it():
    """The "these differ" twin: `p <-> 1-p` must not reproduce the pin.

    A swapped EMA gate -- `h_t = (1-p) z_t + p h_{t-1}` -- is the single most
    likely way this recurrence gets written wrong, and it is invisible to every
    shape and finiteness check. Feeding the oracle `1 - p` reproduces exactly
    that defect on the input side, so the pin must move.
    """
    swapped = dechunk_reference(INNER, 1.0 - DECHUNK_P, DECHUNK_MASK)
    correct = dechunk_reference(INNER, DECHUNK_P, DECHUNK_MASK)

    delta = float(np.max(np.abs(swapped - correct)))
    assert delta > 1.0, f"a swapped gate must move the output, got {delta}"


def test_dechunk_pin_twin_rolling_p_by_one_breaks_it():
    """Second twin: an off-by-one in the probability gather must be visible.

    Rolling `p` by one position leaves its multiset, its mean and its clamp
    behaviour untouched -- a summary-statistic check would pass. The pinned
    output must not.
    """
    rolled = dechunk_reference(INNER, np.roll(DECHUNK_P, 1, axis=1), DECHUNK_MASK)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            rolled,
            np.array(
                [[[1.9998, 0.0], [1.9998, 0.0], [1.49985, 1.0], [1.49985, 1.0]]]
            ),
            rtol=0,
            atol=hand_pin_atol(_DECHUNK_OPS, scale=3.75),
        )


def test_dechunk_repeats_the_last_chunk_value_at_non_boundary_positions():
    """`plug_back_idx = cumsum(mask) - 1` (`dc.py:302-308`), with its twin.

    Positions 0 and 1 must be bit-identical (position 1 is not a boundary, so it
    repeats chunk 0); positions 1 and 2 must NOT be, because position 2 opens a
    new chunk. The second arm is what stops the first from passing on an output
    that is constant everywhere.
    """
    out = dechunk_reference(INNER, DECHUNK_P, DECHUNK_MASK)[0]

    np.testing.assert_allclose(out[1], out[0], rtol=0, atol=0.0)
    np.testing.assert_allclose(out[3], out[2], rtol=0, atol=0.0)
    assert float(np.max(np.abs(out[2] - out[1]))) > 0.4, "a boundary must take a NEW value"


def test_dechunk_clamps_p_at_the_reference_bound():
    """`clamp(p, 1e-4, 1 - 1e-4)` (`dc.py:256`) is applied, not skipped.

    With `p = 1.0` and `h_{-1} = 0`, an unclamped recurrence returns exactly the
    input chunk; the clamped one returns `0.9999 * z_0`. The gap is `1e-4 * z_0`
    -- small, which is precisely why this needs an explicit assertion rather
    than a tolerance that would swallow it. So: assert the clamped value to a
    derived atol AND assert the unclamped value is outside that atol.
    """
    single = np.array([[[2.0, 0.0]]])
    prob = np.array([[1.0]])
    mask = np.array([[True]])

    out = dechunk_reference(single, prob, mask)
    atol = hand_pin_atol(_DECHUNK_OPS, scale=2.0)

    np.testing.assert_allclose(out, np.array([[[2.0 * P_CLAMP_MAX, 0.0]]]), rtol=0, atol=atol)
    assert abs(float(out[0, 0, 0]) - 2.0) > atol, "the unclamped value must be excluded"


def test_dechunk_stays_finite_and_decays_monotonically_at_the_clamp_floor():
    """The long-chunk arm, driven to the clamp rather than sampled.

    At `p = 1e-4` over an inner length of 512, the carry decays by `(1-p)` every
    step. A random `p` would put this arm nowhere near the regime that breaks
    accumulators, so `p` is DRIVEN to its floor. Two assertions, because
    finiteness alone is near-vacuous: the output is finite, and the carry decays
    strictly and monotonically toward zero from a single non-zero first chunk.

    (The closed-form reassociation of this same recurrence returns `nan` here --
    measured, D-010(c). The explicit loop is what the reference itself uses at
    `dc.py:333` and is what this oracle implements.)
    """
    inner_len = 512
    inner = np.zeros((1, inner_len, 2))
    inner[0, 0] = np.array([1.0, -1.0])
    prob = np.full((1, inner_len), 1e-4)
    mask = np.ones((1, inner_len), dtype=bool)

    out = dechunk_reference(inner, prob, mask)

    assert np.all(np.isfinite(out)), "the recurrence must stay finite at the clamp floor"
    trace = np.abs(out[0, :, 0])
    assert trace[0] == pytest.approx(1e-4, rel=1e-12)
    assert np.all(np.diff(trace[1:]) < 0.0), "the carry must decay strictly"
    assert trace[-1] > 0.0, "and must not underflow to zero at L=512 in float64"


# ===========================================================================
# ratio_loss_reference
# ===========================================================================


def test_ratio_loss_matches_the_hand_computed_pin():
    """Pin `ratio_loss_reference` to arithmetic done on paper.

    Inputs are the routing pin's own outputs, so this pin composes with that
    one::

        p          = [1.00, 0.20, 0.02, 0.50]
        mask       = [True, False, False, False]
        N          = 2

    Means over the whole tensor (`train.py:34-35`)::

        true_ratio  = 1 / 4 = 0.25
        average_prob = (1.00 + 0.20 + 0.02 + 0.50) / 4 = 1.72 / 4 = 0.43

    Loss (`train.py:37-40`)::

        (1 - 0.25) * (1 - 0.43)          = 0.75 * 0.57 = 0.4275
        0.25 * 0.43 * (2 - 1)            =             = 0.1075
        sum                              =             = 0.5350
        * N / (N - 1) = * 2 / 1          =             = 1.0700
    """
    prob, mask, _ = routing_reference(HIDDEN)

    loss = ratio_loss_reference(prob, mask, target_ratio=2.0)

    atol = hand_pin_atol(_RATIO_OPS, scale=1.07)
    np.testing.assert_allclose(loss, 1.07, rtol=0, atol=atol)


def test_ratio_loss_pin_twin_a_different_boundary_count_moves_it():
    """The "these differ" twin for the loss pin.

    Hold the probabilities fixed and flip one mask bit. `true_ratio` goes from
    0.25 to 0.50, so the loss must move::

        (1 - 0.5) * (1 - 0.43) + 0.5 * 0.43 * 1 = 0.285 + 0.215 = 0.5
        * 2 / 1                                                  = 1.0

    A loss that ignored `boundary_mask` and used only the probabilities would
    return 1.07 here.
    """
    prob, _, _ = routing_reference(HIDDEN)
    flipped = np.array([[True, True, False, False]])

    loss = ratio_loss_reference(prob, flipped, target_ratio=2.0)

    atol = hand_pin_atol(_RATIO_OPS, scale=1.07)
    np.testing.assert_allclose(loss, 1.0, rtol=0, atol=atol)
    assert abs(loss - 1.07) > atol, "the mask must be load-bearing"


def test_ratio_loss_is_minimised_when_the_realised_ratio_matches_N():
    """A property arm, independent of the pins.

    `train.py:37-40` is minimised when `true_ratio = average_prob = 1/N`. Sweep
    a constant probability `q` at `N = 4` with the mask matching it, and assert
    the minimum lands on `q = 0.25` and that the value there is `N/(N-1) *
    (1 - 1/N) * ... `, i.e. strictly below both endpoints. This uses no number
    from the pins, so it is a second, independent handle on the same formula.
    """
    n = 4.0
    grid = np.linspace(0.05, 0.95, 19)
    losses = []
    for q in grid:
        # A 100-position row whose realised ratio equals q to 1/100.
        count = int(round(q * 100))
        prob = np.full((1, 100), q)
        mask = np.zeros((1, 100), dtype=bool)
        mask[0, :count] = True
        losses.append(ratio_loss_reference(prob, mask, target_ratio=n))

    losses = np.array(losses)
    argmin = int(np.argmin(losses))
    assert grid[argmin] == pytest.approx(1.0 / n, abs=0.03)
    assert losses[argmin] < losses[0]
    assert losses[argmin] < losses[-1]


def test_ratio_loss_rejects_N_equal_to_one_by_upstreams_own_precondition():
    """`N > 1` is required (`train.py:25`); `N = 1` divides by zero.

    Transcribed behaviour, not invented behaviour: the oracle does not add a
    guard upstream does not have. This test records what actually happens, so
    that a Keras loss which DOES add a guard is a documented divergence rather
    than an accident.
    """
    prob, mask, _ = routing_reference(HIDDEN)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = ratio_loss_reference(prob, mask, target_ratio=1.0)
    assert not np.isfinite(value)


# ===========================================================================
# Evidence integrity: the vendored sources and the no-collect rule
# ===========================================================================


@pytest.mark.parametrize(
    "name,expected_sha256",
    [
        ("dc.py", "b0a30a75245a0f6911a92a56e6be80a3725fc27aa859bd0e9905c239fe87bb38"),
        ("train.py", "83534fe9ddf769f78bac331fbb7aacd8a9412a30e2f5cdf180f4d25434c3d27e"),
    ],
)
def test_the_vendored_reference_sources_are_byte_identical_to_upstream(
    name, expected_sha256
):
    """The oracle's `file:line` citations are only meaningful if the file is.

    These hashes are of `github.com/goombalab/hnet` at commit
    `3673fe1217ebeb0d1438c7c71d58d32bdd190ec2`. A reformat, a lint fix or an
    "unused import" cleanup would shift every line number cited in
    `hnet_reference_numpy.py` and silently invalidate the transcription record.
    """
    digest = hashlib.sha256((_REFERENCE_DIR / name).read_bytes()).hexdigest()
    assert digest == expected_sha256, f"_reference/{name} has been modified"


def test_the_reference_directory_is_not_an_importable_package():
    """No `__init__.py` under `_reference/` -- by design, and asserted.

    Without this, `_reference/` could quietly acquire a package marker, become
    importable from `src/`, and turn evidence into a second copy of our own
    code. It also imports `torch`, `einops` and `mamba_ssm`, none of which this
    repo depends on, so importing it would break collection outright.
    """
    assert _REFERENCE_DIR.is_dir()
    assert not (_REFERENCE_DIR / "__init__.py").exists()
    assert sorted(p.name for p in _REFERENCE_DIR.glob("*.py")) == ["dc.py", "train.py"]


def test_pytest_collects_no_tests_from_the_oracle_module():
    """`hnet_reference_numpy.py` carries no `test_`-prefixed callables.

    The no-`test_`-prefix filename is what keeps pytest from collecting the
    oracle; this asserts the complementary half -- that nothing inside it would
    be collected if it were ever renamed or imported by a collected module.
    """
    from . import hnet_reference_numpy as oracle

    collected = [n for n in dir(oracle) if n.startswith("test_") or n.startswith("Test")]
    assert collected == [], f"the oracle must define no test callables, found {collected}"
