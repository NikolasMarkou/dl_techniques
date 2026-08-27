"""Guard: no query position may be left with nothing to attend to.

Why this test exists
--------------------
``_compute_mandatory_indices`` retains two groups on top of the ``top_k`` budget,
and the second exists for one specific reason, stated in its own docstring: the
coarsest windows scatter to ``[F - 1, N - 1]`` with ``F = p^(L-1)``, so base
positions ``0 .. F - 2`` are **unreachable from the coarsest level**. The level-0
prefix entries are the cheapest set that closes that hole.

Deleting that guarantee is caught by ZERO of the file's 12 tests while changing
the output by 3.89. Nothing asserted the property the mechanism exists to provide.

What is asserted here is the PROPERTY, not the mechanism: for a causal layer,
every position ``i > 0`` must attend to at least one EARLIER position. That is
derivable from what the layer is for, without reference to how the index set is
built, so a reimplementation that closes the hole differently still passes.

The self-dependency has to be excluded, and that is the whole difficulty. A first
version of this guard asserted "every output depends on at least one input" and
was VACUOUS: every position depends on itself through the query projection, so it
passed under the mutation too (15 passed, 0 failed). Only the OFF-DIAGONAL support
discriminates.

Measured at ``N=64, L=3, p=4, top_k=8``, non-self contributors for positions
``0..14``::

    HEAD    : [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    mutant  : [0, 0, 0, 3, 4, 4, 4, 15, 15, 15, 15, 3, 4, 4, 4]

Positions 1 and 2 attend to nothing but themselves once the prefix is removed.
The mutant's output differs from HEAD by 3.16, so the mutation is not inert.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.lighthouse_attention import (
    LighthouseAttention,
)

DIM = 32
HEADS = 4
LEVELS = 3
POOLING = 4
TOP_K = 8
N = 64
PERTURBATION = 50.0
SENSITIVITY = 1e-5


def _support_matrix(layer, tokens):
    """``support[i, j]`` = how much output ``i`` moves when input ``j`` moves.

    A per-key perturbation probe: it measures the REALIZED dependency structure
    rather than reading the intended one out of the index arithmetic.
    """
    base = np.asarray(layer(tokens, training=False))
    support = np.zeros((N, N), dtype="float64")
    for j in range(N):
        perturbed = tokens.copy()
        perturbed[0, j, :] += PERTURBATION
        moved = np.abs(np.asarray(layer(perturbed, training=False)) - base)
        support[:, j] = moved.max(axis=-1)[0]
    return support


def _non_self_contributors(layer, tokens):
    """Contributor count per output position, EXCLUDING its own diagonal.

    The diagonal must be excluded or the count is 1 for every position no matter
    what the pyramid does -- see the module docstring.
    """
    support = _support_matrix(layer, tokens)
    np.fill_diagonal(support, 0.0)
    return (support > SENSITIVITY).sum(axis=1)


@pytest.fixture(name="layer")
def _layer():
    keras.utils.set_random_seed(0)
    return LighthouseAttention(
        dim=DIM,
        num_heads=HEADS,
        num_levels=LEVELS,
        pooling_factor=POOLING,
        top_k=TOP_K,
    )


@pytest.fixture(name="tokens")
def _tokens():
    return np.random.default_rng(0).normal(size=(1, N, DIM)).astype("float32")


def test_every_position_after_the_first_attends_to_something_earlier(
    layer, tokens
):
    """The regression itself: a coverage hole strands positions with only self."""
    contributors = _non_self_contributors(layer, tokens)
    # Position 0 legitimately has no earlier position to attend to.
    starved = [i for i in range(1, N) if contributors[i] == 0]
    assert not starved, (
        f"positions {starved} attend to NOTHING but themselves: the pyramid "
        "leaves a coverage hole. The coarsest level only reaches base positions "
        f"[{POOLING ** (LEVELS - 1) - 1}, {N - 1}], so the level-0 mandatory "
        "prefix is what covers the rest."
    )


def test_the_unreachable_prefix_is_specifically_covered(layer, tokens):
    """Target the exact positions the mechanism exists for.

    ``0 .. F - 2`` are the positions the coarsest level cannot reach. Asserting
    them by name means a change that covers most of the sequence but reopens this
    particular hole still fails.
    """
    fanout = POOLING ** (LEVELS - 1)
    prefix = list(range(1, min(fanout - 1, N)))
    contributors = _non_self_contributors(layer, tokens)
    starved = [i for i in prefix if contributors[i] == 0]
    assert not starved, (
        f"prefix positions {starved} are unreachable; these are exactly the "
        f"positions [0, {fanout - 2}] that the coarsest level cannot scatter to"
    )


def test_the_probe_can_detect_a_missing_contributor(layer, tokens):
    """Control: the probe must be able to observe absence at all.

    Without this, a probe whose sensitivity threshold was too high would report
    'no starved positions' for every implementation, correct or not.
    """
    contributors = _non_self_contributors(layer, tokens)
    assert contributors.max() > 1, (
        "no output depends on more than one input, so the probe is not resolving "
        "the dependency structure and the assertions above prove nothing"
    )
    assert contributors.min() < N, (
        "every output depends on every input, i.e. the layer is dense here and "
        "this configuration cannot exhibit a coverage hole"
    )
