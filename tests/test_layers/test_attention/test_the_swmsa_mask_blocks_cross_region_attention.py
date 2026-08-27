"""Guard: the SW-MSA mask must stop a query attending across the cyclic wrap.

Why this test exists
--------------------
The additive shifted-window mask is the entire reason Swin's cyclic-shift
machinery exists: after the roll, one physical window contains tokens that were
NOT adjacent before the shift, and the mask is what keeps them from attending to
each other. Deleting the mask outright passes all 19 tests in
``test_progressive_focused_attention.py``, and the deletion is not inert -- it
moves the output by 1.22 at ``shift_size=2, window_size=4``.

No existing test builds a shifted-window instance and checks the boundary, so a
regression here -- for example a change that desynchronised
``_compute_attention_mask``'s window partition from ``_window_partition``, which
the class docstring flags as a MANUALLY maintained invariant with no shared code
-- would silently corrupt every SW-MSA layer.

The geometry, worked out once so the assertion is not magic
-----------------------------------------------------------
At ``window_size=4, shift_size=2, H=W=8`` the layer rolls by ``(-2, -2)``, so
``shifted[i, j] = x[(i + 2) % 8, (j + 2) % 8]``, and rolls the output back by
``(+2, +2)``, so ``final[i, j] = pre_rollback[(i - 2) % 8, (j - 2) % 8]``.

Shifted-space query ``(4, 4)`` and shifted-space key ``(6, 6)`` land in the SAME
physical window but in DIFFERENT pre-shift regions, so the mask assigns them
``-100``. They correspond to original input positions ``(6, 6)`` for the query --
which surfaces in the final output at ``[..., 6, 6, :]`` -- and ``(0, 0)`` for the
key. So perturbing input ``(0, 0)`` must leave output ``(6, 6)`` unmoved.

Why this can fail if the implementation is wrong: with the mask deleted the delta
at ``(6, 6)`` measures ``6.0e-3``, four orders of magnitude above the ``1e-6``
bound, while the whole-output sanity delta stays at ``56.44`` either way. Both
numbers matter -- the sanity bound is what stops this passing vacuously if the
perturbation ever stops reaching the network at all.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.progressive_focused_attention import (
    ProgressiveFocusedAttention,
)

H = W = 8
DIM = 4
WINDOW = 4
SHIFT = 2

# The query whose value must not move, in ORIGINAL input coordinates.
GUARDED_QUERY = (6, 6)
# The key it is masked away from, in ORIGINAL input coordinates.
MASKED_KEY = (0, 0)


def _unwrap(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@pytest.fixture(name="layer")
def _layer():
    keras.utils.set_random_seed(0)
    # `use_lepe=False` is REQUIRED, not incidental. LePE is a 3x3 depthwise
    # convolution applied to V inside the window; after the cyclic shift it mixes
    # neighbouring shifted-space positions, so it carries the perturbation from
    # the masked key to the guarded query through a path that is NOT attention.
    # Leaving it on makes this guard fail on correct code -- measured: the
    # guarded output moves from -0.803 to -5.616 with LePE enabled.
    return ProgressiveFocusedAttention(
        dim=DIM,
        num_heads=1,
        window_size=WINDOW,
        shift_size=SHIFT,
        use_lepe=False,
    )


@pytest.fixture(name="feature_map")
def _feature_map():
    return np.random.RandomState(0).normal(size=(1, H, W, DIM)).astype("float32")


def test_a_masked_key_does_not_move_its_guarded_query(layer, feature_map):
    """The regression itself."""
    before = np.asarray(_unwrap(layer(feature_map, training=False)))

    perturbed = feature_map.copy()
    perturbed[0, MASKED_KEY[0], MASKED_KEY[1], :] += 100.0
    after = np.asarray(_unwrap(layer(perturbed, training=False)))

    # Sanity first: without this, a perturbation that reached nothing at all
    # would satisfy the real assertion vacuously.
    assert np.abs(before - after).max() > 1.0, (
        "the perturbation did not move the output anywhere, so the guard below "
        "would pass for the wrong reason"
    )

    guarded_before = before[0, GUARDED_QUERY[0], GUARDED_QUERY[1], :]
    guarded_after = after[0, GUARDED_QUERY[0], GUARDED_QUERY[1], :]
    np.testing.assert_allclose(
        guarded_before,
        guarded_after,
        atol=1e-6,
        rtol=0,
        err_msg=(
            f"output at {GUARDED_QUERY} responded to a perturbation at "
            f"{MASKED_KEY}: the two are in the same physical window after the "
            "cyclic shift but different pre-shift regions, so the SW-MSA mask "
            "must separate them"
        ),
    )


def test_an_unmasked_key_does_move_its_query(layer, feature_map):
    """The other side: the mask must not block everything.

    A mask stuck at -100 everywhere would pass the test above trivially.
    """
    before = np.asarray(_unwrap(layer(feature_map, training=False)))

    perturbed = feature_map.copy()
    perturbed[0, GUARDED_QUERY[0], GUARDED_QUERY[1], :] += 100.0
    after = np.asarray(_unwrap(layer(perturbed, training=False)))

    guarded_delta = np.abs(
        before[0, GUARDED_QUERY[0], GUARDED_QUERY[1], :]
        - after[0, GUARDED_QUERY[0], GUARDED_QUERY[1], :]
    ).max()
    assert guarded_delta > 1.0, (
        f"a token's own position moved by only {guarded_delta}: attention is "
        "blocked everywhere, not just across the wrap"
    )


def test_the_unshifted_layer_has_no_mask(layer, feature_map):
    """`shift_size=0` is W-MSA and must build no mask at all."""
    keras.utils.set_random_seed(0)
    unshifted = ProgressiveFocusedAttention(
        dim=DIM, num_heads=1, window_size=WINDOW, shift_size=0, use_lepe=False
    )
    unshifted(feature_map, training=False)
    assert unshifted._attn_mask is None
