"""``forward_cond_scale`` touches the forward direction and nothing else.

Invariant 5 of the plan, plus the per-sample ``direction`` contract (D-005) it
rests on.

Upstream multiplies the RAW ``x_cond`` pixels by ``forward_cond_scale`` before
patch embedding, and does so only on the ``not reverse`` branch
(``dit.py:533-536``). In this port both conditioning embedders always run and
``keras.ops.where`` selects, so "only on the forward branch" is no longer
enforced by a Python ``if``: it is enforced by which tensor is handed to which
embedder. That makes applying the scale to both branches a one-word edit with no
shape symptom, no config symptom and no finiteness symptom -- the reverse output
simply becomes a different plausible number.

The arms below pin it from the reverse side, where the correct answer is
**bit-identical** (``atol = 0``) across values of the scale, and from the forward
side, where it must actually move. Both directions are needed: an
implementation that ignored ``forward_cond_scale`` entirely would pass the
reverse arm alone.

The second class pins the property that makes a mixed-direction batch legitimate
at all: flipping one sample's ``direction`` must move that sample's row and move
every other row by exactly ``0.0``. Step 1's mechanism probe measured exactly
that on a throwaway two-branch model; this is the same measurement on the real
one.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import DiTXA

from ._ditxa_helpers import activate, batch, np_

#: Two clearly distinct scales. ``64.0`` is upstream's own worked example.
SCALES = (1.0, 64.0)


@pytest.fixture(scope="module")
def model() -> DiTXA:
    """A built, non-degenerate ``tiny`` model.

    ``forward_cond_scale`` is a plain Python float read inside ``call()``, so
    the arms below mutate it on ONE model rather than comparing two independently
    initialised ones. That removes the confound entirely: any difference observed
    is the scale's, never the initializer's.
    """
    m = DiTXA.from_variant("tiny", forward_cond_scale=SCALES[0], label_seed=5)
    m(batch(m, batch_size=2))
    return activate(m, seed=4)


def _run(model: DiTXA, data, scale: float) -> np.ndarray:
    """Forward pass at a given ``forward_cond_scale``, restoring it afterwards."""
    previous = model.forward_cond_scale
    try:
        model.forward_cond_scale = float(scale)
        return np_(model(data, training=False))
    finally:
        model.forward_cond_scale = previous


class TestTheScaleIsScopedToTheForwardDirection:
    """Reverse is invariant to it; forward is not."""

    def test_the_reverse_output_is_bit_identical_across_scales(self, model):
        """``atol = 0``, both samples reverse."""
        data = batch(model, batch_size=4, seed=21, direction=[1.0] * 4)
        first = _run(model, data, SCALES[0])
        second = _run(model, data, SCALES[1])
        assert np.array_equal(first, second), (
            "forward_cond_scale must not reach the reverse direction. "
            f"max|delta| between scale {SCALES[0]} and {SCALES[1]} = "
            f"{np.abs(first - second).max():.10g}. The usual cause is feeding "
            "the scaled x_cond to BOTH conditioning embedders instead of only "
            "to cond_embedder_forward."
        )

    def test_the_forward_output_moves_with_the_scale(self, model):
        """Anti-vacuity for the arm above.

        An implementation that dropped ``forward_cond_scale`` on the floor would
        satisfy the reverse arm perfectly. This is what makes the pair
        discriminating rather than merely satisfiable.
        """
        data = batch(model, batch_size=4, seed=21, direction=[0.0] * 4)
        first = _run(model, data, SCALES[0])
        second = _run(model, data, SCALES[1])
        delta = np.abs(first - second).max()
        assert delta > 1e-4, (
            "forward_cond_scale changed the forward-direction output by "
            f"{delta:.10g}, i.e. not at all: the knob is dead and the "
            "reverse-invariance arm above proves nothing"
        )

    def test_a_mixed_batch_scopes_the_scale_row_by_row(self, model):
        """The two claims at once, in one batch.

        Rows 0 and 2 are forward, rows 1 and 3 reverse. Changing the scale must
        move exactly the forward rows and move the reverse rows by ``0.0``.
        """
        directions = [0.0, 1.0, 0.0, 1.0]
        data = batch(model, batch_size=4, seed=22, direction=directions)
        first = _run(model, data, SCALES[0])
        second = _run(model, data, SCALES[1])
        per_row = np.abs(first - second).reshape(4, -1).max(axis=1)
        assert per_row[1] == 0.0 and per_row[3] == 0.0, (
            f"reverse rows moved when forward_cond_scale changed: {per_row}"
        )
        assert per_row[0] > 1e-4 and per_row[2] > 1e-4, (
            f"forward rows did not move: {per_row}"
        )


class TestDirectionIsPerSample:
    """``direction`` is a tensor, and it is local to its own row."""

    def test_flipping_one_samples_direction_moves_only_that_row(self, model):
        """Exactly ``0.0`` on every other row.

        This is the property that makes a mixed-direction batch a legitimate
        data setting rather than an approximation: without it, forward-only and
        reverse-only training would not be reproducible from a mixed batch.
        """
        base = [0.0, 0.0, 0.0, 0.0]
        data = batch(model, batch_size=4, seed=23, direction=base)
        before = np_(model(data, training=False))

        flipped = list(base)
        flipped[2] = 1.0
        after = np_(
            model({**data, "direction": np.array(flipped, "float32")}, training=False)
        )

        per_row = np.abs(before - after).reshape(4, -1).max(axis=1)
        assert per_row[2] > 1e-4, (
            f"flipping row 2's direction changed nothing: {per_row}. Either the "
            "flag is ignored or the two conditioning embedders are identical."
        )
        for row in (0, 1, 3):
            assert per_row[row] == 0.0, (
                f"row {row} moved by {per_row[row]:.10g} when only row 2's "
                "direction changed; the selection is leaking across the batch axis"
            )

    def test_a_mixed_batch_equals_two_single_direction_runs(self, model):
        """Row-for-row, at ``atol = 0``.

        The strongest available statement of D-005: the per-sample selection is
        not an approximation of two specialised call paths, it reproduces them
        exactly.
        """
        data = batch(model, batch_size=4, seed=24, direction=[0.0, 1.0, 1.0, 0.0])
        mixed = np_(model(data, training=False))
        all_forward = np_(
            model({**data, "direction": np.zeros((4,), "float32")}, training=False)
        )
        all_reverse = np_(
            model({**data, "direction": np.ones((4,), "float32")}, training=False)
        )
        expected = np.stack(
            [all_forward[0], all_reverse[1], all_reverse[2], all_forward[3]]
        )
        assert np.array_equal(mixed, expected), (
            "a mixed-direction batch must equal the per-row single-direction "
            f"runs; max|delta| = {np.abs(mixed - expected).max():.10g}"
        )
        assert np.abs(all_forward - all_reverse).max() > 1e-4, (
            "the two directions produce the same output, so the arm above is "
            "vacuous"
        )

    def test_both_conditioning_embedders_carry_their_own_weights(self, model):
        """Structural: D-005 requires BOTH to be built, always.

        A ``build()`` that materialised only the branch the first traced call
        happened to take would produce a weight tree that silently disagrees with
        the graph -- and, being shape-compatible, would reload into the wrong
        slot rather than raise.
        """
        forward_kernel = np_(model.cond_embedder_forward.proj.kernel)
        reverse_kernel = np_(model.cond_embedder_reverse.proj.kernel)
        assert forward_kernel.shape == reverse_kernel.shape
        assert not np.array_equal(forward_kernel, reverse_kernel), (
            "the two conditioning embedders hold identical weights; they are "
            "either the same object or share an Initializer instance"
        )
        assert model.cond_embedder_forward.built
        assert model.cond_embedder_reverse.built
