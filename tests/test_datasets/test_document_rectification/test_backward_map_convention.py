"""RED proofs for the shared convention instrument.

``backward_map_convention.py`` is the single place the ``f_gt`` / ``g`` contract
is stated as executable assertions, and ``test_uvdoc.py`` leans on it to prove
that ``synthetic_warp`` and ``uvdoc`` agree. An instrument that cannot fail
would make every one of those arms vacuous, so this module feeds it each defect
it exists to catch and requires an ``AssertionError`` -- specifically that, not
``Exception``: a ``TypeError`` from an instrument that indexed something it
should not have is the instrument crashing, not judging.
"""

import numpy as np
import pytest

from .backward_map_convention import (
    CONVENTION_HEIGHT,
    CONVENTION_WIDTH,
    assert_is_backward_map,
    assert_is_forward_map,
    assert_same_backward_map_convention,
)


def _healthy_backward_map(height=CONVENTION_HEIGHT, width=CONVENTION_WIDTH):
    """An identity-ish backward map that satisfies the contract."""
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    return np.stack([cols, rows], axis=-1).astype(np.float32)


class TestTheInstrumentAcceptsAHealthyMap:
    """A guard that rejects everything is as useless as one that accepts it."""

    def test_a_healthy_backward_map_passes(self):
        assert_is_backward_map(
            _healthy_backward_map(), CONVENTION_HEIGHT, CONVENTION_WIDTH, "fixture"
        )

    def test_a_healthy_forward_map_passes(self):
        assert_is_forward_map(
            _healthy_backward_map(), CONVENTION_HEIGHT, CONVENTION_WIDTH, "fixture"
        )

    def test_two_healthy_maps_agree(self):
        assert_same_backward_map_convention(
            _healthy_backward_map(), _healthy_backward_map(), "a", "b"
        )


class TestTheInstrumentSeesEachDefect:
    """One arm per defect the contract exists to exclude."""

    def test_a_swapped_channel_order_is_rejected(self):
        """The defect with no shape, dtype or range symptom at H == W."""
        swapped = _healthy_backward_map()[..., ::-1].copy()
        with pytest.raises(AssertionError, match="channel swap"):
            assert_is_backward_map(
                swapped, CONVENTION_HEIGHT, CONVENTION_WIDTH, "swapped"
            )

    def test_a_normalised_map_is_rejected(self):
        """[0, 1] normalisation changes the loss's effective alpha silently."""
        normalised = _healthy_backward_map() / np.array(
            [CONVENTION_WIDTH, CONVENTION_HEIGHT], dtype=np.float32
        )
        with pytest.raises(AssertionError, match="ABSOLUTE PIXELS"):
            assert_is_backward_map(
                normalised, CONVENTION_HEIGHT, CONVENTION_WIDTH, "normalised"
            )

    def test_a_float64_map_is_rejected(self):
        with pytest.raises(AssertionError, match="float32"):
            assert_is_backward_map(
                _healthy_backward_map().astype(np.float64),
                CONVENTION_HEIGHT,
                CONVENTION_WIDTH,
                "float64",
            )

    def test_a_transposed_map_is_rejected(self):
        transposed = np.ascontiguousarray(
            np.transpose(_healthy_backward_map(), (1, 0, 2))
        )
        with pytest.raises(AssertionError, match=r"f_gt must be \(H, W, 2\)"):
            assert_is_backward_map(
                transposed, CONVENTION_HEIGHT, CONVENTION_WIDTH, "transposed"
            )

    def test_a_non_finite_map_is_rejected(self):
        broken = _healthy_backward_map()
        broken[3, 4, 0] = np.nan
        with pytest.raises(AssertionError, match="non-finite"):
            assert_is_backward_map(
                broken, CONVENTION_HEIGHT, CONVENTION_WIDTH, "nan"
            )

    def test_an_unfilled_inversion_nan_is_rejected_in_g(self):
        broken = _healthy_backward_map()
        broken[0, 0, 1] = np.nan
        with pytest.raises(AssertionError, match="non-finite"):
            assert_is_forward_map(
                broken, CONVENTION_HEIGHT, CONVENTION_WIDTH, "nan"
            )

    def test_a_map_in_the_wrong_scale_is_rejected(self):
        """Four times the frame is not "still pixels"; it is a different unit."""
        healthy = _healthy_backward_map()
        with pytest.raises(AssertionError, match=r"live in \[0,"):
            assert_is_backward_map(
                healthy * 4.0, CONVENTION_HEIGHT, CONVENTION_WIDTH, "quadruple"
            )

    def test_two_producers_that_each_pass_but_disagree_are_rejected(self):
        """Both halves are individually legal; only the comparison sees it."""
        healthy = _healthy_backward_map()
        shrunk = (healthy * 0.25).astype(np.float32) + 2.0
        assert_is_backward_map(
            shrunk, CONVENTION_HEIGHT, CONVENTION_WIDTH, "shrunk"
        )
        with pytest.raises(AssertionError, match="not in the same units"):
            assert_same_backward_map_convention(
                healthy, shrunk, "pixels", "quarter-scale"
            )
