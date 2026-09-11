"""The ONE statement of the ``f_gt`` / ``g`` contract, shared by both modules.

``synthetic_warp`` and ``uvdoc`` produce the same two arrays from completely
different machinery -- a closed-form warp composition and an interpolated
correspondence lattice -- and the whole point of the pair is that a training
pipeline can concatenate the two corpora without a per-corpus branch. That only
holds if the two agree on shape, dtype, units, channel order and domain.

Two copies of a comment saying so would not hold anything: comments do not run,
and the two files are 1,000 lines apart. This module is the assertion itself,
applied to both modules' output by ``test_uvdoc.py``. Its own ability to fail
is proven in ``test_backward_map_convention.py`` -- it carries no ``test_``
prefix precisely so pytest does not collect it as a suite.

The contract, stated once:

``f_gt`` (backward, Eq. 11)
    ``(H, W, 2)`` float32. Indexed by the RECTIFIED pixel grid; the value at
    ``[row, col]`` is the ``(x, y)`` ABSOLUTE PIXEL coordinate in the DISTORTED
    image that this rectified pixel is read from. Channel 0 is ``x``, which
    varies along the WIDTH.

``g`` (forward, Eq. 12)
    ``(H, W, 2)`` float32. Indexed by the DISTORTED pixel grid; the value at
    ``[row, col]`` is the ``(x, y)`` absolute pixel coordinate in the RECTIFIED
    image. Same channel order. Values may fall outside the frame -- off-page
    pixels have no true rectified coordinate.
"""

import numpy as np

#: Every convention assertion is run on a NON-SQUARE grid on purpose. A channel
#: swap, an (row, col)-instead-of-(x, y) ordering and a transposed map are ALL
#: invisible at H == W, and the port's training resolution is square 288x288 --
#: so a square fixture would pin nothing at all.
CONVENTION_HEIGHT = 61
CONVENTION_WIDTH = 97


def assert_is_backward_map(f_gt, height, width, source):
    """Assert one array satisfies the ``f_gt`` half of the contract.

    Args:
        f_gt: The array under test.
        height: Expected ``H`` (the rectified grid's height).
        width: Expected ``W``.
        source: Human-readable producer name, quoted in every failure.

    Raises:
        AssertionError: On any contract violation.
    """
    array = np.asarray(f_gt)
    assert array.shape == (height, width, 2), (
        f"{source}: f_gt must be (H, W, 2) = {(height, width, 2)}, got {array.shape}"
    )
    assert array.dtype == np.float32, (
        f"{source}: f_gt must be float32 (the loss's y_true dtype), got {array.dtype}"
    )
    assert np.isfinite(array).all(), f"{source}: f_gt holds non-finite values"

    # Units FIRST. A [0, 1]-normalised map also fails the span checks below,
    # but it fails them for the wrong stated reason -- and "the channels look
    # swapped" sends the next reader after a defect that is not there.
    assert array[..., 0].max() > 1.0 and array[..., 1].max() > 1.0, (
        f"{source}: f_gt must be in ABSOLUTE PIXELS, not normalised to [0, 1] "
        f"or [-1, 1]; maxima are {array[..., 0].max():.4f}, "
        f"{array[..., 1].max():.4f}"
    )

    # The swap check is a RANGE check, not a span check: a real page sits
    # INSIDE its frame (a UVDoc page covers ~56% of it), so "channel 0 must
    # span the width" is false for perfectly good data. What is always true is
    # that x lives in the width's range and y in the height's. On a non-square
    # grid that catches a swap outright; at H == W nothing can, which is why
    # CONVENTION_HEIGHT != CONVENTION_WIDTH and why every caller that has the
    # choice passes a non-square grid.
    margin = 0.05
    for channel, axis_name, extent, other in (
        (0, "x", width, height),
        (1, "y", height, width),
    ):
        reach = float(array[..., channel].max())
        floor = float(array[..., channel].min())
        assert reach <= extent * (1.0 + margin) + 1.0, (
            f"{source}: channel {channel} must be {axis_name} and live in "
            f"[0, {extent}); it reaches {reach:.2f}. On this {height}x{width} "
            f"grid that is what a channel swap looks like (the other axis is "
            f"{other})"
        )
        assert floor >= -extent * margin - 1.0, (
            f"{source}: channel {channel} ({axis_name}) starts at {floor:.2f}, "
            f"far outside the frame"
        )


def assert_is_forward_map(g, height, width, source):
    """Assert one array satisfies the ``g`` half of the contract.

    ``g`` is NOT range-checked against the frame: its whole job is to tell a
    caller where an off-page distorted pixel would land, which is legitimately
    outside ``[0, W) x [0, H)``.

    Args:
        g: The array under test.
        height: Expected ``H`` (the distorted grid's height).
        width: Expected ``W``.
        source: Human-readable producer name, quoted in every failure.

    Raises:
        AssertionError: On any contract violation.
    """
    array = np.asarray(g)
    assert array.shape == (height, width, 2), (
        f"{source}: g must be (H, W, 2) = {(height, width, 2)}, got {array.shape}"
    )
    assert array.dtype == np.float32, (
        f"{source}: g must be float32, got {array.dtype}"
    )
    assert np.isfinite(array).all(), (
        f"{source}: g holds non-finite values -- an unfilled inversion NaN "
        f"reaches the L_line term and poisons the loss"
    )
    x_span = float(array[..., 0].max() - array[..., 0].min())
    y_span = float(array[..., 1].max() - array[..., 1].min())
    assert x_span > 1.0 and y_span > 1.0, (
        f"{source}: g must be in absolute rectified pixels; spans are "
        f"{x_span:.4f}, {y_span:.4f}"
    )


def assert_same_backward_map_convention(first, second, first_name, second_name):
    """Assert two producers' ``f_gt`` arrays are interchangeable.

    Shape, dtype and -- the part a shape check cannot see -- which channel
    carries which axis, established by comparing each channel's span against
    the other producer's.

    Args:
        first: One producer's ``f_gt``.
        second: The other's, for the same ``(H, W)``.
        first_name: Name quoted in failures.
        second_name: Name quoted in failures.

    Raises:
        AssertionError: If the two contracts differ.
    """
    a = np.asarray(first)
    b = np.asarray(second)
    assert a.shape == b.shape, (
        f"{first_name} emits {a.shape}, {second_name} emits {b.shape}"
    )
    assert a.dtype == b.dtype, (
        f"{first_name} emits {a.dtype}, {second_name} emits {b.dtype}"
    )
    height, width = a.shape[:2]
    assert_is_backward_map(a, height, width, first_name)
    assert_is_backward_map(b, height, width, second_name)

    for channel, axis_name, extent in ((0, "x", width), (1, "y", height)):
        a_max = float(a[..., channel].max())
        b_max = float(b[..., channel].max())
        # A quarter of the extent: the two producers are compared on the SAME
        # geometry, so their reaches agree to a fraction of a pixel when the
        # contract holds. A looser gate (0.75) admitted a quarter-scale map.
        assert abs(a_max - b_max) < 0.25 * extent, (
            f"channel {channel} ({axis_name}) reaches {a_max:.2f} in "
            f"{first_name} but {b_max:.2f} in {second_name} on a "
            f"{height}x{width} grid -- the two are not in the same units or "
            f"the channels are ordered differently"
        )
