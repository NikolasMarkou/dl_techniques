"""Spatial-extent guard for AlexNet.

Every convolution in this architecture uses ``padding='same'`` except the final
pool, which is ``'valid'``. That last stage still shrinks the feature map, so a
spatial axis of length ``n`` becomes ``(n - 3) // 2 + 1`` and can reach zero on a
small input. Keras does not raise on a zero-length spatial axis: the model still
returns a correctly shaped tensor, filled with NaN, and the failure surfaces
several steps later as an unexplained all-NaN loss.

This module walks the same downsampling arithmetic ahead of construction so the
failure is a named ``ValueError`` instead. :func:`minimum_spatial_extent`
computes the smallest legal input by simulating each stage rather than hard-coding
a constant, so the floor cannot drift away from the architecture if the layer table
is ever edited.

Adapted from ``models/vision/squeezenet/spatial_guard.py``, which solves the same
problem for an all-``'valid'`` variant. The simulation here is per-stage rather
than driven by a ``MODEL_VARIANTS`` entry because this package has no variant
table -- AlexNet is a single architecture (see ``models/AGENTS.md``, "When the
shape does not apply").
"""

from typing import Optional, Tuple

# ---------------------------------------------------------------------

#: Upper bound for the minimum-extent search. The shipped architecture resolves at
#: 55, so this cap exists only so a pathological caller cannot hang.
_MAX_SEARCH_EXTENT = 8192

#: ``(stage_name, kernel, stride, pad, floor)`` for every spatial-affecting stage,
#: in order. This is the single source of truth: ``AlexNet._build_features`` builds
#: its layers from the same numbers and the guard reads them from here, so the two
#: cannot disagree.
#:
#: ``pad`` is an EXPLICIT symmetric pad applied by a ``ZeroPadding2D`` in front of
#: the layer, and ``floor`` is whether the result is floored at zero. These are not
#: interchangeable with ``padding='same'``, which is the trap this table exists to
#: record:
#:
#:   - Keras/TensorFlow ``'same'`` computes ``pad_total = max((out-1)*stride +
#:     kernel - in, 0)`` and puts ``floor(pad_total/2)`` BEFORE, so an odd total
#:     splits unevenly and the output is ``ceil(in/stride)``. Measured at input 227,
#:     ``'same'`` gives conv1 **57**, not 56.
#:   - Caffe pads by a fixed symmetric amount, which is what the released AlexNet
#:     model does and what the paper's Figure 3 requires.
#:
#: Measured at input 227, explicit pads: conv1 56, pool1 28, conv2 26, pool2 13,
#: conv3-5 13, pool3 6 -- the ``6 x 6 x 256`` of Figure 3 and the 9216 fc6 inputs.
#: Using ``'same'`` throughout instead gives 8 -> 7x7x256 -> 12544, which is NOT
#: the paper. The asymmetry of the released model (padded pools, unpadded final
#: pool) is exactly what produces the paper's figure.
STAGES: Tuple[Tuple[str, int, int, int, bool], ...] = (
    ("conv1", 11, 4, 2, False),
    ("pool1", 3, 2, 1, False),
    ("conv2", 5, 1, 1, False),
    ("pool2", 3, 2, 1, False),
    ("conv3", 3, 1, 1, False),
    ("conv4", 3, 1, 1, False),
    ("conv5", 3, 1, 1, False),
    ("pool3", 3, 2, 0, False),
)

# ---------------------------------------------------------------------


def final_feature_extent(size: int) -> int:
    """Extent of the feature map entering the flatten, after EVERY stage.

    One call to :func:`conv_out` is NOT this. ``conv_out(227, 3, 2, 0)`` is 113 --
    a single 3x3/2 over a 227 input -- whereas the model walks 227 -> 56 -> 28 ->
    26 -> 13 -> 13 -> 13 -> 13 -> **6**. Applying only the last stage's parameters
    to the original input was a measured bug in ``_build_classifier``, which sized
    fc6 off a 113-wide flatten instead of a 6-wide one.

    :param size: Input length along one spatial axis.
    :type size: int
    :return: The extent entering the flatten. 6 at input 227.
    :rtype: int
    """
    current = size
    for _, kernel, stride, pad, _ in STAGES:
        current = conv_out(current, kernel, stride, pad)
    return current


def conv_out(size: int, kernel: int, stride: int, pad: int = 0) -> int:
    """Output length of one stage under an EXPLICIT symmetric pad, floored at zero.

    Note this is deliberately NOT Keras' ``padding='same'``, which pads by
    ``ceil(in/stride)*stride - in + kernel - stride`` split with ``floor`` before and
    the remainder after. See :data:`STAGES` for the measured difference.

    :param size: Input length along the axis.
    :type size: int
    :param kernel: Kernel extent.
    :type kernel: int
    :param stride: Stride.
    :type stride: int
    :param pad: Symmetric padding added to BOTH ends before the kernel slides.
    :type pad: int
    :return: The output length, never negative.
    :rtype: int
    """
    return max(0, (size + 2 * pad - kernel) // stride + 1)


def minimum_spatial_extent() -> int:
    """Smallest per-axis input length that keeps every stage's output >= 1.

    Measured: **55** for the shipped stage table.

    :return: The smallest legal per-axis extent.
    :rtype: int
    :raises ValueError: If no input up to ``_MAX_SEARCH_EXTENT`` survives.
    """
    for size in range(1, _MAX_SEARCH_EXTENT + 1):
        current = size
        for _, kernel, stride, pad, _ in STAGES:
            current = conv_out(current, kernel, stride, pad)
            if current < 1:
                break
        else:
            return size
    raise ValueError(
        f"No input smaller than {_MAX_SEARCH_EXTENT} survives this architecture's "
        f"stages: {[s[0] for s in STAGES]}"
    )


def conv5_output_extent(spatial: Tuple[Optional[int], ...]) -> Optional[int]:
    """Feature-map extent entering the flatten, for a square-agnostic input.

    :param spatial: Per-axis input lengths, channels excluded.
    :type spatial: Tuple[Optional[int], ...]
    :return: The pool3 output length, or ``None`` if any axis is unknown.
    :rtype: Optional[int]
    """
    current: Optional[int] = None
    for axis, size in enumerate(spatial):
        if size is None:
            return None
        axis_size = size
        for _, kernel, stride, same_padding in STAGES:
            axis_size = conv_out(axis_size, kernel, stride, same_padding)
        current = axis_size if current is None else min(current, axis_size)
    return current


def validate_spatial_extent(
        spatial: Tuple[Optional[int], ...],
        model_label: str = "AlexNet",
) -> None:
    """Raise ``ValueError`` if any spatial axis collapses to zero length.

    :param spatial: Spatial axis lengths of the input, channels excluded.
    :type spatial: Tuple[Optional[int], ...]
    :param model_label: Class name used in the error message.
    :type model_label: str
    :raises ValueError: If a stage would produce a zero-length axis. The message
        names the stage, the axis and the computed minimum.
    """
    for axis, size in enumerate(spatial):
        if size is None:
            continue
        current = size
        for stage_name, kernel, stride, pad, _ in STAGES:
            current = conv_out(current, kernel, stride, pad)
            if current < 1:
                floor = minimum_spatial_extent()
                raise ValueError(
                    f"{model_label}: input spatial axis {axis} of length {size} "
                    f"collapses to length 0 at stage '{stage_name}'. The final "
                    f"pool is padding='valid', so a zero-length axis yields an "
                    f"all-NaN output of the correct shape rather than an error. "
                    f"The minimum legal spatial extent is {floor}; got "
                    f"{tuple(spatial)}."
                )