"""
Spatial-extent validation shared by the two SqueezeNet model families.

Both :class:`SqueezeNetV1` and :class:`SqueezeNoduleNetV2` downsample with
``padding='valid'`` at every stage: a strided stem convolution followed by three
``pool_size=3, strides=2`` max-pooling stages. Under valid padding a spatial axis
of length ``n`` becomes ``(n - k) // s + 1``, which reaches **zero** for small
inputs -- and Keras 3.8 does not raise on a zero-length spatial axis. The model
still produces a correctly *shaped* tensor, filled entirely with ``NaN`` (a
``Conv2D`` head over an empty map feeds ``GlobalAveragePooling2D`` a ``0/0``,
and the final softmax propagates it).

This module walks that same arithmetic ahead of construction so the failure is a
named ``ValueError`` at call time instead of a silent all-NaN forward pass.

Interface contract
------------------
``validate_spatial_extent(spatial, variant_config, model_label)``
    ``spatial`` is the tuple of spatial axis lengths (channels excluded);
    ``variant_config`` is an entry of a model's ``MODEL_VARIANTS`` (it is read
    for ``conv1_kernel``, ``conv1_stride`` and ``pool_indices`` only, so it works
    unchanged for 2D and 3D variants). Returns ``None`` on success and raises
    ``ValueError`` naming the first collapsing stage, the axis, and the computed
    minimum legal extent for that exact variant. ``model_label`` only decorates
    the message.

``minimum_spatial_extent(variant_config)``
    Returns the smallest per-axis length that survives every downsampling stage
    of ``variant_config``. It is *computed*, never tabulated: the two shipped
    stem families differ (35 for the 7x7/stride-2 stem with pools after fire4 and
    fire8; 31 for SqueezeNet "1.1"'s 3x3 stem with pools after fire3 and fire5),
    and a hard-coded constant would be wrong for one of them.
"""

from typing import Any, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------

# Upper bound for the minimum-extent search. Every shipped variant resolves far
# below this; the cap exists only so a pathological variant_config cannot hang.
_MAX_SEARCH_EXTENT = 8192


def _conv_out(size: int, kernel: int, stride: int, same_padding: bool) -> int:
    """Output length of one downsampling stage, floored at zero."""
    if same_padding:
        return -(-size // stride)
    return max(0, (size - kernel) // stride + 1)


def _downsampling_stages(
        variant_config: Dict[str, Any]
) -> List[Tuple[str, int, int, bool]]:
    """
    Reconstruct the ordered downsampling stages of a variant.

    Returns a list of ``(stage_name, kernel, stride, same_padding)`` mirroring
    exactly what ``_build_stem`` and ``_build_fire_modules`` emit.
    """
    conv1_stride = variant_config["conv1_stride"]
    pool_indices = variant_config["pool_indices"]

    stages: List[Tuple[str, int, int, bool]] = [
        ("conv1", variant_config["conv1_kernel"], conv1_stride, conv1_stride == 1)
    ]
    if 1 in pool_indices:
        stages.append(("maxpool1 (after conv1)", 3, 2, False))
    for idx in range(len(variant_config["fire_configs"])):
        fire_number = idx + 2
        if fire_number in pool_indices:
            stages.append((f"maxpool after fire{fire_number}", 3, 2, False))
    return stages


def minimum_spatial_extent(variant_config: Dict[str, Any]) -> int:
    """Smallest per-axis input length that keeps every stage output >= 1."""
    stages = _downsampling_stages(variant_config)
    for size in range(1, _MAX_SEARCH_EXTENT + 1):
        current = size
        for _, kernel, stride, same_padding in stages:
            current = _conv_out(current, kernel, stride, same_padding)
            if current < 1:
                break
        else:
            return size
    raise ValueError(
        f"No input smaller than {_MAX_SEARCH_EXTENT} survives this variant's "
        f"downsampling stages: {[s[0] for s in stages]}"
    )


def validate_spatial_extent(
        spatial: Sequence[int],
        variant_config: Dict[str, Any],
        model_label: str,
) -> None:
    """
    Raise ``ValueError`` if any spatial axis collapses to zero length.

    Args:
        spatial: Spatial axis lengths of the input, channels excluded.
        variant_config: A ``MODEL_VARIANTS`` entry.
        model_label: Class name used in the error message.

    Raises:
        ValueError: If a downsampling stage would produce a zero-length axis.
            The message names the stage, the axis, and the computed minimum.
    """
    stages = _downsampling_stages(variant_config)

    for axis, size in enumerate(spatial):
        if size is None:
            continue
        current = size
        for stage_name, kernel, stride, same_padding in stages:
            current = _conv_out(current, kernel, stride, same_padding)
            if current < 1:
                floor = minimum_spatial_extent(variant_config)
                raise ValueError(
                    f"{model_label}: input spatial axis {axis} of length {size} "
                    f"collapses to length 0 at stage '{stage_name}'. All "
                    f"downsampling stages use padding='valid', so a zero-length "
                    f"axis produces an all-NaN output of the correct shape "
                    f"rather than an error. The minimum legal spatial extent "
                    f"for this variant is {floor}; got {tuple(spatial)}."
                )
