"""``validate_spatial_extent`` and ``minimum_spatial_extent``, spatial-size guards shared by the two SqueezeNet model families.

Both `SqueezeNetV1` and `SqueezeNoduleNetV2` downsample with
`padding='valid'` at every stage, so a spatial axis of length `n` becomes
`(n - k) // s + 1` and can reach zero on a small input. Keras does not
raise on a zero-length spatial axis; the model still produces a
correctly shaped tensor, filled with NaN. This module walks the same
downsampling arithmetic ahead of construction, so the failure is a named
`ValueError` instead of a silent all-NaN forward pass.

`minimum_spatial_extent` computes the smallest legal input length by
simulating every downsampling stage, rather than a hard-coded constant,
since the two shipped stem families collapse at different sizes.
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
    """Reconstruct the ordered downsampling stages of a variant.

    :param variant_config: A `MODEL_VARIANTS` entry.
    :return: List of `(stage_name, kernel, stride, same_padding)`, mirroring
        what `_build_stem` and `_build_fire_modules` emit.
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
    """Compute the smallest per-axis input length that keeps every stage output >= 1.

    :param variant_config: A `MODEL_VARIANTS` entry.
    :return: The smallest legal per-axis extent.
    :raises ValueError: If no input up to `_MAX_SEARCH_EXTENT` survives.
    """
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
    """Raise `ValueError` if any spatial axis collapses to zero length.

    :param spatial: Spatial axis lengths of the input, channels excluded.
    :param variant_config: A `MODEL_VARIANTS` entry.
    :param model_label: Class name used in the error message.
    :raises ValueError: If a downsampling stage would produce a zero-length
        axis. The message names the stage, the axis, and the computed minimum.
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
