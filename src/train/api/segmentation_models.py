"""The segmentation model families of ``python -m train``: adapters, variants and the size rule.

The two U-Nets of the library do not take the arguments the API calls a factory with
(``factory(variant=, num_classes=, input_shape=)``): ConvUNext has no ``num_classes``
(it takes ``output_channels``) and AccUNet takes ``(input_channels, num_classes, (H, W))`` and has no
variants table. The adapters below close that gap, and nothing else; the models are not
edited. Importing this module imports no model code (D-012): each adapter imports its model
inside the call.

Interface contract
------------------
``TASK``
    ``"image_segmentation"``, the task these families train.

``create_convunext(*, variant, num_classes, input_shape, **model_args)``
    :param variant: a key of ``CONVUNEXT_CONFIGS`` (``tiny`` .. ``xlarge``).
    :param num_classes: output channels; the logits have one channel per class.
    :param input_shape: ``(height, width, channels)``.
    :param model_args: further keywords of ``create_convunext_variant`` (``depth``,
        ``initial_filters``, ``drop_path_rate`` ...).
    :return: the functional model, linear logits ``(B, H, W, num_classes)``, built with
        ``use_bias=True`` and ``final_activation="linear"`` exactly as the legacy trainer
        builds it (``train.convunext.common``: a softmax head under ``from_logits=True`` would
        softmax twice, and a bias-free stem is the denoiser's arm).
    :raises ValueError: when ``model_args`` names ``use_bias``, ``final_activation`` or
        ``output_channels`` (set here, not overridable), or the model refuses the call.

``create_accunet(*, variant, num_classes, input_shape, **model_args)``
    :param variant: a key of :data:`ACCUNET_VARIANTS`.
    :param num_classes: output channels, greater than 1.
    :param input_shape: ``(height, width, channels)``; ``channels`` becomes ``input_channels``.
    :param model_args: further keywords of ``create_acc_unet_multiclass`` (they replace the
        variant's ``base_filters`` and ``mlfc_iterations``).
    :return: the functional model, linear logits ``(B, H, W, num_classes)``; it forces
        ``jit_compile=False`` on every compile path (``AccUNetFunctional``).
    :raises ValueError: for an unknown variant, ``input_channels`` in ``model_args`` (it is the
        last axis of ``input_shape``), or a refusal by the model.

``ACCUNET_VARIANTS``
    ``{variant: factory keywords}``. AccUNet has no table of its own; ``base`` is what the
    legacy trainer (``train.accunet.train_accunet``) builds: 32 filters, 3 MLFC iterations.

``convunext_stride(variant, model_args)`` / ``ACCUNET_STRIDE``
    The multiple each side of the input must be: ``2 ** depth`` for ConvUNext, where ``depth``
    is the variant's (3 for ``tiny`` and ``small``, 4 for ``base`` and ``large``, 5 for
    ``xlarge``) unless ``model_args`` sets it; 16 for AccUNet (four stride-2 pools).

``check_image_size(spec, variant, model_args, image_size)``
    :param spec: the family's :class:`~train.api.registry.ModelSpec`; it must have a ``stride``.
    :param variant: variant name.
    :param model_args: the caller's ``model_args`` (a ConvUNext ``depth`` changes the stride).
    :param image_size: an int (a square) or ``[height, width]``.
    :return: ``(height, width)``.
    :raises ConfigError: naming ``transform_args.image_size``, the side, the stride and the two
        nearest sizes that satisfy it; also for a value that is not an int or a pair of ints
        above zero.
    :raises ValueError: when ``spec`` has no ``stride`` (a classifier was passed: a bug in the
        caller, not a request).

``sides_of(image_size)``
    :param image_size: an int (a square) or a pair ``[height, width]``.
    :return: ``(height, width)``, the one reading of a segmentation ``image_size`` that
        :func:`check_image_size` and ``train.api.segmentation_args.SegTransformArgs`` share.
    :raises ConfigError: naming ``transform_args.image_size`` for a value that is no positive int
        or pair of positive ints (a bool is no int here).

``builtin_segmentation_models()``
    :return: the two :class:`~train.api.registry.ModelSpec` rows (``convunext``, ``accunet``),
        freshly built. NOT registered by :func:`train.api.recipes.register_builtin_models`: the
        task that trains them registers them (step 28), and the classification registry, which
        a dozen guards enumerate, holds the six classifiers only.

Measured (``plans/.../scratch_step23/logs/p1_build.txt``, CPU)
--------------------------------------------------------------
Both families build and run at non-square sizes with the output the size of the input:
ConvUNext ``tiny`` at 128x256, 256x128, 96x160, 32x64; ``base`` at 128x256, 64x128; ``xlarge`` at
64x128, 96x192, 160x160; AccUNet at 128x256, 256x128, 64x128, 48x80. AccUNet REFUSES a side that is
not a multiple of 16 (100x128, 50x50: ``ValueError`` at the first call). ConvUNext does NOT: it
builds and returns the input's size at sizes that are no multiple of ``2 ** depth`` (50x50, 36x64,
100x128, 80x128 for depth 5), because its decoder resizes an upsampled map to its skip's size
(``create_convunext``, "Odd input sizes make the upsampled dims miss the skip's"). The ``2 ** depth``
rule is therefore a choice of this task, not a limit of the model: a resized map joins its skip
shifted by a fraction of a pixel, and the task trains on skips that meet.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only (D-012).
"""

from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from .errors import ConfigError
from .registry import ModelSpec

# ---------------------------------------------------------------------

TASK = "image_segmentation"

_HERE = "train.api.segmentation_models"
_CONVUNEXT = "dl_techniques.models.vision.convunext.model"

#: AccUNet's four stride-2 max pools (``AccUNet`` requires H and W divisible by 16).
ACCUNET_STRIDE = 16

#: What ``train.accunet.train_accunet`` builds by default (``base_filters`` :93, ``mlfc_iterations`` :94).
ACCUNET_VARIANTS: Dict[str, Dict[str, Any]] = {
    "base": {"base_filters": 32, "mlfc_iterations": 3},
}

#: Keys of ``create_convunext_variant`` this adapter sets itself and does not let a caller change.
#: (``variant``, ``num_classes`` and ``input_shape`` are named parameters of the adapter, so Python
#: refuses a second copy of them; ``recipes._merged`` names them to the user before that.)
_CONVUNEXT_FIXED = ("use_bias", "final_activation", "output_channels")

#: Keys of ``create_acc_unet_multiclass`` this adapter supplies from the data.
_ACCUNET_FIXED = ("input_channels",)


def _refuse_fixed(family: str, model_args: Mapping[str, Any], fixed: Sequence[str]) -> None:
    clash = [key for key in fixed if key in model_args]
    if clash:
        raise ValueError(
            f"{', '.join(map(repr, clash))} is set by the {family!r} adapter from model=, the "
            "data and transform_args (a softmax or biased-off head would break the logits loss); "
            "remove it from model_args"
        )


def _three_sides(input_shape: Sequence[int]) -> Tuple[int, int, int]:
    if len(input_shape) != 3:
        raise ValueError(f"input_shape must be (height, width, channels), got {tuple(input_shape)!r}")
    height, width, channels = (int(side) for side in input_shape)
    return height, width, channels


def create_convunext(
    *, variant: str, num_classes: int, input_shape: Sequence[int], **model_args: Any
) -> Any:
    """Build a ConvUNext segmenter; see the module contract."""
    _refuse_fixed("convunext", model_args, _CONVUNEXT_FIXED)
    height, width, channels = _three_sides(input_shape)
    from dl_techniques.models.vision.convunext.model import create_convunext_variant

    return create_convunext_variant(
        variant,
        (height, width, channels),
        use_bias=True,
        output_channels=num_classes,
        final_activation="linear",
        **model_args,
    )


def create_accunet(
    *, variant: str, num_classes: int, input_shape: Sequence[int], **model_args: Any
) -> Any:
    """Build an AccUNet segmenter; see the module contract."""
    _refuse_fixed("accunet", model_args, _ACCUNET_FIXED)
    if variant not in ACCUNET_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Available variants: {sorted(ACCUNET_VARIANTS)}"
        )
    height, width, channels = _three_sides(input_shape)
    from dl_techniques.models.vision.accunet.model import create_acc_unet_multiclass

    return create_acc_unet_multiclass(
        input_channels=channels,
        num_classes=num_classes,
        input_shape=(height, width),
        **{**ACCUNET_VARIANTS[variant], **model_args},
    )


def convunext_stride(variant: str, model_args: Mapping[str, Any]) -> int:
    """Return ``2 ** depth`` for a ConvUNext variant; see the module contract."""
    depth = model_args.get("depth")
    if depth is None:
        from dl_techniques.models.vision.convunext.model import CONVUNEXT_CONFIGS

        depth = CONVUNEXT_CONFIGS[variant]["depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ConfigError(
            f"Invalid value for 'model_args.depth': a ConvUNext depth is an int of 1 or more, "
            f"got {depth!r}"
        )
    return 2**depth


def sides_of(image_size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    """Return ``(height, width)`` of an int or a pair of ints above zero (a bool is no int here)."""
    sides = (
        [image_size, image_size]
        if isinstance(image_size, int)
        else list(image_size) if isinstance(image_size, (list, tuple)) else None
    )
    if (
        sides is None
        or len(sides) != 2
        or not all(isinstance(side, int) and not isinstance(side, bool) and side > 0 for side in sides)
    ):
        raise ConfigError(
            "Invalid value for 'transform_args.image_size': a segmentation image_size is one "
            f"positive int (a square) or [height, width] of two, got {image_size!r}"
        )
    return sides[0], sides[1]


def check_image_size(
    spec: ModelSpec,
    variant: str,
    model_args: Mapping[str, Any],
    image_size: Union[int, Sequence[int]],
) -> Tuple[int, int]:
    """Return ``(height, width)`` or refuse a side the family cannot take; see the module contract."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-072: the rule is `side % stride == 0` for EVERY
    # segmentation family, ConvUNext included although it builds and returns the input's size
    # without it (measured, the module docstring). Do NOT relax it to "whatever builds": its decoder
    # resizes an upsampled map to the skip's size, so at a side that is no multiple of `2 ** depth`
    # the skip and the map are shifted by a fraction of a pixel and the run would train on that.
    stride = spec.stride_of(variant, model_args)
    if stride is None:
        raise ValueError(
            f"the family {spec.family!r} has no stride rule; check_image_size is for "
            "segmentation families"
        )
    height, width = sides_of(image_size)
    for name, side in (("height", height), ("width", width)):
        if side % stride:
            below = side // stride * stride
            nearest = f"{below} or {below + stride}" if below else str(stride)
            raise ConfigError(
                f"Invalid value for 'transform_args.image_size': the {name} {side} is not a multiple "
                f"of {stride}, which {spec.family}/{variant} needs so that its skip connections "
                f"meet the upsampled maps; use {nearest}"
            )
    return height, width


def builtin_segmentation_models() -> Tuple[ModelSpec, ...]:
    """Return the segmentation family specs, freshly built; see the module contract."""
    return (
        ModelSpec(
            family="convunext",
            factory=f"{_HERE}:create_convunext",
            variants_attr=f"{_CONVUNEXT}:CONVUNEXT_CONFIGS",
            tasks=(TASK,),
            stride=f"{_HERE}:convunext_stride",
        ),
        ModelSpec(
            family="accunet",
            factory=f"{_HERE}:create_accunet",
            variants_attr=f"{_HERE}:ACCUNET_VARIANTS",
            tasks=(TASK,),
            stride=ACCUNET_STRIDE,
        ),
    )
