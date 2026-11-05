"""The segmentation model families of ``python -m train``: adapters, variants and the size rule.

The two U-Nets of the library do not take the arguments the API calls a factory with
(``factory(variant=, num_classes=, input_shape=)``): ConvUNext has no ``num_classes``
(it takes ``output_channels``) and AccUNet takes ``(input_channels, num_classes, (H, W))`` and has no
variants table. The adapters below close that gap, and nothing else; the models are not
edited. Importing this module imports no model code (D-012): each adapter imports its model
inside the call.

The two backbone families (``resnet_seg``, ``convnext_seg``) are a classification family's own factory
with ``include_top=False`` and one single-map ``SegmentationHead`` on its last feature map: a
fully convolutional network of the FCN-32s kind (Long et al. 2015, section 4.1: one stream, the
classifier's final map predicts, learned in-network upsampling), with no skip connection. They are
the coarse baseline of the task, not a competitor of the U-Nets (D-012).

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

``create_resnet_seg(*, variant, num_classes, input_shape, **model_args)`` / ``create_convnext_seg(...)``
    :param variant: a key of ``ResNet.MODEL_VARIANTS`` (``resnet18`` .. ``resnet152``) or of
        ``ConvNeXtV1.MODEL_VARIANTS`` (``cifar10``, ``tiny`` .. ``xlarge``).
    :param num_classes: output channels of the head; the logits have one channel per class.
    :param input_shape: ``(height, width, channels)``; both sides must be multiples of the family's
        stride (:func:`check_image_size` says it before a run, this function again).
    :param model_args: further keywords of ``create_resnet`` / ``create_convnext_v1`` (``stem_type``,
        ``blocks_per_stage``; ``strides``, ``drop_path_rate``, ``depths`` ...).
    :return: a functional model, ``Input -> backbone -> SegmentationHead``, linear logits
        ``(B, H, W, num_classes)``. The backbone is the classification family's model with
        ``include_top=False`` (one nested model named ``backbone``). The head is built from
        ``dl_techniques.layers.heads.vision.segmentation`` only, with EVERY stage knob passed
        (``use_skip_connections=False``, ``use_attention=False``, ``use_ffn=False``,
        ``dropout_rate=0.0``, ``normalization_type="layer_norm"``, ``activation_type="gelu"``,
        ``hidden_dim=min(256, backbone channels)``), so a later change of the head's defaults moves
        neither family (D-012).
    :raises ValueError: when ``model_args`` names a key the adapter sets (:data:`FIXED_MODEL_ARGS`) or the
        family does not offer (:data:`NOT_OFFERED_MODEL_ARGS`), the input is no multiple of the stride, or
        the built output is not the input's size.
    :raises ConfigError: for a stride that is no power of two (see the stride callables).

``resnet_seg_stride(variant, model_args)`` / ``convnext_seg_stride(variant, model_args)``
    The multiple each side of the input must be, which is also the head's ``upsampling_factor``.
    ResNet: 4 for the ImageNet stem (7x7 stride 2, then a stride-2 max pool) or 1 for the CIFAR stem,
    times 2 for each stage after the first: 32 for four stages, 8 with ``stem_type="cifar"``. ConvNeXt V1:
    ``strides ** len(depths)``, because the model uses ONE ``strides`` for the patchify stem and for
    every downsampler (the paper uses 4 and 2): 16 for the four-stage variants and 4 for ``cifar10`` at
    the family's ``strides=2``, measured by building every variant (D-012). A total that is no power of
    two is refused by name; so is a non-int ``strides`` or ``stem_type`` that is not one of the two.

``observe_resnet_seg(model)`` / ``observe_convnext_seg(model)``
    :param model: a loaded functional model.
    :return: the variant-table keys the nested backbone shows (ResNet: ``blocks_per_stage``,
        ``filters_per_stage``, ``block_type``; ConvNeXt V1: ``depths``, ``dims``), or ``None`` when the
        model holds no ``SegmentationHead`` or no layer of that backbone class (ConvNeXt V2 is a different
        class: a ``convnext_v2`` backbone under a ``convnext_seg`` record is refused). ``stem_type``,
        ``strides``, ``kernel_size``, ``drop_path_rate`` are not table keys and are not checked.

``observe_convunext(model)`` / ``observe_accunet(model)``
    :param model: a loaded functional model.
    :return: the variant-table keys the model's own layers show, or ``None`` when it holds no layer of
        that family. ConvUNext: ``depth`` (the count of ``DownsampleAndSkip``), ``initial_filters`` (the
        width of ``ConvStem``), ``blocks_per_level`` (the ConvNeXt blocks over ``2 * depth + 1``, the
        blocks of both paths and the bottleneck) and ``convnext_version`` (``v1`` or ``v2``, from the
        block class); ``drop_path_rate`` is not shown by a layer and is not checked. AccUNet:
        ``base_filters`` and ``mlfc_iterations`` from the ``AccUNet`` layer's config. Nothing is rebuilt
        (D-088, which supersedes the limit D-080 accepted for a segmentation record).

``builtin_segmentation_models()``
    :return: the four :class:`~train.api.registry.ModelSpec` rows (``convunext``, ``accunet``,
        ``resnet_seg``, ``convnext_seg``),
        freshly built, each with its recipe (``defaults``) and ``resolve_segmentation_args``.
        :func:`train.api.recipes.register_builtin_models` registers them with the six classifiers
        (D-077: they were held back in step 23 for want of a recipe and a task, and both exist now).

``resolve_segmentation_args(context, optim_args, model_args)``
    The ``resolve_model_args`` of both families.

    :return: the factory keywords: the recipe's ``model_args`` (none), then the caller's.
    :raises ConfigError: when ``model_args`` names a key the adapter sets itself
        (``use_bias``, ``final_activation``, ``output_channels`` for ConvUNext, ``input_channels``
        for AccUNet) or one the library supplies (``variant``, ``num_classes``, ``input_shape``).
        The task reaches this before the run directory is claimed (``check_model_args``).

Recipes
-------
``CONVUNEXT`` is ``train.convunext.common.SegTrainingConfig``'s defaults, read from it by
``test_the_segmentation_recipes_are_the_legacy_trainers``: AdamW lr 1e-3, weight decay 1e-4, no
warmup, per-variable ``clipnorm`` 1.0, a cosine over all epochs to 1% of the peak, 30 epochs, batch
16, patience 10, a horizontal flip, Oxford-IIIT Pet at 128 px. ``ACCUNET`` is
``train.accunet.train_accunet.AccUNetTrainingConfig``'s: lr 1e-4, weight decay 1e-4, 2 warmup
epochs, ``global_clipnorm`` 1.0, patience 15. Three values are NOT the legacy ones, on purpose and
stated: the monitor is ``val_miou`` (both legacy trainers default to ``val_loss``; D-077), the
loss is cross-entropy (the AccUNet trainer's default is ``combo``, which this task offers from
step 35 on), and AccUNet's batch is 8, not the legacy 16 (D-088): batch 16 runs out of memory at the
first step on the 24 GB reference card in the legacy trainer and in this task alike (measured, 17.4 GB
free with another tenant's 6.6 GB on the card; ``tests/test_train/test_api/accunet_memory_evidence.json``),
batch 8 peaks at 7.6 GB. The learning rate is NOT scaled (D-036's rule: the legacy trainer has no batch rule
and a scaled AdamW rate would be an invented number; the 3-epoch smoke at batch 8 and lr 1e-4 learned, val_miou
0.31, 0.41, 0.47). The evaluation batch is the training batch (16 for ConvUNext, 8 for AccUNet), as the legacy
trainer evaluates; the shared 256 would put 256 full-size masks and logits in one pass.

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

import collections
import copy
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from . import recipes
from .args import AutoContext, OptimArgs
from .errors import ConfigError
from .registry import ModelSpec, load_attr

# ---------------------------------------------------------------------

TASK = "image_segmentation"

_HERE = "train.api.segmentation_models"
_CONVUNEXT = "dl_techniques.models.vision.convunext.model"
_RESNET = "dl_techniques.models.vision.resnet.model"
_CONVNEXT_V1 = "dl_techniques.models.vision.convnext.convnext_v1"

#: AccUNet's four stride-2 max pools (``AccUNet`` requires H and W divisible by 16).
ACCUNET_STRIDE = 16

#: Width of the backbone families' head: the head's own default and DeepLab v3's ASPP width (256), capped
#: by the backbone's last-stage channels (``cifar10`` ConvNeXt ends at 192): D-012.
HEAD_WIDTH = 256

#: ``strides`` of ``convnext_seg``: the model uses one value for its patchify stem and every downsampler
#: (the paper: 4 and 2). At the classifier's 4 the four-stage variants reach a stride of 256 (measured
#: 1x1 at 128x256), at 2 they reach 16 (DeepLab's output stride) and ``cifar10`` reaches 4.
CONVNEXT_SEG_STRIDES = 2

#: Stride of the ResNet stem by ``stem_type``: the 7x7 stride-2 convolution and the stride-2 max pool, or
#: the CIFAR stem's single 3x3 stride-1 convolution. Each later stage downsamples by 2.
_RESNET_STEM_STRIDE: Dict[str, int] = {"imagenet": 4, "cifar": 1}

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

#: Keys of the backbone factories (``create_resnet``, ``create_convnext_v1``) the backbone adapters set
#: themselves: the classifier's head and its deep supervision have no place in a segmenter, and a ``name``
#: would clash with the nested model's.
_BACKBONE_FIXED = ("include_top", "head_config_overrides", "name")
_RESNET_FIXED = _BACKBONE_FIXED + ("enable_deep_supervision",)

#: Keys of the backbone factories the adapters do NOT set but do not offer either: ``pretrained`` (a
#: classifier's weights loaded into the backbone, which the classes do support) is not offered by these
#: families yet, because ``checkpoint=`` loads no layer of a classification run into them (measured,
#: ``scratch_step2/checkpoint_transfer.json``). The adapter never passes it, so the refusal must not
#: say the adapter sets it (review-iter-1 concern 5).
_BACKBONE_NOT_OFFERED = ("pretrained",)

#: The keys of ``model_args`` each family's adapter sets itself.
FIXED_MODEL_ARGS: Dict[str, Tuple[str, ...]] = {
    "convunext": _CONVUNEXT_FIXED,
    "accunet": _ACCUNET_FIXED,
    "resnet_seg": _RESNET_FIXED,
    "convnext_seg": _BACKBONE_FIXED,
}

#: The keys of ``model_args`` each family's backbone factory takes and the adapter does not offer.
NOT_OFFERED_MODEL_ARGS: Dict[str, Tuple[str, ...]] = {
    "resnet_seg": _BACKBONE_NOT_OFFERED,
    "convnext_seg": _BACKBONE_NOT_OFFERED,
}

#: Why each family's adapter keeps those keys; the parenthesis of the refusal.
_FIXED_WHY: Dict[str, str] = {
    "convunext": "a softmax or biased-off head would break the logits loss",
    "accunet": "the input channels are the data's",
    "resnet_seg": "the segmentation head replaces the classification head",
    "convnext_seg": "the segmentation head replaces the classification head",
}


def _fixed_clash(family: str, model_args: Mapping[str, Any]) -> str:
    """Return the sentence for the keys ``model_args`` names that the adapter sets or does not offer, ``""`` for none.

    Interface contract: ``family`` is a key of :data:`FIXED_MODEL_ARGS`; the result names each refused key
    once, in its own clause (a key the adapter sets, a key the family does not offer yet), and ends with
    ``remove it from model_args``. Never raises.
    """
    clash = [key for key in FIXED_MODEL_ARGS[family] if key in model_args]
    missing = [key for key in NOT_OFFERED_MODEL_ARGS.get(family, ()) if key in model_args]
    clauses = []
    if clash:
        clauses.append(
            f"{', '.join(map(repr, clash))} is set by the {family!r} adapter from model=, the "
            f"data and transform_args ({_FIXED_WHY[family]})"
        )
    if missing:
        clauses.append(
            f"{', '.join(map(repr, missing))} is not offered by the {family!r} family yet "
            "(it does not load a classifier's weights)"
        )
    if not clauses:
        return ""
    return "; ".join(clauses) + "; remove it from model_args"


def _refuse_fixed(family: str, model_args: Mapping[str, Any]) -> None:
    message = _fixed_clash(family, model_args)
    if message:
        raise ValueError(message)


def _three_sides(input_shape: Sequence[int]) -> Tuple[int, int, int]:
    if len(input_shape) != 3:
        raise ValueError(f"input_shape must be (height, width, channels), got {tuple(input_shape)!r}")
    height, width, channels = (int(side) for side in input_shape)
    return height, width, channels


def create_convunext(
    *, variant: str, num_classes: int, input_shape: Sequence[int], **model_args: Any
) -> Any:
    """Build a ConvUNext segmenter; see the module contract."""
    _refuse_fixed("convunext", model_args)
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
    _refuse_fixed("accunet", model_args)
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


#: Every setting of ``SegmentationHead`` the backbone families pass, so that a change of the head's
#: defaults (``use_ffn=True``, ``dropout_rate=0.1`` arrived with the heads commits of another session)
#: moves neither family. ``hidden_dim`` and ``upsampling_factor`` are added per model.
_HEAD_SETTINGS: Dict[str, Any] = {
    "use_skip_connections": False,
    "skip_channel_mode": "auto",
    "explicit_skip_channels": None,
    "input_format": "spatial",
    "patch_grid_size": None,
    "use_attention": False,
    "attention_type": None,
    "use_ffn": False,
    "ffn_type": None,
    "ffn_expansion_factor": None,
    "dropout_rate": 0.0,
    "normalization_type": "layer_norm",
    "activation_type": "gelu",
}


def _segmenter(
    family: str, backbone: Any, input_shape: Tuple[int, int, int], num_classes: int, stride: int
) -> Any:
    """Return ``Input -> backbone -> SegmentationHead`` as one functional model; see the module contract."""
    # DECISION plan-2026-11-03T221936-98b866fb/D-012
    # The head is imported from its leaf module and nothing else of `layers/heads` is: Do NOT use
    # `create_head`, `HeadDomain` or `heads.task_types` (the heads package is another session's area
    # and its facade moved twice; a scan in test_backbone_segmentation_families.py holds this). Every
    # stage knob is passed, `use_ffn=False` and `dropout_rate=0.0` included: the head's defaults are
    # `use_ffn=True`, `dropout_rate=0.1` and a change of them must not change a trained family.
    import keras
    from dl_techniques.layers.heads.vision.segmentation import SegmentationHead

    height, width, channels = input_shape
    inputs = keras.Input(shape=(height, width, channels))
    features = backbone(inputs)
    head = SegmentationHead(
        num_classes=num_classes,
        upsampling_factor=stride,
        hidden_dim=min(HEAD_WIDTH, int(features.shape[-1])),
        name="segmentation_head",
        **_HEAD_SETTINGS,
    )
    logits = head(features)
    if (logits.shape[1], logits.shape[2]) != (height, width):
        raise ValueError(
            f"the {family} output is {logits.shape[1]}x{logits.shape[2]} for a {height}x{width} input "
            f"(output stride {stride}); each side must be a multiple of {stride}"
        )
    return keras.Model(inputs, logits, name=family)


def _check_aligned(family: str, input_shape: Tuple[int, int, int], stride: int) -> None:
    height, width, _ = input_shape
    if height % stride or width % stride:
        raise ValueError(
            f"{family} needs each side of input_shape to be a multiple of {stride}, got {height}x{width}"
        )


def create_resnet_seg(
    *, variant: str, num_classes: int, input_shape: Sequence[int], **model_args: Any
) -> Any:
    """Build a ResNet backbone with a single-map segmentation head; see the module contract."""
    _refuse_fixed("resnet_seg", model_args)
    shape = _three_sides(input_shape)
    stride = resnet_seg_stride(variant, model_args)
    _check_aligned("resnet_seg", shape, stride)
    from dl_techniques.models.vision.resnet.model import create_resnet

    backbone = create_resnet(variant, include_top=False, input_shape=shape, name="backbone", **model_args)
    return _segmenter("resnet_seg", backbone, shape, num_classes, stride)


def create_convnext_seg(
    *, variant: str, num_classes: int, input_shape: Sequence[int], **model_args: Any
) -> Any:
    """Build a ConvNeXt V1 backbone with a single-map segmentation head; see the module contract."""
    _refuse_fixed("convnext_seg", model_args)
    shape = _three_sides(input_shape)
    model_args = {"strides": CONVNEXT_SEG_STRIDES, **model_args}
    stride = convnext_seg_stride(variant, model_args)
    _check_aligned("convnext_seg", shape, stride)
    from dl_techniques.models.vision.convnext.convnext_v1 import create_convnext_v1

    backbone = create_convnext_v1(variant, include_top=False, input_shape=shape, name="backbone", **model_args)
    return _segmenter("convnext_seg", backbone, shape, num_classes, stride)


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


def _variant_row(table_path: str, variant: str) -> Mapping[str, Any]:
    """Return the variants table row of ``variant`` from ``module:Class.MODEL_VARIANTS``, or raise ``ValueError``."""
    table = load_attr(table_path)
    if variant not in table:
        raise ValueError(f"Unknown variant {variant!r}. Available variants: {sorted(table)}")
    return table[variant]


def _sequence_of(model_args: Mapping[str, Any], key: str, row: Mapping[str, Any]) -> Sequence[Any]:
    """Return ``model_args[key]`` or the variant row's, a non-empty list; ``ConfigError`` names the key otherwise."""
    value = model_args.get(key, row[key])
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(
            f"Invalid value for 'model_args.{key}': a non-empty list of stage sizes, got {value!r}"
        )
    return value


def resnet_seg_stride(variant: str, model_args: Mapping[str, Any]) -> int:
    """Return the output stride of a ResNet backbone; see the module contract."""
    stem = model_args.get("stem_type", "imagenet")
    if stem not in _RESNET_STEM_STRIDE:
        raise ConfigError(
            f"Invalid value for 'model_args.stem_type': expected one of {sorted(_RESNET_STEM_STRIDE)}, "
            f"got {stem!r}"
        )
    row = _variant_row(f"{_RESNET}:ResNet.MODEL_VARIANTS", variant)
    stages = _sequence_of(model_args, "blocks_per_stage", row)
    return _RESNET_STEM_STRIDE[stem] * 2 ** (len(stages) - 1)


def convnext_seg_stride(variant: str, model_args: Mapping[str, Any]) -> int:
    """Return ``strides ** len(depths)`` of a ConvNeXt V1 backbone, refusing a non power of two; see the module contract."""
    strides = model_args.get("strides", CONVNEXT_SEG_STRIDES)
    if isinstance(strides, bool) or not isinstance(strides, int) or strides < 1:
        raise ConfigError(
            f"Invalid value for 'model_args.strides': an int of 1 or more, got {strides!r}"
        )
    row = _variant_row(f"{_CONVNEXT_V1}:ConvNeXtV1.MODEL_VARIANTS", variant)
    stages = len(_sequence_of(model_args, "depths", row))
    total = strides**stages
    if total & (total - 1):
        raise ConfigError(
            f"Invalid value for 'model_args.strides': {strides} over the {stages} stages of "
            f"convnext_seg/{variant} is an output stride of {total}, which is not a power of two; "
            "the segmentation head upsamples by powers of two (the default is "
            f"{CONVNEXT_SEG_STRIDES})"
        )
    return total


def _layers_of(model: Any) -> Any:
    """Yield every layer of ``model``, depth first through the layers that hold layers."""
    for layer in getattr(model, "layers", None) or []:
        yield layer
        yield from _layers_of(layer)


def observe_convunext(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant keys ConvUNext's layers show, ``None`` for a model without them; see the module contract."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-088
    # The family and variant of a segmentation record are checked by what the layers show, not
    # left unchecked (D-080's limit: review-iter-3 concern 3 loaded a convunext/tiny file under
    # records naming convunext/base, convunext/xlarge and accunet/base, and `export` wrote the lie).
    # Do NOT rebuild the variant and compare weight shapes (a `model_args` key outside the table,
    # `kernel_size` say, changes the shapes: D-032's failure) and do NOT compare `model.name`
    # (`model_args.model_name` renames it). Read only keys a layer shows and a record would carry.
    classes = collections.Counter(type(layer).__name__ for layer in _layers_of(model))
    if not classes["DownsampleAndSkip"] or not classes["ConvStem"]:
        return None
    stem = next(layer for layer in _layers_of(model) if type(layer).__name__ == "ConvStem")
    depth = classes["DownsampleAndSkip"]
    blocks = classes["ConvNextV1Block"] + classes["ConvNextV2Block"]
    observed: Dict[str, Any] = {
        "depth": depth,
        "initial_filters": int(stem.get_config()["filters"]),
        "convnext_version": "v2" if classes["ConvNextV2Block"] else "v1",
    }
    if blocks % (2 * depth + 1) == 0:
        observed["blocks_per_level"] = blocks // (2 * depth + 1)
    return observed


def observe_accunet(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant keys AccUNet's layer shows, ``None`` for a model without it; see the module contract."""
    for layer in _layers_of(model):
        if type(layer).__name__ == "AccUNet":
            config = layer.get_config()
            return {key: config[key] for key in ("base_filters", "mlfc_iterations")}
    return None


def _backbone_table(model: Any, backbone_class: str, keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Return ``keys`` of the nested ``backbone_class`` layer's config, or ``None`` without it or without the head."""
    classes = {type(layer).__name__: layer for layer in _layers_of(model)}
    if "SegmentationHead" not in classes or backbone_class not in classes:
        return None
    config = classes[backbone_class].get_config()
    return {key: config[key] for key in keys}


def observe_resnet_seg(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant keys the nested ResNet shows, ``None`` for a model that is no ResNet plus head; see the module contract."""
    # DECISION plan-2026-11-03T221936-98b866fb/D-012
    # `variants_attr` of this family is the classifier's own table (`ResNet.MODEL_VARIANTS`: one source),
    # and the model is a FUNCTIONAL wrapper, so it is not an instance of `ResNet`: the family and the
    # variant are read from the nested backbone's layer config, as D-088 did for the U-Nets. Do NOT
    # copy the table into this module (two tables to keep equal) and do NOT compare model.name.
    return _backbone_table(model, "ResNet", ("blocks_per_stage", "filters_per_stage", "block_type"))


def observe_convnext_seg(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant keys the nested ConvNeXt V1 shows, ``None`` for a model that is no ConvNeXt V1 plus head; see the module contract."""
    return _backbone_table(model, "ConvNeXtV1", ("depths", "dims"))


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


def resolve_segmentation_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's; see the module contract."""
    message = _fixed_clash(context.family, model_args)
    if message:
        raise ConfigError(f"Invalid value for 'model_args': {message}")
    return recipes.merge_model_args(context, optim_args, model_args, {})


# ---------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------

#: The monitor and the success metric of a segmentation run (D-077).
MONITOR = "val_miou"

#: Side of the Oxford-IIIT Pet images both legacy trainers resize to (``SegTrainingConfig.image_size``
#: 128, ``AccUNetTrainingConfig.image_size`` 128).
PET_IMAGE_SIZE = 128


def _segmentation_recipe(
    *,
    optim_args: Dict[str, Any],
    patience: int,
    epochs: int = 30,
    batch_size: int = 16,
    model_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a segmentation recipe: every AUTO key of the groups the task resolves.

    ``image_size`` is given for Oxford-IIIT Pet only: files have no native size, so the caller
    states one (``seg_facts`` refuses file data without it, before the recipe is read).
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-077
    # The monitor is `val_miou`, NOT the legacy trainers' `val_loss`. Do NOT copy their default
    # "for parity": Keras' `val_loss` is a mean of per-batch valid-pixel means weighted by IMAGES
    # (D-076), not the pooled pixel cross-entropy, and `train.convunext`'s own README measured the
    # final weights beating the `val_loss` checkpoint in all three seeds (the `val_miou` peak came 8
    # to 9 epochs after the `val_loss` best). The success verdict is `val_miou` too, so the epoch
    # kept and the number judged are one metric. The parity gate (step 32) runs the legacy trainer
    # with `--monitor val_miou` for the same reason. The evaluation batch is the training batch
    # (the legacy trainer evaluates at its batch); the shared 256 would put 256 full-size masks
    # and logits in one pass.
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "optim_args": optim_args,
        "transform_args": {"augment": "flip"},
        "metric_args": {"watch_metric": MONITOR},
        "checkpoint_args": {"early_stopping_patience": patience},
        "analysis_args": {"eval_batch_size": batch_size},
        "model_args": dict(model_args or {}),
        "per_dataset": {"oxford_pet": {"transform_args": {"image_size": PET_IMAGE_SIZE}}},
        "per_variant": {},
    }


# train/convunext/common.py SegTrainingConfig: epochs :194, batch :195, lr :196, weight decay :197,
# warmup :198, patience :199, image size :187; AdamW with clipnorm GRADIENT_CLIP_NORM (:680, the
# ConvNeXt trainer's 1.0, per variable); cosine over all epochs (build_lr_schedule, alpha 0.01);
# the train pipeline flips left-right (_random_flip). Its monitor is val_loss (:202), which this
# recipe does NOT copy (D-077, the anchor in `_segmentation_recipe`).
CONVUNEXT = _segmentation_recipe(
    optim_args=recipes.optim_recipe(
        optimizer="adamw", lr=1e-3, weight_decay=1e-4, warmup_epochs=0, decay_over="total",
        clipnorm=1.0,
    ),
    patience=10,
)

# train/accunet/train_accunet.py AccUNetTrainingConfig: epochs :97, batch :98, lr :99, weight decay
# :100, warmup :101 (2.0), gradient_clipping :103 (optimizer_builder "gradient_clipping_by_norm", a
# GLOBAL norm), patience :106, image size :87. Its loss is `combo` (:104) and its monitor val_loss
# (:105); this recipe trains cross-entropy and monitors val_miou (D-077).
# DECISION plan-2026-10-31T180033-2faf890a/D-088
# AccUNet's batch is 8, NOT the legacy trainer's 16. Do NOT restore 16 "for parity": the first
# training step of `batch_size=16` runs out of memory ([16,128,128,128] in `decoder_l3_block2`) in the
# legacy trainer and through this task alike on the 24 GB card (17.4 GB free beside another process's
# 6.6 GB), and the review's smoke is the only GPU run of the family that has completed. D-036's
# procedure: halve until the rule holds (20 steps, no OOM, no allocator collection, peak <= 20,000 MiB,
# device <= 23,000 MiB) -> 8 (peak 7,621 MiB). The learning rate stays 1e-4 (D-036: no batch rule in the
# source, no invented scaling); 3 epochs at 8 and 1e-4 learned.
ACCUNET = _segmentation_recipe(
    optim_args=recipes.optim_recipe(
        optimizer="adamw", lr=1e-4, weight_decay=1e-4, warmup_epochs=2, decay_over="total",
        global_clipnorm=1.0,
    ),
    patience=15,
    batch_size=8,
)


# DECISION plan-2026-11-03T221936-98b866fb/D-012
# The backbone families have NO legacy trainer, so their optimizer settings start from ConvUNext's, read
# from `CONVUNEXT` (one source, not four copied numbers): the same dataset, loss and pipeline, the only
# segmentation recipe measured on Oxford-IIIT Pet. `resnet_seg` and `convnext_seg` keep them as they are (30 epochs on
# Pet: `resnet_seg` val_miou 0.631). The values are an untuned starting point, not a result. The batch is the one D-036's
# procedure measured on the 24 GB card (20 real steps on Pet at 128 px, halved from 16 until it fits): 16 fits
# the default variants (`resnet18`, `convnext_seg/tiny`, and `resnet50`, `resnet152`, `cifar10`),
# `convnext_seg/base` needs 8 and `xlarge` did not fit at any batch from 16 to 1 beside another session's 15.9 GB
# (`tests/test_train/test_api/backbone_seg_memory_evidence.json`).
# DECISION plan-2026-11-03T221936-98b866fb/D-034
# `convnext_seg` takes ConvUNext's learning rate, 1e-3, for EVERY variant (it had a per-variant rate, D-029: 1e-3 for `cifar10`
# and 1e-4 for the rest). The 1e-4 existed because `tiny` at 1e-3 stayed at the pixel prior (test mIoU 0.370 and 0.370 over
# seeds 42 and 43, early stop at epoch 13 and 22), and that was a property of the V1 block's old layer scale (init 1.0, capped,
# L2), not of the rate: with the paper's layer scale (step 10b) `tiny` at 1e-3 trains and is ahead of 1e-4 on both seeds
# (test mIoU 0.550 against 0.507 and 0.537 against 0.529), and `cifar10` is ahead at 1e-3 (0.673 against 0.628, seed 44). Do NOT
# bring the per-variant rate back for `tiny`'s sake and do NOT make 1e-4 the family rate "because it was the safe one": on the
# paper's layer scale it is behind at both measured variants (`tests/test_train/test_api/convnext_seg_lr_evidence.json`,
# held by a test). `small`, `base`, `large` and `xlarge` were NOT measured at either rate and take the same 1e-3; a first run
# of one that stalls at the pixel prior should try a lower rate.
CONVNEXT_SEG = _segmentation_recipe(
    optim_args=copy.deepcopy(CONVUNEXT["optim_args"]), patience=10, batch_size=16,
    model_args={"strides": CONVNEXT_SEG_STRIDES},
)
RESNET_SEG = _segmentation_recipe(
    optim_args=copy.deepcopy(CONVUNEXT["optim_args"]), patience=10, batch_size=16,
)


def builtin_segmentation_models() -> Tuple[ModelSpec, ...]:
    """Return the segmentation family specs, freshly built; see the module contract."""
    return (
        ModelSpec(
            family="convunext",
            factory=f"{_HERE}:create_convunext",
            variants_attr=f"{_CONVUNEXT}:CONVUNEXT_CONFIGS",
            tasks=(TASK,),
            defaults=CONVUNEXT,
            resolve_model_args=f"{_HERE}:resolve_segmentation_args",
            stride=f"{_HERE}:convunext_stride",
            observe=f"{_HERE}:observe_convunext",
        ),
        ModelSpec(
            family="accunet",
            factory=f"{_HERE}:create_accunet",
            variants_attr=f"{_HERE}:ACCUNET_VARIANTS",
            tasks=(TASK,),
            defaults=ACCUNET,
            resolve_model_args=f"{_HERE}:resolve_segmentation_args",
            stride=ACCUNET_STRIDE,
            observe=f"{_HERE}:observe_accunet",
        ),
        ModelSpec(
            family="resnet_seg",
            factory=f"{_HERE}:create_resnet_seg",
            variants_attr=f"{_RESNET}:ResNet.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=RESNET_SEG,
            resolve_model_args=f"{_HERE}:resolve_segmentation_args",
            stride=f"{_HERE}:resnet_seg_stride",
            observe=f"{_HERE}:observe_resnet_seg",
        ),
        ModelSpec(
            family="convnext_seg",
            factory=f"{_HERE}:create_convnext_seg",
            variants_attr=f"{_CONVNEXT_V1}:ConvNeXtV1.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=CONVNEXT_SEG,
            resolve_model_args=f"{_HERE}:resolve_segmentation_args",
            stride=f"{_HERE}:convnext_seg_stride",
            observe=f"{_HERE}:observe_convnext_seg",
        ),
    )
