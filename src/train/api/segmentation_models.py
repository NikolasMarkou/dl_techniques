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
    :return: the two :class:`~train.api.registry.ModelSpec` rows (``convunext``, ``accunet``),
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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from . import recipes
from .args import AutoContext, OptimArgs
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

#: The keys of ``model_args`` each family's adapter sets itself.
FIXED_MODEL_ARGS: Dict[str, Tuple[str, ...]] = {
    "convunext": _CONVUNEXT_FIXED,
    "accunet": _ACCUNET_FIXED,
}


def _fixed_clash(family: str, model_args: Mapping[str, Any]) -> str:
    """Return the sentence for the adapter-set keys ``model_args`` names, ``""`` for none."""
    clash = [key for key in FIXED_MODEL_ARGS[family] if key in model_args]
    if not clash:
        return ""
    return (
        f"{', '.join(map(repr, clash))} is set by the {family!r} adapter from model=, the "
        "data and transform_args (a softmax or biased-off head would break the logits loss); "
        "remove it from model_args"
    )


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
    *, optim_args: Dict[str, Any], patience: int, epochs: int = 30, batch_size: int = 16
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
        "model_args": {},
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
    )
