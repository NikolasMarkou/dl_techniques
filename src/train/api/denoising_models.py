"""The image-denoising model families of ``python -m train``: adapters, refusals, recipes and size rules.

Five families train the ``image_denoising`` task, three bias-free (``properties=("bias_free",)``) and two
that carry additive constants and are never checked for homogeneity (D-010):

``bfcnn``
    ``bias_free_denoisers.bfcnn``: a flat residual stack, no stride, any input size.
``bfunet``
    ``bias_free_denoisers.bfunet``: a U-Net of residual blocks, sides a multiple of ``2 ** depth``.
``bfconvunext``
    ``bias_free_denoisers.bfconvunext``: the ``use_bias=False`` arm of ConvUNext, sides a multiple of
    ``2 ** depth``.
``convunext_biased``
    ``convunext.model.create_convunext_variant`` with ``use_bias=True``: the segmentation family's
    ``convunext`` cannot serve (its adapter forces ``output_channels=num_classes``), so this is its own spec.
``scunet``
    ``image_restoration.scunet``: pads to 64 inside, any input size. It has no variants table in the
    library; ``SCUNET_VARIANTS`` below is this plan's own (D-010).

None of the library factories takes the arguments the API calls a factory with
(``factory(variant=, num_classes=, input_shape=)``): the adapters close that gap and nothing else. Importing this
module imports no model code and no keras: each adapter imports its model inside the call, so
``python -m train --help`` allocates nothing.

Interface contract
------------------
``TASK``
    ``"image_denoising"``, the task these families train.

``create_bfcnn / create_bfunet / create_bfconvunext / create_convunext_biased / create_scunet``
    ``f(*, variant, num_classes=None, input_shape=(None, None, 3), **model_args)``
    :param variant: a key of the family's table (``BFCNN_CONFIGS``, ``BFUNET_CONFIGS``,
        ``CONVUNEXT_CONFIGS`` for both ConvUNext families, :data:`SCUNET_VARIANTS`).
    :param num_classes: must be ``None``: a denoiser has no classes (D-005).
    :param input_shape: ``(height, width, channels)``; a side may be ``None`` (a flexible model).
    :param model_args: the keys the family's table in :data:`ALLOWED` / :data:`REFUSED` / :data:`FORCED` offers.
        ``activation``, ``block_activation``, ``stem_activation`` and ``supervision_activation`` named
        ``"leaky_relu"`` become ONE ``LeakyReLU(negative_slope=0.1)`` instance (:data:`LEAKY_RELU_SLOPE`), as every
        legacy trainer builds it: the bare string is Keras' slope 0.2. ``gabor_activation`` and
        ``final_activation`` stay strings. ``trainable_gabor_stem=false`` (the families with a Gabor stem) freezes
        the stem layer before the model is returned, so before compile.
    :return: a functional ``keras.Model`` with ONE output of the input's size and channels, a linear head
        (``final_activation`` defaults to ``"linear"``), no deep supervision, followed by a weight-free
        ``Activation("linear", dtype="float32")`` named ``output_float32``: under ``mixed_float16`` every family
        emits float16 (measured, D-019), and the loss, PSNR and SSIM must read float32.
    :raises ValueError: for a non-``None`` ``num_classes``, a key the family refuses or does not know, or a
        refusal by the library; :func:`train.api.recipes.build_model` turns it into a ``ConfigError``.

``resolve_denoising_args(context, optim_args, model_args)``
    The ``resolve_model_args`` of every family. :return: the factory keywords: the recipe's ``model_args``, then the
    caller's. :raises ConfigValidationError: naming EVERY refused key at once (``model_args.<key>`` and why), before
    the run directory is claimed (the task's ``check_model_args`` calls it). :raises ConfigError: for an sgd weight
    decay (no model regularizer is built here), a library-supplied key, an adamw ``kernel_regularizer``.

``problems_of(family, model_args)``
    :return: one line per refusal of ``model_args`` for ``family`` (``[]`` for none): an unknown key, a key in
        :data:`REFUSED` outside its rule, a key in :data:`FORCED` with another value, and the cross-key rules
        (``high_freq_blocks`` without ``use_laplacian_pyramid``; ``trainable_gabor_stem=false`` without a Gabor stem).
        Pure; imports the library's activation allowlist on the first value rule it applies.

``factory_signatures(family)``
    :return: ``{"module:callable": [named parameters]}`` of the library factories in :data:`FACTORIES`; a
        ``**kwargs`` is left out (the callable behind it is in the same dict).

``signature_census(family, signatures=None)``
    :param signatures: ``{callable name: parameter names}`` to classify instead of the library's own (a test's
        stand-in for a factory that gained a parameter).
    :return: ``{"unclassified": [...], "stale": [...]}``: parameters of the factories in :data:`FACTORIES` that
        :data:`ALLOWED`, :data:`REFUSED`, :data:`FORCED` and :data:`SUPPLIED` do not name, and keys those tables
        name that no factory has. A passing census is both empty.

``bfunet_stride(variant, model_args)``
    ``2 ** depth`` of a bfunet variant or of ``model_args.depth``. ConvUNext's is
    :func:`train.api.segmentation_models.convunext_stride`; ``bfcnn`` and ``scunet`` have no stride rule.

``observe_bfcnn / observe_bfunet / observe_scunet(model)``
    The variant-table keys the loaded model's own layers show (``num_blocks``, ``filters`` / ``depth``,
    ``initial_filters``, ``blocks_per_level`` / ``config``, ``dim``, ``head_dim``), or ``None`` for a model that holds
    no layer of the family. The ConvUNext families reuse
    :func:`train.api.segmentation_models.observe_convunext`.

``builtin_denoising_models()``
    :return: the five :class:`~train.api.registry.ModelSpec` rows, freshly built.
    :func:`train.api.recipes.register_builtin_models` registers them with the others.

Recipes
-------
Every family trains the legacy denoiser's optimizer (``train.bfunet.common.TrainingConfig``): AdamW, lr 1e-3, weight
decay 4e-3, cosine over all epochs, a GLOBAL clip norm of 1.0 (``gradient_clipping_by_norm``), 10 warmup epochs (10 percent
of the 100 epochs of the recipe), batch 16 (4 for the ``base`` U-Nets at 256 px, the legacy README's 24 GB note). The
bias-free families build what the legacy trainers build: v1 blocks (ConvUNext only), ``batchnorm`` (the variance-only
bias-free layer), LeakyReLU(0.1), the trainable Gabor stem (U-Nets), ``final_projection_groups=1``. Early stopping is
off in the legacy trainers (the curriculum makes the monitor non-monotonic); the API has no off switch, so the recipe's
patience is its 100 epochs. The ``convunext_biased`` recipe keeps the library defaults (the table's v2 blocks, layernorm,
GELU): the bias-free choices exist to keep ``f(a x) = a f(x)`` and a biased family has no use for them.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only.
"""

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import recipes
from .args import AutoContext, OptimArgs
from .config import unknown_key_message
from .errors import ConfigError, ConfigValidationError
from .registry import ModelSpec, load_attr
# The one walk through nested layers (private to segmentation_models by name only: it is the helper its observers use).
from .segmentation_models import _layers_of

# ---------------------------------------------------------------------

TASK = "image_denoising"

_HERE = "train.api.denoising_models"
_SEG = "train.api.segmentation_models"
_BF = "dl_techniques.models.vision.bias_free_denoisers"
_CONVUNEXT = "dl_techniques.models.vision.convunext.model"
_SCUNET = "dl_techniques.models.vision.image_restoration.scunet.model"

#: ``input_shape`` of a flexible model: any height and width, three channels.
DEFAULT_INPUT_SHAPE: Tuple[Optional[int], Optional[int], int] = (None, None, 3)

#: The slope every legacy denoising trainer gives its LeakyReLU (``block_activation_alpha``).
LEAKY_RELU_SLOPE = 0.1

#: The names the API passes to a factory itself; a ``model_args`` key of this name is refused by
#: ``recipes.merge_model_args`` before any table here is read.
SUPPLIED = ("variant", "num_classes", "input_shape")

#: The 100 epochs of the legacy default (``TrainingConfig.epochs``); also the recipe's early-stopping patience.
EPOCHS = 100

#: What ``train.api`` calls SCUNet's three sizes. The library publishes ONE network (``config=[4]*7``, ``dim=64``,
#: ``head_dim=32``, Zhang et al. 2022) and scales it by editing these three arguments, so there is no table to read:
#: ``base`` is the paper's, ``tiny`` is the measured one (1,388,246 parameters, step 2: it trains, exports and pads
#: inside), ``small`` is in between. SCUNet has never trained here (D-010).
SCUNET_VARIANTS: Dict[str, Dict[str, Any]] = {
    "tiny": {"config": [1] * 7, "dim": 32, "head_dim": 16},
    "small": {"config": [2] * 7, "dim": 48, "head_dim": 24},
    "base": {"config": [4] * 7, "dim": 64, "head_dim": 32},
}

# ---------------------------------------------------------------------
# What each family offers, refuses and forces
# ---------------------------------------------------------------------

# DECISION plan-2026-11-10T095307-d483ffab/D-015
# A second rule set beside the library's, held equal to the factories' signatures by `signature_census`. Do NOT
# extend the library's `_validate_bias_free_arguments` instead: it does not check `block_activation`,
# `stem_activation` or `supervision_activation`, and the legacy trainers build a GELU default that must keep
# building. Do NOT refuse `use_gabor_stem=false` or `convnext_version=v2`: the GRN beta they once carried is gone
# (plan-2026-11-03T221936-98b866fb D-039) and both read homogeneous (1e-6, 1.5e-4). Only the three families with
# `properties=("bias_free",)` have refusals; the two biased ones have none beyond what their factories need.

#: Kinds of value rule a :data:`REFUSED` row names.
ACTIVATION = "activation"  # a name in the library's POSITIVELY_HOMOGENEOUS_ACTIVATIONS (None, linear, relu, leaky_relu)
NORM_BIAS_FREE = "norm_bias_free"  # 'batchnorm' (the library resolves it to the bias-free layer) or 'bias_free_batchnorm'
NORM_BATCH = "norm_batch"  # 'batchnorm'
NO_CENTER = "no_center"  # False: a trainable additive offset is a bias by another name
NEVER = "never"  # any value: the key is not offered

_ONLY_VALUES = {NORM_BIAS_FREE: ("batchnorm", "bias_free_batchnorm"), NORM_BATCH: ("batchnorm",), NO_CENTER: (False,)}

#: The keys the adapters themselves add (``trainable_gabor_stem``): classified in :data:`ALLOWED`, not in any signature.
ADAPTER_KEYS = ("trainable_gabor_stem",)

_UNET_PARTS = (
    "kernel_initializer", "kernel_regularizer", "use_gabor_stem", "gabor_filters", "gabor_kernel_size",
    "gabor_stem_projection", "use_laplacian_pyramid", "laplacian_kernel_size", "high_freq_blocks",
    "zero_pad_channels", "downsample_pool_type", "final_projection_groups", "dropout_rate", "model_name",
    "trainable_gabor_stem",
)
_CONVUNEXT_PARTS = (
    "depth", "initial_filters", "filter_multiplier", "blocks_per_level", "convnext_version", "stem_kernel_size",
    "use_gabor_stem", "gabor_filters", "gabor_filters_per_channel", "gabor_kernel_size", "gabor_stem_projection",
    "use_laplacian_pyramid", "laplacian_kernel_size", "high_freq_blocks", "bottleneck_attention_blocks",
    "bottleneck_attention_heads", "zero_pad_channels", "extra_zero_output_channels", "final_projection_groups",
    "downsample_pool_type", "block_kernel_size", "drop_path_rate", "kernel_initializer", "kernel_regularizer",
    "depthwise_initializer", "depthwise_regularizer", "dropout_rate", "supervision_norm_scale", "model_name",
    "trainable_gabor_stem",
)

#: Keys passed through as the caller gave them.
ALLOWED: Dict[str, Tuple[str, ...]] = {
    "bfcnn": (
        "num_blocks", "filters", "initial_kernel_size", "kernel_size", "kernel_initializer", "kernel_regularizer",
        "model_name",
    ),
    "bfunet": ("depth", "initial_filters", "filter_multiplier", "blocks_per_level", "kernel_size",
               "initial_kernel_size", "use_residual_blocks", *_UNET_PARTS),
    "bfconvunext": _CONVUNEXT_PARTS,
    "convunext_biased": (
        *_CONVUNEXT_PARTS, "block_normalization", "block_activation", "stem_activation", "supervision_activation",
        "final_activation", "gabor_activation", "stem_normalization", "supervision_norm_center",
    ),
    "scunet": ("config", "dim", "head_dim", "stochastic_depth_rate", "input_resolution", "head_config_overrides"),
}

_NOT_AN_IMAGE = "it adds outputs, and the task has one"
_HOMOGENEOUS = "use_bias=false builds a model whose output scales with its input only if this holds"

#: Keys refused outside a rule: ``{family: {key: (kind, why)}}``. ``why`` ends the sentence "model_args.<key> is
#: refused: ...". Only the three bias-free families have rows.
REFUSED: Dict[str, Dict[str, Tuple[str, str]]] = {
    "bfcnn": {
        "normalization_type": (NORM_BIAS_FREE, "layernorm and every other norm is not degree 1 homogeneous"),
        "activation": (ACTIVATION, _HOMOGENEOUS),
        "final_activation": (ACTIVATION, _HOMOGENEOUS),
    },
    "bfunet": {
        "block_normalization": (NORM_BIAS_FREE, "layernorm and every other norm is not degree 1 homogeneous"),
        "activation": (ACTIVATION, _HOMOGENEOUS),
        "final_activation": (ACTIVATION, _HOMOGENEOUS),
        "gabor_activation": (ACTIVATION, _HOMOGENEOUS),
        "expose_bottleneck": (NEVER, _NOT_AN_IMAGE),
        "pretrained": (NEVER, "weights come from checkpoint=, not from the model factory"),
        "weights_dataset": (NEVER, "weights come from checkpoint=, not from the model factory"),
        "weights_input_shape": (NEVER, "weights come from checkpoint=, not from the model factory"),
        "cache_dir": (NEVER, "weights come from checkpoint=, not from the model factory"),
    },
    "bfconvunext": {
        "use_bias": (NEVER, "the wrapper pins use_bias=false; the biased arm is model=convunext_biased/<variant>"),
        "stem_normalization": (NEVER, "the wrapper pins stem_normalization to global_response_norm"),
        "block_normalization": (NORM_BATCH, "layernorm is scale invariant (degree 0), not degree 1 homogeneous"),
        "block_activation": (ACTIVATION, _HOMOGENEOUS),
        "stem_activation": (ACTIVATION, _HOMOGENEOUS),
        "supervision_activation": (ACTIVATION, _HOMOGENEOUS),
        "final_activation": (ACTIVATION, _HOMOGENEOUS),
        "gabor_activation": (ACTIVATION, _HOMOGENEOUS),
        "supervision_norm_center": (NO_CENTER, "the supervision norm's center is a trainable additive offset"),
        "expose_bottleneck": (NEVER, _NOT_AN_IMAGE),
    },
    "convunext_biased": {
        "expose_bottleneck": (NEVER, _NOT_AN_IMAGE),
    },
    "scunet": {
        "window_size": (NEVER, "the 64 pixel pad target of the model assumes the default window of 8"),
    },
}

#: Keys the adapter sets itself: ``{family: {key: (value, why)}}``. Naming one with ITS value is accepted (the
#: adapter passes it anyway); another value is refused with ``why``. A key the DATA decides (``output_channels``,
#: ``in_nc``) is in :data:`REFUSED` as ``NEVER`` instead: the resolver does not know the channels.
FORCED: Dict[str, Dict[str, Tuple[Any, str]]] = {
    "bfcnn": {},
    "bfunet": {"enable_deep_supervision": (False, _NOT_AN_IMAGE)},
    "bfconvunext": {
        "enable_deep_supervision": (False, _NOT_AN_IMAGE),
        "include_top": (True, "include_top=false returns decoder features, not an image"),
    },
    "convunext_biased": {
        "use_bias": (True, "use_bias=false is the bias-free arm: train it as model=bfconvunext/<variant>"),
        "enable_deep_supervision": (False, _NOT_AN_IMAGE),
        "include_top": (True, "include_top=false returns decoder features, not an image"),
    },
    "scunet": {"include_top": (True, "include_top=false returns features, not an image")},
}

_CHANNELS_WHY = "the output has the image's channels, which the data decides"
REFUSED["bfconvunext"]["output_channels"] = (NEVER, _CHANNELS_WHY)
REFUSED["convunext_biased"]["output_channels"] = (NEVER, _CHANNELS_WHY)
REFUSED["scunet"]["in_nc"] = (NEVER, _CHANNELS_WHY)

#: The library callables whose signatures :func:`signature_census` classifies: the variant wrapper the adapter calls
#: and the builder behind it (``**kwargs`` of the first lands on the second).
FACTORIES: Dict[str, Tuple[str, ...]] = {
    "bfcnn": (f"{_BF}.bfcnn:create_bfcnn_denoiser", f"{_BF}.bfcnn:create_bfcnn_variant"),
    "bfunet": (f"{_BF}.bfunet:create_bfunet_denoiser", f"{_BF}.bfunet:create_bfunet_variant"),
    "bfconvunext": (f"{_CONVUNEXT}:create_convunext", f"{_BF}.bfconvunext:create_convunext_variant"),
    "convunext_biased": (f"{_CONVUNEXT}:create_convunext", f"{_CONVUNEXT}:create_convunext_variant"),
    "scunet": (f"{_SCUNET}:create_scunet",),
}


def _accepted(kind: str) -> Tuple[Any, ...]:
    """Return the values a value rule of ``kind`` accepts (the activation allowlist is the library's own)."""
    if kind == ACTIVATION:
        from dl_techniques.models.vision.convunext.model import POSITIVELY_HOMOGENEOUS_ACTIVATIONS

        return tuple(sorted(POSITIVELY_HOMOGENEOUS_ACTIVATIONS, key=lambda name: (name is not None, str(name))))
    return _ONLY_VALUES[kind]


def _known_keys(family: str) -> List[str]:
    return sorted({*ALLOWED[family], *REFUSED[family], *FORCED[family]})


def problems_of(family: str, model_args: Mapping[str, Any]) -> List[str]:
    """Return one line per refusal of ``model_args`` for ``family``; see the module contract."""
    lines: List[str] = []
    for key, value in model_args.items():
        name = f"model_args.{key}"
        if key in ALLOWED[family]:
            continue
        if key in FORCED[family]:
            forced, why = FORCED[family][key]
            if value != forced or isinstance(value, bool) != isinstance(forced, bool):
                lines.append(f"Invalid value for {name!r}: the {family!r} adapter sets it to {forced!r} ({why}), got {value!r}")
            continue
        if key not in REFUSED[family]:
            lines.append(unknown_key_message("model_args", key, _known_keys(family)))
            continue
        kind, why = REFUSED[family][key]
        if kind == NEVER:
            lines.append(f"Invalid value for {name!r}: {key} is not offered by {family!r} ({why})")
            continue
        accepted = _accepted(kind)
        if not any(value == ok and type(value) is type(ok) for ok in accepted):
            lines.append(
                f"Invalid value for {name!r}: {value!r} is refused for {family!r}, which needs one of "
                f"{', '.join(map(repr, accepted))} ({why})"
            )
    if "high_freq_blocks" in model_args and "high_freq_blocks" in ALLOWED[family]:
        # `high_freq_blocks` acts on the Laplacian pyramid's high-frequency bands only: with the pyramid off the
        # blocks are never built and the key would be a silent no-op (the recorded silently-inert-argument class).
        blocks = model_args["high_freq_blocks"]
        if isinstance(blocks, int) and not isinstance(blocks, bool) and blocks > 0 and not model_args.get("use_laplacian_pyramid"):
            lines.append(
                f"Invalid value for 'model_args.high_freq_blocks': {blocks} blocks do nothing without "
                "model_args.use_laplacian_pyramid=true (they act on the pyramid's high-frequency bands)"
            )
    if (
        "trainable_gabor_stem" in ALLOWED[family]
        and model_args.get("trainable_gabor_stem") is False
        and not model_args.get("use_gabor_stem")
    ):
        lines.append(
            "Invalid value for 'model_args.trainable_gabor_stem': false freezes the Gabor stem, and "
            f"{family!r} builds none unless model_args.use_gabor_stem=true"
        )
    return lines


def factory_signatures(family: str) -> Dict[str, List[str]]:
    """Return ``{callable path: named parameters}`` of the library factories behind ``family`` (``**kwargs`` left out)."""
    return {
        path: [
            name for name, parameter in inspect.signature(load_attr(path)).parameters.items()
            if parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        ]
        for path in FACTORIES[family]
    }


def signature_census(
    family: str, signatures: Optional[Mapping[str, Sequence[str]]] = None
) -> Dict[str, List[str]]:
    """Return the factory parameters no table names and the table keys no factory has; see the module contract."""
    if signatures is None:
        signatures = factory_signatures(family)
    offered = {*ALLOWED[family], *REFUSED[family], *FORCED[family]}
    present = {name for names in signatures.values() for name in names}
    return {
        "unclassified": sorted(present - offered - set(SUPPLIED)),
        "stale": sorted(offered - present - set(ADAPTER_KEYS)),
    }


# ---------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------


def _shape_of(input_shape: Sequence[Optional[int]]) -> Tuple[Optional[int], Optional[int], int]:
    """Return ``(height, width, channels)``; a height or width may be ``None``, the channels may not."""
    if len(input_shape) != 3:
        raise ValueError(f"input_shape must be (height, width, channels), got {tuple(input_shape)!r}")
    height, width, channels = input_shape
    sides_ok = all(side is None or (isinstance(side, int) and not isinstance(side, bool) and side > 0) for side in (height, width))
    if not sides_ok or not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
        raise ValueError(
            f"input_shape must be (height, width, channels) with each side None or a positive int and "
            f"positive int channels, got {tuple(input_shape)!r}"
        )
    return height, width, channels


def _prepare(
    family: str, num_classes: Optional[int], input_shape: Sequence[Optional[int]], model_args: Mapping[str, Any]
) -> Tuple[Optional[int], Optional[int], int]:
    """Refuse what the family does not take and return the input shape; the adapters' common door."""
    if num_classes is not None:
        raise ValueError(f"the {family!r} family is a denoiser and has no classes: num_classes must be None, got {num_classes!r}")
    shape = _shape_of(input_shape)
    lines = problems_of(family, model_args)
    if lines:
        raise ValueError("; ".join(lines))
    return shape


def _activations(kwargs: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Replace ``"leaky_relu"`` under ``keys`` by one shared ``LeakyReLU(negative_slope=0.1)`` instance."""
    # DECISION plan_2026-06-21_eb7fd829/D-003
    # The bare string "leaky_relu" is Keras' slope 0.2, so an exact 0.1 needs an INSTANCE, which every legacy
    # trainer builds and shares between the stem, the blocks and the supervision heads (the layers wrap it in their
    # own Activation sublayers and serialize it). Do NOT pass the string "for simplicity": the slope changes and
    # nothing fails. `gabor_activation` and `final_activation` take names only and stay strings.
    import keras

    shared = None
    for key in keys:
        if kwargs.get(key) == "leaky_relu":
            shared = shared or keras.layers.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)
            kwargs[key] = shared
    return kwargs


def _freeze_gabor_stem(model: Any, trainable: bool) -> None:
    """Set ``trainable`` of the one ``gabor_stem`` layer to ``False`` when ``trainable`` is false."""
    if trainable:
        return
    stems = {id(layer): layer for layer in _layers_of(model) if layer.name == "gabor_stem"}
    if len(stems) != 1:
        raise ValueError(
            f"trainable_gabor_stem=false needs exactly one 'gabor_stem' layer, found {len(stems)}; "
            "set model_args.use_gabor_stem=true"
        )
    next(iter(stems.values())).trainable = False


def _float32_output(model: Any) -> Any:
    """Return ``model`` with its single output followed by a float32 ``Activation("linear")`` named ``output_float32``."""
    import keras

    # DECISION plan-2026-11-10T095307-d483ffab/D-019
    # Under `mixed_float16` every family emits float16 (measured, step 2): MSE, PSNR and SSIM would be read at half
    # precision. The legacy trainer casts the output back in `train()`; here the adapter does, always, so the
    # architecture does not depend on the precision a run chose and a model trained under one loads under the
    # other. The cast is a weight-free linear layer: homogeneity and weight transfer by layer name are untouched.
    # Do NOT cast in the loss or the metrics instead: `predict` and `export` would still return float16.
    output = keras.layers.Activation("linear", dtype="float32", name="output_float32")(model.output)
    # `model.input`, not `model.inputs`: the list form makes the model expect a LIST of one array and every direct call
    # with a plain array warns about a structure mismatch.
    return keras.Model(model.input, output, name=model.name)


def create_bfcnn(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Optional[int]] = DEFAULT_INPUT_SHAPE,
    **model_args: Any
) -> Any:
    """Build a bias-free CNN denoiser; see the module contract."""
    shape = _prepare("bfcnn", num_classes, input_shape, model_args)
    from dl_techniques.models.vision.bias_free_denoisers.bfcnn import create_bfcnn_variant

    kwargs = _activations({"final_activation": "linear", **model_args}, ("activation",))
    return _float32_output(create_bfcnn_variant(variant, shape, **kwargs))


def create_bfunet(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Optional[int]] = DEFAULT_INPUT_SHAPE,
    **model_args: Any
) -> Any:
    """Build a bias-free U-Net denoiser; see the module contract."""
    shape = _prepare("bfunet", num_classes, input_shape, model_args)
    from dl_techniques.models.vision.bias_free_denoisers.bfunet import create_bfunet_variant

    kwargs = _activations({"final_activation": "linear", **model_args}, ("activation",))
    trainable = kwargs.pop("trainable_gabor_stem", True)
    kwargs.pop("enable_deep_supervision", None)
    model = create_bfunet_variant(variant, shape, enable_deep_supervision=False, **kwargs)
    _freeze_gabor_stem(model, trainable)
    return _float32_output(model)


_SUPERVISED_ACTIVATIONS = ("block_activation", "stem_activation", "supervision_activation")


def create_bfconvunext(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Optional[int]] = DEFAULT_INPUT_SHAPE,
    **model_args: Any
) -> Any:
    """Build the bias-free ConvUNext denoiser; see the module contract."""
    shape = _prepare("bfconvunext", num_classes, input_shape, model_args)
    from dl_techniques.models.vision.bias_free_denoisers.bfconvunext import create_convunext_variant

    kwargs = _activations({"final_activation": "linear", **model_args}, _SUPERVISED_ACTIVATIONS)
    trainable = kwargs.pop("trainable_gabor_stem", True)
    for key in FORCED["bfconvunext"]:
        kwargs.pop(key, None)
    model = create_convunext_variant(
        variant, shape, enable_deep_supervision=False, include_top=True, output_channels=shape[2], **kwargs
    )
    _freeze_gabor_stem(model, trainable)
    return _float32_output(model)


def create_convunext_biased(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Optional[int]] = DEFAULT_INPUT_SHAPE,
    **model_args: Any
) -> Any:
    """Build the biased ConvUNext denoiser (``use_bias=True``); see the module contract."""
    shape = _prepare("convunext_biased", num_classes, input_shape, model_args)
    from dl_techniques.models.vision.convunext.model import create_convunext_variant

    kwargs = _activations({"final_activation": "linear", **model_args}, _SUPERVISED_ACTIVATIONS)
    trainable = kwargs.pop("trainable_gabor_stem", True)
    for key in FORCED["convunext_biased"]:
        kwargs.pop(key, None)
    model = create_convunext_variant(
        variant, shape, use_bias=True, enable_deep_supervision=False, include_top=True, output_channels=shape[2], **kwargs
    )
    _freeze_gabor_stem(model, trainable)
    return _float32_output(model)


def create_scunet(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Optional[int]] = DEFAULT_INPUT_SHAPE,
    **model_args: Any
) -> Any:
    """Build an SCUNet denoiser inside a functional model; see the module contract."""
    shape = _prepare("scunet", num_classes, input_shape, model_args)
    if variant not in SCUNET_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Available variants: {sorted(SCUNET_VARIANTS)}")
    import keras
    from dl_techniques.models.vision.image_restoration.scunet.model import create_scunet as build

    row = {key: list(value) if isinstance(value, list) else value for key, value in SCUNET_VARIANTS[variant].items()}
    kwargs = {**row, **{key: value for key, value in model_args.items() if key not in FORCED["scunet"]}}
    # `create_scunet` returns a subclassed model that no call has built, and a subclassed model saved before a call
    # writes no weights (measured, step 2). Calling it on a symbolic input builds it, and the wrapper is functional.
    core = build(in_nc=shape[2], include_top=True, **kwargs)
    inputs = keras.Input(shape=shape)
    outputs = keras.layers.Activation("linear", dtype="float32", name="output_float32")(core(inputs))
    return keras.Model(inputs, outputs, name=f"scunet_{variant}")


# ---------------------------------------------------------------------
# The stride rule and the observers
# ---------------------------------------------------------------------


def bfunet_stride(variant: str, model_args: Mapping[str, Any]) -> int:
    """Return ``2 ** depth`` for a bfunet variant; see the module contract."""
    depth = model_args.get("depth")
    if depth is None:
        from dl_techniques.models.vision.bias_free_denoisers.bfunet import BFUNET_CONFIGS

        depth = BFUNET_CONFIGS[variant]["depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 2:
        raise ConfigError(f"Invalid value for 'model_args.depth': a bfunet depth is an int of 2 or more, got {depth!r}")
    return 2**depth


def _blocks(model: Any) -> List[Any]:
    """Return the bias-free residual blocks and convolutions of the model, in layer order."""
    return [layer for layer in _layers_of(model) if type(layer).__name__ in ("BiasFreeConv2D", "BiasFreeResidualBlock")]


def _has(model: Any, class_name: str) -> bool:
    return any(type(layer).__name__ == class_name for layer in _layers_of(model))


def observe_bfcnn(model: Any) -> Optional[Dict[str, Any]]:
    """Return ``num_blocks`` and ``filters`` of a BFCNN's layers, ``None`` for any other model; see the module contract."""
    stems = [layer for layer in _blocks(model) if layer.name == "stem"]
    if not stems or _has(model, "DownsampleAndSkip"):
        return None
    return {
        "num_blocks": sum(type(layer).__name__ == "BiasFreeResidualBlock" for layer in _blocks(model)),
        "filters": int(stems[0].get_config()["filters"]),
    }


def observe_bfunet(model: Any) -> Optional[Dict[str, Any]]:
    """Return ``depth``, ``initial_filters`` and ``blocks_per_level`` of a bfunet's layers, ``None`` for any other model."""
    downsamples = sum(type(layer).__name__ == "DownsampleAndSkip" for layer in _layers_of(model))
    first = [layer for layer in _blocks(model) if layer.name.startswith("encoder_level_0_")]
    if not downsamples or not first or _has(model, "ConvStem"):
        return None
    observed: Dict[str, Any] = {"depth": downsamples, "initial_filters": int(first[0].get_config()["filters"])}
    inside = [
        layer for layer in _blocks(model) if layer.name.startswith(("encoder_level_", "decoder_level_", "bottleneck_"))
    ]
    if len(inside) % (2 * downsamples + 1) == 0:
        observed["blocks_per_level"] = len(inside) // (2 * downsamples + 1)
    return observed


def observe_scunet(model: Any) -> Optional[Dict[str, Any]]:
    """Return ``config``, ``dim`` and ``head_dim`` of the SCUNet layer, ``None`` for a model without one."""
    for layer in _layers_of(model):
        if type(layer).__name__ == "SCUNet":
            config = layer.get_config()
            return {key: config[key] for key in ("config", "dim", "head_dim")}
    return None


# ---------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------


def resolve_denoising_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's; see the module contract."""
    decay = recipes.sgd_weight_decay(optim_args)
    if decay is not None:
        # The adapters pass no kernel regularizer, so sgd's decay (an L2 penalty on the kernels in the other tasks)
        # would be dropped without a word; the segmentation resolver refuses it for the same reason (its D-047).
        raise ConfigError(
            f"Invalid value for 'optim_args.weight_decay': optim_args.optimizer=sgd takes its weight decay as an L2 "
            f"penalty on the model's kernels, and the image_denoising task builds none for {context.family!r}, so "
            f"weight_decay={decay!r} would be dropped without a word; set optim_args.weight_decay=null for no "
            "decay, or use optim_args.optimizer=adamw, whose decay is decoupled and applied by the optimizer"
        )
    merged = recipes.merge_model_args(context, optim_args, model_args, {})
    lines = problems_of(context.family, merged)
    if lines:
        raise ConfigValidationError(lines)
    return merged


# ---------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------

# train/bfunet/common.py TrainingConfig: epochs :1394-1396 (100), batch :1360 (16), lr (1e-3), weight decay (0.004),
# cosine over all epochs, clip 1.0 as `gradient_clipping_by_norm` (a GLOBAL norm), warmup `max(1, round(0.1 * epochs))`
# (:1491: 10 at 100 epochs), early stopping off (`early_stopping_patience=-1`, because the curriculum makes the monitor
# non-monotonic). The API has no switch for it, so the patience is the recipe's epochs: it cannot fire in a default run.
_OPTIM = recipes.optim_recipe(
    optimizer="adamw", lr=1e-3, weight_decay=4e-3, warmup_epochs=10, decay_over="total", global_clipnorm=1.0
)


def _denoising_recipe(
    model_args: Mapping[str, Any], batch_size: int = 16, per_variant: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Return a denoising recipe; the optimizer is every family's, the model keys are the family's."""
    return {
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "optim_args": dict(_OPTIM),
        "checkpoint_args": {"early_stopping_patience": EPOCHS},
        "model_args": dict(model_args),
        "per_dataset": {},
        "per_variant": dict(per_variant or {}),
    }


#: `base` at 256 px needs batch 4 on the 24 GB card (train/bfunet/README.md "Memory note"; step 2(e) measured
#: 10.3 GB at batch 4, 256 px, bfconvunext/base).
_BASE_BATCH = {"base": {"batch_size": 4}}

_LEAKY = "leaky_relu"
_BF_NORM = "batchnorm"

BFCNN = _denoising_recipe({"activation": _LEAKY, "normalization_type": _BF_NORM})
BFUNET = _denoising_recipe(
    {"activation": _LEAKY, "block_normalization": _BF_NORM, "use_gabor_stem": True, "final_projection_groups": 1},
    per_variant=_BASE_BATCH,
)
BFCONVUNEXT = _denoising_recipe(
    {
        "convnext_version": "v1",
        "block_normalization": _BF_NORM,
        "block_activation": _LEAKY,
        "stem_activation": _LEAKY,
        "supervision_activation": _LEAKY,
        "use_gabor_stem": True,
        "final_projection_groups": 1,
    },
    per_variant=_BASE_BATCH,
)
CONVUNEXT_BIASED = _denoising_recipe({}, per_variant=_BASE_BATCH)
#: SCUNet tiny peaks at 4,385 MB at 256 px and batch 8 (step 2(d)); batch 16 was measured at 128 px only. Small and base
#: are not measured at any size.
SCUNET = _denoising_recipe({}, batch_size=8)


def builtin_denoising_models() -> Tuple[ModelSpec, ...]:
    """Return the five denoising family specs, freshly built; see the module contract."""
    unet_stride = f"{_SEG}:convunext_stride"
    return (
        ModelSpec(
            family="bfcnn",
            factory=f"{_HERE}:create_bfcnn",
            variants_attr=f"{_BF}.bfcnn:BFCNN_CONFIGS",
            tasks=(TASK,),
            defaults=BFCNN,
            resolve_model_args=f"{_HERE}:resolve_denoising_args",
            observe=f"{_HERE}:observe_bfcnn",
            properties=("bias_free",),
        ),
        ModelSpec(
            family="bfunet",
            factory=f"{_HERE}:create_bfunet",
            variants_attr=f"{_BF}.bfunet:BFUNET_CONFIGS",
            tasks=(TASK,),
            defaults=BFUNET,
            resolve_model_args=f"{_HERE}:resolve_denoising_args",
            stride=f"{_HERE}:bfunet_stride",
            observe=f"{_HERE}:observe_bfunet",
            properties=("bias_free",),
        ),
        ModelSpec(
            family="bfconvunext",
            factory=f"{_HERE}:create_bfconvunext",
            variants_attr=f"{_CONVUNEXT}:CONVUNEXT_CONFIGS",
            tasks=(TASK,),
            defaults=BFCONVUNEXT,
            resolve_model_args=f"{_HERE}:resolve_denoising_args",
            stride=unet_stride,
            observe=f"{_SEG}:observe_convunext",
            properties=("bias_free",),
        ),
        ModelSpec(
            family="convunext_biased",
            factory=f"{_HERE}:create_convunext_biased",
            variants_attr=f"{_CONVUNEXT}:CONVUNEXT_CONFIGS",
            tasks=(TASK,),
            defaults=CONVUNEXT_BIASED,
            resolve_model_args=f"{_HERE}:resolve_denoising_args",
            stride=unet_stride,
            observe=f"{_SEG}:observe_convunext",
        ),
        ModelSpec(
            family="scunet",
            factory=f"{_HERE}:create_scunet",
            variants_attr=f"{_HERE}:SCUNET_VARIANTS",
            tasks=(TASK,),
            defaults=SCUNET,
            resolve_model_args=f"{_HERE}:resolve_denoising_args",
            observe=f"{_HERE}:observe_scunet",
        ),
    )
