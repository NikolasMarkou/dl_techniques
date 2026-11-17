"""The depth-estimation model family of ``python -m train``: the adapter, its tables, the weights check and the recipe.

One family trains the ``depth_estimation`` task: ``depth_anything`` (``DepthAnything``: a DINOv2 encoder and a DPT
head), variants ``vit_s``, ``vit_b`` and ``vit_l``. The library's ``from_variant`` takes ``image_shape``, not the
``input_shape`` the API calls a factory with, and accepts knobs this task does not train (feature alignment, the
semi-supervised branch, a placeholder encoder); the adapter closes that gap and refuses the rest by name. Importing
this module imports no model code and no keras. The spec is built by :func:`builtin_depth_models`.

Interface contract
------------------
``create_depth_anything(*, variant, num_classes=None, input_shape, **model_args)``
    :param variant: a key of ``DepthAnything.MODEL_VARIANTS``.
    :param num_classes: must be ``None``: the output is relative disparity, not classes.
    :param input_shape: ``(s, s, 3)``, ``s`` a positive multiple of :data:`~train.api.depth_args.DINOV2_PATCH`.
    :param model_args: the keys :data:`ALLOWED` offers (``pretrained``) and :data:`FORCED` keys at their value.
    :return: a ``DepthAnything`` BUILT at ``(None, s, s, 3)`` with the :data:`FORCED` values, its encoder loaded from
        the official DINOv2 weights when :func:`~train.api.depth_args.pretrained_start` holds.
    :raises ValueError: for a non-``None`` ``num_classes``, a bad shape, a refused or unknown key, or an unknown variant
        (:func:`train.api.recipes.build_model` turns it into a ``ConfigError``).

``problems_of(model_args)``
    :return: one refusal line per bad key (``[]`` for none): unknown, :data:`REFUSED`, :data:`SUPPLIED`, a
        :data:`FORCED` key at another value, a ``pretrained`` that is not a bool. Pure.

``check_weights(variant, model_args)``
    :raises ConfigError: from a pretrained start whose official DINOv2 file is missing, naming the path, the environment
        variable ``DL_TECHNIQUES_DINOV2_WEIGHTS_DIR`` and the download, through the library's
        ``official_dinov2_weights_path`` (the one home of the path rule). Imports the library (TensorFlow loads, no
        tensor is made, no device is touched); never builds a model.

``resolve_depth_args(context, optim_args, model_args)``
    The family's ``resolve_model_args``. :return: the factory keywords, the recipe's then the caller's.
    :raises ConfigValidationError: naming every refused key at once. :raises ConfigError: for an sgd weight decay (no
        model regularizer is built), an optimizer other than adamw on a pretrained start (the head's 10x table is
        ``LRMultiplierAdamW``'s), and a missing weights file (:func:`check_weights`). The task calls it before the claim.

``factory_signatures()`` / ``signature_census(signatures=None)``
    The denoising pattern: ``{"unclassified": [...], "stale": [...]}``, the parameters of the callables in
    :data:`FACTORIES` the tables do not name, and the table keys no callable has. A passing census is both empty.

``JIT_COMPILE``
    ``False``: the value the task's ``compile_options`` returns explicitly (D-016).

``builtin_depth_models()``
    :return: the one :class:`~train.api.registry.ModelSpec`, freshly built (``tasks=("depth_estimation",)``).

Recipe
------
Every variant trains the legacy trainer's pretrained recipe at 252 px, batch 8 (measured: vit_b 9.5 GiB and
vit_l 16.7 GiB, both at batch 8): AdamW at 5e-6 for the encoder (the head at 10x through the task's
``optimizer_options``), weight decay 1e-5, a GLOBAL clip of 1.0, one warmup epoch, cosine over the whole run to 1
percent, 30 epochs of 100 steps (D-019: comparable with the legacy 3,000 steps), no early stop inside the default run.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import recipes
from .args import AutoContext, OptimArgs, is_count
from .config import unknown_key_message
from .denoising_models import named_parameters
from .depth_args import (
    DINOV2_PATCH,
    PRETRAINED,
    PRETRAINED_DEFAULT,
    PRETRAINED_LR,
    TASK,
    WATCH_METRIC,
    pretrained_start,
)
from .errors import ConfigError, ConfigValidationError
from .registry import ModelSpec, load_attr

# ---------------------------------------------------------------------

_HERE = "train.api.depth_models"
_MODEL = "dl_techniques.models.vision.depth_anything.model"

FAMILY = "depth_anything"
VARIANTS = f"{_MODEL}:DepthAnything.MODEL_VARIANTS"

_DECAY = "the optimizer decays the weights (adamw), a model regularizer would decay them twice"
_SEMI = ("the semi-supervised branch (unlabeled batches, their CutMix and colour jitter, the teacher's loss terms) is "
         "not trained by this task: it trains the labeled step only")
_HEAD = "the DPT head is the published checkpoint's for the variant, the recipe that was measured"

#: Keys passed through as the caller gave them.
ALLOWED: Tuple[str, ...] = (PRETRAINED,)

#: Keys refused whatever their value: ``{key: why}``.
REFUSED: Dict[str, str] = {
    "dpt_features": _HEAD,
    "dpt_out_channels": _HEAD,
    "output_channels": "the output is one channel of relative disparity, which the loss and the metrics read",
    "kernel_initializer": "the head starts from the library's he_normal, the measured start",
    "kernel_regularizer": _DECAY,
    "loss_weights": _SEMI,
    "cutmix_prob": _SEMI,
    "color_jitter_strength": _SEMI,
    "enable_semi_supervised": _SEMI,
    "encoder": "the variant builds the DINOv2 encoder the official weights load into",
}

# DECISION plan-2026-11-17T110935-9c7b9096/D-007
# The four keys the adapter sets, a user's other value refused by name before the claim: the real DINOv2 encoder (the
# official weights load only into it), no feature-alignment term, the head on, and NO input range (the data is
# ImageNet-standardized, and a (0, 1) range would clip it in the model's colour jitter). Do NOT move a key to ALLOWED
# "for experiments": each changes what is trained or what the loss reads, and none was measured.
#: Keys the adapter sets to one value: ``{key: (value, why)}``; any other value is refused, the same is a no-op.
FORCED: Dict[str, Tuple[Any, str]] = {
    "encoder_kind": ("real", "the official DINOv2 weights load only into the real encoder"),
    "use_feature_alignment": (False, "the alignment term needs the frozen teacher and a second loss this task does not train"),
    "include_top": (True, "the DPT head is the depth output"),
    "input_value_range": (None, "the input is ImageNet-standardized, which a declared range would clip"),
}

#: Keywords the adapter takes from the model name or the data: naming one in ``model_args`` is refused.
SUPPLIED: Dict[str, str] = {
    "variant": "model=depth_anything/<variant> chooses it",
    "encoder_type": "it is the variant, model=depth_anything/<variant>",
    "image_shape": "it is transform_args.image_size, square, 3 channels",
}

#: The library callables :func:`signature_census` classifies: the factory and the constructor behind its ``**kwargs``.
FACTORIES: Tuple[str, ...] = (f"{_MODEL}:DepthAnything.from_variant", f"{_MODEL}:DepthAnything")


def _offered() -> List[str]:
    return sorted({*ALLOWED, *REFUSED, *FORCED})


def problems_of(model_args: Mapping[str, Any]) -> List[str]:
    """Return one line per refusal of ``model_args``; see the module contract."""
    lines: List[str] = []
    for key, value in model_args.items():
        name = f"model_args.{key}"
        if key == PRETRAINED:
            if not isinstance(value, bool):
                lines.append(f"Invalid value for {name!r}: expected true or false, got {value!r}")
        elif key in FORCED:
            forced, why = FORCED[key]
            if value != forced or isinstance(value, bool) != isinstance(forced, bool):
                lines.append(f"Invalid value for {name!r}: the {FAMILY!r} adapter sets it to {forced!r}: {why}, got {value!r}")
        elif key in SUPPLIED:
            lines.append(f"Invalid value for {name!r}: {key} is set by the task ({SUPPLIED[key]}), got {value!r}")
        elif key in REFUSED:
            lines.append(f"Invalid value for {name!r}: {key} is not offered by {FAMILY!r} ({REFUSED[key]}), got {value!r}")
        else:
            lines.append(unknown_key_message("model_args", key, _offered()))
    return lines


def factory_signatures() -> Dict[str, List[str]]:
    """Return ``{callable path: named parameters}`` of the library callables behind the family."""
    return {path: named_parameters(path) for path in FACTORIES}


def signature_census(signatures: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, List[str]]:
    """Return the callable parameters no table names and the table keys no callable has; see the module contract."""
    if signatures is None:
        signatures = factory_signatures()
    offered = {*ALLOWED, *REFUSED, *FORCED, *SUPPLIED}
    present = {name for names in signatures.values() for name in names}
    return {"unclassified": sorted(present - offered), "stale": sorted(offered - present)}


def check_weights(variant: str, model_args: Mapping[str, Any]) -> None:
    """Refuse a pretrained start whose official DINOv2 file is missing; see the module contract."""
    # DECISION plan-2026-11-17T110935-9c7b9096/D-007
    # A missing weights file is a ConfigError BEFORE the run directory is claimed, through the library's own path rule.
    # Do NOT fall back to a random encoder (from scratch the model only learns the average layout, and the user asked
    # for no fallbacks), and do NOT leave it to from_variant: that raises after the claim, inside the model build.
    if not pretrained_start(model_args):
        return
    from dl_techniques.models.vision.dino.official_weights import official_dinov2_weights_path

    scale = load_attr(VARIANTS)[variant]["vit_scale"]
    try:
        official_dinov2_weights_path(scale)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"Invalid value for 'model_args.{PRETRAINED}': {FAMILY}/{variant} starts its encoder from the official "
            f"DINOv2 weights. {exc} Or set model_args.{PRETRAINED}=false to train from random weights (the bar is "
            "then expected to fail: from scratch the model reaches the average layout)") from exc


# ---------------------------------------------------------------------
# The adapter and the resolver
# ---------------------------------------------------------------------


def create_depth_anything(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], **model_args: Any
) -> Any:
    """Build a Depth Anything at ``input_shape``; see the module contract."""
    if num_classes is not None:
        raise ValueError(f"{FAMILY!r} predicts relative disparity and has no classes: num_classes must be None, "
                         f"got {num_classes!r}")
    shape = tuple(input_shape)
    if len(shape) != 3 or not all(is_count(side) for side in shape):
        raise ValueError(f"input_shape must be (size, size, 3), each a positive int, got {shape!r}")
    if shape[0] != shape[1]:
        raise ValueError(f"input_shape must be square (the crops are), got {shape!r}")
    if shape[2] != 3:
        raise ValueError(f"input_shape must have 3 channels (RGB), got {shape!r}")
    if shape[0] % DINOV2_PATCH:
        raise ValueError(f"the side must be a multiple of {DINOV2_PATCH} (the DINOv2 patch), got {shape!r}")
    lines = problems_of(model_args)
    if lines:
        raise ValueError("; ".join(lines))
    from dl_techniques.models.vision.depth_anything.model import DepthAnything

    forced = {key: value for key, (value, _) in FORCED.items()}
    return DepthAnything.from_variant(variant, pretrained=pretrained_start(model_args), image_shape=shape, **forced)


def resolve_depth_args(context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's; see the module contract."""
    recipes.refuse_dropped_sgd_decay(optim_args, TASK, context.family)
    merged = recipes.merge_model_args(context, optim_args, model_args, {})
    lines = problems_of(merged)
    if lines:
        raise ConfigValidationError(lines)
    optimizer = str(optim_args.optimizer).strip().lower()
    if pretrained_start(merged) and optimizer != "adamw":
        raise ConfigError(
            f"Invalid value for 'optim_args.optimizer': a pretrained start trains the DPT head at 10x the encoder's rate, "
            f"a table only adamw takes, got {optimizer!r}; use adamw, or set model_args.{PRETRAINED}=false")
    check_weights(context.variant, merged)
    return merged


# ---------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------

EPOCHS = 30
BATCH_SIZE = 8
IMAGE_SIZE = 252
WEIGHT_DECAY = 1e-5
GLOBAL_CLIPNORM = 1.0
WARMUP_EPOCHS = 1
#: Early stopping counts from the first epoch; the default run is 30 epochs, so it never stops early.
PATIENCE = EPOCHS

# DECISION plan-2026-11-17T110935-9c7b9096/D-016
# jit_compile is False and the task returns it EXPLICITLY: XLA was 4x faster per compute step, but the stream is data
# bound (66 to 70 percent wait with XLA), XLA adds about 150 s per process, and a run and its resume would compile
# twice. Do NOT leave it unset (Keras resolves auto to XLA on a GPU, so CPU tests and the GPU gate would build different
# graphs), and do NOT judge a later XLA comparison by a 50-step band (two identical False runs drift 0.036 per batch).
JIT_COMPILE = False

RECIPE: Dict[str, Any] = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optim_args": recipes.optim_recipe(
        optimizer="adamw", lr=PRETRAINED_LR, weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
        decay_over="total", global_clipnorm=GLOBAL_CLIPNORM,
    ),
    "transform_args": {"image_size": IMAGE_SIZE},
    "metric_args": {"watch_metric": WATCH_METRIC},
    "checkpoint_args": {"early_stopping_patience": PATIENCE},
    "model_args": {PRETRAINED: PRETRAINED_DEFAULT},
    "per_dataset": {},
    "per_variant": {},
}


def builtin_depth_models() -> Tuple[ModelSpec, ...]:
    """Return the one depth family spec, freshly built; see the module contract."""
    return (
        ModelSpec(
            family=FAMILY,
            factory=f"{_HERE}:create_depth_anything",
            variants_attr=VARIANTS,
            tasks=(TASK,),
            defaults=RECIPE,
            resolve_model_args=f"{_HERE}:resolve_depth_args",
        ),
    )
