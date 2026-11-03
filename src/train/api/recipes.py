"""The model families of image classification, as data, and their model-argument rules.

Each :class:`~train.api.registry.ModelSpec` below carries the defaults of one
legacy trainer (``vit``, ``resnet``, ``convnext``, ``cliffordnet``), so
``model=vit/vit_pico`` trains the way ``train.vit.train_vit`` does unless the
caller says otherwise (D-003, D-013). A comment beside each value names the
legacy line it was read from.

Interface contract
------------------
``register_builtin_models()``
    Register the six classification families and the two segmentation ones
    (:func:`train.api.segmentation_models.builtin_segmentation_models`) into
    :data:`train.api.registry.MODELS`. Idempotent: a family already registered is
    left alone. Imports no model code.

``resolve_<family>_args(context, optim_args, model_args)``
    The ``ModelSpec.resolve_model_args`` of each family.

    :param context: an :class:`~train.api.args.AutoContext` whose ``defaults``
        is ``spec.recipe(dataset, variant)``.
    :param optim_args: the RESOLVED :class:`~train.api.args.OptimArgs`; the
        weight-decay rule reads ``optimizer`` and ``weight_decay``.
    :param model_args: the caller's free ``model_args`` mapping.
    :return: the keyword arguments for the family factory, without ``variant``,
        ``num_classes`` and ``input_shape``. A caller's key replaces the
        recipe's.
    :raises UnresolvedAutoError: when ``optim_args`` still holds ``"auto"``.
    :raises ConfigError: when ``model_args`` names a key the library supplies
        (``variant``, ``num_classes``, ``input_shape``), or hands an AdamW run
        a ``kernel_regularizer``.

``serialize_model_args(kwargs)``
    :param kwargs: the output of a resolver.
    :return: the same keywords with a regularizer as its serialized config, so
        ``config.json`` can record what the model was built with.

``build_model(spec, variant, num_classes, input_shape, kwargs)``
    :param spec: the family's :class:`~train.api.registry.ModelSpec`.
    :param variant: variant name.
    :param num_classes: output classes.
    :param input_shape: ``(height, width, channels)``.
    :param kwargs: the output of a resolver.
    :return: the model the factory returns (built or not, as the factory leaves it).
    :raises ConfigError: when the factory refuses the call (an unknown
        ``model_args`` key, an input the family cannot take); the factory's
        own message is kept.

Recipe keys
-----------
Every key of a recipe is in :data:`RECIPE_KEYS`. A group's entry lists EVERY
AUTO field the recipe resolves, ``None`` for an option that does not apply to
its optimizer, so a field missing from the recipe is a defect a test finds, not
an ``"auto"`` the user meets. ``image_size``, ``watch_mode`` and
``success_threshold`` are not listed: the data and the metric's name decide
them.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012);
``keras`` is imported inside the resolvers.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

from .args import ERASING_PRESET, AutoContext, OptimArgs
from .config import AUTO
from .errors import ConfigError, UnresolvedAutoError
from . import registry
from .registry import ModelSpec

# ---------------------------------------------------------------------

TASK = "image_classification"

#: Side of the CIFAR images. The vit patch and the resnet stem change here.
SMALL_IMAGE_SIZE = 32

#: Keys a recipe may hold. ``model_args`` is the data half of the model-argument
#: rule, ``loader`` the arguments of ``load_classification_data`` the recipe
#: decides (read by the data step), and the five group keys are those of the
#: task config (``analysis_args`` only in a ``per_dataset`` overlay: no family's base recipe
#: has one, D-040).
RECIPE_KEYS = (
    "epochs",
    "batch_size",
    "optim_args",
    "transform_args",
    "metric_args",
    "checkpoint_args",
    "analysis_args",
    "model_args",
    "loader",
    "per_dataset",
    "per_variant",
)

#: Keys the library passes to a factory itself.
_SUPPLIED = ("variant", "num_classes", "input_shape")

_MODELS = "dl_techniques.models.vision"


def optim_recipe(
    *,
    optimizer: str,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    decay_over: str,
    momentum: Any = None,
    beta_1: Any = None,
    beta_2: Any = None,
    epsilon: Any = None,
    clipnorm: Any = None,
    global_clipnorm: Any = None,
) -> Dict[str, Any]:
    """Return a complete ``optim_args`` recipe: all 14 fields, none left out.

    Public because the segmentation families build their recipes with it
    (:mod:`train.api.segmentation_models`); a second table of the 14 fields would have to be
    kept equal to this one by hand.

    The schedule is cosine to 1% of the peak from 1e-8 in every legacy trainer
    (``alpha`` 0.01, ``warmup_start_lr`` the builder's default), so those
    three are constants here. Everything else is a parameter, so a family
    cannot forget a field.
    """
    return {
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "epsilon": epsilon,
        "schedule": "cosine_decay",
        "warmup_epochs": warmup_epochs,
        "warmup_start_lr": 1e-8,
        "min_lr_ratio": 0.01,
        "decay_over": decay_over,
        "clipnorm": clipnorm,
        "global_clipnorm": global_clipnorm,
    }


def _recipe(
    *,
    epochs: int,
    batch_size: int,
    optim_args: Dict[str, Any],
    watch_metric: str,
    patience: int,
    model_args: Dict[str, Any],
    per_dataset: Dict[str, Any],
    per_variant: Optional[Dict[str, Any]] = None,
    erasing_prob: float = 0.0,
) -> Dict[str, Any]:
    """Return a recipe: the AUTO keys every family lists, plus its overlays.

    :param erasing_prob: the random-erasing probability the family's
        ``cifar_autoaugment`` preset takes; ``TransformArgs.resolve_auto`` uses it
        under that preset only and gives 0.0 under any other.
    """
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "optim_args": optim_args,
        # No augmentation and no erasing unless a dataset overlay says so.
        "transform_args": {"augment": "none", "random_erasing_prob": erasing_prob},
        "metric_args": {"watch_metric": watch_metric},
        "checkpoint_args": {"early_stopping_patience": patience},
        "model_args": model_args,
        "per_dataset": per_dataset,
        "per_variant": per_variant or {},
    }


# ---------------------------------------------------------------------
# The recipes. Paths are under src/train/.
# ---------------------------------------------------------------------

# vit/train_vit.py: epochs :83, batch :67, lr :84, warmup :87, wd :88, clip :89
# (a GLOBAL norm, optimizer_builder "gradient_clipping_by_norm" :296), patience
# :103, monitor val_accuracy :198, schedule decay_steps = spe * epochs :289
# (decay_over total), alpha 0.01 :291. dropout 0.1 and attention dropout 0.1
# :79-80. The CIFAR hook is `cifar_light` (_augment_hook :144-157); ImageNet keeps
# the loader's own crop and flip, so its augment is "none".
# DECISION plan-2026-10-31T180033-2faf890a/D-035: every (family, dataset) pair of ImageNet and
# Imagenette has an entry in `per_dataset`, even one equal to the base recipe, because the "No recipe
# overlay" notice (ImageClassificationTrainer.recipe_notices) looks for the dataset as a KEY. Do NOT
# delete an entry because it changes nothing, and do NOT give an Imagenette pair the ImageNet epochs
# (50 or 100 epochs of 66 steps is 3,300 or 6,600 steps, below the one measured Imagenette budget of
# 13,200). The rule behind each new number is in the table of src/train/README.md.
VIT = _recipe(
    epochs=50,
    batch_size=128,
    optim_args=optim_recipe(
        optimizer="adamw", lr=3e-4, weight_decay=0.05, warmup_epochs=5,
        decay_over="total", global_clipnorm=1.0,
    ),
    watch_metric="val_accuracy",
    patience=15,
    model_args={"dropout_rate": 0.1, "attention_dropout_rate": 0.1},
    per_dataset={
        "cifar10": {"transform_args": {"augment": "cifar_light"}},
        "cifar100": {"transform_args": {"augment": "cifar_light"}},
        # ImageNet is the base recipe (train_vit.py:83-89 and the loader's own crop and flip,
        # augment None :151-157); the entry says so. Label smoothing is 0 (:99, :125).
        "imagenet": {"transform_args": {"augment": "none"}},
        # NEW, no legacy source (train_vit.py offers cifar10, cifar100 and imagenet only, :375).
        # 200 epochs of 66 steps (8,523 images, batch 128) is the step budget of the one legacy
        # Imagenette recipe (cliffordnet, 200 epochs :664); the warmup keeps the legacy 10% of the
        # schedule (5 of 50 epochs, :83 and :87). Every rule is in the README table and D-035.
        "imagenette": {
            "epochs": 200,
            "optim_args": {"warmup_epochs": 20},
            "transform_args": {"augment": "none"},
        },
    },
)

# resnet/train_resnet.py: epochs :74, batch :61, sgd :76, lr :75, warmup :78,
# wd :79 (an L2 regularizer on the model, :366; the optimizer gets none),
# momentum :81, clip :80 (GLOBAL), patience :89, monitor val_loss :307, schedule
# decay_steps = spe * epochs :385 with alpha 0.01 :386. Its label_smoothing 0.1 is a
# dead field (the loss never reads it) and is NOT copied (D-003).
# ImageNet uses the loader's crop and flip plus colour jitter :347. The legacy
# trainer cannot run on CIFAR at all; CIFAR here is new behaviour with the CIFAR
# stem (resolve_resnet_args) and no augmentation, as nothing legacy says otherwise.
RESNET = _recipe(
    epochs=100,
    batch_size=256,
    optim_args=optim_recipe(
        optimizer="sgd", lr=0.1, weight_decay=1e-4, warmup_epochs=5,
        decay_over="total", momentum=0.9, global_clipnorm=1.0,
    ),
    watch_metric="val_loss",
    patience=15,
    model_args={},
    per_dataset={
        "imagenet": {"loader": {"imagenet_augment_color": True}},
        # NEW, no legacy source (train_resnet.py is ImageNet only). Batch 128, because batch 256 is
        # 33 steps an epoch on 8,523 images; lr 0.05 is the linear rule of Goyal et al. 2017
        # (0.1 * 128 / 256); 200 epochs of 66 steps is the cliffordnet Imagenette step budget;
        # warmup 10 epochs keeps the legacy 5% of the schedule (5 of 100, :74 and :78). The loader
        # refuses imagenet_augment_color for Imagenette (classification_data.py:447-448), so the
        # augmentation is the loader's crop and flip alone. README table, D-035.
        "imagenette": {
            "epochs": 200,
            "batch_size": 128,
            "optim_args": {"lr": 0.05, "warmup_epochs": 10},
            "transform_args": {"augment": "none"},
        },
    },
)


def _convnext(strides: int, drop_path_rate: float, dropout_rate: float) -> Dict[str, Any]:
    return {
        "model_args": {
            "strides": strides,
            "drop_path_rate": drop_path_rate,
            "dropout_rate": dropout_rate,
        }
    }


# convnext/common.py: epochs :258, batch :259, lr :260, wd :261, warmup :264,
# patience :265, MONITOR val_loss :176. A direct AdamW with a PER-VARIABLE
# clipnorm 1.0 (GRADIENT_CLIP_NORM :173), the schedule decaying over
# `total - warmup` (decay_over post_warmup; it refuses warmup >= epochs).
# DEFAULT_STRIDES :155 and REGULARIZATION_DEFAULTS :160-165 give the per-dataset
# model arguments; the D-033 anchor in TrainingConfig explains strides 2 on 32 px.
# Augmentation is flip_pad_crop with a flip on CIFAR only (:909); MNIST's legacy
# pad-crop WITHOUT a flip has no preset, so MNIST gets "none" here. MNIST is
# repeated to three channels (:907), a loader argument.
CONVNEXT_PER_DATASET: Dict[str, Any] = {
    "mnist": {**_convnext(2, 0.1, 0.1), "loader": {"channels": 3}},
    "cifar10": {
        **_convnext(2, 0.1, 0.1),
        "transform_args": {"augment": "flip_pad_crop"},
    },
    "cifar100": {
        **_convnext(2, 0.2, 0.2),
        "transform_args": {"augment": "flip_pad_crop"},
    },
    "imagenet": _convnext(4, 0.1, 0.0),
    # NEW, no legacy source: the legacy trainer REFUSES Imagenette (UNSUPPORTED_DATASETS :140). The
    # ImageNet row, because it is the legacy table's only 224 px row (strides 4 :155, the ConvNeXt-T
    # rates :160-165) and the recipe's epochs and batch stay the legacy 100 and 64: 133 steps an
    # epoch is 13,300 steps, the cliffordnet Imagenette budget. Without an entry the factory
    # defaults would apply (drop path 0). README table, D-035.
    "imagenette": _convnext(4, 0.1, 0.0),
}


def _convnext_recipe() -> Dict[str, Any]:
    return _recipe(
        epochs=100,
        batch_size=64,
        optim_args=optim_recipe(
            optimizer="adamw", lr=1e-3, weight_decay=1e-4, warmup_epochs=0,
            decay_over="post_warmup", clipnorm=1.0,
        ),
        watch_metric="val_loss",
        patience=50,
        model_args={},
        per_dataset=CONVNEXT_PER_DATASET,
    )


# cliffordnet/train_cliffordnet.py: epochs 200 :664, batch 128 :653, wd 0.1 :652,
# patience 30 :655, monitor val_accuracy :354, warmup 0 :642, lr 1e-3
# (train/common/args.py:345). AdamW with betas 0.9/0.999 and epsilon 1e-8
# :313-319, NO clipping, the schedule over `total - warmup` with alpha 1e-2 and
# a 1e-8 start :299-311. CIFAR trains with `cifar_autoaugment` and random
# erasing 0.25 (:680-685); other datasets with none. The 0.25 belongs to the
# preset, not to the dataset: `--augment cifar_autoaugment` on any other dataset
# also erases with 0.25 (:684-685), so the base recipe carries 0.25 and
# TransformArgs.resolve_auto applies it under that preset only (D-041).
# Stochastic depth is 0.3, 0.4 for cn_64_5 (_STOCHASTIC_DEPTH_BY_VARIANT
# :98-104), 0.1 for hier_345.
#
# ImageNet and Imagenette are the legacy trainer's own (:241-251): the loader's random-resized-crop
# and flip, `--augment none` (:681), 224 px, label smoothing 0 (:611). The BATCH is not the legacy
# 128: step 17 ran 20 real steps through this API on GPU 0 (RTX 4090, 24,564 MiB) with XLA, halving
# the batch from 128 until a run fit (allocator peak <= 20,000 MiB, device <= 23,000 MiB, no
# allocator garbage collection or out-of-memory line). Measured at 224 px, Imagenette and ImageNet
# alike (memory_fit_evidence.json, README table):
#   cliffordnet/nano      128 OOM (asks 147.6 GiB), 64 OOM, 32 OOM (asks 32.2 GiB, the legacy
#                         docstring's 34.6 GB), 16 completes at peak 18,388 MiB with an allocator
#                         garbage collection (no margin), 8 fits: peak 10,869 MiB
#   cliffordnet/lite      same, 16 still OOM, 8 fits: peak 16,456 MiB
#   cliffordnet_hier/hier_345  128 and 64 OOM, 32 completes with a garbage collection (peak
#                         18,442 MiB), 16 fits: peak 11,601 MiB
# so cliffordnet takes 8 and cliffordnet_hier 16. The learning rate stays 1e-3: the legacy trainer
# has no batch rule (1e-3 at batch 128 in the paper recipe and at batch 32 in its README), so none
# is invented; no run has checked that 1e-3 suits a batch of 8. The image size stays the data's own
# 224: it is not a recipe key here (the data decides it before the recipe is read), and
# `transform_args.image_size=128 batch_size=32` is the legacy README's form, measured to fit for
# nano (peak 14,315 MiB; lite needs 16 there). A variant larger than these is not covered
# (cn_64_5 needs batch 2 at 224 px; its CIFAR recipe is the paper's and cannot take that batch).
# DECISION plan-2026-10-31T180033-2faf890a/D-036: do NOT put these batches into `per_variant` (it
# applies to every dataset and would change the CIFAR recipes) and do NOT restore 128.
def _cliffordnet_per_dataset(large_image_batch: int) -> Dict[str, Any]:
    return {
        **{
            name: {"transform_args": {"augment": ERASING_PRESET}}
            for name in ("cifar10", "cifar100")
        },
        **{
            name: {
                "batch_size": large_image_batch,
                "transform_args": {"augment": "none"},
                # DECISION plan-2026-10-31T180033-2faf890a/D-040
                # The evaluation batch is the batch step 17 measured to TRAIN at, so every
                # prediction after `fit` (figures, analyzer, load check) is bounded by a
                # number already shown to fit. Inference needs a fraction of training memory
                # (nano, real weights, 224 px: 924 MiB at 8, 2,555 at 32, 17,775 at 256).
                # Do NOT drop this entry: `EVAL_BATCH_SIZE` is 256, which asked 18 GB of a
                # 24 GB card for the figures and killed the reporting of the gate's nano run.
                "analysis_args": {"eval_batch_size": large_image_batch},
            }
            for name in ("imagenet", "imagenette")
        },
    }


def _cliffordnet_recipe(
    stochastic_depth_rate: float, per_variant: Dict[str, Any], large_image_batch: int
) -> Dict[str, Any]:
    return _recipe(
        epochs=200,
        batch_size=128,
        optim_args=optim_recipe(
            optimizer="adamw", lr=1e-3, weight_decay=0.1, warmup_epochs=0,
            decay_over="post_warmup", beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        ),
        watch_metric="val_accuracy",
        patience=30,
        model_args={"stochastic_depth_rate": stochastic_depth_rate},
        per_dataset=_cliffordnet_per_dataset(large_image_batch),
        per_variant=per_variant,
        erasing_prob=0.25,
    )


# ---------------------------------------------------------------------
# Model-argument rules
# ---------------------------------------------------------------------


def _weight_decay_regularizer(optim_args: OptimArgs) -> Any:
    """Return ``L2(weight_decay)`` for sgd with a decay, else ``None``.

    :param optim_args: the resolved optimizer arguments.
    :raises UnresolvedAutoError: when ``optimizer`` or ``weight_decay`` is ``"auto"``.
    """
    # DECISION plan-2026-10-30T132432-a865fa08/D-040: weight decay lives in
    # exactly one place. AdamW decays inside the optimizer, so the recipe adds NO
    # model regularizer (a convnext block sets its own gamma L2, as legacy does,
    # D-052); sgd has no decoupled decay, so the same number becomes an L2
    # regularizer on the model and optim.py gives the optimizer none. Do NOT
    # return an L2 for adamw "as well": the decay would apply twice.
    for name in ("optimizer", "weight_decay"):
        value = getattr(optim_args, name)
        if isinstance(value, str) and value == AUTO:
            raise UnresolvedAutoError(
                f"optim_args.{name} is still {AUTO!r}; resolve_auto must run "
                "before the model arguments are resolved"
            )
    if str(optim_args.optimizer).strip().lower() != "sgd":
        return None
    decay = optim_args.weight_decay
    if decay is None or decay <= 0:
        return None
    import keras

    return keras.regularizers.L2(decay)


def merge_model_args(
    context: AutoContext,
    optim_args: OptimArgs,
    model_args: Mapping[str, Any],
    computed: Mapping[str, Any],
    fixed: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Lay the recipe's data, the computed values and the caller's keys, in that order.

    :param computed: the keys the family derives from the data and the optimizer.
    :param fixed: keys of the variant's table that the family's factory sets itself and cannot
        take as an override (measured: a ViT's ``scale`` raises ``TypeError``, "multiple values
        for keyword argument"); naming one in ``model_args`` is refused here, which a task
        reaches before the run directory is claimed (``check_model_args``). Every OTHER key of a
        variant's table may be overridden, and the run records the change in ``task.json``
        (``variant_overrides``).
    """
    fixed_asked = [key for key in model_args if key in fixed]
    if fixed_asked:
        raise ConfigError(
            f"Invalid value for 'model_args': {', '.join(map(repr, fixed_asked))} is fixed by the "
            f"variant {context.variant!r} of {context.family!r} and cannot be overridden; choose "
            f"the variant instead (model={context.family}/<variant>)"
        )
    supplied = [key for key in model_args if key in _SUPPLIED]
    if supplied:
        raise ConfigError(
            f"Invalid value for 'model_args': {', '.join(map(repr, supplied))} "
            "is set by the library from model=, the data and transform_args; "
            "remove it from model_args"
        )
    explicit = model_args.get("kernel_regularizer")
    optimizer = optim_args.optimizer
    if (
        explicit is not None
        and isinstance(optimizer, str)
        and optimizer.strip().lower() == "adamw"
    ):
        raise ConfigError(
            "Invalid value for 'model_args.kernel_regularizer': the adamw "
            "optimizer already decays the weights, a regularizer would apply "
            "the decay twice; set it to null or use optim_args.optimizer=sgd"
        )
    merged: Dict[str, Any] = dict(context.defaults.get("model_args", {}))
    merged.update(computed)
    merged.update(model_args)
    return merged


def resolve_vit_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the ``create_vit`` keywords; see the module contract.

    ``patch_size`` is 4 on 32 px images and 16 otherwise (train_vit.py:451-453, by dataset there).
    ``kernel_regularizer`` is always passed, as the legacy trainer does.
    """
    patch = 4 if context.image_size == SMALL_IMAGE_SIZE else 16
    return merge_model_args(
        context,
        optim_args,
        model_args,
        {"patch_size": patch, "kernel_regularizer": _weight_decay_regularizer(optim_args)},
        fixed=("scale",),
    )


def resolve_resnet_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the ``create_resnet`` keywords; see the module contract.

    ``stem_type`` is ``"cifar"`` on 32 px images and ``"imagenet"`` otherwise.
    The CIFAR stem is new behaviour (D-003): the legacy trainer never passed it.
    """
    stem = "cifar" if context.image_size == SMALL_IMAGE_SIZE else "imagenet"
    return merge_model_args(
        context,
        optim_args,
        model_args,
        {"stem_type": stem, "kernel_regularizer": _weight_decay_regularizer(optim_args)},
    )


def _resolve_regularizer_on_request(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the recipe's model arguments plus an L2 only when sgd asks for one.

    The convnext and cliffordnet trainers never pass a regularizer, because
    both run AdamW. With sgd the weight decay has nowhere else to go, so it is
    passed rather than dropped without a word.
    """
    regularizer = _weight_decay_regularizer(optim_args)
    computed = {} if regularizer is None else {"kernel_regularizer": regularizer}
    return merge_model_args(context, optim_args, model_args, computed)


def resolve_convnext_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the ``create_convnext_v*`` keywords: the recipe's strides and rates."""
    return _resolve_regularizer_on_request(context, optim_args, model_args)


def resolve_cliffordnet_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the cliffordnet keywords: the recipe's stochastic depth."""
    return _resolve_regularizer_on_request(context, optim_args, model_args)


def serialize_model_args(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``kwargs`` with each Keras object as its serialized config.

    The only object a resolver produces is the sgd ``kernel_regularizer``; a
    caller's ``model_args`` are plain JSON values (``config_problems``).

    :param kwargs: the output of a resolver.
    :return: a new dict that ``json.dumps`` takes: a regularizer becomes
        ``{"module": "keras.regularizers", "class_name": "L2", "config": {"l2": ...}, ...}``;
        every other value is returned as it is.
    """
    import keras

    return {
        key: keras.regularizers.serialize(value)
        if isinstance(value, keras.regularizers.Regularizer)
        else value
        for key, value in kwargs.items()
    }


def build_model(
    spec: ModelSpec,
    variant: str,
    num_classes: int,
    input_shape: Tuple[int, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Call the family factory; see the module contract."""
    factory = spec.load_factory()
    try:
        return factory(
            variant=variant,
            num_classes=num_classes,
            input_shape=tuple(input_shape),
            **kwargs,
        )
    except ValueError as exc:
        # DECISION plan-2026-10-30T132432-a865fa08/D-040: only ValueError is
        # turned into a ConfigError, and only around this one call. The
        # factories raise ValueError for an unknown keyword (measured on all
        # six) and for an input they cannot take. Do NOT widen this to
        # Exception: a TypeError or a failed build inside Keras is a bug, and
        # it must arrive as a traceback.
        raise ConfigError(
            f"Model family {spec.family!r} refused variant {variant!r} with "
            f"model_args {sorted(kwargs)}: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# The registry rows
# ---------------------------------------------------------------------

_HERE = "train.api.recipes"


def builtin_models() -> Tuple[ModelSpec, ...]:
    """Return the six classification family specs, freshly built.

    The cliffordnet family is two model classes with disjoint variant tables,
    so it is two families, as the legacy trainer routes them (D-003).
    """
    convnext = f"{_MODELS}.convnext"
    clifford = f"{_MODELS}.cliffordnet"
    return (
        ModelSpec(
            family="vit",
            factory=f"{_MODELS}.vit.model:create_vit",
            variants_attr=f"{_MODELS}.vit.model:ViT.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=VIT,
            resolve_model_args=f"{_HERE}:resolve_vit_args",
        ),
        ModelSpec(
            family="resnet",
            factory=f"{_MODELS}.resnet.model:create_resnet",
            variants_attr=f"{_MODELS}.resnet.model:ResNet.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=RESNET,
            resolve_model_args=f"{_HERE}:resolve_resnet_args",
        ),
        ModelSpec(
            family="convnext_v1",
            factory=f"{convnext}.convnext_v1:create_convnext_v1",
            variants_attr=f"{convnext}.convnext_v1:ConvNeXtV1.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=_convnext_recipe(),
            resolve_model_args=f"{_HERE}:resolve_convnext_args",
        ),
        ModelSpec(
            family="convnext_v2",
            factory=f"{convnext}.convnext_v2:create_convnext_v2",
            variants_attr=f"{convnext}.convnext_v2:ConvNeXtV2.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=_convnext_recipe(),
            resolve_model_args=f"{_HERE}:resolve_convnext_args",
        ),
        ModelSpec(
            family="cliffordnet",
            factory=f"{clifford}.model:create_cliffordnet",
            variants_attr=f"{clifford}.model:CliffordNet.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=_cliffordnet_recipe(
                0.3, {"cn_64_5": {"model_args": {"stochastic_depth_rate": 0.4}}}, 8
            ),
            resolve_model_args=f"{_HERE}:resolve_cliffordnet_args",
        ),
        ModelSpec(
            family="cliffordnet_hier",
            factory=f"{clifford}.hierarchical:create_hierarchical_cliffordnet",
            variants_attr=f"{clifford}.hierarchical:HierarchicalCliffordNet.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=_cliffordnet_recipe(0.1, {}, 16),
            resolve_model_args=f"{_HERE}:resolve_cliffordnet_args",
        ),
    )


def register_builtin_models() -> None:
    """Register the built-in families that are not registered yet; see the module contract."""
    # Imported here: segmentation_models imports this module (``optim_recipe``,
    # ``merge_model_args``), so a module-scope import would be a cycle.
    from .segmentation_models import builtin_segmentation_models

    for spec in (*builtin_models(), *builtin_segmentation_models()):
        if spec.family not in registry.MODELS:
            registry.register_model(spec)
