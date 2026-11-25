"""The model families of image classification, as data, and their model-argument rules.

Each :class:`~train.api.registry.ModelSpec` below carries the defaults of one
legacy trainer (``vit``, ``resnet``, ``convnext``, ``cliffordnet``; ``beit`` and
``energy_transformer`` take their classification trainers' optimizer and the ``vit`` recipe's
epochs, batch and overlays), so
``model=vit/vit_pico`` trains the way ``train.vit.train_vit`` does unless the
caller says otherwise (D-003, D-013). A comment beside each value names the
legacy line it was read from.

Interface contract
------------------
``register_builtin_models()``
    Register the eight classification families and the rows of every other task's builder
    (:func:`train.api.segmentation_models.builtin_segmentation_models`,
    :func:`train.api.denoising_models.builtin_denoising_models`,
    :func:`train.api.pretrain_models.builtin_pretrain_models`,
    :func:`train.api.forecast_models.builtin_forecast_models`,
    :func:`train.api.depth_models.builtin_depth_models`,
    :func:`train.api.text_models.builtin_text_models` and
    :func:`train.api.tokenizer_models.builtin_tokenizer_models`) into
    :data:`train.api.registry.MODELS`. Idempotent: a family already registered is
    left alone. Imports no model code. The count of families is ``len(MODELS)``, never written here.

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
        a ``kernel_regularizer``; for the eight classifiers, also a ``pretrained``
        other than ``false`` (:func:`refuse_classifier_pretrained`). ``resolve_mim_classifier_args`` (the
        ``beit`` and ``energy_transformer`` rows, whose factories take no regularizer) also refuses an sgd
        decay, a key of the row's :data:`NOT_OFFERED` and, for ``energy_transformer``, a null global clip. Both
        ViT-style resolvers refuse a patch that does not tile the image side (:func:`refuse_patch_not_tiling`).

``vit_patch_size(context)``
    :return: the patch of a ViT-style classifier: 4 on images of a side of at most 32 px that 4 divides (CIFAR's 32,
        MNIST's 28), 16 otherwise. On the two datasets of ``train.api.pretrain_args.GEOMETRY`` it is the pretrain patch,
        so a pretrain source and its classifier agree.

``refuse_patch_not_tiling(context, kwargs)``
    :raises ConfigError: naming ``model_args.patch_size`` and ``transform_args.image_size`` when ``kwargs["patch_size"]``
        (computed, pinned by ``init`` or typed) does not divide ``context.image_size``, which the factory would refuse
        after the run directory is claimed. Nothing for a missing patch or size.

``require_global_clip(optim_args, family)``
    :raises UnresolvedAutoError: when ``optim_args.global_clipnorm`` is still ``"auto"``.
    :raises ConfigError: naming ``optim_args.global_clipnorm`` when it is null (a per-variable ``clipnorm``, which
        nulls it, is named as the cause). Called for an Energy Transformer by both of its tasks.

``recipe_notices_of(family, dataset, request=None)``
    :return: the ``(level, text)`` lines the classification task logs about the numbers of ``family``'s recipe on
        ``dataset`` that no run measured: one ``info`` line for ``beit`` and ``energy_transformer``, ``[]`` for the
        six legacy families. ``request`` is the run's unresolved config: a clause about a number the request typed
        (``batch_size``, ``epochs`` or a smoke run, ``optim_args.optimizer`` or ``lr``, ``transform_args.augment``) is
        left out, since the recipe's number is not the run's.

``observe_beit(model)`` / ``observe_energy_transformer(model)``
    The ``ModelSpec.observe`` of the two classifier rows and of their pretrain rows (``beit_mim``, ``et_mim``):
    ``{"scale": ...}`` of the model's backbone layer, ``None`` without one.

``refuse_classifier_pretrained(model_args)``
    :raises ConfigError: when ``model_args`` holds a ``pretrained`` other than ``false``: a
        classifier is built with fresh weights (a path would reach the factory's own name-matching
        loader, and ``true`` reads no official file), and an earlier run's weights come through ``init``.

``serialize_model_args(kwargs)``
    :param kwargs: the output of a resolver.
    :return: the same keywords with a regularizer as its serialized config, so
        ``config.json`` can record what the model was built with.

``build_model(spec, variant, num_classes, input_shape, kwargs)``
    :param spec: the family's :class:`~train.api.registry.ModelSpec`.
    :param variant: variant name.
    :param num_classes: output classes; ``None`` for a task whose data has none (D-005), which
        a factory of that task must accept and a classifier's never receives.
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

import copy
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .args import ERASING_PRESET, AutoContext, OptimArgs, _is_auto, is_count
from .config import AUTO
from .errors import ConfigError, UnresolvedAutoError
from . import registry
from .pretrain_args import BEIT as BEIT_MIM, DECODERS, ET as ET_MIM, TASK as PRETRAIN
from .registry import CrossTask, ModelSpec, TransferSpec

# ---------------------------------------------------------------------

TASK = "image_classification"

#: Side of the CIFAR images: the resnet stem changes here, the vit patch above it (D-047).
SMALL_IMAGE_SIZE = 32

#: Keys a recipe may hold. ``model_args`` is the data half of the model-argument
#: rule, ``loader`` the arguments of ``load_classification_data`` the recipe
#: decides (read by the data step), and the five group keys are those of the
#: task config (``analysis_args`` only in a ``per_dataset`` overlay: no family's base recipe
#: has one, D-040). ``data`` is the ``data`` group, held by the families whose recipe decides a
#: data key: the pretrain families (the mask keys depend on the family, ET masks a ratio, BEiT
#: blocks of patches, and the patch size on the dataset: ``train.api.pretrain_args.PretrainData``),
#: the tokenizer family (the patch size: ``train.api.tokenizer_args.TokenizerData``) and the
#: forecasting families (the lookback and the horizon).
RECIPE_KEYS = (
    "epochs",
    "batch_size",
    "optim_args",
    "transform_args",
    "metric_args",
    "checkpoint_args",
    "analysis_args",
    "data",
    "model_args",
    "loader",
    "per_dataset",
    "per_variant",
)

#: Keys the library passes to a factory itself.
_SUPPLIED = ("variant", "num_classes", "input_shape")

_MODELS = "dl_techniques.models.vision"

#: The two classifier rows whose trunk a pretrain row trains (``beit_mim``, ``et_mim``).
BEIT_CLASSIFIER = "beit"
ET_CLASSIFIER = "energy_transformer"


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
# wd :79 (an L2 penalty on the model, `L2(wd / 2)`, :372; the optimizer gets none),
# momentum :80, patience :87, monitor val_loss :305, schedule decay_steps = spe * epochs
# :391 with alpha 0.01. The legacy trainer had a label_smoothing field the loss never read
# (not copied, D-003; deleted from the trainer, D-042) and a global clip of 1.0 that neither
# He et al. 2015 nor torchvision's reference trainer has (deleted from both arms, D-042):
# the recipe has no clip.
# The decay reaches convolution and dense KERNELS only (BatchNorm gamma and beta and the dense bias get none), where
# torchvision's reference trainer decays every parameter: a sourced choice, not a defect (D-043 c). Goyal et al. 2017
# (arXiv 1706.02677, section 4.2) apply no decay to the BatchNorm coefficients; He et al. 2019, Bag of Tricks (arXiv
# 1812.01187, "No bias decay"), decay only the weights of convolution and fully connected layers.
# ImageNet uses the loader's crop and flip plus colour jitter :345. The legacy
# trainer cannot run on CIFAR at all; CIFAR here is new behaviour with the CIFAR
# stem (resolve_resnet_args) and no augmentation, as nothing legacy says otherwise.
RESNET = _recipe(
    epochs=100,
    batch_size=256,
    optim_args=optim_recipe(
        optimizer="sgd", lr=0.1, weight_decay=1e-4, warmup_epochs=5,
        decay_over="total", momentum=0.9,
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


# beit/train_classification.py and energy_transformer/train_classification.py TrainingConfig: lr 5e-4 :113 and :109,
# adamw :114/:110, cosine :115/:111, warmup 5 :116 and 2 :112, wd 0.05 :117/:113, a GLOBAL clip 1.0 :118/:117,
# patience 15 :121/:120, monitor val_accuracy :405/:376, dropout 0.0 :104/:100, BEiT drop_path_rate 0.1 :103, ET
# num_steps 12 :99. Their epochs (100) and batch (32), :112/:108 and :94/:90, are NOT taken: the plan gives these rows
# the vit recipe's epochs, batch and overlays (D-028), and recipe_notices_of says what that leaves unmeasured.
_LEGACY_EPOCHS, _LEGACY_BATCH = 100, 32


def _mim_classifier_recipe(warmup_epochs: int, model_args: Dict[str, Any]) -> Dict[str, Any]:
    # DECISION plan-2026-11-19T081214-5c6cc105/D-028
    # The optimizer is the legacy classification trainer's and everything else is the vit recipe's, its per-dataset
    # overlays included: cifar_light on CIFAR (the legacy flip and REFLECT pad-crop has no preset; cifar_light is that
    # plus a photometric jitter) and Imagenette's 200 epochs. Do NOT take vit's overlay optim_args (its Imagenette
    # warmup of 20): the warmup is the legacy one. Do NOT copy the numbers: a change of vit's moves these with it.
    per_dataset = {
        name: {key: value for key, value in overlay.items() if key != "optim_args"}
        for name, overlay in copy.deepcopy(VIT["per_dataset"]).items()
    }
    return _recipe(
        epochs=VIT["epochs"],
        batch_size=VIT["batch_size"],
        optim_args=optim_recipe(
            optimizer="adamw", lr=5e-4, weight_decay=0.05, warmup_epochs=warmup_epochs,
            decay_over="total", global_clipnorm=1.0,
        ),
        watch_metric="val_accuracy",
        patience=15,
        model_args={**model_args, "dropout_rate": 0.0},
        per_dataset=per_dataset,
    )


MIM_CLASSIFIER_RECIPES = {
    BEIT_CLASSIFIER: _mim_classifier_recipe(5, {"drop_path_rate": 0.1}),
    ET_CLASSIFIER: _mim_classifier_recipe(2, {"num_steps": 12}),
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


def sgd_weight_decay(optim_args: OptimArgs) -> Optional[float]:
    """Return the SGD weight decay ``wd`` the run asks for, or ``None`` when it asks for none.

    ``None`` unless ``optimizer`` is sgd and ``weight_decay`` is a positive number: adamw decays inside the optimizer,
    and sgd with ``weight_decay=null`` or ``0`` asks for no decay. ``optim_args.weight_decay`` for sgd IS the ``wd`` of
    He et al. 2015 and of ``torch.optim.SGD``, whose update adds ``wd * w`` to the gradient.

    :param optim_args: the resolved optimizer arguments.
    :raises UnresolvedAutoError: when ``optimizer`` or ``weight_decay`` is ``"auto"``.
    """
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
    return decay


def refuse_dropped_sgd_decay(optim_args: OptimArgs, task: str, family: str) -> None:
    """Refuse an sgd weight decay for a task whose adapters build no kernel regularizer.

    The classification families turn it into an L2 penalty on the kernels (:func:`_weight_decay_regularizer`); an adapter
    that builds none would drop the number without a word.

    :param task: the task's name, for the message; ``family`` the family's.
    :raises ConfigError: naming ``optim_args.weight_decay``, when :func:`sgd_weight_decay` finds a decay.
    :raises UnresolvedAutoError: as :func:`sgd_weight_decay`.
    """
    decay = sgd_weight_decay(optim_args)
    if decay is not None:
        raise ConfigError(
            f"Invalid value for 'optim_args.weight_decay': optim_args.optimizer=sgd takes its weight decay as an L2 "
            f"penalty on the model's kernels, and the {task} task builds none for {family!r}, so "
            f"weight_decay={decay!r} would be dropped without a word; set optim_args.weight_decay=null for no "
            "decay, or use optim_args.optimizer=adamw, whose decay is decoupled and applied by the optimizer"
        )


def _weight_decay_regularizer(optim_args: OptimArgs) -> Any:
    """Return ``L2(weight_decay / 2)`` for sgd with a decay, else ``None``.

    The penalty is half of ``weight_decay`` because Keras' ``L2(l)`` is ``l * sum(w^2)``, whose gradient is
    ``2 * l * w``, and the SGD weight decay ``wd`` adds ``wd * w`` to the gradient (:func:`sgd_weight_decay`).

    :param optim_args: the resolved optimizer arguments.
    :raises UnresolvedAutoError: when ``optimizer`` or ``weight_decay`` is ``"auto"``.
    """
    # DECISION plan-2026-10-30T132432-a865fa08/D-040: weight decay lives in
    # exactly one place. AdamW decays inside the optimizer, so the recipe adds NO
    # model regularizer (neither convnext block adds a gamma L2 of its own any
    # more, D-033 and D-036 of plan 98b866fb, so AdamW's decay is the only one
    # that reaches a layer scale); sgd has no decoupled decay, so the same number becomes an L2
    # regularizer on the model and optim.py gives the optimizer none. Do NOT
    # return an L2 for adamw "as well": the decay would apply twice.
    # DECISION plan-2026-11-03T221936-98b866fb/D-042: the sgd penalty is `L2(decay / 2)`, NOT
    # `L2(decay)`. Keras' L2 gradient is `2 * l * w`, the SGD weight decay's is `decay * w`, so
    # `L2(decay)` decayed twice as hard as the paper (measured in D-042, ratio 2.0000). Every
    # family that reaches this rule with sgd (resnet by recipe, the others only when asked) moves
    # with it. Do NOT go back to `L2(decay)` to keep an old run's number.
    from train.common.optimizer import sgd_weight_decay_regularizer

    return sgd_weight_decay_regularizer(sgd_weight_decay(optim_args))


def merge_model_args(
    context: AutoContext,
    optim_args: OptimArgs,
    model_args: Mapping[str, Any],
    computed: Mapping[str, Any],
    fixed: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Lay the recipe's data, the computed values, an ``init`` source's pins and the caller's keys, in that order.

    :param computed: the keys the family derives from the data and the optimizer.
    :param context: its ``pinned`` arguments (an ``init`` source's) replace the computed ones; a caller's value
        that differs from a pin is refused by name (the source's own value is accepted).
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
    changed = [key for key, value in context.pinned.items() if key in model_args and model_args[key] != value]
    if changed:
        raise ConfigError(
            f"Invalid value for 'model_args.{changed[0]}': the source was built with "
            f"{changed[0]}={context.pinned[changed[0]]!r} and init keeps the source's architecture"
        )
    merged: Dict[str, Any] = dict(context.defaults.get("model_args", {}))
    merged.update(computed)
    # DECISION plan-2026-11-18T072904-0fd54b91/D-010
    # The pins go OVER the computed values (a vit source at 32 px keeps patch 4 at 64 px, where the rule above gives
    # 16) and under the caller's keys, which may only repeat them. Do NOT lay them under the computed values or let a
    # differing caller value win: the source's weights fit only the source's architecture.
    merged.update(context.pinned)
    merged.update(model_args)
    return merged


def refuse_classifier_pretrained(model_args: Mapping[str, Any]) -> None:
    """Refuse a classifier ``model_args.pretrained`` other than ``false``; see the module contract."""
    value = model_args.get("pretrained", False)
    if value is not False:
        raise ConfigError(
            f"Invalid value for 'model_args.pretrained': only false is offered, got {value!r}; a classifier is "
            "built with fresh weights (a path is not offered, and no official weights are read); init takes a run "
            "directory"
        )


def require_global_clip(optim_args: OptimArgs, family: str) -> None:
    """Refuse a null global gradient clip for ``family``; see the module contract."""
    clip = optim_args.global_clipnorm
    if isinstance(clip, str) and clip == AUTO:
        raise UnresolvedAutoError(
            f"optim_args.global_clipnorm is still {AUTO!r}; resolve_auto must run before the model arguments "
            "are resolved"
        )
    if clip is None:
        # The energy descent is unrolled num_steps times, so the backward pass composes that many Jacobians and the
        # gradient norm spikes: the paper reports the model fails to train at a learning rate above 1e-4 without the
        # clip. A per-variable `clipnorm` is another clip and leaves the global norm unbounded, so it is refused too.
        clipnorm = optim_args.clipnorm
        cause = (
            f"optim_args.clipnorm={clipnorm!r} was set, and a per-variable clipnorm replaced the global clip "
            "(global_clipnorm resolves to null beside it) and leaves the global norm unbounded: "
            "ET needs global_clipnorm; "
            if clipnorm is not None and not (isinstance(clipnorm, str) and clipnorm == AUTO)
            else ""
        )
        raise ConfigError(
            f"Invalid value for 'optim_args.global_clipnorm': the {family!r} family needs the global gradient "
            "clip (the energy descent is unrolled, and the paper reports the model fails to train at a learning rate "
            f"above 1e-4 without it), got null; {cause}set it to a positive number or leave it "
            "auto, and use global_clipnorm, not clipnorm"
        )


def vit_patch_size(context: AutoContext) -> int:
    """Return the ViT-style patch for ``context.image_size``; see the module contract (train_vit.py:426-428)."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-047
    # 4 at a side of at most 32 px that 4 divides, 16 otherwise. Do NOT go back to 4 at exactly 32 px: MNIST's 28 px then
    # took 16 and failed in the factory after the claim, for vit, beit and energy_transformer alike. A 16 px run takes 4
    # now (a 4x4 grid, where patch 16 gave it one patch); 32 px and every side above it are unchanged.
    side = context.image_size
    return 4 if is_count(side) and side <= SMALL_IMAGE_SIZE and side % 4 == 0 else 16


def refuse_patch_not_tiling(context: AutoContext, kwargs: Mapping[str, Any]) -> None:
    """Refuse a patch that does not divide the image side, before the claim; see the module contract."""
    side, patch = context.image_size, kwargs.get("patch_size")
    if is_count(side) and is_count(patch) and side % patch:
        raise ConfigError(
            f"Invalid value for 'model_args.patch_size': {context.family!r} cuts its input into square patches of "
            f"{patch} px (model_args.patch_size), which do not tile the image side of {side} px "
            "(transform_args.image_size); ask for a patch that divides the side, or a side it divides"
        )


def resolve_vit_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the ``create_vit`` keywords; see the module contract.

    ``patch_size`` is :func:`vit_patch_size` (train_vit.py:426-428 gives 4 on CIFAR and 16 otherwise, by dataset there),
    and a patch that does not tile the side is refused. ``kernel_regularizer`` is always passed, as the legacy trainer does.
    """
    refuse_classifier_pretrained(model_args)
    kwargs = merge_model_args(
        context,
        optim_args,
        model_args,
        {"patch_size": vit_patch_size(context), "kernel_regularizer": _weight_decay_regularizer(optim_args)},
        fixed=("scale",),
    )
    refuse_patch_not_tiling(context, kwargs)
    return kwargs


def resolve_resnet_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the ``create_resnet`` keywords; see the module contract.

    ``stem_type`` is ``"cifar"`` on 32 px images and ``"imagenet"`` otherwise.
    The CIFAR stem is new behaviour (D-003): the legacy trainer never passed it.
    """
    refuse_classifier_pretrained(model_args)
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
    refuse_classifier_pretrained(model_args)
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


_NO_REGULARIZER = ("the library classifier builds no regularizer (adamw decays in the optimizer; sgd's decay is "
                   "refused), and the factory would refuse the key after the run directory is claimed")
_NAMED_BY_FACTORY = "the factory names the model itself (the call fails with a TypeError after the run directory is claimed)"
_ET_TAKES_NO = ("the Energy Transformer constructor does not take the key (it would be refused after the run directory "
                "is claimed)")

# DECISION plan-2026-11-19T081214-5c6cc105/D-045
# Keyed by family: the two constructors treat include_top and use_shared_relative_position_bias differently (measured,
# scratch_step9_1/not_offered_probe.log). Do NOT share one reason per key: energy_transformer was told "the BEiT
# constructor raises" and "beit's false builds a model with no head".
#: ``{family: {key: why}}``: the ``model_args`` keys the ``beit`` and ``energy_transformer`` factories cannot take, and
#: what each would do to that row.
NOT_OFFERED: Dict[str, Dict[str, str]] = {
    BEIT_CLASSIFIER: {
        "kernel_regularizer": _NO_REGULARIZER,
        "include_top": "a classification run trains the head (false builds a BEiT with no head, which would train on "
                       "its pooled features without a word)",
        "name": _NAMED_BY_FACTORY,
        "use_shared_relative_position_bias": "the library does not implement it (the BEiT constructor raises)",
    },
    ET_CLASSIFIER: {
        "kernel_regularizer": _NO_REGULARIZER,
        "include_top": f"a classification run trains the head, and {_ET_TAKES_NO}",
        "name": _NAMED_BY_FACTORY,
        "use_shared_relative_position_bias": f"the Energy Transformer has no relative position bias, and {_ET_TAKES_NO}",
    },
}


def resolve_mim_classifier_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the keywords of the row's library classifier factory; see the module contract."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-027
    # The rows call the library factories directly: their keywords are build_model's (measured), so an adapter would
    # only forward. Do NOT pass a kernel_regularizer, not even None (both factories raise on it), and do NOT let a
    # NOT_OFFERED key reach the factory: it would be refused after the claim, crash it, or train a headless model.
    refuse_classifier_pretrained(model_args)
    refuse_dropped_sgd_decay(optim_args, TASK, context.family)
    if context.family == ET_CLASSIFIER:
        require_global_clip(optim_args, context.family)
    refused = NOT_OFFERED[context.family]
    for key in model_args:
        if key in refused:
            raise ConfigError(
                f"Invalid value for 'model_args.{key}': it is not offered for {context.family!r}: {refused[key]}"
            )
    kwargs = merge_model_args(context, optim_args, model_args, {"patch_size": vit_patch_size(context)}, fixed=("scale",))
    refuse_patch_not_tiling(context, kwargs)
    return kwargs


def _typed_fields(request: Any) -> Set[str]:
    """The request fields a notice clause states that ``request`` typed (not ``"auto"``); a smoke run types its epochs."""
    if request is None:
        return set()
    fields = {"batch_size": request.batch_size, "epochs": request.epochs, "optim_args.optimizer":
              request.optim_args.optimizer, "optim_args.lr": request.optim_args.lr,
              "transform_args.augment": request.transform_args.augment}
    typed = {name for name, value in fields.items() if not _is_auto(value)}
    return (typed | {"epochs"}) if request.debug_args.smoke else typed


def recipe_notices_of(family: str, dataset: str, request: Any = None) -> List[Tuple[str, str]]:
    """Return the lines naming what ``family``'s recipe on ``dataset`` carries unmeasured; see the module contract."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-044
    # A clause stating a recipe number is left out when the request typed that field. Do NOT state the recipe's number
    # as the run's: with batch_size=32 and lr 1e-4 the line said "lr 0.0005 was set for a batch of 32 and is not scaled
    # to the batch of 128", both numbers false for that run. The opening clause names the recipe's optimizer AS the
    # recipe's, true whatever the request typed.
    recipe = MIM_CLASSIFIER_RECIPES.get(family)
    if recipe is None:
        return []
    typed = _typed_fields(request)
    optim, batch = recipe["optim_args"], recipe["batch_size"]
    epochs = recipe["per_dataset"].get(dataset, {}).get("epochs", recipe["epochs"])
    clauses = [
        f"{family}: the recipe's optimizer is the legacy classification trainer's (AdamW, lr {optim['lr']:g}, weight "
        f"decay {optim['weight_decay']:g}, {optim['warmup_epochs']} warmup epochs, a global gradient clip of "
        f"{optim['global_clipnorm']:g}) with the vit recipe's epochs and batch"
    ]
    if not typed & {"batch_size", "optim_args.optimizer", "optim_args.lr"}:
        clauses.append(f"lr {optim['lr']:g} was set for a batch of {_LEGACY_BATCH} and is not scaled to the batch of "
                       f"{batch}, an effect that is unmeasured")
    if "epochs" not in typed:
        clauses.append(f"{epochs} epochs are not the legacy {_LEGACY_EPOCHS}, an effect that is unmeasured")
    if dataset in ("cifar10", "cifar100"):
        if "transform_args.augment" not in typed:
            clauses.append("the augmentation is cifar_light, the legacy flip and REFLECT pad-crop with a photometric "
                           "jitter added")
    elif dataset in ("imagenet", "imagenette") and "batch_size" not in typed:
        clauses.append(f"the batch of {batch} at 224 px is not measured to fit a 24 GB card")
    if family == ET_CLASSIFIER:
        clauses.append("the global clip is load-bearing (the paper reports the model fails to train at a learning rate "
                       "above 1e-4 without it) and cannot be switched off")
    return [("info", "; ".join(clauses))]


def _scale_of(model: Any, backbone_class: str) -> Optional[Dict[str, Any]]:
    # DECISION plan-2026-11-19T081214-5c6cc105/D-029
    # The two rows read MODULE-level variant tables, so the predictor has no class to check a file against and would
    # load an ET file under a beit record. Do NOT copy the tables onto a class: read the backbone layer, as the
    # segmentation rows do (plan 2faf890a D-088).
    for layer in getattr(model, "layers", None) or []:
        if type(layer).__name__ == backbone_class:
            return {"scale": layer.get_config()["scale"]}
    return None


def observe_beit(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant key the BEiT backbone shows, ``None`` for a model without one; see the module contract."""
    return _scale_of(model, "BeitModel")


def observe_energy_transformer(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant key the ET backbone shows, ``None`` for a model without one; see the module contract."""
    return _scale_of(model, "EnergyTransformerBackbone")


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
    num_classes: Optional[int],
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

#: What ``init`` moves between two classifiers of one family (paths relative to the model). The head is
#: ``classification_head``; the 3-channel stem kernel makes ``channels`` fixed. ``pinned`` keeps the source's
#: architecture where a resolver below derives it from the image size or the dataset (vit ``patch_size``, the
#: resnet ``stem_type``, the convnext ``strides``); ``grow`` names the blocks a deeper variant adds.
_CLASSIFIER_HEAD = ("classification_head",)
VIT_TRANSFER = TransferSpec(
    head=_CLASSIFIER_HEAD, resample={"pos_embed/pos_embedding": "vit_pos_bicubic"}, grow=r"transformer_layer_\d+",
    pinned=("patch_size",), fixed=("channels",),
)
RESNET_TRANSFER = TransferSpec(
    head=_CLASSIFIER_HEAD, grow=r"stage\d+_block\d+", pinned=("stem_type",), fixed=("channels",),
)
CONVNEXT_TRANSFER = TransferSpec(head=_CLASSIFIER_HEAD, pinned=("strides",), fixed=("channels",))
CLIFFORDNET_TRANSFER = TransferSpec(head=_CLASSIFIER_HEAD, fixed=("channels",))
# DECISION plan-2026-11-19T081214-5c6cc105/D-031
# BEiT's head is either name, by model_args.use_mean_pooling; a pooling mismatch is refused at the weight level
# (final_norm has no source). use_absolute_position_embeddings reaches the factory (measured), so its table is declared
# under the vit rule (D-007); use_shared_relative_position_bias is refused (NOT_OFFERED). The pretrain sources drop
# their decoders. Do NOT pin use_mean_pooling: a same-pooling source of either kind is legitimate.
BEIT_TRANSFER = TransferSpec(
    head=("mean_pooling_classification_head", "classification_head"),
    resample={
        r"beit_backbone/encoder_layer_\d+/attention/relative_position_bias_table": "beit_relpos_cubic",
        "beit_backbone/pos_embed/pos_embedding": "vit_pos_bicubic",
    },
    pinned=("patch_size",), fixed=("channels",),
    sources={(PRETRAIN, BEIT_MIM): CrossTask(drop=DECODERS[BEIT_MIM])},
)
ET_TRANSFER = TransferSpec(
    head=("head_norm", "head_dense"), resample={"et_backbone/pos_embed/pos_embedding": "et_pos_refused"},
    pinned=("patch_size",), fixed=("channels",),
    sources={(PRETRAIN, ET_MIM): CrossTask(drop=DECODERS[ET_MIM])},
)


def builtin_models() -> Tuple[ModelSpec, ...]:
    """Return the eight classification family specs, freshly built.

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
            transfer=VIT_TRANSFER,
        ),
        ModelSpec(
            family="resnet",
            factory=f"{_MODELS}.resnet.model:create_resnet",
            variants_attr=f"{_MODELS}.resnet.model:ResNet.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=RESNET,
            resolve_model_args=f"{_HERE}:resolve_resnet_args",
            transfer=RESNET_TRANSFER,
        ),
        ModelSpec(
            family="convnext_v1",
            factory=f"{convnext}.convnext_v1:create_convnext_v1",
            variants_attr=f"{convnext}.convnext_v1:ConvNeXtV1.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=_convnext_recipe(),
            resolve_model_args=f"{_HERE}:resolve_convnext_args",
            transfer=CONVNEXT_TRANSFER,
        ),
        ModelSpec(
            family="convnext_v2",
            factory=f"{convnext}.convnext_v2:create_convnext_v2",
            variants_attr=f"{convnext}.convnext_v2:ConvNeXtV2.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=_convnext_recipe(),
            resolve_model_args=f"{_HERE}:resolve_convnext_args",
            transfer=CONVNEXT_TRANSFER,
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
            transfer=CLIFFORDNET_TRANSFER,
        ),
        ModelSpec(
            family="cliffordnet_hier",
            factory=f"{clifford}.hierarchical:create_hierarchical_cliffordnet",
            variants_attr=f"{clifford}.hierarchical:HierarchicalCliffordNet.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=_cliffordnet_recipe(0.1, {}, 16),
            resolve_model_args=f"{_HERE}:resolve_cliffordnet_args",
            transfer=CLIFFORDNET_TRANSFER,
        ),
        ModelSpec(
            family=BEIT_CLASSIFIER,
            factory=f"{_MODELS}.beit.model:create_beit_classifier",
            variants_attr=f"{_MODELS}.beit.model:MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=MIM_CLASSIFIER_RECIPES[BEIT_CLASSIFIER],
            resolve_model_args=f"{_HERE}:resolve_mim_classifier_args",
            observe=f"{_HERE}:observe_beit",
            transfer=BEIT_TRANSFER,
        ),
        ModelSpec(
            family=ET_CLASSIFIER,
            factory=f"{_MODELS}.energy_transformer.model:create_energy_transformer_classifier",
            variants_attr=f"{_MODELS}.energy_transformer.model:MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=MIM_CLASSIFIER_RECIPES[ET_CLASSIFIER],
            resolve_model_args=f"{_HERE}:resolve_mim_classifier_args",
            observe=f"{_HERE}:observe_energy_transformer",
            transfer=ET_TRANSFER,
        ),
    )


def register_builtin_models() -> None:
    """Register the built-in families that are not registered yet; see the module contract."""
    # Imported here: segmentation_models and denoising_models import this module (``optim_recipe``,
    # ``merge_model_args``), so a module-scope import would be a cycle.
    from .denoising_models import builtin_denoising_models
    from .depth_models import builtin_depth_models
    from .forecast_models import builtin_forecast_models
    from .pretrain_models import builtin_pretrain_models
    from .segmentation_models import builtin_segmentation_models
    from .text_models import builtin_text_models
    from .tokenizer_models import builtin_tokenizer_models

    for spec in (
        *builtin_models(), *builtin_segmentation_models(), *builtin_denoising_models(), *builtin_pretrain_models(),
        *builtin_forecast_models(), *builtin_depth_models(), *builtin_text_models(), *builtin_tokenizer_models(),
    ):
        if spec.family not in registry.MODELS:
            registry.register_model(spec)
