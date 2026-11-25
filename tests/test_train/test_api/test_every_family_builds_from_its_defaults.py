"""The eight classification families, their recipes, and the rules that turn a recipe into a model.

Four things are pinned:

* the listing: every family's variants, by literal name, so a model added or
  dropped upstream changes this file on purpose;
* the recipes: a literal table of the defaults of each legacy trainer, read
  through ``resolve_auto`` so the resolved values are what is compared, and a
  literal table of the per-dataset and per-variant overlays;
* the weight-decay rule: AdamW never gets a model regularizer, sgd's decay
  becomes an ``L2`` on the model and the optimizer gets none;
* the build: each family makes a model from its defaults at 32x32x3 with ten
  classes, and one forward call returns ten logits.

The literals were read from the legacy trainers (the comments in
``train/api/recipes.py`` cite the lines), not from the recipes under test. The ``beit`` and
``energy_transformer`` rows (plan 5c6cc105 Step 5) take the legacy classification trainers' optimizer
and the ``vit`` recipe's epochs, batch and per-dataset overlays. The three ViT-style rows share one patch rule, and a
patch that does not tile the side is refused before the claim naming both (D-047).
"""

import dataclasses
import functools
import subprocess
import sys

import numpy as np
import pytest

from train.api import cli, recipes, registry
from train.api.args import (
    AutoContext,
    CheckpointArgs,
    MetricArgs,
    OptimArgs,
    TransformArgs,
)
from train.api.config import AUTO, assert_resolved
from train.api.errors import (
    ConfigError,
    ConfigValidationError,
    UnresolvedAutoError,
)
from train.api.optim import build_optimizer_and_schedule

from ._groups import task_class

# ---------------------------------------------------------------------
# The listing
# ---------------------------------------------------------------------

#: Every variant of every CLASSIFICATION family, by name. 6 + 5 + 6 + 9 + 9 + 1 + 4 + 3 = 43. The rest of this
#: file (recipes, overlays, builds at 32x32 with ten classes) is about these eight; the segmentation
#: families are listed in :data:`SEGMENTATION_VARIANTS` and have their own files
#: (``test_image_segmentation_task.py``, ``test_backbone_segmentation_families.py``).
VARIANTS = {
    "vit": ["vit_pico", "vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge"],
    "resnet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    "convnext_v1": ["cifar10", "tiny", "small", "base", "large", "xlarge"],
    "convnext_v2": [
        "cifar10", "atto", "femto", "pico", "nano", "tiny", "base", "large", "huge",
    ],
    "cliffordnet": [
        "nano", "lite", "lite_g", "nano_g", "cn_12_3", "cn_18_5", "cn_32_3",
        "cn_32_5", "cn_64_5",
    ],
    "cliffordnet_hier": ["hier_345"],
    "beit": ["beit_tiny", "beit_small", "beit_base", "beit_large"],
    "energy_transformer": ["et_tiny", "et_small", "et_base"],
}
#: The two classifiers whose trunk a pretrain row trains (plan 5c6cc105 Step 5); their factories take no regularizer.
MIM_CLASSIFIERS = ("beit", "energy_transformer")
TASK = "image_classification"
SEGMENTATION = "image_segmentation"

#: The segmentation families (step 28, D-077: ``register_builtin_models`` registers them, so the
#: registry-wide listings below hold 36 + 6 + 11 = 53 names). ``convunext`` follows ``CONVUNEXT_CONFIGS``;
#: ``resnet_seg`` and ``convnext_seg`` (plan 98b866fb step 2, D-012) read the classifiers' own tables, so they list
#: the same variants as ``resnet`` and ``convnext_v1`` above.
SEGMENTATION_VARIANTS = {
    "convunext": ["base", "large", "small", "tiny", "xlarge"],
    "accunet": ["base"],
    "resnet_seg": VARIANTS["resnet"],
    "convnext_seg": VARIANTS["convnext_v1"],
}

#: The denoising families (plan-2026-11-10T095307-d483ffab step 7: ``recipes.register_builtin_models`` registers
#: them, so the registry-wide listings below held 36 + 17 + 23 = 76 names until the pretrain families, below). Four read the library's five-name tables
#: (``BFCNN_CONFIGS``, ``BFUNET_CONFIGS``, ``CONVUNEXT_CONFIGS`` twice); ``scunet`` has the three of its own
#: ``SCUNET_VARIANTS``. Their builds and refusals are in ``test_denoising_families.py``.
DENOISING_VARIANTS = {
    "bfcnn": ["base", "large", "small", "tiny", "xlarge"],
    "bfunet": ["base", "large", "small", "tiny", "xlarge"],
    "bfconvunext": ["base", "large", "small", "tiny", "xlarge"],
    "convunext_biased": ["base", "large", "small", "tiny", "xlarge"],
    "scunet": ["base", "small", "tiny"],
}
DENOISING = "image_denoising"

#: The pretrain families (plan-2026-11-12T015747-46e5318f step 10: ``recipes.register_builtin_models`` registers them,
#: so the registry-wide listings below hold the 76 names above and these 7). Both read the model modules'
#: own ``MODEL_VARIANTS`` (``dl_techniques.models.vision.energy_transformer.model`` and ``...beit.model``). Their builds,
#: recipes and refusals are in ``test_pretrain_families.py``.
PRETRAIN_VARIANTS = {
    "et_mim": ["et_tiny", "et_small", "et_base"],
    "beit_mim": ["beit_tiny", "beit_small", "beit_base", "beit_large"],
}
PRETRAIN = "pretrain"

#: The forecasting families (``recipes.register_builtin_models`` registers them,
#: so the registry-wide listings below hold the 83 names above and these). nbeats and mdn read the variants tables written in
#: ``train.api.forecast_models``, ets the library's tuple of names, the others the models' own ``MODEL_VARIANTS``. Their builds,
#: recipes and refusals are in ``test_forecast_families.py``.
FORECAST_VARIANTS = {
    "nbeats": ["base", "small", "tiny"],
    "prism": ["base", "large", "small", "tiny"],
    "tirex": ["large", "medium", "small", "tiny"],
    "xlstm": ["small", "tiny"],
    "ets": ["AAA", "AAN", "ANN"],
    "timestral": ["123M", "209M", "28M", "5M"],
    "mdn": ["base", "small", "tiny"],
}
FORECAST = "time_series_forecasting"

#: The one depth family; it reads the model's own ``DepthAnything.MODEL_VARIANTS``. Its build, recipe and refusals are
#: in ``test_depth_task.py``.
DEPTH_VARIANTS = {"depth_anything": ["vit_b", "vit_l", "vit_s"]}
DEPTH = "depth_estimation"

#: The one language-model family (plan-2026-11-24T092348-7806a290 step 4); it reads the library GPT2's own ``MODEL_VARIANTS``.
#: Its adapter, recipe and refusals are in ``test_language_model_pretraining_families.py``.
LANGUAGE_VARIANTS = {"gpt2_lm": ["large", "medium", "small", "tiny", "xl"]}
LANGUAGE = "language_model_pretraining"

#: The two text-classification families (plan-2026-11-24T092348-7806a290 step 10); each reads its library model's own
#: ``MODEL_VARIANTS`` (``BERT`` and ``GPT2``). Their adapters, recipes and refusals are in ``test_text_classification_families.py``.
TEXT_CLASSIFICATION_VARIANTS = {"bert": ["base", "large", "small", "tiny"], "gpt2": ["large", "medium", "small", "tiny", "xl"]}
TEXT_CLASSIFICATION = "text_classification"

#: The one tokenizer family (plan-2026-11-25T124419-fa94f59c step 6); its one variant is the API's own table (the library has no named scales).
#: Its adapter, recipe and refusals are in ``test_beit_tokenizer_families.py``.
TOKENIZER_VARIANTS = {"vq_vae_rotation": ["small"]}
TOKENIZER = "beit_tokenizer"
ALL_VARIANTS = {
    **VARIANTS, **SEGMENTATION_VARIANTS, **DENOISING_VARIANTS, **PRETRAIN_VARIANTS, **FORECAST_VARIANTS,
    **DEPTH_VARIANTS, **LANGUAGE_VARIANTS, **TEXT_CLASSIFICATION_VARIANTS, **TOKENIZER_VARIANTS,
}


@pytest.fixture
def builtin(monkeypatch):
    """An empty registry holding only the six built-in families."""
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "TASKS", {})
    recipes.register_builtin_models()
    return registry.MODELS


def context(family, variant, dataset, image_size=32, num_classes=10):
    spec = registry.MODELS[family]
    return AutoContext(
        family=family,
        variant=variant,
        dataset=dataset,
        num_classes=num_classes,
        image_size=image_size,
        defaults=spec.recipe(dataset, variant),
    )


def test_the_listing_is_the_literal_variant_tables(builtin):
    assert sorted(builtin) == sorted(ALL_VARIANTS)
    expected = sorted(f"{f}/{v}" for f, names in ALL_VARIANTS.items() for v in names)
    # 76 before the pretrain families (3 + 4), the forecasting families (23), depth (3); 109 + 4 + 3 classifier
    # variants of beit and energy_transformer (plan 5c6cc105 Step 5), the 5 variants of gpt2_lm, the 4 + 5 of the two
    # text classifiers, and the one of the tokenizer.
    assert len(expected) == 131
    assert registry.list_models() == expected
    for family, spec in builtin.items():
        if family in SEGMENTATION_VARIANTS:
            assert spec.tasks == (SEGMENTATION,), family
        elif family in DENOISING_VARIANTS:
            assert spec.tasks == (DENOISING,), family
        elif family in PRETRAIN_VARIANTS:
            assert spec.tasks == (PRETRAIN,), family
        elif family in FORECAST_VARIANTS:
            assert spec.tasks == (FORECAST,), family
        elif family in DEPTH_VARIANTS:
            assert spec.tasks == (DEPTH,), family
        elif family in LANGUAGE_VARIANTS:
            assert spec.tasks == (LANGUAGE,), family
        elif family in TEXT_CLASSIFICATION_VARIANTS:
            assert spec.tasks == (TEXT_CLASSIFICATION,), family
        elif family in TOKENIZER_VARIANTS:
            assert spec.tasks == (TOKENIZER,), family
        else:
            assert spec.tasks == (TASK,), family


def test_the_task_filter_keeps_every_family_of_the_task(builtin):
    registry.register_task(
        registry.TaskSpec(TASK, "x.y:Config", "x.y:run", "image classification")
    )
    registry.register_task(
        registry.TaskSpec(SEGMENTATION, "x.y:Config", "x.y:run", "image segmentation")
    )
    registry.register_task(
        registry.TaskSpec(DENOISING, "x.y:Config", "x.y:run", "image denoising")
    )
    registry.register_task(registry.TaskSpec(PRETRAIN, "x.y:Config", "x.y:run", "pretrain"))
    registry.register_task(registry.TaskSpec(FORECAST, "x.y:Config", "x.y:run", "time series forecasting"))
    registry.register_task(registry.TaskSpec(DEPTH, "x.y:Config", "x.y:run", "depth estimation"))
    registry.register_task(registry.TaskSpec(LANGUAGE, "x.y:Config", "x.y:run", "language model pretraining"))
    registry.register_task(registry.TaskSpec(TEXT_CLASSIFICATION, "x.y:Config", "x.y:run", "text classification"))
    registry.register_task(registry.TaskSpec(TOKENIZER, "x.y:Config", "x.y:run", "tokenizer"))
    classifiers = sorted(f"{f}/{v}" for f, names in VARIANTS.items() for v in names)
    segmenters = sorted(f"{f}/{v}" for f, names in SEGMENTATION_VARIANTS.items() for v in names)
    denoisers = sorted(f"{f}/{v}" for f, names in DENOISING_VARIANTS.items() for v in names)
    pretrainers = sorted(f"{f}/{v}" for f, names in PRETRAIN_VARIANTS.items() for v in names)
    forecasters = sorted(f"{f}/{v}" for f, names in FORECAST_VARIANTS.items() for v in names)
    depth = sorted(f"{f}/{v}" for f, names in DEPTH_VARIANTS.items() for v in names)
    language = sorted(f"{f}/{v}" for f, names in LANGUAGE_VARIANTS.items() for v in names)
    text = sorted(f"{f}/{v}" for f, names in TEXT_CLASSIFICATION_VARIANTS.items() for v in names)
    tokenizers = sorted(f"{f}/{v}" for f, names in TOKENIZER_VARIANTS.items() for v in names)
    assert registry.list_models(TASK) == classifiers
    assert registry.list_models(SEGMENTATION) == segmenters
    assert registry.list_models(DENOISING) == denoisers
    assert registry.list_models(PRETRAIN) == pretrainers
    assert registry.list_models(FORECAST) == forecasters
    assert registry.list_models(DEPTH) == depth
    assert registry.list_models(LANGUAGE) == language
    assert registry.list_models(TEXT_CLASSIFICATION) == text
    assert registry.list_models(TOKENIZER) == tokenizers
    assert sorted(
        classifiers + segmenters + denoisers + pretrainers + forecasters + depth + language + text + tokenizers
    ) == registry.list_models()
    # Control: a family that lists another task drops out of the filter.
    other = dataclasses.replace(builtin["vit"], family="other_vit", tasks=("else",))
    registry.register_model(other)
    assert not [n for n in registry.list_models(TASK) if n.startswith("other_vit/")]
    assert [n for n in registry.list_models() if n.startswith("other_vit/")]


def test_registering_twice_changes_nothing(builtin):
    before = dict(builtin)
    recipes.register_builtin_models()
    assert registry.MODELS == before


def test_the_command_line_lists_them_in_a_fresh_process(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "train", "list_models"],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    expected = sorted(f"{f}/{v}" for f, names in ALL_VARIANTS.items() for v in names)
    assert result.stdout.split() == expected


def test_an_unknown_variant_is_refused_with_a_suggestion(builtin):
    with pytest.raises(registry.UnknownModelError, match="Did you mean 'vit_pico'"):
        registry.parse_model_name("vit/vit_pic")


# ---------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------

#: The legacy defaults, one row per family. ``None`` is "does not apply".
DEFAULTS = {
    "vit": dict(
        epochs=50, batch_size=128, optimizer="adamw", lr=3e-4, weight_decay=0.05,
        momentum=None, betas=(None, None, None), warmup_epochs=5,
        min_lr_ratio=0.01, decay_over="total", clipnorm=None, global_clipnorm=1.0,
        watch_metric="val_accuracy", watch_mode="max", patience=15,
    ),
    # No clip: the legacy trainer's global clip of 1.0 was unsourced (He et al. 2015 and torchvision's reference trainer have none)
    # and is deleted from both arms (D-042 of plan 98b866fb).
    "resnet": dict(
        epochs=100, batch_size=256, optimizer="sgd", lr=0.1, weight_decay=1e-4,
        momentum=0.9, betas=(None, None, None), warmup_epochs=5,
        min_lr_ratio=0.01, decay_over="total", clipnorm=None, global_clipnorm=None,
        watch_metric="val_loss", watch_mode="min", patience=15,
    ),
    "convnext_v1": dict(
        epochs=100, batch_size=64, optimizer="adamw", lr=1e-3, weight_decay=1e-4,
        momentum=None, betas=(None, None, None), warmup_epochs=0,
        min_lr_ratio=0.01, decay_over="post_warmup", clipnorm=1.0, global_clipnorm=None,
        watch_metric="val_loss", watch_mode="min", patience=50,
    ),
    "cliffordnet": dict(
        epochs=200, batch_size=128, optimizer="adamw", lr=1e-3, weight_decay=0.1,
        momentum=None, betas=(0.9, 0.999, 1e-8), warmup_epochs=0,
        min_lr_ratio=0.01, decay_over="post_warmup", clipnorm=None, global_clipnorm=None,
        watch_metric="val_accuracy", watch_mode="max", patience=30,
    ),
}
# beit and energy_transformer: train/beit/train_classification.py and train/energy_transformer/train_classification.py
# TrainingConfig (lr 5e-4, AdamW, cosine, wd 0.05, global clip 1.0, patience 15; warmup 5 and 2) at the vit recipe's
# epochs and batch (the legacy 100 epochs at batch 32 are not taken, and the recipe notice says so).
DEFAULTS["beit"] = dict(DEFAULTS["vit"], lr=5e-4)
DEFAULTS["energy_transformer"] = dict(DEFAULTS["beit"], warmup_epochs=2)
DEFAULTS["convnext_v2"] = DEFAULTS["convnext_v1"]
DEFAULTS["cliffordnet_hier"] = DEFAULTS["cliffordnet"]

FIRST_VARIANT = {family: names[0] for family, names in VARIANTS.items()}
FIRST_VARIANT["convnext_v2"] = "atto"
FIRST_VARIANT["cliffordnet"] = "nano"


@pytest.mark.parametrize("family", sorted(DEFAULTS))
def test_the_defaults_are_the_literal_legacy_table(family, builtin):
    row = DEFAULTS[family]
    ctx = context(family, FIRST_VARIANT[family], "cifar10")
    optim = OptimArgs().resolve_auto(ctx)
    metric = MetricArgs().resolve_auto(ctx)
    checkpoint = CheckpointArgs().resolve_auto(ctx)
    assert ctx.defaults["epochs"] == row["epochs"]
    assert ctx.defaults["batch_size"] == row["batch_size"]
    for name in (
        "optimizer", "lr", "weight_decay", "momentum", "warmup_epochs",
        "min_lr_ratio", "decay_over", "clipnorm", "global_clipnorm",
    ):
        assert getattr(optim, name) == row[name], name
    assert (optim.beta_1, optim.beta_2, optim.epsilon) == row["betas"]
    assert optim.schedule == "cosine_decay"
    assert optim.warmup_start_lr == 1e-8
    assert metric.watch_metric == row["watch_metric"]
    assert metric.watch_mode == row["watch_mode"]
    assert checkpoint.early_stopping_patience == row["patience"]


def auto_fields(cls):
    return [f.name for f in dataclasses.fields(cls) if f.default == AUTO]


#: AUTO fields a recipe does not list: the data or the metric's name decides.
DERIVED = {
    "TransformArgs": {"image_size"},
    "MetricArgs": {"watch_mode", "success_threshold"},
}


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_a_recipe_lists_every_auto_key_literally(family, builtin):
    recipe = builtin[family].defaults
    assert set(recipe) <= set(recipes.RECIPE_KEYS)
    for key, cls in (
        ("optim_args", OptimArgs),
        ("transform_args", TransformArgs),
        ("metric_args", MetricArgs),
        ("checkpoint_args", CheckpointArgs),
    ):
        wanted = set(auto_fields(cls)) - DERIVED.get(cls.__name__, set())
        assert set(recipe[key]) == wanted, key
    # A control on the helper: it finds the fields the groups really have.
    assert len(auto_fields(OptimArgs)) == 14


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_nothing_is_left_auto_after_every_group_resolves(family, builtin):
    ctx = context(family, FIRST_VARIANT[family], "cifar10")
    task = task_class()()
    groups = {}
    for field in dataclasses.fields(task):
        group = getattr(task, field.name)
        if hasattr(group, "resolve_auto"):
            groups[field.name] = group.resolve_auto(ctx)
    assert {"optim_args", "transform_args", "metric_args", "checkpoint_args"} <= set(groups)
    assert_resolved(dataclasses.replace(task, **groups))
    # Control: the unresolved task is refused.
    with pytest.raises(Exception, match="auto"):
        assert_resolved(task)


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_a_recipe_key_that_is_no_field_is_an_error(family, builtin):
    ctx = context(family, FIRST_VARIANT[family], "cifar10")
    for group_key, cls, typo in (
        ("optim_args", OptimArgs, "clipnrom"),
        ("metric_args", MetricArgs, "watch_metrik"),
        ("checkpoint_args", CheckpointArgs, "patience"),
        ("transform_args", TransformArgs, "augmnet"),
    ):
        broken = {**ctx.defaults, group_key: {**ctx.defaults[group_key], typo: 1}}
        bad = dataclasses.replace(ctx, defaults=broken)
        with pytest.raises(ConfigValidationError, match=typo):
            cls().resolve_auto(bad)
    # Control: the unbroken recipe resolves.
    OptimArgs().resolve_auto(ctx)


def test_the_overlay_merges_and_does_not_replace(builtin):
    base = builtin["vit"].recipe("imagenet", "vit_pico")
    cifar = builtin["vit"].recipe("cifar10", "vit_pico")
    assert cifar["transform_args"] == {"augment": "cifar_light", "random_erasing_prob": 0.0}
    assert base["transform_args"] == {"augment": "none", "random_erasing_prob": 0.0}
    assert cifar["optim_args"] == base["optim_args"]
    assert "per_dataset" not in cifar and "per_variant" not in cifar


def test_a_recipe_is_a_copy(builtin):
    spec = builtin["vit"]
    taken = spec.recipe("cifar10", "vit_pico")
    taken["optim_args"]["lr"] = 99.0
    taken["model_args"]["dropout_rate"] = 0.9
    fresh = spec.recipe("cifar10", "vit_pico")
    assert fresh["optim_args"]["lr"] == 3e-4
    assert fresh["model_args"]["dropout_rate"] == 0.1
    assert spec.defaults["optim_args"]["lr"] == 3e-4


def test_every_overlay_key_is_a_recipe_key(builtin):
    overlays = set(registry.OVERLAY_KEYS)
    for spec in builtin.values():
        for table in registry.OVERLAY_KEYS:
            for name, overlay in spec.defaults[table].items():
                assert set(overlay) <= set(recipes.RECIPE_KEYS) - overlays, (
                    spec.family, table, name,
                )


# (family, variant, dataset, image_size, augment, erasing, model_args, loader)
OVERLAYS = [
    ("vit", "vit_pico", "cifar10", 32, "cifar_light", 0.0,
     {"dropout_rate": 0.1, "attention_dropout_rate": 0.1, "patch_size": 4}, {}),
    ("vit", "vit_pico", "cifar100", 32, "cifar_light", 0.0,
     {"dropout_rate": 0.1, "attention_dropout_rate": 0.1, "patch_size": 4}, {}),
    ("vit", "vit_base", "imagenet", 224, "none", 0.0,
     {"dropout_rate": 0.1, "attention_dropout_rate": 0.1, "patch_size": 16}, {}),
    ("resnet", "resnet18", "cifar10", 32, "none", 0.0, {"stem_type": "cifar"}, {}),
    ("resnet", "resnet50", "imagenet", 224, "none", 0.0, {"stem_type": "imagenet"},
     {"imagenet_augment_color": True}),
    ("convnext_v1", "cifar10", "mnist", 28, "none", 0.0,
     {"strides": 2, "drop_path_rate": 0.1, "dropout_rate": 0.1}, {"channels": 3}),
    ("convnext_v1", "cifar10", "cifar10", 32, "flip_pad_crop", 0.0,
     {"strides": 2, "drop_path_rate": 0.1, "dropout_rate": 0.1}, {}),
    ("convnext_v1", "cifar10", "cifar100", 32, "flip_pad_crop", 0.0,
     {"strides": 2, "drop_path_rate": 0.2, "dropout_rate": 0.2}, {}),
    ("convnext_v1", "tiny", "imagenet", 224, "none", 0.0,
     {"strides": 4, "drop_path_rate": 0.1, "dropout_rate": 0.0}, {}),
    ("convnext_v2", "cifar10", "cifar100", 32, "flip_pad_crop", 0.0,
     {"strides": 2, "drop_path_rate": 0.2, "dropout_rate": 0.2}, {}),
    ("convnext_v2", "tiny", "imagenet", 224, "none", 0.0,
     {"strides": 4, "drop_path_rate": 0.1, "dropout_rate": 0.0}, {}),
    ("cliffordnet", "nano", "cifar10", 32, "cifar_autoaugment", 0.25,
     {"stochastic_depth_rate": 0.3}, {}),
    ("cliffordnet", "cn_64_5", "cifar100", 32, "cifar_autoaugment", 0.25,
     {"stochastic_depth_rate": 0.4}, {}),
    ("cliffordnet", "lite", "imagenette", 224, "none", 0.0,
     {"stochastic_depth_rate": 0.3}, {}),
    ("cliffordnet", "cn_64_5", "imagenet", 224, "none", 0.0,
     {"stochastic_depth_rate": 0.4}, {}),
    ("cliffordnet_hier", "hier_345", "cifar100", 32, "cifar_autoaugment", 0.25,
     {"stochastic_depth_rate": 0.1}, {}),
    ("cliffordnet_hier", "hier_345", "imagenet", 224, "none", 0.0,
     {"stochastic_depth_rate": 0.1}, {}),
    ("beit", "beit_tiny", "cifar10", 32, "cifar_light", 0.0,
     {"drop_path_rate": 0.1, "dropout_rate": 0.0, "patch_size": 4}, {}),
    ("beit", "beit_base", "imagenette", 224, "none", 0.0,
     {"drop_path_rate": 0.1, "dropout_rate": 0.0, "patch_size": 16}, {}),
    ("energy_transformer", "et_tiny", "cifar100", 32, "cifar_light", 0.0,
     {"num_steps": 12, "dropout_rate": 0.0, "patch_size": 4}, {}),
    ("energy_transformer", "et_small", "imagenet", 224, "none", 0.0,
     {"num_steps": 12, "dropout_rate": 0.0, "patch_size": 16}, {}),
]


@pytest.mark.parametrize(
    "family, variant, dataset, image_size, augment, erasing, model_args, loader",
    OVERLAYS,
    ids=[f"{o[0]}-{o[1]}-{o[2]}" for o in OVERLAYS],
)
def test_the_per_dataset_and_per_variant_rules(
    family, variant, dataset, image_size, augment, erasing, model_args, loader, builtin
):
    spec = builtin[family]
    ctx = context(family, variant, dataset, image_size)
    transform = TransformArgs().resolve_auto(ctx)
    assert (transform.augment, transform.random_erasing_prob) == (augment, erasing)
    assert ctx.defaults.get("loader", {}) == loader
    optim = OptimArgs().resolve_auto(ctx)
    resolved = spec.load_resolver()(ctx, optim, {})
    resolved.pop("kernel_regularizer", None)
    assert resolved == model_args


@pytest.mark.parametrize(
    "family, variant, dataset, image_size, explicit, erasing",
    [
        # train_cliffordnet.py:684-685: the preset carries 0.25 on ANY dataset.
        ("cliffordnet", "nano", "imagenet", 224, "cifar_autoaugment", 0.25),
        ("cliffordnet", "lite", "imagenette", 224, "cifar_autoaugment", 0.25),
        ("cliffordnet_hier", "hier_345", "imagenet", 224, "cifar_autoaugment", 0.25),
        # Any other preset on cifar: no erasing, as the legacy default for it.
        ("cliffordnet", "nano", "cifar10", 32, "flip", 0.0),
        ("cliffordnet", "nano", "cifar10", 32, "none", 0.0),
        # Control: a family with no erasing recipe gives none under the preset.
        ("vit", "vit_pico", "imagenet", 224, "cifar_autoaugment", 0.0),
        ("convnext_v1", "tiny", "imagenet", 224, "cifar_autoaugment", 0.0),
    ],
)
def test_an_explicit_autoaugment_gets_the_erasing_probability_the_legacy_trainer_gave(
    family, variant, dataset, image_size, explicit, erasing, builtin
):
    ctx = context(family, variant, dataset, image_size)
    transform = TransformArgs(augment=explicit).resolve_auto(ctx)
    assert (transform.augment, transform.random_erasing_prob) == (explicit, erasing)


# ---------------------------------------------------------------------
# The weight-decay rule and the model-argument rules
# ---------------------------------------------------------------------


def l2_strength(regularizer):
    """The ``l`` of a Keras ``L2(l)``. The recipes give the model ``wd / 2`` for an sgd weight decay ``wd``: ``L2(l)`` has the
    gradient ``2 * l * w`` and the SGD weight decay ``wd * w`` (D-042 of plan 98b866fb), so every number below is half the decay."""
    import keras

    assert isinstance(regularizer, keras.regularizers.L2)
    return float(regularizer.l2)


def resolved(family, variant, dataset="cifar10", image_size=32, user=None, **optim):
    ctx = context(family, variant, dataset, image_size)
    args = OptimArgs(**optim).resolve_auto(ctx)
    spec = registry.MODELS[family]
    return spec.load_resolver()(ctx, args, user or {}), args


def test_adamw_gets_no_model_regularizer(builtin):
    for family in ("vit", "convnext_v1", "cliffordnet", *MIM_CLASSIFIERS):
        kwargs, optim = resolved(family, FIRST_VARIANT[family])
        assert optim.optimizer == "adamw" and optim.weight_decay > 0
        assert kwargs.get("kernel_regularizer") is None
    kwargs, _ = resolved("vit", "vit_pico")
    assert "kernel_regularizer" in kwargs and kwargs["kernel_regularizer"] is None


def test_resnet_with_sgd_gets_an_l2_and_its_optimizer_gets_no_decay(builtin):
    kwargs, optim = resolved("resnet", "resnet18")
    assert optim.optimizer == "sgd"
    assert optim.weight_decay == 1e-4
    assert l2_strength(kwargs["kernel_regularizer"]) == pytest.approx(1e-4 / 2)
    optimizer, _ = build_optimizer_and_schedule(optim, 10, 5)
    assert optimizer.weight_decay is None


def test_resnet_with_adamw_decays_in_the_optimizer_only(builtin):
    kwargs, optim = resolved("resnet", "resnet18", optimizer="adamw", lr=1e-3, weight_decay=0.02)
    assert kwargs["kernel_regularizer"] is None
    optimizer, _ = build_optimizer_and_schedule(optim, 10, 5)
    assert optimizer.weight_decay == 0.02


def test_the_decay_is_in_exactly_one_place(builtin):
    """Across both optimizers and every family, the decay is never counted twice."""
    for family in sorted(VARIANTS):
        for name in ("adamw", "sgd"):
            if name == "sgd" and family in MIM_CLASSIFIERS:
                continue  # refused instead: their factories take no regularizer (the test below)
            kwargs, optim = resolved(
                family, FIRST_VARIANT[family], optimizer=name, lr=0.01, weight_decay=0.03,
                momentum=0.9 if name == "sgd" else None,
            )
            optimizer, _ = build_optimizer_and_schedule(optim, 10, 5)
            in_model = kwargs.get("kernel_regularizer") is not None
            in_optimizer = bool(optimizer.weight_decay)
            assert in_model != in_optimizer, (family, name)
            assert in_model == (name == "sgd")
            if in_model:
                assert l2_strength(kwargs["kernel_regularizer"]) == pytest.approx(0.03 / 2)


#: What each row's refusal says of the two keys the factories treat differently (plan 5c6cc105 D-045, review R4); the
#: build loop of the test below measures each.
OWN_REASON = {
    "beit": {"include_top": "builds a BEiT with no head", "use_shared_relative_position_bias": "the BEiT constructor raises"},
    "energy_transformer": {"include_top": "the Energy Transformer constructor does not take the key",
                           "use_shared_relative_position_bias": "the Energy Transformer has no relative position bias"},
}


@pytest.mark.parametrize("family", MIM_CLASSIFIERS)
def test_the_mim_classifiers_refuse_what_their_factory_would_drop_or_crash_on_before_the_claim(family, builtin):
    """sgd's decay has no regularizer to become, and three keys would be refused after the claim (a ValueError for
    ``kernel_regularizer``), crash it (a TypeError for ``name``) or train a model with no head (``include_top``)."""
    variant = FIRST_VARIANT[family]
    with pytest.raises(ConfigError) as caught:
        resolved(family, variant, optimizer="sgd", lr=0.05, weight_decay=0.01, momentum=0.9)
    assert "optim_args.weight_decay" in str(caught.value) and f"builds none for {family!r}" in str(caught.value)
    for key, value in (("kernel_regularizer", None), ("kernel_regularizer", {"class_name": "L2"}),
                       ("include_top", False), ("name", "x"), ("use_shared_relative_position_bias", True)):
        with pytest.raises(ConfigError, match=rf"Invalid value for 'model_args.{key}': it is not offered for '{family}'") \
                as refused:
            resolved(family, variant, user={key: value})
        message = str(refused.value)
        assert OWN_REASON[family].get(key, "") in message, message
        assert family == "beit" or "beit" not in message.lower(), message  # the Energy Transformer's reasons name no BEiT
    # Each reason is still true of the library (a factory that took one would make its row a dead refusal).
    spec = builtin[family]
    built = []
    for key, value in (("kernel_regularizer", None), ("name", "x"), ("include_top", False),
                       ("use_shared_relative_position_bias", True)):
        try:
            model = recipes.build_model(spec, variant, 10, (32, 32, 3), {"patch_size": 4, key: value})
            built.append((key, tuple(model(np.zeros((1, 32, 32, 3), "float32")).shape)))
        except (ConfigError, TypeError) as error:
            assert isinstance(error, TypeError) == (key == "name"), (key, error)
    # Only BEiT builds with include_top=False, and then emits its pooled features where the logits belong.
    assert built == ([("include_top", (1, 192))] if family == "beit" else [])
    with pytest.raises(ConfigError, match="'scale' is fixed by the variant"):
        resolved(family, variant, user={"scale": "base"})
    # The controls: sgd without a decay resolves, and so does a key the factory takes.
    kwargs, _ = resolved(family, variant, optimizer="sgd", lr=0.05, momentum=0.9)
    assert "kernel_regularizer" not in kwargs
    assert resolved(family, variant, user={"dropout_rate": 0.2})[0]["dropout_rate"] == 0.2


def test_the_energy_transformer_classifier_refuses_an_unbounded_clip_with_et_mims_sentence(builtin):
    """One function refuses it for both Energy Transformer rows: the sentences differ only in the family's name."""
    for given in ({"global_clipnorm": None}, {"clipnorm": 0.5}):
        with pytest.raises(ConfigError) as caught:
            resolved("energy_transformer", "et_tiny", **given)
        with pytest.raises(ConfigError) as pretrain:
            resolved("et_mim", "et_tiny", **given)
        assert "'energy_transformer' family needs the global gradient clip" in str(caught.value), given
        assert str(caught.value) == str(pretrain.value).replace("'et_mim'", "'energy_transformer'"), given
    # The controls: a clip of another size, and BEiT with none, resolve.
    assert resolved("energy_transformer", "et_tiny", global_clipnorm=0.5)[1].global_clipnorm == 0.5
    assert resolved("beit", "beit_tiny", global_clipnorm=None)[1].global_clipnorm is None


def test_sgd_on_an_adamw_family_takes_no_decay_from_the_recipe(builtin):
    kwargs, optim = resolved("vit", "vit_pico", optimizer="sgd", lr=0.05)
    assert optim.weight_decay is None
    assert kwargs["kernel_regularizer"] is None


def test_an_explicit_regularizer_with_adamw_is_refused(builtin):
    with pytest.raises(ConfigError, match="twice"):
        resolved("vit", "vit_pico", user={"kernel_regularizer": {"class_name": "L2"}})
    # Controls: null is fine, and so is a regularizer for sgd.
    resolved("vit", "vit_pico", user={"kernel_regularizer": None})
    resolved("resnet", "resnet18", user={"kernel_regularizer": {"class_name": "L2"}})


@pytest.mark.parametrize("key", ["variant", "num_classes", "input_shape"])
def test_a_key_the_library_supplies_is_refused(key, builtin):
    with pytest.raises(ConfigError, match=f"'{key}'"):
        resolved("vit", "vit_pico", user={key: 1})


def test_a_caller_key_replaces_the_recipe_key(builtin):
    kwargs, _ = resolved(
        "vit", "vit_pico", user={"dropout_rate": 0.3, "patch_size": 8, "extra": 1}
    )
    assert kwargs["dropout_rate"] == 0.3
    assert kwargs["patch_size"] == 8
    assert kwargs["attention_dropout_rate"] == 0.1
    assert kwargs["extra"] == 1


def test_an_unresolved_optimizer_is_refused(builtin):
    ctx = context("resnet", "resnet18", "cifar10")
    with pytest.raises(UnresolvedAutoError, match="optim_args.optimizer"):
        builtin["resnet"].load_resolver()(ctx, OptimArgs(), {})


@pytest.mark.parametrize(
    "family, variant",
    [("convnext_v1", "cifar10"), ("cliffordnet", "nano"), ("cliffordnet_hier", "hier_345")],
)
def test_sgd_on_a_family_without_a_regularizer_path_still_decays(family, variant, builtin):
    """The legacy AdamW trainers never pass a regularizer; sgd's decay must not vanish."""
    kwargs, _ = resolved(
        family, variant, optimizer="sgd", lr=0.05, weight_decay=0.01, momentum=0.9
    )
    assert l2_strength(kwargs["kernel_regularizer"]) == pytest.approx(0.01 / 2)
    plain, _ = resolved(family, variant)
    assert "kernel_regularizer" not in plain


# ---------------------------------------------------------------------
# Warmup against epochs, by recipe data
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, refused",
    [
        ("vit", False),
        ("resnet", False),
        ("convnext_v1", True),
        ("convnext_v2", True),
        ("cliffordnet", True),
        ("cliffordnet_hier", True),
        ("beit", False),
        ("energy_transformer", False),
    ],
)
def test_warmup_reaching_epochs_is_a_recipe_decision(family, refused, builtin):
    ctx = context(family, FIRST_VARIANT[family], "cifar10")
    optim = OptimArgs(warmup_epochs=5).resolve_auto(ctx)
    if refused:
        with pytest.raises(ConfigValidationError, match="optim_args.warmup_epochs"):
            build_optimizer_and_schedule(optim, 10, 5)
    else:
        build_optimizer_and_schedule(optim, 10, 5)
    # Control: one more epoch than the warmup trains in every family.
    build_optimizer_and_schedule(optim, 10, 6)


# ---------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------

#: Attributes the built model must carry, so the recipe reached the factory (a dotted name reads the backbone).
BUILT = {
    "vit": {"patch_size": (4, 4), "dropout_rate": 0.1, "attention_dropout_rate": 0.1},
    "resnet": {"stem_type": "cifar"},
    "convnext_v1": {"strides": 2, "drop_path_rate": 0.1, "dropout_rate": 0.1},
    "convnext_v2": {"strides": 2, "drop_path_rate": 0.1, "dropout_rate": 0.1},
    "cliffordnet": {"stochastic_depth_rate": 0.3},
    "cliffordnet_hier": {"stochastic_depth_rate": 0.1},
    "beit": {"backbone.patch_size": 4, "backbone.drop_path_rate": 0.1, "dropout_rate": 0.0},
    "energy_transformer": {"backbone.patch_size": (4, 4), "backbone.num_steps": 12, "dropout_rate": 0.0},
}


@pytest.mark.parametrize("family", sorted(VARIANTS))
def test_each_family_builds_from_its_defaults(family, builtin):
    variant = FIRST_VARIANT[family]
    spec = builtin[family]
    kwargs, _ = resolved(family, variant)
    model = recipes.build_model(spec, variant, 10, (32, 32, 3), kwargs)
    logits = np.asarray(model(np.zeros((2, 32, 32, 3), "float32"), training=False))
    assert logits.shape == (2, 10)
    assert np.isfinite(logits).all()
    for name, value in BUILT[family].items():
        assert functools.reduce(getattr, name.split("."), model) == value, name


def test_an_unknown_model_args_key_is_a_config_error(builtin):
    spec = builtin["vit"]
    kwargs, _ = resolved("vit", "vit_pico", user={"bogus_key": 1})
    with pytest.raises(ConfigError, match="bogus_key") as caught:
        recipes.build_model(spec, "vit_pico", 10, (32, 32, 3), kwargs)
    assert isinstance(caught.value.__cause__, ValueError)
    # Control: without the key the same call builds.
    clean, _ = resolved("vit", "vit_pico")
    recipes.build_model(spec, "vit_pico", 10, (32, 32, 3), clean)


def test_an_input_the_family_cannot_take_is_a_config_error(builtin):
    """vit at 28 px with a 16-pixel patch: the factory refuses (the resolver refuses it first, the test below)."""
    kwargs, _ = resolved("vit", "vit_pico", dataset="mnist", image_size=28)
    assert kwargs["patch_size"] == 4
    with pytest.raises(ConfigError, match="vit"):
        recipes.build_model(builtin["vit"], "vit_pico", 10, (28, 28, 1), {**kwargs, "patch_size": 16})


#: ``side: the patch the rule gives`` (plan 5c6cc105 D-047, review R6): 4 at a side of at most 32 px that 4 divides
#: (CIFAR 32, MNIST 28), 16 otherwise; before, 4 at exactly 32 px only, so MNIST took 16 and failed after the claim.
PATCH_RULE = {32: 4, 28: 4, 16: 4, 8: 4, 64: 16, 224: 16, 30: 16, 36: 16}


@pytest.mark.parametrize("family", ["vit", *MIM_CLASSIFIERS])
def test_the_patch_divides_the_side_or_the_request_is_refused_before_the_claim_naming_both(family, builtin):
    variant = FIRST_VARIANT[family]
    for side, patch in PATCH_RULE.items():
        if side % patch:
            with pytest.raises(ConfigError) as refused:
                resolved(family, variant, image_size=side)
            assert str(refused.value) == (
                f"Invalid value for 'model_args.patch_size': {family!r} cuts its input into square patches of "
                f"{patch} px (model_args.patch_size), which do not tile the image side of {side} px "
                f"(transform_args.image_size); ask for a patch that divides the side, or a side it divides")
        else:
            assert resolved(family, variant, image_size=side)[0]["patch_size"] == patch, side
    with pytest.raises(ConfigError, match=r"square patches of 16 px \(model_args.patch_size\), which do not tile the "
                                          r"image side of 28 px"):
        resolved(family, variant, dataset="mnist", image_size=28, user={"patch_size": 16})
    model = recipes.build_model(builtin[family], variant, 10, (28, 28, 1), resolved(family, variant, image_size=28)[0])
    assert tuple(model(np.zeros((1, 28, 28, 1), "float32")).shape) == (1, 10)  # MNIST builds now


def test_only_value_errors_are_translated(builtin):
    """A TypeError from a factory is a bug and reaches the caller as it is."""

    spec = dataclasses.replace(
        builtin["vit"], factory="builtins:int"  # int(variant=...) raises TypeError
    )
    with pytest.raises(TypeError):
        recipes.build_model(spec, "vit_pico", 10, (32, 32, 3), {})


def test_no_family_is_named_like_a_command(builtin):
    assert set(builtin).isdisjoint(cli.COMMANDS)
