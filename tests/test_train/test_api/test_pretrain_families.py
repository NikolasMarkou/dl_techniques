"""The two pretrain model families and their groups: adapters, refusals, recipes, the mask rule and the registration.

``train.api.pretrain_models`` and ``train.api.pretrain_args`` (plan-2026-11-12T015747-46e5318f step 9) adapt
``et_mim`` (ET masked-patch completion) and ``beit_mim`` (masked image modelling against a frozen
tokenizer's code ids) to the factory call the API makes, hold the per-family refusal tables, the recipes and the
``data`` group with its mask keys. The first part of this file calls everything directly and pins:

* the adapters: a non-``None`` ``num_classes`` is refused, the model is returned BUILT for the two inputs
  ``[image, mask]`` (a task returns a built model, so ``run.model.build`` is skipped), the keywords the task
  supplies (``patch_size``, ``vocab_size``) are required and the keys it reads (``tokenizer_checkpoint``) are not
  passed to the model;
* the refusal tables, held equal to the factories' signatures by a census (the denoising pattern), one row per
  refused key through the real resolver, a key of the other family refused by NAME, every refusal at once;
* the recipes, equal to the legacy ``TrainingConfig`` defaults READ FROM the legacy modules (never typed here),
  with the per-dataset geometry (cifar10 32 / 4, imagenette 224 / 16) read from the legacy command lines, the clip
  as ``global_clipnorm`` and never a literal ``clipnorm``, built into a real optimizer;
* the ET gradient clip cannot be switched off, an sgd run with a positive decay is refused (no L2 penalty is
  built here), as the segmentation and denoising resolvers do;
* the ``data`` group: the mask keys of the other family are refused, ``data.num_mask_patches`` above the grid is
  refused by name, the auto value follows the BEiT recipe's mask fraction (D-008, D-018: 24 and 5 at an 8 x 8 grid);
* a request that can be refused without data is refused with no TensorFlow device initialised: one child process
  runs the refusal rows through the REAL command line (``train.api.main(["pretrain", ...])``) with a recorder on
  TensorFlow's eager context and a spy on the claim, so ``out`` stays absent and no tensor is created. The names the two
  families had before plan-2026-11-19T081214-5c6cc105 step 3 (``energy_transformer``, ``beit``; renamed with no alias) are
  refused there the same way, each told the families that train pretrain.

The second part (plan step 10) is the registration: the task row, the two families under the ``pretrain`` task
(their variants read from the model modules' ``MODEL_VARIANTS``), a family of another task refused in both
directions, and the help and listing paths that stay free of TensorFlow.

Everything writes under ``tmp_path``.
"""

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest

from train.api import data_args, pretrain_args as pa, pretrain_models as pm, recipes, registry
from train.api.args import AutoContext, CheckpointArgs, OptimArgs
from train.api.config import AUTO
from train.api.errors import ConfigError, ConfigValidationError
from train.api.optim import build_optimizer_and_schedule

REPO_ROOT = Path(__file__).resolve().parents[3]

FAMILIES = ("et_mim", "beit_mim")
DATASETS = ("cifar10", "imagenette")
SPECS = {spec.family: spec for spec in pm.builtin_pretrain_models()}


def legacy_modules():
    """The two legacy trainers, imported on use (they import TensorFlow)."""
    from train.beit import train_mim
    from train.energy_transformer import train_masked_completion

    return {"et_mim": train_masked_completion, "beit_mim": train_mim}


def legacy_defaults(family: str) -> Dict[str, Any]:
    """``{field: default}`` of the legacy ``TrainingConfig`` of ``family``, read by looking."""
    config = legacy_modules()[family].TrainingConfig
    return {f.name: f.default for f in dataclasses.fields(config) if f.default is not dataclasses.MISSING}


def legacy_geometry(family: str, dataset: str, tmp_path: Path) -> Tuple[int, int]:
    """``(image_size, patch_size)`` the legacy command line gives ``dataset`` (its ``config_from_args``)."""
    module = legacy_modules()[family]
    argv = ["--dataset", dataset]
    if family == "beit_mim":
        # The BEiT config refuses a default of 75 masked patches on an 8 x 8 grid, and needs a tokenizer file.
        tokenizer = tmp_path / "tokenizer.keras"
        tokenizer.write_bytes(b"")
        argv += ["--num-mask-patches", "1", "--min-mask-patches-per-block", "1", "--tokenizer-checkpoint", str(tokenizer)]
    config = module.config_from_args(module.parse_arguments(argv))
    return config.image_size, config.patch_size


def context(family: str, variant: str = None, dataset: str = "cifar10", image_size: int = None) -> AutoContext:
    spec = SPECS[family]
    variant = variant or ("et_tiny" if family == "et_mim" else "beit_tiny")
    size = pa.GEOMETRY[dataset][0] if image_size is None else image_size
    return AutoContext(
        family=family, variant=variant, dataset=dataset, num_classes=None, image_size=size,
        defaults=spec.recipe(dataset, variant),
    )


def resolved_optim(family: str, dataset: str = "cifar10", **given: Any) -> OptimArgs:
    ctx = context(family, dataset=dataset)
    return OptimArgs(**given).resolve_auto(ctx)


def resolve(family: str, dataset: str = "cifar10", model_args: Dict[str, Any] = None, **optim: Any) -> Dict[str, Any]:
    """The factory keywords the task's ``check_model_args`` gets from the resolver."""
    ctx = context(family, dataset=dataset)
    return SPECS[family].load_resolver()(ctx, OptimArgs(**optim).resolve_auto(ctx), model_args or {})


def data_group(family: str, dataset: str = "cifar10", image_size: int = None, **given: Any) -> pa.PretrainData:
    ctx = context(family, dataset=dataset, image_size=image_size)
    return pa.PretrainData(dataset=dataset, **given).resolve_auto(ctx)


# ---------------------------------------------------------------------
# The constants this module copies, held equal to their homes
# ---------------------------------------------------------------------


def test_the_datasets_are_the_shared_loaders():
    from train.common.image_streams import SUPPORTED_DATASETS

    assert sorted(pa.DATASETS) == sorted(SUPPORTED_DATASETS)
    assert sorted(pa.GEOMETRY) == sorted(pa.DATASETS)


def test_the_methods_are_the_wrappers_methods():
    from train.common import pretrain_data

    assert sorted(pm.METHOD_OF.values()) == sorted(pretrain_data.METHODS)
    assert sorted(pm.METHOD_OF) == sorted(FAMILIES)


def test_the_beit_mask_fractions_are_the_librarys_and_the_legacy_trainers_constants():
    from dl_techniques.datasets.vision import beit_masking

    assert pa.BEIT_NUM_MASK_PATCHES == beit_masking.BEIT_NUM_MASK_PATCHES == legacy_defaults("beit_mim")["num_mask_patches"]
    assert (
        pa.BEIT_MIN_MASK_PATCHES_PER_BLOCK
        == beit_masking.BEIT_MIN_MASK_PATCHES_PER_BLOCK
        == legacy_defaults("beit_mim")["min_mask_patches_per_block"]
    )
    # 196 is the grid the recipe constants are stated at: 224 px at patch 16.
    size, patch = pa.GEOMETRY["imagenette"]
    assert pa.BEIT_GRID_PATCHES == (size // patch) ** 2


# ---------------------------------------------------------------------
# The recipes equal the legacy trainers' defaults
# ---------------------------------------------------------------------

#: ``(optim field, legacy TrainingConfig field)``: the optimizer both families share the shape of.
OPTIM_FIELDS = (
    ("optimizer", "optimizer_type"), ("lr", "learning_rate"), ("weight_decay", "weight_decay"),
    ("warmup_epochs", "warmup_epochs"), ("global_clipnorm", "gradient_clipping"), ("schedule", "lr_schedule_type"),
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dataset", DATASETS)
def test_the_recipe_is_the_legacy_trainers_defaults(family, dataset):
    legacy = legacy_defaults(family)
    variant = "et_tiny" if family == "et_mim" else "beit_tiny"
    recipe = SPECS[family].recipe(dataset, variant)
    assert recipe["epochs"] == legacy["epochs"]
    assert recipe["batch_size"] == legacy["batch_size"]
    assert recipe["checkpoint_args"] == {"early_stopping_patience": legacy["early_stopping_patience"]}
    for field, legacy_field in OPTIM_FIELDS:
        assert recipe["optim_args"][field] == legacy[legacy_field], field
    # The decay of the legacy trainer is `build_optimizer`'s: cosine over ALL the epochs, alpha 0.01.
    assert recipe["optim_args"]["decay_over"] == "total"
    assert recipe["optim_args"]["min_lr_ratio"] == 0.01
    assert recipe["metric_args"] == {"watch_metric": "val_loss"}


def test_the_et_recipe_is_the_paper_regime_the_legacy_trainer_has():
    # The numbers of the task text (ET: 100, 32, 5e-4, 0.05, 2, 1.0, 15), asserted from the legacy module AND
    # as literals, so a legacy trainer that moves is named here and not silently followed.
    legacy = legacy_defaults("et_mim")
    assert (
        legacy["epochs"], legacy["batch_size"], legacy["learning_rate"], legacy["weight_decay"],
        legacy["warmup_epochs"], legacy["gradient_clipping"], legacy["early_stopping_patience"],
    ) == (100, 32, 5e-4, 0.05, 2, 1.0, 15)
    recipe = SPECS["et_mim"].recipe("cifar10", "et_tiny")
    assert recipe["optim_args"]["lr"] == 5e-4 and recipe["optim_args"]["global_clipnorm"] == 1.0


def test_the_model_and_mask_keys_of_the_recipes_are_the_legacy_trainers_defaults():
    et, beit = legacy_defaults("et_mim"), legacy_defaults("beit_mim")
    recipe = SPECS["et_mim"].recipe("cifar10", "et_tiny")
    assert recipe["model_args"] == {"num_steps": et["num_steps"]}
    assert recipe["data"]["mask_ratio"] == et["mask_ratio"]
    assert recipe["data"]["mask_token_frac"] == et["mask_token_frac"]
    recipe = SPECS["beit_mim"].recipe("cifar10", "beit_tiny")
    assert recipe["model_args"] == {"drop_path_rate": beit["drop_path_rate"]}


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dataset", DATASETS)
def test_the_per_dataset_geometry_is_what_the_legacy_command_lines_give(family, dataset, tmp_path):
    size, patch = legacy_geometry(family, dataset, tmp_path)
    assert pa.GEOMETRY[dataset] == (size, patch)
    variant = "et_tiny" if family == "et_mim" else "beit_tiny"
    recipe = SPECS[family].recipe(dataset, variant)
    assert recipe["transform_args"]["image_size"] == size
    assert recipe["data"]["patch_size"] == patch


@pytest.mark.parametrize("family", FAMILIES)
def test_the_clip_is_a_global_clipnorm_and_never_a_literal_clipnorm(family):
    base = SPECS[family].defaults["optim_args"]
    assert base["clipnorm"] is None
    assert base["global_clipnorm"] == legacy_defaults(family)["gradient_clipping"]
    # Built into a real optimizer: the clip reaches Keras as the GLOBAL norm (a literal `clipnorm` clips each variable).
    optim = resolved_optim(family)
    assert optim.clipnorm is None and optim.global_clipnorm == base["global_clipnorm"]
    optimizer, _ = build_optimizer_and_schedule(optim, 10, 100)
    assert optimizer.global_clipnorm == base["global_clipnorm"]
    assert optimizer.clipnorm is None


@pytest.mark.parametrize("family", FAMILIES)
def test_every_auto_field_of_every_group_is_listed_by_the_recipe_or_derived(family):
    recipe = SPECS[family].recipe("cifar10", "et_tiny" if family == "et_mim" else "beit_tiny")
    assert set(recipe) <= set(recipes.RECIPE_KEYS)
    auto = lambda cls: {f.name for f in dataclasses.fields(cls) if f.default == AUTO}
    assert set(recipe["optim_args"]) == auto(OptimArgs)
    assert set(recipe["checkpoint_args"]) == auto(CheckpointArgs)
    assert set(recipe["metric_args"]) == auto(pa.PretrainMetricArgs) - {"watch_mode"}
    assert set(recipe["transform_args"]) == auto(pa.PretrainTransformArgs)  # image_size, from the dataset overlay
    assert set(recipe["data"]) == auto(pa.PretrainData) - ({"num_mask_patches", "min_mask_patches_per_block"} if family == "beit_mim" else set())


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dataset", DATASETS)
def test_nothing_is_left_auto_after_the_groups_resolve(family, dataset):
    ctx = context(family, dataset=dataset)
    for group in (
        OptimArgs(), CheckpointArgs(), pa.PretrainData(dataset=dataset), pa.PretrainTransformArgs(), pa.PretrainMetricArgs(),
    ):
        resolved = group.resolve_auto(ctx)
        for field in dataclasses.fields(resolved):
            assert getattr(resolved, field.name) != AUTO, (type(group).__name__, field.name)


def test_a_recipe_key_that_is_no_field_is_named():
    ctx = context("beit_mim")
    broken = {**ctx.defaults, "data": {**ctx.defaults["data"], "mask_ratiu": 0.5}}
    with pytest.raises(ConfigValidationError, match="mask_ratiu"):
        pa.PretrainData(dataset="cifar10").resolve_auto(dataclasses.replace(ctx, defaults=broken))


@pytest.mark.parametrize("family", FAMILIES)
def test_a_recipe_notice_states_the_regime_it_doubts_with_the_recipes_own_numbers(family):
    notices = pm.recipe_notices_of(family)
    assert notices and all(level == "info" and text for level, text in notices)
    text = " ".join(text for _, text in notices)
    optim = SPECS[family].defaults["optim_args"]
    assert f"{optim['lr']:g}" in text, "the notice names the learning rate"
    assert f"batch {SPECS[family].defaults['batch_size']}" in text, "the notice names the batch"
    assert "unmeasured" in text, "it says the effect is not measured"
    if family == "et_mim":
        assert "global" in text and "clip" in text


# ---------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------

#: ``(family, variant, extra keywords)``: the tiny build of each adapter at 32 px and patch 4.
BUILDS = (("et_mim", "et_tiny", {}), ("beit_mim", "beit_tiny", {"vocab_size": 16}))


def factory(family: str):
    return SPECS[family].load_factory()


def build_tiny(family: str, variant: str, extra: Dict[str, Any], **model_args: Any):
    return factory(family)(variant=variant, num_classes=None, input_shape=(32, 32, 3), patch_size=4, **extra, **model_args)


@pytest.fixture(scope="module")
def tiny_models():
    """One tiny model per family, built once (a build is the cost)."""
    return {family: build_tiny(family, variant, extra) for family, variant, extra in BUILDS}


@pytest.mark.parametrize("family, variant, extra", BUILDS)
def test_a_class_count_is_refused_by_the_adapter_and_by_build_model(family, variant, extra):
    spec = SPECS[family]
    with pytest.raises(ValueError, match="has no classes"):
        factory(family)(variant=variant, num_classes=10, input_shape=(32, 32, 3), patch_size=4, **extra)
    kwargs = {**resolve(family), "patch_size": 4, **extra}
    with pytest.raises(ConfigError, match="has no classes"):
        recipes.build_model(spec, variant, 10, (32, 32, 3), kwargs)
    # The control: None builds, through the real call path of `recipes.build_model`.
    assert recipes.build_model(spec, variant, None, (32, 32, 3), kwargs) is not None


@pytest.mark.parametrize("family, variant, extra", BUILDS)
def test_the_model_comes_back_built_for_the_image_and_the_mask(family, variant, extra, tiny_models):
    model = tiny_models[family]
    assert model.built, "the adapter returns a BUILT model: the sequence's own build call is then skipped"
    assert len(model.weights) > 0
    images = np.zeros((2, 32, 32, 3), "float32")
    mask = np.zeros((2, 64), bool)
    mask[:, :8] = True
    out = model([images, mask], training=False)
    width = 4 * 4 * 3 if family == "et_mim" else 16
    assert tuple(out.shape) == (2, 64, width)


@pytest.mark.parametrize("family, variant, extra", BUILDS)
def test_the_variant_resolves_in_both_spellings(family, variant, extra):
    scale = variant.split("_", 1)[1]
    a = build_tiny(family, variant, extra)
    b = build_tiny(family, scale, extra)
    assert a.count_params() == b.count_params()


@pytest.mark.parametrize("shape", [(32, 32), (32, 32, 3, 1), (0, 32, 3), (32, 32, 0), (32.0, 32, 3), (32, None, 3), (None, None, 3)])
@pytest.mark.parametrize("family, variant, extra", BUILDS)
def test_a_malformed_input_shape_is_refused(shape, family, variant, extra):
    with pytest.raises(ValueError, match="input_shape"):
        factory(family)(variant=variant, num_classes=None, input_shape=shape, patch_size=4, **extra)


@pytest.mark.parametrize("patch", [0, -4, 3, 4.0, True])
@pytest.mark.parametrize("family, variant, extra", BUILDS)
def test_a_patch_size_that_does_not_divide_the_image_is_refused(patch, family, variant, extra):
    with pytest.raises(ValueError, match="patch_size"):
        factory(family)(variant=variant, num_classes=None, input_shape=(32, 32, 3), patch_size=patch, **extra)


@pytest.mark.parametrize("family, variant, extra", BUILDS)
def test_the_task_keywords_are_required_not_defaulted(family, variant, extra):
    with pytest.raises(TypeError, match="patch_size"):
        factory(family)(variant=variant, num_classes=None, input_shape=(32, 32, 3), **extra)
    if family == "beit_mim":
        with pytest.raises(TypeError, match="vocab_size"):
            factory(family)(variant=variant, num_classes=None, input_shape=(32, 32, 3), patch_size=4)


@pytest.mark.parametrize("vocab", [0, 1, -3, True, 16.0, "16"])
def test_a_vocabulary_that_is_not_a_codebook_size_is_refused(vocab):
    with pytest.raises(ValueError, match="vocab_size"):
        factory("beit_mim")(variant="beit_tiny", num_classes=None, input_shape=(32, 32, 3), patch_size=4, vocab_size=vocab)


def test_the_vocabulary_is_the_models_output_width(tiny_models):
    assert tiny_models["beit_mim"].get_config()["vocab_size"] == 16


def test_the_adapter_refuses_a_refused_key_as_a_value_error_not_a_type_error():
    # `return_energy` is a backbone keyword the library itself refuses with its own text, `name` would collide
    # with the library's keyword: both arrive as ValueError, which `recipes.build_model` turns into a ConfigError.
    with pytest.raises(ValueError, match="return_energy"):
        build_tiny("et_mim", "et_tiny", {}, return_energy=True)
    with pytest.raises(ValueError, match="name"):
        build_tiny("beit_mim", "beit_tiny", {"vocab_size": 16}, name="mine")
    with pytest.raises(ValueError, match="Unknown key 'model_args.no_such_key'"):
        build_tiny("et_mim", "et_tiny", {}, no_such_key=1)


def test_the_adapter_does_not_take_the_tokenizer_path_the_task_reads():
    with pytest.raises(ValueError, match="tokenizer_checkpoint"):
        build_tiny("beit_mim", "beit_tiny", {"vocab_size": 16}, tokenizer_checkpoint="t.keras")


def test_a_model_argument_reaches_the_model():
    model = build_tiny("et_mim", "et_tiny", {}, num_steps=3)
    assert model.backbone.num_steps == 3
    beit = build_tiny("beit_mim", "beit_tiny", {"vocab_size": 16}, drop_path_rate=0.3)
    assert beit.backbone.drop_path_rate == 0.3


def test_the_keras_round_trip_is_exact(tiny_models, tmp_path):
    for family, model in tiny_models.items():
        path = tmp_path / f"{family}.keras"
        model.save(path)
        loaded = keras.saving.load_model(path)
        images = np.random.default_rng(0).uniform(size=(2, 32, 32, 3)).astype("float32")
        mask = np.zeros((2, 64), bool)
        mask[:, :6] = True
        np.testing.assert_array_equal(
            model([images, mask], training=False).numpy(), loaded([images, mask], training=False).numpy()
        )


# ---------------------------------------------------------------------
# The refusal tables and the signature census
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_the_census_classifies_every_parameter_of_every_factory(family):
    assert pm.signature_census(family) == {"unclassified": [], "stale": []}


@pytest.mark.parametrize("family", FAMILIES)
def test_the_census_walks_the_backbone_constructor_where_the_overrides_land(family):
    walked = {name for names in pm.factory_signatures(family).values() for name in names}
    assert {"scale", "name", "input_shape", "patch_size"} <= walked
    assert "num_steps" in walked if family == "et_mim" else "drop_path_rate" in walked


@pytest.mark.parametrize("family", FAMILIES)
def test_the_census_fails_when_a_factory_gains_a_parameter_no_table_names(family):
    real = pm.factory_signatures(family)
    assert pm.signature_census(family, real) == {"unclassified": [], "stale": []}
    grown = {path: [*names, "new_option"] for path, names in real.items()}
    assert pm.signature_census(family, grown)["unclassified"] == ["new_option"]
    gone = sorted(pm.ALLOWED[family])[0]
    shrunk = {path: [name for name in names if name != gone] for path, names in real.items()}
    assert pm.signature_census(family, shrunk)["stale"] == [gone]


@pytest.mark.parametrize("family", FAMILIES)
def test_the_classes_of_a_key_do_not_overlap(family):
    tables = [set(pm.ALLOWED[family]), set(pm.REFUSED[family]), set(pm.SUPPLIED[family]), set(pm.TASK_KEYS[family])]
    for index, first in enumerate(tables):
        for second in tables[index + 1:]:
            assert not (first & second), (family, first & second)


def test_the_supplied_keys_are_what_the_task_passes():
    assert set(pm.SUPPLIED["et_mim"]) == {"variant", "input_shape", "patch_size"}
    assert set(pm.SUPPLIED["beit_mim"]) == {"variant", "input_shape", "patch_size", "vocab_size"}
    assert pm.TASK_KEYS == {"et_mim": (), "beit_mim": ("tokenizer_checkpoint",)}


#: ``(family, model_args, fragments)``: a refused key, through the resolver; each fragment names the key and the rule.
REFUSALS: List[Tuple[str, Dict[str, Any], Tuple[str, ...]]] = [
    ("et_mim", {"tokenizer_checkpoint": "t.keras"}, ("model_args.tokenizer_checkpoint", "'beit_mim'")),
    ("et_mim", {"vocab_size": 16}, ("model_args.vocab_size", "'beit_mim'")),
    ("et_mim", {"drop_path_rate": 0.1}, ("model_args.drop_path_rate", "'beit_mim'")),
    ("et_mim", {"return_energy": True}, ("model_args.return_energy", "is not offered by 'et_mim'")),
    ("et_mim", {"name": "x"}, ("model_args.name", "is not offered by 'et_mim'")),
    ("et_mim", {"scale": "small"}, ("model_args.scale", "is not offered by 'et_mim'")),
    ("et_mim", {"patch_size": 8}, ("model_args.patch_size", "data.patch_size")),
    ("et_mim", {"no_such_key": 1}, ("Unknown key 'model_args.no_such_key'",)),
    ("beit_mim", {"num_steps": 6}, ("model_args.num_steps", "'et_mim'")),
    ("beit_mim", {"noise_std": 0.1}, ("model_args.noise_std", "'et_mim'")),
    ("beit_mim", {"vocab_size": 16}, ("model_args.vocab_size", "tokenizer")),
    ("beit_mim", {"patch_size": 8}, ("model_args.patch_size", "data.patch_size")),
    ("beit_mim", {"name": "x"}, ("model_args.name", "is not offered by 'beit_mim'")),
    ("beit_mim", {"scale": "small"}, ("model_args.scale", "is not offered by 'beit_mim'")),
    # plan 5c6cc105 D-031: the constructor raises for it, which would refuse it only after the claim.
    ("beit_mim", {"use_shared_relative_position_bias": True},
     ("model_args.use_shared_relative_position_bias", "is not offered by 'beit_mim'", "does not implement it")),
    ("beit_mim", {"no_such_key": 1}, ("Unknown key 'model_args.no_such_key'",)),
]

#: Requests that must NOT be refused (the control of the rows above: a resolver that refused everything would pass them).
ACCEPTED: List[Tuple[str, Dict[str, Any]]] = [
    ("et_mim", {}),
    ("et_mim", {"num_steps": 6, "step_size": 0.2, "noise_std": 0.01, "embed_dim": 96, "num_heads": 3}),
    ("et_mim", {"hopfield_activation": "softmax", "attn_self": True, "pos_dropout_rate": 0.1}),
    ("beit_mim", {}),
    ("beit_mim", {"drop_path_rate": 0.2, "hidden_dropout_rate": 0.1, "use_mean_pooling": False}),
    ("beit_mim", {"tokenizer_checkpoint": "t.keras"}),
]


@pytest.mark.parametrize("family, model_args, fragments", REFUSALS, ids=[f"{f}:{sorted(a)[0]}" for f, a, _ in REFUSALS])
def test_every_refused_key_is_named_with_its_rule(family, model_args, fragments):
    with pytest.raises(ConfigValidationError) as raised:
        resolve(family, model_args=model_args)
    text = str(raised.value)
    for fragment in fragments:
        assert fragment in text, (fragment, text)


@pytest.mark.parametrize("family, model_args", ACCEPTED, ids=[f"{f}:{'-'.join(sorted(a)) or 'recipe'}" for f, a in ACCEPTED])
def test_the_keys_a_family_offers_are_not_refused(family, model_args):
    assert resolve(family, model_args=model_args) is not None


def test_the_resolver_returns_the_recipes_keys_then_the_callers_without_the_keys_the_task_reads():
    assert resolve("et_mim") == {"num_steps": 12}
    assert resolve("et_mim", model_args={"num_steps": 6, "step_size": 0.2}) == {"num_steps": 6, "step_size": 0.2}
    assert resolve("beit_mim") == {"drop_path_rate": 0.1}
    assert resolve("beit_mim", model_args={"tokenizer_checkpoint": "t.keras", "drop_path_rate": 0.0}) == {"drop_path_rate": 0.0}


def test_the_resolver_names_every_refusal_at_once():
    with pytest.raises(ConfigValidationError) as raised:
        resolve("beit_mim", model_args={"num_steps": 6, "vocab_size": 16, "typo_key": 1})
    text = str(raised.value)
    assert "model_args.num_steps" in text and "model_args.vocab_size" in text and "model_args.typo_key" in text


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("key", ["variant", "num_classes", "input_shape"])
def test_a_key_the_library_supplies_is_refused_before_the_tables_are_read(family, key):
    with pytest.raises(ConfigError, match="is set by the library"):
        resolve(family, model_args={key: 1})


@pytest.mark.parametrize("family", FAMILIES)
def test_an_adamw_run_refuses_an_explicit_kernel_regularizer_like_the_other_tasks(family):
    with pytest.raises(ConfigError, match="kernel_regularizer"):
        resolve(family, model_args={"kernel_regularizer": "l2"})


@pytest.mark.parametrize("family", FAMILIES)
def test_an_sgd_weight_decay_is_refused_because_no_regularizer_is_built(family):
    with pytest.raises(ConfigError) as raised:
        resolve(family, optimizer="sgd", lr=0.1, weight_decay=1e-4)
    assert "optim_args.weight_decay" in str(raised.value)
    assert f"the pretrain task builds none for {family!r}" in str(raised.value)
    # The controls: sgd with no decay, and adamw with its decay, resolve.
    assert resolve(family, optimizer="sgd", lr=0.1, weight_decay=None) is not None
    assert resolve(family) is not None


def test_the_et_global_clip_cannot_be_switched_off():
    for given in ({"global_clipnorm": None}, {"clipnorm": 0.5}):
        with pytest.raises(ConfigError) as raised:
            resolve("et_mim", **given)
        assert "optim_args.global_clipnorm" in str(raised.value), given
        assert "et_mim" in str(raised.value)
    # A per-variable clipnorm turns the global clip off (the group resolves global_clipnorm to null beside it): the sentence
    # says so, instead of blaming a null the caller never wrote.
    with pytest.raises(ConfigError) as raised:
        resolve("et_mim", clipnorm=0.5)
    assert "optim_args.clipnorm=0.5" in str(raised.value)
    assert "replaced the global clip" in str(raised.value) and "needs global_clipnorm" in str(raised.value)
    with pytest.raises(ConfigError) as raised:
        resolve("et_mim", global_clipnorm=None)
    assert "replaced the global clip" not in str(raised.value)
    # A hand-built resolved group is held to it too.
    ctx = context("et_mim")
    nulled = dataclasses.replace(OptimArgs().resolve_auto(ctx), global_clipnorm=None)
    with pytest.raises(ConfigError, match="optim_args.global_clipnorm"):
        SPECS["et_mim"].load_resolver()(ctx, nulled, {})
    # The controls: ET with a clip of another size, and BEiT with none, resolve.
    assert resolve("et_mim", global_clipnorm=0.5) is not None
    assert resolve("beit_mim", global_clipnorm=None) is not None


# ---------------------------------------------------------------------
# The data group and the mask rule
# ---------------------------------------------------------------------


def test_the_auto_mask_rule_is_the_beit_recipes_fractions_of_the_grid():
    # D-008 and D-018: round(75 / 196 * N) and round(16 / 196 * N), the minimum clamped to the budget.
    assert pa.auto_mask_counts(64) == (24, 5)
    assert pa.auto_mask_counts(196) == (75, 16)
    assert pa.auto_mask_counts(1) == (1, 1)
    assert pa.auto_mask_counts(4) == (2, 1)
    for patches in range(1, 400):
        num, block = pa.auto_mask_counts(patches)
        assert 1 <= block <= num <= patches, patches


@pytest.mark.parametrize(
    "dataset, family, expected",
    [
        ("cifar10", "beit_mim", {"patch_size": 4, "num_mask_patches": 24, "min_mask_patches_per_block": 5, "mask_ratio": None, "mask_token_frac": None}),
        ("imagenette", "beit_mim", {"patch_size": 16, "num_mask_patches": 75, "min_mask_patches_per_block": 16, "mask_ratio": None, "mask_token_frac": None}),
        ("cifar10", "et_mim", {"patch_size": 4, "num_mask_patches": None, "min_mask_patches_per_block": None, "mask_ratio": 0.5, "mask_token_frac": 0.9}),
        ("imagenette", "et_mim", {"patch_size": 16, "num_mask_patches": None, "min_mask_patches_per_block": None, "mask_ratio": 0.5, "mask_token_frac": 0.9}),
    ],
)
def test_the_data_group_resolves_to_the_family_and_the_dataset(dataset, family, expected):
    data = data_group(family, dataset)
    assert {name: getattr(data, name) for name in expected} == expected
    assert data.format == "builtin" and data.dataset == dataset


def test_the_auto_count_follows_the_grid_not_the_legacy_absolute_75():
    # cifar10 at 32 px / patch 4 is an 8 x 8 grid of 64 patches: the legacy default of 75 cannot be drawn there.
    assert data_group("beit_mim", "cifar10").num_mask_patches == 24 < 64
    bigger = data_group("beit_mim", "cifar10", image_size=64, patch_size=4)
    assert bigger.num_mask_patches == round(pa.BEIT_NUM_MASK_PATCHES / pa.BEIT_GRID_PATCHES * 256)


def test_an_explicit_count_above_the_grid_is_refused_by_name():
    with pytest.raises(ConfigError) as raised:
        data_group("beit_mim", "cifar10", num_mask_patches=65)
    text = str(raised.value)
    assert "data.num_mask_patches" in text and "64" in text and "8 x 8" in text
    # The legacy default is the same refusal: it is why the legacy command line failed on cifar10.
    with pytest.raises(ConfigError, match="data.num_mask_patches"):
        data_group("beit_mim", "cifar10", num_mask_patches=pa.BEIT_NUM_MASK_PATCHES)
    # The control at the boundary: all 64 patches is a mask the library draws.
    assert data_group("beit_mim", "cifar10", num_mask_patches=64).num_mask_patches == 64


def test_a_block_minimum_above_the_budget_is_refused_by_name():
    with pytest.raises(ConfigError, match="data.min_mask_patches_per_block"):
        data_group("beit_mim", "cifar10", num_mask_patches=8, min_mask_patches_per_block=9)
    assert data_group("beit_mim", "cifar10", num_mask_patches=8, min_mask_patches_per_block=8).min_mask_patches_per_block == 8
    # An explicit budget under the auto minimum clamps the auto minimum, and an explicit minimum over the auto budget is refused.
    assert data_group("beit_mim", "cifar10", num_mask_patches=3).min_mask_patches_per_block == 3
    with pytest.raises(ConfigError, match="data.min_mask_patches_per_block"):
        data_group("beit_mim", "cifar10", min_mask_patches_per_block=40)


@pytest.mark.parametrize(
    "family, key, value, other",
    [
        ("et_mim", "num_mask_patches", 10, "mask_ratio"),
        ("et_mim", "min_mask_patches_per_block", 2, "mask_ratio"),
        ("beit_mim", "mask_ratio", 0.3, "num_mask_patches"),
        ("beit_mim", "mask_token_frac", 0.8, "num_mask_patches"),
    ],
)
def test_a_mask_key_of_the_other_family_is_refused_by_name(family, key, value, other):
    with pytest.raises(ConfigError) as raised:
        data_group(family, "cifar10", **{key: value})
    text = str(raised.value)
    assert f"data.{key}" in text and repr(family) in text and f"data.{other}" in text
    # The control: the same key as an explicit null is what the family already has.
    assert getattr(data_group(family, "cifar10", **{key: None}), key) is None


@pytest.mark.parametrize("family, key", [("et_mim", "mask_ratio"), ("et_mim", "mask_token_frac"), ("beit_mim", "num_mask_patches"), ("beit_mim", "min_mask_patches_per_block")])
def test_a_null_for_a_key_the_family_masks_with_is_refused(family, key):
    with pytest.raises(ConfigError, match=f"data.{key}"):
        data_group(family, "cifar10", **{key: None})


def test_a_patch_that_does_not_divide_the_image_is_refused_by_name():
    with pytest.raises(ConfigError) as raised:
        data_group("et_mim", "cifar10", patch_size=5)
    assert "data.patch_size" in str(raised.value) and "32" in str(raised.value)
    assert data_group("et_mim", "cifar10", patch_size=8).patch_size == 8
    with pytest.raises(ConfigError, match="data.patch_size"):
        data_group("et_mim", "cifar10", image_size=30)


def test_the_group_problems_name_a_value_out_of_range_at_its_dotted_path():
    lines = pa.PretrainData(
        dataset="cifar10", validation_split=1.5, max_samples=0, patch_size=0, mask_ratio=0.0, mask_token_frac=1.5,
        num_mask_patches=0, min_mask_patches_per_block=0,
    ).problems("data")
    text = "\n".join(lines)
    for name in ("validation_split", "max_samples", "patch_size", "mask_ratio", "mask_token_frac", "num_mask_patches", "min_mask_patches_per_block"):
        assert f"'data.{name}'" in text, name
    assert pa.PretrainData(dataset="cifar10").problems("data") == []


def test_a_dataset_the_loader_does_not_serve_is_refused():
    for name in ("cifar100", "imagenet", "mnist", "folder"):
        with pytest.raises(ConfigValidationError, match="dataset"):
            pa.load_pretrain_data_config({"dataset": name})


def test_the_data_config_loads_from_a_mapping_an_instance_and_a_yaml_file(tmp_path):
    data = pa.load_pretrain_data_config({"dataset": "cifar10", "max_samples": 64})
    assert isinstance(data, pa.PretrainData) and data.max_samples == 64 and data.format == "builtin"
    assert pa.load_pretrain_data_config(data) is data
    path = tmp_path / "data.yaml"
    path.write_text("dataset: imagenette\nvalidation_split: 0.2\n")
    from_file = pa.load_pretrain_data_config(str(path))
    assert (from_file.dataset, from_file.validation_split) == ("imagenette", 0.2)
    path.write_text("dataset: cifar10\nbogus: 1\n")
    with pytest.raises(ConfigValidationError, match="bogus") as raised:
        pa.load_pretrain_data_config(str(path))
    assert str(path) in str(raised.value)
    with pytest.raises(ConfigError, match="mapping"):
        pa.load_pretrain_data_config(3)


def test_the_transform_group_takes_the_size_the_data_reports():
    ctx = context("et_mim", dataset="cifar10")
    assert pa.PretrainTransformArgs().resolve_auto(ctx) == pa.PretrainTransformArgs(image_size=32, augment=True)
    assert pa.PretrainTransformArgs(image_size=64).resolve_auto(ctx).image_size == 64
    assert pa.PretrainTransformArgs(augment=False).resolve_auto(ctx).augment is False
    with pytest.raises(ConfigError, match="image_size"):
        pa.PretrainTransformArgs().resolve_auto(dataclasses.replace(ctx, image_size=None))
    assert pa.PretrainTransformArgs(image_size=0).problems("transform_args")


def test_the_metric_group_watches_val_loss_and_minimises_it():
    resolved = pa.PretrainMetricArgs().resolve_auto(context("beit_mim"))
    assert (resolved.watch_metric, resolved.watch_mode) == ("val_loss", "min")
    assert pa.PretrainMetricArgs(watch_metric="val_loss").resolve_auto(context("beit_mim")).watch_mode == "min"


@pytest.mark.parametrize(
    "given, key",
    [
        ({"watch_metric": "loss"}, "watch_metric"),
        ({"watch_metric": "val_psnr"}, "watch_metric"),
        ({"watch_metric": "val_loss", "watch_mode": "max"}, "watch_mode"),
        ({"watch_mode": "max"}, "watch_mode"),
    ],
)
def test_the_metric_group_refuses_what_the_bar_cannot_describe(given, key):
    # The bar is a statement about the MINIMUM `val_loss` of the selection window, so the checkpoint that is saved and
    # reloaded must be that epoch: another metric or the other direction saves weights the verdict does not describe.
    (line,) = pa.PretrainMetricArgs(**given).problems("metric_args")
    assert f"'metric_args.{key}'" in line and "val_loss" in line and "min" in line
    for allowed in ({}, {"watch_metric": "val_loss"}, {"watch_mode": "min"}, {"watch_metric": "val_loss", "watch_mode": "min"}):
        assert pa.PretrainMetricArgs(**allowed).problems("metric_args") == [], allowed


# ---------------------------------------------------------------------
# What is known before the data is loaded
# ---------------------------------------------------------------------


@pytest.fixture
def imagenette_records_present(monkeypatch):
    monkeypatch.setattr(data_args, "_check_imagenette_records", lambda: None)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_facts_are_the_datasets_geometry_and_no_classes(dataset, imagenette_records_present):
    facts = pa.pretrain_facts(pa.PretrainData(dataset=dataset), None)
    assert facts == data_args.DataFacts(dataset=dataset, num_classes=None, image_size=pa.GEOMETRY[dataset][0])
    assert pa.pretrain_facts(pa.PretrainData(dataset=dataset), 64).image_size == 64


def test_imagenette_without_its_records_is_refused_before_any_data_is_read(monkeypatch, tmp_path):
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "no_tfds"))
    with pytest.raises(ConfigError, match="imagenette"):
        pa.pretrain_facts(pa.PretrainData(dataset="imagenette"), None)
    # cifar10 has no records to look for.
    assert pa.pretrain_facts(pa.PretrainData(dataset="cifar10"), None).image_size == 32


# ---------------------------------------------------------------------
# Importing the modules
# ---------------------------------------------------------------------


def test_importing_the_modules_and_building_the_specs_imports_no_model_code():
    code = (
        "import sys\n"
        "from train.api import pretrain_args, pretrain_models\n"
        "specs = pretrain_models.builtin_pretrain_models()\n"
        "assert [s.family for s in specs] == ['et_mim', 'beit_mim']\n"
        "loaded = sorted(m for m in ('keras', 'tensorflow', 'dl_techniques') if m in sys.modules)\n"
        "print('LOADED', loaded)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "LOADED []"


# ---------------------------------------------------------------------
# Registration (plan step 10): the task row and the two families
# ---------------------------------------------------------------------


def variant_tables():
    """``{family: [variant names]}`` read from the model modules' ``MODEL_VARIANTS`` (never typed here)."""
    from dl_techniques.models.vision.beit.model import MODEL_VARIANTS as beit
    from dl_techniques.models.vision.energy_transformer.model import MODEL_VARIANTS as et

    return {"et_mim": list(et), "beit_mim": list(beit)}


@pytest.fixture
def registered(monkeypatch):
    """An empty registry holding what ``main`` registers: every built-in family and task."""
    from train.api.tasks import register_builtin_tasks

    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "TASKS", {})
    recipes.register_builtin_models()
    register_builtin_tasks()
    return registry


def test_the_two_families_are_registered_under_the_pretrain_task_only(registered):
    assert pa.TASK == "pretrain" == pm.TASK
    for family in FAMILIES:
        assert registry.MODELS[family].tasks == (pa.TASK,), family
    assert [spec.tasks for spec in pm.builtin_pretrain_models()] == [(pa.TASK,), (pa.TASK,)]
    # Control: every other family of the registry trains something else.
    assert not [f for f, spec in registry.MODELS.items() if f not in FAMILIES and pa.TASK in spec.tasks]


def test_the_task_row_names_the_config_and_the_function(registered):
    row = registry.TASKS[pa.TASK]
    assert row.config_class == "train.api.tasks.pretrain:PretrainConfig"
    assert row.function == "train.api.tasks.pretrain:train_pretrain"
    assert row.summary
    assert [spec.name for spec in registry.list_tasks()].count(pa.TASK) == 1
    from train.api.tasks.pretrain import PretrainConfig, train_pretrain

    assert row.load_config_class() is PretrainConfig
    assert row.load_function() is train_pretrain


def test_list_models_for_the_task_is_the_variant_tables(registered):
    tables = variant_tables()
    assert tables["et_mim"] and tables["beit_mim"]
    expected = sorted(f"{family}/{name}" for family, names in tables.items() for name in names)
    assert registry.list_models(pa.TASK) == expected
    assert all(name.split("/")[0] in FAMILIES for name in registry.list_models(pa.TASK))
    # The other tasks' listings do not gain them.
    for task in ("image_classification", "image_segmentation", "image_denoising"):
        assert not [n for n in registry.list_models(task) if n.split("/")[0] in FAMILIES], task


def test_the_listing_in_a_fresh_process_is_the_same(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "train", "list_models", "task=pretrain"], cwd=tmp_path, capture_output=True, text=True,
        timeout=300, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    expected = sorted(f"{family}/{name}" for family, names in variant_tables().items() for name in names)
    assert done.stdout.split() == expected


@pytest.mark.parametrize(
    "name, task, family_fragment",
    [
        ("vit/vit_pico", "pretrain", "image_classification"),
        ("bfcnn/tiny", "pretrain", "image_denoising"),
        ("convunext/tiny", "pretrain", "image_segmentation"),
        ("beit_mim/beit_tiny", "image_classification", "pretrain"),
        ("et_mim/et_tiny", "image_classification", "pretrain"),
        ("beit_mim/beit_tiny", "image_denoising", "pretrain"),
        ("et_mim/et_tiny", "image_segmentation", "pretrain"),
    ],
)
def test_a_family_of_another_task_is_refused_both_ways(registered, name, task, family_fragment):
    with pytest.raises(ConfigError) as caught:
        registry.parse_model_name(name, task=task)
    message = str(caught.value)
    assert f"trains {family_fragment}" in message and repr(task) in message, message
    # Control: each name is accepted by its own task.
    own = "pretrain" if name.split("/")[0] in FAMILIES else family_fragment
    assert registry.parse_model_name(name, task=own)[0] == name.split("/")[0]


def test_registering_twice_changes_nothing(registered):
    before = dict(registry.MODELS), dict(registry.TASKS)
    from train.api.tasks import register_builtin_tasks

    recipes.register_builtin_models()
    register_builtin_tasks()
    assert (dict(registry.MODELS), dict(registry.TASKS)) == before


TF_FREE_DRIVER = (
    "import sys\n"
    "from train.api import cli\n"
    "codes = [cli.main(['help']), cli.main(['list_tasks']), cli.main(['pretrain', 'help'])]\n"
    "loaded = sorted(m for m in ('tensorflow', 'keras', 'yaml', 'train.common', 'dl_techniques.models') if m in sys.modules)\n"
    "print('CODES', codes)\n"
    "print('LOADED', loaded)\n"
)


def test_help_and_list_tasks_load_no_tensorflow_and_name_the_task(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", TF_FREE_DRIVER], cwd=tmp_path, capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    lines = done.stdout.splitlines()
    assert lines[-2:] == ["CODES [0, 0, 0]", "LOADED []"], done.stdout[-2000:]
    assert "pretrain" in done.stdout
    # The task's own help names its keys, among them the mask keys of the data group.
    assert "data.mask_ratio" in done.stdout and "data.num_mask_patches" in done.stdout


# ---------------------------------------------------------------------
# A request that can be refused without data is refused with no tensor created, through the real command line
# ---------------------------------------------------------------------

#: The tokenizer token every BEiT row carries so that the refusal under test is the one reported (a BEiT run needs one).
#: The child writes an EMPTY ``tokenizer.keras`` into its working directory: the check before the claim reads existence and
#: the suffix only (the content, the grid and the usage are read after the claim, which a spy replaces), so a real tiny
#: tokenizer would change nothing here. The real tokenizers are in ``test_pretrain_task.py``.
TOKENIZER = "model_args.tokenizer_checkpoint=tokenizer.keras"

#: ``(family, dataset, command-line tokens, fragments of the message)``: one refusal per row, each of a different
#: layer (the data group, the optimizer, the model keys, the facts, the task), all of which need no loaded data. The
#: tokens are what ``python -m train pretrain`` is given after ``out=``, ``model=`` and ``data.dataset=``.
PRE_CLAIM_REFUSALS: List[Tuple[str, str, List[str], Tuple[str, ...]]] = [
    ("beit_mim", "cifar10", ["data.num_mask_patches=75", TOKENIZER], ("data.num_mask_patches", "64")),
    ("beit_mim", "cifar10", ["data.mask_ratio=0.5", TOKENIZER], ("data.mask_ratio", "'beit_mim'")),
    ("et_mim", "cifar10", ["data.num_mask_patches=10"], ("data.num_mask_patches", "'et_mim'")),
    ("et_mim", "cifar10", ["data.patch_size=5"], ("data.patch_size",)),
    ("et_mim", "cifar10", ["optim_args.global_clipnorm=null"], ("optim_args.global_clipnorm",)),
    (
        "et_mim", "cifar10", ["optim_args.optimizer=sgd", "optim_args.lr=0.1", "optim_args.weight_decay=0.0001"],
        ("optim_args.weight_decay",),
    ),
    (
        "beit_mim", "imagenette", ["optim_args.optimizer=sgd", "optim_args.lr=0.1", "optim_args.weight_decay=0.0001", TOKENIZER],
        ("optim_args.weight_decay",),
    ),
    (
        "et_mim", "cifar10", ["model_args.tokenizer_checkpoint=tokenizer.keras"],
        ("model_args.tokenizer_checkpoint", "'beit_mim'"),
    ),
    ("beit_mim", "cifar10", ["model_args.num_steps=6", TOKENIZER], ("model_args.num_steps", "'et_mim'")),
    ("beit_mim", "cifar10", ["model_args.vocab_size=16", TOKENIZER], ("model_args.vocab_size",)),
    ("beit_mim", "cifar10", ["transform_args.image_size=30", TOKENIZER], ("data.patch_size",)),
    # A task-level refusal: a conformal set is a set of classes, and a pretrain run has none.
    ("et_mim", "cifar10", ["analysis_args.conformal_alpha=0.1"], ("analysis_args.conformal_alpha",)),
    # A BEiT run with no tokenizer has no targets; one whose path is absent or is not a .keras is refused by name too.
    ("beit_mim", "cifar10", [], ("model_args.tokenizer_checkpoint", "python -m train beit_tokenizer")),
    ("beit_mim", "imagenette", [], ("model_args.tokenizer_checkpoint", "python -m train beit_tokenizer")),
    ("beit_mim", "cifar10", ["model_args.tokenizer_checkpoint=missing.keras"], ("model_args.tokenizer_checkpoint", "missing.keras", "does not exist")),
    ("beit_mim", "cifar10", ["model_args.tokenizer_checkpoint=tokenizer.h5"], ("model_args.tokenizer_checkpoint", "tokenizer.h5", "does not end in .keras")),
]

#: What the same command line accepts: it reaches the claim (the control of the rows above).
PRE_CLAIM_ACCEPTED: List[Tuple[str, str, List[str]]] = [
    ("et_mim", "cifar10", []),
    ("et_mim", "imagenette", ["data.mask_ratio=0.25"]),
    ("beit_mim", "cifar10", ["data.num_mask_patches=64", TOKENIZER]),
    ("beit_mim", "imagenette", [TOKENIZER]),
]

#: ``(model, dataset, tokens)``: the two pretrain names before plan-2026-11-19T081214-5c6cc105 step 3, which renamed the rows
#: with no alias (U1, D-002). Each request is one every other check accepts (the BEiT one carries a tokenizer), so the name
#: is the only thing refused, and an alias would carry the request to the claim.
OLD_NAMES: List[Tuple[str, str, List[str]]] = [
    ("energy_transformer/et_tiny", "cifar10", []),
    ("beit/beit_tiny", "cifar10", [TOKENIZER]),
]
#: What an old name is told: the families that train pretrain, in the sentence of ``registry.parse_model_name``.
PRETRAIN_FAMILIES = "the families of 'pretrain' are beit_mim, et_mim"

CHILD_DRIVER = textwrap.dedent(
    """
    import contextlib
    import io
    import json
    import os
    import sys

    from tensorflow.python.eager import context as eager

    from train.api import cli, data_args, trainer

    # The path test for the prepared imagenette records is the parent's test (`pretrain_facts`, monkeypatched there); the
    # child must not depend on the machine's TFDS directory.
    data_args._check_imagenette_records = lambda: None


    class Claimed(Exception):
        # Raised by the spy that replaces the claim: the request passed every refusal and reached `open_run`.
        pass


    def spy(path, resume):
        raise Claimed(str(path))


    trainer.open_run = spy


    def initialised():
        return eager.context()._context_handle is not None


    rows = json.load(open(sys.argv[1]))
    seen = {"at_start": initialised(), "results": []}
    for name in ("tokenizer.keras", "tokenizer.h5"):
        open(os.path.join(sys.argv[2], name), "wb").close()
    os.chdir(sys.argv[2])
    for index, row in enumerate(rows):
        out = os.path.join(sys.argv[2], f"run_{index}")
        argv = ["pretrain", f"out={out}", f"model={row['model']}", f"data.dataset={row['dataset']}", *row["tokens"]]
        errors = io.StringIO()
        code, claimed = None, False
        try:
            with contextlib.redirect_stderr(errors):
                code = cli.main(argv)
        except Claimed:
            claimed = True
        seen["results"].append(
            {"code": code, "claimed": claimed, "stderr": errors.getvalue(), "out_exists": os.path.exists(out)}
        )
    seen["initialised"] = initialised()
    import keras

    keras.ops.zeros(1)
    seen["after_op"] = initialised()
    print("SEEN " + json.dumps(seen))
    """
)


@pytest.fixture(scope="module")
def pre_claim_run(tmp_path_factory):
    """Every refusal row, every accepted control and every old name through the real command line in ONE child process."""
    work = tmp_path_factory.mktemp("pre_claim")
    tiny = {family: f"{family}/{'et_tiny' if family == 'et_mim' else 'beit_tiny'}" for family in FAMILIES}
    rows = [{"model": tiny[f], "dataset": d, "tokens": t} for f, d, t, _ in PRE_CLAIM_REFUSALS]
    rows += [{"model": tiny[f], "dataset": d, "tokens": t} for f, d, t in PRE_CLAIM_ACCEPTED]
    rows += [{"model": m, "dataset": d, "tokens": t} for m, d, t in OLD_NAMES]
    (work / "rows.json").write_text(json.dumps(rows))
    runs = work / "runs"
    runs.mkdir()
    env = {**os.environ, "MPLBACKEND": "Agg", "TF_CPP_MIN_LOG_LEVEL": "3", "CUDA_VISIBLE_DEVICES": "-1",
           "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run(
        [sys.executable, "-c", CHILD_DRIVER, str(work / "rows.json"), str(runs)],
        capture_output=True, text=True, timeout=900, cwd=work, env=env,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    line = [row for row in done.stdout.splitlines() if row.startswith("SEEN ")][-1]
    return json.loads(line[len("SEEN "):])


@pytest.mark.parametrize(
    "index", range(len(PRE_CLAIM_REFUSALS)),
    ids=[f"{f}:{d}:{'+'.join(t.split('=')[0] for t in tokens) or 'bare'}" for f, d, tokens, _ in PRE_CLAIM_REFUSALS],
)
def test_a_request_refusable_without_data_exits_2_before_the_claim(index, pre_claim_run):
    result = pre_claim_run["results"][index]
    assert result["code"] == 2, f"the command line accepted a request that must be refused: {result}"
    assert not result["claimed"], "the run directory was claimed before the refusal"
    assert not result["out_exists"]
    for fragment in PRE_CLAIM_REFUSALS[index][3]:
        assert fragment in result["stderr"], (fragment, result["stderr"])


@pytest.mark.parametrize("index", range(len(PRE_CLAIM_ACCEPTED)))
def test_the_requests_a_family_offers_reach_the_claim(index, pre_claim_run):
    result = pre_claim_run["results"][len(PRE_CLAIM_REFUSALS) + index]
    assert result["claimed"], f"a request every refusal accepts must reach the claim: {result}"
    assert result["code"] is None and result["stderr"] == "", result
    assert not result["out_exists"], "the spy replaces the claim, which is the first thing that creates `out`"


@pytest.mark.parametrize("index", range(len(OLD_NAMES)), ids=[model for model, _, _ in OLD_NAMES])
def test_an_old_pretrain_name_exits_2_naming_the_pretrain_families_before_the_claim(index, pre_claim_run):
    result = pre_claim_run["results"][len(PRE_CLAIM_REFUSALS) + len(PRE_CLAIM_ACCEPTED) + index]
    assert result["code"] == 2, f"an old pretrain name was accepted (an alias carried it to the claim?): {result}"
    assert not result["claimed"] and not result["out_exists"], "the run directory was claimed before the refusal"
    assert PRETRAIN_FAMILIES in result["stderr"], result["stderr"]


def test_no_refusal_and_no_resolution_created_a_tensor(pre_claim_run):
    # The control arm: the same child runs one op afterwards, and the recorder must see it. A TensorFlow release
    # that moves the attribute turns this red instead of turning the assertion above blind.
    assert pre_claim_run["at_start"] is False
    assert pre_claim_run["initialised"] is False
    assert pre_claim_run["after_op"] is True
