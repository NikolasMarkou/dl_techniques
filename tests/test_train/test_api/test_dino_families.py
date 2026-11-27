"""The ``dino_v1`` family of the ``dino`` task, its groups and its recipe: adapter, tables, resolver, geometry, row.

``train.api.dino_models`` and ``train.api.dino_args`` (plan-2026-11-26T224507-e3381378 step 5) adapt the library ``DINOTrainingModel`` to the
factory call the API makes, hold the refusal tables, the recipe and the data group. Pinned here:

* the knob census: every field of the legacy ``train.dino.train_dino.TrainingConfig`` (read by looking) has ONE class (offered, common to
  every task, fixed by the task, refused, not applicable) and the adapter holds it, in both directions; every offered key reaches the
  factory or the task, every refused key is refused by name with its reason before the run directory exists;
* the tables, held equal to the constructors' signatures by a census in both directions (the denoising pattern);
* the recipe, equal to the legacy ``TrainingConfig`` defaults at the shared fields, every difference listed with its reason, and the teacher EMA
  warmup denominated in EPOCHS (266 steps per epoch is 266 steps, not the 295 of the measured scale);
* the pre-claim refusals that are not keys: a patch that does not tile the image, a half precision, a constant schedule, a mid-epoch backup, a
  watch metric other than the probe's;
* the adapter: the model is built in the factory with the rank-5 input and its grid and view count are PROVED by real forward passes;
* the family row: its task, its variant table (tiny and small only), a transfer declaration, refused under every other task and every other
  family refused under this one, and a fresh process that imports both modules and builds the row imports no TensorFlow.

Everything writes under ``tmp_path``.
"""

import dataclasses
import math
import os
import re
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any, Dict

import keras
import pytest

from dl_techniques.models.vision.dino.dino_v1 import DINOv1
from train.api import data_args, dino_args as da, dino_models as dm, pretrain_args as pa, recipes, registry
from train.api.args import AutoContext, CheckpointArgs, OptimArgs
from train.api.config import AUTO, validate
from train.api.errors import ConfigError, ConfigValidationError, UnknownModelError

from .test_pretrain_families import imagenette_records_present  # noqa: F401  (fixture: the one definition, the pretrain families file's)

REPO_ROOT = Path(__file__).resolve().parents[3]

DATASETS = pa.DATASETS
SPEC = dm.builtin_dino_models()[0]

#: The CPU test size: tiny at 32 px, patch 8 (a 4 x 4 grid), two local crops, a 64-wide head.
TEST_SIZE = dict(variant="tiny", num_classes=None, input_shape=(32, 32, 3), patch_size=8, n_local_crops=2, dino_out_dim=64)


def legacy_config_class() -> Any:
    """The legacy trainer's ``TrainingConfig`` class, read by looking (it imports TensorFlow)."""
    from train.dino import train_dino

    return train_dino.TrainingConfig


def legacy_defaults() -> Dict[str, Any]:
    """``{field: default}`` of the legacy ``TrainingConfig``."""
    return {f.name: f.default for f in dataclasses.fields(legacy_config_class()) if f.default is not dataclasses.MISSING}


def context(dataset: str = "cifar10", variant: str = "tiny", image_size: int = None) -> AutoContext:
    size = da.GEOMETRY[dataset][0] if image_size is None else image_size
    return AutoContext(
        family=da.FAMILY, variant=variant, dataset=dataset, num_classes=None, image_size=size, defaults=SPEC.recipe(dataset, variant)
    )


def resolve(dataset: str = "cifar10", model_args: Dict[str, Any] = None, **optim: Any) -> Dict[str, Any]:
    """The factory keywords the task's ``check_model_args`` gets from the resolver."""
    ctx = context(dataset)
    return SPEC.load_resolver()(ctx, OptimArgs(**optim).resolve_auto(ctx), model_args or {})


def values(dataset: str = "cifar10", model_args: Dict[str, Any] = None) -> Dict[str, Any]:
    """The task-read values (``teacher_temp``, ``teacher_temp_final``, ``ema_warmup_epochs``) of a request."""
    ctx = context(dataset)
    return dm.task_values(ctx, OptimArgs().resolve_auto(ctx), model_args or {})


def data_group(dataset: str = "cifar10", image_size: int = None, **given: Any) -> da.DinoData:
    return da.DinoData(dataset=dataset, **given).resolve_auto(context(dataset, image_size=image_size))


def build(**changes: Any):
    return dm.create_dino_model(**{**TEST_SIZE, **changes})


@pytest.fixture(scope="module")
def test_size_model():
    return build()


# ---------------------------------------------------------------------
# The knob census: every legacy knob has one class and the adapter holds it
# ---------------------------------------------------------------------

#: The class of every field of the legacy ``TrainingConfig`` and where it lives at the front door (plan e3381378 D-031: read off the
#: measurement record `research/2026_dino_ssl_measurements.md` by the rule of D-002, a knob is OFFERED only where the record names a value).
OFFERED = {
    "global_crop_size": ("transform_args", "image_size"), "n_local_crops": ("data", "n_local_crops"), "patch_size": ("data", "patch_size"),
    "dino_out_dim": ("model_args", "dino_out_dim"), "teacher_temp": ("model_args", "teacher_temp"),
    "teacher_temp_final": ("model_args", "teacher_temp_final"), "ema_warmup_epochs": ("model_args", "ema_warmup_epochs"),
    "variant": ("model", "dino_v1/<variant>"),
}
#: A key every task has, at its place in the shared groups or the recipe's top level.
COMMON = {
    "dataset": ("data", "dataset"), "validation_split": ("data", "validation_split"), "max_samples": ("data", "max_samples"),
    "batch_size": ("recipe", "batch_size"), "epochs": ("recipe", "epochs"), "learning_rate": ("optim_args", "lr"),
    "optimizer_type": ("optim_args", "optimizer"), "lr_schedule_type": ("optim_args", "schedule"),
    "warmup_epochs": ("optim_args", "warmup_epochs"), "weight_decay": ("optim_args", "weight_decay"),
    "gradient_clipping": ("optim_args", "global_clipnorm"), "early_stopping_patience": ("checkpoint_args", "early_stopping_patience"),
    "seed": ("task", "seed"),
}
#: Fixed by the task (the record's value, never a knob) and refused by name when a caller names them.
FIXED_BY_THE_TASK = (
    "local_crop_size", "stateless_augmentation", "teacher_temp_warmup_epochs", "knn_eval_every", "knn_bank_batches", "knn_query_batches",
    "knn_temperature",
)
#: Refused: the record names no measured value for them.
REFUSED_KNOBS = ("source_image_size", "student_temp", "center_momentum", "ema_decay_start", "ema_decay_end", "ema_warmup_steps")
#: Not a knob of a run through the front door.
NOT_APPLICABLE = ("random_init_repeats", "max_steps", "output_dir", "experiment_name", "gpu")


def test_every_field_of_the_legacy_config_has_exactly_one_class_and_every_class_names_real_fields():
    fields = [f.name for f in dataclasses.fields(legacy_config_class())]
    classed = [*OFFERED, *COMMON, *FIXED_BY_THE_TASK, *REFUSED_KNOBS, *NOT_APPLICABLE]
    assert sorted(classed) == sorted(fields), (sorted(set(fields) - set(classed)), sorted(set(classed) - set(fields)))
    assert len(set(classed)) == len(classed), "a knob in two classes"
    assert len(fields) == 39


def test_the_offered_knobs_are_keys_the_front_door_has_and_no_other_model_key_is_offered():
    group_fields = {"data": {f.name for f in dataclasses.fields(da.DinoData)},
                    "transform_args": {f.name for f in dataclasses.fields(da.DinoTransformArgs)}}
    for knob, (group, key) in OFFERED.items():
        if group in group_fields:
            assert key in group_fields[group], knob
        elif group == "model_args":
            assert key in dm.ALLOWED, knob
    assert sorted(OFFERED["variant"][1:]) == ["dino_v1/<variant>"] and sorted(dm.VARIANTS) == ["small", "tiny"]
    assert set(dm.ALLOWED) == {key for group, key in OFFERED.values() if group == "model_args"}
    assert set(dm.ALLOWED) == {*dm.FACTORY_KEYS, *dm.TASK_KEYS}


def test_the_common_knobs_are_fields_of_the_shared_groups_or_the_recipe():
    recipe = SPEC.recipe("imagenette", "tiny")
    fields = {"data": {f.name for f in dataclasses.fields(da.DinoData)}, "optim_args": {f.name for f in dataclasses.fields(OptimArgs)},
              "checkpoint_args": {f.name for f in dataclasses.fields(CheckpointArgs)}}
    for knob, (group, key) in COMMON.items():
        if group in fields:
            assert key in fields[group], knob
        elif group == "recipe":
            assert key in recipe, knob
    assert COMMON["seed"] == ("task", "seed")


@pytest.mark.parametrize("knob", [*FIXED_BY_THE_TASK, *REFUSED_KNOBS])
def test_a_fixed_or_refused_knob_is_refused_by_name_with_its_reason_before_the_claim(knob):
    table = {**dm.REFUSED, **dm.SUPPLIED}
    assert knob in table, knob
    with pytest.raises((ConfigValidationError, ConfigError)) as raised:
        resolve(model_args={knob: 1})
    text = str(raised.value)
    assert f"model_args.{knob}" in text and table[knob] in text, text


@pytest.mark.parametrize("knob", NOT_APPLICABLE)
def test_a_knob_that_is_not_applicable_is_in_no_table(knob):
    assert knob not in {*dm.ALLOWED, *dm.REFUSED, *dm.SUPPLIED}, knob


def test_every_offered_model_key_reaches_the_factory_or_the_task():
    assert resolve() == {"dino_out_dim": 4096} and values() == {"teacher_temp": 0.04, "teacher_temp_final": 0.04, "ema_warmup_epochs": 1.0}
    assert resolve(model_args={"dino_out_dim": 64}) == {"dino_out_dim": 64}, "the factory key is passed through"
    changed = values(model_args={"teacher_temp": 0.05, "teacher_temp_final": 0.07, "ema_warmup_epochs": 2.0})
    assert changed == {"teacher_temp": 0.05, "teacher_temp_final": 0.07, "ema_warmup_epochs": 2.0}, "the task keys are read by the task"
    assert resolve(model_args={"teacher_temp": 0.03}) == {"dino_out_dim": 4096}, "and are no model keyword"
    model = build(dino_out_dim=32)
    assert model.out_dim == 32, "dino_out_dim reaches the built model"


# ---------------------------------------------------------------------
# The tables and the signature census
# ---------------------------------------------------------------------


def test_the_census_classifies_every_parameter_of_the_constructors_in_both_directions():
    assert dm.signature_census() == {"unclassified": [], "stale": []}
    signatures = dm.factory_signatures()
    assert set(signatures) == set(dm.FACTORIES) and len(dm.FACTORIES) == 3
    walked = {name for names in signatures.values() for name in names}
    assert {"dino_out_dim", "n_local_views", "embed_dim", "include_projection_head", "student_temp", "teacher_temp"} <= walked


def test_the_census_fails_when_a_constructor_gains_a_parameter_no_table_names_and_when_a_table_key_loses_its_parameter():
    real = dm.factory_signatures()
    grown = {path: [*names, "new_option"] for path, names in real.items()}
    assert dm.signature_census(grown)["unclassified"] == ["new_option"]
    gone = "dino_out_dim"
    shrunk = {path: [name for name in names if name != gone] for path, names in real.items()}
    assert dm.signature_census(shrunk)["stale"] == [gone]


def test_the_keys_no_constructor_has_are_the_trainers_and_the_task_keys_and_no_constructor_walks_them():
    walked = {name for names in dm.factory_signatures().values() for name in names}
    assert not set(dm.UNSEEN_KEYS) & walked, set(dm.UNSEEN_KEYS) & walked
    assert set(dm.UNSEEN_KEYS) <= {*dm.ALLOWED, *dm.REFUSED, *dm.SUPPLIED}


def test_the_classes_of_a_key_do_not_overlap_and_every_reason_is_a_phrase_that_ends_the_sentence():
    tables = [set(dm.ALLOWED), set(dm.REFUSED), set(dm.SUPPLIED)]
    for index, first in enumerate(tables):
        for second in tables[index + 1:]:
            assert not (first & second), first & second
    for table in (dm.REFUSED, dm.SUPPLIED):
        for key, why in table.items():
            assert why and not why.endswith("."), key
    assert {"variant", "input_shape", "patch_size", "image_size", "num_classes", "n_local_views", "n_local_crops", "local_crop_size"} <= set(dm.SUPPLIED)
    assert {"student_temp", "center_momentum", "ema_warmup_steps", "embed_dim", "depth"} <= set(dm.REFUSED)


@pytest.mark.parametrize("key", sorted(dm.REFUSED))
def test_every_refused_key_is_named_with_its_rule_before_the_claim(key):
    with pytest.raises(ConfigValidationError) as raised:
        resolve(model_args={key: 1})
    text = str(raised.value)
    assert f"model_args.{key}" in text and f"is not offered by {da.FAMILY!r}" in text and dm.REFUSED[key] in text, text


@pytest.mark.parametrize("key", sorted(dm.SUPPLIED))
def test_a_key_the_task_supplies_is_refused_by_name(key):
    with pytest.raises((ConfigValidationError, ConfigError)) as raised:
        resolve(model_args={key: 1})
    assert f"model_args.{key}" in str(raised.value) or f"'{key}'" in str(raised.value)


def test_an_unknown_key_is_refused_with_a_suggestion_and_every_refusal_is_named_at_once():
    with pytest.raises(ConfigValidationError, match="Unknown key 'model_args.no_such_key'"):
        resolve(model_args={"no_such_key": 1})
    with pytest.raises(ConfigValidationError, match="Did you mean 'model_args.dino_out_dim'"):
        resolve(model_args={"dino_out_dims": 64})
    with pytest.raises(ConfigValidationError, match="Unknown key 'model_args.encoder'"):
        resolve(model_args={"encoder": None})
    with pytest.raises(ConfigValidationError) as raised:
        resolve(model_args={"student_temp": 0.1, "center_momentum": 0.9, "ema_warmup_steps": 295, "typo_key": 1})
    text = str(raised.value)
    for key in ("student_temp", "center_momentum", "ema_warmup_steps", "typo_key"):
        assert f"model_args.{key}" in text, key


#: Requests that must NOT be refused (the control of the rows above: a resolver that refused everything would pass them).
ACCEPTED = [
    {}, {"dino_out_dim": 2}, {"dino_out_dim": 64}, {"teacher_temp": 0.03}, {"teacher_temp": 0.04, "teacher_temp_final": 0.07},
    {"teacher_temp_final": 0.5}, {"ema_warmup_epochs": 0}, {"ema_warmup_epochs": 0.5}, {"ema_warmup_epochs": 3},
]


@pytest.mark.parametrize("model_args", ACCEPTED, ids=["-".join(sorted(a)) + "-" + "-".join(map(str, a.values())) if a else "recipe" for a in ACCEPTED])
def test_the_keys_the_family_offers_are_not_refused(model_args):
    got = resolve(model_args=model_args)
    assert got == {"dino_out_dim": model_args.get("dino_out_dim", 4096)}


#: Values of an offered key that no run can use: each is refused before the claim, naming the key.
BAD_VALUES = [
    ("dino_out_dim", 0), ("dino_out_dim", 1), ("dino_out_dim", -4), ("dino_out_dim", 64.0), ("dino_out_dim", True), ("dino_out_dim", "64"),
    ("teacher_temp", 0), ("teacher_temp", -0.04), ("teacher_temp", True), ("teacher_temp", "0.04"), ("teacher_temp", math.nan),
    ("teacher_temp_final", 0), ("teacher_temp_final", math.inf), ("teacher_temp_final", False),
    ("ema_warmup_epochs", -1), ("ema_warmup_epochs", math.nan), ("ema_warmup_epochs", math.inf), ("ema_warmup_epochs", True),
    ("ema_warmup_epochs", "1.0"),
]


@pytest.mark.parametrize("key, value", BAD_VALUES, ids=[f"{key}={value!r}" for key, value in BAD_VALUES])
def test_a_value_no_run_can_use_is_refused_by_name_before_the_claim(key, value):
    with pytest.raises(ConfigValidationError) as raised:
        resolve(model_args={key: value})
    assert f"model_args.{key}" in str(raised.value) and repr(value) in str(raised.value)


def test_a_teacher_temperature_that_falls_during_its_schedule_is_refused_naming_both():
    with pytest.raises(ConfigValidationError) as raised:
        resolve(model_args={"teacher_temp": 0.07, "teacher_temp_final": 0.04})
    text = str(raised.value)
    assert "teacher_temp_final" in text and "0.04" in text and "0.07" in text and "teacher_temp" in text
    with pytest.raises(ConfigValidationError, match="teacher_temp_final"):
        resolve(model_args={"teacher_temp": 0.05})  # the recipe's 0.04 ends the schedule, so a start above it falls
    assert resolve(model_args={"teacher_temp": 0.04, "teacher_temp_final": 0.04}) == {"dino_out_dim": 4096}, "equal is a constant schedule: accepted"


@pytest.mark.parametrize("key", ["variant", "num_classes", "input_shape"])
def test_a_key_the_library_supplies_is_refused_before_the_tables_are_read(key):
    with pytest.raises(ConfigError, match="is set by the library"):
        resolve(model_args={key: 1})


def test_an_adamw_run_refuses_an_explicit_kernel_regularizer_and_an_sgd_decay_is_refused_because_no_regularizer_is_built():
    with pytest.raises(ConfigError, match="kernel_regularizer"):
        resolve(model_args={"kernel_regularizer": "l2"})
    with pytest.raises(ConfigError) as raised:
        resolve(optimizer="sgd", lr=0.1, weight_decay=1e-4)
    assert "optim_args.weight_decay" in str(raised.value) and f"the {da.TASK} task builds none for {da.FAMILY!r}" in str(raised.value)
    assert resolve(optimizer="sgd", lr=0.1, weight_decay=None) == {"dino_out_dim": 4096}, "the control: sgd without a decay resolves"
    assert resolve() == {"dino_out_dim": 4096}, "and adamw with its decay"


def test_a_precision_other_than_float32_is_refused_by_name():
    assert dm.precision_problems("float32", "") == []
    for precision in ("mixed_float16", "mixed_bfloat16"):
        (line,) = dm.precision_problems(precision, "")
        assert "'precision'" in line and "float32" in line and repr(precision) in line and "centre" in line, line
    assert "'run.precision'" in dm.precision_problems("mixed_float16", "run")[0], "the prefix is the caller's dotted path"


def test_a_constant_schedule_is_refused_by_name_because_it_adds_a_plateau_callback_that_watches_a_metric_no_dino_run_has():
    assert dm.schedule_problems("cosine_decay") == []
    (line,) = dm.schedule_problems("constant")
    assert "'optim_args.schedule'" in line and "ReduceLROnPlateau" in line and "val_loss" in line and "'constant'" in line, line
    assert "'run.optim.schedule'" in dm.schedule_problems("constant", "run.optim")[0]


def test_a_mid_epoch_backup_is_refused_by_name_because_it_would_pair_an_old_centre_with_newer_weights():
    assert dm.backup_problems(None) == []
    for every in (2, 100, 100000):
        (line,) = dm.backup_problems(every)
        assert "'checkpoint_args.backup_every_steps'" in line and "centre" in line and repr(every) in line, line
    assert "'run.ck.backup_every_steps'" in dm.backup_problems(5, "run.ck")[0]


# ---------------------------------------------------------------------
# The recipe equals the legacy trainer's defaults at the shared fields
# ---------------------------------------------------------------------

#: ``legacy field -> where the recipe holds it``: every field the recipe shares with the legacy trainer.
SHARED = {
    "learning_rate": lambda r: r["optim_args"]["lr"], "optimizer_type": lambda r: r["optim_args"]["optimizer"],
    "lr_schedule_type": lambda r: r["optim_args"]["schedule"], "warmup_epochs": lambda r: r["optim_args"]["warmup_epochs"],
    "weight_decay": lambda r: r["optim_args"]["weight_decay"], "gradient_clipping": lambda r: r["optim_args"]["global_clipnorm"],
    "batch_size": lambda r: r["batch_size"], "n_local_crops": lambda r: r["data"]["n_local_crops"],
    "teacher_temp": lambda r: r["model_args"]["teacher_temp"], "teacher_temp_final": lambda r: r["model_args"]["teacher_temp_final"],
    "ema_warmup_epochs": lambda r: r["model_args"]["ema_warmup_epochs"],
}
#: The fields the recipe changes on purpose, ``legacy field -> (legacy default, recipe value, where the recipe holds it, the record's line)``.
DIFFERENT = {
    "epochs": (100, 60, lambda r: r["epochs"], "the verdict arms of the record ran 60 epochs"),
    "early_stopping_patience": (30, 70, lambda r: r["checkpoint_args"]["early_stopping_patience"], "70 so the 60-epoch horizon runs to its end"),
    "dino_out_dim": (65536, 4096, lambda r: r["model_args"]["dino_out_dim"], "the record's scale; 65536 does not fit 12 GB"),
}
#: The legacy defaults of the values the task fixes (the record names none of them, so they are the legacy ones) and its two probe sizes.
FIXED_SAME = ("student_temp", "center_momentum", "ema_decay_start", "ema_decay_end", "teacher_temp_warmup_epochs")
FIXED_DIFFERENT = {"knn_bank_batches": (16, 64), "knn_query_batches": (8, 32)}


def test_the_recipe_equals_the_legacy_defaults_at_every_shared_field():
    legacy = legacy_defaults()
    for dataset in DATASETS:
        recipe = SPEC.recipe(dataset, "tiny")
        for field, where in SHARED.items():
            assert where(recipe) == legacy[field], (dataset, field, where(recipe), legacy[field])
    # The legacy decay is `build_optimizer`'s: cosine over ALL the epochs (warmup inside them), alpha 0.01, a GLOBAL clip.
    recipe = SPEC.recipe("imagenette", "tiny")
    assert recipe["optim_args"]["decay_over"] == "total" and recipe["optim_args"]["min_lr_ratio"] == 0.01
    assert recipe["optim_args"]["clipnorm"] is None
    assert recipe["metric_args"] == {"watch_metric": da.WATCH_METRIC}
    assert da.DinoData(dataset="cifar10").validation_split is legacy["validation_split"] is None
    assert da.DinoData(dataset="cifar10").max_samples is legacy["max_samples"] is None


def test_every_difference_from_the_legacy_defaults_is_listed_with_its_value_and_its_reason():
    legacy = legacy_defaults()
    for dataset in DATASETS:
        recipe = SPEC.recipe(dataset, "tiny")
        for field, (was, now, where, reason) in DIFFERENT.items():
            assert legacy[field] == was and where(recipe) == now and was != now and reason, (dataset, field)
    assert legacy["global_crop_size"] == 224 and SPEC.recipe("imagenette", "tiny")["transform_args"]["image_size"] == 96
    assert da.GEOMETRY["imagenette"] == (96, 16) and da.GEOMETRY["cifar10"] == (32, 4)
    # Every offered and common knob is either shared with the legacy trainer, listed as a difference above, or a key with no default to compare.
    listed = {*SHARED, *DIFFERENT, "global_crop_size", "patch_size", "variant", "dataset", "validation_split", "max_samples", "seed"}
    classed = {*OFFERED, *COMMON}
    assert classed - listed == set(), classed - listed


def test_the_fixed_values_are_the_legacy_defaults_and_the_probe_is_the_records_estimator():
    legacy = legacy_defaults()
    assert set(dm.FIXED) == {*FIXED_SAME, *FIXED_DIFFERENT}
    for field in FIXED_SAME:
        assert dm.FIXED[field] == legacy[field], field
    for field, (was, now) in FIXED_DIFFERENT.items():
        assert legacy[field] == was and dm.FIXED[field] == now and was != now, field
    for field in dm.FIXED:
        assert field in {*FIXED_BY_THE_TASK, *REFUSED_KNOBS}, field


def test_the_recipes_geometry_is_the_one_table_per_dataset():
    for dataset, (size, patch) in da.GEOMETRY.items():
        recipe = SPEC.recipe(dataset, "tiny")
        assert (recipe["transform_args"]["image_size"], recipe["data"]["patch_size"]) == (size, patch)
        assert recipe["data"]["n_local_crops"] == 4
    assert set(da.GEOMETRY) == set(DATASETS), "one geometry row per dataset the shared streams serve"
    assert dm.RECIPE["epochs"] == dm.EPOCHS == 60 and dm.RECIPE["batch_size"] == dm.BATCH_SIZE == 32


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_auto_field_of_every_group_is_listed_by_the_recipe_or_derived(dataset):
    recipe = SPEC.recipe(dataset, "tiny")
    assert set(recipe) <= set(recipes.RECIPE_KEYS), set(recipe) - set(recipes.RECIPE_KEYS)
    auto = lambda cls: {f.name for f in dataclasses.fields(cls) if f.default == AUTO}
    assert set(recipe["optim_args"]) == auto(OptimArgs)
    assert set(recipe["checkpoint_args"]) == auto(CheckpointArgs)
    assert set(recipe["metric_args"]) == auto(da.DinoMetricArgs) - {"watch_mode"}
    assert set(recipe["transform_args"]) == auto(da.DinoTransformArgs) == {"image_size"}
    assert set(recipe["data"]) == auto(da.DinoData)
    for group in (OptimArgs(), CheckpointArgs(), da.DinoData(dataset=dataset), da.DinoTransformArgs(), da.DinoMetricArgs()):
        resolved = group.resolve_auto(context(dataset))
        for field in dataclasses.fields(resolved):
            assert getattr(resolved, field.name) != AUTO, (type(group).__name__, field.name)


def test_a_recipe_key_that_is_no_field_is_named():
    ctx = context()
    broken = {**ctx.defaults, "data": {**ctx.defaults["data"], "patch_sizes": 4}}
    with pytest.raises(ConfigValidationError, match="patch_sizes"):
        da.DinoData(dataset="cifar10").resolve_auto(dataclasses.replace(ctx, defaults=broken))


def test_the_teacher_ema_warmup_is_denominated_in_epochs_so_266_steps_an_epoch_is_266_steps_and_not_295():
    # 266 = 8523 fit rows // 32 at HEAD (scratch_step5/measured.md; the 295 of the record is the older protocol's 9,440 // 32).
    assert dm.ema_warmup_steps(1.0, 266) == 266 != 295
    assert dm.ema_warmup_steps(1.0, 295) == 295, "the same epoch at the record's scale is the record's number"
    assert dm.ema_warmup_steps(0.5, 266) == 133 and dm.ema_warmup_steps(0, 266) == 0
    assert values()["ema_warmup_epochs"] == 1.0, "the recipe holds epochs, not a step count"
    config_class = legacy_config_class()
    for epochs in (0.0, 0.5, 1.0, 2.25):
        for steps in (1, 7, 266, 295, 1406):
            from train.dino.train_dino import resolve_ema_warmup_steps

            legacy = resolve_ema_warmup_steps(config_class(ema_warmup_epochs=epochs, experiment_name="x"), steps)
            assert dm.ema_warmup_steps(epochs, steps) == legacy, (epochs, steps)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_recipe_notice_states_what_the_gate_covers_and_no_figure_of_it(dataset):
    """plan e3381378 D-015: a sentence that says 'unmeasured' goes false the night the gate runs, and one that quotes the gate goes stale at the next."""
    notices = dm.recipe_notices_of(dataset)
    assert notices and all(level == "info" and text for level, text in notices)
    text = " ".join(text for _, text in notices)
    for stale in ("unmeasured", "first measurement", "no run in this repository"):
        assert stale not in text.lower(), stale
    assert "imagenette" in text and "tiny" in text and "gate" in text and "cifar10" in text and "small" in text, "what the gate covers and what it does not"
    assert dm.GATE_BLOCK in text and "src/train/README.md" in text and "dino-gate" in text, "where the gate's numbers are"
    optim = SPEC.defaults["optim_args"]
    for words in (f"lr {optim['lr']:g}", f"batch {SPEC.defaults['batch_size']}", f"weight decay {optim['weight_decay']:g}",
                  f"{optim['warmup_epochs']} warmup epochs", f"gradient clip of {optim['global_clipnorm']:g}",
                  f"patience {dm.PATIENCE}", f"{dm.EPOCHS} epochs"):
        assert words in text, words
    # Every number of the notice is one of the RECIPE's or the geometry's: none is the gate's.
    out_dim = SPEC.defaults["model_args"]["dino_out_dim"]
    own = {f"{optim['lr']:g}", str(SPEC.defaults["batch_size"]), f"{optim['weight_decay']:g}", str(optim["warmup_epochs"]),
           f"{optim['global_clipnorm']:g}", str(dm.PATIENCE), str(dm.EPOCHS), str(SPEC.defaults["data"]["n_local_crops"]),
           str(out_dim), f"{out_dim:,}"}
    own |= {str(side) for size, patch in da.GEOMETRY.values() for side in (size, patch, size // patch)}
    numbers = set(re.findall(r"\d[\d,.]*\d|\d", text.replace(dm.GATE_BLOCK, "").replace(da.FAMILY, "")))
    assert numbers <= own, f"a number that is not the recipe's or the geometry's: {sorted(numbers - own)}"


def test_the_cifar10_notice_says_no_dino_run_has_trained_it_and_the_imagenette_notice_does_not():
    cifar = " ".join(text for _, text in dm.recipe_notices_of("cifar10"))
    big = " ".join(text for _, text in dm.recipe_notices_of("imagenette"))
    assert "cifar10" in cifar and "was not trained by the gate" in cifar
    assert "was not trained by the gate" not in big


# ---------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dataset, total", [("imagenette", 23_316_864), ("cifar10", 23_051_136)])
def test_the_default_model_has_the_parameter_count_of_the_records_scale_and_half_of_it_trains(dataset, total):
    # 23,316,864 is the record's smoke model (tiny, 96 px, patch 16, out 4096), READ OFF the built models by scratch_step5/measure.py at this HEAD.
    size, patch = da.GEOMETRY[dataset]
    model = recipes.build_model(SPEC, "tiny", None, (size, size, 3), {**resolve(dataset), "patch_size": patch, "n_local_crops": 4})
    weights = sum(int(math.prod(w.shape)) for w in model.weights)
    trainable = sum(int(math.prod(w.shape)) for w in model.trainable_weights)
    assert weights == total and trainable * 2 == total, "the teacher is the other half and never trains"
    assert model.n_views == 6 and model.n_pairs == 10 and model.out_dim == 4096
    assert model.student.num_patches == (size // patch) ** 2


def test_the_model_at_the_test_size_is_built_in_the_factory_with_the_rank_5_input_and_has_the_grid_and_the_views_the_record_states(test_size_model):
    model = test_size_model
    assert model.built, "the sequence's own `build((None, *input_shape))` is refused by a rank-5 model, so the factory builds it"
    assert (model.n_views, model.n_pairs, model.out_dim, model.student.num_patches) == (4, 6, 64, 16) == (2 + 2, 2 * 4 - 2, 64, (32 // 8) ** 2)
    assert tuple(model.student.input_shape[1:]) == (32, 32, 3)
    packed = keras.ops.convert_to_numpy(model(keras.ops.zeros((1, 4, 32, 32, 3)), training=False))
    assert packed.shape == (6, 128), "the packed (batch * n_pairs, 2 * out_dim) output"
    assert [w.name for w in model.teacher.weights] and model.teacher.trainable is False


def test_the_sequences_own_rank_4_build_call_is_refused_by_the_model_which_is_why_the_factory_builds_it():
    from dl_techniques.models.vision.dino.training import create_dino_training_model

    unbuilt = create_dino_training_model("tiny", image_size=32, patch_size=8, n_local_views=2, dino_out_dim=64)
    with pytest.raises(ValueError, match="rank-5"):
        unbuilt.build((None, 32, 32, 3))
    assert not unbuilt.built


def test_the_small_variant_builds_with_its_own_width():
    model = build(variant="small")
    assert model.student.embed_dim == 384 and model.out_dim == 64


def test_the_variant_table_is_tiny_and_small_of_the_libraries_own_table():
    assert sorted(dm.VARIANTS) == ["small", "tiny"]
    for name, row in dm.VARIANTS.items():
        assert row == {key: DINOv1.MODEL_VARIANTS[name][key] for key in dm.TABLE_KEYS}, name
    assert dm.TABLE_KEYS == ("embed_dim", "depth", "num_heads", "mlp_ratio")


def test_a_grid_that_is_not_the_image_over_the_patch_is_refused_by_the_proof(monkeypatch):
    # The proof is a REAL forward pass through the patch layer compared with `(size // patch) ** 2`, never the model's own `num_patches` (which
    # agrees with itself by construction): a library that built the student at another patch is refused, naming both token counts.
    from dl_techniques.models.vision.dino import training

    real = training.create_dino_training_model
    monkeypatch.setattr(training, "create_dino_training_model", lambda variant, **kw: real(variant, **{**kw, "patch_size": kw["patch_size"] * 2}))
    with pytest.raises(ValueError, match=r"into 4 tokens.*gives 16"):
        build()


def test_a_view_count_that_is_not_two_plus_the_local_crops_is_refused_by_the_proof(monkeypatch):
    from dl_techniques.models.vision.dino import training

    real = training.create_dino_training_model
    monkeypatch.setattr(training, "create_dino_training_model", lambda variant, **kw: real(variant, **{**kw, "n_local_views": kw["n_local_views"] + 1}))
    with pytest.raises(ValueError, match=r"views"):
        build()


@pytest.mark.parametrize("shape", [(32, 32), (32, 32, 3, 1), (0, 32, 3), (32, 32, 0), (32.0, 32, 3), (32, None, 3), (None, None, 3), (32, 16, 3), (32, 32, 1)])
def test_a_malformed_or_non_square_or_non_rgb_input_shape_is_refused(shape):
    with pytest.raises(ValueError, match="input_shape"):
        build(input_shape=shape)


@pytest.mark.parametrize("patch", [0, -4, 3, 5, 6, 8.0, True, 64, 32, 16])
def test_a_patch_that_cannot_tile_the_image_is_refused_by_name(patch):
    with pytest.raises(ValueError, match="patch_size"):
        build(patch_size=patch)


@pytest.mark.parametrize("crops", [0, -1, 2.0, True, None, "4"])
def test_a_local_crop_count_that_is_not_a_positive_int_is_refused(crops):
    with pytest.raises(ValueError, match="n_local_crops"):
        build(n_local_crops=crops)


def test_a_class_count_a_task_key_an_unknown_variant_and_an_unknown_key_are_refused_by_name():
    with pytest.raises(ValueError, match="classes"):
        build(num_classes=10)
    with pytest.raises(ValueError, match="variant"):
        build(variant="base")
    for variant in ("large", "giant", "vit_tiny", ""):
        with pytest.raises(ValueError, match="variant"):
            build(variant=variant)
    with pytest.raises(ValueError, match="teacher_temp.*read by the task"):
        build(teacher_temp=0.04)
    with pytest.raises(ValueError, match="student_temp"):
        build(student_temp=0.1)
    with pytest.raises(ValueError, match="dino_out_dim"):
        build(dino_out_dim=1)


def test_the_task_keywords_are_required_not_defaulted():
    for missing in ("patch_size", "n_local_crops", "dino_out_dim"):
        kwargs = {key: value for key, value in TEST_SIZE.items() if key != missing}
        with pytest.raises(TypeError, match=missing):
            dm.create_dino_model(**kwargs)


def test_observe_reads_the_students_table_keys_and_a_model_of_another_class_has_none(test_size_model):
    assert dm.observe_dino(test_size_model) == dm.VARIANTS["tiny"]
    assert dm.observe_dino(build(variant="small")) == dm.VARIANTS["small"]
    assert dm.observe_dino(keras.Sequential([keras.layers.Dense(2)])) is None
    assert SPEC.load_observer() is dm.observe_dino


# ---------------------------------------------------------------------
# The data group, the geometry and the metric group
# ---------------------------------------------------------------------


def test_the_dataset_literal_is_the_shared_loaders_two_datasets():
    (annotation,) = [f.type for f in dataclasses.fields(da.DinoData) if f.name == "dataset"]
    assert set(typing.get_args(annotation)) == set(DATASETS) == {"cifar10", "imagenette"}


@pytest.mark.parametrize("dataset, patch", [("cifar10", 4), ("imagenette", 16)])
def test_the_patch_and_the_local_crops_are_the_recipes_unless_given(dataset, patch):
    data = data_group(dataset)
    assert data.patch_size == patch == da.GEOMETRY[dataset][1] and data.n_local_crops == 4 and data.format == "builtin"
    assert data_group(dataset, patch_size=2).patch_size == 2, "an explicit patch is kept"
    assert data_group(dataset, n_local_crops=2).n_local_crops == 2


def test_the_help_of_the_patch_states_the_geometry_table_it_is_derived_from():
    (field,) = [f for f in dataclasses.fields(da.DinoData) if f.name == "patch_size"]
    for name, (size, patch) in da.GEOMETRY.items():
        assert f"{patch} for {name} ({size} px, grid {size // patch} x {size // patch})" in field.metadata["help"], name


def test_a_recipe_with_no_patch_or_no_crops_for_the_dataset_is_named():
    ctx = context()
    for key in ("patch_size", "n_local_crops"):
        data = {name: value for name, value in ctx.defaults["data"].items() if name != key}
        with pytest.raises(ConfigError, match=f"data.{key}"):
            da.DinoData(dataset="cifar10").resolve_auto(dataclasses.replace(ctx, defaults={**ctx.defaults, "data": data}))


def test_the_transform_group_has_the_crop_size_and_no_augment_key_that_nothing_would_read():
    assert [f.name for f in dataclasses.fields(da.DinoTransformArgs)] == ["image_size"]
    with pytest.raises(ConfigValidationError, match="augment"):
        validate(da.DinoTransformArgs, {"augment": True})
    (field,) = dataclasses.fields(da.DinoTransformArgs)
    for name, (size, _) in da.GEOMETRY.items():
        assert f"{size} for {name}" in field.metadata["help"], "the help states the 96 px of imagenette, not the pretrain group's 224"
    assert "224" not in field.metadata["help"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_image_size_is_the_datasets_own_unless_given_and_a_size_below_one_is_named(dataset):
    assert da.DinoTransformArgs().resolve_auto(context(dataset)).image_size == da.GEOMETRY[dataset][0]
    assert da.DinoTransformArgs(image_size=64).resolve_auto(context(dataset)).image_size == 64, "an explicit size is kept"
    assert da.DinoTransformArgs().problems("transform_args") == []
    (line,) = da.DinoTransformArgs(image_size=0).problems("transform_args")
    assert "'transform_args.image_size'" in line and "0" in line
    broken = dataclasses.replace(context(dataset), image_size=None)
    with pytest.raises(ConfigError, match="transform_args.image_size"):
        da.DinoTransformArgs().resolve_auto(broken)


def test_the_group_problems_name_a_value_out_of_range_at_its_dotted_path():
    lines = da.DinoData(dataset="cifar10", validation_split=1.5, max_samples=0, patch_size=0, n_local_crops=0).problems("data")
    text = "\n".join(lines)
    for name in ("validation_split", "max_samples", "patch_size", "n_local_crops"):
        assert f"'data.{name}'" in text, name
    assert da.DinoData(dataset="cifar10").problems("data") == []
    assert da.DinoData(dataset="cifar10", n_local_crops=1).problems("data") == [], "one local crop is the least"


def test_a_dataset_the_loader_does_not_serve_is_refused():
    for name in ("cifar100", "imagenet", "mnist", "folder"):
        with pytest.raises(ConfigValidationError, match="dataset"):
            da.load_dino_data_config({"dataset": name})


def test_the_data_config_loads_from_a_mapping_an_instance_and_a_yaml_file(tmp_path):
    data = da.load_dino_data_config({"dataset": "cifar10", "max_samples": 64})
    assert isinstance(data, da.DinoData) and data.max_samples == 64 and data.format == "builtin"
    assert da.load_dino_data_config(data) is data
    path = tmp_path / "data.yaml"
    path.write_text("dataset: imagenette\nvalidation_split: 0.2\n")
    from_file = da.load_dino_data_config(str(path))
    assert (from_file.dataset, from_file.validation_split) == ("imagenette", 0.2)
    path.write_text("dataset: cifar10\nbogus: 1\n")
    with pytest.raises(ConfigValidationError, match="bogus") as raised:
        da.load_dino_data_config(str(path))
    assert str(path) in str(raised.value)
    with pytest.raises(ConfigError, match="mapping"):
        da.load_dino_data_config(3)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_facts_are_the_datasets_geometry_and_no_classes(dataset, imagenette_records_present):
    facts = da.dino_facts(da.DinoData(dataset=dataset), None)
    assert facts == data_args.DataFacts(dataset=dataset, num_classes=None, image_size=da.GEOMETRY[dataset][0])
    assert da.dino_facts(da.DinoData(dataset=dataset), 64).image_size == 64


def test_imagenette_without_its_records_is_refused_before_any_data_is_read(monkeypatch, tmp_path):
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "no_tfds"))
    with pytest.raises(ConfigError, match="imagenette"):
        da.dino_facts(da.DinoData(dataset="imagenette"), None)
    assert da.dino_facts(da.DinoData(dataset="cifar10"), None).image_size == 32, "cifar10 has no records to look for"


@pytest.mark.parametrize("patch, size", [(4, 32), (8, 32), (2, 32), (16, 96), (32, 96), (16, 224), (2, 6)])
def test_a_patch_that_is_a_power_of_two_dividing_the_image_into_at_least_5_tokens_is_accepted(patch, size):
    assert da.geometry_problem(patch, size) is None
    da.check_geometry(patch, size)


@pytest.mark.parametrize("patch, size, words", [
    (3, 96, "power of 2"), (12, 96, "power of 2"), (6, 32, "power of 2"), (0, 32, "positive int"), (-4, 32, "positive int"),
    (4.0, 32, "positive int"), (True, 32, "positive int"), ("4", 32, "positive int"), (64, 96, "divide"), (16, 100, "divide"),
    (16, 32, "tokens"), (32, 32, "tokens"), (4, 8, "tokens"), (8, 8, "tokens"),
])
def test_a_patch_that_cannot_tile_the_image_or_leaves_it_fewer_than_5_tokens_is_refused_naming_both(patch, size, words):
    assert words in da.geometry_problem(patch, size), (patch, size)
    with pytest.raises(ConfigError) as raised:
        da.check_geometry(patch, size)
    text = str(raised.value)
    assert "'data.patch_size'" in text and "transform_args.image_size" in text and repr(size) in text and words in text, text


def test_the_legacy_default_patch_of_16_on_cifar10_is_refused_for_its_four_tokens():
    # The record: cifar10 at 32 px with the library's patch 16 is a 2 x 2 grid, 4 tokens; the recipe's row is patch 4.
    assert "4 tokens" in da.geometry_problem(16, 32) and da.MIN_TOKENS == 5
    for dataset, (size, patch) in da.GEOMETRY.items():
        assert da.geometry_problem(patch, size) is None, dataset


def test_the_metric_group_watches_the_k20_probe_and_maximises_it():
    from train.common.callbacks import resolve_monitor_mode
    from train.dino.knn_eval import DEFAULT_KS

    assert da.WATCH_METRIC == "dino_knn_top1_k20" and da.WATCH_MODE == "max"
    assert da.WATCH_METRIC == f"dino_knn_top1_k{max(DEFAULT_KS)}", "the key the probe writes into every epoch's logs"
    assert resolve_monitor_mode(da.WATCH_METRIC) == da.WATCH_MODE
    resolved = da.DinoMetricArgs().resolve_auto(context())
    assert (resolved.watch_metric, resolved.watch_mode) == (da.WATCH_METRIC, "max")
    assert da.DinoMetricArgs(watch_metric=da.WATCH_METRIC, watch_mode="max").problems("metric_args") == []


@pytest.mark.parametrize(
    "given, key",
    [({"watch_metric": "val_loss"}, "watch_metric"), ({"watch_metric": "dino_knn_top1_k10"}, "watch_metric"),
     ({"watch_metric": "loss"}, "watch_metric"), ({"watch_mode": "min"}, "watch_mode")],
)
def test_the_metric_group_refuses_what_the_bar_cannot_describe(given, key):
    lines = da.DinoMetricArgs(**given).problems("metric_args")
    assert len(lines) == 1 and f"'metric_args.{key}'" in lines[0] and repr(given[key]) in lines[0], lines
    assert "checkpoint" in lines[0] and da.WATCH_METRIC in lines[0]


# ---------------------------------------------------------------------
# The row, its task and the import rule
# ---------------------------------------------------------------------


@pytest.fixture
def registered(monkeypatch):
    """An empty registry holding every built-in family, this one included.

    ``recipes.register_builtin_models`` registers the row WITH its task (step 6, D-043: a family before its task makes ``list_models`` list a
    family no command runs, and the count sentences of the documents and the init census false), so until then the row is added here; once
    the call is there the condition is false and the fixture is the plain registry.
    """
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "TASKS", {})
    recipes.register_builtin_models()
    if da.FAMILY not in registry.MODELS:
        registry.register_model(SPEC)
    return registry


def test_the_row_is_one_family_two_variants_and_declares_its_transfer():
    (row,) = dm.builtin_dino_models()
    assert (row.family, row.tasks, da.TASK, dm.TASK) == (da.FAMILY, (da.TASK,), "dino", da.TASK)
    assert da.FAMILY == "dino_v1", "DINOv1: the only variant with a projection head"
    assert sorted(row.load_variants()) == ["small", "tiny"] and row.load_factory() is dm.create_dino_model
    assert row.load_resolver() is dm.resolve_dino_args and row.defaults is dm.RECIPE
    # The init census wants a declaration of every row; a dino run is no `init` target, so the declaration is empty: no head, no fixed key, no
    # source of another task (step 8 adds the `vit` row's source, which lives on the target's row).
    assert row.transfer == registry.TransferSpec()


def test_the_family_trains_this_task_only_and_every_other_family_is_refused_under_it(registered):
    assert registry.parse_model_name(f"{da.FAMILY}/tiny", task=da.TASK) == (da.FAMILY, "tiny")
    assert registry.parse_model_name(f"{da.FAMILY}/small", task=da.TASK) == (da.FAMILY, "small")
    others = sorted(set(registry.MODELS) - {da.FAMILY})
    assert len(others) >= 20, "the control: the other families are really registered"
    for family in others:
        with pytest.raises(ConfigError, match=f"the family '{family}' trains"):
            registry.parse_model_name(f"{family}/anything", task=da.TASK)


def test_the_family_is_refused_under_every_other_task(registered):
    tasks = sorted({task for spec in registry.MODELS.values() for task in spec.tasks} - {da.TASK})
    assert len(tasks) >= 9, tasks  # the control: the other tasks are really registered
    for task in tasks:
        with pytest.raises(ConfigError) as raised:
            registry.parse_model_name(f"{da.FAMILY}/tiny", task=task)
        assert f"trains {da.TASK}" in str(raised.value) and repr(task) in str(raised.value)


@pytest.mark.parametrize("variant", ["base", "large", "giant", "small_", "vit_tiny"])
def test_a_variant_the_library_has_and_the_family_does_not_offer_is_refused_by_absence(registered, variant):
    with pytest.raises(UnknownModelError, match="variant"):
        registry.parse_model_name(f"{da.FAMILY}/{variant}", task=da.TASK)
    assert [n for n in registry.list_models() if n.startswith(f"{da.FAMILY}/")] == [f"{da.FAMILY}/small", f"{da.FAMILY}/tiny"]


def test_importing_the_modules_and_building_the_row_imports_no_model_code():
    code = (
        "import sys\n"
        "from train.api import dino_args, dino_models\n"
        "specs = dino_models.builtin_dino_models()\n"
        "assert [s.family for s in specs] == ['dino_v1']\n"
        "assert specs[0].recipe('cifar10', 'tiny')['data'] == {'patch_size': 4, 'n_local_crops': 4}\n"
        "assert sorted(specs[0].load_variants()) == ['small', 'tiny'] and dino_models.recipe_notices_of('imagenette')\n"
        "assert dino_args.geometry_problem(16, 96) is None and dino_models.problems_of({'dino_out_dim': 64}) == []\n"
        "loaded = sorted(m for m in ('keras', 'tensorflow', 'dl_techniques', 'yaml', 'train.common') if m in sys.modules)\n"
        "print('LOADED', loaded)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "LOADED []"


