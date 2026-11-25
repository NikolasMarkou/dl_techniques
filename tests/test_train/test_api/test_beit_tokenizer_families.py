"""The ``vq_vae_rotation`` family of the ``beit_tokenizer`` task, its groups and its recipe: adapter, tables, resolver, geometry, row.

``train.api.tokenizer_models`` and ``train.api.tokenizer_args`` (plan-2026-11-25T124419-fa94f59c step 5) adapt the library
``VQVAERotationTrick`` to the factory call the API makes, hold the refusal tables, the recipe and the data group. Pinned here:

* the adapter: the model is built at the run's size with ``downsample_factor`` = ``data.patch_size`` and the code grid is PROVED by a
  real forward pass (a grid that is not size over patch is refused), ``num_classes`` is refused, the default model is the gate's small
  model (the parameter count is read off the built model and pinned, with the grid);
* the tables, held equal to the constructor's signature by a census in both directions (the denoising pattern), every refused key
  refused by name with its reason before the run directory exists, a key of the library or of the task refused as such;
* the recipe, equal to the legacy ``train.beit.train_tokenizer.TrainingConfig`` defaults READ FROM that module (never typed here),
  with the two numbers the CIFAR-10 row changes on purpose (patch 4 and 512 codes, the pairing ``beit_mim`` reads) stated and held;
* the data group (geometry from the shared table), the metric group (``val_reconstruction_loss`` only), the geometry and codebook
  checks (a patch of 3, 5 or 6, a patch that does not divide, a codebook under the floor) and the precision refusal;
* the family row: its task, its variant table, a transfer declaration, refused under every other task and every other family refused
  under this one, and a fresh process that imports both modules and builds the row imports no TensorFlow.

The row is registered with its task by ``recipes.register_builtin_models`` (step 6, D-031); the ``registered`` fixture below is the registry after
that call.

Everything writes under ``tmp_path``.
"""

import dataclasses
import math
import os
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any, Dict

import keras
import pytest

from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick
from train.api import data_args, pretrain_args as pa, recipes, registry, tokenizer_args as ta, tokenizer_models as tm
from train.api.args import AutoContext, CheckpointArgs, OptimArgs
from train.api.config import AUTO
from train.api.errors import ConfigError, ConfigValidationError, UnknownModelError
from train.common.pretrain_run import MIN_TOKENIZER_PERPLEXITY as FLOOR

REPO_ROOT = Path(__file__).resolve().parents[3]

DATASETS = pa.DATASETS
SPEC = tm.builtin_tokenizer_models()[0]

#: A tiny model of the family (a few thousand weights): the widths and the codebook of the CPU tests of the task.
TINY = {"num_embeddings": 16, "embedding_dim": 4, "hidden_channels": 8, "num_res_blocks": 1}


def legacy_defaults() -> Dict[str, Any]:
    """``{field: default}`` of the legacy trainer's ``TrainingConfig``, read by looking (it imports TensorFlow)."""
    from train.beit import train_tokenizer

    return {f.name: f.default for f in dataclasses.fields(train_tokenizer.TrainingConfig) if f.default is not dataclasses.MISSING}


def legacy_command_line(dataset: str) -> Any:
    """The legacy trainer's ``TrainingConfig`` for ``--dataset <dataset>``: what its command line resolves."""
    from train.beit import train_tokenizer

    return train_tokenizer.config_from_args(train_tokenizer.parse_arguments(["--dataset", dataset]))


def context(dataset: str = "cifar10", variant: str = "small", image_size: int = None) -> AutoContext:
    size = pa.GEOMETRY[dataset][0] if image_size is None else image_size
    return AutoContext(
        family=ta.FAMILY, variant=variant, dataset=dataset, num_classes=None, image_size=size, defaults=SPEC.recipe(dataset, variant)
    )


def resolve(dataset: str = "cifar10", model_args: Dict[str, Any] = None, **optim: Any) -> Dict[str, Any]:
    """The factory keywords the task's ``check_model_args`` gets from the resolver."""
    ctx = context(dataset)
    return SPEC.load_resolver()(ctx, OptimArgs(**optim).resolve_auto(ctx), model_args or {})


def data_group(dataset: str = "cifar10", image_size: int = None, **given: Any) -> ta.TokenizerData:
    return ta.TokenizerData(dataset=dataset, **given).resolve_auto(context(dataset, image_size=image_size))


def build(patch: int = 4, shape: Any = (32, 32, 3), **model_args: Any):
    return tm.create_vq_tokenizer(variant="small", num_classes=None, input_shape=shape, patch_size=patch, **{**TINY, **model_args})


def check(patch=4, size=32, codes=512, floor=FLOOR, smoke=False):
    ta.check_geometry(patch, size, codes, floor=floor, smoke=smoke)


@pytest.fixture(scope="module")
def default_model():
    """The model the gate trains, built through the production call path: the resolver's keywords into ``recipes.build_model``."""
    kwargs = {**resolve("cifar10"), "patch_size": 4}
    return recipes.build_model(SPEC, "small", None, (32, 32, 3), kwargs)


# ---------------------------------------------------------------------
# The recipe equals the legacy trainer's defaults
# ---------------------------------------------------------------------

#: ``(optim field, legacy TrainingConfig field)``.
OPTIM_FIELDS = (
    ("optimizer", "optimizer_type"), ("lr", "learning_rate"), ("weight_decay", "weight_decay"),
    ("warmup_epochs", "warmup_epochs"), ("global_clipnorm", "gradient_clipping"), ("schedule", "lr_schedule_type"),
)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_recipe_is_the_legacy_trainers_defaults(dataset):
    legacy = legacy_defaults()
    recipe = SPEC.recipe(dataset, "small")
    assert recipe["epochs"] == legacy["epochs"]
    assert recipe["batch_size"] == legacy["batch_size"]
    assert recipe["checkpoint_args"] == {"early_stopping_patience": legacy["early_stopping_patience"]}
    for field, legacy_field in OPTIM_FIELDS:
        assert recipe["optim_args"][field] == legacy[legacy_field], field
    # The legacy decay is `build_optimizer`'s: cosine over ALL the epochs (warmup inside them), alpha 0.01, a GLOBAL clip.
    assert recipe["optim_args"]["decay_over"] == "total" and recipe["optim_args"]["min_lr_ratio"] == 0.01
    assert recipe["optim_args"]["clipnorm"] is None
    assert recipe["metric_args"] == {"watch_metric": ta.WATCH_METRIC}
    assert pa.PretrainTransformArgs().augment is legacy["augment_data"], "the shared transform group's default is the legacy one"
    assert ta.TokenizerData(dataset=dataset).validation_split is legacy["validation_split"] is None
    assert ta.TokenizerData(dataset=dataset).max_samples is legacy["max_samples"] is None


def test_the_model_defaults_are_the_legacy_trainers(default_model):
    legacy = legacy_defaults()
    table = tm.VARIANTS["small"]
    assert table == {"hidden_channels": legacy["hidden_channels"], "num_res_blocks": legacy["num_res_blocks"],
                     "embedding_dim": legacy["embedding_dim"]}
    # The two legacy knobs the recipe leaves to the library are the library's defaults: read off the built model.
    assert (default_model.commitment_cost, default_model.use_ema) == (legacy["commitment_cost"], legacy["use_ema"])
    assert {key: getattr(default_model, key) for key in table} == table, "the built model carries the variant's widths"


def test_imagenettes_row_is_the_legacy_default_and_cifar10s_two_differences_are_the_beit_mim_pairing():
    legacy = legacy_defaults()
    big = SPEC.recipe("imagenette", "small")
    assert (big["transform_args"]["image_size"], big["data"]["patch_size"], big["model_args"]["num_embeddings"]) == (
        legacy["image_size"], legacy["downsample_factor"], legacy["num_embeddings"])
    # The legacy command line resolves cifar10 to a down-sample of 8 (a 4 x 4 grid) and 8,192 codes: `beit_mim` refuses that grid at
    # its patch of 4, so the API's row is the 4h pairing: patch 4 (an 8 x 8 grid, `pretrain_args.GEOMETRY`) and 512 codes.
    command_line = legacy_command_line("cifar10")
    small = SPEC.recipe("cifar10", "small")
    assert (command_line.image_size, command_line.downsample_factor, command_line.num_embeddings) == (32, 8, 8192)
    assert (small["transform_args"]["image_size"], small["data"]["patch_size"], small["model_args"]["num_embeddings"]) == (32, 4, 512)
    for dataset in DATASETS:
        recipe = SPEC.recipe(dataset, "small")
        assert (recipe["transform_args"]["image_size"], recipe["data"]["patch_size"]) == pa.GEOMETRY[dataset]
        assert recipe["model_args"]["num_embeddings"] == tm.CODEBOOK[dataset]


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_auto_field_of_every_group_is_listed_by_the_recipe_or_derived(dataset):
    recipe = SPEC.recipe(dataset, "small")
    assert set(recipe) <= set(recipes.RECIPE_KEYS), set(recipe) - set(recipes.RECIPE_KEYS)
    auto = lambda cls: {f.name for f in dataclasses.fields(cls) if f.default == AUTO}
    assert set(recipe["optim_args"]) == auto(OptimArgs)
    assert set(recipe["checkpoint_args"]) == auto(CheckpointArgs)
    assert set(recipe["metric_args"]) == auto(ta.TokenizerMetricArgs) - {"watch_mode"}
    assert set(recipe["transform_args"]) == auto(pa.PretrainTransformArgs)
    assert set(recipe["data"]) == auto(ta.TokenizerData)
    for group in (OptimArgs(), CheckpointArgs(), ta.TokenizerData(dataset=dataset), pa.PretrainTransformArgs(), ta.TokenizerMetricArgs()):
        resolved = group.resolve_auto(context(dataset))
        for field in dataclasses.fields(resolved):
            assert getattr(resolved, field.name) != AUTO, (type(group).__name__, field.name)


def test_a_recipe_key_that_is_no_field_is_named():
    ctx = context()
    broken = {**ctx.defaults, "data": {**ctx.defaults["data"], "patch_sizes": 4}}
    with pytest.raises(ConfigValidationError, match="patch_sizes"):
        ta.TokenizerData(dataset="cifar10").resolve_auto(dataclasses.replace(ctx, defaults=broken))


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_recipe_notice_states_the_numbers_it_doubts_and_says_they_are_unmeasured(dataset):
    notices = tm.recipe_notices_of(dataset)
    assert notices and all(level == "info" and text for level, text in notices)
    text = " ".join(text for _, text in notices)
    optim = SPEC.defaults["optim_args"]
    for words in (f"lr {optim['lr']:g}", f"batch {SPEC.defaults['batch_size']}", f"weight decay {optim['weight_decay']:g}",
                  f"{optim['warmup_epochs']} warmup epochs", f"gradient clip of {optim['global_clipnorm']:g}",
                  f"patience {tm.PATIENCE}", f"{tm.EPOCHS} epochs"):
        assert words in text, words
    assert "legacy" in text and "unmeasured" in text and "perplexity floor" in text and "the bar" in text, "whose numbers, and what judges"
    # Only the dataset nobody has run says so (the 8,192-code recipe at 224 px), and it names its geometry.
    assert ("imagenette" in text and "8192" in text.replace(",", "")) is (dataset == "imagenette"), text


# ---------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------


def test_the_default_model_is_the_gates_small_model_with_an_8_by_8_code_grid(default_model):
    # 2,004,003 is READ OFF the built model by `count_params()` at this HEAD (scratch_step5/probe_count.py): 47 weights, the codebook
    # (1, 512, 32) among them; the linear last layer of the decoder (plan D-022) and the sigmoid one it replaced have the same count.
    assert default_model.built and isinstance(default_model, VQVAERotationTrick)
    assert default_model.count_params() == 2_004_003
    assert sum(math.prod(weight.shape) for weight in default_model.weights) == 2_004_003
    assert tuple(default_model.quantizer.weights[0].shape) == (1, 512, 32)
    grid = keras.ops.convert_to_numpy(default_model.encode_to_indices(keras.ops.zeros((1, 32, 32, 3)))).shape
    assert grid == (1, 8, 8)
    assert default_model.downsample_factor == 4 == pa.GEOMETRY["cifar10"][1], "the patch the data and the model share"


@pytest.mark.parametrize("patch", [2, 4, 8, 16])
def test_the_code_grid_is_the_image_over_the_patch_at_every_power_of_two(patch):
    model = build(patch=patch)
    assert model.downsample_factor == patch, "the model's stride is the patch, not the library's default of 4"
    grid = keras.ops.convert_to_numpy(model.encode_to_indices(keras.ops.zeros((2, 32, 32, 3)))).shape
    assert grid == (2, 32 // patch, 32 // patch)


def test_a_rectangular_image_gives_a_rectangular_grid_in_row_major_order():
    model = build(patch=4, shape=(32, 16, 3))
    assert keras.ops.convert_to_numpy(model.encode_to_indices(keras.ops.zeros((1, 32, 16, 3)))).shape == (1, 8, 4)


def test_a_grid_that_is_not_the_image_over_the_patch_is_refused_by_the_proof(monkeypatch):
    # The proof is a REAL forward pass compared with `size // patch`, never the model's own `downsample_factor` (which would agree
    # with itself by construction): a model whose encoder lands on another grid is refused, naming both grids.
    monkeypatch.setattr(VQVAERotationTrick, "encode_to_indices", lambda self, x: keras.ops.zeros((1, 4, 4), "int32"))
    with pytest.raises(ValueError, match=r"code grid.*\(4, 4\).*\(8, 8\)"):
        build(patch=4)


@pytest.mark.parametrize("shape", [(32, 32), (32, 32, 3, 1), (0, 32, 3), (32, 32, 0), (32.0, 32, 3), (32, None, 3), (None, None, 3)])
def test_a_malformed_input_shape_is_refused(shape):
    with pytest.raises(ValueError, match="input_shape"):
        build(shape=shape)


@pytest.mark.parametrize("patch", [0, -4, 3, 5, 6, 4.0, True, 64])
def test_a_patch_that_cannot_cut_the_image_into_code_cells_is_refused_by_name(patch):
    with pytest.raises(ValueError, match="patch_size"):
        build(patch=patch)


def test_a_class_count_a_missing_codebook_and_an_unknown_variant_are_refused_by_name():
    with pytest.raises(ValueError, match="has no classes"):
        tm.create_vq_tokenizer(variant="small", num_classes=10, input_shape=(32, 32, 3), patch_size=4, **TINY)
    with pytest.raises(ConfigError, match="has no classes"):
        recipes.build_model(SPEC, "small", 10, (32, 32, 3), {**TINY, "patch_size": 4})
    assert recipes.build_model(SPEC, "small", None, (32, 32, 3), {**TINY, "patch_size": 4}) is not None, "the control"
    with pytest.raises(ValueError, match="num_embeddings"):
        tm.create_vq_tokenizer(variant="small", input_shape=(32, 32, 3), patch_size=4, embedding_dim=4)
    with pytest.raises(ValueError, match="variant"):
        tm.create_vq_tokenizer(variant="tiny", input_shape=(32, 32, 3), patch_size=4, **TINY)


def test_the_task_keywords_are_required_not_defaulted():
    with pytest.raises(TypeError, match="patch_size"):
        tm.create_vq_tokenizer(variant="small", num_classes=None, input_shape=(32, 32, 3), **TINY)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_recipes_own_rows_pass_the_checks_they_are_held_to(dataset):
    size, patch = pa.GEOMETRY[dataset]
    assert data_group(dataset).patch_size == patch
    check(patch, size, resolve(dataset)["num_embeddings"])


def test_imagenettes_model_has_the_14_by_14_grid_beit_mim_reads_at_224_pixels():
    kwargs = {**resolve("imagenette"), "patch_size": 16, "hidden_channels": 8, "num_res_blocks": 1}  # narrow: this proves the grid only
    model = recipes.build_model(SPEC, "small", None, (224, 224, 3), kwargs)
    assert model.num_embeddings == 8192 and model.downsample_factor == 16
    assert keras.ops.convert_to_numpy(model.encode_to_indices(keras.ops.zeros((1, 224, 224, 3)))).shape == (1, 14, 14)


def test_a_model_argument_reaches_the_model_and_beats_the_variants_table():
    model = build(commitment_cost=0.5, use_ema=True, embedding_dim=8, hidden_channels=16, num_res_blocks=2)
    assert (model.commitment_cost, model.use_ema, model.embedding_dim, model.hidden_channels, model.num_res_blocks) == (
        0.5, True, 8, 16, 2)
    assert model.num_embeddings == TINY["num_embeddings"]


def test_the_watch_metric_is_the_models_own_reconstruction_key():
    model = build()
    names = [metric.name for metric in model.metrics]
    assert ta.WATCH_METRIC == "val_" + "reconstruction_loss" and "reconstruction_loss" in names, names


# ---------------------------------------------------------------------
# The refusal tables and the signature census
# ---------------------------------------------------------------------

#: The keys the plan names as refused: each is a knob no run measured, or one the bar's arithmetic would stop describing.
PLANNED_REFUSALS = ("kmeans_init", "dead_code_threshold", "reconstruction_loss_weight", "encoder", "decoder")


def test_the_census_classifies_every_parameter_of_the_constructor_in_both_directions():
    assert tm.signature_census() == {"unclassified": [], "stale": []}
    signatures = tm.factory_signatures()
    assert set(signatures) == set(tm.FACTORIES) and len(tm.FACTORIES) == 1
    walked = {name for names in signatures.values() for name in names}
    assert {"num_embeddings", "downsample_factor", "input_shape", "kmeans_init", "reconstruction_loss_weight"} <= walked


def test_the_census_fails_when_the_constructor_gains_a_parameter_no_table_names_and_when_a_table_key_loses_its_parameter():
    real = tm.factory_signatures()
    grown = {path: [*names, "new_option"] for path, names in real.items()}
    assert tm.signature_census(grown)["unclassified"] == ["new_option"]
    gone = sorted(tm.ALLOWED)[0]
    shrunk = {path: [name for name in names if name != gone] for path, names in real.items()}
    assert tm.signature_census(shrunk)["stale"] == [gone]


def test_the_classes_of_a_key_do_not_overlap_and_the_planned_refusals_are_refused():
    tables = [set(tm.ALLOWED), set(tm.REFUSED), set(tm.SUPPLIED)]
    for index, first in enumerate(tables):
        for second in tables[index + 1:]:
            assert not (first & second), first & second
    assert set(tm.ALLOWED) == {"num_embeddings", "embedding_dim", "hidden_channels", "num_res_blocks", "commitment_cost", "use_ema"}
    assert set(tm.SUPPLIED) == {"variant", "input_shape", "patch_size", "downsample_factor"}
    assert set(PLANNED_REFUSALS) <= set(tm.REFUSED)
    for key, why in tm.REFUSED.items():
        assert why and not why.endswith("."), key


@pytest.mark.parametrize("key", sorted({*tm.REFUSED, *PLANNED_REFUSALS}))
def test_every_refused_key_is_named_with_its_rule_before_the_claim(key):
    # Over the table AND the planned keys: a key moved out of the table is a failure here, not a smaller table.
    with pytest.raises(ConfigValidationError) as raised:
        resolve(model_args={key: 1})
    text = str(raised.value)
    assert f"model_args.{key}" in text and f"is not offered by {ta.FAMILY!r}" in text and tm.REFUSED[key] in text, text


@pytest.mark.parametrize("key", sorted(tm.SUPPLIED))
def test_a_key_the_task_supplies_is_refused_by_name(key):
    with pytest.raises((ConfigValidationError, ConfigError)) as raised:
        resolve(model_args={key: 1})
    assert f"model_args.{key}" in str(raised.value) or f"'{key}'" in str(raised.value)


def test_an_unknown_key_is_refused_with_a_suggestion_and_every_refusal_is_named_at_once():
    with pytest.raises(ConfigValidationError, match="Unknown key 'model_args.no_such_key'"):
        resolve(model_args={"no_such_key": 1})
    with pytest.raises(ConfigValidationError, match="Did you mean 'model_args.commitment_cost'"):
        resolve(model_args={"commitment_costs": 0.1})
    with pytest.raises(ConfigValidationError) as raised:
        resolve(model_args={"kmeans_init": True, "encoder": None, "typo_key": 1})
    text = str(raised.value)
    assert "model_args.kmeans_init" in text and "model_args.encoder" in text and "model_args.typo_key" in text


#: Requests that must NOT be refused (the control of the rows above: a resolver that refused everything would pass them).
ACCEPTED = [
    {}, {"num_embeddings": 64}, {"embedding_dim": 16}, {"hidden_channels": 64, "num_res_blocks": 1}, {"commitment_cost": 0.5},
    {"use_ema": True},
]


@pytest.mark.parametrize("model_args", ACCEPTED, ids=["-".join(sorted(a)) or "recipe" for a in ACCEPTED])
def test_the_keys_the_family_offers_are_not_refused(model_args):
    assert resolve(model_args=model_args) == {"num_embeddings": 512, **model_args}


def test_the_resolver_returns_the_recipes_codebook_per_dataset_then_the_callers_keys():
    assert resolve("cifar10") == {"num_embeddings": 512}
    assert resolve("imagenette") == {"num_embeddings": 8192}
    assert resolve("cifar10", model_args={"num_embeddings": 64, "use_ema": True}) == {"num_embeddings": 64, "use_ema": True}


@pytest.mark.parametrize("key", ["variant", "num_classes", "input_shape"])
def test_a_key_the_library_supplies_is_refused_before_the_tables_are_read(key):
    with pytest.raises(ConfigError, match="is set by the library"):
        resolve(model_args={key: 1})


def test_an_adamw_run_refuses_an_explicit_kernel_regularizer_and_an_sgd_decay_is_refused_because_no_regularizer_is_built():
    with pytest.raises(ConfigError, match="kernel_regularizer"):
        resolve(model_args={"kernel_regularizer": "l2"})
    with pytest.raises(ConfigError) as raised:
        resolve(optimizer="sgd", lr=0.1, weight_decay=1e-4)
    assert "optim_args.weight_decay" in str(raised.value) and f"the {ta.TASK} task builds none for {ta.FAMILY!r}" in str(raised.value)
    assert resolve(optimizer="sgd", lr=0.1, weight_decay=None) == {"num_embeddings": 512}, "the control: sgd without a decay resolves"
    assert resolve() == {"num_embeddings": 512}, "and adamw with its decay"


def test_a_precision_other_than_float32_is_refused_by_name():
    assert tm.precision_problems("float32", "") == []
    for precision in ("mixed_float16", "mixed_bfloat16"):
        (line,) = tm.precision_problems(precision, "")
        assert "'precision'" in line and "float32" in line and repr(precision) in line and "quantizer" in line, line
    assert "'run.precision'" in tm.precision_problems("mixed_float16", "run")[0], "the prefix is the caller's dotted path"


# ---------------------------------------------------------------------
# The data group, the geometry and the codebook floor
# ---------------------------------------------------------------------


def test_the_dataset_literal_is_the_shared_loaders_two_datasets():
    (annotation,) = [f.type for f in dataclasses.fields(ta.TokenizerData) if f.name == "dataset"]
    assert set(typing.get_args(annotation)) == set(DATASETS) == {"cifar10", "imagenette"}


@pytest.mark.parametrize("dataset, patch", [("cifar10", 4), ("imagenette", 16)])
def test_the_patch_is_the_geometry_tables_unless_given(dataset, patch):
    data = data_group(dataset)
    assert data.patch_size == patch == pa.GEOMETRY[dataset][1] and data.format == "builtin" and data.dataset == dataset
    assert data_group(dataset, patch_size=8).patch_size == 8, "an explicit patch is kept"


def test_the_help_of_the_patch_states_the_geometry_table_it_is_derived_from():
    (field,) = [f for f in dataclasses.fields(ta.TokenizerData) if f.name == "patch_size"]
    for name, (size, patch) in pa.GEOMETRY.items():
        assert f"{patch} for {name} (grid {size // patch} x {size // patch})" in field.metadata["help"], name


def test_a_recipe_with_no_patch_for_the_dataset_is_named():
    ctx = context()
    with pytest.raises(ConfigError, match="data.patch_size"):
        ta.TokenizerData(dataset="cifar10").resolve_auto(dataclasses.replace(ctx, defaults={**ctx.defaults, "data": {}}))


def test_the_group_problems_name_a_value_out_of_range_at_its_dotted_path():
    lines = ta.TokenizerData(dataset="cifar10", validation_split=1.5, max_samples=0, patch_size=0).problems("data")
    text = "\n".join(lines)
    for name in ("validation_split", "max_samples", "patch_size"):
        assert f"'data.{name}'" in text, name
    assert ta.TokenizerData(dataset="cifar10").problems("data") == []


def test_a_dataset_the_loader_does_not_serve_is_refused():
    for name in ("cifar100", "imagenet", "mnist", "folder"):
        with pytest.raises(ConfigValidationError, match="dataset"):
            ta.load_tokenizer_data_config({"dataset": name})


def test_the_data_config_loads_from_a_mapping_an_instance_and_a_yaml_file(tmp_path):
    data = ta.load_tokenizer_data_config({"dataset": "cifar10", "max_samples": 64})
    assert isinstance(data, ta.TokenizerData) and data.max_samples == 64 and data.format == "builtin"
    assert ta.load_tokenizer_data_config(data) is data
    path = tmp_path / "data.yaml"
    path.write_text("dataset: imagenette\nvalidation_split: 0.2\n")
    from_file = ta.load_tokenizer_data_config(str(path))
    assert (from_file.dataset, from_file.validation_split) == ("imagenette", 0.2)
    path.write_text("dataset: cifar10\nbogus: 1\n")
    with pytest.raises(ConfigValidationError, match="bogus") as raised:
        ta.load_tokenizer_data_config(str(path))
    assert str(path) in str(raised.value)
    with pytest.raises(ConfigError, match="mapping"):
        ta.load_tokenizer_data_config(3)


@pytest.fixture
def imagenette_records_present(monkeypatch):
    monkeypatch.setattr(data_args, "_check_imagenette_records", lambda: None)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_facts_are_the_datasets_geometry_and_no_classes(dataset, imagenette_records_present):
    facts = ta.tokenizer_facts(ta.TokenizerData(dataset=dataset), None)
    assert facts == data_args.DataFacts(dataset=dataset, num_classes=None, image_size=pa.GEOMETRY[dataset][0])
    assert ta.tokenizer_facts(ta.TokenizerData(dataset=dataset), 64).image_size == 64


def test_imagenette_without_its_records_is_refused_before_any_data_is_read(monkeypatch, tmp_path):
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "no_tfds"))
    with pytest.raises(ConfigError, match="imagenette"):
        ta.tokenizer_facts(ta.TokenizerData(dataset="imagenette"), None)
    assert ta.tokenizer_facts(ta.TokenizerData(dataset="cifar10"), None).image_size == 32, "cifar10 has no records to look for"


@pytest.mark.parametrize("patch, size", [(4, 32), (8, 32), (2, 32), (16, 224), (1, 32), (32, 32)])
def test_a_patch_that_is_a_power_of_two_dividing_the_image_is_accepted(patch, size):
    assert ta.geometry_problem(patch, size) is None
    check(patch, size)


@pytest.mark.parametrize("patch", [3, 5, 6, 12, 24])
def test_a_patch_that_is_not_a_power_of_two_is_refused_by_name_whatever_the_image(patch):
    for size in (32, 30, patch * 8):
        with pytest.raises(ConfigError, match="data.patch_size") as raised:
            check(patch, size)
        assert "power of 2" in str(raised.value) and repr(patch) in str(raised.value)


@pytest.mark.parametrize("patch, size", [(64, 32), (8, 36), (16, 24), (4, 30)])
def test_a_power_of_two_that_does_not_divide_the_image_is_refused_naming_both_sides(patch, size):
    with pytest.raises(ConfigError) as raised:
        check(patch, size)
    text = str(raised.value)
    assert "data.patch_size" in text and "transform_args.image_size" in text and str(size) in text and "does not divide" in text


@pytest.mark.parametrize("patch", [0, -2, 4.0, True, "4"])
def test_a_patch_that_is_not_a_positive_int_is_refused(patch):
    with pytest.raises(ConfigError, match="data.patch_size"):
        check(patch, 32)


def test_a_codebook_under_the_floor_is_refused_and_the_floor_itself_is_not():
    assert FLOOR == 64.0, "the floor is `pretrain_run`'s one definition (the tests read it, the check is handed it)"
    for codes in (1, 32, 63):
        with pytest.raises(ConfigError, match="model_args.num_embeddings") as raised:
            check(codes=codes)
        assert str(codes) in str(raised.value) and "64" in str(raised.value) and "debug_args.smoke" in str(raised.value)
    check(codes=64)
    check(codes=65)
    check(codes=8192)


def test_the_floor_is_the_one_the_caller_passes_and_a_smoke_run_has_none():
    check(codes=16, floor=8.0)
    with pytest.raises(ConfigError, match="num_embeddings"):
        check(codes=7, floor=8.0)
    check(codes=1, smoke=True)
    with pytest.raises(ConfigError, match="data.patch_size"):
        check(patch=6, codes=1, smoke=True)


@pytest.mark.parametrize("codes", [0, -5, 16.0, True, None, "512"])
def test_a_codebook_that_is_not_a_positive_int_is_refused_even_in_a_smoke_run(codes):
    with pytest.raises(ConfigError, match="model_args.num_embeddings"):
        check(codes=codes, smoke=True)


def test_the_metric_group_watches_the_reconstruction_and_minimises_it():
    resolved = ta.TokenizerMetricArgs().resolve_auto(context())
    assert (resolved.watch_metric, resolved.watch_mode) == ("val_reconstruction_loss", "min")
    assert ta.TokenizerMetricArgs(watch_metric="val_reconstruction_loss").resolve_auto(context()).watch_mode == "min"


@pytest.mark.parametrize(
    "given, key",
    [
        ({"watch_metric": "val_loss"}, "watch_metric"),
        ({"watch_metric": "loss"}, "watch_metric"),
        ({"watch_metric": "val_vq_loss"}, "watch_metric"),
        ({"watch_metric": "val_reconstruction_loss", "watch_mode": "max"}, "watch_mode"),
        ({"watch_mode": "max"}, "watch_mode"),
    ],
)
def test_the_metric_group_refuses_what_the_bar_cannot_describe(given, key):
    # The bar is about the MINIMUM `val_reconstruction_loss` of the selection window (the constant image's score has no VQ terms, and
    # `val_loss` carries them): the saved best checkpoint must be that epoch, so another metric or the other direction is refused.
    (line,) = ta.TokenizerMetricArgs(**given).problems("metric_args")
    assert f"'metric_args.{key}'" in line and "val_reconstruction_loss" in line and "min" in line
    for allowed in ({}, {"watch_metric": "val_reconstruction_loss"}, {"watch_mode": "min"}):
        assert ta.TokenizerMetricArgs(**allowed).problems("metric_args") == [], allowed


# ---------------------------------------------------------------------
# The row, its task and the import rule
# ---------------------------------------------------------------------


@pytest.fixture
def registered(monkeypatch):
    """An empty registry holding every built-in family, this one included (``register_builtin_models`` registers it with its task, D-031)."""
    monkeypatch.setattr(registry, "MODELS", {})
    monkeypatch.setattr(registry, "TASKS", {})
    recipes.register_builtin_models()
    return registry


def test_the_row_is_one_family_one_variant_and_declares_its_transfer():
    (row,) = tm.builtin_tokenizer_models()
    assert (row.family, row.tasks, ta.TASK, tm.TASK) == (ta.FAMILY, (ta.TASK,), "beit_tokenizer", ta.TASK)
    assert ta.FAMILY == "vq_vae_rotation", "the library package's name"
    assert sorted(row.load_variants()) == ["small"] and row.load_factory() is tm.create_vq_tokenizer
    assert row.load_resolver() is tm.resolve_tokenizer_args and row.defaults is tm.RECIPE
    # `init` takes an earlier tokenizer run of the same size, patch and channels only (the codebook size is the engine's geometry key).
    assert row.transfer is not None and row.transfer.fixed == ("channels", "input_size", "patch_size")
    assert row.transfer.head == () and not row.transfer.sources


def test_the_family_trains_this_task_only_and_every_other_family_is_refused_under_it(registered):
    assert registry.parse_model_name(f"{ta.FAMILY}/small", task=ta.TASK) == (ta.FAMILY, "small")
    others = sorted(set(registry.MODELS) - {ta.FAMILY})
    assert len(others) >= 20, "the control: the other families are really registered"
    for family in others:
        with pytest.raises(ConfigError, match=f"the family '{family}' trains"):
            registry.parse_model_name(f"{family}/anything", task=ta.TASK)


def test_the_family_is_refused_under_every_other_task(registered):
    tasks = sorted({task for spec in registry.MODELS.values() for task in spec.tasks} - {ta.TASK})
    assert len(tasks) >= 8, tasks  # the control: the other tasks are really registered
    for task in tasks:
        with pytest.raises(ConfigError) as raised:
            registry.parse_model_name(f"{ta.FAMILY}/small", task=task)
        assert f"trains {ta.TASK}" in str(raised.value) and repr(task) in str(raised.value)
    with pytest.raises(UnknownModelError, match="variant"):
        registry.parse_model_name(f"{ta.FAMILY}/tiny", task=ta.TASK)


def test_importing_the_modules_and_building_the_row_imports_no_model_code():
    code = (
        "import sys\n"
        "from train.api import tokenizer_args, tokenizer_models\n"
        "specs = tokenizer_models.builtin_tokenizer_models()\n"
        "assert [s.family for s in specs] == ['vq_vae_rotation']\n"
        "assert specs[0].recipe('cifar10', 'small')['data'] == {'patch_size': 4}\n"
        "assert sorted(specs[0].load_variants()) == ['small'] and tokenizer_models.recipe_notices_of('imagenette')\n"
        "loaded = sorted(m for m in ('keras', 'tensorflow', 'dl_techniques', 'yaml', 'train.common') if m in sys.modules)\n"
        "print('LOADED', loaded)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "LOADED []"
