"""``analysis_args.eval_batch_size``: every pass after ``fit`` takes its batch from one key (D-040).

The ImageNet gate's ``cliffordnet/nano`` trained at batch 8 and then died in the figures' ``predict``
at the module constant ``EVAL_BATCH_SIZE`` = 256, which asked 18 GB at 224 px. The key replaces
the constant on the API path: ``auto`` is the recipe's value and 256 where the recipe says nothing
(so a default run evaluates exactly as before), an int is taken as given.

What is pinned:

* the resolution: the fallback IS the loader's constant, a recipe value is taken, an explicit value
  wins over the recipe, ``0``, a negative number and a bool are refused naming the key;
* the real recipes: ``cliffordnet`` and ``cliffordnet_hier`` set it on ImageNet and Imagenette to the
  batch they train at, no other (family, dataset) pair sets it, and the CIFAR recipes do not;
* the sites: the in-memory selection and test splits are batched at it, the figures' ``predict``
  runs at it, the analyzer and the load check run at it when it is below Keras' 32 and at Keras'
  own default otherwise (so a default run's calls are the calls it always made);
* the places that do NOT use it are stated: file and ImageNet splits keep the training batch.

Tests write only under ``tmp_path``.
"""

import keras
import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import registry
from train.api.args import AnalysisArgs, AutoContext
from train.api.config import validate
from train.api.data_args import BuiltinImageData, load_image_data
from train.api.errors import ConfigValidationError
from train.api.tasks import register_builtin_tasks
from train.common.classification_data import EVAL_BATCH_SIZE
from train.common.evaluation import KERAS_PREDICT_BATCH_SIZE, predict_batch_cap

from tests.classification_fakes import use_tiny_raw

from .test_task_trainer import make_config


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=60, n_test=21)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def context(defaults):
    return AutoContext(
        family="f", variant="v", dataset="d", num_classes=10, image_size=32, defaults=defaults
    )


# ---------------------------------------------------------------------
# The resolution
# ---------------------------------------------------------------------


def test_auto_is_the_loaders_constant_where_the_recipe_says_nothing():
    """Mutant ``a copied 256``: the fallback is the one constant, not a second number."""
    resolved = AnalysisArgs().resolve_auto(context({}))
    assert resolved.eval_batch_size == EVAL_BATCH_SIZE == 256
    assert resolved.model_analysis is True  # the other fields are as they were


def test_auto_takes_the_recipes_value():
    resolved = AnalysisArgs().resolve_auto(context({"analysis_args": {"eval_batch_size": 8}}))
    assert resolved.eval_batch_size == 8


def test_an_explicit_value_wins_over_the_recipe_even_when_it_equals_the_default():
    recipe = context({"analysis_args": {"eval_batch_size": 8}})
    assert AnalysisArgs(eval_batch_size=64).resolve_auto(recipe).eval_batch_size == 64
    assert AnalysisArgs(eval_batch_size=256).resolve_auto(recipe).eval_batch_size == 256


@pytest.mark.parametrize("bad", [0, -4, True])
def test_a_batch_below_one_or_a_bool_is_refused_naming_the_key(bad):
    with pytest.raises((ConfigValidationError, TypeError, ValueError)) as caught:
        lines = AnalysisArgs(eval_batch_size=bad).problems("analysis_args")
        if lines:
            raise ConfigValidationError(lines)
        pytest.fail("no problem was reported")
    assert "analysis_args.eval_batch_size" in str(caught.value) or "eval_batch_size" in str(caught.value)


def test_a_recipe_key_that_is_a_typo_is_named():
    with pytest.raises(ConfigValidationError, match="eval_batch_siz"):
        AnalysisArgs().resolve_auto(context({"analysis_args": {"eval_batch_siz": 8}}))


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "big"])
def test_a_recipe_value_that_is_not_a_positive_int_is_named_as_the_recipes(bad):
    """Mutant ``recipe value unchecked``: a typo in a recipe must not reach the loader."""
    with pytest.raises(ConfigValidationError, match="Family recipe for 'analysis_args'.*eval_batch_size"):
        AnalysisArgs().resolve_auto(context({"analysis_args": {"eval_batch_size": bad}}))


def test_a_bad_value_elsewhere_in_the_group_is_not_worded_as_the_recipes():
    """Control: only the recipe's own value is judged in ``resolve_auto``; the group's other
    fields are the request's and keep their own wording (the hand-built config arm)."""
    resolved = AnalysisArgs(conformal_alpha=2.0).resolve_auto(context({}))
    assert resolved.conformal_alpha == 2.0 and resolved.eval_batch_size == EVAL_BATCH_SIZE
    assert AnalysisArgs(conformal_alpha=2.0).problems("analysis_args")  # reported by `problems`, not here


def test_validate_accepts_auto_and_an_int_and_refuses_a_string():
    assert validate(AnalysisArgs, {"eval_batch_size": "auto"}).eval_batch_size == "auto"
    assert validate(AnalysisArgs, {"eval_batch_size": 16}).eval_batch_size == 16
    with pytest.raises(ConfigValidationError, match="eval_batch_size"):
        validate(AnalysisArgs, {"eval_batch_size": "big"})


# ---------------------------------------------------------------------
# The real recipes
# ---------------------------------------------------------------------

LARGE = ("imagenet", "imagenette")
SMALL = ("cifar10", "cifar100")
FAMILIES = ("vit", "resnet", "convnext_v1", "convnext_v2", "cliffordnet", "cliffordnet_hier")


CLASSIFICATION = "image_classification"
SEGMENTATION = "image_segmentation"


@pytest.fixture(scope="module")
def table(world):
    """The recipe of every (CLASSIFIER family, dataset) pair.

    Step 28 (D-077): the registry also holds the two segmentation families, whose recipes set the
    key in their BASE recipe (the evaluation batch is the training batch, 16) and are checked in
    :func:`test_a_segmentation_run_evaluates_at_the_batch_it_trains_at`; this table, and the claims
    that "every other run keeps 256", are about the classifiers.
    """
    out = {}
    for family in registry.MODELS:
        spec = registry.MODELS[family]
        if spec.tasks != (CLASSIFICATION,):
            continue
        variant = sorted(spec.load_variants())[0]
        for dataset in LARGE + SMALL:
            out[(family, dataset)] = spec.recipe(dataset, variant)
    return out


def test_the_cliffordnet_families_evaluate_at_the_batch_they_train_at_on_large_images(table):
    """Mutant ``the entry is dropped``: 256 images of 224 px asked 18 GB of a 24 GB card."""
    for family, batch in (("cliffordnet", 8), ("cliffordnet_hier", 16)):
        for dataset in LARGE:
            recipe = table[(family, dataset)]
            assert recipe["batch_size"] == batch
            assert recipe["analysis_args"] == {"eval_batch_size": batch}, (family, dataset)


def test_no_other_recipe_sets_the_key_so_every_other_run_keeps_256(table):
    others = [key for key, recipe in table.items() if "analysis_args" in recipe]
    assert sorted(others) == sorted(
        (family, dataset) for family in ("cliffordnet", "cliffordnet_hier") for dataset in LARGE
    )


def test_a_segmentation_run_evaluates_at_the_batch_it_trains_at(world):
    """The shared 256 would put 256 full-size masks and logits in one pass (D-077)."""
    # AccUNet trains at 8: 16 runs out of memory on the 24 GB card in legacy too (D-088). The backbone families
    # train at 16, which D-036's procedure measured to fit for their default variants (D-012).
    for family, batch in (("convunext", 16), ("accunet", 8), ("resnet_seg", 16), ("convnext_seg", 16)):
        spec = registry.MODELS[family]
        assert spec.tasks == (SEGMENTATION,)
        for dataset in ("oxford_pet", "folder", "csv"):
            recipe = spec.recipe(dataset, sorted(spec.load_variants())[0])
            assert recipe["batch_size"] == batch
            assert recipe["analysis_args"] == {"eval_batch_size": batch}, (family, dataset)
            ctx = AutoContext(family=family, variant="v", dataset=dataset, num_classes=3,
                              image_size=None, defaults=recipe)
            assert AnalysisArgs().resolve_auto(ctx).eval_batch_size == batch


def test_the_resolved_value_of_each_real_pair(table):
    for (family, dataset), recipe in table.items():
        ctx = AutoContext(family=family, variant="v", dataset=dataset, num_classes=10,
                          image_size=32, defaults=recipe)
        wanted = recipe["batch_size"] if "analysis_args" in recipe else EVAL_BATCH_SIZE
        assert AnalysisArgs().resolve_auto(ctx).eval_batch_size == wanted, (family, dataset)


# ---------------------------------------------------------------------
# The sites
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "asked, cap", [(None, None), (1, 1), (8, 8), (31, 31), (32, None), (256, None), (1024, None)]
)
def test_the_predict_cap_is_the_ask_below_keras_32_and_nothing_at_or_above(asked, cap):
    """Mutant ``cap passes the ask through``: a default run would predict at 256, not at 32."""
    assert KERAS_PREDICT_BATCH_SIZE == 32
    assert predict_batch_cap(asked) == cap


def test_the_in_memory_splits_are_batched_at_the_key(world):
    data = BuiltinImageData(dataset="cifar10")
    for asked, expect in ((8, 8), (None, 21)):
        loaded = load_image_data(
            data, batch_size=4, seed=0, augment=None, normalize="standardize", image_size=32,
            eval_batch_size=asked,
        )
        first = next(iter(loaded.test_ds))[0].shape[0]
        assert first == expect, (asked, first)  # None: 256 > the 21 images there are
        assert next(iter(loaded.select_ds))[0].shape[0] == min(
            asked or EVAL_BATCH_SIZE, int(loaded.n_select)
        )
        sizes = [int(x.shape[0]) for x, _ in loaded.test_ds]
        assert sum(sizes) == 21 and max(sizes) <= (asked or EVAL_BATCH_SIZE)


def test_the_trainer_hands_the_resolved_key_to_the_loader(world, tmp_path):
    from train.api.tasks.image_classification import ImageClassificationTrainer

    trainer = ImageClassificationTrainer(make_config(tmp_path / "r", **{
        "analysis_args.eval_batch_size": 5}))
    facts = trainer.describe_data(trainer.config)
    resolved, ctx = trainer._resolve(facts, "vit", "vit_pico")
    assert resolved.analysis_args.eval_batch_size == 5
    data = trainer.build_data(resolved, ctx)
    assert int(next(iter(data.test_ds))[0].shape[0]) == 5
    assert data.x_fig.shape[0] == 21  # the figure array is whole; only its predict is batched


class SpyPredict:
    """Record the ``batch_size`` of every ``keras.Model.predict`` call; call through."""

    def __init__(self, monkeypatch):
        self.calls = []
        original = keras.Model.predict
        spy = self

        def predict(model, x, *args, **kwargs):
            spy.calls.append((len(x), kwargs.get("batch_size", "unset")))
            return original(model, x, *args, **kwargs)

        monkeypatch.setattr(keras.Model, "predict", predict)


@pytest.mark.parametrize("asked", [None, 8])
def test_the_figures_and_the_load_check_predict_at_the_key(world, tmp_path, monkeypatch, asked):
    """A whole tiny run, model analysis off. ``fit`` and ``evaluate`` never call ``predict``, so the
    calls are exactly three: the figures' (all 21 test images, at the resolved key), then the
    load check's two (the model's and the reloaded one's, 21 images, at the CAP: ``None`` is Keras'
    32, so a default run makes the calls it always made, and 8 is below it)."""
    from train.api.tasks.image_classification import ImageClassificationTrainer

    spy = SpyPredict(monkeypatch)
    changes = {"epochs": 1, "metric_args.success_threshold": 0.0}
    if asked is not None:
        changes["analysis_args.eval_batch_size"] = asked
    ImageClassificationTrainer(make_config(tmp_path / "run", **changes)).run()
    cap = predict_batch_cap(asked)
    assert spy.calls == [(21, asked or EVAL_BATCH_SIZE), (21, cap), (21, cap)], spy.calls


@pytest.mark.parametrize("asked, cap", [(8, 8), (31, 31), (32, None), (256, None)])
def test_the_trainers_analysis_hook_hands_the_analyzer_the_cap(world, tmp_path, monkeypatch, asked, cap):
    """``run_analysis`` builds the analyzer's config: the cap, never the raw key (D-040)."""
    from train.api.tasks.image_classification import ImageClassificationTrainer
    from train.common import evaluation

    seen = {}

    def fake(model, test_data, history, name, directory, config=None):
        seen["config"] = config

    monkeypatch.setattr(evaluation, "run_model_analysis", fake)
    config = make_config(tmp_path / "r", **{"analysis_args.eval_batch_size": asked})
    x = np.zeros((2, 32, 32, 3), np.float32)
    model = keras.Sequential([keras.layers.Input((32, 32, 3)), keras.layers.Flatten(),
                              keras.layers.Dense(10)])
    ImageClassificationTrainer(config).run_analysis(
        config, model, (x, np.zeros(2, np.int64)), None, "n", str(tmp_path))
    assert seen["config"].prediction_batch_size == cap


def test_the_key_is_echoed_resolved_in_config_json_and_as_requested_in_the_request(world, tmp_path):
    from train.api.tasks.image_classification import ImageClassificationTrainer
    import json

    run_dir = tmp_path / "run"
    ImageClassificationTrainer(make_config(run_dir, epochs=1, **{
        "metric_args.success_threshold": 0.0})).run()
    assert json.loads((run_dir / "config_requested.json").read_text())[
        "analysis_args"]["eval_batch_size"] == "auto"
    assert json.loads((run_dir / "config.json").read_text())[
        "analysis_args"]["eval_batch_size"] == EVAL_BATCH_SIZE
    assert np.isfinite(float(json.loads((run_dir / "results_summary.json").read_text())["best_epoch"]))
