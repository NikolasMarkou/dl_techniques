"""The ``image_segmentation`` task end to end, per family (plan 2faf890a step 28, D-077).

Real fits on the CPU of ``convunext/tiny`` and ``accunet/base`` on synthetic folders of image and
mask pairs whose class fractions are known by construction, so the numbers the summary reports can be
recomputed by hand rather than read back from the code under test:

* every mask is 20 columns of class 0, 8 of class 1 and 4 of class 2 (62.5%, 25%, 12.5%), whatever
  the random split picks, so the majority-class baseline is class 0 with mIoU ``0.625 / 3`` (three
  classes present); with the last four columns void (255) it is class 0 with ``(20/28) / 2`` (class 2
  never appears: Keras' ``MeanIoU`` leaves a class with an empty union out of the mean);
* the verdict is keyed on ``val_miou`` (a run whose ``val_accuracy`` is higher than its ``val_miou``,
  as these are, would report another number if it were keyed on accuracy), against the baseline;
* the function and ``python -m train image_segmentation`` write the same ``config.json`` bytes, for
  both families (the run stops at ``build_model``, after both files are written, as the
  classification file does);
* a killed run resumes and reports the same bar, because the baseline is worked out again by
  ``build_data``;
* every refusal that needs no loaded data leaves no run directory: a classifier under the segmentation
  task and the reverse, a side that is not a multiple of the family's stride, a key the adapter sets,
  a conformal level, a ``watch_metric`` the task does not report.

The recipes are compared with the legacy trainers' defaults, read from their config classes, and the
two stated deviations (the monitor and AccUNet's loss) are asserted as deviations. The tests write only
under ``tmp_path``.
"""

import json
import re
from pathlib import Path

import keras
import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.args import AutoContext
from train.api.config import validate
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_segmentation import (
    ImageSegmentationConfig,
    ImageSegmentationTrainer,
    train_image_segmentation,
)
from train.common import segmentation_run

from . import _segmentation_fixtures as fx

VOID = 255
SIZE = 32
#: Columns of class 0, 1 and 2 in every mask (the last four are void when a run says so).
COLUMNS = (20, 8, 4)


@pytest.fixture(scope="module")
def world():
    """An empty registry holding the built-in families and the real task rows."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def one_mask(void: bool) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[:, COLUMNS[0]:COLUMNS[0] + COLUMNS[1]] = 1
    mask[:, COLUMNS[0] + COLUMNS[1]:] = VOID if void else 2
    return mask


def write_set(root: Path, count: int, void: bool, prefix: str):
    images, masks = root / f"{prefix}images", root / f"{prefix}masks"
    for index in range(count):
        mask = one_mask(void)
        # A little per-image variety so the images are not one picture: noise in the colour that
        # leaves the nearest palette colour (and so the class) unchanged.
        fx.write_pair(images, masks, f"p{index:03d}", mask)
    return images, masks


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = tmp_path_factory.mktemp("seg_task")
    return {
        "clean": (write_set(root / "clean", 40, False, "train_"), write_set(root / "clean", 8, False, "test_")),
        "void": (write_set(root / "void", 40, True, "train_"), write_set(root / "void", 8, True, "test_")),
    }


def data_of(folders, kind="clean", **extra):
    (train_images, train_masks), (test_images, test_masks) = folders[kind]
    mapping = {
        "format": "folder",
        "train_images": str(train_images),
        "train_masks": str(train_masks),
        "test_images": str(test_images),
        "test_masks": str(test_masks),
        "num_classes": 3,
        "validation_split": 0.25,
    }
    if kind == "void":
        mapping["ignore_index"] = VOID
    mapping.update(extra)
    return mapping


def request(out, folders, model="convunext/tiny", kind="clean", **changes):
    mapping = {
        "out": str(out),
        "model": model,
        "data": data_of(folders, kind),
        "epochs": 2,
        "batch_size": 8,
        "transform_args": {"image_size": SIZE},
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    mapping.update(changes)
    return validate(ImageSegmentationConfig, mapping)


def read_json(path):
    return json.loads(Path(path).read_text())


def history_of(run_dir):
    return read_json(Path(run_dir) / "training_history.json")


# ---------------------------------------------------------------------
# The numbers a run reports, per family, against a hand count
# ---------------------------------------------------------------------

#: (model, mask kind, image_size, model analysis): convunext at a non-square size with a void label and the
#: analyzer (its calibration drops the void pixels), accunet square and clean (its CPU fit is the slow one).
RUNS = [
    pytest.param(("convunext/tiny", "void", [32, 48], True), id="convunext-void-nonsquare"),
    pytest.param(("accunet/base", "clean", SIZE, False), id="accunet-clean"),
]


@pytest.fixture(scope="module", params=RUNS)
def finished(world, folders, request, tmp_path_factory):
    model, kind, size, analysis = request.param
    out = tmp_path_factory.mktemp("run_" + model.replace("/", "_")) / "run"
    result = train_image_segmentation(
        out=str(out), model=model, data=data_of(folders, kind), epochs=2, batch_size=8,
        transform_args={"image_size": size},
        analysis_args={"model_analysis": analysis, "figures": analysis},
    )
    summary = read_json(result.summary_path)
    return model, kind, size, analysis, result, summary, history_of(result.run_dir)


def test_the_baseline_is_the_hand_counted_majority_class(finished):
    _, kind, _, _, _, summary, _ = finished
    baseline = summary["majority_class_baseline"]
    assert baseline["class"] == 0 and baseline["class_name"] == "0"
    if kind == "clean":
        assert baseline["miou"] == pytest.approx(0.625 / 3, abs=1e-12)
        assert baseline["pixel_accuracy"] == pytest.approx(0.625, abs=1e-12)
        assert baseline["per_class_iou"] == pytest.approx([0.625, 0.0, 0.0], abs=1e-12)
    else:
        # Void is left out: 28 valid columns, class 2 absent from the truth and from the prediction.
        assert baseline["miou"] == pytest.approx((20 / 28) / 2, abs=1e-12)
        assert baseline["pixel_accuracy"] == pytest.approx(20 / 28, abs=1e-12)
        assert baseline["per_class_iou"][2] is None


def test_the_verdict_is_val_miou_against_the_baseline(finished):
    _, _, _, _, result, summary, hist = finished
    assert summary["success_metric"] == "val_miou" and summary["monitor"] == "val_miou"
    assert summary["monitor_mode"] == "max"
    assert summary["success_value"] == max(hist["val_miou"])
    # The control for "keyed on val_accuracy": the two maxima are different numbers here.
    assert max(hist["val_accuracy"]) != max(hist["val_miou"])
    assert summary["success_threshold"] == summary["majority_class_baseline"]["miou"]
    assert summary["success"] is bool(summary["success_value"] >= summary["success_threshold"])
    assert result.success is summary["success"]


def test_the_summary_says_what_it_trained_and_what_val_loss_is(finished):
    model, kind, size, _, _, summary, _ = finished
    family, variant = model.split("/")
    assert summary["task"] == "image_segmentation"
    assert (summary["model_family"], summary["variant"]) == (family, variant)
    height, width = (size, size) if isinstance(size, int) else size
    assert summary["input_size"] == [height, width] and summary["input_shape"] == [height, width, 3]
    assert summary["num_classes"] == 3 and summary["class_names"] == ["0", "1", "2"]
    assert summary["ignore_index"] == (VOID if kind == "void" else None)
    assert summary["mask_values"] == ([0, 1, VOID] if kind == "void" else [0, 1, 2])
    assert summary["n_train"] + summary["n_val"] == 40 and summary["n_test"] == 8
    text = " ".join(summary["notes"])
    # D-076, D-077: the number a reader will compare must be named for what it is.
    assert "NOT the cross-entropy pooled over every valid pixel" in text
    assert "weighted by images" in text
    assert "pooled over every valid pixel" in text and "`val_miou`" in text
    assert summary["initial_loss_ratio"] == pytest.approx(1.0, abs=0.35)
    assert any("ln(C)" in note for note in summary["notes"])
    # A segmentation summary has no calibration error of a classifier: neither the top-level `ece` nor
    # the one inside the figure record that `base_summary` would copy up.
    assert "ece" not in summary and "ece" not in summary["visualizations"]
    assert summary["visualizations"] == {"files": [], "failed": [], "skipped": True}
    assert summary["conformal"] is None


def test_a_void_label_is_dropped_from_the_analysis_and_named_in_the_notes(finished):
    _, kind, _, analysis, _, summary, _ = finished
    assert summary["analyzer"]["status"] == ("success" if analysis else "skipped")
    assert "failed_analyses" not in summary["analyzer"]
    notes = " ".join(summary["notes"])
    assert (f"mask value {VOID} is void" in notes) is (kind == "void")


def test_the_run_records_a_version_two_segmentation_task_json(finished):
    model, kind, size, _, result, _, _ = finished
    record = read_json(result.run_dir / "task.json")
    height, width = (size, size) if isinstance(size, int) else size
    assert record["schema_version"] == 2 and record["task"] == "image_segmentation"
    assert record["input_size"] == [height, width] and record["channels"] == 3
    assert record["normalize"] == "unit" and record["mean"] == [0.0] * 3 and record["std"] == [1.0] * 3
    assert record["ignore_index"] == (VOID if kind == "void" else None)
    assert record["output"] == "masks" and record["mask_threshold"] is None
    assert "input_shape" not in record and "eval_transform" not in record


def test_the_model_emits_one_logit_per_class_at_the_input_size(finished):
    _, _, size, _, result, _, _ = finished
    model = keras.models.load_model(result.run_dir / "best_model.keras", compile=False)
    height, width = (size, size) if isinstance(size, int) else size
    assert tuple(model.output_shape) == (None, height, width, 3)


# ---------------------------------------------------------------------
# The function and the command line are one entry point
# ---------------------------------------------------------------------


class StopAtBuild(Exception):
    """Raised in place of ``build_model``: ``config.json`` is already on disk by then."""


@pytest.mark.parametrize("model", ["convunext/tiny", "accunet/base"])
def test_the_function_and_the_cli_write_the_same_config_bytes(model, world, folders, tmp_path):
    reached = []

    def stop(self, config, context, data):
        reached.append(context.family)
        raise StopAtBuild

    data = data_of(folders, "void")
    tokens = [f"data.{key}={value}" for key, value in data.items()]
    out = tmp_path / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setattr(ImageSegmentationTrainer, "build_model", stop)
        with pytest.raises(StopAtBuild):
            train_image_segmentation(
                out=str(out), model=model, data=data, epochs=1, batch_size=8, seed=7,
                transform_args={"image_size": SIZE}, optim_args={"lr": 1e-3},
                metric_args={"success_threshold": 0.1},
                analysis_args={"model_analysis": False, "figures": False},
            )
        out.rename(tmp_path / "function")
        try:
            cli.main([
                "image_segmentation", f"out={out}", f"model={model}", *tokens, "epochs=1",
                "batch_size=8", "seed=7", f"transform_args.image_size={SIZE}", "optim_args.lr=1e-3",
                "metric_args.success_threshold=0.1", "analysis_args.model_analysis=false",
                "analysis_args.figures=false",
            ])
        except StopAtBuild:
            pass
        out.rename(tmp_path / "cli")
    family = model.split("/")[0]
    assert reached == [family, family]
    for name in ("config.json", "config_requested.json"):
        assert (tmp_path / "function" / name).read_bytes() == (tmp_path / "cli" / name).read_bytes(), name
    resolved = read_json(tmp_path / "cli" / "config.json")
    assert resolved["model"] == model and resolved["optim_args"]["lr"] == 1e-3
    assert resolved["metric_args"]["watch_metric"] == "val_miou"
    assert resolved["data"]["ignore_index"] == VOID


# ---------------------------------------------------------------------
# A kill and a resume keep the bar
# ---------------------------------------------------------------------


class KillAt(keras.callbacks.Callback):
    def __init__(self, epoch):
        super().__init__()
        self.epoch = epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch:
            raise RuntimeError("killed")


class Killed(ImageSegmentationTrainer):
    def extra_callbacks(self, config, data):
        return [KillAt(2)]


def test_a_killed_run_resumes_to_the_same_bar_and_the_same_row_count(world, folders, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="killed"):
        Killed(request(out, folders, epochs=3, kind="void")).run()
    assert not (out / "results_summary.json").exists()
    resumed = ImageSegmentationTrainer(
        request(out, folders, epochs=3, kind="void", resume_interrupted=True)
    ).run()
    summary = read_json(resumed.summary_path)
    assert summary["resumed_after_epochs"] == 2 and summary["epochs_run"] == 3
    # `build_data` runs again on a resume, so the bar is there and is the hand-counted one.
    assert summary["success_threshold"] == pytest.approx((20 / 28) / 2, abs=1e-12)
    rows = (out / "training_log.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 3
    assert len(history_of(out)["val_miou"]) == 3


# ---------------------------------------------------------------------
# What is refused before the run directory is claimed
# ---------------------------------------------------------------------


def refused(tmp_path, folders, match, **changes):
    out = tmp_path / "never"
    with pytest.raises((ConfigError, ConfigValidationError), match=match):
        train_image_segmentation(
            out=str(out), model=changes.pop("model", "convunext/tiny"),
            data=changes.pop("data", data_of(folders)), epochs=1, **changes,
        )
    assert not out.exists(), "a refusal that needs no data must leave no run directory"


def test_a_classifier_is_refused_under_the_segmentation_task(world, folders, tmp_path):
    refused(tmp_path, folders, r"the family 'vit' trains image_classification, not 'image_segmentation'",
            model="vit/vit_pico", transform_args={"image_size": SIZE})


def test_a_segmenter_is_refused_under_the_classification_task(world, tmp_path):
    from train.api.tasks.image_classification import train_image_classification

    out = tmp_path / "never"
    with pytest.raises(ConfigError, match=r"the family 'convunext' trains image_segmentation, not 'image_classification'"):
        train_image_classification(out=str(out), model="convunext/tiny", data={"dataset": "cifar10"}, epochs=1)
    assert not out.exists()


@pytest.mark.parametrize(
    "model, size, match",
    [
        ("convunext/tiny", 30, r"the height 30 is not a multiple of 8.*use 24 or 32"),
        ("convunext/base", [32, 40], r"the width 40 is not a multiple of 16"),
        ("accunet/base", [48, 40], r"the width 40 is not a multiple of 16.*use 32 or 48"),
    ],
)
def test_a_side_that_is_not_a_multiple_of_the_stride_is_refused_naming_the_nearest(
    model, size, match, world, folders, tmp_path
):
    refused(tmp_path, folders, match, model=model, transform_args={"image_size": size})


def test_a_key_the_adapter_sets_is_refused_by_name(world, folders, tmp_path):
    refused(tmp_path, folders, r"'use_bias'.*set by the 'convunext' adapter",
            model_args={"use_bias": False}, transform_args={"image_size": SIZE})
    refused(tmp_path, folders, r"'input_channels'.*set by the 'accunet' adapter", model="accunet/base",
            model_args={"input_channels": 1}, transform_args={"image_size": SIZE})


def test_a_conformal_level_is_refused_because_a_segmentation_run_has_no_set_of_classes(world, folders, tmp_path):
    refused(tmp_path, folders, r"analysis_args\.conformal_alpha.*does not produce",
            analysis_args={"conformal_alpha": 0.1}, transform_args={"image_size": SIZE})


def test_a_monitor_the_task_does_not_report_is_refused_and_the_listed_ones_are_val_miou_accuracy_loss(
    world, folders, tmp_path
):
    refused(tmp_path, folders, r"'val_top5_accuracy'.*choose one of .*val_miou",
            metric_args={"watch_metric": "val_top5_accuracy"}, transform_args={"image_size": SIZE})


def test_file_data_without_a_size_is_refused_because_the_recipe_gives_the_size_of_pet_only(world, folders, tmp_path):
    refused(tmp_path, folders, r"transform_args\.image_size.*input size must be given")


def test_the_recipes_never_give_a_size_to_file_data(world):
    for family in ("convunext", "accunet"):
        spec = registry.MODELS[family]
        assert "image_size" not in spec.recipe("folder", "base" if family == "accunet" else "tiny")["transform_args"]
        assert spec.recipe("oxford_pet", "base" if family == "accunet" else "tiny")["transform_args"]["image_size"] == 128


# ---------------------------------------------------------------------
# The families, the hooks and the recipes
# ---------------------------------------------------------------------


def test_the_two_families_train_segmentation_only_and_the_six_classifiers_do_not(world):
    for family in ("convunext", "accunet"):
        assert registry.MODELS[family].tasks == ("image_segmentation",)
    assert all(registry.MODELS[f].tasks == ("image_classification",)
               for f in ("vit", "resnet", "convnext_v1", "convnext_v2", "cliffordnet", "cliffordnet_hier"))
    assert registry.list_models("image_segmentation")[:2] == ["accunet/base", "convunext/base"]
    assert not [m for m in registry.list_models("image_classification") if m.startswith(("convunext", "accunet"))]
    assert {t.name for t in registry.list_tasks()} == {"image_classification", "image_segmentation"}


def test_the_hooks_that_decide_the_verdict_are_the_segmentation_ones():
    assert ImageSegmentationTrainer.task == "image_segmentation"
    assert ImageSegmentationTrainer.success_metric == "val_miou"
    stub = object.__new__(ImageSegmentationTrainer)
    assert stub.success_value({"val_accuracy": [0.9, 0.95], "val_miou": [0.1, 0.3]}) == 0.3


def test_the_analyzer_is_told_the_void_value_and_pet_has_none(world, folders, tmp_path):
    void = ImageSegmentationTrainer(request(tmp_path / "a", folders, kind="void"))
    assert void.analysis_ignore_label(void.config) == VOID
    clean = ImageSegmentationTrainer(request(tmp_path / "b", folders))
    assert clean.analysis_ignore_label(clean.config) is None
    pet = validate(ImageSegmentationConfig, {"out": str(tmp_path / "c"), "model": "convunext/tiny",
                                              "data": {"dataset": "oxford_pet"}})
    assert ImageSegmentationTrainer(pet).analysis_ignore_label(pet) is None


def test_the_bar_is_the_explicit_threshold_else_the_baseline_and_never_a_guess(world, folders, tmp_path):
    explicit = request(tmp_path / "a", folders, metric_args={"success_threshold": 0.42})
    assert ImageSegmentationTrainer(explicit).success_threshold(explicit) == 0.42
    implicit = request(tmp_path / "b", folders)
    with pytest.raises(RuntimeError, match="before the data was built"):
        ImageSegmentationTrainer(implicit).success_threshold(implicit)


def legacy_defaults():
    from train.accunet.train_accunet import AccUNetTrainingConfig
    from train.convunext.common import SegTrainingConfig

    return SegTrainingConfig(), AccUNetTrainingConfig()


def resolved_recipe(family, variant):
    spec = registry.MODELS[family]
    recipe = spec.recipe("oxford_pet", variant)
    from train.api.args import CheckpointArgs, OptimArgs
    from train.api.segmentation_args import SegMetricArgs

    context = AutoContext(family=family, variant=variant, dataset="oxford_pet", num_classes=3,
                          image_size=None, defaults=recipe)
    return recipe, OptimArgs().resolve_auto(context), SegMetricArgs().resolve_auto(context), \
        CheckpointArgs().resolve_auto(context)


def test_the_convunext_recipe_is_the_legacy_trainers_defaults_but_for_the_monitor(world):
    convunext, _ = legacy_defaults()
    recipe, optim, metric, checkpoint = resolved_recipe("convunext", "tiny")
    assert (recipe["epochs"], recipe["batch_size"]) == (convunext.epochs, convunext.batch_size)
    assert optim.optimizer == "adamw" and optim.lr == convunext.learning_rate
    assert optim.weight_decay == convunext.weight_decay and optim.warmup_epochs == convunext.warmup_epochs
    assert optim.clipnorm == 1.0 and optim.global_clipnorm is None
    assert optim.schedule == "cosine_decay" and optim.min_lr_ratio == 0.01 and optim.decay_over == "total"
    assert checkpoint.early_stopping_patience == convunext.patience
    assert recipe["transform_args"] == {"augment": "flip", "image_size": convunext.image_size}
    # The evaluation batch is the training batch, as the legacy trainer evaluates (not the shared 256).
    assert recipe["analysis_args"] == {"eval_batch_size": convunext.batch_size}
    # The stated deviation (D-077): the legacy default monitor is val_loss.
    assert convunext.monitor == "val_loss"
    assert (metric.watch_metric, metric.watch_mode) == ("val_miou", "max")


def test_the_accunet_recipe_is_its_trainers_defaults_but_for_the_monitor_and_the_loss(world):
    _, accunet = legacy_defaults()
    recipe, optim, metric, checkpoint = resolved_recipe("accunet", "base")
    assert (recipe["epochs"], recipe["batch_size"]) == (accunet.epochs, accunet.batch_size)
    assert optim.optimizer == "adamw" and optim.lr == accunet.lr and optim.weight_decay == accunet.weight_decay
    assert optim.warmup_epochs == accunet.warmup_epochs == 2
    assert optim.global_clipnorm == accunet.gradient_clipping == 1.0 and optim.clipnorm is None
    assert checkpoint.early_stopping_patience == accunet.early_stopping_patience
    assert recipe["transform_args"]["image_size"] == accunet.image_size
    assert (accunet.monitor, accunet.loss_name) == ("val_loss", "combo")
    assert (metric.watch_metric, metric.watch_mode) == ("val_miou", "max")
    stub = object.__new__(ImageSegmentationTrainer)
    notices = stub.recipe_notices(type("F", (), {"dataset": "oxford_pet"})(), "accunet")
    assert any("combo" in text for _, text in notices)


def test_the_adapters_build_what_the_legacy_trainers_build(world):
    from train.api import segmentation_models as sm

    assert sm.ACCUNET_VARIANTS["base"] == {"base_filters": 32, "mlfc_iterations": 3}
    _, accunet = legacy_defaults()
    assert (accunet.base_filters, accunet.mlfc_iterations) == (32, 3)


# ---------------------------------------------------------------------
# The numbers the baseline is made of
# ---------------------------------------------------------------------


def batches(masks, batch=2):
    import tensorflow as tf

    images = np.zeros((len(masks), 4, 4, 3), np.float32)
    return tf.data.Dataset.from_tensor_slices((images, np.asarray(masks, np.int32))).batch(batch)


MASKS = [
    [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 2, 2]],
    [[0, 0, 0, 0], [1, 1, 1, 1], [VOID, VOID, 2, 2], [VOID, VOID, 2, 2]],
    [[0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 0], [0, 1, 2, 0]],
]


def test_the_pixel_counts_drop_void_and_count_every_batch():
    counts = segmentation_run.class_pixel_counts(batches(MASKS), 3, VOID)
    flat = np.asarray(MASKS).reshape(-1)
    expected = [int((flat == k).sum()) for k in range(3)]
    assert counts.tolist() == expected and int((flat == VOID).sum()) == 4
    assert counts.sum() == flat.size - 4


def test_a_mask_value_that_is_no_class_and_not_void_is_refused_with_its_value():
    with pytest.raises(ValueError, match="value 7, which is neither a class"):
        segmentation_run.class_pixel_counts(batches([[[0, 7], [1, 1]]]), 3, VOID)


@pytest.mark.parametrize("ignore", [None, VOID])
def test_the_baseline_equals_keras_mean_iou_of_the_constant_prediction(ignore):
    masks = np.asarray(MASKS if ignore is not None else [m for m in MASKS[:1]] + [MASKS[2]], np.int32)
    counts = segmentation_run.class_pixel_counts(batches(masks), 3, ignore)
    baseline = segmentation_run.majority_class_baseline(counts, ["a", "b", "c"])
    metric = keras.metrics.MeanIoU(num_classes=3, sparse_y_true=True, sparse_y_pred=False,
                                   ignore_class=ignore)
    logits = np.zeros(masks.shape + (3,), np.float32)
    logits[..., baseline["class"]] = 1.0
    metric.update_state(masks, logits)
    assert baseline["miou"] == pytest.approx(float(metric.result()), abs=1e-7)
    assert baseline["class_name"] == "abc"[baseline["class"]]


def test_a_tie_goes_to_the_lowest_class_and_all_void_is_refused():
    baseline = segmentation_run.majority_class_baseline(np.asarray([5, 5, 1]), ["a", "b", "c"])
    assert baseline["class"] == 0
    with pytest.raises(ValueError, match="every pixel is void"):
        segmentation_run.majority_class_baseline(np.zeros(3, np.int64), ["a", "b", "c"])


def test_the_readme_and_the_notes_say_the_same_thing_about_val_loss():
    readme = (Path(__file__).resolve().parents[3] / "src" / "train" / "README.md").read_text()
    region = readme[readme.index("image_segmentation"):]
    assert re.search(r"val_loss.*weighted by images", region, re.S), "README must say what val_loss is"
