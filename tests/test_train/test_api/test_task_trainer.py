"""``TaskTrainer.run()``: the one sequence, observed on real fits of a tiny model.

The task is the real ``ImageClassificationTrainer`` (``make_config`` below
builds its config): the real data config, loader, family registry and shared
helpers around a ``vit/vit_pico`` on fake CIFAR-10 arrays. Three fits are shared by the arms that read their outputs:

* the **base** run (defaults, two epochs): the literal file list, the request
  and the resolved config, the log order, the result, the CSV;
* the **knob** run: ``save_best``, ``save_last``, ``figures`` and
  ``tensorboard`` all off, an unreachable threshold;
* the **smoke** run.

Everything else is a refusal or a short run that stops in a callback, so each
arm costs seconds. The tests write only under ``tmp_path``.
"""

import ast
import csv
import dataclasses
import json
from pathlib import Path

import keras
import pytest
import tensorflow as tf

import train.api.recipes as recipes
import train.api.tasks.image_classification as task_module
import train.api.trainer as trainer_module
import train.common.gpu as gpu_module
import train.common.image_augment as augment_module
import train.common.seed as seed_module
from train.api import registry
from train.api.config import AUTO, validate
from train.api.errors import ConfigError, UnknownModelError
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
)
from train.api.trainer import CONFIG_FIELDS, REQUIRED_HOOKS, TaskTrainer, TrainResult

from tests.classification_fakes import use_tiny_raw


def make_config(out, **changes):
    """Return a validated task config for ``vit/vit_pico`` on cifar10.

    :param out: the run directory (an absolute path under ``tmp_path``).
    :param changes: field -> value; a dotted key (``"debug_args.smoke"``) sets a
        field of a group.
    :return: an ``ImageClassificationConfig`` with a batch of 4 and the model
        analysis off (it costs a minute on the CPU).
    """
    mapping = {
        "out": str(out),
        "model": "vit/vit_pico",
        "data": {"dataset": "cifar10"},
        "batch_size": 4,
        "analysis_args": {"model_analysis": False},
    }
    for key, value in changes.items():
        head, _, tail = key.partition(".")
        if tail:
            mapping[head] = {**mapping.get(head, {}), tail: value}
        else:
            mapping[head] = value
    return validate(ImageClassificationConfig, mapping)

# ---------------------------------------------------------------------
# The world: fake arrays, and an empty registry holding the six families
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


def files_of(run_dir: Path):
    """Every file under ``run_dir``, relative, with TensorBoard's event files folded to a directory."""
    found = set()
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir)
        if relative.parts[0] == "tensorboard":
            found.add("tensorboard/")
        elif path.is_file():
            found.add(relative.as_posix())
    return found


def read_json(run_dir: Path, name: str):
    return json.loads((run_dir / name).read_text())


def all_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_strings(item)
    elif isinstance(value, str):
        yield value


@pytest.fixture(scope="module")
def base(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("base") / "run"
    # `warmup_start_lr=0` is an int on purpose: the request must be written as validated (0.0).
    config = make_config(
        run_dir, epochs=2, **{"metric_args.success_threshold": 0.0, "optim_args.warmup_start_lr": 0}
    )
    return ImageClassificationTrainer(config).run(), run_dir, config


@pytest.fixture(scope="module")
def knobs(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("knobs") / "run"
    config = make_config(
        run_dir,
        epochs=1,
        **{
            "checkpoint_args.save_best": False,
            "checkpoint_args.save_last": False,
            "analysis_args.figures": False,
            "logger_args.tensorboard": None,
            "metric_args.success_threshold": 1.0,
        },
    )
    return ImageClassificationTrainer(config).run(), run_dir, config


@pytest.fixture(scope="module")
def smoke(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("smoke") / "run"
    # An explicit threshold nothing can reach: smoke must not apply it (review-iter-2 concern 6).
    config = make_config(
        run_dir, **{"debug_args.smoke": True, "metric_args.success_threshold": 1.0}
    )
    return ImageClassificationTrainer(config).run(), run_dir, config


@pytest.fixture(scope="module")
def reversed_mode(world, tmp_path_factory):
    """Three epochs watching ``val_loss`` with ``watch_mode=max``: the OTHER direction's epoch is saved."""
    run_dir = tmp_path_factory.mktemp("reversed") / "run"
    config = make_config(
        run_dir,
        epochs=3,
        **{
            "metric_args.watch_metric": "val_loss",
            "metric_args.watch_mode": "max",
            "analysis_args.figures": False,
            "logger_args.tensorboard": None,
            "checkpoint_args.save_last": False,
        },
    )
    return ImageClassificationTrainer(config).run(), run_dir, config


# ---------------------------------------------------------------------
# The base run
# ---------------------------------------------------------------------

BASE_FILES = {
    "config.json",
    "checkpoints/backup/latest.weights.h5",
    "checkpoints/backup/latest.weights.h5.bkp",
    "checkpoints/backup/training_metadata.json",
    "checkpoints/backup/training_metadata.json.bkp",
    "checkpoints/fit_state.json",
    "run.lock",
    "config_requested.json",
    "results_summary.json",
    "run.log",
    "task.json",
    "training_history.json",
    "training_log.csv",
    "best_model.keras",
    "final_model.keras",
    "tensorboard/",
    "visualizations/classification_report.json",
    "visualizations/confidence_calibration.png",
    "visualizations/confusion_matrix.png",
    "visualizations/misclassifications.png",
    "visualizations/per_class_metrics.png",
    "visualizations/training_dashboard.png",
}


def test_the_base_run_writes_exactly_these_files(base):
    _, run_dir, _ = base
    assert files_of(run_dir) == BASE_FILES


def test_the_request_keeps_auto_and_the_resolved_config_has_none(base):
    _, run_dir, _ = base
    requested = read_json(run_dir, "config_requested.json")
    resolved = read_json(run_dir, "config.json")
    assert requested["optim_args"]["lr"] == AUTO
    assert requested["devices"] == AUTO
    assert AUTO not in set(all_strings(resolved))
    assert resolved["optim_args"]["lr"] == 3e-4
    assert resolved["devices"] is None
    assert resolved["epochs"] == 2 and resolved["batch_size"] == 4


def test_the_request_is_written_as_the_validated_config_not_as_typed(base):
    _, run_dir, _ = base
    # An int for a float field is a float after validation, and in BOTH files.
    for name in ("config_requested.json", "config.json"):
        text = (run_dir / name).read_text()
        assert '"warmup_start_lr": 0.0' in text, name
        assert '"warmup_start_lr": 0,' not in text, name


def test_args_are_logged_before_the_resolved_args(base):
    _, run_dir, _ = base
    log = (run_dir / "run.log").read_text()
    assert 0 <= log.index("Args:") < log.index("Resolved args:")
    assert '"lr": "auto"' in log.split("Resolved args:")[0]
    assert '"lr": 0.0003' in log.split("Resolved args:")[1].split("Untrained evaluate")[0]


def test_the_result_is_what_the_summary_says(base):
    result, run_dir, _ = base
    summary = read_json(run_dir, "results_summary.json")
    history = read_json(run_dir, "training_history.json")
    assert isinstance(result, TrainResult)
    assert result.run_dir == run_dir
    assert result.summary_path == run_dir / "results_summary.json"
    assert summary["status"] == "ok" and summary["epochs_run"] == 2
    assert result.best_metric == max(history["val_accuracy"])
    assert result.test_metrics_final == summary["test_metrics_final"]
    assert result.test_metrics_best == summary["test_metrics_best"]
    assert result.test_metrics_best is not None
    assert summary["best_val_metrics"]["val_accuracy"] == result.best_metric


def test_success_is_the_best_val_accuracy_against_the_threshold(base):
    result, run_dir, _ = base
    summary = read_json(run_dir, "results_summary.json")
    history = read_json(run_dir, "training_history.json")
    assert summary["success_threshold"] == 0.0
    assert summary["success_value"] == max(history["val_accuracy"])
    assert result.success is True and summary["success"] is True


def test_the_csv_carries_the_learning_rate_of_the_start_of_each_epoch(base):
    _, run_dir, _ = base
    with open(run_dir / "training_log.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "lr" in rows[0]
    # vit warms up from warmup_start_lr (0.0 here) over 5 epochs: epoch 1 starts at 0.
    assert float(rows[0]["lr"]) == 0.0
    assert float(rows[1]["lr"]) > 0.0


def test_the_model_analysis_is_off_when_asked_and_recorded(base):
    _, run_dir, _ = base
    summary = read_json(run_dir, "results_summary.json")
    assert summary["analyzer"]["status"] == "skipped"
    assert not (run_dir / "model_analysis").exists()
    assert not (run_dir / "epoch_analysis").exists()
    assert summary["conformal"] is None
    assert summary["model_loading_validated"] is True


# ---------------------------------------------------------------------
# The knobs of D-022
# ---------------------------------------------------------------------


def test_save_best_false_writes_no_best_model_and_skips_its_test(knobs):
    result, run_dir, _ = knobs
    assert not (run_dir / "best_model.keras").exists()
    assert result.test_metrics_best is None
    summary = read_json(run_dir, "results_summary.json")
    assert summary["test_metrics_best"] is None
    assert summary["best_checkpoint_load_error"] is None


def test_save_last_false_writes_no_final_model_but_still_the_history(knobs):
    result, run_dir, _ = knobs
    assert not (run_dir / "final_model.keras").exists()
    assert (run_dir / "training_history.json").is_file()
    summary = read_json(run_dir, "results_summary.json")
    assert summary["model_loading_validated"] is None
    assert result.test_metrics_final == summary["test_metrics_final"]
    assert set(result.test_metrics_final) >= {"loss", "accuracy"}


def test_figures_false_writes_no_figure_and_no_dashboard(knobs):
    _, run_dir, _ = knobs
    assert not (run_dir / "visualizations").exists()
    summary = read_json(run_dir, "results_summary.json")
    assert summary["visualizations"]["skipped"] is True
    assert summary["visualizations"]["files"] == []
    assert summary["ece"] is None


def test_tensorboard_null_writes_no_tensorboard_directory(knobs):
    _, run_dir, _ = knobs
    assert not (run_dir / "tensorboard").exists()


def test_an_unreachable_threshold_is_a_failed_success_not_an_exception(knobs):
    result, run_dir, _ = knobs
    assert result.success is False
    assert read_json(run_dir, "results_summary.json")["success"] is False


def test_the_knob_run_keeps_what_was_not_turned_off(knobs):
    _, run_dir, _ = knobs
    assert files_of(run_dir) == {
        "config.json",
        "config_requested.json",
        "results_summary.json",
        "run.log",
        "task.json",
        "training_history.json",
        "training_log.csv",
        # the resume files are written whatever the knobs say
        "checkpoints/backup/latest.weights.h5",
        "checkpoints/backup/training_metadata.json",
        "checkpoints/fit_state.json",
        "run.lock",  # one epoch, so no previous backup pair (it is copied before the second save)
    }


def test_smoke_runs_one_epoch_on_at_most_512_samples(smoke):
    _, run_dir, config = smoke
    resolved = read_json(run_dir, "config.json")
    assert resolved["epochs"] == 1
    assert resolved["data"]["max_samples"] == 512
    assert read_json(run_dir, "results_summary.json")["epochs_run"] == 1
    # The request is what was typed: no smoke effect in it.
    requested = read_json(run_dir, "config_requested.json")
    assert requested["epochs"] == AUTO and requested["data"]["max_samples"] is None


def test_smoke_does_not_apply_the_success_threshold_and_exits_zero(smoke):
    result, run_dir, _ = smoke
    summary = read_json(run_dir, "results_summary.json")
    history = read_json(run_dir, "training_history.json")
    # The explicit threshold of 1.0 is not reached (the best val_accuracy is below it) and
    # is still not a failure: the run finished, and the summary says "not applicable".
    assert max(history["val_accuracy"]) < 1.0
    assert result.success is True
    assert summary["success"] is None and summary["success_threshold"] is None
    assert summary["success_value"] == max(history["val_accuracy"])
    assert any("not applied" in note for note in summary["notes"])
    assert "not applicable under debug_args.smoke" in (run_dir / "run.log").read_text()
    assert "Missed success_threshold" not in (run_dir / "run.log").read_text()


def test_watch_mode_reaches_the_epoch_the_summary_and_the_result_report(reversed_mode):
    result, run_dir, _ = reversed_mode
    summary = read_json(run_dir, "results_summary.json")
    losses = read_json(run_dir, "training_history.json")["val_loss"]
    assert len(set(losses)) == 3, "the arm needs three different val_loss values to tell the directions apart"
    highest = losses.index(max(losses)) + 1
    lowest = losses.index(min(losses)) + 1
    assert highest != lowest
    assert summary["monitor_mode"] == "max"
    assert summary["best_epoch"] == highest, "the summary named the epoch the other direction selects"
    assert summary["final_is_best"] is (highest == 3)
    assert result.best_metric == max(losses)
    assert summary["best_val_metrics"]["val_loss"] == max(losses)


def test_the_log_names_the_api_key_and_not_the_legacy_flag(base):
    _, run_dir, _ = base
    log = (run_dir / "run.log").read_text()
    assert "Analyzer skipped (analysis_args.model_analysis=false)" in log
    assert "--no-model-analysis" not in log


def resolve(config, family="vit", variant="vit_pico"):
    trainer = ImageClassificationTrainer(config)
    facts = trainer.describe_data(config)
    return trainer._resolve(facts, family, variant)


def test_smoke_keeps_an_explicit_max_samples(world, tmp_path):
    resolved, _ = resolve(
        make_config(tmp_path / "r", **{"debug_args.smoke": True, "data.max_samples": 16})
    )
    assert (resolved.epochs, resolved.data.max_samples) == (1, 16)


def test_without_smoke_the_recipe_decides_the_epochs_and_the_batch(world, tmp_path):
    config = dataclasses.replace(make_config(tmp_path / "r"), batch_size=AUTO)
    resolved, context = resolve(config)
    assert (resolved.epochs, resolved.batch_size) == (50, 128)
    assert resolved.data.max_samples is None
    assert context.defaults["epochs"] == 50
    explicit, _ = resolve(make_config(tmp_path / "r", epochs=7, batch_size=9))
    assert (explicit.epochs, explicit.batch_size) == (7, 9)


class _Stop(Exception):
    """Raised by a patched ``set_seeds`` so that a run ends right after ``setup_gpu``."""


@pytest.mark.parametrize(
    "given, expected",
    [(1, 1), (0, 0), (AUTO, None), (None, None)],
    ids=["one", "zero", "auto", "null"],
)
def test_devices_reaches_setup_gpu_as_the_index_or_none(given, expected, world, tmp_path, monkeypatch):
    seen = []
    # `devices` is checked against the visible list before the claim; three entries cover 0 and 1.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setattr(gpu_module, "setup_gpu", lambda *args, **kwargs: seen.append((args, kwargs)))

    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(seed_module, "set_seeds", stop)
    with pytest.raises(_Stop):
        ImageClassificationTrainer(make_config(tmp_path / "run", devices=given)).run()
    # The API names its own key in setup_gpu's messages, not the legacy `--gpu`.
    assert seen == [((expected,), {"flag": "devices"})]


def test_the_transform_group_reaches_the_augment_hook_and_the_loader(world, tmp_path, monkeypatch):
    seen = {}

    def augment(name, **kwargs):
        seen["augment"] = (name, kwargs)
        return "hook"

    def loader(data, **kwargs):
        seen["loader"] = kwargs
        return "data"

    monkeypatch.setattr(augment_module, "build_augment", augment)
    monkeypatch.setattr(task_module, "load_image_data", loader)
    config = make_config(
        tmp_path / "run",
        **{
            "data": {"dataset": "imagenette"},
            "transform_args.augment": "cifar_autoaugment",
            "transform_args.random_erasing_prob": 0.25,
            "transform_args.normalize": "unit",
            "transform_args.image_size": 64,
        },
    )
    resolved, context = resolve(config)
    assert ImageClassificationTrainer(resolved).build_data(resolved, context) == "data"
    from dl_techniques.datasets.vision.classification import get_spec

    # D-049: the erasing value is the dataset's mean, as in the legacy cliffordnet trainer.
    assert seen["augment"] == (
        "cifar_autoaugment",
        {"random_erasing_prob": 0.25, "fill": get_spec("imagenette").mean},
    )
    assert seen["loader"]["augment"] == "hook" and seen["loader"]["augment_from_mean"] is None
    assert (seen["loader"]["normalize"], seen["loader"]["image_size"]) == ("unit", 64)
    assert ImageClassificationTrainer(resolved).describe_data(resolved).image_size == 64


def test_the_recipe_loader_mapping_reaches_the_data_of_the_family(world, tmp_path):
    # convnext repeats MNIST's channel (its recipe's loader says channels=3); vit does not.
    def loaded(model):
        config = make_config(tmp_path / "r", model=model, **{"data.dataset": "mnist"})
        trainer = ImageClassificationTrainer(config)
        family, variant = model.split("/")
        resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
        return context, trainer.build_data(resolved, context)

    context, data = loaded("convnext_v1/cifar10")
    assert context.defaults["loader"] == {"channels": 3}
    assert data.input_shape == (28, 28, 3)
    context, data = loaded("vit/vit_pico")
    assert "loader" not in context.defaults
    assert data.input_shape == (28, 28, 1)


# ---------------------------------------------------------------------
# Refusals: nothing is created before the cheap ones
# ---------------------------------------------------------------------


def test_smoke_with_an_explicit_epochs_is_refused_before_anything_is_created(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="'epochs'.*smoke"):
        ImageClassificationTrainer(make_config(out, epochs=3, **{"debug_args.smoke": True})).run()
    assert not out.exists()


def test_resuming_a_run_that_does_not_exist_is_refused_and_creates_nothing(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="checkpoints/backup"):
        ImageClassificationTrainer(make_config(out, resume_interrupted=True)).run()
    assert not out.exists()


def test_an_unknown_model_is_refused_before_the_directory_is_claimed(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(UnknownModelError, match="vit_pic"):
        ImageClassificationTrainer(make_config(out, model="vit/vit_pic")).run()
    assert not out.exists()


def test_a_used_directory_is_refused_and_left_alone(world, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "results_summary.json").write_text("{}")
    with pytest.raises(ConfigError, match="already holds a run"):
        ImageClassificationTrainer(make_config(out)).run()
    assert [p.name for p in out.iterdir()] == ["results_summary.json"]


def test_a_task_that_describes_other_data_than_it_loads_is_a_bug_and_says_so(world, tmp_path):
    class Lying(ImageClassificationTrainer):
        def describe_data(self, config):
            facts = super().describe_data(config)
            return dataclasses.replace(facts, num_classes=facts.num_classes + 1)

    with pytest.raises(RuntimeError, match="describe_data said 11 classes"):
        Lying(make_config(tmp_path / "run")).run()


def test_a_trainer_needs_every_config_field(world, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class Thin:
        out: str = "x"

    with pytest.raises(TypeError, match="model"):
        TaskTrainer(Thin())


def test_a_task_that_forgets_a_required_hook_is_refused_when_built_and_the_hook_is_named(
    world, tmp_path
):
    class Forgetful(TaskTrainer):
        task = "forgetful"
        success_metric = "val_x"

        def describe_data(self, config):
            raise AssertionError("must not run")

    with pytest.raises(TypeError) as caught:
        Forgetful(make_config(tmp_path / "run"))
    text = str(caught.value)
    for name in REQUIRED_HOOKS:
        assert (name in text) == (name != "describe_data"), name

    class NoMetric(ImageClassificationTrainer):
        success_metric = ""

    with pytest.raises(TypeError, match="success_metric"):
        NoMetric(make_config(tmp_path / "run"))


def test_the_neutral_defaults_of_the_optional_hooks_carry_no_classification(world, tmp_path):
    class Bare(ImageClassificationTrainer):
        epoch_line_metrics = TaskTrainer.epoch_line_metrics
        summary_fields = TaskTrainer.summary_fields
        task_notes = TaskTrainer.task_notes
        files_to_scan = TaskTrainer.files_to_scan
        recipe_notices = TaskTrainer.recipe_notices

    task = Bare(make_config(tmp_path / "run"))
    assert task.epoch_line_metrics(task.config) == ("loss", "val_loss")
    assert task.summary_fields(task.config, object()) == {}
    assert task.task_notes(task.config, {}, 0.0, object()) == []
    assert task.files_to_scan(object()) is None
    assert task.recipe_notices(None, "vit") == []
    assert TaskTrainer.success_metric == ""


def test_the_task_config_has_exactly_the_fields_the_trainer_reads():
    assert {f.name for f in dataclasses.fields(ImageClassificationConfig)} == set(CONFIG_FIELDS)


# ---------------------------------------------------------------------
# checkpoint=
# ---------------------------------------------------------------------


def saved_dense(path, name="probe", fill=None):
    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(3, name=name)])
    if fill is not None:
        model.set_weights([w * 0 + fill for w in model.get_weights()])
    model.save(path)
    return model


def test_a_checkpoint_that_is_no_file_is_refused_with_the_key(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="'checkpoint'.*not a file"):
        ImageClassificationTrainer(make_config(out, checkpoint=str(tmp_path / "nope.keras"))).run()


def test_a_checkpoint_that_shares_no_layer_is_a_config_error(world, tmp_path):
    other = tmp_path / "other.keras"
    saved_dense(other, name="unrelated")
    with pytest.raises(ConfigError, match="'checkpoint'"):
        ImageClassificationTrainer(make_config(tmp_path / "run", checkpoint=str(other))).run()


def test_a_checkpoint_with_a_matching_layer_loads_its_weights(world, tmp_path):
    source = tmp_path / "source.keras"
    saved_dense(source, fill=0.5)
    target = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(3, name="probe")])
    assert not (target.get_weights()[0] == 0.5).all()
    ImageClassificationTrainer(make_config(tmp_path / "unused"))._load_checkpoint(target, str(source))
    assert (target.get_weights()[0] == 0.5).all()


# ---------------------------------------------------------------------
# Precision, and the policy that is put back
# ---------------------------------------------------------------------


class _Stop(Exception):
    """Raised from a callback to end a run right after ``fit`` starts."""


class Recording(ImageClassificationTrainer):
    """Records the policy and the optimizer in force when training begins, then stops."""

    seen = {}

    def extra_callbacks(self, config, data):
        outer = type(self)

        class Probe(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                outer.seen = {
                    "policy": keras.mixed_precision.global_policy().name,
                    "optimizer": type(self.model.optimizer).__name__,
                }
                raise _Stop

        return [Probe()]


@pytest.mark.parametrize(
    "precision, policy, optimizer",
    [
        ("float32", "float32", "AdamW"),
        ("mixed_float16", "mixed_float16", "LossScaleOptimizer"),
    ],
)
def test_the_precision_sets_the_policy_and_wraps_the_optimizer_and_is_put_back(
    precision, policy, optimizer, world, tmp_path
):
    before = keras.mixed_precision.global_policy().name
    Recording.seen = {}
    with pytest.raises(_Stop):
        Recording(make_config(tmp_path / "run", precision=precision, epochs=1)).run()
    assert Recording.seen == {"policy": policy, "optimizer": optimizer}
    assert keras.mixed_precision.global_policy().name == before == "float32"


def test_the_top_k_metric_takes_half_precision_logits_where_the_stock_one_cannot():
    import numpy as np

    from dl_techniques.metrics.half_precision_top_k import HalfPrecisionTopKAccuracy

    labels = np.array([1, 2], dtype="int32")
    logits = np.array([[0.1, 0.9, 0.0], [0.5, 0.4, 0.1]], dtype="float16")
    stock = keras.metrics.SparseTopKCategoricalAccuracy(k=2)
    with pytest.raises((TypeError, tf.errors.InvalidArgumentError), match="InTopKV2"):
        stock.update_state(labels, logits)
    ours = HalfPrecisionTopKAccuracy(k=2, name="top2_accuracy")
    ours.update_state(labels, logits)
    assert float(ours.result()) == 0.5


@pytest.mark.parametrize("half_precision", [False, True])
def test_the_metrics_are_named_and_top_k_needs_more_classes_than_k(half_precision):
    from train.common.classification_run import classification_metrics

    def names(num_classes, top_k):
        return [m.name for m in classification_metrics(num_classes, top_k, half_precision)]

    assert names(10, 5) == ["accuracy", "top5_accuracy"]
    assert names(5, 5) == ["accuracy"]
    assert names(100, 3) == ["accuracy", "top3_accuracy"]


def test_float32_gets_the_stock_top_k_metric_and_half_precision_ours():
    # D-051: a float32 model carries no class from this repository.
    from dl_techniques.metrics.half_precision_top_k import HalfPrecisionTopKAccuracy
    from train.common.classification_run import classification_metrics

    stock = classification_metrics(10, 5)[1]
    half = classification_metrics(10, 5, half_precision=True)[1]
    assert type(stock) is keras.metrics.SparseTopKCategoricalAccuracy
    assert type(half) is HalfPrecisionTopKAccuracy


@pytest.mark.parametrize("half_precision", [False, True])
def test_the_top_k_metric_survives_a_save_and_a_reload(half_precision, tmp_path):
    from train.common.classification_run import classification_metrics

    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(10)])
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=classification_metrics(10, 5, half_precision),
    )
    import numpy as np

    # One step builds the optimizer, so the saved and the rebuilt one agree.
    model.train_on_batch(np.zeros((4, 4), "float32"), np.zeros(4, "int32"))
    model.save(tmp_path / "m.keras")
    reloaded = keras.models.load_model(tmp_path / "m.keras")

    scores = reloaded.evaluate(
        np.zeros((4, 4), "float32"), np.zeros(4, "int32"), verbose=0, return_dict=True
    )
    assert set(scores) == {"loss", "accuracy", "top5_accuracy"}


# ---------------------------------------------------------------------
# A diverged fit
# ---------------------------------------------------------------------


def test_a_diverged_fit_writes_the_summary_and_the_history_then_raises(world, tmp_path):
    out = tmp_path / "run"
    config = make_config(out, epochs=1, **{"optim_args.lr": 1e30, "optim_args.warmup_epochs": 0})
    with pytest.raises(RuntimeError, match="Training diverged"):
        ImageClassificationTrainer(config).run()
    summary = read_json(out, "results_summary.json")
    assert summary["status"] == "diverged"
    assert summary["non_finite_metrics"]
    assert (out / "training_history.json").is_file()
    assert not (out / "final_model.keras").exists()


# ---------------------------------------------------------------------
# A bad value met after the claim is a ConfigError that says `out` is used
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, text",
    [
        ({"batch_size": 10000}, "data.max_samples"),
        ({"data.max_samples": 1}, "holds out 0 of 1 samples"),
        ({"model_args": {"dropout_rate": "high"}}, "model factory"),
        ({"model_args": {"no_such_option": 1}}, "model factory"),
    ],
    ids=["batch_above_the_train_split", "one_sample", "string_for_a_rate", "unknown_model_argument"],
)
def test_a_value_the_loader_or_the_factory_refuses_is_a_config_error_naming_out(
    changes, text, world, tmp_path
):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=text) as caught:
        ImageClassificationTrainer(make_config(out, **changes)).run()
    message = str(caught.value)
    assert "'out' is now used" in message and "choose a new 'out'" in message
    assert str(out) in message
    assert (out / "config_requested.json").exists(), "the directory really was claimed"
    assert "--max-samples" not in message


def test_a_legacy_caller_still_reads_the_legacy_flag_text():
    import inspect

    from train.common.classification_data import steps_per_epoch_for
    from train.common.classification_run import report_best_weights

    with pytest.raises(ValueError, match="a larger --max-samples"):
        steps_per_epoch_for(5, 10)
    with pytest.raises(ValueError, match="a larger data.max_samples"):
        steps_per_epoch_for(5, 10, "data.max_samples")
    defaults = inspect.signature(report_best_weights).parameters["analysis_off_hint"].default
    assert defaults == "--no-model-analysis"
    assert inspect.signature(gpu_module.setup_gpu).parameters["flag"].default == "--gpu"


def test_setup_gpu_names_the_callers_key_in_its_refusal(monkeypatch):
    # The host process lists no GPU, so a selection cannot take effect and setup_gpu refuses.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match=r"^devices 0 set CUDA_VISIBLE_DEVICES=0"):
        gpu_module.setup_gpu(0, flag="devices")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match=r"^--gpu 0 set CUDA_VISIBLE_DEVICES=0"):
        gpu_module.setup_gpu(0)


# ---------------------------------------------------------------------
# What the resolution tells the user, in run.log
# ---------------------------------------------------------------------


def test_the_run_log_lists_each_replaced_recipe_value_and_the_warmup_warning(world, tmp_path):
    out = tmp_path / "run"
    Recording.seen = {}
    config = make_config(out, epochs=1, **{"optim_args.optimizer": "sgd", "optim_args.lr": 0.05})
    with pytest.raises(_Stop):
        Recording(config).run()
    log = (out / "run.log").read_text()
    assert "Recipe value not used: optim_args.weight_decay 0.05 -> None" in log
    assert "Recipe value not used: optim_args.momentum None -> 0.9" in log
    assert "WARNING" in log and "warmup_epochs 5 is not below epochs 1" in log


def test_a_monitor_the_compiled_model_does_not_report_is_refused_before_fit(world, tmp_path):
    class Lying(ImageClassificationTrainer):
        """Names a metric that ``compile`` never makes, so only the backstop can see it."""

        def metric_names(self, config, facts):
            return [*super().metric_names(config, facts), "invented"]

    out = tmp_path / "run"
    fitted = []

    class NoFit(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            fitted.append(True)

    class Spy(Lying):
        def extra_callbacks(self, config, data):
            return [NoFit()]

    config = make_config(out, epochs=1, **{"metric_args.watch_metric": "val_invented"})
    with pytest.raises(ConfigError, match="metric_args.watch_metric.*compiled model") as caught:
        Spy(config).run()
    assert "'out' is now used" in str(caught.value)
    assert not fitted, "the epoch loop started before the monitor was checked"
    assert not (out / "best_model.keras").exists()


# ---------------------------------------------------------------------
# The sequence has no handler
# ---------------------------------------------------------------------


def test_run_and_its_steps_hold_no_except_clause():
    def methods_of(module, class_name):
        tree = ast.parse(Path(module.__file__).read_text())
        return {
            node.name: node
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == class_name
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
        }

    methods = methods_of(trainer_module, "TaskTrainer")
    for name in ("run", "_train", "_resolve", "_callbacks", "_notes", "_write_diverged_summary"):
        handlers = [n for n in ast.walk(methods[name]) if isinstance(n, ast.ExceptHandler)]
        assert not handlers, name
    # The reporting sequence is the classification task's since the hooks moved (step 10).
    from train.api.tasks import image_classification

    task_methods = methods_of(image_classification, "ImageClassificationTrainer")
    for name in ("evaluate_and_report", "task_notes", "initial_loss_check"):
        handlers = [n for n in ast.walk(task_methods[name]) if isinstance(n, ast.ExceptHandler)]
        assert not handlers, name
