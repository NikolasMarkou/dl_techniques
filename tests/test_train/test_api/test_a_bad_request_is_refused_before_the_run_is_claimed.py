"""A request that is wrong without any data is refused before the run directory exists.

Review iter 2 (concerns 1, 5, 10, 13, 15) found bad values met only after
``open_run``, or after the fit: a mistyped ``metric_args.watch_metric`` trained
every epoch and saved no model, a negative learning rate was refused for a
directory that the next try could not reuse, ``conformal_alpha=2.0`` failed
after the entire fit. Every case below goes through the public function, so the
rule is observed where a user meets it: the refusal is a ``ConfigError`` (exit 2
at the command line) and ``out`` does not exist afterwards.

The arms also pin the other half of each rule, so a refusal that hits everything
is not mistaken for a check: the neighbouring VALID value is accepted.

Everything here is a refusal or a resolution, so no arm fits a model. The
tests write only under ``tmp_path``.
"""

import dataclasses
import logging
import traceback
from pathlib import Path

import keras
import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.errors import ConfigError
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
    train_image_classification,
)

from tests.classification_fakes import use_tiny_raw

#: The smallest request that is valid; every case changes one thing in it.
BASE = {
    "model": "vit/vit_pico",
    "data": {"dataset": "cifar10"},
    "batch_size": 4,
    "analysis_args": {"model_analysis": False},
}


@pytest.fixture(scope="module", autouse=True)
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


@pytest.fixture(autouse=True)
def one_visible_gpu(monkeypatch):
    """Fix the visible list, so ``devices`` has a known range whatever the host holds."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


def request_with(**changes):
    """Return :data:`BASE` with ``changes``; a dotted key sets a field of a group."""
    mapping = {key: dict(value) if isinstance(value, dict) else value for key, value in BASE.items()}
    for key, value in changes.items():
        head, _, tail = key.partition(".")
        if tail:
            mapping[head] = {**mapping.get(head, {}), tail: value}
        else:
            mapping[head] = value
    return mapping


def run(out, **changes):
    return train_image_classification(out=str(out), **request_with(**changes))


def resolve(**changes):
    """Resolve a request like ``run`` does, without claiming anything or loading data."""
    from train.api.config import validate

    config = validate(ImageClassificationConfig, {"out": "unused", **request_with(**changes)})
    trainer = ImageClassificationTrainer(config)
    family, variant = registry.parse_model_name(config.model)
    resolved, _ = trainer._resolve(trainer.describe_data(config), family, variant)
    return resolved, trainer


L2 = keras.regularizers.L2(1e-4)

#: (id, changes, text the message must contain)
REFUSED = [
    # C1: a monitor the run does not report
    ("monitor_typo", {"metric_args.watch_metric": "val_acc"}, "metric_args.watch_metric"),
    ("monitor_unknown", {"metric_args.watch_metric": "val_foo"}, "metric_args.watch_metric"),
    (
        "monitor_top_k_dropped",
        {"metric_args.watch_metric": "val_top5_accuracy", "metric_args.top_k": 10},
        "metric_args.watch_metric",
    ),
    # C5: optimizer
    ("lr_negative", {"optim_args.lr": -1}, "optim_args.lr"),
    ("lr_zero", {"optim_args.lr": 0}, "optim_args.lr"),
    ("optimizer_typo", {"optim_args.optimizer": "sgdd"}, "optim_args.optimizer"),
    (
        "both_clips",
        {"optim_args.clipnorm": 1.0, "optim_args.global_clipnorm": 1.0},
        "optim_args.clipnorm",
    ),
    ("clip_negative", {"optim_args.clipnorm": -1.0, "optim_args.global_clipnorm": None}, "optim_args.clipnorm"),
    (
        "post_warmup_without_steps_left",
        {"optim_args.decay_over": "post_warmup", "optim_args.warmup_epochs": 5, "epochs": 3},
        "optim_args.warmup_epochs",
    ),
    ("momentum_on_adamw", {"optim_args.momentum": 0.9}, "optim_args.momentum"),
    # C5: top-level numbers
    ("epochs_zero", {"epochs": 0}, "epochs"),
    ("batch_size_zero", {"batch_size": 0}, "batch_size"),
    ("seed_negative", {"seed": -1}, "seed"),
    ("seed_too_large", {"seed": 2**40}, "seed"),
    ("devices_outside_the_visible_list", {"devices": 5}, "devices"),
    ("devices_negative", {"devices": -1}, "devices"),
    # C5: groups
    (
        "erasing_above_one",
        {"transform_args.augment": "cifar_autoaugment", "transform_args.random_erasing_prob": 1.5},
        "transform_args.random_erasing_prob",
    ),
    ("conformal_alpha_above_one", {"analysis_args.conformal_alpha": 2.0}, "analysis_args.conformal_alpha"),
    ("conformal_alpha_zero", {"analysis_args.conformal_alpha": 0.0}, "analysis_args.conformal_alpha"),
    ("conformal_alpha_one", {"analysis_args.conformal_alpha": 1.0}, "analysis_args.conformal_alpha"),
    ("validation_split_above_one", {"data.validation_split": 1.5}, "data.validation_split"),
    ("validation_split_zero", {"data.validation_split": 0.0}, "data.validation_split"),
    (
        "model_args_holds_a_keras_object",
        {"model_args": {"kernel_regularizer": L2}, "optim_args": {"optimizer": "sgd", "lr": 0.05}},
        "model_args.kernel_regularizer",
    ),
    # C15: ranges
    ("top_k_zero", {"metric_args.top_k": 0}, "metric_args.top_k"),
    ("top_k_negative", {"metric_args.top_k": -2}, "metric_args.top_k"),
    ("patience_negative", {"checkpoint_args.early_stopping_patience": -1}, "early_stopping_patience"),
    ("threshold_above_one", {"metric_args.success_threshold": 5.0}, "metric_args.success_threshold"),
    ("threshold_negative", {"metric_args.success_threshold": -1.0}, "metric_args.success_threshold"),
    ("analysis_period_negative", {"analysis_args.epoch_analysis_every": -3}, "epoch_analysis_every"),
    # C10: a switch of optimizer without a learning rate
    ("optimizer_switch_without_lr", {"optim_args.optimizer": "sgd"}, "optim_args.lr"),
    ("optimizer_switch_to_adam_without_lr", {"optim_args.optimizer": "adam"}, "optim_args.lr"),
]


@pytest.mark.parametrize("changes, text", [case[1:] for case in REFUSED], ids=[c[0] for c in REFUSED])
def test_a_wrong_request_is_a_config_error_and_leaves_out_unclaimed(changes, text, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=text):
        run(out, **changes)
    assert not out.exists(), "the run directory was claimed before the refusal"


@pytest.mark.parametrize(
    "build, text",
    [
        (lambda c: dataclasses.replace(c, epochs=0), "'epochs'"),
        (lambda c: dataclasses.replace(c, seed=-1), "'seed'"),
        (
            lambda c: dataclasses.replace(
                c, analysis_args=dataclasses.replace(c.analysis_args, conformal_alpha=2.0)
            ),
            "analysis_args.conformal_alpha",
        ),
        (
            lambda c: dataclasses.replace(
                c, analysis_args=dataclasses.replace(c.analysis_args, epoch_analysis_every=-3)
            ),
            "analysis_args.epoch_analysis_every",
        ),
    ],
    ids=["epochs", "seed", "conformal_alpha_in_a_group", "analysis_period_in_a_group"],
)
def test_a_config_built_by_hand_is_checked_again_before_the_claim(build, text, tmp_path):
    """``dataclasses.replace`` skips ``validate``; the trainer must not trust the config."""
    from train.api.config import validate

    config = build(validate(ImageClassificationConfig, {"out": str(tmp_path / "run"), **request_with()}))
    with pytest.raises(ConfigError, match=text):
        ImageClassificationTrainer(config).run()
    assert not (tmp_path / "run").exists()


def test_every_refusal_is_exit_2_at_the_command_line(tmp_path, capsys):
    out = tmp_path / "run"
    tokens = [
        "image_classification",
        f"out={out}",
        "model=vit/vit_pico",
        "data.dataset=cifar10",
        "batch_size=4",
        "metric_args.watch_metric=val_acc",
    ]
    assert cli.main(tokens) == 2
    assert "metric_args.watch_metric" in capsys.readouterr().err
    assert not out.exists()


def test_a_value_only_the_loader_can_see_is_exit_2_and_says_out_is_used(tmp_path, capsys):
    out = tmp_path / "run"
    tokens = ["image_classification", f"out={out}", "model=vit/vit_pico", "data.dataset=cifar10", "batch_size=10000"]
    assert cli.main(tokens) == 2
    error = capsys.readouterr().err
    assert "'out' is now used" in error and "data.max_samples" in error
    assert (out / "config_requested.json").exists(), "this refusal needs the data, so it comes after the claim"


# ---------------------------------------------------------------------
# The original traceback of a wrapped error survives, at DEBUG
# ---------------------------------------------------------------------

#: (id, changes, the step the message names, the exception type raised underneath)
WRAPPED = [
    ("loader_value_error", {"batch_size": 10000}, "the data loader", ValueError),
    ("factory_type_error", {"model_args": {"dropout_rate": "high"}}, "the model factory", TypeError),
]


@pytest.fixture
def dl_logger():
    """Return the library logger, its level put back afterwards (``-v`` changes it)."""
    logger = logging.getLogger("dl")
    before = logger.level
    yield logger
    logger.setLevel(before)


@pytest.mark.parametrize("changes, step, kind", [case[1:] for case in WRAPPED], ids=[c[0] for c in WRAPPED])
def test_a_wrapped_error_keeps_its_traceback_for_verbose_runs(changes, step, kind, tmp_path, caplog, dl_logger):
    dl_logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="dl")
    with pytest.raises(ConfigError) as info:
        run(tmp_path / "run", **changes)
    cause = info.value.__cause__
    assert isinstance(cause, kind), "the original error is chained"
    records = [r for r in caplog.records if f"Original error of {step}" in r.getMessage()]
    assert len(records) == 1 and records[0].levelno == logging.DEBUG
    shown = "".join(traceback.format_exception(*records[0].exc_info))
    assert type(cause).__name__ in shown and str(cause) in shown
    # The frame that raised it is in the text, not only the wrapper.
    raised_in = traceback.extract_tb(cause.__traceback__)[-1]
    assert raised_in.name in shown and Path(raised_in.filename).name in shown


def test_the_traceback_is_not_logged_at_the_default_level(tmp_path, caplog, dl_logger):
    dl_logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO, logger="dl")
    with pytest.raises(ConfigError):
        run(tmp_path / "run", batch_size=10000)
    assert not [r for r in caplog.records if "Original error of" in r.getMessage()]


def test_dash_v_at_the_command_line_shows_the_traceback(tmp_path, caplog, dl_logger):
    caplog.set_level(logging.DEBUG, logger="dl")
    tokens = [
        "-v",
        "image_classification",
        f"out={tmp_path / 'run'}",
        "model=vit/vit_pico",
        "data.dataset=cifar10",
        "batch_size=10000",
    ]
    assert cli.main(tokens) == 2
    assert [r for r in caplog.records if "Original error of the data loader" in r.getMessage()]


# ---------------------------------------------------------------------
# The valid neighbour of each refusal is accepted
# ---------------------------------------------------------------------

ACCEPTED = [
    ("monitor_val_loss", {"metric_args.watch_metric": "val_loss"}),
    ("monitor_train_loss", {"metric_args.watch_metric": "loss"}),
    ("monitor_val_top5", {"metric_args.watch_metric": "val_top5_accuracy"}),
    ("monitor_val_top3_when_k_is_3", {"metric_args.watch_metric": "val_top3_accuracy", "metric_args.top_k": 3}),
    ("seed_zero", {"seed": 0}),
    ("seed_largest", {"seed": 2**32 - 1}),
    ("devices_zero", {"devices": 0}),
    ("erasing_one", {"transform_args.augment": "cifar_autoaugment", "transform_args.random_erasing_prob": 1.0}),
    ("conformal_alpha_inside", {"analysis_args.conformal_alpha": 0.1}),
    ("top_k_one", {"metric_args.top_k": 1}),
    ("patience_zero", {"checkpoint_args.early_stopping_patience": 0}),
    ("threshold_zero", {"metric_args.success_threshold": 0.0}),
    ("threshold_one", {"metric_args.success_threshold": 1.0}),
    ("analysis_period_zero_is_off", {"analysis_args.epoch_analysis_every": 0}),
    ("analysis_period_one", {"analysis_args.epoch_analysis_every": 1}),
    ("optimizer_switch_with_lr", {"optim_args.optimizer": "sgd", "optim_args.lr": 0.05}),
    ("the_recipes_own_optimizer_by_name", {"optim_args.optimizer": "adamw"}),
]


@pytest.mark.parametrize("changes", [case[1] for case in ACCEPTED], ids=[c[0] for c in ACCEPTED])
def test_the_valid_neighbour_of_each_refusal_resolves(changes):
    resolved, _ = resolve(**changes)
    assert resolved.epochs >= 1


# ---------------------------------------------------------------------
# C10: what a switch of optimizer does
# ---------------------------------------------------------------------


def test_sgd_reached_by_a_switch_gets_momentum_0_9_unless_one_is_given():
    resolved, _ = resolve(**{"optim_args.optimizer": "sgd", "optim_args.lr": 0.05})
    assert resolved.optim_args.momentum == 0.9
    assert resolved.optim_args.lr == 0.05
    kept, _ = resolve(**{"optim_args.optimizer": "sgd", "optim_args.lr": 0.05, "optim_args.momentum": 0.5})
    assert kept.optim_args.momentum == 0.5
    null, _ = resolve(**{"optim_args.optimizer": "sgd", "optim_args.lr": 0.05, "optim_args.momentum": None})
    assert null.optim_args.momentum is None


def test_every_recipe_value_a_switch_replaces_is_listed_for_the_log():
    _, trainer = resolve(**{"optim_args.optimizer": "sgd", "optim_args.lr": 0.05})
    lines = [text for _level, text in trainer._notices]
    # The vit recipe offered decay 0.05; sgd takes none (the regularizer is the model's); momentum became 0.9.
    for name in ("weight_decay", "momentum"):
        assert any(f"optim_args.{name} " in line for line in lines), (name, lines)
    assert not any("optim_args.lr " in line for line in lines), "the caller's lr replaced nothing it was given"


def test_a_family_and_dataset_pair_with_no_overlay_is_warned_about_by_name():
    # vit has overlays for the two CIFAR datasets only.
    _, with_overlay = resolve()
    assert not [t for level, t in with_overlay._notices if level == "warning" and "overlay" in t]
    _, trainer = resolve(**{"data.dataset": "mnist"})
    warnings = [t for level, t in trainer._notices if level == "warning" and "overlay" in t]
    assert len(warnings) == 1 and "vit" in warnings[0] and "mnist" in warnings[0]


def test_a_warmup_that_outlasts_the_run_is_warned_about_and_still_runs():
    # vit warms up for 5 epochs; the run is 3 (decay_over total: allowed, but the peak is never reached).
    resolved, trainer = resolve(epochs=3)
    warnings = [t for level, t in trainer._notices if level == "warning" and "warmup" in t]
    assert len(warnings) == 1 and "warmup_epochs 5" in warnings[0]
    _, quiet = resolve(epochs=6)
    assert not [t for level, t in quiet._notices if "warmup" in t]
    _, smoke = resolve(**{"debug_args.smoke": True})
    assert not [t for level, t in smoke._notices if "warmup" in t], "smoke is one epoch on purpose"


# ---------------------------------------------------------------------
# C13: paths, from the command line and in Python
# ---------------------------------------------------------------------


@pytest.fixture
def seen(monkeypatch):
    """Register a task whose function records its keywords instead of training."""
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return dataclasses.make_dataclass("Result", [("success", bool)])(True)

    monkeypatch.setattr(
        registry,
        "TASKS",
        {
            **registry.TASKS,
            "capture": registry.TaskSpec(
                name="capture",
                config_class="train.api.tasks.image_classification:ImageClassificationConfig",
                function=f"{__name__}:_capture_function",
                summary="records its keywords",
            ),
        },
    )
    monkeypatch.setitem(globals(), "_capture_function", capture)
    return calls


def test_data_as_a_yaml_path_works_from_the_command_line(seen, tmp_path):
    data_file = tmp_path / "d.yaml"
    data_file.write_text("dataset: cifar100\nmax_samples: 50\n")
    code = cli.main(["capture", "out=x", "model=vit/vit_pico", f"data={data_file}"])
    assert code == 0
    assert seen[0]["data"] == {"dataset": "cifar100", "max_samples": 50}


def test_data_as_a_yaml_path_works_from_a_config_file(seen, tmp_path):
    data_file = tmp_path / "d.yaml"
    data_file.write_text("dataset: mnist\n")
    config_file = tmp_path / "c.yaml"
    config_file.write_text(f"out: x\nmodel: vit/vit_pico\ndata: {data_file}\n")
    assert cli.main(["capture", f"config={config_file}"]) == 0
    assert seen[0]["data"] == {"dataset": "mnist"}


def test_a_json_object_and_dotted_keys_still_work_for_data(seen):
    assert cli.main(["capture", "out=x", "model=vit/vit_pico", 'data={"dataset": "cifar10"}']) == 0
    assert seen[0]["data"] == {"dataset": "cifar10"}
    assert cli.main(["capture", "out=x", "model=vit/vit_pico", "data.dataset=cifar10"]) == 0
    assert seen[1]["data"] == {"dataset": "cifar10"}


def test_a_data_file_that_does_not_exist_is_exit_2(seen, tmp_path, capsys):
    missing = tmp_path / "nope.yaml"
    assert cli.main(["capture", "out=x", "model=vit/vit_pico", f"data={missing}"]) == 2
    assert "nope.yaml" in capsys.readouterr().err
    assert not seen


def test_a_data_file_with_a_bad_key_is_exit_2(seen, tmp_path, capsys):
    data_file = tmp_path / "d.yaml"
    data_file.write_text("dataset: cifar10\nmax_sample: 5\n")
    assert cli.main(["capture", "out=x", "model=vit/vit_pico", f"data={data_file}"]) == 2
    assert "max_sample" in capsys.readouterr().err
    assert not seen


def test_a_group_without_the_path_form_still_refuses_a_bare_word(seen, capsys):
    assert cli.main(["capture", "out=x", "model=vit/vit_pico", "data.dataset=cifar10", "optim_args=file.yaml"]) == 2
    assert not seen


def test_out_and_checkpoint_accept_a_path_in_python(tmp_path):
    out = tmp_path / "run"
    missing = tmp_path / "nope.keras"
    with pytest.raises(ConfigError, match="'checkpoint'.*not a file") as caught:
        train_image_classification(out=out, checkpoint=missing, **request_with())
    # A Path was turned into the text the config holds, in the message and in the files.
    assert str(missing) in str(caught.value)
    assert (out / "config.json").read_text().count(str(out)) == 1
    assert str(missing) in (out / "config.json").read_text()
