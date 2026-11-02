"""What the trainer can know before a single step, it refuses (review-iter-1 concerns 11 and 12).

* Concern 11: a ``success_metric`` the run does not report cost the whole run at its end (a
  ``KeyError`` after ``fit`` and all the reporting, no ``results_summary.json``, ``fit_finished``
  true so the run could not be resumed). It is now checked against the names the task reports,
  before the directory is claimed, and again against the names ``compile`` produced, before
  ``fit``. The check holds for the classification names with their ``val_`` prefix and for a
  future ``val_miou``, and it refuses the name a Keras ``MeanIoU`` would be guessed under.
* Concern 12: ``REQUIRED_HOOKS`` required that a subclass differ from the base, so
  ``input_check = None`` passed construction and raised ``TypeError: 'NoneType' object is not
  callable`` after the directory was claimed; ``task = ""`` ran to a summary. Each hook must be
  callable and ``task`` and ``success_metric`` non-empty texts, when the trainer is built.
"""

import keras
import pytest

from train.api.errors import ConfigError
from train.api.tasks.image_classification import ImageClassificationTrainer
from train.api.trainer import REQUIRED_HOOKS

from .test_resume_interrupted import world  # noqa: F401  (fixture used by name)
from .test_task_trainer import make_config, resolve


def build(tmp_path, cls=ImageClassificationTrainer, **changes):
    return cls(make_config(tmp_path / "run", **changes))


@pytest.mark.parametrize("metric", ["accuracy", "val_accuracy", "val_top5_accuracy", "top5_accuracy", "loss", "val_loss"])
def test_a_success_metric_the_task_reports_is_accepted_with_and_without_its_val_prefix(world, tmp_path, metric):
    class Reports(ImageClassificationTrainer):
        success_metric = metric

    resolve(make_config(tmp_path / "run"))  # the stock task
    trainer = Reports(make_config(tmp_path / "run"))
    resolved, _ = trainer._resolve(trainer.describe_data(trainer.config), "vit", "vit_pico")
    assert resolved.epochs  # resolution finished: nothing was refused


@pytest.mark.parametrize("wrong", ["val_mIoU", "val_acc", "accuracy_val", "val_top1_accuracy", "val_"])
def test_a_misspelt_success_metric_is_refused_before_the_directory_is_claimed(world, tmp_path, wrong):
    class Typo(ImageClassificationTrainer):
        success_metric = wrong

    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=rf"success_metric.*{wrong!r} is not a metric this run reports.*val_accuracy") as caught:
        Typo(make_config(out)).run()
    assert "'out' is now used" not in str(caught.value)
    assert not out.exists(), "the refusal came after the claim"


def test_a_segmentation_style_metric_is_accepted_when_the_task_names_it_and_the_guess_is_refused(world, tmp_path):
    """The shape step 28 needs: a task whose metrics are named ``miou`` succeeds on ``val_miou``; a
    guess under Keras's own name for ``MeanIoU`` is refused with the list of names to choose from."""

    class Names(ImageClassificationTrainer):
        def metric_names(self, config, facts):
            return ["accuracy", "miou"]

    class Good(Names):
        success_metric = "val_miou"

    class Guess(Names):
        success_metric = "val_mean_io_u"

    config = make_config(tmp_path / "run")
    good = Good(config)
    good._resolve(good.describe_data(config), "vit", "vit_pico")
    guess = Guess(config)
    with pytest.raises(ConfigError, match=r"'val_mean_io_u' is not a metric this run reports.*val_miou"):
        guess._resolve(guess.describe_data(config), "vit", "vit_pico")


def test_a_success_metric_the_compiled_model_does_not_report_is_refused_before_fit(world, tmp_path):
    """The task names ``invented`` but ``compile`` never makes it: only the backstop can see it."""

    class Lying(ImageClassificationTrainer):
        success_metric = "val_invented"

        def metric_names(self, config, facts):
            return [*super().metric_names(config, facts), "invented"]

    fitted = []

    class NoFit(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            fitted.append(True)

    class Spy(Lying):
        def extra_callbacks(self, config, data):
            return [NoFit()]

    out = tmp_path / "run"
    with pytest.raises(ConfigError, match=r"success_metric.*'val_invented' is not a metric the compiled model reports") as caught:
        Spy(make_config(out, epochs=1)).run()
    assert "'out' is now used" in str(caught.value)
    assert not fitted, "the epoch loop started before the success metric was checked"


@pytest.mark.parametrize("hook", REQUIRED_HOOKS)
def test_a_required_hook_that_is_not_callable_is_refused_when_built_naming_it(world, tmp_path, hook):
    broken = type("Broken", (ImageClassificationTrainer,), {hook: None})
    with pytest.raises(TypeError, match=rf"not callable.*{hook}"):
        broken(make_config(tmp_path / "run"))


@pytest.mark.parametrize("task", ["", None, 7])
def test_a_task_that_is_not_a_non_empty_text_is_refused_when_built(world, tmp_path, task):
    broken = type("Broken", (ImageClassificationTrainer,), {"task": task})
    with pytest.raises(TypeError, match=r"task \(class variable\)"):
        broken(make_config(tmp_path / "run"))


@pytest.mark.parametrize("metric", [None, 7])
def test_a_success_metric_that_is_not_a_text_is_refused_when_built(world, tmp_path, metric):
    broken = type("Broken", (ImageClassificationTrainer,), {"success_metric": metric})
    with pytest.raises(TypeError, match=r"success_metric \(class variable\)"):
        broken(make_config(tmp_path / "run"))


def test_the_stock_task_still_builds(world, tmp_path):
    assert build(tmp_path).task == "image_classification"


def _point(start, wait):
    from types import SimpleNamespace

    return SimpleNamespace(start=start, wait=wait, best=None, state={"initial": {}})


def _resolved(patience):
    from types import SimpleNamespace

    return SimpleNamespace(epochs=9, checkpoint_args=SimpleNamespace(save_best=True, early_stopping_patience=patience))


def test_a_resume_after_epoch_one_is_not_a_stopped_early_run_even_at_patience_zero(tmp_path):
    """Mutant ``tr_es_start_gt1`` (review-iter-1: counted near-equivalent by reading, now measured).

    The rule that a run that had waited out its patience has no epoch left applies from epoch 2 on:
    epoch 1 always improves on nothing, so a run killed after it had not decided anything. The two
    differ only at ``early_stopping_patience`` 0, which is in range (an int >= 0). Since D-064 the
    rule does not refuse the run, it ROUTES it to the reporting-only checks: a run with nothing
    left to train is resumed to report, or refused there for what the report needs."""
    from train.api.trainer import TaskTrainer

    point = _point(start=1, wait=0)
    assert TaskTrainer._training_complete_reason(point, _resolved(0)) is None
    assert TaskTrainer._check_resume_point(tmp_path, _resolved(0), point) is point
    reason = TaskTrainer._training_complete_reason(_point(start=2, wait=0), _resolved(0))
    assert reason == "after epoch 2 EarlyStopping had waited 0 epochs (patience 0) and had decided to stop"
    # Routed, not refused for "already decided to stop": the point holds no epochs in its sidecar, which
    # only the reporting-only checks look at.
    with pytest.raises(ConfigError, match="number of epochs that ran is unknown"):
        TaskTrainer._check_resume_point(tmp_path, _resolved(0), _point(start=2, wait=0))
    # The same rule at an ordinary patience: complete once the wait reaches it, not below it.
    assert "EarlyStopping had waited 3 epochs (patience 3)" in TaskTrainer._training_complete_reason(
        _point(start=5, wait=3), _resolved(3))
    with pytest.raises(ConfigError, match="number of epochs that ran is unknown"):
        TaskTrainer._check_resume_point(tmp_path, _resolved(3), _point(start=5, wait=3))
    assert TaskTrainer._training_complete_reason(_point(start=5, wait=2), _resolved(3)) is None
    assert TaskTrainer._check_resume_point(tmp_path, _resolved(3), _point(start=5, wait=2))


def test_every_way_a_run_has_no_epoch_left_has_a_reason_and_only_those(tmp_path):
    """The three states (D-064): ``fit_finished``; every epoch ended; EarlyStopping decided."""
    from types import SimpleNamespace

    from train.api.trainer import TaskTrainer

    complete = TaskTrainer._training_complete_reason
    assert complete(_point(start=0, wait=0), _resolved(3)) is None
    assert complete(_point(start=8, wait=0), _resolved(3)) is None
    assert complete(_point(start=9, wait=0), _resolved(3)) == "all 9 of its epochs ended"
    assert complete(_point(start=10, wait=0), _resolved(3)) == "all 9 of its epochs ended"
    finished = SimpleNamespace(start=3, wait=0, best=None, state={"initial": {}, "fit_finished": True})
    assert complete(finished, _resolved(3)) == "its fit returned (fit_state.json says fit_finished)"
    # Not finished: a falsy flag is not a state.
    unfinished = SimpleNamespace(start=3, wait=0, best=None, state={"fit_finished": False})
    assert complete(unfinished, _resolved(3)) is None


def test_a_backup_that_settles_on_a_complete_point_is_refused_not_entered_as_a_normal_resume(
    tmp_path, monkeypatch
):
    """``_train`` chose a NORMAL resume (the latest pair said an epoch was left); if the previous
    pair, re-read, says there is none, the files disagree: the back door into the reporting-only
    path (which `_train` did not take, so it would run ``fit`` with ``initial_epoch >= epochs``) is shut."""
    from train.api import trainer as module
    from train.api.trainer import TaskTrainer

    monkeypatch.setattr("train.common.resume_state.settle_backup", lambda directory, steps: ("previous", 9))
    monkeypatch.setattr(module, "load_resume_point", lambda directory, config: _point(start=9, wait=0))
    monkeypatch.setattr(TaskTrainer, "_check_resume_point", staticmethod(lambda d, r, p: p))
    trainer = TaskTrainer.__new__(TaskTrainer)
    trainer.config = None
    with pytest.raises(ConfigError, match=r"previous one records epoch 9, where all 9 of its epochs ended"):
        trainer._settle_backup(tmp_path, _resolved(3), _point(start=10, wait=0), 4)
    # A previous pair that still has an epoch left is the ordinary fallback, returned as read.
    monkeypatch.setattr(module, "load_resume_point", lambda directory, config: _point(start=7, wait=0))
    assert trainer._settle_backup(tmp_path, _resolved(3), _point(start=8, wait=0), 4).start == 7
