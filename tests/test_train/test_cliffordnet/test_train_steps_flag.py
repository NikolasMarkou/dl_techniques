"""The CliffordNet trainer's ``--train-steps`` is an exact optimizer-step budget.

``--train-steps N --eval-every-steps K`` runs ``N // K`` virtual epochs of ``K`` steps over
a repeating train stream (the CIFAR epoch is longer than the budget, Imagenette's is
shorter, ImageNet's is 40x longer), and the cosine schedule spans N steps, not the real
epoch length. The flag excludes ``--epochs``, ``K`` must divide ``N``, and both values
reach ``config.json``.
"""

import sys

import keras
import numpy as np
import pytest
import tensorflow as tf

from tests.classification_fakes import use_raw_from_unit


class _Stop(Exception):
    """Raised by a stub once it has seen what the test needs."""


def _tiny_cifar(*_args, **_kwargs):
    rng = np.random.default_rng(0)
    x = rng.random((8, 32, 32, 3), dtype=np.float32)
    y = rng.integers(0, 10, size=(8,))
    return (x, y), (x, y), (32, 32, 3), 10


def _tiny_classifier(**_kwargs):
    inputs = keras.Input((32, 32, 3))
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    return keras.Model(inputs, keras.layers.Dense(10)(x))


def _argv(tmp_path, *extra):
    return ["prog", "--dataset", "cifar10", "--batch-size", "4",
            "--output-dir", str(tmp_path), *extra]


def _run_until_fit(monkeypatch, tmp_path, extra_argv):
    """Run ``main()`` up to ``model.fit`` and return what the trainer handed it."""
    from train.cliffordnet import train_cliffordnet as module

    seen = {}

    def fake_build_model(**_kwargs):
        model = _tiny_classifier()

        def fake_fit(x=None, **kwargs):
            seen["fit"] = kwargs
            seen["train_ds"] = x
            raise _Stop

        model.fit = fake_fit
        return model

    real_builder = module.learning_rate_schedule_builder

    def recording_builder(config):
        seen["schedule_config"] = config
        return real_builder(config)

    monkeypatch.setattr(module, "learning_rate_schedule_builder", recording_builder)
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    use_raw_from_unit(monkeypatch, _tiny_cifar)
    monkeypatch.setattr(module, "build_model", fake_build_model)
    monkeypatch.setattr(
        module, "create_callbacks", lambda **_kwargs: ([], str(tmp_path)))
    monkeypatch.setattr(module, "save_config_json",
                        lambda config, _dir: seen.update(config=config))
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, *extra_argv))
    with pytest.raises(_Stop):
        module.main()
    return seen


def test_the_budget_becomes_virtual_epochs_of_the_eval_interval(monkeypatch, tmp_path):
    seen = _run_until_fit(
        monkeypatch, tmp_path, ["--train-steps", "8", "--eval-every-steps", "4"])
    assert seen["fit"]["epochs"] == 2
    assert seen["fit"]["steps_per_epoch"] == 4
    # 2 real steps per epoch here: only a repeating stream can feed 8 steps.
    assert int(seen["train_ds"].cardinality()) == tf.data.INFINITE_CARDINALITY


def test_the_schedule_spans_the_budget_not_the_real_epoch(monkeypatch, tmp_path):
    seen = _run_until_fit(
        monkeypatch, tmp_path, ["--train-steps", "8", "--eval-every-steps", "4"])
    assert seen["schedule_config"]["decay_steps"] == 8


def test_the_default_eval_interval_is_250(monkeypatch, tmp_path):
    seen = _run_until_fit(monkeypatch, tmp_path, ["--train-steps", "1000"])
    assert (seen["fit"]["epochs"], seen["fit"]["steps_per_epoch"]) == (4, 250)


def test_the_epoch_path_is_unchanged(monkeypatch, tmp_path):
    seen = _run_until_fit(monkeypatch, tmp_path, ["--epochs", "3"])
    assert seen["fit"]["epochs"] == 3
    assert seen["fit"]["steps_per_epoch"] is None
    assert seen["config"]["train_steps"] is None


def test_the_values_reach_config_json(monkeypatch, tmp_path):
    seen = _run_until_fit(
        monkeypatch, tmp_path, ["--train-steps", "8", "--eval-every-steps", "4"])
    assert seen["config"]["train_steps"] == 8
    assert seen["config"]["eval_every_steps"] == 4


@pytest.mark.parametrize("extra", [
    ["--train-steps", "8", "--epochs", "2"],
    ["--train-steps", "10", "--eval-every-steps", "4"],
    ["--train-steps", "0"],
    ["--eval-every-steps", "4"],
])
def test_a_bad_combination_is_a_parser_error(monkeypatch, tmp_path, extra):
    from train.cliffordnet import train_cliffordnet as module

    monkeypatch.setattr(sys, "argv", _argv(tmp_path, *extra))
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert excinfo.value.code == 2
