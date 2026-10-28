"""The CliffordNet trainer's label smoothing is opt-in and off by default.

``--label-smoothing a`` compiles the shared ``train.convnext.common.build_loss``: at
0.0 the STOCK ``SparseCategoricalCrossentropy(from_logits=True)`` (the default path
stays the object every earlier run used), above 0 the smoothed sparse loss. A value
outside ``[0, 1)`` is a parser error, and the value reaches ``config.json``.
"""

import sys

import keras
import numpy as np
import pytest

from tests.classification_fakes import use_raw_from_unit

from train.convnext.common import SmoothedSparseCategoricalCrossentropy


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


def _run_main(monkeypatch, tmp_path, extra_argv, stop_at):
    """Run ``main()`` until ``stop_at`` (``create_callbacks`` or ``save_config_json``)."""
    from train.cliffordnet import train_cliffordnet as module

    seen = {}

    def fake_build_model(**_kwargs):
        seen["model"] = _tiny_classifier()
        return seen["model"]

    def fake_create_callbacks(**_kwargs):
        if stop_at == "create_callbacks":
            raise _Stop
        return [], str(tmp_path)

    def fake_save_config_json(config, _results_dir):
        seen["config"] = config
        raise _Stop

    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    use_raw_from_unit(monkeypatch, _tiny_cifar)
    monkeypatch.setattr(module, "build_model", fake_build_model)
    monkeypatch.setattr(module, "create_callbacks", fake_create_callbacks)
    monkeypatch.setattr(module, "save_config_json", fake_save_config_json)
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--dataset", "cifar10", "--epochs", "1", "--batch-size", "4",
         "--output-dir", str(tmp_path), *extra_argv],
    )
    with pytest.raises(_Stop):
        module.main()
    return seen


def test_the_default_keeps_the_stock_loss(monkeypatch, tmp_path):
    loss = _run_main(monkeypatch, tmp_path, [], "create_callbacks")["model"].loss
    assert type(loss) is keras.losses.SparseCategoricalCrossentropy
    assert loss.get_config()["from_logits"] is True


def test_a_positive_value_compiles_the_smoothed_loss(monkeypatch, tmp_path):
    seen = _run_main(
        monkeypatch, tmp_path, ["--label-smoothing", "0.1"], "create_callbacks")
    loss = seen["model"].loss
    assert isinstance(loss, SmoothedSparseCategoricalCrossentropy)
    assert loss.label_smoothing == pytest.approx(0.1)


@pytest.mark.parametrize("value,expected", [(None, 0.0), ("0.1", 0.1)])
def test_the_value_reaches_config_json(monkeypatch, tmp_path, value, expected):
    argv = [] if value is None else ["--label-smoothing", value]
    config = _run_main(monkeypatch, tmp_path, argv, "save_config_json")["config"]
    assert config["label_smoothing"] == pytest.approx(expected)


@pytest.mark.parametrize("value", ["1.0", "-0.1"])
def test_a_value_outside_zero_one_is_a_parser_error(monkeypatch, tmp_path, value):
    from train.cliffordnet import train_cliffordnet as module

    monkeypatch.setattr(
        sys, "argv", ["prog", "--output-dir", str(tmp_path), "--label-smoothing", value])
    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert excinfo.value.code == 2
