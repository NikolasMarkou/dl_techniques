"""An SGD weight decay becomes ``L2(wd / 2)`` through ONE function, in every trainer that turns it into a Keras penalty.

Keras ``L2(l)`` adds ``l * sum(w^2)`` to the loss, whose gradient is ``2 * l * w``; the SGD weight decay of He et al. 2015 and of
``torch.optim.SGD(weight_decay=wd)`` adds ``wd * w`` to the gradient. ``L2(wd)`` is therefore twice the decay the number names
(measured in D-042 of plan 98b866fb: a real SGD step shrinks a weight 2.0000 times as far). Step 10e fixed the ResNet trainer and
the API recipes; ``train_vit`` and ``train_rbf_protonet`` kept ``L2(config.weight_decay)`` under ``--optimizer sgd`` until D-048.

What is pinned, each against a gradient and not against a spelling:

* the shared function ``train.common.optimizer.sgd_weight_decay_regularizer``: gradient ``wd`` at ``w = 1`` and ``wd * w`` at a
  random ``w``, for three decays, and no penalty for ``None``, zero or a negative number;
* the API rule ``recipes._weight_decay_regularizer`` hands the model that very penalty for sgd and none for adamw;
* the real ``train_vit`` and ``train_rbf_protonet`` bodies, stopped at the model factory: under sgd the ``kernel_regularizer`` they
  pass has the gradient ``wd``, under adamw they pass none (the decay is the optimizer's: a penalty as well would apply it twice);
* an AST guard over ``src/train`` and ``src/applications``: no module builds an ``L2`` from a name that mentions a decay except
  the shared function itself (the guard's own positive control is a planted source).
"""

import ast
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

import train.api.recipes as recipes
import train.rbf_protonet.train_rbf_protonet as train_rbf_protonet
import train.vit.train_vit as train_vit
from train.api.args import OptimArgs
from train.common.optimizer import sgd_weight_decay_regularizer

from tests.classification_fakes import use_tiny_raw

WEIGHT_DECAYS = (1e-4, 5e-4, 0.05)
SRC = Path(__file__).resolve().parents[2] / "src"
HOME = "train/common/optimizer.py"


def penalty_gradient(regularizer, weights: np.ndarray) -> np.ndarray:
    variable = tf.Variable(weights)
    with tf.GradientTape() as tape:
        penalty = regularizer(variable)
    return tape.gradient(penalty, variable).numpy()


def assert_is_the_sgd_weight_decay(regularizer, decay: float) -> None:
    assert regularizer is not None, "an sgd weight decay was dropped"
    ones = np.ones((4, 3), "float32")
    assert (penalty_gradient(regularizer, ones) == np.float32(decay)).all(), "L2(wd) has the gradient 2 * wd"
    weights = np.random.default_rng(0).normal(size=(4, 3)).astype("float32")
    assert (penalty_gradient(regularizer, weights) == np.float32(decay) * weights).all()


@pytest.mark.parametrize("decay", WEIGHT_DECAYS)
def test_the_shared_function_has_the_gradient_of_the_sgd_weight_decay(decay):
    assert_is_the_sgd_weight_decay(sgd_weight_decay_regularizer(decay), decay)


@pytest.mark.parametrize("decay", [None, 0, 0.0, -1e-4])
def test_no_decay_builds_no_penalty(decay):
    assert sgd_weight_decay_regularizer(decay) is None


@pytest.mark.parametrize("decay", WEIGHT_DECAYS)
def test_the_api_rule_hands_the_model_the_same_penalty_for_sgd_and_none_for_adamw(decay):
    api = recipes._weight_decay_regularizer(OptimArgs(optimizer="sgd", weight_decay=decay))
    assert_is_the_sgd_weight_decay(api, decay)
    assert keras.regularizers.serialize(api) == keras.regularizers.serialize(sgd_weight_decay_regularizer(decay))
    assert recipes._weight_decay_regularizer(OptimArgs(optimizer="adamw", weight_decay=decay)) is None


class Stopped(Exception):
    """Raised by the model-factory spy: the trainer's regularizer is known, nothing needs to train."""


def regularizer_handed_to(monkeypatch, tmp_path, module, factory_name, build_config, **settings):
    """Run the real trainer body up to its model factory and return the ``kernel_regularizer`` it passed."""
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        raise Stopped

    monkeypatch.setattr(module, factory_name, spy)
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    use_tiny_raw(monkeypatch, n_train=40, n_test=8)
    config = build_config(output_dir=str(tmp_path), experiment_name="decay", batch_size=4, epochs=1, **settings)
    with pytest.raises(Stopped):
        getattr(module, module.__name__.rsplit(".", 1)[1])(config)
    return seen["kernel_regularizer"]


def vit_config(**settings):
    return train_vit.TrainingConfig(dataset="cifar10", num_classes=10, model_variant="vit_pico", **settings)


def rbf_config(**settings):
    return train_rbf_protonet.TrainingConfig(num_classes=100, **settings)


TRAINERS = [
    pytest.param(train_vit, "create_vit", vit_config, id="train_vit"),
    pytest.param(train_rbf_protonet, "create_rbf_protonet", rbf_config, id="train_rbf_protonet"),
]


@pytest.mark.parametrize("decay", WEIGHT_DECAYS)
@pytest.mark.parametrize("module, factory_name, build_config", TRAINERS)
def test_a_trainer_under_sgd_hands_its_model_the_sgd_weight_decay(
        monkeypatch, tmp_path, module, factory_name, build_config, decay):
    regularizer = regularizer_handed_to(
        monkeypatch, tmp_path, module, factory_name, build_config, optimizer_type="sgd", weight_decay=decay)
    assert_is_the_sgd_weight_decay(regularizer, decay)


@pytest.mark.parametrize("module, factory_name, build_config", TRAINERS)
def test_a_trainer_under_adamw_hands_its_model_no_penalty(monkeypatch, tmp_path, module, factory_name, build_config):
    """The decay is the optimizer's; a penalty as well would apply it twice."""
    assert regularizer_handed_to(
        monkeypatch, tmp_path, module, factory_name, build_config, optimizer_type="adamw", weight_decay=0.05) is None


def decay_penalties(source: str) -> list:
    """``(line, call)`` of every ``L2`` / ``L1L2`` / ``l2`` construction whose arguments name a decay."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name in {"L2", "L1L2", "l2", "l1_l2"} and "decay" in ast.unparse(node).lower():
            found.append((node.lineno, ast.unparse(node)))
    return found


def test_the_guard_finds_a_planted_decay_penalty_and_ignores_other_strengths():
    assert decay_penalties("import keras\nr = keras.regularizers.L2(config.weight_decay)\n") == [
        (2, "keras.regularizers.L2(config.weight_decay)")]
    assert decay_penalties("r = L2(weight_decay / 2)\n") == [(1, "L2(weight_decay / 2)")]
    assert decay_penalties("r = keras.regularizers.L2(config.depthwise_l2)\nq = L2(1e-5)\n") == []


def test_no_module_builds_an_l2_from_a_weight_decay_except_the_shared_function():
    offenders = []
    for top in ("train", "applications"):
        for path in sorted((SRC / top).rglob("*.py")):
            relative = path.relative_to(SRC).as_posix()
            if relative == HOME:
                continue
            offenders += [f"{relative}:{line}: {call}" for line, call in decay_penalties(path.read_text(encoding="utf-8"))]
    assert not offenders, (
        "an L2 built straight from a weight decay is twice the SGD decay (or a second decay beside the optimizer's); "
        f"use train.common.optimizer.sgd_weight_decay_regularizer: {offenders}"
    )
