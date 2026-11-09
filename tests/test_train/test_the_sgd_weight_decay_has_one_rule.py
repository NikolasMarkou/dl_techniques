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
from types import SimpleNamespace

import keras
import numpy as np
import pytest
import tensorflow as tf

import train.api.recipes as recipes
import train.rbf_protonet.train_rbf_protonet as train_rbf_protonet
import train.vit.train_vit as train_vit
from train.api.args import OptimArgs
from train.common.optimizer import build_optimizer, refuse_a_dropped_weight_decay, sgd_weight_decay_regularizer

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


# ---------------------------------------------------------------------------
# A decay that the optimizer would be handed no part of is REFUSED (D-063, step 13.9 of plan 98b866fb)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("optimizer_type", ["sgd", "SGD", "adam", "rmsprop"])
def test_a_positive_decay_beside_an_optimizer_that_takes_none_is_refused(optimizer_type):
    with pytest.raises(ValueError, match=r"dropped without a word.*--weight-decay 0.*--optimizer adamw"):
        refuse_a_dropped_weight_decay(optimizer_type, 5e-4, trainer="some.trainer")


@pytest.mark.parametrize("optimizer_type, decay", [
    ("adamw", 0.05), ("AdamW", 1e-4), ("sgd", 0.0), ("adam", None), ("rmsprop", -1.0), ("sgd", 0)])
def test_a_decay_the_optimizer_applies_or_no_decay_is_accepted(optimizer_type, decay):
    assert refuse_a_dropped_weight_decay(optimizer_type, decay, trainer="some.trainer") is None


def _shared_config(optimizer_type, decay):
    return SimpleNamespace(lr_schedule_type="cosine_decay", learning_rate=1e-3, epochs=2, warmup_epochs=0,
                           optimizer_type=optimizer_type, weight_decay=decay, gradient_clipping=None)


def test_the_shared_build_optimizer_refuses_what_it_would_drop_and_still_builds_adamw():
    """Serves energy_transformer, beit, graph_energy_transformer, dino, colbert, qwen3_embeddings and mean_shift (D-049)."""
    with pytest.raises(ValueError, match="train.common.build_optimizer"):
        build_optimizer(_shared_config("sgd", 1e-4), steps_per_epoch=10)
    adamw = build_optimizer(_shared_config("adamw", 0.03), steps_per_epoch=10)
    assert type(adamw).__name__ == "AdamW" and adamw.weight_decay == 0.03
    sgd = build_optimizer(_shared_config("sgd", 0.0), steps_per_epoch=10)
    assert type(sgd).__name__ == "SGD" and not sgd.weight_decay


#: Functions that gate a weight decay on AdamW and are deliberately not covered by the refusal, with the reason.
WAIVED_DECAY_GATES = {
    ("train/api/optim.py", "build_optimizer_and_schedule"):
        "the API hands an sgd decay to the MODEL as an L2 penalty (recipes._weight_decay_regularizer) and refuses it where a "
        "family builds none (segmentation_models.resolve_segmentation_args, D-047); the optimizer is given none on purpose",
}
_COVERING_CALLS = {"refuse_a_dropped_weight_decay", "sgd_weight_decay_regularizer"}


def _mentions(nodes, needle):
    for node in nodes:
        for n in ast.walk(node):
            if (isinstance(n, ast.Name) and needle in n.id) or (isinstance(n, ast.Attribute) and needle in n.attr) or \
                    (isinstance(n, ast.keyword) and n.arg and needle in n.arg) or \
                    (isinstance(n, ast.Constant) and isinstance(n.value, str) and needle in n.value):
                return True
    return False


def _tests_for_adamw(test):
    return any((isinstance(n, ast.Name) and "adamw" in n.id.lower())
               or (isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.lower() == "adamw")
               for n in ast.walk(test))


def uncovered_decay_gates(source: str) -> list:
    """Names of functions with the dropped-decay shape: ``if <adamw>:`` carries ``weight_decay`` and no other branch does,
    in a function that neither refuses (``refuse_a_dropped_weight_decay``) nor builds the sgd penalty."""
    found = []
    for fn in [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)]:
        shaped = any(isinstance(n, ast.If) and _tests_for_adamw(n.test) and _mentions(n.body, "weight_decay")
                     and not _mentions(n.orelse, "weight_decay") for n in ast.walk(fn))
        calls = {getattr(c.func, "id", None) or getattr(c.func, "attr", None) for c in ast.walk(fn) if isinstance(c, ast.Call)}
        if shaped and not (calls & _COVERING_CALLS):
            found.append(fn.name)
    return found


def test_the_decay_gate_scan_finds_a_planted_drop_and_skips_the_covered_shapes():
    planted = ("def build(config):\n    cfg = {}\n    if config.optimizer_type == 'adamw':\n"
               "        cfg['weight_decay'] = config.weight_decay\n    return cfg\n")
    refused = "from x import refuse_a_dropped_weight_decay\n" + planted.replace("    cfg = {}", "    refuse_a_dropped_weight_decay(1, 2, trainer='t')\n    cfg = {}")
    both_branches = ("def build(config):\n    if config.optimizer_type == 'adamw':\n        a = config.weight_decay\n"
                     "    else:\n        b = config.weight_decay\n")
    assert uncovered_decay_gates(planted) == ["build"]
    assert uncovered_decay_gates(refused) == []
    assert uncovered_decay_gates(both_branches) == []


def test_no_trainer_gates_its_weight_decay_on_adamw_without_refusing_the_rest():
    """The shape D-049 found in the six trainers that build their own optimizer, in ``dit`` and in the shared
    ``build_optimizer`` (every trainer behind it inherits the refusal): ``if adamw: opt_config['weight_decay'] = ...`` and nothing for the other
    optimizers, so ``--optimizer sgd --weight-decay 5e-4`` trained with no decay and said nothing."""
    offenders = []
    for top in ("train", "applications"):
        for path in sorted((SRC / top).rglob("*.py")):
            relative = path.relative_to(SRC).as_posix()
            for name in uncovered_decay_gates(path.read_text(encoding="utf-8")):
                if (relative, name) not in WAIVED_DECAY_GATES:
                    offenders.append(f"{relative}::{name}")
    assert not offenders, (
        "a function hands a weight decay to AdamW only and does neither refuse nor build the sgd penalty for the other "
        f"optimizers; call train.common.optimizer.refuse_a_dropped_weight_decay (or sgd_weight_decay_regularizer): {offenders}")
