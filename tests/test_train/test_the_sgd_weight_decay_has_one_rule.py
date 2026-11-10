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

import argparse
import ast
import importlib
import inspect
import sys
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
    """Serves energy_transformer, beit, graph_energy_transformer, dino, colbert, qwen3_embeddings, bit_diffusion and mean_shift
    (D-049); each of those trainers also refuses first, before its run directory (``test_a_refused_decay_leaves_nothing_behind``)."""
    with pytest.raises(ValueError, match="train.common.build_optimizer"):
        build_optimizer(_shared_config("sgd", 1e-4), steps_per_epoch=10)
    adamw = build_optimizer(_shared_config("adamw", 0.03), steps_per_epoch=10)
    assert type(adamw).__name__ == "AdamW" and adamw.weight_decay == 0.03
    sgd = build_optimizer(_shared_config("sgd", 0.0), steps_per_epoch=10)
    assert type(sgd).__name__ == "SGD" and not sgd.weight_decay


class Reached(Exception):
    """Raised by a data or model factory spy: the trainer got past the point where the refusal must already have fired."""


def _reached(name):
    def spy(*args, **kwargs):
        raise Reached(name)
    return spy


def _trainer(monkeypatch, name):
    """Import the trainer module with its ``setup_gpu`` stubbed (the test sets no device)."""
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    return module


def _run_siamfc_family(name, data_factory):
    def run(monkeypatch, out):
        module = _trainer(monkeypatch, name)
        monkeypatch.setattr(module, data_factory, _reached(data_factory))
        module.main(["--optimizer", "sgd", "--weight-decay", "5e-4", "--output-dir", str(out), "--experiment-name", "refused"])
    return run


def _run_yolo12_coco(monkeypatch, out):
    """``--save-dir`` is the root under which ``main`` makes ``yolov12_<scale>_coco_pretrain_<timestamp>``."""
    module = _trainer(monkeypatch, "train.yolo12.train_coco")
    for name in ("COCODatasetBuilder", "create_coco_model_and_loss"):
        monkeypatch.setattr(module, name, _reached(name))
    monkeypatch.setattr(module, "check_dependencies", lambda: True)
    monkeypatch.setattr(sys, "argv", ["train_coco", "--optimizer", "sgd", "--weight-decay", "5e-4", "--save-dir", str(out)])
    module.main()


def _run_yolo12_multitask(monkeypatch, out):
    """``train_model`` makes ``<--output-dir>/yolov12_multitask_...``; ``main`` would turn the ValueError into
    ``sys.exit(1)``, so its training function is called."""
    module = _trainer(monkeypatch, "train.yolo12.train_multitask")
    for name in ("create_dataset_splits", "create_model_and_loss"):
        monkeypatch.setattr(module, name, _reached(name))
    module.train_model(argparse.Namespace(optimizer="sgd", weight_decay=5e-4, tasks=["detection"], scale="n",
                                          data_dir=str(out / "data"), patch_size=64, seed=0, output_dir=str(out)))


def _run_som_cifar(monkeypatch, out):
    """No CLI flag picks the optimizer (always adamw); a config built in code could ask for another and lost its decay."""
    module = _trainer(monkeypatch, "train.som_nd_soft.train_cifar")
    monkeypatch.setattr(module, "load_cifar_data", _reached("load_cifar_data"))
    module.train_cifar_som_autoencoder(module.CIFARSOMConfig(optimizer_type="sgd", weight_decay=5e-4, output_dir=str(out)))


def _run_cli(name, flags, spies, extra=lambda out: ()):
    """Run ``name``'s ``main`` on ``extra(out)`` and ``flags`` followed by the output root, with ``spies`` (its data and model
    factories) raising. ``flags`` ends with the trainer's output-root flag; a ``main`` without an ``argv`` reads ``sys.argv``."""
    def run(monkeypatch, out):
        module = _trainer(monkeypatch, name)
        for spy in spies:
            monkeypatch.setattr(module, spy, _reached(spy))
        tokens = [*extra(out), *flags, str(out)]
        if inspect.signature(module.main).parameters:
            module.main(tokens)
        else:
            monkeypatch.setattr(sys, "argv", [name, *tokens])
            module.main()
    return run


#: The CLI tokens of a refused run, by the spelling of the trainer's optimizer and output-root flags.
_SGD_OUTPUT_DIR = ("--optimizer", "sgd", "--weight-decay", "5e-4", "--output-dir")
_SGD_TYPE_OUTPUT_DIR = ("--optimizer-type", "sgd", "--weight-decay", "5e-4", "--output-dir")
_SGD_TYPE_OUTPUT_ROOT = ("--optimizer-type", "sgd", "--weight-decay", "5e-4", "--output-root")


def _a_tokenizer_checkpoint(out):
    """beit MIM's config refuses a missing ``--tokenizer-checkpoint``; an empty file beside the output root passes that check
    (``load_frozen_tokenizer``, which would read it, is a spy)."""
    path = out.parent / "tokenizer.keras"
    path.touch()
    return "--tokenizer-checkpoint", str(path)

#: Every trainer behind the shared ``train.common.optimizer.build_optimizer``, and dit (its own ``build_dit_optimizer``):
#: both builders refuse, but only after the run directory or the data exist, so each trainer refuses first (step 13.9.2).
#: mean_shift is the one other caller of the shared builder and is not here: its ``Recipe.for_epochs`` is the only recipe
#: any entry builds, always adamw (pinned by ``tests/test_train/test_mean_shift/test_cells.py``), and no flag changes it.
_SHARED_BUILDER_TRAINERS = [
    ("train.beit.train_classification", _SGD_OUTPUT_DIR, ("build_datasets", "create_beit_classifier")),
    ("train.beit.train_mim", _SGD_OUTPUT_DIR, ("load_frozen_tokenizer", "build_datasets", "create_beit_mim"),
     _a_tokenizer_checkpoint),
    ("train.beit.train_tokenizer", _SGD_OUTPUT_DIR, ("build_datasets", "build_tokenizer")),
    ("train.energy_transformer.train_classification", _SGD_OUTPUT_DIR,
     ("build_datasets", "create_energy_transformer_classifier")),
    ("train.energy_transformer.train_masked_completion", _SGD_OUTPUT_DIR,
     ("build_datasets", "create_energy_transformer_mim")),
    ("train.graph_energy_transformer.train_anomaly", _SGD_OUTPUT_DIR, ("build_datasets", "create_graph_anomaly_detector")),
    ("train.graph_energy_transformer.train_classification", _SGD_OUTPUT_DIR, ("build_datasets", "create_graph_classifier")),
    ("train.dino.train_dino", _SGD_OUTPUT_DIR, ("build_dataset", "build_model_and_loss")),
    ("train.bit_diffusion.train_bit_diffusion", _SGD_TYPE_OUTPUT_DIR, ("build_sde", "create_model", "create_datasets")),
    ("train.dit.train_dit", _SGD_TYPE_OUTPUT_DIR, ("create_model", "create_datasets")),
    ("train.language.colbert.train_colbert_v1", _SGD_TYPE_OUTPUT_ROOT, ("build_datasets", "build_model")),
    ("train.language.colbert.train_colbert_v2", _SGD_TYPE_OUTPUT_ROOT, ("build_datasets", "build_model")),
    ("train.qwen3_embeddings.train_qwen3_embedding", _SGD_TYPE_OUTPUT_ROOT,
     ("build_embedding_datasets", "build_embedding_model")),
    ("train.qwen3_embeddings.train_qwen3_reranker", _SGD_TYPE_OUTPUT_ROOT,
     ("build_reranker_datasets", "build_reranker_model")),
]


@pytest.mark.parametrize("run", [
    *[pytest.param(_run_cli(*row), id=row[0].rsplit(".", 2)[-2] + "_" + row[0].rsplit(".", 1)[-1])
      for row in _SHARED_BUILDER_TRAINERS],
    pytest.param(_run_siamfc_family("train.siamfc.train_siamfc", "create_pair_dataset"), id="siamfc"),
    pytest.param(_run_siamfc_family("train.dasiamrpn.train_dasiamrpn", "create_pair_dataset"), id="dasiamrpn"),
    pytest.param(_run_siamfc_family("train.deepsort.train_deepsort", "load_split_arrays"), id="deepsort"),
    pytest.param(_run_siamfc_family("train.mambalct.train_mambalct", "create_clip_dataset"), id="mambalct"),
    pytest.param(_run_yolo12_coco, id="yolo12_train_coco"),
    pytest.param(_run_yolo12_multitask, id="yolo12_train_multitask"),
    pytest.param(_run_som_cifar, id="som_nd_soft_train_cifar"),
])
def test_a_refused_decay_leaves_nothing_behind(monkeypatch, tmp_path, run):
    """``--optimizer sgd --weight-decay 5e-4`` is refused BEFORE the run directory, the data or the model exist (review-iter-2-pass2
    NOTE 6: the refusal used to fire after ``prepare_run_dir`` and the model build, leaving a run directory in ``results/`` with a
    config of a run that never happened). The output root is a fresh ``tmp_path`` directory, never the repo's ``results/``."""
    out = tmp_path / "out"
    out.mkdir()
    try:
        run(monkeypatch, out)
        raised = None
    except Exception as error:  # the type is asserted below, after the output root
        raised = error
    left = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    assert not left, f"a refused run left {left} in its output root (raised {raised!r})"
    assert isinstance(raised, ValueError) and "dropped without a word" in str(raised), f"expected the refusal, got {raised!r}"


#: Functions that gate a weight decay on AdamW and are deliberately not covered by the refusal, with the reason.
WAIVED_DECAY_GATES = {
    ("train/api/optim.py", "build_optimizer_and_schedule"):
        "the API hands an sgd decay to the MODEL as an L2 penalty (recipes._weight_decay_regularizer) and refuses it where a "
        "family builds none (segmentation_models.resolve_segmentation_args, D-047); the optimizer is given none on purpose",
}
_REFUSAL, _PENALTY = "refuse_a_dropped_weight_decay", "sgd_weight_decay_regularizer"


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
               or (isinstance(n, ast.Attribute) and "adamw" in n.attr.lower())
               or (isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.lower() == "adamw")
               for n in ast.walk(test))


def _decay_gates(fn) -> list:
    """``(gate, branch)`` of every ``if`` statement or conditional expression that tests for AdamW and carries ``weight_decay``
    in exactly one branch, in either order: ``if adamw: <decay>``, ``if not adamw: ... else: <decay>``, ``wd if adamw else None``
    and ``None if not adamw else wd``. ``branch`` is the side that carries the decay."""
    gates = []
    for node in ast.walk(fn):
        if isinstance(node, (ast.If, ast.IfExp)) and _tests_for_adamw(node.test):
            body, orelse = (node.body, node.orelse) if isinstance(node, ast.If) else ([node.body], [node.orelse])
            in_body, in_else = _mentions(body, "weight_decay"), _mentions(orelse, "weight_decay")
            if in_body != in_else:
                gates.append((node, body if in_body else orelse))
    return gates


def _assigned(fn) -> dict:
    """``name -> [value]`` of the plain assignments in ``fn`` (``use_adamw = config.optimizer_type.lower() == "adamw"``)."""
    values = {}
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    values.setdefault(target.id, []).append(node.value)
    return values


def _receivers(node) -> set:
    """``id`` of every expression under ``node`` that is only the object a field or an item is read from: ``config`` in
    ``config.optimizer_type`` or ``args["optimizer"]``. A method's receiver is NOT one: in ``config.optimizer_type.lower()``
    the value tested is ``config.optimizer_type`` itself."""
    called = {id(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}
    return {id(n.value) for n in ast.walk(node)
            if isinstance(n, ast.Subscript) or (isinstance(n, ast.Attribute) and id(n) not in called)}


def _expressions(nodes, values) -> set:
    """Source text of every value expression in ``nodes`` and in the value assigned to any name they read (one level),
    without the objects those values are read from (:func:`_receivers`): a refusal handed ``config`` is handed no value
    the gate tests, only the object it reads one from (review of step 13.9.1: ``refuse_a_dropped_weight_decay(config, ...)``
    was credited because ``config`` appears in ``config.optimizer_type == "adamw"``)."""
    found = set()
    for node in nodes:
        skip = _receivers(node)
        for n in ast.walk(node):
            if not isinstance(n, ast.expr) or id(n) in skip:
                continue
            found.add(ast.unparse(n))
            if isinstance(n, ast.Name):
                for value in values.get(n.id, ()):
                    inner = _receivers(value)
                    found |= {ast.unparse(m) for m in ast.walk(value) if isinstance(m, ast.expr) and id(m) not in inner}
    return found


def _is_the_value_of(argument, nodes, values) -> bool:
    """``argument`` reads the program (a literal such as ``"adamw"`` or ``0`` reads no name) and is, or names, an expression of
    ``nodes``."""
    if argument is None or not any(isinstance(n, ast.Name) for n in ast.walk(argument)):
        return False
    texts = {ast.unparse(argument)} | {ast.unparse(v) for v in values.get(getattr(argument, "id", None), ())}
    return bool(texts & _expressions(nodes, values))


def _argument(call, position, keyword):
    if len(call.args) > position:
        return call.args[position]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def _covers(call, gate, branch, values) -> bool:
    """The refusal is handed the gate's OWN optimizer value (read in its test) and its OWN decay (read in the decay branch);
    the sgd penalty is built from that decay. A call that names the function but is handed a literal, or another object's
    value, covers nothing (the review-iter-2-pass2 mutant ``refuse_a_dropped_weight_decay("adamw", ...)`` survived that)."""
    name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
    if name == _PENALTY:
        return _is_the_value_of(_argument(call, 0, "weight_decay"), branch, values)
    return name == _REFUSAL and _is_the_value_of(_argument(call, 1, "weight_decay"), branch, values) and \
        _is_the_value_of(_argument(call, 0, "optimizer_type"), [gate.test], values)


def uncovered_decay_gates(source: str) -> list:
    """Names of functions with the dropped-decay shape (see :func:`_decay_gates`) where some gate is neither refused
    (``refuse_a_dropped_weight_decay`` handed that gate's optimizer value and decay) nor turned into the sgd penalty."""
    found = []
    for fn in [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        values = _assigned(fn)
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
        if any(not any(_covers(c, gate, branch, values) for c in calls) for gate, branch in _decay_gates(fn)):
            found.append(fn.name)
    return found


SIAMFC = SRC / "train/siamfc/train_siamfc.py"
SIAMFC_REFUSAL = 'refuse_a_dropped_weight_decay(config.optimizer_type, config.weight_decay, trainer="siamfc.train_siamfc")'
SIAMFC_GATE = ('        "gradient_clipping_by_norm": config.gradient_clipping,\n    }\n    if use_adamw:\n'
               '        opt_config["weight_decay"] = config.weight_decay\n    elif')


def _mutated(text: str, *edits) -> str:
    for old, new in edits:
        assert text.count(old) == 1, f"the planted mutant no longer matches the trainer: {old[:70]!r}"
        text = text.replace(old, new)
    return text


def _siamfc_gate_as(decay_entry: str) -> tuple:
    """The two edits of review-iter-2-pass2 mutant ``139_siamfc_ifexp_no_call``: the refusal deleted, the decay moved into
    the dict as a conditional expression."""
    return ((f"    {SIAMFC_REFUSAL}\n", ""),
            (SIAMFC_GATE, SIAMFC_GATE.replace('    }\n    if use_adamw:\n        opt_config["weight_decay"] = config.weight_decay\n',
                                              f'        "weight_decay": {decay_entry},\n    }}\n    if False:\n        pass\n')))


@pytest.mark.parametrize("edits", [
    pytest.param(_siamfc_gate_as('config.weight_decay if config.optimizer_type == "adamw" else None'), id="ifexp_no_call"),
    pytest.param(_siamfc_gate_as('None if config.optimizer_type != "adamw" else config.weight_decay'), id="mirrored_ifexp"),
    pytest.param(((SIAMFC_REFUSAL, SIAMFC_REFUSAL.replace("(config.optimizer_type,", '("adamw",')),), id="call_hardcoded_adamw"),
    pytest.param(((SIAMFC_REFUSAL, SIAMFC_REFUSAL.replace("config.weight_decay,", "0,")),), id="call_hardcoded_decay"),
    pytest.param(((SIAMFC_REFUSAL, SIAMFC_REFUSAL.replace("(config.optimizer_type,", "(config,")),), id="call_handed_the_config"),
])
def test_the_decay_gate_scan_fails_the_siamfc_mutants_that_survived_review_iter_2_pass2(edits):
    """The real trainer passes; each mutant of it (two survived the 13.9 guard, D-063) is caught."""
    source = SIAMFC.read_text(encoding="utf-8")
    assert uncovered_decay_gates(source) == []
    assert uncovered_decay_gates(_mutated(source, *edits)) == ["train_siamfc"]


def test_the_decay_gate_scan_finds_a_planted_drop_and_skips_the_covered_shapes():
    planted = ("def build(config):\n    cfg = {}\n    if config.optimizer_type == 'adamw':\n"
               "        cfg['weight_decay'] = config.weight_decay\n    return cfg\n")
    refused = planted.replace("    cfg = {}", "    refuse_a_dropped_weight_decay(config.optimizer_type, config.weight_decay, trainer='t')"
                                              "\n    cfg = {}")
    aliased = ("def build(config):\n    use_adamw = config.optimizer_type.lower() == 'adamw'\n    wd = config.weight_decay\n"
               "    refuse_a_dropped_weight_decay(config.optimizer_type, wd, trainer='t')\n    if use_adamw:\n"
               "        return {'weight_decay': wd}\n")
    mirrored_if = ("def build(config):\n    if config.optimizer_type != 'adamw':\n        return {}\n    else:\n"
                   "        return {'weight_decay': config.weight_decay}\n")
    penalty = "def build(config, use_adamw):\n    return None if use_adamw else sgd_weight_decay_regularizer(config.weight_decay)\n"
    both_branches = ("def build(config):\n    if config.optimizer_type == 'adamw':\n        a = config.weight_decay\n"
                     "    else:\n        b = config.weight_decay\n")
    assert uncovered_decay_gates(planted) == ["build"]
    assert uncovered_decay_gates(mirrored_if) == ["build"]
    assert uncovered_decay_gates(refused) == []
    assert uncovered_decay_gates(aliased) == []
    assert uncovered_decay_gates(penalty) == []
    assert uncovered_decay_gates(both_branches) == []
    assert uncovered_decay_gates(planted.replace("    cfg = {}", "    refuse_a_dropped_weight_decay(1, 2, trainer='t')\n    cfg = {}")) \
        == ["build"], "a refusal handed literals covers nothing"
    assert uncovered_decay_gates(refused.replace("(config.optimizer_type,", "(other.optimizer_type,")) == ["build"], \
        "a refusal handed another object's optimizer covers nothing"
    # The optimizer argument is the VALUE the gate tests (or its one-assignment alias), not the object it reads one from.
    assert uncovered_decay_gates(refused.replace("(config.optimizer_type,", "(config,")) == ["build"], \
        "a refusal handed the config object covers nothing: config.optimizer_type is what the gate tests"
    assert uncovered_decay_gates(refused.replace("config.weight_decay, trainer", "config, trainer")) == ["build"], \
        "a refusal handed the config object as the decay covers nothing"
    by_item = ("def build(args):\n    refuse_a_dropped_weight_decay(args, args['weight_decay'], trainer='t')\n"
               "    if args['optimizer'] == 'adamw':\n        return {'weight_decay': args['weight_decay']}\n")
    assert uncovered_decay_gates(by_item) == ["build"], "the mapping an item is read from is not the item"
    assert uncovered_decay_gates(by_item.replace("(args, ", "(args['optimizer'], ")) == []
    aliased_object = aliased.replace("    use_adamw = config.", "    cfg = config\n    use_adamw = cfg.").replace(
        "(config.optimizer_type,", "(cfg.optimizer_type,")
    assert uncovered_decay_gates(aliased_object) == []
    assert uncovered_decay_gates(aliased_object.replace("(cfg.optimizer_type,", "(cfg,")) == ["build"], \
        "an alias of the config object is still the object"
    method_alias = ("def build(config):\n    opt = config.optimizer_type.lower()\n"
                    "    refuse_a_dropped_weight_decay(opt, config.weight_decay, trainer='t')\n"
                    "    if opt == 'adamw':\n        return {'weight_decay': config.weight_decay}\n")
    assert uncovered_decay_gates(method_alias) == [], "a method's receiver is the value tested, and its alias names it"


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
