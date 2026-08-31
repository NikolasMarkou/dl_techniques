"""A config field that nothing compiles is a dead knob.

``BaseTimeSeriesTrainingConfig.multistep_loss`` is opt-in and defaults to
``None``. Two things therefore need pinning, and they pull in opposite
directions:

*   with ``None``, every wired trainer must compile the EXACT loss it compiled
    before -- the feature must not perturb six working pipelines;
*   with a value set, the loss must actually reach ``model.compile`` -- the
    failure mode this repo has shipped before is a field that is declared,
    serialised into ``config.json``, and read by nothing.

The consumption arm is AST-based rather than a substring search over the
trainer source. ``"build_multistep_loss" in source`` would be satisfied by the
string appearing in a comment, a docstring, or a dead branch -- including the
comments this very change added. What is asserted instead is that each wired
trainer's ``_build_model`` contains a CALL whose attribute is
``build_multistep_loss``, and that the call's result is bound and used.

``tirex`` is deliberately NOT wired: ``train_tirex.py`` compiles ``QuantileLoss``
unconditionally and has no point-forecast branch to override. That absence is
asserted here too, so "tirex is missing" reads as a recorded decision rather
than as an oversight for someone to "finish".
"""

import ast
import inspect

import keras
import pytest

from dl_techniques.losses.multistep_loss import MultistepLoss
from train.common.timeseries import BaseTimeSeriesTrainingConfig

# ---------------------------------------------------------------------

WIRED = ("nbeats", "prism", "xlstm")
NOT_WIRED = ("tirex",)

# The ETS trainer is new, so `None` there means plain MSE rather than "keep a
# pre-existing loss". It is wired the same way and gets the same CLI arm.
CLI_WIRED = WIRED + ("ets",)


def _module(name):
    return __import__(f"train.time_series.{name}.train_{name}", fromlist=["*"])


def _trainer_class(name):
    module = _module(name)
    trainers = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and obj.__module__ == module.__name__
        and hasattr(obj, "_build_model")
    ]
    assert len(trainers) == 1, [t.__name__ for t in trainers]
    return trainers[0]


def _calls_build_multistep_loss(cls):
    """True iff ``_build_model`` CALLS ``...build_multistep_loss()``."""
    tree = ast.parse(inspect.getsource(cls._build_model).lstrip())
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_multistep_loss"
        for node in ast.walk(tree)
    )


# ---------------------------------------------------------------------
# The config knob itself
# ---------------------------------------------------------------------

def test_default_is_a_no_op():
    assert BaseTimeSeriesTrainingConfig().build_multistep_loss() is None


@pytest.mark.parametrize("aggregation", ["mseh", "tmse", "gtmse", "msce"])
def test_builds_the_named_aggregation(aggregation):
    config = BaseTimeSeriesTrainingConfig(multistep_loss=aggregation, multistep_h=6)
    loss = config.build_multistep_loss()
    assert isinstance(loss, MultistepLoss)
    assert loss.aggregation == aggregation
    assert loss.h == 6


def test_h_defaults_to_the_full_horizon():
    loss = BaseTimeSeriesTrainingConfig(multistep_loss="tmse").build_multistep_loss()
    assert loss.h is None


def test_a_typo_raises_at_config_time_not_at_compile_time():
    """Catching this after the data pipeline is built wastes the whole run."""
    with pytest.raises(ValueError, match="Unknown multistep_loss"):
        BaseTimeSeriesTrainingConfig(multistep_loss="gtmes")


def test_a_bad_h_raises():
    with pytest.raises(ValueError, match="multistep_h"):
        BaseTimeSeriesTrainingConfig(multistep_loss="tmse", multistep_h=0)


# ---------------------------------------------------------------------
# The trainers
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", WIRED)
def test_wired_trainers_consume_the_knob(name):
    assert _calls_build_multistep_loss(_trainer_class(name)), (
        f"train_{name}.py declares the field via BaseTimeSeriesTrainingConfig "
        f"but its _build_model never calls build_multistep_loss()."
    )


@pytest.mark.parametrize("name", NOT_WIRED)
def test_tirex_is_deliberately_not_wired(name):
    """RECORDED DECISION, not an omission.

    ``train_tirex.py`` compiles ``QuantileLoss`` unconditionally: there is no
    point-forecast branch for a multistep loss to replace. Wiring one would mean
    inventing a point path, which is a different change. If a point branch is
    ever added, this test goes red and the wiring should follow.
    """
    source = inspect.getsource(_trainer_class(name)._build_model)
    assert "QuantileLoss(" in source
    assert not _calls_build_multistep_loss(_trainer_class(name))


@pytest.mark.parametrize("name", WIRED)
def test_the_call_is_reachable_and_bound(name):
    """The call must feed a variable, not be evaluated and thrown away."""
    tree = ast.parse(inspect.getsource(_trainer_class(name)._build_model).lstrip())
    bound = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "build_multistep_loss"
    ]
    assert bound, f"train_{name}.py calls build_multistep_loss() but discards it."


# ---------------------------------------------------------------------
# The CLI -- a flag `main()` forgets to forward parses cleanly and does nothing
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", CLI_WIRED)
def test_the_cli_exposes_both_flags(name):
    args = _module(name).build_parser().parse_args(
        ["--multistep_loss", "gtmse", "--multistep_h", "6"]
    )
    assert args.multistep_loss == "gtmse"
    assert args.multistep_h == 6


@pytest.mark.parametrize("name", CLI_WIRED)
def test_the_cli_defaults_to_off(name):
    args = _module(name).build_parser().parse_args([])
    assert args.multistep_loss is None
    assert args.multistep_h is None


@pytest.mark.parametrize("name", CLI_WIRED)
def test_main_forwards_both_flags_to_the_config(name):
    """``main()`` lists every config field by hand.

    That is the whole failure mode: a flag added to the parser and forgotten in
    ``main()`` parses, validates, prints in ``--help`` -- and silently does
    nothing. Asserting the parser accepts it is NOT enough.
    """
    tree = ast.parse(inspect.getsource(_module(name).main).lstrip())
    forwarded = {
        keyword.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "args"
        and keyword.arg == keyword.value.attr
    }
    missing = {"multistep_loss", "multistep_h"} - forwarded
    assert not missing, f"train_{name}.py main() does not forward {sorted(missing)}"


@pytest.mark.parametrize("name", CLI_WIRED)
def test_the_cli_rejects_an_unknown_aggregation(name):
    with pytest.raises(SystemExit):
        _module(name).build_parser().parse_args(["--multistep_loss", "gtmes"])
