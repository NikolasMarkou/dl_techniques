"""The model and task registries hold strings and import nothing until asked.

The registries are exercised with a synthetic module written to ``tmp_path``,
so no real family is imported (that would load TensorFlow). The module name is
unique, so "was it imported" is answered by ``sys.modules``.
"""

import sys
import textwrap

import pytest

import train.api
from train.api import cli, registry
from train.api.errors import (
    ConfigError,
    ConfigValidationError,
    DLTrainError,
    UnknownModelError,
)

MODULE = "fake_train_api_registry_module"

SOURCE = '''
import dataclasses
from train.api.config import StrictConfig


class Family:
    MODEL_VARIANTS = {"small": 1, "tiny": 2}


def make_model(variant):
    return ("model", variant)


@dataclasses.dataclass(frozen=True)
class FakeConfig(StrictConfig):
    count: int = dataclasses.field(default=1, metadata={"help": "a count"})
    name: str = dataclasses.field(default="x", metadata={"help": "a name"})


CALLS = []


@dataclasses.dataclass
class Result:
    success: bool = True


def run_fake(**kwargs):
    CALLS.append(kwargs)
    return Result(success=True)
'''


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    """Write the synthetic module, make it importable, and forget it afterwards."""
    (tmp_path / f"{MODULE}.py").write_text(textwrap.dedent(SOURCE))
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(MODULE, None)
    yield MODULE
    sys.modules.pop(MODULE, None)


@pytest.fixture
def empty_registries(monkeypatch):
    models, tasks = {}, {}
    monkeypatch.setattr(registry, "MODELS", models)
    monkeypatch.setattr(registry, "TASKS", tasks)
    return models, tasks


def model_spec(family="fam", tasks=("fake_task",), module=MODULE):
    return registry.ModelSpec(
        family=family,
        factory=f"{module}:make_model",
        variants_attr=f"{module}:Family.MODEL_VARIANTS",
        tasks=tasks,
    )


def task_spec(name="fake_task", module=MODULE):
    return registry.TaskSpec(
        name=name,
        config_class=f"{module}:FakeConfig",
        function=f"{module}:run_fake",
        summary="a synthetic task",
    )


# ---------------------------------------------------------------------
# Laziness
# ---------------------------------------------------------------------


def test_storing_a_spec_imports_nothing(fake_module, empty_registries):
    registry.register_model(model_spec())
    registry.register_task(task_spec())
    assert fake_module not in sys.modules


def test_the_loaders_import_on_first_use(fake_module, empty_registries):
    spec = registry.register_model(model_spec())
    assert fake_module not in sys.modules
    assert dict(spec.load_variants()) == {"small": 1, "tiny": 2}
    assert fake_module in sys.modules
    assert spec.load_factory()("tiny") == ("model", "tiny")
    assert spec.load_resolver() is None
    task = registry.register_task(task_spec())
    assert task.load_config_class().__name__ == "FakeConfig"
    assert callable(task.load_function())


def test_a_resolver_string_is_loaded_when_set(fake_module):
    spec = registry.ModelSpec(
        family="fam",
        factory=f"{fake_module}:make_model",
        variants_attr=f"{fake_module}:Family.MODEL_VARIANTS",
        tasks=("fake_task",),
        resolve_model_args=f"{fake_module}:make_model",
    )
    assert spec.load_resolver()("v") == ("model", "v")


def test_registering_twice_is_a_bug_not_a_user_error(empty_registries):
    registry.register_model(model_spec())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_model(model_spec())
    registry.register_task(task_spec())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_task(task_spec())
    assert not issubclass(ValueError, DLTrainError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family": ""},
        {"family": "a/b"},
        {"factory": "no_colon"},
        {"variants_attr": "mod:"},
        {"tasks": ()},
        {"tasks": ["list"]},
        {"resolve_model_args": "nope"},
    ],
    ids=lambda k: next(iter(k)) + "=" + repr(next(iter(k.values()))),
)
def test_a_malformed_model_spec_is_refused(kwargs):
    base = {
        "family": "fam",
        "factory": "m:f",
        "variants_attr": "m:C.V",
        "tasks": ("t",),
    }
    registry.ModelSpec(**base)  # the base itself is valid
    with pytest.raises(ValueError):
        registry.ModelSpec(**{**base, **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [{"name": "Bad"}, {"name": "has space"}, {"config_class": "x"}, {"function": "m:"}],
    ids=lambda k: next(iter(k)),
)
def test_a_malformed_task_spec_is_refused(kwargs):
    base = {"name": "t", "config_class": "m:C", "function": "m:f", "summary": "s"}
    registry.TaskSpec(**base)
    with pytest.raises(ValueError):
        registry.TaskSpec(**{**base, **kwargs})


# ---------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------


def test_a_good_name_splits(fake_module, empty_registries):
    registry.register_model(model_spec())
    assert registry.parse_model_name("fam/tiny") == ("fam", "tiny")


@pytest.mark.parametrize("name", ["fam", "fam/", "/tiny", "fam/tiny/x", ""])
def test_a_malformed_name_names_the_expected_form(name, fake_module, empty_registries):
    registry.register_model(model_spec())
    with pytest.raises(UnknownModelError, match="family/variant"):
        registry.parse_model_name(name)


def test_an_unknown_family_is_suggested(fake_module, empty_registries):
    registry.register_model(model_spec(family="fam"))
    with pytest.raises(UnknownModelError) as caught:
        registry.parse_model_name("fan/tiny")
    assert "Unknown model family 'fan'" in str(caught.value)
    assert "Did you mean 'fam'?" in str(caught.value)


def test_an_unknown_variant_is_suggested(fake_module, empty_registries):
    registry.register_model(model_spec())
    with pytest.raises(UnknownModelError) as caught:
        registry.parse_model_name("fam/tinny")
    message = str(caught.value)
    assert "Unknown variant 'tinny' for family 'fam'" in message
    assert "Did you mean 'tiny'?" in message


def test_an_unknown_name_with_no_close_match_makes_no_suggestion(
    fake_module, empty_registries
):
    registry.register_model(model_spec())
    with pytest.raises(UnknownModelError) as caught:
        registry.parse_model_name("zzzzzz/tiny")
    assert "Did you mean" not in str(caught.value)


def test_the_message_prints_unquoted(fake_module, empty_registries):
    registry.register_model(model_spec())
    with pytest.raises(UnknownModelError) as caught:
        registry.parse_model_name("fam/nope")
    text = str(caught.value)
    # KeyError.__str__ would give repr(arg): a leading quote and escaped text.
    assert text.startswith("Unknown variant")
    assert not text.startswith(("'", '"'))
    assert "\\" not in text


def test_the_exception_bases_keep_their_python_meaning():
    assert issubclass(ConfigError, ValueError)
    assert issubclass(ConfigValidationError, ConfigError)
    with pytest.raises(ValueError):
        raise ConfigError("x")
    with pytest.raises(KeyError):
        raise UnknownModelError("x")
    with pytest.raises(DLTrainError):
        raise UnknownModelError("x")


# ---------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------


def test_list_models_is_sorted_and_filters_by_task(fake_module, empty_registries):
    registry.register_model(model_spec(family="zeta", tasks=("fake_task",)))
    registry.register_model(model_spec(family="alpha", tasks=("fake_task", "other")))
    registry.register_model(model_spec(family="mid", tasks=("other",)))
    registry.register_task(task_spec())
    assert registry.list_models() == [
        "alpha/small", "alpha/tiny",
        "mid/small", "mid/tiny",
        "zeta/small", "zeta/tiny",
    ]
    assert registry.list_models("fake_task") == [
        "alpha/small", "alpha/tiny", "zeta/small", "zeta/tiny",
    ]


def test_list_models_with_an_unknown_task_is_a_config_error(empty_registries):
    registry.register_task(task_spec())
    with pytest.raises(ConfigError, match="Unknown task 'fake_taks'.*Did you mean 'fake_task'"):
        registry.list_models("fake_taks")


def test_list_tasks_is_sorted(empty_registries):
    registry.register_task(task_spec("zebra"))
    registry.register_task(task_spec("apple"))
    assert [t.name for t in registry.list_tasks()] == ["apple", "zebra"]


def test_no_registered_task_shadows_a_command():
    assert set(registry.TASKS).isdisjoint(cli.COMMANDS)


# ---------------------------------------------------------------------
# The command line over the registries
# ---------------------------------------------------------------------


def test_list_tasks_prints_each_registered_task(fake_module, empty_registries, capsys):
    registry.register_task(task_spec())
    assert cli.main(["list_tasks"]) == 0
    out = capsys.readouterr().out
    assert out.split()[:1] == ["fake_task"] and "a synthetic task" in out


def test_list_models_prints_names_and_filters(fake_module, empty_registries, capsys):
    registry.register_model(model_spec(family="fam", tasks=("fake_task",)))
    registry.register_model(model_spec(family="other", tasks=("else",)))
    registry.register_task(task_spec())
    assert cli.main(["list_models"]) == 0
    assert capsys.readouterr().out.split() == [
        "fam/small", "fam/tiny", "other/small", "other/tiny",
    ]
    assert cli.main(["list_models", "task=fake_task"]) == 0
    assert capsys.readouterr().out.split() == ["fam/small", "fam/tiny"]
    assert cli.main(["list_models", "help"]) == 0
    out = capsys.readouterr().out
    assert "usage:" not in out and "fam/tiny" in out


def test_list_models_with_nothing_registered_says_so(empty_registries, capsys):
    assert cli.main(["list_models"]) == 0
    assert "no models are registered" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args, fragment",
    [
        (["task=nope"], "Unknown task 'nope'"),
        (["tsak=x"], "Unknown key 'tsak'. Did you mean 'task'?"),
        (["config=a.yaml"], "takes only task="),
        (["task.x=1"], "not a nested key"),
    ],
)
def test_list_models_refuses_bad_arguments(args, fragment, empty_registries, capsys):
    assert cli.main(["list_models", *args]) == 2
    assert fragment in capsys.readouterr().err


def test_a_registered_task_runs_through_the_cli(fake_module, empty_registries, capsys):
    registry.register_task(task_spec())
    assert cli.main(["fake_task", "count=3", "name=no"]) == 0
    module = sys.modules[fake_module]
    assert module.CALLS == [{"count": 3, "name": "no"}]


def test_a_registered_task_reports_a_bad_value_with_exit_2(
    fake_module, empty_registries, capsys
):
    registry.register_task(task_spec())
    assert cli.main(["fake_task", "count=abc"]) == 2
    err = capsys.readouterr().err
    assert "Invalid value for 'count': expected int, got 'abc' (str)" in err
    assert sys.modules[fake_module].CALLS == []


@pytest.mark.parametrize("args", [[], ["help"], ["-h"], ["--help"]], ids=str)
def test_a_task_command_prints_its_keys_instead_of_running(
    args, fake_module, empty_registries, capsys
):
    registry.register_task(task_spec())
    assert cli.main(["fake_task", *args]) == 0
    out = capsys.readouterr().out
    assert "usage: python -m train fake_task" in out
    assert "count  (int)  a count [default: 1]" in out
    assert "name  (str)  a name [default: 'x']" in out
    assert sys.modules[fake_module].CALLS == []


def test_help_and_unknown_command_know_the_registered_tasks(
    fake_module, empty_registries, capsys
):
    registry.register_task(task_spec())
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    assert "tasks:\n  fake_task  a synthetic task" in out
    assert cli.main(["fake_tsk"]) == 2
    assert "Did you mean 'fake_task'?" in capsys.readouterr().err


# ---------------------------------------------------------------------
# The lazy package surface
# ---------------------------------------------------------------------


def test_dir_lists_the_surface_and_every_name_resolves():
    assert set(train.api.__all__) <= set(dir(train.api))
    assert len(train.api.__all__) >= 20  # floor: the surface is not empty
    for name in train.api.__all__:
        assert getattr(train.api, name) is not None, name


def test_an_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        train.api.nope
