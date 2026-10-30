"""``train.api.config`` refuses what does not match, and says so key by key.

Python callers stay strict (invariant 2): ``"5"`` for an ``int`` field raises,
because strings are coerced only in the CLI layer. A bad config reports every
problem at once, one line per key. The tests use small local dataclasses; the
real groups arrive in ``args.py``.
"""

import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import pytest

from train.api import config as config_module
from train.api.config import (
    AUTO,
    DEFAULT_SEED,
    Auto,
    StrictConfig,
    assert_resolved,
    from_json,
    load_yaml_mapping,
    merge,
    to_json,
    validate,
)
from train.api.errors import (
    ConfigError,
    ConfigValidationError,
    UnresolvedAutoError,
)


@dataclasses.dataclass(frozen=True)
class Optim(StrictConfig):
    lr: Auto[float] = AUTO
    steps: int = 10
    nesterov: bool = False
    schedule: Literal["cosine_decay", "constant"] = "cosine_decay"
    clipnorm: Optional[float] = None
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclasses.dataclass(frozen=True)
class Top(StrictConfig):
    out: str = "auto"
    name: str = "run"
    optim_args: Optim = dataclasses.field(default_factory=Optim)
    model_args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class NeedsKey(StrictConfig):
    must: int
    optional: int = 1


def errors_of(cls, mapping):
    with pytest.raises(ConfigValidationError) as caught:
        validate(cls, mapping)
    return caught.value.errors


# ---------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------


def test_a_bad_float_names_its_dotted_key_in_the_documented_wording():
    assert errors_of(Top, {"optim_args": {"lr": "abc"}}) == (
        "Invalid value for 'optim_args.lr': expected float, got 'abc' (str)",
    )


def test_two_bad_keys_give_two_lines_in_one_error():
    with pytest.raises(ConfigValidationError) as caught:
        validate(Top, {"optim_args": {"lr": "abc", "steps": 2.5}, "name": 3})
    assert len(caught.value.errors) == 3
    assert str(caught.value) == "\n".join(caught.value.errors)
    joined = str(caught.value)
    for key in ("'optim_args.lr'", "'optim_args.steps'", "'name'"):
        assert key in joined


def test_a_literal_violation_names_the_choices():
    (line,) = errors_of(Top, {"optim_args": {"schedule": "cosin"}})
    assert line == (
        "Invalid value for 'optim_args.schedule': expected one of "
        "'cosine_decay', 'constant', got 'cosin' (str)"
    )


def test_an_unknown_key_names_its_dotted_path_and_suggests_a_sibling():
    (line,) = errors_of(Top, {"optim_args": {"lrr": 1.0}})
    assert line == "Unknown key 'optim_args.lrr'. Did you mean 'optim_args.lr'?"
    (top,) = errors_of(Top, {"optim_arg": {}})
    assert top == "Unknown key 'optim_arg'. Did you mean 'optim_args'?"


def test_an_unknown_key_with_no_close_sibling_has_no_suggestion():
    (line,) = errors_of(Top, {"zzzzzz": 1})
    assert line == "Unknown key 'zzzzzz'."


def test_a_missing_required_key_is_reported():
    assert errors_of(NeedsKey, {"optional": 2}) == (
        "Missing required key 'must'.",
    )
    assert validate(NeedsKey, {"must": 3}) == NeedsKey(must=3, optional=1)


def test_a_group_given_a_scalar_is_reported_at_the_group():
    (line,) = errors_of(Top, {"optim_args": 3})
    assert line.startswith("Invalid value for 'optim_args': expected a mapping")


# ---------------------------------------------------------------------
# Strict types
# ---------------------------------------------------------------------


def test_a_string_for_an_int_field_raises():
    (line,) = errors_of(Top, {"optim_args": {"steps": "5"}})
    assert line == (
        "Invalid value for 'optim_args.steps': expected int, got '5' (str)"
    )


def test_a_bool_for_an_int_field_raises():
    assert errors_of(Top, {"optim_args": {"steps": True}})


def test_a_bool_for_a_float_field_raises():
    assert errors_of(Top, {"optim_args": {"lr": True}})
    assert errors_of(Top, {"optim_args": {"clipnorm": False}})


def test_an_int_for_a_bool_field_raises():
    assert errors_of(Top, {"optim_args": {"nesterov": 1}})


def test_a_float_for_an_int_field_raises():
    assert errors_of(Top, {"optim_args": {"steps": 5.0}})


def test_an_int_for_a_float_field_becomes_a_float():
    built = validate(Top, {"optim_args": {"lr": 1, "clipnorm": 2}})
    assert built.optim_args.lr == 1.0 and type(built.optim_args.lr) is float
    assert type(built.optim_args.clipnorm) is float


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_float_is_refused(bad):
    assert errors_of(Top, {"optim_args": {"lr": bad}})
    assert errors_of(Top, {"optim_args": {"clipnorm": bad}})


def test_auto_is_accepted_only_where_the_annotation_admits_it():
    assert validate(Top, {"optim_args": {"lr": "auto"}}).optim_args.lr == AUTO
    assert errors_of(Top, {"optim_args": {"steps": "auto"}})
    assert errors_of(Top, {"optim_args": {"schedule": "auto"}})


def test_optional_accepts_none_and_a_tuple_field_takes_a_list():
    built = validate(Top, {"optim_args": {"clipnorm": None, "betas": [0.8, 0.9]}})
    assert built.optim_args.clipnorm is None
    assert built.optim_args.betas == (0.8, 0.9)
    assert errors_of(Top, {"optim_args": {"betas": [0.8]}})


def test_a_python_caller_may_pass_a_group_instance_and_it_is_checked():
    built = validate(Top, {"optim_args": Optim(lr=1)})
    assert built.optim_args.lr == 1.0
    assert errors_of(Top, {"optim_args": Optim(steps="5")})


def test_defaults_fill_missing_keys_and_a_free_dict_keeps_auto_as_data():
    built = validate(Top, {"model_args": {"depth": "auto", "x": [1, 2]}})
    assert built.optim_args == Optim()
    assert built.model_args == {"depth": "auto", "x": [1, 2]}


def test_a_non_mapping_top_level_is_refused():
    (line,) = errors_of(Top, [("name", "x")])
    assert line.startswith("Invalid config: expected a mapping of Top fields")


# ---------------------------------------------------------------------
# merge, assert_resolved, base class
# ---------------------------------------------------------------------


def test_merge_is_deep_and_mutates_nothing():
    base = {"a": {"x": 1, "y": [1]}, "b": 1}
    overrides = {"a": {"x": 2}, "c": {"d": 3}}
    snapshot = (json.dumps(base), json.dumps(overrides))
    merged = merge(base, overrides)
    assert merged == {"a": {"x": 2, "y": [1]}, "b": 1, "c": {"d": 3}}
    assert (json.dumps(base), json.dumps(overrides)) == snapshot
    merged["a"]["y"].append(2)
    merged["c"]["d"] = 9
    assert base["a"]["y"] == [1] and overrides["c"]["d"] == 3


def test_merge_replaces_a_mapping_with_a_scalar_and_back():
    assert merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}
    assert merge({"a": 5}, {"a": {"x": 1}}) == {"a": {"x": 1}}


def test_assert_resolved_lists_every_auto_field_by_dotted_path():
    config = validate(Top, {"optim_args": {"lr": "auto"}})
    with pytest.raises(UnresolvedAutoError) as caught:
        assert_resolved(config)
    assert "optim_args.lr" in str(caught.value)


def test_assert_resolved_passes_when_nothing_is_auto():
    assert_resolved(validate(Top, {"optim_args": {"lr": 0.1}}))


def test_a_plain_str_field_or_a_free_dict_holding_auto_is_not_unresolved():
    config = validate(
        Top,
        {"optim_args": {"lr": 0.1}, "out": "auto", "model_args": {"k": "auto"}},
    )
    assert config.out == "auto"
    assert_resolved(config)


def test_the_base_class_is_fieldless_and_resolve_auto_returns_self():
    assert dataclasses.fields(StrictConfig) == ()
    config = StrictConfig()
    assert config.resolve_auto(object()) is config
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.x = 1


# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------


def test_json_round_trip_equals_the_original():
    original = validate(
        Top,
        {
            "name": "r",
            "optim_args": {"lr": 3e-4, "clipnorm": 1, "betas": (0.8, 0.95)},
            "model_args": {"depth": 4},
        },
    )
    text = to_json(original)
    assert from_json(Top, text) == original
    assert json.loads(text)["optim_args"]["lr"] == 3e-4


def test_to_json_refuses_nan():
    config = Top(optim_args=Optim(lr=math.nan))
    with pytest.raises(ValueError):
        to_json(config)


def test_from_json_refuses_nan_garbage_and_unknown_keys():
    with pytest.raises(ConfigError):
        from_json(Top, '{"optim_args": {"lr": NaN}}')
    with pytest.raises(ConfigError):
        from_json(Top, "{not json")
    with pytest.raises(ConfigValidationError):
        from_json(Top, '{"nope": 1}')


# ---------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_yaml_reads_3e_minus_4_as_a_float_and_the_stock_loader_is_untouched(tmp_path):
    import yaml

    loaded = load_yaml_mapping(write(tmp_path, "optim_args:\n  lr: 3e-4\n  steps: 5\n"))
    assert loaded == {"optim_args": {"lr": 0.0003, "steps": 5}}
    assert type(loaded["optim_args"]["lr"]) is float
    assert type(loaded["optim_args"]["steps"]) is int
    # The control: the stock loader is YAML 1.1 and still returns a string.
    assert yaml.safe_load("3e-4") == "3e-4"
    assert validate(Top, loaded).optim_args.lr == 0.0003


def test_yaml_one_point_two_floats_and_the_usual_scalars(tmp_path):
    loaded = load_yaml_mapping(
        write(tmp_path, "a: 1E-3\nb: .5\nc: 2.\nd: -1.5e2\ne: abc\nf: null\ng: 7\n")
    )
    assert loaded == {
        "a": 0.001, "b": 0.5, "c": 2.0, "d": -150.0, "e": "abc", "f": None, "g": 7,
    }
    assert type(loaded["g"]) is int


def test_a_yaml_one_point_one_surprise_fails_loudly_in_validation(tmp_path):
    loaded = load_yaml_mapping(write(tmp_path, "name: no\n"))
    assert loaded == {"name": False}
    (line,) = errors_of(Top, loaded)
    assert line == "Invalid value for 'name': expected str, got False (bool)"


def test_a_yaml_non_mapping_top_level_is_refused(tmp_path):
    for text in ("- 1\n- 2\n", "3\n", ""):
        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_yaml_mapping(write(tmp_path, text))


def test_a_yaml_parse_error_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_yaml_mapping(write(tmp_path, "a: [1, 2\n"))


@pytest.mark.parametrize(
    "text, key",
    [
        ("lr: 1\nname: a\nlr: 2\n", "lr"),
        ("optim_args:\n  steps: 1\n  steps: 2\n", "steps"),
    ],
    ids=["top-level", "nested"],
)
def test_a_repeated_yaml_key_is_refused_instead_of_last_wins(tmp_path, text, key):
    # PyYAML alone keeps the last of two equal keys; the CLI refuses a repeated key.
    import yaml

    assert yaml.safe_load("a: 1\na: 2\n") == {"a": 2}
    with pytest.raises(ConfigError, match=f"duplicate key '{key}'"):
        load_yaml_mapping(write(tmp_path, text))
    # A merge key overriding a merged value is not a repeated key.
    merged = load_yaml_mapping(write(tmp_path, "base: &b {x: 1}\nuse:\n  <<: *b\n  x: 2\n"))
    assert merged["use"] == {"x": 2}


def test_an_unreadable_yaml_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read"):
        load_yaml_mapping(tmp_path / "missing.yaml")


def test_a_missing_pyyaml_names_the_train_extra(tmp_path, monkeypatch):
    path = write(tmp_path, "a: 1\n")
    config_module._yaml_loader.cache_clear()
    monkeypatch.setitem(sys.modules, "yaml", None)
    try:
        with pytest.raises(ConfigError, match=r"'train' extra"):
            load_yaml_mapping(path)
    finally:
        config_module._yaml_loader.cache_clear()


# ---------------------------------------------------------------------
# The TF-free seed copy
# ---------------------------------------------------------------------


def test_the_seed_copy_equals_the_real_default_and_the_literal():
    from train.common.seed import DEFAULT_SEED as real

    assert DEFAULT_SEED == real == 42
