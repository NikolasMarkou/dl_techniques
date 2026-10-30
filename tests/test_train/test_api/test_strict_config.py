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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pytest

from train.api import config as config_module
from train.api.cli import coerce
from train.api.config import (
    AUTO,
    DEFAULT_SEED,
    Auto,
    StrictConfig,
    assert_resolved,
    from_json,
    load_yaml_mapping,
    merge,
    parse_int,
    parse_json,
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


@dataclasses.dataclass(frozen=True)
class Inner(StrictConfig):
    lr: Auto[float] = AUTO
    n: int = 1


@dataclasses.dataclass(frozen=True)
class Mid(StrictConfig):
    inner: Inner = dataclasses.field(default_factory=Inner)
    k: int = 1


@dataclasses.dataclass(frozen=True)
class Containers(StrictConfig):
    """Every shape in which an AUTO field can hide below the top level."""

    one: Inner = dataclasses.field(default_factory=lambda: Inner(lr=1.0))
    either: Union[Inner, Mid, None] = None
    floats: List[Auto[float]] = dataclasses.field(default_factory=list)
    by_name: Dict[str, Auto[float]] = dataclasses.field(default_factory=dict)
    groups: List[Inner] = dataclasses.field(default_factory=list)
    pair: Tuple[Auto[int], int] = (1, 1)
    free: Dict[str, Any] = dataclasses.field(default_factory=dict)
    text: str = "auto"


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


def test_an_empty_mapping_over_a_mapping_clears_nothing():
    # What the task help promises about ``model_args={}`` (D-034).
    assert merge({"model_args": {"a": 1}}, {"model_args": {}}) == {"model_args": {"a": 1}}
    assert merge({"model_args": {"a": 1}}, {"model_args": None}) == {"model_args": None}


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


# ---------------------------------------------------------------------
# YAML integers follow the command line, not YAML 1.1 (D-034)
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Count(StrictConfig):
    epochs: int = 1


@dataclasses.dataclass(frozen=True)
class Label(StrictConfig):
    name: str = "x"


def test_yaml_one_point_one_ints_are_not_silently_another_number(tmp_path):
    import yaml

    # The control: the stock loader gives these three different numbers.
    assert yaml.safe_load("a: 010")["a"] == 8
    assert yaml.safe_load("a: 1:30")["a"] == 90
    assert yaml.safe_load("a: 1_000")["a"] == 1000
    assert load_yaml_mapping(write(tmp_path, "a: 010\n"))["a"] == 10


@pytest.mark.parametrize("text", ["010", "007", "0", "-5", "42", "1:30", "1_000", "0x10", "0o17", "+5", "0b11", "5.0"])
def test_a_yaml_int_means_what_the_command_line_means_or_is_refused(tmp_path, text):
    from_cli = None
    try:
        from_cli = coerce(int, text, "epochs")
    except ConfigError:
        pass
    loaded = load_yaml_mapping(write(tmp_path, f"epochs: {text}\n"))
    try:
        from_yaml = validate(Count, loaded).epochs
    except ConfigValidationError:
        from_yaml = None
    # Either door refuses, or both give the same number; neither gives another one.
    if from_yaml is not None:
        assert from_cli == from_yaml, (text, from_cli, from_yaml)
    expected_ints = {"010": 10, "007": 7, "0": 0, "-5": -5, "42": 42}
    if text in expected_ints:
        assert from_yaml == expected_ints[text] and type(from_yaml) is int
    else:
        assert from_yaml is None, (text, from_yaml)


@pytest.mark.parametrize("text", ["1:30", "1_000", "0x10"])
def test_the_same_text_stays_text_in_a_str_field(tmp_path, text):
    loaded = load_yaml_mapping(write(tmp_path, f"name: {text}\n"))
    assert validate(Label, loaded).name == text


def test_an_explicit_int_tag_on_a_non_decimal_is_a_config_error_not_a_traceback(tmp_path):
    for text in ("!!int 0x10", "!!int 1_000", "!!int abc"):
        with pytest.raises(ConfigError, match="not a decimal integer"):
            load_yaml_mapping(write(tmp_path, f"epochs: {text}\n"))


def test_the_yaml_int_pattern_is_the_one_the_command_line_uses():
    from train.api import cli

    assert config_module.INT_PATTERN == r"-?[0-9]+"
    assert cli.parse_int is config_module.parse_int


def test_a_huge_digit_string_is_a_refusal_in_every_door(tmp_path):
    digits = "9" * 5000
    assert parse_int(digits) == (None, ": too many digits")
    assert parse_int("12") == (12, None)
    assert parse_int("1_0") == (None, "")
    with pytest.raises(ConfigError, match="too many digits"):
        load_yaml_mapping(write(tmp_path, f"epochs: {digits}\n"))


def test_yaml_with_bytes_that_are_not_utf_8_is_a_config_error(tmp_path):
    path = tmp_path / "binary.yaml"
    path.write_bytes(b"out: \xff\xfe\n")
    with pytest.raises(ConfigError, match="Cannot read config file"):
        load_yaml_mapping(path)


# ---------------------------------------------------------------------
# JSON refuses a repeated key (D-034)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ['{"lr": 1, "lr": 2}', '{"a": {"b": 1, "b": 2}}', '[{"x": 1, "x": 1}]'],
    ids=["top", "nested", "in-a-list"],
)
def test_parse_json_refuses_a_repeated_key(text):
    value, reason = parse_json(text)
    assert value is None
    assert "found a duplicate key" in reason


def test_a_repeated_key_in_a_config_file_text_is_a_config_error():
    with pytest.raises(ConfigError, match="duplicate key 'lr'"):
        from_json(Inner, '{"lr": 1, "lr": 2}')
    assert from_json(Inner, '{"lr": 1, "n": 2}') == Inner(lr=1.0, n=2)


def test_deep_nesting_comes_back_as_a_value_not_a_traceback():
    value, reason = parse_json("[" * 100000)
    assert value is None
    assert "nested too deeply" in reason


# ---------------------------------------------------------------------
# Free dicts: finite numbers, string keys, a private copy (D-034)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_free_dict_refuses_a_non_finite_number_at_any_depth(bad):
    errors = errors_of(
        Top,
        {"model_args": {"top": bad, "deep": {"deeper": [1, {"x": bad}]}, "ok": 1.5}},
    )
    assert len(errors) == 2, errors
    assert errors[0].startswith("Invalid value for 'model_args.top': expected a finite number")
    assert errors[1].startswith(
        "Invalid value for 'model_args.deep.deeper[1].x': expected a finite number"
    )


def test_a_free_dict_with_a_finite_number_still_validates_and_a_nan_does_not_reach_json():
    top = validate(Top, {"model_args": {"depth": 4, "rate": 0.5, "nested": [1.5, 2]}})
    assert from_json(Top, to_json(top)) == top
    with pytest.raises(ConfigValidationError):
        validate(Top, {"model_args": {"rate": math.nan}})


def test_a_free_dict_refuses_a_non_string_key_at_any_depth():
    errors = errors_of(Top, {"model_args": {1: 2, "a": {3: 4}}})
    assert len(errors) == 2, errors
    assert errors[0] == "Invalid value for 'model_args.1': expected str, got 1 (int)"
    assert errors[1] == "Invalid key 'model_args.a.3': keys must be strings, got 3 (int)"


def test_a_non_string_top_level_key_is_refused_with_the_word_string():
    (line,) = errors_of(Top, {1: 2})
    assert line == "Invalid key '1': keys must be strings, got 1 (int)"


def test_validate_copies_a_free_dict_deeply():
    payload = {"a": [1, 2], "b": {"c": [3]}}
    top = validate(Top, {"model_args": payload})
    payload["a"].append(9)
    payload["b"]["c"].append(9)
    assert top.model_args == {"a": [1, 2], "b": {"c": [3]}}
    assert top.model_args is not payload and top.model_args["a"] is not payload["a"]


def test_validate_copies_an_any_field_too():
    @dataclasses.dataclass(frozen=True)
    class Loose(StrictConfig):
        anything: Any = None

    shared = [1, 2]
    loose = validate(Loose, {"anything": shared})
    shared.append(3)
    assert loose.anything == [1, 2]
    assert validate(Loose, {"anything": (1, 2)}).anything == (1, 2)
    assert errors_of(Loose, {"anything": [math.nan]})


def test_a_dict_annotation_with_a_non_string_key_is_a_programming_error():
    @dataclasses.dataclass(frozen=True)
    class IntKeys(StrictConfig):
        table: Dict[int, float] = dataclasses.field(default_factory=dict)

    with pytest.raises(TypeError, match="Dict key type must be str"):
        validate(IntKeys, {"table": {1: 2.0}})


# ---------------------------------------------------------------------
# assert_resolved walks containers; a union of groups tries each group (D-034)
# ---------------------------------------------------------------------


def unresolved(mapping):
    with pytest.raises(UnresolvedAutoError) as caught:
        assert_resolved(validate(Containers, mapping))
    return str(caught.value).split(": ", 1)[1].split(", ")


def test_assert_resolved_finds_auto_inside_lists_dicts_and_tuples():
    assert unresolved({"floats": [1.0, "auto"]}) == ["floats[1]"]
    assert unresolved({"by_name": {"a": 1.0, "b": "auto"}}) == ["by_name.b"]
    assert unresolved({"groups": [{"lr": 1.0}, {}]}) == ["groups[1].lr"]
    assert unresolved({"pair": ["auto", 2]}) == ["pair[0]"]
    assert unresolved({"either": {"lr": "auto"}}) == ["either.lr"]


def test_assert_resolved_reads_every_member_of_a_union_of_groups():
    assert unresolved({"either": {"inner": {}, "k": 2}}) == ["either.inner.lr"]


def test_assert_resolved_still_ignores_auto_that_is_data():
    ok = validate(
        Containers,
        {"free": {"x": "auto", "y": ["auto"]}, "text": "auto", "floats": [1.0]},
    )
    assert_resolved(ok)


def test_a_mapping_for_a_union_of_groups_is_tried_against_each_member():
    assert validate(Containers, {"either": {"inner": {"lr": 1.0}, "k": 2}}).either == Mid(Inner(lr=1.0), 2)
    assert validate(Containers, {"either": {"n": 3}}).either == Inner(n=3)
    assert validate(Containers, {"either": Mid(k=2)}).either == Mid(k=2)


def test_a_mapping_that_fits_no_group_reports_the_closest_one():
    # Inner has one problem (n), Mid two (both keys unknown): Inner reports.
    errors = errors_of(Containers, {"either": {"n": "z", "lr": 1.0}})
    assert errors == ("Invalid value for 'either.n': expected int, got 'z' (str)",)
    # Inner has two unknown keys, Mid one bad value: Mid reports, though it is second.
    errors = errors_of(Containers, {"either": {"k": 1, "inner": {"lr": "x"}}})
    assert errors == ("Invalid value for 'either.inner.lr': expected float, got 'x' (str)",)
