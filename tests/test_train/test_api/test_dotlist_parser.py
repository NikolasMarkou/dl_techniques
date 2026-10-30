"""``key=value`` tokens become a typed mapping, and every bad token is named.

Two stages live in ``train.api.cli``: ``parse_dotlist`` splits tokens into a
nested dict of raw strings, and ``coerce_mapping`` types those strings by the
field annotations (D-002). Python callers never pass through here (invariant
2). The tests use small local dataclasses; the real groups arrive in
``args.py``.
"""

import dataclasses
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple

import pytest

from train.api.cli import (
    coerce,
    coerce_mapping,
    parse_dotlist,
    run_task,
)
from train.api.config import AUTO, Auto, StrictConfig, validate
from train.api.errors import ConfigError, ConfigValidationError


@dataclasses.dataclass(frozen=True)
class Optim(StrictConfig):
    lr: Auto[float] = AUTO
    steps: int = 10
    nesterov: bool = False
    name: str = "adam"
    clipnorm: Optional[float] = None
    betas: Tuple[float, float] = (0.9, 0.999)
    schedule: Literal["cosine_decay", "constant"] = "cosine_decay"
    warmup: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class Top(StrictConfig):
    out: str = "auto"
    epochs: Auto[int] = AUTO
    seed: Optional[int] = None
    optim_args: Optim = dataclasses.field(default_factory=Optim)
    model_args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    tensorboard: Optional[Dict[str, Any]] = dataclasses.field(default_factory=dict)
    layers: List[int] = dataclasses.field(default_factory=list)


def coerce_error(annotation, text, path="key"):
    with pytest.raises(ConfigError) as caught:
        coerce(annotation, text, path)
    return str(caught.value)


def mapping_errors(raw):
    with pytest.raises(ConfigValidationError) as caught:
        coerce_mapping(Top, raw)
    return caught.value.errors


# ---------------------------------------------------------------------
# Splitting tokens
# ---------------------------------------------------------------------


def test_tokens_nest_on_dots_and_split_at_the_first_equals():
    path, tree = parse_dotlist(
        ["out=runs/a", "optim_args.lr=3e-4", "optim_args.name=a=b", "model_args.x.y=1"]
    )
    assert path is None
    assert tree == {
        "out": "runs/a",
        "optim_args": {"lr": "3e-4", "name": "a=b"},
        "model_args": {"x": {"y": "1"}},
    }


def test_an_empty_value_is_kept_as_empty_text():
    assert parse_dotlist(["name="]) == (None, {"name": ""})


def test_the_config_key_is_returned_apart_and_not_put_in_the_tree():
    path, tree = parse_dotlist(["config=c.yaml", "epochs=3"])
    assert path == "c.yaml"
    assert tree == {"epochs": "3"}


@pytest.mark.parametrize(
    "tokens, wording",
    [
        (["epochs"], "'epochs' is not key=value"),
        (["=3"], "'=3' has an empty key"),
        (["a..b=1"], "'a..b' has an empty segment"),
        (["a.=1"], "'a.' has an empty segment"),
        (["epochs=1", "epochs=2"], "Repeated key 'epochs'"),
        (["config=a.yaml", "config=b.yaml"], "Repeated key 'config'"),
        (["config="], "'config' needs a file path"),
        (["a=1", "a.b=2"], "Repeated key 'a'"),
        (["a.b=2", "a=1"], "Repeated key 'a'"),
        (["a=1", "a.b.c=2"], "Repeated key 'a'"),
        (["optim_args={}", "optim_args.lr=1"], "Repeated key 'optim_args'"),
        (["optim_args.lr=1", "optim_args={}"], "Repeated key 'optim_args'"),
    ],
)
def test_a_bad_token_is_refused_with_its_own_wording(tokens, wording):
    with pytest.raises(ConfigValidationError) as caught:
        parse_dotlist(tokens)
    assert any(wording in line for line in caught.value.errors), caught.value.errors


def test_every_bad_token_is_reported_at_once():
    with pytest.raises(ConfigValidationError) as caught:
        parse_dotlist(["epochs", "=3", "a..b=1", "x=1", "x=2"])
    assert len(caught.value.errors) == 4


def test_a_dotlist_error_is_a_config_error():
    with pytest.raises(ConfigError):
        parse_dotlist(["nope"])


# ---------------------------------------------------------------------
# Coercion, by annotation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [("3e-4", 0.0003), ("1E-3", 0.001), (".5", 0.5), ("5", 5.0), ("-2.5", -2.5), ("1.", 1.0)],
)
def test_a_float_reads_decimal_and_exponent_forms(text, expected):
    value = coerce(float, text, "lr")
    assert value == expected
    assert type(value) is float


@pytest.mark.parametrize(
    "text", ["inf", "-inf", "nan", "NaN", "1e999", "1_000", "abc", "", "--1", "0x10", " 1"]
)
def test_a_float_refuses_everything_else(text):
    assert "expected float" in coerce_error(float, text, "optim_args.lr")


@pytest.mark.parametrize("text, expected", [("5", 5), ("-3", -3), ("0", 0), ("007", 7)])
def test_an_int_reads_only_digits(text, expected):
    value = coerce(int, text, "epochs")
    assert value == expected
    assert type(value) is int


@pytest.mark.parametrize("text", ["5.0", "1_000", "+5", "1e3", "", "five", "0x10", "5 "])
def test_an_int_refuses_a_float_form_or_a_separator(text):
    assert coerce_error(int, text, "epochs") == (
        f"Invalid value for 'epochs': expected int, got {text!r} (str)"
    )


def test_a_digit_string_past_the_interpreter_limit_is_a_refusal_not_a_traceback():
    text = "9" * 5000
    assert coerce_error(int, text, "epochs").endswith("(str): too many digits")
    (line,) = mapping_errors({"optim_args": {"steps": text}})
    assert line.startswith("Invalid value for 'optim_args.steps': expected int")
    assert line.endswith("(str): too many digits")
    assert len(line) < 200  # the token is quoted in part, not whole
    (union_line,) = mapping_errors({"epochs": text})  # Auto[int]: a union drops the note
    assert union_line.startswith("Invalid value for 'epochs': expected int") and len(union_line) < 200
    # An untyped value falls through to the text, as for any other non-number.
    assert coerce_mapping(Top, {"model_args": {"x": text}}) == {"model_args": {"x": text}}


def test_json_nested_past_the_recursion_limit_is_a_refusal_not_a_traceback():
    deep = "[" * 100000
    assert "nested too deeply" in coerce_error(List[int], deep, "layers")
    errors = mapping_errors({"model_args": deep, "layers": deep})
    assert len(errors) == 2, errors
    assert coerce_mapping(Top, {"model_args": {"x": deep}}) == {"model_args": {"x": deep}}


def test_a_repeated_key_in_json_text_is_refused():
    assert "duplicate key 'lr'" in coerce_error(Dict[str, Any], '{"lr": 1, "lr": 2}', "model_args")
    assert "duplicate key 'lr'" in coerce_error(Optim, '{"lr": 1, "lr": 2}', "optim_args")
    errors = mapping_errors({"optim_args": '{"lr": 1, "lr": 2}'})
    assert "duplicate key 'lr'" in errors[0]


@pytest.mark.parametrize("text, expected", [("true", True), ("false", False)])
def test_a_bool_reads_true_and_false(text, expected):
    assert coerce(bool, text, "nesterov") is expected


@pytest.mark.parametrize("text", ["yes", "no", "on", "off", "1", "0", "True", "FALSE", ""])
def test_a_bool_refuses_the_yaml_one_point_one_words(text):
    assert "expected bool" in coerce_error(bool, text, "nesterov")


@pytest.mark.parametrize("text", ["no", "on", "true", "", "a=b", "3e-4", " x "])
def test_a_str_is_kept_verbatim(text):
    assert coerce(str, text, "name") == text


@pytest.mark.parametrize("text", ["null", "none", "None", "NULL"])
def test_optional_accepts_null_and_none_in_any_case(text):
    assert coerce(Optional[float], text, "clipnorm") is None
    assert coerce(Optional[int], text, "seed") is None


def test_optional_otherwise_defers_to_the_member():
    assert coerce(Optional[float], "3e-4", "clipnorm") == 0.0003
    assert "expected float or null" in coerce_error(Optional[float], "abc", "clipnorm")


def test_an_optional_str_still_reads_null_as_none():
    assert coerce(Optional[str], "null", "x") is None
    assert coerce(Optional[str], "hello", "x") == "hello"


def test_auto_matches_its_literal_before_the_other_members():
    assert coerce(Auto[float], "auto", "lr") == "auto"
    assert coerce(Auto[float], "0.1", "lr") == 0.1
    assert coerce(Auto[int], "auto", "epochs") == "auto"
    assert "expected float" in coerce_error(Auto[float], "autox", "lr")


def test_a_literal_takes_one_of_its_choices_only():
    kind = Literal["cosine_decay", "constant"]
    assert coerce(kind, "constant", "schedule") == "constant"
    assert "one of 'cosine_decay', 'constant'" in coerce_error(kind, "linear", "schedule")


def test_a_list_or_tuple_reads_json():
    assert coerce(Tuple[float, float], "[0.9,0.999]", "betas") == [0.9, 0.999]
    assert coerce(List[int], "[1, 2, 3]", "layers") == [1, 2, 3]


@pytest.mark.parametrize("text", ["[NaN]", "[Infinity]", "[1e999]", "{}", "[1,2", "1,2", ""])
def test_a_list_refuses_non_json_non_lists_and_non_finite_numbers(text):
    assert "expected a list" in coerce_error(List[float], text, "betas")


def test_nan_inside_json_names_the_reason():
    assert "NaN is not allowed" in coerce_error(List[float], "[NaN]", "betas")


def test_a_dict_reads_json_and_a_group_reads_a_json_mapping():
    assert coerce(Dict[str, Any], '{"a": 1, "b": [2]}', "model_args") == {"a": 1, "b": [2]}
    assert coerce(Optional[Dict[str, Any]], "null", "tensorboard") is None
    assert coerce(Optional[Dict[str, Any]], "{}", "tensorboard") == {}
    assert coerce(Optim, '{"lr": 3e-4}', "optim_args") == {"lr": 0.0003}
    assert "expected a mapping" in coerce_error(Dict[str, Any], "[1]", "model_args")


def test_an_annotation_the_coercer_does_not_know_is_a_programming_error():
    with pytest.raises(TypeError, match="add a case to _convert"):
        coerce(bytes, "x", "k")


# ---------------------------------------------------------------------
# Untyped dict values
# ---------------------------------------------------------------------


def test_untyped_dict_values_follow_the_documented_rule():
    typed = coerce_mapping(
        Top,
        {
            "model_args": {
                "float_exp": "3e-4",
                "int": "5",
                "bool": "true",
                "word": "on",
                "dot_float": ".5",
                "nan": "nan",
                "leading_zero": "010",
                "null": "null",
                "list": "[1, 2]",
                "json_nan": "NaN",
                "overflow": "1e999",
                "quoted": '"abc"',
            }
        },
    )["model_args"]
    assert typed == {
        "float_exp": 0.0003,
        "int": 5,
        "bool": True,
        "word": "on",
        "dot_float": 0.5,
        "nan": "nan",
        "leading_zero": 10.0,  # pinned: JSON refuses 010, the float fallback reads it
        "null": None,
        "list": [1, 2],
        "json_nan": "NaN",
        "overflow": "1e999",
        "quoted": "abc",
    }
    assert type(typed["int"]) is int
    assert type(typed["leading_zero"]) is float


@pytest.mark.parametrize("text", ["none", "None", "NONE", "Null", "null"])
def test_an_untyped_value_reads_the_null_words_as_none_like_an_optional_field(text):
    typed = coerce_mapping(Top, {"model_args": {"x": text}, "seed": text})
    assert typed == {"model_args": {"x": None}, "seed": None}


def test_a_quoted_null_word_stays_a_string_in_an_untyped_value():
    assert coerce_mapping(Top, {"model_args": {"x": '"none"'}}) == {"model_args": {"x": "none"}}


def test_a_nested_untyped_dict_is_typed_leaf_by_leaf():
    typed = coerce_mapping(Top, {"model_args": {"block": {"drop": "0.1", "n": "2"}}})
    assert typed["model_args"] == {"block": {"drop": 0.1, "n": 2}}


# ---------------------------------------------------------------------
# Whole mappings
# ---------------------------------------------------------------------


def test_the_documented_error_line_for_a_bad_nested_float():
    _, raw = parse_dotlist(["optim_args.lr=abc"])
    assert mapping_errors(raw) == (
        "Invalid value for 'optim_args.lr': expected float, got 'abc' (str)",
    )


def test_every_bad_value_and_unknown_key_is_reported_in_one_error():
    _, raw = parse_dotlist(
        ["epochs=5.0", "optim_args.lr=abc", "optim_args.lrr=1", "outt=x", "seed=nine"]
    )
    errors = mapping_errors(raw)
    assert len(errors) == 5, errors
    assert "Invalid value for 'epochs': expected int, got '5.0' (str)" in errors
    assert "Unknown key 'optim_args.lrr'. Did you mean 'optim_args.lr'?" in errors
    assert "Unknown key 'outt'. Did you mean 'out'?" in errors


def test_sub_keys_under_a_scalar_field_are_refused():
    errors = mapping_errors({"epochs": {"foo": "1"}})
    assert len(errors) == 1
    assert errors[0].startswith("Invalid value for 'epochs': expected int")


def test_a_json_group_and_a_dotted_group_both_reach_the_same_mapping():
    by_json = coerce_mapping(Top, {"optim_args": '{"lr": 3e-4, "steps": 5}'})
    by_dots = coerce_mapping(Top, {"optim_args": {"lr": "3e-4", "steps": "5"}})
    assert by_json == by_dots == {"optim_args": {"lr": 0.0003, "steps": 5}}


def test_a_coerced_mapping_goes_on_to_validate_and_builds_the_config():
    _, raw = parse_dotlist(
        [
            "out=runs/a",
            "epochs=50",
            "optim_args.lr=3e-4",
            "optim_args.clipnorm=null",
            "optim_args.betas=[0.9,0.99]",
            "tensorboard=null",
            "layers=[1,2]",
        ]
    )
    top = validate(Top, coerce_mapping(Top, raw))
    assert top.epochs == 50 and top.out == "runs/a"
    assert top.optim_args.lr == 0.0003
    assert top.optim_args.clipnorm is None
    assert top.optim_args.betas == (0.9, 0.99)
    assert top.tensorboard is None and top.layers == [1, 2]


def test_a_python_caller_stays_strict_where_the_cli_coerces():
    assert validate(Top, coerce_mapping(Top, {"epochs": "5"})).epochs == 5
    with pytest.raises(ConfigValidationError):
        validate(Top, {"epochs": "5"})


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------


def recorder(success=True):
    calls = []

    def function(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=success)

    return function, calls


def test_run_task_calls_the_function_with_the_typed_mapping_and_returns_zero():
    function, calls = recorder(success=True)
    code = run_task(Top, function, ["epochs=3", "optim_args.lr=3e-4"])
    assert code == 0
    assert calls == [{"epochs": 3, "optim_args": {"lr": 0.0003}}]


def test_an_unmet_result_returns_one():
    function, calls = recorder(success=False)
    assert run_task(Top, function, ["epochs=3"]) == 1
    assert len(calls) == 1


def test_the_config_file_loads_first_and_the_dotlist_wins(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("epochs: 9\noptim_args:\n  lr: 3e-4\n  steps: 5\n")
    function, calls = recorder()
    run_task(Top, function, [f"config={path}", "optim_args.steps=7"])
    assert calls == [{"epochs": 9, "optim_args": {"lr": 0.0003, "steps": 7}}]


def test_a_bad_token_or_value_stops_before_the_function_runs(tmp_path):
    function, calls = recorder()
    with pytest.raises(ConfigError):
        run_task(Top, function, ["epochs=1", "epochs=2"])
    with pytest.raises(ConfigValidationError):
        run_task(Top, function, ["epochs=five"])
    with pytest.raises(ConfigError):
        run_task(Top, function, [f"config={tmp_path / 'missing.yaml'}"])
    assert calls == []


@dataclasses.dataclass(frozen=True)
class Needs(StrictConfig):
    out: str
    epochs: int = 1


def test_a_bad_file_or_a_missing_key_is_refused_before_a_keyword_only_function(tmp_path):
    """The function has a real signature, so reaching it would be a TypeError (exit 1)."""
    calls = []

    def keyword_only(*, out, epochs=1):
        calls.append((out, epochs))
        return SimpleNamespace(success=True)

    def run(text, *tokens):
        path = tmp_path / "c.yaml"
        path.write_text(text)
        with pytest.raises(ConfigValidationError) as caught:
            run_task(Needs, keyword_only, [f"config={path}", *tokens])
        return caught.value.errors

    assert run("out: a\nepoch: 3\n") == ("Unknown key 'epoch'. Did you mean 'epochs'?",)
    assert run("epochs: 3\n") == ("Missing required key 'out'.",)
    assert run("out: a\n1: 2\n") == ("Invalid key '1': keys must be strings, got 1 (int)",)
    assert calls == []
    assert run_task(Needs, keyword_only, ["out=a", "epochs=2"]) == 0
    assert calls == [("a", 2)]


def test_a_missing_required_key_on_the_command_line_alone_is_refused():
    function, calls = recorder()
    with pytest.raises(ConfigValidationError, match="Missing required key 'out'"):
        run_task(Needs, function, ["epochs=1"])
    assert calls == []


def test_a_failure_inside_the_function_is_not_swallowed():
    def boom(**_kwargs):
        raise RuntimeError("training exploded")

    with pytest.raises(RuntimeError, match="training exploded"):
        run_task(Top, boom, ["epochs=1"])
