"""Strict configuration objects for ``train.api``.

A config is a tree of frozen dataclasses that derive from
:class:`StrictConfig`. :func:`validate` builds one from a plain mapping and
refuses anything that does not match the field annotations, instead of
coercing it.

Interface contract
------------------
``validate(cls, mapping)``
    :param cls: a :class:`StrictConfig` subclass.
    :param mapping: a mapping of field name to value; nested groups are mappings.
    :return: an instance of ``cls``. Fields absent from ``mapping`` keep their defaults.
    :raises ConfigValidationError: one error whose ``.errors`` holds one line per
        bad key, so a caller sees every problem at once.

``merge(base, overrides)``
    Deep merge of two mappings into a new dict. Neither input is changed.

``assert_resolved(config)``
    :raises UnresolvedAutoError: when a field annotated to accept ``"auto"`` still
        holds it. The message lists every such field by dotted path.

``to_json(config)`` / ``from_json(cls, text)``
    ``from_json`` goes back through :func:`validate`, so a file edited by hand is
    held to the same rules as a Python call.

``load_yaml_mapping(path)``
    :raises ConfigError: for an unreadable file, a parse error (a repeated key
        included), a missing PyYAML, or a top level that is not a mapping.

Strings are never coerced here. ``"5"`` for an ``int`` field is an error; the
command line turns text into typed values before it calls this module, and a
Python caller must pass typed values (invariant 2 of the plan). A ``bool`` is
refused for ``int`` and ``float`` fields, and an ``int`` given to a ``float``
field is stored as a ``float`` so ``config.json`` is the same from either door.

Import rule
-----------
Module scope imports only the standard library and
:mod:`train.api.errors`. ``yaml`` and ``train.common`` are imported inside the
functions that need them (D-012).
"""

import collections.abc
import copy
import dataclasses
import difflib
import functools
import json
import math
import re
import typing
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .errors import ConfigError, ConfigValidationError, UnresolvedAutoError

# ---------------------------------------------------------------------

T = TypeVar("T")

#: The literal a field holds until the family recipe and the data fill it in.
AUTO = "auto"

#: ``Auto[float]`` is ``Union[float, Literal["auto"]]``.
Auto = Union[T, Literal["auto"]]

#: A TF-free copy of ``train.common.seed.DEFAULT_SEED``. Importing the real one
#: loads TensorFlow, which the help path must not do (D-012);
#: ``test_strict_config.py`` holds the two equal.
DEFAULT_SEED = 42

_NONE_TYPE = type(None)

# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StrictConfig:
    """Base class of every config group; it declares no fields.

    Subclasses add fields and may override :meth:`resolve_auto`.
    """

    def resolve_auto(self, context: Any) -> "StrictConfig":
        """Return a copy with every ``"auto"`` field filled in from ``context``.

        :param context: an ``AutoContext`` (defined in ``args.py``).
        :return: ``self`` here; groups that have AUTO fields return a new instance.
        """
        return self


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def _is_config_class(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, StrictConfig)


def _describe_value(value: Any) -> str:
    return f"{value!r} ({type(value).__name__})"


def invalid_value_message(
    path: str, annotation: Any, value: Any, note: str = ""
) -> str:
    """Return the one-line wording for a value that does not fit its field.

    ``validate`` and the command line's text coercion share this function, so
    a bad ``optim_args.lr`` reads the same from a Python call and from a
    ``key=value`` token.

    :param path: the dotted key.
    :param annotation: the field annotation the value was held to.
    :param value: the offending value.
    :param note: text appended after the "got ..." part.
    """
    return (
        f"Invalid value for {path!r}: expected {describe_annotation(annotation)}, "
        f"got {_describe_value(value)}{note}"
    )


def unknown_key_message(prefix: str, name: str, siblings: Any) -> str:
    """Return the wording for an unknown key, with a "did you mean" suggestion.

    :param prefix: the dotted path of the group holding the key; ``""`` at the top.
    :param name: the unknown key, without its prefix.
    :param siblings: the names (without prefix) of the keys that do exist there.
    """
    dotted = (lambda key: f"{prefix}.{key}") if prefix else (lambda key: key)
    message = f"Unknown key {dotted(name)!r}."
    close = difflib.get_close_matches(name, list(siblings), n=1)
    if close:
        message += f" Did you mean {dotted(close[0])!r}?"
    return message


def describe_annotation(annotation: Any) -> str:
    """Return the "expected ..." wording for ``annotation``."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is Any:
        return "any value"
    if annotation is _NONE_TYPE or annotation is None:
        return "null"
    if origin is Literal:
        return "one of " + ", ".join(repr(a) for a in args)
    if origin is Union:
        words = [describe_annotation(a) for a in args if typing.get_origin(a) is not Literal]
        choices = [
            c for a in args if typing.get_origin(a) is Literal
            for c in typing.get_args(a)
        ]
        if not words:
            return "one of " + ", ".join(repr(c) for c in choices)
        # "auto" is the same escape hatch on every AUTO field, so it is left
        # out of the sentence ("expected float", not "float or 'auto'").
        words.extend(repr(c) for c in choices if c != AUTO)
        if len(words) == 1:
            return words[0]
        return ", ".join(words[:-1]) + " or " + words[-1]
    if _is_config_class(annotation):
        return f"a mapping of {annotation.__name__} fields"
    if origin in (list, collections.abc.Sequence):
        return f"a list of {describe_annotation(args[0]) if args else 'values'}"
    if origin is tuple:
        return "a list" if not args else f"a list of {describe_annotation(args[0])}"
    if origin in (dict, collections.abc.Mapping):
        return "a mapping"
    return getattr(annotation, "__name__", str(annotation))


def _check(annotation: Any, value: Any, path: str, errors: List[str]) -> Any:
    """Return ``value`` converted for ``annotation``, or append to ``errors``.

    On failure the returned value is meaningless; the caller discards it
    because ``errors`` has grown.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    def fail(note: str = "") -> None:
        errors.append(invalid_value_message(path, annotation, value, note))

    if annotation is Any:
        return value
    if annotation is _NONE_TYPE or annotation is None:
        if value is not None:
            fail()
        return None
    if origin is Literal:
        for choice in args:
            if type(value) is type(choice) and value == choice:
                return value
        fail()
        return value
    if origin is Union:
        if isinstance(value, collections.abc.Mapping):
            # A mapping can only be meant for a group member; report that
            # member's own problems instead of one vague line.
            for member in args:
                if _is_config_class(member):
                    return _check(member, value, path, errors)
        for member in args:
            attempt: List[str] = []
            converted = _check(member, value, path, attempt)
            if not attempt:
                return converted
        fail()
        return value
    if _is_config_class(annotation):
        return _check_group(annotation, value, path, errors)
    if annotation is bool:
        if isinstance(value, bool):
            return value
        fail()
        return value
    if annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        fail()
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail()
            return value
        try:
            converted = float(value)
        except OverflowError:
            fail(": too large for a float")
            return value
        if not math.isfinite(converted):
            errors.append(
                f"Invalid value for {path!r}: expected a finite float, "
                f"got {_describe_value(value)}"
            )
        return converted
    if annotation is str:
        if isinstance(value, str):
            return value
        fail()
        return value
    if origin in (list, collections.abc.Sequence, tuple):
        if not isinstance(value, (list, tuple)):
            fail()
            return value
        if origin is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
            if len(args) != len(value):
                fail(f": expected {len(args)} items")
                return value
            items = [
                _check(a, v, f"{path}[{i}]", errors)
                for i, (a, v) in enumerate(zip(args, value))
            ]
        else:
            element = args[0] if args else Any
            items = [
                _check(element, v, f"{path}[{i}]", errors)
                for i, v in enumerate(value)
            ]
        return tuple(items) if origin is tuple else items
    if origin in (dict, collections.abc.Mapping):
        if not isinstance(value, collections.abc.Mapping):
            fail()
            return value
        key_type, value_type = args if args else (Any, Any)
        return {
            _check(key_type, k, f"{path}.{k}", errors): _check(
                value_type, v, f"{path}.{k}", errors
            )
            for k, v in value.items()
        }
    raise TypeError(
        f"train.api.config cannot validate the annotation {annotation!r} "
        f"of {path!r}; add a case to _check"
    )


def _check_group(
    cls: Type[StrictConfig], value: Any, path: str, errors: List[str]
) -> Any:
    """Validate ``value`` as the nested group ``cls`` found at ``path``."""
    if isinstance(value, cls):
        value = {
            f.name: getattr(value, f.name)
            for f in dataclasses.fields(value)
            if f.init
        }
    if not isinstance(value, collections.abc.Mapping):
        errors.append(
            f"Invalid value for {path!r}: expected {describe_annotation(cls)}, "
            f"got {_describe_value(value)}"
        )
        return value
    return _build(cls, value, path, errors)


def _build(
    cls: Type[StrictConfig],
    mapping: Mapping[str, Any],
    prefix: str,
    errors: List[str],
) -> Optional[StrictConfig]:
    """Build ``cls`` from ``mapping``, appending every problem to ``errors``."""
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    dotted = (lambda name: f"{prefix}.{name}") if prefix else (lambda name: name)
    problems_before = len(errors)
    kwargs: Dict[str, Any] = {}
    for key, value in mapping.items():
        name = str(key)
        if not isinstance(key, str) or name not in fields:
            errors.append(
                unknown_key_message(
                    prefix, name, fields if isinstance(key, str) else []
                )
            )
            continue
        kwargs[name] = _check(hints[name], value, dotted(name), errors)
    for name, field in fields.items():
        no_default = (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        )
        if name not in kwargs and no_default:
            errors.append(f"Missing required key {dotted(name)!r}.")
    if len(errors) > problems_before:
        return None
    return cls(**kwargs)


def validate(cls: Type[T], mapping: Mapping[str, Any]) -> T:
    """Build ``cls`` from ``mapping``, refusing every mismatch.

    :param cls: a :class:`StrictConfig` subclass.
    :param mapping: field name to value; nested groups are mappings.
    :return: an instance of ``cls``.
    :raises ConfigValidationError: carrying one line per problem key.
    """
    errors: List[str] = []
    if not isinstance(mapping, collections.abc.Mapping):
        raise ConfigValidationError(
            [
                f"Invalid config: expected a mapping of {cls.__name__} fields, "
                f"got {_describe_value(mapping)}"
            ]
        )
    result = _build(cls, mapping, "", errors)
    if errors:
        raise ConfigValidationError(errors)
    return result


def merge(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    """Deep-merge ``overrides`` onto ``base`` into a new dict.

    A key whose value is a mapping on both sides is merged; any other value in
    ``overrides`` replaces the one in ``base``. Neither argument is changed and
    the result shares no mutable object with them.
    """
    result: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and isinstance(
            result.get(key), collections.abc.Mapping
        ):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _admits_auto(annotation: Any) -> bool:
    """Return whether ``annotation`` is a union with ``Literal["auto"]`` in it."""
    if typing.get_origin(annotation) is Literal:
        return AUTO in typing.get_args(annotation)
    if typing.get_origin(annotation) is Union:
        return any(_admits_auto(a) for a in typing.get_args(annotation))
    return False


def _unresolved(config: Any, prefix: str) -> List[str]:
    hints = typing.get_type_hints(type(config))
    found: List[str] = []
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        path = f"{prefix}.{field.name}" if prefix else field.name
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            found.extend(_unresolved(value, path))
        elif (
            isinstance(value, str)
            and value == AUTO
            and _admits_auto(hints[field.name])
        ):
            found.append(path)
    return found


def assert_resolved(config: StrictConfig) -> None:
    """Raise when an AUTO-annotated field of ``config`` is still ``"auto"``.

    Only dataclass fields are walked, and only a field whose own annotation
    admits ``Literal["auto"]`` counts. A literal ``"auto"`` inside a free
    ``model_args`` dict is data, not AUTO.

    :raises UnresolvedAutoError: naming every unresolved field by dotted path.
    """
    paths = _unresolved(config, "")
    if paths:
        raise UnresolvedAutoError(
            "These fields are still 'auto' after resolution, so no recipe "
            "or data supplied them; set them explicitly: " + ", ".join(paths)
        )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def to_json(config: StrictConfig) -> str:
    """Return ``config`` as indented JSON text; NaN and Infinity are refused."""
    from train.common.config_io import json_numpy_default

    return json.dumps(
        dataclasses.asdict(config),
        indent=2,
        default=json_numpy_default,
        allow_nan=False,
    )


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed in a config")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite number")
    return value


def parse_json(text: str) -> Tuple[Any, Optional[str]]:
    """Parse JSON text, refusing NaN, Infinity and overflowing numbers.

    The failure comes back as a value, not as an exception, so a caller that
    collects many problems (the command line's coercion) needs no handler.

    :param text: JSON text.
    :return: ``(value, None)`` on success, ``(None, reason)`` on failure.
    """
    try:
        return (
            json.loads(
                text, parse_constant=_refuse_constant, parse_float=_finite_float
            ),
            None,
        )
    except ValueError as exc:
        return None, str(exc)


def from_json(cls: Type[T], text: str) -> T:
    """Parse ``text`` and build ``cls`` through :func:`validate`.

    :raises ConfigError: when ``text`` is not JSON, or holds NaN or Infinity.
    :raises ConfigValidationError: when the parsed mapping does not fit ``cls``.
    """
    mapping, reason = parse_json(text)
    if reason is not None:
        raise ConfigError(f"Config text is not valid JSON: {reason}")
    return validate(cls, mapping)


# ---------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------

_YAML_FLOAT = re.compile(
    r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
)


@functools.lru_cache(maxsize=1)
def _yaml_loader() -> Any:
    """Build the SafeLoader subclass that reads floats the YAML 1.2 way.

    :raises ConfigError: when PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "Reading a YAML config needs PyYAML; install the 'train' extra "
            "(pip install 'dl_techniques[train]' or pip install PyYAML)."
        ) from exc

    class Yaml12SafeLoader(yaml.SafeLoader):
        """SafeLoader whose float resolver accepts ``3e-4`` and which refuses a repeated key."""

        def construct_mapping(self, node, deep=False):
            # DECISION plan-2026-10-30T132432-a865fa08/D-027: PyYAML keeps the
            # LAST of two equal keys without a word, while the command line
            # refuses a repeated key. Do NOT leave the silent last-wins in a
            # config file: a typo'd copy-paste would override a value and
            # nothing would say so. Only scalar keys are compared, and merge
            # keys (``<<``) are skipped because overriding a merged value is
            # what a merge is for.
            seen = set()
            for key_node, _value_node in node.value:
                if (
                    not isinstance(key_node, yaml.ScalarNode)
                    or key_node.tag == "tag:yaml.org,2002:merge"
                ):
                    continue
                key = self.construct_object(key_node, deep=True)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found a duplicate key {key!r}",
                        key_node.start_mark,
                    )
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

    # DECISION plan-2026-10-30T132432-a865fa08/D-002: PyYAML implements YAML
    # 1.1, whose float pattern needs a dot, so yaml.safe_load("3e-4") returns
    # the STRING "3e-4" (measured) and ``lr: 3e-4`` in a config file would
    # fail validation. Do NOT fall back to the stock yaml.safe_load, and do NOT
    # coerce strings to numbers in validate(); swap in the 1.2 float pattern on
    # this subclass only, so the global SafeLoader is left alone.
    float_tag = "tag:yaml.org,2002:float"
    Yaml12SafeLoader.yaml_implicit_resolvers = {
        first: [(tag, pattern) for tag, pattern in resolvers if tag != float_tag]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    Yaml12SafeLoader.add_implicit_resolver(
        float_tag, _YAML_FLOAT, list("-+0123456789.")
    )
    return Yaml12SafeLoader


def load_yaml_mapping(path: Any) -> Dict[str, Any]:
    """Read a YAML file whose top level is a mapping.

    :param path: the file to read.
    :return: the parsed mapping, with ``3e-4`` read as a float.
    :raises ConfigError: for an unreadable file, a parse error (a repeated
        key included), a missing PyYAML, or a top level that is not a mapping.
    """
    loader_class = _yaml_loader()
    import yaml

    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {str(path)!r}: {exc}") from exc
    loader = loader_class(text)
    try:
        data = loader.get_single_data()
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Config file {str(path)!r} is not valid YAML: {exc}"
        ) from exc
    finally:
        loader.dispose()
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {str(path)!r} must have a mapping at the top level, "
            f"got {_describe_value(data)}"
        )
    return data
