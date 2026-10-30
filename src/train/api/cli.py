"""Command line entry of ``python -m train``.

Interface contract
------------------
``main(argv)`` takes the arguments after the program name and returns the
process exit code: 0 on success, 2 when a :class:`~train.api.errors.DLTrainError`
was raised. Any other exception propagates, so Python exits 1 with a
traceback. Help text goes to stdout, error text to stderr.

Commands are the rows of ``COMMANDS`` (``help``, ``list_tasks``,
``list_models``), then the tasks in :data:`train.api.registry.TASKS`. A task
command with no arguments, or with a lone help word, prints its config keys;
any other arguments go to ``run_task``. ``list_*`` commands list instead of
printing help, even when given ``help``.

``parse_dotlist(tokens)``
    :param tokens: ``key=value`` tokens; dots nest, ``config=path.yaml`` is special.
    :return: ``(config_path or None, nested dict of raw strings)``.
    :raises ConfigValidationError: one line per bad token (no ``=``, empty key,
        empty segment, repeated key, a key that is both a value and a prefix).

``coerce(annotation, text, dotted_path)``
    :return: ``text`` converted by ``annotation``.
    :raises ConfigError: when the text does not fit; the message is the same
        line :func:`~train.api.config.validate` writes.

``coerce_mapping(cls, raw)``
    :param cls: a :class:`~train.api.config.StrictConfig` subclass.
    :param raw: the nested dict from :func:`parse_dotlist`.
    :return: a nested dict of typed values, ready for ``validate``.
    :raises ConfigValidationError: every unknown key and every bad value at once.

``run_task(config_class, function, args)``
    Parse, coerce, merge over ``config=``, call ``function(**mapping)`` and
    return 0 when ``result.success`` else 1. ``function`` is the keyword-only
    function a Python caller uses (D-008); it runs ``validate`` itself.

Coercion rules (the text is the whole token after the first ``=``):

* ``float``: decimal or exponent form (``3e-4``, ``1E-3``, ``.5``), finite.
* ``int``: ``-?[0-9]+`` only; ``5.0`` and ``1_000`` are refused.
* ``bool``: ``true`` or ``false`` only.
* ``str``: verbatim, so ``no`` stays ``"no"``.
* ``Optional``: ``null`` or ``none`` (any case) gives ``None``.
* list, tuple, dict, group: JSON text; NaN and Infinity are refused.
* untyped (``Any``) values: JSON, then a finite float (``.5``, ``010``), else the text.

Import rule
-----------
This module is on the help path, which must stay under one second. Its
module scope therefore imports only the standard library and
``dl_techniques.utils.logger``. Anything heavy (TensorFlow, Keras,
``train.common``, ``dl_techniques.models``, ``yaml``) is imported inside the
function that needs it.
"""

# DECISION plan-2026-10-30T132432-a865fa08/D-012: module scope stays stdlib plus
# the logger. Do NOT import train.common, dl_techniques.models, keras,
# tensorflow or yaml here: train.common's eager __init__ loads TensorFlow
# (about 4.2 s) and the help path must finish in under 1 s.
import collections.abc
import dataclasses
import difflib
import logging
import math
import re
import sys
import typing
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from dl_techniques.utils.logger import logger

from . import registry
from .config import (
    StrictConfig,
    describe_annotation,
    invalid_value_message,
    load_yaml_mapping,
    merge,
    parse_json,
    unknown_key_message,
)
from .errors import ConfigError, ConfigValidationError, DLTrainError

# ---------------------------------------------------------------------

PROGRAM = "python -m train"
HELP_WORDS = ("help", "-h", "--help")
VERBOSE_FLAGS = ("-v", "--verbose")

# ---------------------------------------------------------------------


class Command(NamedTuple):
    """One row of the command table.

    :param name: the word typed after ``python -m train``.
    :param summary: the one-line description shown in the help text.
    :param run: called with the arguments after the command name; returns an exit code.
    """

    name: str
    summary: str
    run: Callable[[List[str]], int]


def _write_help(_args: List[str]) -> int:
    """Print the help text derived from :data:`COMMANDS` and the task registry."""
    lines = [
        f"usage: {PROGRAM} <command> [key=value ...]",
        f"       {PROGRAM} <command> help",
        "",
        "commands:",
    ]
    width = max(len(name) for name in COMMANDS)
    for command in COMMANDS.values():
        lines.append(f"  {command.name.ljust(width)}  {command.summary}")
    tasks = registry.list_tasks()
    if tasks:
        width = max(len(spec.name) for spec in tasks)
        lines += ["", "tasks:"]
        lines += [f"  {spec.name.ljust(width)}  {spec.summary}" for spec in tasks]
    lines += [
        "",
        "options:",
        "  -v, --verbose  log at DEBUG level",
        "  -h, --help     show this help",
        "",
    ]
    sys.stdout.write("\n".join(lines))
    return 0


def _list_tasks(args: List[str]) -> int:
    """Print the registered tasks, one ``name  summary`` line each."""
    if args not in ([], ["help"]):
        raise ConfigError(
            f"list_tasks takes no arguments, got: {' '.join(args)}"
        )
    tasks = registry.list_tasks()
    if not tasks:
        sys.stdout.write("no tasks are registered\n")
        return 0
    width = max(len(spec.name) for spec in tasks)
    sys.stdout.write(
        "".join(f"{spec.name.ljust(width)}  {spec.summary}\n" for spec in tasks)
    )
    return 0


def _list_models(args: List[str]) -> int:
    """Print ``family/variant`` names, one per line; ``task=<name>`` filters.

    This imports the model families' variant tables, so it is not on the help
    path. ``list_models help`` lists, like ``list_tasks help``.
    """
    task = None
    if args != ["help"]:
        config_path, raw = parse_dotlist(args)
        if config_path is not None:
            raise ConfigError("list_models takes only task=<name>, not config=")
        for key in raw:
            if key != "task":
                raise ConfigError(unknown_key_message("", key, ["task"]))
        task = raw.get("task")
        if task is not None and not isinstance(task, str):
            raise ConfigError("list_models: 'task' takes one name, not a nested key")
    names = registry.list_models(task)
    if not names:
        scope = f" for task {task!r}" if task else ""
        sys.stdout.write(f"no models are registered{scope}\n")
        return 0
    sys.stdout.write("".join(f"{name}\n" for name in names))
    return 0


def _field_help_lines(cls: Type[StrictConfig], prefix: str = "") -> List[str]:
    """Return one ``key  type  help [default]`` line per leaf key of ``cls``."""
    hints = typing.get_type_hints(cls)
    lines: List[str] = []
    for item in dataclasses.fields(cls):
        if not item.init:
            continue
        key = f"{prefix}.{item.name}" if prefix else item.name
        annotation = hints[item.name]
        if isinstance(annotation, type) and issubclass(annotation, StrictConfig):
            lines += _field_help_lines(annotation, key)
            continue
        if item.default is not dataclasses.MISSING:
            default = f" [default: {item.default!r}]"
        elif item.default_factory is not dataclasses.MISSING:
            default = f" [default: {item.default_factory()!r}]"
        else:
            default = " [required]"
        text = item.metadata.get("help", "")
        lines.append(f"  {key}  ({describe_annotation(annotation)})  {text}{default}")
    return lines


def _write_task_help(spec: registry.TaskSpec) -> int:
    """Print the keys of a task's config, nested keys written with dots."""
    lines = [
        f"usage: {PROGRAM} {spec.name} [config=path.yaml] [key=value ...]",
        "",
        spec.summary,
        "",
        "keys (dots reach a group, as in optim_args.lr=3e-4):",
        *_field_help_lines(spec.load_config_class()),
        "",
    ]
    sys.stdout.write("\n".join(lines))
    return 0


def _run_registered_task(spec: registry.TaskSpec, args: List[str]) -> int:
    """Run a registered task; a bare command or a lone help word prints its keys."""
    if not args or (len(args) == 1 and args[0] in HELP_WORDS):
        return _write_task_help(spec)
    return run_task(spec.load_config_class(), spec.load_function(), args)


# One table drives dispatch and the help text, so help never lists a command
# that does not exist (D-007). Registered tasks are looked up after it (D-029).
COMMANDS: Dict[str, Command] = {
    command.name: command
    for command in (
        Command("help", "show this help", _write_help),
        Command("list_tasks", "list the registered tasks", _list_tasks),
        Command("list_models", "list family/variant names, task=<name> filters", _list_models),
    )
}


def main(argv: Sequence[str]) -> int:
    """Run one command line and return the exit code.

    :param argv: the arguments after the program name.
    :return: 0 on success, 2 on a :class:`DLTrainError`.
    """
    args = [token for token in argv if token not in VERBOSE_FLAGS]
    if len(args) != len(argv):
        logger.setLevel(logging.DEBUG)
    # DECISION plan-2026-10-30T132432-a865fa08/D-009: catch DLTrainError only.
    # Do NOT widen this to ``except Exception: return 2``: the failed-run
    # censuses reject a broad handler in a ``main`` that has no
    # ``if __name__`` block, and an unexpected failure must reach the user as
    # a traceback with exit code 1, Ctrl-C included.
    try:
        return _dispatch(args)
    except DLTrainError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


def _dispatch(args: List[str]) -> int:
    """Pick the command named by ``args[0]`` and run it."""
    if not args or args[0] in HELP_WORDS:
        if args[1:]:
            raise ConfigError(
                f"{args[0]} takes no arguments, got: {' '.join(args[1:])}"
            )
        return COMMANDS["help"].run([])
    command = COMMANDS.get(args[0])
    if command is not None:
        return command.run(args[1:])
    # DECISION plan-2026-10-30T132432-a865fa08/D-029: a task is a registry row,
    # not a COMMANDS row. Do NOT add one hand-written Command per task: the
    # exit-code tests inject a TaskSpec and expect __main__ to dispatch it.
    task = registry.TASKS.get(args[0])
    if task is None:
        raise ConfigError(_unknown_command_message(args[0]))
    return _run_registered_task(task, args[1:])


def _unknown_command_message(name: str) -> str:
    """Build the unknown-command text, with a suggestion when one is close."""
    message = f"unknown command {name!r}."
    close = difflib.get_close_matches(name, [*COMMANDS, *registry.TASKS], n=1)
    if close:
        message += f" Did you mean {close[0]!r}?"
    return message + f" Run '{PROGRAM} help' for the list."


# ---------------------------------------------------------------------
# Dotlist parsing
# ---------------------------------------------------------------------

CONFIG_KEY = "config"


def _insert(
    tree: Dict[str, Any], key: str, text: str, errors: List[str]
) -> None:
    """Place ``text`` at the nested position ``key`` names, or append to ``errors``."""
    segments = key.split(".")
    node = tree
    for depth, segment in enumerate(segments[:-1]):
        child = node.get(segment)
        if child is None:
            child = node[segment] = {}
        elif not isinstance(child, dict):
            prefix = ".".join(segments[: depth + 1])
            errors.append(
                f"Repeated key {prefix!r}: it is given a value and also as "
                f"{key!r}."
            )
            return
        node = child
    last = segments[-1]
    existing = node.get(last)
    if existing is None:
        node[last] = text
    elif isinstance(existing, dict):
        errors.append(
            f"Repeated key {key!r}: it is given a value and also with "
            f"sub-keys (as '{key}.<name>')."
        )
    else:
        errors.append(f"Repeated key {key!r}: give each key once.")


def parse_dotlist(tokens: Sequence[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Split ``key=value`` tokens into a ``config=`` path and a nested dict.

    A token is split at its first ``=``, so a value may hold further ``=``.
    Every value stays a raw string; :func:`coerce_mapping` types them.

    :param tokens: the arguments after the command name.
    :return: ``(config_path or None, nested dict of raw strings)``.
    :raises ConfigValidationError: naming every bad token.
    """
    errors: List[str] = []
    config_path: Optional[str] = None
    tree: Dict[str, Any] = {}
    for token in tokens:
        key, separator, text = token.partition("=")
        if not separator:
            errors.append(f"Argument {token!r} is not key=value.")
        elif not key:
            errors.append(f"Argument {token!r} has an empty key.")
        elif "" in key.split("."):
            errors.append(f"Key {key!r} has an empty segment.")
        elif key == CONFIG_KEY:
            if config_path is not None:
                errors.append(f"Repeated key {CONFIG_KEY!r}: give each key once.")
            elif not text:
                errors.append(f"{CONFIG_KEY!r} needs a file path.")
            else:
                config_path = text
        else:
            _insert(tree, key, text, errors)
    if errors:
        raise ConfigValidationError(errors)
    return config_path, tree


# ---------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------

_NONE_TYPE = type(None)
_NULL_WORDS = ("null", "none")
_INT = re.compile(r"-?[0-9]+")
_FLOAT = re.compile(r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")

#: ``_convert`` returns ``(value, None)`` on success and ``(None, note)`` on
#: failure, where ``note`` is text appended to the error line (often empty).
#: A tuple, not an exception, because ``cli.py`` holds no handler but
#: ``DLTrainError`` and a Union tries several members in turn.
_Converted = Tuple[Any, Optional[str]]


def _json_container(text: str, container: type) -> _Converted:
    value, reason = parse_json(text)
    if reason is not None:
        return None, f": {reason}"
    if not isinstance(value, container):
        return None, f": the JSON text is not a {container.__name__}"
    return value, None


def _untyped(text: str) -> Any:
    """Type a value nothing declares: JSON, then a finite float, else the text."""
    value, reason = parse_json(text)
    if reason is None:
        return value
    if _FLOAT.fullmatch(text):
        number = float(text)
        if math.isfinite(number):
            return number
    return text


def _convert(annotation: Any, text: str) -> _Converted:
    """Convert ``text`` by ``annotation``; see :data:`_Converted` for the return shape."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is Any:
        return _untyped(text), None
    if annotation is _NONE_TYPE or annotation is None:
        if text.lower() in _NULL_WORDS:
            return None, None
        return None, ""
    if origin is Literal:
        for choice in args:
            if isinstance(choice, str):
                if text == choice:
                    return choice, None
                continue
            value, note = _convert(type(choice), text)
            if note is None and value == choice:
                return choice, None
        return None, ""
    if origin is Union:
        for member in args:
            if typing.get_origin(member) is Literal:
                value, note = _convert(member, text)
                if note is None:
                    return value, None
        if _NONE_TYPE in args and text.lower() in _NULL_WORDS:
            return None, None
        for member in args:
            if member is _NONE_TYPE or typing.get_origin(member) is Literal:
                continue
            value, note = _convert(member, text)
            if note is None:
                return value, None
        return None, ""
    if annotation is bool:
        if text in ("true", "false"):
            return text == "true", None
        return None, ""
    # DECISION plan-2026-10-30T132432-a865fa08/D-002: int and float are read
    # by their own pattern, never one through the other. Do NOT write
    # ``int(float(text))`` or ``float(text)`` bare: ``epochs=5.0`` and
    # ``epochs=1_000`` must be refused for an int field, and ``float("1_000")``,
    # ``float("nan")`` and ``float("inf")`` would all be accepted.
    if annotation is int:
        if _INT.fullmatch(text):
            return int(text), None
        return None, ""
    if annotation is float:
        if not _FLOAT.fullmatch(text):
            return None, ""
        number = float(text)
        if not math.isfinite(number):
            return None, ": not a finite number"
        return number, None
    if annotation is str:
        return text, None
    if isinstance(annotation, type) and issubclass(annotation, StrictConfig):
        return _json_container(text, dict)
    if origin in (list, tuple, collections.abc.Sequence):
        return _json_container(text, list)
    if origin in (dict, collections.abc.Mapping):
        return _json_container(text, dict)
    raise TypeError(
        f"train.api.cli cannot coerce the annotation {annotation!r}; "
        "add a case to _convert"
    )


def coerce(annotation: Any, text: str, dotted_path: str) -> Any:
    """Convert ``text`` as ``annotation`` demands.

    :param annotation: a field annotation.
    :param text: the raw command line text.
    :param dotted_path: the key, for the error line.
    :raises ConfigError: when the text does not fit.
    """
    value, note = _convert(annotation, text)
    if note is not None:
        raise ConfigError(invalid_value_message(dotted_path, annotation, text, note))
    return value


def _members(annotation: Any) -> List[Any]:
    if typing.get_origin(annotation) is Union:
        return list(typing.get_args(annotation))
    return [annotation]


def _coerce_value(
    annotation: Any, raw: Any, path: str, errors: List[str]
) -> Any:
    """Coerce one node of the raw tree: a leaf string or a nested dict."""
    if isinstance(raw, str):
        value, note = _convert(annotation, raw)
        if note is not None:
            errors.append(invalid_value_message(path, annotation, raw, note))
        return value
    for member in _members(annotation):
        if isinstance(member, type) and issubclass(member, StrictConfig):
            return _coerce_group(member, raw, path, errors)
    if annotation is Any:
        return {
            key: _coerce_value(Any, value, f"{path}.{key}", errors)
            for key, value in raw.items()
        }
    for member in _members(annotation):
        if typing.get_origin(member) in (dict, collections.abc.Mapping):
            member_args = typing.get_args(member)
            value_type = member_args[1] if len(member_args) == 2 else Any
            return {
                key: _coerce_value(value_type, value, f"{path}.{key}", errors)
                for key, value in raw.items()
            }
    errors.append(invalid_value_message(path, annotation, raw))
    return None


def _coerce_group(
    cls: Type[StrictConfig], raw: Mapping[str, Any], prefix: str, errors: List[str]
) -> Dict[str, Any]:
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls) if f.init]
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in names:
            errors.append(unknown_key_message(prefix, key, names))
            continue
        path = f"{prefix}.{key}" if prefix else key
        result[key] = _coerce_value(hints[key], value, path, errors)
    return result


def coerce_mapping(cls: Type[StrictConfig], raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Type a raw dotlist tree by the fields of ``cls``.

    :param cls: the config class the tree is meant for.
    :param raw: the nested dict from :func:`parse_dotlist`.
    :return: a nested dict of typed values; ``validate`` still has the last word.
    :raises ConfigValidationError: naming every unknown key and bad value.
    """
    errors: List[str] = []
    result = _coerce_group(cls, raw, "", errors)
    if errors:
        raise ConfigValidationError(errors)
    return result


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------


def run_task(
    config_class: Type[StrictConfig],
    function: Callable[..., Any],
    args: List[str],
) -> int:
    """Run one task command line.

    ``config=path.yaml`` loads first and the dotlist overrides merge on top.

    :param config_class: the task's config class; it types the dotlist.
    :param function: the keyword-only task function, called with the merged mapping.
    :param args: the arguments after the command name.
    :return: 0 when ``result.success`` is true, else 1.
    :raises ConfigError: for a bad token, key, value or config file.
    """
    config_path, raw = parse_dotlist(args)
    mapping = coerce_mapping(config_class, raw)
    if config_path is not None:
        mapping = merge(load_yaml_mapping(config_path), mapping)
    result = function(**mapping)
    return 0 if result.success else 1
