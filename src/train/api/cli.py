"""Command line entry of ``python -m train``.

Interface contract
------------------
``main(argv)`` takes the arguments after the program name and returns the
process exit code: 0 on success, 2 when a :class:`~train.api.errors.DLTrainError`
was raised. Any other exception propagates, so Python exits 1 with a
traceback. Help text goes to stdout, error text to stderr.

Commands are the rows of ``COMMANDS`` (``help``, ``list_tasks``,
``list_models``, ``export``, ``predict``, ``convert``), then the tasks in
:data:`train.api.registry.TASKS`. A task command with no arguments, or with a lone
help word, prints its config keys; any other arguments go to ``run_task``.
``list_*`` commands list instead of printing help, even when given ``help``.

``export model= out= format=``
    Load the run (a run directory, or a ``.keras`` file beside its record) and write it with
    :func:`train.api.exporting.export`; the input shape comes from ``task.json``. ``format``
    is one of ``train.api.exporting.FORMATS``. Every check that needs no model runs before
    the model is loaded and before TensorFlow is imported: the keys (none empty), that
    ``model`` exists, ``format``, the suffix of ``out``, and that ``out`` and
    ``<out>.task.json`` are free. The exported model takes inputs standardized with the run's
    mean and std and returns raw logits (D-068); ``<out>.task.json`` is written beside it and
    the log says so. Exit 0 and ONLY the written path on stdout (whatever Keras prints while
    exporting goes to stderr, failed runs too), else 2.

``predict model= data= out=``
    ``data`` is an image file or a directory of images (searched recursively, hidden
    entries skipped, by the suffixes the folder loader reads, sorted by path). ``out`` is a
    new JSON file holding a list with one ``{"path", "labels", "scores"}`` object per image:
    every class, best first, the :class:`~train.api.predictor.Predictor`'s answer. An
    existing ``out`` (a dangling symlink included), or one under a file, is refused before
    the model is loaded. Exit 0, else 2.

    For a SEGMENTATION run (the record's ``task``) ``out`` is a new DIRECTORY instead: one
    ``<stem>.png`` per image (the class id of every pixel, at the size of that image after its EXIF
    orientation; 8 bit, 16 bit above 255 classes) and, written last, ``index.json`` with the class
    names and one ``{"path", "mask", "size"}`` per image
    (:func:`train.api.segmentation_predictor.write_predictions`). Two images with one stem are
    refused before the model is loaded, naming both; so is a record whose ``output`` is ``logits``.

``convert <dataset> key=value ...``
    Writes a source dataset as index masks and CSV listings the segmentation task reads, outside the
    source tree, resumably and without deleting (:mod:`train.api.convert`; ``ade20k``, ``coco``). Exit 0 and ONLY
    the output directory on stdout, else 2.

``export`` and ``predict`` take exactly their keys (no ``config=``, no nested key), a bare command or
a lone help word prints them, and the help path imports no TensorFlow.

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
    Parse, coerce, merge over ``config=``, run ``validate`` on the merged
    mapping, call ``function(**mapping)`` and return 0 when ``result.success``
    else 1. ``function`` is the keyword-only function a Python caller uses
    (D-008); it validates again, but a bad file or a missing key is refused
    here with exit 2 before any signature can raise a ``TypeError``.

Coercion rules (the text is the whole token after the first ``=``):

* ``float``: decimal or exponent form (``3e-4``, ``1E-3``, ``.5``), finite.
* ``int``: ``-?[0-9]+`` only (the YAML loader reads the same pattern); ``5.0``,
  ``1_000`` and a digit string past the interpreter's limit are refused.
* ``bool``: ``true`` or ``false`` only.
* ``str``: verbatim, so ``no`` stays ``"no"``.
* ``Optional``: ``null`` or ``none`` (any case) gives ``None``.
* list, tuple, dict, group: JSON text; NaN and Infinity are refused.
* a path-form group (``data``, marked with ``PATH_FORM`` in its field metadata):
  text that does not open a JSON object is the path of a YAML file holding the
  group, from the command line and from a ``config=`` file alike (the Python
  function accepts the same form); ``run_task`` reads the file, and an
  unreadable file or a bad key in it is a ``ConfigError``.
* untyped (``Any``) values: JSON, then ``null`` or ``none`` (any case) as
  ``None``, then a finite float (``.5``, ``010``), else the text. Write
  ``'"none"'`` for the string.

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
import contextlib
import dataclasses
import difflib
import json
import logging
import math
import re
import sys
import typing
from pathlib import Path
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
    PATH_FORM,
    StrictConfig,
    candidate_groups,
    describe_annotation,
    invalid_value_message,
    load_yaml_mapping,
    merge,
    parse_int,
    parse_json,
    unknown_key_message,
    unknown_tag_message,
    validate,
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


def _is_help_request(args: List[str]) -> bool:
    """Return whether ``args`` is empty or one help word (``help``, ``-h``, ``--help``)."""
    return not args or (len(args) == 1 and args[0] in HELP_WORDS)


def _list_tasks(args: List[str]) -> int:
    """Print the registered tasks, one ``name  summary`` line each."""
    if not _is_help_request(args):
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
    path. ``list_models`` with a help word lists, like ``list_tasks``.
    """
    task = None
    if not (len(args) == 1 and args[0] in HELP_WORDS):
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


class Option(NamedTuple):
    """One key of a command that is not a task: its name and its help text."""

    key: str
    help: str


EXPORT_OPTIONS = (
    Option("model", "a run directory (its best_model.keras) or a .keras file beside its task.json"),
    Option("out", "the file or directory to write; it must not exist (<out>.task.json is written beside it)"),
    Option("format", "keras, saved_model, tflite or onnx"),
)
EXPORT_NOTES = (
    "input contract: the exported model takes float32 images (batch, height, width, channels)",
    "with pixels scaled to [0, 1] and then standardized, (x - mean) / std, using the mean and",
    "std in <out>.task.json; it returns raw logits (no softmax). A classification model gives one",
    "score per class name in that file for each image: (batch, classes). A segmentation model gives",
    "one score per class per pixel: (batch, height, width, classes), at the input_size in that file",
    "(height and width are fixed to it, the batch is free); the mask is the argmax over the last",
    "axis, after the caller resizes the scores to the picture. mixed_float16 runs differ from Keras",
    "by up to about 1e-4 in onnx.",
)
PREDICT_OPTIONS = (
    Option("model", "a run directory (its best_model.keras) or a .keras file beside its task.json"),
    Option("data", "an image file, or a directory of images (searched recursively)"),
    Option("out", "a new .json file; it must not exist (a segmentation model writes a new directory of PNG masks and an index.json there)"),
)


def _write_command_help(
    name: str, summary: str, options: Sequence[Option], notes: Sequence[str] = ()
) -> int:
    """Print the usage and the keys of an ``export`` or ``predict`` command."""
    width = max(len(o.key) for o in options)
    lines = [
        f"usage: {PROGRAM} {name} " + " ".join(f"{o.key}=<{o.key}>" for o in options),
        "",
        summary,
        "",
        "keys (all required):",
        *[f"  {o.key.ljust(width)}  {o.help}" for o in options],
        "",
        *([*notes, ""] if notes else []),
    ]
    sys.stdout.write("\n".join(lines))
    return 0


def _command_values(
    name: str, args: List[str], options: Sequence[Option]
) -> Dict[str, str]:
    """Return the text of each of ``options`` given in ``args``.

    :raises ConfigValidationError: naming every unknown or nested key and every missing key.
    :raises ConfigError: for ``config=``, which these commands do not read.
    """
    config_path, raw = parse_dotlist(args)
    if config_path is not None:
        raise ConfigError(f"{name} takes no config=; give {', '.join(o.key for o in options)}")
    names = [o.key for o in options]
    errors: List[str] = []
    for key, value in raw.items():
        if key not in names:
            errors.append(unknown_key_message("", key, names))
        elif not isinstance(value, str):
            errors.append(f"{key!r} takes one value, not sub-keys.")
    errors += [f"Missing required key {key!r}." for key in names if key not in raw]
    errors += [f"Invalid value for {key!r}: it is empty." for key in names if raw.get(key) == ""]
    if errors:
        raise ConfigValidationError(errors)
    return {key: raw[key] for key in names}


def _existing_model_path(text: str) -> Path:
    """Return the path ``text`` names, which must exist (checked before TensorFlow is imported)."""
    path = Path(text)
    if not path.exists():
        raise ConfigError(
            f"Invalid value for 'model': {path} does not exist; give a run directory, or a "
            ".keras file beside a task.json"
        )
    return path


def _export(args: List[str]) -> int:
    """Run ``export model= out= format=``; see the module docstring."""
    if _is_help_request(args):
        return _write_command_help(
            "export", COMMANDS["export"].summary, EXPORT_OPTIONS, EXPORT_NOTES
        )
    values = _command_values("export", args, EXPORT_OPTIONS)
    from .exporting import check_target, export

    # DECISION plan-2026-10-30T132432-a865fa08/D-064: the model path and the target are checked
    # BEFORE ``load_model``, and the traced shape is ``task.json``'s ``input_shape``. Do NOT load
    # first and let ``export`` refuse (a bad ``out`` would cost a TensorFlow import and a model
    # load), and do NOT ask the model for its shape (a ViT has no ``inputs``). The model path
    # comes first: ``check_target`` imports tf2onnx (and so TensorFlow) for ``onnx``.
    model_path = _existing_model_path(values["model"])
    target = check_target(values["out"], values["format"], with_record=True)
    from .predictor import input_shape_of, load_model

    # DECISION plan-2026-10-30T132432-a865fa08/D-068: stdout carries ONLY the written path, so
    # ``OUT=$(python -m train export ...)`` works. Keras prints "Saved artifact at ..." and the
    # endpoint list to stdout whatever ``verbose`` says; do NOT filter those lines, send the
    # whole call's stdout to stderr (an exception included, the context restores it).
    with contextlib.redirect_stdout(sys.stderr):
        loaded = load_model(model_path)
        written = export(
            model=loaded.model,
            out=target,
            format=values["format"],
            input_shape=input_shape_of(loaded.record),
            record=loaded.record,
        )
    sys.stdout.write(f"{written}\n")
    return 0


def _convert(args: List[str]) -> int:
    """Run ``convert <dataset> key=value ...``; see :mod:`train.api.convert`."""
    # The module is imported here so the help path loads none of it (D-012).
    from .convert import run_convert

    return run_convert(args)


def _image_files(data: Path) -> List[Path]:
    """Return the image files ``data`` names: the file itself, or a directory's, sorted.

    :raises ConfigError: for a path that is neither, or a directory holding no image.
    """
    from train.common.classification_data import IMAGE_SUFFIXES

    # DECISION plan-2026-10-30T132432-a865fa08/D-064: a directory is searched RECURSIVELY. Do NOT
    # list only its top level: a folder of class directories (the layout the folder loader reads)
    # would then hold "no image". The suffixes are the loader's own, not a second list.
    if data.is_file():
        return [data]
    found = sorted(
        item
        for item in data.rglob("*")
        if item.is_file()
        and item.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in item.relative_to(data).parts)
    )
    if not found:
        raise ConfigError(
            f"Invalid value for 'data': {data} holds no image ({', '.join(IMAGE_SUFFIXES)})"
        )
    return found


def _predict(args: List[str]) -> int:
    """Run ``predict model= data= out=``; see the module docstring."""
    if _is_help_request(args):
        return _write_command_help("predict", COMMANDS["predict"].summary, PREDICT_OPTIONS)
    values = _command_values("predict", args, PREDICT_OPTIONS)
    from .exporting import check_free

    out = Path(values["out"])
    check_free(out, "out", "predict")
    model_path = _existing_model_path(values["model"])
    data = Path(values["data"])
    if not data.exists():
        raise ConfigError(f"Invalid value for 'data': {data} does not exist")
    from .predictor import CHUNK, SEGMENTATION, load_model, record_beside

    files = _image_files(data)
    # The task is read from the record alone, so what only the files and the record decide is refused
    # before the model is loaded (D-064): a stem two images share, an output a PNG cannot hold.
    record = record_beside(model_path)
    if record["task"] == SEGMENTATION:
        return _predict_masks(model_path, record, files, out)
    predictor = load_model(model_path)
    entries: List[Dict[str, Any]] = []
    for start in range(0, len(files), CHUNK):
        chunk = files[start : start + CHUNK]
        answer = predictor.predict([str(file) for file in chunk])
        entries += [
            {"path": str(file), "labels": labels, "scores": scores}
            for file, labels, scores in zip(chunk, answer["labels"], answer["scores"])
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
    sys.stdout.write(f"{out}\n")
    return 0


def _predict_masks(model_path: Path, record: Mapping[str, Any], files: List[Path], out: Path) -> int:
    """Write the masks of ``files`` as PNGs and an index into the new directory ``out``."""
    from .predictor import load_model
    from .segmentation_predictor import check_png_output, mask_names, write_predictions

    check_png_output(record)
    mask_names(files)
    write_predictions(load_model(model_path), files, out)
    sys.stdout.write(f"{out}\n")
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
        members = typing.get_args(annotation) if typing.get_origin(annotation) is Union else ()
        if members and all(isinstance(m, type) and issubclass(m, StrictConfig) for m in members):
            # A union of groups (``data``): the keys of each member, told apart by ``format``.
            for member in members:
                lines += _field_help_lines(member, key)
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
        "A mapping given on the command line is merged into the one in config=,",
        "so model_args={} adds nothing and does not clear a file's model_args;",
        "edit the file to drop a key.",
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
        Command("export", "write a trained run as keras, saved_model, tflite or onnx", _export),
        Command("predict", "label the images in a file or directory with a trained run", _predict),
        Command("convert", "write a source dataset (ade20k, coco) as index masks and CSV listings", _convert),
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
        _register_builtins()
        return _dispatch(args)
    except DLTrainError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


def _register_builtins() -> None:
    """Register the built-in families and tasks (idempotent, imports no TensorFlow).

    ``main`` does this itself, so ``train.api.main([...])`` from Python lists and
    dispatches exactly what ``python -m train`` does (review-iter-3 concern 22): both
    registrations hold ``"module:attr"`` text and import no model or task code, which
    ``test_cli_help_allocates_nothing`` pins for the help path. A row already in
    the registry is left alone.
    """
    # DECISION plan-2026-10-30T132432-a865fa08/D-071: `main` registers, not `__main__`. Do NOT
    # move this back into `__main__.py` (a Python caller of `train.api.main` then gets
    # "unknown command 'image_classification'" for the only task there is), and do NOT make
    # it a module-level call in this file (the registries are swapped by tests and must stay
    # empty at import).
    from .recipes import register_builtin_models
    from .tasks import register_builtin_tasks

    register_builtin_models()
    register_builtin_tasks()


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
    if text.lower() in _NULL_WORDS:
        return None
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
        number, note = parse_int(text)
        return (number, None) if note is None else (None, note)
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


def _takes_json(annotation: Any) -> bool:
    """Whether some member of ``annotation`` is a list, mapping or group, which the command line writes as JSON."""
    return any(
        typing.get_origin(member) in (list, tuple, dict, collections.abc.Sequence, collections.abc.Mapping)
        or (isinstance(member, type) and issubclass(member, StrictConfig))
        for member in _members(annotation)
    )


def _json_hint(annotation: Any, text: str, note: str) -> str:
    """Return ``note`` plus how a list or mapping is written, for a token that looks like one but is not JSON.

    JSON of the wrong shape (a mapping for a list) keeps its own reason and gets no hint.

    ``class_names=[pet,background]`` is the usual attempt: the list is JSON (``'["pet","background"]'``,
    quoted for the shell) for every key of every task, a string inside it is quoted.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-087
    # The rule stays "a list is JSON" for every key of every task; classification has no list-of-text
    # key at all. Do NOT teach the coercer to read `[a,b]` as a list of text: `[1,2]` and `[a,b]` would
    # follow different rules and a text that holds a comma could not be written. The message says how
    # (review-iter-3 concern 8: step 32 lost an attempt to it).
    if text.lstrip()[:1] in ("[", "{") and _takes_json(annotation) and parse_json(text)[1] is not None:
        return (
            f"{note}; a list or a mapping is written as JSON, with every text quoted, e.g. "
            "'[\"a\",\"b\"]' (single quotes around it for the shell)"
        )
    return note


def coerce(annotation: Any, text: str, dotted_path: str) -> Any:
    """Convert ``text`` as ``annotation`` demands.

    :param annotation: a field annotation.
    :param text: the raw command line text.
    :param dotted_path: the key, for the error line.
    :raises ConfigError: when the text does not fit.
    """
    value, note = _convert(annotation, text)
    if note is not None:
        raise ConfigError(
            invalid_value_message(dotted_path, annotation, text, _json_hint(annotation, text, note))
        )
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
            errors.append(
                invalid_value_message(path, annotation, raw, _json_hint(annotation, raw, note))
            )
        return value
    groups = [
        member
        for member in _members(annotation)
        if isinstance(member, type) and issubclass(member, StrictConfig)
    ]
    if groups:
        # A union of groups (``data``) is told apart by its ``format`` key.
        candidates = candidate_groups(groups, raw)
        if not candidates:
            errors.append(unknown_tag_message(path, groups, raw))
            return None
        return _coerce_group(candidates[0], raw, path, errors)
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
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    names = list(fields)
    result: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in names:
            errors.append(unknown_key_message(prefix, key, names))
            continue
        path = f"{prefix}.{key}" if prefix else key
        if _is_path_text(fields[key], value):
            # A group that also takes a file path (``data=data.yaml``) keeps the text
            # as a path: run_task loads the file, as the Python function does.
            result[key] = value
            continue
        result[key] = _coerce_value(hints[key], value, path, errors)
    return result


def _is_path_text(field: dataclasses.Field, value: Any) -> bool:
    """Return whether ``value`` is the path form of a ``path_form`` group field.

    :param field: the field being coerced.
    :param value: its raw token: text, or a nested dict.
    :return: ``True`` for text that does not open a JSON object, on a field whose
        metadata carries :data:`~train.api.config.PATH_FORM`.
    """
    return (
        bool(field.metadata.get(PATH_FORM))
        and isinstance(value, str)
        and not value.lstrip().startswith("{")
    )


def _expand_path_forms(cls: Type[StrictConfig], mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``mapping`` with each path-form field's text replaced by the file's mapping.

    :raises ConfigError: for a file that cannot be read, from ``load_yaml_mapping``.
    """
    expanded = dict(mapping)
    for field in dataclasses.fields(cls):
        value = expanded.get(field.name)
        reader = field.metadata.get(PATH_FORM)
        if reader and isinstance(value, str):
            # ``PATH_FORM`` is ``True`` (read the YAML as it is) or the field's own reader.
            expanded[field.name] = reader(value) if callable(reader) else load_yaml_mapping(value)
    return expanded


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
    # DECISION plan-2026-10-30T132432-a865fa08/D-034: a file is not coerced, so
    # it can hold an unknown key, a missing required key or a non-string key.
    # Do NOT leave that to the keyword-only signature: it raises TypeError,
    # which exits 1, the code for a failed run.
    # A path-form field (``data=data.yaml``, or ``data: data.yaml`` in a config file) is
    # read here, so a bad file or key is refused with exit 2 like any other.
    mapping = _expand_path_forms(config_class, mapping)
    validate(config_class, mapping)
    result = function(**mapping)
    return 0 if result.success else 1
