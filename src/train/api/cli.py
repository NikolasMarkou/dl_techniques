"""Command line entry of ``python -m train``.

Interface contract
------------------
``main(argv)`` takes the arguments after the program name and returns the
process exit code: 0 on success, 2 when a :class:`~train.api.errors.DLTrainError`
was raised. Any other exception propagates, so Python exits 1 with a
traceback. Help text goes to stdout, error text to stderr.

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
import difflib
import logging
import sys
from typing import Callable, Dict, List, NamedTuple, Sequence

from dl_techniques.utils.logger import logger

from .errors import ConfigError, DLTrainError

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
    """Print the help text derived from :data:`COMMANDS`."""
    lines = [
        f"usage: {PROGRAM} <command> [key=value ...]",
        f"       {PROGRAM} <command> help",
        "",
        "commands:",
    ]
    width = max(len(name) for name in COMMANDS)
    for command in COMMANDS.values():
        lines.append(f"  {command.name.ljust(width)}  {command.summary}")
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
    """Print the registered task names, one per line."""
    if args not in ([], ["help"]):
        raise ConfigError(
            f"list_tasks takes no arguments, got: {' '.join(args)}"
        )
    sys.stdout.write("no tasks are registered\n")
    return 0


# One table drives dispatch and the help text, so help never lists a command
# that does not exist (D-007). Later phases add rows.
COMMANDS: Dict[str, Command] = {
    command.name: command
    for command in (
        Command("help", "show this help", _write_help),
        Command("list_tasks", "list the registered tasks", _list_tasks),
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
    if command is None:
        raise ConfigError(_unknown_command_message(args[0]))
    return command.run(args[1:])


def _unknown_command_message(name: str) -> str:
    """Build the unknown-command text, with a suggestion when one is close."""
    message = f"unknown command {name!r}."
    close = difflib.get_close_matches(name, list(COMMANDS), n=1)
    if close:
        message += f" Did you mean {close[0]!r}?"
    return message + f" Run '{PROGRAM} help' for the list."
