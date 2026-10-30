"""Exception types of the ``train.api`` package.

Nothing here imports anything, so the CLI help path can raise and catch them
at no cost. ``cli.main`` catches :class:`DLTrainError` and nothing else.

Hierarchy
---------
* :class:`DLTrainError` -- base of every error the API raises on purpose.
* :class:`ConfigError` -- a bad configuration; also a ``ValueError``.
* :class:`ConfigValidationError` -- ``ConfigError`` carrying one line per bad key.
* :class:`UnknownModelError` -- a model name the registry does not know; also a ``KeyError``.
* :class:`UnresolvedAutoError` -- an ``"auto"`` field no recipe could fill.
"""

from typing import Iterable, Tuple


class DLTrainError(Exception):
    """Base class of every error ``train.api`` raises on purpose."""


class ConfigError(DLTrainError, ValueError):
    """A configuration the API refuses."""


class ConfigValidationError(ConfigError):
    """A configuration with one or more bad keys.

    :param errors: one message per problem, each naming its dotted key.
    :ivar errors: the same messages as a tuple, in the order given.

    ``str(exc)`` joins the messages with newlines, so a caller that prints the
    exception shows every problem at once.
    """

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors: Tuple[str, ...] = tuple(errors)
        super().__init__("\n".join(self.errors))


class UnknownModelError(DLTrainError, KeyError):
    """A model name that is not in the registry.

    ``KeyError.__str__`` returns ``repr`` of its argument, which wraps the
    message in quotes. This class overrides ``__str__`` so the message prints
    as written.
    """

    def __str__(self) -> str:
        return " ".join(str(arg) for arg in self.args)


class UnresolvedAutoError(DLTrainError):
    """A field is still ``"auto"`` after resolution.

    The user can fix this by setting the field, so the CLI exits 2 for it.
    """
