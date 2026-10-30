"""Turns a resolved :class:`~train.api.args.OptimArgs` into an optimizer.

This is a mapping and nothing else. The schedule and the optimizer are built by
``dl_techniques.optimization`` (``learning_rate_schedule_builder`` and
``optimizer_builder``); no ``keras.optimizers`` class is named in ``api/``.

Interface contract
------------------
``build_optimizer_and_schedule(args, steps_per_epoch, epochs)``
    :param args: a resolved :class:`~train.api.args.OptimArgs` (no ``"auto"``).
    :param steps_per_epoch: optimizer steps in one epoch; a positive ``int``.
    :param epochs: epochs the run lasts; a positive ``int``.
    :return: ``(optimizer, schedule)``. ``schedule`` is the learning-rate
        schedule object the optimizer holds, or a plain ``float`` when
        ``args.schedule`` is ``"constant"``.
    :raises ConfigValidationError: one line per problem, each naming its
        ``optim_args.<field>``: a field still ``"auto"``, an unknown
        ``optimizer`` or ``schedule``, a number out of range, both clips set,
        an option the chosen optimizer does not take, a constant schedule with
        a warmup, or a ``post_warmup`` decay with no steps left after warmup.

Where each field goes
---------------------
``clipnorm`` is a per-variable clip (``gradient_clipping_by_norm_local``) and
``global_clipnorm`` a global one (``gradient_clipping_by_norm``). The builder
reads those two keys only; a literal ``clipnorm`` key is dropped without a word
(D-013), which is the mistake ``test_optim_mapping`` guards.
``warmup_epochs`` becomes ``warmup_steps``, ``min_lr_ratio`` becomes ``alpha``,
and ``decay_over`` picks the cosine length. ``None`` leaves a key out of the
builder's config so the builder's own default applies, with one exception:
AdamW with ``weight_decay=None`` is built with ``0.0``, because the builder
would otherwise apply its own 0.004.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012);
``dl_techniques.optimization`` is imported inside the function.
"""

import dataclasses
import math
from typing import Any, Dict, List, Tuple

from .args import OptimArgs
from .config import AUTO, join_path
from .errors import ConfigValidationError

# ---------------------------------------------------------------------

#: Optimizers ``optim_args.optimizer`` accepts.
OPTIMIZERS = ("adamw", "adam", "sgd")
SCHEDULES = ("cosine_decay", "constant")
DECAYS = ("total", "post_warmup")

_BETAS = ("beta_1", "beta_2", "epsilon")

#: Options each optimizer refuses to be given. ``weight_decay`` is refused for
#: adam (use adamw) but not for sgd: sgd's value is the strength of an L2
#: regularizer on the model, which the optimizer never sees and never gets.
_REFUSED_OPTIONS = {
    "adamw": ("momentum",),
    "adam": ("momentum", "weight_decay"),
    "sgd": _BETAS,
}

# ---------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _problems(args: OptimArgs, steps_per_epoch: Any, epochs: Any) -> List[str]:
    """Return one line per reason ``args`` cannot be built; empty when it can."""
    lines: List[str] = []

    def bad(name: str, expected: str, value: Any) -> None:
        lines.append(
            f"Invalid value for {join_path('optim_args', name)!r}: "
            f"expected {expected}, got {value!r}"
        )

    for name, value in (("steps_per_epoch", steps_per_epoch), ("epochs", epochs)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            lines.append(f"Invalid value for {name!r}: expected a positive int, got {value!r}")

    unresolved = [
        field.name
        for field in dataclasses.fields(args)
        if isinstance(getattr(args, field.name), str) and getattr(args, field.name) == AUTO
    ]
    for name in unresolved:
        lines.append(
            f"Unresolved {join_path('optim_args', name)!r}: still {AUTO!r}; "
            "resolve_auto must run before the optimizer is built"
        )
    if unresolved:
        return lines

    optimizer = str(args.optimizer).strip().lower()
    if optimizer not in OPTIMIZERS:
        bad("optimizer", f"one of {', '.join(OPTIMIZERS)}", args.optimizer)
    if args.schedule not in SCHEDULES:
        bad("schedule", f"one of {', '.join(SCHEDULES)}", args.schedule)
    if args.decay_over not in DECAYS:
        bad("decay_over", f"one of {', '.join(DECAYS)}", args.decay_over)

    ranges = (
        ("lr", args.lr, lambda v: v > 0, "a positive finite number", False),
        ("warmup_start_lr", args.warmup_start_lr, lambda v: v >= 0, "a number >= 0", False),
        ("min_lr_ratio", args.min_lr_ratio, lambda v: 0 <= v <= 1, "a number in [0, 1]", False),
        ("warmup_epochs", args.warmup_epochs, lambda v: v >= 0, "an int >= 0", False),
        ("clipnorm", args.clipnorm, lambda v: v > 0, "null or a positive number", True),
        ("global_clipnorm", args.global_clipnorm, lambda v: v > 0, "null or a positive number", True),
        ("weight_decay", args.weight_decay, lambda v: v >= 0, "null or a number >= 0", True),
        ("momentum", args.momentum, lambda v: 0 <= v <= 1, "null or a number in [0, 1]", True),
        ("beta_1", args.beta_1, lambda v: 0 <= v < 1, "null or a number in [0, 1)", True),
        ("beta_2", args.beta_2, lambda v: 0 <= v < 1, "null or a number in [0, 1)", True),
        ("epsilon", args.epsilon, lambda v: v > 0, "null or a positive number", True),
    )
    for name, value, accepts, expected, optional in ranges:
        if value is None and optional:
            continue
        if not _is_number(value) or not math.isfinite(value) or not accepts(value):
            bad(name, expected, value)

    if args.clipnorm is not None and args.global_clipnorm is not None:
        lines.append(
            "Invalid value for 'optim_args.clipnorm' and 'optim_args.global_clipnorm': "
            "set one, not both (clipnorm clips each variable, global_clipnorm clips "
            f"all gradients together), got {args.clipnorm!r} and {args.global_clipnorm!r}"
        )

    if optimizer in _REFUSED_OPTIONS:
        for name in _REFUSED_OPTIONS[optimizer]:
            if getattr(args, name) is not None:
                lines.append(
                    f"Invalid value for {join_path('optim_args', name)!r}: "
                    f"the {optimizer} optimizer does not take it, got "
                    f"{getattr(args, name)!r}; set it to null"
                )

    if args.schedule == "constant" and _is_number(args.warmup_epochs) and args.warmup_epochs > 0:
        lines.append(
            "Invalid value for 'optim_args.warmup_epochs': a constant schedule has "
            f"no warmup, got {args.warmup_epochs!r}; set it to 0 or use cosine_decay"
        )

    if (
        args.schedule == "cosine_decay"
        and args.decay_over == "post_warmup"
        and isinstance(epochs, int)
        and _is_number(args.warmup_epochs)
        and args.warmup_epochs >= epochs
    ):
        lines.append(
            "Invalid value for 'optim_args.warmup_epochs': decay_over 'post_warmup' "
            f"needs epochs > warmup_epochs, got warmup_epochs={args.warmup_epochs!r} "
            f"and epochs={epochs!r}; shorten the warmup or use decay_over=total"
        )
    return lines


def build_optimizer_and_schedule(
    args: OptimArgs, steps_per_epoch: int, epochs: int
) -> Tuple[Any, Any]:
    """Build the optimizer and its learning-rate schedule; see the module docstring."""
    problems = _problems(args, steps_per_epoch, epochs)
    if problems:
        raise ConfigValidationError(problems)

    from dl_techniques.optimization import (
        learning_rate_schedule_builder,
        optimizer_builder,
    )

    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = steps_per_epoch * epochs
    if args.schedule == "constant":
        schedule: Any = float(args.lr)
    else:
        decay_steps = total_steps if args.decay_over == "total" else total_steps - warmup_steps
        schedule = learning_rate_schedule_builder(
            {
                "type": args.schedule,
                "learning_rate": args.lr,
                "decay_steps": decay_steps,
                "warmup_steps": warmup_steps,
                "warmup_start_lr": args.warmup_start_lr,
                "alpha": args.min_lr_ratio,
            }
        )

    optimizer_name = str(args.optimizer).strip().lower()
    config: Dict[str, Any] = {"type": optimizer_name}
    # DECISION plan-2026-10-30T132432-a865fa08/D-039: the two clips go to two
    # different builder keys. Do NOT pass ``clipnorm`` as a key of its own:
    # optimizer_builder reads ``gradient_clipping_by_norm_local`` and
    # ``gradient_clipping_by_norm`` only and drops anything else silently, so a
    # run configured with a clip would train unclipped and never say so.
    if args.clipnorm is not None:
        config["gradient_clipping_by_norm_local"] = args.clipnorm
    if args.global_clipnorm is not None:
        config["gradient_clipping_by_norm"] = args.global_clipnorm
    for name in ("beta_1", "beta_2", "epsilon", "momentum"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if optimizer_name == "adamw":
        # DECISION plan-2026-10-30T132432-a865fa08/D-039: AdamW with no decay
        # is built with 0.0. Do NOT leave the key out: the builder then applies
        # its own 0.004, and ``weight_decay=null`` would mean a decay the
        # caller never asked for.
        config["weight_decay"] = 0.0 if args.weight_decay is None else args.weight_decay

    return optimizer_builder(config, schedule), schedule
