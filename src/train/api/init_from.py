"""Start a run from an earlier run's weights (``init``): the transfer engine.

The unit of a transfer is the individual weight, keyed by its Keras variable path. Every weight of
the target is either copied from the source weight at the same path (after a declared cross-task
prefix), resampled by a rule its family declares, reset as a declared head, or grown as a new
block of a deeper variant; every source weight is either used or dropped as a declared head.
Anything else refuses the whole transfer before the first ``assign`` (complete-or-raise in both
directions), so a refusal leaves every target weight as it was. Pass ``model.weights``: in Keras
3.8 it holds the BatchNorm moving statistics and no ``SeedGenerator`` state (that is in
``model.variables``), so a dropout seed is never transferred.

Nothing here imports TensorFlow or Keras at module scope; the one resample rule imports the
DINOv2 bicubic matrix inside the function.

Interface contract
------------------
``TransferSpec``
    One family's declaration. ``head``: path prefixes of the final layer, reset when the head is
    not reused; ``resample``: target path to rule name (a key of :data:`RULES`); ``grow``: a regex
    matched at the start of a path, naming a depth-indexed block a deeper variant may add;
    ``pinned``: factory arguments the target takes from the source; ``fixed``: geometry keys the
    family cannot change; ``sources``: ``(task, family)`` of another task to the
    :class:`CrossTask` that maps its paths.

    :raises ValueError: for a ``resample`` rule name that is not in :data:`RULES`.

``plan_transfer(source_weights, target_weights, spec, relation=None, head_reset=False, name=...)``
    :param source_weights: the source model's ``weights``.
    :param target_weights: the target model's ``weights``.
    :param spec: the target family's :class:`TransferSpec`.
    :param relation: ``None`` for a source of the same family and task; for a cross-task source,
        the :class:`CrossTask` entry of ``spec.sources``, whose head is always reset.
    :param head_reset: same task only: reset the target's head and drop the source's.
    :param name: what the refusal calls the target (``family/variant``).
    :return: a :class:`TransferPlan`. Nothing is assigned; values are read only to resample.
    :raises ConfigError: a variable path held twice on one side, or a resample rule refusing the
        shapes (a smaller or non-square grid).

``check(plan)``
    :raises ConfigError: when a target weight has no source weight of its shape and dtype and no
        declared rule, or a source weight is neither used nor dropped; the message names
        ``<n> of <m>`` weights, the first five paths with both shapes, ``+<k> more`` and
        "nothing was copied".

``apply(plan)``
    Calls :func:`check`, then assigns every copy and resample. Returns ``None``.

``vit_pos_bicubic(value, target_shape)``
    :param value: a ``(1, 1 + m * m, d)`` table, class row first, then the row-major grid.
    :param target_shape: ``(1, 1 + n * n, d)`` with ``n > m``.
    :return: the class row unchanged and the grid resampled per axis with torch's bicubic kernel
        (``align_corners=False``, no antialiasing), in ``value``'s dtype.
    :raises ConfigError: a smaller grid, a non-square grid, or another batch or width.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from dl_techniques.utils.logger import logger

from .errors import ConfigError

#: How many offending paths a refusal lists before ``+<k> more``.
SHOWN = 5


@dataclass(frozen=True)
class CrossTask:
    """How a source of another task maps onto this family: a target prefix and the dropped source head."""

    prefix: str = ""
    drop: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TransferSpec:
    """One family's transfer declaration (fields in the module docstring)."""

    head: Tuple[str, ...] = ()
    resample: Mapping[str, str] = field(default_factory=dict)
    grow: Optional[str] = None
    pinned: Tuple[str, ...] = ()
    fixed: Tuple[str, ...] = ()
    sources: Mapping[Tuple[str, str], CrossTask] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.resample.values()) - set(RULES))
        if unknown:
            raise ValueError(f"unknown resample rule(s) {unknown}; the rules are {sorted(RULES)}")


@dataclass(frozen=True)
class TransferPlan:
    """What :func:`apply` will do, and what :func:`check` refuses. Paths are the target's unless named."""

    name: str
    n_target: int
    n_source: int
    copies: Tuple[Tuple[Any, Any], ...]
    copied: Tuple[str, ...]
    resampled: Tuple[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], ...]
    reset: Tuple[str, ...]
    grown: Tuple[str, ...]
    dropped: Tuple[str, ...]  # source paths
    missing: Tuple[str, ...]
    unused: Tuple[str, ...]  # source paths


def _by_path(weights: Sequence[Any], side: str) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    for variable in weights:
        if variable.path in found:
            raise ConfigError(f"Invalid value for 'init': the {side} holds two weights at {variable.path}")
        found[variable.path] = variable
    return found


def _under(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)


def _block(pattern: Optional[str], path: str) -> Optional[str]:
    match = re.match(pattern, path) if pattern else None
    return match.group(0) if match else None


def _shape(variable: Any) -> Tuple[int, ...]:
    return tuple(int(n) for n in variable.shape)


def _side(variable: Any) -> str:
    return f"{_shape(variable)} {variable.dtype}" if variable is not None else "none"


def plan_transfer(source_weights: Sequence[Any], target_weights: Sequence[Any], spec: TransferSpec,
                  relation: Optional[CrossTask] = None, head_reset: bool = False,
                  name: str = "the target model") -> TransferPlan:
    """Pair every weight by path under ``spec``; assign nothing (contract in the module docstring)."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-005: one weight per Keras variable path, never
    # whole layers by name (a name loader loses a layer on one differing weight and sees no nested
    # backbone) and never names normalized (auto-named layers need a declared positional rule).
    prefix = relation.prefix if relation is not None else ""
    drop = relation.drop if relation is not None else (spec.head if head_reset else ())
    reset_heads = spec.head if (relation is not None or head_reset) else ()
    raw = _by_path(source_weights, "source")
    dropped = tuple(path for path in raw if _under(path, drop))
    source = {prefix + path: (path, var) for path, var in raw.items() if not _under(path, drop)}
    source_blocks = {_block(spec.grow, path) for path in source} - {None}
    copies, copied, resampled, reset, grown, missing = [], [], [], [], [], []
    used = set()
    for path, var in _by_path(target_weights, "target").items():
        src = source.get(path, (None, None))[1]
        rule = spec.resample.get(path)
        if _under(path, reset_heads):
            reset.append(path)
        elif src is not None and _shape(src) == _shape(var) and src.dtype == var.dtype:
            copies.append((var, src))
            copied.append(path)
            used.add(path)
        elif src is not None and rule is not None and src.dtype == var.dtype:
            copies.append((var, RULES[rule](np.asarray(src.numpy()), _shape(var))))
            resampled.append((path, rule, _shape(src), _shape(var)))
            used.add(path)
        elif src is None and _block(spec.grow, path) not in source_blocks | {None}:
            grown.append(path)
        else:
            missing.append(f"{path}: target {_side(var)}, source {_side(src)}")
    unused = tuple(f"{original} {_side(var)}" for path, (original, var) in source.items() if path not in used)
    return TransferPlan(name, len(target_weights), len(source_weights), tuple(copies), tuple(copied),
                        tuple(resampled), tuple(reset), tuple(grown), dropped, tuple(missing), unused)


def _refuse(problems: Sequence[str], total: int, whose: str, what: str) -> None:
    if problems:
        more = f" (+{len(problems) - SHOWN} more)" if len(problems) > SHOWN else ""
        raise ConfigError(f"Invalid value for 'init': {len(problems)} of {total} weights of {whose} {what}: "
                          f"{'; '.join(problems[:SHOWN])}{more}; nothing was copied")


def check(plan: TransferPlan) -> None:
    """Complete-or-raise in both directions (contract in the module docstring)."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-006: no skip_mismatch and no override; a partial
    # load is refused in BOTH directions, since a loader that returns normally on a partial match
    # has left a trunk random without a word. Do not soften either half into a warning.
    _refuse(plan.missing, plan.n_target, plan.name, "have no source weight of their shape and no declared rule")
    _refuse(plan.unused, plan.n_source, "the source run", "are neither used nor dropped by a declared rule")


def apply(plan: TransferPlan) -> None:
    """Check, then assign every copy and resample; a refusal assigns nothing."""
    check(plan)
    for variable, value in plan.copies:
        variable.assign(value)
    logger.info(f"init: {len(plan.copied)} weights copied, {len(plan.resampled)} resampled, "
                f"{len(plan.reset)} reset, {len(plan.grown)} grown, {len(plan.dropped)} source weights dropped")


def _grid(rows: int) -> int:
    side = int(round(rows ** 0.5))
    if side * side != rows:
        raise ConfigError(f"Invalid value for 'init': vit pos_embed: a table of 1 + {rows} rows is not a square "
                          "grid; only a square grid is resampled")
    return side


def vit_pos_bicubic(value: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Resample a ViT position table to a larger square grid (contract in the module docstring)."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-009: torch's bicubic without antialiasing, to a
    # LARGER grid only. Do not allow a smaller grid: downsampling without antialiasing aliases, and
    # timm antialiases, so a shrink would silently differ from the reference implementation.
    from dl_techniques.models.vision.dino.dino_v2 import torch_bicubic_resize_matrix

    if len(target_shape) != 3 or target_shape[0] != value.shape[0] or target_shape[2] != value.shape[2]:
        raise ConfigError(f"Invalid value for 'init': vit pos_embed {tuple(value.shape)} cannot become "
                          f"{tuple(target_shape)}; only the number of grid rows may change")
    m, n = _grid(value.shape[1] - 1), _grid(target_shape[1] - 1)
    if n < m:
        raise ConfigError(f"Invalid value for 'init': vit pos_embed would shrink its grid from {m}x{m} to {n}x{n}; "
                          "only a larger grid is resampled (bicubic, no antialiasing)")
    matrix = torch_bicubic_resize_matrix(m, n)
    grid = value[:, 1:].astype(np.float64).reshape(value.shape[0], m, m, value.shape[2])
    grid = np.einsum("ai,bijd,cj->bacd", matrix, grid, matrix).reshape(value.shape[0], n * n, value.shape[2])
    return np.concatenate([value[:, :1], grid.astype(value.dtype)], axis=1)


#: Resample rules by the name a :class:`TransferSpec` declares.
RULES: Dict[str, Callable[[np.ndarray, Tuple[int, ...]], np.ndarray]] = {"vit_pos_bicubic": vit_pos_bicubic}
