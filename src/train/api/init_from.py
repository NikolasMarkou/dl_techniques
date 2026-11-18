"""Start a run from an earlier run's weights (``init``): the transfer engine.

The unit of a transfer is the individual weight, keyed by its Keras variable path relative to its
model (a leading ``<model.name>/`` dropped, on both sides; nothing else renamed). Every weight of
the target is either copied from the source weight at the same path (after a declared cross-task
prefix), resampled by a rule its family declares, reset as a declared head, or grown as a new
block of a deeper variant; every source weight is either used or dropped as a declared head.
Anything else refuses the whole transfer before the first ``assign`` (complete-or-raise in both
directions), so a refusal leaves every target weight as it was. Pass ``model.weights``: in Keras
3.8 it holds the BatchNorm moving statistics and no ``SeedGenerator`` state (that is in
``model.variables``), so a dropout seed is never transferred.

The pre-claim half, :func:`read_source`, checks a source run directory before the run directory is
claimed: its files, its record, the task and family pair, the family's ``fixed`` geometry and the
input scaling against what this run will record, the pinned factory arguments, and the archive's
sha256. It reads JSON and hashes bytes; it loads no model, and for an absolute path imports no
TensorFlow, Keras or numpy.

Nothing here imports TensorFlow, Keras or numpy at module scope (``registry`` rows import the
declarations, and ``python -m train help`` stays light); numpy and the DINOv2 bicubic matrix are
imported inside the functions that use them.

Interface contract
------------------
``TransferSpec`` (declared in :mod:`train.api.registry`)
    One family's declaration; every path in it is relative to the model. ``head``: path prefixes
    of the final layer, reset when the head is not reused; ``resample``: target path to rule name
    (a key of :data:`RULES`); ``grow``: a regex matched at the start of a path, naming a
    depth-indexed block a deeper variant may add; ``pinned``: factory arguments the target takes
    from the source; ``fixed``: geometry keys the family cannot change (``channels``,
    ``lookback``, ``horizon``); ``sources``: ``(task, family)`` of another task to the
    :class:`~train.api.registry.CrossTask` that maps its paths; ``positional``: top-level path
    components inside which weights whose paths differ pair by order, when both sides hold the
    same count and every pair has equal shape and dtype.

``plan_transfer(source_model, target_model, spec, relation=None, head_reset=False, name=...)``
    :param source_model: the source model (anything with ``name`` and ``weights``).
    :param target_model: the target model.
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
        ``<n> of <m>`` weights, the first five paths (relative to their model) with both shapes,
        ``+<k> more`` and "nothing was copied".

``apply(plan)``
    Calls :func:`check`, then assigns every copy and resample. Returns ``None``.

``read_source(init, trainer, config, context)``
    :param init: the ``init`` value (text or ``Path``); a relative path is anchored at the repo root as ``out``
        is (:func:`train.api.run_dir.resolve_out`).
    :param trainer: the task's trainer: its ``task`` and its ``init_identity(config, context)`` hook.
    :param config: this run's resolved config (its ``model_args`` are the request's).
    :param context: the ``AutoContext`` of this run (``family``).
    :return: a :class:`Source`. Nothing is written; ``best_model.keras`` is hashed, not opened as a model.
    :raises ConfigError: naming ``init`` (or the ``model_args`` key a pin refuses), in this order: the row
        declares no transfer (its ``transfer_reason``); the trainer's ``init_identity`` refuses (the default
        does); the directory is missing or is a file; ``task.json``, ``config.json`` or ``best_model.keras``
        is missing; the record fails the predictor's record check or ``config.json`` is no JSON object; the
        source's task and family are neither this run's nor a declared cross-task pair; a ``fixed`` key of
        this family or the input scaling differs; a pinned argument is not recorded in the source's
        ``config.json`` ``used.model_args``, or the request's ``model_args`` gives it another value.

``vit_pos_bicubic(value, target_shape)``
    :param value: a ``(1, 1 + m * m, d)`` table, class row first, then the row-major grid.
    :param target_shape: ``(1, 1 + n * n, d)`` with ``n > m``.
    :return: the class row unchanged and the grid resampled per axis with torch's bicubic kernel
        (``align_corners=False``, no antialiasing), in ``value``'s dtype.
    :raises ConfigError: a smaller grid, a non-square grid, or another batch or width.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger

from . import registry
from .errors import ConfigError
from .registry import CrossTask, TransferSpec

#: How many offending paths a refusal lists before ``+<k> more``.
SHOWN = 5


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


def _by_path(model: Any, side: str) -> Dict[str, Any]:
    """``{path relative to the model: variable}``, in ``model.weights`` order."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-030: drop ONLY a leading "<model.name>/". A subclassed
    # model roots every path at its instance name (vision_transformer_pico vs vision_transformer_tiny,
    # res_net vs the segmenter's backbone/), which is not its architecture. Do not strip any other
    # component or suffix: an inner "_1" or "rnn_2" is a real difference (D-031 pairs those by position).
    root = f"{model.name}/"
    found: Dict[str, Any] = {}
    for variable in model.weights:
        path = variable.path[len(root):] if variable.path.startswith(root) else variable.path
        if path in found:
            raise ConfigError(f"Invalid value for 'init': the {side} holds two weights at {path}")
        found[path] = variable
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


def _pair_by_position(source: Dict[str, Any], target: Dict[str, Any], components: Sequence[str]) -> None:
    """Re-key ``source`` inside each component to the target's paths, by order, when that pairs every weight."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-031: order pairing ONLY inside the declared components
    # and ONLY when counts, shapes and dtypes all agree; otherwise leave the paths for the usual
    # refusal. Do not widen it to the whole model or rename paths: xlstm's rnn/m_lstm_cell become
    # rnn_2/m_lstm_cell_1 when a source loads beside its target, and nothing else drifts.
    for component in components:
        theirs = [path for path in source if _under(path, (component,))]
        ours = [path for path in target if _under(path, (component,))]
        pairs = list(zip(theirs, ours))
        if theirs != ours and len(theirs) == len(ours) and all(
                _shape(source[a][1]) == _shape(target[b]) and source[a][1].dtype == target[b].dtype for a, b in pairs):
            source.update({b: source.pop(a) for a, b in pairs})


def plan_transfer(source_model: Any, target_model: Any, spec: TransferSpec,
                  relation: Optional[CrossTask] = None, head_reset: bool = False,
                  name: str = "the target model") -> TransferPlan:
    """Pair every weight by path under ``spec``; assign nothing (contract in the module docstring)."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-005: one weight per Keras variable path, never
    # whole layers by name (a name loader loses a layer on one differing weight and sees no nested
    # backbone) and never names normalized (auto-named layers need a declared positional rule).
    import numpy as np

    prefix = relation.prefix if relation is not None else ""
    drop = relation.drop if relation is not None else (spec.head if head_reset else ())
    reset_heads = spec.head if (relation is not None or head_reset) else ()
    raw = _by_path(source_model, "source")
    dropped = tuple(path for path in raw if _under(path, drop))
    source = {prefix + path: (path, var) for path, var in raw.items() if not _under(path, drop)}
    target = _by_path(target_model, "target")
    _pair_by_position(source, target, spec.positional)
    source_blocks = {_block(spec.grow, path) for path in source} - {None}
    copies, copied, resampled, reset, grown, missing = [], [], [], [], [], []
    used = set()
    for path, var in target.items():
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
    return TransferPlan(name, len(target), len(raw), tuple(copies), tuple(copied),
                        tuple(resampled), tuple(reset), tuple(grown), dropped, tuple(missing), unused)


def _refuse(problems: Sequence[str], total: int, whose: str, what: str) -> None:
    if problems:
        more = f" (+{len(problems) - SHOWN} more)" if len(problems) > SHOWN else ""
        raise ConfigError(f"Invalid value for 'init': {len(problems)} of {total} weights of {whose} {what}: "
                          f"{'; '.join(problems[:SHOWN])}{more}; nothing was copied "
                          "(paths without the model's own name)")


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


def vit_pos_bicubic(value: Any, target_shape: Tuple[int, ...]) -> Any:
    """Resample a ViT position table to a larger square grid (contract in the module docstring)."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-009: torch's bicubic without antialiasing, to a
    # LARGER grid only. Do not allow a smaller grid: downsampling without antialiasing aliases, and
    # timm antialiases, so a shrink would silently differ from the reference implementation.
    import numpy as np

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


# ---------------------------------------------------------------------
# The source run, read before the claim
# ---------------------------------------------------------------------

#: The files a source run directory must hold.
SOURCE_FILES = ("task.json", "config.json", "best_model.keras")

#: How a refusal names a geometry key of the records.
GEOMETRY_NAMES = {"channels": "the input channels", "lookback": "data.lookback", "horizon": "data.horizon"}

#: How a refusal describes an input scaling, by the value :func:`_scaling` gives it.
SCALING_WORDS = {"standardize": "standardizes its input", "unit": "takes its input in [0, 1]",
                 "imagenet": "standardizes its input with the ImageNet statistics"}


@dataclass(frozen=True)
class Source:
    """A source run as read before the claim: what the run after the claim loads and records."""

    path: Path
    record: Mapping[str, Any]
    config: Mapping[str, Any]
    sha256: str
    relation: Optional[CrossTask]
    pins: Mapping[str, Any]


def _refuse_init(text: str) -> ConfigError:
    return ConfigError(f"Invalid value for 'init': {text}")


def _scaling(record: Mapping[str, Any]) -> Optional[str]:
    """The input scaling a record (or a target's identity) states: ``normalize``, else the denoisers' ``data_range``."""
    if "normalize" in record:
        return record["normalize"]
    if "data_range" in record:
        from .predictor import unit_domain_stamp

        return "unit" if record["data_range"] == unit_domain_stamp() else f"data_range {record['data_range']}"
    return None


def _source_dir(init: Any) -> Path:
    text = str(init)
    # Anchored as ``out`` is; an absolute path needs no anchor (and so no import of the anchoring helper).
    path = Path(text) if Path(text).is_absolute() else _anchor(text)
    if not path.exists():
        raise _refuse_init(f"{path} does not exist; give the directory of an earlier python -m train run")
    if not path.is_dir():
        raise _refuse_init(f"{path} is a file; give the run directory that holds it (init reads task.json, "
                           "config.json and best_model.keras)")
    task_file, config_file, model_file = (path / name for name in SOURCE_FILES)
    if not task_file.is_file():
        raise _refuse_init(f"{task_file} does not exist; a source is a python -m train run of a task that writes "
                           "a record (pretrain runs and legacy trainer runs write none)")
    if not config_file.is_file():
        raise _refuse_init(f"{config_file} does not exist; a source is a python -m train run directory")
    if not model_file.is_file():
        raise _refuse_init(f"{path} holds no best_model.keras (the run ended before its first epoch)")
    return path


def _anchor(text: str) -> Path:
    from .run_dir import _anchor as anchor

    return anchor(text)


def _declared_sources(task: str, family: str, spec: TransferSpec) -> str:
    pairs = [(task, family), *sorted(spec.sources)]
    return " or ".join(f"{name} ({kind})" for kind, name in pairs)


def _check_identity(family: str, task: str, spec: TransferSpec, source: Mapping[str, Any],
                    target: Mapping[str, Any]) -> None:
    for key in spec.fixed:
        if key in source and key in target and source[key] != target[key]:
            keeping = sorted(name for name, row in registry.MODELS.items()
                             if task in row.tasks and row.transfer is not None and key not in row.transfer.fixed)
            kept = f" (families of {task} that keep every weight across a {GEOMETRY_NAMES[key]} change: " \
                   f"{', '.join(keeping)})" if keeping else ""
            raise _refuse_init(f"{family} sizes its weights by {GEOMETRY_NAMES[key]}; the source trained at "
                               f"{source[key]}, this run asks for {target[key]}{kept}")
    theirs, ours = _scaling(source), _scaling(target)
    if theirs is not None and ours is not None and theirs != ours:
        raise _refuse_init(f"the source {SCALING_WORDS.get(theirs, f'has {theirs}')} and this run "
                           f"{SCALING_WORDS.get(ours, f'has {ours}')} (transform_args.normalize)")


def _pins(path: Path, spec: TransferSpec, used: Mapping[str, Any], asked: Mapping[str, Any]) -> Dict[str, Any]:
    pins: Dict[str, Any] = {}
    for key in spec.pinned:
        if key not in used:
            raise _refuse_init(f"{path / 'config.json'} records no used.model_args.{key}, which init keeps from the "
                               "source (the family derives it from the data, so a source without it cannot say "
                               "how it was built)")
        if key in asked and asked[key] != used[key]:
            raise ConfigError(f"Invalid value for 'model_args.{key}': the source was built with {key}={used[key]!r} "
                              "and init keeps the source's architecture")
        pins[key] = used[key]
    return pins


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(init: Any, trainer: Any, config: Any, context: Any) -> Source:
    """Check a source run directory before the claim (contract in the module docstring)."""
    from .predictor import read_record

    family, task = context.family, trainer.task
    row = registry.MODELS[family]
    if row.transfer is None:
        raise _refuse_init(row.transfer_reason or f"the family {family!r} declares no transfer, so it takes no "
                                                   "earlier run's weights")
    spec = row.transfer
    identity = trainer.init_identity(config, context)
    path = _source_dir(init)
    try:
        record = read_record(path / "task.json")
    except ConfigError as exc:
        raise _refuse_init(f"the source's record cannot be used: {exc}") from exc
    try:
        recorded = json.loads((path / "config.json").read_text())
    except ValueError as exc:
        raise _refuse_init(f"{path / 'config.json'} is not JSON ({exc})") from exc
    if not isinstance(recorded, dict):
        raise _refuse_init(f"{path / 'config.json'} is not a JSON object")
    pair = (record["task"], record["family"])
    relation = None if pair == (task, family) else spec.sources.get(pair)
    if pair != (task, family) and relation is None:
        raise _refuse_init(f"{record['family']} ({record['task']}) cannot seed {family} ({task}); {family} takes "
                           f"weights from {_declared_sources(task, family, spec)}")
    _check_identity(family, task, spec, record, identity)
    used = (recorded.get("used") or {}).get("model_args") or {}
    pins = _pins(path, spec, used, config.model_args or {}) if relation is None else {}
    return Source(path, record, recorded, _sha256(path / "best_model.keras"), relation, pins)


#: Resample rules by the name a :class:`TransferSpec` declares.
RULES: Dict[str, Callable[[Any, Tuple[int, ...]], Any]] = {"vit_pos_bicubic": vit_pos_bicubic}
