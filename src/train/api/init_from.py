"""Start a run from an earlier run's weights (``init``): the transfer engine.

The unit of a transfer is the individual weight, keyed by its Keras variable path relative to its
model (one leading component dropped, on both sides, when it is the model's root: ``<model.name>/``
wherever a path has it, or the component every weight shares; nothing else renamed). Every weight of
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
TensorFlow, Keras or numpy. After the claim, :func:`transfer` loads the source model, copies its
weights into the run's freshly built model, hashes the archive again and returns the ``init`` block
of ``task.json``. A resume never reads the source: :func:`recorded_init` reads the run's own block.

Nothing here imports TensorFlow, Keras, numpy, scipy or ``dl_techniques`` at module scope (a ``registry``
row that resamples imports this module, and ``python -m train help`` and a forecasting family refusal stay
light); numpy, scipy, the logger and the DINOv2 bicubic matrix are imported inside the functions that use them.

Interface contract
------------------
``TransferSpec`` (declared in :mod:`train.api.registry`)
    One family's declaration; every path in it is relative to the model. ``head``: path prefixes
    of the final layer, reset when the head is not reused; ``resample``: a regex, which must match a
    whole target path (``re.fullmatch``), to a rule name (a key of :data:`RULES`); a path two keys
    match is refused while planning; ``grow``: a regex matched at the start of a path, naming a
    depth-indexed block a deeper variant may add; ``pinned``: factory arguments the target takes
    from the source (a cross-task source pins those its record holds); ``fixed``: geometry keys the
    family cannot change (``channels``, ``lookback``, ``horizon``, ``input_size``, ``patch_size``, for the language
    model ``vocab_size`` and ``tokenizer``, a tokenizer's digest, and for the BEiT tokenizer ``num_embeddings``, its codebook size; each
    needs a name in :data:`GEOMETRY_NAMES`); ``policy``: ``{key: reason}`` for the keys of ``fixed`` the family holds by its task's contract
    although every weight would copy (the tokenizer's image size): the refusal before the claim then says "holds <key> to the source's by
    policy, not because a weight depends on it" and gives the reason, where it would otherwise say the family "sizes its weights by" the key;
    ``sources``: ``(task, family)`` of another task to the
    :class:`~train.api.registry.CrossTask` that maps its paths; ``positional``: regexes, each
    matching whole top-level path components, inside which weights whose paths differ pair by
    order, when both sides hold the same count and every pair has equal shape and dtype.

``plan_transfer(source_model, target_model, spec, relation=None, head_reset=False, name=...)``
    :param source_model: the source model (anything with ``name`` and ``weights``).
    :param target_model: the target model.
    :param spec: the target family's :class:`TransferSpec`.
    :param relation: ``None`` for a source of the same family and task; for a cross-task source,
        the :class:`CrossTask` entry of ``spec.sources``, whose head is always reset. A source path is read as the source
        holds it (``drop``), then renamed (``rename``, ``re.sub`` pair by pair), then prefixed (``prefix``).
    :param head_reset: same task only: reset the target's head and drop the source's.
    :param name: what the refusal calls the target (``family/variant``).
    :return: a :class:`TransferPlan`. Nothing is assigned; values are read only to resample.
    :raises ConfigError: a variable path held twice on one side, two source paths a ``rename`` sends to one path (both
        named), a target path two ``resample`` keys match, or a resample rule refusing the shapes (a smaller or
        non-square grid; any change of the Energy Transformer position table).

``source_view(model, relation)``
    :param model: the loaded source model.
    :param relation: the :class:`CrossTask` of the pair, or ``None`` for a same-task source.
    :return: ``model`` itself, or ``getattr(model, relation.view)`` when the relation names a view (a DINO training model's
        ``student``: the whole model holds a student and a teacher at every path, which :func:`plan_transfer` refuses).
        :func:`transfer` applies it before planning; a test that plans by hand applies it too.
    :raises ConfigError: the model has no attribute of that name.

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
        declares no transfer; the trainer's ``init_identity`` refuses (the default does); the directory is
        missing or is a file; ``task.json``, ``config.json`` or ``best_model.keras``
        is missing; the record fails the predictor's record check or ``config.json`` is no JSON object; the
        source's task and family are neither this run's nor a declared cross-task pair; ``fixed`` keys of
        this family differ (every one named) or the input scaling does; a pinned argument of a same-task source is
        not recorded in its ``config.json`` ``used.model_args`` (a cross-task source pins what its record holds); the
        target's grid (its side over the pinned patch) differs from the source's where its position rule makes no such
        change: any change of ``et_pos_refused``'s; for ``beit_relpos_cubic`` a grid below 3x3, a smaller one or one
        past :func:`beit_reach` (both grids and the tokens named). (A request ``model_args`` value other than a pin
        skips that check and is refused by :func:`train.api.recipes.merge_model_args`, also before the claim.) A
        cross-task pin of ``patch_size`` other than the recipe's patch gives the :class:`Source` a ``notice`` with the
        tokens.

``transfer(source, model, task, family, variant, record)``
    :param source: the :class:`Source` :func:`read_source` returned before the claim.
    :param model: this run's freshly built model (its pins applied).
    :param task: this run's task; ``record``: the ``task.json`` record this run will write, whose
        :data:`HEAD_KEYS` say whether a same-task head means what the source's means.
    :return: the ``init`` block (keys in :data:`train.api.predictor.INIT_KEYS`). The source model is
        released before it returns.
    :raises ConfigError: the source does not load (the predictor's checks), the engine refuses (nothing
        is copied), or ``best_model.keras`` hashes to another sha256 than before the claim (both named;
        the copied weights must not be trained).

``recorded_init(run_dir)``
    :return: the ``init`` block of ``<run_dir>/task.json``, or ``None`` when the run has no ``task.json``
        (a resume of such a run is refused by the run directory's own checks: it has no backup).
    :raises ConfigError: the record fails its check, or holds no ``init`` block.

``split_block(relation, reason, keys=None, unit=None, counts=(None, None))``
    The ``split`` of the ``init`` block, which each task's ``init_overlap`` hook returns after the claim: how the
    source's training data meets this run's selection and test data (``same_split``, ``different_data``,
    ``overlap`` with both counts, or ``unknown`` with the reason). :func:`same_split`, :func:`counted_split` and
    :func:`seeded_rows_split` (classification and segmentation) build it; :func:`bias_of` gives the summary's
    ``selection_biased``, ``test_biased`` and ``overlap_unknown`` and :func:`bias_words` the success line's words.
    Nothing here refuses: an overlap is counted and marked.

``vit_pos_bicubic(value, target_shape)``
    :param value: a ``(1, 1 + m * m, d)`` table, class row first, then the row-major grid.
    :param target_shape: ``(1, 1 + n * n, d)`` with ``n > m``.
    :return: the class row unchanged and the grid resampled per axis with torch's bicubic kernel
        (``align_corners=False``, no antialiasing), in ``value``'s dtype.
    :raises ConfigError: a smaller grid, a non-square grid, or another batch or width.

``beit_relpos_cubic(value, target_shape)``
    :param value: a BEiT ``relative_position_bias_table`` ``(k * k + 3, heads)``: the row-major grid of the
        ``k = 2 W - 1`` displacements of ``W x W`` patches, then 3 extra rows (class to token, token to class,
        class to class).
    :param target_shape: ``(n * n + 3, heads)`` with ``n > k`` (both odd, ``k`` at least 5: a cubic needs 4 points).
    :return: unilm's resize (``beit/run_class_finetuning.py``): per head a cubic spline (``RectBivariateSpline``,
        ``s=0``) through the source grid set at unilm's geometric positions, evaluated at the target's integer
        displacements, then the 3 extra rows unchanged, in ``value``'s dtype.
    :raises ConfigError: another rank or head count, a row count that is no such grid, a grid that is not larger, or
        one past :func:`beit_reach` of the source's (where unilm's grid stops spanning the target, see BEYOND_REACH).

``beit_reach(window)``
    :return: the largest side of a grid of patches ``beit_relpos_cubic`` grows a ``window x window`` grid to
        (``ceil(2 (1.5^(window - 1) - 1))``: 3 for 3, 5 for 4, 33 for 8).

``et_pos_refused(value, target_shape)``
    :raises ConfigError: always, naming both shapes and grids: no reference rule resamples the Energy Transformer
        position table, so ``init`` cannot change its grid.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

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
    # DECISION plan-2026-11-18T072904-0fd54b91/D-030 (amended by D-045): drop ONE leading component, only when it is
    # the model's root: "<model.name>/" wherever a path has it, or the component EVERY weight shares (accunet builds
    # under acc_u_net/, not its name ACC_UNet). A loaded archive can lose the root on some weights and keep it on
    # others (the depth backbone loses depth_anything/, its head keeps it), so do NOT drop only a shared component
    # and do NOT strip any other component or suffix: an inner "_1" or "rnn_2" is a real difference (D-031).
    paths = [variable.path for variable in model.weights]
    tops = {path.split("/", 1)[0] for path in paths}
    roots = {model.name} | (tops if len(tops) == 1 and all("/" in path for path in paths) else set())
    found: Dict[str, Any] = {}
    for variable, path in zip(model.weights, paths):
        top, _, rest = path.partition("/")
        path = rest if rest and top in roots else path
        if path in found:
            raise ConfigError(f"Invalid value for 'init': the {side} holds two weights at {path}")
        found[path] = variable
    return found


def _under(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)


def _block(pattern: Optional[str], path: str) -> Optional[str]:
    match = re.match(pattern, path) if pattern else None
    return match.group(0) if match else None


def _rule_of(spec: TransferSpec, path: str) -> Optional[str]:
    """The rule of the one ``resample`` key that matches the whole ``path``; ``None`` when no key does."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-005: a key is a regex the WHOLE path must match, so one BEiT key names
    # the table of every encoder layer at any depth. Do NOT go back to the exact-path lookup (a deeper variant or a
    # num_layers override would be refused after the claim, the defect 4g's review found for xlstm), do NOT use
    # re.match or re.search (a key would also take longer paths), and do NOT let key order pick between two matches.
    keys = [key for key in spec.resample if re.fullmatch(key, path)]
    if len(keys) > 1:
        raise ConfigError(f"Invalid value for 'init': {path} matches {len(keys)} resample keys of the family "
                          f"({', '.join(keys)}); a weight takes one rule")
    return spec.resample[keys[0]] if keys else None


def _shape(variable: Any) -> Tuple[int, ...]:
    return tuple(int(n) for n in variable.shape)


def _side(variable: Any) -> str:
    return f"{_shape(variable)} {variable.dtype}" if variable is not None else "none"


def _pair_by_position(source: Dict[str, Any], target: Dict[str, Any], patterns: Sequence[str]) -> None:
    """Re-key ``source`` inside each component a pattern matches to the target's paths, by order, if all pair."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-031: order pairing ONLY inside the declared components
    # and ONLY when counts, shapes and dtypes all agree; otherwise leave the paths for the usual
    # refusal. Do not widen it to the whole model or rename paths: xlstm's rnn/m_lstm_cell become
    # rnn_2/m_lstm_cell_1 when a source loads beside its target, and nothing else drifts.
    tops = {path.partition("/")[0] for path in target}
    for component in sorted(top for top in tops if any(re.fullmatch(pattern, top) for pattern in patterns)):
        theirs = [path for path in source if _under(path, (component,))]
        ours = [path for path in target if _under(path, (component,))]
        pairs = list(zip(theirs, ours))
        if theirs != ours and len(theirs) == len(ours) and all(
                _shape(source[a][1]) == _shape(target[b]) and source[a][1].dtype == target[b].dtype for a, b in pairs):
            source.update({b: source.pop(a) for a, b in pairs})


def source_view(model: Any, relation: Optional[CrossTask]) -> Any:
    """Return the model whose weights are the source: ``model``, or its ``relation.view`` attribute (contract in the module docstring)."""
    if relation is None or not relation.view:
        return model
    view = getattr(model, relation.view, None)
    if view is None:
        raise _refuse_init(f"the loaded source model has no attribute {relation.view!r}, which the pair names as the model whose "
                           "weights are the source")
    return view


def _renamed(path: str, relation: Optional[CrossTask]) -> str:
    """``path`` through every ``(regex, replacement)`` of ``relation.rename``, in order."""
    for pattern, replacement in (relation.rename if relation is not None else ()):
        path = re.sub(pattern, replacement, path)
    return path


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
    # DECISION plan-2026-11-26T224507-e3381378/D-069
    # `drop` reads the paths AS THE SOURCE HOLDS THEM, and `rename` acts only on the paths that survive it. Do NOT rename first: a rename
    # that moves a dropped weight out of its prefix would turn a declared drop into an unused weight (refused), and a dropped head would
    # then need its drop written in the target's words. Do NOT let two paths meet silently: the dict below would keep the last one.
    source: Dict[str, Tuple[str, Any]] = {}
    for path, var in raw.items():
        if _under(path, drop):
            continue
        key = prefix + _renamed(path, relation)
        if key in source:
            raise _refuse_init(f"the rename of the pair sends two source weights, {source[key][0]} and {path}, to the same path {key}; "
                               "a rename must keep every weight of the source apart")
        source[key] = (path, var)
    target = _by_path(target_model, "target")
    _pair_by_position(source, target, spec.positional)
    source_blocks = {_block(spec.grow, path) for path in source} - {None}
    copies, copied, resampled, reset, grown, missing = [], [], [], [], [], []
    used = set()
    for path, var in target.items():
        src = source.get(path, (None, None))[1]
        rule = _rule_of(spec, path)
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
                          "(paths without the model's root)")


def check(plan: TransferPlan) -> None:
    """Complete-or-raise in both directions (contract in the module docstring)."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-006: no skip_mismatch and no override; a partial
    # load is refused in BOTH directions, since a loader that returns normally on a partial match
    # has left a trunk random without a word. Do not soften either half into a warning.
    _refuse(plan.missing, plan.n_target, plan.name, "have no source weight of their shape and no declared rule")
    _refuse(plan.unused, plan.n_source, "the source run", "are neither used nor dropped by a declared rule")


def apply(plan: TransferPlan) -> None:
    """Check, then assign every copy and resample; a refusal assigns nothing."""
    from dl_techniques.utils.logger import logger

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


#: The rows a BEiT relative position bias table holds after its grid of displacements (class to token, token to
#: class, class to class); unilm's rule copies them unchanged.
BEIT_EXTRA_ROWS = 3
#: The upper bound of unilm's search for the ratio of its geometric grid (``run_class_finetuning.py`` L376).
BEIT_RATIO_BOUND = 1.5
#: Why :func:`beit_relpos_cubic` grows no grid past :func:`beit_reach` (both refusals say it).
BEYOND_REACH = ("past it the ratio of the rule's geometric grid stops at its upper bound 1.5 and the source grid no "
                "longer spans the target, so every outer target displacement takes the value at the source's edge; "
                "unilm's published fine-tunes grow 14x14 patches to 24x24, far inside the reach of 14 (388), and no "
                "reference measures a fine-tune past it")
#: Why the BEiT rule grows a grid only and the ET grid never changes: each rule says it, and read_source (D-055).
BEIT_LARGER_ONLY = ("only a larger grid is resampled (unilm's rule grows a grid; for a smaller one its search for the "
                    "ratio stops at its lower bound)")
ET_NO_RULE = ("no reference rule resamples it (the Energy Transformer paper describes no fine-tuning at another "
              "resolution); train this run at the source's grid (its image_size and patch_size)")


def beit_reach(window: int) -> int:
    """Return the largest side of a grid of patches :func:`beit_relpos_cubic` grows a ``window x window`` grid to.

    Interface contract: ``window`` is an int of at least 3 (the rule's floor). unilm's search sets the source's
    ``window - 1`` outer displacements on a geometric grid of ratio ``q <= 1.5``, the last at ``(1 - q^(window - 1)) /
    (1 - q)``; a target whose outer displacement (its side minus 1) is not below that at ``q = 1.5`` saturates the
    search. 3 reaches 3 (no growth), 4 reaches 5, 8 reaches 33. Pure; never raises for such a ``window``.
    """
    import math

    return math.ceil((1.0 - BEIT_RATIO_BOUND ** (window - 1)) / (1.0 - BEIT_RATIO_BOUND))


def _reach_words(window: int) -> str:
    reach = beit_reach(window)
    return (f"unilm's rule grows the relative position bias table of a grid of {window}x{window} patches to at most "
            f"{reach}x{reach} ({BEYOND_REACH})")


def _beit_side(shape: Tuple[int, ...]) -> int:
    """The side ``2 W - 1`` of the displacement grid of a BEiT table of ``W x W`` patches (``(2 W - 1)^2 + 3`` rows)."""
    side = int(round(max(shape[0] - BEIT_EXTRA_ROWS, 0) ** 0.5))
    if side * side + BEIT_EXTRA_ROWS != shape[0] or side % 2 == 0 or side < 5:
        raise ConfigError(f"Invalid value for 'init': BEiT relative_position_bias_table {tuple(shape)}: {shape[0]} "
                          "rows are not the (2W - 1)^2 + 3 of a square grid of W x W patches with W at least 3; only "
                          "such a table is resampled")
    return side


def beit_relpos_cubic(value: Any, target_shape: Tuple[int, ...]) -> Any:
    """Resample a BEiT relative position bias table to a larger square grid (contract in the module docstring)."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-004: unilm's own rule (beit/run_class_finetuning.py L351-411): per head
    # a cubic spline through the source displacements set on a GEOMETRIC grid, evaluated at the integer target ones,
    # the 3 extra rows copied. Do NOT use timm's default (linear on the same points, 0.29 to 0.45 off) or a bicubic on
    # the index grid (vit_pos_bicubic's kernel, 0.73 to 1.02 off): neither is the rule BEiT was fine-tuned with.
    # RectBivariateSpline(s=0) stands in for interp2d (gone in scipy 1.14); test_init_from_engine.py holds it to unilm's
    # code run verbatim (tests/test_train/test_api/data/beit_relpos_reference.npz) at 1e-6.
    import numpy as np
    from scipy.interpolate import RectBivariateSpline

    if len(value.shape) != 2 or len(target_shape) != 2 or value.shape[1] != target_shape[1]:
        raise ConfigError(f"Invalid value for 'init': BEiT relative_position_bias_table {tuple(value.shape)} cannot "
                          f"become {tuple(target_shape)}; only its number of rows (the grid of patches) may change")
    src, dst = _beit_side(value.shape), _beit_side(target_shape)
    if dst <= src:
        raise ConfigError(f"Invalid value for 'init': BEiT relative_position_bias_table would resample a grid of "
                          f"{(src + 1) // 2}x{(src + 1) // 2} patches to {(dst + 1) // 2}x{(dst + 1) // 2}; "
                          f"{BEIT_LARGER_ONLY}")
    # DECISION plan-2026-11-19T081214-5c6cc105/D-046 (amended by D-051)
    # Past beit_reach the source grid stops spanning the target and FITPACK clamps, so the margin repeats the source's
    # edge (scratch_step9_2/clamp_probe.log). Do NOT justify it by overshoot or extrapolation (the cubic rings as much
    # inside): no reference measures a growth past it. Refused here, and before the claim by read_source.
    if (dst + 1) // 2 > beit_reach((src + 1) // 2):
        raise ConfigError(f"Invalid value for 'init': BEiT relative_position_bias_table would resample a grid of "
                          f"{(src + 1) // 2}x{(src + 1) // 2} patches to {(dst + 1) // 2}x{(dst + 1) // 2}; "
                          f"{_reach_words((src + 1) // 2)}")
    left, right = 1.01, BEIT_RATIO_BOUND  # unilm's search for the ratio q whose src // 2 geometric steps reach dst // 2
    while right - left > 1e-6:
        q = (left + right) / 2.0
        if (1.0 - q ** (src // 2)) / (1.0 - q) > dst // 2:
            right = q
        else:
            left = q
    steps, step = [], 1.0
    for i in range(src // 2):
        steps.append(step)
        step += q ** (i + 1)
    x = np.array([-s for s in reversed(steps)] + [0.0] + steps)
    t = dst // 2.0
    targets = np.arange(-t, t + 0.1, 1.0)
    grid = np.asarray(value[:-BEIT_EXTRA_ROWS], dtype=np.float32)
    heads = [RectBivariateSpline(x, x, grid[:, h].reshape(src, src), kx=3, ky=3, s=0)(targets, targets).reshape(-1)
             for h in range(value.shape[1])]
    return np.concatenate([np.stack(heads, axis=-1).astype(value.dtype), value[-BEIT_EXTRA_ROWS:]], axis=0)


def _grid_of(shape: Tuple[int, ...]) -> str:
    """``(1, 64, 192) (8x8 patches)``: a position table's shape and, when its rows are a square, its grid."""
    rows = shape[1] if len(shape) == 3 else 0
    side = int(round(rows ** 0.5))
    return f"{tuple(shape)} ({side}x{side} patches)" if rows and side * side == rows else f"{tuple(shape)}"


def et_pos_refused(value: Any, target_shape: Tuple[int, ...]) -> Any:
    """Refuse a change of an Energy Transformer position table (contract in the module docstring)."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-006: a rule that never resamples. No reference resizes this table (the
    # paper describes no fine-tuning at another resolution), so do NOT give it vit_pos_bicubic's kernel or a class-free
    # bicubic: a convention is not a reference (U3). It is declared so the refusal names both grids and the reason;
    # it runs while planning, so nothing is assigned.
    raise ConfigError(f"Invalid value for 'init': the Energy Transformer position table {_grid_of(value.shape)} cannot "
                      f"become {_grid_of(target_shape)}: {ET_NO_RULE}")


# ---------------------------------------------------------------------
# The source run, read before the claim
# ---------------------------------------------------------------------

#: The files a source run directory must hold.
SOURCE_FILES = ("task.json", "config.json", "best_model.keras")

#: How a refusal names a geometry key of the records.
GEOMETRY_NAMES = {"channels": "the input channels", "lookback": "data.lookback", "horizon": "data.horizon",
                  "input_size": "transform_args.image_size", "patch_size": "data.patch_size",
                  "vocab_size": "the model's vocabulary size", "tokenizer": "the tokenizer (data.tokenizer, as its digest)",
                  "num_embeddings": "the codebook size (model_args.num_embeddings)"}

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
    notice: Optional[str] = None  # the moved-patch line; the trainer's notices replay it into run.log (D-053)


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
        raise _refuse_init(f"{task_file} does not exist; a source is a python -m train run, which writes it before its "
                           "first epoch (legacy trainer runs and runs made before their task wrote one have none)")
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
    moved = [key for key in spec.fixed if key in source and key in target and source[key] != target[key]]
    if moved:  # every differing key at once: a dataset change moves a pretrain run's size and patch together
        # DECISION plan-2026-11-25T124419-fa94f59c/D-066
        # A key the row holds by POLICY (its `policy` reason) is refused in its own sentence: the weights would copy, so "sizes its weights by"
        # would be false there (the tokenizer's image size: 47 of 47 weights copy into a 64 px model) and a user who read only the error would
        # conclude the transfer is impossible. Do NOT fold the two kinds of key into one sentence, and do NOT drop `policy` for a plainer
        # message: the sentence of a key the weights DO depend on stays as it was, byte for byte.
        shaped = [key for key in moved if key not in spec.policy]
        sentences = []
        if shaped:
            names = " and ".join(GEOMETRY_NAMES[key] for key in shaped)
            keeping = sorted(name for name, row in registry.MODELS.items()
                             if task in row.tasks and row.transfer is not None and not set(shaped) & set(row.transfer.fixed))
            kept = f" (families of {task} that keep every weight across a {names} change: {', '.join(keeping)})" \
                if keeping else ""
            sentences.append(f"{family} sizes its weights by {names}; the source trained at "
                             f"{' and '.join(str(source[key]) for key in shaped)}, this run asks for "
                             f"{' and '.join(str(target[key]) for key in shaped)}{kept}")
        sentences += [f"{task} holds {GEOMETRY_NAMES[key]} to the source's by policy, not because a weight depends on it: the source trained "
                      f"at {source[key]}, this run asks for {target[key]}; {spec.policy[key]}" for key in moved if key in spec.policy]
        raise _refuse_init("; ".join(sentences))
    theirs, ours = _scaling(source), _scaling(target)
    if theirs is not None and ours is not None and theirs != ours:
        raise _refuse_init(f"the source {SCALING_WORDS.get(theirs, f'has {theirs}')} and this run "
                           f"{SCALING_WORDS.get(ours, f'has {ours}')} (transform_args.normalize)")


def _pins(path: Path, spec: TransferSpec, used: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [key for key in spec.pinned if key not in used]
    if missing:
        raise _refuse_init(f"{path / 'config.json'} records no used.model_args.{missing[0]}, which init keeps from the "
                           "source (the family derives it from the data, so a source without it cannot say "
                           "how it was built)")
    return {key: used[key] for key in spec.pinned}


def file_sha256(path: Path) -> str:
    """Return the 64 lowercase hex digits of the sha256 of the file ``path``'s bytes, read in 1 MiB blocks.

    Interface contract: ``path`` is a readable file (``OSError`` otherwise). The ``init`` block's ``sha256`` of a source
    archive and a pretrain record's ``tokenizer`` (``tasks/pretrain.py``) are this function of their file.
    """
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
        raise _refuse_init(f"the family {family!r} declares no transfer, so it takes no earlier run's weights")
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
    # DECISION plan-2026-11-19T081214-5c6cc105/D-036
    # A cross-task source pins the keys of `spec.pinned` its RECORD holds (a pretrain record's patch_size), under the
    # same precedence as a same-task pin (recipes.merge_model_args). Do NOT read its used.model_args (another family's
    # factory keywords) and do NOT pin nothing: a 64 px beit would take patch 16 and be refused after the claim.
    pins = _pins(path, spec, used) if relation is None else {key: record[key] for key in spec.pinned if key in record}
    _refuse_grid_change(spec, record, pins, context)
    notice = _moved_patch(pins, context) if relation is not None else None
    return Source(path, record, recorded, file_sha256(path / "best_model.keras"), relation, pins, notice)


def _refuse_grid_change(spec: TransferSpec, record: Mapping[str, Any], pins: Mapping[str, Any], context: Any) -> None:
    """Refuse a target grid its position rule cannot make, naming both grids and the tokens (D-046, D-055)."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-055
    # Every grid change the position rules refuse while planning, after the claim, is refused HERE, before it, from
    # the source record's size, the pinned patch and the run's side. Do NOT leave one to the rules: a claimed directory
    # stays behind. A typed patch other than the pin skips it, so merge_model_args's pin refusal is what it gets.
    rules = set(spec.resample.values()) & {"beit_relpos_cubic", "et_pos_refused"}
    if not rules or "patch_size" not in pins:
        return  # no position rule: the context is not read (the source reader's own test passes a bare one)
    asked = context.model_args or {}
    if asked.get("patch_size", pins["patch_size"]) != pins["patch_size"]:
        return  # a typed patch other than the pin: merge_model_args refuses it, before the claim too
    if "et_pos_refused" not in rules and asked.get("use_relative_position_bias") is False:
        return  # the target has no relative position bias table
    patch, side, size = pins["patch_size"], context.image_size, record.get("input_size") or record.get("input_shape")
    if not isinstance(side, int) or not size or side % patch or size[0] % patch or side == size[0]:
        return  # an untiled side is refused by name elsewhere; an equal grid is copied
    theirs, ours = size[0] // patch, side // patch
    if "et_pos_refused" in rules:
        reason = f"the Energy Transformer position table cannot change its grid: {ET_NO_RULE}"
    elif min(theirs, ours) < 3:
        reason = "BEiT's relative position bias table: the rule resamples no grid below 3x3 patches"
    elif ours < theirs:
        reason = f"BEiT's relative position bias table: {BEIT_LARGER_ONLY}"
    elif ours > beit_reach(theirs):
        reason = f"{_reach_words(theirs)}; ask for transform_args.image_size of at most {beit_reach(theirs) * patch} px"
    else:
        return
    raise _refuse_init(f"the source's grid of {theirs}x{theirs} patches ({size[0]} px at patch_size {patch}, which "
                       f"init keeps) would become {ours}x{ours} at {side} px, {ours * ours} tokens; {reason}")


def _moved_patch(pins: Mapping[str, Any], context: Any) -> Optional[str]:
    """The line of a cross-task patch pin that is not the recipe's patch, else None (D-046; a notice since D-053)."""
    from .recipes import vit_patch_size  # the patch rule of the rows a pretrain record pins (beit, energy_transformer)

    patch, side, computed = pins.get("patch_size"), context.image_size, vit_patch_size(context)
    if patch is None or not isinstance(side, int) or side % patch or patch == computed:
        return None
    grid = side // patch
    return (f"init: the source's record pins patch_size {patch} (the recipe gives {computed} at {side} px), so this "
            f"run cuts its {side} px input into {grid}x{grid} = {grid * grid} tokens")


# ---------------------------------------------------------------------
# After the claim: the transfer and its record; on a resume, the run's own record
# ---------------------------------------------------------------------

#: The record keys that say what a task's head means: a same-task head is copied only when the two records agree on
#: every one; across tasks it is always fresh. Depth has one output whose meaning never changes.
HEAD_KEYS: Dict[str, Tuple[str, ...]] = {
    "image_classification": ("class_names",),
    "image_segmentation": ("class_names",),
    "image_denoising": ("channels",),
    "time_series_forecasting": ("horizon", "head", "quantile_levels", "channels"),
    "depth_estimation": (),
    # DECISION plan-2026-11-19T081214-5c6cc105/D-008 (amended by D-048): a pretrain decoder answers in its tokenizer's
    # codes (ET's, with no tokenizer, in pixels) of an input standardized with the record's mean and std, so the
    # tokenizer file's sha256 and the two statistics are the head's identity. Do NOT key it by the path: another file at
    # the same path would keep a head whose outputs mean other codes; do NOT drop the statistics: ET's decoder emits
    # pixels in them, and a run at another dataset's statistics passes the geometry check.
    "pretrain": ("tokenizer", "mean", "std"),
    # DECISION plan-2026-11-24T092348-7806a290/D-041
    # `head_differs` indexes this table by task, so the language task needs its row, but the key never fires: the tokenizer is a
    # `fixed` key of the family (a source of another tokenizer is refused before the claim), so a same-task head always agrees.
    "language_model_pretraining": ("tokenizer",),
    # DECISION plan-2026-11-24T092348-7806a290/D-062
    # A text classifier's head means its CLASS NAMES (the same two outputs under other names, or in another order, are another head), so a
    # same-task source whose record names others has its head reset. The tokenizer is no key here: it is `fixed` by both families, so a
    # source of another one is refused before the claim. Do NOT add "tokenizer" "for symmetry" (it never fires) and do NOT drop the row:
    # `head_differs` indexes this table for every same-task transfer, which a classifier run now is (a source of its own family).
    "text_classification": ("class_names",),
    # DECISION plan-2026-11-25T124419-fa94f59c/D-050
    # A tokenizer has NO head: both convolution stacks and the codebook are what `init` moves, so its row is empty. It exists because
    # `head_differs` indexes this table by task for every same-task transfer (a missing row is a KeyError after the claim). Do NOT add
    # "num_embeddings" or the input statistics "for symmetry with pretrain": another codebook size or geometry is refused BEFORE the claim
    # (`fixed`), so a same-task head can never differ, and the statistics change no shape.
    "beit_tokenizer": (),
}

#: The source record's keys the ``init`` block keeps as its geometry (those its task's record has). The language record's
#: ``tokenizer`` is not one: the pretrain record has a key of that name too (a file's sha256), and its block must not change (D-041).
#: The tokenizer record's ``num_embeddings`` (its codebook size) is, and no other record has a key of that name (D-050).
GEOMETRY_KEYS = ("input_shape", "input_size", "channels", "lookback", "horizon", "patch_size", "seq_len", "vocab_size", "num_embeddings")


def head_differs(task: str, source_record: Mapping[str, Any], target_record: Mapping[str, Any]) -> bool:
    """Whether a same-task source's head means something else than this run's (:data:`HEAD_KEYS`)."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-008
    # Compare the record VALUES (class names as ordered lists), never the head's shape or the class count: 10 other
    # classes, or the same 10 in another order, fit a 10-way head and would be copied as if they were the same, a
    # head that silently answers with the wrong names.
    return any(source_record.get(key) != target_record.get(key) for key in HEAD_KEYS[task])


def block_of(source: Source, plan: TransferPlan) -> Dict[str, Any]:
    """The ``init`` block of ``task.json``: where the weights came from and what the transfer did with them."""
    record = source.record
    return {
        "source": str(source.path),
        "file": SOURCE_FILES[2],
        "sha256": source.sha256,
        "task": record["task"],
        "family": record["family"],
        "variant": record["variant"],
        "geometry": {key: record[key] for key in GEOMETRY_KEYS if key in record},
        "relation": "same_task" if source.relation is None else "cross_task",
        "pinned": dict(source.pins),
        "transfer": {
            "copied": len(plan.copied),
            "resampled": [{"path": path, "rule": rule, "from": list(before), "to": list(after)}
                          for path, rule, before, after in plan.resampled],
            "reset": list(plan.reset),
            "grown": list(plan.grown),
            "dropped": list(plan.dropped),
        },
    }


def transfer(source: Source, model: Any, task: str, family: str, variant: str,
             record: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy the source run's weights into ``model`` after the claim (contract in the module docstring)."""
    from .predictor import load_model

    loaded = load_model(source.path).model
    reset = source.relation is None and head_differs(task, source.record, record)
    plan = plan_transfer(source_view(loaded, source.relation), model, registry.MODELS[family].transfer, source.relation, reset,
                         name=f"{family}/{variant}")
    apply(plan)
    block = block_of(source, plan)
    del loaded, plan  # the source model is released here, before the optimizer is built
    archive = source.path / SOURCE_FILES[2]
    again = file_sha256(archive)
    if again != source.sha256:
        raise _refuse_init(f"{archive} changed while this run read it: its sha256 was {source.sha256} before the run "
                           f"directory was claimed and is {again} now; the copied weights may not be the ones checked, "
                           "so the run stops before it trains")
    return block


def recorded_init(run_dir: Path) -> Optional[Mapping[str, Any]]:
    """The run's own ``task.json`` ``init`` block, read before a resume's claim (contract in the module docstring)."""
    from .predictor import read_record

    task_file = Path(run_dir) / SOURCE_FILES[0]
    if not task_file.is_file():
        return None
    record = read_record(task_file)
    if "init" not in record:
        raise _refuse_init(f"{task_file} holds no init block, so the run did not start from an earlier run's weights; "
                           "resume it without init")
    return record["init"]


#: Resample rules by the name a :class:`TransferSpec` declares.
RULES: Dict[str, Callable[[Any, Tuple[int, ...]], Any]] = {
    "vit_pos_bicubic": vit_pos_bicubic, "beit_relpos_cubic": beit_relpos_cubic, "et_pos_refused": et_pos_refused}


# ---------------------------------------------------------------------
# The split identity: the source's training data against this run's selection and test data
# ---------------------------------------------------------------------

#: The reason of the default ``init_overlap`` hook.
NOT_REBUILT = "this task does not rebuild a source's rows"

#: Built-in datasets of other names that share images: Imagenette's are ImageNet's.
SHARED_DATASETS = (frozenset(("imagenette", "imagenet")),)

#: The ``data`` keys that say WHERE file data comes from (classification and segmentation, folder and csv).
LOCATION_KEYS = ("train_dir", "test_dir", "train_images", "train_masks", "test_images", "test_masks", "train_csv",
                 "test_csv")


def split_block(relation: str, reason: str, keys: Optional[Mapping[str, Any]] = None, unit: Optional[str] = None,
                counts: Tuple[Optional[int], Optional[int]] = (None, None)) -> Dict[str, Any]:
    """The ``split`` of the ``init`` block (keys in :data:`train.api.predictor.INIT_SPLIT_KEYS`).

    :param relation: ``same_split``, ``different_data``, ``overlap`` (counted) or ``unknown``.
    :param reason: one sentence; for ``overlap`` it names the keys that would give 0.
    :param keys: the source's values of the split keys (this run's equal to them is ``same_split``).
    :param unit: what is counted (``rows``, ``files``, ``windows``, ``scenes``).
    :param counts: the source's training data found in this run's selection and test data; ``None`` when ``unknown``.
    """
    return {"relation": relation, "unit": unit, "n_select_overlap": counts[0], "n_test_overlap": counts[1],
            "reason": reason, "keys": dict(keys or {})}


def listed(keys: Mapping[str, Any]) -> str:
    return ", ".join(f"{name}={json.dumps(value)}" for name, value in keys.items())


def same_split(unit: str, keys: Mapping[str, Any]) -> Dict[str, Any]:
    """This run's split keys are the source's: the source trained on none of this run's held-out data."""
    return split_block("same_split", f"{listed(keys)} equal the source's", keys, unit, (0, 0))


def counted_split(unit: str, keys: Mapping[str, Any], in_select: int, n_select: int, in_test: int,
                  n_test: int) -> Dict[str, Any]:
    """The source's training data found in this run's selection and test data, counted."""
    return split_block("overlap", f"the source trained on {in_select} of this run's {n_select} selection {unit} and "
                                  f"{in_test} of its {n_test} test {unit}; {listed(keys)} as the source's give 0",
                       keys, unit, (int(in_select), int(in_test)))


def bias_of(block: Mapping[str, Any]) -> Dict[str, bool]:
    """The summary's flags of an ``init`` block's ``split``: an overlap is counted and marked, never refused."""
    split = block["split"]
    return {"selection_biased": bool(split["n_select_overlap"]), "test_biased": bool(split["n_test_overlap"]),
            "overlap_unknown": split["relation"] == "unknown"}


def bias_words(block: Optional[Mapping[str, Any]]) -> str:
    """What the success line adds for an ``init`` run: ``" (selection-biased, test-biased)"``, or nothing."""
    if block is None:
        return ""
    flags = bias_of(block)
    words = [word for key, word in (("selection_biased", "selection-biased"), ("test_biased", "test-biased"),
                                    ("overlap_unknown", "split overlap unknown")) if flags[key]]
    return f" ({', '.join(words)})" if words else ""


def _fingerprint(path: Path) -> Optional[str]:
    """The ``data_fingerprint`` of a source's fit sidecar, ``None`` when it has none (the sidecar is optional)."""
    try:
        return json.loads((path / "checkpoints" / "fit_state.json").read_text()).get("data_fingerprint")
    except (OSError, ValueError, AttributeError):
        return None


def seeded_rows_split(source: Source, config: Any, data: Any, table: Optional[int] = None) -> Dict[str, Any]:
    """The ``split`` of a classification or segmentation run: the seeded train cut of both runs, rebuilt and compared.

    Both tasks cut ``seeded_split_indices(n, ..., seed, validation_split, max_samples)`` over one dataset or one
    listing, and their test rows are another split the source never trained on (``n_test_overlap`` 0). ``n`` is this
    run's pool when it is not capped below it, else ``table`` (a built-in dataset's train size), else unknown.
    """
    import numpy as np

    from dl_techniques.datasets.vision.splits import seeded_split_indices
    from train.common.classification_data import DEFAULT_VALIDATION_SPLIT

    theirs, ours = source.config.get("data") or {}, config.data
    if theirs.get("format") == ours.format == "builtin" and theirs.get("dataset") != ours.dataset:
        if frozenset((theirs.get("dataset"), ours.dataset)) in SHARED_DATASETS:
            return split_block("unknown", f"the source trained on {theirs.get('dataset')} and this run reads "
                                          f"{ours.dataset}, which share images")
        return split_block("different_data", f"the source trained on {theirs.get('dataset')}, this run reads "
                                             f"{ours.dataset}", {"data.dataset": theirs.get("dataset")}, None, (0, 0))
    moved = [key for key in LOCATION_KEYS if theirs.get(key) != getattr(ours, key, None)]
    if theirs.get("format") != ours.format or moved:
        return split_block("unknown", f"the source read {theirs.get('format')} data "
                                      f"{({key: theirs.get(key) for key in moved})}; rows are matched within one "
                                      "dataset or one listing only")
    if ours.format != "builtin" and _fingerprint(source.path) != data.fingerprint:
        return split_block("unknown", "the source's fit sidecar records another data_fingerprint (or none) for this "
                                      "listing, so its files may have changed")
    def split_keys(seed: Any, split: Optional[float], cap: Optional[int]) -> Dict[str, Any]:
        return {"seed": seed, "data.validation_split": DEFAULT_VALIDATION_SPLIT if split is None else split,
                "data.max_samples": cap}

    keys = split_keys(source.config.get("seed"), theirs.get("validation_split"), theirs.get("max_samples"))
    mine = split_keys(config.seed, ours.validation_split, ours.max_samples)
    if keys == mine:
        return same_split("rows", keys)
    pool = data.n_train + data.n_select
    n = pool if ours.max_samples is None or pool < ours.max_samples else table
    if n is None:
        return split_block("unknown", f"this run caps data.max_samples at {ours.max_samples}, so the size of the "
                                      "listing both cuts permute is not known")
    # DECISION plan-2026-11-18T072904-0fd54b91/D-012: the source's rows are rebuilt with the SOURCE's seed, split and
    # cap (its config.json), this run's with its own. Do NOT take the source's rows from this run's keys or data: the
    # two cuts would then coincide, the count read 0 and a selection-biased verdict pass as unbiased.
    trained = seeded_split_indices(n, 0, keys["seed"], keys["data.validation_split"], keys["data.max_samples"])[0]
    selected = seeded_split_indices(n, 0, config.seed, mine["data.validation_split"], ours.max_samples)[1]
    return counted_split("rows", keys, np.intersect1d(trained, selected).size, len(selected), 0, data.n_test)



# ---------------------------------------------------------------------
# The epoch-zero evaluation and the bar of an init run (D-013)
# ---------------------------------------------------------------------

#: The ``notes`` line of an ``init`` run, whose ``initial_loss_ratio`` and ``init_scale_warning`` are null.
NOT_JUDGED_UNDER_INIT = ("`initial_loss_ratio` and `init_scale_warning` are null (not computed): under `init` the "
                         "epoch-0 evaluation is of the loaded weights, not of an initialization")


def evaluate_loaded(model: Any, data: Any, hint: str) -> Dict[str, float]:
    """The epoch-0 evaluation of an ``init`` run (no ratio, no verdict); a non-finite value raises ``RuntimeError``."""
    import math

    from dl_techniques.utils.logger import logger
    from train.common import classification_run

    metrics = classification_run.evaluate_dict(model, data.select_ds)
    shown = ", ".join(f"val {name} {value:.6g}" for name, value in metrics.items())
    logger.info(f"Evaluate of the loaded weights before fit: {shown} on {data.n_select} selection samples")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Initial validation {shown}: the loaded weights diverge on this run's data. {hint}")
    return metrics


def loaded_value(config: Any, run_dir: Any, key: str, net_of: float = 0.0) -> Optional[float]:
    """Under ``init``, the fit sidecar's ``initial.metrics[key]`` minus ``net_of`` (a resume reads the same number);
    else ``None``, reading nothing. :raises ValueError: under ``init``, a sidecar without that value."""
    if config.init is None:
        return None
    path = Path(run_dir) / "checkpoints" / "fit_state.json"
    value = json.loads(path.read_text()).get("initial", {}).get("metrics", {}).get(key) if path.is_file() else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} holds no initial.metrics.{key}, the loaded weights' epoch-zero value the bar must beat")
    return float(value) - net_of


def beats_loaded(reached: float, loaded: Optional[float], name: str, offset: float = 0.0) -> bool:
    """Whether ``reached + offset`` is above ``loaded + offset`` by more than ``tie_band(1)`` float32 steps of that value;
    true when ``loaded`` is None. ``offset`` turns a success value back into the metric it is net of (denoising)."""
    # DECISION plan-2026-11-18T072904-0fd54b91/D-013
    # Under init the success value must be STRICTLY above the loaded weights' epoch zero, in float32, beyond the tie band:
    # a good source clears an absolute bar at epoch 0. Do NOT drop the term or use `>=` (a fine-tune that only held the
    # source's score would pass), and do NOT apply it without init (`loaded` None keeps every other verdict).
    if loaded is None:
        return True
    import numpy as np

    from dl_techniques.utils.logger import logger
    from train.common.forecast_run import tie_band

    low, band = np.float32(loaded + offset), tie_band(1)
    met = bool(np.float32(reached + offset) > low + np.float32(band) * abs(np.spacing(low)))
    if not met:
        logger.info(f"Under init the best {name} {reached:.6g} does not beat the loaded weights' epoch-zero "
                    f"{loaded:.6g} by more than {band} float32 steps")
    return met
