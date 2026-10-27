"""Generic layer-by-layer weight transfer for Keras 3 models.

This module exists because in Keras 3.8+ the API

    model.load_weights(path.keras, by_name=True, skip_mismatch=True)

raises ``ValueError: Invalid keyword arguments: {'by_name': True}`` — the
``by_name`` path is only valid for legacy ``.h5`` / ``.hdf5`` files.  For our
checkpoints, which are saved via ``model.save(path.keras)``, we need a manual
layer-by-layer transfer that matches weights by layer name, skips
user-specified prefixes (typically task heads), records shape mismatches, and
returns a structured audit report.

Typical use::

    from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

    finetune_model = build_my_model(...)     # any multi-head Keras model
    finetune_model.build((None, 256, 256, 3))
    report = load_weights_from_checkpoint(
        target=finetune_model,
        ckpt_path="results/coco_pretrain/final_model.keras",
        skip_prefixes=("head_primary_", "head_aux_"),
    )
    logger.info(report.summary_string())

Three loaders live here, and they are not interchangeable:

* :func:`load_weights_from_checkpoint` — partial, name-matched transfer between
  two *different* architectures. A zero-layer result is a legitimate outcome, so
  it reports rather than raises (several trainers assert on the report).
* :func:`load_weights_or_raise` — whole-file ``.keras`` restore into the *same*
  architecture. A load that restores nothing is always a defect there, so it
  raises.
* :func:`load_pretrained_checkpoint` — a model's own ``pretrained=<path>`` load:
  builds the model if it is not built, transfers through
  :func:`load_weights_from_checkpoint` (a headless model takes a headed
  checkpoint) and raises unless every layer outside the classifier head is
  restored (only the layers named by ``head_pattern`` may differ), when the
  archive's own config differs from the model's on a structural key, or when no
  layer is restored.

:func:`load_weights_from_checkpoint` reports by default; pass ``allow_unloaded``
to make it complete-or-raise instead (every target layer loads and every
checkpoint layer finds a home, except the ones named there).
:func:`read_checkpoint_config` and :func:`vocab_change` read a ``.keras``
archive's own config, so a caller decides from what the checkpoint actually
holds rather than from a class default. :func:`load_backbone_from_checkpoint`
is the language models' load rule built from those pieces, in one place, and
:func:`pretrained_path_or_raise` is the factories' shared reading of their
``pretrained`` argument; :func:`refuse_pretrained` is its sibling for a model
with no backbone/head split, which refuses a path as well as ``True``.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, NamedTuple, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger


# Layer-by-layer set_weights rather than load_weights(by_name=True).
# Keras 3.8's load_weights on a .keras file rejects the by_name kwarg outright.
# Three existing helpers in the repo (cliffordnet/model.py:413, bfunet.py:515,
# convnext_v2.py:400) carry this latent bug. Documented in plans/LESSONS.md.


@dataclass
class TransferReport:
    """Structured audit of a weight-transfer operation.

    :ivar loaded: names of target layers whose weights were overwritten from the source.
    :ivar skipped_by_prefix: target layer names matched by ``skip_prefixes`` and therefore not touched.
    :ivar shape_mismatch: ``(layer_name, target_shapes, source_shapes)`` tuples where names matched but weights did not.
    :ivar missing_in_source: target layer names absent from the source checkpoint (kept at target's init).
    :ivar unused_in_source: source layer names with weights that had no home in the target.
    :ivar vocab_fresh: ``layer/variable`` paths of vocabulary tables kept at the
        target's init under a ``vocab_change`` (their layer's other variables loaded).
    """

    loaded: List[str] = field(default_factory=list)
    skipped_by_prefix: List[str] = field(default_factory=list)
    shape_mismatch: List[Tuple[str, List[Tuple[int, ...]], List[Tuple[int, ...]]]] = field(
        default_factory=list
    )
    missing_in_source: List[str] = field(default_factory=list)
    unused_in_source: List[str] = field(default_factory=list)
    vocab_fresh: List[str] = field(default_factory=list)

    @property
    def num_loaded(self) -> int:
        return len(self.loaded)

    @property
    def num_shape_mismatch(self) -> int:
        return len(self.shape_mismatch)

    def summary_string(self) -> str:
        """Human-readable single-block summary suitable for logging."""
        lines = [
            "TransferReport:",
            f"  loaded             : {len(self.loaded)} layers",
            f"  skipped_by_prefix  : {len(self.skipped_by_prefix)} layers",
            f"  shape_mismatch     : {len(self.shape_mismatch)} layers",
            f"  missing_in_source  : {len(self.missing_in_source)} layers",
            f"  unused_in_source   : {len(self.unused_in_source)} layers",
            f"  vocab_fresh        : {len(self.vocab_fresh)} variables",
        ]
        if self.shape_mismatch:
            lines.append("  shape_mismatch details:")
            for name, tgt, src in self.shape_mismatch[:10]:
                lines.append(f"    - {name}: target={tgt} source={src}")
            if len(self.shape_mismatch) > 10:
                lines.append(f"    ... ({len(self.shape_mismatch) - 10} more)")
        if self.vocab_fresh:
            lines.append(f"  vocab_fresh: {self.vocab_fresh}")
        return "\n".join(lines)


class VocabChange(NamedTuple):
    """A checkpoint whose vocabulary differs from the receiving model's.

    Interface contract (built by :func:`vocab_change`, consumed by
    :func:`load_weights_from_checkpoint`'s ``vocab_change``):

    :ivar layers: Layer-name prefixes of the layers that hold vocabulary tables.
    :ivar checkpoint_size: The checkpoint's ``vocab_size``.
    :ivar model_size: The receiving model's ``vocab_size``.
    :ivar tables: Variable-path suffixes naming the vocabulary tables (token
        embedding, untied output kernel/bias), for example
        ``("word_embeddings/embeddings", "lm_head/kernel")``. A suffix matches
        whole path components (``embeddings`` does not match
        ``word_embeddings``). ``()`` declares no vocabulary table: every shape
        difference is then a mismatch, so a checkpoint of another vocabulary
        raises instead of loading.
    """

    layers: Tuple[str, ...]
    checkpoint_size: int
    model_size: int
    tables: Tuple[str, ...]

    def is_table(self, path: str) -> bool:
        """True when a variable path ends with a declared table suffix."""
        return any(path == t or path.endswith("/" + t) for t in self.tables)

    def is_vocab_table(
        self,
        path: str,
        source_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...],
    ) -> bool:
        """True for a declared table whose shape differs only by the vocabulary."""
        # DECISION plan-2026-10-11T170958-dc4d2202/D-007: the ROLE (the path) gates,
        # and the shape only confirms. Do NOT go back to the shape test alone: a
        # position table resized from exactly the old vocab size then passes as
        # "vocabulary-shaped" and stays random with a warning. See decisions.md.
        return (
            self.is_table(path)
            and len(source_shape) == len(target_shape)
            and source_shape != target_shape
            and all(
                s == t or (s == self.checkpoint_size and t == self.model_size)
                for s, t in zip(source_shape, target_shape)
            )
        )


# DECISION plan-2026-08-14T233721-d4f9beb2/D-070: this is a SECOND function, not a
# change to `load_weights_from_checkpoint`. Do NOT "unify" them by making the
# transfer function raise when it loads zero layers: `src/train/beit/
# train_classification.py` and `src/train/energy_transformer/
# train_classification.py` build their own warm-start guards precisely BECAUSE it
# does not raise there, and `tests/test_train/test_beit/test_warm_start.py` pins
# that ("if this ever goes RED, load_weights_from_checkpoint grew a real
# no-overlap guard"). The two functions also use different mechanisms —
# layer-by-layer `set_weights` from a loaded source model vs `model.load_weights`
# on the file. See decisions.md D-070.
def load_weights_or_raise(
    model: keras.Model,
    weights_path: str,
    skip_mismatch: bool = False,
) -> int:
    """``model.load_weights`` that refuses to succeed while restoring nothing.

    ``keras.Model.load_weights(path, skip_mismatch=True)`` returns normally when a
    checkpoint's variable names or shapes do not line up with *model*: every
    variable is left at its initialized value and nothing is reported. The call
    site then logs "loaded weights from ..." and hands back an untrained model.
    This wrapper snapshots the variable values, loads, and counts how many
    actually changed — a load that changes zero variables raises.

    Use this for the whole-file ``.keras`` restore path (a checkpoint of the SAME
    architecture). For a partial, name-matched transfer between DIFFERENT
    architectures — a pretrain trunk into a fine-tune model — use
    :func:`load_weights_from_checkpoint` instead; a zero-layer result is a
    legitimate (if usually unintended) outcome there, and several trainers under
    ``src/train/`` deliberately assert on its report rather than on an exception.

    :param model: A **built** Keras model. Weights can only be restored into
        variables that already exist, so call the model once (or ``build``) first;
        the probe input is model-specific, which is why it is not done here.
    :param weights_path: Path to the weights file.
    :param skip_mismatch: Forwarded to ``load_weights``. ``False`` (default) is
        strict — a shape mismatch raises rather than being silently skipped.
    :returns: The number of variables whose value changed.
    :raises FileNotFoundError: *weights_path* does not exist.
    :raises ValueError: *model* is not built; the load failed; or the load
        completed without changing a single variable.

    .. warning::
       **This guard is all-or-nothing, and it is blind to a PARTIAL restore.**
       It raises only when *zero* variables changed. A 9-of-30 restore returns
       normally with a ``logger.warning`` and the caller sees a success, which
       under ``skip_mismatch=True`` is possible. That is why the language
       models' ``pretrained=<path>`` routes no longer use it with
       ``skip_mismatch=True`` (plan-2026-10-11T004932-b83935e1 D-024, D-026):
       they copy by name through :func:`load_weights_from_checkpoint` with
       ``allow_unloaded``, which is complete-or-raise. Compare the RETURNED
       count against ``len(model.weights)`` if a total restore is what you
       require; a non-zero return is not a claim of completeness.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not model.built:
        raise ValueError(
            "model must be built before loading weights (call it once on a probe "
            "input, or invoke model.build(input_shape), first)."
        )

    try:
        logger.info(f"Loading weights from {weights_path}")
        before = [keras.ops.convert_to_numpy(v) for v in model.weights]
        model.load_weights(weights_path, skip_mismatch=skip_mismatch)
        after = [keras.ops.convert_to_numpy(v) for v in model.weights]
    except Exception as e:
        raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    changed = sum(
        1 for b, a in zip(before, after)
        if b.shape != a.shape or not np.array_equal(b, a)
    )
    total = len(model.weights)

    if changed == 0:
        raise ValueError(
            f"Loading {weights_path} changed none of this model's {total} "
            "variables. Nothing was restored -- the file's variable names or "
            "shapes do not match this model. Check the architecture config, "
            "or use keras.models.load_model() to load the saved model as-is."
        )

    if changed == total:
        logger.info(f"Loaded {weights_path}: all {total} variables changed value.")
    elif skip_mismatch:
        logger.warning(
            f"Loaded {weights_path} with skip_mismatch=True: {changed} of "
            f"{total} variables changed value. The other {total - changed} were "
            "either skipped for a shape mismatch or already equal to the stored "
            "value -- this function cannot tell those two apart, so verify the "
            "load if a mismatch was not intended."
        )
    else:
        # Strict load: nothing was skipped, so the unchanged variables were
        # already equal to what the file holds (common for zero/one-init
        # variables of an untrained checkpoint).
        logger.info(
            f"Loaded {weights_path}: {changed} of {total} variables changed "
            f"value; the other {total - changed} already held the stored value."
        )
    return changed


def _matches_any_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def _listed(names: Sequence[str], limit: int = 8) -> str:
    """``names`` as a list literal, cut to the first *limit* with the rest counted.

    A wrong-architecture checkpoint can leave hundreds of layers unloaded; the
    error then names the first ones and says how many more there are.
    """
    names = list(names)
    if len(names) <= limit:
        return str(names)
    return f"{str(names[:limit])[:-1]}, ... ({len(names) - limit} more)]"


def _layer_weight_shapes(layer: keras.layers.Layer) -> List[Tuple[int, ...]]:
    return [tuple(w.shape) for w in layer.get_weights()]


def _vocab_layer_plan(
    tgt_layer: keras.layers.Layer, src_weights: list, change: VocabChange
) -> Optional[List[Optional[np.ndarray]]]:
    """Per-variable plan for a vocabulary layer whose shapes differ.

    Each entry is the source value to copy, or ``None`` for a vocabulary table
    that keeps the target's init. A variable is a vocabulary table only when
    its path ends with one of ``change.tables`` AND its shape differs only by
    the vocabulary size. ``None`` overall when any variable neither fits nor is
    such a table (or the counts differ).
    """
    variables = tgt_layer.weights
    if len(variables) != len(src_weights):
        return None
    plan: List[Optional[np.ndarray]] = []
    for variable, value in zip(variables, src_weights):
        if tuple(value.shape) == tuple(variable.shape):
            plan.append(value)
        elif change.is_vocab_table(
            variable.path, tuple(value.shape), tuple(variable.shape)
        ):
            plan.append(None)
        else:
            return None
    return plan


def _unwrap_single_model(model: keras.Model) -> keras.Model:
    """The model whose layers a by-name transfer compares.

    A functional model whose only non-input layer is itself a
    ``keras.Model`` (an input/output wrapper such as the two-input models the
    retired ``create_qwen3_classification`` returned) holds every weight in
    that one layer,
    so its layers are the ones compared. Any other model is returned as is.
    """
    inner = [
        layer for layer in model.layers
        if not isinstance(layer, keras.layers.InputLayer)
    ]
    if (
        len(inner) == 1
        and len(inner) < len(model.layers)
        and isinstance(inner[0], keras.Model)
    ):
        return inner[0]
    return model


def load_weights_from_checkpoint(
    target: keras.Model,
    ckpt_path: str,
    skip_prefixes: Sequence[str] = ("head_",),
    strict: bool = False,
    custom_objects: Optional[dict] = None,
    allow_unloaded: Optional[Sequence[str]] = None,
    vocab_change: Optional[VocabChange] = None,
    allow_unloaded_pattern: Optional[str] = None,
) -> TransferReport:
    """Copy weights layer-by-layer from a saved ``.keras`` checkpoint into *target*.

    For every layer in the source model whose name does not start with any of
    ``skip_prefixes``, look up the same-named layer in *target* and call
    ``target_layer.set_weights(source_layer.get_weights())`` when shapes match.

    :param target: A *built* Keras model receiving weights.  Must already have
        been called once or had ``build(input_shape)`` invoked so its layers
        exist and are weight-shaped.
        A functional model whose only non-input layer is a ``keras.Model``
        (an input/output wrapper) is compared through that inner model's
        layers; the same holds for the checkpoint's model.
    :param ckpt_path: Path to a ``.keras`` model file (produced by
        ``keras.Model.save(path)``).  Non-``.keras`` paths raise ``ValueError``
        — use ``keras.models.load_model`` manually for legacy formats.
    :param skip_prefixes: Layer-name prefixes whose source weights should NOT
        be copied into *target*.  Default ``("head_",)`` matches this repo's
        multi-head naming convention (``head_<name>``, ``head_<name>_aux_<k>``);
        pass ``()`` to transfer every matching layer including the heads.
    :param strict: If ``True``, any ``shape_mismatch`` during transfer raises
        :class:`ValueError`.  When ``False`` (default), mismatches are recorded
        in the report and skipped so other layers can still load.
    :param custom_objects: Forwarded to :func:`keras.models.load_model`.
    :param allow_unloaded: ``None`` (default) keeps the reporting behavior
        above. A sequence of layer-name prefixes makes the transfer
        complete-or-raise, in both directions: every target layer that holds
        weights must be loaded, and every source layer that holds weights must
        find a target layer, except layers matched by ``skip_prefixes`` and
        layers whose name starts with one of these prefixes. A shape mismatch
        on an allowed layer is recorded and never raises, even with
        ``strict=True``. ``()`` allows none. Any other target layer left at its
        initialization (shape mismatch, or absent from the source), or source
        layer left out (a deeper checkpoint), raises, naming the layers.
    :param vocab_change: ``None`` (default), or the :class:`VocabChange` of a
        checkpoint with another vocabulary. A layer named by its ``layers``
        whose shapes differ is then transferred variable by variable: each
        variable of equal shape is copied, a declared table whose shape differs
        only by the vocabulary size keeps the target's init (listed in
        ``report.vocab_fresh`` and logged), and any other difference makes the
        whole layer a shape mismatch. A variable counts as a vocabulary table
        only when its path ends with one of ``vocab_change.tables``.
    :param allow_unloaded_pattern: ``None`` (default), or a regular expression
        that a layer name must match IN FULL to be allowed like the
        ``allow_unloaded`` prefixes (an exact-name rule: ``classification_head``
        does not let ``classification_head_proj`` through). Giving it turns the
        complete-or-raise check on, with ``allow_unloaded`` ``None`` read as
        ``()``.
    :returns: :class:`TransferReport` describing what happened.
    :raises FileNotFoundError: *ckpt_path* does not exist.
    :raises ValueError: *ckpt_path* does not end in ``.keras``; all source
        layers were filtered / missing in target (no overlap); or, with
        ``allow_unloaded``, a layer outside it was not transferred.

    Every check runs before the first weight is copied, so a raise leaves
    *target* untouched.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not str(ckpt_path).endswith(".keras"):
        raise ValueError(
            f"load_weights_from_checkpoint only supports .keras files; "
            f"got {ckpt_path!r}. Re-save the source via model.save('path.keras')."
        )
    if not target.built:
        raise ValueError(
            "target model must be built before weight transfer "
            "(call model.build(input_shape) or run a probe forward pass first)."
        )

    logger.info(f"Loading source model from {ckpt_path}")
    source = keras.models.load_model(ckpt_path, custom_objects=custom_objects)

    # DECISION plan-2026-10-11T004932-b83935e1/D-027: compare the layers of
    # the one model a functional I/O wrapper holds, on either side. Do not
    # compare the wrapper's own layers: its only weighted layer is the whole
    # nested model, so a wrapped classifier's backbone could never move under
    # another head (no overlap, or one all-or-nothing shape mismatch).
    target_layers = {
        layer.name: layer for layer in _unwrap_single_model(target).layers
    }
    source_layers = {
        layer.name: layer for layer in _unwrap_single_model(source).layers
    }

    report = TransferReport()
    allowed = tuple(allow_unloaded or ())
    allowed_pattern = (
        None if allow_unloaded_pattern is None else re.compile(allow_unloaded_pattern)
    )

    def is_allowed(name: str) -> bool:
        return _matches_any_prefix(name, allowed) or (
            allowed_pattern is not None and allowed_pattern.fullmatch(name) is not None
        )

    # A value of None in a pending list keeps that target variable's init.
    pending: List[Tuple[keras.layers.Layer, list]] = []

    # Walk source layers — decide for each whether to transfer, skip, or record.
    for name, src_layer in source_layers.items():
        src_weights = src_layer.get_weights()
        if not src_weights:
            continue  # layer has no state (e.g. pure reshape)

        if _matches_any_prefix(name, skip_prefixes):
            report.skipped_by_prefix.append(name)
            continue

        tgt_layer = target_layers.get(name)
        if tgt_layer is None:
            report.unused_in_source.append(name)
            continue

        tgt_shapes = _layer_weight_shapes(tgt_layer)
        src_shapes = [tuple(w.shape) for w in src_weights]
        if (
            tgt_shapes != src_shapes
            and vocab_change is not None
            and _matches_any_prefix(name, vocab_change.layers)
        ):
            # DECISION plan-2026-10-11T004932-b83935e1/D-031: per variable; do NOT exempt
            # the whole vocabulary layer -- BERT's `embeddings` also holds the position
            # and token-type tables and a LayerNorm, which would then stay fresh silently.
            plan = _vocab_layer_plan(tgt_layer, src_weights, vocab_change)
            if plan is not None:
                report.vocab_fresh.extend(
                    variable.path
                    for variable, value in zip(tgt_layer.weights, plan)
                    if value is None
                )
                pending.append((tgt_layer, plan))
                report.loaded.append(name)
                continue
        if tgt_shapes != src_shapes:
            report.shape_mismatch.append((name, tgt_shapes, src_shapes))
            if strict and not is_allowed(name):
                raise ValueError(
                    f"Shape mismatch on layer {name!r}: target={tgt_shapes} "
                    f"source={src_shapes}"
                )
            continue

        pending.append((tgt_layer, src_weights))
        report.loaded.append(name)

    # Pass 2: detect target layers that have weights but were never loaded.
    loaded_set = set(report.loaded)
    skipped_set = set(report.skipped_by_prefix)
    for name, tgt_layer in target_layers.items():
        if not tgt_layer.get_weights():
            continue  # layer has no state
        if name in loaded_set or name in skipped_set:
            continue
        # Skip prefixes — target head layers are expected to be uninitialized from source
        if _matches_any_prefix(name, skip_prefixes):
            continue
        report.missing_in_source.append(name)

    if (
        not report.loaded
        and not report.skipped_by_prefix
        and not report.shape_mismatch
    ):
        # No overlap at all — almost certainly a user error (wrong checkpoint?).
        # Note: shape_mismatch still counts as overlap — same names, different shapes.
        raise ValueError(
            f"No overlapping layers between target and source checkpoint "
            f"{ckpt_path!r}. Target has {len(target_layers)} layers, source has "
            f"{len(source_layers)}."
        )

    if allow_unloaded is not None or allowed_pattern is not None:
        # missing_in_source holds every weight-bearing target layer neither
        # loaded nor skipped by prefix: absent from the source OR mismatched.
        unloaded = [
            n for n in report.missing_in_source
            if not is_allowed(n)
        ]
        # DECISION plan-2026-10-11T004932-b83935e1/D-031: the source side too; do NOT
        # let a checkpoint layer without a target home pass -- a deeper checkpoint is
        # then truncated silently and a trained final norm runs after a block it never saw.
        unused = [
            n for n in report.unused_in_source
            if not is_allowed(n)
        ]
        if unloaded or unused:
            raise ValueError(
                f"Checkpoint {ckpt_path!r} leaves {len(unloaded)} target "
                f"layer(s) unloaded: {_listed(unloaded)}, and holds {len(unused)} "
                f"layer(s) with no target layer: {_listed(unused)}. Only layers "
                f"starting with {list(allowed)}"
                + (f" or named in full by {allowed_pattern.pattern!r}" if allowed_pattern else "")
                + f" may be left out on either side.\n{report.summary_string()}"
            )

    for tgt_layer, src_weights in pending:
        if all(value is not None for value in src_weights):
            tgt_layer.set_weights(src_weights)
            continue
        for variable, value in zip(tgt_layer.weights, src_weights):
            if value is not None:
                variable.assign(value)

    if report.vocab_fresh:
        logger.warning(
            f"Vocabulary change ({vocab_change.checkpoint_size} -> "
            f"{vocab_change.model_size}): these variables keep their fresh "
            f"initialization: {report.vocab_fresh}"
        )
    logger.info(report.summary_string())
    return report


def read_checkpoint_config(ckpt_path: str) -> Dict[str, Any]:
    """Return the top-level ``config.json`` of a ``.keras`` archive.

    Interface contract (shared by several model loaders):

    :param ckpt_path: Path to a ``.keras`` archive.
    :returns: The parsed dict (``class_name``, ``config``, ``build_config``,
        ...), or ``{}`` when the file is missing, not a zip archive, holds no
        ``config.json``, or that file is not a JSON object. Never raises for
        an unreadable file: the caller's transfer then reports the real error.
    """
    try:
        with zipfile.ZipFile(ckpt_path) as archive:
            config = json.loads(archive.read("config.json"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return {}
    return config if isinstance(config, dict) else {}


def _backbone_config(ckpt_path: str) -> Dict[str, Any]:
    """The config that records a language model's ``vocab_size`` / ``include_top``.

    The archive's own top-level config; for a functional wrapper around ONE
    model (the two-input interface ``create_qwen3`` used to return, still
    found in old archives; the same shape :func:`_unwrap_single_model`
    compares by), the wrapped model's config
    instead, since the wrapper itself records neither.

    :param ckpt_path: Path to a ``.keras`` archive.
    :returns: The config dict, ``{}`` when unreadable (see
        :func:`read_checkpoint_config`).
    """
    # DECISION plan-2026-10-11T170958-dc4d2202/D-015: read through a one-model wrapper; do NOT
    # require the checkpoint to be the bare class -- the dispatching factories return the wrapper,
    # so that is what users save, and without this a headed wrapper cannot warm-start a headless
    # model and a vocabulary change is a shape mismatch. See decisions.md.
    # Superseded 2026-10-13: the factories return the class, so the wrapper case applies only
    # to old archives.
    config = read_checkpoint_config(ckpt_path).get("config") or {}
    layers = config.get("layers")
    if "vocab_size" in config or not isinstance(layers, list):
        return config
    inner = [
        layer for layer in layers
        if isinstance(layer, dict) and layer.get("class_name") != "InputLayer"
    ]
    if len(inner) == 1 and len(inner) < len(layers):
        return inner[0].get("config") or {}
    return config


def vocab_change(
    ckpt_path: str,
    vocab_size: int,
    vocab_layers: Sequence[str],
    tables: Sequence[str],
) -> Optional[VocabChange]:
    """The checkpoint's vocabulary change, if it records one.

    Interface contract (shared by the language-model loaders through
    :func:`load_backbone_from_checkpoint`):

    :param ckpt_path: Path to a ``.keras`` archive of a model whose top-level
        config carries ``vocab_size`` (or a functional wrapper around one such
        model, whose config is then read, see :func:`_backbone_config`), or
        whose ``arch_config`` dict does (H-Net keeps its vocabulary there).
    :param vocab_size: The receiving model's vocabulary size.
    :param vocab_layers: Layer-name prefixes of the layers holding vocabulary
        tables.
    :param tables: Variable-path suffixes of the vocabulary tables themselves
        (required: there is no shape-only default); see :class:`VocabChange`.
    :returns: A :class:`VocabChange` when the checkpoint records a
        ``vocab_size`` other than *vocab_size*; ``None`` when it records the
        same one or none (an unknown vocabulary is not a change, so nothing is
        exempted).
    """
    config = _backbone_config(ckpt_path)
    recorded = config.get("vocab_size")
    # DECISION plan-2026-10-13T035756-5dd4d94d/D-007: HNet records its vocabulary only in
    # arch_config.vocab_size (origin: plan-2026-10-11T170958-dc4d2202 D-017). Do NOT drop this
    # fallback or replace it with a top-level key or a class default: HNet's top-level config has no
    # vocab_size, so a vocabulary change would read as "no change" and raise as a shape mismatch
    # instead of keeping only the token table fresh.
    if recorded is None and isinstance(config.get("arch_config"), dict):
        recorded = config["arch_config"].get("vocab_size")
    if recorded is None or recorded == vocab_size:
        return None
    return VocabChange(
        tuple(vocab_layers), int(recorded), int(vocab_size), tuple(tables)
    )


def load_backbone_from_checkpoint(
    target: keras.Model,
    ckpt_path: str,
    head_prefixes: Sequence[str],
    vocab_layers: Sequence[str],
    tables: Sequence[str],
    tops: Sequence[str] = (),
    allow_vocab_change: bool = True,
) -> TransferReport:
    """The language models' ``pretrained=<path>`` load rule, in one place.

    Interface contract (shared by the ten language classes with a pretrained
    route):

    :param target: A **built** model with ``vocab_size`` and ``include_top``
        attributes, or a functional wrapper around exactly one such model (its
        layers are then the ones transferred, see :func:`_unwrap_single_model`).
    :param ckpt_path: A ``.keras`` checkpoint of the same class, or of a
        functional wrapper around it, whose config records ``vocab_size`` and
        ``include_top``.
    :param head_prefixes: Prefixes of the task head, never copied (and a
        target head keeps its init).
    :param vocab_layers: Prefixes of the layers holding vocabulary tables.
    :param tables: Variable-path suffixes of the vocabulary tables themselves
        (token embedding, untied output kernel/bias), required. Only these
        variables may keep their init on a vocabulary change; any other shape
        difference in a vocabulary layer, for example a resized position
        table, raises. ``()`` makes every vocabulary change raise.
    :param tops: Prefixes of a headless model's output layer (an untied
        ``lm_head``), absent from a headed model.
    :param allow_vocab_change: ``False`` refuses a checkpoint of another
        vocabulary.
    :returns: The :class:`TransferReport`.
    :raises ValueError: when any backbone layer does not transfer, on either
        side; failure modes as :func:`load_weights_from_checkpoint`, plus a
        refused vocabulary change. A raise copies nothing.

    The rule: every backbone layer loads, and every checkpoint backbone layer
    has a home (a deeper checkpoint is refused, never truncated). On a
    vocabulary change, only the declared ``tables``, shaped by the vocabulary, keep
    their init; every other variable of those layers must load (a resized
    position table raises). A top may be left
    out only when one side carries a task head instead (the checkpoint's
    config or the target says ``include_top``) or the vocabulary changed, so
    a headless tied checkpoint does not fill an untied ``lm_head`` with
    nothing, and an untied ``lm_head`` is not dropped into a tied model.
    """
    target = _unwrap_single_model(target)
    change = vocab_change(
        ckpt_path, target.vocab_size, (*vocab_layers, *tops), tables
    )
    if change is not None and not allow_vocab_change:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} has another vocab_size "
            f"({change.checkpoint_size}) than this model ({change.model_size}), "
            "and the vocabulary exception was refused."
        )
    recorded = _backbone_config(ckpt_path)
    headed = bool(recorded.get("include_top")) or bool(target.include_top)
    return load_weights_from_checkpoint(
        target=target,
        ckpt_path=ckpt_path,
        skip_prefixes=tuple(head_prefixes),
        strict=True,
        allow_unloaded=tuple(tops) if headed or change is not None else (),
        vocab_change=change,
    )


#: Names of the classifier head(s) of the vision classifiers, matched IN FULL: the main
#: ``classification_head`` and ResNet's deep-supervision ``supervision_head_{i}``.
#: The names come from the naming table the house-shape guards enforce
#: (``test_the_layer_names_follow_the_naming_table``), so a model that renames its
#: head fails there before it can fail here. A full match, not a prefix: a trunk
#: layer named ``classification_head_proj`` is not a head and must load.
DEFAULT_HEAD_PATTERN: str = r"classification_head|supervision_head_\d+"

#: Constructor-config keys that never change what a checkpoint holds, for any model:
#: the Keras bookkeeping (``name``, ``trainable``, ``dtype``), the head (its width
#: ``num_classes``, its presence ``include_top``, ``head_config_overrides``: the head's
#: own layer shapes are checked layer by layer). A model passes these plus the keys of
#: its own that leave the weight layout and the computed function alone (initializers,
#: regularizers, dropout rates) as ``ignore_config_keys``.
DEFAULT_NON_STRUCTURAL_CONFIG_KEYS: Tuple[str, ...] = (
    "name", "trainable", "dtype", "num_classes", "include_top", "head_config_overrides",
)


def _plain(value: Any) -> Any:
    """``value`` as plain JSON data: Keras' ``__tuple__`` wrapper becomes a list."""
    if isinstance(value, dict):
        if value.get("class_name") == "__tuple__":
            return _plain(value.get("config", {}).get("value"))
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _short(value: Any, limit: int = 80) -> str:
    """``repr(value)`` cut to *limit* characters, for an error message."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _architecture_differences(
    model: keras.Model, weights_path: str, ignore_config_keys: Collection[str]
) -> List[str]:
    """The constructor-config keys on which *model* and the archive disagree.

    Interface contract (used by :func:`load_pretrained_checkpoint`): the archive's
    top-level config is read with :func:`read_checkpoint_config`; the model is
    serialized the way ``model.save`` serializes it, so both sides carry the same
    encoding. Only a key present on BOTH sides and outside *ignore_config_keys* is
    compared (values as plain JSON data, tuples read as lists). Returns one
    ``"key: checkpoint <value>, model <value>"`` entry per difference, sorted by key,
    values cut to 80 characters; ``[]`` when nothing differs, and also when the
    archive holds no config, is of another class (a functional wrapper around the
    model), or the model is a graph of layers (its config lists the layers, whose
    generated names differ at every build): the layer-level check is then the
    only one.
    """
    recorded = read_checkpoint_config(weights_path)
    if recorded.get("class_name") != type(model).__name__:
        return []
    theirs = _plain(recorded.get("config") or {})
    ours = _plain(keras.saving.serialize_keras_object(model).get("config") or {})
    if "layers" in ours:
        return []
    return [
        f"{key}: checkpoint {_short(theirs[key])}, model {_short(ours[key])}"
        for key in sorted(set(theirs) & set(ours) - set(ignore_config_keys))
        if theirs[key] != ours[key]
    ]


def load_pretrained_checkpoint(
    model: keras.Model,
    weights_path: str,
    build_shape: Sequence[Optional[int]],
    skip_mismatch: bool = True,
    head_pattern: str = DEFAULT_HEAD_PATTERN,
    ignore_config_keys: Collection[str] = DEFAULT_NON_STRUCTURAL_CONFIG_KEYS,
) -> TransferReport:
    """Load a local ``.keras`` checkpoint into *model*; the architecture must be the same.

    Interface contract (the shared body for a model's
    ``load_pretrained_weights``): builds *model* at ``(1, *build_shape)`` only
    when it is not built yet, then transfers layer by layer through
    :func:`load_weights_from_checkpoint` with no skipped prefix and the layers
    named by *head_pattern* allowed to differ. The model is never rebuilt: a
    built model keeps its variables.

    Two checks run before the first weight is copied, so a refused load leaves
    *model* untouched:

    * **Layers.** Only the head may differ. With ``skip_mismatch=True`` (default)
      a layer whose whole name matches *head_pattern* may have another shape
      (another ``num_classes``), be absent from the checkpoint (a headless
      checkpoint into a headed model) or be absent from the model (a headed
      checkpoint into an ``include_top=False`` model); a head left at its
      initialization is logged as a warning. Every other layer must be present
      in both and fit: a checkpoint of another variant, block type, width or
      stem, a deeper one and a shallower one all raise, and the error names the
      first offending layers and carries the report's counts (loaded, shape
      mismatch, missing, unused).
    * **Configuration.** Equal shapes do not make equal architectures (ReLU
      against GELU, another ``shifts`` or ``strides``, the same widths under a
      different context mode). When the archive holds a config of *model*'s own
      class, every constructor-config key present on BOTH sides and outside
      *ignore_config_keys* must be equal. A key one side does not have is unknown,
      not a difference: configs change between revisions (arguments removed,
      renamed or added), and an old checkpoint must keep loading. An archive
      without a config of that class (a functional wrapper around the model, a
      checkpoint written by other code) is checked by layers only.

    :param model: The receiving model, built or not.
    :param weights_path: Path to a ``.keras`` checkpoint.
    :param build_shape: The input shape without the batch axis, used only when
        *model* is not built.
    :param skip_mismatch: ``True`` (default) lets the head layers differ as above.
        ``False`` allows no difference anywhere: every layer of the model and of
        the checkpoint must load, so a classifier of another width raises too.
    :param head_pattern: Regular expression a head layer's name must match in
        full. Default :data:`DEFAULT_HEAD_PATTERN`; pass a pattern that matches
        no layer name when no layer, the head included, may differ.
    :param ignore_config_keys: Config keys not compared: those that leave the
        weight layout and the computed function alone. Each model declares its
        own set as a module constant (``num_classes``, ``include_top``,
        initializers, regularizers and rates are typical); the default holds
        only :data:`DEFAULT_NON_STRUCTURAL_CONFIG_KEYS`. A key that changes a
        weight shape needs no entry: the layer check catches it anyway.
    :returns: The :class:`TransferReport` of the transfer, for a caller that
        wants to inspect what the head kept.
    :raises FileNotFoundError: *weights_path* does not exist.
    :raises ValueError: the archive's configuration differs from the model's
        (the message names each key with both values), the build or the transfer
        failed (chained to the original error), a layer outside the head was not
        restored, or no layer was restored at all.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # DECISION plan-2026-10-26T125959-7af851e5/D-041: compare the archive's own
    # config before anything is built or copied. Do NOT rely on the layer check
    # alone: ReLU to GELU, another `shifts`, `ctx_mode` or strides keep every
    # weight shape, so the layers all load and the model computes another
    # function than the one the checkpoint was trained as. Do NOT compare keys
    # that exist on one side only: revisions add, remove and rename arguments.
    differences = _architecture_differences(model, weights_path, ignore_config_keys)
    if differences:
        raise ValueError(
            f"{weights_path} was saved from another architecture than this "
            f"{type(model).__name__}; the configuration differs on "
            f"{len(differences)} key(s): " + "; ".join(differences)
        )

    try:
        if not model.built:
            model.build((1, *build_shape))
        logger.info(f"Loading pretrained weights from {weights_path}")
        # DECISION plan-2026-10-26T125959-7af851e5/D-040: only the head
        # (`head_prefixes`) may be left out, on either side. Do NOT go back to a
        # bare reporting transfer (`allow_unloaded=None`): resnet50 into resnet18
        # then restored the stem, listed 9 shape mismatches and 9 missing layers
        # and returned, a model that looks pretrained and is mostly random. Do NOT
        # relax it to "at least N layers loaded" either: a count does not say
        # WHICH layers, and the zero-layer rule below is that same count at N=1.
        # DECISION plan-2026-10-26T125959-7af851e5/D-041: (2026-10-27) the D-040 rule above
        # now names its head by `head_pattern`, matched IN FULL (`allow_unloaded_pattern`),
        # not by the `head_prefixes` it was written with: a trunk layer called
        # `classification_head_proj` is not a head and must load. Do NOT go back to prefix
        # matching. The config check above this block is D-041's too; this one is the layer rule.
        report = load_weights_from_checkpoint(
            target=model,
            ckpt_path=weights_path,
            skip_prefixes=(),
            strict=not skip_mismatch,
            allow_unloaded=(),
            allow_unloaded_pattern=head_pattern if skip_mismatch else None,
        )
    except Exception as e:
        raise ValueError(
            f"Failed to load weights from {weights_path}: {e}"
        ) from e

    # DECISION plan-2026-10-26T125959-7af851e5/D-023: raise here, in the models'
    # loader. Do NOT move this check into `load_weights_from_checkpoint`: the D-070
    # anchor above keeps that function reporting, and a same-named checkpoint whose
    # every shape differs returns an empty report there. A model that logged a
    # successful load and then trained from random weights is the defect.
    if report.num_loaded == 0:
        # 2026-10-27 (D-040): the head rule refuses a partial load first; this
        # raise is what remains for a model whose every weighted layer is a head.
        raise ValueError(
            f"Loading {weights_path} restored no layer of this model. "
            f"{report.summary_string()}"
        )
    head = re.compile(head_pattern)
    kept = [
        name for name in report.missing_in_source if head.fullmatch(name)
    ]
    if kept:
        logger.warning(
            f"Head layers kept at their initialization (absent from "
            f"{weights_path} or of another shape): {kept}"
        )
    logger.info(f"Loaded {report.num_loaded} layers from {weights_path}")
    return report


def pretrained_path_or_raise(pretrained: Any, model_label: str) -> Optional[str]:
    """Resolve a model factory's ``pretrained`` argument to a path.

    Interface contract (shared by ``from_variant`` of Qwen3, Qwen3Next, Gemma3,
    Zamba2, HNet, HRM and PTRM, which follow what ``GPT2.from_variant`` does
    with it, and by ``FastVLM.from_variant`` and ``DiTXA.from_variant``):

    :param pretrained: The ``from_variant`` argument. Falsy (``False``, ``None``,
        ``""``) means random initialization. A ``str`` is the path of a local
        ``.keras`` checkpoint. Anything else truthy (``True``) asks for
        released weights.
    :param model_label: Name used in the refusal message.
    :returns: ``None`` for a falsy argument, else the existing path.
    :raises NotImplementedError: for ``True`` (or any non-string truthy value):
        no pretrained weights are distributed, and a silent random
        initialization would be indistinguishable from a successful download.
    :raises FileNotFoundError: when a path is given and does not exist.
    """
    if not pretrained:
        return None
    if not isinstance(pretrained, str):
        raise NotImplementedError(
            f"Pretrained {model_label} weights are not distributed with "
            "dl_techniques. Pass pretrained=<local .keras path> to load a "
            "local checkpoint, or pretrained=False to random-init."
        )
    if not os.path.exists(pretrained):
        raise FileNotFoundError(f"Weights file not found: {pretrained}")
    return pretrained


def refuse_pretrained(pretrained: Any, model_label: str) -> None:
    """Refuse every truthy ``pretrained`` for a model with no backbone/head split.

    Interface contract (callers: ``CLIP.from_variant``,
    ``MobileClipModel.from_variant``, ``MobileClipV2Model.from_variant``,
    ``NanoVLM.from_variant``, ``CliffordCLIP.from_variant``,
    ``SAM.from_variant``, ``SAM2.from_variant``, ``Sam3Image.from_variant``,
    and the variant-table-free factories
    ``create_shared_token_decoder``, ``create_ideogram4_transformer``,
    ``create_ideogram4_autoencoder``, ``create_sd3_mmdit``, ``create_sd3_vae``
    and ``create_sd3_pipeline``). The
    sibling of :func:`pretrained_path_or_raise` for the models whose output IS
    the model, such as a contrastive dual encoder: with no task head there is
    no partial (backbone-only) load to offer, and a whole-archive load is what
    ``keras.models.load_model(path)`` already does.

    :param pretrained: The ``from_variant`` argument. Falsy (``False``,
        ``None``, ``""``) means random initialization and returns.
    :param model_label: Name used in the refusal message.
    :raises NotImplementedError: for any truthy value, ``True`` or a path: no
        pretrained weights are distributed, and a silent random initialization
        would be indistinguishable from a successful load. Raised before any
        model is built.
    """
    if not pretrained:
        return
    raise NotImplementedError(
        f"Pretrained {model_label} weights are not distributed with "
        f"dl_techniques (got pretrained={pretrained!r}), and {model_label} has "
        "no backbone/head split for a partial load. Load a local .keras "
        "archive with keras.models.load_model(path), or pass "
        "pretrained=False to random-init."
    )
