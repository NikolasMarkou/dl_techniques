"""Watch the topography form: a callback that draws the map while TopoLM trains.

``--viz-every-steps N`` makes a run score its own topography at step 0 (the
untrained map), every ``N`` optimizer steps and at the end, and write the figures
where a person can open them while ``fit`` is still running::

    <run_dir>/topography_live/
        step_0000000/ ...            the figure set of that step (t_maps_raw.png, ...)
        latest_t_maps_raw.png        the newest t-maps; open this one
        latest_morans_i_raw.png      the newest Moran's I profile
        topography_curve.png         Moran's I, clusters, significant units against step
        metrics.jsonl                one strict-JSON row per draw (``tail -f`` it)

The last frame shows the model the run ENDS with. ``fit`` draws the last step at
the end of its final batch, and only then does
``ConsecutiveIncreaseEarlyStopping.on_train_end`` put back the weights it snapshot
before the final rising streak, so a run that stops early would otherwise end on a
frame of weights it no longer has. ``on_train_end`` therefore compares a
fingerprint of the trainable weights (per-variable float64 sums, taken at every
draw) with the last draw's, and draws again when the step differs or the weights
do. That second draw of an already-drawn step overwrites the step directory and
the ``latest_*`` files and appends a row with the SAME ``step`` and
``"restored_weights": true``; nothing is redrawn when nothing changed. A run
without a restore therefore has strictly increasing steps and no such field, and
a reader (the curve is one) takes the LAST row of a repeated step. The frame is
labelled with the step the run stopped at, which is not the step the restored
weights were trained to.

The counter counts OPTIMIZER steps, resumed runs included (``initial_step``), and
is independent of the virtual epoch: ``steps_per_epoch`` is ``eval_every_steps``,
so a draw every 500 steps in a run validating every 2000 is four draws per
virtual epoch. The ``val_loss`` in a row is the last one Keras reported, and the
row says at which step it was measured.

A draw reuses :func:`~train.topolm.common.evaluate_topography` and
:func:`~train.topolm.common.plot_topography`, so the last live frame and the
post-hoc figure come out of one code path, for a 2D and a 3D backbone alike. It is
restricted to the two contrast conditions and the raw arm, without a permutation
test, because it runs inside the training loop.

Two promises, and what backs each
---------------------------------
**A draw never changes the run.** Eval mode only (``training=False``, always: a
training-mode forward overwrites each tap's non-trainable ``last_spatial_loss``
and advances every ``SeedGenerator``), no head call, no ``model.predict``. A
tripwire compares every ``SeedGenerator`` state and the Python and NumPy RNG
states around each draw; on any difference it restores them, logs an ERROR and
switches the callback off. ``tests/test_train/test_topolm/test_topolm_live.py``
holds the tripwire and a same-seed CPU A/B (trainable and non-trainable weights,
SeedGenerator states and per-epoch losses bit-equal to a run without the
callback), each with a negative control seen failing.

**A draw never kills the run.** Any ``Exception`` in a draw is logged and the run
goes on; three CONSECUTIVE failures switch the callback off. ``KeyboardInterrupt``
and ``SystemExit`` are not caught. The post-hoc
:func:`~train.topolm.common.run_topographic_evaluation` still raises on a real
failure, so a run cannot exit 0 without its topography.

The stimuli are the smoke set unless the config carries its own (five sentences
per condition), so a live map is noisy and says so in its subtitle: it shows that
a structure is forming, not a result.
"""

import contextlib
import json
import os
import random
import shutil
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import keras

from dl_techniques.utils.logger import logger

from train.topolm import common as C

#: Prompts per forward pass in a draw. The live pass shares the card with the
#: training step, so it stays small.
DRAW_BATCH_SIZE = 8

#: Consecutive failed draws after which the callback switches itself off.
MAX_CONSECUTIVE_FAILURES = 3

#: Figures of a step directory that are mirrored as ``latest_<name>``.
LATEST_PREFIXES = ("t_maps_", "morans_i_")


# ---------------------------------------------------------------------
# The tripwire: what a draw must leave exactly as it found it
# ---------------------------------------------------------------------


def seed_generators(model) -> List[Any]:
    """Every ``keras.random.SeedGenerator`` reachable from ``model``, once each.

    Walks ``model._flatten_layers()`` and reads each layer's ``_seed_generators``
    (Keras 3.8 private attributes, the only registry there is). If a Keras release
    renames them this returns an empty list and the tripwire would pass vacuously,
    which is why :meth:`TopographyLiveCallback.on_train_begin` warns when it finds
    fewer generators than the backbone has taps.

    :param model: A Keras model or layer.
    :type model: keras.layers.Layer
    :return: The generators, in layer order, without duplicates.
    :rtype: List[keras.random.SeedGenerator]
    """
    found: List[Any] = []
    seen = set()
    for layer in model._flatten_layers():
        for generator in getattr(layer, "_seed_generators", ()):
            if id(generator) not in seen:
                seen.add(id(generator))
                found.append(generator)
    return found


def capture_run_state(model) -> Dict[str, Any]:
    """Everything a draw must not move: SeedGenerator states and both global RNGs.

    :param model: The model whose generators to read.
    :type model: keras.layers.Layer
    :return: ``{"seed_generators": [(generator, state_copy), ...], "python":
        random.getstate(), "numpy": np.random.get_state() with a copied key
        array}``. Opaque to callers: pass it to :func:`run_state_changes` and
        :func:`restore_run_state`.
    :rtype: Dict[str, Any]
    """
    numpy_state = np.random.get_state()
    return {
        "seed_generators": [
            (generator, np.array(keras.ops.convert_to_numpy(generator.state)))
            for generator in seed_generators(model)
        ],
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
    }


def run_state_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[str]:
    """Name what differs between two :func:`capture_run_state` results.

    :return: One short label per difference (``"SeedGenerator 2"``,
        ``"Python random state"``, ``"NumPy random state"``); empty when nothing
        moved.
    :rtype: List[str]
    """
    changed = [
        f"SeedGenerator {index}"
        for index, ((_, left), (_, right)) in enumerate(
            zip(before["seed_generators"], after["seed_generators"])
        )
        if not np.array_equal(left, right)
    ]
    if before["python"] != after["python"]:
        changed.append("Python random state")
    left, right = before["numpy"], after["numpy"]
    if (
        left[0] != right[0]
        or not np.array_equal(left[1], right[1])
        or tuple(left[2:]) != tuple(right[2:])
    ):
        changed.append("NumPy random state")
    return changed


def restore_run_state(state: Mapping[str, Any]) -> None:
    """Put back what :func:`capture_run_state` read (generators and both RNGs)."""
    for generator, saved in state["seed_generators"]:
        generator.state.assign(saved)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """``value`` as strict-JSON data: non-finite floats become ``None``.

    ``json.dumps(..., allow_nan=False)`` is the guard; this is what makes it
    pass. NumPy scalars and arrays become Python numbers and lists.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _failure_site(error: BaseException) -> str:
    """Where ``error`` was raised, as ``file.py:line in function`` (the innermost frame)."""
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return "unknown location"
    last = frames[-1]
    return f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}"


def weights_fingerprint(variables: Sequence[Any]) -> np.ndarray:
    """A cheap, exact-enough identity of a set of weights: ``(n, 2)`` float64.

    One row per variable, ``[sum, sum of squares]`` accumulated in float64. It
    reads the values (``convert_to_numpy``) and nothing else: no RNG, no optimizer
    or metric state, no forward pass. Two weight sets that differ in any variable
    by more than float64 rounding of these two sums are told apart; it is an
    equality test for "did the weights change", not a hash for storage.

    :param variables: Keras variables (typically ``backbone.trainable_variables``).
    :type variables: Sequence[keras.Variable]
    :return: The fingerprint; compare two with :func:`same_fingerprint` (a NaN
        weight must not make a run differ from itself).
    :rtype: numpy.ndarray
    """
    rows = []
    for variable in variables:
        values = np.asarray(keras.ops.convert_to_numpy(variable), dtype="float64")
        rows.append([values.sum(dtype="float64"), np.square(values).sum(dtype="float64")])
    return np.asarray(rows, dtype="float64").reshape(-1, 2)


def same_fingerprint(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> bool:
    """Whether two :func:`weights_fingerprint` results are equal (NaN equals NaN)."""
    if left is None or right is None:
        return False
    return left.shape == right.shape and bool(np.array_equal(left, right, equal_nan=True))


def read_metrics_rows(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Every readable row of a ``metrics.jsonl``, and how many lines were not.

    A crash mid-write leaves a truncated last line; a reader must survive it. A
    line that is not a JSON object is skipped and counted. A repeated ``step``
    (the redraw after a weight restore, ``"restored_weights": true``) stays as
    written: take the last row of a step with
    :func:`train.topolm.plotting.last_row_per_step`.

    :param path: The file; a missing file is an empty list.
    :type path: str
    :return: ``(rows, skipped)``.
    :rtype: Tuple[List[Dict[str, Any]], int]
    """
    rows: List[Dict[str, Any]] = []
    skipped = 0
    if not os.path.exists(path):
        return rows, skipped
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                skipped += 1
    return rows, skipped


def _metrics_row(
    step: int,
    seconds: float,
    val_loss: Optional[float],
    val_loss_step: Optional[int],
    report: Mapping[str, Any],
    arrays: Mapping[str, Any],
    restored_weights: bool = False,
) -> Dict[str, Any]:
    """The ``metrics.jsonl`` row of one draw (raw arm), JSON-safe.

    ``per_tap`` holds, for every tap in depth order: standard and islands Moran's
    I, the BH-FDR significant units, the clusters found for each polarity and the
    min, max and mean absolute value of the t-map. ``restored_weights`` adds the
    boolean field of that name (``true``) to the redraw of an already-drawn step
    whose weights changed afterwards; an ordinary row does not carry it.
    """
    arm = report["arms"]["raw"]
    t_maps = arrays["arms"]["raw"]["t_maps"]
    per_tap: Dict[str, Any] = {}
    for name, entry in arm["per_tap"].items():
        t_map = np.asarray(t_maps[name], dtype="float64")
        per_tap[name] = {
            "morans_i": entry["morans_i"]["standard"],
            "morans_i_islands": entry["morans_i"]["islands"],
            "num_significant_units": entry["num_significant_units"],
            "clusters_a": len(entry["cluster_sizes_a"]),
            "clusters_b": len(entry["cluster_sizes_b"]),
            "t_min": float(np.min(t_map)),
            "t_max": float(np.max(t_map)),
            "t_abs_mean": float(np.mean(np.abs(t_map))),
        }
    row = {
        "step": step,
        "seconds": seconds,
        "val_loss": val_loss,
        "val_loss_step": val_loss_step,
        "grid_shape": report["grid_shape"],
        "contrast": report["contrast"],
        "stimuli_are_smoke_set": report["stimuli_are_smoke_set"],
        "mean_morans_i": arm["mean_morans_i"],
        "mean_num_significant_units": arm["mean_num_significant_units"],
        "total_clusters_a": arm["total_clusters_a"],
        "total_clusters_b": arm["total_clusters_b"],
        "per_tap": per_tap,
    }
    if restored_weights:
        row["restored_weights"] = True
    return _json_safe(row)


def _ends_with_newline(path: str) -> bool:
    """Whether ``path`` is missing, empty or ends in a newline (an append starts a line)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return True
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


# ---------------------------------------------------------------------
# The callback
# ---------------------------------------------------------------------


class TopographyLiveCallback(keras.callbacks.Callback):
    """Draw the model's topography during ``fit``, without changing the run.

    Built by :func:`~train.topolm.common.build_callbacks` when
    ``viz_every_steps > 0`` and appended LAST, so ``on_train_end`` runs after
    ``ConsecutiveIncreaseEarlyStopping`` has restored its snapshot. The final
    frame shows the weights the run ends with, like the post-hoc figure: when the
    last step was already drawn (``viz_every_steps`` divides it) and the weights
    have changed since, ``on_train_end`` draws that step again, flagged
    ``restored_weights`` in its row (see the module docstring). See it too for
    the outputs and the two promises.

    What the non-perturbation check covers, and what it does not. The tripwire
    compares every ``SeedGenerator`` in the model tree (the head's and the
    backbone's, found through ``Layer._seed_generators``) and the Python and NumPy
    global RNG states around each draw. It does NOT snapshot Keras's own global
    ``SeedGenerator`` (what ``keras.random.*`` uses when no ``seed`` is given) or
    TensorFlow's global op-seed counter. A draw does not use either: the eval-mode
    pass calls no ``keras.random`` function and no seeded TF op, and that is
    pinned by the same-seed A/B in ``test_topolm_live.py`` (trainable and
    non-trainable weights, SeedGenerator states and per-epoch losses bit-equal to
    a run without the callback), not by the tripwire. A future draw that used a
    global generator would be caught by the A/B, not by the tripwire.

    :param backbone: The TopoLM backbone to score (2D or 3D).
    :type backbone: TopoLM
    :param preprocessor: Tokenizer for the stimuli.
    :type preprocessor: Any
    :param output_dir: The ``topography_live`` directory, created on first write.
    :type output_dir: str
    :param viz_every_steps: Draw every this many optimizer steps; positive.
    :type viz_every_steps: int
    :param stimuli: ``{condition: [sentence, ...]}``; only the two contrast
        conditions are used.
    :type stimuli: Mapping[str, Sequence[str]]
    :param contrast_conditions: The two conditions the t-map contrasts.
    :type contrast_conditions: Tuple[str, str]
    :param min_cluster_size: Smallest cluster kept.
    :type min_cluster_size: int
    :param is_smoke_set: Whether ``stimuli`` is the built-in stand-in; it is said
        in every figure's subtitle and in every row.
    :type is_smoke_set: bool
    :param initial_step: The step the run resumes from. Labels are
        ``initial_step`` plus the optimizer steps this ``fit`` has run, and the
        cadence is on those labels.
    :type initial_step: int
    :param batch_size: Prompts per forward pass in a draw.
    :type batch_size: int
    :raises ValueError: If ``viz_every_steps`` is not positive or a contrast
        condition is absent from ``stimuli`` (at construction, so a misconfigured
        run fails before its first step rather than being swallowed per draw).
    """

    def __init__(
        self,
        backbone,
        preprocessor,
        output_dir: str,
        viz_every_steps: int,
        stimuli: Mapping[str, Sequence[str]],
        contrast_conditions: Tuple[str, str] = ("a", "b"),
        min_cluster_size: int = 10,
        is_smoke_set: bool = False,
        initial_step: int = 0,
        batch_size: int = DRAW_BATCH_SIZE,
    ) -> None:
        super().__init__()
        if viz_every_steps <= 0:
            raise ValueError(
                f"viz_every_steps must be positive, got {viz_every_steps}"
            )
        missing = [name for name in contrast_conditions if name not in stimuli]
        if missing:
            raise ValueError(
                f"contrast conditions {missing} absent from the stimuli "
                f"(available: {sorted(stimuli)})"
            )
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.output_dir = output_dir
        self.viz_every_steps = int(viz_every_steps)
        self.contrast_conditions = tuple(contrast_conditions)
        self.stimuli = {
            name: list(stimuli[name]) for name in self.contrast_conditions
        }
        self.min_cluster_size = int(min_cluster_size)
        self.is_smoke_set = bool(is_smoke_set)
        self.initial_step = int(initial_step)
        self.batch_size = int(batch_size)

        self._steps_run = 0
        self._val_loss: Optional[float] = None
        self._val_loss_step: Optional[int] = None
        self._last_attempt: Optional[int] = None
        self._last_fingerprint: Optional[np.ndarray] = None
        self._warned_unreadable = False
        self._consecutive_failures = 0
        self._draws = 0
        self.disabled = False

    # -- Keras hooks ---------------------------------------------------

    @property
    def step(self) -> int:
        """The step label now: ``initial_step`` plus optimizer steps run."""
        return self.initial_step + self._steps_run

    def on_train_begin(self, logs=None) -> None:
        """Warn if the tripwire would be blind, then draw the untrained map."""
        self._steps_run = 0
        found = len(seed_generators(self._guarded_model()))
        taps = len(getattr(self.backbone, "tap_layers", ()))
        if found < taps:
            logger.warning(
                f"Live topography: the tripwire found {found} SeedGenerator(s) "
                f"for {taps} tap(s); a Keras release may have renamed "
                f"Layer._seed_generators, and the non-perturbation check is "
                f"blind to them."
            )
        self._draw(self.step)

    def on_train_batch_end(self, batch, logs=None) -> None:
        """Count the optimizer step; draw when the label is a multiple of N.

        The ``batch`` index restarts every virtual epoch and is not used.
        """
        self._steps_run += 1
        if self.step % self.viz_every_steps == 0:
            self._draw(self.step)

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Remember the latest ``val_loss`` and the step it was measured at."""
        value = (logs or {}).get("val_loss")
        if value is not None:
            self._val_loss = float(value)
            self._val_loss_step = self.step

    def on_train_end(self, logs=None) -> None:
        """Draw the final frame: the model the run ends with.

        Draws when the last step was never drawn, and also when it was but the
        weights are not the ones that draw saw (the early stopper restored its
        snapshot after the last batch). That redraw replaces the step's frame and
        appends a row with the same step and ``restored_weights: true``. With
        nothing changed it draws nothing, so a run without a restore keeps its
        strictly increasing steps.
        """
        if self._last_attempt != self.step:
            self._draw(self.step)
        elif not same_fingerprint(self._fingerprint(), self._last_fingerprint):
            self._draw(self.step, restored_weights=True)
        logger.info(
            f"Live topography: {self._draws} frame(s) in {self.output_dir}"
            + (" (switched off during the run)" if self.disabled else "")
        )

    # -- one draw ------------------------------------------------------

    def _fingerprint(self) -> np.ndarray:
        """:func:`weights_fingerprint` of the backbone's trainable weights."""
        return weights_fingerprint(self.backbone.trainable_variables)

    def _guarded_model(self):
        """The model whose generators the tripwire watches: the head under
        ``fit``, the backbone when the callback is driven without one."""
        return self.model if self.model is not None else self.backbone

    def _draw(self, step: int, restored_weights: bool = False) -> None:
        """One draw, isolated and checked by the tripwire.

        The frame is written inside ``try``; whatever happens, the run state is
        compared afterwards. A failed frame counts toward the three-in-a-row
        limit and a good one resets the count.

        :param step: The step label of the frame.
        :type step: int
        :param restored_weights: Mark the row (and the figure note) as the redraw
            of ``step`` after the weights changed, i.e. a restore.
        :type restored_weights: bool
        """
        # DECISION plan-2026-10-23T190557-972a9ac5/D-007
        # The draw is an EVAL-MODE pass and nothing else. Do NOT call the backbone
        # or the head with `training=True` in it, not even to warm something up
        # and not even inside a snapshot/restore: that overwrites every tap's
        # non-trainable `last_spatial_loss` (which the restore below does NOT
        # cover, MEASURED) and advances every SeedGenerator, so the run stops
        # being the run it would have been. Do NOT drop the snapshot and the
        # comparison after the pass either: they are what lets this guard fail.
        # The same-seed A/B and the tripwire tests in test_topolm_live.py hold
        # both. See decisions.md D-007, D-008.
        if self.disabled:
            return
        self._last_attempt = step
        model = self._guarded_model()
        self._last_fingerprint = self._fingerprint()
        before = capture_run_state(model)
        try:
            self._write_frame(step, restored_weights)
        except Exception as error:
            # Exception, not BaseException: KeyboardInterrupt and SystemExit
            # must reach the user.
            self._consecutive_failures += 1
            logger.warning(
                f"Live topography draw at step {step} failed "
                f"({self._consecutive_failures}/{MAX_CONSECUTIVE_FAILURES} in a "
                f"row): {type(error).__name__}: {error} (at {_failure_site(error)}). "
                f"The run goes on."
            )
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.disabled = True
                logger.error(
                    f"Live topography switched off after "
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive failed draws; the "
                    f"run continues and the post-hoc evaluation is unaffected."
                )
        else:
            self._consecutive_failures = 0
            self._draws += 1

        changed = run_state_changes(before, capture_run_state(model))
        if changed:
            restore_run_state(before)
            self.disabled = True
            logger.error(
                f"Live topography at step {step} changed run state it must not "
                f"touch ({', '.join(changed)}). The state was restored and the "
                f"live callback is switched off; the run continues. This is a "
                f"bug in the draw, not in the model."
            )

    def _write_frame(self, step: int, restored_weights: bool = False) -> None:
        """Evaluate, plot, publish ``latest_*``, append the row, redraw the curve."""
        started = time.perf_counter()
        report, arrays = C.evaluate_topography(
            self.backbone,
            self.stimuli,
            self.contrast_conditions,
            self.preprocessor,
            readout_fwhm=None,
            min_cluster_size=self.min_cluster_size,
            permutation_p_value=False,
            is_smoke_set=self.is_smoke_set,
            batch_size=self.batch_size,
        )
        written = C.plot_topography(
            report, arrays,
            os.path.join(self.output_dir, f"step_{step:07d}"),
            note=(
                f"step {step}{', restored weights' if restored_weights else ''}"
                f" | {self._stimulus_label()}"
            ),
        )
        for path in written:
            name = os.path.basename(path)
            if name.startswith(LATEST_PREFIXES):
                self._publish(
                    f"latest_{name}",
                    lambda temp, source=path: shutil.copyfile(source, temp),
                )
        row = _metrics_row(
            step, time.perf_counter() - started, self._val_loss,
            self._val_loss_step, report, arrays, restored_weights,
        )
        rows = self._append_row(row)

        # Local, like plot_topography's: importing the callback (it is exported
        # from the package) must not import matplotlib.
        from train.topolm.plotting import plot_topography_curve

        self._publish(
            "topography_curve.png",
            lambda temp: plot_topography_curve(
                rows, temp, note=self._stimulus_label()
            ),
        )

    def _stimulus_label(self) -> str:
        """What the frame was scored on, for the figures: noisy by construction."""
        sizes = "/".join(str(len(v)) for v in self.stimuli.values())
        kind = "smoke stimuli" if self.is_smoke_set else "configured stimuli"
        return f"{kind}, n={sizes} per condition"

    def _append_row(self, row: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Append ``row`` to ``metrics.jsonl`` and return every readable row in it.

        Strict JSON (``allow_nan=False``), flushed and fsynced so a reader
        tailing the file, or a crash, sees whole rows. A file that ends in a
        partial line (a crash mid-write) gets a newline first, so the new row
        starts on its own line instead of being glued to the fragment; unreadable
        lines are skipped when the file is read back, with one warning per
        callback.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "metrics.jsonl")
        prefix = "" if _ends_with_newline(path) else "\n"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(prefix + json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        rows, skipped = read_metrics_rows(path)
        if skipped and not self._warned_unreadable:
            self._warned_unreadable = True
            logger.warning(
                f"Live topography: {path} has {skipped} unreadable line(s) "
                f"(a crash mid-write?); skipped, the view goes on. Further "
                f"unreadable lines are not reported again."
            )
        return rows

    def _publish(self, name: str, write: Callable[[str], Any]) -> str:
        """Make ``<output_dir>/<name>`` appear whole or not at all.

        ``write(temp_path)`` produces the file under a temporary name in the same
        directory and ``os.replace`` moves it into place, so a viewer that
        refreshes ``latest_*.png`` mid-draw never reads a half-written PNG. The
        temporary name ends in ``.png`` because matplotlib picks the format from
        the extension.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        final = os.path.join(self.output_dir, name)
        temp = os.path.join(self.output_dir, f".{os.path.splitext(name)[0]}.tmp.png")
        try:
            write(temp)
            os.replace(temp, final)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(temp)
        return final
