"""Cell runner for the mean shift experiments: idle-GPU gate, manifest, fsynced rows, resume.

One sweep is a list of cells, each ``(experiment, arm, config, seed)``. Every cell trains
one model in its own subprocess, so a failure, an XLA state or a seed state never leaks
into the next cell, and so the idle gate can run before every cell.

Interface contract (callers: ``exp1_mode_pooling`` and ``exp2_mean_shift_attention``, in
their driver mode through :func:`add_sweep_arguments`, :func:`add_run_arguments` and
:func:`run_driver`, in their worker mode through :func:`add_worker_arguments`,
:func:`add_run_arguments`, :func:`load_cell_data`, :func:`fit_cell`,
:func:`selection_split`, :func:`top1`, :func:`physical_devices`, :func:`build_result` and
:func:`write_result`):

* ``run_sweep(cells, worker_module, root, ...)`` writes ``root/manifest.json`` first, then
  for each pending cell (seed-major order) waits for GPU 0 to be idle, runs
  ``python -m <worker_module> --worker --cell-dir D --arm A --config C --seed S <extra>``
  with ``cwd`` at the repository root, and appends ONE fsynced line to ``root/rows.jsonl``.
  A worker writes ``<cell-dir>/result.json`` through :func:`write_result` and exits 0; a
  non-zero exit, a missing result, a timeout or a non-finite loss makes the row ``failed``.
* Resume (``resume=True``) re-reads the rows, skips cells with an ``ok`` row, and runs the
  rest in a new ``cells/<cell_id>__try<k>`` directory. It refuses a manifest whose cell
  list or recipe hash differs. Nothing is ever deleted: a killed cell's directory stays.
* Failure modes: ``ValueError`` (bad cell, bad devices, manifest mismatch),
  ``FileExistsError`` (a fresh sweep into a used root), ``FileNotFoundError`` (resume
  without a manifest), :class:`GateTimeout` (GPU 0 not idle within the sweep's wait cap),
  ``RuntimeError`` (no ``nvidia-smi`` for a GPU sweep).
* Importing this module does not import TensorFlow or Keras: the pieces that need them
  (:func:`build_optimizer`, :func:`epoch_timer`, :func:`load_cell_data`, :func:`fit_cell`,
  :func:`top1`, :func:`physical_devices`, the child environment) import them on call.
"""

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# constants

#: Repository root: children run from here, so any CWD-relative path lands in the repo.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The only GPU the harness may use, as a ``CUDA_VISIBLE_DEVICES`` value ("" is CPU only).
GPU_DEVICE = "0"
CPU_DEVICE = ""

NVIDIA_SMI_COMMAND = [
    "nvidia-smi", "-i", "0", "--query-gpu=memory.used,utilization.gpu",
    "--format=csv,noheader,nounits"]
NVIDIA_SMI_TIMEOUT_S = 20.0

GATE_MAX_MIB = 2048
GATE_MAX_UTIL = 20
GATE_REQUIRED_SAMPLES = 3
GATE_GAP_S = 5.0
GATE_RETRY_S = 60.0
GATE_MAX_WAIT_S = 6 * 3600.0
DEFAULT_CELL_TIMEOUT_S = 7200.0

MANIFEST_NAME = "manifest.json"
ROWS_NAME = "rows.jsonl"
RESULT_NAME = "result.json"
CELL_LOG_NAME = "cell.log"
CELLS_DIR = "cells"
ERROR_TAIL_CHARS = 1500

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_SMI_LINE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*$")

# ---------------------------------------------------------------------
# cells


@dataclasses.dataclass(frozen=True)
class CellSpec:
    """One training run: an arm of an experiment at one config and seed.

    ``extra_args`` are appended verbatim to the worker command line (dataset, epochs,
    sample cap, ...); they are part of the manifest, so a resume with different arguments
    is refused.
    """

    experiment: str
    arm: str
    config: str
    seed: int
    extra_args: Tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("experiment", "arm", "config"):
            value = getattr(self, name)
            if not _NAME.match(value) or "__" in value or ".." in value:
                raise ValueError(
                    f"CellSpec.{name}={value!r} must match {_NAME.pattern} without '__' "
                    "(it becomes part of a directory name)")
        object.__setattr__(self, "extra_args", tuple(str(a) for a in self.extra_args))

    @property
    def key(self) -> Tuple[str, str, str, int]:
        return cell_key(self)

    @property
    def cell_id(self) -> str:
        return f"{self.experiment}__{self.arm}__{self.config}__s{self.seed}"

    def to_dict(self) -> Dict[str, Any]:
        return {"experiment": self.experiment, "arm": self.arm, "config": self.config,
                "seed": int(self.seed), "extra_args": list(self.extra_args)}


def cell_key(cell: Any) -> Tuple[str, str, str, int]:
    """The full identity tuple of a :class:`CellSpec` or of a row dict.

    Every reader and the resume logic use this one function: a key on the seed alone once
    gave a wrong verdict (two arms, same seed).
    """
    if isinstance(cell, Mapping):
        return (cell["experiment"], cell["arm"], cell["config"], int(cell["seed"]))
    return (cell.experiment, cell.arm, cell.config, int(cell.seed))


def seed_major(cells: Sequence[CellSpec]) -> List[CellSpec]:
    """Order cells so every arm completes a seed before any arm starts the next one."""
    return [cell for _, cell in sorted(enumerate(cells), key=lambda p: (p[1].seed, p[0]))]


# ---------------------------------------------------------------------
# the shared recipe


@dataclasses.dataclass(frozen=True)
class Recipe:
    """The training recipe every arm of every experiment shares.

    The field names are the ones :func:`train.common.optimizer.build_optimizer` reads, so a
    recipe is passed to it directly. No early stopping and no analyzer callbacks: every arm
    trains the same number of epochs and its final-epoch weights are scored.
    """

    epochs: int
    warmup_epochs: int
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    batch_size: int = 128
    gradient_clipping: float = 1.0
    lr_schedule_type: str = "cosine_decay"
    optimizer_type: str = "adamw"

    @classmethod
    def for_epochs(cls, epochs: int) -> "Recipe":
        """The recipe for ``epochs``: warmup is a tenth of the run, between 1 and 5 epochs."""
        return cls(epochs=int(epochs), warmup_epochs=min(5, max(1, int(epochs) // 10)))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def recipe_hash(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_optimizer(recipe: Recipe, steps_per_epoch: int):
    """The shared AdamW with linear warmup then cosine decay, built by ``train.common``."""
    from train.common.optimizer import build_optimizer as shared_build_optimizer
    return shared_build_optimizer(recipe, steps_per_epoch)


@functools.lru_cache(maxsize=1)
def _epoch_timer_class():
    import keras

    class EpochTimer(keras.callbacks.Callback):
        """Writes each epoch's wall seconds into the logs, hence into ``History``."""

        def on_epoch_begin(self, epoch, logs=None):
            self._started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                logs["epoch_seconds"] = time.perf_counter() - self._started

    return EpochTimer


def epoch_timer():
    """A Keras callback that adds ``epoch_seconds`` to the history of ``fit``."""
    return _epoch_timer_class()()


# ---------------------------------------------------------------------
# the worker side of a cell: data, fit, scoring


def load_cell_data(dataset: str, recipe: Recipe, seed: int, max_samples: Optional[int]):
    """The CIFAR splits of one cell through the shared loader, in the ``[0, 1]`` model domain.

    ``normalize="unit"`` makes the model input equal the pixel domain the corruptions work
    in; the train pipeline gets the ``cifar_light`` augmentation. Selection is the held-out
    slice of train that the loader cuts (``validation_split`` default); the test split only
    scores. ``max_samples`` caps the train pool and the test split (smoke runs).

    Returns:
        A ``train.common.classification_data.ClassificationData``; ``data.arrays`` holds the
        selection and test images and labels as float32 ``[0, 1]`` arrays.
    """
    from train.common.classification_data import load_classification_data
    from train.common.image_augment import build_augment
    return load_classification_data(
        dataset, batch_size=recipe.batch_size, seed=seed, normalize="unit",
        augment=build_augment("cifar_light"), max_samples=max_samples)


def fit_cell(model, data, recipe: Recipe) -> Dict[str, List[float]]:
    """Compile and train ``model`` with the shared recipe; return the history snapshot.

    Stock ``fit``, float32 policy (refused otherwise), no early stopping and no analyzer
    callbacks, so the weights left in ``model`` are the final-epoch weights. The returned
    dict is copied out of ``History`` here because Keras 3 resets it on the next predict
    or evaluate call.

    Raises:
        RuntimeError: the global dtype policy is not float32.
    """
    import keras
    policy = keras.config.dtype_policy().name
    if policy != "float32":
        raise RuntimeError(f"the harness trains in float32, the dtype policy is {policy!r}")
    model.compile(
        optimizer=build_optimizer(recipe, data.steps_per_epoch),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    history = model.fit(
        data.train_ds, epochs=recipe.epochs, steps_per_epoch=data.steps_per_epoch,
        validation_data=data.select_ds, callbacks=[epoch_timer()], verbose=2)
    return {name: [float(v) for v in values] for name, values in history.history.items()}


def selection_split(data) -> Tuple[np.ndarray, np.ndarray]:
    """``(images, labels)`` of the selection split: the slice of TRAIN the loader held out.

    The only split any ranking or choice may read; the test split only scores.
    """
    return data.arrays.x_select, data.arrays.y_select


def top1(model, x: np.ndarray, y: np.ndarray, batch_size: int = 256) -> float:
    """Top-1 accuracy of a logits model on the images ``x`` with integer labels ``y``."""
    logits = model.predict(x, batch_size=batch_size, verbose=0)
    return float(np.mean(np.argmax(logits, axis=-1) == np.asarray(y).reshape(-1)))


def physical_devices() -> List[str]:
    """The physical devices TensorFlow lists in this process, for the result row."""
    import tensorflow as tf
    return [f"{d.device_type}:{d.name}" for d in tf.config.list_physical_devices()]


# ---------------------------------------------------------------------
# strict JSON and durable writes


def _strict(value: Any) -> Any:
    """``value`` as JSON-safe Python: numpy to Python, non-finite floats to ``None``."""
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dumps(value: Any, **kwargs) -> str:
    return json.dumps(_strict(value), allow_nan=False, **kwargs)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_durable(path: Path, value: Any) -> None:
    """Write ``path`` atomically: temp file, fsync, rename, fsync of the directory."""
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(_dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)


def append_row(root: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line to ``rows.jsonl`` and fsync it (and the directory on creation).

    A torn last line from a power-off is terminated first, so the new row starts a line.
    """
    path = Path(root) / ROWS_NAME
    created = not path.exists()
    needs_newline = False
    if not created and path.stat().st_size:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    with open(path, "ab") as handle:
        if needs_newline:
            handle.write(b"\n")
        handle.write((_dumps(row) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        _fsync_dir(path.parent)


def read_rows(root: Path) -> List[Dict[str, Any]]:
    """All parseable rows of ``rows.jsonl`` in file order (``[]`` when it is missing).

    A line that is not JSON (a torn write) is skipped with a warning; its cell has no
    ``ok`` row, so it runs again.
    """
    path = Path(root) / ROWS_NAME
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"[sweep] {path}:{number} is not valid JSON, skipped")
    return rows


def write_result(cell_dir: Path, result: Mapping[str, Any]) -> None:
    """Worker side: store the cell's result where the driver looks for it."""
    _write_json_durable(Path(cell_dir) / RESULT_NAME, result)


def read_result(cell_dir: Path) -> Optional[Dict[str, Any]]:
    """Driver side: the cell's result, or ``None`` if it is missing or unreadable."""
    path = Path(cell_dir) / RESULT_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_result(
        history: Mapping[str, Sequence[float]],
        *,
        params: int,
        metrics: Optional[Mapping[str, Any]] = None,
        devices: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Worker side: the strict-JSON result of one cell.

    Args:
        history: The snapshot of ``History.history`` taken before any predict or
            evaluate call (Keras 3 resets it), with ``loss`` and, from :func:`epoch_timer`,
            ``epoch_seconds``.
        params: Parameter count of the trained model.
        metrics: Everything else the experiment reports (accuracy per condition, ...).
        devices: The physical devices TensorFlow listed in the worker.

    Returns:
        A dict with ``finite`` (False when any history value is NaN or infinite, which
        the driver turns into a failed row), ``epochs_run``, ``seconds_per_epoch`` (the
        median over epochs 2 and later, since epoch 1 carries the one-off trace) and the
        history with non-finite values as ``None``.
    """
    clean = {name: [float(v) for v in values] for name, values in history.items()}
    finite = all(np.isfinite(v) for values in clean.values() for v in values)
    seconds = clean.get("epoch_seconds", [])
    steady = seconds[1:] if len(seconds) > 1 else seconds
    return _strict({
        "params": int(params),
        "epochs_run": len(clean.get("loss", [])),
        "finite": bool(finite),
        "history": clean,
        "epoch_seconds": seconds,
        "seconds_per_epoch": float(statistics.median(steady)) if steady else None,
        "devices": list(devices) if devices is not None else None,
        "metrics": dict(metrics or {}),
    })


# ---------------------------------------------------------------------
# the idle gate


class GateTimeout(RuntimeError):
    """GPU 0 was not idle within the sweep's wait cap."""


@dataclasses.dataclass(frozen=True)
class GateResult:
    samples: List[Tuple[int, int]]
    refusals: int
    waited_s: float


def parse_smi_output(text: str) -> Optional[Tuple[int, int]]:
    """``(memory MiB, utilization %)`` from one ``nvidia-smi`` CSV line, or ``None``.

    Anything but exactly one line of two non-negative integers is unparsable (an error
    message, ``[N/A]``, several GPUs), and the gate treats unparsable as busy.
    """
    match = _SMI_LINE.match(text.strip()) if len(text.strip().splitlines()) == 1 else None
    return (int(match.group(1)), int(match.group(2))) if match else None


def sample_gpu0(run: Callable = subprocess.run) -> Optional[Tuple[int, int]]:
    """One reading of GPU 0, or ``None`` when ``nvidia-smi`` failed, hung or was unreadable.

    ``-i 0`` is mandatory: without it ``nvidia-smi`` prints a GPU 1 error to stderr, exits
    0, and the stdout of GPU 0 is mixed with it. stderr is dropped for the same reason.
    """
    try:
        done = run(NVIDIA_SMI_COMMAND, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                   text=True, timeout=NVIDIA_SMI_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if done.returncode != 0:
        return None
    return parse_smi_output(done.stdout)


def _idle(sample: Optional[Tuple[int, int]]) -> bool:
    return sample is not None and sample[0] <= GATE_MAX_MIB and sample[1] <= GATE_MAX_UTIL


def wait_for_idle(
        sampler: Callable[[], Optional[Tuple[int, int]]],
        sleeper: Callable[[float], None],
        *,
        max_wait_s: float = GATE_MAX_WAIT_S,
) -> GateResult:
    """Block until ``GATE_REQUIRED_SAMPLES`` consecutive samples are idle.

    Idle means at most ``GATE_MAX_MIB`` MiB used and ``GATE_MAX_UTIL`` % utilization. A busy
    sample, or a failed one (``None``), resets the streak and sleeps ``GATE_RETRY_S``;
    consecutive idle samples are ``GATE_GAP_S`` apart. Waiting is counted from the
    sleeps asked for, so an injected sleeper makes the cap testable.

    Raises:
        GateTimeout: the next refusal sleep would pass ``max_wait_s``.
    """
    streak: List[Tuple[int, int]] = []
    refusals = 0
    waited = 0.0
    while True:
        sample = sampler()
        if _idle(sample):
            streak.append(sample)
            if len(streak) == GATE_REQUIRED_SAMPLES:
                return GateResult(streak, refusals, waited)
            sleeper(GATE_GAP_S)
            waited += GATE_GAP_S
            continue
        streak = []
        refusals += 1
        if waited + GATE_RETRY_S > max_wait_s:
            raise GateTimeout(
                f"GPU 0 was not idle within {max_wait_s:g} s of waiting "
                f"(last reading {sample}, {refusals} refusals)")
        logger.info(f"[gate] GPU 0 busy or unreadable ({sample}), sleeping {GATE_RETRY_S:g} s")
        sleeper(GATE_RETRY_S)
        waited += GATE_RETRY_S


# ---------------------------------------------------------------------
# devices


def resolve_devices(gpu: Optional[int], cpu: bool, inherited: Optional[str]) -> str:
    """The child ``CUDA_VISIBLE_DEVICES`` value: ``"0"`` or ``""`` (CPU), nothing else.

    ``inherited`` is the launcher's own export. GPU mode refuses any value but unset or
    ``"0"``: ``train.common.gpu`` counts ``--gpu`` within an export, so ``"1"`` would
    silently select physical GPU 1, which is broken and forbidden.
    """
    if cpu:
        if gpu is not None:
            raise ValueError("--cpu and --gpu are mutually exclusive")
        return CPU_DEVICE
    if gpu not in (None, 0):
        raise ValueError(f"--gpu accepts only 0 (got {gpu}): this harness uses GPU 0 only")
    if inherited not in (None, GPU_DEVICE):
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES={inherited!r} is exported; unset it or export 0, "
            "this harness runs on GPU 0 only")
    return GPU_DEVICE


def _child_env(devices: str) -> Dict[str, str]:
    from train.common.gpu import env_with_visible_devices
    env = env_with_visible_devices(devices)
    env["MPLBACKEND"] = "Agg"
    return env


def worker_command(
        python_exe: str, worker_module: str, cell: CellSpec, cell_dir: Path) -> List[str]:
    return [python_exe, "-m", worker_module, "--worker", "--cell-dir", str(cell_dir),
            "--arm", cell.arm, "--config", cell.config, "--seed", str(cell.seed),
            *cell.extra_args]


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments a worker module accepts from :func:`run_sweep`; all are required."""
    parser.add_argument("--worker", action="store_true",
                        help="run one cell (set by the sweep driver)")
    parser.add_argument("--cell-dir", required=True, help="directory of this cell")
    parser.add_argument("--arm", required=True, help="arm of the experiment")
    parser.add_argument("--config", required=True, help="config id within the arm")
    parser.add_argument("--seed", type=int, required=True, help="seed of this cell")


def add_run_arguments(parser: argparse.ArgumentParser, default_dataset: str) -> None:
    """The run arguments of both the driver and a worker: dataset, epochs, sample cap."""
    parser.add_argument("--dataset", default=default_dataset, choices=("cifar10", "cifar100"),
                        help=f"classification dataset (default: {default_dataset})")
    parser.add_argument("--epochs", type=int, required=True,
                        help="epochs of every cell (the shared recipe, no early stopping)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="smoke cap on the train pool and the test split")


def run_extra_args(args: argparse.Namespace) -> Tuple[str, ...]:
    """The worker command-line tail that carries :func:`add_run_arguments` to a cell."""
    extra = ["--dataset", args.dataset, "--epochs", str(args.epochs)]
    if args.max_samples is not None:
        extra += ["--max-samples", str(args.max_samples)]
    return tuple(extra)


def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    """The driver arguments shared by every experiment: the root, resume, device, timeout."""
    parser.add_argument("--out-dir", help="NEW sweep root (must not exist or be empty)")
    parser.add_argument("--resume", help="resume the sweep at this existing root")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index; only 0 is accepted (default: GPU 0)")
    parser.add_argument("--cpu", action="store_true",
                        help="run the cells on CPU without the idle gate (tests only)")
    parser.add_argument("--gate-max-wait-hours", type=float, default=GATE_MAX_WAIT_S / 3600,
                        help="cumulative wait for an idle GPU 0 before the sweep raises")
    parser.add_argument("--cell-timeout-s", type=float, default=DEFAULT_CELL_TIMEOUT_S,
                        help="wall-clock limit of one cell (the orchestrator sets it from the "
                             "measured epoch time)")


# ---------------------------------------------------------------------
# the sweep


@dataclasses.dataclass(frozen=True)
class SweepResult:
    root: Path
    ok: int
    skipped: int
    failed: List[Tuple[str, str, str, int]]

    @property
    def exit_code(self) -> int:
        """Non-zero when any cell of this sweep failed."""
        return 1 if self.failed else 0


def _git_sha() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return "unknown"
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "unknown"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest(cells, recipe, devices, resolved) -> Dict[str, Any]:
    return {
        "schema": 1,
        "created_utc": _utc(),
        "cells": [c.to_dict() for c in cells],
        "recipe": recipe.to_dict(),
        "recipe_hash": recipe.recipe_hash,
        "resolved": dict(resolved or {}),
        "devices": devices,
        "mplbackend": "Agg",
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }


def _check_resume(root: Path, cells, recipe) -> None:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"cannot resume: no {MANIFEST_NAME} in {root}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["recipe_hash"] != recipe.recipe_hash:
        raise ValueError(
            f"manifest recipe hash {saved['recipe_hash']} differs from this run's "
            f"{recipe.recipe_hash}: the recipe changed, start a new sweep")

    def canonical(items):
        return sorted(json.dumps(item, sort_keys=True) for item in items)

    if canonical(saved["cells"]) != canonical(c.to_dict() for c in cells):
        raise ValueError(
            f"manifest cell list differs from this run's cells ({len(saved['cells'])} saved, "
            f"{len(cells)} given): start a new sweep")


def _next_try(root: Path, cell: CellSpec) -> int:
    prefix = f"{cell.cell_id}__try"
    tries = [int(p.name[len(prefix):]) for p in (root / CELLS_DIR).glob(prefix + "*")
             if p.name[len(prefix):].isdigit()] if (root / CELLS_DIR).is_dir() else []
    return max(tries, default=0) + 1


def _log_tail(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-ERROR_TAIL_CHARS:]
    except OSError:
        return ""


def _run_cell(cell, cell_dir, attempt, *, worker_module, python_exe, cwd, env, devices,
              timeout_s, gate) -> Dict[str, Any]:
    cmd = worker_command(python_exe, worker_module, cell, cell_dir)
    log_path = cell_dir / CELL_LOG_NAME
    started = time.monotonic()
    started_utc = _utc()
    returncode: Optional[int] = None
    error_tail: Optional[str] = None
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# cmd: {' '.join(cmd)}\n# CUDA_VISIBLE_DEVICES={devices}\n"
                  f"# MPLBACKEND={env['MPLBACKEND']}\n# timeout: {timeout_s:g}s\n"
                  f"# gate samples: {gate.samples if gate else None}\n\n")
        log.flush()
        try:
            # DECISION plan-2026-10-30T025213-f75ab4e9/D-004
            # One subprocess per cell, gated before each: do NOT make this an in-process loop (state leaks, no per-cell gate).
            done = subprocess.run(cmd, env=env, cwd=str(cwd), timeout=timeout_s,
                                  stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT)
            returncode = done.returncode
        except subprocess.TimeoutExpired:
            error_tail = f"cell timeout after {timeout_s:g}s"
    seconds = time.monotonic() - started

    result = read_result(cell_dir) if returncode == 0 else None
    if error_tail is None and returncode != 0:
        error_tail = _log_tail(log_path)
    elif error_tail is None and result is None:
        error_tail = "worker exited 0 but wrote no readable result.json"
    elif error_tail is None and not result.get("finite", False):
        error_tail = "non-finite loss in the cell history"
    return {
        "experiment": cell.experiment, "arm": cell.arm, "config": cell.config,
        "seed": int(cell.seed),
        "status": "ok" if error_tail is None else "failed",
        "attempt": attempt,
        "cell_dir": f"{CELLS_DIR}/{cell_dir.name}",
        "seconds": seconds,
        "started_utc": started_utc,
        "device": f"CUDA_VISIBLE_DEVICES={devices}",
        "gate": None if gate is None else {
            "samples": [list(s) for s in gate.samples], "refusals": gate.refusals,
            "waited_s": gate.waited_s},
        "returncode": returncode,
        "metrics": result,
        "error_tail": error_tail,
    }


def run_sweep(
        cells: Sequence[CellSpec],
        worker_module: str,
        root: Path,
        *,
        recipe: Recipe,
        devices: str,
        cell_timeout_s: float,
        resolved: Optional[Mapping[str, Any]] = None,
        resume: bool = False,
        sampler: Optional[Callable[[], Optional[Tuple[int, int]]]] = None,
        sleeper: Callable[[float], None] = time.sleep,
        max_wait_s: float = GATE_MAX_WAIT_S,
        python_exe: str = sys.executable,
        cwd: Optional[Path] = None,
) -> SweepResult:
    """Run every pending cell, one at a time, and record one row per finished cell.

    Args:
        cells: The sweep's cells; run in seed-major order.
        worker_module: Dotted module run as ``python -m`` for each cell.
        root: Sweep root. Fresh runs need it absent or empty; ``resume`` needs a manifest.
        recipe: The shared recipe; its hash is checked on resume.
        devices: ``"0"`` (GPU 0, gated before every cell) or ``""`` (CPU, no gate).
        cell_timeout_s: Wall-clock limit of one cell.
        resolved: Values the driver resolved (epochs, dataset, ...), recorded in the manifest.
        resume: Skip cells with an ``ok`` row instead of refusing a used root.
        sampler: GPU reading function; defaults to :func:`sample_gpu0` (injectable for tests).
        sleeper: Sleep function (injectable for tests).
        max_wait_s: Wait cap shared by all gates of this sweep.
        python_exe: Interpreter for the children.
        cwd: Working directory of the children; defaults to the repository root.

    Returns:
        A :class:`SweepResult`; ``exit_code`` is non-zero when a cell failed.
    """
    if devices not in (GPU_DEVICE, CPU_DEVICE):
        raise ValueError(f"devices={devices!r}: the harness runs on GPU 0 ('0') or CPU ('') only")
    ordered = seed_major(cells)
    keys = [c.key for c in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate cells in the sweep: {sorted(k for k in keys if keys.count(k) > 1)}")
    root = Path(root)
    gated = devices == GPU_DEVICE
    if gated and sampler is None:
        if shutil.which(NVIDIA_SMI_COMMAND[0]) is None:
            raise RuntimeError("nvidia-smi was not found: the idle gate cannot run")
        sampler = sample_gpu0

    if resume:
        _check_resume(root, ordered, recipe)
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} is not empty: a fresh sweep never reuses a root")
        root.mkdir(parents=True, exist_ok=True)
        _write_json_durable(root / MANIFEST_NAME, _manifest(ordered, recipe, devices, resolved))

    finished = {cell_key(row) for row in read_rows(root) if row.get("status") == "ok"}
    pending = [c for c in ordered if c.key not in finished]
    skipped = len(ordered) - len(pending)
    logger.info(f"[sweep] {root}: {len(pending)} cells to run, {skipped} already ok")

    env = _child_env(devices)
    workdir = REPO_ROOT if cwd is None else Path(cwd)
    ok = 0
    failed: List[Tuple[str, str, str, int]] = []
    waited_total = 0.0
    for cell in pending:
        gate = None
        if gated:
            gate = wait_for_idle(sampler, sleeper, max_wait_s=max_wait_s - waited_total)
            waited_total += gate.waited_s
        attempt = _next_try(root, cell)
        cell_dir = root / CELLS_DIR / f"{cell.cell_id}__try{attempt}"
        cell_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"[sweep] START {cell.cell_id} try {attempt} CUDA_VISIBLE_DEVICES={devices} "
                    f"gate={gate.samples if gate else None}")
        row = _run_cell(cell, cell_dir, attempt, worker_module=worker_module,
                        python_exe=python_exe, cwd=workdir, env=env, devices=devices,
                        timeout_s=cell_timeout_s, gate=gate)
        append_row(root, row)
        if row["status"] == "ok":
            ok += 1
            logger.info(f"[sweep] OK {cell.cell_id} in {row['seconds']:.1f}s")
        else:
            failed.append(cell.key)
            logger.error(f"[sweep] FAILED {cell.cell_id} in {row['seconds']:.1f}s: "
                         f"{(row['error_tail'] or '')[-300:]}")
    return SweepResult(root, ok, skipped, failed)


def run_driver(
        args: argparse.Namespace,
        cells: Sequence[CellSpec],
        *,
        worker_module: str,
        recipe: Recipe,
        cell_timeout_s: float,
        resolved: Optional[Mapping[str, Any]] = None,
        **sweep_kwargs: Any,
) -> int:
    """Driver front end: validate the device and root, hide GPUs from this process, sweep.

    ``args`` come from a parser that ran :func:`add_sweep_arguments`. The driver process
    holds no GPU: it hides them from itself and only the children get ``"0"``. Returns the
    process exit code (non-zero when a cell failed); extra keywords reach :func:`run_sweep`.

    Raises:
        ValueError: both or neither of ``--out-dir`` and ``--resume``, or a bad device.
    """
    if bool(args.out_dir) == bool(args.resume):
        raise ValueError("give exactly one of --out-dir (new sweep) and --resume (existing root)")
    devices = resolve_devices(args.gpu, args.cpu, os.environ.get("CUDA_VISIBLE_DEVICES"))
    from train.common.gpu import hide_gpus
    hide_gpus()
    result = run_sweep(
        cells, worker_module, Path(args.resume or args.out_dir), recipe=recipe,
        devices=devices, cell_timeout_s=cell_timeout_s, resolved=resolved,
        resume=bool(args.resume), max_wait_s=args.gate_max_wait_hours * 3600.0, **sweep_kwargs)
    logger.info(f"[sweep] done: {result.ok} ok, {result.skipped} skipped, "
                f"{len(result.failed)} failed")
    return result.exit_code
