"""Sweep driver for the embeddings study.

Enumerates the Cartesian product of the study's four axes -- model, variant,
pooling strategy, seed -- and runs each cell as its own subprocess.

Structurally a copy of :mod:`train.rms_variants_train.sweep`, which is this
repository's most-evolved sweep harness, adapted from
``(experiment, norm, mode, seed)`` to ``(model, variant, pooling, seed)``. The
properties worth preserving, each of which that harness paid for:

- **One subprocess per cell.** TensorFlow and Keras carry process-global state
  (the dtype policy, the RNG, the XLA cache, allocated device memory), so cells
  run in sequence in fresh processes rather than in one loop.
- **The cell budget is checked BEFORE anything launches.** A four-axis product
  grows fast; discovering that at cell 300 wastes hours.
- **Failures are collected, never fatal.** One bad cell must not end the sweep.
- **``MPLBACKEND`` and ``CUDA_VISIBLE_DEVICES`` are hard-set, not defaulted.**
  ``setdefault`` inherits whatever the parent shell exported, which is how a
  sweep silently lands on the wrong GPU.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .train_embeddings import resolve_output_dir

from .config import (
    BASELINE_MODEL,
    POOLING_STRATEGIES,
    VARIANTS,
    available_models,
)

# ---------------------------------------------------------------------

__all__ = ["RunSpec", "build_run_specs", "collect_results", "main", "run_one"]

TRAINER_MODULE = "train.embeddings_experimental.train_embeddings"

#: Refuse to launch a sweep larger than this without an explicit override.
DEFAULT_MAX_CELLS = 200


@dataclass(frozen=True)
class RunSpec:
    """One cell of the sweep grid.

    Frozen so a spec cannot be mutated after the grid is built and the planned
    cell count stops matching what actually runs.

    :param model: Encoder arm.
    :type model: str
    :param variant: Size ladder rung.
    :type variant: str
    :param pooling: Pooling strategy.
    :type pooling: str
    :param seed: Random seed.
    :type seed: int
    :param sweep_root: Root directory for the whole sweep.
    :type sweep_root: str
    :param extra_args: Additional trainer flags, applied to every cell.
    :type extra_args: Tuple[str, ...]
    """

    model: str
    variant: str
    pooling: str
    seed: int
    sweep_root: str
    extra_args: Tuple[str, ...] = ()

    @property
    def cell_id(self) -> str:
        """Return the cell's grid coordinates.

        :return: ``<model>/<variant>/<pooling>/seed_<n>``.
        :rtype: str
        """
        return f"{self.model}/{self.variant}/{self.pooling}/seed_{self.seed}"

    @property
    def cell_dir(self) -> str:
        """Return the cell's run directory.

        :return: ``<sweep_root>/<cell_id>``.
        :rtype: str
        """
        return os.path.join(self.sweep_root, self.cell_id)

    def command(self, python_exe: str) -> List[str]:
        """Return the argv that runs this cell.

        :param python_exe: Interpreter to run the trainer with.
        :type python_exe: str
        :return: The command.
        :rtype: list[str]
        """
        return [
            python_exe, "-m", TRAINER_MODULE,
            "--model", self.model,
            "--variant", self.variant,
            "--pooling-strategy", self.pooling,
            "--seed", str(self.seed),
            "--output-dir", self.sweep_root,
            "--experiment-name", self.cell_id,
            *self.extra_args,
        ]


def build_run_specs(
    *,
    models: Sequence[str],
    variants: Sequence[str],
    poolings: Sequence[str],
    seeds: Sequence[int],
    sweep_root: str,
    extra_args: Sequence[str] = (),
    max_cells: int = DEFAULT_MAX_CELLS,
) -> List[RunSpec]:
    """Enumerate the sweep grid.

    :param models: Encoder arms.
    :type models: Sequence[str]
    :param variants: Size ladder rungs.
    :type variants: Sequence[str]
    :param poolings: Pooling strategies.
    :type poolings: Sequence[str]
    :param seeds: Random seeds.
    :type seeds: Sequence[int]
    :param sweep_root: Root directory for the sweep.
    :type sweep_root: str
    :param extra_args: Additional trainer flags for every cell.
    :type extra_args: Sequence[str]
    :param max_cells: Refuse to build a grid larger than this.
    :type max_cells: int
    :return: One spec per cell, in a deterministic order.
    :rtype: list[RunSpec]
    :raises ValueError: If an axis is empty, an axis value is unknown, or the
        grid exceeds ``max_cells``.
    """
    unknown_models = sorted(set(models) - set(available_models()))
    if unknown_models:
        raise ValueError(
            f"unknown models {unknown_models}; available {available_models()}"
        )
    for name, axis in (
        ("models", models), ("variants", variants),
        ("poolings", poolings), ("seeds", seeds),
    ):
        if not axis:
            raise ValueError(f"axis {name!r} is empty; a sweep needs every axis")

    planned = len(models) * len(variants) * len(poolings) * len(seeds)
    if planned > max_cells:
        raise ValueError(
            f"this grid is {planned} cells, over the {max_cells}-cell budget. "
            "Narrow an axis or raise --max-cells deliberately; a four-axis "
            "product grows faster than it looks."
        )

    return [
        RunSpec(
            model=model, variant=variant, pooling=pooling, seed=seed,
            sweep_root=sweep_root, extra_args=tuple(extra_args),
        )
        for model, variant, pooling, seed in itertools.product(
            models, variants, poolings, seeds
        )
    ]


def run_one(
    spec: RunSpec,
    *,
    python_exe: str,
    gpu_id: int,
    cell_timeout_s: float,
    deadline_s: Optional[float] = None,
) -> Tuple[bool, str]:
    """Run one cell in its own process.

    :param spec: The cell to run.
    :type spec: RunSpec
    :param python_exe: Interpreter to run the trainer with.
    :type python_exe: str
    :param gpu_id: GPU index; hard-set, never inherited.
    :type gpu_id: int
    :param cell_timeout_s: Per-cell wall-clock limit.
    :type cell_timeout_s: float
    :param deadline_s: Optional absolute deadline for the whole sweep.
    :type deadline_s: float | None
    :return: ``(succeeded, stderr_tail)``.
    :rtype: tuple[bool, str]
    """
    os.makedirs(spec.cell_dir, exist_ok=True)
    log_path = os.path.join(spec.cell_dir, "cell.log")

    env = dict(os.environ)
    # Hard-set, not setdefault: inheriting these is how a sweep silently lands
    # on the wrong GPU or opens an X11 connection on a headless host.
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    timeout = cell_timeout_s
    if deadline_s is not None:
        timeout = min(timeout, max(1.0, deadline_s - time.time()))

    command = spec.command(python_exe)
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = completed.returncode == 0
        stderr_tail = completed.stderr[-2000:]
        stdout = completed.stdout
    except subprocess.TimeoutExpired:
        ok, stderr_tail, stdout = False, f"TIMEOUT after {timeout:.0f}s", ""

    with open(log_path, "w") as handle:
        handle.write(f"# cmd: {' '.join(command)}\n")
        handle.write(f"# gpu: {gpu_id}\n")
        handle.write(f"# elapsed: {time.time() - started:.1f}s\n\n")
        handle.write(stdout)
        handle.write("\n--- stderr ---\n")
        handle.write(stderr_tail)

    return ok, stderr_tail


def collect_results(sweep_root: str) -> List[Dict[str, Any]]:
    """Gather every cell's ``results.json`` into a list of flat records.

    :param sweep_root: Root directory of the sweep.
    :type sweep_root: str
    :return: One record per completed cell.
    :rtype: list[dict[str, Any]]
    """
    records: List[Dict[str, Any]] = []
    for path in sorted(Path(sweep_root).rglob("results.json")):
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"skipping unreadable {path}: {exc}")
            continue

        config = payload.get("config", {})
        record: Dict[str, Any] = {
            "model": config.get("model"),
            "variant": config.get("variant"),
            "pooling": config.get("pooling_strategy"),
            "seed": config.get("seed"),
            "parameters": payload.get("parameters"),
            "run_dir": payload.get("run_dir", str(path.parent)),
        }
        for stage in ("mlm", "contrastive"):
            history = payload.get(stage) or {}
            for metric, values in history.items():
                if values:
                    record[f"{stage}_{metric}_final"] = float(values[-1])
                    record[f"{stage}_{metric}_best"] = float(min(values))
        records.append(record)
    return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse sweep arguments.

    :param argv: Argument vector; ``None`` uses ``sys.argv[1:]``.
    :type argv: Sequence[str] | None
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Run the embeddings_experimental study sweep."
    )
    parser.add_argument("--models", nargs="+", default=available_models())
    parser.add_argument("--variants", nargs="+", default=["tiny"])
    parser.add_argument("--pooling", nargs="+", default=list(POOLING_STRATEGIES))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
        help=(
            "Six by default, and that is a floor rather than a taste: the "
            "report's two-sided sign-flip paired test cannot reach p < 0.05 "
            "with five pairs or fewer, for any effect size."
        ),
    )
    parser.add_argument(
        "--sweep-root", type=str, default="results/embeddings_study",
        help=(
            "Root directory; every cell writes under it. A relative path "
            "resolves against the REPO ROOT, not the working directory, so a "
            "sweep launched from src/ does not create src/results/."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS)
    parser.add_argument(
        "--cell-timeout-s", type=float, default=7200.0,
        help="Per-cell wall-clock limit.",
    )
    parser.add_argument(
        "--global-timeout-s", type=float, default=None,
        help="Optional wall-clock limit for the whole sweep.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the grid and exit without launching anything.",
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument(
        "--trainer-arg", dest="extra_args", action="append", default=[],
        help="Extra trainer flag applied to every cell; repeatable.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    :param argv: Argument vector; ``None`` uses ``sys.argv[1:]``.
    :type argv: Sequence[str] | None
    :return: Process exit code; non-zero if any cell failed.
    :rtype: int
    """
    args = parse_args(argv)

    args.sweep_root = resolve_output_dir(args.sweep_root)

    try:
        specs = build_run_specs(
            models=args.models,
            variants=args.variants,
            poolings=args.pooling,
            seeds=args.seeds,
            sweep_root=args.sweep_root,
            extra_args=args.extra_args,
            max_cells=args.max_cells,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    logger.info(f"Sweep grid: {len(specs)} cells -> {args.sweep_root}")
    for spec in specs:
        logger.info(f"  {spec.cell_id}")
    if args.dry_run:
        return 0

    os.makedirs(args.sweep_root, exist_ok=True)
    deadline = (
        time.time() + args.global_timeout_s if args.global_timeout_s else None
    )

    failures: List[Tuple[str, str]] = []
    for index, spec in enumerate(specs, start=1):
        if deadline is not None and time.time() >= deadline:
            logger.warning("global timeout reached; skipping the remainder")
            break
        logger.info(f"[{index}/{len(specs)}] {spec.cell_id}")
        ok, stderr_tail = run_one(
            spec,
            python_exe=args.python_exe,
            gpu_id=args.gpu,
            cell_timeout_s=args.cell_timeout_s,
            deadline_s=deadline,
        )
        if not ok:
            logger.error(f"cell FAILED: {spec.cell_id}")
            failures.append((spec.cell_id, stderr_tail))

    records = collect_results(args.sweep_root)
    summary_path = os.path.join(args.sweep_root, "all_runs.json")
    with open(summary_path, "w") as handle:
        json.dump(records, handle, indent=2)
    logger.info(f"Collected {len(records)} cells -> {summary_path}")

    if failures:
        failures_path = os.path.join(args.sweep_root, "failures.log")
        with open(failures_path, "w") as handle:
            for cell_id, tail in failures:
                handle.write(f"=== {cell_id} ===\n{tail}\n\n")
        logger.error(f"{len(failures)} cell(s) failed; see {failures_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
