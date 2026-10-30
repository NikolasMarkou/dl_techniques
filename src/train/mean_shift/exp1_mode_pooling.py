"""Experiment 1: Mode Pooling against mean and CLS pooling on vit_pico, clean and corrupted.

Four arms train the same ``vit_pico`` backbone (patch size 4, 64 patches, sequence 65, CLS
at position 0) and differ only in the pooling head that turns the ``[B, 65, D]`` sequence
into one vector before a ``Dense`` layer: ``mean`` (patches only), ``cls``, ``mode`` (Mode
Pooling, one start) and ``mode_s4`` (four starts). Every arm has 2,710,948 parameters at 100
classes. Each cell trains with the shared recipe (``train.mean_shift.common.Recipe``) and
scores its FINAL-epoch weights on the test split under six conditions (``clean``, ``occ25``,
``occ50``, ``noise0.05``, ``noise0.1``, ``noise0.2``); the corruptions are keyed by
``(seed, condition)`` and never by the arm, so all arms see the same images.

Two modes in one module, chosen by ``--worker``:

* driver (default): builds the 4 arms x seeds cells and runs them one subprocess each,
  gated on an idle GPU 0 (``common.run_sweep``)::

      MPLBACKEND=Agg .venv/bin/python -m train.mean_shift.exp1_mode_pooling \\
          --out-dir results/mean_shift/exp1_cifar10 --epochs 30 --dataset cifar10

* worker (started by the driver, one cell)::

      python -m train.mean_shift.exp1_mode_pooling --worker --cell-dir D --arm mode \\
          --config cifar10 --seed 0 --epochs 30

The cell ``config`` is the dataset name. Results land under ``--out-dir`` only; run from the
repository root so a relative path resolves to the repo-root ``results/``.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from dl_techniques.utils.logger import logger
from train.mean_shift import common, corruption

EXPERIMENT = "exp1"
WORKER_MODULE = "train.mean_shift.exp1_mode_pooling"
DEFAULT_SEEDS = (0, 1, 2)
PATCH_SIZE = 4


def pooling_layer(arm: str):
    """The pooling head of ``arm``, applied to the ``[B, 65, D]`` sequence (CLS at 0).

    ``mean`` and both mode arms drop position 0 from the pooled set; ``cls`` reads it.
    """
    from dl_techniques.layers.sequence_pooling import ModePooling, SequencePooling
    if arm == "mean":
        return SequencePooling(strategy="mean", exclude_positions=[0], name="pool")
    if arm == "cls":
        return SequencePooling(strategy="cls", name="pool")
    if arm in ("mode", "mode_s4"):
        return ModePooling(num_steps=5, bandwidth="auto", exclude_positions=[0],
                           num_starts=4 if arm == "mode_s4" else 1, name="pool")
    raise KeyError(f"unknown exp1 arm {arm!r}")


#: The cell ``arm`` values, in the order the driver lists them.
ARMS: Tuple[str, ...] = ("mean", "cls", "mode", "mode_s4")


def build_model(arm: str, num_classes: int):
    """``vit_pico`` backbone, the pooling head of ``arm``, then a logits ``Dense``."""
    import keras
    from dl_techniques.models.vision.vit.model import create_vit
    # DECISION plan-2026-10-30T025213-f75ab4e9/D-005
    # patch_size=4 explicitly: the vit_pico default 16 trains silently on 4 tokens at 32x32; do NOT drop it.
    backbone = create_vit(
        "vit_pico", num_classes=num_classes, input_shape=(32, 32, 3), patch_size=PATCH_SIZE,
        include_top=False, pooling=None, dropout_rate=0.1, name="backbone")
    inputs = keras.Input((32, 32, 3))
    pooled = pooling_layer(arm)(backbone(inputs))
    return keras.Model(inputs, keras.layers.Dense(num_classes, name="logits")(pooled),
                       name=f"exp1_{arm}")


def parse_worker_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1 worker: train and score one cell.")
    common.add_worker_arguments(parser)
    common.add_run_arguments(parser, default_dataset="cifar10")
    return parser.parse_args(argv)


def parse_driver_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1 driver: Mode Pooling against mean and CLS pooling on vit_pico. "
            "Runs one gated subprocess per (arm, seed) cell; the cells call this module "
            "with --worker."),
        epilog="GPU 0 only, and only while idle (common.run_sweep); --cpu is for tests.")
    common.add_sweep_arguments(parser)
    common.add_run_arguments(parser, default_dataset="cifar10")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                        help="arms to run (default: all four)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="seeds of the sweep, one cell per arm and seed")
    return parser.parse_args(argv)


def build_cells(args: argparse.Namespace) -> List[common.CellSpec]:
    """One cell per (arm, seed), in the order arm-fastest so seeds come out major."""
    extra = common.run_extra_args(args)
    return [common.CellSpec(EXPERIMENT, arm, args.dataset, seed, extra)
            for seed in args.seeds for arm in args.arms]


def corrupted_test_sets(x_test: np.ndarray, seed: int) -> Iterator[Tuple[str, np.ndarray]]:
    """``(condition, images)`` for every condition, one at a time; a function of ``(x_test, seed)`` only."""
    for name in corruption.CONDITIONS:
        yield name, corruption.apply_condition(x_test, name, seed)


def run_worker(args: argparse.Namespace) -> int:
    """Train one cell and write its ``result.json``; exceptions propagate (exit non-zero)."""
    from train.common import set_seeds, setup_gpu
    setup_gpu(None)
    set_seeds(args.seed)
    recipe = common.Recipe.for_epochs(args.epochs)
    data = common.load_cell_data(args.dataset, recipe, args.seed, args.max_samples)
    model = build_model(args.arm, data.num_classes)
    logger.info(f"[exp1] {args.arm} seed {args.seed}: {model.count_params():,} parameters, "
                f"{recipe.epochs} epochs")
    # DECISION plan-2026-10-30T025213-f75ab4e9/D-005
    # The weights scored below are the FINAL epoch's: do NOT restore a best epoch picked on any split.
    history = common.fit_cell(model, data, recipe)

    arrays = data.arrays
    accuracy = {name: common.top1(model, images, arrays.y_test)
                for name, images in corrupted_test_sets(arrays.x_test, args.seed)}
    x_select, y_select = common.selection_split(data)
    metrics = {
        "experiment": EXPERIMENT, "arm": args.arm, "dataset": args.dataset,
        "epochs": args.epochs, "max_samples": args.max_samples, "seed": args.seed,
        "accuracy": accuracy,
        "select_accuracy": common.top1(model, x_select, y_select),
        "n_select": int(len(x_select)), "n_test": int(len(arrays.x_test)),
        "dtype_policy": "float32", "patch_size": PATCH_SIZE,
    }
    result = common.build_result(
        history, params=model.count_params(), metrics=metrics, devices=common.physical_devices())
    common.write_result(Path(args.cell_dir), result)
    logger.info(f"[exp1] {args.arm} seed {args.seed}: clean {accuracy['clean']:.4f}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Worker when ``--worker`` is given, otherwise the sweep driver; returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return run_worker(parse_worker_arguments(argv))
    args = parse_driver_arguments(argv)
    return common.run_driver(
        args, build_cells(args), worker_module=WORKER_MODULE,
        recipe=common.Recipe.for_epochs(args.epochs), cell_timeout_s=args.cell_timeout_s,
        resolved={"experiment": EXPERIMENT, "dataset": args.dataset, "epochs": args.epochs,
                  "max_samples": args.max_samples, "arms": list(args.arms),
                  "seeds": list(args.seeds)})


if __name__ == "__main__":
    sys.exit(main())
