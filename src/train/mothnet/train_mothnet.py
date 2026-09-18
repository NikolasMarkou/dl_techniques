"""
MothNet Training Script
====================================================================

Trains `MothNet` (`dl_techniques.models.general_purpose.mothnet.model`) on a small
MNIST subsample via its hand-rolled `train_hebbian()` method — never `model.fit()`,
since `HebbianReadoutLayer.readout_weights` is `trainable=False` and is updated only
by a direct Hebbian `.assign()` inside `train_hebbian` (see
`plans/plan-2026-09-18T045308-c89cdf76/decisions.md` D-001/D-002).

This module currently wires CLI argument parsing (plan Step 1), MNIST data loading
(plan Step 2), model construction (plan Step 3), the run directory / config.json
preamble (plan Step 4), and the core Hebbian training loop with its ordering-
disciplined checkpoint/CSV writes (plan Step 5). Visualization and final-model/
history-JSON finalization land in later plan steps.

Usage:
    python -m train.mothnet.train_mothnet --help
    python -m train.mothnet.train_mothnet --epochs 3 --num-train-samples 2000 \
        --num-val-samples 500
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import keras
import numpy as np
import matplotlib
matplotlib.use("Agg")

from dl_techniques.utils.logger import logger
from dl_techniques.models.general_purpose.mothnet.model import MothNet
from train.common import default_experiment_name, prepare_run_dir
from train.common.callbacks import best_checkpoint_path


# `parents[3]` reaches the repo root from THIS file
# (src/train/mothnet/train_mothnet.py: [0] mothnet, [1] train, [2] src, [3] <repo>),
# matching `src/train/kan/train_kan.py`'s own `REPO_ROOT` derivation exactly.
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "mothnet"


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for the MothNet trainer.

    Built as a fresh, purpose-built `argparse.ArgumentParser` rather than
    `create_base_argument_parser()` — MothNet has no optimizer/LR-schedule/patience/
    dataset-choice surface for that shared parser to usefully cover
    (`decisions.md` D-003).

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Parsed arguments namespace.
    """
    # DECISION plan-2026-09-18T045308-c89cdf76/D-003: fresh ArgumentParser, not
    # create_base_argument_parser(). That shared parser's surface (--dataset,
    # --image-size, --weight-decay, --lr-schedule, --patience) is entirely dead for
    # MothNet (no optimizer, no LR schedule, no early stopping, no image resize) —
    # do not "fix" this by reintroducing the shared parser + a pruning helper; see
    # decisions.md D-003 for the full trade-off.
    parser = argparse.ArgumentParser(
        description=(
            "Train MothNet on a small MNIST subsample via its hand-rolled "
            "train_hebbian() method."
        ),
    )
    parser.add_argument(
        "--mb-units", type=int, default=2000,
        help="Number of Mushroom Body units (default: 2000, matches MothNet's own default).",
    )
    parser.add_argument(
        "--mb-sparsity", type=float, default=0.1,
        help="Fraction of MB units allowed to fire per sample (default: 0.1).",
    )
    parser.add_argument(
        "--al-units", type=int, default=None,
        help=(
            "Number of Antennal Lobe units (default: None, meaning MothNet infers it "
            "from the input dimension)."
        ),
    )
    parser.add_argument(
        "--connection-sparsity", type=float, default=0.1,
        help="Mushroom Body projection connection sparsity (default: 0.1).",
    )
    parser.add_argument(
        "--hebbian-learning-rate", type=float, default=0.01,
        help="Learning rate used by the Hebbian readout update (default: 0.01).",
    )
    parser.add_argument(
        "--inhibition-strength", type=float, default=0.5,
        help="Antennal Lobe inhibition strength (default: 0.5).",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help=(
            "Number of training epochs; each epoch is one "
            "train_hebbian(epochs=1) call (default: 3, must be >= 1)."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size passed to train_hebbian (default: 32).",
    )
    parser.add_argument(
        "--viz-freq", type=int, default=2,
        help=(
            "Render a periodic MB-sparsity visualization every N epochs; "
            "<= 0 disables periodic visualization (default: 2)."
        ),
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to use (default: 0).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=(
            "Output directory for run artifacts (default: None, meaning "
            "repo-root results/, resolved later)."
        ),
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help=(
            "Experiment name (default: None, meaning auto-generated via "
            "default_experiment_name)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for NumPy's global RNG (default: 42).",
    )
    parser.add_argument(
        "--num-train-samples", type=int, default=2000,
        help="Number of MNIST training samples to subsample (default: 2000).",
    )
    parser.add_argument(
        "--num-val-samples", type=int, default=500,
        help="Number of MNIST validation samples to subsample (default: 500).",
    )

    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error(f"--epochs must be >= 1, got {args.epochs}")

    return args


def load_mnist_data(config: argparse.Namespace):
    """Load, normalize, flatten, and one-hot-encode MNIST, then subsample.

    Mirrors `src/train/som_nd_soft/train_mnist.py`'s `load_mnist_data` normalization
    pipeline exactly (`astype("float32")/255.0` -> `reshape(-1, 784)` ->
    `keras.utils.to_categorical(y, 10)`). The training subsample is drawn from MNIST's
    own train split; the validation subsample is drawn from MNIST's own held-out test
    split (never a further split of the training pool) — matching `train_kan.py`'s
    train/val split pattern.

    Both subsamples are drawn via a `config.seed`-seeded shuffle, applied BEFORE
    `train_hebbian`'s own internal unseeded shuffle ever runs (that internal shuffle is
    out of this function's scope — see plan.md Step 2).

    :param config: Parsed CLI namespace (`parse_arguments`'s return value); reads
        `config.num_train_samples`, `config.num_val_samples`, `config.seed`.
    :return: `(x_train, y_train), (x_val, y_val)` as numpy arrays — `x_*` shape
        `(N, 784)` float32 in `[0, 1]`, `y_*` shape `(N, 10)` one-hot float32.
    """
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    num_train_samples = config.num_train_samples
    if num_train_samples > x_train.shape[0]:
        logger.warning(
            f"--num-train-samples {num_train_samples} exceeds the available MNIST "
            f"train set size ({x_train.shape[0]}); clipping to {x_train.shape[0]}."
        )
        num_train_samples = x_train.shape[0]

    num_val_samples = config.num_val_samples
    if num_val_samples > x_test.shape[0]:
        logger.warning(
            f"--num-val-samples {num_val_samples} exceeds the available MNIST "
            f"test set size ({x_test.shape[0]}); clipping to {x_test.shape[0]}."
        )
        num_val_samples = x_test.shape[0]

    # Seeded shuffle for the subsample selection — independent of, and applied before,
    # train_hebbian's own internal unseeded shuffle (see docstring above and
    # plan.md Step 2's scope note).
    rng = np.random.default_rng(config.seed)
    train_indices = rng.permutation(x_train.shape[0])[:num_train_samples]
    val_indices = rng.permutation(x_test.shape[0])[:num_val_samples]

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_val = x_test[val_indices]
    y_val = y_test[val_indices]

    logger.info(
        f"Subsampled MNIST — train: x={x_train.shape}, y={y_train.shape}; "
        f"val: x={x_val.shape}, y={y_val.shape}"
    )
    return (x_train, y_train), (x_val, y_val)


def build_model(args: argparse.Namespace, input_dim: int) -> MothNet:
    """Construct and explicitly build a `MothNet` model.

    Seeds NumPy's global RNG once, before construction, so both any model-init
    randomness and `train_hebbian`'s own internal unseeded shuffle
    (`model.py:308`, no `seed=` kwarg exists there) become reproducible across runs
    for a fixed `--seed`. This is process-global `np.random.seed`, deliberately
    separate from `load_mnist_data`'s own local `np.random.default_rng(config.seed)`
    generator used for subsampling — both consume the same `args.seed` value, they
    just use different RNG mechanisms (`decisions.md` D-009).

    The model is explicitly `.build()`-built here, before the training loop ever
    starts, rather than left to `train_hebbian`'s own `if not self.built:` self-build
    guard (`model.py:295-299`, pinned by `# DECISION
    plan-2026-08-17T183311-79c63e38/D-017`). Calling `build()` first makes that guard
    a no-op on the model's first `train_hebbian` call, which is consistent with (not
    a violation of) D-017's "don't rebuild and destroy learned weights" intent, since
    no training has happened yet at build time. See `decisions.md` D-009.

    :param args: Parsed CLI namespace (`parse_arguments`'s return value); reads
        `args.al_units`, `args.mb_units`, `args.mb_sparsity`,
        `args.connection_sparsity`, `args.hebbian_learning_rate`,
        `args.inhibition_strength`, `args.seed`.
    :param input_dim: Flattened input feature dimension (784 for MNIST).
    :return: A built `MothNet` instance (`model.built is True`).
    """
    # DECISION plan-2026-09-18T045308-c89cdf76/D-009: explicit pre-loop build call.
    # Do not remove this and rely solely on train_hebbian's own self-build guard —
    # an --epochs 0 run (blocked at CLI-parse time, but also any early crash before
    # the first train_hebbian call) would otherwise leave an unbuilt, unsavable
    # model. See decisions.md D-009 for the full trade-off.
    np.random.seed(args.seed)

    model = MothNet(
        num_classes=10,
        al_units=args.al_units,
        mb_units=args.mb_units,
        mb_sparsity=args.mb_sparsity,
        connection_sparsity=args.connection_sparsity,
        hebbian_learning_rate=args.hebbian_learning_rate,
        inhibition_strength=args.inhibition_strength,
    )
    model.build((None, input_dim))

    resolved_al_units = model.antennal_lobe.units
    logger.info(
        f"Built MothNet — input_dim={input_dim}, al_units={resolved_al_units} "
        f"(requested: {args.al_units}), mb_units={args.mb_units}, "
        f"mb_sparsity={args.mb_sparsity}, connection_sparsity={args.connection_sparsity}, "
        f"hebbian_learning_rate={args.hebbian_learning_rate}, "
        f"inhibition_strength={args.inhibition_strength}, "
        f"params={model.count_params()}."
    )
    return model


def main(argv=None) -> int:
    """Entry point for the MothNet trainer.

    The first statement parses argv, so ``--help`` exits 0 with a ``usage:`` line
    and no GPU/dataset/model work happens before parsing returns.

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Process exit code.
    """
    args = parse_arguments(argv)

    logger.info(f"Parsed args: {args}")

    # Unified run directory (matches `train_kan.py`'s own preamble): a single
    # results/<run>/ directory, resolved once here, that `prepare_run_dir` both
    # creates and writes config.json into. `--output-dir` overrides the
    # repo-root `results/` base; `--experiment-name` overrides the timestamped
    # default leaf name. This step only wires the directory + config.json —
    # the training loop itself lands in Step 5.
    base_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results"
    run_dir = base_dir / (args.experiment_name or default_experiment_name(EXPERIMENT_NAME))
    prepare_run_dir(args, output_dir=run_dir)
    logger.info(f"Run directory: {run_dir}")

    (x_train, y_train), (x_val, y_val) = load_mnist_data(args)
    model = build_model(args, input_dim=x_train.shape[1])
    logger.info(f"Model ready — model.built={model.built}, params={model.count_params()}")

    # `history` mirrors Keras' own `History.history` shape (a dict of equal-length
    # lists) so Step 6 (visualization) and Step 7 (`training_history.json`) can
    # consume it unmodified — this step is the ONLY place these values are computed,
    # so it owns creating and populating the dict even though nothing downstream
    # reads it yet.
    history: Dict[str, List[float]] = {
        "epoch": [], "loss": [], "train_accuracy": [], "val_accuracy": [],
    }
    best_val_accuracy = -1.0
    csv_path = run_dir / "training_log.csv"

    # Opened once, before the loop, in append mode with the header written here and
    # every row flushed immediately after it's written (never batched) — a crash
    # after N completed epochs must leave exactly N complete CSV rows, per plan.md's
    # edge-case requirement.
    with open(csv_path, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "loss", "train_accuracy", "val_accuracy"])
        csv_file.flush()

        for epoch in range(args.epochs):
            # DECISION plan-2026-09-18T045308-c89cdf76/D-005
            # ORDERING DISCIPLINE for this loop body — do not reorder without
            # re-reading decisions.md D-005 first.
            #
            # (a) train_hebbian(epochs=1, ...) below is the ONE weight-mutating
            #     step this epoch — the only writer of `readout.readout_weights`
            #     (findings/mothnet-architecture.md finding 4's mutation-kill test
            #     confirms no other call site touches it).
            # (b) val_accuracy/train_accuracy are computed strictly AFTER (a)
            #     returns, never before — a genuine post-update read.
            # (c) the checkpoint-save / CSV-row / history-append decision further
            #     below uses that same post-update measurement.
            #
            # Why this is safe (and why it differs from train_kan.py's D-006
            # hazard): `train_kan.py` shipped a bug where a weight-mutating Keras
            # callback (`KANGridUpdateCallback`) ran BEFORE `ModelCheckpoint`/
            # `CSVLogger` in an `on_epoch_end` callback list, so the checkpoint
            # saved POST-mutation weights against a `val_loss` Keras had already
            # measured PRE-mutation — a silent mismatch caught only by adversarial
            # review (decisions.md D-006, that plan). This loop has no callback
            # list and no second, later mutator: `train_hebbian` is the first
            # statement in the epoch body, and everything after it only reads.
            # That eliminates the D-006 hazard CLASS here — but only
            # CONDITIONALLY, not absolutely: this reasoning holds PROVIDED the
            # metric read in (b) always executes strictly after (a) returns,
            # never reordered ahead of it or interleaved with it. Do not move the
            # accuracy computation above the `train_hebbian` call, and do not add
            # a second weight-mutating call anywhere in this loop without
            # re-deriving this comment.
            epoch_history = model.train_hebbian(
                x_train, y_train, epochs=1, batch_size=args.batch_size, verbose=0,
            )
            loss = float(epoch_history["loss"][0])

            val_logits = keras.ops.convert_to_numpy(model.extract_features(x_val))
            val_accuracy = float(
                np.mean(np.argmax(val_logits, axis=-1) == np.argmax(y_val, axis=-1))
            )
            train_logits = keras.ops.convert_to_numpy(model.extract_features(x_train))
            train_accuracy = float(
                np.mean(np.argmax(train_logits, axis=-1) == np.argmax(y_train, axis=-1))
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model.save(best_checkpoint_path(str(run_dir)))
                logger.info(
                    f"Epoch {epoch}: new best val_accuracy={val_accuracy:.4f} "
                    f"— saved {best_checkpoint_path(str(run_dir))}"
                )

            csv_writer.writerow([epoch, loss, train_accuracy, val_accuracy])
            csv_file.flush()

            history["epoch"].append(epoch)
            history["loss"].append(loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_accuracy"].append(val_accuracy)

            logger.info(
                f"Epoch {epoch}/{args.epochs - 1} — loss={loss:.4f}, "
                f"train_accuracy={train_accuracy:.4f}, val_accuracy={val_accuracy:.4f}"
            )

    return 0


if __name__ == "__main__":
    main()
