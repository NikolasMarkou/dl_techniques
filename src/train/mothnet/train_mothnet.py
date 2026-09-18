"""
MothNet Training Script
====================================================================

Trains `MothNet` (`dl_techniques.models.general_purpose.mothnet.model`) on a small
MNIST subsample via its hand-rolled `train_hebbian()` method — never `model.fit()`,
since `HebbianReadoutLayer.readout_weights` is `trainable=False` and is updated only
by a direct Hebbian `.assign()` inside `train_hebbian` (see
`plans/plan-2026-09-18T045308-c89cdf76/decisions.md` D-001/D-002).

This module provides CLI argument parsing, seeded MNIST data loading/subsampling,
model construction with an explicit pre-loop build, a unified `results/<run>/` output
directory (`config.json`, `best_model.keras`, `final_model.keras`, `training_log.csv`,
`training_history.json`, `visualizations/`), the core Hebbian training loop with its
ordering-disciplined checkpoint/CSV writes, and per-epoch/periodic visualization
rendering (`render_training_dashboard`, `render_mb_sparsity`) — the same output shape
as `src/train/bfunet/` and `src/train/kan/train_kan.py`, hand-assembled around
`train_hebbian()`'s loop rather than a Keras `Callback`/`fit()` event loop.

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
import matplotlib.pyplot as plt

from dl_techniques.utils.logger import logger
from dl_techniques.models.general_purpose.mothnet.model import MothNet
from train.common import (
    default_experiment_name, prepare_run_dir, save_training_history_json, set_seeds,
    setup_gpu,
)
from train.common.callbacks import best_checkpoint_path


# `parents[3]` reaches the repo root from THIS file
# (src/train/mothnet/train_mothnet.py: [0] mothnet, [1] train, [2] src, [3] <repo>),
# matching `src/train/kan/train_kan.py`'s own `REPO_ROOT` derivation exactly.
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "mothnet"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the MothNet trainer's `argparse.ArgumentParser`, unparsed.

    Split out of `parse_arguments()` (Step 2,
    `plan-2026-09-18T060057-c1cfc3d3`) so `tests/test_train/test_mothnet/
    test_cli_contract.py` can drive the REAL parser directly (`_cli_contract.py`'s
    `Contract.build_parser` needs the raw `ArgumentParser`, not a parsed
    `Namespace`) without duplicating its 16-flag definition. Pure extraction —
    `parse_arguments()` below still does exactly what it did before, just by
    calling this helper first. No behavior change.

    Built as a fresh, purpose-built `argparse.ArgumentParser` rather than
    `create_base_argument_parser()` — MothNet has no optimizer/LR-schedule/patience/
    dataset-choice surface for that shared parser to usefully cover
    (`decisions.md` D-003).

    :return: An unparsed `argparse.ArgumentParser` with all 16 MothNet flags.
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
    # DECISION plan-2026-09-18T060057-c1cfc3d3/D-009
    # 16000 is NOT the model's own constructor default (that was 2000) and NOT an
    # arbitrarily-larger round number picked to "match the README range" — it is the
    # value the pre-registered decision rule in decisions.md D-009 committed to before the
    # data existed. Do NOT lower this back toward 2000 without re-running
    # multiseed_sweep.py; a single anecdotal run is exactly the n=1 evidence this rule
    # replaced (see D-009's Context: a prior n=1 claim was refuted by same-config noise).
    parser.add_argument(
        "--mb-units", type=int, default=16000,
        help=(
            "Number of Mushroom Body units (default: 16000, raised from 2000 per the "
            "8-seed paired mb_units sweep in decisions.md D-009: p=0.0086, "
            "mean_diff(val_accuracy)=0.1633, n_pairs=8)."
        ),
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
        "--num-train-samples", type=int, default=60000,
        help=(
            "Number of MNIST training samples to subsample (default: 60000, the "
            "full MNIST train split size). Previously defaulted to 2000 as a "
            "fast-iteration toy subsample; pass a smaller value explicitly to "
            "restore that toy-scale behavior."
        ),
    )
    parser.add_argument(
        "--num-val-samples", type=int, default=10000,
        help=(
            "Number of MNIST validation samples to subsample (default: 10000, "
            "the full MNIST test split size). Previously defaulted to 500 as a "
            "fast-iteration toy subsample; pass a smaller value explicitly to "
            "restore that toy-scale behavior."
        ),
    )
    # DECISION plan-2026-09-18T080513-debe8b11/D-005: a dedicated eval-chunking flag,
    # NOT a reuse of --batch-size. --batch-size is train_hebbian's own training
    # mini-batch — an algorithm-affecting parameter (the Hebbian update is computed
    # per mini-batch). Reusing it here would silently couple two independent
    # concerns: a user raising --batch-size for training-speed reasons would also,
    # as an unintended side effect, change eval-chunk memory pressure. Do not merge
    # these two flags without re-reading decisions.md D-005.
    parser.add_argument(
        "--eval-batch-size", type=int, default=5000,
        help=(
            "Chunk size for the per-epoch accuracy computation's forward pass "
            "(default: 5000). Decoupled from --batch-size, which governs "
            "train_hebbian's own training mini-batch and IS algorithm-affecting "
            "(it changes what the Hebbian update computes); --eval-batch-size is a "
            "pure eval-chunking performance/memory knob with no effect on the "
            "computed accuracy value."
        ),
    )

    return parser


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for the MothNet trainer.

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Parsed arguments namespace.
    """
    parser = _build_parser()
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

    Seeds all RNGs (`train.common.set_seeds`) once, before construction, so both
    Keras/TF-backed weight init and `train_hebbian`'s own internal unseeded shuffle
    (`model.py:308`, no `seed=` kwarg exists there) become reproducible across
    SEPARATE PROCESS invocations for a fixed `--seed` (`decisions.md` D-001).
    Deliberately separate from `load_mnist_data`'s own local
    `np.random.default_rng(config.seed)` generator used for subsampling — both
    consume the same `args.seed` value, they just use different RNG mechanisms
    (`decisions.md` D-009); `set_seeds()` also calls `np.random.seed(seed)`
    internally, so no conflict arises between the two.

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
    #
    # DECISION plan-2026-09-18T060057-c1cfc3d3/D-001: seeding now routes through
    # train.common.set_seeds() (was: np.random.seed(args.seed) alone), called here
    # in the SAME position — strictly before MothNet(...) construction below.
    # MEASURED: Keras' RandomInitializer (base of GlorotUniform, the default
    # kernel_initializer for AntennalLobeLayer/HebbianReadoutLayer) draws its
    # per-instance seed from Python's stdlib `random` module at object-CONSTRUCTION
    # time, not from `np.random` — so np.random.seed alone never made weight init
    # reproducible across separate process invocations. set_seeds() also calls
    # np.random.seed(seed) internally, so D-009's original "seeds train_hebbian's own
    # unseeded shuffle" rationale stays covered unchanged; this is a strict widening
    # of what gets seeded, not a narrowing. D-009's own text above is NOT edited (its
    # "two-separate-RNG-mechanism" framing is now partially superseded — see
    # decisions.md D-001 for the full trade-off; do not re-narrow this back to a bare
    # np.random.seed call).
    set_seeds(args.seed)

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


def _predict_in_batches(model: MothNet, x: np.ndarray, batch_size: int) -> np.ndarray:
    """Run `model.extract_features` over `x` in fixed-size chunks.

    Chunks `x` into `batch_size`-sized slices via plain Python slicing (the final,
    possibly-partial chunk is included automatically — `range(0, len(x), batch_size)`
    plus `x[start:start + batch_size]` never raises or drops rows when `len(x)` is not
    evenly divisible by `batch_size`, and handles `batch_size >= len(x)` identically to
    passing `x` directly, in one chunk). Calls `model.extract_features(chunk)` per
    chunk — an inference-mode-only forward (`extract_features()` already calls
    `self(inputs, training=False)` internally) — and concatenates the per-chunk numpy
    results via `np.concatenate`.

    Exists purely to bound the transient GPU memory of the per-epoch accuracy
    computation at full-MNIST scale (Step 1 of this plan MEASURED a real
    `ResourceExhaustedError` from the prior unbatched call); it is NOT a change to
    `MothNet.extract_features()`'s own contract (`decisions.md` D-002).

    :param model: A built `MothNet` instance.
    :param x: Input array, shape `(N, input_dim)`.
    :param batch_size: Number of rows per chunk; need not evenly divide `N`.
    :return: Concatenated feature array, shape `(N, num_classes)`, as a numpy array.
    """
    # DECISION plan-2026-09-18T080513-debe8b11/D-002: a small, local batching helper
    # inside train_mothnet.py — NOT a change to `MothNet.extract_features()` itself.
    # Pushing batching down into the model would change `extract_features()`'s public
    # contract for every consumer (including `create_cyborg_features`), a broader
    # blast radius than this trainer's own OOM fix. This function must never call
    # `train_hebbian` or anything else that mutates `readout_weights` — the only
    # weight-mutating call site in this file's epoch loop stays `train_hebbian`
    # itself (plan.md invariant 2). See decisions.md D-002.
    chunk_outputs = []
    for start in range(0, len(x), batch_size):
        chunk = x[start:start + batch_size]
        chunk_outputs.append(
            keras.ops.convert_to_numpy(model.extract_features(chunk))
        )
    return np.concatenate(chunk_outputs, axis=0)


# DECISION plan-2026-09-18T045308-c89cdf76/D-004: do NOT convert
# `render_training_dashboard`/`render_mb_sparsity` into `VisualizationManager`
# plugins. A plugin abstraction is earned only when >=2 concrete call sites need it
# (`references/complexity-control.md`); this trainer has exactly one dashboard. See
# decisions.md D-004.
def render_training_dashboard(history: Dict[str, List[float]], out_path: Path) -> None:
    """Render a 2-panel training dashboard PNG from the in-memory `history` dict.

    Left panel: loss vs. epoch. Right panel: train_accuracy and val_accuracy vs.
    epoch, on shared axes with a legend. A pure function — no model/filesystem state
    beyond writing `out_path` — mirroring `train/bfunet/common.py`'s
    `render_training_dashboard` shape (plain dict in, PNG out) without adopting any of
    its image-denoising-specific content (`decisions.md` D-004: hand-rolled function,
    not a `VisualizationManager` plugin).

    Called every epoch by `main()`'s training loop, wrapped in `try/except Exception:
    logger.warning(...)` at the call site — a render failure must never abort training
    (plan.md invariant 9).

    :param history: Per-epoch scalar lists with keys `epoch`, `loss`,
        `train_accuracy`, `val_accuracy` (the exact shape Step 5's loop builds).
    :param out_path: PNG destination (typically
        `run_dir / "visualizations" / "training_dashboard.png"`).
    :return: None.
    """
    epochs = history["epoch"]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history["loss"], color="#d62728", marker="o", markersize=3)
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(
        epochs, history["train_accuracy"], label="train_accuracy",
        color="#1f77b4", marker="o", markersize=3,
    )
    ax_acc.plot(
        epochs, history["val_accuracy"], label="val_accuracy",
        color="#2ca02c", marker="o", markersize=3,
    )
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_mb_sparsity(model: MothNet, x_sample: np.ndarray, out_path: Path) -> None:
    """Render a `plt.spy(...)` sparsity plot of the Mushroom Body's sparse codes.

    Reuses the MothNet README's own Example 2 pattern
    (`src/dl_techniques/models/general_purpose/mothnet/README.md`, "Extracting and
    Visualizing MB Sparse Codes"): `model.extract_mb_features(x_sample)` ->
    `keras.ops.convert_to_numpy(...)` -> `plt.spy(...)`. A pure function — no
    training-loop state beyond `model`/`x_sample`/`out_path`.

    Called periodically by `main()`'s training loop (every `args.viz_freq` epochs,
    `<= 0` disables), wrapped in `try/except Exception: logger.warning(...)` at the
    call site — a render failure must never abort training (plan.md invariant 9).

    :param model: A built `MothNet` instance.
    :param x_sample: A small batch of input samples, shape `(N, input_dim)`.
    :param out_path: PNG destination, expected to be epoch-stamped by the caller
        (e.g. `visualizations/epoch_{epoch+1:03d}_mb_sparsity.png`) so periodic
        renders never overwrite each other.
    :return: None.
    """
    mb_codes = keras.ops.convert_to_numpy(model.extract_mb_features(x_sample))

    fig = plt.figure(figsize=(10, 4))
    plt.spy(mb_codes, markersize=2)
    plt.title("Mushroom Body Sparse Codes")
    plt.xlabel("MB Neuron Index")
    plt.ylabel("Sample Index")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None) -> int:
    """Entry point for the MothNet trainer.

    The first statement parses argv, so ``--help`` exits 0 with a ``usage:`` line
    and no GPU/dataset/model work happens before parsing returns.

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Process exit code.
    """
    args = parse_arguments(argv)

    logger.info(f"Parsed args: {args}")

    # DECISION plan-2026-09-18T045308-c89cdf76/D-011: `--gpu` was declared but never
    # wired at Step 1, a REFLECT-adversarial-review CRITICAL finding — every trainer
    # in this repo supports `--gpu` and calls `setup_gpu(args.gpu)`
    # (`src/train/CLAUDE.md` § What lives in `train.common`); fixed here rather than
    # deferred, since a stale README example (`README.md`'s "Larger-scale run") was
    # already instructing users to pass `--gpu 1` with no effect.
    setup_gpu(gpu_id=args.gpu)

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

    # Created once, before the loop — `render_training_dashboard` (every epoch) and
    # `render_mb_sparsity` (every `--viz-freq` epochs) both write into this single
    # subdirectory (plan.md Step 6).
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # A rerun into the same `--experiment-name` must start `visualizations/` clean
    # of stale periodic PNGs from a PRIOR (possibly longer) run at that name —
    # matching every other artifact class's already-correct fresh-open/overwrite
    # behavior (the CSV below is opened in "w" mode, `best_model.keras`/
    # `final_model.keras` are overwritten by `model.save()`). Scoped to EXACTLY
    # this filename pattern — never a broader glob — since `training_dashboard.png`
    # is overwritten in place every epoch and must be left untouched here
    # (plan-2026-09-18T060057-c1cfc3d3 Step 5 / F-05).
    for stale_png in viz_dir.glob("epoch_*_mb_sparsity.png"):
        try:
            stale_png.unlink()
        except OSError as unlink_error:
            logger.warning(f"Could not remove stale visualization {stale_png}: {unlink_error}")

    # Opened once, before the loop, in write mode (never append: `prepare_run_dir`
    # creates the run directory with `exist_ok=True`, so a rerun into the same
    # directory — e.g. a fixed `--experiment-name` — must start a fresh log, matching
    # how `model.save()` overwrites `best_model.keras`/`final_model.keras` on rerun
    # rather than accumulating a duplicate header + stale rows from a prior run) with
    # the header written here and every row flushed immediately after it's written
    # (never batched) — a crash after N completed epochs must leave exactly N complete
    # CSV rows, per plan.md's edge-case requirement.
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "loss", "train_accuracy", "val_accuracy"])
        csv_file.flush()

        # DECISION plan-2026-09-18T045308-c89cdf76/D-002: do NOT collapse this outer
        # loop into a single `model.train_hebbian(x_train, y_train, epochs=args.epochs,
        # ...)` call. `train_hebbian` has no callback hook, so calling it once per
        # epoch (with `epochs=1`) is the ONLY way to get per-epoch checkpoint/CSV/viz
        # cadence matching bfunet's shape. See decisions.md D-002.
        try:
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
                # (c) val_accuracy/train_accuracy are genuine post-update reads (see
                #     (b)), and the checkpoint-save / CSV-row / history-append
                #     decision further below uses THOSE. `loss`, written into that
                #     same CSV row and history dict, is NOT a post-update read: it is
                #     `epoch_history["loss"][0]`, a within-epoch MEAN over per-batch
                #     losses computed against INTERMEDIATE weight states inside (a)
                #     (`model.py`, unmodified) — never against the epoch-final
                #     weights that (b)'s reads observe. See decisions.md D-007.
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
                #
                # DECISION plan-2026-09-18T060057-c1cfc3d3/D-007: item (c) above was
                # narrowed — it used to imply `loss` is a post-update measurement
                # like val_accuracy/train_accuracy. It is not (see (c)). No
                # checkpoint-selection logic depends on this distinction:
                # `best_model.keras` already gates on val_accuracy only
                # (`decisions.md` D-006 of plan-2026-09-18T045308-c89cdf76). This is
                # a precision fix on the comment, not a behavior change. See
                # decisions.md D-007 for the full trade-off.
                epoch_history = model.train_hebbian(
                    x_train, y_train, epochs=1, batch_size=args.batch_size, verbose=0,
                )
                loss = float(epoch_history["loss"][0])

                val_logits = _predict_in_batches(model, x_val, args.eval_batch_size)
                val_accuracy = float(
                    np.mean(np.argmax(val_logits, axis=-1) == np.argmax(y_val, axis=-1))
                )
                train_logits = _predict_in_batches(model, x_train, args.eval_batch_size)
                train_accuracy = float(
                    np.mean(np.argmax(train_logits, axis=-1) == np.argmax(y_train, axis=-1))
                )

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    model.save(best_checkpoint_path(str(run_dir)))
                    logger.info(
                        f"Epoch {epoch + 1}: new best val_accuracy={val_accuracy:.4f} "
                        f"— saved {best_checkpoint_path(str(run_dir))}"
                    )

                csv_writer.writerow([epoch + 1, loss, train_accuracy, val_accuracy])
                csv_file.flush()

                history["epoch"].append(epoch)
                history["loss"].append(loss)
                history["train_accuracy"].append(train_accuracy)
                history["val_accuracy"].append(val_accuracy)

                # Both render calls are wrapped in try/except — a visualization failure
                # must never abort training (plan.md invariant 9 / Constraints HARD list).
                try:
                    render_training_dashboard(history, viz_dir / "training_dashboard.png")
                except Exception as render_error:
                    logger.warning(
                        f"Epoch {epoch}: render_training_dashboard failed: {render_error}"
                    )

                if args.viz_freq > 0 and (epoch + 1) % args.viz_freq == 0:
                    try:
                        render_mb_sparsity(
                            model, x_val[:8],
                            viz_dir / f"epoch_{epoch + 1:03d}_mb_sparsity.png",
                        )
                    except Exception as render_error:
                        logger.warning(
                            f"Epoch {epoch}: render_mb_sparsity failed: {render_error}"
                        )

                logger.info(
                    f"Epoch {epoch + 1}/{args.epochs} — loss={loss:.4f}, "
                    f"train_accuracy={train_accuracy:.4f}, val_accuracy={val_accuracy:.4f}"
                )
        finally:
            # DECISION plan-2026-09-18T060057-c1cfc3d3/D-008: best-effort final-save
            # net (Invariant 8) — ATTEMPT final_model.keras + training_history.json
            # even when an exception propagated out of the loop body above (from
            # train_hebbian/model.save()/extract_features). `model` was explicitly
            # built before the loop (Step 3 of plan-2026-09-18T045308-c89cdf76), so
            # this save is always valid even under `--epochs 1`, a loop that never
            # beat the `-1.0` best_val_accuracy sentinel, or a loop that raised on
            # its very first iteration. Each save below is wrapped in its OWN
            # try/except so a save failure is logged via logger.error and does NOT
            # mask the ORIGINAL exception — nothing here re-raises or swallows
            # anything, so Python's normal try/finally semantics let that original
            # exception propagate unchanged once this finally block finishes. See
            # decisions.md D-008.
            final_model_path = run_dir / "final_model.keras"
            try:
                model.save(final_model_path)
                logger.info(f"Saved final model to {final_model_path}")
            except Exception as save_error:
                logger.error(
                    f"Failed to save final model to {final_model_path}: {save_error}"
                )

            try:
                save_training_history_json(history, run_dir)
            except Exception as save_error:
                logger.error(f"Failed to save training history JSON: {save_error}")

    logger.info(
        f"Run complete — run_dir={run_dir}, epochs={args.epochs}, "
        f"best_val_accuracy={best_val_accuracy:.4f}, "
        f"final_epoch_loss={history['loss'][-1]:.4f}, "
        f"final_epoch_train_accuracy={history['train_accuracy'][-1]:.4f}, "
        f"final_epoch_val_accuracy={history['val_accuracy'][-1]:.4f}"
    )

    return 0


if __name__ == "__main__":
    main()
