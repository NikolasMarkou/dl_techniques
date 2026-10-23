"""
Neural HMM Training Script
====================================================================

Trains a `NeuralHMM` (`dl_techniques.models.probabilistic.neural_hmm`) on sequences
sampled from a known hidden Markov model (`train.neural_hmm.data`), then scores it
against that ground truth:

- **Likelihood.** The test negative log-likelihood per valid step, next to the same
  quantity under the true HMM (an independent numpy forward algorithm). Generative
  variants report ``-log p(x)``; the supervised ``neural-crf-tagger`` reports
  ``-log p(states | x)`` (``nll_kind`` in the summary says which).
- **State recovery.** Viterbi state accuracy on the test split. An unsupervised model
  names its states arbitrarily, so its labels are first matched to the true ones by
  the assignment that maximizes agreement (``viterbi_acc_matched``); a supervised
  model is also scored directly (``viterbi_acc``). Both sit next to the true HMM's own
  Viterbi accuracy (decoded with the model package's ``inference.viterbi_decode`` on
  the true parameters) and the majority-state rate, the accuracy of a constant
  predictor.

Training is stock ``compile()`` + ``fit()``: the loss is the model's ``compute_loss``
(NLL per valid step), the generative variants get ``y=None`` and the supervised one the
state labels. The train, validation and test splits come from three disjoint random
streams of ``--seed``; the test split is used for nothing but the final scores.

Each run writes one directory under repo-root ``results/`` (or ``--output-dir``):
``config.json``, ``run.log``, ``training_log.csv``, ``training_history.json``,
``best_model.keras`` (the epoch with the lowest ``val_loss``), ``results_summary.json``
(strict JSON; a non-finite number is written as ``null``) and ``visualizations/``
(``loss_curve.png``, ``transitions.png``, ``state_strip.png``). A reused
``--experiment-name`` is refused before anything is written; so is an argument the
data or the model would refuse, and a flag the configuration would ignore
(``--obs-dim``, ``--vocab-size`` or ``--separation`` where the data does not read it,
``--encoder`` on a preset that builds no encoder). ``--n-states`` must be at least 2. A data flag that does
not apply and was not given is recorded as ``null``.

``train.common.create_callbacks()`` is not used: its analyzer callback and default
monitor are built for classification, and this run needs ``restore_best_weights``.
The callbacks are hand-assembled below.

Usage:
    python -m train.neural_hmm.train_neural_hmm --help
    python -m train.neural_hmm.train_neural_hmm --variant neural-hmm --obs-type gaussian \
        --n-states 4 --obs-dim 2 --separation 4.0 --stay-prob 0.9
    python -m train.neural_hmm.train_neural_hmm --variant io-hmm --obs-type categorical \
        --n-states 3 --vocab-size 8
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import keras
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker

from dl_techniques.utils.logger import logger
from dl_techniques.models.probabilistic.neural_hmm import NeuralHMM, create_neural_hmm
from dl_techniques.models.probabilistic.neural_hmm import inference
from dl_techniques.models.probabilistic.neural_hmm.encoders import ENCODER_TYPES
from dl_techniques.optimization import optimizer_builder
from train.common import (
    DEFAULT_SEED, default_experiment_name, prepare_run_dir, save_training_history_json, set_seeds,
    setup_gpu,
)
from train.common.callbacks import EpochCSVLogger, best_checkpoint_path
from train.common.run_artifacts import attach_run_log, refuse_existing_run, write_summary_json
from train.neural_hmm.data import (
    OBS_TYPES, HMMDataset, accuracy, hungarian_mapping, majority_rate, matched_accuracy,
    sample_splits, true_conditional_nll_per_step, true_log_potentials, true_nll_per_step,
    unread_arguments,
)

# ---------------------------------------------------------------------

# `parents[3]` reaches the repo root from THIS file
# (src/train/neural_hmm/train_neural_hmm.py: [0] neural_hmm, [1] train, [2] src, [3] <repo>).
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "neural_hmm"
TRAINING_LOG_NAME = "training_log.csv"
#: Test sequences drawn in the state strip figure.
STRIP_SEQUENCES = 4

# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the Neural HMM trainer's raw `argparse.ArgumentParser`.

    :return: The unparsed parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train a Neural HMM on sequences from a known HMM and score it "
                    "against the truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", choices=sorted(NeuralHMM.MODEL_VARIANTS), default="neural-hmm",
                        help="Model preset (NeuralHMM.MODEL_VARIANTS).")
    parser.add_argument("--obs-type", choices=OBS_TYPES, default="gaussian",
                        help="Observation type of the data and the model.")
    parser.add_argument("--n-states", type=int, default=4,
                        help="Hidden states of the true HMM and of the model; at least 2.")
    parser.add_argument("--obs-dim", type=int, default=2,
                        help="Feature count of a gaussian observation. Refused with categorical "
                             "(recorded as null when not given).")
    parser.add_argument("--vocab-size", type=int, default=8,
                        help="Symbol count of a categorical observation. Refused with gaussian "
                             "(recorded as null when not given).")
    parser.add_argument("--encoder", choices=ENCODER_TYPES, default=None,
                        help="Override the preset's encoder type (default: the preset's). "
                             "Refused for a preset that builds no encoder.")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="Width of the encoder, the observation embedding and the emission head.")

    parser.add_argument("--seq-len", type=int, default=50,
                        help="Padded sequence length; also the transformer's position table size.")
    parser.add_argument("--min-seq-len", type=int, default=25,
                        help="Shortest valid length; valid lengths are uniform on "
                             "[min-seq-len, seq-len].")
    parser.add_argument("--num-train", type=int, default=2000, help="Training sequences.")
    parser.add_argument("--num-val", type=int, default=200,
                        help="Validation sequences (checkpoint selection and early stopping).")
    parser.add_argument("--num-test", type=int, default=500, help="Test sequences (final scores only).")
    parser.add_argument("--separation", type=float, default=4.0,
                        help="Distance between consecutive gaussian means, in standard "
                             "deviations. Refused with categorical (recorded as null when "
                             "not given).")
    parser.add_argument("--stay-prob", type=float, default=0.9,
                        help="Diagonal of the true transition matrix, in (0, 1).")

    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Sequences per batch.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Adam learning rate.")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0,
                        help="Global gradient-norm clip of the optimizer.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Epochs without a lower val_loss before training stops.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Seed of the data splits and of the model initialization.")

    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index; counts within CUDA_VISIBLE_DEVICES when it "
                             "is set, else names a physical GPU.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base directory of the run (default: <repo>/results).")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Run directory name (default: neural_hmm_<variant>_<timestamp>).")
    return parser


def _explicitly_set(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]]) -> Set[str]:
    """Return the destinations ``argv`` sets, whatever the value (the default included).

    Parses again into a namespace pre-filled with a sentinel: argparse writes a default
    only into a missing attribute, so every attribute still holding the sentinel was not
    on the command line. Abbreviations and ``--flag=value`` resolve as in the real parse.
    """
    unset = object()
    dests = vars(parser.parse_args([]))
    seen = parser.parse_args(argv, namespace=argparse.Namespace(**dict.fromkeys(dests, unset)))
    return {dest for dest, value in vars(seen).items() if value is not unset}


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse and validate CLI arguments for the Neural HMM trainer.

    Everything the data module would refuse is refused here, at parse time, so a bad
    value never leaves a half-written run directory behind. So is a data flag the
    configuration would ignore (``train.neural_hmm.data.unread_arguments``: ``--obs-dim``
    for categorical, ``--vocab-size`` for gaussian, ``--separation`` for categorical)
    when it is given explicitly, even at its default value; when it is not given it is
    set to ``None``, so ``config.json`` and the summary record it as ``null``.
    ``--n-states`` below 2 is refused first, as the model refuses it (D-029), so the
    data table's one-state rows are never reached from here.

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Parsed arguments namespace.
    :rtype: argparse.Namespace
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.n_states < 2:
        parser.error(f"--n-states must be >= 2, got {args.n_states}: an HMM needs at least two "
                     f"states; with one, the initial and transition distributions are constants "
                     f"that never train")
    unread = unread_arguments(args.n_states, args.obs_type)
    explicit = _explicitly_set(parser, argv)
    for dest, reason in sorted(unread.items()):
        if dest in explicit:
            parser.error(f"--{dest.replace('_', '-')} would be ignored with --obs-type "
                         f"{args.obs_type} and --n-states {args.n_states}: {reason}. Drop it.")
        setattr(args, dest, None)

    for flag in ("obs_dim", "vocab_size", "hidden_dim", "seq_len", "min_seq_len",
                 "num_train", "num_val", "num_test", "epochs", "batch_size"):
        if getattr(args, flag) is not None and getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be >= 1, got {getattr(args, flag)}")
    if args.min_seq_len > args.seq_len:
        parser.error(f"--min-seq-len ({args.min_seq_len}) must be <= --seq-len ({args.seq_len})")
    if args.vocab_size is not None and args.vocab_size < args.n_states:
        parser.error(f"--vocab-size ({args.vocab_size}) must be >= --n-states ({args.n_states}) "
                     "so that every state owns a symbol")
    if args.separation is not None and not (np.isfinite(args.separation) and args.separation >= 0):
        parser.error(f"--separation must be finite and >= 0, got {args.separation}")
    if args.stay_prob is not None and not 0.0 < args.stay_prob < 1.0:
        parser.error(f"--stay-prob must lie in (0, 1), got {args.stay_prob}")
    for flag in ("learning_rate", "gradient_clip_norm"):
        if not (np.isfinite(getattr(args, flag)) and getattr(args, flag) > 0):
            parser.error(f"--{flag.replace('_', '-')} must be finite and > 0, got {getattr(args, flag)}")
    if args.patience < 0:
        parser.error(f"--patience must be >= 0, got {args.patience}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    return args

# ---------------------------------------------------------------------


def load_splits(args: argparse.Namespace) -> Dict[str, HMMDataset]:
    """Sample the train, validation and test splits named by ``args``.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :return: ``{"train", "val", "test"}`` to :class:`~train.neural_hmm.data.HMMDataset`.
    :rtype: Dict[str, HMMDataset]
    """
    return sample_splits(
        {"train": args.num_train, "val": args.num_val, "test": args.num_test},
        seed=args.seed,
        seq_len=args.seq_len,
        min_seq_len=args.min_seq_len,
        n_states=args.n_states,
        obs_type=args.obs_type,
        obs_dim=args.obs_dim,
        vocab_size=args.vocab_size,
        separation=args.separation,
        stay_prob=args.stay_prob,
    )


def build_model(args: argparse.Namespace) -> NeuralHMM:
    """Construct an unbuilt `NeuralHMM` preset from the parsed arguments.

    ``--seq-len`` becomes ``max_seq_len`` only when the model builds a ``transformer``
    encoder, the one configuration that reads it (``NeuralHMM.unread_arguments``).

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :return: The model.
    :rtype: NeuralHMM
    :raises ValueError: If the model refuses the arguments, or ``--encoder`` is given
        for a preset that builds no encoder (the flag would be ignored).
    """
    kwargs: Dict[str, Any] = {
        "n_states": args.n_states,
        "obs_type": args.obs_type,
        "hidden_dim": args.hidden_dim,
    }
    if args.obs_type == "categorical":
        kwargs["vocab_size"] = args.vocab_size
    else:
        kwargs["obs_dim"] = args.obs_dim
    if args.encoder is not None:
        kwargs["encoder_type"] = args.encoder
    # The model refuses a non-default value for an argument it does not read (D-025), and
    # its own rule says which: --seq-len sizes only a transformer's position table, and
    # --encoder is refused even at the default type when the preset builds no encoder.
    unread = NeuralHMM.unread_arguments({**NeuralHMM.MODEL_VARIANTS[args.variant], **kwargs})
    if args.encoder is not None and "encoder_type" in unread:
        raise ValueError(
            f"--encoder {args.encoder} would be ignored: variant '{args.variant}' builds no "
            "encoder (nothing reads its context). Drop --encoder or pick a variant with one."
        )
    if "max_seq_len" not in unread:
        kwargs["max_seq_len"] = args.seq_len
    return create_neural_hmm(args.variant, **kwargs)


def _inputs(split: HMMDataset, start: int = 0, stop: Optional[int] = None) -> Dict[str, np.ndarray]:
    """The model input dict of rows ``start:stop`` of a split."""
    return {"obs": split.obs[start:stop], "mask": split.mask[start:stop]}


def _labels(model: NeuralHMM, split: HMMDataset) -> Optional[np.ndarray]:
    """``fit``'s ``y``: the state labels in discriminative mode, ``None`` otherwise."""
    return split.states if model.mode == "discriminative" else None


def evaluate(model: NeuralHMM, split: HMMDataset, batch_size: int) -> Dict[str, Any]:
    """Score a model on a split with the model's own public helpers.

    :param model: A trained model.
    :type model: NeuralHMM
    :param split: The split to score.
    :type split: HMMDataset
    :param batch_size: Sequences per forward pass.
    :type batch_size: int
    :return: ``log_likelihood`` ``(N,)`` (``log p(x)``, or ``log p(states | x)`` in
        discriminative mode), ``path`` ``(N, T)`` Viterbi states (``-1`` at padding) and
        ``transition`` ``(K, K)``, the mean of ``exp(log_A[:, t])`` over every valid step
        ``t >= 1`` (the static matrix itself for a static transition).
    :rtype: Dict[str, Any]
    """
    k = model.n_states
    log_lik, paths = [], []
    transition_sum, transition_count = np.zeros((k, k)), 0
    for start in range(0, len(split.obs), batch_size):
        stop = start + batch_size
        inputs = _inputs(split, start, stop)
        states = split.states[start:stop] if model.mode == "discriminative" else None
        log_lik.append(keras.ops.convert_to_numpy(model.log_likelihood(inputs, states)))
        paths.append(keras.ops.convert_to_numpy(model.viterbi(inputs)[0]))
        log_a = keras.ops.convert_to_numpy(model(inputs, training=False)["log_A"]).astype(np.float64)
        moves = (split.mask[start:stop, 1:] > 0)
        transition_sum += np.exp(log_a[:, 1:])[moves].sum(axis=0)
        transition_count += int(moves.sum())
    return {
        "log_likelihood": np.concatenate(log_lik).astype(np.float64),
        "path": np.concatenate(paths),
        "transition": transition_sum / max(transition_count, 1),
    }


def true_viterbi_path(split: HMMDataset, batch_size: int) -> np.ndarray:
    """Decode a split with the TRUE parameters through the model package's Viterbi.

    :param split: The split.
    :type split: HMMDataset
    :param batch_size: Sequences per call.
    :type batch_size: int
    :return: ``(N, T)`` int path, ``-1`` at padding.
    :rtype: np.ndarray
    """
    paths = []
    for start in range(0, len(split.obs), batch_size):
        stop = start + batch_size
        potentials = true_log_potentials(split.true_hmm, split.obs[start:stop])
        path, _ = inference.viterbi_decode(
            potentials["log_pi"], potentials["log_A"], potentials["log_emit"], split.mask[start:stop])
        paths.append(keras.ops.convert_to_numpy(path))
    return np.concatenate(paths)


def _max_abs_output_diff(a: keras.Model, b: keras.Model, split: HMMDataset, batch_size: int) -> float:
    """Largest absolute difference between two models' outputs over a split.

    Equal entries count 0, NaN in both included. A NaN facing a number, or two
    infinities of opposite sign, counts ``inf``: a NaN never hides a disagreement.
    """
    worst = 0.0
    for start in range(0, len(split.obs), batch_size):
        inputs = _inputs(split, start, start + batch_size)
        out_a, out_b = a(inputs, training=False), b(inputs, training=False)
        for key in out_a:
            x = keras.ops.convert_to_numpy(out_a[key]).astype(np.float64)
            y = keras.ops.convert_to_numpy(out_b[key]).astype(np.float64)
            same = (x == y) | (np.isnan(x) & np.isnan(y))
            with np.errstate(invalid="ignore"):
                diff = np.where(same, 0.0, np.abs(x - y))
            diff = np.where(np.isnan(diff), np.inf, diff)
            worst = max(worst, float(diff.max()) if diff.size else 0.0)
    return worst

# ---------------------------------------------------------------------


def _relabel_matrix(matrix: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Move a ``(K, K)`` matrix from predicted-label space to true-label space."""
    out = np.empty_like(matrix)
    out[np.ix_(mapping, mapping)] = matrix
    return out


def render_loss_curve(log_path: Path, reference: float, nll_kind: str, out_path: Path) -> None:
    """Training and validation loss per epoch from ``training_log.csv``.

    The loss is the NLL per valid step; the dashed line is the same quantity under the
    true HMM on the test split.
    """
    with open(log_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) + 1 for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(epochs, [float(row["loss"]) for row in rows], label="train, running mean over the epoch")
    ax.plot(epochs, [float(row["val_loss"]) for row in rows], label="validation, end of epoch (monitored)")
    if np.isfinite(reference):
        ax.axhline(reference, color="black", linestyle="--", linewidth=1, label="true HMM, test split")
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel(f"NLL per valid step ({nll_kind})")
    ax.set_title("Training loss")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_transitions(learned: np.ndarray, true: np.ndarray, title: str, out_path: Path) -> None:
    """Learned and true transition matrices side by side, both in true-label order."""
    k = true.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(3.0 + 1.6 * k, 1.2 + 0.8 * k))
    for ax, matrix, name in ((axes[0], learned, "learned"), (axes[1], true, "true")):
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.6 else "black", fontsize=8)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xlabel("to state")
        ax.set_ylabel("from state")
        ax.set_title(name)
        ax.grid(False)
    fig.colorbar(image, ax=list(axes), shrink=0.8)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_state_strip(
        true_states: np.ndarray, decoded: np.ndarray, n_states: int, title: str, out_path: Path,
) -> None:
    """True and decoded states of a few test sequences; padding is left blank."""
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap("tab10").colors[:max(n_states, 1)])
    cmap.set_bad("white")
    rows = len(true_states)
    fig, axes = plt.subplots(rows, 1, figsize=(10.0, 1.0 + 0.9 * rows), squeeze=False)
    for ax, truth, guess in zip(axes[:, 0], true_states, decoded):
        strip = np.ma.masked_less(np.stack([truth, guess]).astype(np.float64), 0)
        ax.imshow(strip, aspect="auto", cmap=cmap, vmin=-0.5, vmax=n_states - 0.5,
                  interpolation="nearest")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["true", "decoded"])
        ax.grid(False)
    axes[-1, 0].set_xlabel("time step")
    handles = [matplotlib.patches.Patch(color=cmap(k), label=f"state {k}") for k in range(n_states)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_figures(figures: Dict[str, Callable[[], None]], viz_dir: Path) -> Dict[str, List[str]]:
    """Draw every figure; a failing figure is logged and skipped.

    :param figures: Figure name to a zero-argument renderer.
    :param viz_dir: Existing figure directory.
    :return: ``files`` (PNG names written) and ``failed`` (figure names that were not).
    """
    failed = []
    for name, render in figures.items():
        try:
            render()
        except Exception as error:  # a figure must never cost the run its results
            logger.warning(f"Figure '{name}' was not written: {error!r}")
            failed.append(name)
    return {"files": sorted(p.name for p in viz_dir.glob("*.png")), "failed": failed}

# ---------------------------------------------------------------------


def main(argv=None) -> int:
    """Entry point for the Neural HMM trainer.

    The first statement parses argv, so ``--help`` exits 0 with a ``usage:`` line
    before any GPU, data or model work.

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Process exit code.
    :rtype: int
    """
    args = parse_arguments(argv)
    setup_gpu(gpu_id=args.gpu)

    base_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results"
    run_dir = base_dir / (args.experiment_name or default_experiment_name(EXPERIMENT_NAME, args.variant))
    refuse_existing_run(run_dir)
    set_seeds(args.seed)
    # Built before the run directory exists: a combination the model refuses (or an
    # --encoder the preset would ignore) then raises without using up the name.
    model = build_model(args)
    prepare_run_dir(args, output_dir=run_dir)

    with attach_run_log(run_dir):
        start = time.perf_counter()
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Parsed args: {args}")
        splits = load_splits(args)
        train, val, test = splits["train"], splits["val"], splits["test"]
        hmm = test.true_hmm
        discriminative = model.mode == "discriminative"

        model.compile(optimizer=optimizer_builder(
            {"type": "adam", "gradient_clipping_by_norm": args.gradient_clip_norm}, args.learning_rate))
        jit_compile = bool(model.jit_compile)
        logger.info(f"Model: {args.variant}, mode={model.mode}, encoder="
                    f"{model.encoder_type if model.encoder is not None else None}, "
                    f"jit_compile={jit_compile}")

        best_path = Path(best_checkpoint_path(str(run_dir)))
        # Hand-assembled rather than create_callbacks(): see the module docstring.
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-021
        # EarlyStopping RESTORES the best weights. Do NOT set restore_best_weights=False
        # (the plan's first draft): the in-memory model would then hold the last epoch,
        # and the reload check below would compare best_model.keras with a different
        # model. ModelCheckpoint and EarlyStopping select by the same rule (strictly
        # lower val_loss), so the restored weights are those of best_model.keras, and
        # an exact match proves the archive. See decisions.md D-021.
        callbacks = [
            keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", mode="min",
                                            save_best_only=True),
            EpochCSVLogger(str(run_dir / TRAINING_LOG_NAME)),
            keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.patience,
                                          restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: logger.info(
                f"epoch {epoch + 1}: loss {logs.get('loss', float('nan')):.5f}, "
                f"val_loss {logs.get('val_loss', float('nan')):.5f}")),
        ]
        fit_start = time.perf_counter()
        history = model.fit(
            _inputs(train), _labels(model, train),
            validation_data=(_inputs(val), _labels(model, val)),
            epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=0,
        ).history
        fit_seconds = time.perf_counter() - fit_start
        save_training_history_json(history, run_dir)

        # The scored model is the archive a user would load: best_model.keras, checked
        # to reproduce the in-memory (restored) weights exactly.
        best_checkpoint: Dict[str, Any] = {"file": None, "epoch": None, "val_loss": None,
                                           "reload_max_abs_diff": None, "reload_exact": None}
        scored = model
        if best_path.is_file():
            reloaded = keras.models.load_model(best_path)
            diff = _max_abs_output_diff(reloaded, model, test, args.batch_size)
            epoch = int(np.nanargmin(history["val_loss"]))
            best_checkpoint = {"file": best_path.name, "epoch": epoch + 1,
                               "val_loss": float(history["val_loss"][epoch]),
                               "reload_max_abs_diff": diff, "reload_exact": diff == 0.0}
            if diff != 0.0:
                logger.warning(f"best_model.keras differs from the restored weights by {diff:.3g}")
            scored = reloaded
        else:
            logger.warning("No best checkpoint: val_loss was never finite; scoring the final weights.")

        scores = evaluate(scored, test, args.batch_size)
        steps = float(test.mask.sum())
        test_nll = float(-scores["log_likelihood"].sum() / steps)
        if discriminative:
            nll_kind, true_nll = "conditional", true_conditional_nll_per_step(
                hmm, test.obs, test.states, test.mask)
        else:
            nll_kind, true_nll = "marginal", true_nll_per_step(hmm, test.obs, test.mask)
        mapping = hungarian_mapping(scores["path"], test.states, args.n_states)
        true_path = true_viterbi_path(test, args.batch_size)

        summary: Dict[str, Any] = {
            "model": EXPERIMENT_NAME,
            "variant": args.variant,
            "mode": model.mode,
            "obs_type": args.obs_type,
            "n_states": args.n_states,
            "obs_dim": args.obs_dim,
            "vocab_size": args.vocab_size,
            "encoder_type": model.encoder_type if model.encoder is not None else None,
            "hidden_dim": args.hidden_dim,
            "params": int(scored.count_params()),
            "seq_len": args.seq_len,
            "min_seq_len": args.min_seq_len,
            "num_train": args.num_train,
            "num_val": args.num_val,
            "num_test": args.num_test,
            "test_valid_steps": int(steps),
            "separation": args.separation,
            "stay_prob": args.stay_prob,
            "seed": args.seed,
            "jit_compile": jit_compile,
            "epochs_requested": args.epochs,
            "epochs_run": len(history["loss"]),
            "best_checkpoint": best_checkpoint,
            "nll_kind": nll_kind,
            "test_nll_per_step": test_nll,
            "true_nll_per_step": true_nll,
            "nll_gap": test_nll - true_nll,
            "viterbi_acc": accuracy(scores["path"], test.states) if discriminative else None,
            "viterbi_acc_matched": matched_accuracy(scores["path"], test.states, args.n_states),
            "state_mapping": [int(v) for v in mapping],
            "true_viterbi_acc": accuracy(true_path, test.states),
            "majority_rate": majority_rate(test.states),
            "fit_seconds": fit_seconds,
        }

        viz_dir = run_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        # Supervised labels are the true labels; only unsupervised ones are relabelled.
        shown = np.arange(args.n_states) if discriminative else mapping
        order = "true labels" if discriminative else "model labels matched to the true ones"
        decoded = np.where(test.states >= 0, shown[np.maximum(scores["path"], 0)], -1)
        summary["visualizations"] = render_figures({
            "loss_curve": lambda: render_loss_curve(
                run_dir / TRAINING_LOG_NAME, true_nll, nll_kind, viz_dir / "loss_curve.png"),
            "transitions": lambda: render_transitions(
                _relabel_matrix(scores["transition"], shown), hmm.transition,
                f"Transition matrix ({order}; learned: mean over valid test steps)",
                viz_dir / "transitions.png"),
            "state_strip": lambda: render_state_strip(
                test.states[:STRIP_SEQUENCES], decoded[:STRIP_SEQUENCES], args.n_states,
                f"Viterbi decoding of {min(STRIP_SEQUENCES, len(test.states))} test sequences "
                f"({order})", viz_dir / "state_strip.png"),
        }, viz_dir)
        summary["total_seconds"] = time.perf_counter() - start
        write_summary_json(run_dir, summary)
        logger.info(
            f"Done: {args.variant}, test NLL/step {test_nll:.4f} vs true {true_nll:.4f} "
            f"({nll_kind}); Viterbi accuracy matched {summary['viterbi_acc_matched']:.4f}, "
            f"true model {summary['true_viterbi_acc']:.4f}, majority {summary['majority_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------
