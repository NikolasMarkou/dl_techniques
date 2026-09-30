"""
HKAN Training Script
====================================================================

Trains `HKAN` (`dl_techniques.models.general_purpose.hkan`) on a synthetic target of
the HKAN paper or on user-supplied CSV files, in one of three training modes
(`--training-mode`):

- `closed_form` (default): the paper's procedure, `HKAN.fit_closed_form`. Layer by
  layer least squares in float64, no gradients, no epochs.
- `backprop`: stock `compile()` + `fit()` from the model's initial state. No custom
  `train_step`.
- `closed_form_then_backprop`: the closed-form fit, then stock `fit()` as a fine-tune
  that starts from the closed-form weights.

In the two backprop modes a validation part (`--val-fraction`) is carved out of the
TRAIN split and the model is fitted on the remaining rows; the test split is never
used for fitting or for checkpoint selection. `closed_form` fits on every train row.

Each run writes one directory under repo-root `results/` (or `--output-dir`):
`config.json`, `run.log`, `results_summary.json` (strict JSON), `final_model.keras`
and `visualizations/`; the backprop modes add `training_log.csv`,
`training_history.json` and, when an epoch earned it, `best_model.keras`; the
two-phase mode adds `closed_form_model.keras`, the model the fine-tune started
from. With `--repeats N` the fit is repeated with N derived seeds (new basis
centers and initial weights, same data); the summary holds every repeat plus median
and interquartile range, while the saved models, the history and the figures belong
to repeat 0. Reported RMSEs come from the weights at the end of training in every
repeat. A reused `--experiment-name` is refused before anything is written.

`best_checkpoint` in the summary is `null` in `closed_form` mode and otherwise
always a dict with the same eight keys, describing the best model of repeat 0 by
the monitor (`val_loss`, or `loss` when `--val-fraction 0`):

- `monitor`: the monitored name.
- `source`: `"epoch"` (an epoch of `fit`), `"closed_form"` (two-phase mode only:
  no epoch beat the closed-form start) or `"none"` (the monitor was never finite,
  for example a diverged run; nothing was saved).
- `file`: `best_model.keras`, `closed_form_model.keras` or `null`. `best_model.keras`
  exists only when `source` is `"epoch"`.
- `epoch`: the 1-based epoch, `0` for the closed-form start, `null` for `"none"`.
- `monitor_value`: the monitored MSE of that model (`null` for `"none"`).
- `train_rmse`, `test_rmse`: its scores (`null` for `"none"`).
- `reason`: `null` for `"epoch"`, otherwise one sentence saying why.

A run whose loss is not finite still finishes and writes its summary; non-finite
numbers are written as `null`.

`train.common.create_callbacks()` is not used: it always adds early stopping, which
would make repeats stop at different epochs, and it has no role at all in the
closed-form mode. The backprop modes hand-assemble `ModelCheckpoint` + `CSVLogger`.

Usage:
    python -m train.hkan.train_hkan --help
    python -m train.hkan.train_hkan --dataset tf5 --hidden-units 128 \
        --basis tanh,identity --slope 50 --num-basis 23,10 --l2-block 0.01,0.1
    python -m train.hkan.train_hkan --dataset tf2 --training-mode backprop --epochs 200
"""

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import keras
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from dl_techniques.utils.logger import logger
from dl_techniques.models.general_purpose.hkan import HKAN
from dl_techniques.models.general_purpose.hkan.hkan_layer import (
    BASIS_NAMES, CENTER_MODES, DEFAULT_SLOPE,
)
from dl_techniques.models.general_purpose.hkan.model import DEFAULT_L2_BLOCK, DEFAULT_L2_MIX
from dl_techniques.optimization import optimizer_builder
from train.common import (
    default_experiment_name, prepare_run_dir, save_training_history_json, set_seeds,
    setup_gpu,
)
from train.common.callbacks import best_checkpoint_path
from train.common.run_artifacts import (
    RESULTS_SUMMARY_NAME, RUN_LOG_NAME, attach_run_log, refuse_existing_run,
    write_summary_json,
)
from train.hkan.data import TARGETS, load_csv_dataset, make_dataset

# ---------------------------------------------------------------------

# `parents[3]` reaches the repo root from THIS file
# (src/train/hkan/train_hkan.py: [0] hkan, [1] train, [2] src, [3] <repo>).
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "hkan"
TRAINING_MODES = ("closed_form", "backprop", "closed_form_then_backprop")
FINAL_MODEL_NAME = "final_model.keras"
#: The model the two-phase fine-tune starts from, saved before `fit` (repeat 0).
CLOSED_FORM_MODEL_NAME = "closed_form_model.keras"
#: Last entry of the validation split's seed list (the bytes of "SPLT"). Non-zero
#: and last because numpy drops trailing zeros of a seed list; different from the
#: model package's `SEED_DOMAIN_TAG`, so the split shares a stream with nothing.
VALIDATION_SPLIT_TAG = 0x53504C54
#: Files whose presence means the experiment directory already holds an HKAN run.
RUN_ARTIFACTS = (RESULTS_SUMMARY_NAME, "config.json", RUN_LOG_NAME, FINAL_MODEL_NAME)

# ---------------------------------------------------------------------


def _list_of(cast: Callable[[str], Any]) -> Callable[[str], List[Any]]:
    """Return an argparse ``type`` that parses a comma-separated list with ``cast``."""
    def parse(text: str) -> List[Any]:
        try:
            return [cast(item.strip()) for item in text.split(",") if item.strip()]
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
    return parse


def _build_parser() -> argparse.ArgumentParser:
    """Construct the HKAN trainer's raw `argparse.ArgumentParser`.

    A purpose-built parser: `create_base_argument_parser()` carries image, schedule
    and patience flags this model has no use for. Per-layer settings take one value
    (used for every layer) or a comma-separated value per layer, where the number of
    layers is the number of `--hidden-units` entries plus one.

    :return: The unparsed parser.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train HKAN by closed-form least squares, backpropagation, or both.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="closed_form",
                        help="How the model is trained.")
    parser.add_argument("--dataset", choices=sorted(TARGETS) + ["csv"], default="tf5",
                        help="A synthetic target of the paper, or 'csv' for user files.")
    parser.add_argument("--train-csv", type=str, default=None,
                        help="Train file for --dataset csv (no header, last column the target).")
    parser.add_argument("--test-csv", type=str, default=None,
                        help="Test file for --dataset csv.")
    parser.add_argument("--scale-csv", action="store_true",
                        help="Min-max scale the CSV data with the train file's statistics.")
    parser.add_argument("--num-train-samples", type=int, default=None,
                        help="Train rows of a synthetic target (default: the paper's count).")
    parser.add_argument("--num-test-samples", type=int, default=None,
                        help="Test rows of a synthetic target (default: the paper's count).")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Independent fits with derived seeds; the paper uses 50.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the data and of the per-repeat derived seeds.")

    parser.add_argument("--hidden-units", type=_list_of(int), default=[64],
                        help="Comma-separated hidden widths; empty for a one-layer model.")
    parser.add_argument("--num-basis", type=_list_of(int), default=[10],
                        help="Basis functions per block, per layer.")
    parser.add_argument("--basis", type=_list_of(str), default=["sigmoid"],
                        help=f"Basis per layer: {', '.join(BASIS_NAMES)}.")
    parser.add_argument("--slope", type=_list_of(float), default=[DEFAULT_SLOPE],
                        help="Basis slope (the paper's sigma), per layer.")
    parser.add_argument("--centers", type=_list_of(str), default=["random"],
                        help=f"Center placement per layer: {', '.join(CENTER_MODES)}.")
    parser.add_argument("--l2-block", type=_list_of(float), default=[DEFAULT_L2_BLOCK],
                        help="Ridge strength of the block fits, per layer (closed-form modes).")
    parser.add_argument("--l2-mix", type=_list_of(float), default=[DEFAULT_L2_MIX],
                        help="Ridge strength of the connecting fits, per layer (closed-form "
                             "modes); 0 is the paper's plain least squares.")
    parser.add_argument("--no-block-bias", action="store_true",
                        help="Drop the intercept of every block (the paper's equation 13).")
    parser.add_argument("--no-bias", action="store_true",
                        help="Drop the intercept of every layer output (the paper's equation 6).")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Outputs solved at once in the closed-form fit (default: auto).")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Backprop epochs. Ignored in closed_form mode.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Backprop batch size. Ignored in closed_form mode.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Adam learning rate. Ignored in closed_form mode.")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Part of the train split held out for checkpoint selection "
                             "in the backprop modes. Ignored in closed_form mode.")
    parser.add_argument("--predict-batch-size", type=int, default=256,
                        help="Batch size of every prediction pass.")

    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base directory of the run (default: <repo>/results).")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Run directory name (default: hkan_<timestamp>).")
    return parser


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse and validate CLI arguments for the HKAN trainer.

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Parsed arguments namespace.
    :rtype: argparse.Namespace
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.dataset == "csv" and not (args.train_csv and args.test_csv):
        parser.error("--dataset csv needs both --train-csv and --test-csv")
    for flag in ("repeats", "epochs", "batch_size", "predict_batch_size"):
        if getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be >= 1, got {getattr(args, flag)}")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error(f"--val-fraction must be in [0, 1), got {args.val_fraction}")
    if args.learning_rate <= 0:
        parser.error(f"--learning-rate must be > 0, got {args.learning_rate}")
    num_layers = len(args.hidden_units) + 1
    for flag in ("num_basis", "basis", "slope", "centers", "l2_block", "l2_mix"):
        if len(getattr(args, flag)) not in (1, num_layers):
            parser.error(
                f"--{flag.replace('_', '-')} takes 1 value or {num_layers} (one per "
                f"layer), got {len(getattr(args, flag))}"
            )
    # Everything below would otherwise fail inside the model or the data module,
    # after the run directory exists, and leave the experiment name used up.
    for flag, allowed in (("basis", BASIS_NAMES), ("centers", CENTER_MODES)):
        unknown = [item for item in getattr(args, flag) if item not in allowed]
        if unknown:
            parser.error(f"--{flag} items must be among {', '.join(allowed)}; got {unknown}")
    for flag in ("hidden_units", "num_basis"):
        if any(item < 1 for item in getattr(args, flag)):
            parser.error(f"--{flag.replace('_', '-')} items must be >= 1, got {getattr(args, flag)}")
    if not all(np.isfinite(args.slope)):
        parser.error(f"--slope items must be finite, got {args.slope}")
    for flag in ("l2_block", "l2_mix"):
        if not all(np.isfinite(item) and item >= 0 for item in getattr(args, flag)):
            parser.error(
                f"--{flag.replace('_', '-')} items must be finite and >= 0, got {getattr(args, flag)}")
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error(f"--chunk-size must be >= 1, got {args.chunk_size}")
    if args.seed < 0:
        parser.error(f"--seed must be >= 0, got {args.seed}")
    for flag in ("num_train_samples", "num_test_samples"):
        if getattr(args, flag) is not None and getattr(args, flag) < 2:
            parser.error(f"--{flag.replace('_', '-')} must be >= 2, got {getattr(args, flag)}")
    return args

# ---------------------------------------------------------------------


def load_data(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    """Load the dataset named by ``args``.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :return: ``x_train``, ``y_train``, ``x_test``, ``y_test``, all float64.
    :rtype: Dict[str, np.ndarray]
    """
    # DECISION plan-2026-09-30T082355-4d999dbc/D-031
    # The inputs stay float64. Do NOT round them to float32 here "so the fit sees
    # what the model is called on": the targets were computed from the unrounded
    # inputs, so rounding makes them inconsistent with the inputs at about 5e-8,
    # the near-interpolating closed-form fit chases that with large weights, and
    # the float32 forward pass then amplifies it. Measured on TF5 (width 912):
    # train RMSE 1.4e-4 with the rounding, 1.4e-7 without. See decisions.md D-031.
    if args.dataset == "csv":
        data = load_csv_dataset(args.train_csv, args.test_csv, scale=args.scale_csv)
        low, high = data["x_train"].min(axis=0), data["x_train"].max(axis=0)
        outside = np.flatnonzero((low < 0.0) | (high > 1.0))
        if outside.size and args.centers[0] != "data":
            # `random` and `equally_spaced` centers of the first layer lie on [0, 1].
            logger.warning(
                f"CSV input column(s) {outside.tolist()} (0-based) leave [0, 1] on the train "
                f"file (overall range {low.min():.6g} to {high.max():.6g}) while the first "
                f"layer's centers are '{args.centers[0]}', which are placed on [0, 1]: "
                "expect a poor fit. Pass --scale-csv, or 'data' as the first --centers item."
            )
        return data
    return make_dataset(args.dataset, seed=args.seed,
                        num_train=args.num_train_samples, num_test=args.num_test_samples)


def build_model(args: argparse.Namespace, seed: int) -> HKAN:
    """Construct an unbuilt `HKAN` from the parsed arguments.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :param seed: Seed of the basis centers of this repeat.
    :type seed: int
    :return: The model.
    :rtype: HKAN
    """
    def per_layer(values: List[Any]) -> Any:
        return values[0] if len(values) == 1 else list(values)

    return HKAN(
        hidden_units=list(args.hidden_units),
        num_basis=per_layer(args.num_basis),
        basis=per_layer(args.basis),
        slope=per_layer(args.slope),
        centers=per_layer(args.centers),
        l2_block=per_layer(args.l2_block),
        l2_mix=per_layer(args.l2_mix),
        use_block_bias=not args.no_block_bias,
        use_bias=not args.no_bias,
        seed=seed,
    )


def _predict(model: keras.Model, x: np.ndarray, batch_size: int) -> np.ndarray:
    """Predict in batches with plain eager calls, returned as float64 ``(N,)``.

    Not `model.predict`: several models are scored per run, each on a full and a
    short last batch, and that many traced predict functions trips TensorFlow's
    retracing warning. The feature tensor is ``(B, n_out, n_in, m)``, so the batch
    bounds memory.
    """
    return np.concatenate([
        keras.ops.convert_to_numpy(model(x[start:start + batch_size], training=False))[:, 0]
        for start in range(0, len(x), batch_size)
    ]).astype(np.float64)


def _rmse(model: keras.Model, x: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    """RMSE of the model's float32 predictions against float64 targets."""
    return float(np.sqrt(np.mean(np.square(_predict(model, x, batch_size) - y))))


def split_validation(num_rows: int, val_fraction: float, seed: int):
    """Split train row indices into fit rows and validation rows.

    :param num_rows: Rows of the train split.
    :type num_rows: int
    :param val_fraction: Part held out; 0 holds out nothing.
    :type val_fraction: float
    :param seed: Seed of the permutation.
    :type seed: int
    :return: ``(fit_indices, val_indices)``, disjoint, both sorted.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: If fewer than 2 rows would be left to fit on.
    """
    num_val = 0 if val_fraction == 0 else max(1, int(round(val_fraction * num_rows)))
    if num_rows - num_val < 2:
        raise ValueError(
            f"--val-fraction {val_fraction} leaves {num_rows - num_val} of "
            f"{num_rows} train rows to fit on; at least 2 are needed"
        )
    order = np.random.default_rng([seed, VALIDATION_SPLIT_TAG]).permutation(num_rows)
    return np.sort(order[num_val:]), np.sort(order[:num_val])


def run_repeat(
        args: argparse.Namespace,
        data: Dict[str, np.ndarray],
        repeat: int,
        run_dir: Path,
) -> Dict[str, Any]:
    """Train and score one model.

    Repeat 0 also saves its models and, in the backprop modes, its checkpoint and
    per-epoch log into ``run_dir``; its ``best_checkpoint`` entry has the shape the
    module docstring describes.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :param data: Output of :func:`load_data`.
    :type data: Dict[str, np.ndarray]
    :param repeat: Repeat index, from 0.
    :type repeat: int
    :param run_dir: Existing run directory.
    :type run_dir: Path
    :return: The repeat's scores; for repeat 0 also ``_model``, ``_history`` and
        ``_importance`` (dropped before the summary is written).
    :rtype: Dict[str, Any]
    """
    seed = int(np.random.SeedSequence([args.seed, repeat]).generate_state(1)[0] % (2 ** 31 - 1))
    set_seeds(seed)
    backprop = args.training_mode != "closed_form"
    closed_form = args.training_mode != "backprop"
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    fit_rows, val_rows = split_validation(
        len(x_train), args.val_fraction if backprop else 0.0, seed)
    x_fit, y_fit = x_train[fit_rows], y_train[fit_rows]

    def score(model: keras.Model) -> Dict[str, float]:
        return {
            "train_rmse": _rmse(model, x_fit, y_fit, args.predict_batch_size),
            "test_rmse": _rmse(model, x_test, y_test, args.predict_batch_size),
        }

    model = build_model(args, seed)
    # DECISION plan-2026-09-30T082355-4d999dbc/D-043
    # `fit_rows` is RECORDED here, from the rows this repeat is fitted on. Do NOT
    # recompute it in `main` from the arguments: a recomputed value keeps saying
    # "all train rows" when the fit silently holds rows out. See decisions.md D-043.
    result: Dict[str, Any] = {"repeat": repeat, "seed": seed, "fit_rows": int(len(x_fit))}
    diagnostics, history, start_mse = None, None, None
    start = time.perf_counter()

    if closed_form:
        diagnostics = model.fit_closed_form(x_fit, y_fit, chunk_size=args.chunk_size)
        result["closed_form_seconds"] = time.perf_counter() - start
        result["layer_rmse"] = diagnostics["layer_rmse"]
        # How far the model left behind is from the float64 fit it reports.
        result["forward_deviation_rms"] = float(diagnostics["forward_deviation_rms"])
        if backprop:
            result["closed_form_phase"] = score(model)
            if repeat == 0:
                # Before compile(): an archive holding a never-built optimizer
                # does not reload (Keras 3.8 refuses the variable count).
                model.save(run_dir / CLOSED_FORM_MODEL_NAME)
    elif "data" in args.centers:
        model.initialize_centers(x_fit)

    if backprop:
        # The optimizer is created after the closed-form assign; stock fit() then
        # starts from the assigned weights (the optimizer holds no copy of them).
        model.compile(optimizer=optimizer_builder({"type": "adam"}, args.learning_rate),
                      loss="mse")
        monitor = "val_loss" if len(val_rows) else "loss"
        callbacks = []
        if repeat == 0:
            if closed_form:
                # DECISION plan-2026-09-30T082355-4d999dbc/D-042
                # The closed-form model was saved above, before fit, and its own
                # monitored MSE is the checkpoint's starting threshold. Do NOT drop
                # `initial_value_threshold`: without it `best_model.keras` is the
                # best EPOCH even when every epoch is worse than the start, and the
                # better model the run began with is kept nowhere (measured on the
                # tutorial TF5 run, test RMSE: start 1.5e-07, "best" epoch 1.6e-05). The value
                # is computed as Keras computes `val_loss` (float32 predictions
                # against float32 targets) on the validation rows, or on the fit
                # rows when none are held out. See decisions.md D-042.
                rows = val_rows if len(val_rows) else fit_rows
                start_mse = float(np.mean(np.square(
                    _predict(model, x_train[rows], args.predict_batch_size)
                    - y_train[rows].astype(np.float32))))
            # Hand-assembled rather than create_callbacks(): see the module docstring.
            # A non-finite threshold would stop Keras from ever saving, so it is
            # passed only when finite.
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    best_checkpoint_path(str(run_dir)), monitor=monitor, mode="min",
                    save_best_only=True,
                    initial_value_threshold=(
                        start_mse if start_mse is not None and np.isfinite(start_mse) else None)),
                keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
            ]
        history = model.fit(
            x_fit, y_fit.astype(np.float32),
            validation_data=(
                (x_train[val_rows], y_train[val_rows].astype(np.float32))
                if len(val_rows) else None),
            epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=0,
        ).history
        result["final_loss"] = float(history["loss"][-1])

    result["fit_seconds"] = time.perf_counter() - start
    result.update(score(model))
    logger.info(
        f"Repeat {repeat} (seed {seed}): train RMSE {result['train_rmse']:.6g}, "
        f"test RMSE {result['test_rmse']:.6g}, {result['fit_seconds']:.2f} s"
    )

    if repeat == 0:
        model.save(run_dir / FINAL_MODEL_NAME)
        if backprop:
            best_path = Path(best_checkpoint_path(str(run_dir)))
            if best_path.is_file():  # written only on an improvement, so a finite epoch exists
                epoch = int(np.nanargmin(history[monitor])) + 1
                best = {"source": "epoch", "file": best_path.name, "epoch": epoch,
                        "monitor_value": float(history[monitor][epoch - 1]),
                        **score(keras.models.load_model(best_path)), "reason": None}
            elif start_mse is not None and np.isfinite(start_mse):
                best = {"source": "closed_form", "file": CLOSED_FORM_MODEL_NAME, "epoch": 0,
                        "monitor_value": start_mse, **result["closed_form_phase"],
                        "reason": f"no epoch lowered {monitor} below the closed-form start"}
            else:
                best = {"source": "none", "file": None, "epoch": None, "monitor_value": None,
                        "train_rmse": None, "test_rmse": None,
                        "reason": f"{monitor} was never finite, so no checkpoint was written"}
                logger.warning(f"No best checkpoint: {best['reason']}")
            result["best_checkpoint"] = {"monitor": monitor, **best}
        result["_model"] = model
        result["_history"] = history
        result["_importance"] = None if diagnostics is None else diagnostics["importance"]
    return result

# ---------------------------------------------------------------------


def _spread(values: List[float]) -> Dict[str, float]:
    """Median, interquartile range, minimum and maximum of a list of floats."""
    q25, q75 = np.percentile(values, [25, 75])
    return {"median": float(np.median(values)), "iqr": float(q75 - q25),
            "min": float(np.min(values)), "max": float(np.max(values))}


def render_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Predicted against target, and residual against predicted, on the test split."""
    residuals = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    left.scatter(y_true, y_pred, s=4, alpha=0.4, color="tab:blue")
    low, high = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    left.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    left.set_xlabel("target")
    left.set_ylabel("predicted")
    left.set_title("Predicted vs target")
    right.scatter(y_pred, residuals, s=4, alpha=0.4, color="tab:orange")
    right.axhline(0.0, color="black", linestyle="--", linewidth=1)
    right.set_xlabel("predicted")
    right.set_ylabel("predicted - target")
    right.set_title("Residuals")
    for ax in (left, right):
        ax.grid(alpha=0.3)
    fig.suptitle(f"Test split, repeat 0, weights at the end of training: RMSE {rmse:.3e} "
                 f"over {len(y_true)} rows")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_importance(importance: np.ndarray, out_path: Path) -> None:
    """Bar chart of the per-input importance (mean first-layer block R^2)."""
    fig, ax = plt.subplots(figsize=(max(5.0, 0.5 * len(importance) + 3.0), 4.0))
    ax.bar(np.arange(len(importance)), importance, color="tab:blue", zorder=3)
    ax.set_xticks(np.arange(len(importance)))
    ax.set_xticklabels([f"x{i + 1}" for i in range(len(importance))])
    ax.set_ylabel("mean $R^2$ of the first-layer blocks")
    ax.set_title("Input importance (paper eq. 14), repeat 0")
    ax.xaxis.grid(False)  # the repo plot style draws x grid lines through the bars
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_rmse_box(train: List[float], test: List[float], out_path: Path) -> None:
    """Box plot of the train and test RMSE over the repeats, with every point drawn."""
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.boxplot([train, test], showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["train", "test"])
    for position, values in ((1, train), (2, test)):
        ax.scatter(np.full(len(values), position), values, color="tab:red", s=14, zorder=3)
    # Log scale only across a decade or more: inside one decade matplotlib labels
    # at most one tick of a log axis, which left the TF5 backprop plot unreadable.
    values = np.asarray(list(train) + list(test), dtype=np.float64)
    if values.min() > 0 and values.max() / values.min() >= 10.0:
        ax.set_yscale("log")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE over {len(train)} repeats")
    ax.xaxis.grid(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_loss_curve(
        history: Dict[str, List[float]], out_path: Path, start_mse: Optional[float] = None,
) -> None:
    """Training (and validation) MSE per epoch of repeat 0, log scale.

    ``start_mse`` draws the closed-form train MSE the fine-tune started from. The
    epoch-1 train value is Keras' running mean over that epoch's batches, so it can
    sit above the start line even when the epoch ends below it.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    epochs = np.arange(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], label="train (fit rows)")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="validation")
    if start_mse is not None:
        ax.axhline(start_mse, color="black", linestyle="--", linewidth=1,
                   label="closed-form fit (train)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("MSE")
    ax.set_title("Backprop loss, repeat 0")
    # Outside the axes: inside, the legend box landed on the closed-form line.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_figures(
        args: argparse.Namespace,
        data: Dict[str, np.ndarray],
        results: List[Dict[str, Any]],
        run_dir: Path,
) -> Dict[str, List[str]]:
    """Draw every figure of the run; a failing figure is logged and skipped.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :param data: Output of :func:`load_data`.
    :type data: Dict[str, np.ndarray]
    :param results: Output of :func:`run_repeat` for every repeat (repeat 0 first).
    :type results: List[Dict[str, Any]]
    :param run_dir: Existing run directory.
    :type run_dir: Path
    :return: ``files`` (PNG names written) and ``failed`` (figure names that were not).
    :rtype: Dict[str, List[str]]
    """
    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    first = results[0]

    # Hand-rolled rather than the library's `regression_dashboard` plugin: that
    # plugin raises on the installed seaborn ("Got both 'linewidth' and
    # 'linewidths'") and prints RMSE with four decimals, which reads 0.0000 for the
    # errors this model reaches (decisions.md D-028 of plan-2026-09-30T082355-4d999dbc).
    figures: Dict[str, Callable[[], None]] = {
        "test_predictions": lambda: render_predictions(
            data["y_test"],
            _predict(first["_model"], data["x_test"], args.predict_batch_size),
            viz_dir / "test_predictions.png"),
    }
    if first["_importance"] is not None:
        figures["input_importance"] = lambda: render_importance(
            first["_importance"], viz_dir / "input_importance.png")
    if len(results) > 1:
        figures["rmse_over_repeats"] = lambda: render_rmse_box(
            [r["train_rmse"] for r in results], [r["test_rmse"] for r in results],
            viz_dir / "rmse_over_repeats.png")
    if first["_history"] is not None:
        start = first.get("closed_form_phase")
        figures["loss_curve"] = lambda: render_loss_curve(
            first["_history"], viz_dir / "loss_curve.png",
            start_mse=None if start is None else start["train_rmse"] ** 2)

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
    """Entry point for the HKAN trainer.

    The first statement parses argv, so ``--help`` exits 0 with a ``usage:`` line
    before any GPU, dataset or model work.

    :param argv: Argument list to parse; ``None`` defers to ``sys.argv[1:]``.
    :return: Process exit code.
    :rtype: int
    """
    args = parse_arguments(argv)
    setup_gpu(gpu_id=args.gpu)

    base_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results"
    run_dir = base_dir / (args.experiment_name or default_experiment_name(EXPERIMENT_NAME))
    refuse_existing_run(run_dir, artifact_names=RUN_ARTIFACTS)
    prepare_run_dir(args, output_dir=run_dir)

    with attach_run_log(run_dir):
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Parsed args: {args}")
        data = load_data(args)
        results = [run_repeat(args, data, repeat, run_dir) for repeat in range(args.repeats)]
        first = results[0]

        if first["_history"] is not None:
            save_training_history_json(first["_history"], run_dir)
        figures = render_figures(args, data, results, run_dir)
        importance = first["_importance"]
        params = first["_model"].count_params()
        public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]

        summary: Dict[str, Any] = {
            "model": EXPERIMENT_NAME,
            "training_mode": args.training_mode,
            "dataset": args.dataset,
            "num_inputs": int(data["x_train"].shape[1]),
            "train_rows": int(len(data["x_train"])),
            "fit_rows": first["fit_rows"],
            "test_rows": int(len(data["x_test"])),
            "params": int(params),
            "seed": args.seed,
            "repeats": args.repeats,
            "train_rmse": _spread([r["train_rmse"] for r in public]),
            "test_rmse": _spread([r["test_rmse"] for r in public]),
            "fit_seconds": _spread([r["fit_seconds"] for r in public]),
            "closed_form_phase": (
                {key: _spread([r["closed_form_phase"][key] for r in public])
                 for key in ("train_rmse", "test_rmse")}
                if args.training_mode == "closed_form_then_backprop" else None),
            "best_checkpoint": first.get("best_checkpoint"),
            "input_importance": None if importance is None else [float(v) for v in importance],
            "repeat_results": public,
            "visualizations": figures,
        }
        write_summary_json(run_dir, summary)
        logger.info(
            f"Done: {args.training_mode} on {args.dataset}, test RMSE median "
            f"{summary['test_rmse']['median']:.6g} (IQR {summary['test_rmse']['iqr']:.3g}) "
            f"over {args.repeats} repeat(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------
