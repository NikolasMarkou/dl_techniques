"""
KAN Training & Visualization Script
====================================================================

Trains a Kolmogorov-Arnold Network (KAN) on a synthetic regression task
(y = sin(pi*x1) + x2^2) with grid adaptation, custom callbacks, and
comprehensive visualization of learned spline activation functions.

Usage:
    python train_kan.py
    python train_kan.py --epochs 300 --batch-size 256 --learning-rate 0.005
"""

import gc
import argparse
import keras
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

from dl_techniques.models.general_purpose.kan.model import create_kan_model
from dl_techniques.layers.ffn.kan_linear import KANLinear
from dl_techniques.utils.logger import logger
from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    TrainingHistory,
    TrainingCurvesVisualization,
)
from train.common import (
    setup_gpu,
    create_base_argument_parser,
    default_experiment_name,
    prepare_run_dir,
    save_training_history_json,
)
from train.common.callbacks import best_checkpoint_path


# Resolved once in `main()` to compute `run_dir` (plan-2026-09-17T064004-0d194df2
# D-001), which is then the SINGLE source both `create_visualization_manager()`
# (D-004) and `main()`'s post-training `model.save()` read for "the run's
# results directory" — so the two call sites cannot drift apart.
# `parents[3]` reaches the repo root from THIS file
# (src/train/kan/train_kan.py: [0] kan, [1] train, [2] src, [3] <repo>).
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "kan_regression"


# ---------------------------------------------------------------------
# Custom Data Structures
# ---------------------------------------------------------------------

@dataclass
class KANFunctionApproximation:
    """Data container for KAN function approximation visualization."""

    x1_grid: np.ndarray
    x2_grid: np.ndarray
    z_true: np.ndarray
    z_pred: np.ndarray
    model_name: str = "KAN"


@dataclass
class KANSplineData:
    """Data container for KAN spline visualization."""

    layer: KANLinear
    input_dim: int
    x_range: np.ndarray
    feature_names: List[str]
    expected_shapes: List[str]


# ---------------------------------------------------------------------
# Custom Visualization Plugins
# ---------------------------------------------------------------------

class FunctionApproximationVisualization(VisualizationPlugin):
    """Visualizes 3D function approximation comparing ground truth vs prediction."""

    @property
    def name(self) -> str:
        return "function_approximation"

    @property
    def description(self) -> str:
        return "3D surface plot comparing ground truth vs KAN prediction."

    def can_handle(self, data) -> bool:
        return isinstance(data, KANFunctionApproximation)

    def create_visualization(
        self,
        data: KANFunctionApproximation,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(
            data.x1_grid, data.x2_grid, data.z_true,
            cmap='viridis', alpha=0.8
        )
        ax1.set_title(r'Ground Truth: $y = \sin(\pi x_1) + x_2^2$')
        ax1.set_xlabel(r'$x_1$')
        ax1.set_ylabel(r'$x_2$')

        ax2 = fig.add_subplot(121, projection='3d')
        ax2.plot_surface(
            data.x1_grid, data.x2_grid, data.z_pred,
            cmap='plasma', alpha=0.8
        )
        ax2.set_title(f'{data.model_name} Prediction')
        ax2.set_xlabel(r'$x_1$')
        ax2.set_ylabel(r'$x_2$')

        fig.suptitle("Function Approximation Capability", fontsize=16)
        return fig


class KANSplineVisualization(VisualizationPlugin):
    """Visualizes learned spline activation functions from a KANLinear layer."""

    @property
    def name(self) -> str:
        return "kan_splines"

    @property
    def description(self) -> str:
        return "Visualizes learned activation functions from KANLinear layers."

    def can_handle(self, data) -> bool:
        return isinstance(data, KANSplineData)

    def create_visualization(
        self,
        data: KANSplineData,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> plt.Figure:
        layer = data.layer
        num_points = len(data.x_range)
        dummy_batch = np.zeros((num_points, data.input_dim), dtype="float32")

        fig, axes = plt.subplots(1, data.input_dim, figsize=(7 * data.input_dim, 5))
        if data.input_dim == 1:
            axes = [axes]

        for input_idx in range(data.input_dim):
            ax_plot = axes[input_idx]
            dummy_batch[:, :] = 0.0
            dummy_batch[:, input_idx] = data.x_range

            x_tensor = keras.ops.convert_to_tensor(dummy_batch, dtype=layer.dtype)
            basis_vals = layer._compute_bspline_basis(x_tensor)
            base_vals = layer.base_activation_fn(x_tensor)

            scalers = keras.ops.convert_to_numpy(layer.spline_scaler[input_idx, :])
            top_indices = np.argsort(np.abs(scalers))[-3:]

            for output_idx in top_indices:
                c_ijk = layer.spline_weight[input_idx, output_idx, :]
                w_spline = layer.spline_scaler[input_idx, output_idx]
                w_base = layer.base_scaler[input_idx, output_idx]

                b_i = basis_vals[:, input_idx, :]
                spline_part = keras.ops.einsum('bk,k->b', b_i, c_ijk)
                base_part = base_vals[:, input_idx]
                y_plot = (w_base * base_part) + (w_spline * spline_part)
                y_plot_np = keras.ops.convert_to_numpy(y_plot)

                ax_plot.plot(
                    data.x_range, y_plot_np,
                    alpha=0.7, linewidth=2, label=f'To Neuron {output_idx}'
                )

            ax_plot.set_title(
                f'{data.feature_names[input_idx]}\nExpected: {data.expected_shapes[input_idx]}'
            )
            ax_plot.set_xlabel('Input Value')
            ax_plot.set_ylabel('Activation Output')
            ax_plot.legend()
            ax_plot.grid(True)

        fig.suptitle(
            "Interpretability: Visualizing Learned Activation Functions",
            fontsize=16
        )
        return fig


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------

class KANGridUpdateCallback(keras.callbacks.Callback):
    """Updates the B-spline grids periodically during training.

    KANs perform best when the grid range matches the activation distribution
    of the inputs. This callback ensures the grid adapts as weights change.

    Args:
        x_data: Input data for grid updates.
        update_freq: Update frequency in epochs.
    """

    def __init__(self, x_data: np.ndarray, update_freq: int = 5) -> None:
        super().__init__()
        self.x_data = x_data
        self.update_freq = update_freq

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if (epoch + 1) % self.update_freq == 0:
            logger.info(f"Epoch {epoch + 1}: Updating B-spline grids...")
            self.model.update_kan_grids(self.x_data)


class KANVisualizationCallback(keras.callbacks.Callback):
    """Per-epoch cheap loss dashboard + periodic expensive function/spline grid.

    # DECISION plan-2026-09-17T064004-0d194df2/D-002
    Structurally mirrors `DenoisingVisualizationCallback`
    (src/train/bfunet/common.py:1610-1889): a single-use abstraction within this file
    (one instantiation site in `main()`, wired in Step 6) admitted as a charge against
    the 2-max complexity budget specifically because it MIRRORS an already-proven,
    already-amortized repo-wide pattern rather than inventing a new one -- do not
    generalize this into a shared cross-trainer base class on the strength of "it looks
    reusable"; python-software.md Sec.B.16 rules that out for a training script with one
    real consumer. See decisions.md D-002 for the full trade-off.

    Every `on_epoch_end` call re-renders a cheap combined loss/val_loss dashboard via
    the existing `TrainingCurvesVisualization` plugin (through `viz_manager`, never a
    hand-rolled plot). Every `freq` epochs it additionally re-renders the more
    expensive `function_approximation` + `kan_splines` grid via
    `render_function_and_spline_grid()` -- the SAME helper `plot_results()` uses for its
    final post-hoc render, so there is exactly one code path for that grid, called from
    two places (periodic-during-training and final-post-hoc). Both renders are wrapped
    in `try/except Exception` + `logger.warning` so a rendering failure never aborts
    training (verified by Step 8's deliberate-exception fail-soft check, not merely
    assumed from reading the `try/except`).
    """

    def __init__(
        self,
        viz_manager: VisualizationManager,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        freq: int = 5,
    ) -> None:
        """
        Args:
            viz_manager: Registered `VisualizationManager` (see
                `create_visualization_manager`) -- the SAME instance `plot_results()`
                uses at the end of `main()`, so periodic and final renders land in the
                same `run_dir / "visualizations"` directory.
            X_train: Training inputs. Accepted for interface parity with
                `DenoisingVisualizationCallback` (which stores a fixed eval batch) and
                for a future data-dependent render; the current
                `render_function_and_spline_grid()` helper evaluates on a synthetic
                mesh grid (matching `plot_results()`'s pre-existing behavior) and a
                fixed `KANLinear` input range, so `X_train`/`y_train` are not read by
                this callback today.
            y_train: Training targets. See `X_train`.
            freq: Epoch cadence for the expensive function/spline grid render. Every
                epoch still gets the cheap loss dashboard regardless of `freq`.
        """
        super().__init__()
        self.viz_manager = viz_manager
        self.X_train = X_train
        self.y_train = y_train
        self.freq = max(1, int(freq))
        self._hist = {"epoch": [], "loss": [], "val_loss": []}

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Record per-epoch scalars, re-render the cheap dashboard every epoch, and
        the expensive function/spline grid every `self.freq` epochs."""
        logs = logs or {}
        self._hist["epoch"].append(epoch + 1)
        self._hist["loss"].append(logs.get("loss", float("nan")))
        self._hist["val_loss"].append(logs.get("val_loss", float("nan")))

        try:
            train_history = TrainingHistory(
                epochs=self._hist["epoch"],
                train_loss=self._hist["loss"],
                val_loss=self._hist["val_loss"],
            )
            self.viz_manager.visualize(
                data=train_history, plugin_name="training_curves",
                smooth_factor=0.0, show=False,
            )
        except Exception as e:  # visualization must never break training
            logger.warning(f"Per-epoch dashboard render failed at epoch {epoch + 1}: {e}")

        if (epoch + 1) % self.freq != 0:
            return
        try:
            render_function_and_spline_grid(self.model, self.viz_manager, show=False)
        except Exception as e:  # visualization must never break training
            logger.warning(
                f"Periodic function/spline grid render failed at epoch {epoch + 1}: {e}"
            )
        finally:
            gc.collect()


# ---------------------------------------------------------------------
# Data Generation (synthetic function approximation)
# ---------------------------------------------------------------------

def generate_data(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic regression data: y = sin(pi * x1) + x2^2.

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Tuple of input features (N, 2) and target values (N,).
    """
    X = np.random.rand(num_samples, 2) * 2 - 1
    y = np.sin(np.pi * X[:, 0]) + np.square(X[:, 1])
    return X, y


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def create_visualization_manager(run_dir: Path) -> VisualizationManager:
    """Creates visualization manager with KAN-specific plugins."""
    # DECISION plan-2026-09-17T064004-0d194df2/D-004
    # Supersedes plan-2026-09-17T052443-2c932602/D-002, which anchored
    # `output_dir` at the bare "results" root and passed a non-empty
    # `experiment_name` to avoid double-nesting under THAT plan's split
    # layout (a separately-timestamped `results/kan_regression/<ts>/` dir,
    # disconnected from the model/config artifacts). Since Step 1/2 of this
    # plan unified everything under one `run_dir` (already unique: it embeds
    # its own timestamp via `default_experiment_name()`), the only way to
    # land PNGs at exactly `run_dir / "visualizations" / <name>.png` with no
    # second nested subdirectory is to make BOTH of
    # `VisualizationContext.get_save_path()`'s own extra path segments
    # (`dl_techniques/visualization/core.py:206-217`) inert:
    #   save_dir = output_dir
    #   if experiment_name: save_dir = save_dir / experiment_name   # skip
    #   if timestamp:       save_dir = save_dir / timestamp          # skip
    # `experiment_name=""` is falsy so the first append is skipped;
    # `VisualizationManager.__init__`'s `timestamp` parameter defaults to
    # `None`, which lets `VisualizationContext`'s own dataclass default
    # MINT A SECOND, DIFFERENT timestamp (`datetime.now()` at construction
    # time, not `run_dir`'s) unless overridden. Passing `timestamp=""`
    # explicitly (its documented case: "save directly into `output_dir`
    # without a timestamp subdirectory") skips the second append too. Do NOT
    # drop either kwarg, and do NOT re-add a non-empty `experiment_name` or
    # rely on `timestamp`'s default — either alone reintroduces exactly the
    # double-nesting hazard this decision exists to close (see decisions.md
    # D-004).
    viz_manager = VisualizationManager(
        experiment_name="",
        output_dir=run_dir / "visualizations",
        timestamp="",
    )
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("function_approximation", FunctionApproximationVisualization)
    viz_manager.register_template("kan_splines", KANSplineVisualization)
    return viz_manager


def render_function_and_spline_grid(
    model: keras.Model,
    viz_manager: VisualizationManager,
    show: bool = False,
) -> None:
    """Renders the 3D function-approximation surface + KAN spline interpretability grid.

    Shared data-construction+render helper for the `function_approximation` and
    `kan_splines` plugins (plan-2026-09-17T064004-0d194df2/D-002, Step 5): extracted out of
    `plot_results()` so `KANVisualizationCallback`'s periodic expensive render and the
    final post-hoc `plot_results()` call both go through ONE code path instead of two
    copies of this grid-construction logic drifting apart.

    Interface contract (2 call sites: `plot_results()` and `KANVisualizationCallback`):
        Args:
            model: A `KANLinear`-containing Keras model, trained or mid-training. Must
                support `model.predict(...)` and expose `model.layers`.
            viz_manager: A `VisualizationManager` with `"function_approximation"` and
                `"kan_splines"` plugins already registered (see
                `create_visualization_manager`).
            show: Forwarded verbatim to `viz_manager.visualize(show=...)`.
        Returns:
            None. Side effect only: writes PNGs via `viz_manager`. Logs a warning and
            returns early (skipping the spline render only) if `model` has no
            `KANLinear` layers yet -- this is expected mid-training before the first
            `update_kan_grids()` call has run, so it is not raised as an exception.
        Failure mode: propagates any exception from `model.predict()` or
            `viz_manager.visualize()` to the caller -- callers that must not let a
            render failure abort training (e.g. `KANVisualizationCallback`) are
            responsible for their own `try/except`.
    """
    # 3D function approximation surface
    res = 50
    x1 = np.linspace(-1, 1, res)
    x2 = np.linspace(-1, 1, res)
    X1, X2 = np.meshgrid(x1, x2)
    grid_inputs = np.column_stack([X1.ravel(), X2.ravel()])

    Z_true = np.sin(np.pi * X1) + X2 ** 2
    Z_pred = model.predict(grid_inputs, verbose=0).reshape(res, res)

    viz_manager.visualize(
        data=KANFunctionApproximation(
            x1_grid=X1, x2_grid=X2,
            z_true=Z_true, z_pred=Z_pred,
        ),
        plugin_name="function_approximation", show=show,
    )

    # KAN spline interpretability
    kan_layers = [l for l in model.layers if isinstance(l, KANLinear)]
    if not kan_layers:
        logger.warning("No KANLinear layers found for spline visualization.")
        return

    logger.info("Extracting learned activation functions from Layer 0...")
    viz_manager.visualize(
        data=KANSplineData(
            layer=kan_layers[0],
            input_dim=2,
            x_range=np.linspace(-1.5, 1.5, 100),
            feature_names=[r'Input 0 ($x_1$)', r'Input 1 ($x_2$)'],
            expected_shapes=['Sine Wave-like', 'Quadratic-like'],
        ),
        plugin_name="kan_splines", show=show,
    )


def plot_results(
    history: keras.callbacks.History,
    model: keras.Model,
    viz_manager: VisualizationManager,
    show: bool = True,
) -> None:
    """Visualizes training history, 3D prediction surface, and learned splines."""
    logger.info("Generating visualizations...")

    # 1. Training curves
    train_history = TrainingHistory(
        epochs=list(range(len(history.history['loss']))),
        train_loss=history.history['loss'],
        val_loss=history.history.get('val_loss'),
        train_metrics={'mae': history.history.get('mean_absolute_error', [])},
        val_metrics={'mae': history.history.get('val_mean_absolute_error', [])}
    )
    viz_manager.visualize(
        data=train_history, plugin_name="training_curves",
        smooth_factor=0.0, show=show,
    )

    # 2-3. 3D function approximation surface + KAN spline interpretability
    render_function_and_spline_grid(model, viz_manager, show=show)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

# DECISION plan-2026-09-17T052443-2c932602/D-005
# `--dataset`/`--patience` are dead in THIS script but live in several other
# Pattern-1 trainers that share `create_base_argument_parser()`
# (train_capsnet.py, train_convnext_v1.py, train_convnext_v2.py). Do NOT
# "fix" this by deleting the two `add_argument()` calls from
# `create_base_argument_parser()` itself -- that would silently break every
# other caller's live `--dataset`/`--patience` consumption. Instead, this
# script's own parser INSTANCE is pruned post-construction, by `dest`, via
# `argparse`'s private action-removal bookkeeping (no public API for this
# exists). See decisions.md D-005 for the full reasoning and the rejected
# alternatives (parser.add_argument(..., help=SUPPRESS) only hides the flag
# from --help, it does not stop the flag from being accepted and silently
# ignored -- which is exactly the dead-knob problem this step closes).
def _drop_base_parser_args(parser: argparse.ArgumentParser, *dests: str) -> None:
    """Removes CLI arguments this script inherits from
    `create_base_argument_parser()` but never reads, per `src/train/CLAUDE.md`'s
    "config fields must be live" convention.

    `--dataset` and `--patience` are dead here: `train_kan.py`'s data is 100%
    synthetic (`generate_data()` never reads `args.dataset`), and Step 4
    (D-004) deliberately used a plain `model.save()` instead of
    `create_callbacks()`'s EarlyStopping, so there is no patience knob to
    gate. `create_base_argument_parser()` itself is left untouched -- these
    two flags are live for the OTHER Pattern-1 trainers that share it (e.g.
    `train_capsnet.py`, `train_convnext_v1.py`); only THIS script's own
    parser instance is pruned, post-construction, by `dest`.

    Args:
        parser: An `argparse.ArgumentParser` already populated via
            `create_base_argument_parser()` (or any parser using standard
            `add_argument`/action-group bookkeeping).
        *dests: One or more `dest` names (e.g. `"dataset"`) to remove.

    Returns:
        None. Mutates `parser` in place.
    """
    for dest in dests:
        for action in list(parser._actions):
            if action.dest != dest:
                continue
            parser._actions.remove(action)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
            for opt in action.option_strings:
                parser._option_string_actions.pop(opt, None)


def main() -> None:
    """Main training pipeline for KAN model."""
    parser = create_base_argument_parser(
        description="Train a KAN model on synthetic function approximation",
        default_dataset="synthetic",
        dataset_choices=["synthetic"],
    )
    parser.add_argument('--num-samples', type=int, default=3000,
                        help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=600,
                        help='Number of validation samples')
    parser.add_argument('--variant', type=str, default='small',
                        choices=['micro', 'small', 'medium', 'large', 'xlarge'],
                        help='KAN model variant')
    parser.add_argument('--hidden-features', type=int, nargs='+', default=[16, 8],
                        help='Hidden layer feature sizes')
    parser.add_argument('--grid-update-freq', type=int, default=5,
                        help='Grid update frequency in epochs')
    parser.add_argument('--viz-freq', type=int, default=5,
                        help='Periodic (expensive) function/spline visualization '
                             'frequency in epochs, for KANVisualizationCallback')
    # `--image-size`/`--weight-decay`/`--lr-schedule` joined `--dataset`/
    # `--patience` (D-005) as confirmed-dead here: `generate_data()` has no
    # notion of an image size, and `model.compile()` above uses a plain
    # `keras.optimizers.Adam(learning_rate=...)` with no weight-decay term
    # and no schedule object at all. Same reasoning as D-005 applies
    # unchanged (see decisions.md D-006): these three remain live for other
    # Pattern-1 trainers sharing `create_base_argument_parser()`, so only
    # THIS script's own parser instance drops them.
    # `--show-plots` joins the dead-flag set (plan-2026-09-17T064004-0d194df2
    # D-003, Resolution (Step 7)): `matplotlib.use('Agg')` runs at module
    # import, before `args.show_plots` is ever read, so no interactive
    # display can ever occur regardless of this flag's value -- forwarding
    # `show=args.show_plots` to `viz_manager.visualize()` was removed (now
    # unconditionally `show=False`), which makes the flag itself as dead as
    # `--dataset`/`--patience`/etc above. Dropped via the SAME
    # `_drop_base_parser_args()` mechanism for consistency with D-005/D-006,
    # rather than kept as an accepted-but-warned no-op.
    _drop_base_parser_args(
        parser, "dataset", "patience", "image_size", "weight_decay",
        "lr_schedule", "show_plots",
    )
    # DECISION plan-2026-09-17T052443-2c932602/D-006
    # KAN's own preferred defaults (200 epochs / batch 128 / lr 1e-2) differ
    # from `create_base_argument_parser()`'s general defaults (100 / 64 /
    # 1e-3). Do NOT reintroduce the previous "if args.X == <old default>:
    # args.X = <new default>" pattern here -- it cannot distinguish "user
    # never passed --batch-size" from "user explicitly passed --batch-size
    # 64", so an explicit `--batch-size 64` was silently clobbered to 128
    # (confirmed live: this is why the Step 1 baseline run's progress bar
    # showed 2 steps/epoch for 256 samples, not the expected 4 -- see
    # decisions.md D-006). `set_defaults()` changes what argparse falls back
    # to when a flag is ABSENT, so an explicit CLI value of any kind
    # (including one that happens to equal the base parser's old default) is
    # honored correctly.
    parser.set_defaults(epochs=200, batch_size=128, learning_rate=1e-2)
    args = parser.parse_args()

    setup_gpu(args.gpu)

    # Unified run directory (Step 1 of the bfunet-output-layout normalization
    # plan): ONE run directory at results/<prefix>_<variant>_<timestamp>/,
    # replacing the two disconnected paths this file used to write (a flat,
    # overwritten results/kan_regression/kan_model.keras plus a separately-
    # timestamped results/kan_regression/<ts>/ viz dir). `args` is passed
    # directly as `prepare_run_dir`'s `config` -- `save_config_json` falls
    # back to `vars(config)` for anything exposing `__dict__`, which an
    # `argparse.Namespace` has, so no new config dataclass is needed here.
    # Step 2 wires the model/history artifacts into `run_dir`; Step 3
    # re-points visualization output at `run_dir / "visualizations"`.
    run_dir = REPO_ROOT / "results" / default_experiment_name(EXPERIMENT_NAME, args.variant)
    prepare_run_dir(args, output_dir=run_dir)
    logger.info(f"Run directory: {run_dir}")

    logger.info("Initializing KAN Training Pipeline")

    # Data generation (synthetic)
    logger.info(f"Generating {args.num_samples} training samples...")
    X_train, y_train = generate_data(args.num_samples)
    X_val, y_val = generate_data(args.val_samples)

    # Model creation
    logger.info("Building KAN model...")
    model = create_kan_model(
        variant=args.variant,
        input_features=2,
        output_features=1,
        override_config={"hidden_features": args.hidden_features}
    )

    # Grid initialization
    logger.info("Initializing B-spline grids with training data subset...")
    model.update_kan_grids(X_train[:200])

    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )
    model.summary()

    # Visualization manager -- created BEFORE `fit()` (not after, as in the
    # pre-Step-6 layout) so the SAME instance is shared by
    # `KANVisualizationCallback` (periodic, during training) and the final
    # post-hoc `plot_results()` call below: both renders land in the same
    # `run_dir / "visualizations"` directory via one `VisualizationManager`.
    viz_manager = create_visualization_manager(run_dir)

    # Training
    logger.info("Starting training...")
    # DECISION plan-2026-09-17T064004-0d194df2/D-001
    # `ModelCheckpoint`/`CSVLogger` are added directly here, NOT via
    # `train.common.create_callbacks()`. The prior plan's D-004 (superseded
    # by this one) rejected `create_callbacks()` wholesale because its
    # defaults (`monitor='val_accuracy'`, `include_analyzer=True`) target
    # Pattern-1 classification; this plan's D-001 narrows that to: the
    # DEFAULTS were the problem, not the function, but a customized
    # `create_callbacks(monitor='val_loss', include_analyzer=False, ...)`
    # call would still pull in `EarlyStopping`/`patience` surface area that
    # D-005 (prior plan) already pruned as dead for this short-smoke-scale
    # trainer. So checkpoint/logging wiring is hand-rolled from the lower-
    # level `best_checkpoint_path()` primitive instead, matching bfunet's
    # artifact names (`best_model.keras`, `training_log.csv`) without
    # reopening the EarlyStopping question. See decisions.md D-001.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            KANGridUpdateCallback(X_train[:500], update_freq=args.grid_update_freq),
            keras.callbacks.ModelCheckpoint(
                best_checkpoint_path(str(run_dir)),
                monitor='val_loss',
                save_best_only=True,
            ),
            keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
            KANVisualizationCallback(
                viz_manager, X_train, y_train, freq=args.viz_freq,
            ),
        ],
        verbose=1
    )

    logger.info("Training complete.")
    final_loss = history.history['val_loss'][-1]
    logger.info(f"Final Validation MSE: {final_loss:.6f}")
    save_training_history_json(history, run_dir)

    # Persist the final (not necessarily best) model so a run leaves an
    # inspectable artifact even when `save_best_only=True` never fired (e.g.
    # a run with no improving epoch). Named `final_model.keras` to sit
    # alongside `best_model.keras` in the unified `run_dir`, matching the
    # bfunet convention (see decisions.md D-001).
    model_path = run_dir / "final_model.keras"
    model.save(model_path)
    logger.info(f"Saved trained model to {model_path}")

    # Visualization: final, definitive-quality post-hoc render. Reuses the
    # SAME `viz_manager` instance `KANVisualizationCallback` used during
    # training (created above, before `fit()`), so periodic and final
    # renders share one `run_dir / "visualizations"` directory.
    # DECISION plan-2026-09-17T064004-0d194df2/D-003
    # `show=False` unconditionally, never `args.show_plots` -- see
    # decisions.md D-003's Resolution (Step 7): `--show-plots` is dropped
    # from this script's parser entirely (below, via
    # `_drop_base_parser_args`), so `args.show_plots` no longer exists here.
    plot_results(history, model, viz_manager, show=False)


if __name__ == "__main__":
    main()
