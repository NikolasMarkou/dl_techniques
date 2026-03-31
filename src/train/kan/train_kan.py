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

import keras
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from dataclasses import dataclass

from dl_techniques.models.kan.model import create_kan_model
from dl_techniques.layers.kan_linear import KANLinear
from dl_techniques.utils.logger import logger
from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    TrainingHistory,
    TrainingCurvesVisualization,
)
from train.common import setup_gpu, create_base_argument_parser


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

def create_visualization_manager(experiment_name: str) -> VisualizationManager:
    """Creates visualization manager with KAN-specific plugins."""
    viz_manager = VisualizationManager(
        experiment_name=experiment_name,
        output_dir="results"
    )
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("function_approximation", FunctionApproximationVisualization)
    viz_manager.register_template("kan_splines", KANSplineVisualization)
    return viz_manager


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

    # 2. 3D function approximation surface
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

    # 3. KAN spline interpretability
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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
                        choices=['tiny', 'small', 'medium', 'large'],
                        help='KAN model variant')
    parser.add_argument('--hidden-features', type=int, nargs='+', default=[16, 8],
                        help='Hidden layer feature sizes')
    parser.add_argument('--grid-update-freq', type=int, default=5,
                        help='Grid update frequency in epochs')
    args = parser.parse_args()

    # Override defaults for KAN
    if args.epochs == 100:
        args.epochs = 200
    if args.batch_size == 64:
        args.batch_size = 128
    if args.learning_rate == 1e-3:
        args.learning_rate = 1e-2

    setup_gpu(args.gpu)

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

    # Training
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            KANGridUpdateCallback(X_train[:500], update_freq=args.grid_update_freq),
        ],
        verbose=1
    )

    logger.info("Training complete.")
    final_loss = history.history['val_loss'][-1]
    logger.info(f"Final Validation MSE: {final_loss:.6f}")

    # Visualization
    viz_manager = create_visualization_manager("kan_regression")
    plot_results(history, model, viz_manager, show=args.show_plots)


if __name__ == "__main__":
    main()
