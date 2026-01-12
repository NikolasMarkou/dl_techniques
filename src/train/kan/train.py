"""
KAN Training & Visualization Script
====================================================================

This script demonstrates the complete workflow for training a Kolmogorov-Arnold
Network (KAN) on a synthetic regression task. It includes:
1. Synthetic data generation (y = sin(pi*x1) + x2^2).
2. Model creation using the factory pattern.
3. Grid adaptation (critical for KAN performance).
4. Training with a custom callback to keep grids updated.
5. Comprehensive visualization using the visualization module.

Usage:
    python train.py
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


# ---------------------------------------------------------------------
# Custom Data Structures
# ---------------------------------------------------------------------

@dataclass
class KANFunctionApproximation:
    """Data container for KAN function approximation visualization.

    Attributes:
        x1_grid: 2D meshgrid for x1 values.
        x2_grid: 2D meshgrid for x2 values.
        z_true: Ground truth function values.
        z_pred: Model predicted values.
        model_name: Name of the model.
    """

    x1_grid: np.ndarray
    x2_grid: np.ndarray
    z_true: np.ndarray
    z_pred: np.ndarray
    model_name: str = "KAN"


@dataclass
class KANSplineData:
    """Data container for KAN spline visualization.

    Attributes:
        layer: The KANLinear layer to visualize.
        input_dim: Number of input features.
        x_range: Input range for plotting.
        feature_names: Names for each input feature.
        expected_shapes: Expected function shapes for each input.
    """

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

        # Ground Truth
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(
            data.x1_grid, data.x2_grid, data.z_true,
            cmap='viridis', alpha=0.8
        )
        ax1.set_title(r'Ground Truth: $y = \sin(\pi x_1) + x_2^2$')
        ax1.set_xlabel(r'$x_1$')
        ax1.set_ylabel(r'$x_2$')

        # Prediction
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
    of the inputs. As weights change during training, these distributions shift.
    This callback ensures the grid adapts.

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
# Utilities
# ---------------------------------------------------------------------

def generate_data(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic regression data.

    Function: y = sin(pi * x1) + x2^2

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Tuple of input features and target values.
    """
    X = np.random.rand(num_samples, 2) * 2 - 1
    y = np.sin(np.pi * X[:, 0]) + np.square(X[:, 1])
    return X, y


def create_visualization_manager(experiment_name: str) -> VisualizationManager:
    """Creates and configures the visualization manager.

    Args:
        experiment_name: Name for the experiment.

    Returns:
        Configured VisualizationManager instance.
    """
    viz_manager = VisualizationManager(
        experiment_name=experiment_name,
        output_dir="kan_visualizations"
    )

    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("function_approximation", FunctionApproximationVisualization)
    viz_manager.register_template("kan_splines", KANSplineVisualization)

    return viz_manager


def plot_results(
    history: keras.callbacks.History,
    model: keras.Model,
    viz_manager: VisualizationManager
) -> None:
    """Visualizes training history, 3D prediction surface, and learned splines.

    Args:
        history: Keras training history object.
        model: Trained KAN model.
        viz_manager: Visualization manager instance.
    """
    logger.info("Generating visualizations...")

    # 1. Training History
    train_history = TrainingHistory(
        epochs=list(range(len(history.history['loss']))),
        train_loss=history.history['loss'],
        val_loss=history.history.get('val_loss'),
        train_metrics={'mae': history.history.get('mean_absolute_error', [])},
        val_metrics={'mae': history.history.get('val_mean_absolute_error', [])}
    )

    viz_manager.visualize(
        data=train_history,
        plugin_name="training_curves",
        smooth_factor=0.0,
        show=True
    )

    # 2. 3D Function Approximation Surface
    res = 50
    x1 = np.linspace(-1, 1, res)
    x2 = np.linspace(-1, 1, res)
    X1, X2 = np.meshgrid(x1, x2)
    grid_inputs = np.column_stack([X1.ravel(), X2.ravel()])

    Z_true = np.sin(np.pi * X1) + X2 ** 2
    Z_pred = model.predict(grid_inputs, verbose=0).reshape(res, res)

    approx_data = KANFunctionApproximation(
        x1_grid=X1,
        x2_grid=X2,
        z_true=Z_true,
        z_pred=Z_pred,
        model_name="KAN"
    )

    viz_manager.visualize(
        data=approx_data,
        plugin_name="function_approximation",
        show=True
    )

    # 3. KAN Spline Interpretability
    kan_layers = [layer for layer in model.layers if isinstance(layer, KANLinear)]
    if not kan_layers:
        logger.warning("No KANLinear layers found for spline visualization.")
        return

    layer_0 = kan_layers[0]
    spline_data = KANSplineData(
        layer=layer_0,
        input_dim=2,
        x_range=np.linspace(-1.5, 1.5, 100),
        feature_names=[r'Input 0 ($x_1$)', r'Input 1 ($x_2$)'],
        expected_shapes=['Sine Wave-like', 'Quadratic-like']
    )

    logger.info("Extracting learned activation functions from Layer 0...")
    viz_manager.visualize(
        data=spline_data,
        plugin_name="kan_splines",
        show=True
    )


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main() -> None:
    """Main training pipeline for KAN model."""
    logger.info("Initializing KAN Training Pipeline")

    # Configuration
    NUM_SAMPLES = 3000
    VAL_SAMPLES = 600
    EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-2

    # Data Generation
    logger.info(f"Generating {NUM_SAMPLES} training samples...")
    X_train, y_train = generate_data(NUM_SAMPLES)
    X_val, y_val = generate_data(VAL_SAMPLES)

    # Model Creation
    logger.info("Building KAN model...")
    model = create_kan_model(
        variant="small",
        input_features=2,
        output_features=1,
        override_config={"hidden_features": [16, 8]}
    )

    # Grid Initialization
    logger.info("Initializing B-spline grids with training data subset...")
    model.update_kan_grids(X_train[:200])

    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )
    model.summary()

    # Training
    logger.info("Starting training...")
    grid_updater = KANGridUpdateCallback(X_train[:500], update_freq=5)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[grid_updater],
        verbose=1
    )

    logger.info("Training complete.")

    # Evaluation & Visualization
    final_loss = history.history['val_loss'][-1]
    logger.info(f"Final Validation MSE: {final_loss:.6f}")

    viz_manager = create_visualization_manager("kan_regression")
    plot_results(history, model, viz_manager)


if __name__ == "__main__":
    main()