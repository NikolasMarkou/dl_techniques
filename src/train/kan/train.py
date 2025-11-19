"""
KAN Training & Visualization Script
====================================================================

This script demonstrates the complete workflow for training a Kolmogorov-Arnold
Network (KAN) on a synthetic regression task. It includes:
1. Synthetic data generation (y = sin(pi*x1) + x2^2).
2. Model creation using the factory pattern.
3. Grid adaptation (critical for KAN performance).
4. Training with a custom callback to keep grids updated.
5. Comprehensive visualization of the learned activation functions.

Usage:
    python train.py
"""

import keras
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.models.kan.model import create_kan_model
from dl_techniques.layers.kan_linear import KANLinear
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Utilities & Callbacks
# ---------------------------------------------------------------------

class KANGridUpdateCallback(keras.callbacks.Callback):
    """
    Updates the B-spline grids periodically during training.

    KANs perform best when the grid range matches the activation distribution
    of the inputs. As weights change during training, these distributions shift.
    This callback ensures the grid adapts.
    """

    def __init__(self, x_data: np.ndarray, update_freq: int = 5):
        super().__init__()
        self.x_data = x_data
        self.update_freq = update_freq

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if (epoch + 1) % self.update_freq == 0:
            logger.info(f"Epoch {epoch + 1}: Updating B-spline grids...")
            # The model method handles the forward pass to hidden layers
            self.model.update_kan_grids(self.x_data)


def generate_data(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic regression data.
    Function: y = sin(pi * x1) + x2^2
    """
    # Input in range [-1, 1]
    X = np.random.rand(num_samples, 2) * 2 - 1

    # Target function
    # Feature 0: Sine wave
    # Feature 1: Quadratic
    y = np.sin(np.pi * X[:, 0]) + np.square(X[:, 1])

    return X, y


# ---------------------------------------------------------------------
# Visualization Logic
# ---------------------------------------------------------------------

def plot_results(history: keras.callbacks.History, model: keras.Model) -> None:
    """
    Visualizes training history, 3D prediction surface, and learned splines.
    """
    logger.info("Generating visualizations...")
    plt.style.use('seaborn-v0_8-darkgrid')

    # --- 1. Training History ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history.history['loss'], label='Train Loss (MSE)')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Val Loss (MSE)')
    ax1.set_title('Training Dynamics')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    plt.show()

    # --- 2. 3D Function Approximation Surface ---
    # Create a grid for visualization
    res = 50
    x1 = np.linspace(-1, 1, res)
    x2 = np.linspace(-1, 1, res)
    X1, X2 = np.meshgrid(x1, x2)

    # Flatten for prediction
    grid_inputs = np.column_stack([X1.ravel(), X2.ravel()])

    # Ground Truth
    Z_true = np.sin(np.pi * X1) + X2 ** 2

    # KAN Prediction
    Z_pred = model.predict(grid_inputs, verbose=0).reshape(res, res)

    fig2 = plt.figure(figsize=(14, 6))

    # Plot Ground Truth
    ax2 = fig2.add_subplot(122, projection='3d')
    ax2.plot_surface(X1, X2, Z_true, cmap='viridis', alpha=0.8)
    ax2.set_title('Ground Truth\n$y = \sin(\pi x_1) + x_2^2$')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')

    # Plot Prediction
    ax3 = fig2.add_subplot(121, projection='3d')
    ax3.plot_surface(X1, X2, Z_pred, cmap='plasma', alpha=0.8)
    ax3.set_title('KAN Prediction')
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')

    fig2.suptitle("Function Approximation Capability", fontsize=16)
    plt.show()

    # --- 3. Interpretability: Visualizing Learned Splines ---
    # We examine the first KAN layer to see if it decoupled the function correctly.
    # Expected: Input 0 -> Sine shape, Input 1 -> Quadratic shape

    kan_layers = [l for l in model.layers if isinstance(l, KANLinear)]
    if not kan_layers:
        logger.warning("No KANLinear layers found for visualization.")
        return

    layer_0 = kan_layers[0]
    input_dim = 2

    # Create a test range for plotting the spline shape
    num_points = 100
    x_range = np.linspace(-1.5, 1.5, num_points)

    # We need to query the layer carefully. The layer expects input (Batch, Features).
    # We create a dummy batch where we vary one feature and keep others 0.
    dummy_batch = np.zeros((num_points, input_dim), dtype="float32")

    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    feature_names = ['Input 0 ($x_1$)', 'Input 1 ($x_2$)']
    expected_shapes = ['Sine Wave-like', 'Quadratic-like']

    logger.info("Extracting learned activation functions from Layer 0...")

    for input_idx in range(input_dim):
        ax = axes[input_idx]

        # Set the column of interest to the range, reset others
        dummy_batch[:, :] = 0.0
        dummy_batch[:, input_idx] = x_range

        # Convert to tensor
        x_tensor = keras.ops.convert_to_tensor(dummy_batch, dtype=layer_0.dtype)

        # 1. Compute B-Spline Basis for this input
        # Shape: (num_points, input_features, num_basis)
        basis_vals = layer_0._compute_bspline_basis(x_tensor)

        # 2. Compute Base Activation
        # Shape: (num_points, input_features)
        base_vals = layer_0.base_activation_fn(x_tensor)

        # Find the most significant output connections for this input
        # Scalers shape: (input_features, output_features)
        scalers = keras.ops.convert_to_numpy(layer_0.spline_scaler[input_idx, :])
        # Get indices of top 3 strongest connections
        top_indices = np.argsort(np.abs(scalers))[-3:]

        for output_idx in top_indices:
            # --- Reconstruction of phi_ij(x) ---

            # Get Weights
            # c_ijk: (num_basis,)
            c_ijk = layer_0.spline_weight[input_idx, output_idx, :]
            w_spline = layer_0.spline_scaler[input_idx, output_idx]
            w_base = layer_0.base_scaler[input_idx, output_idx]

            # Calculate Spline Component for specific feature
            # Slice basis for specific input: (num_points, num_basis)
            b_i = basis_vals[:, input_idx, :]

            # Dot product: sum_k (c_ijk * B_k(x))
            spline_part = keras.ops.einsum('bk,k->b', b_i, c_ijk)

            # Calculate Base Component
            base_part = base_vals[:, input_idx]

            # Combine
            y_plot = (w_base * base_part) + (w_spline * spline_part)

            # Convert to numpy for plotting
            y_plot_np = keras.ops.convert_to_numpy(y_plot)

            ax.plot(x_range, y_plot_np, alpha=0.7, linewidth=2,
                    label=f'To Neuron {output_idx}')

        ax.set_title(f'{feature_names[input_idx]}\nExpected: {expected_shapes[input_idx]}')
        ax.set_xlabel('Input Value')
        ax.set_ylabel('Activation Output')
        ax.legend()
        ax.grid(True)

    fig3.suptitle("Interpretability: Visualizing Learned Activation Functions", fontsize=16)
    plt.show()


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    logger.info("Initializing KAN Training Pipeline")

    # 1. Configuration
    NUM_SAMPLES = 3000
    VAL_SAMPLES = 600
    EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-2  # KANs often benefit from slightly higher LR

    # 2. Data Generation
    logger.info(f"Generating {NUM_SAMPLES} training samples...")
    X_train, y_train = generate_data(NUM_SAMPLES)
    X_val, y_val = generate_data(VAL_SAMPLES)

    # 3. Model Creation
    # "Small" variant: [64, 32, 16] hidden units.
    # We reduce it slightly for this simple regression task to avoid overfitting
    # and make visualizations cleaner.
    logger.info("Building KAN model...")
    model = create_kan_model(
        variant="small",
        input_features=2,
        output_features=1,
        # Override to make it slightly smaller for this specific demo
        override_config={"hidden_features": [16, 8]}
    )

    # 4. Grid Initialization (CRITICAL STEP)
    # Adapts the B-spline knots to the input distribution [-1, 1]
    logger.info("Initializing B-spline grids with training data subset...")
    model.update_kan_grids(X_train[:200])

    # 5. Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    model.summary()

    # 6. Training
    logger.info("Starting training...")

    # Callback to update grids every 5 epochs using a subset of data
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

    # 7. Evaluation & Visualization
    final_loss = history.history['val_loss'][-1]
    logger.info(f"Final Validation MSE: {final_loss:.6f}")

    plot_results(history, model)


if __name__ == "__main__":
    main()