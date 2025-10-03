"""
Simple MNIST Classification with Soft Self-Organizing Map Features

This script demonstrates a straightforward application of the SoftSOMLayer for
MNIST digit classification. The SOM layer learns a topologically organized
representation of digit features, which is then used for classification.

Architecture:
    Input (28x28) → Flatten → Dense(256) → SoftSOM(10x10) → Dense(10) → Softmax

The SoftSOM layer creates a 10x10 grid of prototype vectors, organizing similar
digit features into nearby grid locations. This provides both good classification
performance and interpretable spatial organization of learned features.

Key Features:
    - Simple end-to-end training with backpropagation
    - Real-time visualization of SOM activation patterns
    - Analysis of which digits activate which grid regions
    - Monitoring of topological organization quality
    - Comprehensive visualizations: U-Matrix, hit counts, trajectories, etc.

Usage:
    python train_mnist_softsom.py --epochs 20 --batch-size 128
"""

import gc
import json
import keras
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.som_nd_soft_layer import SoftSOMLayer


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MNISTSOMConfig:
    """
    Configuration for MNIST + SoftSOM classification training.

    Attributes:
        grid_shape: Shape of SOM grid (height, width)
        som_temperature: Softmax temperature for SOM assignments
        som_reconstruction_weight: Weight for SOM reconstruction loss
        som_topological_weight: Weight for topological preservation
        som_sharpness_weight: Weight for encouraging sharp assignments

        dense_units: Number of units in dense layer before SOM
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate

        output_dir: Directory for saving results
        experiment_name: Unique experiment identifier
        monitor_every_n_epochs: Frequency for visualization updates
    """

    # SOM Configuration
    grid_shape: Tuple[int, int] = (10, 10)
    som_temperature: float = 0.5
    som_reconstruction_weight: float = 1.0
    som_topological_weight: float = 0.2
    som_sharpness_weight: float = 0.05

    # Network Architecture
    dense_units: int = 256

    # Training Configuration
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3

    # Output Configuration
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    monitor_every_n_epochs: int = 2

    def __post_init__(self) -> None:
        """Generate experiment name if not provided."""
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"mnist_softsom_{timestamp}"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mnist_data(config: MNISTSOMConfig) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray]
]:
    """
    Load and preprocess MNIST dataset.

    Args:
        config: Training configuration

    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test)) where:
        - x data is normalized to [0, 1] and flattened to vectors
        - y data is one-hot encoded
    """
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] range
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten images to vectors
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    logger.info(f"Training samples: {x_train.shape[0]}")
    logger.info(f"Test samples: {x_test.shape[0]}")
    logger.info(f"Input dimension: {x_train.shape[1]}")

    return (x_train, y_train), (x_test, y_test)


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_mnist_som_classifier(config: MNISTSOMConfig) -> keras.Model:
    """
    Create MNIST classifier with SoftSOM feature extraction layer.

    Architecture:
        Input(784) → Dense(256, ReLU) → SoftSOM(10x10) → Dense(10, Softmax)

    The SoftSOM layer learns a topologically organized grid of prototype
    vectors, where similar digit features are mapped to nearby grid locations.

    Args:
        config: Model configuration

    Returns:
        Compiled Keras model
    """
    logger.info("Creating MNIST + SoftSOM classification model...")

    inputs = keras.Input(shape=(784,), name="input_images")

    # Feature extraction with dense layer
    x = keras.layers.Dense(
        config.dense_units,
        activation='relu',
        name="dense_features"
    )(inputs)

    # Soft Self-Organizing Map layer for topological feature organization
    som_features = SoftSOMLayer(
        grid_shape=config.grid_shape,
        input_dim=config.dense_units,
        temperature=config.som_temperature,
        use_per_dimension_softmax=True,
        use_reconstruction_loss=True,
        reconstruction_weight=config.som_reconstruction_weight,
        topological_weight=config.som_topological_weight,
        sharpness_weight=config.som_sharpness_weight,
        name="soft_som"
    )(x)

    # Classification head
    outputs = keras.layers.Dense(
        10,
        activation='softmax',
        name="digit_classifier"
    )(som_features)

    model = keras.Model(inputs, outputs, name="mnist_softsom_classifier")

    logger.info("Model architecture:")
    model.summary()

    return model


# =============================================================================
# VISUALIZATION AND MONITORING
# =============================================================================

class SOMVisualizationCallback(keras.callbacks.Callback):
    """
    Comprehensive callback for visualizing SOM activation patterns and organization.

    Creates visualizations including:
    - Activation maps for each digit class
    - Best Matching Unit (BMU) distribution across classes
    - Prototype vector visualizations (learned codebook)
    - U-Matrix (unified distance matrix for topology quality)
    - Hit count maps (usage frequency per neuron)
    - Confusion analysis on the SOM grid
    - Training trajectory tracking for sample digits
    - Quantization and topological error metrics
    """

    def __init__(
            self,
            config: MNISTSOMConfig,
            x_test: np.ndarray,
            y_test: np.ndarray
    ) -> None:
        """
        Initialize visualization callback.

        Args:
            config: Training configuration
            x_test: Test images for visualization
            y_test: Test labels (one-hot encoded)
        """
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs

        # Select subset of test data for visualization
        self.x_viz = x_test[:1000]
        self.y_viz = np.argmax(y_test[:1000], axis=1)

        # Select specific samples to track over time (one per digit)
        self.tracked_samples = []
        self.tracked_labels = []
        for digit in range(10):
            digit_mask = self.y_viz == digit
            digit_indices = np.where(digit_mask)[0]
            if len(digit_indices) > 0:
                self.tracked_samples.append(self.x_viz[digit_indices[0]])
                self.tracked_labels.append(digit)
        self.tracked_samples = np.array(self.tracked_samples)

        # Track BMU trajectories over epochs
        self.bmu_trajectories: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {digit: [] for digit in range(10)}

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name / "som_visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track metrics over time
        self.quantization_errors: List[float] = []
        self.topological_errors: List[float] = []
        self.epochs_recorded: List[int] = []

        logger.info(f"SOM visualization callback initialized with {len(self.x_viz)} test samples")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Create comprehensive SOM visualizations at specified intervals.

        Args:
            epoch: Current epoch number
            logs: Training logs
        """
        if (epoch + 1) % self.monitor_freq != 0:
            return

        logger.info(f"Creating SOM visualizations for epoch {epoch + 1}...")

        try:
            # Get SOM layer and models
            som_layer = self.model.get_layer("soft_som")

            # Create feature extraction model up to SOM input
            feature_model = keras.Model(
                self.model.input,
                som_layer.input
            )

            # Extract features for visualization data
            features = feature_model.predict(self.x_viz, verbose=0, batch_size=256)

            # Get soft assignments from SOM
            assignments = som_layer.get_soft_assignments(features)
            assignments = keras.ops.convert_to_numpy(assignments)

            # Get SOM weights (prototype vectors)
            weights_map = som_layer.get_weights_map()
            weights_map = keras.ops.convert_to_numpy(weights_map)

            # Create all visualizations
            self._visualize_class_activation_maps(assignments, epoch + 1)
            self._visualize_bmu_distribution(assignments, epoch + 1)
            self._visualize_prototype_vectors(weights_map, epoch + 1)
            self._visualize_u_matrix(weights_map, epoch + 1)
            self._visualize_hit_counts(assignments, epoch + 1)
            self._visualize_confusion_on_grid(assignments, epoch + 1)

            # Track metrics and trajectories
            self._compute_and_track_metrics(features, weights_map, assignments, epoch + 1)
            self._track_sample_trajectories(epoch + 1)

            # Create summary visualizations
            if len(self.quantization_errors) > 1:
                self._plot_error_metrics(epoch + 1)
                self._visualize_bmu_trajectories(epoch + 1)

            # Cleanup
            del features, assignments, weights_map
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to create SOM visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_class_activation_maps(
            self,
            assignments: np.ndarray,
            epoch: int
    ) -> None:
        """
        Create activation heatmaps showing which grid regions respond to each digit.

        Args:
            assignments: Soft assignment arrays [N, grid_h, grid_w]
            epoch: Current epoch number
        """
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'SOM Activation Maps by Digit Class - Epoch {epoch}', fontsize=16)

        for digit in range(10):
            ax = axes[digit // 5, digit % 5]

            # Get assignments for this digit class
            digit_mask = (self.y_viz == digit)
            digit_assignments = assignments[digit_mask]

            # Average activation across all samples of this digit
            mean_activation = np.mean(digit_assignments, axis=0)

            # Plot heatmap
            im = ax.imshow(mean_activation, cmap='hot', interpolation='nearest')
            ax.set_title(f'Digit {digit}', fontsize=14)
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = self.output_dir / f"epoch_{epoch:03d}_activation_maps.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_bmu_distribution(
            self,
            assignments: np.ndarray,
            epoch: int
    ) -> None:
        """
        Visualize the distribution of Best Matching Units across digit classes.

        Args:
            assignments: Soft assignment arrays [N, grid_h, grid_w]
            epoch: Current epoch number
        """
        # Find BMU for each sample
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        # Convert flat indices to 2D coordinates
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))
        bmu_coords = np.stack(bmu_coords, axis=1)

        # Create figure with subplots for each digit
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'BMU Distribution by Digit Class - Epoch {epoch}', fontsize=16)

        for digit in range(10):
            ax = axes[digit // 5, digit % 5]

            # Get BMU coordinates for this digit
            digit_mask = (self.y_viz == digit)
            digit_bmus = bmu_coords[digit_mask]

            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(
                digit_bmus[:, 0],
                digit_bmus[:, 1],
                bins=[grid_h, grid_w],
                range=[[0, grid_h], [0, grid_w]]
            )

            # Plot heatmap
            im = ax.imshow(H, cmap='viridis', interpolation='nearest', origin='lower')
            ax.set_title(f'Digit {digit} (n={np.sum(digit_mask)})', fontsize=12)
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = self.output_dir / f"epoch_{epoch:03d}_bmu_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_prototype_vectors(
            self,
            weights_map: np.ndarray,
            epoch: int
    ) -> None:
        """
        Visualize the learned prototype vectors as pseudo-images.

        Each SOM grid cell learns a 256-dimensional prototype. We project it back
        to understand what patterns each neuron has learned to represent.

        Args:
            weights_map: SOM weight matrix [grid_h, grid_w, input_dim]
            epoch: Current epoch number
        """
        grid_h, grid_w, input_dim = weights_map.shape

        # Create visualization showing what each grid cell "looks like"
        fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 1.5, grid_h * 1.5))
        fig.suptitle(f'SOM Prototype Vectors (Learned Codebook) - Epoch {epoch}', fontsize=16)

        # Normalize each prototype for better visualization
        for i in range(grid_h):
            for j in range(grid_w):
                if grid_h > 1 and grid_w > 1:
                    ax = axes[i, j]
                elif grid_h > 1:
                    ax = axes[i]
                elif grid_w > 1:
                    ax = axes[j]
                else:
                    ax = axes

                # Get prototype vector and normalize for visualization
                prototype = weights_map[i, j, :]
                prototype_normalized = (prototype - prototype.min()) / (prototype.max() - prototype.min() + 1e-8)

                # Reshape to image-like representation (16x16 for 256-dim)
                grid_size = int(np.sqrt(input_dim))
                if grid_size * grid_size == input_dim:
                    prototype_img = prototype_normalized.reshape(grid_size, grid_size)
                else:
                    # If not perfect square, pad to nearest square
                    pad_size = int(np.ceil(np.sqrt(input_dim))) ** 2
                    padded = np.pad(prototype_normalized, (0, pad_size - input_dim), mode='constant')
                    grid_size = int(np.sqrt(pad_size))
                    prototype_img = padded.reshape(grid_size, grid_size)

                ax.imshow(prototype_img, cmap='viridis', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'{i},{j}', fontsize=6)

        plt.tight_layout()
        save_path = self.output_dir / f"epoch_{epoch:03d}_prototypes.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_u_matrix(
            self,
            weights_map: np.ndarray,
            epoch: int
    ) -> None:
        """
        Create U-Matrix (Unified Distance Matrix) visualization.

        The U-Matrix shows the average distance between each neuron and its
        neighbors. High values (bright) indicate cluster boundaries, low values
        (dark) indicate cluster centers. This reveals the topological quality.

        Args:
            weights_map: SOM weight matrix [grid_h, grid_w, input_dim]
            epoch: Current epoch number
        """
        grid_h, grid_w, input_dim = weights_map.shape
        u_matrix = np.zeros((grid_h, grid_w))

        # Compute average distance to neighbors for each cell
        for i in range(grid_h):
            for j in range(grid_w):
                distances = []
                current = weights_map[i, j, :]

                # Check all 8 neighbors (4-connected and diagonal)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_h and 0 <= nj < grid_w:
                            neighbor = weights_map[ni, nj, :]
                            dist = np.linalg.norm(current - neighbor)
                            distances.append(dist)

                u_matrix[i, j] = np.mean(distances) if distances else 0

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(u_matrix, cmap='bone', interpolation='nearest')
        ax.set_title(f'U-Matrix (Topological Quality) - Epoch {epoch}\n'
                    f'Dark=Cluster Centers, Bright=Boundaries', fontsize=14)
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')

        # Add grid lines
        for i in range(grid_h + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(grid_w + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        plt.colorbar(im, ax=ax, label='Average Distance to Neighbors')
        plt.tight_layout()
        save_path = self.output_dir / f"epoch_{epoch:03d}_umatrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_hit_counts(
            self,
            assignments: np.ndarray,
            epoch: int
    ) -> None:
        """
        Visualize hit count map showing neuron usage frequency.

        Shows how many samples are mapped to each grid cell (BMU).
        Reveals if all neurons are being used or if some are "dead".

        Args:
            assignments: Soft assignment arrays [N, grid_h, grid_w]
            epoch: Current epoch number
        """
        # Find BMU for each sample
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        # Convert to grid coordinates
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        # Create hit count matrix
        hit_counts = np.zeros((grid_h, grid_w))
        for i, j in zip(bmu_coords[0], bmu_coords[1]):
            hit_counts[i, j] += 1

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Linear scale
        im1 = ax1.imshow(hit_counts, cmap='YlOrRd', interpolation='nearest')
        ax1.set_title(f'Hit Counts (Linear Scale) - Epoch {epoch}', fontsize=14)
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')

        # Add count annotations
        for i in range(grid_h):
            for j in range(grid_w):
                count = int(hit_counts[i, j])
                ax1.text(j, i, str(count), ha='center', va='center',
                        color='white' if count > hit_counts.max() * 0.5 else 'black',
                        fontsize=8)

        plt.colorbar(im1, ax=ax1, label='Number of Samples')

        # Log scale for better visibility of small counts
        hit_counts_log = np.log10(hit_counts + 1)
        im2 = ax2.imshow(hit_counts_log, cmap='YlOrRd', interpolation='nearest')
        ax2.set_title(f'Hit Counts (Log Scale) - Epoch {epoch}', fontsize=14)
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')
        plt.colorbar(im2, ax=ax2, label='Log10(Samples + 1)')

        # Compute statistics
        total_neurons = grid_h * grid_w
        active_neurons = np.sum(hit_counts > 0)
        dead_neurons = total_neurons - active_neurons
        max_hits = int(hit_counts.max())
        mean_hits = hit_counts.mean()

        fig.suptitle(f'Active: {active_neurons}/{total_neurons} neurons | '
                    f'Dead: {dead_neurons} | Max: {max_hits} | Mean: {mean_hits:.1f}',
                    fontsize=12)

        plt.tight_layout()
        save_path = self.output_dir / f"epoch_{epoch:03d}_hit_counts.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_confusion_on_grid(
            self,
            assignments: np.ndarray,
            epoch: int
    ) -> None:
        """
        Visualize which digit classes get confused and where on the grid.

        For each grid cell, shows the distribution of digit classes mapped to it.
        Helps identify regions where different digits overlap.

        Args:
            assignments: Soft assignment arrays [N, grid_h, grid_w]
            epoch: Current epoch number
        """
        # Find BMU for each sample
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        # Convert to grid coordinates
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        # Create confusion map: for each cell, count samples per class
        confusion_grid = np.zeros((grid_h, grid_w, 10))
        for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1])):
            label = self.y_viz[idx]
            confusion_grid[i, j, label] += 1

        # Create visualization
        fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 1.2, grid_h * 1.2))
        fig.suptitle(f'Class Distribution per Neuron - Epoch {epoch}', fontsize=16)

        for i in range(grid_h):
            for j in range(grid_w):
                if grid_h > 1 and grid_w > 1:
                    ax = axes[i, j]
                elif grid_h > 1:
                    ax = axes[i]
                elif grid_w > 1:
                    ax = axes[j]
                else:
                    ax = axes

                # Get class distribution for this cell
                class_counts = confusion_grid[i, j, :]
                total = class_counts.sum()

                if total > 0:
                    # Plot pie chart of class distribution
                    colors = plt.cm.tab10(np.arange(10))
                    non_zero = class_counts > 0
                    ax.pie(class_counts[non_zero], colors=colors[non_zero],
                          startangle=90, counterclock=False)

                    # Annotate with dominant class
                    dominant_class = np.argmax(class_counts)
                    purity = class_counts[dominant_class] / total
                    ax.text(0, -1.3, f'{dominant_class}\n({purity:.1%})',
                           ha='center', va='top', fontsize=6,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax.text(0, 0, 'Empty', ha='center', va='center', fontsize=8)

                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.5, 1.2)
                ax.axis('off')

        plt.tight_layout()
        save_path = self.output_dir / f"epoch_{epoch:03d}_confusion_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _compute_and_track_metrics(
            self,
            features: np.ndarray,
            weights_map: np.ndarray,
            assignments: np.ndarray,
            epoch: int
    ) -> None:
        """
        Compute and track quantization and topological errors over time.

        Args:
            features: Latent features [N, input_dim]
            weights_map: SOM weights [grid_h, grid_w, input_dim]
            assignments: Soft assignments [N, grid_h, grid_w]
            epoch: Current epoch number
        """
        # Quantization Error: average distance from input to its BMU
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        quantization_error = 0.0
        for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1])):
            bmu_prototype = weights_map[i, j, :]
            dist = np.linalg.norm(features[idx] - bmu_prototype)
            quantization_error += dist
        quantization_error /= len(features)

        # Topological Error: fraction of samples where BMU and 2nd BMU are not neighbors
        topological_error = 0.0
        for idx in range(len(features)):
            # Get distances to all prototypes
            distances = flat_assignments[idx]
            sorted_indices = np.argsort(-distances)  # Sort by distance (descending)

            # Get BMU and 2nd BMU coordinates
            bmu1_idx = sorted_indices[0]
            bmu2_idx = sorted_indices[1]

            bmu1_coords = np.unravel_index(bmu1_idx, (grid_h, grid_w))
            bmu2_coords = np.unravel_index(bmu2_idx, (grid_h, grid_w))

            # Check if they are neighbors (Manhattan distance <= 1)
            manhattan_dist = abs(bmu1_coords[0] - bmu2_coords[0]) + abs(bmu1_coords[1] - bmu2_coords[1])
            if manhattan_dist > 1:
                topological_error += 1
        topological_error /= len(features)

        self.quantization_errors.append(quantization_error)
        self.topological_errors.append(topological_error)
        self.epochs_recorded.append(epoch)

        logger.info(f"Epoch {epoch} - Quantization Error: {quantization_error:.4f}, "
                   f"Topological Error: {topological_error:.4f}")

    def _track_sample_trajectories(self, epoch: int) -> None:
        """
        Track BMU positions for specific sample digits over training.

        Args:
            epoch: Current epoch number
        """
        try:
            # Get feature model
            som_layer = self.model.get_layer("soft_som")
            feature_model = keras.Model(self.model.input, som_layer.input)

            # Extract features for tracked samples
            features = feature_model.predict(self.tracked_samples, verbose=0, batch_size=32)

            # Get assignments
            assignments = som_layer.get_soft_assignments(features)
            assignments = keras.ops.convert_to_numpy(assignments)

            # Find BMU for each tracked sample
            flat_assignments = assignments.reshape(assignments.shape[0], -1)
            bmu_indices = np.argmax(flat_assignments, axis=1)

            grid_h, grid_w = assignments.shape[1], assignments.shape[2]
            bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

            # Store trajectories
            for idx, label in enumerate(self.tracked_labels):
                coord = (int(bmu_coords[0][idx]), int(bmu_coords[1][idx]))
                self.bmu_trajectories[label].append((epoch, coord))

        except Exception as e:
            logger.warning(f"Failed to track sample trajectories: {e}")

    def _plot_error_metrics(self, epoch: int) -> None:
        """
        Plot quantization and topological errors over training.

        Args:
            epoch: Current epoch number
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Quantization error
        ax1.plot(self.epochs_recorded, self.quantization_errors, 'b-', linewidth=2, marker='o')
        ax1.set_title('Quantization Error over Training', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average Distance to BMU')
        ax1.grid(True, alpha=0.3)

        # Topological error
        ax2.plot(self.epochs_recorded, self.topological_errors, 'r-', linewidth=2, marker='s')
        ax2.set_title('Topological Error over Training', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Fraction of Non-Neighbor BMU Pairs')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        save_path = self.output_dir / f"epoch_{epoch:03d}_error_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_bmu_trajectories(self, epoch: int) -> None:
        """
        Visualize how BMU positions evolve over training for tracked samples.

        Shows the "path" each sample takes on the SOM grid during training,
        revealing how the organization stabilizes.

        Args:
            epoch: Current epoch number
        """
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'BMU Trajectories During Training (up to Epoch {epoch})', fontsize=16)

        grid_h, grid_w = self.config.grid_shape

        for digit in range(10):
            ax = axes[digit // 5, digit % 5]

            # Create empty grid
            ax.set_xlim(-0.5, grid_w - 0.5)
            ax.set_ylim(-0.5, grid_h - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Digit {digit}', fontsize=12)
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            ax.grid(True, alpha=0.3)

            # Plot trajectory if available
            if digit in self.bmu_trajectories and len(self.bmu_trajectories[digit]) > 0:
                trajectory = self.bmu_trajectories[digit]
                epochs_list = [t[0] for t in trajectory]
                coords = [t[1] for t in trajectory]

                # Extract x and y coordinates
                x_coords = [c[1] for c in coords]
                y_coords = [c[0] for c in coords]

                # Plot trajectory with color gradient (blue to red = early to late)
                for i in range(len(x_coords) - 1):
                    color = plt.cm.coolwarm(i / max(1, len(x_coords) - 1))
                    ax.plot(x_coords[i:i+2], y_coords[i:i+2], 'o-',
                           color=color, linewidth=2, markersize=6, alpha=0.7)

                # Mark start and end
                ax.plot(x_coords[0], y_coords[0], 'go', markersize=12,
                       label='Start', markeredgecolor='black', markeredgewidth=2)
                ax.plot(x_coords[-1], y_coords[-1], 'r*', markersize=16,
                       label='Current', markeredgecolor='black', markeredgewidth=2)

                ax.legend(loc='upper right', fontsize=8)
            else:
                ax.text(grid_w / 2, grid_h / 2, 'No trajectory data',
                       ha='center', va='center', fontsize=10)

            ax.invert_yaxis()

        plt.tight_layout()
        save_path = self.output_dir / f"epoch_{epoch:03d}_trajectories.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()


# =============================================================================
# TRAINING ORCHESTRATION
# =============================================================================

def train_mnist_som_classifier(config: MNISTSOMConfig) -> keras.Model:
    """
    Train MNIST classifier with SoftSOM feature extraction.

    Args:
        config: Training configuration

    Returns:
        Trained Keras model
    """
    logger.info(f"Starting MNIST + SoftSOM training: {config.experiment_name}")

    # Setup output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data(config)

    # Create model
    model = create_mnist_som_classifier(config)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Setup callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            str(output_dir / "training_log.csv"),
            append=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "tensorboard"),
            histogram_freq=1
        ),
        SOMVisualizationCallback(config, x_test, y_test)
    ]

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate final model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Final test accuracy: {test_acc:.4f}")

    # Save training history
    history_path = output_dir / "training_history.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Save final metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'total_params': int(model.count_params())
    }
    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training completed successfully!")
    return model


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MNIST classifier with SoftSOM layer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--grid-size', type=int, default=10, help='SOM grid size (square)')
    parser.add_argument('--som-temperature', type=float, default=0.5, help='SOM softmax temperature')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    return parser.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main training function."""
    args = parse_arguments()

    config = MNISTSOMConfig(
        grid_shape=(args.grid_size, args.grid_size),
        som_temperature=args.som_temperature,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )

    logger.info("=== MNIST + SoftSOM Classification ===")
    logger.info(f"SOM Grid Shape: {config.grid_shape}")
    logger.info(f"SOM Temperature: {config.som_temperature}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Epochs: {config.epochs}")

    model = train_mnist_som_classifier(config)
    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()