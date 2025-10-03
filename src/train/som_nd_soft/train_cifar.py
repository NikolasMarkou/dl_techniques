"""
Complex CIFAR Autoencoder with Soft Self-Organizing Map Bottleneck

This module implements a sophisticated autoencoder architecture using the SoftSOMLayer
as a topological bottleneck. The SOM enforces structured organization in the latent
space, creating a continuous, topologically-ordered representation of CIFAR images.

Architecture:
    Encoder: Conv layers → Reshape
    Bottleneck: TimeDistributed(SoftSOM Layer) on spatial feature vectors
    Decoder: Reshape → TransposeConv layers

The SOM bottleneck creates a grid of prototype vectors that organize the latent
representations spatially, enabling:
    - Structured latent space exploration via grid traversal
    - Semantic interpolation between image concepts
    - Topologically ordered feature clustering
    - Visualization of learned image manifold structure

Key Features:
    1. **Multi-Scale Training**: Progressive resolution training for stability
    2. **Advanced Augmentation**: Cutout, mixup, and color jittering
    3. **Comprehensive Visualizations**: U-Matrix, hit counts, trajectories, etc.
    4. **Latent Space Visualization**: t-SNE and grid activation heatmaps
    5. **Manifold Walking**: Interpolation along SOM grid paths
    6. **Quality Metrics**: Quantization and topological error tracking

Usage:
    python train_cifar.py --epochs 100 --grid-size 16
"""

import gc
import json
import argparse
import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
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
class CIFARSOMConfig:
    """
    Comprehensive configuration for CIFAR + SoftSOM autoencoder training.
    """

    # === Data Configuration ===
    dataset_name: str = 'cifar10'
    image_size: int = 32
    use_data_augmentation: bool = True
    augmentation_strength: float = 0.5

    # === Architecture Configuration ===
    encoder_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])

    grid_shape: Tuple[int, int] = (16, 16)
    som_temperature: float = 1.0
    som_use_per_dim_softmax: bool = True
    som_reconstruction_weight: float = 0.0
    som_topological_weight: float = 1.0
    som_sharpness_weight: float = 0.0

    decoder_filters: List[int] = field(default_factory=lambda: [256, 128, 64])
    use_batch_norm: bool = True
    use_dropout: bool = True
    dropout_rate: float = 0.3

    # === Training Configuration ===
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    lr_schedule: str = 'cosine'
    optimizer_type: str = 'adamw'
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    # === Progressive Training ===
    use_progressive_training: bool = False
    progressive_epochs: List[int] = field(
        default_factory=lambda: [20, 30, 50])
    initial_size: int = 16

    # === Loss Configuration ===
    reconstruction_loss_type: str = 'mse'
    perceptual_weight: float = 0.5

    # === Monitoring Configuration ===
    monitor_every_n_epochs: int = 5
    num_reconstruction_samples: int = 16
    num_interpolation_samples: int = 8
    save_latent_projections: bool = True

    # === Output Configuration ===
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10

    def __post_init__(self) -> None:
        """Validate configuration and generate experiment name."""
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"cifar_softsom_ae_{timestamp}"

        if self.dataset_name not in ['cifar10', 'cifar100']:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


# =============================================================================
# DATA LOADING AND AUGMENTATION
# =============================================================================

def load_cifar_data(config: CIFARSOMConfig) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray]
]:
    """
    Load and preprocess CIFAR dataset.

    Args:
        config: Training configuration

    Returns:
        Tuple of ((x_train, y_train), (x_test, y_test)) normalized to [0, 1]
    """
    logger.info(f"Loading {config.dataset_name} dataset...")

    if config.dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.info(f"Training samples: {x_train.shape[0]}")
    logger.info(f"Test samples: {x_test.shape[0]}")
    logger.info(f"Image shape: {x_train.shape[1:]}")

    return (x_train, y_train), (x_test, y_test)


def create_augmentation_layer(config: CIFARSOMConfig) -> keras.layers.Layer:
    """
    Create data augmentation preprocessing layer.

    Args:
        config: Training configuration

    Returns:
        Sequential layer with augmentation operations
    """
    strength = config.augmentation_strength

    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1 * strength),
        keras.layers.RandomZoom(0.1 * strength),
        keras.layers.RandomTranslation(0.1 * strength, 0.1 * strength),
        keras.layers.RandomBrightness(0.2 * strength),
        keras.layers.RandomContrast(0.2 * strength),
    ], name="augmentation")

    return augmentation


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def create_encoder(config: CIFARSOMConfig) -> keras.Model:
    """
    Create convolutional encoder network that preserves spatial information.
    """
    inputs = keras.Input(
        shape=(config.image_size, config.image_size, 3),
        name="image_input"
    )
    x = inputs

    for i, (filters, kernel_size) in enumerate(
            zip(config.encoder_filters, config.encoder_kernel_sizes)
    ):
        x = keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=2,
            padding='same',
            name=f"encoder_conv_{i + 1}"
        )(x)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization(name=f"encoder_bn_{i + 1}")(x)
        x = keras.layers.Activation('relu', name=f"encoder_relu_{i + 1}")(x)
        if config.use_dropout:
            x = keras.layers.Dropout(
                config.dropout_rate, name=f"encoder_dropout_{i + 1}"
            )(x)

    return keras.Model(inputs, x, name="encoder")


def create_decoder(config: CIFARSOMConfig) -> keras.Model:
    """
    Create convolutional decoder network with transpose convolutions.
    """
    final_spatial_size = config.image_size // (2 ** len(config.encoder_filters))
    input_channels = config.encoder_filters[-1]

    inputs = keras.Input(
        shape=(final_spatial_size, final_spatial_size, input_channels),
        name="latent_map_input"
    )
    x = inputs

    for i, filters in enumerate(config.decoder_filters):
        x = keras.layers.Conv2DTranspose(
            filters,
            kernel_size=3,
            strides=2,
            padding='same',
            name=f"decoder_deconv_{i + 1}"
        )(x)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization(name=f"decoder_bn_{i + 1}")(x)
        x = keras.layers.Activation('relu', name=f"decoder_relu_{i + 1}")(x)
        if config.use_dropout and i < len(config.decoder_filters) - 1:
            x = keras.layers.Dropout(
                config.dropout_rate, name=f"decoder_dropout_{i + 1}"
            )(x)

    outputs = keras.layers.Conv2D(
        3,
        kernel_size=3,
        padding='same',
        activation='sigmoid',
        name="reconstructed_image"
    )(x)
    return keras.Model(inputs, outputs, name="decoder")


def create_cifar_som_autoencoder(config: CIFARSOMConfig) -> keras.Model:
    """
    Create complete autoencoder with SoftSOM topological bottleneck.
    """
    logger.info("Creating CIFAR SoftSOM Autoencoder...")

    encoder = create_encoder(config)
    decoder = create_decoder(config)

    inputs = keras.Input(
        shape=(config.image_size, config.image_size, 3),
        name="input_images"
    )

    feature_map = encoder(inputs)
    map_shape = keras.ops.shape(feature_map)
    h, w, channels = map_shape[1], map_shape[2], map_shape[3]

    reshaped_features = keras.layers.Reshape((-1, channels))(feature_map)

    som_layer = SoftSOMLayer(
        grid_shape=config.grid_shape,
        input_dim=channels,
        temperature=config.som_temperature,
        use_per_dimension_softmax=config.som_use_per_dim_softmax,
        use_reconstruction_loss=config.som_reconstruction_weight > 0,
        reconstruction_weight=config.som_reconstruction_weight,
        topological_weight=config.som_topological_weight,
        sharpness_weight=config.som_sharpness_weight,
        name="soft_som_bottleneck"
    )
    time_distributed_som = keras.layers.TimeDistributed(
        som_layer, name="time_distributed_som"
    )
    som_features = time_distributed_som(reshaped_features)

    som_map = keras.layers.Reshape((h, w, channels))(som_features)
    reconstructed = decoder(som_map)

    autoencoder = keras.Model(
        inputs, reconstructed, name="cifar_softsom_autoencoder"
    )

    logger.info("Autoencoder architecture:")
    autoencoder.summary()
    return autoencoder


# =============================================================================
# ADVANCED VISUALIZATION
# =============================================================================

class ComprehensiveMonitoringCallback(keras.callbacks.Callback):
    """
    Advanced monitoring callback with comprehensive visualizations.
    """

    def __init__(
            self,
            config: CIFARSOMConfig,
            x_test: np.ndarray,
            y_test: np.ndarray
    ) -> None:
        """
        Initialize comprehensive monitoring callback.
        """
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs

        self.x_viz = x_test[:config.num_reconstruction_samples]
        self.y_viz = y_test[:config.num_reconstruction_samples].flatten()
        self.x_latent_analysis = x_test[:1000]
        self.y_latent_analysis = y_test[:1000].flatten()

        num_classes = 10 if config.dataset_name == 'cifar10' else 100
        num_classes_to_track = min(10, num_classes)
        self.tracked_samples = []
        self.tracked_labels = []
        for class_id in range(num_classes_to_track):
            class_mask = self.y_latent_analysis == class_id
            class_indices = np.where(class_mask)[0]
            if len(class_indices) > 0:
                self.tracked_samples.append(
                    self.x_latent_analysis[class_indices[0]]
                )
                self.tracked_labels.append(class_id)
        self.tracked_samples = np.array(self.tracked_samples)

        self.bmu_trajectories: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {
            label: [] for label in self.tracked_labels
        }

        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.viz_dir = self.output_dir / "visualizations"
        self.metrics_dir = self.output_dir / "metrics"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_metrics: Dict[str, List[float]] = {
            'reconstruction_mse': [], 'psnr': [],
            'quantization_error': [], 'topological_error': []
        }
        self.epochs_recorded: List[int] = []
        logger.info(f"Monitoring initialized with {len(self.x_viz)} samples")

    def on_epoch_end(
            self,
            epoch: int,
            logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create comprehensive visualizations and metrics.
        """
        if (epoch + 1) % self.monitor_freq != 0:
            return

        logger.info(f"Visualizing for epoch {epoch + 1}...")

        try:
            reconstructed = self.model.predict(
                self.x_viz, verbose=0, batch_size=32
            )

            wrapper_layer = self.model.get_layer("time_distributed_som")
            som_layer = wrapper_layer.layer
            encoder = self.model.get_layer("encoder")
            decoder = self.model.get_layer("decoder")

            feature_map = encoder.predict(
                self.x_latent_analysis, verbose=0, batch_size=128
            )
            features = feature_map.reshape(-1, som_layer.input_dim)
            h, w = feature_map.shape[1], feature_map.shape[2]
            vectors_per_image = h * w

            expanded_y = np.repeat(self.y_latent_analysis, vectors_per_image)

            assignments_list = []
            inference_batch_size = 512
            for i in range(0, len(features), inference_batch_size):
                batch = features[i:i + inference_batch_size]
                assignments_list.append(som_layer.get_soft_assignments(batch))
            assignments = np.concatenate(assignments_list, axis=0)
            assignments = keras.ops.convert_to_numpy(assignments)

            weights_map = keras.ops.convert_to_numpy(som_layer.get_weights_map())

            self._compute_and_store_metrics(
                reconstructed, features, weights_map, assignments, epoch + 1
            )
            self._visualize_reconstructions(reconstructed, epoch + 1)
            self._visualize_som_activations(assignments, expanded_y, epoch + 1)
            self._visualize_prototype_vectors(weights_map, epoch + 1)
            self._visualize_u_matrix(weights_map, epoch + 1)
            self._visualize_hit_counts(assignments, epoch + 1)
            self._visualize_confusion_on_grid(assignments, expanded_y, epoch + 1)
            self._visualize_grid_interpolations(
                decoder, weights_map, h, w, epoch + 1
            )
            self._track_sample_trajectories(encoder, som_layer, epoch + 1)

            if len(self.epochs_recorded) > 1:
                self._plot_metrics_curves(epoch + 1)
                self._visualize_bmu_trajectories(epoch + 1)

            if (self.config.save_latent_projections and
                    (epoch + 1) % (self.monitor_freq * 2) == 0):
                self._visualize_latent_space(features, expanded_y, epoch + 1)

            del reconstructed, features, assignments, weights_map, expanded_y
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _compute_and_store_metrics(
            self, reconstructed: np.ndarray, features: np.ndarray,
            weights_map: np.ndarray, assignments: np.ndarray, epoch: int
    ) -> None:
        """Compute and store reconstruction quality and SOM metrics."""
        mse = np.mean((self.x_viz - reconstructed) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))

        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        quant_error = 0.0
        for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1])):
            dist = np.linalg.norm(features[idx] - weights_map[i, j, :])
            quant_error += dist
        quant_error /= len(features)

        topo_error = 0.0
        for idx in range(len(features)):
            sorted_indices = np.argsort(-flat_assignments[idx])
            bmu1_idx, bmu2_idx = sorted_indices[0], sorted_indices[1]
            bmu1_coords = np.unravel_index(bmu1_idx, (grid_h, grid_w))
            bmu2_coords = np.unravel_index(bmu2_idx, (grid_h, grid_w))
            dist = abs(bmu1_coords[0] - bmu2_coords[0]) + \
                abs(bmu1_coords[1] - bmu2_coords[1])
            if dist > 1:
                topo_error += 1
        topo_error /= len(features)

        self.epoch_metrics['reconstruction_mse'].append(float(mse))
        self.epoch_metrics['psnr'].append(float(psnr))
        self.epoch_metrics['quantization_error'].append(float(quant_error))
        self.epoch_metrics['topological_error'].append(float(topo_error))
        self.epochs_recorded.append(epoch)

        logger.info(f"Epoch {epoch} - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
        logger.info(f"Epoch {epoch} - Quant Error: {quant_error:.4f}, "
                    f"Topo Error: {topo_error:.4f}")

    def _visualize_reconstructions(
            self, reconstructed: np.ndarray, epoch: int
    ) -> None:
        """Create side-by-side comparison of original and reconstructed."""
        n_samples = len(self.x_viz)
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows * 2, n_cols, figsize=(n_cols * 3, n_rows * 6)
        )
        fig.suptitle(f'Image Reconstructions - Epoch {epoch}', fontsize=16)

        for i in range(n_samples):
            row, col = (i // n_cols) * 2, i % n_cols
            ax_orig = axes[row, col] if n_rows > 1 else axes[0, col]
            ax_recon = axes[row + 1, col] if n_rows > 1 else axes[1, col]

            ax_orig.imshow(self.x_viz[i])
            ax_orig.set_title(f'Original\nClass: {self.y_viz[i]}', fontsize=10)
            ax_orig.axis('off')

            ax_recon.imshow(np.clip(reconstructed[i], 0, 1))
            mse = np.mean((self.x_viz[i] - reconstructed[i]) ** 2)
            ax_recon.set_title(f'Reconstructed\nMSE: {mse:.4f}', fontsize=10)
            ax_recon.axis('off')

        for i in range(n_samples, n_rows * n_cols):
            row, col = (i // n_cols) * 2, i % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
                axes[row + 1, col].axis('off')

        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_reconstructions.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_som_activations(
            self, assignments: np.ndarray, y_expanded: np.ndarray, epoch: int
    ) -> None:
        """Visualize SOM activation patterns for different image classes."""
        try:
            unique_classes = np.unique(self.y_latent_analysis)
            n_classes = min(10, len(unique_classes))
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            title = f'SOM Activation Heatmaps by Class - Epoch {epoch}'
            fig.suptitle(title, fontsize=16)

            for idx, class_label in enumerate(unique_classes[:n_classes]):
                ax = axes.flatten()[idx]
                class_mask = (y_expanded == class_label)
                if np.any(class_mask):
                    mean_activation = np.mean(assignments[class_mask], axis=0)
                    im = ax.imshow(mean_activation, cmap='hot')
                    ax.set_title(f'Class {class_label}')
                    plt.colorbar(im, ax=ax)
                else:
                    ax.set_title(f'Class {class_label} (No Data)')
                ax.axis('off')

            plt.tight_layout()
            save_path = self.viz_dir / f"epoch_{epoch:03d}_som_activations.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to visualize SOM activations: {e}")

    def _visualize_prototype_vectors(
            self, weights_map: np.ndarray, epoch: int
    ) -> None:
        """Visualize learned prototype vectors as pseudo-images."""
        grid_h, grid_w, input_dim = weights_map.shape
        fig, axes = plt.subplots(
            grid_h, grid_w, figsize=(grid_w * 1.5, grid_h * 1.5)
        )
        title = f'SOM Prototype Vectors (Learned Codebook) - Epoch {epoch}'
        fig.suptitle(title, fontsize=16)

        for i in range(grid_h):
            for j in range(grid_w):
                ax = axes[i, j] if grid_h > 1 and grid_w > 1 else axes
                prototype = weights_map[i, j, :]
                prototype_norm = (prototype - prototype.min()) / \
                    (prototype.max() - prototype.min() + 1e-8)
                pad_size = int(np.ceil(np.sqrt(input_dim))) ** 2
                padded = np.pad(
                    prototype_norm, (0, pad_size - input_dim), mode='constant'
                )
                grid_size = int(np.sqrt(pad_size))
                prototype_img = padded.reshape(grid_size, grid_size)
                ax.imshow(prototype_img, cmap='viridis')
                ax.axis('off')

        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_prototypes.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_u_matrix(self, weights_map: np.ndarray, epoch: int) -> None:
        """Create U-Matrix visualization showing topological quality."""
        grid_h, grid_w, _ = weights_map.shape
        u_matrix = np.zeros((grid_h, grid_w))

        for i in range(grid_h):
            for j in range(grid_w):
                distances = []
                current = weights_map[i, j, :]
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

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(u_matrix, cmap='bone')
        ax.set_title(f'U-Matrix - Epoch {epoch}')
        plt.colorbar(im, ax=ax, label='Average Distance to Neighbors')
        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_umatrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_hit_counts(self, assignments: np.ndarray, epoch: int) -> None:
        """Visualize neuron usage frequency (hit counts)."""
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))
        hit_counts = np.zeros((grid_h, grid_w))
        for i, j in zip(bmu_coords[0], bmu_coords[1]):
            hit_counts[i, j] += 1

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(hit_counts, cmap='YlOrRd')
        ax.set_title(f'Hit Counts - Epoch {epoch}')
        plt.colorbar(im, ax=ax, label='Number of Samples')

        total_neurons = grid_h * grid_w
        active = np.sum(hit_counts > 0)
        fig.suptitle(f'Active: {active}/{total_neurons} neurons')
        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_hit_counts.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_confusion_on_grid(
            self, assignments: np.ndarray, y_expanded: np.ndarray, epoch: int
    ) -> None:
        """Visualize class distribution per grid cell."""
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        num_classes = len(np.unique(self.y_latent_analysis))
        confusion_grid = np.zeros((grid_h, grid_w, num_classes))
        for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1])):
            label = y_expanded[idx]
            if label < num_classes:
                confusion_grid[i, j, int(label)] += 1

        fig, axes = plt.subplots(
            grid_h, grid_w, figsize=(grid_w * 1.2, grid_h * 1.2)
        )
        fig.suptitle(f'Class Distribution - Epoch {epoch}', fontsize=16)
        for i in range(grid_h):
            for j in range(grid_w):
                ax = axes[i, j]
                class_counts = confusion_grid[i, j, :]
                if class_counts.sum() > 0:
                    colors = plt.cm.tab10(np.arange(num_classes))
                    ax.pie(class_counts[class_counts > 0],
                           colors=colors[class_counts > 0])
                ax.axis('off')
        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_confusion_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_grid_interpolations(
            self, decoder: keras.Model, weights_map: np.ndarray,
            h: int, w: int, epoch: int
    ) -> None:
        """Create interpolations by walking along the SOM grid."""
        try:
            grid_h, grid_w, _ = weights_map.shape
            fig, axes = plt.subplots(2, grid_w, figsize=(grid_w * 2, 4))
            fig.suptitle(f'SOM Grid Traversals - Epoch {epoch}', fontsize=16)
            middle_row = grid_h // 2
            for col in range(grid_w):
                prototype = weights_map[middle_row, col, :]
                pure_map = np.tile(prototype, (h, w, 1))
                decoded = decoder.predict(np.expand_dims(pure_map, axis=0))
                axes[0, col].imshow(np.clip(decoded[0], 0, 1))
                axes[0, col].axis('off')
            middle_col = grid_w // 2
            for row in range(min(grid_h, grid_w)):
                prototype = weights_map[row, middle_col, :]
                pure_map = np.tile(prototype, (h, w, 1))
                decoded = decoder.predict(np.expand_dims(pure_map, axis=0))
                axes[1, row].imshow(np.clip(decoded[0], 0, 1))
                axes[1, row].axis('off')
            plt.tight_layout()
            save_path = self.viz_dir / f"epoch_{epoch:03d}_grid_traversals.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create grid interpolations: {e}")

    def _track_sample_trajectories(
            self, encoder: keras.Model, som_layer: SoftSOMLayer, epoch: int
    ) -> None:
        """Track BMU positions for specific samples over training."""
        try:
            feature_maps = encoder.predict(self.tracked_samples, verbose=0)
            grid_h, grid_w = self.config.grid_shape

            for idx, label in enumerate(self.tracked_labels):
                features = feature_maps[idx].reshape(-1, som_layer.input_dim)
                assignments = som_layer.get_soft_assignments(features)
                flat_assign = keras.ops.reshape(assignments, (len(features), -1))
                bmu_indices = keras.ops.argmax(flat_assign, axis=1)
                bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

                mean_y = np.mean(bmu_coords[0])
                mean_x = np.mean(bmu_coords[1])
                self.bmu_trajectories[label].append((epoch, (mean_y, mean_x)))
        except Exception as e:
            logger.warning(f"Failed to track sample trajectories: {e}")

    def _plot_metrics_curves(self, epoch: int) -> None:
        """Plot all metrics evolution over training."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Metrics Evolution', fontsize=16)

        metrics_map = {
            (0, 0): ('reconstruction_mse', 'MSE'),
            (0, 1): ('psnr', 'PSNR (dB)'),
            (1, 0): ('quantization_error', 'Quantization Error'),
            (1, 1): ('topological_error', 'Topological Error')
        }
        for (r, c), (key, label) in metrics_map.items():
            if key in self.epoch_metrics:
                ax = axes[r, c]
                ax.plot(self.epochs_recorded, self.epoch_metrics[key])
                ax.set_title(label)
                ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = self.metrics_dir / "metrics_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_bmu_trajectories(self, epoch: int) -> None:
        """Visualize BMU position evolution for tracked samples."""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        title = f'BMU Trajectories - Epoch {epoch}'
        fig.suptitle(title, fontsize=16)
        grid_h, grid_w = self.config.grid_shape

        for idx, label in enumerate(self.tracked_labels):
            ax = axes.flatten()[idx]
            ax.set_xlim(-0.5, grid_w - 0.5)
            ax.set_ylim(-0.5, grid_h - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Class {label}')
            if self.bmu_trajectories[label]:
                coords = [t[1] for t in self.bmu_trajectories[label]]
                y, x = [c[0] for c in coords], [c[1] for c in coords]
                for i in range(len(x) - 1):
                    color = plt.cm.coolwarm(i / max(1, len(x) - 1))
                    ax.plot(x[i:i + 2], y[i:i + 2], 'o-', color=color)
            ax.invert_yaxis()

        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_trajectories.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_latent_space(
            self, features: np.ndarray, y_expanded: np.ndarray, epoch: int
    ) -> None:
        """Create t-SNE visualization of latent space."""
        try:
            from sklearn.manifold import TSNE
            logger.info("Computing t-SNE projection...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            subset_size = min(len(features), 5000)
            indices = np.random.choice(len(features), subset_size, replace=False)
            features_2d = tsne.fit_transform(features[indices])

            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(
                features_2d[:, 0], features_2d[:, 1],
                c=y_expanded[indices], cmap='tab10', alpha=0.6, s=20
            )
            plt.colorbar(scatter, ax=ax, label='Class')
            ax.set_title(f'Latent Space t-SNE - Epoch {epoch}')
            plt.tight_layout()
            save_path = self.viz_dir / f"epoch_{epoch:03d}_tsne.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create t-SNE visualization: {e}")


# =============================================================================
# TRAINING ORCHESTRATION
# =============================================================================

def create_learning_rate_schedule(
        config: CIFARSOMConfig, steps_per_epoch: int
) -> keras.optimizers.schedules.LearningRateSchedule:
    """Create learning rate schedule."""
    total_steps = config.epochs * steps_per_epoch

    if config.lr_schedule == 'cosine':
        return keras.optimizers.schedules.CosineDecay(
            config.learning_rate, decay_steps=total_steps, alpha=0.01
        )
    return config.learning_rate


def train_cifar_som_autoencoder(config: CIFARSOMConfig) -> keras.Model:
    """Train CIFAR autoencoder with SoftSOM topological bottleneck."""
    logger.info(f"Starting training: {config.experiment_name}")

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    (x_train, y_train), (x_test, y_test) = load_cifar_data(config)
    x_train_aug = create_augmentation_layer(config)(x_train, training=True) \
        if config.use_data_augmentation else x_train

    model = create_cifar_som_autoencoder(config)
    steps_per_epoch = len(x_train) // config.batch_size
    lr_schedule = create_learning_rate_schedule(config, steps_per_epoch)

    if config.optimizer_type == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=config.weight_decay
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    if config.gradient_clip_norm > 0:
        optimizer.clipnorm = config.gradient_clip_norm

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(output_dir / "tensorboard")),
        ComprehensiveMonitoringCallback(config, x_test, y_test)
    ]

    logger.info("Starting training...")
    model.fit(
        x_train_aug, x_train, batch_size=config.batch_size,
        epochs=config.epochs, validation_data=(x_test, x_test),
        callbacks=callbacks, verbose=1
    )

    test_loss, _ = model.evaluate(x_test, x_test, verbose=0)
    logger.info(f"Final test loss: {test_loss:.6f}")
    return model


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CIFAR autoencoder with SoftSOM bottleneck',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'],
                        default='cifar10')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--grid-size', type=int, default=16)
    parser.add_argument('--som-temperature', type=float, default=1.0)
    parser.add_argument('--no-augmentation', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    return parser.parse_args()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main training function."""
    args = parse_arguments()

    config = CIFARSOMConfig(
        dataset_name=args.dataset,
        grid_shape=(args.grid_size, args.grid_size),
        som_temperature=args.som_temperature,
        use_data_augmentation=not args.no_augmentation,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )

    logger.info("=== CIFAR + SoftSOM Autoencoder ===")
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"SOM Grid: {config.grid_shape}")
    logger.info(f"Temperature: {config.som_temperature}")
    aug_status = 'Enabled' if config.use_data_augmentation else 'Disabled'
    logger.info(f"Augmentation: {aug_status}")

    train_cifar_som_autoencoder(config)
    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()