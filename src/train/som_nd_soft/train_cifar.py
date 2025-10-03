"""
Complex CIFAR Autoencoder with Soft Self-Organizing Map Bottleneck

This module implements a sophisticated autoencoder architecture using the SoftSOMLayer
as a topological bottleneck. The SOM enforces structured organization in the latent
space, creating a continuous, topologically-ordered representation of CIFAR images.

Architecture:
    Encoder: Conv layers → Dense → SoftSOM bottleneck
    Decoder: Dense → TransposeConv layers

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
    python train_cifar_softsom_autoencoder.py --epochs 100 --grid-size 16
"""

import gc
import json
import keras
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
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

    Attributes:
        # Data Configuration
        dataset_name: Which CIFAR dataset ('cifar10' or 'cifar100')
        image_size: Input image dimensions
        use_data_augmentation: Enable training augmentation
        augmentation_strength: Augmentation intensity [0, 1]

        # Architecture Configuration
        encoder_filters: Number of filters per encoder conv layer
        encoder_kernel_sizes: Kernel sizes for encoder layers
        latent_dim: Dimension of features before SOM

        grid_shape: SOM grid dimensions (height, width)
        som_temperature: Softmax temperature for assignments
        som_use_per_dim_softmax: Use per-dimension vs global softmax
        som_reconstruction_weight: Weight for SOM reconstruction loss
        som_topological_weight: Weight for topological preservation
        som_sharpness_weight: Weight for encouraging sharp assignments

        decoder_filters: Number of filters per decoder conv layer
        use_batch_norm: Enable batch normalization
        use_dropout: Enable dropout regularization
        dropout_rate: Dropout probability

        # Training Configuration
        batch_size: Training batch size
        epochs: Total training epochs
        learning_rate: Initial learning rate
        lr_schedule: Learning rate schedule type
        optimizer_type: Optimizer choice ('adam', 'adamw', 'rmsprop')
        weight_decay: L2 regularization weight
        gradient_clip_norm: Gradient clipping threshold

        # Progressive Training
        use_progressive_training: Enable multi-resolution training
        progressive_epochs: Epochs at each resolution stage
        initial_size: Starting image size for progressive training

        # Loss Configuration
        reconstruction_loss_type: Loss function ('mse', 'mae', 'perceptual')
        perceptual_weight: Weight for perceptual loss component

        # Monitoring Configuration
        monitor_every_n_epochs: Visualization update frequency
        num_reconstruction_samples: Number of samples to visualize
        num_interpolation_samples: Number of interpolation paths
        save_latent_projections: Enable t-SNE visualization

        # Output Configuration
        output_dir: Base output directory
        experiment_name: Unique experiment identifier
        save_checkpoints: Enable model checkpointing
        checkpoint_frequency: Epochs between checkpoints
    """

    # === Data Configuration ===
    dataset_name: str = 'cifar10'
    image_size: int = 32
    use_data_augmentation: bool = True
    augmentation_strength: float = 0.5

    # === Architecture Configuration ===
    encoder_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    latent_dim: int = 512

    grid_shape: Tuple[int, int] = (16, 16)
    som_temperature: float = 0.3
    som_use_per_dim_softmax: bool = True
    som_reconstruction_weight: float = 1.5
    som_topological_weight: float = 0.5
    som_sharpness_weight: float = 0.1

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
    progressive_epochs: List[int] = field(default_factory=lambda: [20, 30, 50])
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
    Create convolutional encoder network.

    Architecture:
        Input → Conv blocks (with BN, activation) → GlobalAvgPool → Dense → Output

    Args:
        config: Model configuration

    Returns:
        Keras encoder model mapping images to latent vectors
    """
    inputs = keras.Input(shape=(config.image_size, config.image_size, 3), name="image_input")

    x = inputs

    # Convolutional feature extraction
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

        if i < len(config.encoder_filters) - 1:
            if config.use_batch_norm:
                x = keras.layers.BatchNormalization(name=f"encoder_bn_{i + 1}")(x)

            x = keras.layers.Activation('relu', name=f"encoder_relu_{i + 1}")(x)

            if config.use_dropout:
                x = keras.layers.Dropout(config.dropout_rate, name=f"encoder_dropout_{i + 1}")(x)

    # Global pooling and dense projection
    x = keras.layers.GlobalAveragePooling2D(name="encoder_pool")(x)

    outputs = keras.layers.Dense(config.latent_dim, activation='relu', name="latent_features")(x)

    return keras.Model(inputs, outputs, name="encoder")


def create_decoder(config: CIFARSOMConfig) -> keras.Model:
    """
    Create convolutional decoder network with transpose convolutions.

    Architecture:
        Input → Dense → Reshape → TransposeConv blocks → Output

    Args:
        config: Model configuration

    Returns:
        Keras decoder model mapping latent vectors to reconstructed images
    """
    inputs = keras.Input(shape=(config.latent_dim,), name="latent_input")

    # Calculate spatial size after encoder
    final_spatial_size = config.image_size // (2 ** len(config.encoder_filters))

    # Dense projection to spatial features
    x = keras.layers.Dense(
        final_spatial_size * final_spatial_size * config.decoder_filters[0],
        activation='relu',
        name="decoder_dense"
    )(inputs)

    x = keras.layers.Reshape(
        (final_spatial_size, final_spatial_size, config.decoder_filters[0]),
        name="decoder_reshape"
    )(x)

    # Transposed convolutional upsampling
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
            x = keras.layers.Dropout(config.dropout_rate, name=f"decoder_dropout_{i + 1}")(x)

    # Final reconstruction layer
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

    Architecture:
        Encoder → SoftSOM (topological bottleneck) → Decoder

    The SoftSOM layer enforces spatial organization in the latent space,
    creating a structured manifold where similar images map to nearby locations.

    Args:
        config: Model configuration

    Returns:
        Compiled Keras autoencoder model
    """
    logger.info("Creating CIFAR SoftSOM Autoencoder...")

    # Build components
    encoder = create_encoder(config)
    decoder = create_decoder(config)

    # Create end-to-end model with SOM bottleneck
    inputs = keras.Input(shape=(config.image_size, config.image_size, 3), name="input_images")

    # Encode
    latent_features = encoder(inputs)

    # SoftSOM topological bottleneck
    som_features = SoftSOMLayer(
        grid_shape=config.grid_shape,
        input_dim=config.latent_dim,
        temperature=config.som_temperature,
        use_per_dimension_softmax=config.som_use_per_dim_softmax,
        use_reconstruction_loss=True,
        reconstruction_weight=config.som_reconstruction_weight,
        topological_weight=config.som_topological_weight,
        sharpness_weight=config.som_sharpness_weight,
        name="soft_som_bottleneck"
    )(latent_features)

    # Decode
    reconstructed = decoder(som_features)

    # Build complete model
    autoencoder = keras.Model(inputs, reconstructed, name="cifar_softsom_autoencoder")

    logger.info("Autoencoder architecture:")
    autoencoder.summary()

    return autoencoder


# =============================================================================
# ADVANCED VISUALIZATION
# =============================================================================

class ComprehensiveMonitoringCallback(keras.callbacks.Callback):
    """
    Advanced monitoring callback with comprehensive visualizations.

    Creates:
        - Reconstruction quality comparisons
        - SOM activation heatmaps by class
        - Prototype vector visualizations
        - U-Matrix (topology quality)
        - Hit count maps
        - Confusion analysis on grid
        - Quantization and topological errors
        - BMU trajectories over training
        - Latent space projections (t-SNE)
        - Grid-based interpolations
        - Quality metrics over time
    """

    def __init__(
            self,
            config: CIFARSOMConfig,
            x_test: np.ndarray,
            y_test: np.ndarray
    ) -> None:
        """
        Initialize comprehensive monitoring callback.

        Args:
            config: Training configuration
            x_test: Test images
            y_test: Test labels
        """
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs

        # Select diverse samples for visualization
        self.x_viz = x_test[:config.num_reconstruction_samples]
        self.y_viz = y_test[:config.num_reconstruction_samples].flatten()

        # Larger set for latent space analysis
        self.x_latent_analysis = x_test[:1000]
        self.y_latent_analysis = y_test[:1000].flatten()

        # Select specific samples to track over time (one per class)
        num_classes = 10 if config.dataset_name == 'cifar10' else 100
        num_classes_to_track = min(10, num_classes)
        self.tracked_samples = []
        self.tracked_labels = []
        for class_id in range(num_classes_to_track):
            class_mask = self.y_latent_analysis == class_id
            class_indices = np.where(class_mask)[0]
            if len(class_indices) > 0:
                self.tracked_samples.append(self.x_latent_analysis[class_indices[0]])
                self.tracked_labels.append(class_id)
        self.tracked_samples = np.array(self.tracked_samples)

        # Track BMU trajectories over epochs
        self.bmu_trajectories: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {
            label: [] for label in self.tracked_labels
        }

        # Setup directories
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.viz_dir = self.output_dir / "visualizations"
        self.metrics_dir = self.output_dir / "metrics"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Track metrics
        self.epoch_metrics: Dict[str, List[float]] = {
            'reconstruction_mse': [],
            'reconstruction_mae': [],
            'psnr': [],
            'ssim': [],
            'quantization_error': [],
            'topological_error': []
        }
        self.epochs_recorded: List[int] = []

        logger.info(f"Comprehensive monitoring initialized with {len(self.x_viz)} viz samples")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Create comprehensive visualizations and metrics.

        Args:
            epoch: Current epoch number
            logs: Training logs
        """
        if (epoch + 1) % self.monitor_freq != 0:
            return

        logger.info(f"Creating comprehensive visualizations for epoch {epoch + 1}...")

        try:
            # Get reconstructions
            reconstructed = self.model.predict(self.x_viz, verbose=0, batch_size=32)

            # Get SOM layer and models
            som_layer = self.model.get_layer("soft_som_bottleneck")
            encoder = self.model.get_layer("encoder")

            # Extract features for analysis
            features = encoder.predict(self.x_latent_analysis, verbose=0, batch_size=128)

            # Get soft assignments and weights
            assignments = som_layer.get_soft_assignments(features)
            assignments = keras.ops.convert_to_numpy(assignments)

            weights_map = som_layer.get_weights_map()
            weights_map = keras.ops.convert_to_numpy(weights_map)

            # Compute and store metrics
            self._compute_and_store_metrics(reconstructed, features, weights_map, assignments, epoch + 1)

            # Create all visualizations
            self._visualize_reconstructions(reconstructed, epoch + 1)
            self._visualize_som_activations(assignments, epoch + 1)
            self._visualize_prototype_vectors(weights_map, epoch + 1)
            self._visualize_u_matrix(weights_map, epoch + 1)
            self._visualize_hit_counts(assignments, epoch + 1)
            self._visualize_confusion_on_grid(assignments, epoch + 1)
            self._visualize_grid_interpolations(epoch + 1)

            # Track sample trajectories
            self._track_sample_trajectories(epoch + 1)

            # Create summary visualizations
            if len(self.epochs_recorded) > 1:
                self._plot_metrics_curves(epoch + 1)
                self._visualize_bmu_trajectories(epoch + 1)

            if self.config.save_latent_projections and (epoch + 1) % (self.monitor_freq * 2) == 0:
                self._visualize_latent_space(features, epoch + 1)

            # Cleanup
            del reconstructed, features, assignments, weights_map
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _compute_and_store_metrics(
            self,
            reconstructed: np.ndarray,
            features: np.ndarray,
            weights_map: np.ndarray,
            assignments: np.ndarray,
            epoch: int
    ) -> None:
        """
        Compute and store reconstruction quality and SOM metrics.

        Args:
            reconstructed: Reconstructed images
            features: Latent features
            weights_map: SOM weights
            assignments: Soft assignments
            epoch: Current epoch
        """
        # Reconstruction metrics
        mse = np.mean((self.x_viz - reconstructed) ** 2)
        mae = np.mean(np.abs(self.x_viz - reconstructed))

        # PSNR calculation
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))

        # Simple SSIM approximation
        mean_x = np.mean(self.x_viz)
        mean_y = np.mean(reconstructed)
        var_x = np.var(self.x_viz)
        var_y = np.var(reconstructed)
        cov_xy = np.mean((self.x_viz - mean_x) * (reconstructed - mean_y))

        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        ssim = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / \
               ((mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2))

        # SOM metrics: Quantization and topological errors
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        # Quantization error
        quantization_error = 0.0
        for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1])):
            bmu_prototype = weights_map[i, j, :]
            dist = np.linalg.norm(features[idx] - bmu_prototype)
            quantization_error += dist
        quantization_error /= len(features)

        # Topological error
        topological_error = 0.0
        for idx in range(len(features)):
            distances = flat_assignments[idx]
            sorted_indices = np.argsort(-distances)

            bmu1_idx = sorted_indices[0]
            bmu2_idx = sorted_indices[1]

            bmu1_coords = np.unravel_index(bmu1_idx, (grid_h, grid_w))
            bmu2_coords = np.unravel_index(bmu2_idx, (grid_h, grid_w))

            manhattan_dist = abs(bmu1_coords[0] - bmu2_coords[0]) + abs(bmu1_coords[1] - bmu2_coords[1])
            if manhattan_dist > 1:
                topological_error += 1
        topological_error /= len(features)

        # Store metrics
        self.epoch_metrics['reconstruction_mse'].append(float(mse))
        self.epoch_metrics['reconstruction_mae'].append(float(mae))
        self.epoch_metrics['psnr'].append(float(psnr))
        self.epoch_metrics['ssim'].append(float(ssim))
        self.epoch_metrics['quantization_error'].append(float(quantization_error))
        self.epoch_metrics['topological_error'].append(float(topological_error))
        self.epochs_recorded.append(epoch)

        logger.info(f"Epoch {epoch} - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        logger.info(f"Epoch {epoch} - Quant Error: {quantization_error:.4f}, Topo Error: {topological_error:.4f}")

        # Save metrics
        metrics_file = self.metrics_dir / f"epoch_{epoch:03d}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'epoch': epoch,
                'mse': float(mse),
                'mae': float(mae),
                'psnr': float(psnr),
                'ssim': float(ssim),
                'quantization_error': float(quantization_error),
                'topological_error': float(topological_error)
            }, f, indent=2)

    def _visualize_reconstructions(
            self,
            reconstructed: np.ndarray,
            epoch: int
    ) -> None:
        """
        Create side-by-side comparison of original and reconstructed images.

        Args:
            reconstructed: Reconstructed images
            epoch: Current epoch
        """
        n_samples = len(self.x_viz)
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 3, n_rows * 6))
        fig.suptitle(f'Image Reconstructions - Epoch {epoch}', fontsize=16)

        for i in range(n_samples):
            row = (i // n_cols) * 2
            col = i % n_cols

            # Original image
            if n_rows > 1:
                ax_orig = axes[row, col]
                ax_recon = axes[row + 1, col]
            else:
                ax_orig = axes[0, col]
                ax_recon = axes[1, col]

            ax_orig.imshow(self.x_viz[i])
            ax_orig.set_title(f'Original\nClass: {self.y_viz[i]}', fontsize=10)
            ax_orig.axis('off')

            # Reconstructed image
            ax_recon.imshow(np.clip(reconstructed[i], 0, 1))
            mse = np.mean((self.x_viz[i] - reconstructed[i]) ** 2)
            ax_recon.set_title(f'Reconstructed\nMSE: {mse:.4f}', fontsize=10)
            ax_recon.axis('off')

        # Hide unused subplots
        for i in range(n_samples, n_rows * n_cols):
            row = (i // n_cols) * 2
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
                axes[row + 1, col].axis('off')

        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_reconstructions.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_som_activations(self, assignments: np.ndarray, epoch: int) -> None:
        """
        Visualize SOM activation patterns for different image classes.

        Args:
            assignments: Soft assignments
            epoch: Current epoch
        """
        try:
            unique_classes = np.unique(self.y_latent_analysis)
            n_classes = min(10, len(unique_classes))

            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f'SOM Activation Heatmaps by Class - Epoch {epoch}', fontsize=16)

            for idx in range(n_classes):
                class_label = unique_classes[idx]
                ax = axes[idx // 5, idx % 5]

                class_mask = (self.y_latent_analysis == class_label)
                class_assignments = assignments[class_mask]

                mean_activation = np.mean(class_assignments, axis=0)

                im = ax.imshow(mean_activation, cmap='hot', interpolation='nearest')
                ax.set_title(f'Class {class_label}', fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            save_path = self.viz_dir / f"epoch_{epoch:03d}_som_activations.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to visualize SOM activations: {e}")

    def _visualize_prototype_vectors(self, weights_map: np.ndarray, epoch: int) -> None:
        """
        Visualize learned prototype vectors as pseudo-images.

        Args:
            weights_map: SOM weight matrix
            epoch: Current epoch
        """
        grid_h, grid_w, input_dim = weights_map.shape

        fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 1.5, grid_h * 1.5))
        fig.suptitle(f'SOM Prototype Vectors (Learned Codebook) - Epoch {epoch}', fontsize=16)

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

                prototype = weights_map[i, j, :]
                prototype_normalized = (prototype - prototype.min()) / (prototype.max() - prototype.min() + 1e-8)

                # Reshape for visualization
                grid_size = int(np.sqrt(input_dim))
                if grid_size * grid_size == input_dim:
                    prototype_img = prototype_normalized.reshape(grid_size, grid_size)
                else:
                    pad_size = int(np.ceil(np.sqrt(input_dim))) ** 2
                    padded = np.pad(prototype_normalized, (0, pad_size - input_dim), mode='constant')
                    grid_size = int(np.sqrt(pad_size))
                    prototype_img = padded.reshape(grid_size, grid_size)

                ax.imshow(prototype_img, cmap='viridis', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'{i},{j}', fontsize=6)

        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_prototypes.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_u_matrix(self, weights_map: np.ndarray, epoch: int) -> None:
        """
        Create U-Matrix visualization showing topological quality.

        Args:
            weights_map: SOM weight matrix
            epoch: Current epoch
        """
        grid_h, grid_w, input_dim = weights_map.shape
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
        im = ax.imshow(u_matrix, cmap='bone', interpolation='nearest')
        ax.set_title(f'U-Matrix (Topological Quality) - Epoch {epoch}\n'
                     f'Dark=Cluster Centers, Bright=Boundaries', fontsize=14)
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')

        for i in range(grid_h + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(grid_w + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        plt.colorbar(im, ax=ax, label='Average Distance to Neighbors')
        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_umatrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_hit_counts(self, assignments: np.ndarray, epoch: int) -> None:
        """
        Visualize neuron usage frequency (hit counts).

        Args:
            assignments: Soft assignments
            epoch: Current epoch
        """
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        hit_counts = np.zeros((grid_h, grid_w))
        for i, j in zip(bmu_coords[0], bmu_coords[1]):
            hit_counts[i, j] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Linear scale
        im1 = ax1.imshow(hit_counts, cmap='YlOrRd', interpolation='nearest')
        ax1.set_title(f'Hit Counts (Linear Scale) - Epoch {epoch}', fontsize=14)
        ax1.set_xlabel('Grid X')
        ax1.set_ylabel('Grid Y')

        for i in range(grid_h):
            for j in range(grid_w):
                count = int(hit_counts[i, j])
                ax1.text(j, i, str(count), ha='center', va='center',
                         color='white' if count > hit_counts.max() * 0.5 else 'black',
                         fontsize=8)

        plt.colorbar(im1, ax=ax1, label='Number of Samples')

        # Log scale
        hit_counts_log = np.log10(hit_counts + 1)
        im2 = ax2.imshow(hit_counts_log, cmap='YlOrRd', interpolation='nearest')
        ax2.set_title(f'Hit Counts (Log Scale) - Epoch {epoch}', fontsize=14)
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')
        plt.colorbar(im2, ax=ax2, label='Log10(Samples + 1)')

        total_neurons = grid_h * grid_w
        active_neurons = np.sum(hit_counts > 0)
        dead_neurons = total_neurons - active_neurons
        max_hits = int(hit_counts.max())
        mean_hits = hit_counts.mean()

        fig.suptitle(f'Active: {active_neurons}/{total_neurons} neurons | '
                     f'Dead: {dead_neurons} | Max: {max_hits} | Mean: {mean_hits:.1f}',
                     fontsize=12)

        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_hit_counts.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_confusion_on_grid(self, assignments: np.ndarray, epoch: int) -> None:
        """
        Visualize class distribution per grid cell.

        Args:
            assignments: Soft assignments
            epoch: Current epoch
        """
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)

        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        num_classes = len(np.unique(self.y_latent_analysis))
        confusion_grid = np.zeros((grid_h, grid_w, num_classes))
        for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1])):
            label = self.y_latent_analysis[idx]
            confusion_grid[i, j, label] += 1

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

                class_counts = confusion_grid[i, j, :]
                total = class_counts.sum()

                if total > 0:
                    colors = plt.cm.tab10(np.arange(10))
                    non_zero = class_counts > 0
                    ax.pie(class_counts[non_zero], colors=colors[non_zero],
                           startangle=90, counterclock=False)

                    dominant_class = np.argmax(class_counts)
                    purity = class_counts[dominant_class] / total
                    ax.text(0, -1.3, f'{int(dominant_class)}\n({purity:.1%})',
                            ha='center', va='top', fontsize=6,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax.text(0, 0, 'Empty', ha='center', va='center', fontsize=8)

                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.5, 1.2)
                ax.axis('off')

        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_confusion_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_grid_interpolations(self, epoch: int) -> None:
        """
        Create interpolations by walking along the SOM grid.

        Args:
            epoch: Current epoch
        """
        try:
            som_layer = self.model.get_layer("soft_som_bottleneck")
            decoder = self.model.get_layer("decoder")

            weights_map = som_layer.get_weights_map()
            weights_map = keras.ops.convert_to_numpy(weights_map)

            grid_h, grid_w = self.config.grid_shape

            fig, axes = plt.subplots(2, grid_w, figsize=(grid_w * 2, 4))
            fig.suptitle(f'SOM Grid Traversals - Epoch {epoch}', fontsize=16)

            # Horizontal traversal
            middle_row = grid_h // 2
            for col in range(grid_w):
                prototype = weights_map[middle_row, col:col + 1, :]
                decoded = decoder.predict(prototype, verbose=0)

                axes[0, col].imshow(np.clip(decoded[0], 0, 1))
                axes[0, col].set_title(f'({middle_row},{col})', fontsize=8)
                axes[0, col].axis('off')

            # Vertical traversal
            middle_col = grid_w // 2
            for row in range(min(grid_h, grid_w)):
                prototype = weights_map[row:row + 1, middle_col, :]
                decoded = decoder.predict(prototype, verbose=0)

                axes[1, row].imshow(np.clip(decoded[0], 0, 1))
                axes[1, row].set_title(f'({row},{middle_col})', fontsize=8)
                axes[1, row].axis('off')

            for col in range(grid_h, grid_w):
                axes[1, col].axis('off')

            plt.tight_layout()
            save_path = self.viz_dir / f"epoch_{epoch:03d}_grid_traversals.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to create grid interpolations: {e}")

    def _track_sample_trajectories(self, epoch: int) -> None:
        """
        Track BMU positions for specific samples over training.

        Args:
            epoch: Current epoch
        """
        try:
            som_layer = self.model.get_layer("soft_som_bottleneck")
            encoder = self.model.get_layer("encoder")

            features = encoder.predict(self.tracked_samples, verbose=0, batch_size=32)
            assignments = som_layer.get_soft_assignments(features)
            assignments = keras.ops.convert_to_numpy(assignments)

            flat_assignments = assignments.reshape(assignments.shape[0], -1)
            bmu_indices = np.argmax(flat_assignments, axis=1)

            grid_h, grid_w = assignments.shape[1], assignments.shape[2]
            bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

            for idx, label in enumerate(self.tracked_labels):
                coord = (int(bmu_coords[0][idx]), int(bmu_coords[1][idx]))
                self.bmu_trajectories[label].append((epoch, coord))

        except Exception as e:
            logger.warning(f"Failed to track sample trajectories: {e}")

    def _plot_metrics_curves(self, epoch: int) -> None:
        """
        Plot all metrics evolution over training.

        Args:
            epoch: Current epoch
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Metrics Evolution', fontsize=16)

        epochs_list = self.epochs_recorded

        # MSE
        axes[0, 0].plot(epochs_list, self.epoch_metrics['reconstruction_mse'], 'b-', linewidth=2)
        axes[0, 0].set_title('Reconstruction MSE')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)

        # MAE
        axes[0, 1].plot(epochs_list, self.epoch_metrics['reconstruction_mae'], 'g-', linewidth=2)
        axes[0, 1].set_title('Reconstruction MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3)

        # PSNR
        axes[0, 2].plot(epochs_list, self.epoch_metrics['psnr'], 'r-', linewidth=2)
        axes[0, 2].set_title('PSNR')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('PSNR (dB)')
        axes[0, 2].grid(True, alpha=0.3)

        # SSIM
        axes[1, 0].plot(epochs_list, self.epoch_metrics['ssim'], 'm-', linewidth=2)
        axes[1, 0].set_title('SSIM')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].grid(True, alpha=0.3)

        # Quantization Error
        axes[1, 1].plot(epochs_list, self.epoch_metrics['quantization_error'], 'c-', linewidth=2)
        axes[1, 1].set_title('Quantization Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Avg Distance to BMU')
        axes[1, 1].grid(True, alpha=0.3)

        # Topological Error
        axes[1, 2].plot(epochs_list, self.epoch_metrics['topological_error'], 'y-', linewidth=2)
        axes[1, 2].set_title('Topological Error')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Fraction Non-Neighbor BMU Pairs')
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.metrics_dir / "metrics_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_bmu_trajectories(self, epoch: int) -> None:
        """
        Visualize BMU position evolution for tracked samples.

        Args:
            epoch: Current epoch
        """
        n_tracked = len(self.tracked_labels)
        n_cols = 5
        n_rows = (n_tracked + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'BMU Trajectories During Training (up to Epoch {epoch})', fontsize=16)

        grid_h, grid_w = self.config.grid_shape

        for idx, label in enumerate(self.tracked_labels):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            ax.set_xlim(-0.5, grid_w - 0.5)
            ax.set_ylim(-0.5, grid_h - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Class {label}', fontsize=12)
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            ax.grid(True, alpha=0.3)

            if label in self.bmu_trajectories and len(self.bmu_trajectories[label]) > 0:
                trajectory = self.bmu_trajectories[label]
                coords = [t[1] for t in trajectory]

                x_coords = [c[1] for c in coords]
                y_coords = [c[0] for c in coords]

                for i in range(len(x_coords) - 1):
                    color = plt.cm.coolwarm(i / max(1, len(x_coords) - 1))
                    ax.plot(x_coords[i:i + 2], y_coords[i:i + 2], 'o-',
                            color=color, linewidth=2, markersize=6, alpha=0.7)

                ax.plot(x_coords[0], y_coords[0], 'go', markersize=12,
                        label='Start', markeredgecolor='black', markeredgewidth=2)
                ax.plot(x_coords[-1], y_coords[-1], 'r*', markersize=16,
                        label='Current', markeredgecolor='black', markeredgewidth=2)

                ax.legend(loc='upper right', fontsize=8)
            else:
                ax.text(grid_w / 2, grid_h / 2, 'No trajectory data',
                        ha='center', va='center', fontsize=10)

            ax.invert_yaxis()

        # Hide unused subplots
        for idx in range(n_tracked, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        save_path = self.viz_dir / f"epoch_{epoch:03d}_trajectories.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_latent_space(self, features: np.ndarray, epoch: int) -> None:
        """
        Create t-SNE visualization of latent space.

        Args:
            features: Latent features
            epoch: Current epoch
        """
        try:
            from sklearn.manifold import TSNE

            logger.info("Computing t-SNE projection...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features)

            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(
                features_2d[:, 0],
                features_2d[:, 1],
                c=self.y_latent_analysis,
                cmap='tab10',
                alpha=0.6,
                s=20
            )
            plt.colorbar(scatter, ax=ax, label='Class')
            ax.set_title(f'Latent Space t-SNE Projection - Epoch {epoch}', fontsize=16)
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')

            plt.tight_layout()
            save_path = self.viz_dir / f"epoch_{epoch:03d}_tsne.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()

        except ImportError:
            logger.warning("sklearn not available, skipping t-SNE visualization")
        except Exception as e:
            logger.warning(f"Failed to create t-SNE visualization: {e}")


# =============================================================================
# TRAINING ORCHESTRATION
# =============================================================================

def create_learning_rate_schedule(config: CIFARSOMConfig,
                                  steps_per_epoch: int) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create learning rate schedule.

    Args:
        config: Training configuration
        steps_per_epoch: Number of training steps per epoch

    Returns:
        Keras learning rate schedule
    """
    total_steps = config.epochs * steps_per_epoch

    if config.lr_schedule == 'cosine':
        return keras.optimizers.schedules.CosineDecay(
            config.learning_rate,
            decay_steps=total_steps,
            alpha=0.01
        )
    elif config.lr_schedule == 'exponential':
        return keras.optimizers.schedules.ExponentialDecay(
            config.learning_rate,
            decay_steps=steps_per_epoch * 10,
            decay_rate=0.9
        )
    else:
        return config.learning_rate


def train_cifar_som_autoencoder(config: CIFARSOMConfig) -> keras.Model:
    """
    Train CIFAR autoencoder with SoftSOM topological bottleneck.

    Args:
        config: Training configuration

    Returns:
        Trained Keras model
    """
    logger.info(f"Starting CIFAR SoftSOM Autoencoder training: {config.experiment_name}")

    # Setup output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar_data(config)

    # Create data augmentation
    if config.use_data_augmentation:
        augmentation_layer = create_augmentation_layer(config)
        x_train_augmented = augmentation_layer(x_train, training=True)
    else:
        x_train_augmented = x_train

    # Create model
    model = create_cifar_som_autoencoder(config)

    # Setup optimizer
    steps_per_epoch = len(x_train) // config.batch_size
    lr_schedule = create_learning_rate_schedule(config, steps_per_epoch)

    if config.optimizer_type == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    elif config.optimizer_type == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

    # Apply gradient clipping
    if config.gradient_clip_norm > 0:
        optimizer.clipnorm = config.gradient_clip_norm

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    # Setup callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
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
        ComprehensiveMonitoringCallback(config, x_test, y_test)
    ]

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        x_train_augmented, x_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_test, x_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate final model
    test_loss, test_mae = model.evaluate(x_test, x_test, verbose=0)
    logger.info(f"Final test loss: {test_loss:.6f}, MAE: {test_mae:.6f}")

    # Save training history
    history_path = output_dir / "training_history.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Save final metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
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
        description='Train CIFAR autoencoder with SoftSOM bottleneck',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10', help='Dataset choice')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--grid-size', type=int, default=10, help='SOM grid size (square)')
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension before SOM')
    parser.add_argument('--som-temperature', type=float, default=0.5, help='SOM softmax temperature')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

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
        latent_dim=args.latent_dim,
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
    logger.info(f"Latent Dimension: {config.latent_dim}")
    logger.info(f"Temperature: {config.som_temperature}")
    logger.info(f"Augmentation: {'Enabled' if config.use_data_augmentation else 'Disabled'}")

    model = train_cifar_som_autoencoder(config)
    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()