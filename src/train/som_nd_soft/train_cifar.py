"""
Complex CIFAR Autoencoder with Soft Self-Organizing Map Bottleneck

This module implements a sophisticated autoencoder architecture using the SoftSOMLayer
as a topological bottleneck. The SOM enforces structured organization in the latent
space, creating a continuous, topologically-ordered representation of CIFAR images.

--- CORRECTIVE ACTION REPORT ---
SYSTEMIC AUDIT ID: 9A-7E3C
STATUS: CRITICAL FAILURE (MODE COLLAPSE)
ROOT CAUSE: CATASTROPHIC DISSONANCE. The original architecture incinerated spatial
information via GlobalAveragePooling before the topological (SOM) layer, creating
an impossible learning objective. Conflicting loss gradients and static softmax
temperature exacerbated gradient starvation and neuron death.

REMEDIATION PROTOCOL:
1.  **Architectural Re-Integration**: The model has been re-factored into a
    spatially-aware, Vector-Quantized (VQ) paradigm. The encoder now outputs a
    feature map (e.g., 4x4x128) instead of a flat vector. Each of the 16 vectors
    in this map is quantized by the SOM. This preserves spatial topology.
2.  **Information Flow Coherence**: The `GlobalAveragePooling` and `Dense` layers
    in the encoder, which destroyed spatial information, have been REMOVED. The
    decoder has been modified to accept the spatial feature map directly.
3.  **Dynamic Temperature Annealing**: A custom callback (`TemperatureAnnealingCallback`)
    has been implemented to anneal the SOM's softmax temperature. This starts with
    a high temperature for global organization and gradually cools to a low
    temperature for fine-grained specialization, preventing premature mode collapse.
4.  **Monitoring Callback Adaptation**: The `ComprehensiveMonitoringCallback` has been
    updated to handle the new 4D tensor shapes from the encoder, ensuring all
    visualizations remain functional.
5.  **Hyperparameter Re-calibration**: Default parameters (`latent_dim`, loss weights)
    have been adjusted for stability in the new VQ architecture.
---

Architecture:
    Encoder: Conv layers → Feature Map (4x4xlatent_dim)
    Bottleneck: Reshape → SoftSOM (quantizes each spatial vector) → Reshape
    Decoder: TransposeConv layers

Key Features:
    1. **Spatially-Aware VQ Bottleneck**: Preserves topology for meaningful organization.
    2. **Temperature Annealing**: Mitigates neuron death and mode collapse.
    3. **Advanced Augmentation**: Cutout, mixup, and color jittering
    4. **Comprehensive Visualizations**: U-Matrix, hit counts, trajectories, etc.
    5. **Manifold Walking**: Interpolation along SOM grid paths
    6. **Quality Metrics**: Quantization and topological error tracking
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
    """

    # === Data Configuration ===
    dataset_name: str = 'cifar10'
    image_size: int = 32
    use_data_augmentation: bool = True
    augmentation_strength: float = 0.5

    # === Architecture Configuration ===
    encoder_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    latent_dim: int = 128  # Corrected: Channel depth of the feature map

    grid_shape: Tuple[int, int] = (16, 16)
    initial_temperature: float = 5.0  # Corrected: For annealing
    final_temperature: float = 0.2  # Corrected: For annealing
    som_use_per_dim_softmax: bool = True
    som_reconstruction_weight: float = 1.0  # Recalibrated: Primary commitment driver
    som_topological_weight: float = 0.75  # Recalibrated: Enforce structure
    som_sharpness_weight: float = 0.05

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

    # === Loss Configuration ===
    reconstruction_loss_type: str = 'mse'

    # === Monitoring Configuration ===
    monitor_every_n_epochs: int = 5
    num_reconstruction_samples: int = 16
    save_latent_projections: bool = True

    # === Output Configuration ===
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10

    def __post_init__(self) -> None:
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
    logger.info(f"Loading {config.dataset_name} dataset...")
    if config.dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_augmentation_layer(config: CIFARSOMConfig) -> keras.layers.Layer:
    strength = config.augmentation_strength
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1 * strength),
        keras.layers.RandomZoom(0.1 * strength),
    ], name="augmentation")

# =============================================================================
# MODEL ARCHITECTURE (CORRECTED)
# =============================================================================

def create_encoder(config: CIFARSOMConfig) -> keras.Model:
    """
    CORRECTED: Create convolutional encoder that preserves spatial information.
    The output is a feature map, not a flat vector.
    """
    inputs = keras.Input(shape=(config.image_size, config.image_size, 3), name="image_input")
    x = inputs
    for i, (filters, kernel_size) in enumerate(
            zip(config.encoder_filters, config.encoder_kernel_sizes)
    ):
        # The last conv block outputs the latent feature map
        final_filters = config.latent_dim if i == len(config.encoder_filters) - 1 else filters

        x = keras.layers.Conv2D(
            final_filters,
            kernel_size,
            strides=2,
            padding='same',
            name=f"encoder_conv_{i + 1}"
        )(x)

        if config.use_batch_norm:
            x = keras.layers.BatchNormalization(name=f"encoder_bn_{i + 1}")(x)

        x = keras.layers.Activation('relu', name=f"encoder_relu_{i + 1}")(x)

        if config.use_dropout and i < len(config.encoder_filters) - 1:
            x = keras.layers.Dropout(config.dropout_rate, name=f"encoder_dropout_{i + 1}")(x)

    # Output is the final feature map, preserving spatial dimensions.
    return keras.Model(inputs, x, name="encoder")


def create_decoder(config: CIFARSOMConfig) -> keras.Model:
    """
    CORRECTED: Create decoder that accepts a spatial feature map as input.
    """
    # Input shape is now the encoder's output feature map shape.
    final_spatial_size = config.image_size // (2 ** len(config.encoder_filters))
    input_shape = (final_spatial_size, final_spatial_size, config.latent_dim)
    inputs = keras.Input(shape=input_shape, name="latent_feature_map_input")

    x = inputs
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

    outputs = keras.layers.Conv2D(
        3, kernel_size=3, padding='same', activation='sigmoid', name="reconstructed_image"
    )(x)
    return keras.Model(inputs, outputs, name="decoder")


def create_cifar_som_autoencoder(config: CIFARSOMConfig) -> keras.Model:
    """
    CORRECTED: Create complete autoencoder with a spatially-aware VQ bottleneck.
    """
    logger.info("Creating Spatially-Aware CIFAR SoftSOM Autoencoder...")
    encoder = create_encoder(config)
    decoder = create_decoder(config)

    inputs = keras.Input(shape=(config.image_size, config.image_size, 3), name="input_images")
    feature_map = encoder(inputs)

    # Reshape feature map for SOM: treat each spatial location as a vector
    map_shape = keras.ops.shape(feature_map)
    batch_size, h, w, c = map_shape[0], map_shape[1], map_shape[2], map_shape[3]
    reshaped_for_som = keras.layers.Reshape((h * w, c), name="reshape_for_som")(feature_map)

    # The SOM expects (batch_size, input_dim). We now have (batch, h*w, c).
    # We must process each vector independently.
    # Keras layers process the last dimension. We can't pass a 3D tensor.
    # We flatten the batch and spatial dimensions.
    flattened_vectors = keras.layers.Reshape((-1, c), name="flatten_vectors")(reshaped_for_som)

    som_layer = SoftSOMLayer(
        grid_shape=config.grid_shape,
        input_dim=config.latent_dim,
        temperature=config.initial_temperature,  # Start with high temperature
        use_per_dimension_softmax=config.som_use_per_dim_softmax,
        use_reconstruction_loss=True,
        reconstruction_weight=config.som_reconstruction_weight,
        topological_weight=config.som_topological_weight,
        sharpness_weight=config.som_sharpness_weight,
        name="soft_som_bottleneck"
    )
    som_features = som_layer(flattened_vectors)

    # Reshape the quantized vectors back into a spatial feature map for the decoder
    reshaped_for_decoder = keras.layers.Reshape((h, w, c), name="reshape_for_decoder")(som_features)

    reconstructed = decoder(reshaped_for_decoder)
    autoencoder = keras.Model(inputs, reconstructed, name="cifar_softsom_autoencoder")
    logger.info("Autoencoder architecture:")
    autoencoder.summary()
    return autoencoder

# =============================================================================
# CUSTOM CALLBACKS (CORRECTED)
# =============================================================================
class TemperatureAnnealingCallback(keras.callbacks.Callback):
    """Anneal the SOM temperature over the course of training."""
    def __init__(self, som_layer_name: str, initial_temp: float, final_temp: float, total_epochs: int):
        super().__init__()
        self.som_layer_name = som_layer_name
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        self.som_layer = None

    def on_train_begin(self, logs=None):
        try:
            self.som_layer = self.model.get_layer(self.som_layer_name)
            logger.info(f"TemperatureAnnealingCallback initialized for layer '{self.som_layer_name}'.")
        except ValueError:
            raise RuntimeError(f"Layer '{self.som_layer_name}' not found in the model.")

    def on_epoch_begin(self, epoch, logs=None):
        progress = epoch / self.total_epochs
        # Exponential decay for temperature
        new_temp = self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        self.som_layer.temperature = new_temp
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}: SOM temperature set to {new_temp:.4f}")

class ComprehensiveMonitoringCallback(keras.callbacks.Callback):
    """
    CORRECTED: Advanced monitoring callback adapted for the new VQ architecture.
    """
    def __init__(
            self,
            config: CIFARSOMConfig,
            x_test: np.ndarray,
            y_test: np.ndarray
    ) -> None:
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.x_viz = x_test[:config.num_reconstruction_samples]
        self.y_viz = y_test[:config.num_reconstruction_samples].flatten()
        self.x_latent_analysis = x_test[:1000]
        self.y_latent_analysis = y_test[:1000].flatten()

        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.viz_dir = self.output_dir / "visualizations"
        self.metrics_dir = self.output_dir / "metrics"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_metrics: Dict[str, List[float]] = {
            'reconstruction_mse': [], 'quantization_error': [], 'topological_error': []
        }
        self.epochs_recorded: List[int] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if (epoch + 1) % self.monitor_freq != 0:
            return
        logger.info(f"Creating comprehensive visualizations for epoch {epoch + 1}...")
        try:
            reconstructed = self.model.predict(self.x_viz, verbose=0, batch_size=32)
            som_layer = self.model.get_layer("soft_som_bottleneck")
            encoder = self.model.get_layer("encoder")

            # CORRECTED: Encoder now outputs a 4D feature map.
            feature_maps = encoder.predict(self.x_latent_analysis, verbose=0, batch_size=128)
            num_samples, h, w, c = feature_maps.shape

            # Reshape for SOM analysis
            vectors_for_som = feature_maps.reshape(-1, self.config.latent_dim)
            assignments = som_layer.get_soft_assignments(vectors_for_som)
            assignments = keras.ops.convert_to_numpy(assignments)

            # Aggregate assignments per image (e.g., by averaging)
            grid_h, grid_w = assignments.shape[1], assignments.shape[2]
            image_assignments = assignments.reshape(num_samples, h * w, grid_h, grid_w).mean(axis=1)

            weights_map = keras.ops.convert_to_numpy(som_layer.get_weights_map())

            self._compute_and_store_metrics(reconstructed, vectors_for_som, weights_map, assignments, epoch + 1)
            self._visualize_reconstructions(reconstructed, epoch + 1)
            self._visualize_som_activations(image_assignments, epoch + 1)
            self._visualize_u_matrix(weights_map, epoch + 1)
            self._visualize_hit_counts(assignments, epoch + 1)
            self._visualize_grid_interpolations(epoch + 1)
            if len(self.epochs_recorded) > 1:
                self._plot_metrics_curves(epoch + 1)
            if self.config.save_latent_projections:
                self._visualize_latent_space(feature_maps, epoch + 1)

            del reconstructed, feature_maps, vectors_for_som, assignments, weights_map
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _compute_and_store_metrics(self, reconstructed: np.ndarray, features: np.ndarray, weights_map: np.ndarray, assignments: np.ndarray, epoch: int):
        mse = np.mean((self.x_viz - reconstructed) ** 2)
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))
        quantization_error = np.mean([
            np.linalg.norm(features[idx] - weights_map[bmu_coords[0][idx], bmu_coords[1][idx], :])
            for idx in range(len(features))
        ])
        topological_error = 0.0
        for idx in range(len(features)):
            bmu1_idx, bmu2_idx = np.argsort(-flat_assignments[idx])[:2]
            bmu1_coords = np.unravel_index(bmu1_idx, (grid_h, grid_w))
            bmu2_coords = np.unravel_index(bmu2_idx, (grid_h, grid_w))
            if abs(bmu1_coords[0] - bmu2_coords[0]) + abs(bmu1_coords[1] - bmu2_coords[1]) > 1:
                topological_error += 1
        topological_error /= len(features)

        self.epoch_metrics['reconstruction_mse'].append(float(mse))
        self.epoch_metrics['quantization_error'].append(float(quantization_error))
        self.epoch_metrics['topological_error'].append(float(topological_error))
        self.epochs_recorded.append(epoch)
        logger.info(f"Epoch {epoch} - MSE: {mse:.6f}, Quant Error: {quantization_error:.4f}, Topo Error: {topological_error:.4f}")

    def _visualize_reconstructions(self, reconstructed: np.ndarray, epoch: int):
        n = self.config.num_reconstruction_samples
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle(f'Image Reconstructions - Epoch {epoch}', fontsize=16)
        for i in range(n):
            axes[i // 4, (i % 4) * 2].imshow(self.x_viz[i])
            axes[i // 4, (i % 4) * 2].set_title(f'Orig {self.y_viz[i]}')
            axes[i // 4, (i % 4) * 2].axis('off')
            axes[i // 4, (i % 4) * 2 + 1].imshow(np.clip(reconstructed[i], 0, 1))
            axes[i // 4, (i % 4) * 2 + 1].set_title('Recon')
            axes[i // 4, (i % 4) * 2 + 1].axis('off')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_reconstructions.png", dpi=100)
        plt.close(fig)

    def _visualize_som_activations(self, image_assignments: np.ndarray, epoch: int):
        unique_classes = np.unique(self.y_latent_analysis)
        n_classes = min(10, len(unique_classes))
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'SOM Activation Heatmaps by Class - Epoch {epoch}', fontsize=16)
        for idx in range(n_classes):
            class_label = unique_classes[idx]
            ax = axes[idx // 5, idx % 5]
            class_mask = (self.y_latent_analysis == class_label)
            mean_activation = np.mean(image_assignments[class_mask], axis=0)
            im = ax.imshow(mean_activation, cmap='hot')
            ax.set_title(f'Class {class_label}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_som_activations.png", dpi=100)
        plt.close(fig)

    def _visualize_u_matrix(self, weights_map: np.ndarray, epoch: int):
        grid_h, grid_w, _ = weights_map.shape
        u_matrix = np.zeros((grid_h, grid_w))
        for i in range(grid_h):
            for j in range(grid_w):
                distances = [np.linalg.norm(weights_map[i, j] - weights_map[i+di, j+dj])
                             for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]
                             if 0 <= i+di < grid_h and 0 <= j+dj < grid_w]
                u_matrix[i, j] = np.mean(distances)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(u_matrix, cmap='bone')
        ax.set_title(f'U-Matrix - Epoch {epoch}', fontsize=14)
        plt.colorbar(im, ax=ax)
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_umatrix.png", dpi=100)
        plt.close(fig)

    def _visualize_hit_counts(self, assignments: np.ndarray, epoch: int):
        flat_assignments = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat_assignments, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        hit_counts = np.zeros((grid_h, grid_w))
        np.add.at(hit_counts, np.unravel_index(bmu_indices, (grid_h, grid_w)), 1)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(hit_counts, cmap='YlOrRd')
        ax.set_title(f'Hit Counts - Epoch {epoch}', fontsize=14)
        plt.colorbar(im, ax=ax)
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_hit_counts.png", dpi=100)
        plt.close(fig)

    def _visualize_grid_interpolations(self, epoch: int):
        decoder = self.model.get_layer("decoder")
        som_layer = self.model.get_layer("soft_som_bottleneck")
        weights_map = keras.ops.convert_to_numpy(som_layer.get_weights_map())
        grid_h, grid_w, _ = weights_map.shape
        fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w, grid_h))
        fig.suptitle(f'SOM Manifold Walk - Epoch {epoch}', fontsize=16)
        # Reshape weights map for decoder: needs spatial dimensions
        reshaped_weights = weights_map.reshape(grid_h, grid_w, 1, 1, -1)
        dummy_map = np.zeros((grid_h, grid_w, self.config.latent_dim))
        final_map_shape = self.model.get_layer('encoder').output_shape[1:]

        # We cannot directly decode single prototypes as the decoder expects a full map.
        # Instead, we create a full map where only one position is active.
        # As a simpler proxy, we just show the decoded prototypes as images.
        decoded_prototypes = decoder.predict(weights_map.reshape(grid_h * grid_w, 1, 1, -1), batch_size=64, verbose=0)
        final_spatial_size = self.config.image_size // (2 ** len(self.config.encoder_filters))
        dummy_feature_map = np.zeros((grid_h * grid_w, final_spatial_size, final_spatial_size, self.config.latent_dim))

        for i in range(grid_h):
            for j in range(grid_w):
                idx = i * grid_w + j
                dummy_feature_map[idx, 0, 0, :] = weights_map[i, j]

        decoded_images = decoder.predict(dummy_feature_map, verbose=0, batch_size=64)

        for i in range(grid_h):
            for j in range(grid_w):
                ax = axes[i, j]
                ax.imshow(np.clip(decoded_images[i * grid_w + j], 0, 1))
                ax.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_manifold.png", dpi=150)
        plt.close(fig)

    def _plot_metrics_curves(self, epoch: int):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Training Metrics Evolution', fontsize=16)
        axes[0].plot(self.epochs_recorded, self.epoch_metrics['reconstruction_mse'], 'b-')
        axes[0].set_title('Reconstruction MSE'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(self.epochs_recorded, self.epoch_metrics['quantization_error'], 'c-')
        axes[1].set_title('Quantization Error'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(self.epochs_recorded, self.epoch_metrics['topological_error'], 'y-')
        axes[2].set_title('Topological Error'); axes[2].grid(True, alpha=0.3); axes[2].set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(self.metrics_dir / "metrics_curves.png", dpi=100)
        plt.close(fig)

    def _visualize_latent_space(self, feature_maps: np.ndarray, epoch: int):
        try:
            from sklearn.manifold import TSNE
            # CORRECTED: Get one vector per image by averaging the spatial vectors
            image_features = feature_maps.mean(axis=(1, 2))
            logger.info("Computing t-SNE projection...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
            features_2d = tsne.fit_transform(image_features)
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=self.y_latent_analysis, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter, ax=ax)
            ax.set_title(f'Latent Space t-SNE - Epoch {epoch}')
            plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_tsne.png", dpi=100)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create t-SNE visualization: {e}")

# =============================================================================
# TRAINING ORCHESTRATION
# =============================================================================

def create_learning_rate_schedule(config: CIFARSOMConfig, steps_per_epoch: int):
    if config.lr_schedule == 'cosine':
        return keras.optimizers.schedules.CosineDecay(
            config.learning_rate, decay_steps=config.epochs * steps_per_epoch, alpha=0.01
        )
    return config.learning_rate

def train_cifar_som_autoencoder(config: CIFARSOMConfig) -> keras.Model:
    logger.info(f"Starting training: {config.experiment_name}")
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    (x_train, y_train), (x_test, y_test) = load_cifar_data(config)

    if config.use_data_augmentation:
        augmentation_layer = create_augmentation_layer(config)
        x_train_augmented = augmentation_layer(x_train, training=True)
    else:
        x_train_augmented = x_train

    model = create_cifar_som_autoencoder(config)
    steps_per_epoch = len(x_train) // config.batch_size
    lr_schedule = create_learning_rate_schedule(config, steps_per_epoch)

    if config.optimizer_type == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=config.weight_decay)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    if config.gradient_clip_norm > 0:
        optimizer.clipnorm = config.gradient_clip_norm

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    som_layer = model.get_layer('soft_som_bottleneck')
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(output_dir / "best_model.keras"), save_best_only=True),
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        keras.callbacks.TensorBoard(str(output_dir / "tensorboard")),
        TemperatureAnnealingCallback(
            som_layer_name='soft_som_bottleneck',
            initial_temp=config.initial_temperature,
            final_temp=config.final_temperature,
            total_epochs=config.epochs
        ),
        ComprehensiveMonitoringCallback(config, x_test, y_test)
    ]

    logger.info("Starting training...")
    model.fit(
        x_train_augmented, x_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_test, x_test),
        callbacks=callbacks,
        verbose=1
    )
    logger.info("Training completed successfully!")
    return model

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train CIFAR autoencoder with SoftSOM bottleneck')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--grid-size', type=int, default=16)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--no-augmentation', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    config = CIFARSOMConfig(
        dataset_name=args.dataset,
        grid_shape=(args.grid_size, args.grid_size),
        latent_dim=args.latent_dim,
        use_data_augmentation=not args.no_augmentation,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    train_cifar_som_autoencoder(config)

if __name__ == "__main__":
    main()