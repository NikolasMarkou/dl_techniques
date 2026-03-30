"""CIFAR Autoencoder with Soft Self-Organizing Map Bottleneck."""

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

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.som_nd_soft_layer import SoftSOMLayer


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CIFARSOMConfig:
    """Configuration for CIFAR + SoftSOM autoencoder training."""

    dataset_name: str = 'cifar10'
    image_size: int = 32
    use_data_augmentation: bool = True
    augmentation_strength: float = 0.5

    encoder_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3])

    grid_shape: Tuple[int, int] = (10, 10)
    som_temperature: float = 1.0
    som_use_per_dim_softmax: bool = True
    som_reconstruction_weight: float = 0.0
    som_topological_weight: float = 1.0
    som_sharpness_weight: float = 0.0

    decoder_filters: List[int] = field(default_factory=lambda: [256, 128, 64])
    use_batch_norm: bool = True
    use_dropout: bool = True
    dropout_rate: float = 0.1

    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    lr_schedule: str = 'cosine'
    optimizer_type: str = 'adamw'
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    use_progressive_training: bool = False
    progressive_epochs: List[int] = field(default_factory=lambda: [20, 30, 50])
    initial_size: int = 16

    reconstruction_loss_type: str = 'mse'
    perceptual_weight: float = 0.5

    monitor_every_n_epochs: int = 5
    num_reconstruction_samples: int = 16
    num_interpolation_samples: int = 8
    save_latent_projections: bool = True

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
# DATA LOADING
# =============================================================================

def load_cifar_data(config: CIFARSOMConfig) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """Load and preprocess CIFAR dataset, normalized to [0, 1]."""
    logger.info(f"Loading {config.dataset_name}...")
    if config.dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    logger.info(f"Train: {x_train.shape[0]}, Test: {x_test.shape[0]}, Shape: {x_train.shape[1:]}")
    return (x_train, y_train), (x_test, y_test)


def create_augmentation_layer(config: CIFARSOMConfig) -> keras.layers.Layer:
    """Create data augmentation preprocessing layer."""
    s = config.augmentation_strength
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1 * s),
        keras.layers.RandomZoom(0.1 * s),
        keras.layers.RandomTranslation(0.1 * s, 0.1 * s),
        keras.layers.RandomBrightness(0.2 * s),
        keras.layers.RandomContrast(0.2 * s),
    ], name="augmentation")


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def create_encoder(config: CIFARSOMConfig) -> keras.Model:
    """Create convolutional encoder preserving spatial information."""
    inputs = keras.Input(shape=(config.image_size, config.image_size, 3), name="image_input")
    x = inputs
    for i, (filters, ks) in enumerate(zip(config.encoder_filters, config.encoder_kernel_sizes)):
        x = keras.layers.Conv2D(filters, ks, strides=2, padding='same', name=f"encoder_conv_{i+1}")(x)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization(name=f"encoder_bn_{i+1}")(x)
        x = keras.layers.Activation('gelu', name=f"encoder_relu_{i+1}")(x)
        if config.use_dropout:
            x = keras.layers.Dropout(config.dropout_rate, name=f"encoder_dropout_{i+1}")(x)
    return keras.Model(inputs, x, name="encoder")


def create_decoder(config: CIFARSOMConfig) -> keras.Model:
    """Create convolutional decoder with transpose convolutions."""
    final_spatial = config.image_size // (2 ** len(config.encoder_filters))
    channels = config.encoder_filters[-1]

    inputs = keras.Input(shape=(final_spatial, final_spatial, channels), name="latent_map_input")
    x = inputs
    for i, filters in enumerate(config.decoder_filters):
        x = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        x = keras.layers.Conv2D(filters, 3, padding='same', name=f"decoder_conv_{i+1}")(x)
        if config.use_batch_norm:
            x = keras.layers.BatchNormalization(name=f"decoder_bn_{i+1}")(x)
        x = keras.layers.Activation('gelu', name=f"decoder_relu_{i+1}")(x)
        if config.use_dropout and i < len(config.decoder_filters) - 1:
            x = keras.layers.Dropout(config.dropout_rate, name=f"decoder_dropout_{i+1}")(x)

    x = keras.layers.Conv2D(config.decoder_filters[-1], 3, padding='same', activation='linear')(x)
    if config.use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('gelu')(x)
    outputs = keras.layers.Conv2D(3, 1, padding='same', activation='sigmoid', name="reconstructed_image")(x)
    return keras.Model(inputs, outputs, name="decoder")


def create_cifar_som_autoencoder(config: CIFARSOMConfig) -> keras.Model:
    """Create autoencoder with SoftSOM topological bottleneck."""
    logger.info("Creating CIFAR SoftSOM Autoencoder...")

    encoder = create_encoder(config)
    decoder = create_decoder(config)

    inputs = keras.Input(shape=(config.image_size, config.image_size, 3), name="input_images")
    feature_map = encoder(inputs)
    map_shape = keras.ops.shape(feature_map)
    h, w, channels = map_shape[1], map_shape[2], map_shape[3]

    reshaped = keras.layers.Reshape((-1, channels))(feature_map)

    som_layer = SoftSOMLayer(
        grid_shape=config.grid_shape, input_dim=channels,
        temperature=config.som_temperature,
        use_per_dimension_softmax=config.som_use_per_dim_softmax,
        use_reconstruction_loss=config.som_reconstruction_weight > 0,
        reconstruction_weight=config.som_reconstruction_weight,
        topological_weight=config.som_topological_weight,
        sharpness_weight=config.som_sharpness_weight,
        name="soft_som_bottleneck"
    )
    som_features = keras.layers.TimeDistributed(som_layer, name="time_distributed_som")(reshaped)
    som_map = keras.layers.Reshape((h, w, channels))(som_features)
    reconstructed = decoder(som_map)

    model = keras.Model(inputs, reconstructed, name="cifar_softsom_autoencoder")
    model.summary()
    return model


# =============================================================================
# MONITORING CALLBACK
# =============================================================================

class ComprehensiveMonitoringCallback(keras.callbacks.Callback):
    """Advanced monitoring with reconstruction, SOM, and latent space visualizations."""

    def __init__(self, config: CIFARSOMConfig, x_test: np.ndarray, y_test: np.ndarray):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs

        self.x_viz = x_test[:config.num_reconstruction_samples]
        self.y_viz = y_test[:config.num_reconstruction_samples].flatten()
        self.x_latent_analysis = x_test[:1000]
        self.y_latent_analysis = y_test[:1000].flatten()

        num_classes = 10 if config.dataset_name == 'cifar10' else 100
        self.tracked_samples, self.tracked_labels = [], []
        for cid in range(min(10, num_classes)):
            indices = np.where(self.y_latent_analysis == cid)[0]
            if len(indices) > 0:
                self.tracked_samples.append(self.x_latent_analysis[indices[0]])
                self.tracked_labels.append(cid)
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

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.monitor_freq != 0:
            return
        logger.info(f"Visualizing epoch {epoch + 1}...")

        try:
            reconstructed = self.model.predict(self.x_viz, verbose=0, batch_size=32)
            wrapper = self.model.get_layer("time_distributed_som")
            som_layer = wrapper.layer
            encoder = self.model.get_layer("encoder")
            decoder = self.model.get_layer("decoder")

            feature_map = encoder.predict(self.x_latent_analysis, verbose=0, batch_size=128)
            features = feature_map.reshape(-1, som_layer.input_dim)
            h, w = feature_map.shape[1], feature_map.shape[2]
            vectors_per_image = h * w
            expanded_y = np.repeat(self.y_latent_analysis, vectors_per_image)

            assignments_list = []
            for i in range(0, len(features), 512):
                assignments_list.append(som_layer.get_soft_assignments(features[i:i+512]))
            assignments = keras.ops.convert_to_numpy(np.concatenate(assignments_list, axis=0))
            weights_map = keras.ops.convert_to_numpy(som_layer.get_weights_map())

            self._compute_and_store_metrics(reconstructed, features, weights_map, assignments, epoch + 1)
            self._visualize_reconstructions(reconstructed, epoch + 1)
            self._visualize_som_activations(assignments, expanded_y, epoch + 1)
            self._visualize_prototype_vectors(weights_map, epoch + 1)
            self._visualize_u_matrix(weights_map, epoch + 1)
            self._visualize_hit_counts(assignments, epoch + 1)
            self._visualize_confusion_on_grid(assignments, expanded_y, epoch + 1)
            self._visualize_grid_interpolations(decoder, weights_map, h, w, epoch + 1)
            self._track_sample_trajectories(encoder, som_layer, epoch + 1)

            if len(self.epochs_recorded) > 1:
                self._plot_metrics_curves(epoch + 1)
                self._visualize_bmu_trajectories(epoch + 1)

            if self.config.save_latent_projections and (epoch + 1) % (self.monitor_freq * 2) == 0:
                self._visualize_latent_space(features, expanded_y, epoch + 1)

            del reconstructed, features, assignments, weights_map, expanded_y
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _compute_and_store_metrics(self, reconstructed, features, weights_map, assignments, epoch):
        mse = np.mean((self.x_viz - reconstructed) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))

        flat = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        quant_error = sum(
            np.linalg.norm(features[idx] - weights_map[i, j, :])
            for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1]))
        ) / len(features)

        topo_error = 0.0
        for idx in range(len(features)):
            sorted_idx = np.argsort(-flat[idx])
            c1 = np.unravel_index(sorted_idx[0], (grid_h, grid_w))
            c2 = np.unravel_index(sorted_idx[1], (grid_h, grid_w))
            if abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) > 1:
                topo_error += 1
        topo_error /= len(features)

        self.epoch_metrics['reconstruction_mse'].append(float(mse))
        self.epoch_metrics['psnr'].append(float(psnr))
        self.epoch_metrics['quantization_error'].append(float(quant_error))
        self.epoch_metrics['topological_error'].append(float(topo_error))
        self.epochs_recorded.append(epoch)
        logger.info(f"Epoch {epoch} - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB, "
                     f"Quant: {quant_error:.4f}, Topo: {topo_error:.4f}")

    def _visualize_reconstructions(self, reconstructed, epoch):
        n = len(self.x_viz)
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 3, n_rows * 6))
        fig.suptitle(f'Reconstructions - Epoch {epoch}', fontsize=16)

        for i in range(n):
            row, col = (i // n_cols) * 2, i % n_cols
            ax_orig = axes[row, col] if n_rows > 1 else axes[0, col]
            ax_recon = axes[row + 1, col] if n_rows > 1 else axes[1, col]

            ax_orig.imshow(self.x_viz[i])
            ax_orig.set_title(f'Original (Class {self.y_viz[i]})', fontsize=10)
            ax_orig.axis('off')

            ax_recon.imshow(np.clip(reconstructed[i], 0, 1))
            mse = np.mean((self.x_viz[i] - reconstructed[i]) ** 2)
            ax_recon.set_title(f'Recon (MSE: {mse:.4f})', fontsize=10)
            ax_recon.axis('off')

        for i in range(n, n_rows * n_cols):
            row, col = (i // n_cols) * 2, i % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
                axes[row + 1, col].axis('off')

        plt.tight_layout()
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_reconstructions.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_som_activations(self, assignments, y_expanded, epoch):
        try:
            unique_classes = np.unique(self.y_latent_analysis)[:10]
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f'SOM Activations by Class - Epoch {epoch}', fontsize=16)
            for idx, cl in enumerate(unique_classes):
                ax = axes.flatten()[idx]
                mask = (y_expanded == cl)
                if np.any(mask):
                    im = ax.imshow(np.mean(assignments[mask], axis=0), cmap='hot')
                    ax.set_title(f'Class {cl}')
                    plt.colorbar(im, ax=ax)
                else:
                    ax.set_title(f'Class {cl} (No Data)')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_som_activations.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to visualize SOM activations: {e}")

    def _visualize_prototype_vectors(self, weights_map, epoch):
        grid_h, grid_w, input_dim = weights_map.shape
        fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 1.5, grid_h * 1.5))
        fig.suptitle(f'SOM Prototypes - Epoch {epoch}', fontsize=16)
        for i in range(grid_h):
            for j in range(grid_w):
                ax = axes[i, j] if grid_h > 1 and grid_w > 1 else axes
                proto = weights_map[i, j, :]
                proto_norm = (proto - proto.min()) / (proto.max() - proto.min() + 1e-8)
                pad_size = int(np.ceil(np.sqrt(input_dim))) ** 2
                padded = np.pad(proto_norm, (0, pad_size - input_dim), mode='constant')
                gs = int(np.sqrt(pad_size))
                ax.imshow(padded.reshape(gs, gs), cmap='viridis')
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_prototypes.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_u_matrix(self, weights_map, epoch):
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
                            distances.append(np.linalg.norm(current - weights_map[ni, nj, :]))
                u_matrix[i, j] = np.mean(distances) if distances else 0

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(u_matrix, cmap='bone')
        ax.set_title(f'U-Matrix - Epoch {epoch}')
        plt.colorbar(im, ax=ax, label='Avg Distance to Neighbors')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_umatrix.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_hit_counts(self, assignments, epoch):
        flat = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))
        hit_counts = np.zeros((grid_h, grid_w))
        for i, j in zip(bmu_coords[0], bmu_coords[1]):
            hit_counts[i, j] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(hit_counts, cmap='YlOrRd')
        ax.set_title(f'Hit Counts - Epoch {epoch}')
        plt.colorbar(im, ax=ax, label='Number of Samples')
        total = grid_h * grid_w
        active = np.sum(hit_counts > 0)
        fig.suptitle(f'Active: {active}/{total} neurons')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_hit_counts.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_confusion_on_grid(self, assignments, y_expanded, epoch):
        flat = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        num_classes = len(np.unique(self.y_latent_analysis))
        confusion_grid = np.zeros((grid_h, grid_w, num_classes))
        for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1])):
            label = y_expanded[idx]
            if label < num_classes:
                confusion_grid[i, j, int(label)] += 1

        fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 1.2, grid_h * 1.2))
        fig.suptitle(f'Class Distribution - Epoch {epoch}', fontsize=16)
        for i in range(grid_h):
            for j in range(grid_w):
                ax = axes[i, j]
                counts = confusion_grid[i, j, :]
                if counts.sum() > 0:
                    colors = plt.cm.tab10(np.arange(num_classes))
                    ax.pie(counts[counts > 0], colors=colors[counts > 0])
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_confusion_grid.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_grid_interpolations(self, decoder, weights_map, h, w, epoch):
        try:
            grid_h, grid_w, _ = weights_map.shape
            fig, axes = plt.subplots(2, grid_w, figsize=(grid_w * 2, 4))
            fig.suptitle(f'SOM Grid Traversals - Epoch {epoch}', fontsize=16)

            middle_row = grid_h // 2
            for col in range(grid_w):
                pure_map = np.tile(weights_map[middle_row, col, :], (h, w, 1))
                decoded = decoder.predict(np.expand_dims(pure_map, 0), verbose=0)
                axes[0, col].imshow(np.clip(decoded[0], 0, 1))
                axes[0, col].axis('off')

            middle_col = grid_w // 2
            for row in range(min(grid_h, grid_w)):
                pure_map = np.tile(weights_map[row, middle_col, :], (h, w, 1))
                decoded = decoder.predict(np.expand_dims(pure_map, 0), verbose=0)
                axes[1, row].imshow(np.clip(decoded[0], 0, 1))
                axes[1, row].axis('off')

            plt.tight_layout()
            plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_grid_traversals.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create grid interpolations: {e}")

    def _track_sample_trajectories(self, encoder, som_layer, epoch):
        try:
            feature_maps = encoder.predict(self.tracked_samples, verbose=0)
            grid_h, grid_w = self.config.grid_shape
            for idx, label in enumerate(self.tracked_labels):
                features = feature_maps[idx].reshape(-1, som_layer.input_dim)
                assignments = som_layer.get_soft_assignments(features)
                flat = keras.ops.reshape(assignments, (len(features), -1))
                bmu_indices = keras.ops.argmax(flat, axis=1)
                bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))
                self.bmu_trajectories[label].append(
                    (epoch, (np.mean(bmu_coords[0]), np.mean(bmu_coords[1])))
                )
        except Exception as e:
            logger.warning(f"Failed to track trajectories: {e}")

    def _plot_metrics_curves(self, epoch):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        for (r, c), (key, label) in {
            (0, 0): ('reconstruction_mse', 'MSE'),
            (0, 1): ('psnr', 'PSNR (dB)'),
            (1, 0): ('quantization_error', 'Quantization Error'),
            (1, 1): ('topological_error', 'Topological Error')
        }.items():
            if key in self.epoch_metrics:
                axes[r, c].plot(self.epochs_recorded, self.epoch_metrics[key])
                axes[r, c].set_title(label)
                axes[r, c].grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.metrics_dir / "metrics_curves.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_bmu_trajectories(self, epoch):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'BMU Trajectories - Epoch {epoch}', fontsize=16)
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
                    ax.plot(x[i:i+2], y[i:i+2], 'o-', color=color)
            ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_trajectories.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_latent_space(self, features, y_expanded, epoch):
        try:
            from sklearn.manifold import TSNE
            logger.info("Computing t-SNE projection...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            subset = min(len(features), 5000)
            indices = np.random.choice(len(features), subset, replace=False)
            features_2d = tsne.fit_transform(features[indices])

            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                                 c=y_expanded[indices], cmap='tab10', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax, label='Class')
            ax.set_title(f'Latent Space t-SNE - Epoch {epoch}')
            plt.tight_layout()
            plt.savefig(self.viz_dir / f"epoch_{epoch:03d}_tsne.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create t-SNE visualization: {e}")


# =============================================================================
# TRAINING
# =============================================================================

def create_learning_rate_schedule(config: CIFARSOMConfig, steps_per_epoch: int):
    """Create learning rate schedule based on config."""
    if config.lr_schedule == 'cosine':
        return keras.optimizers.schedules.CosineDecay(
            config.learning_rate, decay_steps=config.epochs * steps_per_epoch, alpha=0.01
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
        optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=config.weight_decay)
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
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(output_dir / "tensorboard")),
        ComprehensiveMonitoringCallback(config, x_test, y_test)
    ]

    model.fit(
        x_train_aug, x_train, batch_size=config.batch_size,
        epochs=config.epochs, validation_data=(x_test, x_test),
        callbacks=callbacks, verbose=1
    )

    test_loss, _ = model.evaluate(x_test, x_test, verbose=0)
    logger.info(f"Final test loss: {test_loss:.6f}")
    return model


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train CIFAR autoencoder with SoftSOM bottleneck',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--grid-size', type=int, default=16)
    parser.add_argument('--som-temperature', type=float, default=1.0)
    parser.add_argument('--no-augmentation', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_gpu()

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

    aug_status = 'on' if config.use_data_augmentation else 'off'
    logger.info(f"CIFAR SoftSOM AE | {config.dataset_name} | Grid: {config.grid_shape} | "
                f"Temp: {config.som_temperature} | Aug: {aug_status}")

    train_cifar_som_autoencoder(config)


if __name__ == "__main__":
    main()
