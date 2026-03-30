"""MNIST Classification with Soft Self-Organizing Map Features."""

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

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.som_nd_soft_layer import SoftSOMLayer


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MNISTSOMConfig:
    """Configuration for MNIST + SoftSOM classification training."""
    grid_shape: Tuple[int, int] = (10, 10)
    som_temperature: float = 0.5
    som_reconstruction_weight: float = 1.0
    som_topological_weight: float = 0.2
    som_sharpness_weight: float = 0.05
    dense_units: int = 256
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    monitor_every_n_epochs: int = 2

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"mnist_softsom_{timestamp}"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_mnist_data(config: MNISTSOMConfig) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """Load and preprocess MNIST dataset (normalized, flattened, one-hot)."""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    logger.info(f"Train: {x_train.shape[0]}, Test: {x_test.shape[0]}, Dim: {x_train.shape[1]}")
    return (x_train, y_train), (x_test, y_test)


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_mnist_som_classifier(config: MNISTSOMConfig) -> keras.Model:
    """Create MNIST classifier: Input(784) -> Dense -> SoftSOM -> Dense(10)."""
    logger.info("Creating MNIST + SoftSOM model...")

    inputs = keras.Input(shape=(784,), name="input_images")
    x = keras.layers.Dense(config.dense_units, activation='relu', name="dense_features")(inputs)

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

    outputs = keras.layers.Dense(10, activation='softmax', name="digit_classifier")(som_features)
    model = keras.Model(inputs, outputs, name="mnist_softsom_classifier")
    model.summary()
    return model


# =============================================================================
# VISUALIZATION CALLBACK
# =============================================================================

class SOMVisualizationCallback(keras.callbacks.Callback):
    """Visualizes SOM activation patterns, prototypes, U-Matrix, and metrics."""

    def __init__(self, config: MNISTSOMConfig, x_test: np.ndarray, y_test: np.ndarray):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs

        self.x_viz = x_test[:1000]
        self.y_viz = np.argmax(y_test[:1000], axis=1)

        self.tracked_samples = []
        self.tracked_labels = []
        for digit in range(10):
            indices = np.where(self.y_viz == digit)[0]
            if len(indices) > 0:
                self.tracked_samples.append(self.x_viz[indices[0]])
                self.tracked_labels.append(digit)
        self.tracked_samples = np.array(self.tracked_samples)

        self.bmu_trajectories: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {d: [] for d in range(10)}
        self.output_dir = Path(config.output_dir) / config.experiment_name / "som_visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.quantization_errors: List[float] = []
        self.topological_errors: List[float] = []
        self.epochs_recorded: List[int] = []

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.monitor_freq != 0:
            return
        logger.info(f"Creating SOM visualizations for epoch {epoch + 1}...")

        try:
            som_layer = self.model.get_layer("soft_som")
            feature_model = keras.Model(self.model.input, som_layer.input)
            features = feature_model.predict(self.x_viz, verbose=0, batch_size=256)

            assignments = keras.ops.convert_to_numpy(som_layer.get_soft_assignments(features))
            weights_map = keras.ops.convert_to_numpy(som_layer.get_weights_map())

            self._visualize_class_activation_maps(assignments, epoch + 1)
            self._visualize_bmu_distribution(assignments, epoch + 1)
            self._visualize_prototype_vectors(weights_map, epoch + 1)
            self._visualize_u_matrix(weights_map, epoch + 1)
            self._visualize_hit_counts(assignments, epoch + 1)
            self._visualize_confusion_on_grid(assignments, epoch + 1)
            self._compute_and_track_metrics(features, weights_map, assignments, epoch + 1)
            self._track_sample_trajectories(epoch + 1)

            if len(self.quantization_errors) > 1:
                self._plot_error_metrics(epoch + 1)
                self._visualize_bmu_trajectories(epoch + 1)

            del features, assignments, weights_map
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create SOM visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_class_activation_maps(self, assignments: np.ndarray, epoch: int):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'SOM Activation Maps by Digit - Epoch {epoch}', fontsize=16)
        for digit in range(10):
            ax = axes[digit // 5, digit % 5]
            mean_act = np.mean(assignments[self.y_viz == digit], axis=0)
            im = ax.imshow(mean_act, cmap='hot', interpolation='nearest')
            ax.set_title(f'Digit {digit}', fontsize=14)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_activation_maps.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_bmu_distribution(self, assignments: np.ndarray, epoch: int):
        flat = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.stack(np.unravel_index(bmu_indices, (grid_h, grid_w)), axis=1)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'BMU Distribution by Digit - Epoch {epoch}', fontsize=16)
        for digit in range(10):
            ax = axes[digit // 5, digit % 5]
            digit_bmus = bmu_coords[self.y_viz == digit]
            H, _, _ = np.histogram2d(digit_bmus[:, 0], digit_bmus[:, 1],
                                     bins=[grid_h, grid_w], range=[[0, grid_h], [0, grid_w]])
            im = ax.imshow(H, cmap='viridis', interpolation='nearest', origin='lower')
            ax.set_title(f'Digit {digit} (n={np.sum(self.y_viz == digit)})', fontsize=12)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_bmu_distribution.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_prototype_vectors(self, weights_map: np.ndarray, epoch: int):
        grid_h, grid_w, input_dim = weights_map.shape
        fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 1.5, grid_h * 1.5))
        fig.suptitle(f'SOM Prototypes - Epoch {epoch}', fontsize=16)

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

                proto = weights_map[i, j, :]
                proto_norm = (proto - proto.min()) / (proto.max() - proto.min() + 1e-8)
                grid_size = int(np.sqrt(input_dim))
                if grid_size * grid_size == input_dim:
                    proto_img = proto_norm.reshape(grid_size, grid_size)
                else:
                    pad_size = int(np.ceil(np.sqrt(input_dim))) ** 2
                    padded = np.pad(proto_norm, (0, pad_size - input_dim), mode='constant')
                    grid_size = int(np.sqrt(pad_size))
                    proto_img = padded.reshape(grid_size, grid_size)
                ax.imshow(proto_img, cmap='viridis', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'{i},{j}', fontsize=6)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_prototypes.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_u_matrix(self, weights_map: np.ndarray, epoch: int):
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
        im = ax.imshow(u_matrix, cmap='bone', interpolation='nearest')
        ax.set_title(f'U-Matrix - Epoch {epoch}\nDark=Clusters, Bright=Boundaries', fontsize=14)
        for i in range(grid_h + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(grid_w + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Average Distance to Neighbors')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_umatrix.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_hit_counts(self, assignments: np.ndarray, epoch: int):
        flat = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        hit_counts = np.zeros((grid_h, grid_w))
        for i, j in zip(bmu_coords[0], bmu_coords[1]):
            hit_counts[i, j] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        im1 = ax1.imshow(hit_counts, cmap='YlOrRd', interpolation='nearest')
        ax1.set_title(f'Hit Counts (Linear) - Epoch {epoch}', fontsize=14)
        for i in range(grid_h):
            for j in range(grid_w):
                count = int(hit_counts[i, j])
                ax1.text(j, i, str(count), ha='center', va='center',
                         color='white' if count > hit_counts.max() * 0.5 else 'black', fontsize=8)
        plt.colorbar(im1, ax=ax1, label='Number of Samples')

        im2 = ax2.imshow(np.log10(hit_counts + 1), cmap='YlOrRd', interpolation='nearest')
        ax2.set_title(f'Hit Counts (Log) - Epoch {epoch}', fontsize=14)
        plt.colorbar(im2, ax=ax2, label='Log10(Samples + 1)')

        total = grid_h * grid_w
        active = np.sum(hit_counts > 0)
        fig.suptitle(f'Active: {active}/{total} | Dead: {total - active} | '
                     f'Max: {int(hit_counts.max())} | Mean: {hit_counts.mean():.1f}', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_hit_counts.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_confusion_on_grid(self, assignments: np.ndarray, epoch: int):
        flat = assignments.reshape(assignments.shape[0], -1)
        bmu_indices = np.argmax(flat, axis=1)
        grid_h, grid_w = assignments.shape[1], assignments.shape[2]
        bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

        confusion_grid = np.zeros((grid_h, grid_w, 10))
        for idx, (i, j) in enumerate(zip(bmu_coords[0], bmu_coords[1])):
            confusion_grid[i, j, self.y_viz[idx]] += 1

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

                counts = confusion_grid[i, j, :]
                total = counts.sum()
                if total > 0:
                    colors = plt.cm.tab10(np.arange(10))
                    non_zero = counts > 0
                    ax.pie(counts[non_zero], colors=colors[non_zero], startangle=90, counterclock=False)
                    dominant = np.argmax(counts)
                    purity = counts[dominant] / total
                    ax.text(0, -1.3, f'{dominant}\n({purity:.1%})', ha='center', va='top',
                            fontsize=6, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax.text(0, 0, 'Empty', ha='center', va='center', fontsize=8)
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.5, 1.2)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_confusion_grid.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _compute_and_track_metrics(self, features, weights_map, assignments, epoch):
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

        self.quantization_errors.append(quant_error)
        self.topological_errors.append(topo_error)
        self.epochs_recorded.append(epoch)
        logger.info(f"Epoch {epoch} - Quant Error: {quant_error:.4f}, Topo Error: {topo_error:.4f}")

    def _track_sample_trajectories(self, epoch: int):
        try:
            som_layer = self.model.get_layer("soft_som")
            feature_model = keras.Model(self.model.input, som_layer.input)
            features = feature_model.predict(self.tracked_samples, verbose=0, batch_size=32)
            assignments = keras.ops.convert_to_numpy(som_layer.get_soft_assignments(features))

            flat = assignments.reshape(assignments.shape[0], -1)
            bmu_indices = np.argmax(flat, axis=1)
            grid_h, grid_w = assignments.shape[1], assignments.shape[2]
            bmu_coords = np.unravel_index(bmu_indices, (grid_h, grid_w))

            for idx, label in enumerate(self.tracked_labels):
                coord = (int(bmu_coords[0][idx]), int(bmu_coords[1][idx]))
                self.bmu_trajectories[label].append((epoch, coord))
        except Exception as e:
            logger.warning(f"Failed to track trajectories: {e}")

    def _plot_error_metrics(self, epoch: int):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(self.epochs_recorded, self.quantization_errors, 'b-', linewidth=2, marker='o')
        ax1.set_title('Quantization Error')
        ax1.set_xlabel('Epoch')
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.epochs_recorded, self.topological_errors, 'r-', linewidth=2, marker='s')
        ax2.set_title('Topological Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_error_metrics.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    def _visualize_bmu_trajectories(self, epoch: int):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'BMU Trajectories (up to Epoch {epoch})', fontsize=16)
        grid_h, grid_w = self.config.grid_shape

        for digit in range(10):
            ax = axes[digit // 5, digit % 5]
            ax.set_xlim(-0.5, grid_w - 0.5)
            ax.set_ylim(-0.5, grid_h - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Digit {digit}', fontsize=12)
            ax.grid(True, alpha=0.3)

            if digit in self.bmu_trajectories and self.bmu_trajectories[digit]:
                trajectory = self.bmu_trajectories[digit]
                x_coords = [t[1][1] for t in trajectory]
                y_coords = [t[1][0] for t in trajectory]

                for i in range(len(x_coords) - 1):
                    color = plt.cm.coolwarm(i / max(1, len(x_coords) - 1))
                    ax.plot(x_coords[i:i+2], y_coords[i:i+2], 'o-', color=color, linewidth=2, markersize=6, alpha=0.7)

                ax.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start',
                        markeredgecolor='black', markeredgewidth=2)
                ax.plot(x_coords[-1], y_coords[-1], 'r*', markersize=16, label='Current',
                        markeredgecolor='black', markeredgewidth=2)
                ax.legend(loc='upper right', fontsize=8)
            else:
                ax.text(grid_w / 2, grid_h / 2, 'No data', ha='center', va='center')
            ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"epoch_{epoch:03d}_trajectories.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()


# =============================================================================
# TRAINING
# =============================================================================

def train_mnist_som_classifier(config: MNISTSOMConfig) -> keras.Model:
    """Train MNIST classifier with SoftSOM feature extraction."""
    logger.info(f"Starting training: {config.experiment_name}")

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    (x_train, y_train), (x_test, y_test) = load_mnist_data(config)
    model = create_mnist_som_classifier(config)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv"), append=True),
        keras.callbacks.TensorBoard(log_dir=str(output_dir / "tensorboard"), histogram_freq=1),
        SOMVisualizationCallback(config, x_test, y_test)
    ]

    history = model.fit(
        x_train, y_train, batch_size=config.batch_size,
        epochs=config.epochs, validation_data=(x_test, y_test),
        callbacks=callbacks, verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Final test accuracy: {test_acc:.4f}")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history_dict, f, indent=2)

    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'total_params': int(model.count_params())
    }
    with open(output_dir / "final_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return model


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train MNIST classifier with SoftSOM layer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--som-temperature', type=float, default=0.5)
    parser.add_argument('--output-dir', type=str, default='results')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_gpu()

    config = MNISTSOMConfig(
        grid_shape=(args.grid_size, args.grid_size),
        som_temperature=args.som_temperature,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )

    logger.info(f"MNIST + SoftSOM | Grid: {config.grid_shape} | "
                f"Temp: {config.som_temperature} | Batch: {config.batch_size} | "
                f"Epochs: {config.epochs}")

    train_mnist_som_classifier(config)


if __name__ == "__main__":
    main()
