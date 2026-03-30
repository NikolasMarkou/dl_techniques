"""Training pipeline for Anchored Branched UPT (AB-UPT) CFD surrogate model.

Implements data preprocessing/normalization, custom data generators,
multi-task loss functions, training loop with validation, and checkpointing.
Based on the PyTorch implementation, adapted for Keras 3.x.
"""

import os
import json
import numpy as np
import keras
from keras import ops
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

from dl_techniques.utils.logger import logger
from dl_techniques.models.anchored_branched_upt import AnchoredBranchedUPT, create_abupt_model

from train.common import setup_gpu


# ---------------------------------------------------------------------
# CFD normalization statistics
# ---------------------------------------------------------------------

@dataclass
class NormalizationStats:
    """Data normalization statistics for CFD data."""
    raw_pos_min: Tuple[float, ...] = (-40.0,)
    raw_pos_max: Tuple[float, ...] = (80.0,)

    surface_pressure_mean: Tuple[float, ...] = (-2.29772e02,)
    surface_pressure_std: Tuple[float, ...] = (2.69345e02,)
    surface_wallshearstress_mean: Tuple[float, float, float] = (-1.20054e00, 1.49358e-03, -7.20107e-02)
    surface_wallshearstress_std: Tuple[float, float, float] = (2.07670e00, 1.35628e00, 1.11426e00)

    volume_totalpcoeff_mean: Tuple[float, ...] = (1.71387e-01,)
    volume_totalpcoeff_std: Tuple[float, ...] = (5.00826e-01,)
    volume_velocity_mean: Tuple[float, float, float] = (1.67909e01, -3.82238e-02, 4.07968e-01)
    volume_velocity_std: Tuple[float, float, float] = (1.64115e01, 8.63614e00, 6.64996e00)
    volume_vorticity_logscale_mean: Tuple[float, float, float] = (-1.47814e-02, 7.87642e-01, 2.81023e-03)
    volume_vorticity_logscale_std: Tuple[float, float, float] = (5.45681e00, 5.77081e00, 5.46175e00)


# ---------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------

class DataPreprocessor:
    """Data preprocessing for CFD data with normalization/denormalization."""

    def __init__(self, stats: NormalizationStats, scale: float = 1000.0):
        self.stats = stats
        self.scale = scale

        self.raw_pos_min = ops.array(stats.raw_pos_min)
        self.raw_pos_max = ops.array(stats.raw_pos_max)
        self.raw_size = self.raw_pos_max - self.raw_pos_min

        self.surface_pressure_mean = ops.array(stats.surface_pressure_mean)
        self.surface_pressure_std = ops.array(stats.surface_pressure_std)
        self.surface_wss_mean = ops.array(stats.surface_wallshearstress_mean)
        self.surface_wss_std = ops.array(stats.surface_wallshearstress_std)

        self.volume_pressure_mean = ops.array(stats.volume_totalpcoeff_mean)
        self.volume_pressure_std = ops.array(stats.volume_totalpcoeff_std)
        self.volume_velocity_mean = ops.array(stats.volume_velocity_mean)
        self.volume_velocity_std = ops.array(stats.volume_velocity_std)
        self.volume_vorticity_mean = ops.array(stats.volume_vorticity_logscale_mean)
        self.volume_vorticity_std = ops.array(stats.volume_vorticity_logscale_std)

    def normalize_positions(self, positions):
        return (positions - self.raw_pos_min) / self.raw_size * self.scale

    def denormalize_positions(self, positions):
        return (positions / self.scale) * self.raw_size + self.raw_pos_min

    def normalize_surface_pressure(self, pressure):
        return (pressure - self.surface_pressure_mean) / self.surface_pressure_std

    def normalize_surface_wallshearstress(self, wss):
        return (wss - self.surface_wss_mean) / self.surface_wss_std

    def normalize_volume_pressure(self, pressure):
        return (pressure - self.volume_pressure_mean) / self.volume_pressure_std

    def normalize_volume_velocity(self, velocity):
        return (velocity - self.volume_velocity_mean) / self.volume_velocity_std

    def normalize_volume_vorticity(self, vorticity):
        log_vorticity = ops.sign(vorticity) * ops.log1p(ops.abs(vorticity))
        return (log_vorticity - self.volume_vorticity_mean) / self.volume_vorticity_std

    def denormalize_surface_pressure(self, pressure):
        return pressure * self.surface_pressure_std + self.surface_pressure_mean

    def denormalize_surface_wallshearstress(self, wss):
        return wss * self.surface_wss_std + self.surface_wss_mean

    def denormalize_volume_pressure(self, pressure):
        return pressure * self.volume_pressure_std + self.volume_pressure_mean

    def denormalize_volume_velocity(self, velocity):
        return velocity * self.volume_velocity_std + self.volume_velocity_mean

    def denormalize_volume_vorticity(self, vorticity):
        log_vorticity = vorticity * self.volume_vorticity_std + self.volume_vorticity_mean
        return ops.sign(log_vorticity) * (ops.exp(ops.abs(log_vorticity)) - 1)


# ---------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------

class CFDDataGenerator(keras.utils.PyDataset):
    """Data generator for CFD training data."""

    def __init__(
            self,
            data_samples: List[Dict[str, Any]],
            preprocessor: DataPreprocessor,
            batch_size: int = 1,
            num_geometry_points: int = 1000,
            num_surface_anchor_points: int = 500,
            num_volume_anchor_points: int = 800,
            num_geometry_supernodes: int = 200,
            use_query_positions: bool = True,
            shuffle: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.data_samples = data_samples
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.num_geometry_points = num_geometry_points
        self.num_surface_anchor_points = num_surface_anchor_points
        self.num_volume_anchor_points = num_volume_anchor_points
        self.num_geometry_supernodes = num_geometry_supernodes
        self.use_query_positions = use_query_positions
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data_samples))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return (len(self.data_samples) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.data_samples[i] for i in batch_indices]
        return self._generate_batch(batch_samples)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_batch(self, samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a batch from samples (currently batch_size=1)."""
        sample = samples[0]

        geometry_position = self._sample_points(sample["geometry_position"], self.num_geometry_points)
        surface_position = self._sample_points(sample["surface_position"], self.num_surface_anchor_points * 2)
        volume_position = self._sample_points(sample["volume_position"], self.num_volume_anchor_points * 2)

        surface_pressure = self._sample_points(sample["surface_pressure"], self.num_surface_anchor_points * 2)
        surface_wallshearstress = self._sample_points(
            sample["surface_wallshearstress"], self.num_surface_anchor_points * 2
        )
        volume_totalpcoeff = self._sample_points(sample["volume_totalpcoeff"], self.num_volume_anchor_points * 2)
        volume_velocity = self._sample_points(sample["volume_velocity"], self.num_volume_anchor_points * 2)
        volume_vorticity = self._sample_points(sample["volume_vorticity"], self.num_volume_anchor_points * 2)

        # Normalize positions
        geometry_position = self.preprocessor.normalize_positions(geometry_position)
        surface_position = self.preprocessor.normalize_positions(surface_position)
        volume_position = self.preprocessor.normalize_positions(volume_position)

        # Sample supernodes
        supernode_indices = np.random.choice(
            len(geometry_position),
            size=min(self.num_geometry_supernodes, len(geometry_position)),
            replace=False
        )
        supernode_indices = ops.array(supernode_indices, dtype="int32")

        # Split anchor/query positions
        if self.use_query_positions:
            s_split = self.num_surface_anchor_points
            surface_anchor_pos = surface_position[:s_split]
            surface_query_pos = surface_position[s_split:s_split * 2]
            surface_anchor_pressure = surface_pressure[:s_split]
            surface_query_pressure = surface_pressure[s_split:s_split * 2]
            surface_anchor_wss = surface_wallshearstress[:s_split]
            surface_query_wss = surface_wallshearstress[s_split:s_split * 2]

            v_split = self.num_volume_anchor_points
            volume_anchor_pos = volume_position[:v_split]
            volume_query_pos = volume_position[v_split:v_split * 2]
            volume_anchor_pressure = volume_totalpcoeff[:v_split]
            volume_query_pressure = volume_totalpcoeff[v_split:v_split * 2]
            volume_anchor_velocity = volume_velocity[:v_split]
            volume_query_velocity = volume_velocity[v_split:v_split * 2]
            volume_anchor_vorticity = volume_vorticity[:v_split]
            volume_query_vorticity = volume_vorticity[v_split:v_split * 2]
        else:
            surface_anchor_pos = surface_position[:self.num_surface_anchor_points]
            volume_anchor_pos = volume_position[:self.num_volume_anchor_points]
            surface_anchor_pressure = surface_pressure[:self.num_surface_anchor_points]
            surface_anchor_wss = surface_wallshearstress[:self.num_surface_anchor_points]
            volume_anchor_pressure = volume_totalpcoeff[:self.num_volume_anchor_points]
            volume_anchor_velocity = volume_velocity[:self.num_volume_anchor_points]
            volume_anchor_vorticity = volume_vorticity[:self.num_volume_anchor_points]

        # Normalize quantities
        surface_anchor_pressure = self.preprocessor.normalize_surface_pressure(
            ops.expand_dims(surface_anchor_pressure, -1)
        )
        surface_anchor_wss = self.preprocessor.normalize_surface_wallshearstress(surface_anchor_wss)
        volume_anchor_pressure = self.preprocessor.normalize_volume_pressure(
            ops.expand_dims(volume_anchor_pressure, -1)
        )
        volume_anchor_velocity = self.preprocessor.normalize_volume_velocity(volume_anchor_velocity)
        volume_anchor_vorticity = self.preprocessor.normalize_volume_vorticity(volume_anchor_vorticity)

        inputs = {
            "geometry_position": geometry_position,
            "geometry_supernode_idx": supernode_indices,
            "geometry_batch_idx": None,
            "surface_anchor_position": ops.expand_dims(surface_anchor_pos, 0),
            "volume_anchor_position": ops.expand_dims(volume_anchor_pos, 0),
        }

        targets = {
            "surface_anchor_pressure": surface_anchor_pressure,
            "surface_anchor_wallshearstress": surface_anchor_wss,
            "volume_anchor_totalpcoeff": volume_anchor_pressure,
            "volume_anchor_velocity": volume_anchor_velocity,
            "volume_anchor_vorticity": volume_anchor_vorticity,
        }

        if self.use_query_positions:
            inputs["surface_query_position"] = ops.expand_dims(surface_query_pos, 0)
            inputs["volume_query_position"] = ops.expand_dims(volume_query_pos, 0)

            surface_query_pressure = self.preprocessor.normalize_surface_pressure(
                ops.expand_dims(surface_query_pressure, -1)
            )
            surface_query_wss = self.preprocessor.normalize_surface_wallshearstress(surface_query_wss)
            volume_query_pressure = self.preprocessor.normalize_volume_pressure(
                ops.expand_dims(volume_query_pressure, -1)
            )
            volume_query_velocity = self.preprocessor.normalize_volume_velocity(volume_query_velocity)
            volume_query_vorticity = self.preprocessor.normalize_volume_vorticity(volume_query_vorticity)

            targets.update({
                "surface_query_pressure": surface_query_pressure,
                "surface_query_wallshearstress": surface_query_wss,
                "volume_query_totalpcoeff": volume_query_pressure,
                "volume_query_velocity": volume_query_velocity,
                "volume_query_vorticity": volume_query_vorticity,
            })

        return inputs, targets

    def _sample_points(self, data: np.ndarray, num_points: int) -> np.ndarray:
        if len(data) <= num_points:
            return ops.array(data, dtype="float32")
        indices = np.random.choice(len(data), size=num_points, replace=False)
        return ops.array(data[indices], dtype="float32")


# ---------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------

class MultiTaskLoss(keras.losses.Loss):
    """Multi-task loss for CFD predictions with per-task weighting."""

    def __init__(self, loss_weights: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_weights = loss_weights or {}
        self.mse = keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        total_loss = 0.0
        for key in y_true.keys():
            if key in y_pred:
                task_loss = self.mse(y_true[key], y_pred[key])
                total_loss += task_loss * self.loss_weights.get(key, 1.0)
        return total_loss


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------

class CFDTrainer:
    """Training pipeline for AB-UPT model."""

    def __init__(
            self,
            model: AnchoredBranchedUPT,
            preprocessor: DataPreprocessor,
            loss_weights: Optional[Dict[str, float]] = None,
            learning_rate: float = 1e-4,
            **kwargs
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.loss_weights = loss_weights or {
            "surface_anchor_pressure": 1.0,
            "surface_anchor_wallshearstress": 1.0,
            "volume_anchor_totalpcoeff": 1.0,
            "volume_anchor_velocity": 1.0,
            "volume_anchor_vorticity": 1.0,
        }

        self.optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        self.loss_fn = MultiTaskLoss(loss_weights=self.loss_weights)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, run_eagerly=False)

        self.history = {"train_loss": [], "val_loss": [], "learning_rate": []}

    def train(
            self,
            train_generator: CFDDataGenerator,
            val_generator: CFDDataGenerator,
            epochs: int = 100,
            save_dir: str = "checkpoints",
            save_best_only: bool = True,
            patience: int = 10,
            verbose: int = 1
    ):
        """Train the model with early stopping and checkpointing."""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"Training for {epochs} epochs, train batches: {len(train_generator)}, val batches: {len(val_generator)}")

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_generator, verbose=verbose)
            val_loss = self._validate_epoch(val_generator)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(float(self.optimizer.learning_rate))

            logger.info(
                f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.6f}, "
                f"val_loss: {val_loss:.6f}, lr: {float(self.optimizer.learning_rate):.2e}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best_only:
                    checkpoint_path = os.path.join(save_dir, "best_model.keras")
                    self.model.save(checkpoint_path)
                    logger.info(f"New best model saved: {checkpoint_path}")
            else:
                patience_counter += 1

            if not save_best_only:
                self.model.save(os.path.join(save_dir, f"model_epoch_{epoch + 1}.keras"))

            if patience_counter >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break

            self._save_history(os.path.join(save_dir, "training_history.json"))

        logger.info("Training completed")
        return self.history

    def _train_epoch(self, generator: CFDDataGenerator, verbose: int = 1) -> float:
        epoch_losses = []
        for batch_idx in range(len(generator)):
            inputs, targets = generator[batch_idx]
            with keras.backend.eager_scope():
                loss = self.model.train_on_batch(inputs, targets)
            epoch_losses.append(float(loss))
            if verbose > 1 and batch_idx % 10 == 0:
                logger.info(f"  Batch {batch_idx}/{len(generator)}: loss={loss:.6f}")
        return np.mean(epoch_losses)

    def _validate_epoch(self, generator: CFDDataGenerator) -> float:
        epoch_losses = []
        for batch_idx in range(len(generator)):
            inputs, targets = generator[batch_idx]
            loss = self.model.test_on_batch(inputs, targets)
            epoch_losses.append(float(loss))
        return np.mean(epoch_losses)

    def _save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history (loss and learning rate)."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(self.history["train_loss"], label="Train Loss", alpha=0.8)
        axes[0].plot(self.history["val_loss"], label="Val Loss", alpha=0.8)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history["learning_rate"], alpha=0.8)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training plots saved to {save_path}")
        plt.show()


# ---------------------------------------------------------------------
# Synthetic data for testing
# ---------------------------------------------------------------------

def create_synthetic_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Create synthetic CFD data for testing."""
    samples = []
    for i in range(num_samples):
        num_geometry = np.random.randint(800, 1200)
        num_surface = np.random.randint(600, 1000)
        num_volume = np.random.randint(1000, 1500)

        samples.append({
            "geometry_position": np.random.uniform(-40, 80, (num_geometry, 3)).astype(np.float32),
            "surface_position": np.random.uniform(-40, 80, (num_surface, 3)).astype(np.float32),
            "surface_pressure": np.random.normal(-229, 269, num_surface).astype(np.float32),
            "surface_wallshearstress": np.random.normal(
                [-1.2, 0.0015, -0.072], [2.08, 1.36, 1.11], (num_surface, 3)
            ).astype(np.float32),
            "volume_position": np.random.uniform(-40, 80, (num_volume, 3)).astype(np.float32),
            "volume_totalpcoeff": np.random.normal(0.17, 0.50, num_volume).astype(np.float32),
            "volume_velocity": np.random.normal(
                [16.8, -0.038, 0.41], [16.4, 8.64, 6.65], (num_volume, 3)
            ).astype(np.float32),
            "volume_vorticity": np.random.exponential(2.0, (num_volume, 3)).astype(np.float32),
        })
    return samples


def main():
    """Main training function with synthetic CFD data."""
    setup_gpu()

    logger.info("Starting AB-UPT Training Pipeline")

    logger.info("Generating synthetic CFD data...")
    train_data = create_synthetic_data(num_samples=80)
    val_data = create_synthetic_data(num_samples=20)

    stats = NormalizationStats()
    preprocessor = DataPreprocessor(stats)

    generator_kwargs = dict(
        batch_size=1,
        num_geometry_points=800,
        num_surface_anchor_points=300,
        num_volume_anchor_points=400,
        num_geometry_supernodes=150,
        use_query_positions=True,
    )
    train_generator = CFDDataGenerator(train_data, preprocessor, shuffle=True, **generator_kwargs)
    val_generator = CFDDataGenerator(val_data, preprocessor, shuffle=False, **generator_kwargs)

    logger.info("Building AB-UPT model...")
    model = create_abupt_model(
        dim=192, num_heads=3, geometry_depth=1,
        blocks="pscscs", num_surface_blocks=4, num_volume_blocks=4,
        radius=2.5, dropout=0.1
    )

    loss_weights = {
        "surface_anchor_pressure": 1.0, "surface_anchor_wallshearstress": 0.5,
        "surface_query_pressure": 1.0, "surface_query_wallshearstress": 0.5,
        "volume_anchor_totalpcoeff": 1.0, "volume_anchor_velocity": 0.8,
        "volume_anchor_vorticity": 0.3, "volume_query_totalpcoeff": 1.0,
        "volume_query_velocity": 0.8, "volume_query_vorticity": 0.3,
    }

    trainer = CFDTrainer(
        model=model, preprocessor=preprocessor,
        loss_weights=loss_weights, learning_rate=1e-4
    )

    # Validate with single batch
    logger.info("Testing single batch...")
    try:
        inputs, targets = train_generator[0]
        logger.info("Input shapes: " + ", ".join(f"{k}: {v.shape}" for k, v in inputs.items() if v is not None))
        outputs = model(inputs)
        logger.info("Output shapes: " + ", ".join(f"{k}: {v.shape}" for k, v in outputs.items()))
        logger.info("Single batch test passed")
    except Exception as e:
        logger.error(f"Single batch test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Train
    logger.info("Starting training...")
    try:
        history = trainer.train(
            train_generator=train_generator,
            val_generator=val_generator,
            epochs=10,
            save_dir="abupt_checkpoints",
            save_best_only=True,
            patience=5,
            verbose=1
        )
        trainer.plot_training_history("training_history.png")
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
