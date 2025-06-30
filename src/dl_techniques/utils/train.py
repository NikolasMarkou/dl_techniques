import os
import keras
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .logger import logger


# ---------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Configuration class for model training parameters.

    Args:
        epochs: Number of training epochs.
        batch_size: Size of training batches.
        output_dir: Directory for saving model outputs.
        model_name: Name of the model for saving.
        early_stopping_patience: Number of epochs with no improvement after which training will stop.
        reduce_lr_patience: Number of epochs with no improvement after which learning rate will be reduced.
        reduce_lr_factor: Factor by which learning rate will be reduced.
        min_learning_rate: Minimum learning rate.
        monitor_metric: Metric to monitor for callbacks ('val_loss' or 'val_accuracy').
    """
    epochs: int = 10
    batch_size: int = 128
    output_dir: Optional[Path] = None
    model_name: str = "model"
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    monitor_metric: str = 'val_loss'


# ---------------------------------------------------------------------

def train_model(
        model: keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        config: TrainingConfig
) -> keras.callbacks.History:
    """Train a Keras model with configurable parameters and callbacks.

    This function trains a model using early stopping, learning rate reduction,
    and model checkpointing. It saves the best model based on validation metrics.

    Args:
        model: Pre-compiled Keras model to train.
        x_train: Training input data.
        y_train: Training target data.
        x_test: Validation input data.
        y_test: Validation target data.
        config: TrainingConfig instance containing training parameters.

    Returns:
        keras.callbacks.History: Training history containing metrics.

    Raises:
        ValueError: If input shapes are incompatible with model architecture.
    """
    # Validate input shapes
    if x_train.shape[1:] != model.input_shape[1:]:
        raise ValueError(
            f"Training data shape {x_train.shape[1:]} does not match "
            f"model input shape {model.input_shape[1:]}"
        )

    if config.output_dir is not None:
        os.makedirs(str(config.output_dir), exist_ok=True)

    # Define callbacks
    callbacks = [

        keras.callbacks.EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=config.monitor_metric,
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
            verbose=1
        ),
    ]

    if config.output_dir is not None:
        callbacks += [
            keras.callbacks.ModelCheckpoint(
                filepath=config.output_dir / f'{config.model_name}.keras',
                save_best_only=True,
                monitor=config.monitor_metric,
                mode='min' if 'loss' in config.monitor_metric else 'max',
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                filename=config.output_dir / f'{config.model_name}_training_log.csv',
                separator=',',
                append=False
            )
        ]

    try:
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=config.batch_size,
            epochs=config.epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        # Save training configuration
        if config.output_dir is not None:
            with open(config.output_dir / f'{config.model_name}_config.txt', 'w') as f:
                f.write(str(config))

        return history

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

# ---------------------------------------------------------------------
