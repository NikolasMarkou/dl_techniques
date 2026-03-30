"""Common callback and learning rate schedule utilities for training scripts."""

import os
import keras
from datetime import datetime
from typing import Tuple, List, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback


# ---------------------------------------------------------------------

def create_callbacks(
        model_name: str,
        results_dir_prefix: str = "model",
        monitor: str = 'val_accuracy',
        patience: int = 15,
        use_lr_schedule: bool = True,
        analyzer_epoch_frequency: int = 1,
) -> Tuple[List, str]:
    """
    Create standard training callbacks.

    Parameters
    ----------
    model_name : str
        Name identifier for the model (used in directory naming).
    results_dir_prefix : str
        Prefix for the results directory (e.g., 'convnext_v1', 'convnext_v2').
    monitor : str
        Metric to monitor for checkpointing/early stopping.
    patience : int
        Early stopping patience.
    use_lr_schedule : bool
        If True, skip ReduceLROnPlateau (assumes external LR schedule).
    analyzer_epoch_frequency : int
        How often to run the EpochAnalyzerCallback (every N epochs).

    Returns
    -------
    Tuple[List, str]
        List of callbacks and the results directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"{results_dir_prefix}_{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    monitor_mode = 'max' if 'accuracy' in monitor else 'min'

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode=monitor_mode,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode=monitor_mode,
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv')
        ),
        EpochAnalyzerCallback(
            output_dir=os.path.join(results_dir, "epoch_analysis"),
            model_name=model_name,
            epoch_frequency=analyzer_epoch_frequency,
        ),
    ]

    if not use_lr_schedule:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )

    logger.info(f"Results will be saved to: {results_dir}")
    return callbacks, results_dir


# ---------------------------------------------------------------------

def create_learning_rate_schedule(
        initial_lr: float,
        schedule_type: str = 'cosine',
        total_epochs: int = 100,
        warmup_epochs: int = 5,
        steps_per_epoch: Optional[int] = None,
) -> keras.optimizers.schedules.LearningRateSchedule:
    """
    Create learning rate schedule.

    Parameters
    ----------
    initial_lr : float
        Initial learning rate.
    schedule_type : str
        Type of schedule ('cosine', 'exponential', 'constant').
    total_epochs : int
        Total number of training epochs.
    warmup_epochs : int
        Number of warmup epochs (reserved for future use).
    steps_per_epoch : Optional[int]
        Steps per epoch (for step-based schedules like ImageNet).

    Returns
    -------
    Learning rate schedule or float for constant.
    """
    if schedule_type == 'cosine':
        decay_steps = total_epochs if steps_per_epoch is None else total_epochs * steps_per_epoch
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=0.01
        )
    elif schedule_type == 'exponential':
        decay_steps = (total_epochs // 4) if steps_per_epoch is None else (total_epochs // 4) * steps_per_epoch
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.9
        )
    else:  # constant
        return initial_lr
