"""
COCO Pre-training Script for YOLOv12 Feature Extractor

This script implements Phase 1 of the two-phase training approach:
pre-training the YOLOv12 multi-task model on COCO dataset to learn
powerful, general-purpose visual features.

The script will:
1. Load and preprocess the COCO 2017 dataset
2. Create a YOLOv12 multi-task model (detection + segmentation)
3. Train the model on COCO for a substantial number of epochs
4. Save the trained feature extractor weights for later fine-tuning

Usage:
    python coco_pretrain.py --scale s --epochs 50 --batch-size 16 \
                           --img-size 640 --cache-dir /path/to/cache

Requirements:
    - tensorflow-datasets: pip install tensorflow-datasets
    - Sufficient disk space for COCO dataset (~37GB)
    - GPU with adequate memory for chosen batch size and image size

File: coco_pretrain.py
"""

import os
import sys
import argparse
import json
import keras
import tensorflow as tf
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dl_techniques.utils.logger import logger
from dl_techniques.utils.datasets.coco import COCODatasetBuilder
from dl_techniques.models.yolo12_multitask import create_yolov12_multitask
from dl_techniques.losses.yolo12_multitask_loss import create_yolov12_multitask_loss
from dl_techniques.utils.vision_task_types import TaskType


def setup_gpu():
    """Configure GPU settings for optimal training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")


def create_coco_model_and_loss(
        scale: str,
        img_size: int,
        num_classes: int = 80,
        use_uncertainty_weighting: bool = False
) -> Tuple[keras.Model, keras.losses.Loss]:
    """
    Create YOLOv12 multi-task model and loss for COCO pre-training.

    Args:
        scale: Model scale ('n', 's', 'm', 'l', 'x').
        img_size: Input image size.
        num_classes: Number of COCO classes (80).
        use_uncertainty_weighting: Whether to use uncertainty-based loss weighting.

    Returns:
        Tuple of (model, loss_function).
    """
    logger.info(f"Creating YOLOv12-{scale} model for COCO pre-training...")

    # Create model for detection and segmentation tasks
    model = create_yolov12_multitask(
        num_classes=num_classes,
        input_shape=(img_size, img_size, 3),
        scale=scale,
        tasks=[TaskType.DETECTION, TaskType.SEGMENTATION]
    )

    # Create loss function
    loss_fn = create_yolov12_multitask_loss(
        tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
        num_classes=num_classes,
        input_shape=(img_size, img_size),
        use_uncertainty_weighting=use_uncertainty_weighting,
        # COCO-specific loss weights (can be tuned)
        detection_weight=1.0,
        segmentation_weight=0.5  # Lower weight for segmentation
    )

    return model, loss_fn


def create_coco_callbacks(
        results_dir: str,
        monitor: str = 'val_loss',
        patience: int = 10,
        save_best_only: bool = True
) -> list:
    """
    Create callbacks for COCO pre-training.

    Args:
        results_dir: Directory to save results.
        monitor: Metric to monitor for early stopping and checkpointing.
        patience: Early stopping patience.
        save_best_only: Whether to save only the best model.

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='min',
            verbose=1,
            min_delta=1e-4
        ),

        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_coco_model.keras'),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),

        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(patience // 2, 3),
            min_lr=1e-7,
            verbose=1,
            cooldown=2
        ),

        # CSV logging
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'coco_training_log.csv'),
            append=False
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0
        ),

        # Progress logging callback
        ProgressLoggingCallback()
    ]

    return callbacks


class ProgressLoggingCallback(keras.callbacks.Callback):
    """Custom callback for enhanced progress logging during COCO pre-training."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Enhanced logging every epoch
        loss_str = f"Loss: {logs.get('loss', 0):.4f}"

        if 'val_loss' in logs:
            loss_str += f", Val Loss: {logs.get('val_loss', 0):.4f}"

        # Add task-specific losses if available
        task_losses = []
        for task in ['detection', 'segmentation']:
            if f'{task}_loss' in logs:
                task_losses.append(f"{task}: {logs[f'{task}_loss']:.4f}")

        if task_losses:
            loss_str += f" | {', '.join(task_losses)}"

        # Add learning rate if available
        if 'lr' in logs:
            loss_str += f" | LR: {logs['lr']:.2e}"

        logger.info(f"Epoch {epoch + 1}/{self.params.get('epochs', '?')} - {loss_str}")


def save_feature_extractor_weights(
        model: keras.Model,
        save_path: str,
        scale: str,
        img_size: int
) -> None:
    """
    Extract and save the feature extractor weights for fine-tuning.

    Args:
        model: Trained YOLOv12 model.
        save_path: Path to save the weights.
        scale: Model scale.
        img_size: Input image size.
    """
    try:
        # Get the shared feature extractor layer
        feature_extractor = model.get_layer('shared_feature_extractor')

        # Create the weights save path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the weights
        feature_extractor.save_weights(save_path)

        logger.info(f"✅ Feature extractor weights saved to: {save_path}")

        # Also save metadata about the pre-trained model
        metadata = {
            'model_scale': scale,
            'input_size': img_size,
            'num_classes': 80,  # COCO classes
            'tasks': ['detection', 'segmentation'],
            'pretrained_on': 'COCO 2017',
            'save_timestamp': datetime.now().isoformat(),
            'weights_path': save_path
        }

        metadata_path = save_path.replace('.weights.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Metadata saved to: {metadata_path}")

    except Exception as e:
        logger.error(f"❌ Failed to save feature extractor weights: {e}")
        raise


def calculate_steps_per_epoch(
        dataset_size: int,
        batch_size: int,
        min_steps: int = 100
) -> int:
    """
    Calculate appropriate steps per epoch.

    Args:
        dataset_size: Total number of examples in dataset.
        batch_size: Batch size.
        min_steps: Minimum steps per epoch.

    Returns:
        Steps per epoch.
    """
    steps = max(dataset_size // batch_size, min_steps)
    logger.info(f"Calculated steps per epoch: {steps}")
    return steps


def main():
    """Main COCO pre-training function."""
    parser = argparse.ArgumentParser(
        description='Pre-train YOLOv12 Feature Extractor on COCO Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--scale', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv12 model scale')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')

    # Dataset arguments
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache COCO dataset')
    parser.add_argument('--max-boxes', type=int, default=100,
                        help='Maximum boxes per image')

    # Loss arguments
    parser.add_argument('--uncertainty-weighting', action='store_true',
                        help='Use uncertainty-based adaptive task weighting')

    # Control arguments
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save-dir', type=str, default='coco_pretrain_results',
                        help='Directory to save results')
    parser.add_argument('--weights-name', type=str, default=None,
                        help='Custom name for saved weights file')
    parser.add_argument('--run-eagerly', action='store_true',
                        help='Run model in eager mode')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Setup
    setup_gpu()

    # Set random seeds
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.save_dir}/yolov12_{args.scale}_coco_pretrain_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    logger.info("🚀 Starting COCO Pre-training for YOLOv12 Feature Extractor")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Results directory: {results_dir}")

    # Create COCO dataset
    logger.info("📁 Creating COCO dataset...")
    dataset_builder = COCODatasetBuilder(
        img_size=args.img_size,
        batch_size=args.batch_size,
        max_boxes_per_image=args.max_boxes,
        cache_dir=args.cache_dir,
        use_detection=True,
        use_segmentation=True,
        augment_data=True
    )

    train_ds, val_ds = dataset_builder.create_datasets()
    dataset_info = dataset_builder.get_dataset_info()
    logger.info(f"Dataset info: {dataset_info}")

    # Create model and loss
    logger.info("🏗️ Creating model and loss function...")
    model, loss_fn = create_coco_model_and_loss(
        scale=args.scale,
        img_size=args.img_size,
        num_classes=80,  # COCO classes
        use_uncertainty_weighting=args.uncertainty_weighting
    )

    # Build model
    sample_input = tf.zeros((1, args.img_size, args.img_size, 3))
    _ = model(sample_input, training=False)

    logger.info("Model summary:")
    model.summary(print_fn=logger.info)
    logger.info(f"Total parameters: {model.count_params():,}")

    # Create optimizer
    if args.optimizer.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            clipnorm=1.0
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            clipnorm=1.0
        )

    logger.info(f"Using optimizer: {type(optimizer).__name__}")

    # Compile model
    logger.info("⚙️ Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        run_eagerly=args.run_eagerly
    )

    # Create callbacks
    callbacks = create_coco_callbacks(
        results_dir=results_dir,
        monitor='val_loss',
        patience=args.patience
    )

    # Calculate steps (estimate for COCO train: ~118k images)
    steps_per_epoch = calculate_steps_per_epoch(118000, args.batch_size)

    # Train model
    logger.info("🏋️ Starting COCO pre-training...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=None,  # Use full validation set
            callbacks=callbacks,
            verbose=1
        )

        logger.info("✅ COCO pre-training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

    # Save feature extractor weights
    logger.info("💾 Saving feature extractor weights...")

    if args.weights_name:
        weights_filename = f"{args.weights_name}.weights.h5"
    else:
        weights_filename = f"yolov12_{args.scale}_coco_pretrained_feature_extractor.weights.h5"

    weights_path = os.path.join(results_dir, weights_filename)

    save_feature_extractor_weights(
        model=model,
        save_path=weights_path,
        scale=args.scale,
        img_size=args.img_size
    )

    # Save training configuration
    config = {
        'training_args': vars(args),
        'dataset_info': dataset_info,
        'model_info': {
            'scale': args.scale,
            'input_size': args.img_size,
            'total_parameters': model.count_params(),
            'num_classes': 80
        },
        'training_results': {
            'epochs_completed': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history.get('val_loss', [0])[-1]),
            'best_validation_loss': float(min(history.history.get('val_loss', [float('inf')])))
        },
        'saved_weights': {
            'weights_path': weights_path,
            'relative_path': weights_filename
        }
    }

    config_path = os.path.join(results_dir, 'coco_pretrain_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ Configuration saved to: {config_path}")

    # Create usage instructions
    instructions = f"""
🎉 COCO Pre-training Completed Successfully!

The feature extractor has been trained on COCO and saved to:
{weights_path}

To use these pre-trained weights in your SUT crack detection fine-tuning:

1. In your fine-tuning script, load the weights:
   ```python
   # After creating your SUT model
   feature_extractor = model.get_layer('shared_feature_extractor')
   feature_extractor.load_weights('{weights_path}')
   ```

2. For Phase 2a (freeze feature extractor, train heads only):
   ```python
   feature_extractor.trainable = False
   model.compile(optimizer=optimizer, loss=loss_fn)
   # Train for ~20-30 epochs
   ```

3. For Phase 2b (end-to-end fine-tuning):
   ```python
   feature_extractor.trainable = True
   # Use very low learning rate (e.g., 1e-6)
   model.compile(optimizer=low_lr_optimizer, loss=loss_fn)
   # Train for ~10-15 epochs
   ```

Next steps:
- Run your SUT fine-tuning script with these pre-trained weights
- Use the two-phase approach for best results
- Monitor validation loss carefully during fine-tuning

Happy fine-tuning! 🚀
"""

    instructions_path = os.path.join(results_dir, 'usage_instructions.txt')
    with open(instructions_path, 'w') as f:
        f.write(instructions)

    print(instructions)
    logger.info(f"📋 Usage instructions saved to: {instructions_path}")


if __name__ == '__main__':
    main()