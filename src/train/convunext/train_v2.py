"""
ConvUNext MAE Segmentation Training Pipeline (ImageNet + COCO)
==============================================================

Refactored to use COCODatasetBuilder for robust COCO dataset handling.

Two-Stage training workflow:
1. **Self-Supervised Pretraining**: Using Masked Autoencoders (MAE) on ImageNet.
2. **Supervised Fine-tuning**: Fine-tuning the pretrained U-Net on COCO for
   semantic segmentation using the robust COCODatasetBuilder.

Configuration:
--------------
Data locations can be configured via command line arguments or a .env file.
Define `TFDS_DATA_DIR` in your .env file to set a custom TFDS download location.
"""

import os
import math
import argparse
import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Framework Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.convunext import ConvUNextModel
from dl_techniques.models.masked_autoencoder import (
    MaskedAutoencoder,
    visualize_reconstruction
)
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)
from dl_techniques.datasets.vision.coco import COCODatasetBuilder


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def setup_environment():
    """Configure GPU settings, precision, and load environment variables."""
    load_dotenv()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")


# ---------------------------------------------------------------------
# Data Pipelines
# ---------------------------------------------------------------------

def get_imagenet_dataset(
    split: str,
    image_size: int,
    batch_size: int,
    data_dir: Optional[str] = None
) -> tf.data.Dataset:
    """
    Load ImageNet dataset for MAE pretraining.

    Returns (image, image) tuples because MAE is an autoencoder.

    :param split: Dataset split ('train', 'validation').
    :param image_size: Target resolution (H, W).
    :param batch_size: Batch size.
    :param data_dir: Optional directory for TFDS data.
    :return: A tf.data.Dataset yielding (image, image).
    """
    try:
        dataset_name = "imagenet2012"
        ds, info = tfds.load(
            dataset_name,
            split=split,
            with_info=True,
            data_dir=data_dir,
            as_supervised=True
        )
        logger.info(f"Loaded {dataset_name} ({split})")
    except Exception as e:
        # Fallback to Imagenette
        logger.warning(
            f"ImageNet load failed: {e}. Falling back to 'imagenette' "
            "for demonstration."
        )
        dataset_name = "imagenette/320px-v2"
        ds, info = tfds.load(
            dataset_name,
            split=split,
            with_info=True,
            data_dir=data_dir,
            as_supervised=True
        )

    def preprocess_mae(image, label):
        """Prepare image for MAE: Resize -> Scale -> Return (x, x)."""
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32) / 255.0
        # MAE target is the image itself (reconstruction)
        return image, image

    ds = ds.map(preprocess_mae, num_parallel_calls=tf.data.AUTOTUNE)

    if split == 'train':
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_coco_segmentation_dataset(
    image_size: int,
    batch_size: int,
    num_classes: int,
    cache_dir: Optional[str] = None,
    shuffle_buffer_size: int = 100,
    limit_train_samples: Optional[int] = None,
    augment_data: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]:
    """
    Load COCO dataset for semantic segmentation using COCODatasetBuilder.

    Adapts the multi-task output format to pure segmentation format
    for ConvUNext training.

    :param image_size: Target resolution (H, W).
    :param batch_size: Batch size.
    :param num_classes: Number of segmentation classes.
    :param cache_dir: Optional directory for caching.
    :param shuffle_buffer_size: Shuffle buffer size for memory management.
    :param limit_train_samples: Optional limit on training samples.
    :param augment_data: Whether to apply data augmentation.
    :return: Tuple of (train_ds, val_ds, dataset_info).
    """
    logger.info("Creating COCO segmentation dataset via COCODatasetBuilder...")

    builder = COCODatasetBuilder(
        img_size=image_size,
        batch_size=batch_size,
        max_boxes_per_image=100,
        cache_dir=cache_dir,
        use_detection=False,
        use_segmentation=True,
        segmentation_classes=num_classes,
        augment_data=augment_data,
        shuffle_buffer_size=shuffle_buffer_size,
        limit_train_samples=limit_train_samples
    )

    train_ds_raw, val_ds_raw = builder.create_datasets()
    dataset_info = builder.get_dataset_info()

    def adapt_for_segmentation(image, targets):
        """
        Adapt COCODatasetBuilder output to segmentation format.

        COCODatasetBuilder produces:
          - image: (B, H, W, 3)
          - targets: {'segmentation': (B, H, W, num_classes)} one-hot

        ConvUNext with sparse_categorical_crossentropy expects:
          - image: (B, H, W, 3)
          - mask: (B, H, W, 1) or (B, H, W) with integer class indices

        We convert one-hot to integer class indices.
        """
        seg_mask = targets['segmentation']

        if num_classes == 1:
            # Binary segmentation - threshold at 0.5
            mask = tf.cast(seg_mask > 0.5, tf.int32)
        else:
            # Multi-class - argmax to get class indices
            mask = tf.argmax(seg_mask, axis=-1, output_type=tf.int32)
            mask = tf.expand_dims(mask, axis=-1)

        return image, mask

    train_ds = train_ds_raw.map(
        adapt_for_segmentation,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds_raw.map(
        adapt_for_segmentation,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    logger.info(f"COCO segmentation dataset created: {num_classes} classes")
    logger.info(f"Using dummy data: {dataset_info.get('using_dummy_data', False)}")

    return train_ds, val_ds, dataset_info


# ---------------------------------------------------------------------
# Encoder Wrapper
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextEncoderWrapper(keras.Model):
    """
    Wraps the Encoder path of a ConvUNextModel for MAE compatibility.

    :param convunext_model: The instance of ConvUNextModel to wrap.
    """

    def __init__(self, convunext_model: ConvUNextModel, **kwargs):
        super().__init__(**kwargs)

        self.stem = convunext_model.stem
        self.encoder_stages = convunext_model.encoder_stages
        self.encoder_downsamples = convunext_model.encoder_downsamples
        self.bottleneck_entry = convunext_model.bottleneck_entry
        self.bottleneck_blocks = convunext_model.bottleneck_blocks

        self.input_shape_config = convunext_model.input_shape_config
        self.depth = convunext_model.depth
        self.filter_sizes = convunext_model.filter_sizes

        self.input_spec = keras.layers.InputSpec(
            shape=(None,) + self.input_shape_config
        )

    def call(self, inputs, training=None):
        x = inputs
        x = self.stem(x, training=training)

        for level in range(self.depth):
            for block in self.encoder_stages[level]:
                x = block(x, training=training)
            if level < self.depth - 1:
                x = self.encoder_downsamples[level](x, training=training)

        x = self.bottleneck_entry(x, training=training)
        for block in self.bottleneck_blocks:
            x = block(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        h, w, c = self.input_shape_config
        downsample_factor = 2 ** self.depth
        new_h = h // downsample_factor if h else None
        new_w = w // downsample_factor if w else None
        final_channels = self.filter_sizes[self.depth]
        return (batch_size, new_h, new_w, final_channels)

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------

class ImageReconstructionCallback(keras.callbacks.Callback):
    """
    Keras callback to generate MAE image reconstructions at the end
    of every epoch.

    :param val_images: Numpy array of images (B, H, W, 3) to reconstruct.
    :param save_dir: Directory path to save the visualization images.
    :param num_samples: Number of images from the batch to visualize.
    """

    def __init__(
        self,
        val_images: np.ndarray,
        save_dir: str,
        num_samples: int = 4
    ):
        super().__init__()
        self.val_images = val_images
        self.save_dir = save_dir
        self.num_samples = min(num_samples, len(val_images))
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        viz_data = self.val_images[:self.num_samples]
        grid = visualize_reconstruction(
            self.model,
            viz_data,
            num_samples=self.num_samples
        )

        plt.figure(figsize=(12, 8))
        plt.imshow(grid)
        plt.axis('off')
        plt.title(f"MAE Reconstruction - Epoch {epoch + 1}")

        filename = os.path.join(self.save_dir, f"epoch_{epoch + 1:03d}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved reconstruction visualization to {filename}")


class SegmentationVisualizationCallback(keras.callbacks.Callback):
    """
    Callback for visualizing segmentation predictions during training.

    :param validation_dataset: Validation dataset to sample from.
    :param results_dir: Directory to save visualizations.
    :param num_classes: Number of segmentation classes.
    :param num_samples: Number of samples to visualize.
    :param visualization_freq: Frequency of visualization (every N epochs).
    """

    def __init__(
        self,
        validation_dataset: tf.data.Dataset,
        results_dir: str,
        num_classes: int,
        num_samples: int = 4,
        visualization_freq: int = 5
    ):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.results_dir = results_dir
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.visualization_freq = visualization_freq

        self.viz_dir = os.path.join(results_dir, 'segmentation_visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

        self._prepare_samples()

    def _prepare_samples(self):
        """Prepare fixed samples for consistent visualization."""
        try:
            sample_batch = next(iter(self.validation_dataset.take(1)))
            if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
                self.sample_images, self.sample_masks = sample_batch
                if tf.shape(self.sample_images)[0] > self.num_samples:
                    self.sample_images = self.sample_images[:self.num_samples]
                    self.sample_masks = self.sample_masks[:self.num_samples]
                logger.info(
                    f"Prepared {tf.shape(self.sample_images)[0]} samples "
                    "for segmentation visualization"
                )
            else:
                logger.error("Unexpected sample batch format")
                self.sample_images = None
                self.sample_masks = None
        except Exception as e:
            logger.error(f"Failed to prepare visualization samples: {e}")
            self.sample_images = None
            self.sample_masks = None

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.visualization_freq != 0:
            return

        if self.sample_images is None:
            return

        try:
            predictions = self.model(self.sample_images, training=False)

            epoch_dir = os.path.join(self.viz_dir, f'epoch_{epoch + 1:03d}')
            os.makedirs(epoch_dir, exist_ok=True)

            self._visualize_segmentation(predictions, epoch_dir, epoch + 1)
            logger.info(f"Generated segmentation visualizations for epoch {epoch + 1}")

        except Exception as e:
            logger.error(
                f"Failed to generate visualizations for epoch {epoch + 1}: {e}"
            )

    def _visualize_segmentation(
        self,
        predictions: tf.Tensor,
        save_dir: str,
        epoch: int
    ):
        """Generate segmentation visualization."""
        images_np = self.sample_images.numpy()
        masks_np = self.sample_masks.numpy()
        pred_np = predictions.numpy()

        # Get predicted class indices
        if pred_np.shape[-1] > 1:
            pred_classes = np.argmax(pred_np, axis=-1)
        else:
            pred_classes = (pred_np[..., 0] > 0.5).astype(np.int32)

        # Ground truth class indices
        if masks_np.shape[-1] == 1:
            gt_classes = masks_np[..., 0]
        else:
            gt_classes = masks_np

        n_samples = min(4, len(images_np))
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

        if n_samples == 1:
            axes = axes[np.newaxis, :]

        for i in range(n_samples):
            # Original image
            axes[i, 0].imshow(images_np[i])
            axes[i, 0].set_title(f'Image {i + 1}')
            axes[i, 0].axis('off')

            # Ground truth mask
            vmax = self.num_classes - 1 if self.num_classes > 1 else 1
            im_gt = axes[i, 1].imshow(
                gt_classes[i],
                cmap='tab20' if self.num_classes > 2 else 'gray',
                vmin=0,
                vmax=vmax
            )
            axes[i, 1].set_title(f'Ground Truth {i + 1}')
            axes[i, 1].axis('off')

            # Predicted mask
            im_pred = axes[i, 2].imshow(
                pred_classes[i],
                cmap='tab20' if self.num_classes > 2 else 'gray',
                vmin=0,
                vmax=vmax
            )
            axes[i, 2].set_title(f'Prediction {i + 1}')
            axes[i, 2].axis('off')

            # Calculate IoU per sample
            intersection = np.sum(
                (pred_classes[i] == gt_classes[i]) &
                (gt_classes[i] > 0)
            )
            union = np.sum((pred_classes[i] > 0) | (gt_classes[i] > 0))
            iou = intersection / (union + 1e-8)

            axes[i, 2].text(
                5, pred_classes[i].shape[0] - 10,
                f'IoU: {iou:.3f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.8),
                fontsize=10,
                fontweight='bold'
            )

        plt.suptitle(
            f'Segmentation Results - Epoch {epoch}',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, 'segmentation_results.png'),
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()


class ProgressLoggingCallback(keras.callbacks.Callback):
    """Custom callback for enhanced progress logging."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        loss_str = f"Loss: {logs.get('loss', 0):.4f}"
        if 'val_loss' in logs:
            loss_str += f", Val Loss: {logs.get('val_loss', 0):.4f}"
        if 'accuracy' in logs:
            loss_str += f", Acc: {logs.get('accuracy', 0):.4f}"
        if 'val_accuracy' in logs:
            loss_str += f", Val Acc: {logs.get('val_accuracy', 0):.4f}"

        logger.info(
            f"Epoch {epoch + 1}/{self.params.get('epochs', '?')} - {loss_str}"
        )


# ---------------------------------------------------------------------
# Training Workflow
# ---------------------------------------------------------------------

def run_mae_pretraining(
    convunext_model: ConvUNextModel,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    args: argparse.Namespace,
    results_dir: str
) -> MaskedAutoencoder:
    """
    Stage 1: Pretrain on ImageNet using Masked Autoencoders.

    :param convunext_model: The base UNext model instance.
    :param train_ds: Training dataset.
    :param val_ds: Validation dataset.
    :param args: Parsed command line arguments.
    :param results_dir: Directory to save artifacts.
    :return: Trained MaskedAutoencoder wrapper.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: MAE SELF-SUPERVISED PRETRAINING (ImageNet)")
    logger.info("=" * 60)

    encoder_wrapper = ConvUNextEncoderWrapper(convunext_model)

    downsample_factor = 2 ** convunext_model.depth
    decoder_depth = int(math.log2(downsample_factor))

    mae = MaskedAutoencoder(
        encoder=encoder_wrapper,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        decoder_depth=decoder_depth,
        decoder_dims=None,
        input_shape=(args.image_size, args.image_size, 3)
    )

    cardinality = int(train_ds.cardinality())
    steps_per_epoch = cardinality if cardinality > 0 else 1000

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.mae_lr,
        "decay_steps": args.mae_epochs * steps_per_epoch,
        "warmup_steps": 1000
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "weight_decay": 0.05,
        "gradient_clipping_by_norm": 1.0
    }, lr_schedule)

    mae.compile(optimizer=optimizer)

    viz_dir = os.path.join(results_dir, "mae_viz")
    logger.info("Preparing validation batch for reconstruction visualization...")
    val_iter = iter(val_ds)
    val_batch = next(val_iter)
    val_images = val_batch[0].numpy()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "mae_weights.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "mae_history.csv")),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, "logs", "mae")
        ),
        ImageReconstructionCallback(
            val_images=val_images,
            save_dir=viz_dir,
            num_samples=8
        ),
        ProgressLoggingCallback()
    ]

    mae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.mae_epochs,
        callbacks=callbacks
    )

    return mae


def run_segmentation_finetuning(
    convunext_model: ConvUNextModel,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int,
    args: argparse.Namespace,
    results_dir: str,
    dataset_info: Dict[str, Any]
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Stage 2: Fine-tune on COCO for semantic segmentation.

    :param convunext_model: The pretrained UNext model.
    :param train_ds: Training dataset.
    :param val_ds: Validation dataset.
    :param num_classes: Number of segmentation classes.
    :param args: Parsed command line arguments.
    :param results_dir: Directory to save artifacts.
    :param dataset_info: Dataset configuration info.
    :return: Tuple of (Finetuned Model, History object).
    """
    logger.info("=" * 60)
    logger.info(f"STAGE 2: SEGMENTATION FINE-TUNING (Classes: {num_classes})")
    logger.info("=" * 60)

    # Determine monitoring strategy based on data source
    using_dummy_data = dataset_info.get('using_dummy_data', False)
    monitor = 'loss' if using_dummy_data else 'val_loss'
    logger.info(f"Using {'training loss' if using_dummy_data else 'validation loss'} monitoring")

    # ---------------------------------------------
    # Phase 2a: Freeze Encoder
    # ---------------------------------------------
    logger.info("Phase 2a: Freezing encoder, training decoder only...")
    convunext_model.stem.trainable = False
    for stage in convunext_model.encoder_stages:
        for block in stage:
            block.trainable = False
    for ds in convunext_model.encoder_downsamples:
        ds.trainable = False
    convunext_model.bottleneck_entry.trainable = False
    for block in convunext_model.bottleneck_blocks:
        block.trainable = False

    convunext_model.compile(
        optimizer=keras.optimizers.Adam(args.finetune_lr_stage1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Calculate steps per epoch
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else 500
    validation_steps = args.validation_steps if args.validation_steps else 50

    phase1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=3,
            restore_best_weights=True
        ),
        ProgressLoggingCallback()
    ]

    convunext_model.fit(
        train_ds,
        validation_data=val_ds if not using_dummy_data else None,
        epochs=args.finetune_epochs_stage1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps if not using_dummy_data else None,
        callbacks=phase1_callbacks
    )

    # ---------------------------------------------
    # Phase 2b: Unfreeze All
    # ---------------------------------------------
    logger.info("Phase 2b: Unfreezing all layers for end-to-end fine-tuning...")
    convunext_model.trainable = True
    convunext_model.stem.trainable = True
    for stage in convunext_model.encoder_stages:
        for block in stage:
            block.trainable = True

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.finetune_lr_stage2,
        "decay_steps": args.finetune_epochs_stage2 * steps_per_epoch,
        "warmup_steps": 500
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "weight_decay": 0.01,
        "gradient_clipping_by_norm": 1.0
    }, lr_schedule)

    convunext_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "best_segmentation.keras"),
            save_best_only=True,
            monitor=monitor
        ),
        keras.callbacks.CSVLogger(
            os.path.join(results_dir, "segmentation_history.csv")
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=args.patience,
            restore_best_weights=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, "logs", "finetune")
        ),
        ProgressLoggingCallback()
    ]

    # Add visualization callback if validation data available
    if not using_dummy_data:
        callbacks.append(
            SegmentationVisualizationCallback(
                validation_dataset=val_ds,
                results_dir=results_dir,
                num_classes=num_classes,
                num_samples=4,
                visualization_freq=args.visualization_freq
            )
        )

    history = convunext_model.fit(
        train_ds,
        validation_data=val_ds if not using_dummy_data else None,
        epochs=args.finetune_epochs_stage2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps if not using_dummy_data else None,
        callbacks=callbacks
    )

    return convunext_model, history


def transfer_encoder_weights(
    source_model: ConvUNextModel,
    target_model: ConvUNextModel
) -> None:
    """
    Transfer encoder and bottleneck weights from source to target model.

    :param source_model: Model with pretrained weights.
    :param target_model: Model to transfer weights to.
    """
    logger.info("Transferring encoder weights...")

    target_model.stem.set_weights(source_model.stem.get_weights())

    for i in range(len(source_model.encoder_stages)):
        for j in range(len(source_model.encoder_stages[i])):
            target_model.encoder_stages[i][j].set_weights(
                source_model.encoder_stages[i][j].get_weights()
            )
        if i < len(source_model.encoder_downsamples):
            target_model.encoder_downsamples[i].set_weights(
                source_model.encoder_downsamples[i].get_weights()
            )

    target_model.bottleneck_entry.set_weights(
        source_model.bottleneck_entry.get_weights()
    )
    for i in range(len(source_model.bottleneck_blocks)):
        target_model.bottleneck_blocks[i].set_weights(
            source_model.bottleneck_blocks[i].get_weights()
        )

    logger.info("Weight transfer complete.")


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    setup_environment()

    parser = argparse.ArgumentParser(
        description="ConvUNext (ImageNet MAE -> COCO Seg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data params
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="TFDS data directory. Overrides TFDS_DATA_DIR env var."
    )
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory for caching processed data")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)

    # Model params
    parser.add_argument(
        "--variant", type=str, default="tiny",
        choices=["tiny", "small", "base"]
    )
    parser.add_argument("--num-classes", type=int, default=80,
                        help="Number of segmentation classes (80 for COCO)")

    # MAE params
    parser.add_argument("--mae-epochs", type=int, default=5)
    parser.add_argument("--mae-lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # Finetune params
    parser.add_argument("--finetune-epochs-stage1", type=int, default=5)
    parser.add_argument("--finetune-epochs-stage2", type=int, default=50)
    parser.add_argument("--finetune-lr-stage1", type=float, default=1e-3)
    parser.add_argument("--finetune-lr-stage2", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    # Memory management
    parser.add_argument("--shuffle-buffer", type=int, default=100,
                        help="Shuffle buffer size (reduce if out of memory)")
    parser.add_argument("--limit-train-samples", type=int, default=None,
                        help="Limit training samples for memory efficiency")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Steps per epoch (auto-calculated if not set)")
    parser.add_argument("--validation-steps", type=int, default=None,
                        help="Validation steps (auto-calculated if not set)")

    # Control flags
    parser.add_argument(
        "--skip-mae", action="store_true", help="Skip MAE pretraining"
    )
    parser.add_argument("--visualization-freq", type=int, default=5,
                        help="Visualization frequency (every N epochs)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Resolve Data Directory
    if args.data_dir is None:
        env_data_dir = os.getenv("TFDS_DATA_DIR", "~/tensorflow_datasets/")
        if env_data_dir:
            args.data_dir = env_data_dir
            logger.info(f"Resolved TFDS data directory from .env: {args.data_dir}")
        else:
            logger.info("No custom data directory provided. Using TFDS default.")
    else:
        logger.info(f"Using CLI provided data directory: {args.data_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        "results", f"convunext_{args.variant}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ConvUNext MAE + COCO Segmentation Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Results directory: {results_dir}")

    # 1. Instantiate Main Model (placeholder output channels for MAE)
    logger.info(f"Instantiating ConvUNext ({args.variant})...")
    convunext_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=1,  # Temporary for MAE
        use_bias=True
    )

    dummy_in = keras.ops.zeros((1, args.image_size, args.image_size, 3))
    convunext_model(dummy_in)

    # 2. Stage 1: MAE Pretraining (ImageNet)
    if not args.skip_mae:
        train_ds_mae = get_imagenet_dataset(
            'train', args.image_size, args.batch_size, args.data_dir
        )
        val_ds_mae = get_imagenet_dataset(
            'validation', args.image_size, args.batch_size, args.data_dir
        )

        run_mae_pretraining(
            convunext_model, train_ds_mae, val_ds_mae, args, results_dir
        )
    else:
        logger.info("Skipping MAE pretraining...")

    # 3. Stage 2: Segmentation Fine-tuning (COCO)
    logger.info("Loading COCO segmentation dataset...")
    train_ds_seg, val_ds_seg, dataset_info = get_coco_segmentation_dataset(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        cache_dir=args.cache_dir,
        shuffle_buffer_size=args.shuffle_buffer,
        limit_train_samples=args.limit_train_samples,
        augment_data=True
    )

    logger.info(f"Dataset info: {dataset_info}")

    # Create new model with correct number of output classes
    logger.info(
        f"Creating segmentation model with {args.num_classes} classes..."
    )
    final_seg_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=args.num_classes,
        final_activation="softmax",
        use_bias=True
    )
    final_seg_model(dummy_in)

    # Transfer pretrained weights
    if not args.skip_mae:
        transfer_encoder_weights(convunext_model, final_seg_model)

    # Run Fine-tuning
    run_segmentation_finetuning(
        final_seg_model,
        train_ds_seg,
        val_ds_seg,
        args.num_classes,
        args,
        results_dir,
        dataset_info
    )

    # Save final model
    final_model_path = os.path.join(results_dir, "final_model.keras")
    final_seg_model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

    logger.info("=" * 60)
    logger.info(f"Pipeline complete. Results saved to {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)