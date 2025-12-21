"""
ConvUNext MAE Deep Supervision Training Pipeline (ImageNet + COCO)
==================================================================

Multi-Label Segmentation Pipeline with Sigmoid Activation:
1. Supports overlapping classes (multiple objects per pixel)
2. Careful background handling (background = all classes inactive)
3. Multi-Scale Visualization with per-class predictions
4. Binary metrics (precision, recall, F1) per scale
5. Weight compatibility: Train with include_top=True, reuse with include_top=False
"""

import os
import argparse
import keras
from keras import ops
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List, Union
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.convunext.model_v2 import ConvUNextModel
from dl_techniques.models.masked_autoencoder import (
    MaskedAutoencoder,
    visualize_reconstruction
)
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)
from dl_techniques.datasets.vision.coco import COCODatasetBuilder

from dl_techniques.losses.multi_labels_loss import create_multilabel_segmentation_loss


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
# Multi-Label Metrics
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiLabelMetrics(keras.metrics.Metric):
    """
    Comprehensive multi-label metrics: precision, recall, F1.

    Computes metrics across all classes and provides mean values.
    Handles background class (channel 0) separately.
    """

    def __init__(self, num_classes: int, threshold: float = 0.5, name='multilabel_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold

        # Per-class metrics
        self.true_positives = self.add_weight(
            name='tp',
            shape=(num_classes,),
            initializer='zeros'
        )
        self.false_positives = self.add_weight(
            name='fp',
            shape=(num_classes,),
            initializer='zeros'
        )
        self.false_negatives = self.add_weight(
            name='fn',
            shape=(num_classes,),
            initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state.

        Args:
            y_true: Ground truth, shape (B, H, W, num_classes), binary
            y_pred: Predictions, shape (B, H, W, num_classes), probabilities
        """
        # Threshold predictions
        y_pred_binary = ops.cast(y_pred >= self.threshold, 'float32')
        y_true = ops.cast(y_true, 'float32')

        # Flatten spatial dimensions: (B, H, W, C) -> (B*H*W, C)
        y_true_flat = ops.reshape(y_true, (-1, self.num_classes))
        y_pred_flat = ops.reshape(y_pred_binary, (-1, self.num_classes))

        # Compute TP, FP, FN per class
        tp = ops.sum(y_true_flat * y_pred_flat, axis=0)
        fp = ops.sum((1 - y_true_flat) * y_pred_flat, axis=0)
        fn = ops.sum(y_true_flat * (1 - y_pred_flat), axis=0)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        """Compute mean F1 score across all classes."""
        precision = self.true_positives / (
                self.true_positives + self.false_positives + 1e-7
        )
        recall = self.true_positives / (
                self.true_positives + self.false_negatives + 1e-7
        )
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        # Return mean F1 (excluding background if it's all zeros)
        return ops.mean(f1)

    def reset_state(self):
        """Reset all metric accumulators."""
        self.true_positives.assign(ops.zeros_like(self.true_positives))
        self.false_positives.assign(ops.zeros_like(self.false_positives))
        self.false_negatives.assign(ops.zeros_like(self.false_negatives))

# ---------------------------------------------------------------------
# Data Pipelines
# ---------------------------------------------------------------------

def get_imagenet_dataset(
    split: str,
    image_size: int,
    batch_size: int,
    data_dir: Optional[str] = None
) -> tf.data.Dataset:
    """Load ImageNet dataset for MAE pretraining."""
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
        logger.warning(
            f"ImageNet load failed: {e}. Falling back to 'imagenette'."
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
    Load COCO dataset for multi-label semantic segmentation.

    COCO has 80 object categories. We output 80 channels (one per object).
    Background is NOT a separate channel - it's represented by all 80 channels = 0.

    Multi-label approach:
    - Each pixel can belong to multiple classes (overlapping instances)
    - Background = all 80 object channels are 0 (no foreground classes present)
    - Output shape: (B, H, W, 80) with binary indicators for each COCO category

    Args:
        num_classes: Should be 80 for COCO (number of object categories)

    Returns:
        train_ds, val_ds, dataset_info
        Each sample: (image, mask) where mask is (B, H, W, 80)
    """
    logger.info("Creating COCO multi-label segmentation dataset via COCODatasetBuilder...")

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

    def adapt_for_multilabel_segmentation(image, targets):
        """
        Convert from categorical segmentation to multi-label format.

        COCO Convention:
        - COCO has 80 object categories
        - Input may have 81 channels: [background, object1, object2, ..., object80]
        - Output should have 80 channels: [object1, object2, ..., object80]
        - Background = all 80 object channels are 0

        Input:
            image: (B, H, W, 3)
            targets['segmentation']:
                - (B, H, W, 81) if includes background at channel 0
                - (B, H, W, 80) if only objects

        Output:
            image: (B, H, W, 3)
            mask: (B, H, W, 80) - binary multi-hot for 80 COCO objects
                  Background pixels have all 80 channels = 0
        """
        seg_mask = targets['segmentation']
        seg_mask = tf.cast(seg_mask, tf.float32)

        num_channels = tf.shape(seg_mask)[-1]

        # Determine if background channel is present
        # If we have 81 channels, assume channel 0 is background
        has_background_channel = num_channels > num_classes

        if num_classes == 1:
            # Binary segmentation: foreground vs background
            mask = tf.cast(seg_mask > 0.5, tf.float32)
        else:
            # Check if one-hot encoded (mutually exclusive classes)
            mask_sum = tf.reduce_sum(seg_mask, axis=-1, keepdims=True)
            is_one_hot = tf.reduce_all(
                tf.abs(mask_sum - 1.0) < 0.1
            )

            def process_one_hot():
                """
                Convert one-hot to multi-hot, removing background.

                If input has 81 channels: remove channel 0 (background)
                If input has 80 channels: keep all (assuming background already handled)
                """
                if has_background_channel:
                    # Remove background channel (channel 0)
                    # Keep channels 1 to 80 (the COCO objects)
                    foreground_mask = seg_mask[..., 1:]  # (B, H, W, 80)
                else:
                    # Already 80 channels, keep as is
                    foreground_mask = seg_mask

                # Ensure binary
                return tf.cast(foreground_mask > 0.5, tf.float32)

            def process_multi_hot():
                """
                Already multi-hot, remove background if present.
                """
                if has_background_channel:
                    # Remove background channel
                    return tf.cast(seg_mask[..., 1:] > 0.5, tf.float32)
                else:
                    # Already correct format
                    return tf.cast(seg_mask > 0.5, tf.float32)

            # Process based on encoding type
            mask = tf.cond(
                is_one_hot,
                process_one_hot,
                process_multi_hot
            )

        return image, mask

    train_ds = train_ds_raw.map(
        adapt_for_multilabel_segmentation,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds_raw.map(
        adapt_for_multilabel_segmentation,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    logger.info("Multi-label segmentation dataset created")
    logger.info(f"Output format: (image, mask) where mask shape = (B, H, W, {num_classes})")
    logger.info(f"COCO has {num_classes} object categories (no separate background channel)")
    logger.info("Background handling: background pixels have all channels = 0")

    return train_ds, val_ds, dataset_info


# ---------------------------------------------------------------------
# MAE Components
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextWrapper(keras.Model):
    """
    Wraps ConvUNextModel to act as an MAE 'Encoder'.

    For MAE pretraining, the wrapped model should have include_top=True
    to output RGB predictions (3 channels) for reconstruction.
    """
    def __init__(self, convunext_model: ConvUNextModel, **kwargs):
        super().__init__(**kwargs)
        self.convunext_model = convunext_model

    def call(self, inputs, training=None):
        return self.convunext_model(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return self.convunext_model.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        return {
            "convunext_model": keras.saving.serialize_keras_object(self.convunext_model)
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvUNextWrapper":
        config["convunext_model"] = keras.saving.deserialize_keras_object(
            config["convunext_model"]
        )
        return cls(**config)


@keras.saving.register_keras_serializable()
class MultiScaleIdentityDecoder(keras.layers.Layer):
    """Pass-through decoder for MAE deep supervision."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        return inputs


@keras.saving.register_keras_serializable()
class DeepSupervisionMAE(MaskedAutoencoder):
    """MAE that calculates loss across multiple scales."""
    def __init__(self, loss_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_weights = loss_weights
        self.loss_tracker_main = keras.metrics.Mean(name="loss_main")
        self.loss_tracker_aux = keras.metrics.Mean(name="loss_aux_avg")

    def _create_decoder(self):
        return MultiScaleIdentityDecoder(name="identity_decoder")

    def compute_loss(
        self,
        x: keras.KerasTensor,
        y: Optional[keras.KerasTensor] = None,
        y_pred: Optional[Dict[str, Union[keras.KerasTensor, List[keras.KerasTensor]]]] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        reconstructions = y_pred["reconstruction"]
        mask = y_pred["mask"]

        target_full = ops.cast(x, "float32")
        mask_float = ops.cast(mask, "float32")

        if not isinstance(reconstructions, list):
            reconstructions = [reconstructions]

        if self.loss_weights is None:
            weights = [1.0 / (2**i) for i in range(len(reconstructions))]
        elif len(self.loss_weights) != len(reconstructions):
            raise ValueError(
                f"loss_weights length {len(self.loss_weights)} != "
                f"reconstructions length {len(reconstructions)}"
            )
        else:
            weights = self.loss_weights

        total_loss = ops.convert_to_tensor(0.0)

        for i, (recon, weight) in enumerate(zip(reconstructions, weights)):
            recon_shape = ops.shape(recon)
            target_resized = ops.image.resize(
                target_full,
                size=(recon_shape[1], recon_shape[2]),
                interpolation='bilinear'
            )
            target_resized = ops.cast(target_resized, "float32")
            recon = ops.cast(recon, "float32")

            loss_per_pixel = ops.square(target_resized - recon)
            loss_per_pixel = ops.mean(loss_per_pixel, axis=-1)

            H, W = recon_shape[1], recon_shape[2]
            h_patch = ops.cast(self.input_shape_config[0], "int32") // self.patch_size
            w_patch = ops.cast(self.input_shape_config[1], "int32") // self.patch_size

            scale_h = ops.cast(H, "float32") / ops.cast(h_patch * self.patch_size, "float32")
            scale_w = ops.cast(W, "float32") / ops.cast(w_patch * self.patch_size, "float32")

            mask_scaled = self._reshape_mask_for_loss_scaled(
                mask_float,
                target_resized,
                scale_h,
                scale_w
            )
            mask_scaled = ops.maximum(mask_scaled, self.non_mask_value)

            loss_masked = loss_per_pixel * mask_scaled
            num_masked = ops.sum(mask_float, axis=-1) + 1e-6
            pixels_per_patch = self.patch_size * self.patch_size

            scale_factor = (scale_h * scale_w)
            adjusted_pixels = ops.cast(pixels_per_patch, "float32") * scale_factor

            loss_sum = ops.sum(loss_masked, axis=[1, 2])
            scale_loss = loss_sum / (num_masked * adjusted_pixels)

            total_loss = total_loss + (weight * ops.mean(scale_loss))

            if i == 0:
                self.loss_tracker_main.update_state(ops.mean(scale_loss))
            else:
                self.loss_tracker_aux.update_state(ops.mean(scale_loss))

        return total_loss

    def _reshape_mask_for_loss_scaled(
        self,
        mask: keras.KerasTensor,
        target: keras.KerasTensor,
        scale_h: keras.KerasTensor,
        scale_w: keras.KerasTensor
    ) -> keras.KerasTensor:
        B = ops.shape(mask)[0]
        H, W = ops.shape(target)[1], ops.shape(target)[2]

        h_patch = ops.cast(self.input_shape_config[0], "int32") // self.patch_size
        w_patch = ops.cast(self.input_shape_config[1], "int32") // self.patch_size

        mask_grid = ops.reshape(mask, (B, h_patch, w_patch))
        mask_img = ops.image.resize(
            ops.expand_dims(mask_grid, axis=-1),
            size=(H, W),
            interpolation='nearest'
        )
        return ops.squeeze(mask_img, axis=-1)

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.loss_tracker_main,
            self.loss_tracker_aux
        ]


# ---------------------------------------------------------------------
# Training Callbacks
# ---------------------------------------------------------------------

class MAEVisualizationCallback(keras.callbacks.Callback):
    """Callback to visualize MAE reconstructions during training."""
    def __init__(self, val_batch: Tuple, save_dir: str, frequency: int = 5):
        super().__init__()
        self.val_batch = val_batch
        self.save_dir = save_dir
        self.frequency = frequency
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
            return

        images, _ = self.val_batch
        sample = images[:4]

        outputs = self.model(sample, training=True)
        masked = outputs["masked_input"]
        if isinstance(outputs["reconstruction"], list):
            reconstructed = outputs["reconstruction"][0]
        else:
            reconstructed = outputs["reconstruction"]

        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        for i in range(min(4, len(sample))):
            axes[i, 0].imshow(keras.ops.convert_to_numpy(sample[i]))
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(keras.ops.convert_to_numpy(masked[i]))
            axes[i, 1].set_title("Masked")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(keras.ops.convert_to_numpy(reconstructed[i]))
            axes[i, 2].set_title("Reconstructed")
            axes[i, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"mae_epoch_{epoch+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved MAE visualization: {save_path}")


class MultiLabelSegmentationCallback(keras.callbacks.Callback):
    """
    Callback to visualize multi-label segmentation at multiple scales.

    Shows:
    - Input image
    - Ground truth multi-label mask (composite visualization)
    - Predictions at each scale with class overlays
    """
    def __init__(
        self,
        val_batch: Tuple,
        save_dir: str,
        num_classes: int,
        threshold: float = 0.5,
        frequency: int = 5
    ):
        super().__init__()
        self.val_batch = val_batch
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.threshold = threshold
        self.frequency = frequency
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
            return

        images, masks = self.val_batch
        sample_images = images[:2]
        sample_masks = masks[0][:2] if isinstance(masks, tuple) else masks[:2]

        predictions = self.model.predict(sample_images, verbose=0)
        if not isinstance(predictions, list):
            predictions = [predictions]

        num_scales = len(predictions)
        fig, axes = plt.subplots(2, 2 + num_scales, figsize=(4 * (2 + num_scales), 8))

        for i in range(2):
            # Input image
            axes[i, 0].imshow(keras.ops.convert_to_numpy(sample_images[i]))
            axes[i, 0].set_title("Input")
            axes[i, 0].axis('off')

            # Ground truth - composite visualization
            mask_gt = keras.ops.convert_to_numpy(sample_masks[i])
            mask_gt_composite = self._create_composite_mask(mask_gt)
            axes[i, 1].imshow(mask_gt_composite)
            axes[i, 1].set_title(f"GT ({self._count_active_classes(mask_gt)} classes)")
            axes[i, 1].axis('off')

            # Predictions at each scale
            for j, pred in enumerate(predictions):
                pred_mask = keras.ops.convert_to_numpy(pred[i])
                # Threshold predictions
                pred_binary = (pred_mask >= self.threshold).astype(np.float32)
                pred_composite = self._create_composite_mask(pred_binary)
                axes[i, 2 + j].imshow(pred_composite)
                axes[i, 2 + j].set_title(
                    f"Pred Scale {j} ({self._count_active_classes(pred_binary)} classes)"
                )
                axes[i, 2 + j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"seg_epoch_{epoch+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved multi-label segmentation visualization: {save_path}")

    def _create_composite_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create composite visualization of multi-label mask.

        Args:
            mask: (H, W, num_classes) binary mask where num_classes = 80 (COCO objects)
                  All channels represent object categories (no background channel)

        Returns:
            RGB image showing class overlays with transparency
        """
        H, W, C = mask.shape

        # Create RGB canvas
        composite = np.zeros((H, W, 3), dtype=np.float32)

        # Color map for all object classes
        colors = plt.cm.tab20(np.linspace(0, 1, max(C, 20)))[:, :3]

        # Overlay each object class with blending
        for c in range(C):
            class_mask = mask[..., c]
            if class_mask.sum() > 0:  # Only if class is present
                color = colors[c % len(colors)]
                for ch in range(3):
                    composite[..., ch] += class_mask * color[ch]

        # Normalize to [0, 1]
        composite = np.clip(composite, 0, 1)

        return composite

    def _count_active_classes(self, mask: np.ndarray) -> int:
        """
        Count number of active object classes.

        Args:
            mask: (H, W, num_classes) where num_classes = 80 for COCO

        Returns:
            Number of object classes present (not counting background)
        """
        # Sum over spatial dimensions, check which classes are present
        class_present = (mask.sum(axis=(0, 1)) > 0)
        # All channels represent objects, so just count how many are present
        return int(class_present.sum())


class ProgressLoggingCallback(keras.callbacks.Callback):
    """Callback to log training progress."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_str = f"Epoch {epoch + 1}"
        for key, value in logs.items():
            log_str += f" - {key}: {value:.4f}"
        logger.info(log_str)


# ---------------------------------------------------------------------
# Training Functions
# ---------------------------------------------------------------------

def run_mae_pretraining(
    mae_backbone: ConvUNextModel,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    args: argparse.Namespace,
    results_dir: str
) -> None:
    """
    Run MAE pretraining on the backbone.

    The backbone should be configured with include_top=True and output_channels=3
    to output RGB predictions for reconstruction. Later, these weights can be loaded
    into models with different include_top settings or output_channels.
    """
    logger.info("Starting MAE Pretraining...")

    # Wrap backbone as encoder
    encoder_wrapper = ConvUNextWrapper(mae_backbone)

    # Determine loss weights for deep supervision
    if mae_backbone.enable_deep_supervision:
        num_outputs = mae_backbone.depth
        base_weights = [1.0, 0.4, 0.2, 0.1]
        loss_weights = base_weights[:num_outputs]
        logger.info(f"Deep Supervision MAE with {num_outputs} scales")
        logger.info(f"Loss weights: {loss_weights}")
    else:
        loss_weights = None
        logger.info("Single-scale MAE")

    # Create MAE model
    mae_model = DeepSupervisionMAE(
        encoder=encoder_wrapper,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=False,
        input_shape=(args.image_size, args.image_size, 3),
        loss_weights=loss_weights
    )

    # Build the model
    dummy_in = keras.ops.zeros((1, args.image_size, args.image_size, 3))
    mae_model(dummy_in, training=True)

    # Log model summary
    logger.info("=" * 80)
    logger.info("MAE Model Architecture Summary")
    logger.info("=" * 80)
    mae_model.summary(print_fn=lambda x: logger.info(x))
    logger.info("=" * 80)

    # Compile
    lr_schedule = learning_rate_schedule_builder({
        'type': 'cosine_decay',
        'learning_rate': args.mae_lr,
        'decay_steps': args.mae_epochs * 1000,
        'alpha': 0.0
    })

    optimizer = optimizer_builder({
        'type': 'adamw',
        'learning_rate': args.mae_lr,
        'weight_decay': 0.05,
        'clipnorm': 1.0
    }, lr_schedule)

    mae_model.compile(optimizer=optimizer)

    # Get validation batch for visualization
    viz_batch = next(iter(val_ds.take(1)))

    # Callbacks
    mae_viz_dir = os.path.join(results_dir, "mae_viz")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "best_mae_model.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "mae_history.csv")),
        keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True),
        MAEVisualizationCallback(viz_batch, mae_viz_dir, frequency=5),
        ProgressLoggingCallback()
    ]

    # Train
    mae_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.mae_epochs,
        callbacks=callbacks
    )

    logger.info("MAE Pretraining complete.")


def run_segmentation_finetuning(
    convunext_model: ConvUNextModel,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    args: argparse.Namespace,
    results_dir: str,
    dataset_info: Dict
) -> keras.Model:
    """
    Stage 2: Multi-Label Segmentation Fine-tuning.

    Uses sigmoid activation for overlapping classes.
    Background is handled as "no foreground classes present".
    """
    logger.info("=" * 60)
    logger.info(f"STAGE 2: MULTI-LABEL SEGMENTATION FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Activation: sigmoid (allows overlapping classes)")
    logger.info(f"Background handling: all classes = 0")

    # Extract visualization batch BEFORE deep supervision mapping
    viz_batch = next(iter(val_ds.take(1)))

    steps_per_epoch = args.steps_per_epoch or 500
    total_steps = args.finetune_epochs * steps_per_epoch

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.finetune_lr,
        "decay_steps": total_steps,
        "warmup_steps": 500
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "weight_decay": 0.05,
        "gradient_clipping_by_norm": 1.0
    }, lr_schedule)

    # Configure loss for Deep Supervision with Per-Channel Focal Loss
    if convunext_model.enable_deep_supervision:
        # Weights for [Main, Aux1, Aux2...]
        base_weights = [1.0, 0.4, 0.2, 0.1]
        num_outputs = 1 + (convunext_model.depth - 1)
        loss_weights = base_weights[:num_outputs]

        # CRITICAL: Per-channel focal loss for multi-label segmentation
        # Each of the 80 channels is treated as independent binary problem
        focal_loss = create_multilabel_segmentation_loss(
            loss_type='focal',
            alpha=0.75,  # 75% weight on positive class (objects)
            gamma=2.0  # Focus on hard examples (down-weight easy background)
        )

        # Apply to all scales
        loss = [focal_loss] * num_outputs

        # Multi-label metrics (keep these)
        metrics = [
            [
                "binary_accuracy",
                MultiLabelMetrics(num_classes=args.num_classes, name=f"f1_scale_{i}")
            ]
            for i in range(num_outputs)
        ]

        logger.info(f"Deep Supervision enabled: {num_outputs} heads.")
        logger.info(f"Loss: Per-Channel Binary Focal Loss")
        logger.info(f"  - alpha=0.75 (75% weight on objects)")
        logger.info(f"  - gamma=2.0 (focus on hard examples)")
        logger.info(f"  - Applied to each of {args.num_classes} channels independently")
        logger.info(f"  - This is CRITICAL for multi-label segmentation")
        logger.info(f"Loss weights per scale: {loss_weights}")

        def multiscale_target_map(image, mask):
            """
            Maps (B,H,W,C), (B,H,W,num_classes) -> (B,H,W,C), [Target1, Target2...]

            Creates downsampled versions of multi-label mask for each scale.
            """
            targets = [mask]
            h = tf.shape(image)[1]
            w = tf.shape(image)[2]

            for i in range(num_outputs - 1):
                factor = 2 ** (i + 1)
                # Resize multi-label mask
                # Use 'nearest' to preserve binary nature
                m_resized = tf.image.resize(
                    mask,
                    (h // factor, w // factor),
                    method='nearest'
                )
                # Ensure still binary after resize
                m_resized = tf.cast(m_resized > 0.5, tf.float32)
                targets.append(m_resized)

            return image, tuple(targets)

        train_ds = train_ds.map(multiscale_target_map, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(multiscale_target_map, num_parallel_calls=tf.data.AUTOTUNE)

    else:
        # Single scale with per-channel focal loss
        loss = create_multilabel_segmentation_loss(
            loss_type='focal',
            alpha=0.75,
            gamma=2.0
        )
        loss_weights = None

        metrics = [
            "binary_accuracy",
            MultiLabelMetrics(num_classes=args.num_classes, name='f1')
        ]

        logger.info("Single scale mode")
        logger.info("Loss: Per-Channel Binary Focal Loss")
        logger.info("  alpha=0.75, gamma=2.0, per-channel application")

    # Compile (no changes here)
    convunext_model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics
    )

    # Setup Callbacks
    viz_dir = os.path.join(results_dir, "seg_viz")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "best_seg_model.keras"),
            save_best_only=True,
            monitor="val_f1_scale_0" if convunext_model.enable_deep_supervision else "val_f1",
            mode='max'  # Higher F1 is better
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "seg_history.csv")),
        keras.callbacks.EarlyStopping(
            patience=args.patience,
            restore_best_weights=True,
            monitor="val_dice_0" if convunext_model.enable_deep_supervision else "val_dice",
            mode='max'
        ),
        MultiLabelSegmentationCallback(
            val_batch=viz_batch,
            save_dir=viz_dir,
            num_classes=args.num_classes,
            threshold=0.5
        ),
        ProgressLoggingCallback()
    ]

    convunext_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=args.validation_steps,
        callbacks=callbacks
    )

    return convunext_model


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    setup_environment()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)

    # Model
    parser.add_argument("--variant", type=str, default="tiny")
    parser.add_argument("--num-classes", type=int, default=80,
                       help="Number of classes (for COCO: 80 categories, background handled separately)")

    # MAE
    parser.add_argument("--mae-epochs", type=int, default=2)
    parser.add_argument("--mae-lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # Fine-tuning
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)

    # Misc
    parser.add_argument("--skip-mae", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()
    keras.utils.set_random_seed(args.random_seed)

    if args.data_dir is None:
        args.data_dir = os.getenv("TFDS_DATA_DIR", "~/tensorflow_datasets/")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"convunext_multilabel_{args.variant}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    dummy_in = ops.zeros((1, args.image_size, args.image_size, 3))

    # -----------------------------------------------------------
    # 1. Instantiate MAE Backbone
    # -----------------------------------------------------------
    logger.info(f"Instantiating MAE Backbone ({args.variant})...")
    logger.info("Configuration: include_top=True, output_channels=3 (RGB reconstruction)")
    mae_backbone = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=3,
        include_top=True,
        enable_deep_supervision=True,
        use_bias=True
    )
    mae_backbone(dummy_in)

    # -----------------------------------------------------------
    # 2. Stage 1: MAE Pretraining
    # -----------------------------------------------------------
    if not args.skip_mae:
        logger.info("=" * 80)
        logger.info("MAE Backbone (ConvUNext Encoder) Architecture Summary")
        logger.info("=" * 80)
        mae_backbone.summary(print_fn=lambda x: logger.info(x))
        logger.info("=" * 80)

        train_ds_mae = get_imagenet_dataset('train', args.image_size, args.batch_size, args.data_dir)
        val_ds_mae = get_imagenet_dataset('validation', args.image_size, args.batch_size, args.data_dir)

        run_mae_pretraining(mae_backbone, train_ds_mae, val_ds_mae, args, results_dir)
    else:
        logger.info("Skipping MAE pretraining...")

    # -----------------------------------------------------------
    # 3. Stage 2: Multi-Label Segmentation Fine-tuning
    # -----------------------------------------------------------
    train_ds_seg, val_ds_seg, dataset_info = get_coco_segmentation_dataset(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        limit_train_samples=args.steps_per_epoch * args.finetune_epochs if args.steps_per_epoch else None
    )

    logger.info(f"Creating Multi-Label Segmentation Model ({args.num_classes} classes)...")
    logger.info("Configuration: include_top=True, activation=sigmoid (multi-label)")

    # CRITICAL: Use sigmoid for multi-label classification
    seg_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=args.num_classes,
        include_top=True,
        enable_deep_supervision=True,
        final_activation="sigmoid",  # CRITICAL: sigmoid for multi-label
        use_bias=True
    )
    seg_model(dummy_in)

    logger.info("=" * 80)
    logger.info("Multi-Label Segmentation Model Architecture Summary")
    logger.info("=" * 80)
    seg_model.summary(print_fn=lambda x: logger.info(x))
    logger.info("=" * 80)

    # Transfer weights from MAE
    if not args.skip_mae:
        logger.info("Transferring weights from MAE to Segmentation model...")
        logger.info("Note: Backbone weights transferred (stem, encoder, decoder)")
        logger.info(f"Prediction heads NOT transferred (3 RGB vs {args.num_classes} classes)")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as f:
            temp_path = f.name

        try:
            mae_backbone.save_weights(temp_path)
            seg_model.load_weights(temp_path, skip_mismatch=True)
            logger.info("Weight transfer complete.")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    run_segmentation_finetuning(
        seg_model,
        train_ds_seg,
        val_ds_seg,
        args,
        results_dir,
        dataset_info
    )

    final_path = os.path.join(results_dir, "final_multilabel_seg_model.keras")
    seg_model.save(final_path)
    logger.info(f"Pipeline complete. Saved to {final_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)