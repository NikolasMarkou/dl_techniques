"""
ConvUNext MAE & Deep Supervision Training Pipeline
==================================================

A production-grade training script for the ConvUNext architecture, featuring:

1.  **Two-Stage Training**:
    -   Stage 1: Masked Autoencoder (MAE) Pre-training on ImageNet.
    -   Stage 2: Multi-Label Segmentation Fine-tuning on COCO.

2.  **Advanced Data Pipelines**:
    -   Utilizes ``dl_techniques.utils.datasets.coco.COCODatasetBuilder``.
    -   Handles multi-scale target generation for deep supervision.
    -   Efficient tf.data processing with prefetching and parallel mapping.

3.  **Deep Supervision**:
    -   Aligns dataset targets with ConvUNext's multi-scale outputs.
    -   Output order: [Full Res, 1/2 Res, 1/4 Res, 1/8 Res, ...].

4.  **Serialization & Resumption**:
    -   Saves/Loads weights compatible between ``include_top=True`` (training)
        and ``include_top=False`` (inference) modes.

Usage:
    .. code-block:: bash

        python train_v5.py --variant base --batch-size 32 --mae-epochs 50 --finetune-epochs 100

"""

import os
import keras
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import ops
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List
from dotenv import load_dotenv

# Set backend-agnostic matplotlib backend
matplotlib.use('Agg')

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.convunext.model import ConvUNextModel
from dl_techniques.datasets.vision.coco import COCODatasetBuilder

# Optimization & Loss Factories
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)
from dl_techniques.losses.multi_labels_loss import create_multilabel_segmentation_loss
from dl_techniques.metrics.multi_label_metrics import MultiLabelMetrics

# MAE Specifics (Assumed to be in local package)
from dl_techniques.models.masked_autoencoder import (
    MaskedAutoencoder,
    visualize_reconstruction
)


# ---------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------

def setup_environment() -> None:
    """
    Configure GPU settings, precision policies, and load environment variables.

    Sets memory growth to True to prevent TensorFlow from allocating all GPU
    memory at startup, allowing for better resource sharing.
    """
    load_dotenv()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"❌ GPU setup error: {e}")
    else:
        logger.warning("⚠️ No GPUs found, using CPU")


# ---------------------------------------------------------------------
# Data Pipeline: ImageNet (MAE Pre-training)
# ---------------------------------------------------------------------

def get_imagenet_dataset(
    split: str,
    image_size: int,
    batch_size: int,
    data_dir: Optional[str] = None
) -> tf.data.Dataset:
    """
    Load and preprocess ImageNet (or fallback) for MAE pre-training.

    The MAE task requires the input image to serve as both the input (x)
    and the reconstruction target (y).

    :param split: Dataset split ('train', 'validation').
    :param image_size: Target resolution for resizing.
    :param batch_size: Batch size.
    :param data_dir: Directory for TFDS data.
    :return: ``tf.data.Dataset`` yielding ``(image, image)``.
    """
    dataset_name = "imagenet2012"
    try:
        ds, info = tfds.load(
            dataset_name,
            split=split,
            with_info=True,
            data_dir=data_dir,
            as_supervised=True
        )
        logger.info(f"Loaded {dataset_name} ({split})")
    except Exception as e:
        logger.warning(f"ImageNet load failed: {e}. Falling back to 'imagenette'.")
        dataset_name = "imagenette/320px-v2"
        ds, info = tfds.load(
            dataset_name,
            split=split,
            with_info=True,
            data_dir=data_dir,
            as_supervised=True
        )

    def preprocess_mae(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Resize, normalize, and format for Autoencoder.

        :param image: Raw uint8 image.
        :param label: Ignored for MAE.
        :return: (normalized_image, normalized_image).
        """
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, image

    # Optimization pipeline
    if split == 'train':
        ds = ds.shuffle(buffer_size=10000)

    ds = ds.map(preprocess_mae, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ---------------------------------------------------------------------
# Data Pipeline: COCO (Segmentation Fine-tuning)
# ---------------------------------------------------------------------

def get_coco_segmentation_dataset(
    image_size: int,
    batch_size: int,
    num_classes: int,
    augment_data: bool = True,
    limit_train_samples: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]:
    """
    Create COCO datasets for Multi-Label Semantic Segmentation.

    Utilizes ``COCODatasetBuilder`` to generate aligned image/mask pairs.
    Handles the unpacking of the builder's dictionary format into the tuple
    format required by Keras models ``(x, y)``.

    **Format Note**:
    The builder returns masks of shape ``(H, W, num_classes)``. The background
    class (index 0) is stripped by the builder configuration, meaning a pixel
    with all zeros is background.

    :param image_size: Target height/width.
    :param batch_size: Batch size.
    :param num_classes: Number of object classes (e.g., 80 for COCO).
    :param augment_data: Apply data augmentation to training set.
    :param limit_train_samples: Cap training data size (for debugging).
    :return: (train_ds, val_ds, dataset_info_dict).
    """
    logger.info("Initializing COCO Segmentation Pipeline...")

    # 1. Initialize Builder
    # We use the factory function or class directly. The builder handles
    # loading, filtering, and basic augmentation internally.
    builder = COCODatasetBuilder(
        img_size=image_size,
        batch_size=batch_size,
        max_boxes_per_image=100,  # Required for internal graph logic, though unused for seg output
        use_detection=False,
        use_segmentation=True,
        segmentation_classes=num_classes,
        augment_data=augment_data,
        limit_train_samples=limit_train_samples
    )

    # 2. Create raw datasets (Yields: (image, targets_dict))
    train_ds_raw, val_ds_raw = builder.create_datasets()
    dataset_info = builder.get_dataset_info()

    # 3. Adapter: Convert Dict -> Tuple (x, y)
    def unpack_targets(image: tf.Tensor, targets: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Extract segmentation mask from targets dictionary.

        Input targets['segmentation'] is (B, H, W, num_classes).
        Values are 0.0 or 1.0 (float32).
        """
        mask = targets['segmentation']
        # Builder guarantees float32 and correct shape (B, H, W, C)
        return image, mask

    train_ds = train_ds_raw.map(unpack_targets, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds_raw.map(unpack_targets, num_parallel_calls=tf.data.AUTOTUNE)

    logger.info(f"✅ Dataset Ready: (B, {image_size}, {image_size}, 3) -> (B, {image_size}, {image_size}, {num_classes})")

    return train_ds, val_ds, dataset_info


# ---------------------------------------------------------------------
# MAE Wrappers & Logic
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextWrapper(keras.Model):
    """
    Wrapper to treat ConvUNextModel as a pure Encoder for MAE.

    Necessary because MAE expects an 'encoder' attribute that takes inputs
    and returns features. While ConvUNextModel is a Model, wrapping it
    ensures clean serialization of the 'encoder' slot in the MAE class.
    """
    def __init__(self, convunext_model: ConvUNextModel, **kwargs):
        super().__init__(**kwargs)
        self.convunext_model = convunext_model

    def call(self, inputs, training=None):
        return self.convunext_model(inputs, training=training)

    def compute_output_shape(self, input_shape):
        return self.convunext_model.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        return {"convunext_model": keras.saving.serialize_keras_object(self.convunext_model)}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvUNextWrapper":
        config["convunext_model"] = keras.saving.deserialize_keras_object(config["convunext_model"])
        return cls(**config)


@keras.saving.register_keras_serializable()
class MultiScaleIdentityDecoder(keras.layers.Layer):
    """Pass-through decoder that allows MAE to calculate loss on encoder outputs directly."""
    def call(self, inputs, training=None):
        return inputs


@keras.saving.register_keras_serializable()
class DeepSupervisionMAE(MaskedAutoencoder):
    """
    Extended Masked Autoencoder that calculates reconstruction loss across multiple scales.

    Used when the backbone (ConvUNext) outputs a list of features [Full, 1/2, 1/4...]
    due to deep supervision being enabled.
    """
    def __init__(self, loss_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_weights = loss_weights
        self.loss_tracker_main = keras.metrics.Mean(name="loss_main")
        self.loss_tracker_aux = keras.metrics.Mean(name="loss_aux_avg")

    def _create_decoder(self):
        # ConvUNext include_top=True essentially acts as the decoder for reconstruction
        return MultiScaleIdentityDecoder(name="identity_decoder")

    def compute_loss(self, x, y=None, y_pred=None, **kwargs):
        """
        Calculate MSE loss weighted across multiple scales.
        """
        reconstructions = y_pred["reconstruction"]
        mask = y_pred["mask"]

        # Ensure reconstructions is a list
        if not isinstance(reconstructions, list):
            reconstructions = [reconstructions]

        # Default weights: decay by factor of 2 for each smaller scale
        if self.loss_weights is None:
            weights = [1.0 / (2**i) for i in range(len(reconstructions))]
        else:
            weights = self.loss_weights

        total_loss = ops.convert_to_tensor(0.0)
        target_full = ops.cast(x, "float32")
        mask_float = ops.cast(mask, "float32")

        for i, (recon, weight) in enumerate(zip(reconstructions, weights)):
            # Resize target to match current reconstruction scale
            recon_shape = ops.shape(recon) # (B, H_s, W_s, C)
            target_resized = ops.image.resize(
                target_full,
                size=(recon_shape[1], recon_shape[2]),
                interpolation='bilinear'
            )

            # Standard MSE per pixel
            loss_per_pixel = ops.square(target_resized - recon)
            loss_per_pixel = ops.mean(loss_per_pixel, axis=-1)  # (B, H_s, W_s)

            # Reshape mask to match current scale (using Nearest Neighbor)
            mask_scaled = self._reshape_mask_for_loss_scaled(
                mask_float, recon_shape[1], recon_shape[2]
            )

            # Apply mask (learn on masked patches)
            # MAE typically computes loss on invisible patches
            loss_masked = loss_per_pixel * mask_scaled

            # Normalize loss
            sum_loss = ops.sum(loss_masked)
            sum_mask = ops.sum(mask_scaled) + 1e-6
            scale_loss = sum_loss / sum_mask

            total_loss = total_loss + (weight * scale_loss)

            # Metrics
            if i == 0:
                self.loss_tracker_main.update_state(scale_loss)
            else:
                self.loss_tracker_aux.update_state(scale_loss)

        return total_loss

    def _reshape_mask_for_loss_scaled(self, mask, height, width):
        """
        Reshape the (B, num_patches) mask to (B, height, width) spatial map.
        """
        B = ops.shape(mask)[0]
        # Get patch grid dims from config
        h_grid = self.input_shape_config[0] // self.patch_size
        w_grid = self.input_shape_config[1] // self.patch_size

        mask_grid = ops.reshape(mask, (B, h_grid, w_grid))
        # Upsample grid to current feature map resolution
        mask_img = ops.image.resize(
            ops.expand_dims(mask_grid, -1),
            size=(height, width),
            interpolation='nearest'
        )
        return ops.squeeze(mask_img, -1)

    @property
    def metrics(self):
        return [self.loss_tracker_main, self.loss_tracker_aux]


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------

class MultiLabelSegmentationCallback(keras.callbacks.Callback):
    """
    Callback to visualize multi-label segmentation predictions during training.

    Generates composite images overlaying predicted masks on input images
    to debug model performance visually.
    """
    def __init__(self, val_batch, save_dir, num_classes, threshold=0.5, frequency=5):
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

        # Unpack data: val_batch might be mapped to (x, (y1, y2...))
        images, targets = self.val_batch

        # If tuple (deep supervision), take main target
        if isinstance(targets, (list, tuple)):
            main_mask = targets[0]
        else:
            main_mask = targets

        # Predict
        preds = self.model.predict(images[:4], verbose=0)

        # Handle list output (deep supervision)
        if isinstance(preds, list):
            preds = preds[0] # Take main output

        fig, axes = plt.subplots(4, 3, figsize=(12, 16))

        for i in range(min(4, len(images))):
            # 1. Original
            axes[i, 0].imshow(keras.ops.convert_to_numpy(images[i]))
            axes[i, 0].set_title("Input")
            axes[i, 0].axis('off')

            # 2. GT
            gt = keras.ops.convert_to_numpy(main_mask[i])
            axes[i, 1].imshow(self._make_composite(gt))
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            # 3. Pred
            pr = keras.ops.convert_to_numpy(preds[i])
            axes[i, 2].imshow(self._make_composite(pr > self.threshold))
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"epoch_{epoch+1}.png")
        plt.savefig(path)
        plt.close()

    def _make_composite(self, mask):
        """Simple strategy: collapse channels to color map."""
        # mask: (H, W, C)
        if mask.shape[-1] == 1:
            return mask[:, :, 0]

        # Collapse to RGB via simple linear projection for visualization
        h, w, c = mask.shape
        composite = np.zeros((h, w, 3))
        colors = plt.cm.jet(np.linspace(0, 1, c))[:, :3]

        for ch in range(c):
            # Add color weighted by mask presence
            composite += mask[:, :, ch:ch+1] * colors[ch]

        return np.clip(composite, 0, 1)


# ---------------------------------------------------------------------
# Training Stages
# ---------------------------------------------------------------------

def run_mae_pretraining(
    backbone: ConvUNextModel,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    args: argparse.Namespace,
    results_dir: str
) -> None:
    """
    Stage 1: Pre-train ConvUNext as a Masked Autoencoder.
    """
    logger.info("🚀 Starting Stage 1: MAE Pre-training")

    # Wrap model (ConvUNext acts as encoder and projection head)
    encoder = ConvUNextWrapper(backbone)

    # Calculate Deep Supervision weights if enabled
    loss_weights = None
    if backbone.enable_deep_supervision:
        # Weights for [Level 0, Level 1, Level 2, Level 3]
        # Level 0 is main output (highest res), others are aux
        loss_weights = [1.0, 0.5, 0.25, 0.125][:backbone.depth]
        logger.info(f"Using Deep Supervision loss weights: {loss_weights}")

    mae = DeepSupervisionMAE(
        encoder=encoder,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        input_shape=(args.image_size, args.image_size, 3),
        loss_weights=loss_weights,
        norm_pix_loss=True  # Normalize pixel target
    )

    # Compile
    lr_schedule = learning_rate_schedule_builder({
        'type': 'cosine_decay',
        'learning_rate': args.mae_lr,
        'decay_steps': args.mae_epochs * 1000,
        'warmup_steps': 1000
    })

    optimizer = optimizer_builder({
        'type': 'adamw',
        'weight_decay': 0.05,
        'clipnorm': 1.0
    }, lr_schedule)

    mae.compile(optimizer=optimizer)

    # Build
    mae(keras.ops.zeros((1, args.image_size, args.image_size, 3)))

    # Train
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "mae_best.keras"),
            save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "mae_log.csv"))
    ]

    mae.fit(train_ds, validation_data=val_ds, epochs=args.mae_epochs, callbacks=callbacks)
    logger.info("✅ MAE Pre-training Complete")


def run_segmentation_finetuning(
    model: ConvUNextModel,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    args: argparse.Namespace,
    results_dir: str
) -> None:
    """
    Stage 2: Fine-tune for Multi-Label Segmentation.
    """
    logger.info("🚀 Starting Stage 2: Segmentation Fine-tuning")

    # 1. Deep Supervision Data Mapping
    if model.enable_deep_supervision:
        logger.info("Configuring Deep Supervision targets...")
        # ConvUNext returns [Main, Aux1, Aux2, Aux3]
        # Aux1 is scale 1/2, Aux2 is scale 1/4, etc.
        num_aux = model.depth - 1

        def multiscale_mapper(img, mask):
            targets = [mask]
            h, w = args.image_size, args.image_size

            for i in range(num_aux):
                factor = 2 ** (i + 1) # 2, 4, 8...
                # Nearest neighbor for masks to preserve binary values
                scaled = tf.image.resize(mask, (h // factor, w // factor), method='nearest')
                targets.append(scaled)

            return img, tuple(targets)

        train_ds = train_ds.map(multiscale_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(multiscale_mapper, num_parallel_calls=tf.data.AUTOTUNE)

        # Loss weights
        loss_weights = [1.0] + [0.5 / (2**i) for i in range(num_aux)]
    else:
        loss_weights = None # Single output

    # 2. Optimization
    total_steps = args.finetune_epochs * (args.steps_per_epoch or 1000)
    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.finetune_lr,
        "decay_steps": total_steps,
        "warmup_steps": 500
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "weight_decay": 0.01
    }, lr_schedule)

    # 3. Loss & Metrics
    # Soft Dice Loss is standard for segmentation
    loss_fn = create_multilabel_segmentation_loss('dice', smooth=1.0)

    # Keras automatically applies list of losses to list of outputs if single loss obj provided?
    # No, usually need list if multiple outputs.
    if model.enable_deep_supervision:
        losses = [loss_fn] * (1 + num_aux)
        metrics = [[
            "binary_accuracy",
            MultiLabelMetrics(num_classes=args.num_classes, name=f"f1_L{i}")
        ] for i in range(1 + num_aux)]
    else:
        losses = loss_fn
        metrics = ["binary_accuracy", MultiLabelMetrics(num_classes=args.num_classes, name="f1")]

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    # 4. Callbacks
    viz_batch = next(iter(val_ds.take(1)))
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "seg_best.keras"),
            save_best_only=True, monitor="val_loss"
        ),
        MultiLabelSegmentationCallback(
            viz_batch, os.path.join(results_dir, "viz"), args.num_classes
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "seg_log.csv")),
        keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        callbacks=callbacks
    )


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    setup_environment()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data Config
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=80)

    # Model Config
    parser.add_argument("--variant", type=str, default="tiny", choices=['tiny', 'small', 'base', 'large'])

    # Training Config
    parser.add_argument("--mae-epochs", type=int, default=50)
    parser.add_argument("--mae-lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--skip-mae", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    keras.utils.set_random_seed(args.seed)

    # Directory Setup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"convunext_{args.variant}_{ts}")
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Instantiate Backbone
    # -----------------------------------------------------------------
    # We use include_top=True with output_channels=3 for MAE (reconstructing RGB)
    # For Segmentation, we will create a new model and load weights.
    dummy_in = ops.zeros((1, args.image_size, args.image_size, 3))

    logger.info("Initializing MAE Backbone...")
    mae_backbone = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=3, # RGB Reconstruction
        include_top=True,
        enable_deep_supervision=True, # MAE uses multi-scale loss
        use_bias=True
    )
    mae_backbone(dummy_in) # Build

    # -----------------------------------------------------------------
    # 2. Stage 1: MAE
    # -----------------------------------------------------------------
    if not args.skip_mae:
        train_ds_mae = get_imagenet_dataset('train', args.image_size, args.batch_size, args.data_dir)
        val_ds_mae = get_imagenet_dataset('validation', args.image_size, args.batch_size, args.data_dir)

        run_mae_pretraining(mae_backbone, train_ds_mae, val_ds_mae, args, results_dir)

        # Save MAE weights
        mae_weights_path = os.path.join(results_dir, "mae_weights_transfer.weights.h5")
        mae_backbone.save_weights(mae_weights_path)
    else:
        logger.warning("Skipping MAE pre-training")
        mae_weights_path = None

    # -----------------------------------------------------------------
    # 3. Stage 2: Segmentation
    # -----------------------------------------------------------------
    # COCO Data
    train_ds_seg, val_ds_seg, _ = get_coco_segmentation_dataset(
        args.image_size, args.batch_size, args.num_classes,
        limit_train_samples=args.steps_per_epoch * args.finetune_epochs if args.steps_per_epoch else None
    )

    # Create Segmentation Model
    # Note: We must enable deep supervision here if we want to use the transferred weights
    # structure correctly, though we could disable it if transferring only encoder.
    # To keep things simple and robust, we match the architecture.
    logger.info("Initializing Segmentation Model...")
    seg_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=args.num_classes, # COCO Classes
        include_top=True,
        enable_deep_supervision=True,
        final_activation='sigmoid', # Multi-label
        use_bias=True
    )
    seg_model(dummy_in) # Build

    # Weight Transfer
    if mae_weights_path and os.path.exists(mae_weights_path):
        logger.info("Transffering MAE weights...")
        # skip_mismatch=True allows loading backbone weights while ignoring
        # the shape mismatch in the final classification/projection heads
        # (3 channels vs 80 channels)
        seg_model.load_weights(mae_weights_path, skip_mismatch=True)

    run_segmentation_finetuning(seg_model, train_ds_seg, val_ds_seg, args, results_dir)

    # Final Save
    seg_model.save(os.path.join(results_dir, "final_model.keras"))
    logger.info("✅ Pipeline Completed Successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal Pipeline Error: {e}", exc_info=True)