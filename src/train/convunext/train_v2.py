"""
ConvUNext MAE Segmentation Training Pipeline (ImageNet + COCO)
==============================================================

This script implements a production-grade Two-Stage training workflow:
1. **Self-Supervised Pretraining**: Using Masked Autoencoders (MAE) on **ImageNet**.
2. **Supervised Fine-tuning**: Fine-tuning the pretrained U-Net on **COCO-Stuff**
   for semantic segmentation.

Configuration:
--------------
Data locations can be configured via command line arguments or a .env file.
Define `TFDS_DATA_DIR` in your .env file to set a custom TFDS download location.

Architecture Strategy:
----------------------
We utilize a "Shared-Weight Wrapper" pattern. The `ConvUNextModel` is instantiated
once. A lightweight `ConvUNextEncoderWrapper` is created to expose ONLY the
encoder/bottleneck path to the MAE. Training the MAE updates the wrapper,
which in turn updates the shared layer instances in the main ConvUNextModel.
"""

import os
import math
import argparse
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from typing import Tuple, Optional
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


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def setup_environment():
    """Configure GPU settings, precision, and load environment variables."""
    # Load environment variables from .env file
    load_dotenv()

    # NOTE: Mixed Precision disabled to ensure stability with ConvNeXtV2 blocks
    # keras.mixed_precision.set_global_policy("mixed_float16")

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
# Data Pipelines (TFDS)
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
        # standard ImageNet-1k
        dataset_name = "imagenet2012"
        ds, info = tfds.load(
            dataset_name,
            split=split,
            with_info=True,
            data_dir=data_dir,
            as_supervised=True  # Returns (img, label)
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
    split: str,
    image_size: int,
    batch_size: int,
    data_dir: Optional[str] = None
) -> Tuple[tf.data.Dataset, int]:
    """
    Load COCO-Stuff dataset for Semantic Segmentation.
    'coco_stuff/2017' provides pixel-wise semantic labels.

    :param split: Dataset split ('train', 'validation').
    :param image_size: Target resolution (H, W).
    :param batch_size: Batch size.
    :param data_dir: Optional directory for TFDS data.
    :return: Tuple of (Dataset, num_classes).
    """
    dataset_name = "coco_stuff/2017"
    try:
        ds, info = tfds.load(
            dataset_name,
            split=split,
            with_info=True,
            data_dir=data_dir,
            as_supervised=True  # Returns (image, label_map)
        )
        num_classes = info.features['label'].num_classes
        logger.info(
            f"Loaded {dataset_name} ({split}). Num Classes: {num_classes}"
        )
    except Exception as e:
        logger.warning(
            f"COCO-Stuff load failed: {e}. Falling back to 'oxford_iiit_pet'."
        )
        dataset_name = "oxford_iiit_pet"
        ds, info = tfds.load(
            dataset_name,
            split=split,
            with_info=True,
            data_dir=data_dir,
            as_supervised=True
        )
        # Oxford pets returns (image, label), where label is mask-1 (0,1,2)
        num_classes = 3

    def preprocess_seg(image, mask):
        """Prepare (image, mask) pair."""
        # Resize Image (Bilinear)
        image = tf.image.resize(
            image, (image_size, image_size), method='bilinear'
        )
        image = tf.cast(image, tf.float32) / 255.0

        # Resize Mask (Nearest Neighbor to keep integers)
        mask = tf.image.resize(
            mask, (image_size, image_size), method='nearest'
        )
        mask = tf.cast(mask, tf.int32)

        if dataset_name == "oxford_iiit_pet":
            mask = mask - 1

        return image, mask

    ds = ds.map(preprocess_seg, num_parallel_calls=tf.data.AUTOTUNE)

    if split == 'train':
        ds = ds.shuffle(2000)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, num_classes


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

    def __init__(self, val_images: np.ndarray, save_dir: str, num_samples: int = 4):
        super().__init__()
        self.val_images = val_images
        self.save_dir = save_dir
        self.num_samples = min(num_samples, len(val_images))
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Generate reconstruction grid using the utility function
        # Note: self.model refers to the MAE being trained
        viz_data = self.val_images[:self.num_samples]
        grid = visualize_reconstruction(
            self.model,
            viz_data,
            num_samples=self.num_samples
        )

        # Plot and save
        plt.figure(figsize=(12, 8))
        plt.imshow(grid)
        plt.axis('off')
        plt.title(f"MAE Reconstruction - Epoch {epoch + 1}")

        filename = os.path.join(self.save_dir, f"epoch_{epoch + 1:03d}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved reconstruction visualization to {filename}")


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

    # Estimate steps per epoch
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

    # Prepare visualization callback data
    viz_dir = os.path.join(results_dir, "mae_viz")
    logger.info("Preparing validation batch for reconstruction visualization...")
    # Get a single batch from val_ds. val_ds yields (x, x), take first x.
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
        )
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
    results_dir: str
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Stage 2: Fine-tune on COCO-Stuff.

    :param convunext_model: The pretrained UNext model.
    :param train_ds: Training dataset.
    :param val_ds: Validation dataset.
    :param num_classes: Number of segmentation classes.
    :param args: Parsed command line arguments.
    :param results_dir: Directory to save artifacts.
    :return: Tuple of (Finetuned Model, History object).
    """
    logger.info("=" * 60)
    logger.info(f"STAGE 2: SEGMENTATION FINE-TUNING (Classes: {num_classes})")
    logger.info("=" * 60)

    # ---------------------------------------------
    # Phase 2a: Freeze Encoder
    # ---------------------------------------------
    logger.info("Phase 2a: Freezing encoder...")
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

    convunext_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs_stage1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=2, restore_best_weights=True
            )
        ]
    )

    # ---------------------------------------------
    # Phase 2b: Unfreeze All
    # ---------------------------------------------
    logger.info("Phase 2b: Unfreezing all layers...")
    convunext_model.trainable = True
    convunext_model.stem.trainable = True
    for stage in convunext_model.encoder_stages:
        for block in stage:
            block.trainable = True

    cardinality = int(train_ds.cardinality())
    steps_per_epoch = cardinality if cardinality > 0 else 500

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.finetune_lr_stage2,
        "decay_steps": args.finetune_epochs_stage2 * steps_per_epoch,
        "warmup_steps": 500
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "weight_decay": 0.01
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
            monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(
            os.path.join(results_dir, "segmentation_history.csv")
        ),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, "logs", "finetune")
        )
    ]

    history = convunext_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs_stage2,
        callbacks=callbacks
    )

    return convunext_model, history


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    setup_environment()

    parser = argparse.ArgumentParser(
        description="ConvUNext (ImageNet MAE -> COCO Seg)"
    )

    # Data params
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="TFDS data directory. Overrides TFDS_DATA_DIR env var."
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)

    # Model params
    parser.add_argument(
        "--variant", type=str, default="tiny",
        choices=["tiny", "small", "base"]
    )

    # MAE params
    parser.add_argument("--mae-epochs", type=int, default=100)
    parser.add_argument("--mae-lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # Finetune params
    parser.add_argument("--finetune-epochs-stage1", type=int, default=5)
    parser.add_argument("--finetune-epochs-stage2", type=int, default=50)
    parser.add_argument("--finetune-lr-stage1", type=float, default=1e-3)
    parser.add_argument("--finetune-lr-stage2", type=float, default=5e-5)

    # Flags
    parser.add_argument(
        "--skip-mae", action="store_true", help="Skip pretraining"
    )

    args = parser.parse_args()

    # Resolve Data Directory
    # Priority: 1. CLI Arg, 2. Env Var, 3. None (Default)
    if args.data_dir is None:
        env_data_dir = os.getenv("TFDS_DATA_DIR", "~/tensorflow_datasets/")
        if env_data_dir:
            args.data_dir = env_data_dir
            logger.info(
                f"Resolved TFDS data directory from .env: {args.data_dir}"
            )
        else:
            logger.info(
                "No custom data directory provided. Using TFDS default."
            )
    else:
        logger.info(f"Using CLI provided data directory: {args.data_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        "results", f"full_run_{args.variant}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # 1. Instantiate Main Model (Placeholder Output Channels)
    # We will need to re-initialize the head once we know the exact number of
    # COCO classes. But for MAE, the head doesn't matter.
    logger.info(f"Instantiating ConvUNext ({args.variant})...")

    # We start with a dummy output channel count.
    # The weights that matter (Encoder) are shared.
    # We will handle the head mismatch logic in Stage 2.
    convunext_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=1,  # Temporary
        use_bias=True
    )
    # Build
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
    # Load COCO first to get num_classes
    train_ds_seg, num_classes = get_coco_segmentation_dataset(
        'train', args.image_size, args.batch_size, args.data_dir
    )
    val_ds_seg, _ = get_coco_segmentation_dataset(
        'validation', args.image_size, args.batch_size, args.data_dir
    )

    logger.info(
        f"Reconfiguring model for {num_classes} segmentation classes..."
    )

    # CRITICAL: We need to update the final head of the model to match
    # num_classes. Since ConvUNextModel has a fixed head at init, we have
    # two options:
    # 1. Create a NEW model and transfer encoder weights (Safest).
    # 2. Hack the existing model (Not recommended in Keras 3).

    # Option 1: Create New Model & Transfer Weights
    final_seg_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=num_classes,
        final_activation="softmax",
        use_bias=True
    )
    final_seg_model(dummy_in)  # Build

    # Transfer Weights (Encoder + Bottleneck)
    # This works because 'convunext_model' holds the trained MAE weights
    logger.info("Transferring MAE-pretrained weights to Segmentation Model...")

    # We iterate layers and match by name/path logic manually or just use
    # set_weights on the specific sub-blocks which we know match.
    final_seg_model.stem.set_weights(convunext_model.stem.get_weights())

    for i in range(len(convunext_model.encoder_stages)):
        # Stage blocks
        for j in range(len(convunext_model.encoder_stages[i])):
            final_seg_model.encoder_stages[i][j].set_weights(
                convunext_model.encoder_stages[i][j].get_weights()
            )
        # Downsample
        if i < len(convunext_model.encoder_downsamples):
            final_seg_model.encoder_downsamples[i].set_weights(
                convunext_model.encoder_downsamples[i].get_weights()
            )

    # Bottleneck
    final_seg_model.bottleneck_entry.set_weights(
        convunext_model.bottleneck_entry.get_weights()
    )
    for i in range(len(convunext_model.bottleneck_blocks)):
        final_seg_model.bottleneck_blocks[i].set_weights(
            convunext_model.bottleneck_blocks[i].get_weights()
        )

    logger.info("Weight transfer complete.")

    # Run Fine-tuning
    run_segmentation_finetuning(
        final_seg_model, train_ds_seg, val_ds_seg,
        num_classes, args, results_dir
    )

    logger.info(f"Pipeline complete. Results saved to {results_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)