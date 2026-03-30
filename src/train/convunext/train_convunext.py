"""
ConvUNext MAE Segmentation Training Pipeline
============================================

This script implements a production-grade Two-Stage training workflow:
1. **Self-Supervised Pretraining**: Using Masked Autoencoders (MAE) to learn robust
   features from unlabeled data (simulated here by ignoring labels).
2. **Supervised Fine-tuning**: Fine-tuning the pretrained U-Net for semantic segmentation.

Architecture Strategy:
----------------------
We utilize a "Shared-Weight Wrapper" pattern. The `ConvUNextModel` is instantiated
once. A lightweight `ConvUNextEncoderWrapper` is created to expose ONLY the
encoder/bottleneck path to the MAE. Training the MAE updates the wrapper,
which in turn updates the shared layer instances in the main ConvUNextModel.
"""

import os
import math
import keras
import argparse
import numpy as np
import tensorflow as tf
from typing import Tuple
from datetime import datetime

# ---------------------------------------------------------------------
# Framework Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.convunext import ConvUNextModel
from dl_techniques.models.masked_autoencoder import MaskedAutoencoder, visualize_reconstruction
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def setup_environment():
    """Configure GPU settings and precision."""
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

def generate_synthetic_data(
    num_samples: int,
    height: int,
    width: int,
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic geometric shapes for segmentation testing.
    Ensures the script runs out-of-the-box without external datasets.
    """
    logger.info(f"Generating {num_samples} synthetic samples ({height}x{width})...")

    x_data = np.zeros((num_samples, height, width, 3), dtype="float32")
    y_data = np.zeros((num_samples, height, width), dtype="int32")

    for i in range(num_samples):
        # Background noise
        x_data[i] = np.random.rand(height, width, 3) * 0.1

        # Shape 1: Circle (Class 1)
        cy, cx = np.random.randint(0, height), np.random.randint(0, width)
        cr = np.random.randint(10, height // 4)
        y, x = np.ogrid[:height, :width]
        mask_circle = ((y - cy) ** 2 + (x - cx) ** 2) <= cr ** 2
        x_data[i][mask_circle] += [0.5, 0.0, 0.0]  # Red tint
        y_data[i][mask_circle] = 1

        # Shape 2: Rectangle (Class 2)
        if num_classes > 2:
            ry, rx = np.random.randint(0, height - 20), np.random.randint(0, width - 20)
            rh, rw = np.random.randint(10, 50), np.random.randint(10, 50)
            mask_rect = np.zeros((height, width), dtype=bool)
            mask_rect[ry:ry + rh, rx:rx + rw] = True
            # Overwrite overlapping regions
            x_data[i][mask_rect] = np.random.rand(np.sum(mask_rect), 3) * 0.2 + [0.0, 0.5, 0.0]
            y_data[i][mask_rect] = 2

    return np.clip(x_data, 0, 1), y_data

# ---------------------------------------------------------------------
# Encoder Wrapper
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextEncoderWrapper(keras.Model):
    """
    Wraps the Encoder path of a ConvUNextModel for MAE compatibility.

    This model shares the exact layer instances with the provided ConvUNextModel.
    When MAE trains this wrapper, the weights in the main ConvUNextModel are
    updated automatically.

    Path: Input -> Stem -> Encoder Stages (Downsamples) -> Bottleneck -> Output
    """

    def __init__(self, convunext_model: ConvUNextModel, **kwargs):
        super().__init__(**kwargs)
        
        # NOTE: We map specific layers instead of storing the whole model to 
        # avoid Keras tracking unused variables (Decoder) which causes 
        # "Gradients do not exist" warnings.
        self.stem = convunext_model.stem
        self.encoder_stages = convunext_model.encoder_stages
        self.encoder_downsamples = convunext_model.encoder_downsamples
        self.bottleneck_entry = convunext_model.bottleneck_entry
        self.bottleneck_blocks = convunext_model.bottleneck_blocks
        
        # Store config for shape calculation
        self.input_shape_config = convunext_model.input_shape_config
        self.depth = convunext_model.depth
        self.filter_sizes = convunext_model.filter_sizes

        # We need to construct the input spec for the MAE to introspect shapes
        self.input_spec = keras.layers.InputSpec(
            shape=(None,) + self.input_shape_config
        )

    def call(self, inputs, training=None):
        """Forward pass through the encoder and bottleneck."""
        x = inputs

        # 1. Stem
        x = self.stem(x, training=training)

        # 2. Encoder Stages
        for level in range(self.depth):
            # Process blocks in this stage
            for block in self.encoder_stages[level]:
                x = block(x, training=training)

            # Downsample (if not last stage)
            if level < self.depth - 1:
                x = self.encoder_downsamples[level](x, training=training)

        # 3. Bottleneck
        x = self.bottleneck_entry(x, training=training)

        for block in self.bottleneck_blocks:
            x = block(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        """Calculate the shape of the latent representation."""
        batch_size = input_shape[0]
        h, w, c = self.input_shape_config

        downsample_factor = 2 ** self.depth

        new_h = h // downsample_factor if h else None
        new_w = w // downsample_factor if w else None

        # Final channels is the bottleneck dimension
        final_channels = self.filter_sizes[self.depth]

        return (batch_size, new_h, new_w, final_channels)

    def get_config(self):
        # Note: Deserialization requires passing the model instance again
        # or reconstructing it. For strict serialization workflows, 
        # ensure the main model is available.
        return super().get_config()

# ---------------------------------------------------------------------
# Training Workflow
# ---------------------------------------------------------------------

def run_mae_pretraining(
    convunext_model: ConvUNextModel,
    train_data: np.ndarray,
    val_data: np.ndarray,
    args: argparse.Namespace,
    results_dir: str
) -> MaskedAutoencoder:
    """
    Stage 1: Pretrain the encoder using Masked Autoencoding.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: MAE SELF-SUPERVISED PRETRAINING")
    logger.info("=" * 60)

    # 1. Wrap the encoder
    # This shares weights: updating 'encoder_wrapper' updates 'convunext_model'
    encoder_wrapper = ConvUNextEncoderWrapper(convunext_model)

    # 2. Configure MAE
    # Auto-calculate depth for the lightweight decoder to match encoder stride
    downsample_factor = 2 ** convunext_model.depth
    decoder_depth = int(math.log2(downsample_factor))

    mae = MaskedAutoencoder(
        encoder=encoder_wrapper,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        decoder_depth=decoder_depth,  # Symmetric depth usually works best
        decoder_dims=None, # Let it auto-calculate
        input_shape=(args.image_size, args.image_size, 3)
    )

    # 3. Compile MAE
    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.mae_lr,
        "decay_steps": args.mae_epochs * (len(train_data) // args.batch_size),
        "warmup_steps": 500
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "weight_decay": 0.05,
        "gradient_clipping_by_norm": 1.0
    }, lr_schedule)

    mae.compile(optimizer=optimizer)

    # 4. Train
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "mae_weights.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "mae_history.csv"))
    ]

    mae.fit(
        train_data,
        epochs=args.mae_epochs,
        batch_size=args.batch_size,
        validation_data=(val_data, val_data), # MAE is self-supervised (x=y)
        callbacks=callbacks
    )

    # 5. Visualize
    logger.info("Generating MAE visualizations...")
    viz_dir = os.path.join(results_dir, "mae_viz")
    os.makedirs(viz_dir, exist_ok=True)

    grid = visualize_reconstruction(mae, val_data, num_samples=4)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.imshow(grid)
    plt.axis('off')
    plt.title("MAE: Original | Masked | Reconstructed")
    plt.savefig(os.path.join(viz_dir, "reconstruction_sample.png"))
    plt.close()

    return mae

def run_segmentation_finetuning(
    convunext_model: ConvUNextModel,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    num_classes: int,
    args: argparse.Namespace,
    results_dir: str
) -> keras.Model:
    """
    Stage 2: Fine-tune the full U-Net for segmentation.
    Includes explicit "Freeze Encoder" -> "Train Decoder" -> "Unfreeze All" flow.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: SEGMENTATION FINE-TUNING")
    logger.info("=" * 60)

    x_train, y_train = train_data
    x_val, y_val = val_data

    # ---------------------------------------------
    # Phase 2a: Freeze Encoder (Train Randomly Init Decoder)
    # ---------------------------------------------
    logger.info("Phase 2a: Freezing encoder, training decoder...")

    # Recursively freeze encoder components in the model
    convunext_model.stem.trainable = False
    for stage in convunext_model.encoder_stages:
        for block in stage:
            block.trainable = False
    for ds in convunext_model.encoder_downsamples:
        ds.trainable = False

    # Freeze bottleneck
    convunext_model.bottleneck_entry.trainable = False
    for block in convunext_model.bottleneck_blocks:
        block.trainable = False

    # Compile
    convunext_model.compile(
        optimizer=keras.optimizers.Adam(args.finetune_lr_stage1),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    convunext_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.finetune_epochs_stage1,
        batch_size=args.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )

    # ---------------------------------------------
    # Phase 2b: Unfreeze All (Full Fine-tuning)
    # ---------------------------------------------
    logger.info("Phase 2b: Unfreezing all layers...")

    # Unfreeze everything via standard API
    convunext_model.trainable = True
    # Ensure specific blocks are unflagged
    convunext_model.stem.trainable = True
    for stage in convunext_model.encoder_stages:
        for block in stage:
            block.trainable = True

    # Lower Learning Rate for fine-tuning
    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.finetune_lr_stage2,
        "decay_steps": args.finetune_epochs_stage2 * (len(x_train) // args.batch_size),
        "warmup_steps": 200
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "weight_decay": 0.01
    }, lr_schedule)

    convunext_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        # Metric fixed: SparseCategoricalIoU is not standard in all Keras versions
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "best_segmentation.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "segmentation_history.csv")),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = convunext_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.finetune_epochs_stage2,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    return convunext_model, history

# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ConvUNext + MAE Training")

    # Data params
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)

    # Model params
    parser.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base"])

    # MAE params
    parser.add_argument("--mae-epochs", type=int, default=20)
    parser.add_argument("--mae-lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=8) # Smaller patch for 128px
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # Finetune params
    parser.add_argument("--finetune-epochs-stage1", type=int, default=5)
    parser.add_argument("--finetune-epochs-stage2", type=int, default=20)
    parser.add_argument("--finetune-lr-stage1", type=float, default=1e-3)
    parser.add_argument("--finetune-lr-stage2", type=float, default=5e-5)

    # Flags
    parser.add_argument("--skip-mae", action="store_true", help="Skip pretraining")
    parser.add_argument("--analyze", action="store_true", default=True)

    args = parser.parse_args()

    # Setup
    setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"run_{args.variant}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Prepare Data
    num_classes = 3
    x, y = generate_synthetic_data(args.num_samples, args.image_size, args.image_size, num_classes)

    # Split
    split_idx = int(0.8 * len(x))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 2. Instantiate Main Model
    # We create the model with the final segmentation head configuration immediately.
    # The MAE wrapper will simply ignore the decoder/head parts.
    logger.info(f"Instantiating ConvUNext ({args.variant})...")

    convunext_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=num_classes, # Configure for segmentation
        final_activation="softmax",  # Multi-class segmentation
        use_bias=True
    )

    # Build to initialize weights
    dummy_in = keras.ops.zeros((1, args.image_size, args.image_size, 3))
    convunext_model(dummy_in)

    # 3. Stage 1: MAE Pretraining
    if not args.skip_mae:
        # Trains 'convunext_model' weights via the wrapper
        mae_model = run_mae_pretraining(
            convunext_model, x_train, x_val, args, results_dir
        )
    else:
        logger.info("Skipping MAE pretraining...")

    # 4. Stage 2: Segmentation Fine-tuning
    # 'convunext_model' now contains MAE-pretrained encoder weights
    final_model, history = run_segmentation_finetuning(
        convunext_model, (x_train, y_train), (x_val, y_val),
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
