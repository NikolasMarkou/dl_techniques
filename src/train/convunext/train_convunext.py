"""
ConvUNext MAE Segmentation Training Pipeline

Two-stage training workflow:
1. Self-supervised pretraining via Masked Autoencoders (MAE)
2. Supervised fine-tuning for semantic segmentation

Uses a shared-weight wrapper pattern: ConvUNextModel is instantiated once,
and ConvUNextEncoderWrapper exposes only the encoder/bottleneck path to MAE.
Training the MAE updates the shared layer instances in the main model.
"""

import os
import math
import keras
import argparse
import numpy as np
import tensorflow as tf
from typing import Tuple
from datetime import datetime

from train.common import setup_gpu
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

def generate_synthetic_data(
    num_samples: int, height: int, width: int, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic geometric shapes for segmentation testing."""
    logger.info(f"Generating {num_samples} synthetic samples ({height}x{width})...")

    x_data = np.zeros((num_samples, height, width, 3), dtype="float32")
    y_data = np.zeros((num_samples, height, width), dtype="int32")

    for i in range(num_samples):
        x_data[i] = np.random.rand(height, width, 3) * 0.1

        # Circle (Class 1)
        cy, cx = np.random.randint(0, height), np.random.randint(0, width)
        cr = np.random.randint(10, height // 4)
        y, x = np.ogrid[:height, :width]
        mask_circle = ((y - cy) ** 2 + (x - cx) ** 2) <= cr ** 2
        x_data[i][mask_circle] += [0.5, 0.0, 0.0]
        y_data[i][mask_circle] = 1

        # Rectangle (Class 2)
        if num_classes > 2:
            ry, rx = np.random.randint(0, height - 20), np.random.randint(0, width - 20)
            rh, rw = np.random.randint(10, 50), np.random.randint(10, 50)
            mask_rect = np.zeros((height, width), dtype=bool)
            mask_rect[ry:ry + rh, rx:rx + rw] = True
            x_data[i][mask_rect] = np.random.rand(np.sum(mask_rect), 3) * 0.2 + [0.0, 0.5, 0.0]
            y_data[i][mask_rect] = 2

    return np.clip(x_data, 0, 1), y_data


# ---------------------------------------------------------------------
# Encoder Wrapper
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextEncoderWrapper(keras.Model):
    """Wraps the encoder path of ConvUNextModel for MAE compatibility.

    Shares exact layer instances with the provided ConvUNextModel so that
    MAE training updates the main model's weights automatically.
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
        x = self.stem(inputs, training=training)

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
        return (batch_size, new_h, new_w, self.filter_sizes[self.depth])

    def get_config(self):
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
    """Stage 1: Pretrain the encoder using Masked Autoencoding."""
    logger.info("=" * 60)
    logger.info("STAGE 1: MAE SELF-SUPERVISED PRETRAINING")
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

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.mae_lr,
        "decay_steps": args.mae_epochs * (len(train_data) // args.batch_size),
        "warmup_steps": 500
    })

    optimizer = optimizer_builder({
        "type": "adamw", "weight_decay": 0.05, "gradient_clipping_by_norm": 1.0
    }, lr_schedule)

    mae.compile(optimizer=optimizer)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "mae_weights.keras"),
            save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "mae_history.csv"))
    ]

    mae.fit(
        train_data, epochs=args.mae_epochs, batch_size=args.batch_size,
        validation_data=(val_data, val_data), callbacks=callbacks
    )

    # Visualize reconstructions
    viz_dir = os.path.join(results_dir, "mae_viz")
    os.makedirs(viz_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    grid = visualize_reconstruction(mae, val_data, num_samples=4)
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
    """Stage 2: Fine-tune the full U-Net for segmentation.

    Phase 2a: Freeze encoder, train decoder.
    Phase 2b: Unfreeze all, full fine-tuning with lower LR.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: SEGMENTATION FINE-TUNING")
    logger.info("=" * 60)

    x_train, y_train = train_data
    x_val, y_val = val_data

    # Phase 2a: Freeze encoder
    logger.info("Phase 2a: Freezing encoder, training decoder...")
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
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    convunext_model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=args.finetune_epochs_stage1, batch_size=args.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )

    # Phase 2b: Unfreeze all
    logger.info("Phase 2b: Unfreezing all layers...")
    convunext_model.trainable = True
    convunext_model.stem.trainable = True
    for stage in convunext_model.encoder_stages:
        for block in stage:
            block.trainable = True

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.finetune_lr_stage2,
        "decay_steps": args.finetune_epochs_stage2 * (len(x_train) // args.batch_size),
        "warmup_steps": 200
    })

    optimizer = optimizer_builder({"type": "adamw", "weight_decay": 0.01}, lr_schedule)

    convunext_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "best_segmentation.keras"),
            save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "segmentation_history.csv")),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = convunext_model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=args.finetune_epochs_stage2, batch_size=args.batch_size,
        callbacks=callbacks
    )

    return convunext_model, history


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ConvUNext + MAE Training")

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base"])

    parser.add_argument("--mae-epochs", type=int, default=20)
    parser.add_argument("--mae-lr", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    parser.add_argument("--finetune-epochs-stage1", type=int, default=5)
    parser.add_argument("--finetune-epochs-stage2", type=int, default=20)
    parser.add_argument("--finetune-lr-stage1", type=float, default=1e-3)
    parser.add_argument("--finetune-lr-stage2", type=float, default=5e-5)

    parser.add_argument("--skip-mae", action="store_true", help="Skip pretraining")
    parser.add_argument("--analyze", action="store_true", default=True)

    args = parser.parse_args()

    setup_gpu()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"run_{args.variant}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Prepare data
    num_classes = 3
    x, y = generate_synthetic_data(args.num_samples, args.image_size, args.image_size, num_classes)
    split_idx = int(0.8 * len(x))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create model
    logger.info(f"Instantiating ConvUNext ({args.variant})...")
    convunext_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=num_classes,
        final_activation="softmax",
        use_bias=True
    )

    dummy_in = keras.ops.zeros((1, args.image_size, args.image_size, 3))
    convunext_model(dummy_in)

    # Stage 1: MAE Pretraining
    if not args.skip_mae:
        run_mae_pretraining(convunext_model, x_train, x_val, args, results_dir)
    else:
        logger.info("Skipping MAE pretraining...")

    # Stage 2: Segmentation Fine-tuning
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
