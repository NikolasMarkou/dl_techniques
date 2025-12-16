"""
ConvUNext MAE Deep Supervision Training Pipeline (ImageNet + COCO)
==================================================================

Complete training pipeline implementing:
1. **Multi-Scale MAE Pretraining**: Self-supervised learning calculating
   reconstruction loss at every decoder resolution (Deep Supervision).
2. **Robust Weight Transfer**: Correctly handles nested Keras layer structures.
3. **Segmentation Fine-tuning**: Transfers knowledge to COCO segmentation
   with automatic multi-scale target generation.

Key Features:
- Deep Supervision (Multi-scale Loss & Targets)
- Zero-Dropout Modern Architecture
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
    """Load COCO dataset for semantic segmentation."""
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
        """Adapts dictionary targets to mask tensors. Handles batched inputs."""
        seg_mask = targets['segmentation']

        # Check rank to handle both batched and unbatched (just in case)
        # Usually builder returns batched data

        if num_classes == 1:
            mask = tf.cast(seg_mask > 0.5, tf.int32)
        else:
            mask = tf.argmax(seg_mask, axis=-1, output_type=tf.int32)
            mask = tf.expand_dims(mask, axis=-1)

        return image, mask

    train_ds = train_ds_raw.map(adapt_for_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds_raw.map(adapt_for_segmentation, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, val_ds, dataset_info


# ---------------------------------------------------------------------
# MAE Components (Deep Supervision Support)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConvUNextWrapper(keras.Model):
    """
    Wraps ConvUNextModel to act as an MAE 'Encoder'.
    Passes through list outputs for deep supervision.
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
    """
    Pass-through decoder.

    Since ConvUNextModel (with output_channels=3) already projects feature maps
    to RGB at every scale via its internal deep supervision heads, the MAE
    decoder doesn't need to do any work. It just passes the list of tensors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        # inputs is [Main_Output, Aux_1, Aux_2...]
        return inputs


@keras.saving.register_keras_serializable()
class DeepSupervisionMAE(MaskedAutoencoder):
    """
    Masked Autoencoder that calculates loss across multiple scales.

    Expects the encoder to return a list of tensors [main, aux1, aux2...].
    """
    def __init__(
        self,
        loss_weights: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_weights = loss_weights

        self.loss_tracker_main = keras.metrics.Mean(name="loss_main")
        self.loss_tracker_aux = keras.metrics.Mean(name="loss_aux_avg")

    def _create_decoder(self):
        # ConvUNextModel does the projection, so we use Identity
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

        # 1. Normalize Inputs
        target_full = ops.cast(x, "float32")
        mask_float = ops.cast(mask, "float32")

        # 2. Handle List vs Single Output
        if not isinstance(reconstructions, list):
            reconstructions = [reconstructions]

        # 3. Initialize Weights
        if self.loss_weights is None:
            weights = [1.0 / (2**i) for i in range(len(reconstructions))]
        elif len(self.loss_weights) != len(reconstructions):
            # Default fallback if weights don't match
            weights = [1.0] * len(reconstructions)
        else:
            weights = self.loss_weights

        total_loss = 0.0

        # 4. Iterate scales (Main is index 0)
        for i, pred in enumerate(reconstructions):
            pred = ops.cast(pred, "float32")
            pred_shape = ops.shape(pred)
            h_pred, w_pred = pred_shape[1], pred_shape[2]

            # Resize Target (Bilinear for RGB)
            target_resized = ops.image.resize(
                target_full,
                size=(h_pred, w_pred),
                interpolation="bilinear"
            )

            # Resize Mask (Nearest for Binary)
            # a. Get (B, H, W) mask from patches
            mask_img_full = self._reshape_mask_for_loss(mask_float, target_full)
            # b. Expand to (B, H, W, 1) so resize treats it as an image channel
            mask_img_full = ops.expand_dims(mask_img_full, axis=-1)
            # c. Resize to current scale
            mask_resized = ops.image.resize(
                mask_img_full,
                size=(h_pred, w_pred),
                interpolation="nearest"
            )
            # mask_resized is (B, h_pred, w_pred, 1)

            mask_resized = ops.maximum(mask_resized, self.non_mask_value)

            # MSE
            diff = ops.square(target_resized - pred)

            # Keep dims to match mask shape (B, H, W, 1) during channel averaging
            mse_pixel = ops.mean(diff, axis=-1, keepdims=True)

            # Apply Mask
            masked_loss = mse_pixel * mask_resized

            # Normalize by mask sum
            # Sum over spatial [1, 2] AND channel [3] dimensions
            loss_sum = ops.sum(masked_loss, axis=[1, 2, 3])
            mask_sum = ops.sum(mask_resized, axis=[1, 2, 3]) + 1e-6

            scale_loss = ops.mean(loss_sum / mask_sum)

            total_loss += weights[i] * scale_loss

            # Metrics
            if i == 0:
                self.loss_tracker_main.update_state(scale_loss)
            elif i == 1:
                self.loss_tracker_aux.update_state(scale_loss)

        return total_loss

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.loss_tracker_main,
            self.loss_tracker_aux
        ]

    def visualize(self, image, return_arrays=True):
        """Handle list output for visualization (take main output)."""
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image

        outputs = self(image_batch, training=True)
        recon_list = outputs["reconstruction"]

        if isinstance(recon_list, list):
            recon_t = recon_list[0]
        else:
            recon_t = recon_list

        if return_arrays:
            return (
                ops.convert_to_numpy(image_batch[0]),
                ops.convert_to_numpy(outputs["masked_input"][0]),
                ops.convert_to_numpy(recon_t[0])
            )
        return image_batch, outputs["masked_input"], recon_t


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------

class ImageReconstructionCallback(keras.callbacks.Callback):
    def __init__(self, val_images, save_dir, num_samples=4):
        super().__init__()
        self.val_images = val_images
        self.save_dir = save_dir
        self.num_samples = min(num_samples, len(val_images))
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        viz_data = self.val_images[:self.num_samples]
        grid = visualize_reconstruction(self.model, viz_data, num_samples=self.num_samples)

        plt.figure(figsize=(12, 8))
        plt.imshow(grid)
        plt.axis('off')
        plt.title(f"MAE Reconstruction - Epoch {epoch + 1}")
        plt.savefig(os.path.join(self.save_dir, f"epoch_{epoch + 1:03d}.png"))
        plt.close()


class ProgressLoggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch+1}"
        for k, v in logs.items():
            msg += f" - {k}: {v:.4f}"
        logger.info(msg)


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
    """Stage 1: Multi-Scale MAE Pretraining."""
    logger.info("=" * 60)
    logger.info("STAGE 1: MAE PRETRAINING (Deep Supervision Enabled)")
    logger.info("=" * 60)

    encoder_wrapper = ConvUNextWrapper(convunext_model)

    # Weights: Main (1.0), Aux1 (0.5), Aux2 (0.25)...
    loss_weights = [1.0, 0.5, 0.25, 0.125]

    mae = DeepSupervisionMAE(
        encoder=encoder_wrapper,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        input_shape=(args.image_size, args.image_size, 3),
        loss_weights=loss_weights,
        decoder_dims=[] # Not used by IdentityDecoder
    )

    total_steps = args.mae_epochs * (int(train_ds.cardinality()) if int(train_ds.cardinality()) > 0 else 1000)

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.mae_lr,
        "decay_steps": total_steps,
        "warmup_steps": int(total_steps * 0.05)
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "weight_decay": 0.05,
        "gradient_clipping_by_norm": 1.0,
        "beta_1": 0.9,
        "beta_2": 0.95
    }, lr_schedule)

    mae.compile(optimizer=optimizer)

    viz_dir = os.path.join(results_dir, "mae_viz")
    val_images = next(iter(val_ds))[0].numpy()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "mae_weights.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "mae_history.csv")),
        ImageReconstructionCallback(val_images, viz_dir),
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
    args: argparse.Namespace,
    results_dir: str,
    dataset_info: Dict
) -> keras.Model:
    """Stage 2: Segmentation Fine-tuning."""
    logger.info("=" * 60)
    logger.info(f"STAGE 2: SEGMENTATION FINE-TUNING")
    logger.info("=" * 60)

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

    # Configure loss for Deep Supervision
    if convunext_model.enable_deep_supervision:
        # Weights for [Main, Aux1, Aux2...]
        base_weights = [1.0, 0.4, 0.2, 0.1]
        num_outputs = 1 + (convunext_model.depth - 1)
        loss_weights = base_weights[:num_outputs]
        loss = ["sparse_categorical_crossentropy"] * num_outputs

        # CRITICAL FIX: Metrics must also match output length for Keras 3 strictness
        metrics = ["accuracy"] * num_outputs

        logger.info(f"Deep Supervision enabled: Preparing multi-scale targets for {num_outputs} heads...")

        def multiscale_target_map(image, mask):
            """
            Maps (B,H,W,C), (B,H,W,1) -> (B,H,W,C), [Target1, Target2...]
            """
            targets = [mask]

            # Shape for batched input: [Batch, Height, Width, Channels]
            h = tf.shape(image)[1] # Height is index 1
            w = tf.shape(image)[2] # Width is index 2

            for i in range(num_outputs - 1):
                # Scale factor: 2**(i+1) means 2, 4, 8...
                factor = 2 ** (i + 1)

                # Resize mask (Nearest Neighbor)
                m_resized = tf.image.resize(
                    mask,
                    (h // factor, w // factor),
                    method='nearest'
                )
                targets.append(m_resized)

            return image, tuple(targets)

        # Apply mapping
        train_ds = train_ds.map(multiscale_target_map, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(multiscale_target_map, num_parallel_calls=tf.data.AUTOTUNE)

    else:
        loss = "sparse_categorical_crossentropy"
        loss_weights = None
        metrics = ["accuracy"]

    convunext_model.compile(
        optimizer=optimizer,
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, "best_seg_model.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, "seg_history.csv")),
        keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True),
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


def transfer_model_weights(
    source_model: ConvUNextModel,
    target_model: ConvUNextModel
) -> None:
    """
    Transfer weights from source to target model, excluding the final heads.
    Correctly handles nested list structures of ConvUNext blocks.
    """
    logger.info("Transferring pretrained weights (Backbone + Decoder)...")

    # 1. Stem (Layer)
    target_model.stem.set_weights(source_model.stem.get_weights())

    # 2. Encoder Stages (List[List[Layer]])
    for src_stage, tgt_stage in zip(source_model.encoder_stages, target_model.encoder_stages):
        # Iterate blocks in stage
        for src_block, tgt_block in zip(src_stage, tgt_stage):
            tgt_block.set_weights(src_block.get_weights())

    # 2b. Encoder Downsamples (List[Layer])
    # Note: len(downsamples) is depth-1, so checking index safe
    for i, (src_ds, tgt_ds) in enumerate(zip(source_model.encoder_downsamples, target_model.encoder_downsamples)):
        tgt_ds.set_weights(src_ds.get_weights())

    # 3. Bottleneck
    target_model.bottleneck_entry.set_weights(source_model.bottleneck_entry.get_weights())

    for src_block, tgt_block in zip(source_model.bottleneck_blocks, target_model.bottleneck_blocks):
        tgt_block.set_weights(src_block.get_weights())

    # 4. Decoder
    # Decoder Blocks is List[List[Layer]]
    # Decoder Upsamples is List[Layer]
    for i in range(len(source_model.decoder_blocks)):
        # Upsample
        target_model.decoder_upsamples[i].set_weights(
            source_model.decoder_upsamples[i].get_weights()
        )

        # Blocks (List)
        src_stage = source_model.decoder_blocks[i]
        tgt_stage = target_model.decoder_blocks[i]

        for src_block, tgt_block in zip(src_stage, tgt_stage):
            tgt_block.set_weights(src_block.get_weights())

    # NOTE: Deep Supervision Heads and Final Output Layer are intentionally SKIPPED.
    # MAE Heads: Output 3 channels (RGB)
    # Seg Heads: Output N channels (Classes)
    # They are incompatible in shape.

    logger.info("Weight transfer complete.")


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------

def main():
    setup_environment()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)

    # Model
    parser.add_argument("--variant", type=str, default="tiny")
    parser.add_argument("--num-classes", type=int, default=80)

    # MAE
    parser.add_argument("--mae-epochs", type=int, default=10)
    parser.add_argument("--mae-lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # Fine-tuning
    parser.add_argument("--finetune-epochs", type=int, default=10)
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
    results_dir = os.path.join("results", f"convunext_ds_{args.variant}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    dummy_in = ops.zeros((1, args.image_size, args.image_size, 3))

    # -----------------------------------------------------------
    # 1. Instantiate MAE Model (Outputs RGB, Deep Sup Enabled)
    # -----------------------------------------------------------
    logger.info(f"Instantiating MAE Backbone ({args.variant}) - Outputs RGB...")
    mae_backbone = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=3,
        enable_deep_supervision=True, # Critical
        final_activation="sigmoid",   # 0-1 for pixels
        use_bias=True
    )
    mae_backbone(dummy_in) # Build

    # -----------------------------------------------------------
    # 2. Stage 1: MAE Pretraining
    # -----------------------------------------------------------
    if not args.skip_mae:
        train_ds_mae = get_imagenet_dataset('train', args.image_size, args.batch_size, args.data_dir)
        val_ds_mae = get_imagenet_dataset('validation', args.image_size, args.batch_size, args.data_dir)

        run_mae_pretraining(mae_backbone, train_ds_mae, val_ds_mae, args, results_dir)
    else:
        logger.info("Skipping MAE pretraining...")

    # -----------------------------------------------------------
    # 3. Stage 2: Segmentation Fine-tuning
    # -----------------------------------------------------------
    train_ds_seg, val_ds_seg, dataset_info = get_coco_segmentation_dataset(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        limit_train_samples=args.steps_per_epoch * args.finetune_epochs if args.steps_per_epoch else None
    )

    logger.info(f"Creating Segmentation Model ({args.num_classes} classes)...")

    # Target model also has Deep Supervision enabled (architecture match)
    seg_model = ConvUNextModel.from_variant(
        args.variant,
        input_shape=(args.image_size, args.image_size, 3),
        output_channels=args.num_classes,
        enable_deep_supervision=True,
        final_activation="softmax" if args.num_classes > 1 else "sigmoid",
        use_bias=True
    )
    seg_model(dummy_in) # Build

    # Transfer Weights (Manually handling lists)
    if not args.skip_mae:
        transfer_model_weights(mae_backbone, seg_model)

    run_segmentation_finetuning(
        seg_model,
        train_ds_seg,
        val_ds_seg,
        args,
        results_dir,
        dataset_info
    )

    final_path = os.path.join(results_dir, "final_seg_model.keras")
    seg_model.save(final_path)
    logger.info(f"Pipeline complete. Saved to {final_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)