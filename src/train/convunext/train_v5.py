"""ConvUNext MAE Deep Supervision Training Pipeline (ImageNet + COCO)."""

import os
import keras
import tempfile
import argparse
import numpy as np
from keras import ops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List, Union
from dotenv import load_dotenv

from dl_techniques.utils.logger import logger
from dl_techniques.models.convunext.model import ConvUNextModel
from dl_techniques.models.masked_autoencoder import MaskedAutoencoder
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder
from dl_techniques.datasets.vision.coco import COCODatasetBuilder
from dl_techniques.metrics.multi_label_metrics import MultiLabelMetrics
from dl_techniques.losses.multi_labels_loss import create_multilabel_segmentation_loss
from train.common import setup_gpu, create_callbacks as create_common_callbacks


def setup_environment():
    """Configure GPU and load environment variables."""
    load_dotenv()
    setup_gpu()


def get_imagenet_dataset(
    split: str, image_size: int, batch_size: int, data_dir: Optional[str] = None
) -> tf.data.Dataset:
    """Load ImageNet dataset for MAE pretraining."""
    try:
        dataset_name = "imagenet2012"
        ds, info = tfds.load(
            dataset_name, split=split, with_info=True,
            data_dir=data_dir, as_supervised=True
        )
        logger.info(f"Loaded {dataset_name} ({split})")
    except Exception as e:
        logger.warning(f"ImageNet load failed: {e}. Falling back to 'imagenette'.")
        dataset_name = "imagenette/320px-v2"
        ds, info = tfds.load(
            dataset_name, split=split, with_info=True,
            data_dir=data_dir, as_supervised=True
        )

    def preprocess_mae(image, label):
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
    image_size: int, batch_size: int, num_classes: int,
    cache_dir: Optional[str] = None, shuffle_buffer_size: int = 100,
    limit_train_samples: Optional[int] = None, augment_data: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]:
    """Load COCO dataset for multi-label semantic segmentation (80 object channels, no background)."""
    logger.info("Creating COCO multi-label segmentation dataset...")

    builder = COCODatasetBuilder(
        img_size=image_size, batch_size=batch_size, max_boxes_per_image=100,
        cache_dir=cache_dir, use_detection=False, use_segmentation=True,
        segmentation_classes=num_classes, augment_data=augment_data,
        shuffle_buffer_size=shuffle_buffer_size, limit_train_samples=limit_train_samples
    )

    train_ds_raw, val_ds_raw = builder.create_datasets()
    dataset_info = builder.get_dataset_info()

    def adapt_for_multilabel_segmentation(image, targets):
        seg_mask = targets['segmentation']
        seg_mask = tf.cast(seg_mask, tf.float32)
        num_channels = tf.shape(seg_mask)[-1]
        has_background_channel = num_channels > num_classes

        if num_classes == 1:
            mask = tf.cast(seg_mask > 0.5, tf.float32)
        else:
            mask_sum = tf.reduce_sum(seg_mask, axis=-1, keepdims=True)
            is_one_hot = tf.reduce_all(tf.abs(mask_sum - 1.0) < 0.1)

            def process_one_hot():
                foreground_mask = seg_mask[..., 1:] if has_background_channel else seg_mask
                return tf.cast(foreground_mask > 0.5, tf.float32)

            def process_multi_hot():
                if has_background_channel:
                    return tf.cast(seg_mask[..., 1:] > 0.5, tf.float32)
                return tf.cast(seg_mask > 0.5, tf.float32)

            mask = tf.cond(is_one_hot, process_one_hot, process_multi_hot)

        return image, mask

    train_ds = train_ds_raw.map(adapt_for_multilabel_segmentation, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds_raw.map(adapt_for_multilabel_segmentation, num_parallel_calls=tf.data.AUTOTUNE)

    logger.info(f"Multi-label dataset ready: mask shape = (B, H, W, {num_classes})")
    return train_ds, val_ds, dataset_info


@keras.saving.register_keras_serializable()
class ConvUNextWrapper(keras.Model):
    """Wraps ConvUNextModel to act as an MAE encoder."""
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
        self, x: keras.KerasTensor, y: Optional[keras.KerasTensor] = None,
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
                target_full, size=(recon_shape[1], recon_shape[2]), interpolation='bilinear'
            )
            target_resized = ops.cast(target_resized, "float32")
            recon = ops.cast(recon, "float32")

            loss_per_pixel = ops.mean(ops.square(target_resized - recon), axis=-1)

            H, W = recon_shape[1], recon_shape[2]
            h_patch = ops.cast(self.input_shape_config[0], "int32") // self.patch_size
            w_patch = ops.cast(self.input_shape_config[1], "int32") // self.patch_size

            scale_h = ops.cast(H, "float32") / ops.cast(h_patch * self.patch_size, "float32")
            scale_w = ops.cast(W, "float32") / ops.cast(w_patch * self.patch_size, "float32")

            mask_scaled = self._reshape_mask_for_loss_scaled(mask_float, target_resized, scale_h, scale_w)
            mask_scaled = ops.maximum(mask_scaled, self.non_mask_value)

            loss_masked = loss_per_pixel * mask_scaled
            num_masked = ops.sum(mask_float, axis=-1) + 1e-6
            pixels_per_patch = self.patch_size * self.patch_size
            scale_factor = scale_h * scale_w
            adjusted_pixels = ops.cast(pixels_per_patch, "float32") * scale_factor

            loss_sum = ops.sum(loss_masked, axis=[1, 2])
            scale_loss = loss_sum / (num_masked * adjusted_pixels)
            total_loss = total_loss + (weight * ops.mean(scale_loss))

            if i == 0:
                self.loss_tracker_main.update_state(ops.mean(scale_loss))
            else:
                self.loss_tracker_aux.update_state(ops.mean(scale_loss))

        return total_loss

    def _reshape_mask_for_loss_scaled(self, mask, target, scale_h, scale_w):
        B = ops.shape(mask)[0]
        H, W = ops.shape(target)[1], ops.shape(target)[2]
        h_patch = ops.cast(self.input_shape_config[0], "int32") // self.patch_size
        w_patch = ops.cast(self.input_shape_config[1], "int32") // self.patch_size

        mask_grid = ops.reshape(mask, (B, h_patch, w_patch))
        mask_img = ops.image.resize(
            ops.expand_dims(mask_grid, axis=-1), size=(H, W), interpolation='nearest'
        )
        return ops.squeeze(mask_img, axis=-1)

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker, self.loss_tracker_main, self.loss_tracker_aux]


class MAEVisualizationCallback(keras.callbacks.Callback):
    """Visualize MAE reconstructions during training."""
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
        reconstructed = outputs["reconstruction"][0] if isinstance(outputs["reconstruction"], list) else outputs["reconstruction"]

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
    """Visualize multi-label segmentation at multiple scales."""
    def __init__(self, val_batch: Tuple, save_dir: str, num_classes: int,
                 threshold: float = 0.5, frequency: int = 5):
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
            axes[i, 0].imshow(keras.ops.convert_to_numpy(sample_images[i]))
            axes[i, 0].set_title("Input")
            axes[i, 0].axis('off')

            mask_gt = keras.ops.convert_to_numpy(sample_masks[i])
            axes[i, 1].imshow(self._create_composite_mask(mask_gt))
            axes[i, 1].set_title(f"GT ({self._count_active_classes(mask_gt)} classes)")
            axes[i, 1].axis('off')

            for j, pred in enumerate(predictions):
                pred_mask = keras.ops.convert_to_numpy(pred[i])
                pred_binary = (pred_mask >= self.threshold).astype(np.float32)
                axes[i, 2 + j].imshow(self._create_composite_mask(pred_binary))
                axes[i, 2 + j].set_title(f"Pred Scale {j} ({self._count_active_classes(pred_binary)} classes)")
                axes[i, 2 + j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"seg_epoch_{epoch+1:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved segmentation visualization: {save_path}")

    def _create_composite_mask(self, mask: np.ndarray) -> np.ndarray:
        H, W, C = mask.shape
        composite = np.zeros((H, W, 3), dtype=np.float32)
        colors = plt.cm.tab20(np.linspace(0, 1, max(C, 20)))[:, :3]
        for c in range(C):
            class_mask = mask[..., c]
            if class_mask.sum() > 0:
                color = colors[c % len(colors)]
                for ch in range(3):
                    composite[..., ch] += class_mask * color[ch]
        return np.clip(composite, 0, 1)

    def _count_active_classes(self, mask: np.ndarray) -> int:
        return int((mask.sum(axis=(0, 1)) > 0).sum())


class ProgressLoggingCallback(keras.callbacks.Callback):
    """Log training progress per epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_str = f"Epoch {epoch + 1}"
        for key, value in logs.items():
            log_str += f" - {key}: {value:.4f}"
        logger.info(log_str)


def run_mae_pretraining(
    mae_backbone: ConvUNextModel, train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset, args: argparse.Namespace, results_dir: str
) -> None:
    """Run MAE pretraining on the backbone."""
    logger.info("Starting MAE Pretraining...")

    encoder_wrapper = ConvUNextWrapper(mae_backbone)

    if mae_backbone.enable_deep_supervision:
        num_outputs = mae_backbone.depth
        base_weights = [1.0, 0.4, 0.2, 0.1]
        loss_weights = base_weights[:num_outputs]
        logger.info(f"Deep Supervision MAE with {num_outputs} scales, weights: {loss_weights}")
    else:
        loss_weights = None

    mae_model = DeepSupervisionMAE(
        encoder=encoder_wrapper, patch_size=args.patch_size,
        mask_ratio=args.mask_ratio, norm_pix_loss=False,
        input_shape=(args.image_size, args.image_size, 3),
        loss_weights=loss_weights, non_mask_value=0.1
    )

    dummy_in = keras.ops.zeros((1, args.image_size, args.image_size, 3))
    mae_model(dummy_in, training=True)
    mae_model.summary(print_fn=lambda x: logger.info(x))

    lr_schedule = learning_rate_schedule_builder({
        'type': 'cosine_decay', 'learning_rate': args.mae_lr,
        'decay_steps': args.mae_epochs * 1000, 'alpha': 0.0
    })
    optimizer = optimizer_builder({
        'type': 'adamw', 'learning_rate': args.mae_lr,
        'weight_decay': 0.01, 'clipnorm': 1.0
    }, lr_schedule)
    mae_model.compile(optimizer=optimizer)

    viz_batch = next(iter(val_ds.take(1)))
    mae_viz_dir = os.path.join(results_dir, "mae_viz")
    callbacks, _ = create_common_callbacks(
        model_name="ConvUNext-MAE",
        results_dir_prefix=results_dir,
        monitor="val_loss",
        patience=args.patience,
        use_lr_schedule=True,
        include_analyzer=False,
    )
    callbacks.append(MAEVisualizationCallback(viz_batch, mae_viz_dir, frequency=5))
    callbacks.append(ProgressLoggingCallback())

    mae_model.fit(train_ds, validation_data=val_ds, epochs=args.mae_epochs, callbacks=callbacks)
    logger.info("MAE Pretraining complete.")


def run_segmentation_finetuning(
    convunext_model: ConvUNextModel, train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset, args: argparse.Namespace,
    results_dir: str, dataset_info: Dict
) -> keras.Model:
    """Stage 2: Multi-Label Segmentation Fine-tuning with sigmoid activation."""
    logger.info("Starting multi-label segmentation fine-tuning...")

    viz_batch = next(iter(val_ds.take(1)))
    steps_per_epoch = args.steps_per_epoch or 500
    total_steps = args.finetune_epochs * steps_per_epoch

    lr_schedule = learning_rate_schedule_builder(config={
        "type": "cosine_decay", "learning_rate": args.finetune_lr,
        "decay_steps": total_steps, "warmup_steps": 500
    })
    optimizer = optimizer_builder(config={
        "type": "adamw", "weight_decay": 0.05, "gradient_clipping_by_norm": 1.0
    }, lr_schedule=lr_schedule)

    if convunext_model.enable_deep_supervision:
        base_weights = [1.0, 0.4, 0.2, 0.1]
        num_outputs = 1 + (convunext_model.depth - 1)
        loss_weights = base_weights[:num_outputs]

        dice_loss = create_multilabel_segmentation_loss('dice', smooth=1.0)
        loss = [dice_loss] * num_outputs
        metrics = [
            ["binary_accuracy", MultiLabelMetrics(num_classes=args.num_classes, name=f"f1_scale_{i}")]
            for i in range(num_outputs)
        ]

        def multiscale_target_map(image, labels):
            targets = [labels]
            h = tf.shape(image)[1]
            w = tf.shape(image)[2]
            for i in range(num_outputs - 1):
                factor = 2 ** (i + 1)
                m_resized = tf.image.resize(labels, (h // factor, w // factor), method='nearest')
                m_resized = tf.cast(m_resized > 0.5, tf.float32)
                targets.append(m_resized)
            return image, tuple(targets)

        train_ds = train_ds.map(multiscale_target_map, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(multiscale_target_map, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        loss = create_multilabel_segmentation_loss('dice', smooth=1.0)
        loss_weights = None
        metrics = ["binary_accuracy", MultiLabelMetrics(num_classes=args.num_classes, name='f1')]

    convunext_model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)

    viz_dir = os.path.join(results_dir, "seg_viz")
    seg_monitor = "val_f1_scale_0" if convunext_model.enable_deep_supervision else "val_f1"
    callbacks, _ = create_common_callbacks(
        model_name="ConvUNext-Seg",
        results_dir_prefix=results_dir,
        monitor=seg_monitor,
        patience=args.patience,
        use_lr_schedule=True,
        include_analyzer=False,
    )
    callbacks.append(MultiLabelSegmentationCallback(
        val_batch=viz_batch, save_dir=viz_dir,
        num_classes=args.num_classes, threshold=0.5
    ))
    callbacks.append(ProgressLoggingCallback())

    convunext_model.fit(
        train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
        steps_per_epoch=steps_per_epoch, validation_steps=args.validation_steps,
        callbacks=callbacks
    )
    return convunext_model


def main():
    setup_environment()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--variant", type=str, default="tiny")
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--mae-epochs", type=int, default=100)
    parser.add_argument("--mae-lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
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

    # Stage 1: MAE Pretraining
    mae_backbone = ConvUNextModel.from_variant(
        args.variant, input_shape=(args.image_size, args.image_size, 3),
        output_channels=3, include_top=True, enable_deep_supervision=True, use_bias=True
    )
    mae_backbone(dummy_in)

    if not args.skip_mae:
        mae_backbone.summary(print_fn=lambda x: logger.info(x))
        train_ds_mae = get_imagenet_dataset('train', args.image_size, args.batch_size, args.data_dir)
        val_ds_mae = get_imagenet_dataset('validation', args.image_size, args.batch_size, args.data_dir)
        run_mae_pretraining(mae_backbone, train_ds_mae, val_ds_mae, args, results_dir)
    else:
        logger.info("Skipping MAE pretraining...")

    # Stage 2: Multi-Label Segmentation Fine-tuning
    train_ds_seg, val_ds_seg, dataset_info = get_coco_segmentation_dataset(
        image_size=args.image_size, batch_size=args.batch_size,
        num_classes=args.num_classes,
        limit_train_samples=args.steps_per_epoch * args.finetune_epochs if args.steps_per_epoch else None
    )

    seg_model = ConvUNextModel.from_variant(
        args.variant, input_shape=(args.image_size, args.image_size, 3),
        output_channels=args.num_classes, include_top=True,
        enable_deep_supervision=True, final_activation="sigmoid", use_bias=True
    )
    seg_model(dummy_in)
    seg_model.summary(print_fn=lambda x: logger.info(x))

    # Transfer weights from MAE backbone
    if not args.skip_mae:
        logger.info("Transferring weights from MAE to segmentation model...")
        with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as f:
            temp_path = f.name
        try:
            mae_backbone.save_weights(temp_path)
            seg_model.load_weights(temp_path, skip_mismatch=True)
            logger.info("Weight transfer complete.")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    run_segmentation_finetuning(seg_model, train_ds_seg, val_ds_seg, args, results_dir, dataset_info)

    final_path = os.path.join(results_dir, "final_multilabel_seg_model.keras")
    seg_model.save(final_path)
    logger.info(f"Pipeline complete. Saved to {final_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
