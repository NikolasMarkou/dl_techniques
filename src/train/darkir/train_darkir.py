"""
Training Script for DarkIR (Low-Light Image Restoration).

Trains the DarkIR model on paired low/high light image data using
DarkIRCompositeLoss (Charbonnier + SSIM + Perceptual) with PSNR/SSIM metrics.
"""

import os
import argparse
import keras
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt

from train.common import setup_gpu, create_base_argument_parser, create_callbacks

from dl_techniques.utils.logger import logger
from dl_techniques.models.darkir.model import create_darkir_model
from dl_techniques.losses.image_restoration_loss import DarkIRCompositeLoss
from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    PlotConfig,
    PlotStyle,
    TrainingCurvesVisualization,
)
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder


# ---------------------------------------------------------------------

def load_restoration_dataset(
        dataset_path: Optional[str],
        img_size: Tuple[int, int] = (256, 256),
        batch_size: int = 8,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Load paired image restoration dataset (low/high light pairs).

    If dataset_path is None, generates synthetic noisy data from CIFAR-10.
    """
    if dataset_path is None:
        logger.warning("No dataset path provided. Generating SYNTHETIC low-light data using CIFAR-10.")
        return _create_synthetic_dataset(img_size, batch_size)

    logger.info(f"Loading dataset from: {dataset_path}")

    train_low_dir = os.path.join(dataset_path, 'train', 'low')
    train_high_dir = os.path.join(dataset_path, 'train', 'high')
    val_low_dir = os.path.join(dataset_path, 'val', 'low')
    val_high_dir = os.path.join(dataset_path, 'val', 'high')

    def load_paired_paths(low_dir, high_dir):
        low_files = sorted(
            [os.path.join(low_dir, f) for f in os.listdir(low_dir) if f.lower().endswith(('.png', '.jpg'))])
        high_files = sorted(
            [os.path.join(high_dir, f) for f in os.listdir(high_dir) if f.lower().endswith(('.png', '.jpg'))])
        return low_files, high_files

    train_x, train_y = load_paired_paths(train_low_dir, train_high_dir)
    val_x, val_y = load_paired_paths(val_low_dir, val_high_dir)
    logger.info(f"Found {len(train_x)} training pairs and {len(val_x)} validation pairs.")

    def process_path(low_path, high_path):
        low_img = tf.io.read_file(low_path)
        low_img = tf.image.decode_png(low_img, channels=3)
        low_img = tf.cast(tf.image.resize(low_img, img_size), tf.float32) / 255.0

        high_img = tf.io.read_file(high_path)
        high_img = tf.image.decode_png(high_img, channels=3)
        high_img = tf.cast(tf.image.resize(high_img, img_size), tf.float32) / 255.0
        return low_img, high_img

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, len(val_x)


def _create_synthetic_dataset(img_size, batch_size):
    """Create synthetic low-light data from CIFAR-10 for testing."""
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

    x_train = tf.image.resize(x_train, img_size).numpy() / 255.0
    x_test = tf.image.resize(x_test, img_size).numpy() / 255.0

    def darken(img):
        img_dark = img ** 2.5
        noise = np.random.normal(0, 0.02, img.shape)
        return np.clip(img_dark + noise, 0, 1).astype('float32')

    x_train_low = darken(x_train)
    x_test_low = darken(x_test)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train_low, x_train))
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test_low, x_test))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    logger.info(f"Generated synthetic dataset: {len(x_train)} train, {len(x_test)} val samples")
    return train_ds, val_ds, len(x_test)


# ---------------------------------------------------------------------

def create_darkir_config(variant: str) -> Dict[str, Any]:
    """Create DarkIR configuration. Variants: 'medium' (width=32), 'large' (width=64)."""
    config = {
        'img_channels': 3,
        'middle_blk_num_enc': 2,
        'middle_blk_num_dec': 2,
        'enc_blk_nums': [1, 2, 3],
        'dec_blk_nums': [3, 1, 1],
        'dilations': [1, 4, 9],
        'extra_depth_wise': True,
        'use_side_loss': False,
    }

    if variant == 'medium':
        config['width'] = 32
    elif variant == 'large':
        config['width'] = 64
    else:
        logger.warning(f"Unknown variant {variant}, defaulting to medium (width=32)")
        config['width'] = 32

    return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PSNRMetric(keras.metrics.Metric):
    """PSNR metric wrapper."""

    def __init__(self, max_val=1.0, name='psnr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.mean_psnr = keras.metrics.Mean(name='mean_psnr')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean_psnr.update_state(
            tf.image.psnr(y_true, y_pred, max_val=self.max_val), sample_weight)

    def result(self):
        return self.mean_psnr.result()

    def reset_state(self):
        self.mean_psnr.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({'max_val': self.max_val})
        return config


@keras.saving.register_keras_serializable()
class SSIMMetric(keras.metrics.Metric):
    """SSIM metric wrapper."""

    def __init__(self, max_val=1.0, name='ssim', **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.mean_ssim = keras.metrics.Mean(name='mean_ssim')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean_ssim.update_state(
            tf.image.ssim(y_true, y_pred, max_val=self.max_val), sample_weight)

    def result(self):
        return self.mean_ssim.result()

    def reset_state(self):
        self.mean_ssim.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({'max_val': self.max_val})
        return config


# ---------------------------------------------------------------------

def visualize_restoration_results(
        model: keras.Model,
        val_ds: tf.data.Dataset,
        results_dir: str,
        num_samples: int = 4,
):
    """Generate comparison plots: Input vs Prediction vs Ground Truth."""
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    for inputs, targets in val_ds.take(1):
        preds = np.clip(model.predict(inputs, verbose=0), 0.0, 1.0)

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        plt.suptitle("DarkIR Restoration Results (Low Light | Restored | Ground Truth)")

        for i in range(min(num_samples, len(inputs))):
            axes[i, 0].imshow(inputs[i])
            axes[i, 0].set_title("Input (Low)")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(preds[i])
            axes[i, 1].set_title("DarkIR Prediction")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(targets[i])
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(viz_dir, "restoration_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved restoration visualization to {save_path}")


# ---------------------------------------------------------------------

def run_model_analysis(
        model: keras.Model,
        training_history: keras.callbacks.History,
        model_name: str,
        results_dir: str,
):
    """Run weight distribution and training dynamics analysis."""
    logger.info("Running model analysis...")
    try:
        analysis_config = AnalysisConfig(
            analyze_weights=True,
            analyze_training_dynamics=True,
            analyze_calibration=False,
            save_plots=True,
            save_format='png',
        )
        analysis_dir = os.path.join(results_dir, "model_analysis")
        analyzer = ModelAnalyzer(
            models={model_name: model},
            training_history={model_name: training_history.history},
            config=analysis_config,
            output_dir=analysis_dir,
        )
        analyzer.analyze(data=None)
        logger.info(f"Model analysis saved to {analysis_dir}")
    except Exception as e:
        logger.warning(f"Model analysis failed: {e}")


# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace):
    """Main training loop."""
    logger.info("Starting DarkIR Training...")
    setup_gpu(gpu_id=args.gpu)

    # Data
    train_ds, val_ds, val_samples = load_restoration_dataset(
        args.dataset_path,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    # Model
    model_config = create_darkir_config(args.variant)
    logger.info(f"Creating DarkIR model (Variant: {args.variant})")
    model = create_darkir_model(**model_config)

    # Optimization
    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "warmup_steps": args.warmup_steps,
        "warmup_start_lr": 1e-7,
        "learning_rate": args.learning_rate,
        "decay_steps": args.epochs * (1000 if args.dataset_path is None else 500),
        "alpha": 1e-6,
    })
    optimizer = optimizer_builder({
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "weight_decay": args.weight_decay,
        "gradient_clipping_by_norm": 1.0,
    }, lr_schedule)

    # Loss & Metrics
    loss_fn = DarkIRCompositeLoss(
        charbonnier_weight=1.0,
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight,
        name='darkir_loss',
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[PSNRMetric(max_val=1.0), SSIMMetric(max_val=1.0)],
    )
    model.build((None, args.img_size, args.img_size, 3))
    model.summary(print_fn=logger.info)

    # Callbacks
    callbacks, results_dir = create_callbacks(
        model_name=f"darkir_{args.variant}",
        results_dir_prefix="darkir",
        monitor='val_psnr',
        patience=args.patience,
        use_lr_schedule=True,
    )
    callbacks.append(
        keras.callbacks.BackupAndRestore(os.path.join(results_dir, 'backup')),
    )

    # Visualization Manager
    viz_manager = VisualizationManager(
        experiment_name=f"darkir_{args.variant}",
        output_dir=os.path.join(results_dir, "visualizations"),
        config=PlotConfig(style=PlotStyle.PUBLICATION),
    )
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)

    # Train
    logger.info("Starting Fit...")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks, verbose=1,
    )

    # Post-Training
    logger.info("Training complete. Generating visualizations...")
    visualize_restoration_results(model, val_ds, results_dir)

    history_viz = TrainingHistory(
        epochs=list(range(len(history.history['loss']))),
        train_loss=history.history['loss'],
        val_loss=history.history['val_loss'],
        train_metrics={k: v for k, v in history.history.items()
                       if k not in ['loss', 'val_loss'] and not k.startswith('val_')},
        val_metrics={k.replace('val_', ''): v for k, v in history.history.items()
                     if k.startswith('val_') and k != 'val_loss'},
    )
    viz_manager.visualize(data=history_viz, plugin_name="training_curves", show=False)

    run_model_analysis(model, history, f"darkir_{args.variant}", results_dir)

    final_path = os.path.join(results_dir, "final_model.keras")
    model.save(final_path)
    logger.info(f"Done. Model saved to {final_path}")


# ---------------------------------------------------------------------

def main():
    parser = create_base_argument_parser(description="Train DarkIR for Image Restoration")

    # DarkIR-specific args
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to paired dataset (train/val with low/high subfolders). If None, uses synthetic.')
    parser.add_argument('--img-size', type=int, default=128, help='Input image resolution')
    parser.add_argument('--variant', type=str, default='medium', choices=['medium', 'large'], help='DarkIR variant')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='LR warmup steps')
    parser.add_argument('--ssim-weight', type=float, default=0.2, help='Weight for SSIM loss component')
    parser.add_argument('--perceptual-weight', type=float, default=0.01, help='Weight for VGG perceptual loss')

    # Override base defaults for restoration task
    parser.set_defaults(
        batch_size=8,
        learning_rate=2e-4,
        patience=20,
    )

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
