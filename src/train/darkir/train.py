"""
Training Script for DarkIR (Low-Light Image Restoration).

This script trains the DarkIR model using the dl_techniques framework.
It includes:
1. Data loading for Image Restoration (Paired Low/High light images).
2. Custom DarkIR configuration options.
3. Integration of Image Restoration Losses (Charbonnier, Perceptual, etc.).
4. Visualization of restoration results (Input vs Output vs GT).
5. Comprehensive model analysis.
"""

import os
import argparse
import keras
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# dl_techniques imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from dl_techniques.models.darkir.model import create_darkir_model

# Import the loss functions provided in the context
from dl_techniques.losses.image_restoration_loss import (
    DarkIRCompositeLoss,
)

from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    PlotConfig,
    PlotStyle,
    TrainingCurvesVisualization
)
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder


# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------

def load_restoration_dataset(
        dataset_path: Optional[str],
        img_size: Tuple[int, int] = (256, 256),
        batch_size: int = 8
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """
    Load image restoration dataset (Paired).

    If dataset_path is None, generates synthetic noisy data based on CIFAR-10
    for testing the pipeline.
    """
    if dataset_path is None:
        logger.warning("No dataset path provided. Generating SYNTHETIC low-light data using CIFAR-10.")
        return _create_synthetic_dataset(img_size, batch_size)

    logger.info(f"Loading dataset from: {dataset_path}")

    # Expected structure:
    # dataset_path/
    #   train/
    #     low/ (input images)
    #     high/ (ground truth)
    #   val/
    #     low/
    #     high/

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
        # Load Low
        low_img = tf.io.read_file(low_path)
        low_img = tf.image.decode_png(low_img, channels=3)
        low_img = tf.image.resize(low_img, img_size)
        low_img = tf.cast(low_img, tf.float32) / 255.0

        # Load High
        high_img = tf.io.read_file(high_path)
        high_img = tf.image.decode_png(high_img, channels=3)
        high_img = tf.image.resize(high_img, img_size)
        high_img = tf.cast(high_img, tf.float32) / 255.0

        return low_img, high_img

    # Create TF Datasets
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

    # Resize to target size
    x_train = tf.image.resize(x_train, img_size).numpy() / 255.0
    x_test = tf.image.resize(x_test, img_size).numpy() / 255.0

    # Simulate low light: gamma correction + noise
    def darken(img):
        img_dark = img ** 2.5  # Gamma correction to darken
        noise = np.random.normal(0, 0.02, img.shape)  # Add noise
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
    """
    Create DarkIR configuration based on variant.
    Variants based on original paper: 'medium' (DarkIR-m) and 'large' (DarkIR-l).
    """
    config = {
        'img_channels': 3,
        'middle_blk_num_enc': 2,
        'middle_blk_num_dec': 2,
        'enc_blk_nums': [1, 2, 3],
        'dec_blk_nums': [3, 1, 1],
        'dilations': [1, 4, 9],
        'extra_depth_wise': True,
        'use_side_loss': False  # Simplify training loop by disabling deep supervision by default
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
    """Custom PSNR Metric wrapper for better logging."""

    def __init__(self, max_val=1.0, name='psnr', **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.mean_psnr = keras.metrics.Mean(name='mean_psnr')

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr = tf.image.psnr(y_true, y_pred, max_val=self.max_val)
        self.mean_psnr.update_state(psnr, sample_weight)

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
    """Custom SSIM Metric wrapper."""

    def __init__(self, max_val=1.0, name='ssim', **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.mean_ssim = keras.metrics.Mean(name='mean_ssim')

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim(y_true, y_pred, max_val=self.max_val)
        self.mean_ssim.update_state(ssim, sample_weight)

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
        num_samples: int = 4
):
    """
    Generate comparison plots (Input vs Prediction vs GT).
    Unlike classification, we need to see the actual pixels.
    """
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Get a batch
    for inputs, targets in val_ds.take(1):
        preds = model.predict(inputs, verbose=0)

        # Clip values for display
        preds = np.clip(preds, 0.0, 1.0)

        # Plot
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        plt.suptitle("DarkIR Restoration Results (Low Light | Restored | Ground Truth)")

        for i in range(min(num_samples, len(inputs))):
            # Input (Low Light)
            axes[i, 0].imshow(inputs[i])
            axes[i, 0].set_title("Input (Low)")
            axes[i, 0].axis('off')

            # Prediction
            axes[i, 1].imshow(preds[i])
            axes[i, 1].set_title("DarkIR Prediction")
            axes[i, 1].axis('off')

            # Ground Truth
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
        results_dir: str
):
    """Run model analysis focusing on weight distributions and training dynamics."""
    logger.info("Running model analysis...")
    try:
        analysis_config = AnalysisConfig(
            analyze_weights=True,
            analyze_training_dynamics=True,
            analyze_calibration=False,  # Not relevant for regression
            save_plots=True,
            save_format='png'
        )

        analysis_dir = os.path.join(results_dir, "model_analysis")

        analyzer = ModelAnalyzer(
            models={model_name: model},
            training_history={model_name: training_history.history},
            config=analysis_config,
            output_dir=analysis_dir
        )

        # Passing None for data because regression analysis isn't fully supported
        # by the generic analyzer yet, but weight analysis works without data.
        analyzer.analyze(data=None)
        logger.info(f"Model analysis saved to {analysis_dir}")

    except Exception as e:
        logger.warning(f"Model analysis failed (likely due to regression task nature): {e}")


# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace):
    """Main training loop."""
    logger.info("Starting DarkIR Training...")
    setup_gpu()

    # 1. Load Data
    train_ds, val_ds, val_samples = load_restoration_dataset(
        args.dataset_path,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )

    # 2. Configure Model
    model_config = create_darkir_config(args.variant)

    logger.info(f"Creating DarkIR model (Variant: {args.variant})")
    model = create_darkir_model(**model_config)

    # 3. Setup Optimization
    # Cosine Decay is standard for DarkIR
    lr_config = {
        "type": "cosine_decay",
        "warmup_steps": args.warmup_steps,
        "warmup_start_lr": 1e-7,
        "learning_rate": args.learning_rate,
        "decay_steps": args.epochs * (1000 if args.dataset_path is None else 500),  # Approximate steps
        "alpha": 1e-6
    }
    lr_schedule = learning_rate_schedule_builder(lr_config)

    optimizer_config = {
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "weight_decay": args.weight_decay,
        "gradient_clipping_by_norm": 1.0
    }
    optimizer = optimizer_builder(optimizer_config, lr_schedule)

    # 4. Setup Loss & Metrics
    # Using composite loss from image_restoration_loss.py
    loss_fn = DarkIRCompositeLoss(
        charbonnier_weight=1.0,
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight,
        name='darkir_loss'
    )

    metrics = [
        PSNRMetric(max_val=1.0),
        SSIMMetric(max_val=1.0)
    ]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # Build & Summary
    model.build((None, args.img_size, args.img_size, 3))
    model.summary(print_fn=logger.info)

    # 5. Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"darkir_{args.variant}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(results_dir, 'best_model.keras'),
            monitor='val_psnr',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(os.path.join(results_dir, 'log.csv')),
        keras.callbacks.EarlyStopping(
            monitor='val_psnr',
            patience=args.patience,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.BackupAndRestore(os.path.join(results_dir, 'backup'))
    ]

    # 6. Visualization Manager (For Training Curves)
    viz_manager = VisualizationManager(
        experiment_name=f"darkir_{args.variant}",
        output_dir=os.path.join(results_dir, "visualizations"),
        config=PlotConfig(style=PlotStyle.PUBLICATION)
    )
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)

    # 7. Training
    logger.info("Starting Fit...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Post-Training Evaluation & Visualization
    logger.info("Training complete. Generating visualizations...")

    # Visualizing Predictions (Images)
    visualize_restoration_results(model, val_ds, results_dir)

    # Visualizing Curves
    history_viz = TrainingHistory(
        epochs=list(range(len(history.history['loss']))),
        train_loss=history.history['loss'],
        val_loss=history.history['val_loss'],
        train_metrics={k: v for k, v in history.history.items() if
                       k not in ['loss', 'val_loss'] and not k.startswith('val_')},
        val_metrics={k.replace('val_', ''): v for k, v in history.history.items() if
                     k.startswith('val_') and k != 'val_loss'}
    )

    viz_manager.visualize(data=history_viz, plugin_name="training_curves", show=False)

    # 9. Analysis
    run_model_analysis(model, history, f"darkir_{args.variant}", results_dir)

    # 10. Save Final Model
    final_path = os.path.join(results_dir, "final_model.keras")
    model.save(final_path)
    logger.info(f"Done. Model saved to {final_path}")


# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DarkIR for Image Restoration")

    # Dataset
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to dataset (containing train/val subfolders). If None, uses synthetic data.')
    parser.add_argument('--img-size', type=int, default=128, help='Input image resolution')

    # Model
    parser.add_argument('--variant', type=str, default='medium', choices=['medium', 'large'], help='DarkIR Variant')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=20)

    # Loss Weights
    parser.add_argument('--ssim-weight', type=float, default=0.2, help='Weight for SSIM loss component')
    parser.add_argument('--perceptual-weight', type=float, default=0.01, help='Weight for VGG perceptual loss')

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()