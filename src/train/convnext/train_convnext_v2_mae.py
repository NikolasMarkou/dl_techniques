"""
ConvNeXt V2 Training with MAE Pretraining and Comprehensive Analysis
====================================================================

Two-stage training approach with deep model analysis:
1. Self-supervised MAE pretraining on unlabeled images (using ConvNeXtV2 encoder)
2. Supervised fine-tuning on labeled classification task
3. Comprehensive model analysis using ModelAnalyzer

This script demonstrates the power of self-supervised learning for improving
model performance, especially with limited labeled data, followed by deep
analysis of the trained models.
"""

import os
import math
import keras
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Tuple, List, Dict

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2, create_convnext_v2
from dl_techniques.layers.masked_autoencoder import (
    MaskedAutoencoder,
    visualize_reconstruction,
)

from dl_techniques.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    DataInput,
)

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

def load_dataset(
        dataset_name: str
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[int, int, int], int]:
    """Load and preprocess dataset."""
    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # Convert grayscale to RGB
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
        input_shape = (28, 28, 3)
        num_classes = 10

    elif dataset_name.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)
        num_classes = 10

    elif dataset_name.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)
        num_classes = 100

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.info(f"Dataset loaded: {x_train.shape[0]} train, {x_test.shape[0]} test samples")
    logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


# ---------------------------------------------------------------------

def get_class_names(dataset: str, num_classes: int) -> List[str]:
    """Get class names for the dataset."""
    if dataset.lower() == 'mnist':
        return [str(i) for i in range(10)]
    elif dataset.lower() == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset.lower() == 'cifar100':
        return [f'class_{i}' for i in range(num_classes)]
    else:
        return [f'class_{i}' for i in range(num_classes)]


# ---------------------------------------------------------------------

def create_convnext_encoder(
        variant: str,
        input_shape: Tuple[int, int, int],
        strides: int = 4,
        kernel_size: int = 7
) -> ConvNeXtV2:
    """Create ConvNeXtV2 encoder (feature extractor without classification head)."""
    logger.info(f"Creating ConvNeXt V2 {variant} encoder...")

    encoder = create_convnext_v2(
        variant=variant,
        num_classes=0,  # No classification head
        input_shape=input_shape,
        include_top=False,
        strides=strides,
        kernel_size=kernel_size,
        drop_path_rate=0.1,
        dropout_rate=0.0,  # No dropout in encoder for MAE
        use_gamma=True
    )

    # Build the encoder and get its output shape
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    output_tensor = encoder(dummy_input, training=False)

    logger.info(f"ConvNeXt V2 encoder created:")
    logger.info(f"  Variant: {variant}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Output shape: {output_tensor.shape}")
    logger.info(f"  Parameters: {encoder.count_params():,}")

    return encoder


# ---------------------------------------------------------------------

def create_mae_with_convnext(
        encoder: ConvNeXtV2,
        input_shape: Tuple[int, int, int],
        patch_size: int = 4,
        mask_ratio: float = 0.75,
        decoder_depth: int = -1
) -> MaskedAutoencoder:
    """Create MAE model using ConvNeXtV2 as encoder."""
    logger.info("Creating Masked Autoencoder with ConvNeXt V2 encoder...")

    # Get encoder output shape by performing a dummy forward pass
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    encoder_output = encoder(dummy_input, training=False)
    encoder_output_shape = encoder_output.shape[1:]

    # Dynamically calculate decoder depth if set to auto (-1)
    if decoder_depth == -1:
        input_height = input_shape[0]
        encoder_height = encoder_output_shape[0]

        if input_height % encoder_height != 0:
            raise ValueError(
                f"Input height ({input_height}) must be divisible by "
                f"encoder output height ({encoder_height})."
            )
        downsample_factor = input_height // encoder_height
        if downsample_factor <= 0 or (downsample_factor & (downsample_factor - 1)) != 0:
            raise ValueError(f"Total downsample factor ({downsample_factor}) must be a power of 2 for the ConvDecoder.")

        final_decoder_depth = int(math.log2(downsample_factor))
        logger.info(f"Automatically calculated decoder depth: {final_decoder_depth}")
    else:
        final_decoder_depth = decoder_depth
        logger.info(f"Using user-specified decoder depth: {final_decoder_depth}")

    # Create MAE model by passing the encoder instance directly
    mae = MaskedAutoencoder(
        encoder=encoder,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        decoder_depth=final_decoder_depth,
        norm_pix_loss=False,
        mask_value='learnable',
        input_shape=input_shape
    )

    # Build MAE by calling it with a dummy input to initialize all weights
    _ = mae(dummy_input, training=False)

    logger.info(f"MAE model created:")
    logger.info(f"  Total parameters: {mae.count_params():,}")
    logger.info(f"  Encoder parameters: {mae.encoder.count_params():,}")
    logger.info(f"  Decoder parameters: {mae.decoder.count_params():,}")
    logger.info(f"  Patch size: {patch_size}")
    logger.info(f"  Mask ratio: {mask_ratio}")

    return mae


# ---------------------------------------------------------------------

def create_mae_pretrain_callbacks(
        results_dir: str,
        patience: int = 10
) -> List:
    """Create callbacks for MAE pretraining."""
    mae_dir = os.path.join(results_dir, "mae_pretraining")
    os.makedirs(mae_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(mae_dir, 'best_mae.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(mae_dir, 'mae_training_log.csv')
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    logger.info(f"MAE pretraining results will be saved to: {mae_dir}")
    return callbacks


# ---------------------------------------------------------------------

def visualize_mae_reconstructions(
        mae: MaskedAutoencoder,
        test_images: np.ndarray,
        results_dir: str,
        num_samples: int = 8
):
    """Visualize MAE reconstructions."""
    import matplotlib.pyplot as plt

    mae_dir = os.path.join(results_dir, "mae_pretraining")
    viz_dir = os.path.join(mae_dir, "reconstructions")
    os.makedirs(viz_dir, exist_ok=True)

    logger.info("Generating MAE reconstruction visualizations...")

    # Create grid visualization
    grid = visualize_reconstruction(mae, test_images, num_samples=min(num_samples, len(test_images)))

    plt.figure(figsize=(15, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.title('MAE Reconstructions (Original | Masked | Reconstructed)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mae_reconstructions_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Individual samples
    for i in range(min(4, len(test_images))):
        original, masked, reconstructed = mae.visualize(test_images[i])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original)
        axes[0].set_title('Original', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(masked)
        axes[1].set_title('Masked Input (75% masked)', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(reconstructed)
        axes[2].set_title('Reconstructed', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'mae_sample_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"MAE reconstructions saved to: {viz_dir}")


# ---------------------------------------------------------------------

def pretrain_mae(
        mae: MaskedAutoencoder,
        x_train: np.ndarray,
        x_test: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        results_dir: str,
        patience: int = 10
) -> Tuple[MaskedAutoencoder, keras.callbacks.History]:
    """Pretrain MAE model on unlabeled images."""
    logger.info("=" * 70)
    logger.info("STAGE 1: MAE SELF-SUPERVISED PRETRAINING")
    logger.info("=" * 70)

    # Compile MAE
    mae.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.05
        )
    )

    # Create callbacks
    callbacks = create_mae_pretrain_callbacks(results_dir, patience=patience)

    # Train MAE (no labels needed!)
    logger.info(f"Training MAE for {epochs} epochs...")
    logger.info(f"  Training on {len(x_train)} unlabeled images")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")

    mae_history = mae.fit(
        x_train,  # No labels - self-supervised!
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, x_test),
        callbacks=callbacks,
        verbose=1
    )

    # Visualize reconstructions
    visualize_mae_reconstructions(mae, x_test, results_dir, num_samples=8)

    # Save MAE history
    mae_dir = os.path.join(results_dir, "mae_pretraining")
    np.save(os.path.join(mae_dir, 'mae_history.npy'), mae_history.history)

    logger.info("MAE pretraining completed!")
    logger.info(f"Final reconstruction loss: {mae_history.history['loss'][-1]:.4f}")
    logger.info(f"Best validation loss: {min(mae_history.history['val_loss']):.4f}")

    return mae, mae_history


# ---------------------------------------------------------------------

def build_classifier_from_mae(
        mae: MaskedAutoencoder,
        num_classes: int,
        dropout_rate: float = 0.3
) -> keras.Model:
    """Build classification model using pretrained MAE encoder."""
    logger.info("Building classifier from pretrained MAE encoder...")

    # Extract the ConvNeXtV2 encoder
    encoder = mae.encoder

    # Build classifier
    inputs = keras.Input(shape=mae.input_shape_config)

    # Pretrained ConvNeXt encoder (frozen initially)
    features = encoder(inputs, training=False)

    # Classification head
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(features)
    x = keras.layers.LayerNormalization(name='head_norm')(x)
    x = keras.layers.Dropout(dropout_rate, name='head_dropout')(x)
    outputs = keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='classifier'
    )(x)

    # Create model
    classifier = keras.Model(inputs, outputs, name='convnext_classifier')

    logger.info(f"Classifier built with {classifier.count_params():,} total parameters")
    logger.info(f"  ConvNeXt encoder (frozen): {encoder.count_params():,} parameters")

    return classifier


# ---------------------------------------------------------------------

def create_finetune_callbacks(
        results_dir: str,
        stage: str,
        patience: int = 10
) -> List:
    """Create callbacks for fine-tuning."""
    finetune_dir = os.path.join(results_dir, f"finetuning_{stage}")
    os.makedirs(finetune_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(finetune_dir, f'best_model_{stage}.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(finetune_dir, f'training_log_{stage}.csv')
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callbacks


# ---------------------------------------------------------------------

def finetune_classifier(
        classifier: keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        num_classes: int,
        epochs_stage1: int,
        epochs_stage2: int,
        batch_size: int,
        lr_stage1: float,
        lr_stage2: float,
        results_dir: str,
        patience: int = 10
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Two-stage fine-tuning: freeze encoder then unfreeze."""
    logger.info("=" * 70)
    logger.info("STAGE 2: SUPERVISED FINE-TUNING")
    logger.info("=" * 70)

    # Get ConvNeXt encoder reference
    encoder = None
    for layer in classifier.layers:
        if isinstance(layer, ConvNeXtV2):
            encoder = layer
            break

    if encoder is None:
        logger.warning("Could not find ConvNeXt encoder layer, searching by name...")
        for layer in classifier.layers:
            if 'convnext' in layer.name.lower() or 'encoder' in layer.name.lower():
                encoder = layer
                break

    if encoder is None:
        logger.warning("Could not find encoder layer, treating all as trainable")

    # -------------------------
    # Stage 2a: Freeze encoder, train only head
    # -------------------------
    logger.info("Stage 2a: Training classification head (ConvNeXt encoder frozen)")

    if encoder is not None:
        encoder.trainable = False

    metrics = ['accuracy']
    if num_classes > 10:
        metrics.append('top_5_accuracy')

    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_stage1),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )

    trainable_params_stage1 = np.sum([np.prod(w.shape) for w in classifier.trainable_weights])
    logger.info(f"  Trainable parameters: {trainable_params_stage1:,}")

    callbacks_stage1 = create_finetune_callbacks(results_dir, 'stage1_frozen', patience=patience)

    history_stage1 = classifier.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs_stage1,
        validation_data=(x_test, y_test),
        callbacks=callbacks_stage1,
        verbose=1
    )

    logger.info(f"Stage 2a completed!")
    logger.info(f"  Best validation accuracy: {max(history_stage1.history['val_accuracy']):.4f}")

    # -------------------------
    # Stage 2b: Unfreeze encoder, fine-tune entire model
    # -------------------------
    logger.info("Stage 2b: Fine-tuning entire model (ConvNeXt encoder unfrozen)")

    if encoder is not None:
        encoder.trainable = True

    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_stage2),
        loss='sparse_categorical_crossentropy',
        metrics=metrics
    )

    trainable_params_stage2 = np.sum([np.prod(w.shape) for w in classifier.trainable_weights])
    logger.info(f"  Trainable parameters: {trainable_params_stage2:,}")

    callbacks_stage2 = create_finetune_callbacks(results_dir, 'stage2_unfrozen', patience=patience)

    history_stage2 = classifier.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs_stage2,
        validation_data=(x_test, y_test),
        callbacks=callbacks_stage2,
        verbose=1
    )

    logger.info(f"Stage 2b completed!")
    logger.info(f"  Best validation accuracy: {max(history_stage2.history['val_accuracy']):.4f}")

    # Combine histories
    combined_history = keras.callbacks.History()
    combined_history.history = {}

    for key in history_stage1.history.keys():
        combined_history.history[key] = (
                history_stage1.history[key] + history_stage2.history[key]
        )

    return classifier, combined_history


# ---------------------------------------------------------------------

def run_comprehensive_analysis(
        classifier: keras.Model,
        training_history: keras.callbacks.History,
        x_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        results_dir: str,
        config: AnalysisConfig
) -> None:
    """
    Run comprehensive model analysis using ModelAnalyzer.

    Parameters
    ----------
    classifier : keras.Model
        The trained classifier model.
    training_history : keras.callbacks.History
        The training history from fine-tuning.
    x_test : np.ndarray
        Test images.
    y_test : np.ndarray
        Test labels.
    model_name : str
        Name identifier for the model.
    results_dir : str
        Directory to save analysis results.
    config : AnalysisConfig
        Configuration for the analysis.
    """
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE MODEL ANALYSIS")
    logger.info("=" * 70)

    # Create analyzer output directory
    analysis_dir = os.path.join(results_dir, "model_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Prepare data for analyzer
    test_data = DataInput(x_data=x_test, y_data=y_test)

    # Create model dictionary (can be extended for multi-model comparison)
    models = {model_name: classifier}

    # Create training history dictionary
    training_histories = {model_name: training_history}

    # Initialize ModelAnalyzer
    logger.info(f"Initializing ModelAnalyzer for model: {model_name}")
    analyzer = ModelAnalyzer(
        models=models,
        training_history=training_histories,
        config=config,
        output_dir=analysis_dir
    )

    # Run comprehensive analysis
    logger.info("Running comprehensive analysis...")
    logger.info("  - Weight distribution and health")
    logger.info("  - Calibration and confidence metrics")
    logger.info("  - Information flow through layers")
    logger.info("  - Training dynamics and convergence")

    results = analyzer.analyze(test_data)

    logger.info("Analysis complete!")
    logger.info(f"Analysis results saved to: {analysis_dir}")
    logger.info("Generated visualizations:")
    logger.info("  - summary_dashboard.png: High-level overview")
    logger.info("  - training_dynamics.png: Training curves and convergence")
    logger.info("  - weight_learning_journey.png: Weight health analysis")
    logger.info("  - confidence_calibration_analysis.png: Calibration metrics")
    logger.info("  - information_flow_analysis.png: Layer-wise information flow")
    logger.info("  - analysis_results.json: Raw metrics data")


# ---------------------------------------------------------------------

def train_with_mae_pretraining(args: argparse.Namespace):
    """Main training function with MAE pretraining followed by fine-tuning."""
    logger.info("=" * 70)
    logger.info("ConvNeXt V2 Training with MAE Pretraining")
    logger.info("=" * 70)

    setup_gpu()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        "results",
        f"mae_convnext_{args.dataset}_{args.variant}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results directory: {results_dir}")

    # Load dataset
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(args.dataset)
    class_names = get_class_names(args.dataset, num_classes)

    # -------------------------------------------------------------------------
    # CREATE CONVNEXT V2 ENCODER
    # -------------------------------------------------------------------------

    convnext_encoder = create_convnext_encoder(
        variant=args.variant,
        input_shape=input_shape,
        strides=args.strides,
        kernel_size=args.kernel_size
    )

    # -------------------------------------------------------------------------
    # CREATE MAE WITH CONVNEXT V2 ENCODER
    # -------------------------------------------------------------------------

    mae = create_mae_with_convnext(
        encoder=convnext_encoder,
        input_shape=input_shape,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth
    )

    # -------------------------------------------------------------------------
    # STAGE 1: MAE PRETRAINING (Self-supervised on unlabeled images)
    # -------------------------------------------------------------------------

    mae, mae_history = pretrain_mae(
        mae=mae,
        x_train=x_train,
        x_test=x_test,
        epochs=args.mae_epochs,
        batch_size=args.batch_size,
        learning_rate=args.mae_lr,
        results_dir=results_dir,
        patience=args.mae_patience
    )

    # Save MAE model
    mae_path = os.path.join(results_dir, "mae_pretraining", "mae_final.keras")
    mae.save(mae_path)
    logger.info(f"MAE model saved to: {mae_path}")

    # Save just the encoder weights
    encoder_path = os.path.join(results_dir, "mae_pretraining", "convnext_encoder_pretrained.keras")
    mae.encoder.save(encoder_path)
    logger.info(f"Pretrained ConvNeXt encoder saved to: {encoder_path}")

    # -------------------------------------------------------------------------
    # STAGE 2: SUPERVISED FINE-TUNING (Using labeled data)
    # -------------------------------------------------------------------------

    # Build classifier from pretrained encoder
    classifier = build_classifier_from_mae(
        mae=mae,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate
    )

    # Fine-tune classifier
    classifier, finetune_history = finetune_classifier(
        classifier=classifier,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        epochs_stage1=args.finetune_epochs_stage1,
        epochs_stage2=args.finetune_epochs_stage2,
        batch_size=args.batch_size,
        lr_stage1=args.finetune_lr_stage1,
        lr_stage2=args.finetune_lr_stage2,
        results_dir=results_dir,
        patience=args.finetune_patience
    )

    # Save final classifier
    classifier_path = os.path.join(results_dir, "classifier_final.keras")
    classifier.save(classifier_path)
    logger.info(f"Classifier saved to: {classifier_path}")

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------

    logger.info("=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)

    test_results = classifier.evaluate(
        x_test, y_test,
        batch_size=args.batch_size,
        verbose=1,
        return_dict=True
    )
    logger.info(f"Final Test Results: {test_results}")

    # -------------------------------------------------------------------------
    # COMPREHENSIVE MODEL ANALYSIS
    # -------------------------------------------------------------------------

    if args.run_analysis:
        # Configure analysis
        analysis_config = AnalysisConfig(
            analyze_weights=True,
            analyze_calibration=True,
            analyze_information_flow=True,
            analyze_training_dynamics=True,
            n_samples=min(1000, len(x_test)),  # Sample size for analysis
            compute_weight_pca=False,  # Disable for single model
            smooth_training_curves=True,
            smoothing_window=5,
            plot_style='publication',
            save_plots=True,
            save_format='png',
            dpi=300,
            verbose=True
        )

        model_name = f"ConvNeXt_{args.variant}_MAE"

        run_comprehensive_analysis(
            classifier=classifier,
            training_history=finetune_history,
            x_test=x_test,
            y_test=y_test,
            model_name=model_name,
            results_dir=results_dir,
            config=analysis_config
        )

    # -------------------------------------------------------------------------
    # SAVE SUMMARY
    # -------------------------------------------------------------------------

    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("ConvNeXt V2 Training with MAE Pretraining - Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Variant: {args.variant}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")

        f.write("CONVNEXT V2 ENCODER\n")
        f.write("-" * 70 + "\n")
        if hasattr(mae.encoder, 'depths'):
            f.write(f"Depths: {mae.encoder.depths}\n")
            f.write(f"Dimensions: {mae.encoder.dims}\n")
        f.write(f"Parameters: {mae.encoder.count_params():,}\n\n")

        f.write("STAGE 1: MAE PRETRAINING\n")
        f.write("-" * 70 + "\n")
        f.write(f"Epochs: {args.mae_epochs}\n")
        f.write(f"Learning Rate: {args.mae_lr}\n")
        f.write(f"Mask Ratio: {args.mask_ratio}\n")
        f.write(f"Patch Size: {args.patch_size}\n")
        f.write(f"Final Reconstruction Loss: {mae_history.history['loss'][-1]:.4f}\n")
        f.write(f"Best Val Loss: {min(mae_history.history['val_loss']):.4f}\n\n")

        f.write("STAGE 2: SUPERVISED FINE-TUNING\n")
        f.write("-" * 70 + "\n")
        f.write(f"Stage 1 (Frozen) Epochs: {args.finetune_epochs_stage1}\n")
        f.write(f"Stage 1 Learning Rate: {args.finetune_lr_stage1}\n")
        f.write(f"Stage 2 (Unfrozen) Epochs: {args.finetune_epochs_stage2}\n")
        f.write(f"Stage 2 Learning Rate: {args.finetune_lr_stage2}\n")
        f.write(f"Total Training Epochs: {len(finetune_history.history['loss'])}\n\n")

        f.write("FINAL RESULTS\n")
        f.write("-" * 70 + "\n")
        for key, val in test_results.items():
            f.write(f"{key}: {val:.4f}\n")

        f.write(f"\nBest Validation Accuracy: {max(finetune_history.history['val_accuracy']):.4f}\n")
        f.write(f"Final Validation Accuracy: {finetune_history.history['val_accuracy'][-1]:.4f}\n")

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")

    if args.run_analysis:
        logger.info(f"Model Analysis: {os.path.join(results_dir, 'model_analysis')}")


# ---------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train ConvNeXt V2 with MAE pretraining and comprehensive analysis.'
    )

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100'],
                        help='Dataset to use')

    # Model arguments
    parser.add_argument('--variant', type=str, default='cifar10',
                        choices=['cifar10', 'pico', 'nano', 'tiny', 'base'],
                        help='ConvNeXt V2 model variant')
    parser.add_argument('--kernel-size', type=int, default=7,
                        help='ConvNeXt depthwise kernel size')
    parser.add_argument('--strides', type=int, default=4,
                        help='ConvNeXt downsampling strides')

    # MAE pretraining arguments
    parser.add_argument('--mae-epochs', type=int, default=100,
                        help='Number of MAE pretraining epochs')
    parser.add_argument('--mae-lr', type=float, default=1e-4,
                        help='MAE pretraining learning rate')
    parser.add_argument('--mae-patience', type=int, default=15,
                        help='MAE early stopping patience')
    parser.add_argument('--patch-size', type=int, default=4,
                        help='MAE patch size for masking')
    parser.add_argument('--mask-ratio', type=float, default=0.5,
                        help='MAE mask ratio (0-1)')
    parser.add_argument('--decoder-depth', type=int, default=-1,
                        help='MAE decoder depth. Set to -1 for automatic calculation.')

    # Fine-tuning arguments
    parser.add_argument('--finetune-epochs-stage1', type=int, default=50,
                        help='Epochs for stage 1 (frozen encoder)')
    parser.add_argument('--finetune-epochs-stage2', type=int, default=100,
                        help='Epochs for stage 2 (unfrozen encoder)')
    parser.add_argument('--finetune-lr-stage1', type=float, default=1e-3,
                        help='Learning rate for stage 1')
    parser.add_argument('--finetune-lr-stage2', type=float, default=1e-4,
                        help='Learning rate for stage 2')
    parser.add_argument('--finetune-patience', type=int, default=10,
                        help='Fine-tuning early stopping patience')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                        help='Dropout rate in classification head')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')

    # Analysis arguments
    parser.add_argument('--run-analysis', action='store_true', default=True,
                        help='Run comprehensive model analysis')
    parser.add_argument('--no-analysis', dest='run_analysis', action='store_false',
                        help='Skip model analysis')

    args = parser.parse_args()

    try:
        train_with_mae_pretraining(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()