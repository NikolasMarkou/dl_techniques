"""ConvNeXt V2 Training with MAE Pretraining and Comprehensive Analysis."""

import os
import math
import keras
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Tuple, List

from dl_techniques.utils.logger import logger
from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2, create_convnext_v2
from dl_techniques.models.masked_autoencoder import MaskedAutoencoder, visualize_reconstruction
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput
from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback
from train.common import setup_gpu, load_dataset, get_class_names


def create_convnext_encoder(
        variant: str, input_shape: Tuple[int, int, int],
        strides: int = 4, kernel_size: int = 7
) -> ConvNeXtV2:
    """Create ConvNeXtV2 encoder without classification head."""
    encoder = create_convnext_v2(
        variant=variant, num_classes=0, input_shape=input_shape,
        include_top=False, strides=strides, kernel_size=kernel_size,
        drop_path_rate=0.1, dropout_rate=0.0, use_gamma=True,
        use_softorthonormal_regularizer=True
    )
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    output_tensor = encoder(dummy_input, training=False)
    logger.info(f"ConvNeXt V2 {variant} encoder: input={input_shape}, output={output_tensor.shape}, params={encoder.count_params():,}")
    return encoder


def create_mae_with_convnext(
        encoder: ConvNeXtV2, input_shape: Tuple[int, int, int],
        patch_size: int = 4, mask_ratio: float = 0.75, decoder_depth: int = -1
) -> MaskedAutoencoder:
    """Create MAE model using ConvNeXtV2 as encoder."""
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    encoder_output = encoder(dummy_input, training=False)
    encoder_output_shape = encoder_output.shape[1:]

    if decoder_depth == -1:
        input_height = input_shape[0]
        encoder_height = encoder_output_shape[0]
        if input_height % encoder_height != 0:
            raise ValueError(f"Input height ({input_height}) must be divisible by encoder output height ({encoder_height}).")
        downsample_factor = input_height // encoder_height
        if downsample_factor <= 0 or (downsample_factor & (downsample_factor - 1)) != 0:
            raise ValueError(f"Total downsample factor ({downsample_factor}) must be a power of 2.")
        final_decoder_depth = int(math.log2(downsample_factor))
        logger.info(f"Auto decoder depth: {final_decoder_depth}")
    else:
        final_decoder_depth = decoder_depth

    mae = MaskedAutoencoder(
        encoder=encoder, patch_size=patch_size, mask_ratio=mask_ratio,
        decoder_depth=final_decoder_depth, norm_pix_loss=False,
        mask_value='zero', input_shape=input_shape
    )
    _ = mae(dummy_input, training=False)
    logger.info(f"MAE: total={mae.count_params():,}, encoder={mae.encoder.count_params():,}, decoder={mae.decoder.count_params():,}")
    return mae


def create_mae_pretrain_callbacks(results_dir: str, patience: int = 10) -> List:
    """Create callbacks for MAE pretraining."""
    mae_dir = os.path.join(results_dir, "mae_pretraining")
    os.makedirs(mae_dir, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1, mode='min'),
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(mae_dir, 'best_mae.keras'), monitor='val_loss', save_best_only=True, verbose=1, mode='min'),
        keras.callbacks.CSVLogger(filename=os.path.join(mae_dir, 'mae_training_log.csv')),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]


def visualize_mae_reconstructions(mae: MaskedAutoencoder, test_images: np.ndarray, results_dir: str, num_samples: int = 8):
    """Visualize MAE reconstructions."""
    import matplotlib.pyplot as plt

    viz_dir = os.path.join(results_dir, "mae_pretraining", "reconstructions")
    os.makedirs(viz_dir, exist_ok=True)

    grid = visualize_reconstruction(mae, test_images, num_samples=min(num_samples, len(test_images)))
    plt.figure(figsize=(15, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.title('MAE Reconstructions (Original | Masked | Reconstructed)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mae_reconstructions_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for i in range(min(4, len(test_images))):
        original, masked, reconstructed = mae.visualize(test_images[i])
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(axes, [original, masked, reconstructed], ['Original', 'Masked Input (75% masked)', 'Reconstructed']):
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'mae_sample_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"MAE reconstructions saved to: {viz_dir}")


def pretrain_mae(
        mae: MaskedAutoencoder, x_train: np.ndarray, x_test: np.ndarray,
        epochs: int, batch_size: int, learning_rate: float,
        results_dir: str, patience: int = 10
) -> Tuple[MaskedAutoencoder, keras.callbacks.History]:
    """Pretrain MAE model on unlabeled images."""
    logger.info("STAGE 1: MAE self-supervised pretraining")

    mae.compile(optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.05))
    callbacks = create_mae_pretrain_callbacks(results_dir, patience=patience)

    logger.info(f"Training MAE: {len(x_train)} images, {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    mae_history = mae.fit(
        x_train, batch_size=batch_size, epochs=epochs,
        validation_data=(x_test, x_test), callbacks=callbacks, verbose=1
    )

    visualize_mae_reconstructions(mae, x_test, results_dir, num_samples=8)
    mae_dir = os.path.join(results_dir, "mae_pretraining")
    np.save(os.path.join(mae_dir, 'mae_history.npy'), mae_history.history)

    logger.info(f"MAE pretraining done. Final loss: {mae_history.history['loss'][-1]:.4f}, best val: {min(mae_history.history['val_loss']):.4f}")
    return mae, mae_history


def build_classifier_from_mae(mae: MaskedAutoencoder, num_classes: int, dropout_rate: float = 0.3) -> keras.Model:
    """Build classification model using pretrained MAE encoder."""
    encoder = mae.encoder
    inputs = keras.Input(shape=mae.input_shape_config)
    features = encoder(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(features)
    x = keras.layers.LayerNormalization(name='head_norm')(x)
    x = keras.layers.Dropout(dropout_rate, name='head_dropout')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='classifier')(x)
    classifier = keras.Model(inputs, outputs, name='convnext_classifier')
    logger.info(f"Classifier: {classifier.count_params():,} params (encoder frozen: {encoder.count_params():,})")
    return classifier


def create_finetune_callbacks(results_dir: str, stage: str, patience: int = 10) -> List:
    """Create callbacks for fine-tuning."""
    finetune_dir = os.path.join(results_dir, f"finetuning_{stage}")
    os.makedirs(finetune_dir, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True, verbose=1, mode='max'),
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(finetune_dir, f'best_model_{stage}.keras'), monitor='val_accuracy', save_best_only=True, verbose=1, mode='max'),
        keras.callbacks.CSVLogger(filename=os.path.join(finetune_dir, f'training_log_{stage}.csv')),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        EpochAnalyzerCallback(output_dir=os.path.join(finetune_dir, "epoch_analysis"), model_name=f"convnext_v2_{stage}", epoch_frequency=1),
    ]


def finetune_classifier(
        classifier: keras.Model, x_train: np.ndarray, y_train: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray, num_classes: int,
        epochs_stage1: int, epochs_stage2: int, batch_size: int,
        lr_stage1: float, lr_stage2: float, results_dir: str, patience: int = 10
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Two-stage fine-tuning: freeze encoder then unfreeze."""
    logger.info("STAGE 2: Supervised fine-tuning")

    encoder = None
    for layer in classifier.layers:
        if isinstance(layer, ConvNeXtV2):
            encoder = layer
            break
    if encoder is None:
        for layer in classifier.layers:
            if 'convnext' in layer.name.lower() or 'encoder' in layer.name.lower():
                encoder = layer
                break

    metrics = ['accuracy']
    if num_classes > 10:
        metrics.append('top_5_accuracy')

    # Stage 2a: Freeze encoder, train head
    logger.info("Stage 2a: Training head (encoder frozen)")
    if encoder is not None:
        encoder.trainable = False

    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_stage1), loss='sparse_categorical_crossentropy', metrics=metrics)
    callbacks_stage1 = create_finetune_callbacks(results_dir, 'stage1_frozen', patience=patience)
    history_stage1 = classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_stage1, validation_data=(x_test, y_test), callbacks=callbacks_stage1, verbose=1)
    logger.info(f"Stage 2a best val accuracy: {max(history_stage1.history['val_accuracy']):.4f}")

    # Stage 2b: Unfreeze encoder, fine-tune all
    logger.info("Stage 2b: Fine-tuning entire model (encoder unfrozen)")
    if encoder is not None:
        encoder.trainable = True

    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_stage2), loss='sparse_categorical_crossentropy', metrics=metrics)
    callbacks_stage2 = create_finetune_callbacks(results_dir, 'stage2_unfrozen', patience=patience)
    history_stage2 = classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_stage2, validation_data=(x_test, y_test), callbacks=callbacks_stage2, verbose=1)
    logger.info(f"Stage 2b best val accuracy: {max(history_stage2.history['val_accuracy']):.4f}")

    combined_history = keras.callbacks.History()
    combined_history.history = {}
    for key in history_stage1.history.keys():
        combined_history.history[key] = history_stage1.history[key] + history_stage2.history[key]

    return classifier, combined_history


def run_comprehensive_analysis(
        classifier: keras.Model, training_history: keras.callbacks.History,
        x_test: np.ndarray, y_test: np.ndarray, model_name: str,
        results_dir: str, config: AnalysisConfig
) -> None:
    """Run comprehensive model analysis using ModelAnalyzer."""
    analysis_dir = os.path.join(results_dir, "model_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    test_data = DataInput(x_data=x_test, y_data=y_test)
    analyzer = ModelAnalyzer(
        models={model_name: classifier},
        training_history={model_name: training_history.history},
        config=config, output_dir=analysis_dir
    )
    results = analyzer.analyze(test_data)
    logger.info(f"Analysis complete, results saved to: {analysis_dir}")


def train_with_mae_pretraining(args: argparse.Namespace):
    """Main training function with MAE pretraining followed by fine-tuning."""
    setup_gpu()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"mae_convnext_{args.dataset}_{args.variant}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(args.dataset)
    class_names = get_class_names(args.dataset, num_classes)

    convnext_encoder = create_convnext_encoder(variant=args.variant, input_shape=input_shape, strides=args.strides, kernel_size=args.kernel_size)
    mae = create_mae_with_convnext(encoder=convnext_encoder, input_shape=input_shape, patch_size=args.patch_size, mask_ratio=args.mask_ratio, decoder_depth=args.decoder_depth)

    mae, mae_history = pretrain_mae(
        mae=mae, x_train=x_train, x_test=x_test, epochs=args.mae_epochs,
        batch_size=args.batch_size, learning_rate=args.mae_lr,
        results_dir=results_dir, patience=args.mae_patience
    )

    mae_path = os.path.join(results_dir, "mae_pretraining", "mae_final.keras")
    mae.save(mae_path)
    encoder_path = os.path.join(results_dir, "mae_pretraining", "convnext_encoder_pretrained.keras")
    mae.encoder.save(encoder_path)

    classifier = build_classifier_from_mae(mae=mae, num_classes=num_classes, dropout_rate=args.dropout_rate)
    classifier, finetune_history = finetune_classifier(
        classifier=classifier, x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test, num_classes=num_classes,
        epochs_stage1=args.finetune_epochs_stage1, epochs_stage2=args.finetune_epochs_stage2,
        batch_size=args.batch_size, lr_stage1=args.finetune_lr_stage1,
        lr_stage2=args.finetune_lr_stage2, results_dir=results_dir, patience=args.finetune_patience
    )

    classifier_path = os.path.join(results_dir, "classifier_final.keras")
    classifier.save(classifier_path)

    test_results = classifier.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results: {test_results}")

    if args.run_analysis:
        analysis_config = AnalysisConfig(
            analyze_weights=True, analyze_calibration=True,
            analyze_information_flow=True, analyze_training_dynamics=True,
            n_samples=min(1000, len(x_test)), compute_weight_pca=False,
            smooth_training_curves=True, smoothing_window=5,
            plot_style='publication', save_plots=True, save_format='png', dpi=300, verbose=True
        )
        run_comprehensive_analysis(
            classifier=classifier, training_history=finetune_history,
            x_test=x_test, y_test=y_test,
            model_name=f"ConvNeXt_{args.variant}_MAE",
            results_dir=results_dir, config=analysis_config
        )

    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("ConvNeXt V2 Training with MAE Pretraining - Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {args.dataset}\nVariant: {args.variant}\nInput Shape: {input_shape}\nClasses: {num_classes}\n\n")
        if hasattr(mae.encoder, 'depths'):
            f.write(f"Encoder Depths: {mae.encoder.depths}\nEncoder Dims: {mae.encoder.dims}\n")
        f.write(f"Encoder Params: {mae.encoder.count_params():,}\n\n")
        f.write(f"MAE Epochs: {args.mae_epochs}\nMAE LR: {args.mae_lr}\nMask Ratio: {args.mask_ratio}\nPatch Size: {args.patch_size}\n")
        f.write(f"Final Reconstruction Loss: {mae_history.history['loss'][-1]:.4f}\nBest Val Loss: {min(mae_history.history['val_loss']):.4f}\n\n")
        f.write(f"Frozen Epochs: {args.finetune_epochs_stage1}\nFrozen LR: {args.finetune_lr_stage1}\n")
        f.write(f"Unfrozen Epochs: {args.finetune_epochs_stage2}\nUnfrozen LR: {args.finetune_lr_stage2}\n")
        f.write(f"Total Training Epochs: {len(finetune_history.history['loss'])}\n\n")
        for key, val in test_results.items():
            f.write(f"{key}: {val:.4f}\n")
        f.write(f"\nBest Val Accuracy: {max(finetune_history.history['val_accuracy']):.4f}\n")
        f.write(f"Final Val Accuracy: {finetune_history.history['val_accuracy'][-1]:.4f}\n")

    logger.info(f"Training complete. Results: {results_dir}, Test Accuracy: {test_results['accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train ConvNeXt V2 with MAE pretraining.')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--variant', type=str, default='cifar10', choices=['cifar10', 'pico', 'nano', 'tiny', 'base'])
    parser.add_argument('--kernel-size', type=int, default=7)
    parser.add_argument('--strides', type=int, default=4)
    parser.add_argument('--mae-epochs', type=int, default=100)
    parser.add_argument('--mae-lr', type=float, default=1e-4)
    parser.add_argument('--mae-patience', type=int, default=25)
    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--decoder-depth', type=int, default=-1)
    parser.add_argument('--finetune-epochs-stage1', type=int, default=50)
    parser.add_argument('--finetune-epochs-stage2', type=int, default=100)
    parser.add_argument('--finetune-lr-stage1', type=float, default=1e-3)
    parser.add_argument('--finetune-lr-stage2', type=float, default=1e-4)
    parser.add_argument('--finetune-patience', type=int, default=25)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--run-analysis', action='store_true', default=True)
    parser.add_argument('--no-analysis', dest='run_analysis', action='store_false')

    args = parser.parse_args()

    try:
        train_with_mae_pretraining(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
