"""Training script for CoShNet (Complex Shearlet Network) model."""

import os
import keras
import numpy as np
from typing import Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.shearlet_transform import ShearletTransform
from dl_techniques.layers.complex_layers import ComplexDense, ComplexConv2D, ComplexReLU
from dl_techniques.models.coshnet.model import CoShNet, create_coshnet

from train.common import (
    setup_gpu,
    create_base_argument_parser,
    load_dataset,
    create_callbacks,
    create_learning_rate_schedule,
    validate_model_loading,
    run_model_analysis,
)


# Custom objects needed for model serialization
CUSTOM_OBJECTS = {
    "CoShNet": CoShNet,
    "ShearletTransform": ShearletTransform,
    "ComplexDense": ComplexDense,
    "ComplexConv2D": ComplexConv2D,
    "ComplexReLU": ComplexReLU,
}


# ---------------------------------------------------------------------

def create_model_config(dataset: str, variant: str, input_shape, num_classes: int) -> Dict[str, Any]:
    """Create CoShNet model configuration based on dataset and variant."""
    config = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'kernel_regularizer': None,
        'epsilon': 1e-7,
    }

    # Dataset-specific adjustments
    if dataset.lower() == 'mnist':
        config.update({'dropout_rate': 0.05, 'shearlet_scales': 3, 'shearlet_directions': 6})
    elif dataset.lower() == 'cifar10':
        config.update({'dropout_rate': 0.1, 'shearlet_scales': 4, 'shearlet_directions': 8})
    elif dataset.lower() == 'cifar100':
        config.update({'dropout_rate': 0.15, 'shearlet_scales': 4, 'shearlet_directions': 8,
                       'kernel_regularizer': keras.regularizers.l2(1e-4)})

    # Variant-specific adjustments
    if variant == 'tiny':
        config.update({'conv_filters': (16, 32), 'dense_units': (256, 128),
                       'dropout_rate': config.get('dropout_rate', 0.1) + 0.05})
    elif variant == 'large':
        config.update({'conv_filters': (64, 128, 256), 'dense_units': (2048, 1024, 512),
                       'dropout_rate': config.get('dropout_rate', 0.1) + 0.05,
                       'shearlet_scales': 5, 'shearlet_directions': 12})

    return config


# ---------------------------------------------------------------------

def train_model(args) -> None:
    """Main CoShNet training function."""
    logger.info("Starting CoShNet training")
    setup_gpu(gpu_id=args.gpu)

    # Load dataset
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(
        args.dataset, batch_size=args.batch_size
    )

    # Create model
    logger.info(f"Creating CoShNet model (variant: {args.variant})...")
    model = create_coshnet(variant=args.variant, num_classes=num_classes, input_shape=input_shape)

    # Build and verify
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    output = model(dummy_input, training=False)
    logger.info(f"Model built. Output shape: {output.shape}, Params: {model.count_params():,}")

    # LR schedule and optimizer
    use_lr_schedule = args.lr_schedule != 'constant'
    lr = create_learning_rate_schedule(
        initial_lr=args.learning_rate, schedule_type=args.lr_schedule,
        total_epochs=args.epochs
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
    )

    # Compile
    metrics = ['accuracy']
    if num_classes > 10:
        metrics.append('top_5_accuracy')

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=metrics
    )

    # Callbacks
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.variant}",
        results_dir_prefix="coshnet",
        patience=args.patience,
        use_lr_schedule=use_lr_schedule,
    )

    # Train
    logger.info(f"Training | dataset={args.dataset}, variant={args.variant}, "
                f"epochs={args.epochs}, batch={args.batch_size}, lr={args.learning_rate}")

    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Save and validate
    test_sample = x_test[:4]
    pre_save_output = model.predict(test_sample, verbose=0)

    final_path = os.path.join(results_dir, f"coshnet_{args.dataset}_{args.variant}_final.keras")
    try:
        model.save(final_path)
        logger.info(f"Final model saved to: {final_path}")
        validate_model_loading(final_path, test_sample, pre_save_output, CUSTOM_OBJECTS)
    except Exception as e:
        logger.warning(f"Failed to save final model: {e}")

    # Load best and evaluate
    best_path = os.path.join(results_dir, 'best_model.keras')
    best_model = model
    if os.path.exists(best_path):
        try:
            best_model = keras.models.load_model(best_path, custom_objects=CUSTOM_OBJECTS)
            logger.info("Loaded best checkpoint")
        except Exception as e:
            logger.warning(f"Could not load best checkpoint: {e}")

    test_results = best_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Test results: {test_results}")

    # Post-training analysis
    run_model_analysis(
        model=best_model, test_data=(x_test, y_test), training_history=history,
        model_name=f"coshnet_{args.variant}_{args.dataset}", results_dir=results_dir,
    )

    # Save summary
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"CoShNet Training Summary\n{'=' * 40}\n\n")
        f.write(f"Dataset: {args.dataset}, Variant: {args.variant}\n")
        f.write(f"Input: {input_shape}, Classes: {num_classes}\n")
        f.write(f"Params: {model.count_params():,}\n\n")
        f.write(f"Epochs: {len(history.history['loss'])}, Batch: {args.batch_size}\n")
        f.write(f"LR: {args.learning_rate}, Schedule: {args.lr_schedule}\n\n")
        f.write("Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key}: {val:.4f}\n")
        f.write(f"\nBest val_accuracy: {best_val_acc:.4f}\n")

    logger.info(f"Training complete. Best val_accuracy={best_val_acc:.4f}")


# ---------------------------------------------------------------------

def main() -> None:
    parser = create_base_argument_parser(
        description='Train CoShNet (Complex Shearlet Network) model',
        dataset_choices=['mnist', 'cifar10', 'cifar100'],
    )
    parser.add_argument('--variant', type=str, default='base',
                        choices=['tiny', 'base', 'large', 'cifar10', 'custom'],
                        help='CoShNet model variant')
    parser.set_defaults(epochs=100, batch_size=64, learning_rate=2e-3, patience=10)
    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
