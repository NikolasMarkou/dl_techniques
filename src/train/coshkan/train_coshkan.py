"""Training script for CoshKan (Complex Shearlet KAN Network) model."""

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
from dl_techniques.layers.kan_linear import KANLinear
from dl_techniques.models.coshkan.model import CoshKan, create_coshkan, create_coshkan_variant

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
    "CoshKan": CoshKan,
    "ShearletTransform": ShearletTransform,
    "ComplexConv2D": ComplexConv2D,
    "ComplexDense": ComplexDense,
    "ComplexReLU": ComplexReLU,
    "KANLinear": KANLinear,
}


# ---------------------------------------------------------------------

def create_model_config(dataset: str, input_shape, num_classes: int) -> Dict[str, Any]:
    """Create CoshKan model configuration based on dataset."""
    config = {'input_shape': input_shape, 'num_classes': num_classes}

    if dataset.lower() == 'mnist':
        config.update({'dropout_rate': 0.1, 'kan_grid_size': 3,
                       'shearlet_scales': 3, 'shearlet_directions': 6})
    elif dataset.lower() == 'cifar10':
        config.update({'dropout_rate': 0.1, 'kan_grid_size': 5,
                       'shearlet_scales': 4, 'shearlet_directions': 8})
    elif dataset.lower() == 'cifar100':
        config.update({'dropout_rate': 0.2, 'kan_grid_size': 7,
                       'shearlet_scales': 5, 'shearlet_directions': 12})
    else:
        config.update({'dropout_rate': 0.15, 'kan_grid_size': 5,
                       'shearlet_scales': 4, 'shearlet_directions': 8})
    return config


VARIANT_CONFIGS = {
    "micro": {"conv_filters": (8, 16), "dense_units": (64, 32), "kan_units": (16,),
              "shearlet_scales": 2, "shearlet_directions": 4, "kan_grid_size": 3,
              "dropout_rate": 0.2, "conv_kernel_size": 3, "pool_size": 2},
    "small": {"conv_filters": (16, 32), "dense_units": (128, 64), "kan_units": (32, 16),
              "shearlet_scales": 3, "shearlet_directions": 6, "kan_grid_size": 3,
              "dropout_rate": 0.15, "conv_kernel_size": 3, "pool_size": 2},
    "base": {"conv_filters": (32, 64), "dense_units": (512, 256), "kan_units": (128, 64),
             "shearlet_scales": 4, "shearlet_directions": 8, "kan_grid_size": 5,
             "dropout_rate": 0.1, "conv_kernel_size": 5, "pool_size": 2},
    "large": {"conv_filters": (64, 128, 256), "dense_units": (1024, 512, 256),
              "kan_units": (256, 128, 64), "shearlet_scales": 5, "shearlet_directions": 12,
              "kan_grid_size": 7, "kan_spline_order": 4, "dropout_rate": 0.15,
              "conv_kernel_size": 5, "pool_size": 2},
    "imagenet": {"conv_filters": (64, 128, 256), "dense_units": (2048, 1024, 512),
                 "kan_units": (512, 256, 128), "shearlet_scales": 5, "shearlet_directions": 16,
                 "kan_grid_size": 8, "kan_spline_order": 4, "dropout_rate": 0.2,
                 "conv_kernel_size": 7, "conv_strides": 2, "pool_size": 3},
}


def create_model(variant: str, dataset: str, input_shape, num_classes: int):
    """Create CoshKan model for given variant and dataset."""
    model_config = create_model_config(dataset, input_shape, num_classes)

    if variant in VARIANT_CONFIGS:
        model = create_coshkan_variant(variant)
        if model.input_shape_config != input_shape or model.num_classes != num_classes:
            logger.info(f"Adjusting variant for dataset: {input_shape}, {num_classes} classes")
            config = {**VARIANT_CONFIGS[variant], **model_config}
            model = create_coshkan(**config)
    elif variant == 'cifar10':
        model = create_coshkan_variant('cifar10')
        if num_classes != 10 or input_shape != (32, 32, 3):
            model = create_coshkan(**model_config)
    else:
        model = create_coshkan(**model_config)

    return model


# ---------------------------------------------------------------------

def train_model(args) -> None:
    """Main training function."""
    logger.info("Starting CoshKan training")
    setup_gpu(gpu_id=args.gpu)

    # Load dataset
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(
        args.dataset, batch_size=args.batch_size
    )

    # Create model
    logger.info(f"Creating CoshKan model (variant: {args.variant})...")
    model = create_model(args.variant, args.dataset, input_shape, num_classes)

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
        learning_rate=lr, weight_decay=args.weight_decay, clipnorm=1.0
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
        results_dir_prefix="coshkan",
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

    final_path = os.path.join(results_dir, f"coshkan_{args.dataset}_{args.variant}_final.keras")
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
        model_name=f"coshkan_{args.variant}_{args.dataset}", results_dir=results_dir,
    )

    # Save summary
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"CoshKan Training Summary\n{'=' * 40}\n\n")
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

def main():
    parser = create_base_argument_parser(
        description='Train CoshKan model',
        dataset_choices=['mnist', 'cifar10', 'cifar100'],
    )
    parser.add_argument('--variant', type=str, default='base',
                        choices=['micro', 'small', 'base', 'large', 'cifar10', 'imagenet'],
                        help='Model variant')
    parser.set_defaults(epochs=50, batch_size=32, patience=15)
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
