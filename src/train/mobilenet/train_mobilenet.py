"""
MobileNet training script for V1/V2/V3/V4 on classification datasets.

Trains any MobileNet variant on MNIST/CIFAR-10/CIFAR-100 with comprehensive
callbacks, learning rate scheduling, and post-training analysis.

Usage:
    python -m train.mobilenet.train_mobilenet --version v1 --variant large --dataset cifar10
    python -m train.mobilenet.train_mobilenet --version v2 --variant medium --dataset cifar100 --epochs 100
    python -m train.mobilenet.train_mobilenet --version v3 --variant small --dataset mnist --gpu 1
    python -m train.mobilenet.train_mobilenet --version v4 --variant small --dataset cifar10 --gpu 1
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Dict, Any

from dl_techniques.utils.logger import logger
from dl_techniques.models.mobilenet.mobilenet_v1 import MobileNetV1, create_mobilenetv1
from dl_techniques.models.mobilenet.mobilenet_v2 import MobileNetV2, create_mobilenetv2
from dl_techniques.models.mobilenet.mobilenet_v3 import MobileNetV3, create_mobilenetv3
from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4, create_mobilenetv4
from train.common import (
    setup_gpu,
    create_base_argument_parser,
    create_callbacks,
    create_learning_rate_schedule,
    load_dataset,
    get_class_names,
    run_model_analysis,
)


# =============================================================================
# MODEL CREATION
# =============================================================================

# Version → (factory_function, available_variants, default_variant)
VERSION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "v1": {
        "factory": create_mobilenetv1,
        "variants": list(MobileNetV1.MODEL_VARIANTS.keys()),
        "default": "large",
        "custom_objects": {"MobileNetV1": MobileNetV1},
    },
    "v2": {
        "factory": create_mobilenetv2,
        "variants": list(MobileNetV2.MODEL_VARIANTS.keys()),
        "default": "medium",
        "custom_objects": {"MobileNetV2": MobileNetV2},
    },
    "v3": {
        "factory": create_mobilenetv3,
        "variants": list(MobileNetV3.MODEL_VARIANTS.keys()),
        "default": "large",
        "custom_objects": {"MobileNetV3": MobileNetV3},
    },
    "v4": {
        "factory": create_mobilenetv4,
        "variants": list(MobileNetV4.MODEL_VARIANTS.keys()),
        "default": "small",
        "custom_objects": {"MobileNetV4": MobileNetV4},
    },
}


def get_input_shape(dataset: str) -> tuple:
    """Get input shape for the given dataset."""
    shapes = {
        "mnist": (28, 28, 1),
        "cifar10": (32, 32, 3),
        "cifar100": (32, 32, 3),
    }
    return shapes[dataset]


def create_model(version: str, variant: str, dataset: str, num_classes: int) -> keras.Model:
    """Create a MobileNet model for the given version, variant, and dataset."""
    info = VERSION_REGISTRY[version]
    input_shape = get_input_shape(dataset)

    model = info["factory"](
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
    )

    # Build the model
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    _ = model(dummy_input, training=False)

    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_model(args) -> None:
    """Main training function."""
    setup_gpu(gpu_id=args.gpu)

    version = args.version
    variant = args.variant or VERSION_REGISTRY[version]["default"]
    info = VERSION_REGISTRY[version]

    if variant not in info["variants"]:
        raise ValueError(
            f"Unknown variant '{variant}' for MobileNet{version.upper()}. "
            f"Available: {info['variants']}"
        )

    logger.info(f"Starting MobileNet{version.upper()}-{variant} training")
    logger.info(f"Dataset: {args.dataset}, Epochs: {args.epochs}, "
                f"Batch size: {args.batch_size}, LR: {args.learning_rate}")

    # Load dataset
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(
        args.dataset, args.batch_size
    )
    class_names = get_class_names(args.dataset, num_classes)
    logger.info(f"Dataset loaded: {x_train.shape[0]} train, {x_test.shape[0]} test, "
                f"{num_classes} classes")

    # Create model
    model = create_model(version, variant, args.dataset, num_classes)
    model.summary()

    # Register custom objects for serialization
    keras.saving.get_custom_objects().update(info["custom_objects"])

    # Learning rate schedule
    steps_per_epoch = x_train.shape[0] // args.batch_size
    lr = create_learning_rate_schedule(
        initial_lr=args.learning_rate,
        schedule_type=args.lr_schedule,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Compile
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=args.weight_decay,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    model_name = f"{args.dataset}_mobilenet{version}_{variant}"
    callbacks, results_dir = create_callbacks(
        model_name=model_name,
        results_dir_prefix=f"mobilenet{version}",
        monitor="val_accuracy",
        patience=args.patience,
        use_lr_schedule=True,
    )

    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Post-training analysis
    run_model_analysis(
        model=model,
        test_data=(x_test, y_test),
        training_history=history,
        model_name=model_name,
        results_dir=results_dir,
    )

    logger.info(f"Training complete. Results saved to: {results_dir}")

    # Log final metrics
    final_metrics = {k: v[-1] for k, v in history.history.items() if "val_" in k}
    for name, value in final_metrics.items():
        logger.info(f"  {name}: {value:.4f}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = create_base_argument_parser(
        description="Train MobileNet on image classification datasets",
        default_dataset="cifar10",
    )

    # MobileNet-specific arguments
    parser.add_argument(
        "--version", type=str, default="v1",
        choices=["v1", "v2", "v3", "v4"],
        help="MobileNet version (default: v1)",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Model variant (version-specific, default: version default)",
    )

    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
