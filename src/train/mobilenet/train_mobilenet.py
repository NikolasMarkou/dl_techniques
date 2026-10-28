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
from dl_techniques.models.vision.mobilenet.mobilenet_v1 import MobileNetV1, create_mobilenetv1
from dl_techniques.models.vision.mobilenet.mobilenet_v2 import MobileNetV2, create_mobilenetv2
from dl_techniques.models.vision.mobilenet.mobilenet_v3 import MobileNetV3, create_mobilenetv3
from dl_techniques.models.vision.mobilenet.mobilenet_v4 import MobileNetV4, create_mobilenetv4
from train.common import (
    setup_gpu,
    set_seeds,
    create_base_argument_parser,
    create_callbacks,
    create_learning_rate_schedule,
    run_model_analysis,
)
from train.common.classification_data import add_selection_arguments, load_classification_data


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


def create_model(version: str, variant: str, input_shape: tuple, num_classes: int) -> keras.Model:
    """Create a MobileNet model for the given version, variant and input shape."""
    info = VERSION_REGISTRY[version]

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
    set_seeds(args.seed)

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

    # Load dataset: [0, 1] inputs in the dataset's native shape, selection on a held-out
    # slice of train, the test split only for the final analysis.
    data = load_classification_data(
        args.dataset, batch_size=args.batch_size, seed=args.seed, normalize="unit",
        validation_split=args.validation_split, max_samples=args.max_samples,
    )
    logger.info(f"Dataset loaded: {data.n_train} train, {data.n_select} selection, "
                f"{data.n_test} test, {data.num_classes} classes")

    # Create model
    model = create_model(version, variant, data.input_shape, data.num_classes)
    model.summary()

    # Register custom objects for serialization
    keras.saving.get_custom_objects().update(info["custom_objects"])

    # Learning rate schedule
    steps_per_epoch = data.steps_per_epoch
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
        # The model returns logits (its head's classifier is linear).
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Callbacks
    model_name = f"{args.dataset}_mobilenet{version}_{variant}"
    callbacks, results_dir = create_callbacks(
        model_name=model_name,
        results_dir_prefix=f"mobilenet{version}",
        output_root=args.output_dir,
        monitor="val_accuracy",
        patience=args.patience,
        use_lr_schedule=True,
    )

    # Train
    history = model.fit(
        data.train_ds,
        validation_data=data.select_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Post-training analysis
    run_model_analysis(
        model=model,
        test_data=(data.x_fig, data.y_fig),
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
        include_image_size=False,
    )
    add_selection_arguments(parser, imagenet=False)

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
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory under which the timestamped run directory is created "
             "(default: results)",
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for Python, NumPy and Keras (train.common.set_seeds)')

    args = parser.parse_args()

    try:
        train_model(args)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
