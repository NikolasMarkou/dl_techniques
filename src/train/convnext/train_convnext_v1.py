"""Train ConvNeXt V1 model with ImageNet support using TensorFlow Datasets."""

import os
import keras
import argparse
import numpy as np
from typing import Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.models.convnext.convnext_v1 import ConvNeXtV1, create_convnext_v1

from dl_techniques.visualization import (
    VisualizationManager,
)

from train.common import (
    setup_gpu,
    load_dataset,
    get_class_names,
    create_callbacks,
    create_learning_rate_schedule,
    validate_model_loading,
    convert_keras_history_to_training_history,
    create_classification_results,
    generate_comprehensive_visualizations,
    run_model_analysis,
)


# ---------------------------------------------------------------------

def create_model_config(
        dataset: str,
        variant: str,
        input_shape: Tuple[int, int, int],
        num_classes: int
) -> Dict[str, Any]:
    """Create ConvNeXt V1 model configuration based on dataset."""
    config = {
        'include_top': True,
        'kernel_regularizer': None,
    }

    if dataset.lower() == 'imagenet':
        config.update({
            'drop_path_rate': 0.1 if variant in ['tiny', 'small'] else 0.2,
            'dropout_rate': 0.0,
            'use_gamma': True,
        })
    elif dataset.lower() == 'mnist':
        config.update({
            'drop_path_rate': 0.1,
            'dropout_rate': 0.1,
            'use_gamma': True,
        })
    elif dataset.lower() in ['cifar10', 'cifar100']:
        config.update({
            'drop_path_rate': 0.1 if num_classes == 10 else 0.2,
            'dropout_rate': 0.1 if num_classes == 10 else 0.2,
            'use_gamma': True,
        })
    else:
        config.update({
            'drop_path_rate': 0.1,
            'dropout_rate': 0.1,
            'use_gamma': True,
        })

    return config


# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace) -> None:
    """Train ConvNeXt V1 model."""
    setup_gpu()

    # Determine image size for ImageNet
    image_size = None
    if args.dataset.lower() == 'imagenet':
        image_size = (args.image_size, args.image_size)

    # Load dataset
    train_data, test_data, input_shape, num_classes = load_dataset(
        args.dataset,
        batch_size=args.batch_size,
        image_size=image_size
    )

    class_names = get_class_names(args.dataset, num_classes)

    # Create model
    logger.info(f"Creating ConvNeXt V1 model (variant: {args.variant})...")
    model_config = create_model_config(args.dataset, args.variant, input_shape, num_classes)

    model = create_convnext_v1(
        variant=args.variant,
        input_shape=input_shape,
        num_classes=num_classes,
        kernel_size=args.kernel_size,
        strides=args.strides,
        **model_config
    )

    # Calculate steps per epoch for ImageNet
    is_imagenet = args.dataset.lower() == 'imagenet'
    steps_per_epoch = None
    if is_imagenet:
        steps_per_epoch = 1281167 // args.batch_size

    # Create learning rate schedule
    use_lr_schedule = args.lr_schedule != 'constant'
    if use_lr_schedule:
        lr = create_learning_rate_schedule(
            initial_lr=args.learning_rate,
            schedule_type=args.lr_schedule,
            total_epochs=args.epochs,
            steps_per_epoch=steps_per_epoch
        )
    else:
        lr = args.learning_rate

    # Compile model
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=args.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    logger.info("\nModel Summary:")
    model.summary(print_fn=logger.info)

    # Create callbacks
    custom_objects = {"ConvNeXtV1": ConvNeXtV1, "ConvNextV1Block": ConvNextV1Block}

    callbacks, results_dir = create_callbacks(
        model_name=f"{args.variant}_{args.dataset}",
        results_dir_prefix="convnext_v1",
        monitor='val_accuracy',
        patience=args.patience,
        use_lr_schedule=use_lr_schedule
    )

    # Initialize visualization manager
    viz_manager = VisualizationManager(
        experiment_name="visualizations", output_dir=os.path.join(results_dir))

    # Train model
    logger.info("\nStarting training...")
    logger.info(f"  Dataset: {args.dataset}, Variant: {args.variant}")
    logger.info(f"  Input shape: {input_shape}, Classes: {num_classes}")
    logger.info(f"  Batch size: {args.batch_size}, Epochs: {args.epochs}")
    logger.info(f"  LR: {args.learning_rate}, Schedule: {args.lr_schedule}, Weight decay: {args.weight_decay}")

    if is_imagenet:
        history = model.fit(
            train_data,
            epochs=args.epochs,
            validation_data=test_data,
            callbacks=callbacks,
            verbose=1
        )
    else:
        x_train, y_train = train_data
        x_test, y_test = test_data

        history = model.fit(
            x_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

    # Validate model serialization
    logger.info("Validating model serialization...")
    if is_imagenet:
        for test_sample, _ in test_data.take(1):
            test_sample = test_sample[:4]
            break
    else:
        test_sample = x_test[:4]

    pre_save_output = model.predict(test_sample, verbose=0)

    # Save final model
    final_model_path = os.path.join(results_dir, f"convnext_v1_{args.dataset}_{args.variant}_final.keras")
    try:
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        is_loading_valid = validate_model_loading(final_model_path, test_sample, pre_save_output, custom_objects)
        if not is_loading_valid:
            logger.warning("Model loading validation failed - using current model for evaluation")
    except Exception as e:
        logger.warning(f"Failed to save final model: {e}")

    # Load best model for evaluation
    best_model_path = os.path.join(results_dir, 'best_model.keras')
    best_model = model
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from: {best_model_path}")
        try:
            is_loading_valid = validate_model_loading(best_model_path, test_sample, pre_save_output, custom_objects)
            if is_loading_valid:
                best_model = keras.models.load_model(best_model_path, custom_objects=custom_objects)
                logger.info("Successfully loaded best model from checkpoint")
            else:
                logger.warning("Best model loading validation failed - using current model")
        except Exception as e:
            logger.warning(f"Failed to load saved model: {e}")

    # Final evaluation
    logger.info("Evaluating final model on test set...")
    if is_imagenet:
        test_results = best_model.evaluate(test_data, verbose=1, return_dict=True)
    else:
        test_results = best_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results: {test_results}")

    # Generate predictions for visualization
    logger.info("Generating predictions for visualization...")

    if is_imagenet:
        logger.info("Note: Using validation subset for detailed analysis due to dataset size")
        y_true_list, y_pred_list, y_prob_list = [], [], []
        max_samples = 10000
        sample_count = 0

        for images, labels in test_data:
            if sample_count >= max_samples:
                break
            predictions = best_model.predict(images, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            y_true_list.append(labels.numpy())
            y_pred_list.append(pred_classes)
            y_prob_list.append(predictions)
            sample_count += len(labels)

        y_test_subset = np.concatenate(y_true_list)
        predicted_classes = np.concatenate(y_pred_list)
        all_predictions = np.concatenate(y_prob_list)
    else:
        all_predictions = best_model.predict(x_test, batch_size=args.batch_size, verbose=1)
        predicted_classes = np.argmax(all_predictions, axis=1)
        y_test_subset = y_test

    accuracy = np.mean(predicted_classes == y_test_subset)
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Visualizations
    training_history_viz = convert_keras_history_to_training_history(history)
    classification_results = create_classification_results(
        y_true=y_test_subset,
        y_pred=predicted_classes,
        y_prob=all_predictions,
        class_names=class_names if not is_imagenet else class_names[:10],
        model_name=f"{args.variant}_{args.dataset}"
    )

    generate_comprehensive_visualizations(
        viz_manager=viz_manager,
        training_history=training_history_viz,
        classification_results=classification_results,
        model=best_model,
        show_plots=args.show_plots,
    )

    # Run model analysis
    if is_imagenet:
        analysis_test_data = (test_data, y_test_subset)
    else:
        analysis_test_data = (x_test, y_test)

    run_model_analysis(
        model=best_model,
        test_data=analysis_test_data,
        training_history=history,
        model_name=f"convnext_v1_{args.variant}_{args.dataset}",
        results_dir=results_dir
    )

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"ConvNeXt V1 Training Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Variant: {args.variant}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Number of Classes: {num_classes}\n")

        if hasattr(model, 'depths'):
            f.write(f"\nModel Configuration:\n")
            f.write(f"  Depths: {model.depths}\n")
            f.write(f"  Dimensions: {model.dims}\n")
            f.write(f"  Drop Path Rate: {model.drop_path_rate}\n")

        f.write(f"\nTraining Configuration:\n")
        f.write(f"  Epochs: {len(history.history['loss'])}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Initial Learning Rate: {args.learning_rate}\n")
        f.write(f"  LR Schedule: {args.lr_schedule}\n")
        f.write(f"  Weight Decay: {args.weight_decay}\n")

        f.write(f"\nFinal Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

        f.write(f"\nAccuracy: {accuracy:.4f}\n")

        f.write(f"\nTraining History:\n")
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        f.write(f"  Final Training Accuracy: {final_train_acc:.4f}\n")
        f.write(f"  Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"  Best Validation Accuracy: {best_val_acc:.4f}\n")

    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {results_dir}")


# ---------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ConvNeXt V1 model with ImageNet support.')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size for ImageNet (default: 224)')
    parser.add_argument('--variant', type=str, default='tiny',
                        choices=['cifar10', 'tiny', 'small', 'base', 'large', 'xlarge'],
                        help='Model variant')
    parser.add_argument('--kernel-size', type=int, default=7,
                        help='Depthwise kernel size')
    parser.add_argument('--strides', type=int, default=4,
                        help='Downsampling strides')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'exponential', 'constant'],
                        help='Learning rate schedule')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Show plots interactively')

    args = parser.parse_args()

    if args.dataset.lower() == 'imagenet':
        if args.batch_size == 64:
            logger.info("Note: Using default batch size 64. Consider 128-256 for ImageNet.")
        if args.learning_rate == 1e-3:
            logger.info("Note: Using default lr 1e-3. Consider 4e-3 for ImageNet.")

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()
