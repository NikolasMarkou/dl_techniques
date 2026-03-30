import os
import keras
import argparse
import numpy as np
from typing import Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2, create_convnext_v2

from dl_techniques.visualization import (
    VisualizationManager,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    NetworkArchitectureVisualization,
    ModelComparisonBarChart,
    ROCPRCurves
)
from dl_techniques.analyzer import AnalysisConfig

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

def create_model_config(dataset: str, variant: str, input_shape: Tuple[int, int, int], num_classes: int) -> Dict[
    str, Any]:
    """Create ConvNeXt V2 model configuration based on dataset."""
    config = {
        'include_top': True,
        'kernel_regularizer': None,
    }

    if dataset.lower() == 'mnist':
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

def setup_visualization_manager(experiment_name: str, results_dir: str) -> VisualizationManager:
    """Setup and configure the visualization manager."""
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    config = PlotConfig(
        style=PlotStyle.PUBLICATION,
        color_scheme=ColorScheme(
            primary="#2E86AB",
            secondary="#A23B72",
            success="#06D6A0",
            warning="#FFD166"
        ),
        title_fontsize=14,
        label_fontsize=12,
        save_format="png",
        dpi=300,
        fig_size=(12, 8)
    )

    viz_manager = VisualizationManager(
        experiment_name=experiment_name,
        output_dir=viz_dir,
        config=config
    )

    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("network_architecture", NetworkArchitectureVisualization)
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
    viz_manager.register_template("roc_pr_curves", ROCPRCurves)

    logger.info(f"Visualization manager setup complete. Plots will be saved to: {viz_dir}")
    return viz_manager


# ---------------------------------------------------------------------

def train_model(args: argparse.Namespace):
    """Main training function with comprehensive visualization and analysis."""
    logger.info("Starting ConvNeXt V2 training script with comprehensive visualizations")
    setup_gpu()

    # Load dataset
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(args.dataset)
    class_names = get_class_names(args.dataset, num_classes)

    logger.info(f"After load_dataset:")
    logger.info(f"  input_shape = {input_shape} (type: {type(input_shape)})")
    logger.info(f"  num_classes = {num_classes} (type: {type(num_classes)})")
    logger.info(f"  class_names = {class_names[:5]}..." if len(class_names) > 5 else f"  class_names = {class_names}")

    # Create model configuration
    model_config = create_model_config(args.dataset, args.variant, input_shape, num_classes)

    # Create learning rate schedule
    use_lr_schedule = args.lr_schedule != 'constant'
    lr_schedule = create_learning_rate_schedule(
        initial_lr=args.learning_rate,
        schedule_type=args.lr_schedule,
        total_epochs=args.epochs
    )

    # Create model
    logger.info(f"Creating ConvNeXt V2 model (variant: {args.variant})...")

    model = create_convnext_v2(
        variant=args.variant,
        num_classes=num_classes,
        input_shape=input_shape,
        strides=args.strides,
        kernel_size=args.kernel_size,
        **{k: v for k, v in model_config.items() if k not in ['num_classes']}
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
        clipnorm=1.0
    )

    metrics = ['accuracy']
    if num_classes > 10:
        metrics.append('top_5_accuracy')

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics
    )

    # Build model and show summary
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    try:
        output = model(dummy_input, training=False)
        logger.info(f"Model built successfully. Output shape: {output.shape}")
    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise e

    model.summary(print_fn=logger.info)

    # Create callbacks and results directory
    custom_objects = {"ConvNeXtV2": ConvNeXtV2, "ConvNextV2Block": ConvNextV2Block}

    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.variant}",
        results_dir_prefix="convnext_v2",
        use_lr_schedule=use_lr_schedule,
        patience=args.patience
    )

    # Setup visualization manager
    experiment_name = f"convnext_v2_{args.dataset}_{args.variant}"
    viz_manager = setup_visualization_manager(experiment_name, results_dir)

    # Train model
    logger.info("Starting model training...")
    logger.info(f"  Dataset: {args.dataset}, Variant: {args.variant}")
    logger.info(f"  Input shape: {input_shape}, LR: {args.learning_rate}, Schedule: {args.lr_schedule}")
    logger.info(f"  Batch size: {args.batch_size}, Epochs: {args.epochs}, Weight decay: {args.weight_decay}")

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
    test_sample = x_test[:4]
    pre_save_output = model.predict(test_sample, verbose=0)

    # Save final model
    final_model_path = os.path.join(results_dir, f"convnext_v2_{args.dataset}_{args.variant}_final.keras")
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
    test_results = best_model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1, return_dict=True)
    logger.info(f"Final Test Results: {test_results}")

    # Generate predictions for visualization
    all_predictions = best_model.predict(x_test, batch_size=args.batch_size, verbose=1)
    predicted_classes = np.argmax(all_predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_test)
    logger.info(f"Manually calculated accuracy: {accuracy:.4f}")

    # Visualizations
    training_history_viz = convert_keras_history_to_training_history(history)
    classification_results = create_classification_results(
        y_true=y_test,
        y_pred=predicted_classes,
        y_prob=all_predictions,
        class_names=class_names,
        model_name=f"{args.variant}_{args.dataset}"
    )

    generate_comprehensive_visualizations(
        viz_manager=viz_manager,
        training_history=training_history_viz,
        classification_results=classification_results,
        model=best_model,
        show_plots=args.show_plots,
        max_roc_classes=20,
    )

    # Run model analysis
    analysis_config = AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        analyze_training_dynamics=True,
        analyze_spectral=True,
        save_plots=True,
        save_format='png',
        dpi=300,
        plot_style='publication'
    )

    analysis_results = run_model_analysis(
        model=best_model,
        test_data=(x_test, y_test),
        training_history=history,
        model_name=f"convnext_v2_{args.variant}_{args.dataset}",
        results_dir=results_dir,
        config=analysis_config,
    )

    # Log key findings from analysis
    if analysis_results:
        if hasattr(analysis_results, 'training_metrics') and analysis_results.training_metrics:
            training_metrics = analysis_results.training_metrics
            if hasattr(training_metrics, 'peak_performance'):
                peak_acc = training_metrics.peak_performance.get(
                    f"convnext_v2_{args.variant}_{args.dataset}", {}
                ).get('val_accuracy')
                if peak_acc:
                    logger.info(f"Peak validation accuracy: {peak_acc:.4f}")

            if hasattr(training_metrics, 'overfitting_index'):
                overfit_idx = training_metrics.overfitting_index.get(
                    f"convnext_v2_{args.variant}_{args.dataset}"
                )
                if overfit_idx:
                    logger.info(f"Overfitting index: {overfit_idx:.4f}")

    # Save training summary
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"ConvNeXt V2 Training Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Variant: {args.variant}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")

        if hasattr(model, 'depths'):
            f.write(f"Model Depths: {model.depths}\n")
            f.write(f"Model Dimensions: {model.dims}\n")
            f.write(f"Drop Path Rate: {model.drop_path_rate}\n")

        f.write(f"Training Epochs: {len(history.history['loss'])}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Initial Learning Rate: {args.learning_rate}\n")
        f.write(f"LR Schedule: {args.lr_schedule}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n\n")

        f.write(f"Final Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key.replace('_', ' ').title()}: {val:.4f}\n")

        f.write(f"\nManually Calculated Accuracy: {accuracy:.4f}\n")

        f.write(f"\nTraining History Summary:\n")
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
    parser = argparse.ArgumentParser(description='Train a ConvNeXt V2 model with comprehensive visualizations.')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--variant', type=str, default='cifar10',
                        choices=['cifar10', 'tiny', 'small', 'base', 'large', 'xlarge'], help='Model variant')
    parser.add_argument('--kernel-size', type=int, default=7, help='Depthwise kernel size')
    parser.add_argument('--strides', type=int, default=4, help='Downsampling strides')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'exponential', 'constant'], help='Learning rate schedule')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help='Show plots interactively during execution')

    args = parser.parse_args()

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
