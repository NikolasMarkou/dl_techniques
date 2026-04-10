"""Common evaluation, visualization, and analysis utilities for training scripts."""

import os
import keras
import numpy as np
import tensorflow as tf
from typing import Tuple, Any, List, Dict, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.visualization import (
    VisualizationManager,
    PlotConfig,
    TrainingHistory,
    TrainingCurvesVisualization,
    ClassificationResults,
)
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput


# ---------------------------------------------------------------------

def validate_model_loading(
        model_path: str,
        test_sample: Any,
        expected_output: np.ndarray,
        custom_objects: Optional[Dict[str, Any]] = None,
        tolerance: float = 1e-5,
) -> bool:
    """
    Validate that a saved model loads correctly and produces expected outputs.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    test_sample : Any
        Input sample for prediction (numpy array or tensor).
    expected_output : np.ndarray
        Expected output from the original model.
    custom_objects : Optional[Dict[str, Any]]
        Custom objects dict for model loading.
    tolerance : float
        Absolute tolerance for output comparison.

    Returns
    -------
    bool
        True if model loads correctly and outputs match.
    """
    try:
        loaded_model = keras.models.load_model(
            model_path, custom_objects=custom_objects or {}
        )

        if isinstance(test_sample, tf.Tensor):
            test_sample = test_sample.numpy()

        loaded_output = loaded_model.predict(test_sample, verbose=0)

        if np.allclose(expected_output, loaded_output, atol=tolerance):
            logger.info("Model loading validation passed")
            return True
        else:
            max_diff = np.max(np.abs(loaded_output - expected_output))
            logger.warning(f"Model outputs differ after loading (max_diff={max_diff:.6f})")
            return False

    except Exception as e:
        logger.warning(f"Model loading validation failed: {e}")
        return False


# ---------------------------------------------------------------------

def convert_keras_history_to_training_history(
        keras_history: keras.callbacks.History,
) -> TrainingHistory:
    """Convert Keras training history to visualization framework TrainingHistory."""
    history_dict = keras_history.history
    epochs = list(range(len(history_dict['loss'])))

    train_metrics = {}
    val_metrics = {}

    for key, values in history_dict.items():
        if key.startswith('val_') and key != 'val_loss':
            val_metrics[key.replace('val_', '')] = values
        elif not key.startswith('val_') and key != 'loss':
            train_metrics[key] = values

    return TrainingHistory(
        epochs=epochs,
        train_loss=history_dict['loss'],
        val_loss=history_dict.get('val_loss', []),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )


# ---------------------------------------------------------------------

def create_classification_results(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        model_name: str,
) -> ClassificationResults:
    """Create ClassificationResults object for visualization."""
    return ClassificationResults(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        model_name=model_name,
    )


# ---------------------------------------------------------------------

def generate_training_curves(
        history,
        results_dir: str,
        filename: str = "training_curves",
) -> None:
    """
    Generate training curve plots using the visualization framework.

    Accepts either a Keras History object or a raw dict of metric lists
    (as used by in-training callbacks that track metrics manually).

    Parameters
    ----------
    history : keras.callbacks.History or Dict[str, List[float]]
        Keras training history from model.fit(), or a dict mapping
        metric names to lists of per-epoch values. Dict keys follow
        Keras convention: 'loss', 'val_loss', 'metric_name',
        'val_metric_name'.
    results_dir : str
        Directory to save the plot.
    filename : str
        Filename for the saved plot (without extension).
    """
    if isinstance(history, dict):
        history_dict = history
    else:
        history_dict = history.history

    epochs = list(range(len(history_dict['loss'])))
    train_metrics = {}
    val_metrics = {}
    for key, values in history_dict.items():
        if key == 'loss' or key == 'val_loss':
            continue
        if key.startswith('val_'):
            val_metrics[key[4:]] = values
        else:
            train_metrics[key] = values

    training_history = TrainingHistory(
        epochs=epochs,
        train_loss=history_dict['loss'],
        val_loss=history_dict.get('val_loss'),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )

    plot_config = PlotConfig(fig_size=(14, 10), save_dpi=150)
    viz_manager = VisualizationManager(
        experiment_name="training",
        output_dir=results_dir,
        config=plot_config,
    )
    viz_manager.register_plugin(TrainingCurvesVisualization(
        config=plot_config,
        context=viz_manager.context,
    ))
    viz_manager.visualize(
        data=training_history,
        plugin_name="training_curves",
        show=False,
        filename=filename,
    )


# ---------------------------------------------------------------------

def generate_comprehensive_visualizations(
        viz_manager: VisualizationManager,
        training_history: TrainingHistory,
        classification_results: ClassificationResults,
        model: keras.Model,
        show_plots: bool = False,
        max_roc_classes: int = 20,
) -> None:
    """
    Generate comprehensive visualizations using the VisualizationManager.

    Parameters
    ----------
    viz_manager : VisualizationManager
        Configured visualization manager.
    training_history : TrainingHistory
        Training history data.
    classification_results : ClassificationResults
        Classification results with predictions.
    model : keras.Model
        The trained model.
    show_plots : bool
        Whether to show plots interactively.
    max_roc_classes : int
        Maximum number of classes to generate ROC/PR curves for.
    """
    logger.info("Generating comprehensive visualizations...")

    logger.info("  - Training curves")
    viz_manager.visualize(
        data=training_history,
        plugin_name="training_curves",
        show=show_plots,
        filename="training_curves",
    )

    logger.info("  - Confusion matrix")
    viz_manager.visualize(
        data=classification_results,
        plugin_name="confusion_matrix",
        show=show_plots,
        filename="confusion_matrix",
    )

    if len(classification_results.class_names) <= max_roc_classes:
        logger.info("  - ROC and PR curves")
        viz_manager.visualize(
            data=classification_results,
            plugin_name="roc_pr_curves",
            show=show_plots,
            filename="roc_pr_curves",
        )

    logger.info("  - Network architecture")
    viz_manager.visualize(
        data=model,
        plugin_name="network_architecture",
        show=show_plots,
        filename="architecture",
    )

    logger.info("Visualizations generated successfully")


# ---------------------------------------------------------------------

def run_model_analysis(
        model: keras.Model,
        test_data: Tuple[Any, np.ndarray],
        training_history: keras.callbacks.History,
        model_name: str,
        results_dir: str,
        config: Optional[AnalysisConfig] = None,
) -> Any:
    """
    Run comprehensive model analysis using ModelAnalyzer.

    Parameters
    ----------
    model : keras.Model
        Trained model to analyze.
    test_data : Tuple[Any, np.ndarray]
        Test data as (x_test, y_test). x_test can be numpy array or tf.data.Dataset.
    training_history : keras.callbacks.History
        Keras training history.
    model_name : str
        Name identifier for the model.
    results_dir : str
        Directory to save analysis results.
    config : Optional[AnalysisConfig]
        Analysis configuration. Uses sensible defaults if None.

    Returns
    -------
    Analysis results or None if analysis fails.
    """
    logger.info("Running model analysis...")

    if config is None:
        config = AnalysisConfig(
            analyze_weights=True,
            analyze_calibration=True,
            analyze_information_flow=True,
            analyze_training_dynamics=True,
            analyze_spectral=True,
        )

    try:
        # Extract numpy arrays from tf.data.Dataset if needed
        if isinstance(test_data[0], tf.data.Dataset):
            logger.info("Extracting subset from tf.data.Dataset for analysis...")
            for batch_images, batch_labels in test_data[0].take(1):
                x_test_subset = batch_images.numpy()
                y_test_subset = batch_labels.numpy()
                break
        else:
            x_test_subset = test_data[0][:1000]
            y_test_subset = test_data[1][:1000]

        analyzer = ModelAnalyzer(
            models={model_name: model},
            config=config,
            output_dir=os.path.join(results_dir, "model_analysis"),
            training_history={model_name: training_history.history},
        )

        data = DataInput(x_data=x_test_subset, y_data=y_test_subset)
        analysis_results = analyzer.analyze(data=data)

        logger.info("Model analysis completed successfully")
        return analysis_results

    except Exception as e:
        logger.warning(f"Model analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None
