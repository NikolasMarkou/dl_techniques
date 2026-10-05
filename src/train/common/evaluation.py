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
    PlotStyle,
    ColorScheme,
    TrainingHistory,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    NetworkArchitectureVisualization,
    ModelComparisonBarChart,
    ROCPRCurves,
    ClassificationResults,
    RegressionResults,
    MultiModelRegression,
    TimeSeriesEvaluationResults,
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

        if isinstance(expected_output, dict) or isinstance(loaded_output, dict):
            if not (isinstance(expected_output, dict) and isinstance(loaded_output, dict)):
                logger.warning(
                    "Model output structure mismatch after loading "
                    f"(expected={type(expected_output).__name__}, "
                    f"loaded={type(loaded_output).__name__})"
                )
                return False
            if set(expected_output.keys()) != set(loaded_output.keys()):
                logger.warning(
                    "Model output keys differ after loading "
                    f"(expected={sorted(expected_output.keys())}, "
                    f"loaded={sorted(loaded_output.keys())})"
                )
                return False
            max_diff = 0.0
            for k in expected_output:
                diff = np.max(np.abs(
                    np.asarray(loaded_output[k]) - np.asarray(expected_output[k])
                ))
                max_diff = max(max_diff, float(diff))
            if max_diff <= tolerance:
                logger.info("Model loading validation passed")
                return True
            logger.warning(
                f"Model outputs differ after loading (max_diff={max_diff:.6f})"
            )
            return False

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

def _as_prediction_pair(y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten two prediction arrays to ``(N,)`` and assert they agree in length.

    Every ``ClassificationResults`` / ``RegressionResults`` consumer indexes
    ``y_true[i]`` against ``y_pred[i]``, so a length mismatch is a silent
    truncation once a boolean mask is applied to one of them. Assert here,
    where the message can name both shapes.
    """
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length; got "
            f"{yt.shape[0]} and {yp.shape[0]}."
        )
    if yt.shape[0] == 0:
        raise ValueError("y_true and y_pred are both empty; nothing to visualize.")
    return yt, yp


def create_classification_results(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        model_name: Optional[str] = None,
) -> ClassificationResults:
    """Create ClassificationResults object for visualization.

    Args:
        y_true: True labels, any shape; flattened to ``(N,)``.
        y_pred: Predicted labels, flattened to ``(N,)``. Must match
            ``y_true`` in length.
        y_prob: Optional class probabilities. **Omitting it is a supported
            mode, not a degraded one** — several classification plugins
            degrade gracefully without it (``PerClassAnalysis`` renders a
            ``'No probability data'`` panel) but ``ROCPRCurves`` does not:
            its per-model loop is ``if results.y_prob is None: continue``,
            so it builds an EMPTY figure and saves a blank PNG without
            raising. Never register ``roc_pr_curves`` for a model built
            without probabilities. Pinned by
            ``tests/test_train/test_common_evaluation_results.py``.
        class_names: One name per class.
        model_name: Display name used in figure titles.
    """
    yt, yp = _as_prediction_pair(y_true, y_pred)
    return ClassificationResults(
        y_true=yt,
        y_pred=yp,
        y_prob=y_prob,
        class_names=class_names,
        model_name=model_name,
    )


def create_regression_results(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
) -> RegressionResults:
    """Create ``RegressionResults``, the input every regression plugin requires.

    ``dl_techniques.visualization.regression`` ships six plugins —
    ``prediction_error``, ``residuals_plot``, ``residual_distribution``,
    ``qq_plot``, ``regression_evaluation_dashboard`` and the
    ``MultiModelRegression`` comparison dashboard — and every one of them
    type-checks its input with ``isinstance(data, RegressionResults)``.
    Nothing in the tree constructed this container, which is why all six
    were unreachable; this is that constructor.

    Args:
        y_true: Ground-truth targets, flattened to ``(N,)``.
        y_pred: Predicted targets, flattened to ``(N,)``. Must match
            ``y_true`` in length.
        model_name: Display name used in figure titles.
        feature_names: Optional per-feature names.

    Returns:
        A ``RegressionResults`` carrying the flattened pair.
    """
    yt, yp = _as_prediction_pair(y_true, y_pred)
    return RegressionResults(
        y_true=yt,
        y_pred=yp,
        model_name=model_name,
        feature_names=feature_names,
    )


def create_multi_model_regression(
        results: Dict[str, Any],
        dataset_name: Optional[str] = None,
) -> MultiModelRegression:
    """Bundle per-model ``RegressionResults`` for the comparison dashboard.

    Args:
        results: ``{model_name: RegressionResults}``, or ``{model_name:
            (y_true, y_pred)}`` which is passed through
            :func:`create_regression_results`. Mixing the two is allowed.
        dataset_name: Optional dataset label for the figure title.
    """
    built = {
        name: (
            value
            if isinstance(value, RegressionResults)
            else create_regression_results(value[0], value[1], model_name=name)
        )
        for name, value in results.items()
    }
    if not built:
        raise ValueError("results must contain at least one model.")
    return MultiModelRegression(results=built, dataset_name=dataset_name)


def create_timeseries_results(
        all_inputs: np.ndarray,
        all_true_forecasts: np.ndarray,
        all_predicted_forecasts: Optional[np.ndarray] = None,
        all_predicted_quantiles: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
        quantile_levels: Optional[List[float]] = None,
) -> TimeSeriesEvaluationResults:
    """Create ``TimeSeriesEvaluationResults``, the input ``ForecastVisualization`` requires.

    ``ForecastVisualization`` is the only consumer of this container and it
    type-checks with ``isinstance(data, TimeSeriesEvaluationResults)``, so
    without a constructor it was unreachable from the tree as well.

    Shape contract, taken from the container's own annotations:
    ``all_inputs`` is ``(num_samples, input_length)`` and
    ``all_true_forecasts`` is ``(num_samples, forecast_length)``. The two
    must agree on ``num_samples`` — the plugin indexes forecasts by sample
    index — and the forecast axis is kept as-is rather than squeezed,
    because a ``forecast_length`` of 1 must stay two-dimensional.

    A **1-D array on either axis describes ONE window** and is promoted to a
    batch of one, so ``(10,)`` inputs with ``(3,)`` forecasts is the single
    window that predicts three steps ahead. The alternative reading —
    ``(3,)`` meaning three samples each one step ahead — is deliberately NOT
    taken: it would make the promotion rule differ between the two axes, so
    the same call site would mean different things depending on which array
    it touched. Pass ``(N, 1)`` explicitly for N single-step samples.

    Args:
        all_inputs: Windowed history, ``(num_samples, input_length)``.
        all_true_forecasts: Ground-truth futures,
            ``(num_samples, forecast_length)``.
        all_predicted_forecasts: Optional point forecasts, same shape as
            ``all_true_forecasts``.
        all_predicted_quantiles: Optional quantile forecasts,
            ``(num_samples, forecast_length, len(quantile_levels))``.
        model_name: Display name used in figure titles.
        quantile_levels: The levels ``all_predicted_quantiles`` was built
            at. Supplying quantiles without levels is an error rather than
            a silent no-band, because the plugin needs the levels to draw
            the legend.
    """
    inputs = np.asarray(all_inputs)
    true_forecasts = np.asarray(all_true_forecasts)

    # A 1-D array is ONE window on either axis — see the docstring's shape contract.
    if inputs.ndim == 1:
        inputs = inputs[None, :]
    if true_forecasts.ndim == 1:
        true_forecasts = true_forecasts[None, :]

    if inputs.ndim != 2 or true_forecasts.ndim != 2:
        raise ValueError(
            f"all_inputs and all_true_forecasts must be 2-D after promotion; got "
            f"{inputs.ndim}-D and {true_forecasts.ndim}-D."
        )
    if inputs.shape[0] != true_forecasts.shape[0]:
        raise ValueError(
            f"all_inputs and all_true_forecasts must agree on num_samples; got "
            f"{inputs.shape[0]} and {true_forecasts.shape[0]}."
        )
    if inputs.shape[0] == 0:
        raise ValueError("all_inputs is empty; nothing to visualize.")

    predicted = None
    if all_predicted_forecasts is not None:
        predicted = np.asarray(all_predicted_forecasts)
        if predicted.ndim == 1:
            predicted = predicted[None, :]
        if predicted.shape != true_forecasts.shape:
            raise ValueError(
                f"all_predicted_forecasts must match all_true_forecasts "
                f"({true_forecasts.shape}); got {predicted.shape}."
            )

    quantiles = None
    if all_predicted_quantiles is not None:
        quantiles = np.asarray(all_predicted_quantiles)
        if not quantile_levels:
            raise ValueError(
                "quantile_levels is required when all_predicted_quantiles is "
                "given; without them the uncertainty bands cannot be labelled."
            )
        if quantiles.shape[:2] != true_forecasts.shape:
            raise ValueError(
                f"all_predicted_quantiles must lead with all_true_forecasts' "
                f"shape ({true_forecasts.shape}); got {quantiles.shape}."
            )
        if quantiles.shape[2] != len(quantile_levels):
            raise ValueError(
                f"all_predicted_quantiles has {quantiles.shape[2]} quantile "
                f"levels but {len(quantile_levels)} were declared."
            )

    return TimeSeriesEvaluationResults(
        all_inputs=inputs,
        all_true_forecasts=true_forecasts,
        all_predicted_forecasts=predicted,
        all_predicted_quantiles=quantiles,
        model_name=model_name,
        quantile_levels=quantile_levels,
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
        experiment_name="",
        output_dir=results_dir,
        config=plot_config,
        timestamp="",
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


# ---------------------------------------------------------------------

def setup_visualization_manager(
        experiment_name: str,
        results_dir: str,
) -> VisualizationManager:
    """Set up and configure a publication-style visualization manager.

    Creates a ``visualizations`` subdirectory under ``results_dir`` and returns
    a :class:`VisualizationManager` pre-registered with the five standard
    vision-classification templates (training curves, confusion matrix, network
    architecture, model-comparison bars, ROC/PR curves).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment, used to namespace the visualization manager.
    results_dir : str
        Base results directory. Plots are written under
        ``{results_dir}/visualizations``.

    Returns
    -------
    VisualizationManager
        Configured manager with the five standard templates registered.
    """
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
