"""
Model Analyzer Main Module

A comprehensive model analysis toolkit that orchestrates multiple analysis types including
weight analysis, calibration analysis, information flow analysis, and training dynamics
analysis. Provides automated visualization generation and result serialization.

Main coordinator class that orchestrates all analysis and visualization components.
"""

import json
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Set

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .config import AnalysisConfig
from .data_types import DataInput, AnalysisResults
from .utils import find_pareto_front, normalize_metric

from .analyzers.weight_analyzer import WeightAnalyzer
from .analyzers.calibration_analyzer import CalibrationAnalyzer
from .analyzers.information_flow_analyzer import InformationFlowAnalyzer
from .analyzers.training_dynamics_analyzer import TrainingDynamicsAnalyzer

from .visualizers.weight_visualizer import WeightVisualizer
from .visualizers.summary_visualizer import SummaryVisualizer
from .visualizers.calibration_visualizer import CalibrationVisualizer
from .visualizers.information_flow_visualizer import InformationFlowVisualizer
from .visualizers.training_dynamics_visualizer import TrainingDynamicsVisualizer

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

# Time formatting
TIMESTAMP_FORMAT: str = '%Y%m%d_%H%M%S'

# Color palette for consistent model visualization (avoiding yellow for readability)
MODEL_COLOR_PALETTE: List[str] = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b'   # Brown
]

# Analysis type identifiers
ANALYSIS_TYPE_WEIGHTS: str = 'weights'
ANALYSIS_TYPE_CALIBRATION: str = 'calibration'
ANALYSIS_TYPE_INFORMATION_FLOW: str = 'information_flow'
ANALYSIS_TYPE_TRAINING_DYNAMICS: str = 'training_dynamics'

# Analysis types that require input data to function
DATA_REQUIRED_ANALYSES: Set[str] = {ANALYSIS_TYPE_CALIBRATION, ANALYSIS_TYPE_INFORMATION_FLOW}

# File naming constants
DEFAULT_RESULTS_FILENAME: str = "analysis_results.json"

# Metric name patterns for evaluation standardization
ACCURACY_METRIC_PATTERNS: List[str] = [
    'compile_metrics',
    'categorical_accuracy',
    'sparse_categorical_accuracy'
]

# Default directory naming
OUTPUT_DIR_PREFIX: str = "analysis_"

# Status indicators for model processing
STATUS_SUCCESS: str = 'success'
STATUS_MULTI_INPUT_SKIPPED: str = 'multi_input_skipped'
STATUS_EVALUATION_FAILED: str = 'evaluation_failed'
STATUS_ERROR: str = 'error'
STATUS_SKIPPED_MULTI_INPUT: str = 'skipped_multi_input'
STATUS_UNKNOWN: str = 'unknown'

# Cache entry keys
CACHE_KEY_PREDICTIONS: str = 'predictions'
CACHE_KEY_X_DATA: str = 'x_data'
CACHE_KEY_Y_DATA: str = 'y_data'
CACHE_KEY_MULTI_INPUT: str = 'multi_input'
CACHE_KEY_STATUS: str = 'status'
CACHE_KEY_ERROR: str = 'error'

# Visualization parameters
PARETO_SCATTER_SIZE: int = 100
PARETO_STAR_SIZE: int = 200
PARETO_LINE_WIDTH: int = 2
PARETO_ALPHA: float = 0.7
PARETO_STAR_ALPHA: float = 0.5

# Default metric values
DEFAULT_METRIC_VALUE: float = 0.0

# JSON serialization parameters
JSON_INDENT: int = 2

# ---------------------------------------------------------------------

class ModelAnalyzer:
    """
    Model analyzer with training dynamics and improved visualizations.

    This is the main coordinator class that orchestrates all analysis and
    visualization components. It supports multiple types of analysis including
    weight distribution analysis, model calibration assessment, information
    flow analysis, and training dynamics evaluation.

    The analyzer is designed to handle both single-input and multi-input models,
    with special handling and warnings for multi-input architectures that may
    have limited compatibility with certain analyses.

    Attributes:
        models (Dict[str, keras.Model]): Dictionary mapping model names to Keras models
        config (AnalysisConfig): Configuration object controlling analysis behavior
        output_dir (Path): Directory where results and visualizations are saved
        results (AnalysisResults): Container for all analysis results
        model_colors (Dict[str, str]): Consistent color mapping for visualizations
        analyzers (Dict[str, Any]): Dictionary of analysis modules
    """

    def __init__(
        self,
        models: Dict[str, keras.Model],
        config: Optional[AnalysisConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
        training_history: Optional[Dict[str, Dict[str, List[float]]]] = None
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            models: Dictionary mapping model names to Keras models. At least one model
                   must be provided.
            config: Analysis configuration object. If None, default configuration is used.
            output_dir: Output directory for plots and results. If None, creates timestamped
                       directory automatically.
            training_history: Optional training history for each model, used for training
                            dynamics analysis. Should map model names to dictionaries
                            containing metric histories.

        Raises:
            ValueError: If no models are provided in the models dictionary.
        """
        if not models:
            raise ValueError("At least one model must be provided")

        self.models: Dict[str, keras.Model] = models
        self.config: AnalysisConfig = config or AnalysisConfig()

        # Create output directory with timestamp if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            output_dir = f"{OUTPUT_DIR_PREFIX}{timestamp}"
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style based on configuration
        self.config.setup_plotting_style()

        # Initialize results container
        self.results: AnalysisResults = AnalysisResults(config=self.config)

        # Store training history if provided for training dynamics analysis
        if training_history:
            self.results.training_history = training_history

        # Cache for storing model predictions to avoid redundant computation
        # Structure: {model_name: {predictions: array, x_data: array, y_data: array, ...}}
        self._prediction_cache: Dict[str, Dict[str, Any]] = {}

        # Track multi-input models for special handling
        # Multi-input models require different data handling and may have limited analysis support
        self._multi_input_models: Set[str] = set()
        self._identify_multi_input_models()

        # Initialize model colors for consistent visualization across all plots
        self.model_colors: Dict[str, str] = {}
        self._setup_model_colors()

        # Initialize all analyzer modules
        self._init_analyzers()

        logger.info(f"ModelAnalyzer initialized with {len(models)} models")
        if self._multi_input_models:
            logger.info(f"Detected multi-input models: {self._multi_input_models}")

    def _identify_multi_input_models(self) -> None:
        """
        Identify models with multiple inputs for special handling.

        Multi-input models require different data preprocessing and may not be
        compatible with all analysis types. This method identifies such models
        early to provide appropriate warnings and handling.
        """
        for model_name, model in self.models.items():
            try:
                # Check if model has multiple input tensors
                if hasattr(model, 'inputs') and len(model.inputs) > 1:
                    self._multi_input_models.add(model_name)
                    logger.debug(f"Model {model_name} identified as multi-input ({len(model.inputs)} inputs)")
            except Exception as e:
                logger.debug(f"Could not determine input structure for {model_name}: {e}")

    def _setup_model_colors(self) -> None:
        """
        Set up consistent colors for models across all visualizations.

        Assigns colors from a predefined palette to ensure consistent
        representation of models across different plots and analyses.
        Colors are assigned alphabetically by model name for consistency.
        """
        # Sort model names for consistent color assignment across runs
        model_names = sorted(self.models.keys())

        # Cycle through color palette if we have more models than colors
        color_palette = MODEL_COLOR_PALETTE[:len(model_names)]
        if len(model_names) > len(MODEL_COLOR_PALETTE):
            # Extend palette by repeating colors if necessary
            color_palette = (MODEL_COLOR_PALETTE * ((len(model_names) // len(MODEL_COLOR_PALETTE)) + 1))[:len(model_names)]

        self.model_colors = dict(zip(model_names, color_palette))

    def _init_analyzers(self) -> None:
        """
        Initialize all analyzer instances.

        Creates instances of all available analyzers with the current
        models and configuration. Each analyzer is responsible for
        one type of analysis (weights, calibration, etc.).
        """
        self.analyzers: Dict[str, Any] = {
            ANALYSIS_TYPE_WEIGHTS: WeightAnalyzer(self.models, self.config),
            ANALYSIS_TYPE_CALIBRATION: CalibrationAnalyzer(self.models, self.config),
            ANALYSIS_TYPE_INFORMATION_FLOW: InformationFlowAnalyzer(self.models, self.config),
            ANALYSIS_TYPE_TRAINING_DYNAMICS: TrainingDynamicsAnalyzer(self.models, self.config)
        }

    def analyze(
        self,
        data: Optional[Union[DataInput, tuple, Any]] = None,
        analysis_types: Optional[Set[str]] = None
    ) -> AnalysisResults:
        """
        Run comprehensive or selected analyses on models.

        This is the main entry point for running analyses. It coordinates
        data preprocessing, analysis execution, visualization creation,
        and result serialization.

        Args:
            data: Input data for analyses that require it. Can be DataInput object,
                 tuple of (x_data, y_data), or any object with x_data/y_data attributes.
                 Required for calibration and information flow analyses.
            analysis_types: Set of analysis types to run. If None, runs analyses
                          based on configuration flags. Valid types: 'weights',
                          'calibration', 'information_flow', 'training_dynamics'.

        Returns:
            AnalysisResults object containing all computed metrics, statistics,
            and analysis outputs.

        Raises:
            ValueError: If data is required for selected analyses but not provided.
        """
        # Determine which analyses to run based on config if not specified
        if analysis_types is None:
            analysis_types = {
                ANALYSIS_TYPE_WEIGHTS if self.config.analyze_weights else None,
                ANALYSIS_TYPE_CALIBRATION if self.config.analyze_calibration else None,
                ANALYSIS_TYPE_INFORMATION_FLOW if self.config.analyze_information_flow else None,
                ANALYSIS_TYPE_TRAINING_DYNAMICS if self.config.analyze_training_dynamics else None,
            }
            # Remove None values from the set
            analysis_types.discard(None)

        # Validate that data is provided for analyses that require it
        if analysis_types & DATA_REQUIRED_ANALYSES and data is None:
            raise ValueError(f"Data is required for: {analysis_types & DATA_REQUIRED_ANALYSES}")

        logger.info(f"Running analyses: {analysis_types}")

        # Convert data to standard DataInput format for consistent handling
        if data is not None:
            if isinstance(data, tuple):
                data = DataInput.from_tuple(data)
            elif not isinstance(data, DataInput):
                data = DataInput.from_object(data)

        # Cache model predictions if needed for data-dependent analyses
        # This avoids redundant computation across different analyses
        if data is not None and analysis_types & DATA_REQUIRED_ANALYSES:
            self._cache_predictions(data)

        # Run each selected analysis
        for analysis_type in analysis_types:
            if analysis_type in self.analyzers:
                analyzer = self.analyzers[analysis_type]

                # Check if analyzer requires data and skip if not available
                if analyzer.requires_data() and data is None:
                    logger.warning(f"Skipping {analysis_type} analysis - requires data")
                    continue

                # Special handling for multi-input models
                # Some analyses may have limited functionality with multi-input architectures
                if (analyzer.requires_data() and
                    analysis_type in [ANALYSIS_TYPE_CALIBRATION, ANALYSIS_TYPE_INFORMATION_FLOW] and
                    self._multi_input_models):
                    # Check if any models to be analyzed are multi-input
                    affected_models = (analysis_types & self._multi_input_models
                                     if hasattr(analysis_types, 'intersection')
                                     else self._multi_input_models)
                    if affected_models:
                        logger.warning(f"Limited {analysis_type} analysis for multi-input models: {affected_models}")

                # Execute the analysis
                analyzer.analyze(self.results, data, self._prediction_cache)

        # Generate visualizations for completed analyses
        self._create_visualizations(analysis_types)

        # Create comprehensive summary dashboard
        self.create_summary_dashboard()

        # Serialize and save all results
        self.save_results()

        return self.results

    def _cache_predictions(self, data: DataInput) -> None:
        """
        Cache model predictions to avoid redundant computation.

        Pre-computes and stores model predictions and evaluations to be reused
        across different analyses. Also handles sampling if dataset is too large.

        Args:
            data: Input data containing x_data and y_data for predictions.
        """
        x_data, y_data = data.x_data, data.y_data

        # Sample data if it exceeds configured maximum to maintain performance
        if len(x_data) > self.config.n_samples:
            indices = np.random.choice(len(x_data), self.config.n_samples, replace=False)
            x_data = x_data[indices]
            y_data = y_data[indices]

        logger.info("Caching model predictions...")

        # Compute predictions for each model
        for model_name, model in tqdm(self.models.items(), desc="Computing predictions"):
            try:
                # Handle multi-input models with clear logging
                if model_name in self._multi_input_models:
                    logger.warning(f"Model {model_name} has multiple inputs. "
                                   "Prediction caching is limited for multi-input architectures. "
                                   "Some analyses may be skipped or have reduced functionality.")

                    # Set cache to indicate multi-input status without attempting prediction
                    self._prediction_cache[model_name] = {
                        CACHE_KEY_PREDICTIONS: None,
                        CACHE_KEY_X_DATA: x_data,
                        CACHE_KEY_Y_DATA: y_data,
                        CACHE_KEY_MULTI_INPUT: True,
                        CACHE_KEY_STATUS: STATUS_SKIPPED_MULTI_INPUT
                    }
                    continue

                # Standard single-input model prediction
                predictions = model.predict(x_data, verbose=0)
                self._prediction_cache[model_name] = {
                    CACHE_KEY_PREDICTIONS: predictions,
                    CACHE_KEY_X_DATA: x_data,
                    CACHE_KEY_Y_DATA: y_data,
                    CACHE_KEY_MULTI_INPUT: False,
                    CACHE_KEY_STATUS: STATUS_SUCCESS
                }
            except Exception as e:
                logger.error(f"Failed to cache predictions for {model_name}: {e}")
                self._prediction_cache[model_name] = {
                    CACHE_KEY_PREDICTIONS: None,
                    CACHE_KEY_X_DATA: x_data,
                    CACHE_KEY_Y_DATA: y_data,
                    CACHE_KEY_ERROR: str(e),
                    CACHE_KEY_STATUS: STATUS_ERROR
                }

        # Evaluate models with improved error handling
        self._evaluate_models(x_data, y_data)

    def _evaluate_models(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Evaluate models and store performance metrics.

        Computes standard metrics (loss, accuracy) for each model and handles
        various edge cases including multi-input models and evaluation failures.

        Args:
            x_data: Input data for evaluation.
            y_data: Target data for evaluation.
        """
        for model_name, model in self.models.items():
            try:
                # Skip evaluation for multi-input models to avoid errors
                if model_name in self._multi_input_models:
                    logger.info(f"Skipping evaluation for multi-input model {model_name}")
                    self.results.model_metrics[model_name] = {
                        'loss': DEFAULT_METRIC_VALUE,
                        'accuracy': DEFAULT_METRIC_VALUE,
                        CACHE_KEY_STATUS: STATUS_MULTI_INPUT_SKIPPED
                    }
                    continue

                # Attempt standard model evaluation
                try:
                    metrics = model.evaluate(x_data, y_data, verbose=0)
                except (ValueError, TypeError) as eval_error:
                    logger.warning(f"Model evaluation failed for {model_name}: {eval_error}")
                    self.results.model_metrics[model_name] = {
                        'loss': DEFAULT_METRIC_VALUE,
                        'accuracy': DEFAULT_METRIC_VALUE,
                        CACHE_KEY_STATUS: STATUS_EVALUATION_FAILED,
                        CACHE_KEY_ERROR: str(eval_error)
                    }
                    continue

                # Process evaluation results
                metric_names = getattr(model, 'metrics_names', ['loss'])
                metric_dict = {CACHE_KEY_STATUS: STATUS_SUCCESS}

                # Handle both single metric and multiple metrics cases
                if isinstance(metrics, (list, tuple)):
                    for i, name in enumerate(metric_names):
                        if i < len(metrics):
                            metric_dict[name] = float(metrics[i])
                else:
                    # Single metric case
                    metric_dict[metric_names[0] if metric_names else 'loss'] = float(metrics)

                # Standardize accuracy metric naming by checking existing metric names
                for name in list(metric_dict.keys()):
                    if name in ACCURACY_METRIC_PATTERNS:
                        metric_dict['accuracy'] = metric_dict[name]
                        break  # Stop after finding the first suitable accuracy metric

                self.results.model_metrics[model_name] = metric_dict

            except Exception as e:
                logger.warning(f"Could not evaluate model {model_name}: {e}")
                self.results.model_metrics[model_name] = {
                    'loss': DEFAULT_METRIC_VALUE,
                    'accuracy': DEFAULT_METRIC_VALUE,
                    CACHE_KEY_STATUS: STATUS_ERROR,
                    CACHE_KEY_ERROR: str(e)
                }

    def _create_visualizations(self, analysis_types: Set[str]) -> None:
        """
        Create visualizations for completed analyses.

        Instantiates and runs appropriate visualizers for each analysis type
        that was executed. Each visualizer is responsible for creating
        its specific set of plots and charts.

        Args:
            analysis_types: Set of analysis types that were completed and need visualization.
        """
        # Mapping of analysis types to their corresponding visualizer classes
        visualizers = {
            ANALYSIS_TYPE_WEIGHTS: WeightVisualizer,
            ANALYSIS_TYPE_CALIBRATION: CalibrationVisualizer,
            ANALYSIS_TYPE_INFORMATION_FLOW: InformationFlowVisualizer,
            ANALYSIS_TYPE_TRAINING_DYNAMICS: TrainingDynamicsVisualizer
        }

        # Create visualizations for each completed analysis
        for analysis_type in analysis_types:
            if analysis_type in visualizers:
                visualizer_class = visualizers[analysis_type]
                visualizer = visualizer_class(
                    self.results, self.config, self.output_dir, self.model_colors
                )
                visualizer.create_visualizations()

    def create_summary_dashboard(self) -> None:
        """
        Create summary dashboard visualization.

        Generates a comprehensive overview dashboard that combines key insights
        from all completed analyses into a single, easy-to-interpret visualization.
        """
        visualizer = SummaryVisualizer(
            self.results, self.config, self.output_dir, self.model_colors
        )
        visualizer.create_visualizations()

    def save_results(self, filename: str = DEFAULT_RESULTS_FILENAME) -> None:
        """
        Save analysis results to JSON file.

        Serializes all analysis results into a JSON format for later review
        or programmatic access. Handles numpy arrays and pandas DataFrames
        appropriately for JSON serialization.

        Args:
            filename: Name of the output JSON file. Defaults to "analysis_results.json".
        """
        # Create serializable version of config (exclude matplotlib state)
        serializable_config = {
            k: v for k, v in self.config.__dict__.items()
            if k != '_original_rcParams'
        }

        # Compile all results into a single dictionary structure
        results_dict = {
            'timestamp': self.results.analysis_timestamp,
            'config': serializable_config,
            'model_metrics': self.results.model_metrics,
            'weight_stats': self.results.weight_stats,
            'weight_pca': self.results.weight_pca,
            'calibration_metrics': self.results.calibration_metrics,
            'confidence_metrics': self.results.confidence_metrics,
            'information_flow': self.results.information_flow,
            'activation_stats': self.results.activation_stats,
            'training_metrics': (self._serialize_training_metrics()
                               if self.results.training_metrics else None),
            'multi_input_models': list(self._multi_input_models),
        }

        def convert_numpy(obj: Any) -> Any:
            """
            Recursively convert numpy types and pandas DataFrames to JSON-serializable formats.

            Args:
                obj: Object to convert, may contain nested numpy arrays or pandas DataFrames.

            Returns:
                JSON-serializable version of the object.
            """
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, dict):
                # Respect non-serializable field markers if present
                skip_fields = getattr(self.results, '_non_serializable_fields', set())
                return {k: convert_numpy(v) for k, v in obj.items()
                       if k not in skip_fields}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        # Convert all numpy types and complex objects to JSON-serializable format
        results_dict = convert_numpy(results_dict)

        # Write results to file with error handling
        try:
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=JSON_INDENT)
            logger.info(f"Saved results to: {filepath}")
        except Exception as e:
            logger.error(f"Could not save results: {e}")

    def _serialize_training_metrics(self) -> Dict[str, Any]:
        """
        Serialize training metrics for JSON storage.

        Extracts and formats training metrics data structure for JSON serialization,
        ensuring all nested objects are properly handled.

        Returns:
            Dictionary containing serializable training metrics data.
        """
        metrics = self.results.training_metrics
        return {
            'epochs_to_convergence': metrics.epochs_to_convergence,
            'training_stability_score': metrics.training_stability_score,
            'overfitting_index': metrics.overfitting_index,
            'peak_performance': metrics.peak_performance,
            'final_gap': metrics.final_gap,
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of the analysis.

        Compiles key metrics and statistics from all completed analyses into
        a structured summary format for easy programmatic access and reporting.

        Returns:
            Dictionary containing summary statistics organized by analysis type,
            including model performance, calibration metrics, confidence metrics,
            weight statistics, and training dynamics summaries.

        Note:
            Fixed to access entropy from confidence_metrics instead of calibration_metrics.
        """
        summary = {
            'n_models': len(self.models),
            'n_multi_input_models': len(self._multi_input_models),
            'multi_input_models': list(self._multi_input_models),
            'analyses_performed': [],
            'model_performance': {},
            'calibration_summary': {},
            'confidence_summary': {},  # Separate confidence summary for clarity
            'weight_summary': {},
            'training_summary': {}
        }

        # Determine which analyses were actually performed based on results
        if any(self.results.weight_stats.values()):
            summary['analyses_performed'].append('weight_analysis')
        if self.results.calibration_metrics:
            summary['analyses_performed'].append('calibration_analysis')
        if self.results.confidence_metrics:
            summary['analyses_performed'].append('confidence_analysis')
        if self.results.information_flow:
            summary['analyses_performed'].append('information_flow_analysis')
        if self.results.training_metrics:
            summary['analyses_performed'].append('training_dynamics_analysis')

        # Compile model performance summaries
        for model_name, metrics in self.results.model_metrics.items():
            summary['model_performance'][model_name] = {
                'accuracy': metrics.get('accuracy', DEFAULT_METRIC_VALUE),
                'loss': metrics.get('loss', DEFAULT_METRIC_VALUE),
                CACHE_KEY_STATUS: metrics.get(CACHE_KEY_STATUS, STATUS_UNKNOWN)
            }

        # Compile calibration summaries
        for model_name, metrics in self.results.calibration_metrics.items():
            summary['calibration_summary'][model_name] = {
                'ece': metrics.get('ece', DEFAULT_METRIC_VALUE),
                'brier_score': metrics.get('brier_score', DEFAULT_METRIC_VALUE),
            }

        # Compile confidence summaries (fixed to use correct source)
        for model_name, metrics in self.results.confidence_metrics.items():
            max_prob_values = metrics.get('max_probability', [DEFAULT_METRIC_VALUE])
            summary['confidence_summary'][model_name] = {
                'mean_entropy': metrics.get('mean_entropy', DEFAULT_METRIC_VALUE),
                'mean_confidence': float(np.mean(max_prob_values)),
                'entropy_std': metrics.get('entropy_std', DEFAULT_METRIC_VALUE)
            }

        # Compile weight summaries
        for model_name, weight_stats in self.results.weight_stats.items():
            # Calculate total parameters from weight tensor shapes
            n_params = sum(np.prod(s['shape']) for s in weight_stats.values())
            summary['weight_summary'][model_name] = {
                'total_parameters': n_params,
                'n_weight_tensors': len(weight_stats)
            }

        # Compile training summaries if available
        if self.results.training_metrics:
            for model_name in self.models:
                peak_performance = self.results.training_metrics.peak_performance.get(model_name, {})
                summary['training_summary'][model_name] = {
                    'epochs_to_convergence': self.results.training_metrics.epochs_to_convergence.get(
                        model_name, DEFAULT_METRIC_VALUE),
                    'overfitting_index': self.results.training_metrics.overfitting_index.get(
                        model_name, DEFAULT_METRIC_VALUE),
                    'peak_accuracy': peak_performance.get('val_accuracy', DEFAULT_METRIC_VALUE)
                }

        return summary

    def create_pareto_analysis(self, save_plot: bool = True) -> Optional[plt.Figure]:
        """
        Create Pareto front analysis for hyperparameter sweep scenarios.

        Generates a Pareto front analysis that helps identify optimal models
        based on multiple competing objectives (accuracy, overfitting, convergence speed).
        Useful for hyperparameter sweeps or model comparison scenarios.

        Args:
            save_plot: Whether to automatically save the generated plot to the output directory.

        Returns:
            matplotlib Figure object containing the Pareto analysis plots, or None
            if insufficient data is available for the analysis.

        Note:
            Requires training metrics to be available. Analysis needs at least
            the number of models specified in config.pareto_analysis_threshold.
        """
        # Verify training metrics are available
        if not self.results.training_metrics or not self.results.training_metrics.peak_performance:
            logger.warning("No training metrics available for Pareto analysis")
            return None

        # Collect data for Pareto analysis
        models = []
        peak_accuracies = []
        overfitting_indices = []
        convergence_speeds = []

        # Extract metrics for all models with training data
        for model_name in sorted(self.models.keys()):
            if model_name in self.results.training_metrics.peak_performance:
                models.append(model_name)
                peak_accuracies.append(
                    self.results.training_metrics.peak_performance[model_name].get(
                        'val_accuracy', DEFAULT_METRIC_VALUE)
                )
                overfitting_indices.append(
                    self.results.training_metrics.overfitting_index.get(
                        model_name, DEFAULT_METRIC_VALUE)
                )
                convergence_speeds.append(
                    self.results.training_metrics.epochs_to_convergence.get(
                        model_name, DEFAULT_METRIC_VALUE)
                )

        # Check if we have enough models for meaningful Pareto analysis
        if len(models) < self.config.pareto_analysis_threshold:
            logger.warning(f"Need at least {self.config.pareto_analysis_threshold} models for Pareto analysis")
            return None

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Peak Accuracy vs Overfitting Index with convergence speed as color
        scatter = ax1.scatter(
            overfitting_indices, peak_accuracies,
            c=convergence_speeds, s=PARETO_SCATTER_SIZE, alpha=PARETO_ALPHA,
            cmap='viridis', edgecolors='black', linewidth=1
        )

        # Find and plot Pareto front
        # Note: We negate overfitting index because we want to minimize it (maximize -overfitting)
        pareto_indices = find_pareto_front(
            np.array(overfitting_indices) * -1,  # Minimize overfitting (negate for maximization)
            np.array(peak_accuracies)  # Maximize accuracy
        )

        # Highlight Pareto optimal models
        pareto_x = [overfitting_indices[i] for i in pareto_indices]
        pareto_y = [peak_accuracies[i] for i in pareto_indices]
        ax1.plot(pareto_x, pareto_y, 'r--', alpha=PARETO_STAR_ALPHA, linewidth=PARETO_LINE_WIDTH)
        ax1.scatter(pareto_x, pareto_y, color='red', s=PARETO_STAR_SIZE, marker='*',
                   edgecolors='black', linewidth=PARETO_LINE_WIDTH, zorder=5, label='Pareto Optimal')

        # Add labels for Pareto optimal models
        for idx in pareto_indices:
            ax1.annotate(models[idx], (overfitting_indices[idx], peak_accuracies[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=PARETO_ALPHA))

        ax1.set_xlabel('Overfitting Index (lower is better)')
        ax1.set_ylabel('Peak Validation Accuracy')
        ax1.set_title('Pareto Front: Accuracy vs Overfitting')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add colorbar for convergence speed
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Convergence Speed (epochs)', rotation=270, labelpad=20)

        # Plot 2: Model ranking heatmap
        # Create normalized metrics matrix for comparison
        metrics_matrix = np.array([
            normalize_metric(peak_accuracies, higher_better=True),
            normalize_metric(overfitting_indices, higher_better=False),
            normalize_metric(convergence_speeds, higher_better=False)
        ])

        # Create heatmap
        im = ax2.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Configure heatmap axes
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_yticks(range(3))
        ax2.set_yticklabels(['Peak Accuracy', 'Low Overfitting', 'Fast Convergence'])
        ax2.set_title('Normalized Performance Metrics')

        # Add text annotations to heatmap cells
        for i in range(3):
            for j in range(len(models)):
                text = ax2.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)

        # Add colorbar for normalized scores
        cbar2 = plt.colorbar(im, ax=ax2)
        cbar2.set_label('Normalized Score', rotation=270, labelpad=20)

        # Finalize plot
        plt.suptitle('Pareto Analysis for Model Selection', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot if requested and configuration allows
        if save_plot and self.config.save_plots:
            summary_visualizer = SummaryVisualizer(
                self.results, self.config, self.output_dir, self.model_colors
            )
            summary_visualizer._save_figure(fig, 'pareto_analysis')

        return fig

# ---------------------------------------------------------------------
