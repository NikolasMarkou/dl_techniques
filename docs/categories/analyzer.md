# Analyzer

Model analysis and evaluation tools

**24 modules in this category**

## Analyzers

### analyzer.analyzers
Analyzer Components

*📁 File: `src/dl_techniques/analyzer/analyzers/__init__.py`*

### analyzer.analyzers.base
Base Analyzer Interface

**Classes:**
- `BaseAnalyzer`

**Functions:** `analyze`, `requires_data`

*📁 File: `src/dl_techniques/analyzer/analyzers/base.py`*

### analyzer.analyzers.calibration_analyzer
Assess the reliability and uncertainty of model predictions.

**Classes:**
- `CalibrationAnalyzer`

**Functions:** `requires_data`, `analyze`

*📁 File: `src/dl_techniques/analyzer/analyzers/calibration_analyzer.py`*

### analyzer.analyzers.information_flow_analyzer
Analyze the flow of information and feature dimensionality through layers.

**Classes:**
- `InformationFlowAnalyzer`

**Functions:** `requires_data`, `analyze`, `forward_hook`

*📁 File: `src/dl_techniques/analyzer/analyzers/information_flow_analyzer.py`*

### analyzer.analyzers.spectral_analyzer
Analyze model generalization and training quality via spectral properties.

**Classes:**
- `SpectralAnalyzer`

**Functions:** `requires_data`, `analyze`

*📁 File: `src/dl_techniques/analyzer/analyzers/spectral_analyzer.py`*

### analyzer.analyzers.training_dynamics_analyzer
Quantify the learning process by analyzing training history.

**Classes:**
- `TrainingDynamicsAnalyzer`

**Functions:** `requires_data`, `analyze`

*📁 File: `src/dl_techniques/analyzer/analyzers/training_dynamics_analyzer.py`*

### analyzer.analyzers.weight_analyzer
Analyze the statistical properties and structural similarity of model weights.

**Classes:**
- `WeightAnalyzer`

**Functions:** `requires_data`, `analyze`

*📁 File: `src/dl_techniques/analyzer/analyzers/weight_analyzer.py`*

## Calibration_Metrics

### analyzer.calibration_metrics
Model Calibration Metrics

**Functions:** `compute_ece`, `compute_adaptive_ece`, `compute_mce`, `compute_reliability_data`, `compute_brier_score` (and 2 more)

*📁 File: `src/dl_techniques/analyzer/calibration_metrics.py`*

## Config

### analyzer.config
Configuration for Model Analyzer

**Classes:**
- `AnalysisConfig`

**Functions:** `get_figure_size`, `setup_plotting_style`

*📁 File: `src/dl_techniques/analyzer/config.py`*

## Constants

### analyzer.constants
Constants for Model Analyzer

**Classes:**
- `LayerType`
- `SmoothingMethod`
- `StatusCode`
- `MetricNames`

*📁 File: `src/dl_techniques/analyzer/constants.py`*

## Core

### analyzer
Model Analyzer Module

*📁 File: `src/dl_techniques/analyzer/__init__.py`*

## Data_Types

### analyzer.data_types
Data Type Definitions for Model Analyzer

**Classes:**
- `DataInput`
- `TrainingMetrics`
- `AnalysisResults`

**Functions:** `from_tuple`, `from_object`, `add_non_serializable_field`, `get_serializable_dict`

*📁 File: `src/dl_techniques/analyzer/data_types.py`*

## Model_Analyzer

### analyzer.model_analyzer
Orchestrate a multi-faceted analysis of deep learning models.

**Classes:**
- `ModelAnalyzer`

**Functions:** `analyze`, `create_summary_dashboard`, `save_results`, `get_summary_statistics`, `create_pareto_analysis` (and 2 more)

*📁 File: `src/dl_techniques/analyzer/model_analyzer.py`*

## Spectral_Metrics

### analyzer.spectral_metrics
The mathematical core of spectral analysis for neural networks.

**Functions:** `compute_eigenvalues`, `fit_powerlaw`, `calculate_matrix_entropy`, `calculate_spectral_metrics`, `calculate_gini_coefficient` (and 10 more)

*📁 File: `src/dl_techniques/analyzer/spectral_metrics.py`*

## Spectral_Utils

### analyzer.spectral_utils
Provide utilities to adapt Keras layer weights for spectral analysis.

**Functions:** `infer_layer_type`, `get_layer_weights_and_bias`, `get_weight_matrices`, `create_weight_visualization`, `compute_weight_statistics` (and 1 more)

*📁 File: `src/dl_techniques/analyzer/spectral_utils.py`*

## Utils

### analyzer.utils
Utility Functions for Model Analyzer

**Classes:**
- `DataSampler`

**Functions:** `safe_set_xticklabels`, `safe_tight_layout`, `smooth_curve`, `find_metric_in_history`, `find_model_metric` (and 7 more)

*📁 File: `src/dl_techniques/analyzer/utils.py`*

## Visualizers

### analyzer.visualizers
Visualizer Components

*📁 File: `src/dl_techniques/analyzer/visualizers/__init__.py`*

### analyzer.visualizers.base
Base Visualizer Interface

**Classes:**
- `BaseVisualizer`

**Functions:** `create_visualizations`

*📁 File: `src/dl_techniques/analyzer/visualizers/base.py`*

### analyzer.visualizers.calibration_visualizer
Calibration Visualization Module

**Classes:**
- `CalibrationVisualizer`

**Functions:** `create_visualizations`

*📁 File: `src/dl_techniques/analyzer/visualizers/calibration_visualizer.py`*

### analyzer.visualizers.information_flow_visualizer
Information Flow Visualization Module

**Classes:**
- `InformationFlowVisualizer`

**Functions:** `create_visualizations`

*📁 File: `src/dl_techniques/analyzer/visualizers/information_flow_visualizer.py`*

### analyzer.visualizers.spectral_visualizer
Spectral Analysis Visualization Module (WeightWatcher Integration)

**Classes:**
- `SpectralVisualizer`

**Functions:** `create_visualizations`

*📁 File: `src/dl_techniques/analyzer/visualizers/spectral_visualizer.py`*

### analyzer.visualizers.summary_visualizer
Summary Dashboard Visualization Module

**Classes:**
- `SummaryVisualizer`

**Functions:** `create_visualizations`

*📁 File: `src/dl_techniques/analyzer/visualizers/summary_visualizer.py`*

### analyzer.visualizers.training_dynamics_visualizer
Training Dynamics Visualization Module

**Classes:**
- `TrainingDynamicsVisualizer`

**Functions:** `create_visualizations`

*📁 File: `src/dl_techniques/analyzer/visualizers/training_dynamics_visualizer.py`*

### analyzer.visualizers.weight_visualizer
Weight Visualization Module

**Classes:**
- `WeightVisualizer`

**Functions:** `create_visualizations`

*📁 File: `src/dl_techniques/analyzer/visualizers/weight_visualizer.py`*