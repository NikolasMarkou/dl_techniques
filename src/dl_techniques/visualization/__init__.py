__author__ = "Nikolas Markou"
__version__ = "1.0.0"

# ---------------------------------------------------------------------
# Core Framework Components
# ---------------------------------------------------------------------
# Expose the main classes for managing, configuring, and extending the framework.

from .core import (
    VisualizationManager,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    VisualizationContext,
    VisualizationPlugin,
    CompositeVisualization,
    setup_logging,
)

# ---------------------------------------------------------------------
# Standardized Data Structures
# ---------------------------------------------------------------------
# Expose all data containers for easy type hinting and instantiation by the user.

from .training_performance import (
    TrainingHistory,
    ModelComparison,
)
from .classification import (
    ClassificationResults,
    MultiModelClassification,
)
from .data_nn import (
    DatasetInfo,
    ActivationData,
    WeightData,
    GradientData,
)

# ---------------------------------------------------------------------
# Visualization Plugin Templates
# ---------------------------------------------------------------------
# Users can import these classes to register them with the VisualizationManager.

# --- Training & Performance ---
from .training_performance import (
    TrainingCurvesVisualization,
    LearningRateScheduleVisualization,
    ModelComparisonBarChart,
    PerformanceRadarChart,
    ConvergenceAnalysis,
    OverfittingAnalysis,
    PerformanceDashboard,
)

# --- Classification Analysis ---
from .classification import (
    ConfusionMatrixVisualization,
    ROCPRCurves,
    ClassificationReportVisualization,
    PerClassAnalysis,
    ErrorAnalysisDashboard,
)

# --- Data & Neural Network Inspection ---
from .data_nn import (
    DataDistributionAnalysis,
    ClassBalanceVisualization,
    NetworkArchitectureVisualization,
    ActivationVisualization,
    WeightVisualization,
    FeatureMapVisualization,
    GradientVisualization,
)

# ---------------------------------------------------------------------
# Public API Control
# ---------------------------------------------------------------------

__all__ = [
    # Core Components
    "VisualizationManager",
    "PlotConfig",
    "PlotStyle",
    "ColorScheme",
    "VisualizationContext",
    "VisualizationPlugin",
    "CompositeVisualization",
    "setup_logging",

    # Data Structures
    "TrainingHistory",
    "ModelComparison",
    "ClassificationResults",
    "MultiModelClassification",
    "DatasetInfo",
    "ActivationData",
    "WeightData",
    "GradientData",

    # Training & Performance Plugins
    "TrainingCurvesVisualization",
    "LearningRateScheduleVisualization",
    "ModelComparisonBarChart",
    "PerformanceRadarChart",
    "ConvergenceAnalysis",
    "OverfittingAnalysis",
    "PerformanceDashboard",

    # Classification Plugins
    "ConfusionMatrixVisualization",
    "ROCPRCurves",
    "ClassificationReportVisualization",
    "PerClassAnalysis",
    "ErrorAnalysisDashboard",

    # Data & NN Plugins
    "DataDistributionAnalysis",
    "ClassBalanceVisualization",
    "NetworkArchitectureVisualization",
    "ActivationVisualization",
    "WeightVisualization",
    "FeatureMapVisualization",
    "GradientVisualization",
]