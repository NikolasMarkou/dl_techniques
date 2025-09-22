__author__ = "Nikolas Markou"
__version__ = "1.0.0"


# --- Core Components ---
# Make the most important classes available at the top level.
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

# --- Data Structures ---
# Expose all data containers for easy type hinting and instantiation.
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
)

# --- Visualization Plugin Templates ---
# Users can import these to register them manually if needed.
from .training_performance import (
    TrainingCurvesVisualization,
    ModelComparisonBarChart,
    PerformanceRadarChart,
    ConvergenceAnalysis,
    OverfittingAnalysis,
    PerformanceDashboard,
    LearningRateScheduleVisualization,
)
from .classification import (
    ConfusionMatrixVisualization,
    ROCPRCurves,
    ClassificationReportVisualization,
    PerClassAnalysis,
    ErrorAnalysisDashboard,
)
from .data_nn import (
    DataDistributionAnalysis,
    ClassBalanceVisualization,
    NetworkArchitectureVisualization,
    ActivationVisualization,
    WeightVisualization,
    FeatureMapVisualization,
    GradientVisualization,
)
from .examples import (
    ExperimentComparisonDashboard,
    LossLandscapeVisualization,
    AttentionVisualization,
    EmbeddingVisualization,
    MLExperimentWorkflow, # Exposing this is great for users
)


# --- Public API Control ---
# Defines what `from vizframework import *` will import.
__all__ = [
    # Core
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

    # Training & Performance Plugins
    "TrainingCurvesVisualization",
    "ModelComparisonBarChart",
    "PerformanceRadarChart",
    "ConvergenceAnalysis",
    "OverfittingAnalysis",
    "PerformanceDashboard",
    "LearningRateScheduleVisualization",

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

    # Specialized/Example Plugins
    "ExperimentComparisonDashboard",
    "LossLandscapeVisualization",
    "AttentionVisualization",
    "EmbeddingVisualization",
    "MLExperimentWorkflow",
]