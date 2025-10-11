"""
Data Type Definitions for Model Analyzer
"""

import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, NamedTuple, Set


class DataInput(NamedTuple):
    """Structured data input type."""
    x_data: np.ndarray
    y_data: np.ndarray

    @classmethod
    def from_tuple(cls, data: Tuple[np.ndarray, np.ndarray]) -> 'DataInput':
        """Create from tuple."""
        return cls(x_data=data[0], y_data=data[1])

    @classmethod
    def from_object(cls, data: Any) -> 'DataInput':
        """Create from object with x_test and y_test attributes."""
        return cls(x_data=data.x_test, y_data=data.y_test)


@dataclass
class TrainingMetrics:
    """Container for computed training metrics.

    Attributes:
        epochs_to_convergence: Number of epochs to reach 95% of peak performance
        training_stability_score: Standard deviation of recent validation losses (lower = more stable)
        overfitting_index: Average gap between validation and training loss in final third of training
                          Positive values indicate overfitting, negative indicate underfitting
        peak_performance: Best validation metrics achieved during training with epoch info
        final_gap: Difference between validation and training loss at end of training
        smoothed_curves: Smoothed versions of training curves for cleaner visualization
    """
    epochs_to_convergence: Dict[str, int] = field(default_factory=dict)
    training_stability_score: Dict[str, float] = field(default_factory=dict)
    overfitting_index: Dict[str, float] = field(default_factory=dict)
    peak_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_gap: Dict[str, float] = field(default_factory=dict)
    smoothed_curves: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)


@dataclass
class AnalysisResults:
    """
    Container for all analysis results.

    FIXED: All attributes are now properly declared to eliminate dynamic injection.
    This makes the contract explicit and prevents AttributeError issues.
    """

    # Model performance
    model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Activation analysis (now part of information flow)
    activation_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Weight analysis
    weight_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    weight_pca: Optional[Dict[str, Any]] = None
    weight_stats_layer_order: Dict[str, List[str]] = field(default_factory=dict)  # FIXED: Properly declared

    # Calibration analysis
    calibration_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Metrics related to the model's calibration (e.g., ECE, Brier Score)."""

    reliability_data: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    """Data for plotting reliability diagrams (bin accuracies, confidences, etc.)."""

    confidence_metrics: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    """Metrics related to the distribution of prediction confidences (e.g., entropy, max probability)."""

    # Information flow
    information_flow: Dict[str, Any] = field(default_factory=dict)

    # Training history and dynamics
    training_history: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    training_metrics: Optional[TrainingMetrics] = None

    # Metadata
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Optional['AnalysisConfig'] = None  # Forward reference

    # FIXED: Explicitly declare fields for serialization control
    _non_serializable_fields: Set[str] = field(
        default_factory=lambda: {'_non_serializable_fields'},
        init=False,
        repr=False
    )

    def __post_init__(self):
        """Post-initialization to set up any derived fields."""
        # Ensure all dict fields are properly initialized
        if not isinstance(self.weight_stats_layer_order, dict):
            self.weight_stats_layer_order = {}
        if not isinstance(self.activation_stats, dict):
            self.activation_stats = {}
        if not isinstance(self.information_flow, dict):
            self.information_flow = {}

    def add_non_serializable_field(self, field_name: str) -> None:
        """Add a field to the non-serializable set."""
        self._non_serializable_fields.add(field_name)

    def get_serializable_dict(self) -> Dict[str, Any]:
        """Get a dictionary representation excluding non-serializable fields."""
        result = {}
        for key, value in self.__dict__.items():
            if key not in self._non_serializable_fields:
                result[key] = value
        return result