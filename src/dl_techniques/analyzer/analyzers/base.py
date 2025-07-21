"""
Base Analyzer Interface
============================================================================

Abstract base class for all analyzers to ensure consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import keras
from ..data_types import AnalysisResults, DataInput
from ..config import AnalysisConfig


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""

    def __init__(self, models: Dict[str, keras.Model], config: AnalysisConfig):
        """
        Initialize the analyzer.

        Args:
            models: Dictionary mapping model names to Keras models
            config: Analysis configuration
        """
        self.models = models
        self.config = config
        self.results = None

    @abstractmethod
    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Perform the analysis and update results.

        Args:
            results: AnalysisResults object to update
            data: Optional input data for analysis
            cache: Optional prediction cache to avoid recomputation
        """
        pass

    @abstractmethod
    def requires_data(self) -> bool:
        """Check if this analyzer requires input data."""
        pass