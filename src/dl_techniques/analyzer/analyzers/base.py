"""
Base Analyzer Interface

Abstract base class for all analyzers to ensure consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import keras
from ..data_types import AnalysisResults, DataInput
from ..config import AnalysisConfig


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers.

    DECISION plan-2026-10-05-analyzer-audit/F-076: two contracts are now stated here
    rather than left implicit, because the two analyzers disagreed on both and nothing
    documented either convention.

    **Error isolation is per-MODEL, never per-run.** One bad model must not abort the
    analysis of the others. ``ModelAnalyzer._evaluate_models`` and
    ``CalibrationAnalyzer`` (F-046) and ``InformationFlowAnalyzer`` all skip the
    offending model and log it; an earlier ``CalibrationAnalyzer`` had no guard at all
    and a single 1-D ``predict`` output lost every metric every model produced. A new
    analyzer MUST wrap its per-model body.

    **The ``requires_data`` contract is inverted in the base and was undocumented.**
    ``ModelAnalyzer.analyze`` already raises ``ValueError`` when a data-requiring
    analysis is requested without ``data``, so an analyzer must not re-raise on that. What
    it must honour is the SECOND half: ``cache`` is declared ``Optional`` but is
    MANDATORY for any analyzer that reads predictions, and the base docstring called it
    merely "optional". Treat ``cache=None`` as a caller error and raise.
    """

    def __init__(self, models: Dict[str, keras.Model], config: AnalysisConfig):
        """
        Initialize the analyzer.

        Args:
            models: Dictionary mapping model names to Keras models
            config: Analysis configuration
        """
        self.models = models
        self.config = config
        # DECISION plan-2026-10-05-analyzer-audit/F-076
        # `self.results = None` is GONE. Every analyzer receives `results` as the first
        # argument of `analyze` and writes into THAT object; none of them ever read
        # `self.results`, so the attribute was permanently `None` and shadowed the
        # parameter name. It was a trap for the next subclass to write
        # `self.results[...] = ...`, which would fail with a `TypeError` far from the
        # cause. Do NOT reintroduce it as a convenience alias for the parameter.

    @abstractmethod
    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Perform the analysis and update results.

        Args:
            results: AnalysisResults object to update. This is the ONLY results sink;
                do not cache it on ``self`` (F-076).
            data: Optional input data for analysis. Guaranteed non-None when
                ``requires_data()`` is True, because ``ModelAnalyzer.analyze`` validates
                that before dispatching.
            cache: Optional prediction cache to avoid recomputation. Declared optional
                for the data-independent analyzers, but MANDATORY for any analyzer that
                reads model predictions -- raise rather than degrade (F-076).
        """
        pass

    @abstractmethod
    def requires_data(self) -> bool:
        """Check if this analyzer requires input data.

        Returns:
            ``True`` when the analyzer cannot run without ``DataInput``. The
            orchestrator uses this to raise before dispatch, so do not re-implement that
            check here.
        """
        pass