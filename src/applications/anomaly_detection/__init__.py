"""Patch-entropy anomaly detection application.

Encoder-only, KL-divergence per-patch anomaly detector built on the trained
``HierarchicalConvNeXtPatchVAE``. The Gradio GUI lives in ``app.py``; the
GUI-free core is :class:`PatchEntropyAnomalyDetector`.
"""

from .anomaly_detector import PatchEntropyAnomalyDetector

__all__ = ["PatchEntropyAnomalyDetector"]
