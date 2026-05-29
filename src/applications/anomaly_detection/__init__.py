"""Patch-entropy anomaly detection application.

Encoder-only, KL-divergence per-patch anomaly detector built on the trained
``HierarchicalConvNeXtPatchVAE``. The Streamlit GUI (live webcam + image) lives
in ``streamlit_app.py``; the GUI-free core is :class:`PatchEntropyAnomalyDetector`.
"""

from .anomaly_detector import PatchEntropyAnomalyDetector

__all__ = ["PatchEntropyAnomalyDetector"]
