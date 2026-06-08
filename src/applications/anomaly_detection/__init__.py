"""Reconstruction-error anomaly detection application.

Per-patch reconstruction-error anomaly detector built on the trained
``ConvNeXtPatchVAE``. The anomaly signal is the squared error between the input
and a deterministic decode (``sample_from(x, temperature=0.0)``), average-pooled
to the patch grid; high values mark regions the trained decoder reconstructs
poorly. The Streamlit GUI (live webcam + image) lives in ``streamlit_app.py``;
the GUI-free core is :class:`PatchReconstructionAnomalyDetector`.
"""

from .anomaly_detector import PatchReconstructionAnomalyDetector

__all__ = ["PatchReconstructionAnomalyDetector"]
