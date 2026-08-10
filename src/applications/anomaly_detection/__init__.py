"""Reconstruction-error anomaly detection application — **non-functional**.

Per-patch reconstruction-error anomaly detector, built on the trained
``ConvNeXtPatchVAE``. The anomaly signal is the squared error between the input
and a deterministic decode (``sample_from(x, temperature=0.0)``), average-pooled
to the patch grid; high values mark regions the trained decoder reconstructs
poorly. The Streamlit GUI (live webcam + image) lives in ``streamlit_app.py``;
the GUI-free core is :class:`PatchReconstructionAnomalyDetector`.

**The model this application runs on no longer exists.**
``dl_techniques.models.convnext_patch_vae`` and ``src/train/convnext_patch_vae/``
were removed from the repository. It was the only architecture the detector
supported, so :meth:`PatchReconstructionAnomalyDetector.from_pretrained` raises
``RuntimeError`` and the Streamlit app cannot start. The module stays importable
and its scoring/threshold/overlay helpers stay correct, but nothing in this repo
can supply a compatible model. See ``README.md``.
"""

from .anomaly_detector import PatchReconstructionAnomalyDetector

__all__ = ["PatchReconstructionAnomalyDetector"]
