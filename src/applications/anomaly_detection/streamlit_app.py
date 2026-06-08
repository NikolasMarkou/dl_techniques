"""Streamlit GUI for patch-reconstruction anomaly detection.

Live webcam (via ``streamlit-webrtc``) and still-image modes. Each frame runs the
``ConvNeXtPatchVAE`` deterministic decode and overlays the per-patch
reconstruction-error heatmap / anomaly mask. The decoder runs every frame
(reconstruction-error scoring).

Run::

    CUDA_VISIBLE_DEVICES=1 ANOMALY_MODEL=results/convnext_patch_vae_ade20k+coco_custom_20260606_214647/best_model.keras \\
        .venv/bin/streamlit run src/applications/anomaly_detection/streamlit_app.py \\
        --server.address 127.0.0.1 --server.port 8501

then open http://127.0.0.1:8501 . The webcam needs a browser with camera access
(localhost is a secure context, so it works directly).
"""

import os
import sys

# Defensive headless guard (detector uses matplotlib colormaps, no pyplot).
os.environ.setdefault("MPLBACKEND", "Agg")

# `streamlit run` executes this file as __main__, so make `src/` importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

from applications.anomaly_detection.anomaly_detector import (
    PatchReconstructionAnomalyDetector,
)

_DEFAULT_MODEL = os.environ.get(
    "ANOMALY_MODEL",
    "results/convnext_patch_vae_ade20k+coco_custom_20260606_214647/best_model.keras",
)
_THRESHOLD_METHODS = ["zscore", "percentile", "absolute"]
_RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@st.cache_resource(show_spinner="Loading anomaly-detection model...")
def load_detector(model_path: str) -> PatchReconstructionAnomalyDetector:
    """Load the detector once per model path (cached across reruns/sessions)."""
    return PatchReconstructionAnomalyDetector.from_pretrained(model_path)


class AnomalyProcessor(VideoProcessorBase):
    """Per-frame webrtc processor; tunables are pushed in from the main thread."""

    def __init__(self, detector: PatchReconstructionAnomalyDetector) -> None:
        self.detector = detector
        self.view = "heatmap"
        self.method = "zscore"
        self.k = 3.0
        self.percentile = 95.0
        self.abs_threshold = 0.01
        self.max_side = 384

    def _overlay(self, rgb: np.ndarray) -> np.ndarray:
        x, (h, w) = self.detector.preprocess(
            rgb, max_size=int(self.max_side) or None
        )
        score_map = self.detector.anomaly_maps(x, orig_hw=(h, w))["anomaly"]
        img01 = x[0][:h, :w]
        if self.view == "mask":
            mask, _ = self.detector.anomaly_mask(
                score_map, method=self.method, k=float(self.k),
                percentile=float(self.percentile),
                abs_threshold=float(self.abs_threshold),
            )
            return self.detector.mask_overlay(img01, mask)
        return self.detector.overlay(img01, score_map)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        rgb = frame.to_ndarray(format="rgb24")
        try:
            out = self._overlay(rgb)
            if out.shape[:2] != rgb.shape[:2]:
                # Keep the output track at the input resolution (stable size
                # even when Max side downscales the processing resolution).
                import cv2

                out = cv2.resize(
                    out, (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        except Exception:  # noqa: BLE001 - never kill the video track
            out = rgb
        return av.VideoFrame.from_ndarray(np.ascontiguousarray(out), format="rgb24")


def _sidebar_controls() -> dict:
    """Render shared controls in the sidebar and return their values."""
    st.sidebar.header("Controls")
    return {
        "view": st.sidebar.radio("Overlay", ["heatmap", "mask"], index=0),
        "method": st.sidebar.selectbox("Threshold method", _THRESHOLD_METHODS),
        "k": st.sidebar.slider("z-score k (mean + k·std)", 0.0, 6.0, 3.0, 0.1),
        "percentile": st.sidebar.slider("percentile", 50.0, 99.9, 95.0, 0.5),
        "abs_threshold": st.sidebar.slider(
            "absolute threshold (MSE units)", 0.0, 0.2, 0.01, 0.005
        ),
        "max_side": st.sidebar.slider(
            "Max side px (0 = native; aspect kept, padded to patch size)",
            0, 1024, 384, 64,
        ),
    }


def main() -> None:
    st.set_page_config(
        page_title="Patch-Reconstruction Anomaly Detection", layout="wide"
    )
    st.title("Patch-Reconstruction Anomaly Detection")
    st.caption(
        "Per-patch reconstruction error from a ConvNeXt patch-VAE. "
        "High reconstruction error = poorly decoded = anomalous. Error is the "
        "per-patch MSE between the input and the deterministic decode."
    )

    model_path = st.sidebar.text_input("Model checkpoint", value=_DEFAULT_MODEL)
    detector = load_detector(model_path)
    c = _sidebar_controls()

    live_tab, image_tab = st.tabs(["Live (webcam)", "Image"])

    with live_tab:
        st.write("Allow camera access, then **Start**. Lower *Max side* for higher FPS.")
        ctx = webrtc_streamer(
            key="anomaly",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=_RTC_CONFIG,
            video_processor_factory=lambda: AnomalyProcessor(detector),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        # Push current control values into the running processor each rerun.
        if ctx.video_processor:
            for key, value in c.items():
                setattr(ctx.video_processor, key, value)

    with image_tab:
        upload = st.file_uploader(
            "Image", type=["png", "jpg", "jpeg", "bmp", "webp"]
        )
        if upload is not None:
            from PIL import Image

            rgb = np.array(Image.open(upload).convert("RGB"))
            x, (h, w) = detector.preprocess(rgb, max_size=int(c["max_side"]) or None)
            score_map = detector.anomaly_maps(x, orig_hw=(h, w))["anomaly"]
            img01 = x[0][:h, :w]
            mask, thr = detector.anomaly_mask(
                score_map, method=c["method"], k=float(c["k"]),
                percentile=float(c["percentile"]),
                abs_threshold=float(c["abs_threshold"]),
            )
            orig = (np.clip(img01, 0.0, 1.0) * 255.0).astype("uint8")
            col1, col2 = st.columns(2)
            col1.image(orig, caption=f"Original {w}x{h}", use_container_width=True)
            col2.image(
                detector.mask_overlay(img01, mask),
                caption=f"Anomaly mask ({c['method']})",
                use_container_width=True,
            )
            st.image(
                detector.overlay(img01, score_map),
                caption=f"Anomaly heatmap {score_map.shape}",
                use_container_width=True,
            )
            scores = detector.score(score_map, mask)
            scores.update({"threshold": thr, "method": c["method"]})
            st.json(scores)


if __name__ == "__main__":
    main()
