"""Gradio web GUI for patch-entropy anomaly detection.

Two tabs:

* **Image** — upload an image; the ``HierarchicalConvNeXtPatchVAE`` encoder runs
  once and shows per-patch KL "surprise" heatmaps (fine L2 + coarse L1) plus a
  binary anomaly mask. Threshold sliders re-render instantly without re-encoding
  (the KL maps are cached in session state).
* **Live (webcam)** — streams frames from your webcam and overlays the live
  per-patch KL heatmap / anomaly mask in real time. Each frame is a single
  encoder forward (the decoder is never run).

Run::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m applications.anomaly_detection.app \\
        --model results/.../best_model.keras

then open the printed http://127.0.0.1:7860 URL (use --share for a public link,
or SSH-forward the port for a headless box). The webcam tab needs a browser with
camera access (https or localhost).
"""

import os

# Defensive headless guard (this module uses matplotlib colormaps via the
# detector; no pyplot is touched, but keep the repo convention).
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gradio as gr

from dl_techniques.utils.logger import logger
from .anomaly_detector import PatchEntropyAnomalyDetector

_DEFAULT_MODEL = (
    "results/hierarchical_convnext_patch_vae_ade20k+coco_large_20260528_205245/"
    "best_model.keras"
)
_THRESHOLD_METHODS = ["zscore", "percentile", "absolute"]


def _threshold_controls(
    *, max_side_default: int, max_side_max: int
) -> Dict[str, Any]:
    """Build the shared level / threshold / max-side control widgets."""
    return {
        "level": gr.Radio(
            choices=["l2", "l1"], value="l2", label="Level (l2 fine / l1 coarse)"
        ),
        "method": gr.Dropdown(
            choices=_THRESHOLD_METHODS, value="zscore", label="Threshold method"
        ),
        "k": gr.Slider(
            0.0, 6.0, value=3.0, step=0.1, label="z-score k (mean + k·std)"
        ),
        "percentile": gr.Slider(
            50.0, 99.9, value=95.0, step=0.5, label="percentile"
        ),
        "abs_threshold": gr.Slider(
            0.0, 50.0, value=5.0, step=0.5, label="absolute threshold (nats)"
        ),
        "max_side": gr.Slider(
            0, max_side_max, value=max_side_default, step=64,
            label="Max side px (0 = native; aspect kept, padded to /32)",
        ),
    }


def build_interface(detector: PatchEntropyAnomalyDetector) -> gr.Blocks:
    """Construct the Gradio Blocks app bound to a detector.

    Args:
        detector: A ready :class:`PatchEntropyAnomalyDetector`.

    Returns:
        An un-launched ``gr.Blocks`` interface.
    """

    # ---- Image tab callbacks ----------------------------------------
    def analyze(
        image: Optional[np.ndarray], max_side: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """Encode once; cache the [0,1] image + both KL maps in session state."""
        if image is None:
            return None
        x, (h, w) = detector.preprocess(image, max_size=int(max_side) or None)
        maps = detector.kl_maps(x, orig_hw=(h, w))  # decoder NOT executed
        return {"image01": x[0][:h, :w], "l1": maps["l1"], "l2": maps["l2"]}

    def render(
        state: Optional[Dict[str, np.ndarray]],
        level: str,
        method: str,
        k: float,
        percentile: float,
        abs_threshold: float,
    ) -> Tuple[Any, Any, Any, Any, Dict[str, float]]:
        """Re-threshold/redraw from cached state — no re-encode."""
        if not state:
            return None, None, None, None, {}
        img01 = state["image01"]
        kl_map = state[level]
        mask, thr = detector.anomaly_mask(
            kl_map, method=method, k=float(k),
            percentile=float(percentile), abs_threshold=float(abs_threshold),
        )
        orig = (np.clip(img01, 0.0, 1.0) * 255.0).astype("uint8")
        scores = detector.score(kl_map, mask)
        scores.update({"threshold": thr, "level": level, "method": method})
        return (
            orig,
            detector.overlay(img01, state["l2"]),
            detector.overlay(img01, state["l1"]),
            detector.mask_overlay(img01, mask),
            scores,
        )

    # ---- Live tab callback ------------------------------------------
    def stream_frame(
        frame: Optional[np.ndarray],
        view: str,
        level: str,
        method: str,
        k: float,
        percentile: float,
        abs_threshold: float,
        max_side: int,
    ) -> Optional[np.ndarray]:
        """Encode one webcam frame and return the anomaly overlay (uint8)."""
        if frame is None:
            return None
        try:
            x, (h, w) = detector.preprocess(frame, max_size=int(max_side) or None)
            maps = detector.kl_maps(x, orig_hw=(h, w))
            img01 = x[0][:h, :w]
            kl_map = maps[level]
            if view == "mask":
                mask, _ = detector.anomaly_mask(
                    kl_map, method=method, k=float(k),
                    percentile=float(percentile),
                    abs_threshold=float(abs_threshold),
                )
                return detector.mask_overlay(img01, mask)
            return detector.overlay(img01, kl_map)
        except Exception as exc:  # noqa: BLE001 - never kill the stream
            logger.warning("stream frame failed: %s", exc)
            return frame

    with gr.Blocks(title="Patch-Entropy Anomaly Detection") as demo:
        gr.Markdown(
            "# Patch-Entropy Anomaly Detection\n"
            "Per-patch **KL divergence** from a hierarchical ConvNeXt patch-VAE "
            "encoder. High KL = high-entropy / surprising = anomalous. "
            "**L2** (fine) uses the trained conditional prior; **L1** (coarse) "
            "uses KL vs N(0, I). The decoder is never run."
        )

        with gr.Tabs():
            # ---------- Image tab ----------
            with gr.Tab("Image"):
                state = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=1):
                        inp = gr.Image(type="numpy", label="Input image")
                        c = _threshold_controls(max_side_default=1024, max_side_max=2048)
                        run = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=2):
                        with gr.Row():
                            g_orig = gr.Image(label="Original", height=256)
                            g_mask = gr.Image(label="Anomaly mask", height=256)
                        with gr.Row():
                            g_l2 = gr.Image(label="L2 KL (fine) overlay", height=256)
                            g_l1 = gr.Image(label="L1 KL (coarse) overlay", height=256)
                        g_score = gr.JSON(label="Scores")

                outputs: List[Any] = [g_orig, g_l2, g_l1, g_mask, g_score]
                render_inputs: List[Any] = [
                    state, c["level"], c["method"], c["k"],
                    c["percentile"], c["abs_threshold"],
                ]
                run.click(
                    analyze, inputs=[inp, c["max_side"]], outputs=state
                ).then(render, inputs=render_inputs, outputs=outputs)
                for ctrl in (c["level"], c["method"], c["k"],
                             c["percentile"], c["abs_threshold"]):
                    ctrl.change(render, inputs=render_inputs, outputs=outputs)

            # ---------- Live (webcam) tab ----------
            with gr.Tab("Live (webcam)"):
                gr.Markdown(
                    "Streams your webcam; each frame is one encoder forward. "
                    "Lower **Max side** for higher FPS."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        cam = gr.Image(
                            sources=["webcam"], streaming=True, type="numpy",
                            label="Webcam",
                        )
                        view = gr.Radio(
                            choices=["heatmap", "mask"], value="heatmap",
                            label="Live overlay",
                        )
                        lc = _threshold_controls(max_side_default=384, max_side_max=1024)
                    with gr.Column(scale=2):
                        live_out = gr.Image(label="Live anomaly overlay", height=480)

                cam.stream(
                    stream_frame,
                    inputs=[
                        cam, view, lc["level"], lc["method"], lc["k"],
                        lc["percentile"], lc["abs_threshold"], lc["max_side"],
                    ],
                    outputs=live_out,
                    stream_every=0.2,
                    concurrency_limit=1,  # drop, don't queue, stale frames
                    show_progress="hidden",
                )

    return demo


def main() -> None:
    """CLI entrypoint: load the model and launch the Gradio server."""
    parser = argparse.ArgumentParser(
        description="Patch-entropy anomaly detection GUI."
    )
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL, help="Path to the .keras checkpoint."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link."
    )
    args = parser.parse_args()

    detector = PatchEntropyAnomalyDetector.from_pretrained(args.model)
    demo = build_interface(detector)
    logger.info(
        "Launching Gradio on %s:%d (share=%s)", args.host, args.port, args.share
    )
    demo.queue().launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
