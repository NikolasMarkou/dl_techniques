"""Streamlit GUI for bias-free-denoiser inverse problems.

Interactive front-end over the SAME :class:`UniversalInverseSolver` loop the CLI
(``main.py``) drives: pick one of the six problems (``prior``, ``inpaint``,
``random_pixels``, ``super_resolution``, ``deblur``, ``compressive_sensing``),
tune the solver + per-problem knobs in the sidebar, pick a configurable input
source (synthetic in-domain target, an uploaded image, or a live webcam
snapshot), and view the target, the measured / degraded view, the
reconstruction, and the ``sigma_t`` convergence curve.

This is the ONLY module in the package that imports streamlit (INV-7 / H7); the
core (:class:`DenoiserPrior`, :class:`UniversalInverseSolver`, the operator
family) stays fully GUI-free and headless-usable. All pixels live in the model's
``[-0.5, +0.5]`` domain (INV-1 / D-002).

Run::

    CUDA_VISIBLE_DEVICES=1 BFDENOISER_MODEL=results/cliffordunet_denoiser_base_20260705_004751/best_model.keras \\
        .venv/bin/streamlit run src/applications/bias_free_denoiser/streamlit_app.py \\
        --server.address 127.0.0.1 --server.port 8501

then open http://127.0.0.1:8501 .
"""

import os
import sys

# INV-7 / H7: force a non-interactive backend BEFORE any matplotlib import so the
# app never touches an X server on headless / remote GPU boxes.
os.environ.setdefault("MPLBACKEND", "Agg")

# `streamlit run` executes this file as __main__, so make `src/` importable
# (mirrors anomaly_detection/streamlit_app.py:25-28).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from dl_techniques.utils.logger import logger

from applications.bias_free_denoiser.denoiser_prior import DenoiserPrior
from applications.bias_free_denoiser.main import (
    create_synthetic_test_image,
    run_problem,
    _plot_convergence,
    _ALL_PROBLEMS,
    _PROBLEM_TITLES,
)

_DEFAULT_MODEL = os.environ.get(
    "BFDENOISER_MODEL",
    "results/cliffordunet_denoiser_base_20260705_004751/best_model.keras",
)


@st.cache_resource(show_spinner="Loading bias-free denoiser prior...")
def load_prior(checkpoint_path: str) -> DenoiserPrior:
    """Load the denoiser prior once per checkpoint path (cached across reruns).

    Args:
        checkpoint_path: Path to ``best_model.keras`` or its results directory.

    Returns:
        The loaded :class:`DenoiserPrior` (registrar-first, ``compile=False``).
    """
    return DenoiserPrior.from_pretrained(checkpoint_path)


def _ingest_upload(upload: Any, size: int) -> np.ndarray:
    """Resize an uploaded image to ``size`` and ingest it to ``[-0.5, +0.5]``.

    Args:
        upload: A streamlit ``UploadedFile`` (file-like) of an RGB-convertible image.
        size: Square edge length (must be divisible by 8 for the depth-3 U-Net).

    Returns:
        A float32 ``[1, size, size, 3]`` array in ``[-0.5, +0.5]``.
    """
    from PIL import Image

    img = Image.open(upload).convert("RGB").resize((size, size))
    arr = np.asarray(img)  # uint8 [size, size, 3]
    normalized = DenoiserPrior.ingest(arr)  # -> [-0.5, +0.5]
    return normalized[None, ...].astype(np.float32)


def _make_args(
    problem: str,
    knobs: Dict[str, Any],
) -> argparse.Namespace:
    """Assemble the namespace :func:`run_problem` / ``build_operator`` consume.

    Reuses the CLI's exact per-problem + solver contract so the GUI and the CLI
    stay in lockstep (one code path, no duplicated solver wiring).

    Args:
        problem: The selected problem id.
        knobs: Sidebar values (solver + the active per-problem knob). Missing
            per-problem knobs fall back to the CLI defaults.

    Returns:
        An ``argparse.Namespace`` with every field ``build_operator`` /
        :func:`run_problem` may read.
    """
    return argparse.Namespace(
        # Solver knobs.
        iterations=int(knobs["iterations"]),
        sigma0=float(knobs["sigma0"]),
        beta=float(knobs["beta"]),
        seed=int(knobs["seed"]),
        # Per-problem knobs (defaults mirror main.py; only the active one is shown).
        block=knobs.get("block"),
        keep_ratio=float(knobs.get("keep_ratio", 0.3)),
        sr_factor=int(knobs.get("sr_factor", 4)),
        keep_fraction=float(knobs.get("keep_fraction", 0.15)),
        measurement_ratio=float(knobs.get("measurement_ratio", 0.2)),
        noise_sigma=float(knobs.get("noise_sigma", 0.0)),
    )


def _sidebar_controls() -> Dict[str, Any]:
    """Render the sidebar (problem, solver knobs, conditional per-problem knob)."""
    st.sidebar.header("Problem")
    problem = st.sidebar.selectbox(
        "Inverse problem",
        list(_ALL_PROBLEMS),
        format_func=lambda p: _PROBLEM_TITLES[p],
        index=0,
    )

    st.sidebar.header("Solver")
    knobs: Dict[str, Any] = {
        "iterations": st.sidebar.slider("Iterations", 20, 1000, 200, 10),
        "sigma0": st.sidebar.slider("sigma_0 (initial noise std)", 0.05, 1.0, 0.4, 0.05),
        "beta": st.sidebar.slider("beta (noise injection)", 0.0, 0.5, 0.01, 0.01),
        "seed": st.sidebar.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0, step=1),
    }

    st.sidebar.header("Image")
    knobs["size"] = st.sidebar.number_input(
        "Size (divisible by 8)", min_value=64, max_value=512, value=256, step=8,
    )

    # Conditional per-problem knob (only the one relevant to the selection).
    st.sidebar.header("Problem knob")
    if problem == "prior":
        st.sidebar.caption("Prior sampling has no measurement knob.")
    elif problem == "inpaint":
        knobs["block"] = st.sidebar.slider(
            "Missing-block edge (px, 0 = size//4)", 0, 256, 0, 8,
        ) or None
    elif problem == "random_pixels":
        knobs["keep_ratio"] = st.sidebar.slider("Keep ratio", 0.01, 1.0, 0.3, 0.01)
    elif problem == "super_resolution":
        knobs["sr_factor"] = st.sidebar.select_slider(
            "Downsample factor", options=[2, 4, 8], value=4,
        )
    elif problem == "deblur":
        knobs["keep_fraction"] = st.sidebar.slider("Low-pass keep fraction", 0.02, 1.0, 0.15, 0.01)
    elif problem == "compressive_sensing":
        knobs["measurement_ratio"] = st.sidebar.slider("Measurement ratio", 0.02, 1.0, 0.2, 0.01)
    elif problem == "denoise":
        knobs["noise_sigma"] = st.sidebar.slider(
            "Noise sigma (0 = denoise as-is)", 0.0, 0.25, 0.1, 0.01,
        )

    knobs["problem"] = problem
    return knobs


def _convergence_figure(info: Dict[str, Any], title: str) -> "plt.Figure":
    """Build a single-axes ``sigma_t`` convergence figure (reuses the CLI plotter)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    _plot_convergence(ax, info, title)
    fig.set_layout_engine("constrained")
    return fig


def _display(problem: str, result: Dict[str, Any]) -> None:
    """Render one problem's panels (images + convergence curve)."""
    title = result["title"]
    if problem == "prior":
        # No input image for prior sampling: show only the sample + convergence.
        col1, col2 = st.columns(2)
        col1.image(
            np.clip(result["recon"], 0.0, 1.0),
            caption="Prior sample (reconstruction)",
            use_container_width=True,
        )
        col2.pyplot(_convergence_figure(result["info"], title))
        return

    col1, col2, col3 = st.columns(3)
    col1.image(np.clip(result["target"], 0.0, 1.0), caption="Target", use_container_width=True)
    if result["degraded"] is not None:
        col2.image(
            np.clip(result["degraded"], 0.0, 1.0),
            caption="Measured / degraded",
            use_container_width=True,
        )
    col3.image(
        np.clip(result["recon"], 0.0, 1.0),
        caption="Reconstruction",
        use_container_width=True,
    )
    st.pyplot(_convergence_figure(result["info"], title))


def main() -> None:
    """Streamlit entry point (guarded by ``__main__`` because streamlit runs the file)."""
    st.set_page_config(page_title="Bias-Free Denoiser: Inverse Problems", layout="wide")
    st.title("Bias-Free Denoiser: Inverse Problems")
    st.caption(
        "One universal stochastic-ascent loop (Kadkhodaie & Simoncelli 2021) solves "
        "prior sampling, inpainting, random pixels, super-resolution, spectral "
        "deblurring, and compressive sensing using the prior implicit in a bias-free "
        "denoiser. Only the measurement operator changes."
    )

    checkpoint_path = st.sidebar.text_input("Model checkpoint", value=_DEFAULT_MODEL)
    knobs = _sidebar_controls()
    problem = knobs["problem"]
    size = int(knobs["size"])

    try:
        prior = load_prior(checkpoint_path)
    except Exception as exc:  # noqa: BLE001 - surface load errors in the UI, don't crash.
        logger.error("failed to load prior from %s: %s", checkpoint_path, exc)
        st.error(f"Could not load the denoiser from '{checkpoint_path}': {exc}")
        return

    if size % 8 != 0:
        st.error(f"Image size must be divisible by 8 (depth-3 U-Net); got {size}.")
        return

    # Configurable input source (prior sampling synthesises from noise, so it
    # needs no input image and the selector is hidden for it).
    source = "Synthetic"
    upload: Optional[Any] = None
    camera: Optional[Any] = None
    if problem != "prior":
        source = st.radio(
            "Input source",
            ["Synthetic", "Upload", "Webcam"],
            horizontal=True,
            help=(
                "Synthetic in-domain target, an uploaded image, or a live webcam "
                "snapshot (the camera is read by your browser and the frame is sent "
                "to the server)."
            ),
        )
        if source == "Upload":
            upload = st.file_uploader(
                "Image", type=["png", "jpg", "jpeg", "bmp", "webp"],
            )
        elif source == "Webcam":
            camera = st.camera_input("Take a photo")

    run = st.button("Run", type="primary")
    if not run:
        st.info("Configure the sidebar, then press **Run**.")
        return

    try:
        if problem != "prior" and source == "Upload":
            if upload is None:
                st.warning("Select an image to upload, or switch the input source.")
                return
            target = _ingest_upload(upload, size)
            logger.info("using uploaded image (resized to %dx%d) for '%s'", size, size, problem)
        elif problem != "prior" and source == "Webcam":
            if camera is None:
                st.warning("Take a webcam photo, or switch the input source.")
                return
            target = _ingest_upload(camera, size)
            logger.info("using webcam snapshot (resized to %dx%d) for '%s'", size, size, problem)
        else:
            target = create_synthetic_test_image((1, size, size, 3))
            logger.info("using synthetic in-domain target %dx%d for '%s'", size, size, problem)

        args = _make_args(problem, knobs)
        with st.spinner(f"Solving '{_PROBLEM_TITLES[problem]}' ({args.iterations} iterations)..."):
            result = run_problem(problem, prior, target, args)
    except Exception as exc:  # noqa: BLE001 - a bad input must never crash the app.
        logger.error("solve failed for problem '%s': %s", problem, exc)
        st.error(f"Solve failed: {exc}")
        return

    _display(problem, result)

    info = result["info"]
    final_sigma = info["sigma_values"][-1] if info.get("sigma_values") else float("nan")
    st.caption(
        f"{len(info.get('iterations', []))} iterations, "
        f"final sigma_t = {final_sigma:.5f}, best sigma_t = {info.get('best_sigma', float('nan')):.5f}."
    )


if __name__ == "__main__":
    main()
