"""Generate qualitative figures for the bfunet paper.

Produces ``figures/qualitative_examples.png``: a 3-row x 3-col grid
(rows = benchmark images, columns = clean / noisy / denoised) with per-panel
PSNR/SSIM computed on the FULL image. Full-image inference (reflect-pad to a
multiple of 16, single forward pass, crop back) is used for every panel; the
displayed detail crop is taken from the already-denoised full-size output, so
the figure faithfully reflects the paper's full-image protocol. An isolated
small crop is NEVER denoised on its own (receptive-field / padding artifacts
would misrepresent the protocol).

Reuses the exact inference/noise/metric helpers behind the paper's PSNR table
from ``train.bfunet.eval_psnr_vs_noise``.

Usage (GPU1, headless)::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg \\
        .venv/bin/python research/papers/bfunet/generate_figures.py --sigma255 25
"""
from __future__ import annotations

import os
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from train.common import setup_gpu

# Importing the model module registers every custom Keras layer / initializer the
# saved denoiser needs (ConvNeXt blocks, Gabor stem, Laplacian pyramid, LayerScale,
# ...) so ``keras.models.load_model`` resolves them from the serialization registry.
import dl_techniques.models.bias_free_denoisers.bfconvunext  # noqa: F401

# Reuse the paper's exact inference / noise / metric protocol (single source of truth).
from train.bfunet.eval_psnr_vs_noise import (
    load_denoiser,
    _to_flexible_input,
    _load_full_image,
    add_awgn,
    _denoise_full,
    psnr_per_image,
    MAX_VAL,  # noqa: F401 - re-exported for protocol provenance; images live in [-0.5,+0.5]
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))

OUT = os.path.join(_THIS_DIR, "figures")
os.makedirs(OUT, exist_ok=True)

# Frozen checkpoint the paper evaluates (SYSTEM.md / plan A4).
CHECKPOINT = os.path.join(
    _REPO_ROOT, "results", "convunext_denoiser_base_20260707_122133", "best_model.keras"
)

CHANNELS = 3
SIZE_MULTIPLE = 16  # U-Net downsample factor: _denoise_full reflect-pads to this.

# --------------------------------------------------------------------------- examples
# One image per dataset for diversity. Crop offsets (y0, x0) are into the loaded
# array of shape (H, W, C) and the crop is CROP_SIZE x CROP_SIZE. These are
# deliberately top-of-file constants: adjust after eyeballing the first render.
DATA_ROOT = "/media/arxwn/data0_4tb/datasets"
CROP_SIZE = 224

EXAMPLES = [
    # Kodak24 lighthouse: smooth sky + fine railing/fence texture. Image is 768H x 512W.
    {
        "label": "Kodak24",
        "path": os.path.join(DATA_ROOT, "kodak24", "kodim19.png"),
        "y0": 300,
        "x0": 150,
    },
    # Urban100 building: fine repetitive straight-line structure (denoiser stress test).
    # Image is 768H x 1024W.
    {
        "label": "Urban100",
        "path": os.path.join(DATA_ROOT, "urban100", "Urban100_HR", "img_005.png"),
        "y0": 220,
        "x0": 360,
    },
    # McMaster saturated color patch. Image is 500H x 500W.
    {
        "label": "McMaster",
        "path": os.path.join(DATA_ROOT, "mcmaster_src", "color", "data", "McMaster", "5.png"),
        "y0": 150,
        "x0": 150,
    },
]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _disp(x: np.ndarray) -> np.ndarray:
    """Denorm [-0.5,+0.5] -> [0,1] for display (matches the eval/training convention)."""
    return np.clip(x + 0.5, 0.0, 1.0)


def _ssim_full(denoised: np.ndarray, clean: np.ndarray) -> float:
    """Per-image SSIM on the FULL image (tf.image.ssim, max_val=1.0 in [0,1] display space)."""
    return float(
        tf.image.ssim(
            _disp(denoised)[None], _disp(clean)[None], max_val=1.0
        )[0]
    )


def _crop(x: np.ndarray, y0: int, x0: int, cs: int) -> np.ndarray:
    """Take a cs x cs display crop, clamping offsets so it always fits the image."""
    h, w = x.shape[:2]
    cs = min(cs, h, w)
    y0 = max(0, min(y0, h - cs))
    x0 = max(0, min(x0, w - cs))
    return x[y0:y0 + cs, x0:x0 + cs, :]


def fig_qualitative(model, sigma255: float):
    sigma_norm = sigma255 / 255.0
    rng = np.random.RandomState(0)  # fixed noise realization -> reproducible figure

    n_rows = len(EXAMPLES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(3 * 2.6, n_rows * 2.6))
    axes = np.atleast_2d(axes)
    col_titles = ["Clean", f"Noisy ($\\sigma$={sigma255:.0f})", "Denoised"]

    for i, ex in enumerate(EXAMPLES):
        clean = _load_full_image(ex["path"], CHANNELS)
        if clean is None:
            raise FileNotFoundError(f"could not load image: {ex['path']}")

        # Full-image protocol: noise -> full-image denoise -> full-image metrics.
        noisy = add_awgn(clean, sigma_norm, clip=True, rng=rng)
        denoised = _denoise_full(model, noisy, size_multiple=SIZE_MULTIPLE)

        psnr_in = float(psnr_per_image(noisy[None], clean[None])[0])
        psnr_out = float(psnr_per_image(denoised[None], clean[None])[0])
        ssim_out = _ssim_full(denoised, clean)

        # Display crops (same window on all three), taken from the already-denoised
        # FULL output -- never from an isolated re-denoised crop.
        y0, x0 = ex["y0"], ex["x0"]
        panels = [
            _disp(_crop(clean, y0, x0, CROP_SIZE)),
            _disp(_crop(noisy, y0, x0, CROP_SIZE)),
            _disp(_crop(denoised, y0, x0, CROP_SIZE)),
        ]

        for j, panel in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(panel, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])  # NOT axis("off"): keep xlabel/ylabel visible.
            if i == 0:
                ax.set_title(col_titles[j])
        axes[i, 0].set_ylabel(ex["label"])
        axes[i, 1].set_xlabel(f"input PSNR {psnr_in:.2f} dB")
        axes[i, 2].set_xlabel(f"PSNR {psnr_out:.2f} dB / SSIM {ssim_out:.3f}")

        print(
            f"[{ex['label']}] input PSNR {psnr_in:.2f} dB -> "
            f"denoised PSNR {psnr_out:.2f} dB / SSIM {ssim_out:.3f}"
        )

    save(fig, "qualitative_examples.png")


def main():
    parser = argparse.ArgumentParser(description="Generate bfunet qualitative figure")
    parser.add_argument("--sigma255", type=float, default=25.0,
                        help="AWGN std on the [0,255] scale (default 25).")
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU id for setup_gpu (default 1; GPU0 is reserved).")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT,
                        help="Path to the saved .keras denoiser.")
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    model = _to_flexible_input(load_denoiser(args.checkpoint))
    fig_qualitative(model, args.sigma255)


if __name__ == "__main__":
    main()
