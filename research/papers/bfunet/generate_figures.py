"""Generate qualitative figures for the bfunet paper.

Produces ``figures/qualitative_examples.png``: a grid with one benchmark image
per row and, for a range of noise levels, a (noisy, denoised) column pair --
preceded by a single clean reference column. Per-panel PSNR/SSIM are computed
on the FULL image. Full-image inference (reflect-pad to a multiple of 16, single
forward pass, crop back) is used for every panel; the displayed detail crop is
taken from the already-denoised full-size output, so the figure faithfully
reflects the paper's full-image protocol. An isolated small crop is NEVER
denoised on its own (receptive-field / padding artifacts would misrepresent the
protocol).

The default noise levels span moderate to extreme -- including levels past the
training ceiling ($\\sigma_{255}\\le64$) -- so the figure visually corroborates
the out-of-range generalization of Figure~2: heavy static in, clean image out,
even where the model was never trained.

Reuses the exact inference/noise/metric helpers behind the paper's PSNR table
from ``train.bfunet.eval_psnr_vs_noise``.

Usage (GPU0, headless)::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg \\
        .venv/bin/python research/papers/bfunet/generate_figures.py \\
        --sigmas255 50 100 200 --gpu 0
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
    MAX_VAL,  # noqa: F401 - re-exported for protocol provenance; images live in [0,1]
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))

OUT = os.path.join(_THIS_DIR, "figures")
os.makedirs(OUT, exist_ok=True)

# Frozen checkpoint the paper evaluates (SYSTEM.md / plan A4).
CHECKPOINT = os.path.join(
    _REPO_ROOT, "results", "20260715_convunext_denoiser", "best_model.keras"
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
    """Clip to [0,1] for display (the model is [0,1]-native; matches eval/training convention)."""
    return np.clip(x, 0.0, 1.0)


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


# Training ceiling (sigma_norm <= 0.25). Noise levels above this are extrapolation.
TRAIN_SIGMA255_MAX = 64.0


def fig_qualitative(model, sigmas255):
    """One row per image; a (noisy, denoised) column pair per noise level, after a
    single clean reference column. Smaller panels + a wider noise sweep than the
    original single-sigma figure."""
    sigmas255 = list(sigmas255)
    n_rows = len(EXAMPLES)
    n_cols = 1 + 2 * len(sigmas255)  # clean + (noisy, denoised) per sigma

    # Smaller panels (2.0" vs the old 2.6") -> the whole grid shrinks under
    # \includegraphics[width=\linewidth]; the extra columns shrink it further.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))
    axes = np.atleast_2d(axes)

    for i, ex in enumerate(EXAMPLES):
        clean = _load_full_image(ex["path"], CHANNELS)
        if clean is None:
            raise FileNotFoundError(f"could not load image: {ex['path']}")
        y0, x0 = ex["y0"], ex["x0"]

        # Column 0: clean reference.
        ax = axes[i, 0]
        ax.imshow(_disp(_crop(clean, y0, x0, CROP_SIZE)), vmin=0.0, vmax=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title("Clean")
        ax.set_ylabel(ex["label"])

        for k, sigma255 in enumerate(sigmas255):
            sigma_norm = sigma255 / 255.0
            rng = np.random.RandomState(0)  # fixed realization -> reproducible figure

            # Full-image protocol: noise -> full-image denoise -> full-image metrics.
            noisy = add_awgn(clean, sigma_norm, clip=True, rng=rng)
            denoised = _denoise_full(model, noisy, size_multiple=SIZE_MULTIPLE)
            psnr_in = float(psnr_per_image(noisy[None], clean[None])[0])
            psnr_out = float(psnr_per_image(denoised[None], clean[None])[0])
            ssim_out = _ssim_full(denoised, clean)

            oor = sigma255 > TRAIN_SIGMA255_MAX  # out of training range
            tag = r"$\,\dagger$" if oor else ""

            jn, jd = 1 + 2 * k, 2 + 2 * k  # noisy col, denoised col
            for j, panel in ((jn, noisy), (jd, denoised)):
                ax = axes[i, j]
                ax.imshow(_disp(_crop(panel, y0, x0, CROP_SIZE)), vmin=0.0, vmax=1.0)
                ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                axes[i, jn].set_title(f"Noisy $\\sigma$={sigma255:.0f}{tag}")
                axes[i, jd].set_title("Denoised")
            axes[i, jn].set_xlabel(f"{psnr_in:.1f} dB")
            axes[i, jd].set_xlabel(f"{psnr_out:.1f} dB / {ssim_out:.3f}")

            print(
                f"[{ex['label']}] sigma255={sigma255:.0f}{' (OOR)' if oor else ''}: "
                f"input {psnr_in:.2f} dB -> denoised {psnr_out:.2f} dB / SSIM {ssim_out:.3f} "
                f"| denoised px range [{float(denoised.min()):.3f}, {float(denoised.max()):.3f}]"
            )

    save(fig, "qualitative_examples.png")


def main():
    parser = argparse.ArgumentParser(description="Generate bfunet qualitative figure")
    parser.add_argument("--sigmas255", type=float, nargs="+", default=[50.0, 100.0, 200.0],
                        help="AWGN stds on the [0,255] scale, one (noisy,denoised) pair each "
                             "(default 50 100 200; 200 is beyond the sigma255<=64 training ceiling).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id for setup_gpu (default 0).")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT,
                        help="Path to the saved .keras denoiser.")
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    model = _to_flexible_input(load_denoiser(args.checkpoint))
    fig_qualitative(model, args.sigmas255)


if __name__ == "__main__":
    main()
