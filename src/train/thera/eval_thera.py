"""THERA multi-scale super-resolution evaluation.

Mirrors the inference path of the reference THERA ``super_resolve.py`` /
``run_eval.py``: standardize the LR source, encode it once, build a query grid of
the TARGET shape, set the heat-kernel time ``t = (target_h / source_h) ** -2``,
decode the residual field at those coordinates, denormalize with the THERA
channel statistics, add the nearest-upsampled source, and clip to ``[0, 1]``.

The :func:`evaluate_multiscale` driver reproduces the standard arbitrary-scale SR
benchmark: an HR image is downscaled by an integer factor ``s`` (antialiased
bicubic) to synthesize the LR input, super-resolved back to the HR size, and
compared (PSNR / SSIM) against a plain bicubic-upscale baseline. Following the
reference protocol a ``s``-pixel border is cropped before computing metrics.

This module is inference-only (no training, no wandb, ``logger`` not ``print``).

Typical usage::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.thera.eval_thera \\
        --checkpoint results/thera_.../thera_model.keras \\
        --data-dir /path/to/benchmark/HR --scales 2,3,4
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import keras
import tensorflow as tf

from dl_techniques.utils.logger import logger
from dl_techniques.layers.grid_sample import make_grid

# THERA per-channel image statistics. Imported from the trainer so the eval and
# training paths share one source of truth (DRY); redefined here only if that
# import is unavailable.
try:
    from train.thera.train_thera import THERA_MEAN, THERA_VAR
except Exception:  # pragma: no cover - fallback when trainer import is unavailable
    THERA_MEAN = np.array([0.4488, 0.4371, 0.4040], dtype=np.float32)
    THERA_VAR = np.array([0.25, 0.25, 0.25], dtype=np.float32)


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# =====================================================================
# Core inference
# =====================================================================


def super_resolve(
    thera_model: keras.Model,
    lr_image_01: np.ndarray,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Super-resolve one LR image to ``target_hw`` via THERA's inference path.

    Reproduces the reference ``super_resolve.py`` process: standardize, encode
    once, decode the residual field at a target-shaped query grid with heat-time
    ``t = (target_h / source_h) ** -2``, denormalize, add the nearest-upsampled
    source, and clip.

    Args:
        thera_model: A built / loaded :class:`~dl_techniques.models.thera.model.Thera`
            (raw residual-field predictor).
        lr_image_01: Low-resolution image ``(H, W, 3)``, ``float`` in ``[0, 1]``.
        target_hw: Target output size ``(target_h, target_w)``.

    Returns:
        Super-resolved image ``(target_h, target_w, 3)`` as a ``float32`` numpy
        array in ``[0, 1]``.
    """
    lr = np.asarray(lr_image_01, dtype=np.float32)
    if lr.ndim != 3 or lr.shape[-1] != 3:
        raise ValueError(f"lr_image_01 must be (H, W, 3), got {lr.shape}")
    src_h, src_w = int(lr.shape[0]), int(lr.shape[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])

    mean = tf.constant(THERA_MEAN, dtype=tf.float32)
    std = tf.sqrt(tf.constant(THERA_VAR, dtype=tf.float32))

    lr_t = tf.convert_to_tensor(lr, dtype=tf.float32)

    # Standardize and add a batch dim -> (1, H, W, 3).
    source_std = ((lr_t - mean) / std)[tf.newaxis, ...]
    encoding = thera_model.apply_encoder(source_std, training=False)

    # Target-shaped query grid (pixel-center; numpy -> tensor + batch dim).
    coords = make_grid((target_h, target_w))  # (target_h, target_w, 2) numpy
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)[tf.newaxis, ...]

    # Heat-kernel time t = (target_h / source_h) ** -2, shape (1, 1).
    t = tf.constant(
        [[float(target_h) / float(src_h)]], dtype=tf.float32
    ) ** -2.0

    out = thera_model.apply_decoder(encoding, coords, t, training=False)

    # Denormalize the residual field and add the nearest-upsampled source.
    out = out * std + mean
    source_nearest = tf.image.resize(
        lr_t[tf.newaxis, ...], (target_h, target_w), method="nearest"
    )
    sr = tf.clip_by_value(out + source_nearest, 0.0, 1.0)[0]
    return np.asarray(sr, dtype=np.float32)


# =====================================================================
# Multi-scale benchmark
# =====================================================================


def _load_image_01(path: str) -> np.ndarray:
    """Load an image file as ``float32`` ``(H, W, 3)`` in ``[0, 1]``."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0, 1]
    return np.asarray(img, dtype=np.float32)


def _rgb_to_y(img: tf.Tensor) -> tf.Tensor:
    """Convert a ``[0, 1]`` RGB tensor to the luma (Y) channel of YCbCr.

    Uses the ITU-R BT.601 coefficients on the standard ``[16, 235]`` luma range
    (matching the common SR-benchmark Y-only protocol). Input/output keep the
    leading batch + spatial dims; the channel dim becomes 1.
    """
    r, g, b = img[..., 0:1], img[..., 1:2], img[..., 2:3]
    y = 16.0 / 255.0 + (
        65.481 * r + 128.553 * g + 24.966 * b
    ) / 255.0
    return y


def _crop_border(img: tf.Tensor, border: int) -> tf.Tensor:
    """Crop ``border`` pixels from each spatial edge of a ``(..., H, W, C)`` tensor."""
    if border <= 0:
        return img
    return img[..., border:-border, border:-border, :]


def _psnr_ssim(
    pred_01: np.ndarray,
    target_01: np.ndarray,
    border: int,
    y_only: bool,
) -> Tuple[float, float]:
    """Border-cropped PSNR + SSIM between two ``[0, 1]`` images (``max_val=1.0``)."""
    pred = tf.convert_to_tensor(pred_01, dtype=tf.float32)[tf.newaxis, ...]
    target = tf.convert_to_tensor(target_01, dtype=tf.float32)[tf.newaxis, ...]

    pred = _crop_border(pred, border)
    target = _crop_border(target, border)

    if y_only:
        pred = _rgb_to_y(pred)
        target = _rgb_to_y(target)

    psnr = float(tf.image.psnr(pred, target, max_val=1.0)[0])
    ssim = float(tf.image.ssim(pred, target, max_val=1.0)[0])
    return psnr, ssim


def evaluate_multiscale(
    thera_model: keras.Model,
    image_paths: Sequence[str],
    scales: Sequence[int] = (2, 3, 4),
    border_crop: Optional[int] = None,
    max_images: Optional[int] = None,
    y_only: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run the multi-scale arbitrary-scale SR benchmark on a set of HR images.

    For each image and each integer scale ``s``: crop the HR image to a multiple
    of ``s``, synthesize the LR input by antialiased bicubic downscale, THERA
    super-resolves it back to the cropped-HR size, and PSNR/SSIM are computed for
    BOTH the THERA output and a plain bicubic-upscale baseline (after cropping an
    ``s``-pixel border, per the reference protocol).

    Args:
        thera_model: A built / loaded :class:`Thera` model.
        image_paths: Paths to HR ground-truth images.
        scales: Integer downscale/upscale factors to evaluate.
        border_crop: Border (pixels) to crop before metrics. ``None`` -> use the
            scale ``s`` itself (the reference convention).
        max_images: If set, evaluate at most this many images.
        y_only: When ``True``, compute metrics on the Y (luma) channel only.

    Returns:
        ``{f"x{s}": {"psnr", "ssim", "bicubic_psnr", "bicubic_ssim"}}`` with the
        mean over evaluated images per scale.
    """
    paths = list(image_paths)
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise ValueError("evaluate_multiscale received no image paths")

    # Accumulators: per scale -> list of (sr_psnr, sr_ssim, bic_psnr, bic_ssim).
    acc: Dict[int, List[Tuple[float, float, float, float]]] = {
        int(s): [] for s in scales
    }

    for path in paths:
        hr = _load_image_01(path)  # (H, W, 3) [0, 1]
        H, W = int(hr.shape[0]), int(hr.shape[1])

        for s in scales:
            s = int(s)
            # Crop H, W to multiples of s so the LR/HR shapes are exact.
            h = (H // s) * s
            w = (W // s) * s
            if h < s or w < s:
                logger.warning(
                    f"Skipping {Path(path).name} at x{s}: too small ({H}x{W})"
                )
                continue
            hr_c = hr[:h, :w, :]
            lr_h, lr_w = h // s, w // s

            # LR input: antialiased bicubic downscale by s.
            lr = tf.image.resize(
                hr_c, (lr_h, lr_w), method="bicubic", antialias=True
            )
            lr = np.asarray(tf.clip_by_value(lr, 0.0, 1.0), dtype=np.float32)

            # THERA super-resolution back to the cropped-HR size.
            sr = super_resolve(thera_model, lr, (h, w))

            # Plain bicubic-upscale baseline.
            bicubic = tf.image.resize(lr, (h, w), method="bicubic")
            bicubic = np.asarray(
                tf.clip_by_value(bicubic, 0.0, 1.0), dtype=np.float32
            )

            border = s if border_crop is None else int(border_crop)
            sr_psnr, sr_ssim = _psnr_ssim(sr, hr_c, border, y_only)
            bic_psnr, bic_ssim = _psnr_ssim(bicubic, hr_c, border, y_only)
            acc[s].append((sr_psnr, sr_ssim, bic_psnr, bic_ssim))

    results: Dict[str, Dict[str, float]] = {}
    for s, rows in acc.items():
        if not rows:
            results[f"x{s}"] = {
                "psnr": float("nan"),
                "ssim": float("nan"),
                "bicubic_psnr": float("nan"),
                "bicubic_ssim": float("nan"),
            }
            continue
        arr = np.asarray(rows, dtype=np.float64)  # (N, 4)
        results[f"x{s}"] = {
            "psnr": float(arr[:, 0].mean()),
            "ssim": float(arr[:, 1].mean()),
            "bicubic_psnr": float(arr[:, 2].mean()),
            "bicubic_ssim": float(arr[:, 3].mean()),
        }
    return results


# =====================================================================
# CLI
# =====================================================================


def _collect_image_paths(data_dir: str) -> List[str]:
    """Recursively collect image files under ``data_dir`` (sorted)."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data-dir does not exist: {data_dir}")
    paths = sorted(
        str(p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}")
    return paths


def _log_table(results: Dict[str, Dict[str, float]]) -> None:
    """Log a per-scale PSNR/SSIM table via the project logger."""
    logger.info("THERA multi-scale evaluation:")
    logger.info(
        f"{'scale':>6} | {'PSNR':>8} {'SSIM':>8} | "
        f"{'bic-PSNR':>9} {'bic-SSIM':>9}"
    )
    logger.info("-" * 52)
    for scale in sorted(results, key=lambda k: int(k[1:])):
        m = results[scale]
        logger.info(
            f"{scale:>6} | {m['psnr']:>8.3f} {m['ssim']:>8.4f} | "
            f"{m['bicubic_psnr']:>9.3f} {m['bicubic_ssim']:>9.4f}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-scale evaluation for a trained THERA model."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a saved thera_model.keras (the deployable inner Thera).",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory of HR ground-truth images (searched recursively).",
    )
    parser.add_argument(
        "--scales", type=str, default="2,3,4",
        help="Comma-separated integer scales, e.g. '2,3,4'.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--border-crop", type=int, default=None,
        help="Border pixels to crop before metrics (default: the scale s).",
    )
    parser.add_argument("--y-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    scales = tuple(int(s) for s in str(args.scales).split(",") if s.strip())

    logger.info(f"Loading THERA model: {args.checkpoint}")
    thera_model = keras.models.load_model(args.checkpoint)

    image_paths = _collect_image_paths(args.data_dir)
    logger.info(
        f"Evaluating {len(image_paths)} image(s) at scales {scales} "
        f"(y_only={args.y_only})"
    )

    results = evaluate_multiscale(
        thera_model,
        image_paths,
        scales=scales,
        border_crop=args.border_crop,
        max_images=args.max_images,
        y_only=args.y_only,
    )
    _log_table(results)


if __name__ == "__main__":
    main()
