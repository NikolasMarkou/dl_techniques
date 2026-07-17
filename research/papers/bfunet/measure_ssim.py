"""One-off: measure full-image PSNR + SSIM for the [0,1] ConvUNeXt denoiser (tab:ssim).

Produces the SSIM table data for ``bfunet.tex`` (tab:ssim) for the checkpoint
``results/20260715_convunext_denoiser/best_model.keras`` over the four color-AWGN
benchmark sets (kodak24 / cbsd68 / mcmaster / urban100) at sigma_255 in {15, 25, 50}.

Protocol
--------
This REUSES the shipped ``train.bfunet.eval_psnr_vs_noise`` helpers and mirrors its
``--full-image`` branch (``evaluate_dataset``) EXACTLY -- same seed (42), same
``num_samples`` (100), same reflect-pad-to-16 single-forward-pass full-image denoise,
same clip -- so the PSNR this script recomputes reproduces the on-disk full-image CSV
(``results/eval_convunext_20260715_psnr_fullimage/psnr_vs_noise.csv``) to <0.1 dB. That
PSNR match is the sanity gate that proves SSIM was computed under the SAME protocol; if
PSNR disagrees the SSIM is not trustworthy (plan Pre-Mortem #3).

SSIM note
---------
SSIM is ``tf.image.ssim(denoised[None], clean[None], max_val=1.0)`` on the [0,1]-NATIVE
tensors. Unlike ``generate_figures.py::_ssim_full`` we do NOT apply the legacy ``_disp``
``+0.5`` shift: that shift denormalized the OLD [-0.5,+0.5] checkpoint for display. This
checkpoint is [0,1]-native, so the tensors are already in display space and a ``+0.5``
would push them out of [0,1] and corrupt SSIM.

Run::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python \\
        research/papers/bfunet/measure_ssim.py
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

# setup_gpu / set_seeds match run_evaluation's startup so the numeric protocol is identical.
from train.common import setup_gpu, set_seeds, collect_image_paths
from train.bfunet.eval_psnr_vs_noise import (
    EvalConfig,
    IMAGE_EXTENSIONS,
    load_denoiser,
    _to_flexible_input,
    sample_full_images,
    add_awgn,
    _denoise_full,
    psnr_per_image,
)

# --- Fixed protocol (must match results/eval_convunext_20260715_psnr_fullimage) --------
CHECKPOINT = "results/20260715_convunext_denoiser/best_model.keras"
SEED = 42
NUM_SAMPLES = 100          # >= every set's size -> all images loaded, permutation order fixed
SIZE_MULTIPLE = 16
CHANNELS = 3
SIGMAS_255: List[float] = [15.0, 25.0, 50.0]
DATASETS: Dict[str, str] = {
    "kodak24": "/media/arxwn/data0_4tb/datasets/kodak24",
    "cbsd68": "/media/arxwn/data0_4tb/datasets/cbsd68_src/CBSD68/original",
    "mcmaster": "/media/arxwn/data0_4tb/datasets/mcmaster_src/color/data/McMaster",
    "urban100": "/media/arxwn/data0_4tb/datasets/urban100/Urban100_HR",
}
OUT_JSON = Path(__file__).resolve().parent / "ssim_results.json"


def _ssim_full(denoised: np.ndarray, clean: np.ndarray) -> float:
    """Per-image SSIM on the full [0,1]-native image (NO legacy +0.5 _disp shift)."""
    return float(
        tf.image.ssim(denoised[None], clean[None], max_val=1.0)[0]
    )


def main() -> None:
    setup_gpu(gpu_id=0)
    set_seeds(SEED)

    # EvalConfig only supplies num_samples / channels / size_multiple / clip to the reused
    # helpers; the models/datasets fields are unused here (we drive the loop directly).
    cfg = EvalConfig(
        models={"convunext_20260715": CHECKPOINT},
        datasets={},
        sigmas_255=SIGMAS_255,
        num_samples=NUM_SAMPLES,
        channels=CHANNELS,
        size_multiple=SIZE_MULTIPLE,
        clip_noise=True,
        seed=SEED,
    )

    model = _to_flexible_input(load_denoiser(CHECKPOINT))

    results: List[Dict] = []
    for ds_name, ds_dir in DATASETS.items():
        paths = collect_image_paths([ds_dir], extensions=IMAGE_EXTENSIONS, sort=True)
        if not paths:
            raise RuntimeError(f"[{ds_name}] no images found in {ds_dir}")
        # Fresh per-dataset RNG seeded at SEED, exactly like evaluate_dataset -> the noise
        # realizations (and thus PSNR) reproduce the on-disk CSV bit-for-bit.
        rng = np.random.RandomState(SEED)
        clean_list = sample_full_images(cfg, paths, rng)

        for sigma_255 in SIGMAS_255:
            sigma_norm = sigma_255 / 255.0
            # Generate ALL noisy images first (same rng consumption order as evaluate_dataset).
            noisy_list = [add_awgn(c, sigma_norm, cfg.clip_noise, rng) for c in clean_list]
            psnrs, ssims = [], []
            for noisy, clean in zip(noisy_list, clean_list):
                denoised = _denoise_full(model, noisy, cfg.size_multiple)
                psnrs.append(float(psnr_per_image(denoised[None], clean[None])[0]))
                ssims.append(_ssim_full(denoised, clean))
            entry = {
                "dataset": ds_name,
                "sigma_255": sigma_255,
                "n": len(clean_list),
                "psnr_mean": float(np.mean(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
            }
            results.append(entry)
            print(f"[{ds_name}] sigma_255={sigma_255:5.1f}  n={entry['n']:3d}  "
                  f"PSNR={entry['psnr_mean']:6.3f} dB  SSIM={entry['ssim_mean']:.4f}")

    payload = {
        "checkpoint": CHECKPOINT,
        "seed": SEED,
        "num_samples": NUM_SAMPLES,
        "size_multiple": SIZE_MULTIPLE,
        "sigmas_255": SIGMAS_255,
        "results": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT_JSON}  ({len(results)} (set,sigma) entries)")


if __name__ == "__main__":
    main()
