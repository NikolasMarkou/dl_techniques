"""Measure degree-1 homogeneity error of the NEW [0,1] checkpoint.

One-off research probe (NOT library code). Companion to ``measure_jacobian.py``:
reuses the SAME checkpoint + the SAME noisy DIV2K-val patch (crop/sigma/seed) so the
two claim measurements sit on one identical input.

Homogeneity here is ``||D(a*y) - a*D(y)|| / ||a*D(y)||`` for a set of scale factors a.
This checkpoint uses BiasFreeBatchNorm (fixed running-var at inference -> linear scale),
LeakyReLU (degree-1 homogeneous for a>0) and a frozen bias-free Gabor stem, so it SHOULD
be homogeneous to float32 round-off (~2.5e-5 floor, see ddnm.py:104).

Two alpha sets are probed:
  - the paper's set {1/4,1/2,2,4,8} (powers of two -> bit-exact float32 scaling, can
    report a spuriously perfect 0.0);
  - ddnm's DEFAULT non-power-of-two set {0.3,1.3,3.7,6.1}, a STRONGER cross-check that a
    bit-exact power-of-two scaling is not masking a real homogeneity break.

FALSIFICATION: if any alpha (especially the non-pow2 set) gives relative error > 1e-2,
OR the error clearly GROWS with |log alpha|, that is a REAL non-homogeneity -- report the
true magnitude, do NOT call it round-off.

Run:
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python \\
        research/papers/bfunet/measure_homogeneity.py \\
        [--checkpoint PATH] [--output PATH]

``--checkpoint`` defaults to the module constant below, so an argument-less call
behaves exactly as it always has. The checkpoint actually loaded is recorded in the
output JSON's ``checkpoint`` field -- always read that field, never assume the default.

Emits ``research/papers/bfunet/homogeneity_results.json``.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dl_techniques.utils.logger import logger
from applications.bias_free_denoiser.ddnm import homogeneity_error
from train.bfunet.eval_psnr_vs_noise import (
    load_denoiser,
    add_awgn,
    sample_full_images,
    EvalConfig,
)
from train.common import collect_image_paths

# --- Frozen protocol constants (mirror measure_jacobian.py exactly) -------------
CKPT = "results/20260715_convunext_denoiser/best_model.keras"
DIV2K_VAL = "/media/arxwn/data0_4tb/datasets/div2k/validation"
CROP = 256
CHANNELS = 3
SIGMA_255 = 25.0
SEED = 42
OUT_JSON = "research/papers/bfunet/homogeneity_results.json"

PAPER_ALPHAS = (0.25, 0.5, 2.0, 4.0, 8.0)  # {1/4,1/2,2,4,8}: bit-exact in float32
# The non-power-of-two default set {0.3,1.3,3.7,6.1} lives in homogeneity_error's
# signature -- call it with no alphas= to exercise that stronger cross-check.

ROUNDOFF_FLOOR = 2.5e-5  # float32 rounding through a deep net (ddnm.py:104)
FALSIFICATION_TOL = 1e-2  # > this on any alpha => real non-homogeneity (Pre-Mortem #3)


def _build_noisy_y():
    """Reproduce measure_jacobian.py's noisy input y: a (1,256,256,3) [0,1] AWGN
    sigma_255=25 seed-42 crop of the first DIV2K-val image."""
    cfg = EvalConfig(models={}, datasets={}, num_samples=1,
                     patch_size=CROP, channels=CHANNELS, seed=SEED)
    paths = collect_image_paths([DIV2K_VAL], max_files=None)
    if not paths:
        raise FileNotFoundError(f"No DIV2K-val images at {DIV2K_VAL}")
    rng = np.random.RandomState(SEED)
    images = sample_full_images(cfg, paths, rng)
    if not images:
        raise RuntimeError("sample_full_images returned no readable images")
    full = images[0]
    if full.shape[0] < CROP or full.shape[1] < CROP:
        raise RuntimeError(f"image too small for a {CROP} crop: {full.shape}")
    clean = full[:CROP, :CROP, :].astype(np.float32)
    noise_rng = np.random.RandomState(SEED)
    noisy = add_awgn(clean, SIGMA_255 / 255.0, clip=True, rng=noise_rng)  # [0,1]
    y = noisy[None, ...].astype(np.float32)
    logger.info(f"noisy input y: shape={y.shape} range=[{y.min():.4f},{y.max():.4f}]")
    return y


def _parse_args():
    parser = argparse.ArgumentParser(description="Measure degree-1 homogeneity error")
    parser.add_argument("--checkpoint", type=str, default=CKPT,
                        help="Path to the saved .keras denoiser.")
    parser.add_argument("--output", type=str, default=OUT_JSON,
                        help="Path of the results JSON to write.")
    return parser.parse_args()


def main():
    args = _parse_args()
    checkpoint = args.checkpoint

    logger.info(f"Loading checkpoint via load_denoiser: {checkpoint}")
    model = load_denoiser(checkpoint)  # goes through require_unit_domain_checkpoint gate

    y = _build_noisy_y()

    logger.info(f"homogeneity_error over PAPER alphas {PAPER_ALPHAS} ...")
    paper = homogeneity_error(model, y, alphas=PAPER_ALPHAS)
    logger.info(f"homogeneity_error over DEFAULT non-pow2 alphas (0.3,1.3,3.7,6.1) ...")
    nonpow2 = homogeneity_error(model, y)  # ddnm default {0.3,1.3,3.7,6.1}

    paper_str = {str(a): float(e) for a, e in paper.items()}
    nonpow2_str = {str(a): float(e) for a, e in nonpow2.items()}
    all_errors = list(paper.values()) + list(nonpow2.values())
    assert all(math.isfinite(e) for e in all_errors), "non-finite homogeneity error"
    max_error = float(max(all_errors))

    results = {
        # The checkpoint ACTUALLY loaded (not the module default) -- the paper must key
        # off this field to know which model produced these numbers.
        "checkpoint": checkpoint,
        "sigma_255": SIGMA_255,
        "seed": SEED,
        "roundoff_floor_note": (
            "float32 rounding through a deep net lands ~2.5e-5 (ddnm.py:104); "
            "a non-homogeneous (e.g. LayerNorm) checkpoint lands ~0.9"
        ),
        "roundoff_floor": ROUNDOFF_FLOOR,
        "falsification_tol": FALSIFICATION_TOL,
        "paper_alphas": paper_str,
        "nonpow2_alphas": nonpow2_str,
        "max_error": max_error,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"wrote {out_path}")
    logger.info(f"checkpoint recorded in JSON: {checkpoint}")

    # --- Report + Pre-Mortem #3 falsification gate -----------------------------
    logger.info("=" * 68)
    logger.info("PAPER alphas {1/4,1/2,2,4,8} (bit-exact float32):")
    for a in PAPER_ALPHAS:
        logger.info(f"  alpha={a:<5} rel_error = {paper[float(a)]:.3e}")
    logger.info("DEFAULT non-pow2 alphas (stronger cross-check):")
    for a in sorted(nonpow2.keys()):
        logger.info(f"  alpha={a:<5} rel_error = {nonpow2[a]:.3e}")
    logger.info(f"MAX error over BOTH sets = {max_error:.3e}")
    logger.info(f"float32 round-off floor  = {ROUNDOFF_FLOOR:.1e}")

    fired = max_error > FALSIFICATION_TOL
    # spread test: does the non-pow2 error grow with |log alpha|?
    nz = sorted(nonpow2.items(), key=lambda kv: abs(math.log(kv[0])))
    growing = len(nz) >= 2 and nz[-1][1] > 10.0 * (nz[0][1] + 1e-30)
    if fired:
        logger.error(f"FALSIFICATION FIRED (Pre-Mortem #3): max error {max_error:.3e} "
                     f"> {FALSIFICATION_TOL:.0e} -> REAL non-homogeneity, NOT round-off.")
    elif growing:
        logger.warning(f"SPREAD WARNING: non-pow2 error grows with |log alpha| "
                       f"({nz[0][1]:.2e} -> {nz[-1][1]:.2e}); inspect before calling round-off.")
    else:
        logger.info("VERDICT: genuine float32 round-off (flat, all errors <= tol). "
                    "BiasFreeBatchNorm + LeakyReLU homogeneity confirmed.")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
