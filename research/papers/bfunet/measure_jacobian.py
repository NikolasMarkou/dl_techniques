"""Measure the denoiser's LOCAL Jacobian asymmetry on the NEW [0,1] checkpoint.

One-off research probe (NOT library code). Reproduces the protocol frozen in
``analyses/analysis_2026-07-12_103e465c/probe2_jacobian.json`` against the new
checkpoint ``results/20260715_convunext_denoiser/best_model.keras``:

    - co-located 12x12x3 = 432-dim input/output block at top_left [122,122] of a
      256x256 noisy DIV2K-val crop (AWGN sigma_255=25, seed 42);
    - full 432x432 Jacobian J = dD/dy by CENTRAL finite differences (eps=1e-3) --
      reverse-mode autodiff is a documented TF2.18 jit-conv dead-end, do NOT use it;
    - asymmetry_ratio = ||J - J^T||_F / ||J||_F, stable_rank = ||J||_F^2 / ||J||_2^2;
    - an identically-measured symmetric box-blur CONTROL validates the pipeline
      (a true symmetric operator MUST give asymmetry << the model's and <~ 1e-3).

Run:
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python \\
        research/papers/bfunet/measure_jacobian.py \\
        [--checkpoint PATH] [--output PATH]

``--checkpoint`` defaults to the module constant below, so an argument-less call
behaves exactly as it always has. The checkpoint actually loaded is recorded in the
output JSON's ``checkpoint`` field -- always read that field, never assume the default.

Emits ``research/papers/bfunet/jacobian_results.json``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dl_techniques.utils.logger import logger
from train.bfunet.eval_psnr_vs_noise import (
    load_denoiser,
    add_awgn,
    sample_full_images,
    EvalConfig,
)
from train.common import collect_image_paths

# --- Frozen protocol constants (mirror probe2_jacobian.json) --------------------
CKPT = "results/20260715_convunext_denoiser/best_model.keras"
DIV2K_VAL = "/media/arxwn/data0_4tb/datasets/div2k/validation"
CROP = 256
BLOCK_H = BLOCK_W = 12
CHANNELS = 3
TOP_LEFT = (122, 122)
EPS = 1e-3
SIGMA_255 = 25.0
SEED = 42
OUT_JSON = "research/papers/bfunet/jacobian_results.json"
BOX_BLUR_K = 5  # odd uniform-average kernel size for the symmetric control


# --- Block index bookkeeping ----------------------------------------------------
def _block_coords():
    """Return the 432 (row, col, channel) coordinates of the co-located block,
    flattened in C-order (row-major over h, w, c)."""
    t, l = TOP_LEFT
    coords = []
    for dh in range(BLOCK_H):
        for dw in range(BLOCK_W):
            for c in range(CHANNELS):
                coords.append((t + dh, l + dw, c))
    return coords


def _extract_block(img):
    """Pull the co-located 12x12x3 patch from a (1,H,W,C) image -> (432,) vector."""
    t, l = TOP_LEFT
    patch = img[0, t:t + BLOCK_H, l:l + BLOCK_W, :]
    return np.asarray(patch, dtype=np.float64).reshape(-1)


# --- Finite-difference Jacobian over the co-located block -----------------------
def finite_diff_jacobian(op, y, coords):
    """Central-difference Jacobian of the co-located output block w.r.t. the
    co-located input block.

    Args:
        op:     callable (1,H,W,C) float32 array -> (1,H,W,C) float32 array.
        y:      base input, (1,H,W,C) float32.
        coords: list of 432 (row,col,ch) input coordinates (== output block).

    Returns:
        (432, 432) float64 Jacobian; column i is d(output block)/d(y at coord i).
    """
    n = len(coords)
    J = np.empty((n, n), dtype=np.float64)
    for i, (r, cc, ch) in enumerate(coords):
        yp = y.copy()
        ym = y.copy()
        yp[0, r, cc, ch] += EPS
        ym[0, r, cc, ch] -= EPS
        out_p = _extract_block(op(yp))
        out_m = _extract_block(op(ym))
        J[:, i] = (out_p - out_m) / (2.0 * EPS)
        if (i + 1) % 48 == 0:
            logger.info(f"  finite-diff column {i + 1}/{n}")
    return J


def spectral_metrics(J):
    """asymmetry_ratio, frob_norm, spectral_norm, stable_rank, top-10 singular values."""
    frob = float(np.linalg.norm(J, "fro"))
    asym = float(np.linalg.norm(J - J.T, "fro") / frob)
    svals = np.linalg.svd(J, compute_uv=False)
    spectral = float(svals[0])
    stable_rank = float(frob ** 2 / spectral ** 2)
    return {
        "asymmetry_ratio": asym,
        "frob_norm": frob,
        "spectral_norm": spectral,
        "stable_rank": stable_rank,
        "n_dims": int(J.shape[0]),
        "top_10_singular_values": [float(v) for v in svals[:10]],
    }


def make_box_blur(k):
    """Fixed odd-sized uniform-average conv applied per-channel (a symmetric linear
    operator). Uses reflect padding so the interior block matches the model's
    boundary treatment as closely as a pure blur can."""
    pad = k // 2

    def op(y):
        yp = np.pad(y, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(y, dtype=np.float64)
        for di in range(k):
            for dj in range(k):
                out += yp[:, di:di + y.shape[1], dj:dj + y.shape[2], :]
        out /= (k * k)
        return out.astype(np.float32)

    return op


def _parse_args():
    parser = argparse.ArgumentParser(description="Measure local Jacobian asymmetry")
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

    # --- Build the noisy input y in [0,1], (1,256,256,3) -----------------------
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
    sigma_norm = SIGMA_255 / 255.0
    noisy = add_awgn(clean, sigma_norm, clip=True, rng=noise_rng)  # [0,1]
    y = noisy[None, ...].astype(np.float32)
    logger.info(f"noisy input y: shape={y.shape} range=[{y.min():.4f},{y.max():.4f}]")

    coords = _block_coords()
    assert len(coords) == BLOCK_H * BLOCK_W * CHANNELS == 432

    # --- Denoiser operator (reduce multi-output to output 0) -------------------
    def denoiser_op(inp):
        pred = model.predict(inp, batch_size=1, verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return np.asarray(pred)

    logger.info("Building denoiser Jacobian (central finite differences, 432 dims)...")
    J_model = finite_diff_jacobian(denoiser_op, y, coords)
    assert np.all(np.isfinite(J_model)), "denoiser Jacobian has non-finite entries"
    m_metrics = spectral_metrics(J_model)
    m_metrics["name"] = "denoiser_local_jacobian"
    logger.info(f"denoiser asymmetry_ratio = {m_metrics['asymmetry_ratio']:.6f}")

    logger.info(f"Building box-blur CONTROL Jacobian ({BOX_BLUR_K}x{BOX_BLUR_K})...")
    box_op = make_box_blur(BOX_BLUR_K)
    J_box = finite_diff_jacobian(box_op, y, coords)
    assert np.all(np.isfinite(J_box)), "box-blur Jacobian has non-finite entries"
    b_metrics = spectral_metrics(J_box)
    b_metrics["name"] = "box_blur_baseline"
    b_metrics["kernel_size"] = BOX_BLUR_K
    logger.info(f"box-blur asymmetry_ratio = {b_metrics['asymmetry_ratio']:.6e}")

    ratio = m_metrics["asymmetry_ratio"] / b_metrics["asymmetry_ratio"]

    results = {
        "block": {"h": BLOCK_H, "w": BLOCK_W, "channels": CHANNELS,
                  "top_left": list(TOP_LEFT), "eps": EPS},
        "sigma_255": SIGMA_255,
        "seed": SEED,
        "finite_difference": "central",
        # The checkpoint ACTUALLY loaded (not the module default) -- the paper must key
        # off this field to know which model produced these numbers.
        "checkpoint": checkpoint,
        "denoiser": m_metrics,
        "baseline_box_blur": b_metrics,
        "ratio": ratio,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"wrote {out_path}")
    logger.info(f"checkpoint recorded in JSON: {checkpoint}")

    # --- Falsification gate (Pre-Mortem #1): control must be near-zero ---------
    ctrl = b_metrics["asymmetry_ratio"]
    den = m_metrics["asymmetry_ratio"]
    logger.info("=" * 68)
    logger.info(f"denoiser asymmetry   = {den:.6f}")
    logger.info(f"box-blur asymmetry   = {ctrl:.6e}")
    logger.info(f"ratio (den/control)  = {ratio:.1f}x")
    logger.info(f"denoiser stable_rank = {m_metrics['stable_rank']:.3f}")
    logger.info(f"denoiser top sv      = {m_metrics['top_10_singular_values'][:3]}")
    gate_ok = (ctrl < den) and (ctrl <= 1e-3)
    logger.info(f"FALSIFICATION GATE (control << denoiser AND <= 1e-3): "
                f"{'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok:
        logger.error("CONTROL NOT NEAR-ZERO -> finite-diff pipeline is suspect; "
                     "do NOT trust the denoiser number.")
    # Pre-Mortem #2 status
    if den <= 10.0 * ctrl:
        logger.warning("PRE-MORTEM #2: denoiser asymmetry within ~1 order of the "
                       "control -> (near-)CONSERVATIVE residual (real finding).")
    else:
        logger.info("PRE-MORTEM #2: denoiser asymmetry >> control -> "
                    "NON-CONSERVATIVE residual (paper conclusion holds).")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
