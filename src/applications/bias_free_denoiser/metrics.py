"""GUI-free reconstruction-quality measurement harness for the inverse-problem app.

This module scores how well the :class:`UniversalInverseSolver` (driven by the
frozen :class:`DenoiserPrior`) reconstructs a clean target after each operator's
forward degradation. It reports **PSNR** and **SSIM** in the ``[0, 1]`` display
domain, per task and per solver config, plus a mean-over-tasks scalar — the empirical
spine the plan uses to accept or reject each regime / default change.

Boundary (INV-7 / H7 — GUI-free core)
--------------------------------------
This module imports **NO** GUI framework (no ``streamlit``). It reuses the existing
app machinery verbatim:

* :func:`applications.bias_free_denoiser.main.build_operator` — the operator factory
  (do NOT re-implement the per-problem operator switch).
* :class:`applications.bias_free_denoiser.denoiser_prior.DenoiserPrior` — ``denoise``
  and the domain helpers ``ingest`` / ``denorm`` (``[-0.5, +0.5]`` <-> ``[0, 1]``).
* :class:`applications.bias_free_denoiser.solver.UniversalInverseSolver` — constructed
  with EXPLICIT kwargs so callers A/B configs without touching library defaults.

Domain (INV-1 / D-002)
----------------------
All solver/operator math stays in ``[-0.5, +0.5]`` (center ``c0 = 0.0``); PSNR/SSIM
are computed on the ``[0, 1]`` denormed reconstruction against the ``[0, 1]`` denormed
clean target. Spectral-deblur / compressive-sensing intermediates may be complex, but
the solver returns a REAL signal-domain iterate (``best_y``), so scoring is always on a
real image (plan edge case).
"""

import argparse
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger

from .denoiser_prior import DenoiserPrior
from .main import build_operator
from .solver import UniversalInverseSolver

# Tasks the harness can score. Mirrors main._ALL_PROBLEMS (single source of truth for
# the problem ids); "denoise" is the single-pass control with no operator/solver.
DENOISE_TASK = "denoise"
PRIOR_TASK = "prior"

# Default per-problem operator knobs (mirror main.parse_args defaults). block=None ->
# build_operator resolves it to size//4. Fed to build_operator via a lightweight
# namespace so we reuse the exact operator switch instead of re-implementing it.
_DEFAULT_OP_KWARGS: Dict[str, Any] = dict(
    block=None,
    keep_ratio=0.3,
    sr_factor=4,
    keep_fraction=0.15,
    measurement_ratio=0.2,
)

# Default synthetic-noise std for the denoise control (in-domain, matches main.py).
_DEFAULT_NOISE_SIGMA = 0.1

# PSNR guard: floor the MSE and cap the dB so identical images report a finite ceiling
# (100 dB) rather than +inf (plan edge case).
_PSNR_MSE_FLOOR = 1e-10
_PSNR_CAP_DB = 100.0


# ---------------------------------------------------------------------------
# Metric primitives (operate on [0, 1]-domain images)
# ---------------------------------------------------------------------------


def _as_float01(x: Any) -> np.ndarray:
    """Materialize an array-like as a contiguous float32 numpy array."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def _to_bhwc(x: Any) -> tf.Tensor:
    """Coerce a ``[H, W, C]`` or ``[B, H, W, C]`` array to a float32 4-D tensor."""
    arr = _as_float01(x)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"expected a 3-D or 4-D image, got shape {arr.shape}")
    return tf.convert_to_tensor(arr, dtype=tf.float32)


def psnr(a: Any, b: Any, *, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB) between two ``[0, 1]``-domain images.

    Computed as ``10 * log10(max_val^2 / MSE)`` with the MSE floored at
    ``1e-10`` and the result capped at ``100.0`` dB, so two identical images
    return a finite ceiling instead of ``+inf`` (plan edge case). Accepts
    ``[H, W, C]`` or ``[B, H, W, C]`` inputs (global MSE over all elements).

    Args:
        a: Reconstruction, values in ``[0, 1]``.
        b: Ground-truth reference, values in ``[0, 1]``.
        max_val: Dynamic range of the images (``1.0`` for ``[0, 1]``).

    Returns:
        The (capped) PSNR in decibels as a Python float.
    """
    a_np, b_np = _as_float01(a), _as_float01(b)
    mse = float(np.mean((a_np - b_np) ** 2))
    mse = max(mse, _PSNR_MSE_FLOOR)
    value = 10.0 * float(np.log10((max_val ** 2) / mse))
    return float(min(value, _PSNR_CAP_DB))


def ssim(a: Any, b: Any, *, max_val: float = 1.0) -> float:
    """Structural similarity between two ``[0, 1]``-domain images.

    Thin wrapper over :func:`tf.image.ssim` (default ``11 x 11`` Gaussian window),
    so inputs must be at least ``11 x 11`` spatially. Accepts ``[H, W, C]`` or
    ``[B, H, W, C]`` inputs and returns the mean SSIM over the batch.

    Args:
        a: Reconstruction, values in ``[0, 1]``.
        b: Ground-truth reference, values in ``[0, 1]``.
        max_val: Dynamic range of the images (``1.0`` for ``[0, 1]``).

    Returns:
        The mean SSIM in ``[-1, 1]`` (typically ``[0, 1]``) as a Python float.
    """
    value = tf.image.ssim(_to_bhwc(a), _to_bhwc(b), max_val=max_val)
    return float(tf.reduce_mean(value))


# ---------------------------------------------------------------------------
# Forward-degrade -> reconstruct -> score
# ---------------------------------------------------------------------------


def _op_args(seed: int, op_kwargs: Optional[Dict[str, Any]]) -> argparse.Namespace:
    """Build the lightweight namespace ``build_operator`` reads its knobs from."""
    merged = dict(_DEFAULT_OP_KWARGS)
    if op_kwargs:
        merged.update(op_kwargs)
    merged["seed"] = int(seed)
    return SimpleNamespace(**merged)


def degrade_and_reconstruct(
    prior: Any,
    clean_pm05: np.ndarray,
    task: str,
    solver_kwargs: Optional[Dict[str, Any]] = None,
    *,
    seed: int = 0,
    noise_sigma: float = _DEFAULT_NOISE_SIGMA,
    op_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Forward-degrade a clean image, reconstruct it, and score PSNR/SSIM.

    For the six solver tasks this builds the task operator (via
    :func:`build_operator`), measures the clean target, runs
    :meth:`UniversalInverseSolver.solve`, and denorms the returned real
    signal-domain iterate to ``[0, 1]``. For the ``"denoise"`` control there is
    NO operator/solver: the harness adds in-domain Gaussian noise and scores a
    single-pass ``prior.denoise`` (mirroring ``main.run_problem``).

    Args:
        prior: A denoiser prior duck-typed to ``residual`` (and ``denoise`` for the
            denoise task) — e.g. a :class:`DenoiserPrior` or a compatible stub.
        clean_pm05: Clean signal ``[1, H, W, C]`` in ``[-0.5, +0.5]``.
        task: One of ``main._ALL_PROBLEMS`` (``denoise``/``prior``/``inpaint``/
            ``random_pixels``/``super_resolution``/``deblur``/``compressive_sensing``).
        solver_kwargs: Explicit kwargs forwarded to :class:`UniversalInverseSolver`
            (``sigma_0``, ``sigma_l``, ``h0``, ``h_max``, ``beta``, ``max_iterations``,
            ``patience``, ``seed``-independent). Ignored for the denoise task.
        seed: RNG seed for the operator, solver init, and denoise noise.
        noise_sigma: In-domain Gaussian noise std for the denoise control.
        op_kwargs: Optional per-problem operator overrides (``block``, ``keep_ratio``,
            ``sr_factor``, ``keep_fraction``, ``measurement_ratio``).

    Returns:
        A dict with ``task``, ``recon01`` (``[H, W, C]`` in ``[0, 1]``), ``psnr``,
        ``ssim`` (floats), and ``info`` (the solver convergence dict; ``{}`` for
        denoise).
    """
    clean = _as_float01(clean_pm05)
    if clean.ndim == 3:
        clean = clean[None, ...]
    # Ground truth for scoring: the clean image denormed to [0, 1].
    gt01 = np.clip(DenoiserPrior.denorm(clean[0]), 0.0, 1.0)

    if task == DENOISE_TASK:
        rng = np.random.default_rng(seed)
        if noise_sigma > 0.0:
            noise = rng.normal(0.0, noise_sigma, clean.shape).astype(np.float32)
            noisy = np.clip(clean + noise, -0.5, 0.5)
        else:
            noisy = clean
        denoised = np.asarray(keras.ops.convert_to_numpy(prior.denoise(noisy)))
        recon01 = np.clip(DenoiserPrior.denorm(denoised[0]), 0.0, 1.0)
        info: Dict[str, Any] = {}
    else:
        image_shape = tuple(int(s) for s in clean.shape[1:])
        operator = build_operator(task, image_shape, _op_args(seed, op_kwargs))
        solver = UniversalInverseSolver(prior, **(solver_kwargs or {}))
        if task == PRIOR_TASK:
            # Unconstrained prior sampling: no measurements, size via shape.
            recon, info = solver.solve(
                operator, measurements=None, shape=clean.shape, seed=seed
            )
        else:
            measurements = operator.measure(clean)
            recon, info = solver.solve(operator, measurements=measurements, seed=seed)
        # best_y is a REAL signal-domain iterate even for spectral/CS (plan edge case).
        recon_np = np.asarray(keras.ops.convert_to_numpy(recon))
        recon01 = np.clip(DenoiserPrior.denorm(recon_np[0]), 0.0, 1.0)

    return {
        "task": task,
        "recon01": recon01,
        "psnr": psnr(recon01, gt01),
        "ssim": ssim(recon01, gt01),
        "info": info,
    }


# ---------------------------------------------------------------------------
# Full harness: tasks x configs x images
# ---------------------------------------------------------------------------


def run_harness(
    prior: Any,
    images: Sequence[np.ndarray],
    tasks: Sequence[str],
    configs: Dict[str, Dict[str, Any]],
    *,
    seed: int = 0,
    noise_sigma: float = _DEFAULT_NOISE_SIGMA,
    op_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score every ``(task, config, image)`` and summarize PSNR/SSIM per config.

    Args:
        prior: The denoiser prior (see :func:`degrade_and_reconstruct`).
        images: Clean targets, each ``[1, H, W, C]`` or ``[H, W, C]`` in ``[-0.5, +0.5]``.
        tasks: Task ids to score (subset of ``main._ALL_PROBLEMS``).
        configs: Mapping ``name -> solver_kwargs`` (e.g. ``{"baseline": {...},
            "fixed": {...}}``); each value is forwarded to
            :class:`UniversalInverseSolver`.
        seed: Base RNG seed (offset per image for variety).
        noise_sigma: In-domain noise std for the denoise control.
        op_kwargs: Optional per-problem operator overrides.

    Returns:
        A dict with ``rows`` (one dict per ``(task, config, image_idx)`` holding
        ``task``, ``config``, ``image_idx``, ``psnr``, ``ssim``) and ``config_means``
        (``name -> {"psnr": mean, "ssim": mean}`` averaged over all rows of that
        config). The table is also logged.
    """
    rows: List[Dict[str, Any]] = []
    for cfg_name, solver_kwargs in configs.items():
        for img_idx, image in enumerate(images):
            for task in tasks:
                result = degrade_and_reconstruct(
                    prior,
                    image,
                    task,
                    solver_kwargs=solver_kwargs,
                    seed=seed + img_idx,
                    noise_sigma=noise_sigma,
                    op_kwargs=op_kwargs,
                )
                rows.append({
                    "task": task,
                    "config": cfg_name,
                    "image_idx": img_idx,
                    "psnr": result["psnr"],
                    "ssim": result["ssim"],
                })

    config_means: Dict[str, Dict[str, float]] = {}
    for cfg_name in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg_name]
        if cfg_rows:
            config_means[cfg_name] = {
                "psnr": float(np.mean([r["psnr"] for r in cfg_rows])),
                "ssim": float(np.mean([r["ssim"] for r in cfg_rows])),
            }

    _log_table(rows, config_means)
    return {"rows": rows, "config_means": config_means}


def _log_table(
    rows: Sequence[Dict[str, Any]],
    config_means: Dict[str, Dict[str, float]],
) -> None:
    """Log the per-row PSNR/SSIM table and the per-config mean scalars."""
    logger.info("=" * 68)
    logger.info("%-20s %-12s %6s %8s %8s", "task", "config", "img", "PSNR", "SSIM")
    logger.info("-" * 68)
    for r in rows:
        logger.info(
            "%-20s %-12s %6d %8.3f %8.4f",
            r["task"], r["config"], r["image_idx"], r["psnr"], r["ssim"],
        )
    logger.info("-" * 68)
    for cfg_name, means in config_means.items():
        logger.info(
            "MEAN %-15s over tasks: PSNR=%.3f dB  SSIM=%.4f",
            cfg_name, means["psnr"], means["ssim"],
        )
    logger.info("=" * 68)


# ---------------------------------------------------------------------------
# Optional CLI driver (NICE-TO-HAVE; the unit test is the Step-1 gate)
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args for the standalone harness driver."""
    p = argparse.ArgumentParser(
        description="Bias-free-denoiser reconstruction-quality harness (GUI-free).",
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best_model.keras or its results directory.")
    p.add_argument("--tasks", type=str, default="super_resolution,inpaint",
                   help="Comma-separated task ids (default super_resolution,inpaint).")
    p.add_argument("--size", type=int, default=256,
                   help="Square image edge (divisible by 8; default 256).")
    p.add_argument("--iterations", type=int, default=200,
                   help="Solver max iterations for the single default config.")
    p.add_argument("--h-max", type=str, default="0.1",
                   help="Step cap; 'none' for the uncapped paper schedule (default 0.1).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    return p.parse_args(argv)


def _synthetic_target(size: int) -> np.ndarray:
    """A smooth synthetic ``[1, size, size, 3]`` target in ``[-0.5, +0.5]``."""
    from .main import create_synthetic_test_image

    return create_synthetic_test_image((1, size, size, 3))


def main(argv: Optional[List[str]] = None) -> int:
    """Load a checkpoint, run the harness over the given tasks, and log the table."""
    args = _parse_args(argv)
    if args.size % 8 != 0:
        raise ValueError(f"--size must be divisible by 8; got {args.size}")

    h_max: Optional[float] = None if args.h_max.strip().lower() == "none" else float(args.h_max)
    solver_kwargs = dict(max_iterations=args.iterations, h_max=h_max)

    logger.info("loading denoiser prior from %s", args.checkpoint)
    prior = DenoiserPrior.from_pretrained(args.checkpoint)
    target = _synthetic_target(args.size)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    run_harness(prior, [target], tasks, {"default": solver_kwargs}, seed=args.seed)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
