"""Post-hoc per-pixel uncertainty validation for a FROZEN bias-free denoiser.

Standalone eval script (a sibling of ``eval_psnr_vs_noise.py``) that wires the
post-hoc uncertainty-quantification utilities onto a trained, frozen additive-
Gaussian denoiser checkpoint and produces a coverage/width validation table plus
per-pixel diagnostic PNG maps. NOTHING here retrains or edits the model — the
denoiser is called black-box (a forward pass only).

Pipeline
--------
1. Load a frozen ``.keras`` denoiser (``compile=False``) with the Gabor-stem +
   bfconvunext custom objects registered (mirrors
   ``multiplicative_miyasawa.run_checkpoint_diagnostic``). Deep-supervision list
   outputs are unwrapped to index-0 downstream.
2. Build ONE fixed clean val batch (reusing ``common.build_fixed_val_batch`` over
   the checkpoint's sibling ``config.json`` val dirs), then split it into disjoint
   calibration and test patch sets.
3. For each sigma bin (all inside the trained curriculum ``[~0.025, 0.25]``), add
   additive Gaussian noise clipped to ``[-0.5, +0.5]`` and:
   * ``calibrate_per_sigma`` on the calibration split -> a per-sigma radius ``q``;
   * ``evaluate_coverage`` on the INDEPENDENT test split -> empirical coverage +
     mean interval width; plus a test-split PSNR (``common._mean_psnr``).
   The per-sigma table (``sigma | n_calib | q | test_coverage | target | width |
   test_PSNR``) is printed to stdout.
4. Save per-pixel diagnostic PNGs for a few example patches at a representative
   sigma: the ``|clean - denoised|`` error map, the conformal interval-width map
   (constant ``2q`` here, annotated), and the additive MC-SURE risk map
   (``additive_sure_risk_map``). Pixels within ``epsilon`` of the ``+/-0.5`` clip
   boundary are masked in the SURE panel and a one-line caveat is printed, because
   the clip breaks strict Gaussianity there and biases the SURE estimate.

Caveats (see ``conformal_denoiser_intervals`` / ``multiplicative_miyasawa`` docs):
coverage is MARGINAL and per-sigma (Mondrian), the interval is homoscedastic
(a single scalar ``q`` per sigma), and the MC-SURE map is ADDITIVE-ONLY and
systematically biased near the ``+/-0.5`` saturation boundary.

Usage::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.eval_per_pixel_uncertainty \\
        --checkpoint results/20260701_convunext_denoiser/best_model.keras --gpu 1

GPU note: GPU0 is typically busy with live training, so this script defaults to
GPU1 and sets ``CUDA_VISIBLE_DEVICES`` BEFORE importing TensorFlow. NEVER run it
on GPU0 in parallel with a training job.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# GPU selection MUST happen before TensorFlow is imported: TF grabs the
# visible CUDA devices at import time, so CUDA_VISIBLE_DEVICES has to be
# set first. Default is GPU1 (RTX 4070) because GPU0 (RTX 4090) is usually
# occupied by a live training run and parallel GPU jobs are forbidden.
# DECISION plan_2026-07-06_2d62fdd4/D-002: validate on the frozen convunext
# checkpoint on GPU1, never in parallel with the GPU0 training job. Do NOT
# switch the default to GPU0 or move this below the TF import.
# ---------------------------------------------------------------------

DEFAULT_GPU: int = 1


def _preparse_gpu(argv: List[str], default: int = DEFAULT_GPU) -> str:
    """Extract ``--gpu N`` / ``--gpu=N`` from argv before argparse/TF import.

    Returns the GPU id as a string suitable for ``CUDA_VISIBLE_DEVICES``. Falls
    back to ``default`` when the flag is absent or malformed (never GPU0 by
    default). ``--gpu -1`` disables CUDA (empty ``CUDA_VISIBLE_DEVICES``).
    """
    gpu = default
    for i, tok in enumerate(argv):
        if tok == "--gpu" and i + 1 < len(argv):
            try:
                gpu = int(argv[i + 1])
            except ValueError:
                gpu = default
        elif tok.startswith("--gpu="):
            try:
                gpu = int(tok.split("=", 1)[1])
            except ValueError:
                gpu = default
    return "" if gpu is not None and gpu < 0 else str(gpu)


os.environ["CUDA_VISIBLE_DEVICES"] = _preparse_gpu(sys.argv)

import matplotlib
matplotlib.use("Agg")  # headless: avoid X11 crashes
import matplotlib.pyplot as plt

import keras

from dl_techniques.utils.logger import logger
from dl_techniques.utils.conformal_denoiser_intervals import (
    calibrate_per_sigma,
    calibrate_per_sigma_normalized,
    predict_intervals,
    evaluate_coverage,
    evaluate_coverage_normalized,
)
from dl_techniques.utils.multiplicative_miyasawa import additive_sure_risk_map

# Importing these registers every custom Keras object the saved denoiser needs
# (Gabor-stem initializer + bfconvunext ConvNeXt blocks / Laplacian pyramid /
# LayerScale) so ``keras.models.load_model`` resolves them from the registry.
import dl_techniques.initializers.gabor_filters_initializer  # noqa: F401,E402
import dl_techniques.models.bias_free_denoisers.bfconvunext  # noqa: F401,E402

# Reuse the bfunet trainer's val-batch loader, PSNR convention, and the sibling
# eval script's additive-noise helper (single source of truth for each) rather
# than re-implementing them here.
from train.bfunet.common import (  # noqa: E402
    BFUnetTrainingConfig,
    build_fixed_val_batch,
    _mean_psnr,
)
from train.bfunet.eval_psnr_vs_noise import add_awgn  # noqa: E402
from train.common import collect_image_paths, set_seeds  # noqa: E402


DEFAULT_SIGMAS: List[float] = [0.05, 0.10, 0.15, 0.20, 0.25]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
BOUNDARY_EPS: float = 0.02  # |value| within this of +/-0.5 -> masked in SURE panel

# Conformal schemes: (calibrator, coverage-evaluator). The naive per-sigma scheme
# is the historical scalar-radius baseline (kept byte-identical); 'normalized' is
# the variance-scaled per-image-normalized band. Both share the SAME calib/test
# split so they are compared apples-to-apples.
_SCHEME_SPECS = {
    "naive": (calibrate_per_sigma, evaluate_coverage),
    "normalized": (calibrate_per_sigma_normalized, evaluate_coverage_normalized),
}
SCHEME_LABELS = {"naive": "naive per-s", "normalized": "variance-scaled"}
VALID_SCHEMES = tuple(_SCHEME_SPECS.keys())


def _parse_schemes(raw: str) -> List[str]:
    """Parse a comma-separated ``--schemes`` string into an ordered, de-duped list.

    Unknown names raise (fail loud rather than silently skipping a requested
    scheme). Order is preserved so ``naive,normalized`` prints naive first.
    """
    seen: List[str] = []
    for tok in str(raw).split(","):
        name = tok.strip().lower()
        if not name:
            continue
        if name not in VALID_SCHEMES:
            raise ValueError(
                f"unknown scheme {name!r}; valid: {', '.join(VALID_SCHEMES)}"
            )
        if name not in seen:
            seen.append(name)
    if not seen:
        raise ValueError("no valid scheme selected via --schemes")
    return seen


def _overall_row(rows: List[dict]) -> dict:
    """Sigma-pooled summary of per-sigma rows.

    Each sigma bin contributes the SAME number of test pixels (same
    ``clean_test``), so the pixel-pooled coverage/width equals the equal-weight
    mean of the per-sigma values; PSNR is likewise averaged over sigmas.
    """
    return {
        "coverage": float(np.mean([r["coverage"] for r in rows])),
        "mean_width": float(np.mean([r["mean_width"] for r in rows])),
        "test_psnr": float(np.mean([r["test_psnr"] for r in rows])),
    }


# ---------------------------------------------------------------------
# Checkpoint + data
# ---------------------------------------------------------------------


def _load_eval_config(checkpoint: str) -> BFUnetTrainingConfig:
    """Build a config from the checkpoint's sibling ``config.json`` (val dirs, patch).

    Reads ``patch_size`` / ``channels`` / ``val_image_dirs`` (and the sigma range,
    for logging) from the training config saved next to the checkpoint. Falls back
    to ``BFUnetTrainingConfig`` defaults when the file is absent.
    """
    cfg_path = Path(checkpoint).parent / "config.json"
    kwargs = {}
    if cfg_path.is_file():
        raw = json.loads(cfg_path.read_text())
        for key in ("patch_size", "channels", "val_image_dirs"):
            if key in raw and raw[key]:
                kwargs[key] = raw[key]
        if raw.get("noise_type", "additive") != "additive":
            logger.warning(
                "checkpoint config noise_type=%r is not 'additive'; MC-SURE map "
                "is additive-only and its unbiased-MSE identity will not hold.",
                raw.get("noise_type"),
            )
        logger.info(
            "loaded eval config from %s (sigma range %.3f..%.3f)", cfg_path,
            raw.get("noise_sigma_min", 0.0), raw.get("sigma_max_end", 0.25),
        )
    else:
        logger.warning("no config.json beside checkpoint; using config defaults")
    return BFUnetTrainingConfig(**kwargs)


def _load_denoiser(checkpoint: str) -> keras.Model:
    """Load the frozen ``.keras`` denoiser (weights + graph), compile-free."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    model = keras.models.load_model(path, compile=False)
    n_out = len(model.outputs) if isinstance(model.outputs, (list, tuple)) else 1
    logger.info(
        "loaded frozen denoiser '%s' (%s params, %d output(s)) from %s",
        model.name, f"{model.count_params():,}", n_out, checkpoint,
    )
    return model


def _prepare_clean_batch(
    config: BFUnetTrainingConfig, n_total: int, seed: int
) -> np.ndarray:
    """Load one fixed clean [-0.5,+0.5] batch of ``n_total`` patches from val dirs.

    Reuses ``common.build_fixed_val_batch`` (which loads one random crop per val
    image). Fails loudly when no readable images are found rather than silently
    emitting uncalibrated intervals.
    """
    set_seeds(seed)
    val_paths = collect_image_paths(
        config.val_image_dirs, extensions=IMAGE_EXTENSIONS, sort=True
    )
    if not val_paths:
        raise RuntimeError(
            f"no images found under val dirs {config.val_image_dirs}; cannot "
            "build a calibration/test batch."
        )
    batch = build_fixed_val_batch(val_paths, config, n=n_total)
    if batch is None:
        raise RuntimeError(
            "build_fixed_val_batch returned None (no readable patches); cannot "
            "calibrate — refusing to emit uncalibrated intervals."
        )
    clean = np.asarray(batch, dtype=np.float32)
    if clean.shape[0] < n_total:
        logger.warning(
            "requested %d patches but only loaded %d (too few readable val images?)",
            n_total, clean.shape[0],
        )
    return clean


# ---------------------------------------------------------------------
# Calibration + coverage table
# ---------------------------------------------------------------------


def _build_calibration_sets(
    clean_calib: np.ndarray, sigmas: List[float], seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-sigma noisy calibration patches with per-sample sigma labels.

    For every sigma the SAME clean calibration patches are re-noised (a fresh
    additive-Gaussian realization clipped to [-0.5,+0.5]) and concatenated, with a
    per-sample sigma label array so ``calibrate_per_sigma`` can form Mondrian bins.
    """
    rng = np.random.RandomState(seed)
    clean_stack, noisy_stack, sigma_stack = [], [], []
    for sigma in sigmas:
        noisy = add_awgn(clean_calib, float(sigma), clip=True, rng=rng)
        clean_stack.append(clean_calib)
        noisy_stack.append(noisy)
        sigma_stack.append(np.full(clean_calib.shape[0], float(sigma), np.float64))
    return (
        np.concatenate(clean_stack, axis=0),
        np.concatenate(noisy_stack, axis=0),
        np.concatenate(sigma_stack, axis=0),
    )


def _print_table(rows: List[dict], alpha: float) -> None:
    """Print the per-sigma coverage-vs-target + mean-width validation table."""
    target = 1.0 - alpha
    header = (
        f"{'sigma':>7} | {'n_calib':>7} | {'q':>9} | {'test_cov':>9} | "
        f"{'target':>7} | {'mean_width':>10} | {'test_PSNR':>9}"
    )
    print("=" * len(header))
    print("Per-pixel split-conformal coverage (per-sigma / Mondrian)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['sigma']:>7.3f} | {r['n_calib']:>7d} | {r['q']:>9.5f} | "
            f"{r['coverage']:>9.4f} | {target:>7.2f} | {r['mean_width']:>10.5f} | "
            f"{r['test_psnr']:>9.3f}"
        )
    print("=" * len(header))


def _print_combined_table(scheme_rows: Dict[str, List[dict]], alpha: float) -> None:
    """Print the multi-scheme coverage table: per sigma + a sigma-pooled overall row.

    One block per scheme (naive per-sigma, variance-scaled), each with its
    per-sigma rows followed by an ``overall`` (sigma-pooled) summary. ``q`` is a
    scalar radius for the naive scheme and a NORMALIZED radius for the variance-
    scaled scheme (multiplied by the per-image scale at test time), so the two
    ``q`` columns are not directly comparable — coverage/width are.
    """
    target = 1.0 - alpha
    header = (
        f"{'scheme':>14} | {'sigma':>7} | {'q':>9} | {'test_cov':>9} | "
        f"{'target':>7} | {'mean_width':>10} | {'test_PSNR':>9}"
    )
    print("=" * len(header))
    print("Per-pixel split-conformal coverage — multi-scheme (per-sigma / Mondrian)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for scheme, rows in scheme_rows.items():
        label = SCHEME_LABELS.get(scheme, scheme)
        for r in rows:
            print(
                f"{label:>14} | {r['sigma']:>7.3f} | {r['q']:>9.5f} | "
                f"{r['coverage']:>9.4f} | {target:>7.2f} | "
                f"{r['mean_width']:>10.5f} | {r['test_psnr']:>9.3f}"
            )
        ov = _overall_row(rows)
        print("-" * len(header))
        print(
            f"{label:>14} | {'overall':>7} | {'':>9} | "
            f"{ov['coverage']:>9.4f} | {target:>7.2f} | "
            f"{ov['mean_width']:>10.5f} | {ov['test_psnr']:>9.3f}"
        )
        print("-" * len(header))
    print("=" * len(header))


# ---------------------------------------------------------------------
# Per-pixel PNG maps
# ---------------------------------------------------------------------


def _to_display(patch: np.ndarray) -> np.ndarray:
    """Shift a [-0.5,+0.5] patch to [0,1] for imshow (grayscale collapse if 1ch)."""
    img = np.clip(patch + 0.5, 0.0, 1.0)
    return img[..., 0] if img.shape[-1] == 1 else img


def _save_uncertainty_maps(
    model: keras.Model,
    clean_test: np.ndarray,
    sigma: float,
    q: float,
    out_dir: Path,
    n_examples: int,
    n_hutchinson: int,
    seed: int,
) -> List[Path]:
    """Save per-pixel |error| / interval-width / MC-SURE PNGs at a representative sigma.

    Mirrors the multi-panel grid style of ``DenoisingVisualizationCallback``: one
    row per example patch, columns = clean, noisy, denoised, |error|, width, SURE.
    The SURE panel masks pixels within ``BOUNDARY_EPS`` of the +/-0.5 clip boundary
    (where the clip breaks Gaussianity and the additive-SURE estimate is biased).
    """
    rng = np.random.RandomState(seed)
    n = min(n_examples, clean_test.shape[0])
    clean_ex = clean_test[:n]
    noisy_ex = add_awgn(clean_ex, float(sigma), clip=True, rng=rng)

    mu, lower, upper = predict_intervals(model, noisy_ex, q)
    error = np.abs(clean_ex - mu)                       # per-pixel |clean - denoised|
    width = upper - lower                               # constant 2q here (annotated)

    # Additive MC-SURE per-pixel risk map over the example batch (additive-only).
    denoiser = lambda t: model(t, training=False)  # noqa: E731
    sure_map = np.asarray(
        additive_sure_risk_map(
            denoiser, noisy_ex.astype(np.float32), float(sigma),
            n_hutchinson=n_hutchinson, seed=seed,
        )
    )  # shape [H, W, C], batch-reduced

    # Saturation mask: pixels near +/-0.5 in the noisy input are boundary-biased.
    sat_mask = (np.abs(noisy_ex) >= (0.5 - BOUNDARY_EPS))  # per-example [n,H,W,C]
    sat_frac = float(np.mean(sat_mask))
    logger.info(
        "SURE map: sigma=%.3f mean=%.4g; %.2f%% of pixels within eps=%.3f of the "
        "+/-0.5 clip boundary (masked in the SURE panel — boundary-biased).",
        sigma, float(np.mean(sure_map)), 100.0 * sat_frac, BOUNDARY_EPS,
    )

    err_max = float(max(error.max(), 1e-6))
    fig, axes = plt.subplots(n, 6, figsize=(6 * 2.6, n * 2.6), squeeze=False)
    col_titles = ["clean", f"noisy (σ={sigma:.3f})", "denoised μ",
                  "|clean-μ|", f"width (=2q={2*q:.4f})", "MC-SURE risk"]
    for i in range(n):
        panels = [
            (_to_display(clean_ex[i]), None, None),
            (_to_display(noisy_ex[i]), None, None),
            (_to_display(mu[i]), None, None),
            (error[i].mean(-1), 0.0, err_max),
            (width[i].mean(-1), 0.0, float(width.max())),
            (None, None, None),  # SURE handled below (batch-reduced, masked)
        ]
        for j, (data, vmin, vmax) in enumerate(panels):
            ax = axes[i, j]
            if j == 5:
                sure_disp = np.ma.masked_where(
                    sat_mask[i].any(-1), sure_map.mean(-1)
                )
                im = ax.imshow(sure_disp, cmap="inferno")
                fig.colorbar(im, ax=ax, fraction=0.046)
            elif j >= 3:
                im = ax.imshow(data, cmap="magma", vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.imshow(data, cmap=None if data.ndim == 3 else "gray")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"Per-pixel uncertainty maps — σ={sigma:.3f}, q={q:.5f}  "
        f"(SURE additive-only; {100*sat_frac:.1f}% boundary pixels masked)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    grid_path = out_dir / f"uncertainty_maps_sigma_{sigma:.3f}.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved per-pixel uncertainty maps -> %s", grid_path)
    return [grid_path]


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------


def run_evaluation(args: argparse.Namespace) -> Path:
    """Load the frozen checkpoint, calibrate per sigma, print the table, save maps."""
    config = _load_eval_config(args.checkpoint)
    if args.patch_size:
        config.patch_size = args.patch_size
    if args.val_dirs:
        config.val_image_dirs = list(args.val_dirs)
        logger.info("val dirs overridden -> %s", config.val_image_dirs)
    model = _load_denoiser(args.checkpoint)

    sigmas = [float(s) for s in args.sigmas]
    n_total = args.n_calib + args.n_test
    clean = _prepare_clean_batch(config, n_total, seed=args.seed)

    # Disjoint calibration / test splits (no patch appears in both).
    n_calib = min(args.n_calib, clean.shape[0] - 1)
    clean_calib = clean[:n_calib]
    clean_test = clean[n_calib:]
    logger.info(
        "split %d patches -> %d calibration / %d test (patch=%d, channels=%d)",
        clean.shape[0], clean_calib.shape[0], clean_test.shape[0],
        config.patch_size, config.channels,
    )

    schemes = _parse_schemes(args.schemes)
    logger.info("conformal schemes: %s", ", ".join(schemes))

    # Per-sigma Mondrian calibration set (shared across ALL schemes so they are
    # calibrated on the identical calib split).
    clean_c, noisy_c, sigma_c = _build_calibration_sets(
        clean_calib, sigmas, seed=args.seed
    )

    # Pre-draw the independent test-split noise ONCE per sigma so every scheme is
    # evaluated on byte-identical test inputs. Drawing them in sigma order with a
    # RandomState(seed+1) reproduces the historical naive-path noise draws exactly
    # (test noise != calibration noise), keeping the existing 0.877 result stable.
    rng = np.random.RandomState(args.seed + 1)
    noisy_test_by_sigma = {
        float(sigma): add_awgn(clean_test, float(sigma), clip=True, rng=rng)
        for sigma in sigmas
    }

    scheme_rows: Dict[str, List[dict]] = {}
    q_by_sigma_by_scheme: Dict[str, Dict[float, float]] = {}
    for scheme in schemes:
        calibrate_fn, evaluate_fn = _SCHEME_SPECS[scheme]
        q_by_sigma = calibrate_fn(
            model, clean_c, noisy_c, sigma_c, alpha=args.alpha,
            batch_size=args.batch_size,
        )
        rows: List[dict] = []
        for sigma in sigmas:
            q = q_by_sigma[float(sigma)]
            noisy_test = noisy_test_by_sigma[float(sigma)]
            cov = evaluate_fn(model, clean_test, noisy_test, q, batch_size=args.batch_size)
            mu, _, _ = predict_intervals(model, noisy_test, q, batch_size=args.batch_size)  # PSNR (mu is
            rows.append({                                       # scheme-agnostic)
                "sigma": float(sigma),
                "n_calib": int(clean_calib.shape[0]),
                "q": float(q),
                "coverage": cov["coverage"],
                "mean_width": cov["mean_width"],
                "test_psnr": _mean_psnr(mu, clean_test),
            })
        scheme_rows[scheme] = rows
        q_by_sigma_by_scheme[scheme] = q_by_sigma

    # Backward-compatible output: only the naive scheme -> the EXACT historical
    # single-scheme table (existing 0.877 result stays reproducible). Otherwise
    # emit the combined multi-scheme table (per sigma + sigma-pooled overall).
    if schemes == ["naive"]:
        _print_table(scheme_rows["naive"], alpha=args.alpha)
    else:
        _print_combined_table(scheme_rows, alpha=args.alpha)

    # Output dir + PNG maps at a representative sigma (median of the requested set).
    name = args.experiment_name or (
        f"per_pixel_uncertainty_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.output_dir) / name
    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    # The uncertainty-map panel annotates a scalar radius (width = 2q). Use the
    # naive scheme's q when available (historical behavior); otherwise the first
    # selected scheme's q (a normalized radius, noted in the panel semantics).
    map_scheme = "naive" if "naive" in schemes else schemes[0]
    map_q_by_sigma = q_by_sigma_by_scheme[map_scheme]

    repr_sigma = sigmas[len(sigmas) // 2]
    _save_uncertainty_maps(
        model, clean_test, repr_sigma, map_q_by_sigma[float(repr_sigma)],
        out_dir / "visualizations", n_examples=args.n_examples,
        n_hutchinson=args.n_hutchinson, seed=args.seed,
    )

    # JSON: naive-only dumps the EXACT historical shape (byte-identical); the
    # multi-scheme case dumps a per-scheme structure plus sigma-pooled overalls.
    if schemes == ["naive"]:
        payload = {
            "checkpoint": args.checkpoint, "alpha": args.alpha,
            "target_coverage": 1.0 - args.alpha, "rows": scheme_rows["naive"],
            "q_by_sigma": {
                str(k): v for k, v in q_by_sigma_by_scheme["naive"].items()
            },
        }
    else:
        payload = {
            "checkpoint": args.checkpoint, "alpha": args.alpha,
            "target_coverage": 1.0 - args.alpha, "schemes": schemes,
            "rows_by_scheme": scheme_rows,
            "overall_by_scheme": {
                s: _overall_row(r) for s, r in scheme_rows.items()
            },
            "q_by_sigma_by_scheme": {
                s: {str(k): v for k, v in q.items()}
                for s, q in q_by_sigma_by_scheme.items()
            },
        }
    with open(out_dir / "coverage_table.json", "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("done; results in %s", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc per-pixel split-conformal + MC-SURE uncertainty validation "
            "for a FROZEN bias-free additive-Gaussian denoiser checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to the frozen denoiser best_model.keras")
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS,
                        help="noise stds to calibrate/report (all within the trained "
                             "curriculum range ~[0.025, 0.25])")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="miscoverage level; target coverage is 1-alpha (0.1 -> 90%%)")
    parser.add_argument("--schemes", type=str, default="naive",
                        help="comma-separated conformal schemes to run/report: "
                             "'naive' (per-sigma scalar radius) and/or "
                             "'normalized' (variance-scaled per-image band). "
                             "Default 'naive' keeps the historical single-scheme "
                             "output byte-identical.")
    parser.add_argument("--n-calib", type=int, default=32,
                        help="clean patches used for the calibration split")
    parser.add_argument("--n-test", type=int, default=32,
                        help="disjoint clean patches used for the test split")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="patch size (matches training; overrides config.json)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="forward-pass batch size (lower to fit a 12GB GPU at "
                             "256px full-res expand activations; e.g. 4)")
    parser.add_argument("--val-dirs", type=str, nargs="+", default=None,
                        help="override config.json val_image_dirs (one crop per image); "
                             "point at a large held-out pool (e.g. COCO val2017) for "
                             "a statistically powerful conformal calibration set")
    parser.add_argument("--n-examples", type=int, default=4,
                        help="example patches rendered in the per-pixel map grid")
    parser.add_argument("--n-hutchinson", type=int, default=8,
                        help="Hutchinson probes for the MC-SURE per-pixel map")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="repo-root results dir for the eval output")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU,
                        help="GPU id (default 1; GPU0 is usually busy with training)")
    args = parser.parse_args()

    logger.info("CUDA_VISIBLE_DEVICES=%r", os.environ.get("CUDA_VISIBLE_DEVICES"))
    run_evaluation(args)


if __name__ == "__main__":
    main()
