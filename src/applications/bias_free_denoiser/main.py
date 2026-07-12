"""CLI demo for the bias-free-denoiser inverse-problem application.

Loads the frozen CliffordUNet denoiser as an implicit prior and runs the SAME
:class:`UniversalInverseSolver` loop over any subset of the six supported problems
(``prior``, ``inpaint``, ``random_pixels``, ``super_resolution``, ``deblur``,
``compressive_sensing``). It saves a matplotlib grid — target, measured/degraded
view, reconstruction, and the effective-noise convergence curve per problem — to
``results/`` and runs fully headless (``MPLBACKEND=Agg``, INV-7 / H7).

Every problem is selected purely by swapping a :class:`MeasurementOperator`; the
solver never branches on a problem type (INV-6). All pixels live in the model's
``[0, 1]`` domain (INV-1).

Examples:
    Run all six problems on the synthetic in-domain target at a modest budget::

        CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python \\
            -m applications.bias_free_denoiser.main --problem all --iterations 200

    Crank iterations for higher quality (convergence is coarse at low budgets —
    hundreds-to-1000 iterations give the paper-quality result)::

        CUDA_VISIBLE_DEVICES=1 .venv/bin/python \\
            -m applications.bias_free_denoiser.main --problem inpaint --iterations 1000

    Reconstruct a real image (resized to ``--size``, RGB)::

        CUDA_VISIBLE_DEVICES=1 .venv/bin/python \\
            -m applications.bias_free_denoiser.main --image path/to/img.png
"""

import os

# INV-7 / H7: force a non-interactive backend BEFORE importing pyplot so the demo
# never touches an X server on headless / remote GPU boxes.
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from dl_techniques.utils.logger import logger

from .denoiser_prior import DenoiserPrior
from .solver import UniversalInverseSolver
from .operators import (
    MeasurementOperator,
    NullOperator,
    InpaintingOperator,
    RandomPixelsOperator,
    SuperResolutionOperator,
    SpectralDeblurOperator,
    CompressiveSensingOperator,
)

# Repo-root results/ is the sanctioned output dir (never src/results/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results"
_DEFAULT_CHECKPOINT = (
    _REPO_ROOT / "results" / "convunext_denoiser_base_20260707_122133" / "best_model.keras"
)

# Canonical problem order for the "all" selection. "denoise" is the trivial
# single-forward-pass task D(y): it has NO MeasurementOperator/solver, so it is
# handled as a sibling branch in run_problem (H1 / D-001), never via build_operator.
# Adding it here auto-wires the --problem choices, the "all" loop, and the Streamlit
# selectbox (F2 A3) with no further wiring.
_ALL_PROBLEMS: Tuple[str, ...] = (
    "denoise",
    "prior",
    "inpaint",
    "random_pixels",
    "super_resolution",
    "deblur",
    "compressive_sensing",
)

# Human-readable panel titles keyed by problem id.
_PROBLEM_TITLES: Dict[str, str] = {
    "denoise": "Denoising",
    "prior": "Prior Sampling",
    "inpaint": "Inpainting",
    "random_pixels": "Random Pixels",
    "super_resolution": "Super-Resolution",
    "deblur": "Spectral Deblur",
    "compressive_sensing": "Compressive Sensing",
}


# ---------------------------------------------------------------------------
# Synthetic in-domain target
# ---------------------------------------------------------------------------


def create_synthetic_test_image(shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Create a smooth synthetic RGB test image in the ``[0, 1]`` domain.

    The pattern (a gradient background plus a circle and a rectangle) gives structure
    the operators can visibly degrade and the prior can plausibly restore.

    Args:
        shape: ``(batch, H, W, C)``. Only ``batch == 1`` is used by the demo; ``C``
            is broadcast across channels (the checkpoint is RGB, ``C == 3``).

    Returns:
        A float32 ``numpy.ndarray`` of ``shape`` with values in ``[0, 1]``.
    """
    _, h, w, c = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= float(h)
    xx /= float(w)

    dist = np.sqrt((yy - 0.25) ** 2 + (xx - 0.25) ** 2)
    circle = (dist < 0.15).astype(np.float32)
    rect = ((yy > 0.6) & (yy < 0.9) & (xx > 0.6) & (xx < 0.9)).astype(np.float32)
    gradient = xx * 0.3

    # The [0, 1] assembly IS the model domain now — no shift, just a clip.
    img01 = np.clip(gradient + circle * 0.5 + rect * 0.7, 0.0, 1.0)
    img = np.repeat(img01[:, :, None], c, axis=-1)
    return img[None, ...].astype(np.float32)


def load_real_image(path: str, size: int) -> np.ndarray:
    """Load a real image, resize to ``size`` and ingest to ``[0, 1]``.

    Args:
        path: Path to an image file (any format Keras/Pillow can open).
        size: Target square edge length (must be divisible by 8 for the U-Net).

    Returns:
        A float32 ``[1, size, size, 3]`` array in ``[0, 1]``.
    """
    import keras

    img = keras.utils.load_img(path, color_mode="rgb", target_size=(size, size))
    arr = keras.utils.img_to_array(img)  # [size, size, 3] in [0, 255]
    normalized = DenoiserPrior.ingest(arr)  # -> [0, 1]
    return normalized[None, ...].astype(np.float32)


# ---------------------------------------------------------------------------
# Operator factory
# ---------------------------------------------------------------------------


def build_operator(
    problem: str,
    image_shape: Tuple[int, int, int],
    args: argparse.Namespace,
) -> MeasurementOperator:
    """Construct the :class:`MeasurementOperator` for a given problem id.

    Args:
        problem: One of the six problem ids (see ``_ALL_PROBLEMS``).
        image_shape: Signal shape ``(H, W, C)``.
        args: Parsed CLI namespace supplying per-problem knobs.

    Returns:
        The matching operator instance.

    Raises:
        ValueError: If ``problem`` is not a recognized id.
    """
    h, w, _ = image_shape
    if problem == "prior":
        return NullOperator()
    if problem == "inpaint":
        block = args.block if args.block is not None else max(1, min(h, w) // 4)
        return InpaintingOperator(image_shape, block_size=block)
    if problem == "random_pixels":
        return RandomPixelsOperator(image_shape, keep_ratio=args.keep_ratio, seed=args.seed)
    if problem == "super_resolution":
        return SuperResolutionOperator(image_shape, factor=args.sr_factor)
    if problem == "deblur":
        return SpectralDeblurOperator(image_shape, keep_fraction=args.keep_fraction)
    if problem == "compressive_sensing":
        return CompressiveSensingOperator(
            image_shape, measurement_ratio=args.measurement_ratio, seed=args.seed
        )
    raise ValueError(f"unknown problem {problem!r}")


# ---------------------------------------------------------------------------
# Single-problem run
# ---------------------------------------------------------------------------


def run_problem(
    problem: str,
    prior: DenoiserPrior,
    target: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build the operator, measure the target, and solve one inverse problem.

    Args:
        problem: The problem id.
        prior: The loaded denoiser prior.
        target: The clean signal ``[1, H, W, C]`` in ``[0, 1]``.
        args: Parsed CLI namespace (solver + per-problem knobs).

    Returns:
        A dict with ``title``, ``target`` (display, ``[H, W, C]`` in ``[0, 1]``),
        ``degraded`` (display or ``None`` for prior sampling), ``recon`` (display),
        and ``info`` (solver convergence dict).
    """
    import keras

    logger.info("--- problem '%s' (%s) ---", problem, _PROBLEM_TITLES[problem])

    if problem == "denoise":
        # H1 / D-001: denoise is a single forward pass D(y). It has NO
        # MeasurementOperator and NO UniversalInverseSolver — it returns BEFORE
        # build_operator and must NOT be routed through a fake IdentityOperator
        # (the deliberately-rejected anti-pattern; see decisions.md D-001 / F2).
        # H2: all pixel math stays in [0, 1]; synthetic Gaussian noise is added
        # post-ingest, in-domain, and lightly clipped to the domain (matching the
        # trainer) before the single denoise call.
        rng = np.random.default_rng(args.seed)
        if args.noise_sigma > 0.0:
            noise = rng.normal(0.0, args.noise_sigma, target.shape).astype(np.float32)
            noisy = np.clip(target + noise, 0.0, 1.0)
        else:
            noisy = target  # denoise-as-is
        denoised = np.asarray(keras.ops.convert_to_numpy(prior.denoise(noisy)))
        return {
            "title": _PROBLEM_TITLES["denoise"],
            "target": DenoiserPrior.denorm(target[0]),
            "degraded": DenoiserPrior.denorm(noisy[0]),
            "recon": DenoiserPrior.denorm(denoised[0]),
            "info": {},
        }

    image_shape = tuple(int(s) for s in target.shape[1:])
    operator = build_operator(problem, image_shape, args)
    # patience==0 means "disabled": resolve to iterations+1 so the no-improvement
    # counter (which can reach at most max_iterations) never trips an early stop
    # on a full-budget quality run. See parse_args --patience help / D-003.
    resolved_patience = args.patience if args.patience > 0 else args.iterations + 1
    solver = UniversalInverseSolver(
        prior,
        sigma_0=args.sigma0,
        beta=args.beta,
        max_iterations=args.iterations,
        sigma_l=args.sigma_l,
        h0=args.h0,
        h_max=args.h_max,
        patience=resolved_patience,
        final_projection=args.final_projection,
    )

    if problem == "prior":
        # Algorithm-1 unconstrained prior sampling: no measurements, size via shape.
        measurements: Optional[Any] = None
        shape: Optional[Tuple[int, ...]] = target.shape
        degraded_disp: Optional[np.ndarray] = None
    else:
        measurements = operator.measure(target)
        shape = None
        # Uniform signal-shaped observable view (real for spectral/CS): M M^T target.
        degraded = np.asarray(keras.ops.convert_to_numpy(operator.project(target)))
        degraded_disp = DenoiserPrior.denorm(degraded[0])

    # DECISION plan_2026-07-06_c9c7a81a/D-005: --num-samples > 1 averages N independent
    # solve() runs (seeds args.seed + i) in the [0,1] denormed domain here, as an OUTER
    # loop — the averaging is deliberately OUTSIDE solve() to preserve the single unified
    # loop (INV-6). num_samples=1 (default) runs exactly one solve at args.seed and the
    # `/ 1.0` mean is byte-identical to a single-solve reconstruction. See decisions.md D-005.
    num_samples = max(1, int(args.num_samples))
    recon01_sum: Optional[np.ndarray] = None
    info: Dict[str, Any] = {}
    for i in range(num_samples):
        recon, info = solver.solve(
            operator, measurements=measurements, shape=shape, seed=args.seed + i
        )
        recon_np = np.asarray(keras.ops.convert_to_numpy(recon))
        recon01_i = DenoiserPrior.denorm(recon_np[0])
        recon01_sum = recon01_i if recon01_sum is None else recon01_sum + recon01_i
    recon01 = recon01_sum / float(num_samples)

    return {
        "title": _PROBLEM_TITLES[problem],
        "target": DenoiserPrior.denorm(target[0]),
        "degraded": degraded_disp,
        "recon": recon01,
        "info": info,
    }


def denoise_frame(prior: DenoiserPrior, rgb_uint8: np.ndarray, size: int = 256) -> np.ndarray:
    """Denoise one RGB frame with a single forward pass ``D(y)`` — GUI-free primitive.

    This is the shared per-frame primitive reused by BOTH a headless unit test and the
    streamlit-webrtc ``recv()`` loop (H3 / D-002), so it stays free of any streamlit
    import. The forward pass runs at a fixed ``size`` x ``size`` (``size`` must be
    divisible by 8 for the depth-3 U-Net); the input may be an arbitrary ``[H, W, 3]``
    uint8 frame (non-÷8, non-square) and the denoised output is resized back to the
    original ``(H, W)`` so a live video track's resolution stays stable (H4).

    Args:
        prior: The loaded denoiser prior (exposes ``.denoise``).
        rgb_uint8: A ``[H, W, 3]`` uint8 RGB frame in ``[0, 255]``.
        size: Square edge for the forward pass (divisible by 8; default 256).

    Returns:
        A ``[H, W, 3]`` uint8 RGB frame in ``[0, 255]`` (finite), same ``H`` x ``W`` as
        the input.
    """
    from PIL import Image
    import keras

    frame = np.asarray(rgb_uint8).astype(np.uint8)
    h, w = int(frame.shape[0]), int(frame.shape[1])

    # Fixed ÷8 square pass, independent of input aspect ratio (H4). PIL resize takes
    # (width, height); a square (size, size) is orientation-agnostic here.
    resized = np.asarray(Image.fromarray(frame).resize((size, size)))
    x = DenoiserPrior.ingest(resized)[None, ...]  # -> [1, size, size, 3] in [0, 1]
    y = np.asarray(keras.ops.convert_to_numpy(prior.denoise(x)))
    out01 = DenoiserPrior.denorm(y[0])  # already [0,1]; denorm clips out-of-domain
    out_uint8 = (out01 * 255.0).astype(np.uint8)
    # Resize back to the original (H, W) -> PIL wants (width, height) == (w, h).
    return np.asarray(Image.fromarray(out_uint8).resize((w, h))).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _plot_convergence(ax: "plt.Axes", info: Dict[str, Any], title: str) -> None:
    """Plot the effective-noise ``sigma_t`` curve for one problem."""
    sigmas = [s for s in info.get("sigma_values", []) if s and np.isfinite(s) and s > 0]
    if sigmas:
        ax.plot(sigmas, "b-", linewidth=1.5)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"{title} sigma_t", fontsize=9)
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("effective sigma", fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=7)


def _show_image(ax: "plt.Axes", img: Optional[np.ndarray], title: str) -> None:
    """Render a ``[H, W, C]`` image in ``[0, 1]`` (or a placeholder if ``None``)."""
    if img is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    else:
        disp = img if img.shape[-1] == 3 else img[..., 0]
        ax.imshow(np.clip(disp, 0.0, 1.0), cmap=None if img.shape[-1] == 3 else "gray",
                  vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def visualize_results(results: List[Dict[str, Any]], save_path: Path) -> None:
    """Save a grid PNG (one row per problem) headless.

    Columns: target, measured/degraded view, reconstruction, convergence curve.

    Args:
        results: Per-problem dicts from :func:`run_problem`.
        save_path: Destination PNG path.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n), squeeze=False)
    fig.suptitle("Bias-Free Denoiser Prior: Inverse Problems", fontsize=15)

    for row, res in enumerate(results):
        _show_image(axes[row][0], res["target"], f"{res['title']}\nTarget")
        _show_image(axes[row][1], res["degraded"], "Measured / Degraded")
        _show_image(axes[row][2], res["recon"], "Reconstruction")
        _plot_convergence(axes[row][3], res["info"], res["title"])

    fig.set_layout_engine("constrained")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved grid to %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _hmax(s: str) -> Optional[float]:
    """Parse the ``--h-max`` value: ``none``/empty -> ``None`` (uncapped), else ``float``.

    Lets the CLI express the uncapped paper step schedule (``h_max=None``) as a plain
    string token, since argparse ``type=float`` cannot yield ``None``.
    """
    return None if s.strip().lower() in ("none", "") else float(s)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the demo."""
    p = argparse.ArgumentParser(
        description="Bias-free-denoiser inverse-problem demo (headless, MPLBACKEND=Agg).",
    )
    p.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT),
                   help="Path to best_model.keras or its results directory.")
    p.add_argument("--image", type=str, default=None,
                   help="Optional real image path; if omitted a synthetic in-domain target is used.")
    p.add_argument("--problem", type=str, default="all",
                   choices=(*_ALL_PROBLEMS, "all"),
                   help="Which problem(s) to run (default: all six).")
    p.add_argument("--size", type=int, default=256,
                   help="Square image edge (must be divisible by 8; default 256).")
    # DECISION plan_2026-07-06_c9c7a81a/D-003: paper-DEVIATING defaults changed
    # (iterations 200->500, and h_max 0.1->None below) per the Step-2 harness A/B
    # (+13.4 dB mild-regime mean PSNR, all five inverse tasks improved, every
    # reconstruction finite+bounded). Paper-EXACT literals beta/sigma_l/h0 keep
    # their 0.01 values. See decisions.md D-003.
    p.add_argument("--iterations", type=int, default=500,
                   help="Solver max iterations (default 500; paper-quality regime per "
                        "Step-2 measurement). Lower budgets (e.g. 200) converge coarser.")
    p.add_argument("--sigma0", type=float, default=0.4, help="Initial noise std (default 0.4).")
    p.add_argument("--beta", type=float, default=0.01, help="Noise-injection parameter (default 0.01).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    # --- Solver-regime knobs: the iteration-starving levers UniversalInverseSolver
    # already accepts (solver.py:92-104) but main.py did not forward.
    # DECISION plan_2026-07-06_c9c7a81a/D-002: SUPERSEDES plan_2026-07-06_b89e65ab/D-003
    # ("h_max/h0 are library-only; no --h-max/--h0 flag exists or is planned"). The
    # approved quality goal requires user-tunable regime knobs, so h_max/sigma_l/h0/
    # patience are now surfaced on the CLI (and the GUI) and forwarded to the solver.
    # Do NOT re-hide them. See decisions.md D-002.
    p.add_argument("--h-max", type=_hmax, default=None,
                   help="Step-size cap; 'none' = uncapped paper schedule (default; "
                        "empirically best per measurement).")
    p.add_argument("--sigma-l", type=float, default=0.01,
                   help="Effective-noise stop threshold (paper-exact 0.01; exposed for "
                        "completeness).")
    p.add_argument("--h0", type=float, default=0.01,
                   help="Step-size schedule parameter (paper-exact 0.01).")
    p.add_argument("--patience", type=int, default=0,
                   help="No-improvement iterations before early stop; 0 = disabled "
                        "(resolved to iterations+1 so a full-budget run never early-stops; "
                        "default 0).")
    # --- Optional additive quality levers (OFF-by-default, opt-in; D-005).
    # DECISION plan_2026-07-06_c9c7a81a/D-005: both levers are ADDITIVE and OFF by
    # default. --final-projection re-imposes exact measurements ONCE at solve() return
    # (data consistency; NullOperator no-op). --num-samples averages N independent
    # solve() trajectories (different seeds) in the [0,1] domain as an OUTER loop in
    # run_problem — NOT inside solve() (that would break the single unified loop, INV-6).
    # num_samples=1 (default) is byte-identical to a single solve. Do NOT flip either
    # default without a broad measured gain. See decisions.md D-005.
    p.add_argument("--final-projection", action="store_true",
                   help="Impose exact hard data consistency once at solve() return "
                        "(additive, OFF by default; NullOperator no-op).")
    p.add_argument("--num-samples", type=int, default=1,
                   help="Ours-avg: average N independent solve trajectories (different "
                        "seeds) in [0,1]; default 1 = single solve (byte-identical).")
    p.add_argument("--output-dir", type=str, default=str(_DEFAULT_OUTPUT_DIR),
                   help="Directory for the grid PNG (default repo-root results/).")
    # Per-problem knobs.
    p.add_argument("--block", type=int, default=None,
                   help="Inpainting missing-block edge (default size//4).")
    p.add_argument("--keep-ratio", type=float, default=0.3,
                   help="Random-pixels keep fraction (default 0.3).")
    p.add_argument("--sr-factor", type=int, default=4,
                   help="Super-resolution downsample factor (default 4).")
    p.add_argument("--keep-fraction", type=float, default=0.15,
                   help="Spectral-deblur low-pass keep fraction (default 0.15).")
    p.add_argument("--measurement-ratio", type=float, default=0.2,
                   help="Compressive-sensing measurement ratio (default 0.2).")
    p.add_argument("--noise-sigma", type=float, default=0.1,
                   help="Denoise task: synthetic Gaussian noise std added in-domain "
                        "before denoising (0 = denoise as-is; default 0.1).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the demo end to end and save a grid PNG to ``results/``.

    Each selected problem runs inside its own ``try/except`` so a single failure
    (NaN, divergence, or a raised exception) does NOT abort the remaining problems
    or suppress the partial grid — the demo stays resilient. But failures are
    SURFACED, not swallowed: every failed problem is logged at ``ERROR`` level and
    the process exits with a NON-ZERO status so ``--problem all`` fails loudly if
    any problem misbehaves on the real checkpoint (closes the hollow-criterion-8
    gap, D-009 / WARNING-1).

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` if every selected problem succeeded; ``1`` if one or more failed
        (including the degenerate "no problem produced a result" case).
    """
    args = parse_args(argv)

    if args.size % 8 != 0:
        raise ValueError(f"--size must be divisible by 8 (depth-3 U-Net); got {args.size}")

    logger.info("loading denoiser prior from %s", args.checkpoint)
    prior = DenoiserPrior.from_pretrained(args.checkpoint)

    if args.image:
        logger.info("using real image %s (resized to %dx%d)", args.image, args.size, args.size)
        target = load_real_image(args.image, args.size)
    else:
        logger.info("using synthetic in-domain target %dx%d", args.size, args.size)
        target = create_synthetic_test_image((1, args.size, args.size, 3))

    problems = list(_ALL_PROBLEMS) if args.problem == "all" else [args.problem]
    results: List[Dict[str, Any]] = []
    failed: List[str] = []
    for problem in problems:
        try:
            results.append(run_problem(problem, prior, target, args))
        except Exception as exc:  # noqa: BLE001 — one bad problem must not sink the demo.
            logger.error("problem '%s' failed: %s", problem, exc, exc_info=True)
            failed.append(problem)

    # Still produce the partial grid + summary from whatever succeeded.
    if results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = output_dir / f"bias_free_denoiser_demo_{stamp}.png"
        visualize_results(results, save_path)

        logger.info("=" * 60)
        logger.info("SUMMARY (iterations=%d, sigma0=%.3f, beta=%.3f):",
                    args.iterations, args.sigma0, args.beta)
        for res in results:
            info = res["info"]
            n_iter = len(info.get("iterations", []))
            final_sigma = info["sigma_values"][-1] if info.get("sigma_values") else float("nan")
            logger.info("  %-20s %4d iters, final sigma=%.5f", res["title"], n_iter, final_sigma)
        logger.info("=" * 60)
    else:
        logger.error("no problems produced a result; nothing to plot")

    # Surface failures with a non-zero exit so a broken problem fails loudly even
    # though the surviving problems still ran and were plotted (D-009).
    if failed:
        logger.error("%d/%d problem(s) FAILED: %s", len(failed), len(problems), ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
