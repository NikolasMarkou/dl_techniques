"""Fair, mode-agnostic evaluation of VAE latent samplers on MNIST.

This script answers the question "does the hypersphere sampling contribution in
``models/vae/`` actually work?" by comparing three trained VAE arms whose only
difference is the latent ``sampling_type`` in
``{gaussian, hypersphere_controlled, hypersphere_faithful}``.

Fairness rationale
------------------
The prior 20-epoch A/B "verdict" ranked arms by ``total_loss``, which is NOT
comparable across modes: ``hypersphere_faithful`` uses a scalar radius-variance
KL while ``gaussian``/``hypersphere_controlled`` use a D-dimensional Gaussian KL.
``kl_loss`` and ``total_loss`` are therefore apples-to-oranges and are
deliberately NOT compared here. Only two signals are comparable across modes:

1. **Reconstruction error** -- the per-pixel binary-crossentropy and MSE on the
   held-out MNIST test set use the IDENTICAL formula in all three modes (the BCE
   matches ``VAE._compute_reconstruction_loss``: mean-over-pixels BCE per sample,
   then mean over the batch). This is the primary fair scalar.
2. **MMD-to-real** -- a mode-agnostic generative metric. We decode ``n_gen``
   prior samples via ``model.sample()`` (correct per-mode prior after the
   step-2 fix), project both real-test and generated images into a PCA-50 space
   fit ONCE on the real test set (shared across arms for comparability), and
   compute an unbiased RBF-kernel MMD^2 at the median-heuristic bandwidth plus
   0.5x and 2.0x that bandwidth. LOWER MMD = generated distribution closer to
   real = better.

Mode-aware latent diagnostics (D-002)
-------------------------------------
For the hypersphere modes the decoder consumes the ON-SPHERE latent
``z = radius * normalize(z_mean + eps)``; the dict ``z_mean`` is the
UNNORMALIZED mean and can have ``||z_mean|| >> 1``, so plotting raw ``z_mean``
on a fixed axis is misleading (a healthy direction-spread renders as a
"collapsed point"). The honest, dimension-agnostic collapse signal is therefore
the concentration of the per-sample UNIT direction ``u = z_mean/||z_mean||``:

- ``dir_concentration`` = ``||mean(u)||`` over all test samples (0.0 = directions
  uniformly spread = healthy latent usage; 1.0 = all directions identical =
  collapsed). Computed for ALL modes; this is the primary collapse metric and is
  dimension-agnostic (works for latent_dim=2 and latent_dim=16). LOWER is better.
- ``angular_coverage`` = fraction of 36 angle-bins (latent_dim==2 only; ``nan``
  otherwise) that hold more than ``uniform/10`` of the samples.
- ``mean_z_mean_norm`` = mean ``||z_mean||``. For hypersphere modes this norm is
  IRRELEVANT (only the direction matters); it is reported for context only.

For latent_dim==2 the scatter PNG plots the on-sphere direction on the unit
circle (hypersphere modes) or the raw 2D ``z_mean`` blob (gaussian, autoscaled),
plus an angle-per-class step-histogram for the hypersphere modes.

Verdict decision-rule (baked into ``print_verdict``)
---------------------------------------------------
A "hypersphere WORKS" conclusion requires, for a hypersphere arm vs gaussian:
  - recon_bce within ~5-10% of gaussian (competitive on reconstruction), AND
  - mmd2_median lower-or-comparable to gaussian (not worse generatively).
"Does NOT work" = a clear, reproducible deficit on recon AND/OR a clearly higher
MMD with structured (non-noise) prior samples. The script prints a tentative
PASS/FAIL per arm; final human judgment is the orchestrator's.

CLI
---
    python -m train.vae.evaluate_samplers --runs DIR1 DIR2 DIR3 \
        [--out OUTDIR] [--n-gen 5000] [--seed 42]

If ``--runs`` is omitted, the three most-recent ``results/vae_vae_mnist_*``
dirs (newest ``config.json`` per distinct sampler) are auto-discovered.
"""

import os
import csv
import glob
import json
import keras
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from sklearn.decomposition import PCA

from dl_techniques.utils.logger import logger
from dl_techniques.models.vae.model import VAE
from dl_techniques.layers.sampling import Sampling, HypersphereSampling

# Reuse the trainer's plot helpers + custom-objects so the eval stays in lockstep
# with how the model was trained/serialized (DRY: do not redefine these).
from train.vae.train_vae import (
    CUSTOM_OBJECTS,
    plot_reconstruction_comparison,
)

ACTIVE_UNIT_VAR_THRESHOLD = 0.01  # z_mean per-dim variance floor for "active"
MMD_SUBSAMPLE = 2000  # real/gen pairs per MMD estimate (tractable + stable)
MMD_SEEDS = 3  # average MMD over this many subsample seeds

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_mnist_test() -> Tuple[np.ndarray, np.ndarray]:
    """Load the MNIST test split with the EXACT trainer preprocessing.

    Replicates ``train_vae.py``: ``float32`` / 255.0, channel-last shape
    ``(N, 28, 28, 1)``.

    Returns:
        Tuple of (x_test, y_test). ``x_test`` is ``(10000, 28, 28, 1)`` float32
        in [0, 1]; ``y_test`` is ``(10000,)`` int labels.
    """
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
    return x_test, y_test.flatten()


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #


def discover_runs() -> List[str]:
    """Auto-discover the newest ``results/vae_vae_mnist_*`` dir per sampler.

    Returns:
        List of run-dir paths, one per distinct sampler, newest ``config.json``
        mtime first. May be fewer than 3 if a sampler has no run yet.
    """
    candidates = glob.glob("results/vae_vae_mnist_*")
    by_sampler: Dict[str, Tuple[float, str]] = {}
    for d in candidates:
        cfg_path = os.path.join(d, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path) as f:
                sampler = json.load(f).get("sampler")
        except (json.JSONDecodeError, OSError):
            continue
        if sampler is None:
            continue
        mtime = os.path.getmtime(cfg_path)
        if sampler not in by_sampler or mtime > by_sampler[sampler][0]:
            by_sampler[sampler] = (mtime, d)
    chosen = [d for _, d in sorted(by_sampler.values(), reverse=True)]
    return chosen


def load_arm(run_dir: str) -> Tuple[VAE, str]:
    """Load an arm's best checkpoint and its sampler name.

    Args:
        run_dir: Path to a ``results/vae_vae_mnist_*`` run directory.

    Returns:
        Tuple of (model, sampler_name).

    Raises:
        FileNotFoundError: If ``best_model.keras`` or ``config.json`` is missing.
    """
    model_path = os.path.join(run_dir, "best_model.keras")
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No best_model.keras in {run_dir}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"No config.json in {run_dir}")
    with open(cfg_path) as f:
        sampler = json.load(f).get("sampler", "unknown")
    model = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    return model, sampler


# --------------------------------------------------------------------------- #
# Comparable reconstruction error (the fair scalar)
# --------------------------------------------------------------------------- #


def reconstruction_errors(
    model: VAE, x_test: np.ndarray, batch_size: int = 256
) -> Tuple[float, float]:
    """Mean BCE and mean MSE over the full test set (identical formula per mode).

    The BCE matches ``VAE._compute_reconstruction_loss`` exactly: per-sample mean
    over pixels via ``keras.losses.binary_crossentropy`` on clipped predictions,
    then mean over the batch.

    Args:
        model: A loaded VAE arm.
        x_test: Test images ``(N, 28, 28, 1)`` in [0, 1].
        batch_size: Forward-pass batch size.

    Returns:
        Tuple of (mean_bce, mean_mse).
    """
    outputs = model.predict(x_test, batch_size=batch_size, verbose=0)
    recon = outputs["reconstruction"]

    y_true = keras.ops.reshape(x_test, (x_test.shape[0], -1))
    y_pred = keras.ops.reshape(recon, (recon.shape[0], -1))
    y_pred_clipped = keras.ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    bce = keras.ops.mean(keras.losses.binary_crossentropy(y_true, y_pred_clipped))
    mse = keras.ops.mean(keras.ops.square(y_true - y_pred))
    return float(keras.ops.convert_to_numpy(bce)), float(
        keras.ops.convert_to_numpy(mse)
    )


# --------------------------------------------------------------------------- #
# MMD-to-real generative metric (mode-agnostic)
# --------------------------------------------------------------------------- #


def _rbf_mmd2_unbiased(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Unbiased RBF-kernel MMD^2 estimate between samples ``x`` and ``y``.

    Args:
        x: ``(m, d)`` samples from distribution P.
        y: ``(n, d)`` samples from distribution Q.
        gamma: RBF bandwidth parameter (kernel = exp(-gamma * ||a - b||^2)).

    Returns:
        Unbiased MMD^2 estimate (may be slightly negative; caller clamps).
    """

    def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1)[:, None]
        bb = np.sum(b * b, axis=1)[None, :]
        return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)

    m, n = x.shape[0], y.shape[0]
    kxx = np.exp(-gamma * _sq_dists(x, x))
    kyy = np.exp(-gamma * _sq_dists(y, y))
    kxy = np.exp(-gamma * _sq_dists(x, y))
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    term_xx = kxx.sum() / (m * (m - 1))
    term_yy = kyy.sum() / (n * (n - 1))
    term_xy = kxy.mean()
    return float(term_xx + term_yy - 2.0 * term_xy)


def _median_bandwidth(pooled: np.ndarray) -> float:
    """Median-heuristic RBF bandwidth: median pairwise L2 distance on a pool.

    Args:
        pooled: ``(k, d)`` pooled subsample of real+gen points.

    Returns:
        Median pairwise Euclidean distance (>= small floor to avoid div-by-zero).
    """
    sq = np.sum(pooled * pooled, axis=1)[:, None] + np.sum(
        pooled * pooled, axis=1
    )[None, :] - 2.0 * pooled @ pooled.T
    sq = np.maximum(sq, 0.0)
    iu = np.triu_indices(pooled.shape[0], k=1)
    dists = np.sqrt(sq[iu])
    med = float(np.median(dists))
    return max(med, 1e-8)


def mmd_to_real(
    real_pca: np.ndarray, gen_pca: np.ndarray, seed: int
) -> Dict[str, float]:
    """Average RBF-MMD^2 at median, 0.5x, and 2.0x median-heuristic bandwidths.

    Subsamples ``MMD_SUBSAMPLE`` real/gen points per estimate and averages over
    ``MMD_SEEDS`` seeds. Tiny negatives are clamped to 0.

    Args:
        real_pca: Real test images projected into PCA space ``(N, k)``.
        gen_pca: Generated images projected into the SAME PCA space ``(M, k)``.
        seed: Base RNG seed for subsampling.

    Returns:
        Dict with keys ``mmd2_median``, ``mmd2_half``, ``mmd2_double`` (means).
    """
    n_real = real_pca.shape[0]
    n_gen = gen_pca.shape[0]
    size = min(MMD_SUBSAMPLE, n_real, n_gen)

    acc = {"mmd2_median": [], "mmd2_half": [], "mmd2_double": []}
    for s in range(MMD_SEEDS):
        rng = np.random.default_rng(seed + s)
        ri = rng.choice(n_real, size=size, replace=False)
        gi = rng.choice(n_gen, size=size, replace=False)
        rx, gx = real_pca[ri], gen_pca[gi]

        pool_n = min(size, 1000)
        pool = np.concatenate([rx[:pool_n], gx[:pool_n]], axis=0)
        sigma = _median_bandwidth(pool)

        for key, scale in (
            ("mmd2_median", 1.0),
            ("mmd2_half", 0.5),
            ("mmd2_double", 2.0),
        ):
            gamma = 1.0 / (2.0 * (scale * sigma) ** 2)
            acc[key].append(max(_rbf_mmd2_unbiased(rx, gx, gamma), 0.0))

    return {k: float(np.mean(v)) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# Latent diagnostics
# --------------------------------------------------------------------------- #


N_ANGLE_BINS = 36  # angular-coverage bins (latent_dim==2)


def latent_diagnostics(
    model: VAE, x_test: np.ndarray, save_path: str, batch_size: int = 256
) -> Dict[str, float]:
    """Encode the test set and report mode-aware latent collapse diagnostics.

    The honest, dimension-agnostic collapse signal is the concentration of the
    per-sample UNIT direction ``u = z_mean/||z_mean||`` (D-002): a hypersphere
    decoder consumes the on-sphere latent, so ``||z_mean||`` magnitude is
    irrelevant — only the DIRECTION matters.

    Args:
        model: A loaded VAE arm.
        x_test: Test images ``(N, 28, 28, 1)``.
        save_path: Where to write the ``||z_mean||`` histogram PNG.
        batch_size: Forward-pass batch size.

    Returns:
        Dict with keys ``active_units``, ``mean_z_mean_norm``,
        ``dir_concentration``, ``angular_coverage`` (``nan`` unless
        latent_dim==2).
    """
    z_mean = model.predict(x_test, batch_size=batch_size, verbose=0)["z_mean"]
    z_mean = np.asarray(z_mean)
    per_dim_var = z_mean.var(axis=0)
    active_units = int(np.sum(per_dim_var > ACTIVE_UNIT_VAR_THRESHOLD))
    norms = np.linalg.norm(z_mean, axis=1)
    mean_norm = float(norms.mean())

    # Per-sample UNIT direction (the REAL latent for hypersphere modes).
    unit = z_mean / np.maximum(norms[:, None], 1e-12)
    # dir_concentration = ||mean(u)||: 0 = directions uniformly spread (healthy),
    # 1 = all directions identical (collapsed). Dimension-agnostic.
    dir_concentration = float(np.linalg.norm(unit.mean(axis=0)))

    if model.latent_dim == 2:
        ang = np.arctan2(unit[:, 1], unit[:, 0])
        occ = np.histogram(ang, bins=N_ANGLE_BINS, range=(-np.pi, np.pi))[0]
        uniform = len(ang) / N_ANGLE_BINS
        angular_coverage = float((occ > uniform / 10.0).mean())
    else:
        angular_coverage = float("nan")

    plt.figure(figsize=(7, 4))
    plt.hist(norms, bins=60, color="steelblue", alpha=0.85)
    plt.axvline(mean_norm, color="crimson", linestyle="--", label=f"mean={mean_norm:.3f}")
    plt.xlabel("||z_mean|| (L2 norm per sample) [irrelevant for hypersphere modes]")
    plt.ylabel("count")
    plt.title(
        f"z_mean norm distribution ({model.sampling_type}) | "
        f"dir_concentration={dir_concentration:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return {
        "active_units": active_units,
        "mean_z_mean_norm": mean_norm,
        "dir_concentration": dir_concentration,
        "angular_coverage": angular_coverage,
    }


def plot_latent_diagnostics_2d(
    model: VAE,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scatter_path: str,
    angle_path: str,
    batch_size: int = 256,
) -> List[str]:
    """Mode-aware 2D latent scatter (+ angle-per-class hist for hypersphere modes).

    For hypersphere modes the scatter shows the ON-SPHERE direction
    ``u = z_mean/||z_mean||`` on the unit circle (the real latent), and an
    angle-per-class step-histogram is also written (cf. ``/tmp/inspect_latent.py``).
    For gaussian the raw 2D ``z_mean`` blob is shown autoscaled (no [-4,4] clamp).

    Args:
        model: A loaded VAE arm (latent_dim must be 2).
        x_test: Test images.
        y_test: Test labels.
        scatter_path: PNG path for the scatter.
        angle_path: PNG path for the angle-per-class hist (hypersphere only).
        batch_size: Forward-pass batch size.

    Returns:
        List of PNG paths actually written.
    """
    z_mean = np.asarray(
        model.predict(x_test, batch_size=batch_size, verbose=0)["z_mean"]
    )
    labels = y_test.flatten() if y_test.ndim > 1 else y_test
    is_sphere = str(model.sampling_type).startswith("hypersphere")
    written: List[str] = []

    if is_sphere:
        norms = np.linalg.norm(z_mean, axis=1, keepdims=True)
        coords = z_mean / np.maximum(norms, 1e-12)
        title = (
            f"On-sphere direction (real latent) -- {model.sampling_type}\n"
            "u = z_mean / ||z_mean|| (raw z_mean norm is irrelevant)"
        )
        lim = 1.3
    else:
        coords = z_mean
        title = f"Raw z_mean (2D latent, autoscaled) -- {model.sampling_type}"
        lim = None

    plt.figure(figsize=(8, 7))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=4, alpha=0.5)
    plt.colorbar(sc, ticks=np.arange(len(np.unique(labels)))).set_label("digit class")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.title(title, fontsize=10)
    if lim is not None:
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.gca().set_aspect("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    written.append(scatter_path)

    if is_sphere:
        ang = np.arctan2(coords[:, 1], coords[:, 0])
        plt.figure(figsize=(8, 5))
        for c in range(int(labels.max()) + 1):
            plt.hist(
                ang[labels == c], bins=48, range=(-np.pi, np.pi),
                histtype="step", lw=1.2, label=str(c),
            )
        plt.xlabel("on-sphere angle (rad)")
        plt.ylabel("count")
        plt.title(f"On-sphere angle per class -- {model.sampling_type}", fontsize=10)
        plt.legend(title="class", ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(angle_path, dpi=150)
        plt.close()
        written.append(angle_path)

    return written


# --------------------------------------------------------------------------- #
# Visual grids
# --------------------------------------------------------------------------- #


def plot_prior_grid(model: VAE, save_path: str) -> None:
    """Decode ``model.sample(64)`` into an 8x8 grid (correct per-mode prior)."""
    imgs = np.asarray(keras.ops.convert_to_numpy(model.sample(64)))
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.clip(imgs[i].squeeze(), 0, 1), cmap="gray")
        ax.axis("off")
    fig.suptitle(f"Prior samples ({model.sampling_type})", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    plt.close()


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two vectors at fraction ``t``."""
    a_n = a / max(np.linalg.norm(a), 1e-12)
    b_n = b / max(np.linalg.norm(b), 1e-12)
    dot = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-6:
        return (1.0 - t) * a + t * b
    so = np.sin(omega)
    return (np.sin((1.0 - t) * omega) / so) * a_n + (np.sin(t * omega) / so) * b_n


def plot_interpolation_grid(model: VAE, save_path: str, n_steps: int = 10) -> None:
    """Mode-aware latent interpolation between two prior endpoints.

    Gaussian -> LINEAR interpolation in z. Hypersphere modes -> SPHERICAL (slerp)
    between the two unit directions, scaled by the layer radius (great-circle).

    Args:
        model: A loaded VAE arm.
        save_path: PNG path.
        n_steps: Number of decode steps along the path.
    """
    z = np.asarray(keras.ops.convert_to_numpy(model._sample_prior(2)))
    z0, z1 = z[0], z[1]
    is_sphere = model.sampling_type != "gaussian"
    radius = (
        float(model.get_layer("vae_sampling").radius) if is_sphere else None
    )

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.2, 1.6))
    for j, t in enumerate(np.linspace(0, 1, n_steps)):
        if is_sphere:
            zt = _slerp(z0, z1, float(t))
            zt = radius * zt / max(np.linalg.norm(zt), 1e-12)
        else:
            zt = (1.0 - t) * z0 + t * z1
        recon = np.asarray(
            keras.ops.convert_to_numpy(model.decode(zt[np.newaxis]))
        )
        axes[j].imshow(np.clip(recon[0].squeeze(), 0, 1), cmap="gray")
        axes[j].axis("off")
    interp_type = "slerp (great-circle)" if is_sphere else "linear"
    fig.suptitle(
        f"Interpolation [{interp_type}] ({model.sampling_type})", fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(save_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Per-arm evaluation
# --------------------------------------------------------------------------- #


def evaluate_arm(
    run_dir: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    real_pca: np.ndarray,
    pca: PCA,
    n_gen: int,
    seed: int,
    out_dir: str,
) -> Dict[str, float]:
    """Evaluate one arm end-to-end and write its PNGs into run dir + ``out_dir``.

    Args:
        run_dir: The arm's run directory (holds ``best_model.keras``).
        x_test: MNIST test images.
        y_test: MNIST test labels.
        real_pca: Real test images projected into the shared PCA-50 space.
        pca: The PCA fit ONCE on real test pixels (reused for the gen projection).
        n_gen: Number of prior samples to draw for the MMD estimate.
        seed: Base RNG seed.
        out_dir: Combined output directory (PNG copies land here too).

    Returns:
        Per-arm metrics dict.
    """
    model, sampler = load_arm(run_dir)
    logger.info(f"Evaluating arm '{sampler}' from {run_dir}")

    recon_bce, recon_mse = reconstruction_errors(model, x_test)

    # Generative MMD: decode n_gen prior samples (correct per-mode prior), flatten,
    # project into the SHARED PCA space fit on real test pixels.
    gen = np.asarray(keras.ops.convert_to_numpy(model.sample(n_gen)))
    gen_flat = gen.reshape(gen.shape[0], -1)
    gen_pca = pca.transform(gen_flat)
    mmd = mmd_to_real(real_pca, gen_pca, seed=seed)

    # Latent diagnostics + PNGs (mode-aware; D-002).
    hist_path = os.path.join(run_dir, "eval_latent_norm_hist.png")
    diag = latent_diagnostics(model, x_test, hist_path)

    prior_path = os.path.join(run_dir, "eval_prior_samples.png")
    recon_path = os.path.join(run_dir, "eval_reconstructions.png")
    interp_path = os.path.join(run_dir, "eval_interpolation.png")
    plot_prior_grid(model, prior_path)

    sample_imgs = x_test[np.random.choice(len(x_test), 10, replace=False)]
    recon_imgs = model.predict(sample_imgs, verbose=0)["reconstruction"]
    plot_reconstruction_comparison(sample_imgs, recon_imgs, recon_path, dataset="mnist")
    plot_interpolation_grid(model, interp_path)

    pngs = [hist_path, prior_path, recon_path, interp_path]
    if model.latent_dim == 2:
        scatter_path = os.path.join(run_dir, "eval_latent_scatter.png")
        angle_path = os.path.join(run_dir, "eval_latent_angle_hist.png")
        pngs.extend(
            plot_latent_diagnostics_2d(
                model, x_test[:5000], y_test[:5000], scatter_path, angle_path
            )
        )

    # Copy PNGs into the combined out dir, prefixed by sampler.
    for p in pngs:
        if os.path.isfile(p):
            dst = os.path.join(out_dir, f"{sampler}_{os.path.basename(p)}")
            with open(p, "rb") as src_f, open(dst, "wb") as dst_f:
                dst_f.write(src_f.read())

    return {
        "sampler": sampler,
        "recon_bce": recon_bce,
        "recon_mse": recon_mse,
        "mmd2_median": mmd["mmd2_median"],
        "mmd2_half": mmd["mmd2_half"],
        "mmd2_double": mmd["mmd2_double"],
        "active_units": diag["active_units"],
        "dir_concentration": diag["dir_concentration"],
        "angular_coverage": diag["angular_coverage"],
        "mean_z_mean_norm": diag["mean_z_mean_norm"],
    }


# --------------------------------------------------------------------------- #
# Output + verdict
# --------------------------------------------------------------------------- #

COLUMNS = [
    "sampler",
    "recon_bce",
    "recon_mse",
    "mmd2_median",
    "mmd2_half",
    "mmd2_double",
    "active_units",
    "dir_concentration",
    "angular_coverage",
    "mean_z_mean_norm",
]


def write_table(rows: List[Dict[str, float]], out_dir: str) -> None:
    """Write the combined metrics as CSV + markdown."""
    csv_path = os.path.join(out_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(out_dir, "eval_metrics.md")
    with open(md_path, "w") as f:
        f.write("# VAE sampler comparison (fair metrics)\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(COLUMNS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r[c]) for c in COLUMNS) + " |\n")
        f.write(
            "\n> **Note (D-002).** For the hypersphere modes the decoder consumes "
            "the ON-SPHERE latent `z = radius * normalize(z_mean + eps)`, so "
            "`mean_z_mean_norm` (the raw z_mean magnitude) is **irrelevant** — only "
            "the direction matters. The honest, dimension-agnostic collapse signal "
            "is `dir_concentration` = `||mean(u)||` over per-sample unit directions "
            "`u = z_mean/||z_mean||` (0.0 = directions uniformly spread = healthy "
            "latent usage; 1.0 = all directions identical = collapsed; LOWER is "
            "better). `angular_coverage` (latent_dim==2 only) is the fraction of 36 "
            "angle-bins holding > uniform/10 of samples (HIGHER = more of the circle "
            "used). `total_loss`/`kl_loss` are NOT comparable across modes and are "
            "deliberately omitted.\n"
        )


def _fmt(v) -> str:
    """Format a cell: floats to 5 sig digits, else str."""
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def print_verdict(rows: List[Dict[str, float]]) -> None:
    """Print the combined table + a tentative PASS/FAIL per hypersphere arm.

    Decision-rule: a hypersphere arm "works" vs gaussian if recon_bce is within
    ~10% AND mmd2_median is lower-or-comparable (<= 1.10x gaussian). The
    ``dir_concentration`` (LOWER = better latent usage; the honest, dimension-
    agnostic collapse signal of the on-sphere DIRECTION, D-002) is reported
    alongside recon + MMD to flag a real (vs viz-artifact) latent collapse.
    Final human judgment is the orchestrator's.
    """
    lines = ["", "=" * 78, "FAIR VAE SAMPLER COMPARISON (recon + MMD; total_loss NOT compared)", "=" * 78]
    header = "| " + " | ".join(f"{c:>16}" for c in COLUMNS) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["-" * 18] * len(COLUMNS)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(f"{_fmt(r[c]):>16}" for c in COLUMNS) + " |")

    gauss = next((r for r in rows if r["sampler"] == "gaussian"), None)
    lines.append("-" * 78)
    if gauss is None:
        lines.append("No gaussian baseline arm found -> cannot apply verdict rule.")
    else:
        lines.append(
            "Verdict (hypersphere arm vs gaussian baseline; "
            "dir_concentration LOWER = better latent usage):"
        )
        lines.append(
            f"  {'gaussian (baseline)':>22}: "
            f"dir_concentration {gauss['dir_concentration']:.3f}"
        )
        for r in rows:
            if r["sampler"] == "gaussian":
                continue
            recon_delta = (r["recon_bce"] - gauss["recon_bce"]) / max(
                abs(gauss["recon_bce"]), 1e-12
            )
            g_mmd = gauss["mmd2_median"]
            mmd_ratio = r["mmd2_median"] / g_mmd if g_mmd > 1e-12 else float("inf")
            mmd_delta = r["mmd2_median"] - g_mmd
            recon_ok = recon_delta <= 0.10
            mmd_ok = mmd_ratio <= 1.10
            flag = "PASS" if (recon_ok and mmd_ok) else "FAIL"
            lines.append(
                f"  {r['sampler']:>22}: recon_bce {recon_delta:+.1%} vs gauss | "
                f"mmd2_median {mmd_delta:+.4g} ({mmd_ratio:.2f}x) | "
                f"dir_concentration {r['dir_concentration']:.3f} | {flag}"
            )
    lines.append("=" * 78)
    lines.append("")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run_evaluation(
    run_dirs: List[str], out_dir: str, n_gen: int, seed: int
) -> List[Dict[str, float]]:
    """Evaluate all arms against a shared real-PCA reference and write artifacts.

    Args:
        run_dirs: The arm run directories.
        out_dir: Combined output directory.
        n_gen: Prior samples per arm for MMD.
        seed: Base RNG seed.

    Returns:
        List of per-arm metrics dicts.
    """
    os.makedirs(out_dir, exist_ok=True)
    x_test, y_test = load_mnist_test()

    # Fit PCA-50 ONCE on the FULL real test set; reuse for every arm so the MMD
    # is comparable across arms (same reference geometry).
    real_flat = x_test.reshape(x_test.shape[0], -1)
    n_comp = min(50, real_flat.shape[1], real_flat.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed).fit(real_flat)
    real_pca = pca.transform(real_flat)
    logger.info(f"PCA fit on real test set: {n_comp} components")

    rows: List[Dict[str, float]] = []
    for run_dir in run_dirs:
        rows.append(
            evaluate_arm(run_dir, x_test, y_test, real_pca, pca, n_gen, seed, out_dir)
        )

    write_table(rows, out_dir)
    print_verdict(rows)
    logger.info(f"Wrote eval_metrics.csv / eval_metrics.md to {out_dir}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Fair mode-agnostic VAE sampler evaluation (recon + MMD)."
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Run dirs (one per sampler). If omitted, auto-discover newest per sampler.",
    )
    parser.add_argument(
        "--out", default="results/vae_sampler_compare_mnist",
        help="Combined output directory.",
    )
    parser.add_argument("--n-gen", type=int, default=5000, dest="n_gen",
                        help="Prior samples per arm for the MMD estimate.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    run_dirs = args.runs if args.runs else discover_runs()
    if not run_dirs:
        raise SystemExit("No run dirs given and none auto-discovered.")
    logger.info("Evaluating arms:")
    for d in run_dirs:
        logger.info(f"  - {d}")

    run_evaluation(run_dirs, args.out, args.n_gen, args.seed)


if __name__ == "__main__":
    main()
