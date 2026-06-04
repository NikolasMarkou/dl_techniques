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
    plot_latent_space,
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


def latent_diagnostics(
    model: VAE, x_test: np.ndarray, save_path: str, batch_size: int = 256
) -> Tuple[int, float, np.ndarray]:
    """Encode the test set and report active units + ``||z_mean||`` stats.

    Args:
        model: A loaded VAE arm.
        x_test: Test images ``(N, 28, 28, 1)``.
        save_path: Where to write the ``||z_mean||`` histogram PNG.
        batch_size: Forward-pass batch size.

    Returns:
        Tuple of (active_units, mean_latent_norm, per_dim_variance).
    """
    z_mean = model.predict(x_test, batch_size=batch_size, verbose=0)["z_mean"]
    z_mean = np.asarray(z_mean)
    per_dim_var = z_mean.var(axis=0)
    active_units = int(np.sum(per_dim_var > ACTIVE_UNIT_VAR_THRESHOLD))
    norms = np.linalg.norm(z_mean, axis=1)
    mean_norm = float(norms.mean())

    plt.figure(figsize=(7, 4))
    plt.hist(norms, bins=60, color="steelblue", alpha=0.85)
    plt.axvline(mean_norm, color="crimson", linestyle="--", label=f"mean={mean_norm:.3f}")
    plt.xlabel("||z_mean|| (L2 norm per sample)")
    plt.ylabel("count")
    plt.title(f"Latent norm distribution ({model.sampling_type})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return active_units, mean_norm, per_dim_var


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

    # Latent diagnostics + PNGs.
    hist_path = os.path.join(run_dir, "eval_latent_norm_hist.png")
    active_units, mean_norm, _ = latent_diagnostics(model, x_test, hist_path)

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
        plot_latent_space(model, x_test[:5000], y_test[:5000], scatter_path)
        if os.path.isfile(scatter_path):
            pngs.append(scatter_path)

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
        "active_units": active_units,
        "mean_latent_norm": mean_norm,
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
    "mean_latent_norm",
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


def _fmt(v) -> str:
    """Format a cell: floats to 5 sig digits, else str."""
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def print_verdict(rows: List[Dict[str, float]]) -> None:
    """Print the combined table + a tentative PASS/FAIL per hypersphere arm.

    Decision-rule: a hypersphere arm "works" vs gaussian if recon_bce is within
    ~10% AND mmd2_median is lower-or-comparable (<= 1.10x gaussian). Final human
    judgment is the orchestrator's.
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
        lines.append("Verdict (hypersphere arm vs gaussian baseline):")
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
                f"mmd2_median {mmd_delta:+.4g} ({mmd_ratio:.2f}x) | {flag}"
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
