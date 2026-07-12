"""Evaluate a bias-free denoiser's PSNR vs noise level across one or more datasets.

For each dataset this samples ``num_samples`` clean image patches, corrupts them with
additive white Gaussian noise at a sweep of standard deviations, denoises with the
supplied model, and plots **mean PSNR with confidence-interval bands** versus noise
level (one curve per dataset).

Design notes
------------
- Noise / PSNR conventions match ``train_convunext_denoiser.py``: images are normalized
  to ``[0, 1]`` (peak-to-peak range 1.0), noise is ``y = x + N(0, sigma^2)`` optionally
  clipped back to ``[0, 1]`` (the trainer clips), and per-image PSNR is
  ``10 * log10(max_val^2 / MSE)`` with ``max_val = 1.0`` -- so the dB values are directly
  comparable to the trainer's ``val_psnr`` and to published ``max_val=255`` numbers
  (PSNR is scale-invariant). The domain migration from the legacy ``[-0.5, +0.5]`` is a
  pure DC shift: peak-to-peak width is 1.0 in BOTH domains, so sigma and PSNR/SSIM are
  numerically UNCHANGED (``common.py`` D-001 / plan INV-2).
- Noise std is specified on the ``[0, 255]`` pixel scale (``--sigmas-255``, the benchmark
  convention) and converted internally via ``sigma_norm = sigma_255 / 255``.
- **Paired design**: the same sampled patches are reused across every noise level (only
  the noise realization changes), which removes content variance from the curve and
  tightens the confidence intervals. Noise draws are seeded for reproducibility.
- The model output is the denoised estimate directly (the denoiser is trained with MSE
  against the clean target, ``final_activation='linear'``). Multi-output models (deep
  supervision / exposed bottleneck) are reduced to output 0 (the final denoised image).

Usage::

    # 100 DIV2K-validation images per noise level, on GPU 1, default sigma sweep:
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.eval_psnr_vs_noise \\
        --model maxpool=results/convunext_denoiser_base_20260622_133223/best_model.keras \\
        --dataset div2k=/media/arxwn/data0_4tb/datasets/div2k/validation \\
        --num-samples 100 --gpu 1

    # Overlay several models on one plot (each on identical patches+noise), custom sweep:
    ... --model maxpool=run_a/best_model.keras --model laplacian=run_b/best_model.keras \\
        --dataset div2k=/path/div2k/validation --sigmas-255 5 10 15 25 35 50 65 80 100
"""

import gc
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # headless: avoid X11 crashes
import matplotlib.pyplot as plt
from scipy import stats

import keras

from dl_techniques.utils.logger import logger
from train.common import setup_gpu, collect_image_paths, set_seeds
# Single source of truth for the pixel domain (see common.py D-001). This eval script
# re-implements the *decode* path (it needs a no-crop full-image variant and an EvalConfig,
# not a BFUnetTrainingConfig) but must NOT re-declare the bounds: import them.
from train.bfunet.common import DATA_MIN, DATA_MAX

# Importing the model module registers every custom layer / initializer the saved
# denoiser needs (ConvNeXt blocks, Gabor stem, Laplacian pyramid, LayerScale, ...) so
# ``keras.models.load_model`` resolves them from the serialization registry.
import dl_techniques.models.bias_free_denoisers.bfconvunext  # noqa: F401

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

# Benchmark-convention noise stds on the [0, 255] pixel scale. Spans the classic
# 15/25/50 AWGN regimes plus the edges of this trainer's curriculum (sigma_255 ~6.4..63.75).
DEFAULT_SIGMAS_255: List[float] = [5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 65.0]
# Images live in [0, 1] -> peak-to-peak range = 1.0. This value is domain-INVARIANT (it was
# also 1.0 on the legacy [-0.5,+0.5] domain, whose peak-to-peak width is likewise 1.0). Do
# NOT "fix" it to match the new domain -- see common.py D-001 / plan INV-2.
MAX_VAL: float = 1.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}

# Published color AWGN full-image PSNR (dB) reference points, keyed by dataset (lowercase)
# then reference name then sigma_255. Source: Table 2 of Zhang et al., "SCUNet"
# (arXiv:2203.13278). "DnCNN" is the classic-CNN floor; "SOTA best" is the per-cell max over
# {DRUNet, SwinIR, Restormer, SCUNet} (modern transformer / large-UNet ceiling). These are
# overlaid on the plot at matching (dataset, sigma) for context -- not a like-for-like claim
# unless the run used --full-image on the same set.
SOTA_REFERENCE: Dict[str, Dict[str, Dict[float, float]]] = {
    "cbsd68": {"DnCNN": {15: 33.90, 25: 31.24, 50: 27.95},
               "SOTA best": {15: 34.42, 25: 31.79, 50: 28.61}},
    "kodak24": {"DnCNN": {15: 34.60, 25: 32.14, 50: 28.95},
                "SOTA best": {15: 35.47, 25: 33.04, 50: 30.01}},
    "mcmaster": {"DnCNN": {15: 33.45, 25: 31.52, 50: 28.62},
                 "SOTA best": {15: 35.61, 25: 33.34, 50: 30.30}},
    "urban100": {"DnCNN": {15: 32.98, 25: 30.81, 50: 27.59},
                 "SOTA best": {15: 35.18, 25: 33.03, 50: 30.14}},
}
SOTA_STYLE = {"DnCNN": dict(marker="v", color="0.45"),
              "SOTA best": dict(marker="*", color="black")}


@dataclass
class EvalConfig:
    """Configuration for the PSNR-vs-noise evaluation."""
    models: Dict[str, str]                    # name -> saved .keras denoiser path
    datasets: Dict[str, List[str]]            # name -> list of image directories
    sigmas_255: List[float] = field(default_factory=lambda: list(DEFAULT_SIGMAS_255))
    num_samples: int = 100
    patch_size: int = 256
    channels: int = 3
    batch_size: int = 16
    full_image: bool = False                  # evaluate whole images (SOTA protocol) vs patches
    size_multiple: int = 16                   # full-image: reflect-pad H,W to this multiple
    clip_noise: bool = True                   # match the trainer (clips to [0, 1])
    confidence: float = 0.95
    seed: int = 42
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    @property
    def sigmas_norm(self) -> List[float]:
        """Noise stds in the model's normalized [0, 1] space (unchanged by the domain
        migration: sigma is a width, and peak-to-peak width is 1.0 in both domains)."""
        return [s / 255.0 for s in self.sigmas_255]


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_denoiser(model_path: str) -> keras.Model:
    """Load a saved ``.keras`` denoiser (weights + graph), compile-free."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = keras.models.load_model(path, compile=False)
    n_out = len(model.outputs) if isinstance(model.outputs, (list, tuple)) else 1
    logger.info(
        f"Loaded denoiser '{model.name}' ({model.count_params():,} params, "
        f"{n_out} output(s)) from {model_path}"
    )
    return model


def _to_flexible_input(model: keras.Model) -> keras.Model:
    """Rebuild a fixed-input functional denoiser with a spatially-flexible ``(None, None, C)``
    input so it accepts whole images of any size.

    The trainer bakes a static ``(patch, patch, C)`` ``Input`` into the saved model, which
    rejects full-size images. The denoiser is fully convolutional (conv / pooling / channel
    norm / fixed-kernel Gabor + Laplacian), so every weight is spatially independent and
    transfers 1:1 to the rebuilt graph.
    """
    cfg = model.get_config()
    patched = False
    for layer in cfg.get("layers", []):
        if layer.get("class_name") == "InputLayer":
            lc = layer["config"]
            key = "batch_shape" if "batch_shape" in lc else (
                "batch_input_shape" if "batch_input_shape" in lc else None)
            if key and lc.get(key) and len(lc[key]) == 4:
                b = list(lc[key])
                lc[key] = [b[0], None, None, b[3]]
                patched = True
    if not patched:
        logger.warning(f"'{model.name}': could not relax input shape; full-image eval may fail")
        return model
    flex = keras.Model.from_config(cfg)
    flex.set_weights(model.get_weights())
    logger.info(f"rebuilt '{model.name}' with flexible (None,None,C) input for full-image eval")
    return flex


def _predict_denoised(model: keras.Model, noisy: np.ndarray, batch_size: int) -> np.ndarray:
    """Run the denoiser; reduce multi-output models to output 0 (final denoised image)."""
    pred = model.predict(noisy, batch_size=batch_size, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]  # deep supervision / exposed bottleneck -> final denoised output
    return np.asarray(pred)


# ---------------------------------------------------------------------
# Data sampling (clean patches in [0, 1])
# ---------------------------------------------------------------------

def _load_clean_patch(
    path: str, patch_size: int, channels: int, rng: np.random.RandomState
) -> Optional[np.ndarray]:
    """Decode one image, normalize to [0, 1], and crop a random patch.

    Returns ``None`` on a decode failure (the caller skips and resamples). Upscales
    (aspect-preserving) any image smaller than ``patch_size`` before cropping, mirroring
    ``decode_full_image`` in the trainer.

    NOTE: the ``/ 255.0`` normalization DUPLICATES ``common.decode_full_image`` (which
    takes a ``BFUnetTrainingConfig`` and always crops, neither of which fits this script's
    ``EvalConfig`` + optional full-image path). The bounds themselves are imported
    (``DATA_MIN``/``DATA_MAX``) so the two cannot drift apart silently.
    """
    try:
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=channels, expand_animations=False)
    except Exception as exc:  # noqa: BLE001 - corrupt files are skipped, not fatal
        logger.warning(f"skip unreadable image {path}: {exc}")
        return None
    img = tf.cast(img, tf.float32) / 255.0
    h, w = int(tf.shape(img)[0]), int(tf.shape(img)[1])
    if h < patch_size or w < patch_size:
        scale = patch_size / min(h, w)
        img = tf.image.resize(img, [int(np.ceil(h * scale)), int(np.ceil(w * scale))])
        h, w = int(tf.shape(img)[0]), int(tf.shape(img)[1])
    top = rng.randint(0, h - patch_size + 1)
    left = rng.randint(0, w - patch_size + 1)
    patch = img[top:top + patch_size, left:left + patch_size, :]
    arr = patch.numpy()
    if arr.shape != (patch_size, patch_size, channels):
        return None
    # Drop blank/corrupt decodes (all-zero) so they do not skew the curve. SEMANTIC FLIP
    # under [0,1] (no code change needed, mirrors common.py's streaming filter): this now
    # drops flat BLACK images, where on the legacy [-0.5,+0.5] domain it dropped flat
    # MID-GREY ones. Both are degenerate; black is the more sensible thing to drop.
    return None if not np.any(np.abs(arr) > 0) else arr


def sample_clean_patches(cfg: EvalConfig, paths: List[str], rng: np.random.RandomState) -> np.ndarray:
    """Sample ``cfg.num_samples`` clean patches (one random crop per drawn image)."""
    if not paths:
        raise ValueError("No image paths to sample from")
    order = rng.permutation(len(paths))
    patches: List[np.ndarray] = []
    i = 0
    # Walk the shuffled path list (wrapping if num_samples > #images) until we have enough.
    while len(patches) < cfg.num_samples and i < cfg.num_samples * 4:
        path = paths[order[i % len(paths)]]
        patch = _load_clean_patch(path, cfg.patch_size, cfg.channels, rng)
        if patch is not None:
            patches.append(patch)
        i += 1
    if len(patches) < cfg.num_samples:
        logger.warning(
            f"requested {cfg.num_samples} patches but only loaded {len(patches)} "
            f"(too few readable images?)"
        )
    return np.stack(patches, axis=0).astype(np.float32)


def _load_full_image(path: str, channels: int) -> Optional[np.ndarray]:
    """Decode a whole image, normalize to [0, 1]; no cropping. None on failure.

    Same duplicated-normalizer note as ``_load_clean_patch``.
    """
    try:
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=channels, expand_animations=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"skip unreadable image {path}: {exc}")
        return None
    arr = (tf.cast(img, tf.float32) / 255.0).numpy()
    return None if not np.any(np.abs(arr) > 0) else arr


def sample_full_images(cfg: EvalConfig, paths: List[str], rng: np.random.RandomState) -> List[np.ndarray]:
    """Load up to ``num_samples`` whole images (deterministic order) for full-image eval."""
    order = rng.permutation(len(paths))
    images: List[np.ndarray] = []
    for idx in order:
        if len(images) >= cfg.num_samples:
            break
        img = _load_full_image(paths[idx], cfg.channels)
        if img is not None:
            images.append(img)
    return images


def _denoise_full(model: keras.Model, noisy: np.ndarray, size_multiple: int) -> np.ndarray:
    """Denoise one whole image: reflect-pad H,W to a multiple of ``size_multiple`` (the
    U-Net downsample factor), run a single forward pass, then crop back to the original size."""
    h, w = noisy.shape[:2]
    ph = (size_multiple - h % size_multiple) % size_multiple
    pw = (size_multiple - w % size_multiple) % size_multiple
    padded = np.pad(noisy, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    out = _predict_denoised(model, padded[None, ...], batch_size=1)[0]
    return out[:h, :w, :]


# ---------------------------------------------------------------------
# Noise + PSNR
# ---------------------------------------------------------------------

def add_awgn(clean: np.ndarray, sigma_norm: float, clip: bool, rng: np.random.RandomState) -> np.ndarray:
    """Additive white Gaussian noise ``y = x + N(0, sigma^2)`` in normalized space."""
    noisy = clean + rng.standard_normal(clean.shape).astype(np.float32) * np.float32(sigma_norm)
    return np.clip(noisy, DATA_MIN, DATA_MAX) if clip else noisy


def psnr_per_image(denoised: np.ndarray, clean: np.ndarray, max_val: float = MAX_VAL) -> np.ndarray:
    """Per-image PSNR (dB): ``10 * log10(max_val^2 / MSE)`` -- matches PsnrMetric."""
    mse = np.mean((denoised - clean) ** 2, axis=(1, 2, 3))
    mse = np.maximum(mse, 1e-12)  # guard against perfect reconstruction / log(0)
    return 10.0 * np.log10((max_val ** 2) / mse)


def mean_ci(values: np.ndarray, confidence: float) -> Tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` using a Student-t interval on the mean."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    sem = float(stats.sem(values))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * sem
    return mean, mean - half, mean + half


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_dataset(
    models: Dict[str, keras.Model], cfg: EvalConfig, name: str,
    paths: List[str], rng: np.random.RandomState,
) -> List[Dict]:
    """Evaluate every model on one dataset across all noise levels.

    All models see the SAME sampled clean patches and the SAME noise realization at
    each sigma (the noisy batch is built once per sigma, then fed to each model), so
    the per-model curves are a fair paired comparison. Returns per (model, sigma) rows.
    """
    if cfg.full_image:
        logger.info(f"[{name}] loading up to {cfg.num_samples} FULL images from {len(paths)} "
                    f"(full-image PSNR; SOTA protocol)")
        clean_list = sample_full_images(cfg, paths, rng)
    else:
        logger.info(f"[{name}] sampling {cfg.num_samples} clean {cfg.patch_size}px patches "
                    f"from {len(paths)} images")
        clean = sample_clean_patches(cfg, paths, rng)

    rows: List[Dict] = []
    for sigma_255, sigma_norm in zip(cfg.sigmas_255, cfg.sigmas_norm):
        if cfg.full_image:
            # Per-image noise realization (shared across models), variable sizes -> lists.
            noisy_list = [add_awgn(c, sigma_norm, cfg.clip_noise, rng) for c in clean_list]
            psnr_in = np.array([psnr_per_image(n[None], c[None])[0]
                                for n, c in zip(noisy_list, clean_list)])
        else:
            noisy = add_awgn(clean, sigma_norm, cfg.clip_noise, rng)  # one realization, shared
            psnr_in = psnr_per_image(noisy, clean)
        in_mean, _, _ = mean_ci(psnr_in, cfg.confidence)  # model-free baseline
        for model_name, model in models.items():
            if cfg.full_image:
                psnr_out = np.array([
                    psnr_per_image(_denoise_full(model, n, cfg.size_multiple)[None], c[None])[0]
                    for n, c in zip(noisy_list, clean_list)
                ])
            else:
                denoised = _predict_denoised(model, noisy, cfg.batch_size)
                psnr_out = psnr_per_image(denoised, clean)
            mean, lo, hi = mean_ci(psnr_out, cfg.confidence)
            rows.append({
                "model": model_name, "dataset": name, "sigma_255": sigma_255,
                "sigma_norm": sigma_norm, "n": int(psnr_out.size), "psnr_mean": mean,
                "psnr_ci_low": lo, "psnr_ci_high": hi, "psnr_std": float(np.std(psnr_out, ddof=1)),
                "input_psnr_mean": in_mean, "gain_db": mean - in_mean,
            })
            logger.info(
                f"[{name}/{model_name}] sigma_255={sigma_255:6.1f} (norm={sigma_norm:.4f}) -> "
                f"PSNR {mean:5.2f} dB [{lo:5.2f}, {hi:5.2f}]  (input {in_mean:5.2f}, "
                f"+{mean - in_mean:4.2f} dB)"
            )
    if cfg.full_image:
        del clean_list
    else:
        del clean
    gc.collect()
    return rows


# ---------------------------------------------------------------------
# Output: plot + CSV/JSON
# ---------------------------------------------------------------------

def _plot_dataset_panel(ax, rows: List[Dict], ds_name: str, models: List[str],
                        cmap, eval_sigmas: set) -> None:
    """Draw one dataset's panel: per-model mean+CI curves, noisy baseline, SOTA overlay."""
    for mi, model_name in enumerate(models):
        d = sorted([r for r in rows if r["model"] == model_name and r["dataset"] == ds_name],
                   key=lambda r: r["sigma_255"])
        if not d:
            continue
        x = [r["sigma_255"] for r in d]
        color = cmap(mi % 10)  # consistent per-model color across panels
        ax.plot(x, [r["psnr_mean"] for r in d], "-o", color=color, label=model_name,
                linewidth=2, markersize=5)
        ax.fill_between(x, [r["psnr_ci_low"] for r in d], [r["psnr_ci_high"] for r in d],
                        color=color, alpha=0.20)

    # Dashed noisy-input baseline (model-independent; dedupe repeated sigmas).
    d = sorted([r for r in rows if r["dataset"] == ds_name], key=lambda r: r["sigma_255"])
    seen, base = set(), []
    for r in d:
        if r["sigma_255"] not in seen:
            seen.add(r["sigma_255"]); base.append(r)
    ax.plot([r["sigma_255"] for r in base], [r["input_psnr_mean"] for r in base],
            "--", color="0.5", alpha=0.7, linewidth=1, label="noisy input")

    # Published SOTA reference points at matching sigmas.
    ref = SOTA_REFERENCE.get(ds_name.lower())
    if ref:
        for ref_name, pts in ref.items():
            xs = sorted(s for s in pts if s in eval_sigmas)
            if xs:
                ax.plot(xs, [pts[s] for s in xs], linestyle=":", linewidth=1.3, markersize=9,
                        **SOTA_STYLE[ref_name], label=f"{ref_name} [lit.]", alpha=0.85)

    ax.set_title(ds_name)
    ax.set_xlabel("Noise std  σ  (on [0, 255] scale)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_results(rows: List[Dict], cfg: EvalConfig, out_path: Path) -> None:
    """Plot mean PSNR with 95% CI band vs noise level, one panel per dataset.

    Each panel overlays the per-model curves (mean + CI), the dashed noisy-input baseline,
    and any published SOTA reference points for that dataset. Models keep a consistent
    color across panels. Panels share the y-axis for cross-dataset comparison.
    """
    models = sorted({r["model"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    eval_sigmas = {r["sigma_255"] for r in rows}
    cmap = plt.get_cmap("tab10")

    n = len(datasets)
    ncols = min(n, 2)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.0 * nrows),
                             squeeze=False, sharey=True)
    axes_flat = axes.flatten()
    for di, ds_name in enumerate(datasets):
        _plot_dataset_panel(axes_flat[di], rows, ds_name, models, cmap, eval_sigmas)
        if di % ncols == 0:
            axes_flat[di].set_ylabel("PSNR (dB)")
    for j in range(n, len(axes_flat)):  # hide unused cells
        axes_flat[j].set_visible(False)

    sample_desc = ("full images" if cfg.full_image
                   else f"{cfg.num_samples} × {cfg.patch_size}px patches")
    fig.suptitle(f"Denoiser PSNR vs noise level — {sample_desc}/level, "
                 f"{int(cfg.confidence * 100)}% CI of the mean")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"Saved plot -> {out_path}")


def write_tables(rows: List[Dict], out_dir: Path) -> None:
    """Dump the per-sigma stats as CSV + JSON for downstream use."""
    fields = ["model", "dataset", "sigma_255", "sigma_norm", "n", "psnr_mean", "psnr_ci_low",
              "psnr_ci_high", "psnr_std", "input_psnr_mean", "gain_db"]
    csv_path = out_dir / "psnr_vs_noise.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "psnr_vs_noise.json", "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Saved tables -> {csv_path} (+ .json)")


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig) -> Path:
    """Load all models, evaluate every dataset, write plot + tables; return out dir."""
    set_seeds(cfg.seed)
    models = {name: load_denoiser(path) for name, path in cfg.models.items()}
    if cfg.full_image:  # relax the fixed (patch,patch) input so whole images are accepted
        models = {name: _to_flexible_input(m) for name, m in models.items()}

    name = cfg.experiment_name or f"psnr_vs_noise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for ds_name, dirs in cfg.datasets.items():
        paths = collect_image_paths(dirs, extensions=IMAGE_EXTENSIONS, sort=True)
        if not paths:
            logger.warning(f"[{ds_name}] no images found in {dirs}; skipping")
            continue
        rng = np.random.RandomState(cfg.seed)  # per-dataset reproducible sampling/noise
        all_rows.extend(evaluate_dataset(models, cfg, ds_name, paths, rng))

    if not all_rows:
        raise RuntimeError("No datasets produced results (no readable images?)")

    # Persist config alongside the results for reproducibility.
    with open(out_dir / "eval_config.json", "w") as f:
        json.dump({
            "models": cfg.models, "datasets": cfg.datasets,
            "sigmas_255": cfg.sigmas_255, "num_samples": cfg.num_samples,
            "patch_size": cfg.patch_size, "channels": cfg.channels,
            "full_image": cfg.full_image, "size_multiple": cfg.size_multiple,
            "clip_noise": cfg.clip_noise, "confidence": cfg.confidence, "seed": cfg.seed,
        }, f, indent=2)
    write_tables(all_rows, out_dir)
    plot_results(all_rows, cfg, out_dir / "psnr_vs_noise.png")
    logger.info(f"Done. Results in {out_dir}")
    return out_dir


def _parse_dataset_arg(items: List[str]) -> Dict[str, List[str]]:
    """Parse ``--dataset`` values of the form ``name=dir`` or bare ``dir``."""
    datasets: Dict[str, List[str]] = {}
    for item in items:
        if "=" in item:
            name, directory = item.split("=", 1)
        else:
            name, directory = Path(item).name, item
        datasets.setdefault(name, []).append(directory)
    return datasets


def _parse_model_arg(items: List[str]) -> Dict[str, str]:
    """Parse ``--model`` values of the form ``name=path`` or bare ``path``.

    A bare path is named after its parent directory (the run dir), e.g.
    ``results/convunext_denoiser_base_.../best_model.keras`` -> ``convunext_denoiser_base_...``.
    """
    models: Dict[str, str] = {}
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name, path = Path(item).parent.name or Path(item).stem, item
        if name in models:
            raise ValueError(f"duplicate model name '{name}'; give explicit name=path")
        models[name] = path
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate denoiser(s) PSNR vs noise level (mean + CI) across datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                        help="denoiser as 'name=path.keras' (or bare path); repeatable to "
                             "overlay multiple models on the same plot")
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR",
                        help="dataset as 'name=dir' (or bare 'dir'); repeatable for multiple")
    parser.add_argument("--sigmas-255", type=float, nargs="+", default=DEFAULT_SIGMAS_255,
                        help="noise stds on the [0,255] scale to sweep")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="image patches sampled per dataset (reused across noise levels)")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full-image", action="store_true",
                        help="evaluate whole images (SOTA protocol: reflect-pad to a multiple "
                             "of --size-multiple, denoise, crop back) instead of random patches")
    parser.add_argument("--size-multiple", type=int, default=16,
                        help="full-image mode: pad H,W up to this multiple (U-Net downsample factor)")
    parser.add_argument("--no-clip", action="store_true",
                        help="do NOT clip noisy images to [0,1] (trainer clips by default)")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    cfg = EvalConfig(
        models=_parse_model_arg(args.model),
        datasets=_parse_dataset_arg(args.dataset),
        sigmas_255=args.sigmas_255,
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        channels=args.channels,
        batch_size=args.batch_size,
        full_image=args.full_image,
        size_multiple=args.size_multiple,
        clip_noise=not args.no_clip,
        confidence=args.confidence,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
