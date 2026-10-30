"""The exact empirical Tweedie denoiser: an oracle for patch denoisers.

For a noisy patch ``y = x + N(0, sigma^2 I)`` and a bank ``{b_i}`` of N clean training
patches, the posterior mean under the empirical prior (a uniform mixture of point masses at
the bank patches) is

    x_hat(y) = sum_i b_i softmax_i(-||y - b_i||^2 / (2 sigma^2)),

which is one mean-shift step with bandwidth ``h = sigma`` over the bank
(``dl_techniques.layers.mixtures.mean_shift_ops.mean_shift_step``). No network can beat
it on patches drawn from the bank, and it says how much of a denoiser's score is
memorization of the training set. This script scores it, per noise level, on two query sets:

- ``in_bank``: patches that ARE in the bank (the oracle can copy them);
- ``held_out``: patches cropped from images that are NOT in the bank (the validation
  images), where the oracle can only be as good as the bank covers the patch manifold.

Metrics per (query set, sigma), each over the queries of that set:

- PSNR of the oracle, as mean and MEDIAN, plus the count at the per-image cap (120 dB:
  ``psnr_per_image`` floors the MSE at 1e-12);
- ``exp(H)``, the effective number of bank patches that contribute (``H`` is the entropy
  of the softmax weights in nats), as mean and median;
- the memorization ratio ``||x_hat - b*|| / ||x - b*||`` where ``b*`` is the bank patch
  nearest to ``x_hat`` (searched in float64): about 0 means the output is a copy of a bank
  patch, about 1 means it is as far from its nearest bank patch as the truth is. Reported
  as a median, the fraction below 0.5, a degenerate-query count (the truth is within
  ``DEGENERATE_RMS`` of ``b*``, so the denominator is about 0) and, for in-bank queries,
  the self-retrieval rate (``b*`` is the query's own clean patch).

Conventions:

- Noise is unclipped additive Gaussian by default, because Tweedie's identity holds only
  for unclipped noise (``--clip`` reproduces the trainer's clipped noise and changes the
  oracle by about 2.6 dB at sigma 50). Sigma is on the [0, 255] scale and converted by
  ``sigma / 255``; images are in [0, 1] and PSNR uses ``max_val = 1``, as the trainers do.
- The default ``--dtype float32`` is adequate for PSNR and the mean of ``exp(H)``; the
  expanded-square distance cancels below sigma 10/255, so per-query weights need
  ``--dtype float64``.
- Every noise level draws its own noise from one seeded generator per query set, in the
  order of ``--sigmas-255``, so a run is reproducible and the oracle and the model see the
  same noisy patches.

With ``--model-path`` a stamped (``data_range == "[0,1]"``) checkpoint whose input is
``(32, 32, 1)`` is scored on the same noisy patches and a side-by-side table is written;
without it the run is oracle only and says so. Nothing in this script trains.

Usage (from the repo root; the oracle needs no GPU and runs on CPU unless ``--gpu`` is given)::

    MPLBACKEND=Agg .venv/bin/python -m train.bfunet.eval_empirical_mean_shift \\
        --bank-size 16384 --num-queries 1000

Writes ``results/mean_shift_denoiser/<timestamp>/`` holding ``summary.json``,
``oracle_vs_sigma.csv``, ``psnr_vs_sigma.png`` and ``eval_config.json``. A run directory that
already holds a run is refused, never overwritten.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import keras
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: avoid X11 crashes
import matplotlib.pyplot as plt  # noqa: E402

from dl_techniques.layers.mixtures.mean_shift_ops import mean_shift_step
from dl_techniques.utils.logger import logger
from train.bfunet import eval_psnr_vs_noise as ev
from train.bfunet.common import decode_full_image
from train.common import collect_image_paths, set_seeds, setup_gpu
from train.common.config_io import json_numpy_default
from train.common.gpu import hide_gpus
from train.common.run_artifacts import refuse_existing_run
from train.common.run_io import run_timestamp

# Importing the model modules registers every custom layer a saved BFUNet or BFCNN needs
# (``eval_psnr_vs_noise`` registers only the ConvUNeXt one), so ``load_denoiser`` resolves them.
import dl_techniques.models.vision.bias_free_denoisers.bfcnn  # noqa: E402,F401
import dl_techniques.models.vision.bias_free_denoisers.bfunet  # noqa: E402,F401

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: Patch side and channel count of the oracle's setting: grayscale 32 x 32.
PATCH_SIZE = 32
CHANNELS = 1

#: Per-image PSNR ceiling: ``ev.psnr_per_image`` floors the MSE at 1e-12, i.e. 120 dB.
PSNR_CAP_DB = 120.0
#: Slack when counting images at the cap (the cap is an exact float, the slack is for safety).
PSNR_CAP_SLACK_DB = 1e-6

#: A query whose truth lies within this RMS (per pixel, [0, 1] scale) of its nearest bank
#: patch has a memorization denominator of about 0 and is counted, not scored.
DEGENERATE_RMS = 1e-3
#: The ratio below which an output counts as close to a copy of a bank patch.
COPY_RATIO = 0.5

#: The nearest bank patch is searched in this dtype. float32 cannot resolve two bank
#: patches that differ by 1e-3 when the patches are far from the origin.
SEARCH_DTYPE = np.float64

#: Batch size of the model arm's predict (padded to one trace, see ``ev._predict_denoised``).
MODEL_BATCH_SIZE = 64

DEFAULT_TRAIN_DIR = "/media/arxwn/data0_4tb/datasets/div2k/train"
DEFAULT_TEST_DIR = "/media/arxwn/data0_4tb/datasets/div2k/validation"
DEFAULT_OUTPUT_DIR = "results/mean_shift_denoiser"
DEFAULT_BANK_SIZE = 16384
DEFAULT_CHUNK_SIZE = 250
DEFAULT_NUM_QUERIES = 1000

SUMMARY_NAME = "summary.json"
CONFIG_NAME = "eval_config.json"
CSV_NAME = "oracle_vs_sigma.csv"
PLOT_NAME = "psnr_vs_sigma.png"
#: Files whose presence marks a directory that already holds a run of this script.
RUN_ARTIFACTS: Tuple[str, ...] = (SUMMARY_NAME, CONFIG_NAME, CSV_NAME, PLOT_NAME)

ORACLE_ONLY_STATEMENT = (
    "oracle only: no --model-path was given, so no denoiser was scored; the numbers are the "
    "exact empirical Tweedie denoiser over the bank."
)
SCORED_STATEMENT = (
    "the exact empirical Tweedie denoiser and the checkpoint at model.path were scored on the "
    "same noisy patches."
)

IN_BANK = "in_bank"
HELD_OUT = "held_out"


@dataclass(frozen=True)
class OracleConfig:
    """Everything a run depends on; written to ``eval_config.json``."""

    train_dir: str
    test_dir: str
    output_dir: str
    bank_size: int
    chunk_size: int
    num_queries: int
    sigmas_255: Tuple[float, ...]
    dtype: str
    clip: bool
    model_path: Optional[str]
    seed: int


@dataclass(frozen=True)
class _CropSpec:
    """The two attributes ``common.decode_full_image`` reads from its config argument."""

    patch_size: int
    channels: int


# ---------------------------------------------------------------------
# Data: decode once, crop many
# ---------------------------------------------------------------------

def load_crop_patches(
        paths: Sequence[str], count: int, patch_size: int, channels: int,
        rng: np.random.RandomState,
) -> Tuple[np.ndarray, int]:
    """Crop ``count`` patches from ``paths``, decoding each image once.

    Interface contract:

    * ``paths`` are image files; the order is shuffled by ``rng`` and the ``count`` crops
      are spread as evenly as possible over the images (the first ``count % len(paths)``
      of the shuffled order take one more). With ``count < len(paths)`` only the first
      ``count`` shuffled images are decoded.
    * Each image is decoded and normalised to [0, 1] by ``common.decode_full_image`` (the
      one normaliser of the pipeline); an image smaller than the patch is upscaled by it.
      Crop offsets are drawn from ``rng``, so a call is reproducible from the generator.
    * Returns ``(patches, n_images)``: patches ``(count, patch_size, patch_size, channels)``
      float32 and the number of images decoded.
    * Raises ``ValueError`` for an empty ``paths`` or ``count < 1``; a file that cannot be
      decoded raises out of TensorFlow (no image is skipped silently).

    Args:
        paths: Image file paths.
        count: Number of patches to return.
        patch_size: Crop side in pixels.
        channels: 1 (grayscale) or 3.
        rng: Generator for the image order and the crop offsets.

    Returns:
        The patches and the number of images decoded.
    """
    if not paths:
        raise ValueError("no image paths to crop from")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    # DECISION plan-2026-10-30T025213-f75ab4e9/D-003: decode each image once and crop many times.
    # Do not call ev.sample_clean_patches here: it decodes per patch (16384 decodes, about 18 min).
    spec = _CropSpec(patch_size=patch_size, channels=channels)
    order = rng.permutation(len(paths))
    per_image, extra = divmod(count, len(paths))
    patches = np.empty((count, patch_size, patch_size, channels), dtype=np.float32)
    filled = 0
    decoded = 0
    for rank, image_index in enumerate(order):
        take = per_image + (1 if rank < extra else 0)
        if take == 0:
            break
        image = decode_full_image(paths[image_index], spec).numpy()
        height, width = image.shape[:2]
        for _ in range(take):
            top = rng.randint(0, height - patch_size + 1)
            left = rng.randint(0, width - patch_size + 1)
            patches[filled] = image[top:top + patch_size, left:left + patch_size, :]
            filled += 1
        decoded += 1
    return patches, decoded


def duplicate_statistics(patches: np.ndarray) -> Dict[str, Any]:
    """Count the bank patches that are byte-identical to another bank patch.

    Args:
        patches: ``(N, ...)`` array; each patch is compared as a whole.

    Returns:
        ``n``, ``n_unique``, ``duplicate_share`` (copies beyond the first, over ``n``) and
        ``share_with_twin`` (patches that have at least one identical other patch, over ``n``).
    """
    flat = np.ascontiguousarray(patches.reshape(len(patches), -1))
    counts = Counter(hashlib.sha256(row.tobytes()).digest() for row in flat)
    n = len(flat)
    n_unique = len(counts)
    with_twin = sum(count for count in counts.values() if count > 1)
    return {
        "n": n,
        "n_unique": n_unique,
        "duplicate_share": (n - n_unique) / n,
        "share_with_twin": with_twin / n,
    }


# ---------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------

def effective_number(weights: np.ndarray) -> np.ndarray:
    """``exp(H)`` per row of a weight matrix, with ``H`` the entropy in nats.

    ``0 ln 0 = 0``. A uniform row over k entries gives k; ``(0.5, 0.5)`` gives 2.0 and
    ``(0.9, 0.1)`` gives 1.3841. (``1 / sum w^2`` would give 1.2195 for the latter, so the
    definition is named here.)

    Args:
        weights: ``(Q, N)`` rows summing to 1.

    Returns:
        ``(Q,)`` float64 effective numbers of contributing patches.
    """
    w = np.asarray(weights, dtype=np.float64)
    positive = w > 0
    log_w = np.log(np.where(positive, w, 1.0))
    entropy = -np.sum(np.where(positive, w * log_w, 0.0), axis=1)
    return np.exp(entropy)


def empirical_tweedie(
        noisy: np.ndarray, bank: np.ndarray, sigma: float, chunk_size: int,
        dtype: str = "float32",
) -> Tuple[np.ndarray, np.ndarray]:
    """The posterior mean under the empirical prior over ``bank``, one query chunk at a time.

    Interface contract:

    * ``noisy`` ``(Q, D)`` and ``bank`` ``(N, D)`` are flat patches in [0, 1]; ``sigma`` is
      the noise std on the same scale (``sigma_255 / 255``), used as the mean-shift
      bandwidth ``h``.
    * Queries are processed ``chunk_size`` at a time (the bank stays whole), so the
      ``(chunk, N)`` weight matrix is the peak allocation. The result does not depend on
      ``chunk_size`` apart from float rounding, and a final partial chunk is processed.
    * ``dtype`` is the arithmetic of the step (``float64`` removes the float32 cancellation
      of the expanded-square distance at small sigma).
    * Returns ``(estimate, effective)``: the estimate ``(Q, D)`` in ``dtype`` and ``exp(H)``
      of the weights ``(Q,)`` float64.
    * Raises ``ValueError`` for ``chunk_size < 1`` or no queries.

    Args:
        noisy: Noisy patches ``(Q, D)``.
        bank: Clean bank patches ``(N, D)``.
        sigma: Noise std in the [0, 1] domain.
        chunk_size: Queries per ``mean_shift_step`` call.
        dtype: ``"float32"`` or ``"float64"``.

    Returns:
        The denoised estimate and the effective number of contributing patches.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if len(noisy) == 0:
        raise ValueError("no queries")
    keys = keras.ops.convert_to_tensor(np.asarray(bank, dtype=dtype))
    estimates: List[np.ndarray] = []
    effective: List[np.ndarray] = []
    for start in range(0, len(noisy), chunk_size):
        queries = np.asarray(noisy[start:start + chunk_size], dtype=dtype)
        estimate, weights = mean_shift_step(queries, keys, keys, sigma)
        estimates.append(np.asarray(estimate))
        effective.append(effective_number(np.asarray(weights)))
    return np.concatenate(estimates), np.concatenate(effective)


# ---------------------------------------------------------------------
# Memorization
# ---------------------------------------------------------------------

def nearest_bank_patch(
        points: np.ndarray, bank: np.ndarray, chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Index of, and Euclidean distance to, the nearest bank patch of every point.

    The search runs in ``SEARCH_DTYPE`` (float64) on the expanded square after centring
    by the bank mean; the returned distance is a direct difference to the chosen patch.
    Ties go to the lowest index.

    Args:
        points: ``(Q, D)`` points (flattened patches).
        bank: ``(N, D)`` bank patches.
        chunk_size: Points per distance matrix.

    Returns:
        ``(index, distance)``, each ``(Q,)``.
    """
    bank = np.asarray(bank, dtype=SEARCH_DTYPE).reshape(len(bank), -1)
    points = np.asarray(points, dtype=SEARCH_DTYPE).reshape(len(points), -1)
    centre = bank.mean(axis=0, keepdims=True)
    centred_bank = bank - centre
    bank_sq = np.sum(centred_bank * centred_bank, axis=1)
    index = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk_size):
        chunk = points[start:start + chunk_size] - centre
        # The query norm is the same for every bank patch, so it cannot change the argmin.
        score = bank_sq[None, :] - 2.0 * (chunk @ centred_bank.T)
        index[start:start + len(chunk)] = np.argmin(score, axis=1)
    distance = np.linalg.norm(points - bank[index], axis=1)
    return index, distance


class MemorizationScores(NamedTuple):
    """Per-query memorization outcome (all arrays ``(Q,)``).

    ``ratio`` is NaN where the query is not scored (self-retrieved or degenerate).
    """

    ratio: np.ndarray
    self_retrieved: np.ndarray
    degenerate: np.ndarray
    in_bank: bool


def memorization_scores(
        clean: np.ndarray, estimate: np.ndarray, bank: np.ndarray, chunk_size: int,
        in_bank: bool,
) -> MemorizationScores:
    """``||x_hat - b*|| / ||x - b*||`` per query, with the unscorable queries set aside.

    Interface contract:

    * ``clean`` and ``estimate`` are ``(Q, ...)`` and flatten to ``(Q, D)``; ``bank`` is
      ``(N, ...)`` and flattens to ``(N, D)``. ``b*`` is the bank patch nearest to the
      estimate.
    * With ``in_bank`` True a query whose ``b*`` is bit-identical to its clean patch is
      ``self_retrieved`` (the oracle copied the patch it was built from; an exact twin of it
      counts the same) and is not scored. With ``in_bank`` False nothing is self-retrieved.
    * A query that is not self-retrieved but whose truth lies within ``DEGENERATE_RMS`` of
      ``b*`` (RMS over the D pixels) is ``degenerate`` and is not scored: its denominator is
      about 0 (flat or saturated patches have exact twins).
    * ``ratio`` is the score of every other query, NaN elsewhere.

    Args:
        clean: Clean query patches.
        estimate: Denoised estimates of the same queries.
        bank: Bank patches.
        chunk_size: Points per distance matrix in the nearest-patch search.
        in_bank: Whether the queries are bank members.

    Returns:
        The per-query scores.
    """
    n = len(clean)
    truth = np.asarray(clean, dtype=np.float64).reshape(n, -1)
    flat_bank = np.asarray(bank, dtype=np.float64).reshape(len(bank), -1)
    index, distance_out = nearest_bank_patch(estimate.reshape(n, -1), flat_bank, chunk_size)
    distance_ref = np.linalg.norm(truth - flat_bank[index], axis=1)
    self_retrieved = (distance_ref == 0.0) if in_bank else np.zeros(n, dtype=bool)
    degenerate = ~self_retrieved & (distance_ref / math.sqrt(truth.shape[1]) < DEGENERATE_RMS)
    scored = ~self_retrieved & ~degenerate
    ratio = np.full(n, np.nan)
    ratio[scored] = distance_out[scored] / distance_ref[scored]
    return MemorizationScores(ratio, self_retrieved, degenerate, in_bank)


def summarize_memorization(scores: MemorizationScores) -> Dict[str, Any]:
    """Median ratio, fraction below ``COPY_RATIO``, and the counts a mean would hide.

    Returns:
        ``n_scored``, ``median`` and ``frac_below_half`` (``None`` when nothing is scored),
        ``n_degenerate`` and ``self_retrieval_rate`` (``None`` for a held-out set).
    """
    # DECISION plan-2026-10-30T025213-f75ab4e9/D-002: report the median ratio, never the mean.
    # Do not average the ratio: exact twins make the denominator 0 (a measured held-out mean of 8.5e7).
    scored = scores.ratio[~np.isnan(scores.ratio)]
    return {
        "n_scored": int(scored.size),
        "median": float(np.median(scored)) if scored.size else None,
        "frac_below_half": float(np.mean(scored < COPY_RATIO)) if scored.size else None,
        "n_degenerate": int(scores.degenerate.sum()),
        "self_retrieval_rate": float(scores.self_retrieved.mean()) if scores.in_bank else None,
    }


# ---------------------------------------------------------------------
# Per-arm metrics
# ---------------------------------------------------------------------

def psnr_summary(psnr: np.ndarray) -> Dict[str, Any]:
    """Mean, median and the count at the per-image cap of a PSNR vector."""
    return {
        "psnr_mean": float(np.mean(psnr)),
        "psnr_median": float(np.median(psnr)),
        "psnr_at_cap": int(np.sum(psnr >= PSNR_CAP_DB - PSNR_CAP_SLACK_DB)),
    }


def arm_metrics(
        clean: np.ndarray, estimate: np.ndarray, bank: np.ndarray, chunk_size: int,
        in_bank: bool,
) -> Dict[str, Any]:
    """PSNR summary and memorization summary of one arm's estimates (flat metric keys).

    Args:
        clean: Clean patches ``(Q, H, W, C)``.
        estimate: The arm's denoised patches, the same number of elements per query.
        bank: Bank patches.
        chunk_size: Points per distance matrix.
        in_bank: Whether the queries are bank members.

    Returns:
        ``psnr_*`` keys and ``mem_*`` keys.
    """
    estimate = np.asarray(estimate, dtype=np.float64).reshape(clean.shape)
    psnr = ev.psnr_per_image(estimate, clean.astype(np.float64), ev.MAX_VAL)
    memorization = summarize_memorization(
        memorization_scores(clean, estimate, bank, chunk_size, in_bank))
    metrics: Dict[str, Any] = dict(psnr_summary(psnr))
    metrics.update({f"mem_{key}": value for key, value in memorization.items()})
    return metrics


def evaluate_query_set(
        name: str, clean: np.ndarray, bank: np.ndarray, config: OracleConfig,
        rng: np.random.RandomState, model: Optional[keras.Model],
) -> List[Dict[str, Any]]:
    """Score the oracle (and the model, if any) on one query set at every sigma.

    Args:
        name: ``IN_BANK`` or ``HELD_OUT``.
        clean: Clean query patches ``(Q, H, W, C)``.
        bank: Bank patches ``(N, H, W, C)``.
        config: The run configuration.
        rng: Generator for the noise; one draw per sigma, in order.
        model: A loaded denoiser or ``None``.

    Returns:
        One row per sigma with the keys ``query_set``, ``sigma_255``, ``n``, ``noisy_*``,
        ``oracle_*`` and (with a model) ``model_*``.
    """
    in_bank = name == IN_BANK
    flat_bank = bank.reshape(len(bank), -1)
    rows: List[Dict[str, Any]] = []
    for sigma_255 in config.sigmas_255:
        sigma = sigma_255 / 255.0
        started = time.time()
        noisy = ev.add_awgn(clean, sigma, config.clip, rng)
        noisy_psnr = ev.psnr_per_image(noisy.astype(np.float64), clean.astype(np.float64), ev.MAX_VAL)
        row: Dict[str, Any] = {"query_set": name, "sigma_255": sigma_255, "n": int(len(clean))}
        row.update({f"noisy_{k}": v for k, v in psnr_summary(noisy_psnr).items()})
        estimate, effective = empirical_tweedie(
            noisy.reshape(len(noisy), -1), flat_bank, sigma, config.chunk_size, config.dtype)
        row.update({f"oracle_{k}": v for k, v in arm_metrics(
            clean, estimate, flat_bank, config.chunk_size, in_bank).items()})
        row["oracle_neff_mean"] = float(np.mean(effective))
        row["oracle_neff_median"] = float(np.median(effective))
        if model is not None:
            denoised = ev._predict_denoised(model, noisy, MODEL_BATCH_SIZE)
            row.update({f"model_{k}": v for k, v in arm_metrics(
                clean, denoised, flat_bank, config.chunk_size, in_bank).items()})
        rows.append(row)
        logger.info(
            f"[{name}] sigma_255={sigma_255:6.1f}: oracle PSNR median "
            f"{row['oracle_psnr_median']:7.2f} dB (noisy {row['noisy_psnr_median']:6.2f}), "
            f"exp(H) median {row['oracle_neff_median']:.2f}, memorization median "
            f"{row['oracle_mem_median']} ({time.time() - started:.1f} s)")
    return rows


# ---------------------------------------------------------------------
# Model arm
# ---------------------------------------------------------------------

def load_model_arm(model_path: str) -> keras.Model:
    """Load a stamped checkpoint and refuse one that cannot score 32 x 32 x 1 patches.

    The provenance gate (``data_range == "[0,1]"``) runs inside ``ev.load_denoiser`` before
    anything is loaded. A trained checkpoint has a static ``(patch, patch, C)`` input, so a
    256 x 256 x 3 model cannot be fed the oracle's patches and is refused, never resized.

    Args:
        model_path: A ``.keras`` file with a sibling ``config.json``.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: The file is missing.
        ValueError: The checkpoint is unstamped or its input shape is not (32, 32, 1).
    """
    model = ev.load_denoiser(model_path)
    expected = (None, PATCH_SIZE, PATCH_SIZE, CHANNELS)
    shape = model.input_shape
    if not isinstance(shape, tuple) or tuple(shape) != expected:
        raise ValueError(
            f"REFUSING checkpoint {model_path}: its input shape is {shape} but this script "
            f"scores {expected} patches. Train a denoiser with --patch-size {PATCH_SIZE} "
            f"--channels {CHANNELS}; the patches are never resized.")
    return model


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

def write_strict_json(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Write ``payload`` as strict JSON (non-finite floats become null) and return it.

    Numpy values go through ``json_numpy_default``; ``allow_nan=False`` makes a NaN token
    unable to reach the file (the dump raises before the file is opened).

    Args:
        path: Target file.
        payload: A JSON-compatible dict, possibly holding numpy scalars.

    Returns:
        The sanitised dict that was written.
    """
    # DECISION plan-2026-10-30T025213-f75ab4e9/D-003: a thin writer owns the name summary.json.
    # Do not call write_summary_json: it writes results_summary.json, a constant other trainers read.
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    written = clean(json.loads(json.dumps(payload, default=json_numpy_default)))
    path.write_text(json.dumps(written, indent=2, allow_nan=False))
    return written


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write the side-by-side table, one row per (query set, sigma)."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_psnr_vs_sigma(rows: List[Dict[str, Any]], path: Path, has_model: bool) -> None:
    """Median PSNR against sigma, one panel per query set: noisy input, oracle, model."""
    names = [name for name in (IN_BANK, HELD_OUT) if any(r["query_set"] == name for r in rows)]
    fig, axes = plt.subplots(1, len(names), figsize=(6.0 * len(names), 4.5), squeeze=False)
    for axis, name in zip(axes[0], names):
        subset = sorted((r for r in rows if r["query_set"] == name), key=lambda r: r["sigma_255"])
        sigmas = [r["sigma_255"] for r in subset]
        axis.plot(sigmas, [r["noisy_psnr_median"] for r in subset], "--", color="0.5", label="noisy input")
        axis.plot(sigmas, [r["oracle_psnr_median"] for r in subset], "-o", label="empirical Tweedie oracle")
        if has_model:
            axis.plot(sigmas, [r["model_psnr_median"] for r in subset], "-s", label="model")
        axis.set_title(f"{name} (median per-image PSNR)")
        axis.set_xlabel("noise std sigma (on the [0, 255] scale)")
        axis.set_ylabel("PSNR (dB)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run(config: OracleConfig, run_name: Optional[str] = None) -> Path:
    """Build the bank and the query sets, score the arms, write the run directory.

    Nothing is written until the run directory, the image paths and the checkpoint (if
    any) have been validated; a directory that already holds a run is refused.

    Args:
        config: The run configuration.
        run_name: Run directory name under ``config.output_dir``; a timestamp by default.

    Returns:
        The run directory.

    Raises:
        FileExistsError: The run directory already holds a run.
        ValueError: An image directory holds no images, or the checkpoint is refused.
    """
    set_seeds(config.seed)
    train_paths = collect_image_paths([config.train_dir], extensions=ev.IMAGE_EXTENSIONS, sort=True)
    test_paths = collect_image_paths([config.test_dir], extensions=ev.IMAGE_EXTENSIONS, sort=True)
    for label, directory, paths in (("train", config.train_dir, train_paths),
                                    ("test", config.test_dir, test_paths)):
        if not paths:
            raise ValueError(f"no images found in the {label} directory {directory}")
    run_dir = Path(config.output_dir) / (run_name or run_timestamp())
    refuse_existing_run(run_dir, RUN_ARTIFACTS)
    model = load_model_arm(config.model_path) if config.model_path else None

    run_dir.mkdir(parents=True, exist_ok=True)
    write_strict_json(run_dir / CONFIG_NAME, {
        **asdict(config), "patch_size": PATCH_SIZE, "channels": CHANNELS,
        "noise": "unclipped additive Gaussian" if not config.clip else "clipped to [0, 1]",
    })

    started = time.time()
    bank, n_train_images = load_crop_patches(
        train_paths, config.bank_size, PATCH_SIZE, CHANNELS, np.random.RandomState(config.seed))
    bank_stats = duplicate_statistics(bank)
    bank_seconds = time.time() - started
    logger.info(
        f"bank: {bank_stats['n']} patches from {n_train_images} images in {bank_seconds:.0f} s, "
        f"duplicate share {bank_stats['duplicate_share']:.4f}")
    held_out, n_test_images = load_crop_patches(
        test_paths, config.num_queries, PATCH_SIZE, CHANNELS, np.random.RandomState(config.seed + 1))
    chosen = np.random.RandomState(config.seed + 2).choice(
        len(bank), size=min(config.num_queries, len(bank)), replace=False)
    in_bank = bank[chosen]

    rows = evaluate_query_set(IN_BANK, in_bank, bank, config, np.random.RandomState(config.seed + 3), model)
    rows += evaluate_query_set(HELD_OUT, held_out, bank, config, np.random.RandomState(config.seed + 4), model)

    summary: Dict[str, Any] = {
        "oracle_only": model is None,
        "statement": ORACLE_ONLY_STATEMENT if model is None else SCORED_STATEMENT,
        "model": None if model is None else {
            "path": config.model_path, "input_shape": list(model.input_shape),
            "params": int(model.count_params())},
        "config": asdict(config),
        "noise": {"clipped": config.clip, "scale": "sigma_255 / 255 on a [0, 1] image"},
        "psnr_cap_db": PSNR_CAP_DB,
        "bank": {**bank_stats, "train_dir": config.train_dir, "n_images": n_train_images,
                 "seconds": bank_seconds},
        "held_out": {"n": int(len(held_out)), "test_dir": config.test_dir, "n_images": n_test_images},
        "in_bank": {"n": int(len(in_bank))},
        "rows": rows,
    }
    write_csv(run_dir / CSV_NAME, rows)
    plot_psnr_vs_sigma(rows, run_dir / PLOT_NAME, has_model=model is not None)
    write_strict_json(run_dir / SUMMARY_NAME, summary)
    logger.info(f"Done. Results in {run_dir}")
    return run_dir


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line (``--help`` prints ``usage:`` and exits before any work)."""
    parser = argparse.ArgumentParser(
        description="Score the exact empirical Tweedie denoiser (a mean-shift step with "
                    "h = sigma over a bank of training patches) on in-bank and held-out "
                    "patches, optionally beside a stamped denoiser checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bank-size", type=int, default=DEFAULT_BANK_SIZE,
                        help="number of clean training patches in the bank")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="queries per mean-shift call (lower it if memory is short)")
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES,
                        help="queries per query set (in-bank and held-out)")
    parser.add_argument("--sigmas-255", type=float, nargs="+", default=list(ev.DEFAULT_SIGMAS_255),
                        help="noise stds on the [0, 255] scale to sweep")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="arithmetic of the oracle (float64 for per-query weights below sigma 10)")
    # DECISION plan-2026-10-30T025213-f75ab4e9/D-002: the default noise is unclipped.
    # Do not clip by default: Tweedie holds for unclipped noise, and clipping moves the oracle 2.6 dB at sigma 50.
    parser.add_argument("--clip", action="store_true",
                        help="clip the noisy patches to [0, 1] like the trainers (not Tweedie-exact)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="a stamped (data_range [0,1]) .keras denoiser with a (32, 32, 1) input "
                             "to score beside the oracle; omit for an oracle-only run")
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR,
                        help="images the bank is cropped from")
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR,
                        help="images the held-out queries are cropped from (disjoint from the bank)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="parent of the timestamped run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index; by default the run is CPU only (the oracle needs no GPU)")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> OracleConfig:
    """Build the run configuration from parsed arguments."""
    return OracleConfig(
        train_dir=args.train_dir, test_dir=args.test_dir, output_dir=args.output_dir,
        bank_size=args.bank_size, chunk_size=args.chunk_size, num_queries=args.num_queries,
        sigmas_255=tuple(args.sigmas_255), dtype=args.dtype, clip=args.clip,
        model_path=args.model_path, seed=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: parse argv first, select the device, run."""
    args = parse_arguments(argv)
    if args.gpu is None:
        hide_gpus()
    else:
        setup_gpu(args.gpu)
    run(config_from_args(args))


if __name__ == "__main__":
    main()
