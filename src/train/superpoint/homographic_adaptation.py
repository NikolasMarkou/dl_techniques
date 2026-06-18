"""Homographic Adaptation self-labeling (SuperPoint stage 3).

This is stage 3 of the SuperPoint training recipe (DeTone et al., CVPRW 2018).
A detector trained on synthetic shapes (MagicPoint, stage 1) generalizes poorly
to real images; Homographic Adaptation closes that gap by self-labeling a corpus
of unlabeled REAL images (COCO ``train2017`` in real use) into pseudo
ground-truth keypoints.

Algorithm (per image)
----------------------
1. Sample ``N`` random homographies (plus the identity).
2. Warp the image by each homography ``H`` (``warp_image``).
3. Run the model -> detector logits ``(Hc, Wc, 65)`` -> softmax -> drop the
   dustbin channel -> ``depth_to_space`` the 64 within-cell channels back to a
   dense ``H x W`` keypoint-probability heatmap (``grid_logits_to_heatmap``).
4. Unwarp each heatmap back to the ORIGINAL frame with ``warp_image(heatmap,
   inv(H))`` (``warp_image`` expects a forward homography and inverts it
   internally, so passing ``inv(H)`` undoes the forward ``H`` applied in step 2).
5. Aggregate (average) the unwarped heatmaps across all runs.
6. NMS + threshold the aggregate -> discrete pseudo keypoints
   (``simple_nms``).
7. Encode pseudo keypoints into the 65-class grid label
   (``keypoints_to_grid_labels``) and save one ``.npz`` per image plus a JSON
   manifest under the repo-root ``results/`` directory.

**Inference, NOT XLA.** Model inference here is an eager ``model(x,
training=False)`` call. The descriptor head's bicubic upsample
(``keras.ops.image.resize`` -> ``ResizeBicubic``) has no XLA_GPU_JIT OpKernel in
TF 2.18 (see decisions.md D-004); an eager call sidesteps XLA entirely, which is
exactly what we want for a label-generation pass.

**Smoke vs. real.** ``--smoke`` builds a FRESH untrained SuperPoint and labels a
handful of tiny SYNTHETIC images (``generate_synthetic_sample``) -- it NEVER
reads real COCO. The real run loads a trained checkpoint (``--checkpoint``) and
the COCO image directory; the user owns that multi-hour job.

Usage::

    MPLBACKEND=Agg python -m train.superpoint.homographic_adaptation \\
        --checkpoint results/magicpoint_tiny_*/final_model.keras \\
        --image-dirs /media/arxwn/data0_4tb/datasets/COCO/train2017 \\
                     /media/arxwn/data0_4tb/datasets/div2k/train \\
        --dataset-weights 0.5 0.5 \\
        --n-homographies 100 --num-images 5000 --gpu 1

    # Fast smoke (seconds on GPU1, fresh model, synthetic images):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg \\
        python -m train.superpoint.homographic_adaptation --smoke --gpu 1
"""

import json
import keras
import argparse
import collections
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from train.common import (
    setup_gpu,
    set_seeds,
    save_config_json,
    collect_image_paths,
)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.homography import sample_homography, warp_image
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.models.superpoint import create_superpoint
from dl_techniques.datasets.synthetic_shapes import (
    generate_synthetic_sample,
    keypoints_to_grid_labels,
    DEFAULT_CELL,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class HomographicAdaptationConfig:
    """Configuration for Homographic Adaptation self-labeling (stage 3)."""

    # Model / checkpoint. None -> build a fresh untrained model (smoke).
    checkpoint_path: Optional[str] = None
    variant: str = "tiny"

    # Data. Multiple real-image datasets sourced simultaneously; the HA worklist
    # is drawn per-dataset by normalized ``dataset_weights`` (balanced default)
    # so DIV2K's ~800 images are not drowned by COCO's ~118K. See D-002.
    image_dirs: List[str] = field(
        default_factory=lambda: [
            "/media/arxwn/data0_4tb/datasets/COCO/train2017",
            "/media/arxwn/data0_4tb/datasets/div2k/train",
        ]
    )
    dataset_weights: Optional[List[float]] = None
    input_size: int = 240
    channels: int = 1
    cell: int = DEFAULT_CELL
    num_images: int = 5000

    # Adaptation
    n_homographies: int = 100
    nms_radius: int = 4
    detection_threshold: float = 0.015

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Smoke mode: fresh model, 3 tiny synthetic images, 4 homographies.
    smoke: bool = False

    def __post_init__(self):
        if self.smoke:
            self.checkpoint_path = None
            self.variant = "tiny"
            self.input_size = 64
            self.num_images = 3
            self.n_homographies = 4
            self.nms_radius = 2
            self.detection_threshold = 0.0  # keep some points on an untrained net

        if self.input_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid input size or channel configuration")
        if self.input_size % self.cell != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be divisible by cell "
                f"({self.cell}) so the detector grid matches the heatmap"
            )
        if self.variant not in ("tiny", "base", "large"):
            raise ValueError(f"Unknown variant: {self.variant}")

        if not self.image_dirs:
            raise ValueError("image_dirs must contain at least one directory")
        if self.dataset_weights is not None:
            if len(self.dataset_weights) != len(self.image_dirs):
                raise ValueError(
                    f"dataset_weights ({len(self.dataset_weights)}) must match "
                    f"image_dirs ({len(self.image_dirs)}) in length"
                )
            if any(w < 0 for w in self.dataset_weights):
                raise ValueError("dataset_weights must all be >= 0")
            total = float(sum(self.dataset_weights))
            if total <= 0:
                raise ValueError("dataset_weights must sum to > 0")
            self.dataset_weights = [w / total for w in self.dataset_weights]

        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"ha_selflabel_{self.variant}_{timestamp}"


# ---------------------------------------------------------------------
# DETECTOR DECODE: 65-class grid -> dense heatmap
# ---------------------------------------------------------------------


def grid_logits_to_heatmap(logits: np.ndarray, cell: int = DEFAULT_CELL) -> np.ndarray:
    """Decode detector logits ``(Hc, Wc, cell*cell+1)`` to a dense heatmap.

    Applies softmax over the 65 classes, drops the dustbin channel (index
    ``cell*cell``), and scatters the remaining ``cell*cell`` within-cell channels
    back to full resolution via a ``depth_to_space``-style reshape. Channel index
    ``k`` corresponds to within-cell position ``(row=k // cell, col=k % cell)``,
    matching :func:`keypoints_to_grid_labels`.

    Args:
        logits: ``(Hc, Wc, cell*cell+1)`` raw detector logits for one image.
        cell: Detector cell size (8 -> 65 classes).

    Returns:
        ``(Hc*cell, Wc*cell)`` ``float32`` keypoint-probability heatmap in
        ``[0, 1]``.
    """
    logits = np.asarray(logits, dtype=np.float32)
    Hc, Wc, ch = logits.shape
    expected = cell * cell + 1
    if ch != expected:
        raise ValueError(
            f"grid_logits_to_heatmap expects last dim {expected} for cell={cell}, "
            f"got {ch}."
        )

    # Softmax over the 65 classes (numerically stable).
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / (exp.sum(axis=-1, keepdims=True) + 1e-12)

    # Drop the dustbin channel -> (Hc, Wc, cell*cell).
    no_dust = probs[..., :-1]

    # depth_to_space: (Hc, Wc, cell*cell) -> (Hc, cell, Wc, cell) -> (Hc*cell, Wc*cell)
    # channel k -> (row=k//cell, col=k%cell).
    grid = no_dust.reshape(Hc, Wc, cell, cell)          # (Hc, Wc, row, col)
    grid = grid.transpose(0, 2, 1, 3)                   # (Hc, row, Wc, col)
    heatmap = grid.reshape(Hc * cell, Wc * cell)
    return heatmap.astype(np.float32)


# ---------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------


def simple_nms(
    heatmap: np.ndarray,
    radius: int,
    threshold: float,
) -> np.ndarray:
    """Greedy local-maximum NMS on a dense heatmap.

    A pixel survives if (a) its value is ``> threshold`` and (b) it equals the
    maximum over the ``(2*radius+1) x (2*radius+1)`` window centered on it (i.e.
    it is a strict-or-tied local max). Implemented with a max-pool comparison
    (``tf.nn.max_pool2d``), which is the standard SuperPoint NMS.

    Args:
        heatmap: ``(H, W)`` non-negative score map.
        radius: NMS radius in pixels; window size is ``2*radius+1``.
        threshold: Minimum score for a pixel to be considered a keypoint.

    Returns:
        ``(N, 2)`` ``float32`` array of ``(x, y)`` keypoint pixel coordinates
        (possibly empty).
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    H, W = hm.shape
    ksize = 2 * radius + 1

    x = tf.constant(hm[np.newaxis, ..., np.newaxis])  # (1, H, W, 1)
    pooled = tf.nn.max_pool2d(x, ksize=ksize, strides=1, padding="SAME")
    pooled = pooled[0, ..., 0].numpy()

    is_max = (hm == pooled) & (hm > threshold)
    ys, xs = np.nonzero(is_max)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)


# ---------------------------------------------------------------------
# ONE ADAPTATION ROUND
# ---------------------------------------------------------------------


def homographic_adaptation_round(
    model: keras.Model,
    image: np.ndarray,
    n_homographies: int,
    cell: int = DEFAULT_CELL,
    nms_radius: int = 4,
    detection_threshold: float = 0.015,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Self-label one image via Homographic Adaptation.

    Aggregates detector heatmaps over the identity plus ``n_homographies`` random
    homographies (each warped, inferred, and unwarped back to the original
    frame), then NMS + thresholds the average into discrete pseudo keypoints.

    Args:
        model: A built SuperPoint model returning ``{"keypoints", ...}``.
        image: ``(H, W, C)`` ``float32`` image in ``[0, 1]``.
        n_homographies: Number of RANDOM homographies (identity is always added
            on top, so the aggregate averages ``n_homographies + 1`` heatmaps).
        cell: Detector cell size.
        nms_radius: NMS radius in pixels.
        detection_threshold: Minimum aggregated score for a keypoint.
        rng: Optional NumPy generator for deterministic homography sampling.

    Returns:
        ``(keypoints (N, 2) float32 (x, y), aggregate_heatmap (H, W) float32)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = np.asarray(image, dtype=np.float32)
    H, W = img.shape[0], img.shape[1]

    # Identity first, then n random homographies.
    homographies = [np.eye(3, dtype=np.float32)]
    for _ in range(n_homographies):
        homographies.append(sample_homography((H, W), rng=rng))

    aggregate = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for h_mat in homographies:
        # Warp the image by H (forward).
        warped = warp_image(img, h_mat)  # (H, W, C)
        warped_batch = warped[tf.newaxis, ...]

        # Eager inference -> detector logits (avoids XLA, see D-004).
        outputs = model(warped_batch, training=False)
        logits = np.asarray(outputs["keypoints"][0])  # (Hc, Wc, 65)

        # Decode to a dense heatmap in the WARPED frame.
        heatmap = grid_logits_to_heatmap(logits, cell=cell)  # (H, W)

        # Unwarp back to the ORIGINAL frame: undo the forward H by warping the
        # heatmap with inv(H). warp_image inverts the passed matrix internally,
        # so passing inv(H) applies H^{-1} to the content. A unit "validity" map
        # tracks which original pixels actually received a sample (so border
        # regions that fall outside the warped frame don't bias the average).
        inv_h = np.linalg.inv(h_mat.astype(np.float64)).astype(np.float32)
        unwarped = warp_image(heatmap[..., np.newaxis], inv_h)  # (H, W, 1)
        unwarped = np.asarray(unwarped)[..., 0]

        valid = warp_image(
            np.ones((H, W, 1), dtype=np.float32), inv_h, interpolation="nearest"
        )
        valid = np.asarray(valid)[..., 0]

        aggregate += unwarped
        count += valid

    # Average over the number of contributing runs per pixel.
    aggregate = aggregate / np.maximum(count, 1.0)

    keypoints = simple_nms(aggregate, radius=nms_radius, threshold=detection_threshold)
    return keypoints, aggregate.astype(np.float32)


# ---------------------------------------------------------------------
# IMAGE SOURCES
# ---------------------------------------------------------------------


def _load_real_image(path: str, input_size: int, channels: int) -> np.ndarray:
    """Read + decode + grayscale + resize one real image to ``(H, W, C)`` in [0,1]."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if channels == 1:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, (input_size, input_size), method="bilinear")
    return img.numpy().astype(np.float32)


def select_weighted_image_paths(
    image_dirs: List[str],
    weights: Optional[List[float]],
    num_images: Optional[int],
    seed: int,
    collect_fn=collect_image_paths,
) -> List[Tuple[str, str]]:
    """Assemble a per-dataset weighted HA worklist of ``(dataset_name, path)``.

    # DECISION plan_2026-06-18_1cca4fc1/D-002: collect each directory's pool
    # SEPARATELY (one ``collect_fn([d])`` call per dir, UNCAPPED) and draw
    # per-dir integer quotas from normalized weights. Do NOT flat-concat all
    # dirs and cap the merged list (``collect_image_paths([d1, d2], max_files)``):
    # COCO's ~118K images would draw ~99% of any cap and drown DIV2K's ~800.
    # The pure-Python quota draw — not ``collect_image_paths`` — owns per-dir
    # counts. See decisions.md D-002.

    Args:
        image_dirs: Real-image directories, one pool collected per dir.
        weights: Optional per-dir sampling weights (aligned to ``image_dirs``).
            ``None`` -> equal weight over the NON-empty dirs. Empty dirs are
            dropped and the remaining weights renormalized to sum 1.
        num_images: Total worklist size. ``None`` -> use every collected path
            exactly once (no quota cap, no wraparound), weight-interleaved.
        seed: Base seed; each kept dir is permuted with
            ``np.random.default_rng([seed, dir_index])`` for a reproducible draw.
        collect_fn: Path-collection callable (injectable for testing); must
            accept ``(dirs, shuffle_seed=..., sort=...)`` and return a path list.

    Returns:
        A list of ``(dataset_name, path)`` pairs (``dataset_name = Path(dir).name``).
        Exactly ``num_images`` long (with wraparound for undersized dirs) when
        ``num_images`` is set; otherwise every collected path once. ``[]`` when
        no images are found.
    """
    # Collect + permute each dir's OWN pool; remember the original index so the
    # per-dir RNG and any supplied weights stay aligned to the kept dirs.
    kept_names: List[str] = []
    kept_pools: List[List[str]] = []
    kept_indices: List[int] = []
    for dir_index, d in enumerate(image_dirs):
        pool = collect_fn([d], shuffle_seed=seed, sort=True)
        if not pool:
            logger.warning(f"No images found under {d!r}; dropping from worklist")
            continue
        rng = np.random.default_rng([seed, dir_index])
        permuted = [pool[i] for i in rng.permutation(len(pool))]
        kept_names.append(Path(d).name)
        kept_pools.append(permuted)
        kept_indices.append(dir_index)

    k = len(kept_pools)
    if k == 0:
        return []

    # Weights over the kept (non-empty) dirs.
    if weights is None:
        kept_weights = [1.0 / k] * k
    else:
        sliced = [max(0.0, float(weights[i])) for i in kept_indices]
        total = sum(sliced)
        if total <= 0:
            kept_weights = [1.0 / k] * k
        else:
            kept_weights = [w / total for w in sliced]

    # num_images is None -> every path once, weighted round-robin interleave.
    if num_images is None:
        return _interleave_by_weight(
            kept_names, [list(p) for p in kept_pools], kept_weights
        )

    # Integer quotas via largest-fractional-remainder so they sum EXACTLY to
    # num_images (deterministic; ties broken by dir index).
    raw = [w * num_images for w in kept_weights]
    quotas = [int(np.floor(r)) for r in raw]
    leftover = num_images - sum(quotas)
    if leftover > 0:
        remainders = sorted(
            range(k), key=lambda i: (-(raw[i] - np.floor(raw[i])), i)
        )
        for i in remainders[:leftover]:
            quotas[i] += 1

    # Per-dir selection with wraparound when quota_i > len(pool).
    per_dir: List[List[str]] = []
    for pool, quota in zip(kept_pools, quotas):
        n = len(pool)
        per_dir.append([pool[j % n] for j in range(quota)])

    return _interleave_by_weight(kept_names, per_dir, kept_weights)


def _interleave_by_weight(
    names: List[str],
    selections: List[List[str]],
    weights: List[float],
) -> List[Tuple[str, str]]:
    """Deterministic weighted round-robin interleave of per-dir selections.

    Emits one path at a time from the dir whose served fraction most lags its
    target ``weight`` (ties broken by dir index), preserving each dir's internal
    order. The result contains every supplied path exactly once.
    """
    cursors = [0] * len(selections)
    served = [0] * len(selections)
    total = sum(len(s) for s in selections)
    out: List[Tuple[str, str]] = []
    for _ in range(total):
        best = None
        best_deficit = None
        for i, sel in enumerate(selections):
            if cursors[i] >= len(sel):
                continue
            # Larger deficit (target share minus served share) wins.
            deficit = weights[i] - (served[i] / total)
            if best is None or deficit > best_deficit:
                best = i
                best_deficit = deficit
        out.append((names[best], selections[best][cursors[best]]))
        cursors[best] += 1
        served[best] += 1
    return out


def iter_images(config: HomographicAdaptationConfig):
    """Yield ``(image_id, dataset_name, source_path, image (H, W, C) float32)``.

    Smoke -> deterministic tiny SYNTHETIC images (never touches real COCO);
    ``dataset_name="synthetic"`` and ``source_path=None`` (no file on disk yet;
    ``run_adaptation`` persists it and fills in the path).
    Real  -> images drawn per-dataset from ``config.image_dirs`` by weight;
    ``dataset_name`` is the dir name and ``source_path`` the absolute resolved
    path of the on-disk file.
    """
    if config.smoke:
        rng = np.random.default_rng(config.seed)
        for i in range(config.num_images):
            img, _ = generate_synthetic_sample(
                config.input_size, config.input_size, rng
            )
            yield f"synthetic_{i:04d}", "synthetic", None, img.astype(np.float32)
        return

    # Real path: weighted per-dataset draw across config.image_dirs.
    selected = select_weighted_image_paths(
        config.image_dirs,
        config.dataset_weights,
        config.num_images,
        config.seed,
    )
    if not selected:
        raise FileNotFoundError(
            f"No images found under image_dirs={config.image_dirs!r}. "
            f"Point --image-dirs at directories of real images (e.g. COCO "
            f"train2017 and DIV2K train)."
        )
    counts = collections.Counter(name for name, _ in selected)
    logger.info(f"HA worklist: {dict(counts)} (total {len(selected)})")
    for ds_name, path in selected:
        image_id = Path(path).stem
        source_path = str(Path(path).resolve())
        yield (
            image_id,
            ds_name,
            source_path,
            _load_real_image(path, config.input_size, config.channels),
        )


# ---------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------


def _build_model(config: HomographicAdaptationConfig) -> keras.Model:
    """Build the SuperPoint model and (in non-smoke) load checkpoint weights."""
    input_shape = (config.input_size, config.input_size, config.channels)
    model = create_superpoint(config.variant, input_shape=input_shape)
    model.build((None, *input_shape))

    if config.checkpoint_path is not None:
        logger.info(f"Loading weights from checkpoint: {config.checkpoint_path}")
        load_weights_from_checkpoint(model, config.checkpoint_path)
    else:
        logger.warning(
            "No checkpoint_path given -> using a FRESH untrained model. "
            "Pseudo-labels will be meaningless (smoke-only mode)."
        )
    return model


def _build_manifest_entry(
    image_id: str,
    dataset_name: str,
    source_path: str,
    npz_rel: str,
    num_keypoints: int,
) -> dict:
    """Build one self-describing manifest entry for a pseudo-labeled image.

    # DECISION plan_2026-06-18_8ecab001/D-001: entries record ``source_path``
    # (absolute) + ``dataset_name`` so stage-4 can RELOAD the source image
    # directly. Do NOT drop these keys back to the bare ``image_id`` schema:
    # stage-4 would then have to stem-match against ~118K COCO + DIV2K files
    # (fragile, collision-prone). See decisions.md D-001.

    Args:
        image_id: Stable image identifier (file stem or ``synthetic_NNNN``).
        dataset_name: Source dataset name (dir name, or ``"synthetic"``).
        source_path: Absolute path to the source image on disk.
        npz_rel: Path to the ``.npz`` label, relative to the experiment dir.
        num_keypoints: Number of pseudo keypoints in this image.

    Returns:
        The per-entry manifest dict.
    """
    return {
        "image_id": image_id,
        "dataset_name": dataset_name,
        "source_path": source_path,
        "npz": npz_rel,
        "num_keypoints": num_keypoints,
    }


def run_adaptation(config: HomographicAdaptationConfig) -> Path:
    """Run one Homographic Adaptation self-labeling pass over the image corpus.

    Writes one ``<image_id>.npz`` per image (containing ``keypoints`` and the
    65-class ``grid_label``) plus a ``manifest.json`` index under
    ``results/<experiment_name>/pseudo_labels/``.

    Returns:
        The output directory ``Path``.
    """
    logger.info(f"Starting Homographic Adaptation: {config.experiment_name}")

    output_dir = Path(config.output_dir) / config.experiment_name
    labels_dir = output_dir / "pseudo_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    save_config_json(config, str(output_dir), "config.json")

    model = _build_model(config)

    rng = np.random.default_rng(config.seed)
    manifest: List[dict] = []

    for idx, (image_id, dataset_name, source_path, image) in enumerate(
        iter_images(config)
    ):
        H, W = image.shape[0], image.shape[1]

        # Smoke synthetic images have no file on disk; persist each one as a
        # grayscale PNG so the smoke pseudo-label set is self-contained and
        # reloadable end-to-end (stage-4 reads source_path). Real images
        # already exist -> just record their absolute resolved path.
        if source_path is None:
            images_dir.mkdir(parents=True, exist_ok=True)
            png_path = images_dir / f"{image_id}.png"
            uint8_hwc = tf.cast(
                tf.clip_by_value(tf.convert_to_tensor(image), 0.0, 1.0) * 255.0,
                tf.uint8,
            )
            tf.io.write_file(str(png_path), tf.io.encode_png(uint8_hwc))
            source_path = str(png_path.resolve())

        keypoints, _aggregate = homographic_adaptation_round(
            model,
            image,
            n_homographies=config.n_homographies,
            cell=config.cell,
            nms_radius=config.nms_radius,
            detection_threshold=config.detection_threshold,
            rng=rng,
        )

        grid_label = keypoints_to_grid_labels(keypoints, H, W, cell=config.cell)

        npz_path = labels_dir / f"{image_id}.npz"
        np.savez_compressed(
            npz_path,
            keypoints=keypoints.astype(np.float32),
            grid_label=grid_label.astype(np.int32),
        )

        if not np.all(np.isfinite(keypoints)):
            raise ValueError(f"Non-finite keypoints produced for image {image_id}")

        manifest.append(
            _build_manifest_entry(
                image_id=image_id,
                dataset_name=dataset_name,
                source_path=source_path,
                npz_rel=str(npz_path.relative_to(output_dir)),
                num_keypoints=int(keypoints.shape[0]),
            )
        )
        logger.info(
            f"[{idx + 1}/{config.num_images}] {image_id}: "
            f"{keypoints.shape[0]} pseudo keypoints -> {npz_path.name}"
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "experiment_name": config.experiment_name,
                "num_images": len(manifest),
                "input_size": config.input_size,
                "cell": config.cell,
                "n_homographies": config.n_homographies,
                "entries": manifest,
            },
            f,
            indent=2,
        )

    total_kp = sum(e["num_keypoints"] for e in manifest)
    logger.info(
        f"Adaptation complete: {len(manifest)} images, {total_kp} total pseudo "
        f"keypoints. Labels written under: {labels_dir}"
    )
    return output_dir


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Homographic Adaptation self-labeling (SuperPoint stage 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained .keras checkpoint (MagicPoint / "
                             "SuperPoint). Omit in --smoke (fresh model).")
    parser.add_argument("--variant", choices=["tiny", "base", "large"],
                        default="tiny")
    parser.add_argument(
        "--image-dirs", type=str, nargs="+",
        default=[
            "/media/arxwn/data0_4tb/datasets/COCO/train2017",
            "/media/arxwn/data0_4tb/datasets/div2k/train",
        ],
        help="One or more real-image directories sourced simultaneously.",
    )
    parser.add_argument(
        "--dataset-weights", type=float, nargs="+", default=None,
        help="Per-dataset sampling weights aligned to --image-dirs "
             "(normalized to sum 1). Omit for a balanced split.",
    )
    parser.add_argument("--input-size", type=int, default=240)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--n-homographies", type=int, default=100)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--detection-threshold", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force tiny config (fresh model, 3 tiny SYNTHETIC images, 4 "
        "homographies) for a fast self-labeling smoke check. NEVER reads COCO.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_gpu(gpu_id=args.gpu)
    set_seeds(args.seed)

    config = HomographicAdaptationConfig(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        image_dirs=args.image_dirs,
        dataset_weights=args.dataset_weights,
        input_size=args.input_size,
        num_images=args.num_images,
        n_homographies=args.n_homographies,
        nms_radius=args.nms_radius,
        detection_threshold=args.detection_threshold,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        smoke=args.smoke,
    )

    logger.info(
        f"Config: variant={config.variant}, input={config.input_size}, "
        f"num_images={config.num_images}, n_homographies={config.n_homographies}, "
        f"nms_radius={config.nms_radius}, smoke={config.smoke}"
    )

    try:
        run_adaptation(config)
        logger.info("Homographic Adaptation completed successfully!")
    except Exception as e:
        logger.error(f"Homographic Adaptation failed: {e}")
        raise


if __name__ == "__main__":
    main()
