"""Qualitative keypoint / matching figures emitted during training.

Headless (Agg) matplotlib only, no seaborn/pandas/tf at module scope beyond
keras + numpy. Every callback is fail-soft-but-LOUD: plotting failures log a
WARNING with traceback and never abort the run. Directories are created at
construction AND at save time.

Contents:

- :func:`plot_superpoint_panel` — grayscale image + detector heatmap +
  NMS keypoint overlay for one image.
- :func:`plot_match_panel` — two grayscale views with top-K predicted
  correspondences drawn across them, color-coded by matching score.
- :class:`SuperPointVizCallback` — epoch-end detector/descriptor figures for
  the MagicPoint and SuperPoint trainers. Runs the backbone on a FIXED viz
  split (a few batches captured before ``fit``), decodes with
  :mod:`dl_techniques.utils.keypoint_extraction`, and writes
  ``viz/epoch_{N:03d}_detection.png`` plus a descriptor PCA projection.
- :class:`LightGlueVizCallback` — epoch-end match panels for the LightGlue
  trainer. Runs the full ``LightGlueTrainingModel`` pipeline on a fixed pair
  batch and writes ``viz/epoch_{N:03d}_matches.png``.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import: headless-safe

import keras  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dl_techniques.utils.logger import logger  # noqa: E402

PathLike = Union[str, Path]
DPI = 120


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Squeeze a ``(H, W, C)`` viz image to ``(H, W)`` float in [0, 1]."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., 0] if img.shape[-1] == 1 else img.mean(axis=-1)
    return np.clip(img, 0.0, 1.0)


def plot_superpoint_panel(
    image: np.ndarray,
    heatmap: np.ndarray,
    keypoints: np.ndarray,
    out_path: PathLike,
    title: str = "",
) -> None:
    """One 3-panel figure: image | heatmap | keypoint overlay.

    :param image: ``(H, W[, C])`` in [0, 1].
    :param heatmap: ``(H, W)`` detector probability map.
    :param keypoints: ``(K, 2)`` xy pixels (may be empty).
    :param out_path: PNG path (parent created).
    :param title: Figure suptitle.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gray = _to_gray(image)
    hm = np.asarray(heatmap, dtype=np.float32)
    kps = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("image")
    axes[0].axis("off")
    im = axes[1].imshow(hm, cmap="hot", vmin=0.0, vmax=float(hm.max() + 1e-12))
    axes[1].set_title(f"heatmap (max {float(hm.max()):.3f})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    if len(kps):
        axes[2].scatter(kps[:, 0], kps[:, 1], s=12, c="lime", marker="x",
                        linewidths=0.8)
    axes[2].set_title(f"keypoints: {len(kps)}")
    axes[2].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_descriptor_pca(
    descriptors: np.ndarray,
    keypoints: np.ndarray,
    image_shape: tuple,
    out_path: PathLike,
    title: str = "",
) -> None:
    """PCA-3 projection of a dense descriptor map as an RGB image.

    A cheap diversity check: a collapsed head renders flat gray, a trained
    head shows spatial structure. Keypoints are overlaid when given.

    :param descriptors: ``(H, W, D)`` float descriptors.
    :param keypoints: ``(K, 2)`` xy pixels (may be empty).
    :param image_shape: ``(H, W)`` for the axes (informational only).
    :param out_path: PNG path (parent created).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    desc = np.asarray(descriptors, dtype=np.float64)
    h, w, d = desc.shape
    flat = desc.reshape(-1, d)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # 3 principal components via thin SVD on the (pixels x D) matrix.
    try:
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        proj = flat @ vt[:3].T  # (H*W, 3)
    except np.linalg.LinAlgError:
        proj = flat[:, :3] if d >= 3 else np.pad(flat, ((0, 0), (0, 3 - d)))
    for c in range(3):
        col = proj[:, c]
        lo, hi = np.percentile(col, 1), np.percentile(col, 99)
        proj[:, c] = (col - lo) / max(hi - lo, 1e-12)
    rgb = np.clip(proj, 0.0, 1.0).reshape(h, w, 3)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    kps = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if len(kps):
        ax.scatter(kps[:, 0], kps[:, 1], s=10, c="white", marker="x",
                   linewidths=0.8)
    ax.set_title(title or f"descriptor PCA (map {h}x{w})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_match_panel(
    image0: np.ndarray,
    image1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches0: np.ndarray,
    scores0: np.ndarray,
    out_path: PathLike,
    max_draw: int = 50,
    title: str = "",
) -> None:
    """Side-by-side views with predicted correspondences drawn across.

    :param image0/image1: ``(H, W[, C])`` grayscale images in [0, 1].
    :param kpts0/kpts1: ``(M, 2)`` / ``(N, 2)`` xy pixels.
    :param matches0: ``(M,)`` int partner index in image 1 or -1.
    :param scores0: ``(M,)`` float matching scores.
    :param out_path: PNG path (parent created).
    :param max_draw: Draw at most this many matches (top by score).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g0, g1 = _to_gray(image0), _to_gray(image1)
    h = max(g0.shape[0], g1.shape[0])
    canvas = np.zeros((h, g0.shape[1] + g1.shape[1]), dtype=np.float32)
    canvas[:g0.shape[0], :g0.shape[1]] = g0
    canvas[:g1.shape[0], g0.shape[1]:] = g1

    m0 = np.asarray(matches0, dtype=np.int64).reshape(-1)
    s0 = np.asarray(scores0, dtype=np.float32).reshape(-1)
    k0 = np.asarray(kpts0, dtype=np.float32).reshape(-1, 2)
    k1 = np.asarray(kpts1, dtype=np.float32).reshape(-1, 2)
    valid = np.nonzero(m0 >= 0)[0]
    order = valid[np.argsort(-s0[valid])][:max_draw] if len(valid) else []

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas, cmap="gray", vmin=0.0, vmax=1.0)
    off = g0.shape[1]
    cmap = plt.get_cmap("jet")
    for idx in order:
        j = int(m0[idx])
        if j < 0 or j >= len(k1):
            continue
        color = cmap(float(np.clip(s0[idx], 0.0, 1.0)))
        x0, y0 = float(k0[idx, 0]), float(k0[idx, 1])
        x1, y1 = float(k1[j, 0]) + off, float(k1[j, 1])
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=0.8, alpha=0.9)
        ax.scatter([x0], [y0], s=10, c=[color], marker="o")
        ax.scatter([x1], [y1], s=10, c=[color], marker="o")
    ax.set_title(title or f"matches: {len(valid)} (top {len(order)} drawn)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


class SuperPointVizCallback(keras.callbacks.Callback):
    """Epoch-end SuperPoint figures on a fixed viz split.

    Runs the backbone (or joint wrapper, via its ``superpoint`` attribute)
    on ``viz_images`` (``(V, H, W, C)`` float32) and writes, per epoch on
    cadence, ``viz/epoch_{N:03d}_detection.png`` (first image: image |
    heatmap | keypoints) and ``viz/epoch_{N:03d}_descriptors.png`` (PCA).
    With ``viz_warped`` also writes the warped companion panel, which is how
    a joint-training run checks the pair the descriptor loss sees.
    With ``viz_homographies`` additionally writes
    ``viz/epoch_{N:03d}_correspondence.png`` (pair 0: clean detections
    reprojected by ``H`` vs warped detections, greedy 3px matches) and appends
    one line per epoch to ``viz/repeatability.jsonl`` (mean over ALL stored
    pairs, not just the plotted one) -- a plottable repeatability curve.

    The callback never raises: inference/plotting failures log WARNING.

    :param viz_dir: Directory for the PNGs (created).
    :param viz_images: Fixed ``(V, H, W, C)`` images.
    :param viz_warped: Optional fixed warped companions ``(V, H, W, C)``.
    :param viz_homographies: Optional ``(V, 3, 3)`` forward homographies
        (image -> warped). When given WITH ``viz_warped``, an extra
        ``viz/epoch_{N:03d}_correspondence.png`` overlays the clean-view
        detections reprojected by ``H`` against the warped-view detections
        (greedy 3px matches, repeatability in the title) -- the direct "do the
        keypoints warp" check.
    :param every_n: Figure cadence in epochs (1 = every epoch).
    :param max_keypoints: Decode slot count.
    :param threshold: Heatmap probability threshold.
    :param nms_radius: NMS radius in pixels.
    :param border: Edge margin in pixels.
    """

    def __init__(
        self,
        viz_dir: PathLike,
        viz_images: np.ndarray,
        viz_warped: Optional[np.ndarray] = None,
        viz_homographies: Optional[np.ndarray] = None,
        every_n: int = 5,
        max_keypoints: int = 256,
        threshold: float = 0.005,
        nms_radius: int = 4,
        border: int = 4,
    ) -> None:
        super().__init__()
        os.makedirs(viz_dir, exist_ok=True)
        self.viz_dir = str(viz_dir)
        self.viz_images = np.asarray(viz_images, dtype=np.float32)
        self.viz_warped = (
            None if viz_warped is None else np.asarray(viz_warped, dtype=np.float32)
        )
        self.viz_homographies = (
            None if viz_homographies is None
            else np.asarray(viz_homographies, dtype=np.float32)
        )
        self.every_n = max(1, int(every_n))
        self.max_keypoints = int(max_keypoints)
        self.threshold = float(threshold)
        self.nms_radius = int(nms_radius)
        self.border = int(border)

    def _backbone(self) -> Any:
        model = self.model
        return getattr(model, "superpoint", model)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if (epoch + 1) % self.every_n != 0 and epoch != 0:
            return
        os.makedirs(self.viz_dir, exist_ok=True)
        try:
            from dl_techniques.utils.keypoint_extraction import decode_superpoint
            backbone = self._backbone()
            outs = backbone(self.viz_images, training=False)
            heat_fn = None
            try:
                from dl_techniques.utils.keypoint_extraction import (
                    superpoint_heatmap,
                )
                heat_fn = superpoint_heatmap
            except ImportError:
                heat_fn = None
            dec = decode_superpoint(
                {"keypoints": np.asarray(outs["keypoints"]),
                 "descriptors": np.asarray(outs["descriptors"])},
                max_keypoints=self.max_keypoints,
                threshold=self.threshold,
                nms_radius=self.nms_radius,
                border=self.border,
            )
            kp_all = np.asarray(dec["keypoints"])
            mask_all = np.asarray(dec["mask"]).astype(bool)
            kps = kp_all[0][mask_all[0]]
            heat = (
                np.asarray(heat_fn(np.asarray(outs["keypoints"])))[0]
                if heat_fn is not None else np.zeros(_to_gray(self.viz_images[0]).shape, dtype=np.float32)
            )
            plot_superpoint_panel(
                self.viz_images[0], heat, kps,
                Path(self.viz_dir) / f"epoch_{epoch + 1:03d}_detection.png",
                title=f"epoch {epoch + 1} detection",
            )
            plot_descriptor_pca(
                np.asarray(outs["descriptors"])[0], kps,
                _to_gray(self.viz_images[0]).shape,
                Path(self.viz_dir) / f"epoch_{epoch + 1:03d}_descriptors.png",
                title=f"epoch {epoch + 1} descriptor PCA",
            )
            if self.viz_warped is not None:
                outs_w = backbone(self.viz_warped, training=False)
                dec_w = decode_superpoint(
                    {"keypoints": np.asarray(outs_w["keypoints"]),
                     "descriptors": np.asarray(outs_w["descriptors"])},
                    max_keypoints=self.max_keypoints,
                    threshold=self.threshold,
                    nms_radius=self.nms_radius,
                    border=self.border,
                )
                kpw_all = np.asarray(dec_w["keypoints"])
                maskw_all = np.asarray(dec_w["mask"]).astype(bool)
                kps_w = kpw_all[0][maskw_all[0]]
                heat_w = (
                    np.asarray(heat_fn(np.asarray(outs_w["keypoints"])))[0]
                    if heat_fn is not None else np.zeros_like(heat)
                )
                plot_superpoint_panel(
                    self.viz_warped[0], heat_w, kps_w,
                    Path(self.viz_dir) / f"epoch_{epoch + 1:03d}_detection_warped.png",
                    title=f"epoch {epoch + 1} detection (warped)",
                )
                if self.viz_homographies is not None:
                    from dl_techniques.utils.homography import warp_points
                    n_pairs = min(len(self.viz_images), len(self.viz_warped),
                                  len(self.viz_homographies))
                    reps, reproj_counts = [], []
                    first = None
                    for i in range(n_pairs):
                        ki = kp_all[i][mask_all[i]]
                        wi = kpw_all[i][maskw_all[i]]
                        h_mat = np.asarray(self.viz_homographies[i], dtype=np.float32)
                        reproj = warp_points(ki, h_mat)
                        finite = np.all(np.isfinite(reproj), axis=1)
                        h_img, w_img = _to_gray(self.viz_warped[i]).shape
                        inside = (
                            finite & (reproj[:, 0] >= 0) & (reproj[:, 0] < w_img)
                            & (reproj[:, 1] >= 0) & (reproj[:, 1] < h_img)
                        )
                        matches, scores, rep = reprojection_matches(
                            reproj, inside, wi, thresh=3.0)
                        reps.append(rep)
                        reproj_counts.append(int(inside.sum()))
                        if i == 0:
                            first = (ki, wi, matches, scores, rep, int(inside.sum()))
                    import json as _json
                    with open(Path(self.viz_dir) / "repeatability.jsonl", "a") as f:
                        f.write(_json.dumps({
                            "epoch": epoch + 1,
                            "mean_repeatability": float(np.mean(reps)) if reps else 0.0,
                            "per_pair": [float(r) for r in reps],
                            "reprojectable": reproj_counts,
                        }) + "\n")
                    if first is not None:
                        ki, wi, matches, scores, rep, n_reproj = first
                        plot_match_panel(
                            self.viz_images[0], self.viz_warped[0], ki, wi,
                            matches, scores,
                            Path(self.viz_dir) / f"epoch_{epoch + 1:03d}_correspondence.png",
                            title=(f"epoch {epoch + 1} reprojection: "
                                   f"repeatability {rep:.2f} (pair 0, "
                                   f"mean {float(np.mean(reps)):.2f} over {len(reps)})"),
                        )
                    logger.info(
                        f"SuperPointViz epoch {epoch + 1}: mean repeatability "
                        f"{float(np.mean(reps)):.3f} over {len(reps)} pairs")
        except Exception as e:
            logger.warning(
                f"SuperPointVizCallback: failed at epoch {epoch + 1}: {e}",
                exc_info=True,
            )


def reprojection_matches(
    src_warped: np.ndarray,
    src_valid: np.ndarray,
    dst: np.ndarray,
    thresh: float = 3.0,
) -> tuple:
    """Greedy nearest-neighbor matches of reprojected points to detections.

    For each VALID reprojected source point, the nearest destination detection
    within ``thresh`` pixels is its match (greedy, non-mutual -- the
    repeatability convention, not the matcher convention). This answers "do the
    keypoints warp": a detector covariant with the homography scores near 1.

    :param src_warped: ``(M, 2)`` source points mapped into the target frame.
    :param src_valid: ``(M,)`` bool (warped-out / horizon-invalid excluded).
    :param dst: ``(N, 2)`` target-frame detections (may be empty).
    :param thresh: Maximum match distance in pixels (strict ``<``).
    :return: ``(matches0 (M,) int with -1, scores0 (M,) float in [0, 1],
        repeatability float)``; repeatability is matched / valid, 0 when no
        valid source point. Scores are ``1 - dist / thresh``.
    """
    src = np.asarray(src_warped, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(src_valid, dtype=bool).reshape(-1)
    pts = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    matches = -np.ones(len(src), dtype=np.int64)
    scores = np.zeros(len(src), dtype=np.float32)
    if not valid.any() or len(pts) == 0:
        return matches, scores, 0.0
    dist = np.sqrt(((src[valid, None, :] - pts[None, :, :]) ** 2).sum(-1))
    nn = np.argmin(dist, axis=1)
    best = dist[np.arange(dist.shape[0]), nn]
    hit = best < thresh
    idx = np.nonzero(valid)[0][hit]
    matches[idx] = nn[hit]
    scores[idx] = 1.0 - best[hit] / thresh
    return matches, scores, float(hit.mean())


class LightGlueVizCallback(keras.callbacks.Callback):
    """Epoch-end LightGlue match panels on a fixed pair batch.

    Runs the full ``LightGlueTrainingModel`` (frozen SuperPoint + LightGlue)
    on ``viz_batch`` (the dict of :func:`make_pair_dataset`, first ``max_pairs``
    pairs) and writes ``viz/epoch_{N:03d}_matches.png`` (first pair) plus a
    small per-pair match-count summary to the log. Never raises.

    :param viz_dir: Directory for the PNGs (created).
    :param viz_batch: Fixed batch dict with ``image0/image1/H0to1/...``.
    :param every_n: Figure cadence in epochs.
    :param max_draw: Matches drawn on the panel (top by score).
    :param max_pairs: Pairs of the batch to summarize (first pair plotted).
    """

    def __init__(
        self,
        viz_dir: PathLike,
        viz_batch: Dict[str, np.ndarray],
        every_n: int = 5,
        max_draw: int = 50,
        max_pairs: int = 4,
    ) -> None:
        super().__init__()
        os.makedirs(viz_dir, exist_ok=True)
        self.viz_dir = str(viz_dir)
        self.viz_batch = {k: np.asarray(v) for k, v in viz_batch.items()}
        self.every_n = max(1, int(every_n))
        self.max_draw = int(max_draw)
        self.max_pairs = int(max_pairs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if (epoch + 1) % self.every_n != 0 and epoch != 0:
            return
        os.makedirs(self.viz_dir, exist_ok=True)
        try:
            batch = {
                k: keras.ops.convert_to_tensor(v[:self.max_pairs])
                for k, v in self.viz_batch.items()
            }
            out = self.model(batch, training=False)
            m0 = np.asarray(out["matches0"])
            s0 = np.asarray(out["matching_scores0"])
            det0, det1, _, _, _, _ = self.model._labelled(batch)
            k0 = np.asarray(det0["keypoints"])
            k1 = np.asarray(det1["keypoints"])
            img0 = np.asarray(batch["image0"])
            img1 = np.asarray(batch["image1"])
            counts: List[int] = []
            for p in range(min(self.max_pairs, m0.shape[0])):
                n = int(np.sum(m0[p] >= 0))
                counts.append(n)
                if p == 0:
                    plot_match_panel(
                        img0[p], img1[p], k0[p], k1[p], m0[p], s0[p],
                        Path(self.viz_dir) / f"epoch_{epoch + 1:03d}_matches.png",
                        max_draw=self.max_draw,
                        title=f"epoch {epoch + 1} matches: {n}",
                    )
            logger.info(f"LightGlueViz epoch {epoch + 1}: match counts {counts}")
        except Exception as e:
            logger.warning(
                f"LightGlueVizCallback: failed at epoch {epoch + 1}: {e}",
                exc_info=True,
            )
