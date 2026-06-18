"""
Synthetic Shapes Renderer (MagicPoint stage-1 data)
===================================================

NumPy/OpenCV-based renderers that draw geometric primitives with EXACTLY-KNOWN
corner / junction keypoints. This is the MagicPoint synthetic pre-training data
used to bootstrap the SuperPoint detector head before Homographic Adaptation.

Each ``draw_*`` generator returns a tuple ``(image, keypoints)`` where:
    - ``image``     : ``np.ndarray`` of shape ``(H, W)``, ``float32`` in ``[0, 1]``.
    - ``keypoints`` : ``np.ndarray`` of shape ``(N, 2)``, ``float32``, columns are
                      ``(x, y)`` in pixel coordinates (x = column, y = row).

The high-level :func:`generate_synthetic_sample` randomly selects a primitive,
renders it on a (possibly noisy) background, applies mild Gaussian noise/blur for
realism and returns ``(image (H, W, 1), keypoints (N, 2))``.

:func:`keypoints_to_grid_labels` is the critical label encoder for the 65-class
detector head: every ``cell x cell`` block (default ``8x8``) is assigned the
flattened within-cell index ``[0..63]`` of a keypoint that falls inside it, or
``64`` (the dustbin class) when the cell is empty. The output
``(H // cell, W // cell)`` ``int32`` map matches the detector head / loss contract
``(B, Hc, Wc, 65)`` logits  <->  ``(B, Hc, Wc)`` integer labels.

Example:
    >>> import numpy as np
    >>> from dl_techniques.datasets.synthetic_shapes import (
    ...     generate_synthetic_sample, keypoints_to_grid_labels)
    >>> rng = np.random.default_rng(0)
    >>> img, kp = generate_synthetic_sample(128, 128, rng)
    >>> labels = keypoints_to_grid_labels(kp, 128, 128, cell=8)
    >>> labels.shape
    (16, 16)
"""

from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

DUSTBIN_CLASS: int = 64
"""Label assigned to a grid cell that contains no keypoint."""

DEFAULT_CELL: int = 8
"""Default detector-head cell size (8x8 -> 64 within-cell positions + dustbin)."""

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


def _blank_image(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Create a (possibly mid-grey, possibly noisy) blank canvas.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        rng: NumPy random generator for reproducibility.

    Returns:
        A ``(H, W)`` ``float32`` array in ``[0, 1]``.
    """
    base = float(rng.uniform(0.0, 0.4))
    img = np.full((H, W), base, dtype=np.float32)
    return img


def _clip01(img: np.ndarray) -> np.ndarray:
    """Clip an image to ``[0, 1]`` and cast to ``float32``."""
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _filter_in_bounds(pts: np.ndarray, H: int, W: int) -> np.ndarray:
    """Keep only ``(x, y)`` points strictly inside the image bounds.

    Args:
        pts: ``(N, 2)`` array of ``(x, y)`` points.
        H: Image height.
        W: Image width.

    Returns:
        A ``(M, 2)`` ``float32`` array (``M <= N``) of in-bounds points.
    """
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    mask = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] <= W - 1)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] <= H - 1)
    )
    return pts[mask].astype(np.float32)


def _rand_color(rng: np.random.Generator, low: float = 0.5, high: float = 1.0) -> float:
    """Sample a random foreground grey intensity in ``[low, high]``."""
    return float(rng.uniform(low, high))


# ------------------------------------------------------------------------------
# shape generators
# ------------------------------------------------------------------------------


def draw_checkerboard(
    H: int, W: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a checkerboard; keypoints are the internal grid corners.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        rng: NumPy random generator.

    Returns:
        ``(image (H, W) float32 in [0,1], keypoints (N, 2) (x, y))``.
    """
    img = _blank_image(H, W, rng)

    rows = int(rng.integers(3, 7))
    cols = int(rng.integers(3, 7))
    # Grid line coordinates (sorted, unique, in-bounds).
    ys = np.unique(np.linspace(0, H - 1, rows + 1).round().astype(np.int32))
    xs = np.unique(np.linspace(0, W - 1, cols + 1).round().astype(np.int32))

    c0 = _rand_color(rng, 0.0, 0.3)
    c1 = _rand_color(rng, 0.6, 1.0)
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            color = c1 if (r + c) % 2 == 0 else c0
            img[ys[r]:ys[r + 1], xs[c]:xs[c + 1]] = color

    # Internal corners only (exclude the outer border lines).
    kps: List[Tuple[float, float]] = []
    for y in ys[1:-1]:
        for x in xs[1:-1]:
            kps.append((float(x), float(y)))
    keypoints = (
        np.asarray(kps, dtype=np.float32)
        if kps
        else np.zeros((0, 2), dtype=np.float32)
    )
    return _clip01(img), _filter_in_bounds(keypoints, H, W)


def draw_lines(
    H: int, W: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Render random line segments; keypoints are the segment endpoints.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        rng: NumPy random generator.

    Returns:
        ``(image (H, W) float32 in [0,1], keypoints (N, 2) (x, y))``.
    """
    img = _blank_image(H, W, rng)
    n = int(rng.integers(1, 5))
    kps: List[Tuple[float, float]] = []
    for _ in range(n):
        x1 = int(rng.integers(0, W))
        y1 = int(rng.integers(0, H))
        x2 = int(rng.integers(0, W))
        y2 = int(rng.integers(0, H))
        color = _rand_color(rng)
        thickness = int(rng.integers(1, 4))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        kps.append((float(x1), float(y1)))
        kps.append((float(x2), float(y2)))
    keypoints = np.asarray(kps, dtype=np.float32)
    return _clip01(img), _filter_in_bounds(keypoints, H, W)


def draw_polygons(
    H: int, W: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a filled convex polygon; keypoints are the vertices.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        rng: NumPy random generator.

    Returns:
        ``(image (H, W) float32 in [0,1], keypoints (N, 2) (x, y))``.
    """
    img = _blank_image(H, W, rng)

    n_verts = int(rng.integers(3, 7))
    cx = float(rng.uniform(0.3, 0.7) * W)
    cy = float(rng.uniform(0.3, 0.7) * H)
    radius = float(rng.uniform(0.15, 0.4) * min(H, W))

    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_verts))
    verts: List[Tuple[float, float]] = []
    for a in angles:
        jitter = rng.uniform(0.7, 1.0)
        x = cx + radius * jitter * np.cos(a)
        y = cy + radius * jitter * np.sin(a)
        verts.append((float(x), float(y)))

    poly = np.asarray(verts, dtype=np.int32).reshape(-1, 1, 2)
    color = _rand_color(rng)
    cv2.fillPoly(img, [poly], color, lineType=cv2.LINE_AA)

    keypoints = np.asarray(verts, dtype=np.float32)
    return _clip01(img), _filter_in_bounds(keypoints, H, W)


def draw_stars(
    H: int, W: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a star/asterisk of segments; keypoints are the tips + center.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        rng: NumPy random generator.

    Returns:
        ``(image (H, W) float32 in [0,1], keypoints (N, 2) (x, y))``.
    """
    img = _blank_image(H, W, rng)

    n_points = int(rng.integers(4, 8))
    cx = float(rng.uniform(0.3, 0.7) * W)
    cy = float(rng.uniform(0.3, 0.7) * H)
    radius = float(rng.uniform(0.2, 0.45) * min(H, W))
    color = _rand_color(rng)
    thickness = int(rng.integers(1, 3))

    kps: List[Tuple[float, float]] = [(cx, cy)]
    for k in range(n_points):
        a = 2.0 * np.pi * k / n_points + float(rng.uniform(0.0, 0.5))
        tx = cx + radius * np.cos(a)
        ty = cy + radius * np.sin(a)
        cv2.line(
            img,
            (int(round(cx)), int(round(cy))),
            (int(round(tx)), int(round(ty))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        kps.append((float(tx), float(ty)))

    keypoints = np.asarray(kps, dtype=np.float32)
    return _clip01(img), _filter_in_bounds(keypoints, H, W)


def draw_cube(
    H: int, W: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a wireframe cube; keypoints are the projected cube corners.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        rng: NumPy random generator.

    Returns:
        ``(image (H, W) float32 in [0,1], keypoints (N, 2) (x, y))``.
    """
    img = _blank_image(H, W, rng)

    s = float(rng.uniform(0.18, 0.32) * min(H, W))  # front-face half-extent proxy
    cx = float(rng.uniform(0.35, 0.55) * W)
    cy = float(rng.uniform(0.45, 0.65) * H)
    dx = float(rng.uniform(0.3, 0.6) * s)  # depth offset x
    dy = -float(rng.uniform(0.3, 0.6) * s)  # depth offset y (up-right look)

    # Front face corners (clockwise from top-left).
    front = np.array(
        [[cx, cy - s], [cx + s, cy - s], [cx + s, cy], [cx, cy]], dtype=np.float32
    )
    back = front + np.array([dx, dy], dtype=np.float32)

    corners = np.concatenate([front, back], axis=0)
    color = _rand_color(rng)
    thickness = int(rng.integers(1, 3))

    def _ln(p: np.ndarray, q: np.ndarray) -> None:
        cv2.line(
            img,
            (int(round(p[0])), int(round(p[1]))),
            (int(round(q[0])), int(round(q[1]))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    # Front face, back face, connecting edges.
    for i in range(4):
        _ln(front[i], front[(i + 1) % 4])
        _ln(back[i], back[(i + 1) % 4])
        _ln(front[i], back[i])

    keypoints = corners.astype(np.float32)
    return _clip01(img), _filter_in_bounds(keypoints, H, W)


def draw_ellipses(
    H: int, W: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Render ellipses (distractors). Ellipses have NO stable corners.

    These contribute background/distractor structure with NO keypoints, teaching
    the detector to ignore smooth curvature.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        rng: NumPy random generator.

    Returns:
        ``(image (H, W) float32 in [0,1], keypoints (0, 2) empty)``.
    """
    img = _blank_image(H, W, rng)
    n = int(rng.integers(1, 4))
    for _ in range(n):
        cx = int(rng.integers(0, W))
        cy = int(rng.integers(0, H))
        ax = int(rng.integers(max(2, W // 16), max(3, W // 4)))
        ay = int(rng.integers(max(2, H // 16), max(3, H // 4)))
        angle = float(rng.uniform(0.0, 360.0))
        color = _rand_color(rng)
        thickness = -1 if rng.uniform() < 0.5 else int(rng.integers(1, 4))
        cv2.ellipse(
            img, (cx, cy), (ax, ay), angle, 0, 360, color, thickness,
            lineType=cv2.LINE_AA,
        )
    # No stable corners -> empty keypoint set (distractor / background).
    return _clip01(img), np.zeros((0, 2), dtype=np.float32)


# Registry of shape generators (name -> callable).
SHAPE_GENERATORS: Dict[
    str, Callable[[int, int, np.random.Generator], Tuple[np.ndarray, np.ndarray]]
] = {
    "checkerboard": draw_checkerboard,
    "lines": draw_lines,
    "polygons": draw_polygons,
    "stars": draw_stars,
    "cube": draw_cube,
    "ellipses": draw_ellipses,
}


# ------------------------------------------------------------------------------
# high-level sample generator
# ------------------------------------------------------------------------------


def generate_synthetic_sample(
    H: int,
    W: int,
    rng: np.random.Generator,
    shape_types: Optional[Sequence[str]] = None,
    noise_std: float = 0.05,
    blur_prob: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render one random synthetic-shapes sample with exact GT keypoints.

    A shape type is selected uniformly at random, rendered on the canvas, then
    mild Gaussian blur (with probability ``blur_prob``) and additive Gaussian
    noise are applied for realism. The result is fully deterministic given
    ``rng`` and the arguments.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        rng: NumPy random generator (drives shape choice, geometry, and noise).
        shape_types: Optional subset of shape names from :data:`SHAPE_GENERATORS`.
            ``None`` uses all registered shapes.
        noise_std: Standard deviation of additive Gaussian noise (in ``[0, 1]``).
        blur_prob: Probability of applying a mild Gaussian blur.

    Returns:
        ``(image (H, W, 1) float32 in [0,1], keypoints (N, 2) float32 (x, y))``.
    """
    if shape_types is None:
        names = list(SHAPE_GENERATORS.keys())
    else:
        names = [s for s in shape_types if s in SHAPE_GENERATORS]
        if not names:
            raise ValueError(
                f"No valid shape_types in {shape_types}; "
                f"available: {sorted(SHAPE_GENERATORS)}"
            )

    name = names[int(rng.integers(0, len(names)))]
    img, keypoints = SHAPE_GENERATORS[name](H, W, rng)

    # Mild Gaussian blur for realism.
    if rng.uniform() < blur_prob:
        ksize = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Additive Gaussian noise.
    if noise_std > 0.0:
        img = img + rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)

    img = _clip01(img)
    img = img[..., np.newaxis]  # (H, W, 1)
    return img, keypoints


# ------------------------------------------------------------------------------
# 65-class grid label encoder
# ------------------------------------------------------------------------------


def keypoints_to_grid_labels(
    keypoints: np.ndarray,
    H: int,
    W: int,
    cell: int = DEFAULT_CELL,
) -> np.ndarray:
    """Encode keypoints into the 65-class detector-head grid label map.

    The image is divided into ``cell x cell`` blocks. For each block, if a
    keypoint falls inside it, the label is the flattened within-cell index
    ``row_in_cell * cell + col_in_cell`` in ``[0, cell*cell - 1]``. Empty cells
    receive the dustbin class ``cell*cell`` (``64`` for ``cell=8``). When several
    keypoints share a cell the one nearest the cell center is chosen
    deterministically.

    This matches the detector head / loss contract:
    ``(B, Hc, Wc, cell*cell + 1)`` logits  <->  ``(B, Hc, Wc)`` integer labels.

    Args:
        keypoints: ``(N, 2)`` array of ``(x, y)`` keypoints. May be empty.
        H: Image height in pixels.
        W: Image width in pixels.
        cell: Cell size (default 8 -> 65 classes). ``H`` and ``W`` are
            floor-divided by ``cell``.

    Returns:
        ``(H // cell, W // cell)`` ``int32`` label map with values in
        ``[0, cell*cell]``.
    """
    Hc = H // cell
    Wc = W // cell
    dustbin = cell * cell
    labels = np.full((Hc, Wc), dustbin, dtype=np.int32)

    if keypoints is None or np.asarray(keypoints).size == 0:
        return labels

    pts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    # Track the best (smallest) center-distance per chosen cell for determinism.
    best_dist = np.full((Hc, Wc), np.inf, dtype=np.float32)
    center = (cell - 1) / 2.0

    for x, y in pts:
        cx = int(np.floor(x / cell))
        cy = int(np.floor(y / cell))
        if cx < 0 or cx >= Wc or cy < 0 or cy >= Hc:
            continue
        col_in = int(np.floor(x)) - cx * cell
        row_in = int(np.floor(y)) - cy * cell
        # Guard against floating edges landing exactly on the next cell border.
        col_in = min(max(col_in, 0), cell - 1)
        row_in = min(max(row_in, 0), cell - 1)

        dist = (col_in - center) ** 2 + (row_in - center) ** 2
        if dist < best_dist[cy, cx]:
            best_dist[cy, cx] = dist
            labels[cy, cx] = row_in * cell + col_in

    return labels


# ------------------------------------------------------------------------------
# tf.data-friendly generator
# ------------------------------------------------------------------------------


def synthetic_shapes_generator(
    H: int,
    W: int,
    seed: int = 0,
    cell: int = DEFAULT_CELL,
    shape_types: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(image (H, W, 1), grid_label (Hc, Wc))`` pairs for tf.data.

    Designed to be wrapped with ``tf.data.Dataset.from_generator``. Deterministic
    given ``seed``. With ``n_samples=None`` the generator is infinite.

    Args:
        H: Image height in pixels (should be divisible by ``cell``).
        W: Image width in pixels (should be divisible by ``cell``).
        seed: Seed for the internal ``np.random.default_rng``.
        cell: Detector-head cell size (default 8).
        shape_types: Optional subset of shape names.
        n_samples: Number of samples to yield; ``None`` for an infinite stream.

    Yields:
        Tuples ``(image (H, W, 1) float32, grid_label (H//cell, W//cell) int32)``.
    """
    rng = np.random.default_rng(seed)
    count = 0
    logger.info(
        f"synthetic_shapes_generator: H={H} W={W} cell={cell} seed={seed} "
        f"n_samples={'inf' if n_samples is None else n_samples}"
    )
    while n_samples is None or count < n_samples:
        img, kps = generate_synthetic_sample(H, W, rng, shape_types=shape_types)
        labels = keypoints_to_grid_labels(kps, H, W, cell=cell)
        yield img, labels
        count += 1
