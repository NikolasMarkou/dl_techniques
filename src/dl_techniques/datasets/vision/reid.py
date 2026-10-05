"""Person re-identification data: synthetic identities and Market1501 reading.

Two image sources feed the same ``(image, identity)`` contract (images
``(128, 64, 3)`` in [0, 1], matching the appearance network input):

- :func:`synthetic_reid_generator` -- seeded identities with a stable
  per-identity color/shape signature plus unannotated distractor clutter,
  for tests, smoke runs and offline development;
- :func:`read_market1501_split` -- the standard on-disk Market1501 layout
  (``bounding_box_train/`` etc., filenames ``NNNN_cXsSf_dddddd.jpg``),
  transcribing ``datasets/market1501.py`` from
  https://github.com/nwojke/cosine_metric_learning. Junk identities
  (``pid < 0``) are skipped by default; the reference keeps them, which
  breaks identity classification, so the deviation is deliberate and noted.
  TFDS has no Market1501 builder in the pinned version, hence a directory
  reader rather than ``tfds.load``.

Also here: :func:`create_id_validation_split` (disjoint-identity split,
transcribing ``datasets/util.py::create_validation_split``) and
:func:`cmc_and_map` (CMC curve + mAP over cosine distances, the batch form
of ``metrics.py``).

Only the standard library, NumPy and Pillow run at module scope (Pillow is
a ``data`` extra: imported lazily, like ``uvdoc.py``). TensorFlow is
imported lazily inside the dataset builders.
"""

import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

#: Appearance network input geometry (ref ``IMAGE_SHAPE``).
REID_IMAGE_SHAPE = (128, 64, 3)

#: Reference train image directory name.
MARKET_TRAIN_DIR = "bounding_box_train"

#: Reference held-out identity fraction (ref ``num_validation_y=0.1``).
VALIDATION_ID_FRACTION = 0.1


# ---------------------------------------------------------------------
# Market1501 filename parsing (transcribed)
# ---------------------------------------------------------------------


def parse_market1501_filename(filename: str) -> Optional[Tuple[int, int]]:
    """Parse ``(person_id, camera_index)`` from a Market1501 filename.

    Transcribes ``_parse_filename``: handles double extensions, requires
    ``.jpg``, splits ``person_camseq_frame_detection``.

    :param filename: Bare filename (not a path).
    :type filename: str
    :return: ``(person_id, camera_index)`` or None for invalid names.
    :rtype: tuple or None
    """
    base, ext = os.path.splitext(os.path.basename(filename))
    if "." in base:
        base, ext = os.path.splitext(base)
    if ext != ".jpg":
        return None
    try:
        person_id, cam_seq, _, _ = base.split("_")
        if len(cam_seq) < 2 or not cam_seq[1].isdigit():
            return None
        return int(person_id), int(cam_seq[1])
    except (ValueError, IndexError):
        return None


def read_market1501_split(
    dataset_dir: str, image_dir_name: str = MARKET_TRAIN_DIR, skip_junk: bool = True
) -> Tuple[List[str], List[int], List[int]]:
    """List image paths, person IDs and camera indices for one split.

    :param dataset_dir: Market1501 root (contains ``bounding_box_train/``).
    :type dataset_dir: str
    :param image_dir_name: Split directory name.
    :type image_dir_name: str
    :param skip_junk: Drop ``pid < 0`` junk/detector-failure images. The
        reference keeps them; they carry no identity and break identity
        classification, so skipping is deliberate.
    :type skip_junk: bool
    :return: ``(filenames, person_ids, camera_indices)`` in sorted order.
    :rtype: tuple
    :raises FileNotFoundError: If the split directory does not exist.
    """
    image_dir = os.path.join(dataset_dir, image_dir_name)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Market1501 split not found: {image_dir}")
    filenames, ids, cameras = [], [], []
    for filename in sorted(os.listdir(image_dir)):
        parsed = parse_market1501_filename(filename)
        if parsed is None:
            continue
        person_id, camera_index = parsed
        if skip_junk and person_id < 0:
            continue
        filenames.append(os.path.join(image_dir, filename))
        ids.append(person_id)
        cameras.append(camera_index)
    return filenames, ids, cameras


def reindex_person_ids(ids: List[int]) -> Tuple[List[int], Dict[int, int]]:
    """Map raw person IDs to contiguous ``[0, C)`` class indices.

    :param ids: Raw person IDs.
    :type ids: list of int
    :return: ``(contiguous_ids, raw_to_index)``.
    :rtype: tuple
    """
    mapping = {raw: i for i, raw in enumerate(sorted(set(ids)))}
    return [mapping[raw] for raw in ids], mapping


def create_id_validation_split(
    ids: np.ndarray, fraction: float = VALIDATION_ID_FRACTION, seed: int = 1234
) -> Tuple[np.ndarray, np.ndarray]:
    """Split sample indices so train/val identities are DISJOINT.

    Transcribes ``create_validation_split`` (ref seed 1234).

    :param ids: Identity per sample.
    :type ids: numpy.ndarray
    :param fraction: Fraction of identities held out for validation.
    :type fraction: float
    :param seed: Seed for the identity draw.
    :type seed: int
    :return: ``(train_indices, validation_indices)``.
    :rtype: tuple
    """
    unique = np.unique(ids)
    num_val = int(fraction * len(unique))
    rng = np.random.RandomState(seed)
    val_ids = set(rng.choice(unique, max(num_val, 1), replace=False).tolist())
    mask = np.array([i in val_ids for i in ids])
    return np.nonzero(~mask)[0], np.nonzero(mask)[0]


# ---------------------------------------------------------------------
# synthetic identities
# ---------------------------------------------------------------------


def synthetic_reid_generator(
    num_ids: int,
    shots_per_id: int,
    image_shape: Tuple[int, int, int] = REID_IMAGE_SHAPE,
    seed: int = 0,
) -> Iterator[Tuple[np.ndarray, int]]:
    """Seeded identity-labeled person-crop generator.

    Each identity owns a stable signature -- a base hue plus a bright chest
    stripe at an identity-specific height -- over a dim noise background with
    unannotated distractor clutter elsewhere in the frame. Signatures survive
    the 128x64 render and are mutually discriminable by construction (the
    smoke tests learn them).

    :param num_ids: Distinct identities.
    :type num_ids: int
    :param shots_per_id: Images per identity.
    :type shots_per_id: int
    :param image_shape: ``(H, W, C)`` render size.
    :type image_shape: tuple
    :param seed: Seed.
    :type seed: int
    :return: Iterator of ``(image [0, 1], identity)`` tuples.
    :rtype: iterator
    """
    rng = np.random.default_rng(seed)
    height, width, channels = image_shape
    for identity in range(num_ids):
        stripe_row = int(height * (0.25 + 0.5 * ((identity * 0.37) % 1.0)))
        for _ in range(shots_per_id):
            image = (
                rng.random((height, width, channels), dtype=np.float32) * 0.25
            ).astype(np.float32)
            # Identity signature: saturated palette color at 70% plus a
            # complementary chest stripe at an identity-specific height.
            palette = np.array(
                [
                    [0.9, 0.2, 0.2],
                    [0.2, 0.9, 0.2],
                    [0.2, 0.2, 0.9],
                    [0.9, 0.9, 0.2],
                    [0.9, 0.2, 0.9],
                    [0.2, 0.9, 0.9],
                ],
                dtype=np.float32,
            )
            color = palette[identity % len(palette)] * float(
                rng.uniform(0.85, 1.0)
            )
            image[:, :, :] = image * 0.3 + color * 0.7
            stripe = slice(max(stripe_row - 3, 0), stripe_row + 3)
            image[stripe, :, :] = 1.0 - color
            image += rng.normal(0, 0.02, image.shape).astype(np.float32)
            yield np.clip(image, 0.0, 1.0).astype(np.float32), identity


# ---------------------------------------------------------------------
# evaluation: CMC + mAP over cosine distances
# ---------------------------------------------------------------------


def cmc_and_map(
    query_features: np.ndarray,
    query_ids: np.ndarray,
    gallery_features: np.ndarray,
    gallery_ids: np.ndarray,
    topk: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """CMC recall at k plus mean average precision over cosine ranking.

    For each query, gallery entries rank by cosine distance (same-camera
    matches excluded is a Market1501-eval refinement NOT applied here --
    camera indices are not threaded through; documented, not silent).
    CMC@k is the fraction of queries with a same-identity hit in the top k;
    AP per query is the standard ranked-precision average over its hits.

    :param query_features: Array ``(Nq, M)`` (need not be normalized).
    :type query_features: numpy.ndarray
    :param query_ids: Identity per query ``(Nq,)``.
    :type query_ids: numpy.ndarray
    :param gallery_features: Array ``(Ng, M)``.
    :type gallery_features: numpy.ndarray
    :param gallery_ids: Identity per gallery entry ``(Ng,)``.
    :type gallery_ids: numpy.ndarray
    :param topk: CMC cutoffs reported.
    :type topk: tuple
    :return: Dict with ``cmc@k`` entries and ``mAP``.
    :rtype: dict
    """
    query = np.asarray(query_features, dtype=np.float64)
    gallery = np.asarray(gallery_features, dtype=np.float64)
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-9)
    gallery = gallery / np.maximum(
        np.linalg.norm(gallery, axis=1, keepdims=True), 1e-9
    )
    distances = 1.0 - query @ gallery.T
    order = np.argsort(distances, axis=1)
    ranked_ids = np.asarray(gallery_ids)[order]
    query_ids = np.asarray(query_ids)
    out: Dict[str, float] = {}
    for k in topk:
        hits = (ranked_ids[:, :k] == query_ids[:, None]).any(axis=1)
        out[f"cmc@{k}"] = float(hits.mean())
    aps = []
    for i in range(len(query)):
        relevant = ranked_ids[i] == query_ids[i]
        if not relevant.any():
            continue
        positions = np.nonzero(relevant)[0] + 1
        aps.append(float(np.mean(np.arange(1, len(positions) + 1) / positions)))
    out["mAP"] = float(np.mean(aps)) if aps else 0.0
    return out


# ---------------------------------------------------------------------
# image loading + tf.data builders (lazy imports)
# ---------------------------------------------------------------------


def _load_image_128x64(path: str) -> np.ndarray:
    """Load an image file to ``(128, 64, 3)`` [0, 1] float32 (Pillow, lazy)."""
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((64, 128))
    return np.asarray(image, dtype=np.float32) / 255.0


def pk_batch_generator(
    images: List[np.ndarray],
    ids: List[int],
    p_ids: int,
    k_shots: int,
    seed: int = 0,
    augment: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield PK batches (``p_ids`` identities x ``k_shots`` shots) forever.

    Identities sample uniformly per batch; shots sample with replacement
    within an identity. Training batches flip left-right with probability
    0.5 (the reference's only training augmentation).

    :param images: In-memory images ``(128, 64, 3)``.
    :type images: list
    :param ids: Contiguous identity per image.
    :type ids: list
    :param p_ids: Identities per batch.
    :type p_ids: int
    :param k_shots: Shots per identity (K >= 2 for the triplet loss).
    :type k_shots: int
    :param seed: Seed.
    :type seed: int
    :param augment: Whether to flip.
    :type augment: bool
    :return: Iterator of ``(batch (P*K, 128, 64, 3), labels (P*K,))``.
    :rtype: iterator
    """
    rng = np.random.default_rng(seed)
    by_id: Dict[int, List[int]] = {}
    for idx, identity in enumerate(ids):
        by_id.setdefault(int(identity), []).append(idx)
    identity_list = sorted(by_id)
    if p_ids > len(identity_list):
        raise ValueError(
            f"p_ids ({p_ids}) exceeds available identities ({len(identity_list)})"
        )
    while True:
        chosen = rng.choice(identity_list, p_ids, replace=False)
        batch_images, batch_labels = [], []
        for identity in chosen:
            shots = rng.choice(by_id[int(identity)], k_shots, replace=True)
            for shot in shots:
                image = images[int(shot)].astype(np.float32)
                if augment and rng.random() < 0.5:
                    image = image[:, ::-1, :]
                batch_images.append(image)
                batch_labels.append(int(identity))
        yield np.stack(batch_images, axis=0), np.array(batch_labels, dtype=np.int32)
