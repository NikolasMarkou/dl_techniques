"""Shared helpers for image–text pair datasets (CLIP, VLM pretraining).

This module owns the dataset-handling surface that image–text training
scripts share:

- Image preprocessing constants and decode/augment/normalize helpers.
- Caption tokenization against a raw ``tiktoken`` encoder (CLIP uses the
  encoder's native ``eot_token`` as both end-of-sequence and pad; no
  BERT-style special tokens).
- Split loaders for the datasets we currently support on-disk:

  - Synthetic random pairs (smoke tests).
  - MS-COCO 2017 from a locally-extracted tree.
  - Conceptual Captions 3M from the layout written by
    ``train/cliffordnet/prepare_cc3m.py``.

- A ``tf.data`` pipeline builder that streams paths from disk and
  returns batches shaped ``{"image": float32, "text": int32}`` — with an
  opt-in ``cache_decoded`` mode that trades RAM for a per-step speedup
  on small datasets.

All loaders return the same tuple ``(list_of_image_paths_or_arrays,
(N, L) int32 token matrix)`` so a training script can dispatch between
them via a single ``--dataset`` flag without branching in its pipeline.

Scaling note: every loader returns paths, not decoded arrays. The
pipeline defaults to streaming from disk (``cache_decoded=False``) so
RAM usage is constant in dataset size. Use ``cache_decoded=True`` only
when the full post-resize cache fits comfortably in RAM (roughly
``N <= 200 k`` for typical 160²-ish images).
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
#
# ImageNet-standard RGB normalization matches OpenAI CLIP / OpenCLIP, so
# models trained with these constants are directly compatible with the
# CLIP ecosystem's preprocessing expectations.

IMAGE_MEAN: tf.Tensor = tf.constant(
    [0.48145466, 0.4578275, 0.40821073], dtype=tf.float32
)
IMAGE_STD: tf.Tensor = tf.constant(
    [0.26862954, 0.26130258, 0.27577711], dtype=tf.float32
)


def read_decode_resize_uint8(
    path: tf.Tensor, image_size: int, training: bool
) -> tf.Tensor:
    """Load + decode a JPEG and resize to the pre-augment size.

    For training we resize to ``ceil(image_size * 1.15)`` square so a
    subsequent random crop can operate; for evaluation we resize directly
    to the target side. Output stays ``uint8`` so a downstream
    ``.cache()`` (if enabled) keeps a compact representation (4× smaller
    than float32).

    :param path: Scalar string tensor pointing at a JPEG file.
    :param image_size: Target post-crop side length in pixels.
    :param training: Whether augmentation will follow (affects pre-augment
        resize target).
    :return: ``(H, W, 3)`` uint8 tensor.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    if training:
        target = tf.cast(image_size, tf.float32)
        larger = tf.cast(tf.math.ceil(target * 1.15), tf.int32)
        img = tf.image.resize(img, (larger, larger), method="bilinear")
    else:
        img = tf.image.resize(
            img, (image_size, image_size), method="bilinear"
        )
    return tf.cast(img, tf.uint8)


def augment_and_normalize(
    img_uint8: tf.Tensor, image_size: int, training: bool
) -> tf.Tensor:
    """Stochastic augmentation + ImageNet/CLIP normalization.

    Runs on a ``uint8`` tensor (optionally from a cache), applying random
    crop + horizontal flip during training and only the normalization
    during evaluation.
    """
    img = tf.cast(img_uint8, tf.float32) / 255.0
    if training:
        img = tf.image.random_crop(img, (image_size, image_size, 3))
        img = tf.image.random_flip_left_right(img)
    img = (img - IMAGE_MEAN) / IMAGE_STD
    return img


# ---------------------------------------------------------------------------
# Caption tokenization
# ---------------------------------------------------------------------------


def tokenize_captions(
    captions: List[str],
    encoder: Any,
    context_length: int,
) -> np.ndarray:
    """Tokenize a list of captions into a right-padded ``(N, L)`` int32 array.

    Sequences are right-padded with the encoder's ``eot_token``, which
    doubles as both an end-of-sequence marker and the pad sentinel used
    by the CLIP text tower's last-token gather. Using the native EOT ID
    avoids collisions with real caption tokens (e.g. tiktoken ``gpt2``
    encoding's token 0 is ``'!'``).

    :param captions: Python strings. ``bytes`` are decoded as utf-8,
        errors ignored.
    :param encoder: Raw tiktoken encoding object — anything with an
        ``encode`` method and an ``eot_token`` attribute.
    :param context_length: Fixed sequence length.
    :return: ``(N, context_length)`` int32 token matrix.
    """
    eot = int(encoder.eot_token)
    ids = np.full((len(captions), context_length), eot, dtype=np.int32)
    for i, cap in enumerate(captions):
        if isinstance(cap, bytes):
            cap = cap.decode("utf-8", errors="ignore")
        toks = encoder.encode(cap)[: context_length - 1]
        # Append EOT so the last-token gather has a stable termination.
        toks = toks + [eot]
        ids[i, : len(toks)] = toks
    return ids


# ---------------------------------------------------------------------------
# Split loaders
# ---------------------------------------------------------------------------


def build_synthetic_image_text_dataset(
    num_samples: int,
    image_size: int,
    context_length: int,
    vocab_size: int,
    eot_token_id: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic synthetic image–caption dataset for smoke tests.

    Images are random-normal tensors (pre-normalized); captions are random
    token sequences right-padded with ``eot_token_id``. This exists only
    so training scripts stay runnable without any downloaded data.

    :return: ``(images, tokens)`` where images is
        ``(N, image_size, image_size, 3)`` float32 and tokens is
        ``(N, context_length)`` int32.
    """
    rng = np.random.default_rng(seed)
    images = rng.standard_normal(
        (num_samples, image_size, image_size, 3)
    ).astype(np.float32)
    tokens = np.full(
        (num_samples, context_length), eot_token_id, dtype=np.int32
    )
    token_lens = rng.integers(8, context_length, size=num_samples)
    for i, ln in enumerate(token_lens):
        tokens[i, : ln - 1] = rng.integers(0, vocab_size, size=ln - 1)
        tokens[i, ln - 1] = eot_token_id
    return images, tokens


def load_coco2017_local_split(
    split: str,
    coco_root: str,
    max_samples: Optional[int],
    encoder: Any,
    context_length: int,
) -> Tuple[List[str], np.ndarray]:
    """Load a split of MS-COCO 2017 captions from a local extracted tree.

    ``coco_root`` must contain:

    - ``train2017/*.jpg`` and ``val2017/*.jpg`` image folders
    - ``annotations/captions_train2017.json`` and
      ``annotations/captions_val2017.json``

    The tfds ``coco_captions`` builder is deliberately avoided — it uses
    the 2014 split and would trigger a fresh ~20 GB download. This loader
    reads directly from the extracted 2017 tree and returns image paths
    (not decoded arrays) so the tf.data pipeline can stream JPEGs lazily.

    For ``split="train"`` every annotation becomes a pair (~5 captions per
    image, ~591k pairs). For ``split="val"`` exactly one caption per image
    is emitted so the 5k-pair retrieval R@K eval keeps its standard
    one-correct-per-query semantics.

    :param split: ``"train"`` or ``"val"``.
    :param coco_root: Directory containing ``train2017/``, ``val2017/``,
        and ``annotations/``.
    :param max_samples: Optional cap on the number of pairs returned.
    :param encoder: Raw tiktoken encoder.
    :param context_length: Fixed text sequence length.
    :return: (list of absolute image paths, ``(N, L)`` int32 token array).
    :raises FileNotFoundError: If the annotations or image dir are missing.
    """
    ann_path = os.path.join(
        coco_root, "annotations", f"captions_{split}2017.json"
    )
    img_dir = os.path.join(coco_root, f"{split}2017")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"COCO annotations not found: {ann_path}. "
            f"Set coco_root to the directory containing train2017/, "
            f"val2017/, and annotations/."
        )
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"COCO image dir not found: {img_dir}")

    logger.info(
        f"Loading COCO 2017 split={split} max_samples={max_samples} "
        f"from {coco_root}..."
    )
    with open(ann_path) as f:
        data = json.load(f)

    image_id_to_file = {
        img["id"]: img["file_name"] for img in data["images"]
    }

    paths: List[str] = []
    captions: List[str] = []
    if split == "train":
        for ann in data["annotations"]:
            filename = image_id_to_file.get(ann["image_id"])
            if filename is None:
                continue
            paths.append(os.path.join(img_dir, filename))
            captions.append(ann["caption"])
            if max_samples is not None and len(paths) >= max_samples:
                break
    else:
        seen: set = set()
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            if image_id in seen:
                continue
            filename = image_id_to_file.get(image_id)
            if filename is None:
                continue
            seen.add(image_id)
            paths.append(os.path.join(img_dir, filename))
            captions.append(ann["caption"])
            if max_samples is not None and len(paths) >= max_samples:
                break

    token_ids = tokenize_captions(captions, encoder, context_length)
    logger.info(
        f"Loaded {len(paths)} (image, caption) pairs from COCO/{split}"
    )
    return paths, token_ids


def _cc3m_shard_of(img_id: str) -> str:
    """Deterministic byte-hash shard for a CC3M image id.

    Matches the layout written by ``train/cliffordnet/prepare_cc3m.py``
    so the loader can rebuild the image path from the id alone.
    """
    h = 0
    for ch in img_id:
        h = (h * 31 + ord(ch)) & 0xFFFF
    return f"{h & 0xFF:02x}"


def load_cc3m_local_split(
    split: str,
    cc3m_root: str,
    max_samples: Optional[int],
    encoder: Any,
    context_length: int,
) -> Tuple[List[str], np.ndarray]:
    """Load a CC3M split from a locally-extracted tree.

    Expects the layout written by ``prepare_cc3m.py``::

        <cc3m_root>/
          <split>/XX/cc3m_<split>_NNNNNNNN.jpg
          <split>_captions.jsonl       <- one JSON-per-line {id, caption}

    The JSONL file is the source of truth for caption order; image paths
    are reconstructed from the id via :func:`_cc3m_shard_of`. A flat
    ``<split>/<id>.jpg`` layout is also accepted so hand-assembled CC3M
    trees (e.g. from ``img2dataset``) work without re-sharding.

    CC3M normalises to roughly one caption per image (unlike COCO's
    five), so no dedup step is needed.

    :param split: ``"train"`` or ``"validation"``.
    :param cc3m_root: Directory containing the split folder and JSONL.
    :param max_samples: Optional cap on the number of pairs returned.
    :param encoder: Raw tiktoken encoder.
    :param context_length: Fixed text sequence length.
    :return: (list of absolute image paths, ``(N, L)`` int32 token array).
    :raises FileNotFoundError: If the JSONL or image dir are missing.
    """
    jsonl_path = os.path.join(cc3m_root, f"{split}_captions.jsonl")
    img_root = os.path.join(cc3m_root, split)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"CC3M captions file not found: {jsonl_path}. "
            f"Run train/cliffordnet/prepare_cc3m.py --dst {cc3m_root} first."
        )
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"CC3M image dir not found: {img_root}")

    logger.info(
        f"Loading CC3M split={split} max_samples={max_samples} "
        f"from {cc3m_root}..."
    )

    paths: List[str] = []
    captions: List[str] = []
    missing = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_id = rec["id"]
            caption = rec.get("caption", "")
            if not caption:
                continue
            candidates = (
                os.path.join(
                    img_root, _cc3m_shard_of(img_id), f"{img_id}.jpg"
                ),
                os.path.join(img_root, f"{img_id}.jpg"),
            )
            img_path = next(
                (p for p in candidates if os.path.exists(p)), None
            )
            if img_path is None:
                missing += 1
                continue
            paths.append(img_path)
            captions.append(caption)
            if max_samples is not None and len(paths) >= max_samples:
                break

    if missing:
        logger.warning(
            f"CC3M/{split}: {missing:,} captions skipped because the "
            f"image file was missing from {img_root}."
        )
    token_ids = tokenize_captions(captions, encoder, context_length)
    logger.info(
        f"Loaded {len(paths)} (image, caption) pairs from CC3M/{split}"
    )
    return paths, token_ids


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------


def make_image_text_tf_dataset(
    images: Any,
    token_ids: np.ndarray,
    image_size: int,
    batch_size: int,
    training: bool,
    cache_decoded: bool = False,
    shuffle_buffer: int = 16384,
) -> tf.data.Dataset:
    """Build an image–text ``tf.data`` pipeline.

    ``images`` may be one of:

    - a numpy array of pre-normalized images (synthetic mode); the
      pipeline just attaches tokens and batches,
    - a Python list of filesystem JPEG paths (COCO / CC3M / anything
      path-addressable); JPEGs are streamed and decoded lazily via
      ``tf.io.read_file``.

    **Caching**. The file-path path has two variants:

    - ``cache_decoded=False`` (default, scalable): stream every epoch.
      Paths are shuffled, then JPEGs are decoded + augmented in parallel
      via ``num_parallel_calls=AUTOTUNE``. RAM usage is constant in
      dataset size, so the pipeline scales from 32 k samples to
      hundreds of millions. The OS page cache transparently accelerates
      repeat reads if the dataset fits in RAM, but nothing assumes it.
    - ``cache_decoded=True`` (opt-in small-dataset speedup): decode each
      image once, cache the resulting uint8 tensor at the pre-augment
      size, then shuffle + augment + normalize each epoch. Cache memory
      grows linearly with dataset size, so only use this when the full
      cache fits comfortably in RAM.

    The pipeline is ``.repeat()``-ed on the training side so a fixed
    ``steps_per_epoch`` can exceed one natural pass through the data
    without raising a dataset-exhausted error.

    :param images: Numpy float32 array (synthetic) or list of JPEG paths.
    :param token_ids: ``(N, L)`` int32 token matrix aligned with ``images``.
    :param image_size: Target post-crop side length in pixels.
    :param batch_size: Batch size. ``drop_remainder=True`` during training.
    :param training: Whether to shuffle/augment/repeat.
    :param cache_decoded: Enable the RAM-cached uint8 branch.
    :param shuffle_buffer: Max shuffle buffer when caching; path shuffling
        uses the full dataset size regardless.
    :return: ``tf.data.Dataset`` yielding
        ``{"image": float32[B,H,W,3], "text": int32[B,L]}`` batches.
    """
    if isinstance(images, np.ndarray):
        ds = tf.data.Dataset.from_tensor_slices((images, token_ids))

        def _passthrough(img, tok):
            return {"image": img, "text": tok}

        if training:
            ds = ds.shuffle(
                min(shuffle_buffer, len(images)),
                reshuffle_each_iteration=True,
            )
        ds = ds.map(_passthrough, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.repeat()
        ds = ds.batch(batch_size, drop_remainder=training)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # File-path path: stream JPEGs off disk.
    paths = [str(p) for p in images]
    n = len(paths)

    # Sort paths (and tokens alongside) so the filesystem read order is
    # sequential/locality-friendly regardless of the caller's input order.
    # At large N (>~30k files) a globally random read order exceeds the VFS
    # dentry cache — each open() costs a metadata lookup even on a warm page
    # cache, adding ~900 ms/batch for 128 random opens against a 118k-file
    # directory. Sequential opens stay in cache. Randomness is recovered below
    # via the tf.data shuffle buffer (bounded so we don't regress to global
    # random order via a full-N buffer).
    if n > 1 and not isinstance(images, np.ndarray):
        sort_idx = np.argsort(paths, kind="stable")
        paths = [paths[i] for i in sort_idx]
        token_ids = token_ids[sort_idx]

    if cache_decoded:
        # --- Cached variant: decode once, cache uint8 tensors, reuse ---
        ds = tf.data.Dataset.from_tensor_slices((paths, token_ids))

        def _decode_step(path, tok):
            img_u8 = read_decode_resize_uint8(path, image_size, training)
            return img_u8, tok

        ds = ds.map(_decode_step, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()

        if training:
            ds = ds.shuffle(
                min(shuffle_buffer, n),
                reshuffle_each_iteration=True,
            )

        def _augment_step(img_u8, tok):
            img = augment_and_normalize(img_u8, image_size, training)
            return {"image": img, "text": tok}

        ds = ds.map(_augment_step, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # --- Streaming variant: shuffle (small window) + decode + augment each epoch ---
        # Cap the shuffle buffer at 1024 on the streaming path: the argsort
        # above already randomizes globally (COCO/CC3M filenames have no
        # semantic ordering), so a small local jitter window is sufficient for
        # training randomness. Larger buffers (>=4k) add a per-batch cost that
        # dominates step time at N=100k+ (measured ~900 ms/batch at N=full).
        ds = tf.data.Dataset.from_tensor_slices((paths, token_ids))
        if training:
            ds = ds.shuffle(
                min(1024, n),
                reshuffle_each_iteration=True,
            )

        def _full_preprocess_step(path, tok):
            img_u8 = read_decode_resize_uint8(path, image_size, training)
            img = augment_and_normalize(img_u8, image_size, training)
            return {"image": img, "text": tok}

        ds = ds.map(
            _full_preprocess_step, num_parallel_calls=tf.data.AUTOTUNE
        )

    if training:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
