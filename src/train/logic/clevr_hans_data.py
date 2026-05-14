"""CLEVR-Hans3 dataset utilities for E5.

Plan: plan_2026-05-14_c95e848c (decision D-001).

Provides:
- ``download_clevr_hans3``: headless download + zip extract with 2h wall-clock leash.
- ``find_splits``: locate the conventional CLEVR layout on disk.
- ``load_scenes_json``: parse the per-split scene-graph JSON.
- ``encode_scene_graph``: encode a single scene as ``(max_objects, 18)`` float32.
- ``build_image_dataset``: ``tf.data`` pipeline for ResNet50-preprocessed PNGs.
- ``build_symbolic_dataset``: ``tf.data`` pipeline for symbolic scene-graph input
  (perfect-perception oracle).

CLEVR-Hans3 attribute taxonomy (per CVPR'21 paper):
- size:     small, large                          -> 3 onehot slots (3rd reserved)
- color:    gray, red, blue, green, brown, purple,
            cyan, yellow                          -> 8 onehot slots
- material: rubber, metal                         -> 2 onehot slots
- shape:    cube, sphere, cylinder                -> 3 onehot slots
- presence + reserved                             -> 2 binary slots
Total per-object feature width: 18.
"""

# DECISION plan_2026-05-14_c95e848c/D-001
# This module ships with E5 (CLEVR-Hans3). Uses keras.applications.ResNet50
# (the only live pretrained Keras CNN). NS-CL substituted with perfect-perception
# oracle. See plans/plan_2026-05-14_c95e848c/decisions.md entry D-001.

from __future__ import annotations

import json
import os
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

CLEVR_HANS3_URL = (
    "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2611/"
    "CLEVR-Hans3.zip"
)
CLEVR_HANS3_DIRNAME = "CLEVR-Hans3"

SIZES = ["small", "large"]
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIALS = ["rubber", "metal"]
SHAPES = ["cube", "sphere", "cylinder"]

# Width layout: 3 size + 8 color + 2 material + 3 shape + 2 reserved (presence,
# pad) = 18.
FEATURE_WIDTH = 18
SIZE_OFFSET = 0
COLOR_OFFSET = 3
MATERIAL_OFFSET = 11
SHAPE_OFFSET = 13
PRESENCE_OFFSET = 16


# ---------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------


def download_clevr_hans3(
    data_dir: str,
    *,
    url: str = CLEVR_HANS3_URL,
    timeout_s: int = 7200,
    chunk_size: int = 1 << 20,
) -> bool:
    """Download the CLEVR-Hans3 zip and extract it.

    Returns True on success (extracted directory present + zip removed),
    False on any error. Honors a hard wall-clock leash.

    The leash is enforced both on the streaming-download loop (timeout_s
    cap on download alone) and on the total elapsed time.
    """
    import urllib.request
    import urllib.error

    os.makedirs(data_dir, exist_ok=True)
    target_dir = os.path.join(data_dir, CLEVR_HANS3_DIRNAME)
    zip_path = os.path.join(data_dir, "CLEVR-Hans3.zip")

    if os.path.isdir(target_dir):
        logger.info(f"CLEVR-Hans3 already present at {target_dir}; skipping download.")
        return True

    t0 = time.time()
    logger.info(f"Downloading CLEVR-Hans3 from {url} -> {zip_path}")

    try:
        # Stream the zip with a wall-clock budget.
        req = urllib.request.Request(url, headers={"User-Agent": "dl-techniques/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            written = 0
            with open(zip_path, "wb") as f:
                while True:
                    if time.time() - t0 > timeout_s:
                        logger.error(
                            f"Download exceeded wall-clock leash {timeout_s}s; "
                            f"aborting at {written}/{total} bytes."
                        )
                        return False
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
        dl_s = time.time() - t0
        logger.info(
            f"Downloaded {written} bytes in {dl_s:.1f}s "
            f"({written / max(dl_s, 1e-3) / 1024 / 1024:.1f} MB/s)."
        )

        # Verify zip magic bytes (PK\x03\x04).
        with open(zip_path, "rb") as f:
            magic = f.read(4)
        if magic[:2] != b"PK":
            logger.error(f"Bad zip magic bytes: {magic!r}. Aborting.")
            try:
                os.remove(zip_path)
            except OSError:
                pass
            return False

        # Extract.
        logger.info(f"Extracting {zip_path} -> {data_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)

        # Verify extraction landed somewhere we recognise.
        if not os.path.isdir(target_dir):
            # Some upstream zips wrap the dir differently; search for a CLEVR-Hans3*
            candidates = [
                d for d in os.listdir(data_dir)
                if d.startswith("CLEVR-Hans") and os.path.isdir(os.path.join(data_dir, d))
            ]
            if len(candidates) == 1 and candidates[0] != CLEVR_HANS3_DIRNAME:
                src = os.path.join(data_dir, candidates[0])
                logger.info(f"Renaming {src} -> {target_dir}")
                os.rename(src, target_dir)

        if not os.path.isdir(target_dir):
            logger.error(
                f"Extraction did not produce {target_dir}; contents: "
                f"{os.listdir(data_dir)}"
            )
            return False

        # Remove zip to free disk (CRITICAL — only 26 GB free).
        try:
            os.remove(zip_path)
            logger.info(f"Removed zip {zip_path} to reclaim disk.")
        except OSError as e:
            logger.warning(f"Could not remove zip {zip_path}: {e}")

        logger.info(f"CLEVR-Hans3 ready at {target_dir} (elapsed {time.time() - t0:.1f}s).")
        return True

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        logger.error(f"Download failed: {type(e).__name__}: {e}")
        # Clean up partial zip.
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except OSError:
            pass
        return False
    except zipfile.BadZipFile as e:
        logger.error(f"Bad zip file: {e}")
        try:
            os.remove(zip_path)
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------


def find_splits(data_dir: str) -> Dict[str, Dict[str, str]]:
    """Locate the CLEVR-Hans3 split layout on disk.

    Returns a dict ``{"train"|"val"|"test": {"images_dir": ..., "scenes_json": ...}}``.
    Returns ``{}`` if the data is not present.

    Tolerates both ``data_dir`` being the parent of ``CLEVR-Hans3/`` and being
    ``CLEVR-Hans3/`` itself.
    """
    root = data_dir
    if os.path.isdir(os.path.join(data_dir, CLEVR_HANS3_DIRNAME)):
        root = os.path.join(data_dir, CLEVR_HANS3_DIRNAME)

    out: Dict[str, Dict[str, str]] = {}
    for split in ("train", "val", "test"):
        # Actual upstream layout (verified 2026-05-14):
        #   <root>/<split>/images/*.png
        #   <root>/<split>/CLEVR_HANS_scenes_<split>.json
        # Also tolerate the "flat" convention some CLEVR releases use:
        #   <root>/images/<split>/*.png
        #   <root>/scenes/CLEVR_Hans3_<split>_scenes.json
        images_dir_candidates = [
            os.path.join(root, split, "images"),
            os.path.join(root, "images", split),
        ]
        scenes_json_candidates = [
            os.path.join(root, split, f"CLEVR_HANS_scenes_{split}.json"),
            os.path.join(root, "scenes", f"CLEVR_Hans3_{split}_scenes.json"),
            os.path.join(root, "scenes", f"CLEVR-Hans3_{split}_scenes.json"),
            os.path.join(root, "scenes", f"{split}_scenes.json"),
        ]
        images_dir = next((p for p in images_dir_candidates if os.path.isdir(p)), None)
        scenes_json = next((p for p in scenes_json_candidates if os.path.isfile(p)), None)
        if images_dir is not None and scenes_json is not None:
            out[split] = {"images_dir": images_dir, "scenes_json": scenes_json}
        else:
            logger.warning(
                f"find_splits: split '{split}' missing "
                f"(images_dir={images_dir}, scenes_json={scenes_json})."
            )
    return out


# ---------------------------------------------------------------------
# Scene-graph parsing + encoding
# ---------------------------------------------------------------------


def load_scenes_json(path: str) -> List[Dict[str, Any]]:
    """Parse a CLEVR-Hans3 scenes JSON file.

    Returns a list of per-scene dicts. CLEVR convention: top-level dict has
    ``{"info": ..., "scenes": [...]}``; each scene has ``image_filename``,
    ``class_id``, and ``objects`` (list of attribute dicts).
    """
    with open(path, "r") as f:
        blob = json.load(f)
    if isinstance(blob, dict) and "scenes" in blob:
        return list(blob["scenes"])
    if isinstance(blob, list):
        return blob
    raise ValueError(f"Unrecognized scenes JSON structure in {path}: top-level={type(blob)}")


def _onehot_index(value: str, vocab: List[str], default: int = -1) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        return default


def encode_scene_graph(scene: Dict[str, Any], max_objects: int = 10) -> np.ndarray:
    """Encode a single scene as a ``(max_objects, FEATURE_WIDTH)`` float32 matrix.

    Unknown attribute values are silently encoded as zero. Objects past
    ``max_objects`` are truncated (with a one-shot logged warning).
    """
    objects = scene.get("objects", [])
    if len(objects) > max_objects:
        logger.warning(
            f"encode_scene_graph: scene has {len(objects)} objects, "
            f"truncating to {max_objects}."
        )
        objects = objects[:max_objects]

    out = np.zeros((max_objects, FEATURE_WIDTH), dtype=np.float32)
    for i, obj in enumerate(objects):
        si = _onehot_index(obj.get("size", ""), SIZES)
        if si >= 0:
            out[i, SIZE_OFFSET + si] = 1.0
        ci = _onehot_index(obj.get("color", ""), COLORS)
        if ci >= 0:
            out[i, COLOR_OFFSET + ci] = 1.0
        mi = _onehot_index(obj.get("material", ""), MATERIALS)
        if mi >= 0:
            out[i, MATERIAL_OFFSET + mi] = 1.0
        shi = _onehot_index(obj.get("shape", ""), SHAPES)
        if shi >= 0:
            out[i, SHAPE_OFFSET + shi] = 1.0
        out[i, PRESENCE_OFFSET] = 1.0  # presence bit
    return out


def encode_scenes(
    scenes: List[Dict[str, Any]],
    max_objects: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a list of scenes into ``(features, labels)`` numpy arrays.

    - features: shape ``(N, max_objects, FEATURE_WIDTH)`` float32.
    - labels: shape ``(N,)`` int32; missing ``class_id`` -> -1 (filtered upstream).
    """
    feats = np.stack([encode_scene_graph(s, max_objects=max_objects) for s in scenes], axis=0)
    labels = np.asarray(
        [int(s.get("class_id", -1)) for s in scenes], dtype=np.int32
    )
    return feats, labels


# ---------------------------------------------------------------------
# tf.data pipelines (image + symbolic)
# ---------------------------------------------------------------------


def _scene_to_image_paths_labels(
    scenes: List[Dict[str, Any]], images_dir: str
) -> Tuple[List[str], List[int]]:
    paths, labels = [], []
    for s in scenes:
        fn = s.get("image_filename")
        cid = s.get("class_id")
        if fn is None or cid is None:
            continue
        p = os.path.join(images_dir, fn)
        if os.path.isfile(p):
            paths.append(p)
            labels.append(int(cid))
    return paths, labels


def build_image_dataset(
    split_info: Dict[str, str],
    *,
    image_size: int = 128,
    batch_size: int = 32,
    shuffle: bool = False,
    cache: bool = True,
    seed: Optional[int] = None,
):
    """Build a ``tf.data.Dataset`` of (preprocessed image, label) for one split.

    Pipeline: PNG-decode -> resize -> ``keras.applications.resnet.preprocess_input``
    -> cache -> shuffle (if requested) -> batch -> prefetch.

    Returns the dataset and an integer ``num_samples`` (best-effort, from the
    initial list).
    """
    import tensorflow as tf
    from keras.applications.resnet import preprocess_input

    scenes = load_scenes_json(split_info["scenes_json"])
    paths, labels = _scene_to_image_paths_labels(scenes, split_info["images_dir"])
    num_samples = len(paths)
    if num_samples == 0:
        raise ValueError(
            f"No images found for split: images_dir={split_info['images_dir']} "
            f"scenes_json={split_info['scenes_json']}"
        )
    logger.info(f"build_image_dataset: {num_samples} samples from {split_info['images_dir']}")

    paths_t = tf.constant(paths)
    labels_t = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_png(raw, channels=3)
        img = tf.image.resize(img, (image_size, image_size), method="bilinear")
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=min(num_samples, 2048), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, num_samples


def build_symbolic_dataset(
    split_info: Dict[str, str],
    *,
    max_objects: int = 10,
    batch_size: int = 128,
    shuffle: bool = False,
    seed: Optional[int] = None,
):
    """Build a ``tf.data.Dataset`` for the perfect-perception oracle.

    Each example is ``((max_objects, FEATURE_WIDTH) float32, class_id int32)``.
    """
    import tensorflow as tf

    scenes = load_scenes_json(split_info["scenes_json"])
    feats, labels = encode_scenes(scenes, max_objects=max_objects)
    # Drop samples with missing class_id.
    valid = labels >= 0
    feats = feats[valid]
    labels = labels[valid]
    num_samples = int(feats.shape[0])
    if num_samples == 0:
        raise ValueError(f"No valid (class_id>=0) scenes in {split_info['scenes_json']}")
    logger.info(f"build_symbolic_dataset: {num_samples} samples (max_objects={max_objects})")

    ds = tf.data.Dataset.from_tensor_slices((feats, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(num_samples, 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, num_samples


# ---------------------------------------------------------------------
# Convenience: number of classes inferred from the labels seen.
# ---------------------------------------------------------------------


def infer_num_classes(split_info: Dict[str, str]) -> int:
    scenes = load_scenes_json(split_info["scenes_json"])
    labels = sorted(set(int(s["class_id"]) for s in scenes if "class_id" in s))
    if not labels:
        return 0
    return max(labels) + 1
