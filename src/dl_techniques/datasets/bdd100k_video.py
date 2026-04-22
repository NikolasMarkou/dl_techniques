"""BDD100K video clip dataset for Video-JEPA-Clifford training.

Produces ``tf.data.Dataset`` batches matching the VideoJEPA training schema:

.. code-block:: python

    ({
        "pixels": (B, T, H, W, 3),   # float32, in [0, 1]
     },
     dummy_y)  # zero scalar so model.fit with loss=None works

**Source.** BDD100K train videos (``.mov``), assumed flat-listed under
``videos_root``. Each call to the generator randomly picks a video, seeks
to a random starting frame, and reads ``T`` consecutive frames. Frames
are BGR→RGB converted, resized to ``img_size × img_size`` via
``cv2.INTER_AREA``, and normalized to ``[0, 1]`` float32.

**Dependencies.** Requires ``opencv-python`` (``pip install opencv-python``).

Typical usage (sanity run at B=4, T=8, 112×112, 200 steps):

.. code-block:: python

    ds = bdd100k_video_dataset(
        videos_root="/path/to/bdd_data/train/videos",
        batch_size=4, T=8, img_size=112,
        num_steps=200, seed=0,
    )
    for x, y in ds:
        # x["pixels"].shape == (4, 8, 112, 112, 3)
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

try:
    import tensorflow as tf
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "bdd100k_video_dataset requires TensorFlow. "
        "Install tensorflow>=2.18."
    ) from _e

try:
    import cv2
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "bdd100k_video_dataset requires opencv-python. "
        "Install with `pip install opencv-python`."
    ) from _e

from dl_techniques.utils.logger import logger


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _list_video_files(videos_root: Path) -> list[Path]:
    """Non-recursive listing of ``.mov`` files in ``videos_root``.

    BDD100K train videos are flat-listed, so no recursion is needed.
    """
    if not videos_root.exists():
        raise FileNotFoundError(
            f"BDD100K videos_root does not exist: {videos_root}"
        )
    # Accept both .mov and .MOV for robustness.
    paths = sorted(
        p for p in videos_root.iterdir()
        if p.is_file() and p.suffix.lower() == ".mov"
    )
    if not paths:
        raise FileNotFoundError(
            f"No .mov files found in {videos_root}"
        )
    return paths


def _read_clip(
    path: Path, T: int, img_size: int,
) -> Optional[np.ndarray]:
    """Read T consecutive frames starting at a random offset.

    :return: ``(T, img_size, img_size, 3)`` float32 in [0, 1], or
        ``None`` if the video is too short / unreadable.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < T:
            return None
        start = int(np.random.randint(0, frame_count - T + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = np.empty((T, img_size, img_size, 3), dtype=np.float32)
        for t in range(T):
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            # BGR -> RGB, resize, normalize.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (img_size, img_size),
                interpolation=cv2.INTER_AREA,
            )
            frames[t] = frame.astype(np.float32) / 255.0
        return frames
    except Exception as e:  # pragma: no cover
        logger.warning(f"cv2 failure on {path.name}: {e}")
        return None
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def bdd100k_video_dataset(
    videos_root: Union[str, Path],
    batch_size: int,
    T: int,
    img_size: int,
    num_steps: Optional[int] = None,
    seed: int = 0,
    max_resample_attempts: int = 8,
    split: str = "all",
    val_fraction: float = 0.1,
) -> "tf.data.Dataset":
    """Build a tf.data.Dataset of BDD100K video clips.

    :param videos_root: directory containing ``.mov`` files (flat layout).
    :param batch_size: batch size.
    :param T: number of frames per clip.
    :param img_size: spatial size (square). Frames are resized via INTER_AREA.
    :param num_steps: if provided, the dataset yields exactly
        ``num_steps`` batches per iteration (via ``.take``).
    :param seed: numpy RNG seed for shuffling + per-clip offsets. Also
        determines the train/val split when ``split != "all"``.
    :param max_resample_attempts: per-draw cap on retries when a video is
        unreadable or shorter than ``T``. Raises after exhausting.
    :param split: one of ``"all"``, ``"train"``, ``"val"``. The files under
        ``videos_root`` are deterministically shuffled by ``seed`` and the
        last ``val_fraction`` become val.
    :param val_fraction: fraction of files assigned to val when ``split``
        is ``"train"`` or ``"val"``. Ignored for ``"all"``.
    :return: ``tf.data.Dataset`` yielding ``({"pixels": clip}, 0.0)`` where
        ``clip`` has shape ``(B, T, img_size, img_size, 3)`` float32 in [0, 1].
    """
    if split not in ("all", "train", "val"):
        raise ValueError(f"split must be one of all|train|val, got {split!r}.")
    videos_root = Path(videos_root)
    all_paths = _list_video_files(videos_root)
    if split == "all":
        paths = all_paths
    else:
        order = np.random.default_rng(seed).permutation(len(all_paths))
        n_val = max(1, int(round(len(all_paths) * val_fraction)))
        val_idx = set(order[-n_val:].tolist())
        if split == "val":
            paths = [all_paths[i] for i in sorted(val_idx)]
        else:
            paths = [all_paths[i] for i in range(len(all_paths)) if i not in val_idx]
    logger.info(
        f"bdd100k_video_dataset(split={split}): {len(paths)} of "
        f"{len(all_paths)} video files selected under {videos_root}. "
        f"T={T}, img_size={img_size}, B={batch_size}."
    )

    # Seed numpy globally for both cv2 seek offsets and path selection.
    # (cv2 uses np.random.randint inside _read_clip.)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    def _gen() -> Iterator[Tuple[np.ndarray, np.float32]]:
        while True:
            clip: Optional[np.ndarray] = None
            for _attempt in range(max_resample_attempts):
                idx = int(rng.integers(0, len(paths)))
                clip = _read_clip(paths[idx], T=T, img_size=img_size)
                if clip is not None:
                    break
            if clip is None:
                raise RuntimeError(
                    f"bdd100k_video_dataset: exhausted "
                    f"{max_resample_attempts} resample attempts without "
                    f"finding a readable clip. Check videos_root."
                )
            yield clip, np.float32(0.0)

    output_signature = (
        tf.TensorSpec(shape=(T, img_size, img_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        _gen, output_signature=output_signature,
    )
    ds = ds.map(
        lambda c, y: ({"pixels": c}, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=True)
    if num_steps is not None:
        ds = ds.take(num_steps)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
