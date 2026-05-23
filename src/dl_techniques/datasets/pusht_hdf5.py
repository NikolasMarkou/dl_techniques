"""
PushT HDF5 dataset loader + synthetic LeWM dataset generator.

Two dataset producers for LeWM training:

- :func:`synthetic_lewm_dataset` — random-image smoke-test generator.
  Useful for end-to-end pipeline validation without real data.

- :class:`PushTHDF5Dataset` — thin skeleton for the PushT HDF5 schema
  used by upstream LeWM. Loads once into memory, extracts per-episode
  (history_size + num_preds)-length windows, resizes frames to img_size,
  applies ImageNet normalization, and replaces NaN actions with zeros.
  **This skeleton is not tested on real data** — included per plan as a
  starting point for future work.

Both producers emit `tf.data.Dataset` batches with the same schema:

.. code-block:: python

    ({
        "pixels": (B, T, H, W, C),   # float32, normalized
        "action": (B, T - 1, A),     # float32
     },
     dummy_y)  # zero-scalar placeholder so model.fit with loss=None works

`T = history_size + num_preds` throughout.
"""

from typing import Any, Iterator, Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
except ImportError as _e:
    raise ImportError(
        "PushT HDF5 dataset requires TensorFlow. Install tensorflow>=2.18."
    ) from _e


# ImageNet normalization constants (matches upstream LeWM preprocessing).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------
# Synthetic smoke-test dataset
# ---------------------------------------------------------------------

def synthetic_lewm_dataset(
    num_episodes: int = 64,
    num_steps: int = 4,
    img_size: int = 224,
    action_dim: int = 2,
    batch_size: int = 2,
    history_size: int = 3,
    num_preds: int = 1,
    seed: int = 0,
) -> "tf.data.Dataset":
    """Yield synthetic batches matching the LeWM training schema.

    :param num_episodes: number of pseudo-episodes (dataset length).
    :param num_steps: unused placeholder — kept for future parity.
    :param img_size: frame edge length (square).
    :param action_dim: action vector dimension.
    :param batch_size: mini-batch size.
    :param history_size: temporal context length.
    :param num_preds: temporal prediction horizon.
    :param seed: RNG seed.
    :return: `tf.data.Dataset` batched, prefetched.
    """
    T = history_size + num_preds
    rng = np.random.default_rng(seed)

    def gen() -> Iterator[Tuple[dict, float]]:
        for _ in range(num_episodes):
            # Random uint8 image, normalize with ImageNet stats.
            pixels_u8 = rng.integers(0, 256, size=(T, img_size, img_size, 3), dtype=np.uint8)
            pixels = (pixels_u8.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            action = rng.standard_normal((T - 1, action_dim)).astype(np.float32)
            # Dummy scalar "y" — training loss is via self.add_loss in the model.
            yield ({"pixels": pixels, "action": action}, np.float32(0.0))

    output_signature = (
        {
            "pixels": tf.TensorSpec(shape=(T, img_size, img_size, 3), dtype=tf.float32),
            "action": tf.TensorSpec(shape=(T - 1, action_dim), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=True)
    # Repeat so a `model.fit(steps_per_epoch=..., epochs=...)` whose total
    # step budget exceeds num_episodes does not fail mid-fit with
    # "ran out of data". The sole consumer (train_lewm.py) always passes
    # steps_per_epoch, so an unbounded dataset is safe here.
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# PushT HDF5 dataset skeleton (untested on real data)
# ---------------------------------------------------------------------

class PushTHDF5Dataset:
    """Thin skeleton loader for PushT-style HDF5 datasets used by LeWM.

    **UNTESTED SKELETON** — included as a starting point; not validated on
    real data.

    Expected layout (per upstream):
        ``/pixels`` — uint8 tensor `(N, H0, W0, 3)` stacked across episodes.
        ``/action`` — float tensor `(N, A)` — A = action_dim (2 for PushT).
        ``/episode_ends`` — int tensor marking episode boundaries (exclusive
        ends), length = num_episodes.

    Reads `/action` and `/episode_ends` eagerly (small), then performs
    on-demand h5py-indexed reads of `/pixels` for each (T)-length window
    inside the generator. Per-window indices are shuffled (seeded) before
    iteration so the dataset is not produced in file order.

    Frames are resized (via `tf.image.resize`) to `img_size`,
    ImageNet-normalized; actions with NaN (episode-break markers) are
    replaced with 0.

    :param h5_path: path to the .h5 file.
    :param img_size: target frame size (square).
    :param action_dim: action vector dimension.
    :param history_size: context length.
    :param num_preds: prediction horizon.
    :param frameskip: stride between windows.
    :param batch_size: batch size.
    :param shuffle_seed: seed for the per-epoch window-index shuffle.
    """

    def __init__(
        self,
        h5_path: str,
        img_size: int = 224,
        action_dim: int = 2,
        history_size: int = 3,
        num_preds: int = 1,
        frameskip: int = 1,
        batch_size: int = 2,
        shuffle_seed: int = 0,
    ) -> None:
        self.h5_path = h5_path
        self.img_size = img_size
        self.action_dim = action_dim
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed

    def _load_metadata(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Load small metadata (actions, episode_ends, total N) eagerly.
        /pixels is NOT loaded here — it is indexed on demand per window."""
        import h5py
        with h5py.File(self.h5_path, "r") as f:
            n_pixels = f["pixels"].shape[0]
            action = f["action"][...].astype(np.float32)
            if "episode_ends" in f:
                ends = f["episode_ends"][...].astype(np.int64)
            else:
                ends = np.array([n_pixels], dtype=np.int64)
        action = np.where(np.isnan(action), 0.0, action).astype(np.float32)
        return action, ends, int(n_pixels)

    def _window_starts(self, ends: np.ndarray) -> np.ndarray:
        """Enumerate valid (T)-length window start indices across all episodes."""
        T = self.history_size + self.num_preds
        starts = []
        prev_end = 0
        for end in ends:
            i = prev_end
            while i + T <= end:
                starts.append(i)
                i += self.frameskip
            prev_end = int(end)
        return np.asarray(starts, dtype=np.int64)

    def _preprocess_pair(
        self, pixels_u8: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize + ImageNet-normalize pixels; pass action through."""
        pixels_f = pixels_u8.astype(np.float32) / 255.0
        resized = tf.image.resize(pixels_f, [self.img_size, self.img_size]).numpy()
        normalized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
        return normalized.astype(np.float32), action.astype(np.float32)

    def as_tf_dataset(self) -> "tf.data.Dataset":
        """Build a `tf.data.Dataset` from the HDF5 file (on-demand reads)."""
        import h5py

        action, ends, _n_pixels = self._load_metadata()
        starts = self._window_starts(ends)
        T = self.history_size + self.num_preds

        # Index-level shuffle (deterministic via shuffle_seed). This shuffles
        # window starts once; the generator iterates the shuffled order.
        rng = np.random.default_rng(self.shuffle_seed)

        h5_path = self.h5_path  # close over local

        def gen() -> Iterator[Tuple[dict, float]]:
            # Open the file once per generator instantiation; on-demand
            # indexed reads avoid the eager full-pixels load (a larger
            # HDF5 file would OOM the original implementation).
            order = starts.copy()
            rng.shuffle(order)
            with h5py.File(h5_path, "r") as f:
                pixels_ds = f["pixels"]
                for i in order:
                    win_pixels = pixels_ds[i : i + T]      # (T, H0, W0, 3)
                    win_action = action[i : i + T - 1]
                    p, a = self._preprocess_pair(win_pixels, win_action)
                    yield ({"pixels": p, "action": a}, np.float32(0.0))

        output_signature = (
            {
                "pixels": tf.TensorSpec(
                    shape=(T, self.img_size, self.img_size, 3), dtype=tf.float32
                ),
                "action": tf.TensorSpec(shape=(T - 1, self.action_dim), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        # Repeat — same rationale as synthetic_lewm_dataset: train_lewm.py
        # always drives this with steps_per_epoch, so an unbounded dataset
        # avoids a mid-fit "ran out of data" when the step budget exceeds the
        # number of sliceable windows.
        ds = ds.repeat()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
