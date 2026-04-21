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
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# PushT HDF5 dataset skeleton (untested on real data)
# ---------------------------------------------------------------------

class PushTHDF5Dataset:
    """Thin skeleton loader for PushT-style HDF5 datasets used by LeWM.

    Expected layout (per upstream):
        ``/pixels`` — uint8 tensor `(N, H0, W0, 3)` stacked across episodes.
        ``/action`` — float tensor `(N, A)` — A = action_dim (2 for PushT).
        ``/episode_ends`` — int tensor marking episode boundaries (exclusive
        ends), length = num_episodes. Optional: ``/proprio``.

    This skeleton reads the whole file into memory (fine for the small
    PushT replay) and slices per-episode windows of length T = history_size +
    num_preds at `frameskip` temporal stride. Frames are resized (via
    `tf.image.resize`) to `img_size`, ImageNet-normalized; actions with
    NaN (episode-break markers) are replaced with 0.

    **Not validated against real data** — included as a starting point per
    plan.md Step 7.

    :param h5_path: path to the .h5 file.
    :param img_size: target frame size (square).
    :param history_size: context length.
    :param num_preds: prediction horizon.
    :param frameskip: stride between windows.
    :param batch_size: batch size.
    :param proprio_key: optional dataset key for proprioceptive state.
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
        proprio_key: Optional[str] = None,
    ) -> None:
        self.h5_path = h5_path
        self.img_size = img_size
        self.action_dim = action_dim
        self.history_size = history_size
        self.num_preds = num_preds
        self.frameskip = frameskip
        self.batch_size = batch_size
        self.proprio_key = proprio_key

    def _load_raw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import h5py  # local import to keep top-level import lightweight
        with h5py.File(self.h5_path, "r") as f:
            pixels = f["pixels"][...]          # (N, H0, W0, 3), uint8
            action = f["action"][...].astype(np.float32)
            if "episode_ends" in f:
                ends = f["episode_ends"][...].astype(np.int64)
            else:
                ends = np.array([pixels.shape[0]], dtype=np.int64)
        # NaN -> 0 in actions (episode-break markers).
        action = np.where(np.isnan(action), 0.0, action).astype(np.float32)
        return pixels, action, ends

    def _slice_windows(
        self, pixels: np.ndarray, action: np.ndarray, ends: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        T = self.history_size + self.num_preds
        start = 0
        for end in ends:
            # Produce (T)-length contiguous windows at stride frameskip.
            i = start
            while i + T <= end:
                win_pixels = pixels[i : i + T]          # (T, H0, W0, 3)
                win_action = action[i : i + T - 1]      # (T-1, A)
                yield win_pixels, win_action
                i += self.frameskip
            start = int(end)

    def _preprocess_pair(
        self, pixels_u8: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize + ImageNet-normalize pixels; pass action through."""
        # Resize: (T, H0, W0, 3) -> (T, img, img, 3) via tf.image.resize (CPU).
        pixels_f = pixels_u8.astype(np.float32) / 255.0
        resized = tf.image.resize(pixels_f, [self.img_size, self.img_size]).numpy()
        normalized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
        return normalized.astype(np.float32), action.astype(np.float32)

    def as_tf_dataset(self) -> "tf.data.Dataset":
        """Build a `tf.data.Dataset` from the HDF5 file."""
        pixels, action, ends = self._load_raw()

        def gen() -> Iterator[Tuple[dict, float]]:
            for win_p, win_a in self._slice_windows(pixels, action, ends):
                p, a = self._preprocess_pair(win_p, win_a)
                yield ({"pixels": p, "action": a}, np.float32(0.0))

        T = self.history_size + self.num_preds
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
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
