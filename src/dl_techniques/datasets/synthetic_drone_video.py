"""Synthetic drone-video dataset generator for Video-JEPA-Clifford smoke training.

Produces ``tf.data.Dataset`` batches matching the VideoJEPA training schema:

.. code-block:: python

    ({
        "pixels":    (B, T, H, W, C),   # float32, in [0, 1]
        "telemetry": (B, T, k),         # float32, weakly correlated with motion
     },
     dummy_y)  # zero scalar so model.fit with loss=None works

**Scene model.** Each "drone pass" is a short sequence of frames where a
colored moving rectangle translates linearly over a stationary random-noise
background. The translation velocity drives a k-dim telemetry vector — this
gives the predictor *some* real coupling between pixels and telemetry
(useful to validate that AdaLN conditioning at least doesn't hurt), but the
scene is deliberately simple: the goal is smoke-test end-to-end training,
not learning anything meaningful.

Mirrors the usage pattern of
:func:`dl_techniques.datasets.pusht_hdf5.synthetic_lewm_dataset`.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "synthetic_drone_video_dataset requires TensorFlow. "
        "Install tensorflow>=2.18."
    ) from _e


def _render_sequence(
    T: int, H: int, W: int, C: int, k: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a single (pixels, telemetry) pair of length T.

    :return: ``pixels: (T, H, W, C)`` float32 in [0, 1];
             ``telemetry: (T, k)`` float32, signed.
    """
    # Stationary random background (same across frames).
    bg = rng.random((H, W, C), dtype=np.float32) * 0.3

    # Rectangle: size, color, initial position, velocity.
    rect_h = max(4, H // 6)
    rect_w = max(4, W // 6)
    color = rng.random((C,), dtype=np.float32) * 0.8 + 0.2
    # Initial top-left, with some margin so the rectangle stays mostly in-frame.
    x0 = float(rng.integers(0, max(1, W - rect_w)))
    y0 = float(rng.integers(0, max(1, H - rect_h)))
    # Velocity in px/frame — small so the full trajectory stays visible.
    vx = float(rng.normal(0.0, 1.5))
    vy = float(rng.normal(0.0, 1.5))

    pixels = np.empty((T, H, W, C), dtype=np.float32)
    telemetry = np.empty((T, k), dtype=np.float32)

    for t in range(T):
        x = int(np.clip(x0 + vx * t, 0, W - rect_w - 1))
        y = int(np.clip(y0 + vy * t, 0, H - rect_h - 1))

        frame = bg.copy()
        frame[y : y + rect_h, x : x + rect_w, :] = color
        # A tiny bit of per-frame noise so SIGReg sees non-constant features.
        frame = frame + rng.normal(0.0, 0.02, size=frame.shape).astype(
            np.float32
        )
        frame = np.clip(frame, 0.0, 1.0)
        pixels[t] = frame

        # Telemetry: first 2 channels = (vx, vy) (signed), 3rd = altitude proxy
        # (constant per sequence), rest = independent noise. Weakly couples
        # pixel motion to telemetry (by construction of vx, vy).
        tel = np.zeros((k,), dtype=np.float32)
        if k >= 1:
            tel[0] = vx
        if k >= 2:
            tel[1] = vy
        if k >= 3:
            tel[2] = float(y0) / float(H)  # altitude-ish proxy
        if k > 3:
            tel[3:] = rng.normal(0.0, 0.1, size=k - 3).astype(np.float32)
        telemetry[t] = tel

    return pixels, telemetry


def synthetic_drone_video_dataset(
    batch_size: int = 2,
    num_batches: int = 8,
    T: int = 4,
    img_size: int = 64,
    img_channels: int = 3,
    telemetry_dim: int = 7,
    seed: int = 0,
) -> "tf.data.Dataset":
    """Yield synthetic drone-video batches for VideoJEPA training.

    :param batch_size: Mini-batch size ``B`` (must be >= 2 for Clifford
        BatchNorm stability — a hard constraint documented in
        ``plans/plan_2026-04-21_421088a1/findings.md``).
    :param num_batches: Number of batches per epoch (before re-init).
    :param T: Frames per clip.
    :param img_size: Square frame edge length.
    :param img_channels: Number of channels.
    :param telemetry_dim: ``k`` — telemetry channels per frame.
    :param seed: RNG seed.
    :return: ``tf.data.Dataset`` emitting ``(inputs_dict, zero_scalar)``
        tuples, already batched and prefetched.
    """
    if batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 (CliffordNetBlock BN stability); "
            f"got {batch_size}."
        )
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if telemetry_dim <= 0:
        raise ValueError(
            f"telemetry_dim must be positive, got {telemetry_dim}"
        )

    rng = np.random.default_rng(seed)
    num_episodes = batch_size * num_batches

    def gen() -> Iterator[Tuple[dict, float]]:
        for _ in range(num_episodes):
            pixels, telemetry = _render_sequence(
                T=T, H=img_size, W=img_size, C=img_channels,
                k=telemetry_dim, rng=rng,
            )
            yield (
                {"pixels": pixels, "telemetry": telemetry},
                np.float32(0.0),
            )

    output_signature = (
        {
            "pixels": tf.TensorSpec(
                shape=(T, img_size, img_size, img_channels),
                dtype=tf.float32,
            ),
            "telemetry": tf.TensorSpec(
                shape=(T, telemetry_dim), dtype=tf.float32,
            ),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
