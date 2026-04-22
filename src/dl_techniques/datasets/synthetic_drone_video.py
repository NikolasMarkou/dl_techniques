"""Synthetic drone-video dataset generator for Video-JEPA-Clifford smoke training.

Produces ``tf.data.Dataset`` batches matching the VideoJEPA training schema:

.. code-block:: python

    ({
        "pixels":    (B, T, H, W, C),   # float32, in [0, 1]
     },
     dummy_y)  # zero scalar so model.fit with loss=None works

**Scene model.** Each "drone pass" is a short sequence of frames where a
colored moving rectangle translates linearly over a stationary random-noise
background. The goal is smoke-test end-to-end training, not learning
anything meaningful.

Iter-3 (D-013): telemetry emission was dropped alongside the removal of
telemetry conditioning from the model.
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
    T: int, H: int, W: int, C: int, rng: np.random.Generator,
) -> np.ndarray:
    """Render a single pixel sequence of length T.

    :return: ``pixels: (T, H, W, C)`` float32 in [0, 1].
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

    return pixels


def synthetic_drone_video_dataset(
    batch_size: int = 2,
    num_batches: int = 8,
    T: int = 4,
    img_size: int = 64,
    img_channels: int = 3,
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
    :param seed: RNG seed.
    :return: ``tf.data.Dataset`` emitting ``(inputs_dict, zero_scalar)``
        tuples, already batched and prefetched. ``inputs_dict`` has a
        single key ``pixels`` (iter-3: telemetry removed, D-013).
    """
    if batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 (CliffordNetBlock BN stability); "
            f"got {batch_size}."
        )
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    rng = np.random.default_rng(seed)
    num_episodes = batch_size * num_batches

    def gen() -> Iterator[Tuple[dict, float]]:
        for _ in range(num_episodes):
            pixels = _render_sequence(
                T=T, H=img_size, W=img_size, C=img_channels, rng=rng,
            )
            yield (
                {"pixels": pixels},
                np.float32(0.0),
            )

    output_signature = (
        {
            "pixels": tf.TensorSpec(
                shape=(T, img_size, img_size, img_channels),
                dtype=tf.float32,
            ),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=True)
    # Repeat infinitely so model.fit(epochs=N, steps_per_epoch=M) works
    # regardless of num_batches. `steps_per_epoch` is the caller's contract
    # for per-epoch iteration length.
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
