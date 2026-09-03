"""Kernel initializers transcribed from the official SqueezeNet Caffe prototxts, for both SqueezeNet families in this package.

Caffe's default `xavier` filler normalizes by `fan_in` only, while Keras'
`glorot_uniform` normalizes by `(fan_in + fan_out) / 2` — a different
distribution, not an approximation of it. `CAFFE_XAVIER_INITIALIZER`
points at `lecun_uniform` instead, which matches Caffe's fan-in-only
normalization; on the v1.0 stem kernel `(7, 7, 3, 96)` the two give
`+-0.1429` versus `+-0.0352`, a 4x narrower `glorot_uniform` draw.
`CAFFE_HEAD_INITIALIZER` holds `conv10`'s `N(0, 0.01)` filler as a
serialized config, so every consumer resolves its own fresh instance
rather than sharing one seeded object.

These initializers only affect a freshly constructed model's starting
weights; loading a checkpoint overwrites every kernel it names, so no
saved artifact is affected by this module.
"""

from typing import Any, Dict

import keras

# Matches Caffe's default xavier filler (fan_in normalization), not Keras' glorot_uniform (fan_avg).
CAFFE_XAVIER_INITIALIZER: str = "lecun_uniform"

# Caffe `weight_filler { type: "gaussian" mean: 0.0 std: 0.01 }` on `conv10`.
CAFFE_HEAD_MEAN: float = 0.0
CAFFE_HEAD_STDDEV: float = 0.01


CAFFE_HEAD_INITIALIZER: Dict[str, Any] = keras.initializers.serialize(
    keras.initializers.RandomNormal(mean=CAFFE_HEAD_MEAN, stddev=CAFFE_HEAD_STDDEV)
)
