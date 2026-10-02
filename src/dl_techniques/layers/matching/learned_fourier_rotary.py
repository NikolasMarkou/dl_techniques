"""
Learned-Fourier rotary position encoding for keypoint matching (LightGlue).

LightGlue does not rotate its queries and keys with a fixed position ladder. It
learns the frequencies. A keypoint ``p`` (normalised xy, optionally with scale
and orientation) is projected by a bias-free linear map to ``head_dim // 2``
phases, and the cosine and sine of those phases form a rotation table that every
self-attention layer reuses. The table is computed once per image.

Architecture:
    ::

        keypoints (B, N, M)
              |
              |  phases = keypoints @ W          W: (M, head_dim // 2), no bias,
              v                                     init N(0, gamma^-4 variance)
        phases (B, N, head_dim // 2)
              |
              |  cos, sin, then repeat_interleave(2) along channels
              v
        table (2, B, 1, N, head_dim)      slot 0 = cos, slot 1 = sin
              |
              |  apply_rotary(table, t) = t * cos + rotate_half(t) * sin
              v
        rotated q / k (B, H, N, head_dim)

Foundational Mathematics:
    The channels of a head are read as ``head_dim / 2`` ADJACENT pairs
    ``(t_{2j}, t_{2j+1})``. Pair ``j`` is rotated by the phase ``phi_j``::

        [t'_{2j}  ]   [cos phi_j  -sin phi_j] [t_{2j}  ]
        [t'_{2j+1}] = [sin phi_j   cos phi_j] [t_{2j+1}]

    This is the INTERLEAVED convention, the one the reference code and its
    published weights use. Two things must agree with it or every pretrained
    weight is silently wrong: the cos/sin tables are built with
    ``repeat_interleave(2)`` (``[c0, c0, c1, c1, ...]``, NEVER ``tile``, which
    gives ``[c0, c1, ..., c0, c1, ...]``), and ``rotate_half`` maps each adjacent
    pair ``(a, b)`` to ``(-b, a)`` (NEVER the split-half ``(-x2, x1)`` of the two
    contiguous halves). Neither mistake changes a shape.

References:
    - Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). "LightGlue: Local
      Feature Matching at Light Speed". arXiv:2306.13643.
    - Li, Y., Si, S., Li, G., Hsieh, C.-J., & Bengio, S. (2021). "Learnable
      Fourier Features for Multi-Dimensional Spatial Positional Encoding".
      arXiv:2106.02795.
    - Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
      Embedding".
"""

import keras
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def rotate_half_interleaved(x: Any) -> Any:
    """Map every ADJACENT channel pair ``(a, b)`` to ``(-b, a)``.

    This is the reference ``rotate_half`` (unflatten the last axis to
    ``(-1, 2)``, stack ``(-x2, x1)``, flatten). It is NOT the split-half rotation
    used by the Llama-style RoPE layers elsewhere in this repository.

    :param x: Tensor whose last dimension is even.
    :type x: Any
    :return: Tensor of the same shape and dtype.
    :rtype: Any
    """
    shape = keras.ops.shape(x)
    pairs = keras.ops.reshape(x, (*shape[:-1], shape[-1] // 2, 2))
    even = pairs[..., 0]
    odd = pairs[..., 1]
    rotated = keras.ops.stack([-odd, even], axis=-1)
    return keras.ops.reshape(rotated, shape)


def apply_rotary_interleaved(freqs: Any, t: Any) -> Any:
    """Apply a cached rotation table: ``t * cos + rotate_half(t) * sin``.

    The table is cast to ``t``'s dtype here, at the point of use, because the
    layer emits it in a never-narrowing work dtype (see
    :class:`LearnedFourierRotaryEncoding`).

    :param freqs: ``(2, ..., N, head_dim)`` table, slot 0 cosine and slot 1 sine,
        each already ``repeat_interleave(2)``. ``freqs[0]`` must broadcast
        against ``t``, e.g. ``(B, 1, N, head_dim)`` against ``(B, H, N, head_dim)``.
    :type freqs: Any
    :param t: Queries or keys, ``(..., N, head_dim)``.
    :type t: Any
    :return: Rotated tensor, same shape and dtype as ``t``.
    :rtype: Any
    """
    cos = keras.ops.cast(freqs[0], t.dtype)
    sin = keras.ops.cast(freqs[1], t.dtype)
    return t * cos + rotate_half_interleaved(t) * sin


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.matching.learned_fourier_rotary")
class LearnedFourierRotaryEncoding(keras.layers.Layer):
    """Learned-Fourier rotation table from keypoint coordinates (LightGlue ``posenc``).

    Projects ``(B, N, M)`` keypoints with a trainable bias-free matrix to
    ``head_dim // 2`` phases and returns the stacked cosine/sine table
    ``(2, B, 1, N, head_dim)`` built with ``repeat_interleave(2)``. The singleton
    axis is the head axis, so the table broadcasts over heads. One instance is
    shared by all self-attention layers of a model.

    The phases and the trigonometry run in a never-narrowing WORK dtype
    (float32, or float64 under a float64 policy) and the table is RETURNED in
    that work dtype, not in the compute dtype: cos/sin of a phase that was first
    rounded to float16 would be wrong by up to the float16 resolution times the
    phase magnitude. Consumers cast at use, which :meth:`apply_rotary` does. The
    keypoints and the kernel reach ``call`` already narrowed by Keras'
    autocast under a mixed policy; that rounding is outside what this layer can
    recover.

    :param head_dim: Per-head channel count the table is built for. Must be a
        positive even integer.
    :type head_dim: int
    :param num_input_features: Width ``M`` of the coordinates: 2 for normalised
        xy, 4 when scale and orientation are appended. Must be positive.
    :type num_input_features: int
    :param gamma: Scale of the frequency initialiser. The kernel is drawn from a
        normal with standard deviation ``gamma ** -2`` (reference convention).
        Must be positive.
    :type gamma: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar kernel: Trainable phase projection, shape ``(num_input_features,
        head_dim // 2)``. A torch ``Wr.weight`` of shape ``(head_dim // 2, M)``
        converts by transpose alone.
    :vartype kernel: keras.Variable

    Input shape:
        ``(batch, num_points, num_input_features)``, normalised keypoints.

    Output shape:
        ``(2, batch, 1, num_points, head_dim)``.

    :raises ValueError: If ``head_dim`` is not a positive even integer, or
        ``num_input_features`` or ``gamma`` is not positive. Raised from
        ``__init__``.
    :raises ValueError: If the input is not rank 3 or its last dimension is not
        ``num_input_features``. Raised from ``build()``.

    Example:

    .. code-block:: python

        enc = LearnedFourierRotaryEncoding(head_dim=64)
        table = enc(keras.random.uniform((2, 100, 2)))      # (2, 2, 1, 100, 64)
        q = keras.random.normal((2, 4, 100, 64))
        q_rot = LearnedFourierRotaryEncoding.apply_rotary(table, q)
    """

    #: ``apply_rotary(freqs, t)``: the interleaved rotation, usable without an instance.
    apply_rotary = staticmethod(apply_rotary_interleaved)

    def __init__(
            self,
            head_dim: int,
            num_input_features: int = 2,
            gamma: float = 1.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        if num_input_features <= 0:
            raise ValueError(f"num_input_features must be positive, got {num_input_features}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.head_dim = head_dim
        self.num_input_features = num_input_features
        self.gamma = gamma

        # Created in build()
        self.kernel = None

    def build(self, input_shape: Tuple[Any, ...]) -> None:
        """Create the phase projection.

        :param input_shape: ``(batch, num_points, num_input_features)``.
        :type input_shape: Tuple[Any, ...]
        :raises ValueError: If the input is not rank 3 or its last dimension is not
            ``num_input_features``.
        """
        if self.built:
            return
        if len(input_shape) != 3:
            raise ValueError(f"Input must be (batch, num_points, features), got {input_shape}")
        if input_shape[-1] != self.num_input_features:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match "
                f"num_input_features ({self.num_input_features})"
            )

        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.num_input_features, self.head_dim // 2),
            initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=self.gamma ** -2
            ),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, keypoints: keras.KerasTensor) -> keras.KerasTensor:
        """Build the interleaved cosine/sine table.

        :param keypoints: Normalised keypoints ``(B, N, M)``.
        :type keypoints: keras.KerasTensor
        :return: ``(2, B, 1, N, head_dim)`` table in the work dtype.
        :rtype: keras.KerasTensor
        """
        # Never-narrowing work dtype, same rule as ContinuousRoPE: a hard
        # `cast(.., "float32")` would break float64, a cast to the compute dtype would
        # do the trigonometry in float16. The table stays in the work dtype.
        work_dtype = "float64" if self.compute_dtype == "float64" else "float32"
        phases = keras.ops.matmul(
            keras.ops.cast(keypoints, work_dtype),
            keras.ops.cast(self.kernel, work_dtype),
        )
        # repeat is repeat_interleave: [c0, c1] -> [c0, c0, c1, c1]. NOT tile.
        cos = keras.ops.repeat(keras.ops.cos(phases), 2, axis=-1)
        sin = keras.ops.repeat(keras.ops.sin(phases), 2, axis=-1)
        # (2, B, N, Dh) -> (2, B, 1, N, Dh): the singleton is the head axis.
        return keras.ops.expand_dims(keras.ops.stack([cos, sin], axis=0), axis=2)

    def compute_output_shape(self, input_shape: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Return ``(2, batch, 1, num_points, head_dim)``.

        :param input_shape: ``(batch, num_points, num_input_features)``.
        :type input_shape: Tuple[Any, ...]
        :return: Output shape.
        :rtype: Tuple[Any, ...]
        """
        return (2, input_shape[0], 1, input_shape[1], self.head_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Dictionary with ``head_dim``, ``num_input_features`` and ``gamma``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "num_input_features": self.num_input_features,
            "gamma": self.gamma,
        })
        return config

# ---------------------------------------------------------------------
