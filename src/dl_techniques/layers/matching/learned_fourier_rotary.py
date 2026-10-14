"""Learned-Fourier rotary position encoding for LightGlue keypoint matching.

:class:`LearnedFourierRotaryEncoding` turns keypoint coordinates into a
cosine/sine table that self-attention uses to rotate its queries and keys.
Instead of a fixed frequency ladder, the frequencies are learned: a bias-free
linear map projects each keypoint to ``head_dim // 2`` phases. The table is
computed once per image and reused by every self-attention layer. Channels
rotate in adjacent pairs, the interleaved convention of the reference code
and its published weights::

    [t'_{2j}  ]   [cos phi_j  -sin phi_j] [t_{2j}  ]
    [t'_{2j+1}] = [sin phi_j   cos phi_j] [t_{2j+1}]

A split-half layout keeps every shape and silently breaks pretrained weights.
The table comes back in float32, or float64 under a float64 policy, never in
a narrower compute dtype; :func:`apply_rotary_interleaved` casts it at use.

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
    """Map every adjacent channel pair ``(a, b)`` to ``(-b, a)``.

    Matches the reference ``rotate_half``: unflatten the last axis to
    ``(-1, 2)``, stack ``(-x2, x1)``, flatten. The split-half rotation of the
    Llama-style RoPE layers elsewhere in this repository is a different map.

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

    The table is cast to ``t``'s dtype here, because the layer returns it in
    float32 or float64 (see :class:`LearnedFourierRotaryEncoding`).

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
    """Build a learned-Fourier rotation table from keypoints (LightGlue ``posenc``).

    Projects ``(B, N, M)`` keypoints with a trainable bias-free matrix to
    ``head_dim // 2`` phases and returns the stacked cosine/sine table
    ``(2, B, 1, N, head_dim)``. Axis 2 is the head axis, so the table
    broadcasts over heads; one instance is meant to serve every
    self-attention layer of a model.

    The phases and the trigonometry run in float32, or float64 under a
    float64 policy, and the table is returned in that dtype, not the compute
    dtype: cos/sin of a phase rounded to float16 would be off by up to the
    float16 resolution times the phase. :meth:`apply_rotary` casts at use.
    Under a mixed policy Keras autocasts the keypoints and the kernel before
    ``call``, and this layer cannot undo that rounding.

    Table construction:

    .. code-block:: text

        keypoints [B, N, M]
                 │
                 ▼
        cast to work dtype
                 │
                 ▼
        ┌─────────────────────┐
        │ @ kernel [M, D/2]   │  no bias
        └─────────────────────┘
                 │  phases [B, N, D/2]
                 ├────────────────┐
                 ▼                ▼
        cos, repeat 2      sin, repeat 2
                 │                │
                 └───────┬────────┘
                         ▼
        stack on axis 0, expand axis 2
                         │
                         ▼
        table [2, B, 1, N, D]

    D is head_dim; M is num_input_features; repeat is repeat_interleave.

    Channel layout, interleaved:

    .. code-block:: text

        phases       phi0         phi1         ...
        cos slot     c0    c0     c1    c1     ...
        t            t0    t1     t2    t3     ...
        rotate_half  -t1   t0     -t3   t2     ...

    ``tile`` would give ``c0 c1 ... c0 c1``; both keep every shape.

    :param head_dim: Per-head channel count the table is built for. Must be a
        positive even integer; the type is not checked.
    :type head_dim: int
    :param num_input_features: Width ``M`` of the coordinates: 2 for normalised
        xy, 4 when scale and orientation are appended. Must be positive.
    :type num_input_features: int
    :param gamma: Scale of the frequency initialiser. The kernel is drawn from a
        normal with standard deviation ``gamma ** -2`` (reference convention).
        Must be positive.
    :type gamma: float
    :param **kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar kernel: Trainable phase projection, shape ``(num_input_features,
        head_dim // 2)``. A torch ``Wr.weight`` of shape ``(head_dim // 2, M)``
        converts by transpose alone.
    :vartype kernel: keras.Variable

    Input shape:
        ``(batch, num_points, num_input_features)``, normalised keypoints.

    Output shape:
        ``(2, batch, 1, num_points, head_dim)``.

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
        """Validate and store the arguments.

        Arguments are documented on the class.

        :raises ValueError: If ``head_dim`` is not positive and even, or
            ``num_input_features`` or ``gamma`` is not positive.
        """
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

        self.kernel = None

    def build(self, input_shape: Tuple[Any, ...]) -> None:
        """Create the phase projection. Returns at once if already built.

        :param input_shape: ``(batch, num_points, num_input_features)``.
        :type input_shape: Tuple[Any, ...]
        :raises ValueError: If the input is not rank 3 or its last dimension is
            not ``num_input_features``; an unknown (``None``) last dimension
            fails too.
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
        :return: ``(2, B, 1, N, head_dim)`` table in float32, or float64 under
            a float64 policy.
        :rtype: keras.KerasTensor
        """
        # Same rule as ContinuousRoPE: keep float64, never do the trigonometry in
        # float16. The table stays in this dtype.
        work_dtype = "float64" if self.compute_dtype == "float64" else "float32"
        phases = keras.ops.matmul(
            keras.ops.cast(keypoints, work_dtype),
            keras.ops.cast(self.kernel, work_dtype),
        )
        # repeat is repeat_interleave: [c0, c1] -> [c0, c0, c1, c1], not tile.
        cos = keras.ops.repeat(keras.ops.cos(phases), 2, axis=-1)
        sin = keras.ops.repeat(keras.ops.sin(phases), 2, axis=-1)
        # The new axis 2 is the head axis.
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
