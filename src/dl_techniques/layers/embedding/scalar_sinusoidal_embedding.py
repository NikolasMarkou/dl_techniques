"""
Sinusoidal embedding of a scalar, refined by a small SiLU MLP.

This module provides :class:`ScalarSinusoidalEmbedding`, the Keras port of
Ideogram4's ``Ideogram4EmbedScalar``. One scalar per token, typically a
diffusion timestep in ``[0, 1]``, is mapped to a ``dim``-wide sinusoidal
feature vector and then passed through ``Dense -> SiLU -> Dense``. Both
Dense layers use a bias and both are ``dim`` wide, so the output width
equals ``dim`` whatever the input rank.

Architecture:
    The scalar is first rescaled from its declared ``input_range`` onto
    ``[0, 1e4]``. That rescaled value is the position fed to a fixed
    sinusoidal ladder of ``dim // 2`` frequencies. Sine and cosine of the
    ladder are concatenated, giving ``2 * (dim // 2)`` channels, and an odd
    ``dim`` gets one trailing zero so the width is exactly ``dim``. The MLP
    then mixes those channels.

    Two separate factors of ``1e4`` are involved and they are NOT the same
    number used twice. The OUTER ``1e4`` is the range rescale. The INNER
    ``1e4`` is the base of the frequency ladder. Both appear in the PyTorch
    reference and both are reproduced here. Do not collapse them.

Foundational Mathematics:
    With ``half = dim // 2``, the frequency ladder is a geometric
    progression that spans four decades::

        freq_i = exp(-i * log(1e4) / (half - 1))    for i in [0, half)

    so ``freq_0 = 1`` and ``freq_{half-1} = 1e-4``. For a rescaled scalar
    ``s`` the embedding is::

        e_i     = s * freq_i
        emb     = concat([sin(e), cos(e)])          width 2 * half
        emb     = pad(emb, one trailing zero)       only when dim is odd

    The division by ``half - 1`` is why ``dim`` must be at least 4. At
    ``dim = 2`` or ``dim = 3`` the ladder divides by zero and every
    frequency becomes NaN, so the constructor rejects those widths.

Serialization:
    ``freq`` is a constant, but it is stored with
    ``add_weight(trainable=False)`` rather than as a plain attribute. A
    plain tensor attribute does not round-trip through ``.keras`` save and
    load. That was the legacy ``TimestepEmbedding`` bug. The values are
    computed with NumPy in :meth:`build` and installed by a constant
    initializer.

PyTorch reference, ported faithfully::

    def _sinusoidal_embedding(t, dim, scale=1e4):
        half = dim // 2
        freq = math.log(scale) / (half - 1)
        freq = torch.exp(torch.arange(half).float() * -freq)
        emb = t.unsqueeze(-1) * freq
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1: emb = F.pad(emb, (0, 1))
        return emb

    class Ideogram4EmbedScalar(nn.Module):
        def __init__(self, dim, input_range):
            self.range_min, self.range_max = input_range
            self.mlp_in = nn.Linear(dim, dim)
            self.mlp_out = nn.Linear(dim, dim)
        def forward(self, x):
            scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
            emb = _sinusoidal_embedding(scaled, self.dim)
            return self.mlp_out(F.silu(self.mlp_in(emb)))
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------

# Base of the frequency ladder, matching the PyTorch reference's
# ``scale=1e4`` inside ``_sinusoidal_embedding``. The same constant also
# serves as the outer rescale target in ``call()``. The two uses are
# separate factors that happen to share a value. Do not fold them.
_SINUSOID_SCALE: float = 1e4


@register_dl_technique("dl_techniques.layers.embedding.scalar_sinusoidal_embedding")
class ScalarSinusoidalEmbedding(keras.layers.Layer):
    """Embed one scalar per token, then refine it with a SiLU MLP.

    Takes a scalar per token, typically a diffusion timestep, and returns a
    ``dim``-wide feature vector. The scalar is rescaled from ``input_range``
    onto ``[0, 1e4]``, mapped through a fixed sinusoidal ladder of
    ``dim // 2`` frequencies, and then mixed by ``Dense -> SiLU -> Dense``.
    Both Dense layers are ``dim`` wide, so the output width is always
    ``dim``.

    A trailing axis of size 1 is squeezed, so ``(batch, 1)`` and
    ``(batch,)`` give the same ``(batch, dim)`` result. Any other trailing
    size is treated as an extra token axis, not as a squeeze candidate.

    The frequency vector is a constant, but it is held as a non-trainable
    weight rather than a plain attribute. A plain attribute does not
    round-trip through ``.keras`` save and load. That was the legacy
    ``TimestepEmbedding`` bug.

    **Architecture Overview:**

    .. code-block:: text

        x   (..., 1) or (...,)   one scalar per token
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  squeeze a trailing axis of size 1   │
        │  cast to float32                     │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  s = 1e4 * (x - min) / (max - min)   │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  half = dim // 2 frequency bands     │
        │  freq_i = exp(-i*log(1e4)/(half-1))  │
        │  e = s[..., None] * freq             │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  concat(sin(e), cos(e))              │
        │  -> (..., 2 * half)                  │
        │  one trailing zero if dim is odd     │
        │  -> (..., dim)                       │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Dense(dim) -> SiLU -> Dense(dim)    │
        └──────────────────────────────────────┘
                           │
                           ▼
        out (..., dim)

    :param dim: Output width, and the width of the sinusoidal basis. Must
        be at least 4, because the frequency ladder divides by
        ``dim // 2 - 1``.
    :type dim: int
    :param input_range: ``(min, max)`` bounds used to rescale the scalar
        before the sinusoidal map. ``max`` must be strictly greater than
        ``min``. Defaults to ``(0.0, 1.0)``.
    :type input_range: Sequence[float]
    :param kwargs: Additional keyword arguments for the ``Layer`` base
        class.

    :ivar half: ``dim // 2``, the number of frequency bands.
    :vartype half: int

    Input shape:
        Any shape ending in a scalar axis, for example ``(batch,)`` or
        ``(batch, 1)`` or ``(batch, tokens, 1)``.

    Output shape:
        The input shape with the trailing size-1 axis dropped, if present,
        and ``dim`` appended.

    :raises ValueError: If ``dim`` is below 4. A smaller width makes the
        ladder divide by zero and every frequency becomes NaN. Raised from
        ``__init__``.
    :raises ValueError: If ``input_range`` is not a pair, or if its maximum
        is not strictly above its minimum. Raised from ``__init__``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding import (
            scalar_sinusoidal_embedding as sse,
        )

        layer = sse.ScalarSinusoidalEmbedding(dim=64)
        t = keras.ops.convert_to_tensor([[0.0], [0.5], [1.0]])
        layer(t).shape  # (3, 64)
    """

    def __init__(
            self,
            dim: int,
            input_range: Sequence[float] = (0.0, 1.0),
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the two Dense sub-layers.

        The frequency weight is not created here. :meth:`build` computes and
        installs it.

        :param dim: Output width, and the width of the sinusoidal basis.
        :type dim: int
        :param input_range: ``(min, max)`` bounds of the scalar input.
        :type input_range: Sequence[float]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``dim`` is below 4, if ``input_range`` is not
            a pair, or if its maximum is not strictly above its minimum.
        """
        super().__init__(**kwargs)

        if dim < 4:
            # half = dim // 2 must be at least 2. The frequency schedule
            # divides by (half - 1), so dim < 4 gives half <= 1, a division
            # by zero, and a NaN frequency weight.
            raise ValueError(f"dim must be >= 4 (got {dim}); the sinusoidal "
                             f"frequency schedule needs dim // 2 >= 2.")
        if len(input_range) != 2:
            raise ValueError(
                f"input_range must have length 2, got {len(input_range)}"
            )
        range_min, range_max = float(input_range[0]), float(input_range[1])
        if range_max <= range_min:
            raise ValueError(
                f"input_range max must be > min, got "
                f"(min={range_min}, max={range_max})"
            )

        self.dim = int(dim)
        self.range_min = range_min
        self.range_max = range_max
        self.half = self.dim // 2

        # Sub-layers created in __init__ (built in build()).
        self.mlp_in = keras.layers.Dense(self.dim, use_bias=True, name="mlp_in")
        self.mlp_out = keras.layers.Dense(self.dim, use_bias=True, name="mlp_out")

        # Non-trainable frequency weight, materialized in build().
        self.freq = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the frequency weight and build both Dense sub-layers.

        :param input_shape: Shape of the scalar input, with or without a
            trailing axis of size 1.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # DECISION plan_2026-06-12_59a18a10/D-002
        # Store the sinusoidal frequencies as a NON-TRAINABLE WEIGHT, with
        # the values computed by NumPy here. Do NOT revert to a plain
        # attribute such as `self.freq = ops.exp(...)`. A plain tensor
        # attribute does not round-trip through `.keras` save and load; that
        # was the legacy TimestepEmbedding bug this layer exists to avoid.
        # The originating plan directory is gone, so this comment is the
        # only record of the rationale. Do not delete it.
        freq_np = np.exp(
            np.arange(self.half, dtype="float32")
            * -(np.log(_SINUSOID_SCALE) / (self.half - 1))
        )
        self.freq = self.add_weight(
            name="freq",
            shape=(self.half,),
            initializer=keras.initializers.Constant(freq_np),
            trainable=False,
            dtype="float32",
        )

        # The MLP operates on the (..., dim) sinusoidal embedding.
        emb_shape = tuple(input_shape[:-1]) + (self.dim,) \
            if (len(input_shape) > 0 and input_shape[-1] == 1) \
            else tuple(input_shape) + (self.dim,)
        self.mlp_in.build(emb_shape)
        self.mlp_out.build(tuple(emb_shape[:-1]) + (self.dim,))

        super().build(input_shape)

    def _sinusoidal(self, scaled: keras.KerasTensor) -> keras.KerasTensor:
        """Map an already-rescaled scalar onto the sinusoidal basis.

        :param scaled: Rescaled scalar of shape ``(...,)``.
        :type scaled: keras.KerasTensor
        :return: Sinusoidal embedding of shape ``(..., dim)``.
        :rtype: keras.KerasTensor
        """
        # Shape (..., half).
        e = keras.ops.expand_dims(scaled, axis=-1) * self.freq
        # Shape (..., 2 * half).
        emb = keras.ops.concatenate(
            [keras.ops.sin(e), keras.ops.cos(e)], axis=-1
        )
        if self.dim % 2 == 1:
            # An odd dim leaves the width one short. Pad one trailing zero.
            rank = len(emb.shape)
            pad_width = [(0, 0)] * (rank - 1) + [(0, 1)]
            emb = keras.ops.pad(emb, pad_width)
        return emb

    def call(
            self,
            x: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Embed the scalar input and refine it with the MLP.

        :param x: Scalar input, shaped ``(...,)`` or ``(..., 1)``.
        :type x: keras.KerasTensor
        :param training: Training flag forwarded to both Dense sub-layers.
        :type training: Optional[bool]
        :return: Embedding of shape ``(..., dim)``.
        :rtype: keras.KerasTensor
        """
        # Accept both (...,) and (..., 1) by squeezing a trailing singleton.
        # Read the STATIC rank with `len(x.shape)`. The rank is known at
        # trace time and `x.shape` is a static TensorShape. Do NOT switch to
        # `len(keras.ops.shape(x))`. That calls len() on a symbolic shape
        # and is not graph-safe. The squeeze semantics are the same either
        # way.
        if len(x.shape) > 0 and x.shape[-1] == 1:
            x = keras.ops.squeeze(x, axis=-1)

        x = keras.ops.cast(x, "float32")
        scaled = _SINUSOID_SCALE * (x - self.range_min) / (
            self.range_max - self.range_min
        )
        emb = self._sinusoidal(scaled)
        h = keras.activations.silu(self.mlp_in(emb, training=training))
        return self.mlp_out(h, training=training)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Report the output shape for a given input shape.

        A trailing axis of size 1 is dropped, then ``dim`` is appended.

        :param input_shape: Shape of the scalar input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape ending in ``dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        if len(input_shape) > 0 and input_shape[-1] == 1:
            return tuple(input_shape[:-1]) + (self.dim,)
        return tuple(input_shape) + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary carrying every ``__init__``
            argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "input_range": (self.range_min, self.range_max),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ScalarSinusoidalEmbedding":
        """Rebuild the layer, restoring ``input_range`` as a tuple.

        :param config: Configuration produced by :meth:`get_config`, after a
            JSON round trip.
        :type config: Dict[str, Any]
        :return: A new layer with the stored configuration.
        :rtype: ScalarSinusoidalEmbedding
        """
        # JSON turns the `input_range` tuple into a list. Coerce it back so
        # the reconstruction argument matches what get_config emitted. The
        # constructor accepts any Sequence, so this is a shape guarantee for
        # the caller rather than a functional fix.
        config = dict(config)
        if "input_range" in config and config["input_range"] is not None:
            config["input_range"] = tuple(config["input_range"])
        return cls(**config)
