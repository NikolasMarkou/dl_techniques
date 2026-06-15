"""
Scalar (time) sinusoidal embedding with a SiLU MLP head for the Ideogram4 DiT.

This layer ports the Ideogram4 ``Ideogram4EmbedScalar`` module: a single scalar
per token (e.g. a diffusion timestep in ``[0, 1]``) is mapped to a sinusoidal
position embedding of dimensionality ``dim`` and then refined by a two-layer
MLP with a SiLU non-linearity (``Dense -> SiLU -> Dense``, both with bias).

Architecture::

    x      : (..., 1) or (...,)  scalar per token
    scaled = 1e4 * (x - range_min) / (range_max - range_min)
    emb    = sinusoidal(scaled, dim)          # (..., dim)
    out    = Dense(dim)(SiLU(Dense(dim)(emb))) # (..., dim)

Sinusoidal embedding (``scale = 1e4`` is a fixed internal constant, distinct
from the OUTER ``1e4`` rescale above — both factors are present in the PyTorch
reference and are replicated exactly here, NOT collapsed)::

    half = dim // 2
    freq = exp(arange(half) * -(log(1e4) / (half - 1)))   # (half,)
    e    = scaled[..., None] * freq                        # (..., half)
    emb  = concat([sin(e), cos(e)], axis=-1)               # (..., 2*half)
    if dim is odd: emb = pad(emb, last_dim += 1)           # trailing zero

The frequency vector ``freq`` is a CONSTANT derived from ``dim`` and the fixed
``scale=1e4``. It is stored via ``add_weight(trainable=False)`` (values computed
with numpy at ``build()``) so it survives ``.keras`` serialization. This fixes
the known ``TimestepEmbedding`` bug, where the frequencies were kept as a plain
tensor attribute and did NOT round-trip through save/load.

PyTorch reference (faithfully ported)::

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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# Fixed internal sinusoidal scale (PyTorch ``scale=1e4`` inside
# ``_sinusoidal_embedding``). This is distinct from the OUTER ``1e4`` applied to
# the rescaled input; both are intentionally present.
_SINUSOID_SCALE: float = 1e4


@keras.saving.register_keras_serializable(package="dl_techniques.layers")
class ScalarSinusoidalEmbedding(keras.layers.Layer):
    """Sinusoidal embedding of a scalar input followed by a SiLU MLP.

    Maps a scalar per token (typically a diffusion timestep in ``input_range``)
    to a ``dim``-dimensional embedding via a fixed sinusoidal basis and a
    learnable two-layer MLP (``Dense -> SiLU -> Dense``).

    The sinusoidal frequency vector is a constant derived from ``dim`` and a
    fixed internal scale of ``1e4``. It is stored as a non-trainable weight via
    ``add_weight(trainable=False)`` so it survives ``.keras`` serialization
    (fixing the prior ``TimestepEmbedding`` round-trip bug).

    :param dim: Output (and sinusoidal) dimensionality. Must be ``>= 2``.
    :type dim: int
    :param input_range: ``(min, max)`` range of the scalar input used to rescale
        it before the sinusoidal map. ``max`` must be strictly greater than
        ``min``.
    :type input_range: Sequence[float]
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``dim < 2``.
    :raises ValueError: If ``input_range`` does not have length 2 or
        ``input_range[1] <= input_range[0]``.

    Example:
        >>> layer = ScalarSinusoidalEmbedding(dim=64, input_range=(0.0, 1.0))
        >>> t = keras.ops.convert_to_tensor([[0.0], [0.5], [1.0]])
        >>> out = layer(t)
        >>> out.shape
        TensorShape([3, 64])
    """

    def __init__(
            self,
            dim: int,
            input_range: Sequence[float] = (0.0, 1.0),
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
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
        if self.built:
            return

        # DECISION plan_2026-06-12_59a18a10/D-002: store the sinusoidal
        # frequencies as a NON-TRAINABLE weight (values computed with numpy
        # here), NOT as a plain tensor attribute. The plain-attr form (the
        # legacy TimestepEmbedding bug) does not round-trip through
        # .keras save/load. Do NOT revert to `self.freq = ops.exp(...)`.
        # See decisions.md D-002.
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
        """Compute the sinusoidal embedding of an already-rescaled scalar.

        :param scaled: Tensor of shape ``(...,)`` (the rescaled scalar input).
        :type scaled: keras.KerasTensor
        :returns: Sinusoidal embedding of shape ``(..., dim)``.
        :rtype: keras.KerasTensor
        """
        e = keras.ops.expand_dims(scaled, axis=-1) * self.freq  # (..., half)
        emb = keras.ops.concatenate(
            [keras.ops.sin(e), keras.ops.cos(e)], axis=-1
        )  # (..., 2*half)
        if self.dim % 2 == 1:
            # Pad the last dim by one trailing zero (odd dim case).
            rank = len(emb.shape)
            pad_width = [(0, 0)] * (rank - 1) + [(0, 1)]
            emb = keras.ops.pad(emb, pad_width)
        return emb

    def call(
            self,
            x: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        # Accept both (...,) and (..., 1) — squeeze a trailing singleton.
        # Use the STATIC rank (`len(x.shape)`): the rank is known at trace time
        # and `x.shape` is a static TensorShape. Do NOT use
        # `len(keras.ops.shape(x))` — that calls len() on a symbolic shape,
        # which is not graph-safe. The squeeze semantics are identical: a
        # trailing dim of size 1 is squeezed iff present.
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
        # (..., [1]) -> (..., dim)
        if len(input_shape) > 0 and input_shape[-1] == 1:
            return tuple(input_shape[:-1]) + (self.dim,)
        return tuple(input_shape) + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "input_range": (self.range_min, self.range_max),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ScalarSinusoidalEmbedding":
        # JSON round-trips the `input_range` tuple to a list. The guide
        # requires explicit coercion back to a tuple so the reconstruction
        # arg matches what get_config emitted (the ctor accepts a Sequence,
        # so this is robustness, not a functional fix).
        config = dict(config)
        if "input_range" in config and config["input_range"] is not None:
            config["input_range"] = tuple(config["input_range"])
        return cls(**config)
