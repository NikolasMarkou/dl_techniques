"""Sinusoidal timestep embedding with a ``Dense -> SiLU -> Dense`` head.

This module provides :class:`TimestepEmbedding`, the diffusion-transformer
timestep embedder of Peebles & Xie's DiT -- itself OpenAI's GLIDE
``timestep_embedding`` followed by a two-layer MLP. It maps a scalar timestep
per batch element onto a ``hidden_size`` conditioning vector that the model's
adaLN modulation consumes.

Architecture:

.. code-block:: text

    t  [B]   raw timesteps, integer or fractional. NO rescale happens here.
      │
      ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ freqs = exp(-log(max_period) * arange(half) / half)           │
    │   half = frequency_embedding_size // 2                        │
    │   a NON-TRAINABLE weight, materialized from NumPy             │
    └──────────────────────────────────────────────────────────────┘
      │
      ▼
    args = t[:, None] * freqs[None]                       [B, half]
      │
      ├────────────► cos(args)  [B, half] ──┐
      │                                      ⊕ concat, COS FIRST
      └────────────► sin(args)  [B, half] ──┘
                                             │
                                             ▼
                          basis  [B, 2*half]  (one trailing ZERO
                                               column when the width
                                               is odd)
                                             │
                                             ▼
             ┌────────────────────────────────────────────┐
             │ Dense(hidden_size)  RandomNormal(0.02)     │
             │            │                               │
             │            ▼   SiLU                        │
             │ Dense(hidden_size)  RandomNormal(0.02)     │
             └────────────────────────────────────────────┘
                                             │
                                             ▼
                                     t_emb  [B, hidden_size]

Three numerics that look like style and are not:
    1. **The ladder denominator is ``half``, not ``half - 1``.** Dividing by
       ``half - 1`` makes the last frequency land exactly on
       ``1 / max_period``; upstream never reaches it. Same ``max_period``,
       different ladder, no shape symptom.
    2. **The concat is ``[cos, sin]``, not ``[sin, cos]``.** The swap is a
       column permutation: shape, norm, finiteness and ``.keras`` round-trip
       are all blind to it, and the model then trains on a basis no published
       checkpoint uses. Note that
       :func:`~dl_techniques.layers.embedding.sincos_pos_embed_2d.get_1d_sincos_pos_embed_from_grid`
       is deliberately sin-first -- the two are specified independently
       upstream and must NOT be unified.
    3. **``frequency_embedding_size`` is independent of ``hidden_size``.**
       Upstream uses 256 for every variant while ``hidden_size`` ranges over
       384..1152, so the MLP is ``256 -> hidden -> hidden``.

Why this is not
:class:`~dl_techniques.layers.embedding.scalar_sinusoidal_embedding.ScalarSinusoidalEmbedding`:
    That layer looks like a drop-in and is not, on all three numeric axes above
    plus a fourth: it rescales its input onto ``[0, 1e4]`` before the
    sinusoidal map. The divergences were MEASURED (see
    ``bit_diffusion/blocks.py:494-527``, which records
    ``max|delta| = 1.0`` for the concat order, ``0.0534`` for the ladder and
    ``1.88`` for the rescale). Do not substitute one for the other.

Relationship to ``bit_diffusion``:
    ``DiTXATimestepEmbedder``
    (``src/dl_techniques/models/vision_language/bit_diffusion/blocks.py:454-721``)
    is a bit-exact sibling of this class, model-package-private. This module is
    the shared promotion of it (plan ``plan-2026-09-02T170923-1285ed83``
    D-001); ``bit_diffusion`` was deliberately left untouched because moving a
    registered class changes its serialization key. The duplication is
    recorded, and
    ``tests/test_layers/test_embedding/test_timestep_embedding.py``
    cross-checks the two implementations elementwise so a drift reddens.

Serialization:
    ``get_config()`` returns every constructor argument under the names
    ``__init__`` takes. The frequency ladder is a non-trainable weight
    installed through a ``Constant`` initializer, so it round-trips normally --
    it is deliberately NOT a plain tensor attribute (which does not survive a
    ``.keras`` round trip) and deliberately NOT an ``add_weight`` +
    ``.assign()`` inside ``build()`` (``StatelessScope`` discards the assign
    and the table stays all zeros in every real model).

References:
    - Peebles & Xie, 2022. Scalable Diffusion Models with Transformers.
      (https://arxiv.org/abs/2212.09748)
    - Nichol et al., 2021. GLIDE: Towards Photorealistic Image Generation and
      Editing with Text-Guided Diffusion Models.
      (https://arxiv.org/abs/2112.10741) -- the origin of the sinusoidal
      ``timestep_embedding`` helper.
    - Ho et al., 2020. Denoising Diffusion Probabilistic Models.
      (https://arxiv.org/abs/2006.11239)
"""

import math
from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

__all__ = ["TimestepEmbedding"]


# DECISION plan-2026-09-02T170923-1285ed83/D-001
# A deliberate promotion, NOT a duplicate to delete. Do NOT replace this with a
# cross-package import of `bit_diffusion`'s `DiTXATimestepEmbedder` (models ->
# models inverts the dependency direction), do NOT move that class here (its
# registration key would change), and do NOT alias it to
# `ScalarSinusoidalEmbedding`, which differs on three measured numerics.
# decisions.md D-001.
@register_dl_technique(
    package="dl_techniques.layers.embedding.timestep_embedding"
)
class TimestepEmbedding(keras.layers.Layer):
    """Embed scalar timesteps into ``hidden_size``-dimensional vectors.

    A cos-first sinusoidal basis of width ``frequency_embedding_size``,
    refined by ``Dense -> SiLU -> Dense``. Ported from upstream
    ``TimestepEmbedder`` (``reference/models.py:21-57``).

    .. code-block:: text

        t  [B]
          │
          ▼
        ┌────────────────────────────────────────────┐
        │ freqs = exp(-log(max_period)               │
        │              * arange(half) / half)        │  half = F // 2
        │ NON-TRAINABLE weight, F = frequency_       │
        │ embedding_size                             │
        └────────────────────────────────────────────┘
          │
          ▼
        args = t[:, None] * freqs[None]        [B, half]
          │
          ▼
        cos(args)  ⊕  sin(args)                [B, F]   COS FIRST
          │
          ▼
        ┌────────────────────────────────────────────┐
        │ Dense(hidden_size) → SiLU → Dense(hidden)  │
        └────────────────────────────────────────────┘
          │
          ▼
        t_emb  [B, hidden_size]

    :param hidden_size: Output width, and the width of both Dense layers.
    :type hidden_size: int
    :param frequency_embedding_size: Width of the sinusoidal basis feeding the
        MLP. Deliberately independent of ``hidden_size``; upstream's default is
        256 for every variant.
    :type frequency_embedding_size: int
    :param max_period: Base of the frequency ladder. Upstream's ``10000``.
    :type max_period: float
    :param kernel_stddev: Standard deviation of the ``RandomNormal`` kernel
        initializer of both Dense layers, upstream's
        ``nn.init.normal_(std=0.02)`` (``reference/models.py:187-189``). A
        **fresh** initializer instance is constructed per Dense: one shared
        instance draws bit-identically forever.
    :type kernel_stddev: float
    :param kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    :ivar hidden_size: Output width.
    :ivar frequency_embedding_size: Width of the sinusoidal basis.
    :ivar max_period: Base of the frequency ladder.
    :ivar kernel_stddev: Dense kernel initializer standard deviation.
    :ivar half: ``frequency_embedding_size // 2``, the number of frequencies.
    :ivar freqs: Non-trainable ``(half,)`` frequency ladder, materialized in
        :meth:`build` from NumPy through a ``Constant`` initializer.
    :ivar mlp_in: First ``Dense(hidden_size)``.
    :ivar mlp_out: Second ``Dense(hidden_size)``.

    :raises ValueError: If ``hidden_size`` is not positive, if
        ``frequency_embedding_size`` is less than 2 (no frequency ladder
        exists), if ``max_period`` is not greater than 1, or if
        ``kernel_stddev`` is not positive.

    Input shape:
        ``(B,)`` or ``(B, 1)``. A trailing singleton axis is squeezed.

    Output shape:
        ``(B, hidden_size)``.

    Example:
        >>> import keras
        >>> emb = TimestepEmbedding(hidden_size=64, frequency_embedding_size=32)
        >>> t = keras.ops.convert_to_tensor([0.0, 250.0, 999.0])
        >>> emb(t).shape
        (3, 64)
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: float = 10000.0,
        kernel_stddev: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if frequency_embedding_size < 2:
            raise ValueError(
                "frequency_embedding_size must be at least 2 (half = "
                "frequency_embedding_size // 2 must be positive), got "
                f"{frequency_embedding_size}"
            )
        if max_period <= 1.0:
            raise ValueError(
                "max_period must be greater than 1 so that log(max_period) is "
                f"positive and the ladder decreases, got {max_period}"
            )
        if kernel_stddev <= 0.0:
            raise ValueError(
                f"kernel_stddev must be positive, got {kernel_stddev}"
            )

        self.hidden_size = int(hidden_size)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.max_period = float(max_period)
        self.kernel_stddev = float(kernel_stddev)
        self.half = self.frequency_embedding_size // 2

        # A FRESH RandomNormal per Dense. Sharing one instance across layers
        # makes every one of them draw the same numbers forever, and no default
        # exposes it.
        self.mlp_in = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=self.kernel_stddev
            ),
            name="mlp_in",
        )
        self.mlp_out = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=self.kernel_stddev
            ),
            name="mlp_out",
        )

        self.freqs = None

    def build(self, input_shape: Any) -> None:
        """Materialize the frequency ladder and build both Dense sub-layers.

        :param input_shape: ``(B,)`` or ``(B, 1)``.
        :type input_shape: Any
        """
        if self.built:
            return

        # Computed with NumPy, installed as a NON-TRAINABLE WEIGHT through a
        # constant initializer. Do NOT replace this with
        # `self.freqs = keras.ops.exp(...)` in __init__ or build: a plain
        # tensor attribute does not survive a `.keras` round trip, and an
        # `.assign()` inside build() is DISCARDED by StatelessScope, leaving
        # the table all zeros in every real model. Both are recorded repo
        # failures.
        #
        # The denominator is `self.half`, NOT `self.half - 1`. See the module
        # docstring: `half - 1` is the house `ScalarSinusoidalEmbedding`
        # convention and produces a different ladder with no shape symptom.
        freqs_np = np.exp(
            -math.log(self.max_period)
            * np.arange(self.half, dtype="float32")
            / self.half
        )
        self.freqs = self.add_weight(
            name="freqs",
            shape=(self.half,),
            initializer=keras.initializers.Constant(freqs_np),
            trainable=False,
            dtype="float32",
        )

        batch = tuple(input_shape)[0] if len(tuple(input_shape)) > 0 else None
        self.mlp_in.build((batch, self.frequency_embedding_size))
        self.mlp_out.build((batch, self.hidden_size))

        super().build(input_shape)

    def timestep_embedding(self, t: keras.KerasTensor) -> keras.KerasTensor:
        """Map a scalar timestep onto the sinusoidal basis, cos first.

        Exposed as a method so a guard can compare the BASIS against another
        implementation's basis without the two Dense layers in the way.

        :param t: Timesteps, shape ``(B,)``.
        :type t: keras.KerasTensor
        :return: ``(B, frequency_embedding_size)``.
        :rtype: keras.KerasTensor
        """
        args = (
            keras.ops.expand_dims(keras.ops.cast(t, "float32"), axis=-1)
            * self.freqs
        )
        # COS FIRST (`reference/models.py:48`). Do NOT swap to sin-first and do
        # NOT unify with the sin-first 1D helper in `sincos_pos_embed_2d`: the
        # swap is a column permutation, so shape/norm/round-trip are blind and
        # the model trains on a basis no checkpoint uses.
        embedding = keras.ops.concatenate(
            [keras.ops.cos(args), keras.ops.sin(args)], axis=-1
        )
        if self.frequency_embedding_size % 2:
            # An odd width leaves the basis one column short; upstream pads one
            # trailing ZERO column, it does not drop a frequency.
            embedding = keras.ops.pad(embedding, [(0, 0), (0, 1)])
        return embedding

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Embed the timestep.

        :param inputs: Timesteps, ``(B,)`` or ``(B, 1)``. **No rescale happens
            here**: the value is fed to the sinusoidal basis as supplied.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to both Dense sub-layers.
        :type training: Optional[bool]
        :return: ``(B, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        t = inputs
        # Static rank read: `len(t.shape)` is known at trace time.
        # `len(keras.ops.shape(t))` is not graph-safe.
        if len(t.shape) > 1 and t.shape[-1] == 1:
            t = keras.ops.squeeze(t, axis=-1)
        t_freq = self.timestep_embedding(t)
        h = keras.activations.silu(self.mlp_in(t_freq, training=training))
        return self.mlp_out(h, training=training)

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        """Return ``(B, hidden_size)``.

        :param input_shape: ``(B,)`` or ``(B, 1)``.
        :type input_shape: Any
        :return: ``(B, hidden_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(input_shape)
        batch = shape[0] if len(shape) > 0 else None
        return (batch, self.hidden_size)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: A JSON-serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "frequency_embedding_size": self.frequency_embedding_size,
                "max_period": self.max_period,
                "kernel_stddev": self.kernel_stddev,
            }
        )
        return config
