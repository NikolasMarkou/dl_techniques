"""
Fixed 2D sinusoidal position encoding for image-like feature maps.

This module provides :class:`PositionEmbeddingSine2D`, the DETR-style 2D
extension of the original Transformer's sinusoidal encoding. Attention is
permutation-equivariant, so a vision stack has no idea where a patch or a
feature-map pixel sits until something tells it. This layer supplies that
signal. It owns no weights and learns nothing.

Architecture:
    Row and column coordinate grids are produced by a cumulative sum over
    the valid positions, so a padding mask shortens the grid instead of
    shifting it. Each axis is encoded independently with the sinusoid below,
    and the two encodings are concatenated on the channel axis. The result
    is ``2 * num_pos_feats`` channels wide.

    The layer takes a channels-LAST input and returns a channels-FIRST
    encoding. That asymmetry is real and unconditional. See the class
    docstring's diagram for where the transpose happens.

Foundational Mathematics:
    For a position ``pos`` on one axis and a channel index ``i`` within that
    axis's ``d = num_pos_feats`` channels::

        PE(pos, 2i)     = sin(pos / T^(2i / d))
        PE(pos, 2i + 1) = cos(pos / T^(2i / d))

    ``T`` is the ``temperature``, typically ``10000``. The wavelengths form
    a geometric progression from ``2*pi`` to ``2*pi*T``, so the encoding
    carries both coarse and fine position at once. For a fixed offset ``k``,
    ``PE(pos + k)`` is a linear map of ``PE(pos)``, which is what lets
    attention learn to work in relative spatial terms from an absolute code.

    ``d`` must be even. Sine takes the even channels and cosine the odd
    ones, and the two halves are stacked pairwise, so an odd ``d`` gives the
    two halves different widths and the stack fails.

References:
  - Vaswani, A., et al. (2017). "Attention Is All You Need".
    https://arxiv.org/abs/1706.03762
  - Carion, N., et al. (2020). "End-to-End Object Detection with
    Transformers". https://arxiv.org/abs/2005.12872
"""

import math
import keras
from keras import ops
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.embedding.positional_embedding_sine_2d")
class PositionEmbeddingSine2D(keras.layers.Layer):
    """Build a fixed 2D sinusoidal position code for a feature map.

    Encodes the row and column of every spatial location with sine and
    cosine at a geometric ladder of frequencies, then concatenates the two
    axes. The layer owns no weights and ignores the input VALUES entirely;
    only the shape of the first three axes and the optional mask matter.

    Two shape facts catch callers out, so read them before wiring this up.

    First, the LAYOUT FLIPS. The input is channels-last ``(batch, H, W, C)``
    and the output is channels-first ``(batch, 2 * num_pos_feats, H, W)``.
    A caller that writes ``x + pos(x)`` on a channels-last ``x`` will either
    broadcast wrongly or fail on the shapes.

    Second, ``num_pos_feats`` must be EVEN, and ``__init__`` enforces it.
    Sine takes the even channels and cosine the odd ones, and the two are
    stacked pairwise; with an odd ``num_pos_feats`` the halves have different
    widths and the stack cannot run. An odd value now raises ``ValueError``
    at construction rather than dying inside the backend at call time.

    ``from_config`` is deliberately more lenient than ``__init__``: a stored
    config written before that check existed can carry an odd value, so it is
    rounded UP to the next even number and a warning is logged, rather than
    making the archive unloadable.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────┐
        │  Input  (batch, H, W, C)  channels LAST    │
        │  only the first 3 axes are read            │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  not_mask = ~mask   (batch, H, W)          │
        │  mask True marks padding                   │
        │  no mask means every position valid        │
        └────────────────────────────────────────────┘
                              │
                              ▼
                cumsum axis 1          cumsum axis 2
                       │                      │
                       ▼                      ▼
                ┌─────────────┐        ┌─────────────┐
                │  y_embed    │        │  x_embed    │
                │  normalize  │        │  normalize  │
                │  (optional) │        │  (optional) │
                └──────┬──────┘        └──────┬──────┘
                       │                      │
                       ▼                      ▼
                sin on even ch         sin on even ch
                cos on odd ch          cos on odd ch
                dim_t = T^(2*(i//2)/d), shared by both
                       │                      │
                  (B, H, W, d)           (B, H, W, d)
                       └──────────┬───────────┘
                                  ▼
        ┌────────────────────────────────────────────┐
        │  concatenate y then x on axis 3            │
        │  -> (batch, H, W, 2d)                      │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  transpose [0, 3, 1, 2]                    │
        │  -> (batch, 2d, H, W)  channels FIRST      │
        │  unconditional, there is no layout branch  │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  cast to compute_dtype                     │
        │  Output (batch, 2*num_pos_feats, H, W)     │
        └────────────────────────────────────────────┘

    :param num_pos_feats: Channels per spatial axis. The output is twice
        this wide. Must be positive, and must be even for the layer to run.
        Defaults to ``64``.
    :type num_pos_feats: int
    :param temperature: Base of the frequency ladder. Larger values stretch
        the wavelengths and make the code vary more slowly across the grid.
        Must be positive. Defaults to ``10000.0``.
    :type temperature: float
    :param normalize: Rescale each coordinate grid by its own last row or
        column before the sinusoid, so the code does not depend on the
        absolute feature-map size. Defaults to ``True``.
    :type normalize: bool
    :param scale: Multiplier applied to the normalized coordinates. Ignored
        when ``normalize`` is ``False``. Defaults to ``2 * pi``.
    :type scale: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor ``(batch, H, W, C)``. Only ``batch``, ``H`` and ``W`` are
        read.

    Output shape:
        4D tensor ``(batch, 2 * num_pos_feats, H, W)``, channels FIRST.

    :raises ValueError: If ``num_pos_feats`` or ``temperature`` is not
        positive. Raised from ``__init__``.

    Example:

    .. code-block:: python

        import numpy as np
        from dl_techniques.layers.embedding import (
            positional_embedding_sine_2d as pe2d,
        )

        pos = pe2d.PositionEmbeddingSine2D(num_pos_feats=8)
        x = np.zeros((2, 3, 4, 5), dtype="float32")
        pos(x).shape  # (2, 16, 3, 4)
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = 2 * math.pi,
        **kwargs: Any
    ) -> None:
        """Validate and store the configuration.

        The layer has no weights, so there is nothing for ``build()`` to do
        and none is defined.

        :param num_pos_feats: Channels per spatial axis.
        :type num_pos_feats: int
        :param temperature: Base of the frequency ladder.
        :type temperature: float
        :param normalize: Rescale the coordinate grids before the sinusoid.
        :type normalize: bool
        :param scale: Multiplier applied to normalized coordinates.
        :type scale: float
        :param kwargs: Additional keyword arguments for the Layer base class.
        :type kwargs: Any
        :raises ValueError: If ``num_pos_feats`` or ``temperature`` is not
            positive, or if ``num_pos_feats`` is odd.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if num_pos_feats <= 0:
            raise ValueError(f"num_pos_feats must be positive, got {num_pos_feats}")
        # DECISION plan-2026-08-28T181715-3870472c/D-004
        # Enforce evenness HERE. This layer owns no weights and therefore has
        # no `build()`, so `__init__` is the only construction-time site
        # available. Without this the failure is
        # `InvalidArgumentError: Shapes of all inputs must match:
        # values[0].shape=[2,6,5,4] != values[1].shape=[2,6,5,3] [Op:Pack]`
        # raised from `ops.stack` at call time. Do NOT copy this raise into
        # `from_config`: an archive predating this check can carry an odd
        # value and must still load. See decisions.md D-004.
        if num_pos_feats % 2 != 0:
            raise ValueError(
                f"num_pos_feats must be even, got {num_pos_feats}. The sine "
                f"half takes the even channels and the cosine half the odd "
                f"ones, so an odd width leaves the two halves unequal and "
                f"they cannot be stacked."
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        # Store configuration
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def call(self, inputs: keras.KerasTensor, mask: Optional[keras.KerasTensor] = None) -> keras.KerasTensor:
        """Build the position code for one feature map.

        The input values are not read. Only its first three axes are.

        :param inputs: Input of shape ``(batch, H, W, C)``, channels last.
        :type inputs: keras.KerasTensor
        :param mask: Optional boolean mask of shape ``(batch, H, W)``. A
            ``True`` entry marks a padding position, which is excluded from
            the coordinate count.
        :type mask: Optional[keras.KerasTensor]
        :return: Position code of shape
            ``(batch, 2 * num_pos_feats, H, W)``, channels FIRST.
        :rtype: keras.KerasTensor
        :raises Exception: From the backend, if ``num_pos_feats`` is somehow
            odd. ``__init__`` rejects an odd value and ``from_config``
            rounds one up, so this is no longer reachable through either
            construction path; it remains possible only if the attribute is
            overwritten after construction.
        """
        if mask is None:
            # No mask means every position is valid. The input is assumed
            # 4D (B, H, W, C), so the mask covers (B, H, W).
            mask = ops.zeros(ops.shape(inputs)[:3], dtype="bool")

        not_mask = ~mask
        y_embed = ops.cumsum(ops.cast(not_mask, "float32"), axis=1)
        x_embed = ops.cumsum(ops.cast(not_mask, "float32"), axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.arange(self.num_pos_feats, dtype="float32")
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = ops.stack(
            [ops.sin(pos_x[..., 0::2]), ops.cos(pos_x[..., 1::2])], axis=4
        )
        pos_x = ops.reshape(pos_x, (*ops.shape(pos_x)[:3], -1))

        pos_y = ops.stack(
            [ops.sin(pos_y[..., 0::2]), ops.cos(pos_y[..., 1::2])], axis=4
        )
        pos_y = ops.reshape(pos_y, (*ops.shape(pos_y)[:3], -1))

        pos = ops.concatenate([pos_y, pos_x], axis=3)

        # Channels-last (batch, H, W, 2d) -> channels-first
        # (batch, 2d, H, W). Unconditional: there is no layout option.
        pos = ops.transpose(pos, [0, 3, 1, 2])

        # DECISION plan-2026-08-19T163559-499b6f0e/D-011
        # The sinusoid is COMPUTED in float32 on purpose, because the cumsum
        # over positions and `temperature ** (2i/d)` both lose resolution in
        # half precision. A Keras layer must still RETURN `compute_dtype`.
        # Without this cast, a consumer writing `x + pos_embed(x)` under
        # mixed_float16 gets an AddV2 dtype mismatch. That is how the
        # `video_jepa` encoder died, at the line that adds this layer's output
        # to its patch tokens. Do NOT move the float32 literals above to
        # `self.compute_dtype`: the float32 computation is the point and only
        # the boundary is cast. Under float32 this cast is a no-op.
        # See decisions.md D-011.
        return ops.cast(pos, self.compute_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Report the channels-first output shape.

        :param input_shape: Shape of the input, ``(batch, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, 2 * num_pos_feats, H, W)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size, h, w = input_shape[0], input_shape[1], input_shape[2]
        return batch_size, 2 * self.num_pos_feats, h, w

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary carrying every ``__init__``
            argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_pos_feats": self.num_pos_feats,
            "temperature": self.temperature,
            "normalize": self.normalize,
            "scale": self.scale
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PositionEmbeddingSine2D":
        """Rebuild the layer, repairing an odd ``num_pos_feats`` if present.

        The evenness check in ``__init__`` is a NEW raise on a value the old
        code accepted, so applying it here would turn a previously loadable
        archive into an unloadable one. Instead the odd value is rounded UP
        to the next even number and a warning is logged: the constructor
        stays strict, the load path substitutes and warns. The caller's
        mapping is copied, not modified.

        :param config: Configuration produced by ``get_config()``. Not
            modified.
        :type config: Dict[str, Any]
        :return: A new layer instance.
        :rtype: PositionEmbeddingSine2D
        """
        num_pos_feats = config.get("num_pos_feats")
        if isinstance(num_pos_feats, int) and num_pos_feats > 0 and num_pos_feats % 2 != 0:
            config = dict(config)
            config["num_pos_feats"] = num_pos_feats + 1
            logger.warning(
                f"Stored config carries an odd num_pos_feats={num_pos_feats}, "
                f"which this layer cannot run: substituting "
                f"{num_pos_feats + 1}. The output width changes from "
                f"{2 * num_pos_feats} to {2 * (num_pos_feats + 1)} channels. "
                f"Re-save the model to make this permanent."
            )
        return cls(**config)

# ---------------------------------------------------------------------
