"""
3D multi-axis Rotary Position Embedding (mRoPE) for the Ideogram4 DiT.

This layer ports the Ideogram4 ``Ideogram4MRoPE`` module: a rotary positional
embedding driven by a 3-component position id per token ``(t, h, w)`` (time /
height / width). Unlike standard 1D RoPE, the rotary frequency table is built
per-axis and then **band-interleaved**: starting from the time-axis (t)
frequency table, a fixed subset of the ``head_dim/2`` frequency slots is
reassigned to the height (h) and width (w) axes according to ``mrope_section``.

Architecture:
    For each of the three axes (t, h, w) a frequency table is produced by the
    outer product of that axis' position ids with the shared inverse-frequency
    vector ``inv_freq`` (length ``head_dim/2``). The final per-token table is
    assembled by selecting, for each frequency slot ``j in [0, head_dim/2)``,
    which axis' table the slot is drawn from:

        - slot ``j`` defaults to the time axis (t),
        - slots ``j in arange(1, mrope_section[1]*3, 3)`` come from the h axis,
        - slots ``j in arange(2, mrope_section[2]*3, 3)`` come from the w axis.

    The h-slots (``j % 3 == 1``) and w-slots (``j % 3 == 2``) never collide.
    ``mrope_section[0]`` (the t band length) is informational and is not used
    by the interleave loop, matching the PyTorch reference exactly.

    The resulting half-table is concatenated with itself (``[freqs, freqs]``)
    to span the full ``head_dim`` before ``cos``/``sin`` are taken, yielding
    cos/sin tensors of shape ``(B, L, head_dim)``.

PyTorch reference (faithfully ported)::

    inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))   # (head_dim/2,)
    pos = position_ids.permute(2, 0, 1).float()                      # (3, B, L)
    freqs = (inv_freq[None,None,:,None] @ pos[:,:,None,:]).transpose(2,3)  # (3,B,L,head_dim/2)
    freqs_t = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        idx = arange(offset, mrope_section[axis]*3, 3)
        freqs_t[..., idx] = freqs[axis][..., idx]
    emb = cat((freqs_t, freqs_t), dim=-1)
    return emb.cos(), emb.sin()
"""

import keras
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.layers")
class Ideogram4MRoPE(keras.layers.Layer):
    """3D multi-axis rotary position embedding (mRoPE) for Ideogram4.

    Produces ``cos`` and ``sin`` rotary tables of shape ``(B, L, head_dim)``
    from integer position ids of shape ``(B, L, 3)`` carrying ``(t, h, w)``
    coordinates per token. The frequency table is band-interleaved across the
    three axes according to ``mrope_section`` (see module docstring).

    The shared inverse-frequency vector is stored as a non-trainable weight via
    ``add_weight(trainable=False)`` so it survives ``.keras`` serialization.

    :param head_dim: Per-head dimensionality. Must be a positive even integer.
    :type head_dim: int
    :param rope_theta: Rotary base frequency (PyTorch ``base``). Must be > 0.
    :type rope_theta: float
    :param mrope_section: 3-tuple ``(t_band, h_band, w_band)``. Each entry is
        the number of 3-strided frequency slots assigned to that axis. The h
        and w bands consume slots ``arange(offset, band*3, 3)`` and must fit
        inside ``head_dim/2``.
    :type mrope_section: Sequence[int]
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``head_dim`` is not a positive even integer.
    :raises ValueError: If ``rope_theta`` is not positive.
    :raises ValueError: If ``mrope_section`` does not have length 3, contains
        non-positive entries, or its h/w bands exceed ``head_dim/2`` bounds.

    Example:
        >>> layer = Ideogram4MRoPE(head_dim=256, rope_theta=5_000_000,
        ...                        mrope_section=(24, 20, 20))
        >>> position_ids = keras.ops.zeros((2, 16, 3), dtype="int32")
        >>> cos, sin = layer(position_ids)
        >>> cos.shape, sin.shape
        (TensorShape([2, 16, 256]), TensorShape([2, 16, 256]))
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        mrope_section: Sequence[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- validation -------------------------------------------------
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be a positive integer, got {head_dim}")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")

        mrope_section = tuple(int(s) for s in mrope_section)
        if len(mrope_section) != 3:
            raise ValueError(
                f"mrope_section must have length 3 (t, h, w), got {mrope_section}"
            )
        if any(s <= 0 for s in mrope_section):
            raise ValueError(
                f"mrope_section entries must be positive, got {mrope_section}"
            )

        half = head_dim // 2
        # The h band consumes slots arange(1, h*3, 3); the w band arange(2, w*3, 3).
        # The largest consumed index must stay strictly below `half`.
        for axis, offset, name in ((1, 1, "h"), (2, 2, "w")):
            length = mrope_section[axis] * 3
            consumed = np.arange(offset, length, 3)
            if consumed.size and consumed.max() >= half:
                raise ValueError(
                    f"mrope_section[{axis}] ({name} band = {mrope_section[axis]}) "
                    f"reaches frequency slot {int(consumed.max())} which exceeds "
                    f"head_dim/2 - 1 = {half - 1}. Reduce the {name} band."
                )

        # --- store config ----------------------------------------------
        self.head_dim = head_dim
        self.rope_theta = float(rope_theta)
        self.mrope_section = mrope_section
        self._half = half

        # Precompute, at construction, the static per-slot "source axis"
        # selector (length head_dim/2): 0=t (default), 1=h, 2=w. This drives
        # an XLA-safe one-hot select in call() instead of a dynamic scatter.
        source_axis = np.zeros((half,), dtype="int64")
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = np.arange(offset, length, 3)
            source_axis[idx] = axis
        self._source_axis = source_axis  # (half,) int

        # weights created in build()
        self.inv_freq = None
        self._select_onehot = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the non-trainable ``inv_freq`` weight and selector one-hot.

        :param input_shape: Expected ``(B, L, 3)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last input dimension is not 3.
        """
        if self.built:
            return

        if len(input_shape) != 3 or input_shape[-1] != 3:
            raise ValueError(
                f"Ideogram4MRoPE expects position_ids of shape (B, L, 3), "
                f"got input_shape {input_shape}"
            )

        # inv_freq = 1 / (theta ** (arange(0, head_dim, 2) / head_dim))  -> (head_dim/2,)
        inv_freq_values = 1.0 / (
            self.rope_theta
            ** (np.arange(0, self.head_dim, 2, dtype="float32") / self.head_dim)
        )

        # Stored as a NON-TRAINABLE weight so it serializes (raw tensor attrs
        # do not round-trip through .keras).
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(self._half,),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        self.inv_freq.assign(inv_freq_values.astype("float32"))

        # DECISION plan_2026-06-12_59a18a10/D-003: the t/h/w band interleave is
        # implemented as a STATIC one-hot select over the stacked (3, ...) freqs,
        # NOT a dynamic scatter `freqs_t[..., idx] = freqs[axis][..., idx]`. The
        # selector is fixed at build (mrope_section is static), so a precomputed
        # (head_dim/2, 3) one-hot multiplied into the axis dim is XLA-safe and
        # avoids backend-specific in-place / scatter ops. Do NOT replace with a
        # dynamic `keras.ops.scatter`/`slice_update` in call(): position ids are
        # dynamic but the slot->axis map is not, and scatter on the freq axis is
        # not reliably XLA-traceable across backends. See decisions.md D-003.
        onehot = np.eye(3, dtype="float32")[self._source_axis]  # (half, 3)
        self._select_onehot = self.add_weight(
            name="select_onehot",
            shape=(self._half, 3),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        self._select_onehot.assign(onehot)

        super().build(input_shape)

    def call(
        self,
        position_ids: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Compute the mRoPE ``cos`` and ``sin`` tables.

        :param position_ids: Integer tensor of shape ``(B, L, 3)`` with
            ``(t, h, w)`` coordinates per token.
        :type position_ids: keras.KerasTensor
        :param training: Unused (this layer has no training-specific behavior).
        :type training: Optional[bool]
        :return: ``(cos, sin)``, each of shape ``(B, L, head_dim)`` (float32).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        pos = keras.ops.cast(position_ids, "float32")  # (B, L, 3)

        # Per-axis frequency tables via outer product of positions and inv_freq.
        # pos[..., a]: (B, L);  inv_freq: (half,)
        # freqs_per_axis: (B, L, half, 3) where the last dim indexes (t, h, w).
        # einsum over the position/frequency outer product, keeping the axis dim.
        inv_freq = self.inv_freq  # (half,)
        freqs_per_axis = keras.ops.einsum("bla,f->blfa", pos, inv_freq)
        # -> (B, L, half, 3)

        # Static one-hot select over the axis dim: for each frequency slot f,
        # pick the axis assigned to it. _select_onehot: (half, 3).
        # sum_a freqs_per_axis[b,l,f,a] * onehot[f,a]  -> (B, L, half)
        freqs = keras.ops.einsum("blfa,fa->blf", freqs_per_axis, self._select_onehot)

        # Concatenate the half-table with itself to span head_dim, then cos/sin.
        emb = keras.ops.concatenate([freqs, freqs], axis=-1)  # (B, L, head_dim)
        cos = keras.ops.cos(emb)
        sin = keras.ops.sin(emb)
        return cos, sin

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return the ``(cos, sin)`` output shapes.

        :param input_shape: Input shape ``(B, L, 3)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: A pair of shapes, each ``(B, L, head_dim)``.
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        out_shape = (input_shape[0], input_shape[1], self.head_dim)
        return out_shape, out_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        :return: Dictionary with all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "rope_theta": self.rope_theta,
                "mrope_section": list(self.mrope_section),
            }
        )
        return config


# ---------------------------------------------------------------------
# Static rotary application helpers (imported by the attention layer, step 2).
# ---------------------------------------------------------------------


def _rotate_half(x: keras.KerasTensor) -> keras.KerasTensor:
    """Rotate the last-dim halves: ``[-x2, x1]`` for ``x = [x1, x2]``.

    :param x: Tensor whose last dimension is even.
    :type x: keras.KerasTensor
    :return: Rotated tensor of the same shape.
    :rtype: keras.KerasTensor
    """
    half = keras.ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return keras.ops.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: keras.KerasTensor,
    k: keras.KerasTensor,
    cos: keras.KerasTensor,
    sin: keras.KerasTensor,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Apply mRoPE rotary embedding to query and key tensors.

    Ports the PyTorch ``_apply_rotary_pos_emb``: the head axis is unsqueezed
    into ``cos``/``sin`` so they broadcast over heads.

    :param q: Query tensor of shape ``(B, num_heads, L, head_dim)``.
    :type q: keras.KerasTensor
    :param k: Key tensor of shape ``(B, num_heads, L, head_dim)``.
    :type k: keras.KerasTensor
    :param cos: Cosine table of shape ``(B, L, head_dim)``.
    :type cos: keras.KerasTensor
    :param sin: Sine table of shape ``(B, L, head_dim)``.
    :type sin: keras.KerasTensor
    :return: ``(q_embed, k_embed)``, each of shape ``(B, num_heads, L, head_dim)``.
    :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
    """
    # (B, L, head_dim) -> (B, 1, L, head_dim) to broadcast over the head axis.
    cos = keras.ops.expand_dims(cos, axis=1)
    sin = keras.ops.expand_dims(sin, axis=1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------
