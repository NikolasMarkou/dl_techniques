"""
3D multi-axis Rotary Position Embedding (mRoPE) for the Ideogram4 DiT.

This module ports Ideogram4's ``Ideogram4MRoPE``. Each token carries a
3-component position id ``(t, h, w)`` for time, height and width. The layer
returns ``cos`` and ``sin`` tables, not rotated vectors; a separate helper
applies them to query and key.

Architecture:
    Each of the three axes gets a frequency table: the outer product of that
    axis' position ids with the shared inverse-frequency vector ``inv_freq``
    of length ``head_dim/2``. The final half-table is assembled slot by slot,
    picking which axis' table each slot ``j`` is drawn from:

        - slot ``j`` defaults to the time axis (t),
        - slots ``arange(1, mrope_section[1]*3, 3)`` come from the h axis,
        - slots ``arange(2, mrope_section[2]*3, 3)`` come from the w axis.

    The h slots satisfy ``j % 3 == 1`` and the w slots ``j % 3 == 2``, so the
    two bands never collide. ``mrope_section[0]``, the t band length, is
    informational. The interleave loop never uses it. That matches the
    PyTorch reference exactly.

    The half-table is then concatenated with itself, ``[freqs, freqs]``, to
    span the full ``head_dim`` before ``cos`` and ``sin`` are taken. The
    result is a pair of ``(B, L, head_dim)`` tables.

Pairing convention:
    ``apply_rotary_pos_emb`` in this module uses SPLIT-HALF pairing, the
    GPT-NeoX convention: channel ``j`` rotates with channel
    ``j + head_dim/2``. That is what ``_rotate_half`` and the
    ``[freqs, freqs]`` duplication together produce, and it is verified by
    execution. It differs from the INTERLEAVED pairing in
    ``rotary_position_embedding.py`` and ``axial_rope_2d.py``, which pair
    ``x[2i]`` with ``x[2i+1]``. Do not mix a table built here with a rotation
    written for the other convention.

PyTorch reference (faithfully ported)::

    # inv_freq: (head_dim/2,)
    inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))
    # pos: (3, B, L)
    pos = position_ids.permute(2, 0, 1).float()
    # freqs: (3, B, L, head_dim/2)
    freqs = (inv_freq[None, None, :, None]
             @ pos[:, :, None, :]).transpose(2, 3)
    freqs_t = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        idx = arange(offset, mrope_section[axis] * 3, 3)
        freqs_t[..., idx] = freqs[axis][..., idx]
    emb = cat((freqs_t, freqs_t), dim=-1)
    return emb.cos(), emb.sin()
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.embedding.multi_axis_rope")
class Ideogram4MRoPE(keras.layers.Layer):
    """Build 3D multi-axis mRoPE ``cos``/``sin`` tables for Ideogram4.

    Takes integer position ids of shape ``(B, L, 3)`` carrying ``(t, h, w)``
    per token and returns two tables of shape ``(B, L, head_dim)``. The layer
    rotates nothing; :func:`apply_rotary_pos_emb` does that.

    Which axis feeds which frequency slot is fixed by ``mrope_section`` and
    is decided at construction, so the selector is a static constant rather
    than a runtime scatter.

    **Pairing convention: SPLIT-HALF.** ``call()`` returns
    ``concat([freqs, freqs])``, and :func:`apply_rotary_pos_emb` applies
    :func:`_rotate_half`, so channel ``j`` rotates with channel
    ``j + head_dim/2`` -- verified by impulse through the pair of them (at
    ``head_dim=8`` a one-hot at channel 0 leaks into channel 4 and nowhere
    else). This is the GPT-NeoX / HF ``rotate_half`` form, NOT the INTERLEAVED
    pairing used by ``rotary_position_embedding.py`` and ``axial_rope_2d.py``
    in this same package. Both are valid rotations and both keep the
    relative-position property, so the wrong one TRAINS FINE; the difference
    is invisible in a config, in a shape and at load time, and surfaces only
    as plausible, wrong numbers.

    **Do not confuse the two "splits".** The per-axis assignment of frequency
    SLOTS to ``(t, h, w)`` (what ``mrope_section`` controls) is a different
    question from the intra-pair rotation form. Only the latter has to match a
    checkpoint's ``q_proj``/``k_proj``.

    *Which checkpoints this can consume.* Qwen2-VL / Qwen2.5-VL M-RoPE, the
    family this layer's factory key ``mrope_ideogram4`` belongs to, uses the
    identical split-half rotation (HF ``modeling_qwen2_vl.py`` defines the same
    ``rotate_half``), as do GPT-NeoX, HF ``LlamaModel`` and HF Gemma. Weights
    from an INTERLEAVED implementation -- GPT-J's ``rotate_every_two`` or
    Meta's official LLaMA ``apply_rotary_emb`` -- need their
    ``q_proj``/``k_proj`` rows permuted first; HF ships that permutation as
    ``convert_llama_weights_to_hf.py::permute`` (huggingface/transformers
    issue #25199).

    *References*: Su, J., et al. (2021). "RoFormer: Enhanced Transformer with
    Rotary Position Embedding". arXiv:2104.09864 (RoPE itself). Wang, P., et
    al. (2024). "Qwen2-VL: Enhancing Vision-Language Model's Perception of the
    World at Any Resolution". arXiv:2409.12191 (the M-RoPE multi-axis design).
    The layer itself is a port of Ideogram4's ``Ideogram4MRoPE``, which
    publishes no arXiv id of its own.

    The shared inverse-frequency vector is a non-trainable weight, so it
    survives ``.keras`` serialization. Its value comes from an
    ``initializer`` callable and never from a post-creation ``.assign()``.
    Surviving serialization and being correctly initialized are different
    properties: the ``.assign()`` form was discarded by the
    ``StatelessScope`` of Keras 3's symbolic build pass, leaving the vector
    all zeros and mRoPE the identity in every real model. See the anchor in
    :meth:`build`.

    **Architecture Overview:**

    .. code-block:: text

        position_ids (B, L, 3) = (t, h, w), integer
                        │
                        ▼
        pos = cast(position_ids, float32)
                        │
        einsum("bla,f->blfa", pos, inv_freq)
        inv_freq (half,), half = head_dim/2, non-trainable
                        │
                        ▼
        freqs_per_axis (B, L, half, 3)
                        │
        einsum("blfa,fa->blf", freqs_per_axis, select_onehot)
        select_onehot (half, 3), STATIC, fixed at construction
                        │
                        ▼
        freqs (B, L, half)
                        │
        concatenate([freqs, freqs], axis=-1)
                        │
                        ▼
        emb (B, L, head_dim)
                        │
                 ┌──────┴──────┐
                 ▼             ▼
              cos(emb)      sin(emb)     each (B, L, head_dim)

        Which axis feeds slot j of the half-table:

          slot j       source axis   consumed by
          j % 3 == 1   h (index 1)   arange(1, sec[1]*3, 3)
          j % 3 == 2   w (index 2)   arange(2, sec[2]*3, 3)
          otherwise    t (index 0)   the default, no explicit index

        sec = mrope_section. sec[0], the t band, is informational and
        never indexes anything, matching the PyTorch reference. With
        head_dim=8 and sec=(1, 1, 1) the selector is [0, 1, 2, 0].

    :param head_dim: Per-head dimensionality. Must be a positive even
        integer.
    :type head_dim: int
    :param rope_theta: Rotary base frequency, the PyTorch ``base``. Must be
        positive.
    :type rope_theta: float
    :param mrope_section: 3-tuple ``(t_band, h_band, w_band)``. Each entry is
        the number of 3-strided frequency slots given to that axis. The h and
        w bands consume ``arange(offset, band*3, 3)`` and must fit inside
        ``head_dim/2``. The t entry is informational.
    :type mrope_section: Sequence[int]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar inv_freq: Non-trainable float32 vector of shape ``(head_dim/2,)``.
        ``None`` until ``build()`` runs.
    :vartype inv_freq: keras.Variable or None

    Input shape:
        3D integer tensor with shape ``(B, L, 3)``.

    Output shape:
        A pair of 3D tensors, each ``(B, L, head_dim)``, both float32.

    :raises ValueError: If ``head_dim`` is not a positive even integer, if
        ``rope_theta`` is not positive, or if ``mrope_section`` is not three
        positive entries whose h/w bands fit inside ``head_dim/2``. Raised
        from ``__init__``.
    :raises ValueError: If the input is not rank-3 with a last dimension of
        3. Raised from ``build()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding.multi_axis_rope import (
            Ideogram4MRoPE,
        )

        layer = Ideogram4MRoPE(
            head_dim=256, rope_theta=5_000_000,
            mrope_section=(24, 20, 20),
        )
        position_ids = keras.ops.zeros((2, 16, 3), dtype="int32")
        cos, sin = layer(position_ids)
        cos.shape  # (2, 16, 256)
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        mrope_section: Sequence[int],
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and precompute the slot selector.

        No weight is created here; :meth:`build` creates both.

        :param head_dim: Per-head dimensionality.
        :type head_dim: int
        :param rope_theta: Rotary base frequency.
        :type rope_theta: float
        :param mrope_section: 3-tuple ``(t_band, h_band, w_band)``.
        :type mrope_section: Sequence[int]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``head_dim`` is not a positive even integer,
            if ``rope_theta`` is not positive, or if ``mrope_section`` is not
            three positive entries whose h/w bands fit in ``head_dim/2``.
        """
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
        # Shape (half,), integer.
        self._source_axis = source_axis

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

        # A NON-TRAINABLE weight, so the value serializes; a raw tensor
        # attribute does not round-trip through `.keras`.
        #
        # Do NOT restore
        #     self.inv_freq = self.add_weight(..., initializer="zeros")
        #     self.inv_freq.assign(inv_freq_values.astype("float32"))
        # Keras 3 runs a symbolic build pass inside a `StatelessScope` whenever
        # this layer is first reached from a parent's `call()`, which covers
        # every real model and every factory-built (`mrope_ideogram4`) path.
        # That scope records the `.assign()` and then discards it. Measured on
        # CPU 2026-08-15: a direct `.build(...)` gives `inv_freq[0] == 1.0`,
        # while through a parent's `call()` the whole vector was zero, so every
        # rotary angle was 0 and mRoPE was the identity at every position.
        # Being non-trainable makes the value SERIALIZE; it does not make it
        # CORRECT. Initializers run at variable-CREATION time and survive the
        # stateless scope. Same defect and fix as
        # `rotary_position_embedding.py` (D-021). See decisions.md D-027
        # of plan-2026-08-14T233721-d4f9beb2.
        # `inv_freq_values` is NumPy, so closing over it carries no `FuncGraph`
        # tensor; a `keras.ops` tensor built here would raise "out of scope".
        inv_freq_f32 = inv_freq_values.astype("float32")
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(self._half,),
            initializer=lambda shape, dtype=None: keras.ops.convert_to_tensor(
                inv_freq_f32, dtype=dtype or "float32"
            ),
            trainable=False,
            dtype="float32",
        )

        # DECISION plan_2026-06-12_59a18a10/D-003: the t/h/w band interleave is
        # implemented as a STATIC one-hot select over the stacked (3, ...) freqs,
        # NOT a dynamic scatter `freqs_t[..., idx] = freqs[axis][..., idx]`. The
        # selector is fixed at build (mrope_section is static), so a precomputed
        # (head_dim/2, 3) one-hot multiplied into the axis dim is XLA-safe and
        # avoids backend-specific in-place / scatter ops. Do NOT replace with a
        # dynamic `keras.ops.scatter`/`slice_update` in call(): position ids are
        # dynamic but the slot->axis map is not, and scatter on the freq axis is
        # not reliably XLA-traceable across backends.
        # Materialized by an INITIALIZER for the same reason as `inv_freq` above:
        # an `.assign()` inside the symbolic build pass is discarded, and an
        # all-zero selector zeroes the selected frequency for EVERY slot, not just
        # the h/w bands. Measured on CPU 2026-08-15: direct `.build(...)` gives
        # `select_onehot[0, 0] == 1.0`, through a parent's `call()` it was `0.0`.
        # The originating plan directory is gone, so this comment is the only
        # record of the decision. Keep every clause above.
        # Shape (half, 3), one row per frequency slot.
        onehot = np.eye(3, dtype="float32")[self._source_axis]
        self._select_onehot = self.add_weight(
            name="select_onehot",
            shape=(self._half, 3),
            initializer=lambda shape, dtype=None: keras.ops.convert_to_tensor(
                onehot, dtype=dtype or "float32"
            ),
            trainable=False,
            dtype="float32",
        )

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
        # Shape (B, L, 3).
        pos = keras.ops.cast(position_ids, "float32")

        # Per-axis frequency tables, the outer product of positions and
        # inv_freq, keeping the axis dim. pos[..., a] is (B, L), inv_freq is
        # (half,), and the result is (B, L, half, 3) with the last dim
        # indexing (t, h, w).
        inv_freq = self.inv_freq
        freqs_per_axis = keras.ops.einsum("bla,f->blfa", pos, inv_freq)

        # Static one-hot select over the axis dim: for each frequency slot f,
        # pick the axis assigned to it. _select_onehot: (half, 3).
        # sum_a freqs_per_axis[b,l,f,a] * onehot[f,a]  -> (B, L, half)
        freqs = keras.ops.einsum("blfa,fa->blf", freqs_per_axis, self._select_onehot)

        # Concatenate the half-table with itself to span head_dim, then take
        # cos and sin. The duplication is what makes the paired rotation in
        # `apply_rotary_pos_emb` SPLIT-HALF: slot j serves channel j and
        # channel j + head_dim/2. Result shape (B, L, head_dim).
        emb = keras.ops.concatenate([freqs, freqs], axis=-1)
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
# Static rotary application helpers, imported by the attention layer.
# ---------------------------------------------------------------------


def _rotate_half(x: keras.KerasTensor) -> keras.KerasTensor:
    """Rotate the last-dim halves: ``[-x2, x1]`` for ``x = [x1, x2]``.

    This is the SPLIT-HALF (GPT-NeoX) form. It pairs channel ``j`` with
    channel ``j + d/2``, not with ``j + 1``.

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

    Ports the PyTorch ``_apply_rotary_pos_emb``. The head axis is unsqueezed
    into ``cos`` and ``sin`` so they broadcast over heads. The pairing is
    SPLIT-HALF: channel ``j`` rotates with channel ``j + head_dim/2``.

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
