"""
Real-valued 2D axial Rotary Position Embedding (RoPE) for post-head-split q/k.

This module provides :class:`AxialRoPE2D`, the rotary position embedding used by
SAM 2's memory-attention stack. It rotates query and key vectors *after* the
attention head split, encoding a token's ``(x, y)`` position on a 2D grid.

Architecture:
    Tokens are assumed to be a row-major flattening of an ``(H, W)`` spatial
    grid, so flat index ``t`` maps to ``t_x = t % W`` (column) and
    ``t_y = t // W`` (row). The head dimension ``D`` is split into two axial
    halves of ``D // 2`` rotary channels each: the first ``D // 4`` frequency
    bands carry the x-position, the second ``D // 4`` carry the y-position. Both
    axes use the SAME geometric frequency ladder — the axes are distinguished by
    *which channels they occupy* and by *which coordinate multiplies them*, not
    by different frequencies.

Foundational Mathematics:
    The head dimension is read as ``D // 2`` complex numbers formed from
    **adjacent** pairs ``(x[2i], x[2i + 1])`` (the "interleaved" pairing, NOT the
    "split-half" GPT-NeoX pairing). Rotating by angle ``phi_i`` is the complex
    multiply ``(a + bi)(cos phi + i sin phi)``, which expands to::

        out[2i]     = x[2i] * cos(phi_i) - x[2i + 1] * sin(phi_i)
        out[2i + 1] = x[2i] * sin(phi_i) + x[2i + 1] * cos(phi_i)

    equivalently ``out = x * cos_e + rotate_pairs(x) * sin_e`` where ``cos_e``
    and ``sin_e`` duplicate each angle across its own adjacent pair and
    ``rotate_pairs`` maps ``(a, b) -> (-b, a)``.

    The angles are ``phi = concat([outer(t_x, f), outer(t_y, f)])`` of width
    ``D // 2``, with ``f = 1 / theta ** (arange(0, D, 4)[:D // 4] / D)``. This is
    why ``D`` must be divisible by 4.

    Because the rotation is orthogonal and additive in the angle, the inner
    product ``<R_p q, R_p' k>`` depends only on the per-axis displacement
    ``p - p'`` — the relative-position property that makes RoPE useful.

Implementation note (no complex dtype):
    ``keras.ops`` exposes no complex dtype on the TensorFlow backend, so
    ``torch.polar`` / ``view_as_complex`` / ``view_as_real`` have no direct
    equivalent. The rotation is therefore derived from scratch with real-valued
    cos/sin tables; it is mathematically identical to the complex formulation
    above, not an approximation of it.

References:
    - Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
      "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    - Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos"
      (axial RoPE in the memory-attention blocks).
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AxialRoPE2D(keras.layers.Layer):
    """Real-valued 2D axial rotary position embedding for query/key tensors.

    Applies a per-token 2D rotation to query (and optionally key) tensors that
    have ALREADY been split into attention heads, i.e. of shape
    ``(batch, num_heads, num_tokens, head_dim)``. The token axis is interpreted
    as a row-major flattening of the configured ``(H, W)`` grid.

    The layer owns no weights. Its cos/sin table is a pure function of
    ``(head_dim, feat_shape, theta, scale_pos)`` and is materialized in
    :meth:`build` as a
    float64 NumPy constant, then cast per call. It is deliberately NOT an
    ``add_weight`` variable: under a mixed-precision policy Keras autocasts
    variables to the compute dtype, which would silently narrow the angle table
    to float16 (see the work-dtype note in :meth:`call`).

    **Architecture Overview:**

    .. code-block:: text

        q: (B, heads, N_q, D)          k: (B, heads, N_k, D)
                │                                │
                │                        split at N_k - num_k_exclude
                │                          ┌─────┴──────┐
                │                       rotated       tail
                │                       (spatial)   (obj ptrs,
                │                          │        untouched)
                ▼                          ▼             │
        ┌────────────────────────────────────────┐       │
        │ angles[t] = [t_x * f , t_y * f]  (D/2)  │       │
        │ cos_e/sin_e = duplicate per adj. pair   │       │
        │ out = x*cos_e + rot_pairs(x)*sin_e      │       │
        └────────────────────────────────────────┘       │
                │                          │             │
                ▼                          └──── concat ─┘
             q_rot                              k_rot

    :param head_dim: Per-head feature width ``D``. Must be positive and
        divisible by 4 (the ``D // 4`` frequency bands per axis).
    :type head_dim: int
    :param feat_shape: Spatial grid ``(H, W)`` whose row-major flattening
        produces the query token axis. ``H * W`` must equal the query token
        count. Defaults to ``(64, 64)``, SAM 2's memory-attention grid.
    :type feat_shape: Tuple[int, int]
    :param theta: Base of the geometric frequency ladder. Defaults to
        ``10000.0``.
    :type theta: float
    :param scale_pos: Multiplier applied to the ``(t_x, t_y)`` COORDINATES
        before the frequency outer product, i.e. ``angles =
        concat([outer(scale_pos * t_x, f), outer(scale_pos * t_y, f)])``.
        Defaults to ``1.0``, which is bit-identical to the unscaled table. Use
        it when a model computes its frequency ladder at one grid size but wants
        the positions interpolated onto another: SAM 3's global ViTDet blocks
        run a ``72x72`` token grid through a RoPE pre-training grid of
        ``24x24``, i.e. ``scale_pos = 24 / 72 = 1/3``. This is a DISTINCT
        mechanism from ``repeat_k``: ``scale_pos`` compresses the coordinate
        ladder within one grid, ``repeat_k`` broadcasts a finished table across
        extra key blocks.
    :type scale_pos: float
    :param repeat_k: When ``True``, a key sequence may be an integer multiple
        ``r`` of the query grid; the SAME angle table is broadcast across all
        ``r`` blocks. This is spatial-only repetition — it deliberately does NOT
        give each block a distinct phase, because temporal position is carried
        additively elsewhere. When ``False``, the rotated key length must equal
        the query grid exactly.
    :type repeat_k: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``head_dim`` is not a positive multiple of 4, if
        ``feat_shape`` is not a pair of positive ints, if ``theta <= 0``, or if
        ``scale_pos <= 0``.

    Example:
        >>> import numpy as np
        >>> rope = AxialRoPE2D(head_dim=8, feat_shape=(2, 2))
        >>> q = np.zeros((1, 1, 4, 8), dtype="float32")
        >>> out = rope(q)
        >>> out.shape
        (1, 1, 4, 8)
    """

    def __init__(
            self,
            head_dim: int,
            feat_shape: Tuple[int, int] = (64, 64),
            theta: float = 10000.0,
            scale_pos: float = 1.0,
            repeat_k: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be a positive int, got {head_dim!r}")
        # The angle table is `concat([t_x * f, t_y * f])` with `head_dim // 4`
        # bands per axis and width `head_dim // 2`. A head_dim that is even but
        # not a multiple of 4 would silently drop a band (integer division) and
        # produce an angle vector narrower than head_dim // 2. Raise here, at
        # construction, rather than at the first call.
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4 for 2D axial RoPE "
                f"(head_dim // 4 frequency bands per axis), got {head_dim}"
            )

        feat_shape = tuple(feat_shape)
        if len(feat_shape) != 2 or any(
                (not isinstance(s, (int, np.integer))) or s <= 0 for s in feat_shape
        ):
            raise ValueError(
                f"feat_shape must be a (H, W) pair of positive ints, got {feat_shape!r}"
            )
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if scale_pos <= 0:
            raise ValueError(f"scale_pos must be positive, got {scale_pos}")

        # Store ALL configuration parameters.
        self.head_dim = head_dim
        self.feat_shape = (int(feat_shape[0]), int(feat_shape[1]))
        self.theta = float(theta)
        self.scale_pos = float(scale_pos)
        self.repeat_k = bool(repeat_k)

        # Derived, non-config.
        self.num_grid_tokens = self.feat_shape[0] * self.feat_shape[1]

        # Constant cos/sin tables, materialized in build().
        self._cos_table: Optional[np.ndarray] = None
        self._sin_table: Optional[np.ndarray] = None

    # -----------------------------------------------------------------
    # table construction
    # -----------------------------------------------------------------

    def _build_angle_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the float64 cos/sin tables from stored config only.

        :return: ``(cos_e, sin_e)``, each of shape ``(H * W, head_dim)``, where
            each angle is duplicated across its own ADJACENT channel pair.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        height, width = self.feat_shape
        num_tokens = height * width

        # Row-major flattening: x (column) varies fastest.
        flat = np.arange(num_tokens, dtype=np.float64)
        t_x = np.mod(flat, width)
        t_y = np.floor_divide(flat, width)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-083
        # `scale_pos` scales the COORDINATES, applied to BOTH axes, here -- before
        # the frequency outer product. Do NOT "simplify" this by folding it into
        # `freqs` below (`freqs * scale_pos`) even though the products
        # `outer(s*t, f)` and `outer(t, s*f)` are algebraically identical for the
        # single-table case: the two forms diverge the moment either axis gains a
        # per-axis scale or a non-linear coordinate map (NTK/YaRN-style band-wise
        # rescaling touches `freqs` and would then compose wrongly with a
        # frequency-folded position scale). Keeping the scale on the coordinate is
        # also what makes `scale_pos = rope_pt_size / input_size` readable as the
        # grid-interpolation ratio it is. Do NOT apply it to `t_x` only -- the
        # axial halves share one frequency ladder, so an asymmetric scale silently
        # anisotropizes the embedding with no shape error. See decisions.md D-083.
        if self.scale_pos != 1.0:
            t_x = t_x * self.scale_pos
            t_y = t_y * self.scale_pos

        # `head_dim // 4` bands, shared by BOTH axes. This is not a typo and not
        # a simplification: the x and y halves use the identical frequency
        # ladder. The axes are separated by which coordinate multiplies the
        # ladder and by which half of the rotary channels the result lands in.
        bands = np.arange(0, self.head_dim, 4, dtype=np.float64)[: self.head_dim // 4]
        freqs = 1.0 / (self.theta ** (bands / self.head_dim))

        # (N, head_dim // 2): x-phases then y-phases.
        angles = np.concatenate(
            [np.outer(t_x, freqs), np.outer(t_y, freqs)], axis=-1
        )

        # DECISION plan-2026-08-04T044628-4c240b4c/D-006
        # Duplicate each angle across its ADJACENT pair -> [a0, a0, a1, a1, ...].
        # `np.repeat` (interleaved), NOT `np.tile` (split-half GPT-NeoX packing).
        #
        # Do NOT "simplify" this to `np.tile` and a split-half `rotate_half`. Both
        # conventions are valid orthogonal rotations and BOTH satisfy the
        # relative-position property, so the property control in
        # `tests/.../test_axial_rope_2d.py::TestRelativePositionInvariance` stays
        # GREEN under the swap (measured). Only the float64 complex oracle catches
        # it. The choice is not free: adjacent-pair packing is what makes this
        # equivalent to upstream's `view_as_complex` on a `(..., -1, 2)` reshape,
        # so a future upstream-checkpoint conversion depends on it. Swapping the
        # convention would train fine and load a converted checkpoint wrong.
        cos_e = np.repeat(np.cos(angles), 2, axis=-1)
        sin_e = np.repeat(np.sin(angles), 2, axis=-1)
        return cos_e, sin_e

    def build(
            self,
            query_shape: Tuple[Optional[int], ...],
            key_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> None:
        """Materialize the constant angle tables and validate the input rank.

        :param query_shape: Shape of the query tensor,
            ``(batch, num_heads, num_tokens, head_dim)``.
        :type query_shape: Tuple[Optional[int], ...]
        :param key_shape: Shape of the optional key tensor, same rank.
        :type key_shape: Optional[Tuple[Optional[int], ...]]
        :raises ValueError: If a shape is not rank-4, if its last dimension does
            not equal ``head_dim``, or if the query token count is statically
            known and differs from ``H * W``.
        """
        if self.built:
            return

        self._validate_shape(query_shape, "query")
        if key_shape is not None:
            self._validate_shape(key_shape, "key")

        if query_shape[-2] is not None and query_shape[-2] != self.num_grid_tokens:
            raise ValueError(
                f"query token count ({query_shape[-2]}) must equal "
                f"feat_shape H*W ({self.feat_shape[0]}*{self.feat_shape[1]}="
                f"{self.num_grid_tokens})"
            )

        self._cos_table, self._sin_table = self._build_angle_tables()
        logger.debug(
            "AxialRoPE2D built: head_dim=%d feat_shape=%s theta=%.1f "
            "scale_pos=%s repeat_k=%s table=%s", self.head_dim, self.feat_shape,
            self.theta, self.scale_pos, self.repeat_k, self._cos_table.shape,
        )

        super().build(query_shape)

    def _validate_shape(
            self, shape: Tuple[Optional[int], ...], name: str
    ) -> None:
        """Validate one input shape against the stored config.

        :param shape: Shape tuple to validate.
        :type shape: Tuple[Optional[int], ...]
        :param name: Human-readable tensor name for the error message.
        :type name: str
        :raises ValueError: If the rank is not 4 or the head width mismatches.
        """
        if len(shape) != 4:
            raise ValueError(
                f"{name} must be rank-4 (batch, num_heads, num_tokens, head_dim), "
                f"got shape {shape}"
            )
        if shape[-1] is not None and shape[-1] != self.head_dim:
            raise ValueError(
                f"{name} last dimension ({shape[-1]}) must equal head_dim "
                f"({self.head_dim})"
            )

    # -----------------------------------------------------------------
    # rotation
    # -----------------------------------------------------------------

    def _rotate_adjacent_pairs(self, x: Any) -> Any:
        """Map each adjacent channel pair ``(a, b)`` to ``(-b, a)``.

        :param x: Tensor whose last dimension is ``head_dim``.
        :type x: Any
        :return: Tensor of the same shape with each adjacent pair rotated.
        :rtype: Any
        """
        lead = ops.shape(x)[:-1]
        pairs = ops.reshape(x, (*lead, self.head_dim // 2, 2))
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated = ops.stack([-odd, even], axis=-1)
        return ops.reshape(rotated, (*lead, self.head_dim))

    def _apply(self, x: Any, cos_e: Any, sin_e: Any) -> Any:
        """Apply the rotation ``x * cos_e + rotate_pairs(x) * sin_e``.

        :param x: Tensor of shape ``(..., num_tokens, head_dim)``.
        :type x: Any
        :param cos_e: Cosine table broadcastable to ``x``.
        :type cos_e: Any
        :param sin_e: Sine table broadcastable to ``x``.
        :type sin_e: Any
        :return: The rotated tensor, in the same dtype as ``x``.
        :rtype: Any
        """
        return x * cos_e + self._rotate_adjacent_pairs(x) * sin_e

    def _tables(self, num_tokens: int, work_dtype: str) -> Tuple[Any, Any]:
        """Return cos/sin tables tiled to ``num_tokens`` rows in ``work_dtype``.

        :param num_tokens: Number of token rows required. Must be a positive
            integer multiple of ``H * W`` (and exactly ``H * W`` unless
            ``repeat_k`` is enabled by the caller's context).
        :type num_tokens: int
        :param work_dtype: Dtype to emit the tables in.
        :type work_dtype: str
        :return: ``(cos_e, sin_e)`` tensors of shape ``(num_tokens, head_dim)``.
        :rtype: Tuple[Any, Any]
        """
        repeats = num_tokens // self.num_grid_tokens
        cos_e = self._cos_table
        sin_e = self._sin_table
        if repeats > 1:
            # The SAME per-position angle is reused for every block. No block
            # index enters this table -- doing so would encode temporal position
            # in the rotation, which is carried additively elsewhere.
            cos_e = np.tile(cos_e, (repeats, 1))
            sin_e = np.tile(sin_e, (repeats, 1))
        return (
            ops.convert_to_tensor(cos_e, dtype=work_dtype),
            ops.convert_to_tensor(sin_e, dtype=work_dtype),
        )

    def _static_token_count(self, x: Any, name: str) -> int:
        """Read a statically-known token count from a tensor.

        :param x: Tensor of shape ``(batch, num_heads, num_tokens, head_dim)``.
        :type x: Any
        :param name: Human-readable tensor name for the error message.
        :type name: str
        :return: The token count.
        :rtype: int
        :raises ValueError: If the token axis is dynamic.
        """
        num_tokens = x.shape[-2]
        if num_tokens is None:
            raise ValueError(
                f"AxialRoPE2D requires a STATIC token axis on {name}; got a "
                f"dynamic dimension in shape {x.shape}. Trace with a static "
                f"input signature."
            )
        return int(num_tokens)

    def call(
            self,
            query: Any,
            key: Optional[Any] = None,
            num_k_exclude: int = 0,
            training: Optional[bool] = None,
    ) -> Union[Any, Tuple[Any, Any]]:
        """Rotate ``query`` and, when given, ``key``.

        :param query: Post-head-split queries,
            ``(batch, num_heads, H * W, head_dim)``.
        :type query: Any
        :param key: Optional post-head-split keys,
            ``(batch, num_heads, num_k, head_dim)``.
        :type key: Optional[Any]
        :param num_k_exclude: Number of trailing key rows to leave completely
            untouched (SAM 2's object-pointer tokens, which carry no spatial
            position). Must be in ``[0, num_k]``.
        :type num_k_exclude: int
        :param training: Unused; present for the Keras call contract.
        :type training: Optional[bool]
        :return: ``query_rot`` when ``key is None``, else
            ``(query_rot, key_rot)``.
        :rtype: Union[Any, Tuple[Any, Any]]
        :raises ValueError: If ``num_k_exclude`` is out of range, if the rotated
            key length is not a valid multiple of the query grid, or if a token
            axis is dynamic.
        """
        # DECISION plan-2026-08-04T044628-4c240b4c/D-005
        # Compute the rotation in a NEVER-NARROWING working dtype and cast back
        # at the end. Do NOT "simplify" this to `ops.cast(x, self.compute_dtype)`
        # -- that narrows the rotation to float16 under mixed_float16, where a
        # 64x64 grid's largest angle (t=63 at the lowest frequency band) loses
        # roughly three decimal digits and the rotation stops being orthogonal to
        # the tolerance downstream attention assumes. Do NOT hardcode
        # `"float32"` either -- that would silently pin a float64 model to
        # float32 rotations.
        #
        # This is the SAME rule as `continuous_rope_embedding.py:259` /
        # `continuous_sin_cos_embedding.py:335`, but this site is materially
        # simpler than those two: the angle table here is a NumPy constant, not
        # an `add_weight` variable, so the autocast `Mul` dtype-mismatch failure
        # those two comments describe cannot occur here at all. Only the
        # never-narrow half of the rule applies. See decisions.md D-005 for why
        # this was not promoted to a shared helper yet (the prior plan's
        # pre-commitment triggers at the FIFTH site; this is the third).
        work_dtype = "float64" if self.compute_dtype == "float64" else "float32"

        num_q = self._static_token_count(query, "query")
        if num_q != self.num_grid_tokens:
            raise ValueError(
                f"query token count ({num_q}) must equal feat_shape H*W "
                f"({self.num_grid_tokens})"
            )

        cos_q, sin_q = self._tables(num_q, work_dtype)
        query_rot = ops.cast(
            self._apply(ops.cast(query, work_dtype), cos_q, sin_q),
            self.compute_dtype,
        )

        if key is None:
            return query_rot

        num_k = self._static_token_count(key, "key")
        if not 0 <= num_k_exclude <= num_k:
            raise ValueError(
                f"num_k_exclude ({num_k_exclude}) must be in [0, {num_k}]"
            )

        num_k_rope = num_k - num_k_exclude
        if num_k_rope == 0:
            # Every key row is excluded (e.g. a memory sequence of object
            # pointers only). Rotation is a no-op; return the key untouched
            # rather than dividing by the grid size.
            return query_rot, key

        if self.repeat_k:
            if num_k_rope % self.num_grid_tokens != 0:
                raise ValueError(
                    f"with repeat_k=True the rotated key length ({num_k_rope} = "
                    f"{num_k} - {num_k_exclude}) must be an exact multiple of "
                    f"feat_shape H*W ({self.num_grid_tokens})"
                )
        elif num_k_rope != self.num_grid_tokens:
            raise ValueError(
                f"with repeat_k=False the rotated key length ({num_k_rope} = "
                f"{num_k} - {num_k_exclude}) must equal feat_shape H*W "
                f"({self.num_grid_tokens}); pass repeat_k=True to broadcast the "
                f"spatial angle table across multiple memory frames"
            )

        key_head = key[..., :num_k_rope, :]
        cos_k, sin_k = self._tables(num_k_rope, work_dtype)
        key_rot = ops.cast(
            self._apply(ops.cast(key_head, work_dtype), cos_k, sin_k),
            self.compute_dtype,
        )

        if num_k_exclude > 0:
            key_rot = ops.concatenate(
                [key_rot, ops.cast(key[..., num_k_rope:, :], self.compute_dtype)],
                axis=-2,
            )
        return query_rot, key_rot

    # -----------------------------------------------------------------
    # shape / config
    # -----------------------------------------------------------------

    def compute_output_shape(
            self,
            query_shape: Tuple[Optional[int], ...],
            key_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """Return the output shape(s); the rotation is shape-preserving.

        Derived from the input shapes and stored config only — never from weight
        shapes (this layer owns no weights).

        :param query_shape: Query shape.
        :type query_shape: Tuple[Optional[int], ...]
        :param key_shape: Optional key shape.
        :type key_shape: Optional[Tuple[Optional[int], ...]]
        :return: ``query_shape`` alone, or ``[query_shape, key_shape]``.
        :rtype: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        """
        if key_shape is None:
            return tuple(query_shape)
        return [tuple(query_shape), tuple(key_shape)]

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "feat_shape": self.feat_shape,
            "theta": self.theta,
            "scale_pos": self.scale_pos,
            "repeat_k": self.repeat_k,
        })
        return config

# ---------------------------------------------------------------------
