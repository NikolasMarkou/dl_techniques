"""
3-axis (frame/height/width) Rotary Position Embedding for video tokens.

Ports LeVJEPA's ``RoPEAttention`` rotary machinery (``rotate_queries_or_keys``
plus ``separate_positions``, ``module.py``) as a standalone layer. Each token
carries a flat grid index that decomposes into ``(frame_id, height_id,
width_id)``; the head dimension is split into three equal-ish bands, one per
axis, and each band is rotated using ONLY its own axis' position -- the frame
band never sees height or width, and so on.

This is deliberately a NEW, standalone layer rather than an extension of
:class:`~dl_techniques.layers.embedding.multi_axis_rope.Ideogram4MRoPE`. See
``decisions.md`` D-004 of this plan for why: the two mechanisms differ in
kind (an EQUAL three-way band split here vs. ``mrope_section``'s asymmetric,
3-strided slot allocation there), not just in parameters.

Pairing convention -- read this before touching the rotation math:
    The reference's ``rotate_queries_or_keys`` reshapes each band's last
    dimension into ``(D/2, 2)`` (``x.unflatten(-1, (-1, 2))``), unbinds the
    pair axis into ``(y1, y2)``, and recombines as ``(-y2, y1)``. That pairs
    ADJACENT channels ``(2i, 2i+1)`` -- the INTERLEAVED convention, matching
    :class:`~dl_techniques.layers.embedding.axial_rope_2d.AxialRoPE2D` in
    this same package -- NOT the SPLIT-HALF (GPT-NeoX) convention that
    :class:`~dl_techniques.layers.embedding.multi_axis_rope.Ideogram4MRoPE`
    uses (channel ``j`` paired with ``j + D/2``). Verify this by hand before
    changing it: ``y = unflatten(x, (-1, 2))`` groups ``x[2i]`` and
    ``x[2i+1]`` into one pair (reshape does not skip channels), so
    ``y1 = x[0::2]``, ``y2 = x[1::2]``, and the output
    ``stack((-y2, y1)).flatten(-2)`` is exactly PyTorch's/HF's
    ``rotate_every_two``, the textbook INTERLEAVED form -- the same
    conclusion :func:`~dl_techniques.layers.embedding.axial_rope_2d.
    AxialRoPE2D._rotate_adjacent_pairs` reaches for an identical reshape.
    ``multi_axis_rope.py``'s own docstring calls this same construction
    "SPLIT-HALF" when describing ITS layer, but that is because
    ``Ideogram4MRoPE`` builds its table differently (``concat([freqs,
    freqs])`` then ``_rotate_half`` slices the FIRST/SECOND HALVES, not
    adjacent pairs) -- a different code path from this reference's
    ``unflatten``-based one. Do not import that terminology here without
    re-deriving it; this layer's rotation is interleaved.

Architecture:
    .. code-block:: text

        flat token index i  (per (t, h, w) grid, row-major t/h/w flatten)
                │
                ▼  separate_positions(i, H_patches, W_patches)
        frame_id = i // (H_patches*W_patches)
        rem      = i %  (H_patches*W_patches)
        height_id = rem // W_patches
        width_id  = rem %  W_patches

        head_dim D  ->  band = 2 * ((D // 3) // 2)   (rounded DOWN, even)
                │
        ┌───────┼────────────────┬────────────────┬──────────────┐
        ▼       ▼                ▼                 ▼              ▼
      [0, band)          [band, 2*band)     [2*band, 3*band)   [3*band, D)
      frame band           height band          width band     unrotated
      rotated by            rotated by           rotated by     (pass
      frame_id               height_id             width_id     through)

        Each band's rotation, independently:

          omega = 1 / theta ** (arange(band/2) / (band/2))
          freq  = pos[..., None] * omega              # (..., N, band/2)
          sin_e, cos_e = repeat(sin(freq), 2), repeat(cos(freq), 2)  # (..., N, band)
          rotate_pairs(x): (x[2i], x[2i+1]) -> (-x[2i+1], x[2i])     # INTERLEAVED
          out = x * cos_e + rotate_pairs(x) * sin_e

    The three rotated bands and the unrotated remainder are concatenated back
    to width ``D`` in that order: ``[frame | height | width | pass-through]``.

Foundational Mathematics:
    Standard 1D RoPE (Su et al., 2021) applied independently per axis, each
    over its own band of channels: rotating a 2D sub-vector by angle ``phi``
    is the complex multiply ``(a + bi)(cos phi + i sin phi)``, which is
    exactly ``out = x*cos_e + rotate_pairs(x)*sin_e`` above. Because each
    axis' rotation is orthogonal and additive in its own angle, the inner
    product between a rotated query and a rotated key, restricted to one
    band, depends only on that axis' relative displacement -- the
    relative-position property RoPE is built to have, reproduced
    independently along frame, height and width.

References:
    - Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
      "RoFormer: Enhanced Transformer with Rotary Position Embedding".
      arXiv:2104.09864.
    - LeVJEPA PyTorch reference, ``module.py::RoPEAttention`` /
      ``rotate_queries_or_keys`` / ``separate_positions`` (pasted transcript;
      no public arXiv id in this plan's context).
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.embedding.video_rope")
class VideoRoPE3D(keras.layers.Layer):
    """Rotate post-head-split ``q``/``k`` by a token's 3D video grid position.

    Takes ``q``/``k`` tensors of shape ``(batch, num_heads, num_tokens,
    head_dim)`` and, for each token, a flat grid index that decomposes into
    ``(frame_id, height_id, width_id)`` via ``separate_positions``. The head
    dimension is split into three equal-ish bands (frame, height, width),
    each rotated by ONLY its own axis' position, using the INTERLEAVED
    pairing convention (see the module docstring's "Pairing convention"
    section -- do not assume split-half).

    The layer owns no weights; every table is a pure function of the
    per-call position ids, recomputed each call (unlike
    :class:`~dl_techniques.layers.embedding.axial_rope_2d.AxialRoPE2D`'s
    static build-time table, this layer's grid shape -- and, after token
    dropping, its ordering -- is genuinely dynamic per call, not fixed at
    construction).

    :param head_dim: Per-head dimensionality. Must be a positive integer
        large enough that the equal three-way band split (rounded down to an
        even width per band) leaves at least one rotated channel per axis.
    :type head_dim: int
    :param rope_theta: Rotary base frequency, shared by all three axes.
        Defaults to ``10000.0``.
    :type rope_theta: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    Input shape:
        ``q``, ``k``: 4D tensors ``(batch, num_heads, num_tokens, head_dim)``.
        ``token_ids`` (optional): integer tensor of shape ``(num_tokens,)``
        or ``(batch, num_tokens)`` giving each surviving token's TRUE flat
        grid index (needed after token dropping, where sequence order no
        longer matches grid order). Defaults to
        ``arange(num_frames * height_patches * width_patches)``, the
        no-dropping case.

    Output shape:
        A pair ``(q_rotated, k_rotated)``, each the same shape as its input.

    :raises ValueError: If ``head_dim`` is not a positive integer, if
        ``rope_theta`` is not positive, or if ``head_dim`` is too small for
        the equal three-way band split to leave a positive band width.
        Raised from ``__init__``.
    :raises ValueError: If ``q`` or ``k`` is not rank-4, or if their last
        dimension does not equal ``head_dim``. Raised from ``call()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding.video_rope import VideoRoPE3D

        rope = VideoRoPE3D(head_dim=24)
        q = keras.random.normal((2, 4, 2 * 3 * 3, 24))
        k = keras.random.normal((2, 4, 2 * 3 * 3, 24))
        q_rot, k_rot = rope(
            q, k, num_frames=2, height_patches=3, width_patches=3,
        )
        q_rot.shape  # (2, 4, 18, 24)
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10000.0,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and precompute the static band widths.

        No table is built here; every table is a function of the per-call
        position ids and is computed inside :meth:`call`.

        :param head_dim: Per-head dimensionality.
        :type head_dim: int
        :param rope_theta: Rotary base frequency.
        :type rope_theta: float
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``head_dim`` is not a positive integer, if
            ``rope_theta`` is not positive, or if the equal three-way band
            split leaves a non-positive band width.
        """
        super().__init__(**kwargs)

        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be a positive integer, got {head_dim}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")

        self.head_dim = head_dim
        self.rope_theta = float(rope_theta)

        # DECISION plan-2026-09-03T113223-2a714a91/D-010
        # Equal three-way band split, each band rounded DOWN to the nearest
        # even width so an intra-band pair never straddles a band boundary;
        # any remainder is passed through unrotated. Ported from the
        # reference's `d_dim = h_dim = w_dim = 2 * ((head_dim // 3) // 2)`
        # plus its leftover-passthrough (`if s < head_dim: concat with
        # q[..., s:]`). Do NOT round UP (e.g. ceil-to-even): that can make
        # bands overlap and double-rotate shared channels. Do NOT
        # redistribute the remainder round-robin across bands: the
        # reference leaves it unrotated, not reassigned. See decisions.md
        # D-010.
        band = 2 * ((head_dim // 3) // 2)
        if band <= 0:
            raise ValueError(
                f"head_dim ({head_dim}) is too small for a 3-axis equal-band "
                f"RoPE split: 2 * ((head_dim // 3) // 2) must be positive"
            )
        self._band = band
        self._rotated_dim = 3 * band
        self._pass_dim = head_dim - self._rotated_dim

        logger.info(
            f"Initialized VideoRoPE3D with head_dim={self.head_dim}, "
            f"band_width={self._band} (x3 axes), pass_through={self._pass_dim}"
        )

    # -----------------------------------------------------------------
    # position decomposition
    # -----------------------------------------------------------------

    @staticmethod
    def _separate_positions(
        token_ids: Any,
        height_patches: int,
        width_patches: int,
    ) -> Tuple[Any, Any, Any]:
        """Decompose a flat grid index into ``(frame_id, height_id, width_id)``.

        Ports the reference's ``separate_positions``: the grid is read as a
        row-major flattening of ``(frame, height, width)``, matching
        :func:`~dl_techniques.layers.embedding.patch_embed_3d.PatchEmbed3D`'s
        and
        :func:`~dl_techniques.layers.embedding.sincos_pos_embed_3d.get_3d_sincos_pos_embed`'s
        flatten order.

        :param token_ids: Integer tensor of flat grid indices, any shape.
        :type token_ids: Any
        :param height_patches: Number of patches along the height axis.
        :type height_patches: int
        :param width_patches: Number of patches along the width axis.
        :type width_patches: int
        :return: ``(frame_id, height_id, width_id)``, each the same shape as
            ``token_ids``.
        :rtype: Tuple[Any, Any, Any]
        """
        tokens_per_frame = height_patches * width_patches
        frame_id = token_ids // tokens_per_frame
        rem = token_ids % tokens_per_frame
        height_id = rem // width_patches
        width_id = rem % width_patches
        return frame_id, height_id, width_id

    # -----------------------------------------------------------------
    # rotation
    # -----------------------------------------------------------------

    def _rotate_band(self, x_band: Any, pos: Any) -> Any:
        """Rotate one axis' band using that axis' position ids.

        :param x_band: Tensor of shape ``(batch, num_heads, num_tokens,
            band_width)``.
        :type x_band: Any
        :param pos: Integer position ids, shape ``(num_tokens,)`` or
            ``(batch, num_tokens)``.
        :type pos: Any
        :return: Rotated tensor, same shape as ``x_band``.
        :rtype: Any
        """
        d = self._band
        half = d // 2

        omega = 1.0 / (
            self.rope_theta ** (ops.arange(half, dtype="float32") / float(half))
        )
        pos_f = ops.cast(pos, "float32")
        # (..., N, half), where ... is empty (pos rank 1) or (batch,) (rank 2).
        freq = ops.expand_dims(pos_f, axis=-1) * omega

        # Each angle serves an ADJACENT channel pair -- INTERLEAVED, see the
        # module docstring. repeat(..., 2) turns [a0, a1, ...] into
        # [a0, a0, a1, a1, ...].
        sin_e = ops.repeat(ops.sin(freq), 2, axis=-1)
        cos_e = ops.repeat(ops.cos(freq), 2, axis=-1)

        if len(pos.shape) == 2:
            # (batch, N, band) -> (batch, 1, N, band) to broadcast over heads.
            sin_e = ops.expand_dims(sin_e, axis=1)
            cos_e = ops.expand_dims(cos_e, axis=1)

        sin_e = ops.cast(sin_e, x_band.dtype)
        cos_e = ops.cast(cos_e, x_band.dtype)

        # INTERLEAVED pair rotation: (x[2i], x[2i+1]) -> (-x[2i+1], x[2i]).
        # Reshape-based, matching AxialRoPE2D._rotate_adjacent_pairs exactly
        # (same conclusion, independently re-derived from this reference's
        # `unflatten(-1, (-1, 2))`).
        lead = ops.shape(x_band)[:-1]
        pairs = ops.reshape(x_band, (*lead, half, 2))
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated = ops.reshape(ops.stack([-odd, even], axis=-1), (*lead, d))

        return x_band * cos_e + rotated * sin_e

    def _rotate(self, x: Any, frame_id: Any, height_id: Any, width_id: Any) -> Any:
        """Rotate all three bands of one tensor (``q`` or ``k``) and reassemble.

        :param x: Tensor of shape ``(batch, num_heads, num_tokens, head_dim)``.
        :type x: Any
        :param frame_id: Frame position ids.
        :type frame_id: Any
        :param height_id: Height position ids.
        :type height_id: Any
        :param width_id: Width position ids.
        :type width_id: Any
        :return: Rotated tensor, same shape as ``x``.
        :rtype: Any
        """
        d = self._band
        x_frame = x[..., :d]
        x_height = x[..., d:2 * d]
        x_width = x[..., 2 * d:3 * d]

        rotated_frame = self._rotate_band(x_frame, frame_id)
        rotated_height = self._rotate_band(x_height, height_id)
        rotated_width = self._rotate_band(x_width, width_id)

        parts = [rotated_frame, rotated_height, rotated_width]
        if self._pass_dim > 0:
            # Leftover channels the equal-band split could not cover; passed
            # through unrotated, matching the reference's concat-remainder.
            parts.append(x[..., 3 * d:])
        return ops.concatenate(parts, axis=-1)

    # -----------------------------------------------------------------
    # call / shape / config
    # -----------------------------------------------------------------

    def call(
        self,
        q: Any,
        k: Any,
        num_frames: int,
        height_patches: int,
        width_patches: int,
        token_ids: Optional[Any] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Any]:
        """Rotate ``q`` and ``k`` by each token's 3D grid position.

        :param q: Query tensor, ``(batch, num_heads, num_tokens, head_dim)``.
        :type q: Any
        :param k: Key tensor, same shape convention as ``q``.
        :type k: Any
        :param num_frames: Number of frame positions in the grid. Used only
            to build the default ``token_ids`` when none is given.
        :type num_frames: int
        :param height_patches: Number of patches along the height axis.
        :type height_patches: int
        :param width_patches: Number of patches along the width axis.
        :type width_patches: int
        :param token_ids: Optional integer tensor of each token's TRUE flat
            grid index, shape ``(num_tokens,)`` or ``(batch, num_tokens)``.
            Required after token dropping, where sequence order no longer
            matches grid order. Defaults to
            ``arange(num_frames * height_patches * width_patches)``.
        :type token_ids: Optional[Any]
        :param training: Unused (this layer has no training-specific
            behavior).
        :type training: Optional[bool]
        :return: ``(q_rotated, k_rotated)``, each the same shape as its
            input.
        :rtype: Tuple[Any, Any]
        :raises ValueError: If ``q`` or ``k`` is not rank-4, or if their last
            dimension does not equal ``head_dim``.
        """
        for x, name in ((q, "q"), (k, "k")):
            if len(x.shape) != 4:
                raise ValueError(
                    f"{name} must be rank-4 (batch, num_heads, num_tokens, "
                    f"head_dim), got shape {x.shape}"
                )
            if x.shape[-1] is not None and x.shape[-1] != self.head_dim:
                raise ValueError(
                    f"{name} last dimension ({x.shape[-1]}) must equal "
                    f"head_dim ({self.head_dim})"
                )

        if token_ids is None:
            num_tokens = num_frames * height_patches * width_patches
            token_ids = ops.arange(num_tokens, dtype="int32")

        frame_id, height_id, width_id = self._separate_positions(
            token_ids, height_patches, width_patches
        )

        q_rot = self._rotate(q, frame_id, height_id, width_id)
        k_rot = self._rotate(k, frame_id, height_id, width_id)
        return q_rot, k_rot

    def compute_output_shape(
        self,
        q_shape: Tuple[Optional[int], ...],
        k_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Return the ``(q, k)`` output shapes -- rotation preserves shape.

        :param q_shape: Shape of the query tensor.
        :type q_shape: Tuple[Optional[int], ...]
        :param k_shape: Shape of the key tensor; defaults to ``q_shape``.
        :type k_shape: Optional[Tuple[Optional[int], ...]]
        :return: ``(q_shape, k_shape)`` unchanged.
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        return q_shape, (k_shape if k_shape is not None else q_shape)

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
            }
        )
        return config


# ---------------------------------------------------------------------
