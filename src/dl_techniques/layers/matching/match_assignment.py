"""
LightGlue soft assignment: log-space partial assignment with a dustbin.

Given the final descriptors of two images, :class:`MatchAssignment` produces an
``(M + 1, N + 1)`` matrix of log scores. The extra row and column are the dustbin
(``"this keypoint has no match"``). :func:`filter_matches` turns that matrix into hard
matches.

Architecture:
    ::

        desc0 (B,M,D)                          desc1 (B,N,D)
          |                                      |
          | final_proj: Dense(D), * D^-1/4       | final_proj (SHARED), * D^-1/4
          v                                      v
        md0 ------------ sim = md0 md1^T --------- md1          sim (B,M,N)
                              |
        matchability: Dense(1) on each side  ->  z0 (B,M), z1 (B,N)
                              |
        interior[i,j] = log_softmax_j(sim)[i,j]      (over image-1 keypoints)
                      + log_softmax_i(sim)[i,j]      (over image-0 keypoints)
                      + logsigmoid(z0_i) + logsigmoid(z1_j)
        last column   = logsigmoid(-z0)              (image-0 keypoint unmatched)
        last row      = logsigmoid(-z1)              (image-1 keypoint unmatched)
        corner        = 0

What the output is NOT. It is not a normalised distribution. For a real row ``i`` the
interior is ``r_ij * c_ij * sigma(z0_i) * sigma(z1_j)`` in probability space, with ``r``
the row softmax and ``c`` the column softmax. Since ``sum_j r_ij = 1`` and ``c_ij <= 1``,
``sigma(-z0_i) <= sum_j exp(S[i, j]) + exp(S[i, dustbin]) <= 1``. The row total is bounded
by one and below by the dustbin mass, but it is generally strictly below one. The
same holds for columns with ``z1``.

Weight mapping from the official PyTorch state dict (``log_assignment.i.*`` and
``token_confidence.i.token.0.*``): every sublayer is a Keras attribute named after its
torch counterpart, a ``Linear`` weight ``(out, in)`` goes to the Dense ``kernel`` by
TRANSPOSE (``kernel = W.T``).

    ==========================================  ====================================
    torch                                       Keras
    ==========================================  ====================================
    ``matchability.{weight,bias}``              ``layer.matchability.{kernel,bias}``
    ``final_proj.{weight,bias}``                ``layer.final_proj.{kernel,bias}``
    ==========================================  ====================================

References:
    - Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). "LightGlue: Local
      Feature Matching at Light Speed". arXiv:2306.13643.
    - Sarlin, P.-E. et al. (2020). "SuperGlue: Learning Feature Matching with Graph
      Neural Networks". arXiv:1911.11763.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.common import apply_attention_mask, mask_dtype
from dl_techniques.utils.dtype_policy import mask_sentinel
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _real_mask(mask: Optional[Any], dtype: str) -> Optional[Any]:
    """``(B, N)`` boolean keep mask (True = real) from a bool or 0/1 mask, or ``None``."""
    if mask is None:
        return None
    return keras.ops.cast(mask, dtype) > 0.0


def _double_log_softmax(logits: Any) -> Any:
    """``log_softmax`` over image-1 keypoints plus ``log_softmax`` over image-0 keypoints.

    :param logits: ``(B, M, N)``, at least float32, padded entries already biased down.
    :return: ``(B, M, N)``; row ``i`` of the first term and column ``j`` of the second
        each normalise to one over their axis.
    """
    return keras.ops.log_softmax(logits, axis=2) + keras.ops.log_softmax(logits, axis=1)


def _certainties(z0: Any, z1: Any) -> Any:
    """``logsigmoid(z0_i) + logsigmoid(z1_j)`` as ``(B, M, N)`` from ``(B, M, 1)``, ``(B, N, 1)``."""
    return keras.ops.log_sigmoid(z0) + keras.ops.transpose(keras.ops.log_sigmoid(z1), (0, 2, 1))


def _assemble_scores(interior: Any, z0: Any, z1: Any) -> Any:
    """Append the dustbin column ``logsigmoid(-z0)``, row ``logsigmoid(-z1)`` and a 0 corner.

    :param interior: ``(B, M, N)``.
    :param z0: ``(B, M, 1)`` matchability logits of image 0.
    :param z1: ``(B, N, 1)`` matchability logits of image 1.
    :return: ``(B, M + 1, N + 1)``.
    """
    dustbin_col = keras.ops.log_sigmoid(-z0)                                      # (B, M, 1)
    dustbin_row = keras.ops.transpose(keras.ops.log_sigmoid(-z1), (0, 2, 1))      # (B, 1, N)
    corner = keras.ops.zeros_like(dustbin_col[:, :1, :])                          # (B, 1, 1)
    top = keras.ops.concatenate([interior, dustbin_col], axis=2)
    bottom = keras.ops.concatenate([dustbin_row, corner], axis=2)
    return keras.ops.concatenate([top, bottom], axis=1)


@register_dl_technique("dl_techniques.layers.matching.match_assignment")
class MatchAssignment(keras.layers.Layer):
    """Dustbin-augmented log assignment between two sets of descriptors.

    The scores are always computed in at least float32 (``mask_dtype``), so the
    output dtype is float32 under any half-precision policy and float64 under a float64
    policy. Padded keypoints (``mask`` 0) are removed from both softmaxes through
    :func:`apply_attention_mask`, so they take no probability mass from the real ones.

    **Padded rows and columns of the output.** The reference has no mask: it slices the
    padding off first. This layer's padded-row/column masking is an extension, and its
    contract is: every entry of the ``(M + 1, N + 1)`` output whose row is a padded
    image-0 keypoint or whose column is a padded image-1 keypoint is the finite constant
    ``0.0`` (no sentinel, no NaN, no gradient). That value means nothing; consumers
    (:func:`filter_matches`, the loss, the metric) must use the masks. Entries of real
    rows and real columns, the dustbin included, equal what the reference computes on the
    unpadded keypoints.

    :param dim: Descriptor width ``D``.
    :type dim: int
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar final_proj: ``Dense(dim)`` applied to both images before the similarity.
    :ivar matchability: ``Dense(1)`` producing the per-keypoint matchability logit.

    Call arguments:
        desc0: ``(B, M, dim)``.
        desc1: ``(B, N, dim)``.
        mask0: Optional ``(B, M)`` keep mask, bool or 0/1, 1 = real keypoint.
        mask1: Optional ``(B, N)`` keep mask.

    Output:
        ``(scores, sim)``: ``scores`` ``(B, M + 1, N + 1)`` log scores as described
        above, and ``sim`` the raw ``(B, M, N)`` similarity (unmasked, the reference
        returns it too).

    :raises ValueError: From ``__init__`` for a non-positive ``dim``; from ``build()``
        for a wrong rank or last dimension.

    Example:

    .. code-block:: python

        layer = MatchAssignment(dim=64)
        scores, sim = layer(keras.random.normal((2, 30, 64)), keras.random.normal((2, 40, 64)))
        # scores: (2, 31, 41)
    """

    def __init__(self, dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        policy = self.dtype_policy
        self.final_proj = keras.layers.Dense(dim, dtype=policy, name="final_proj")
        self.matchability = keras.layers.Dense(1, dtype=policy, name="matchability")

    def build(self, desc0_shape: Tuple[Any, ...], desc1_shape: Optional[Tuple[Any, ...]] = None,
              mask0_shape: Optional[Tuple[Any, ...]] = None,
              mask1_shape: Optional[Tuple[Any, ...]] = None) -> None:
        """Build both sublayers explicitly.

        :raises ValueError: If a descriptor shape is not rank 3 with last dim ``dim``.
        """
        if self.built:
            return
        for name, shape in (("desc0", desc0_shape), ("desc1", desc1_shape)):
            if shape is None:
                continue
            if len(shape) != 3 or (shape[-1] is not None and shape[-1] != self.dim):
                raise ValueError(f"{name} must be (batch, num_points, {self.dim}), got {shape}")
        self.final_proj.build((None, None, self.dim))
        self.matchability.build((None, None, self.dim))
        super().build(desc0_shape)

    def get_matchability(self, desc: Any) -> Any:
        """``sigmoid(matchability(desc))`` as ``(B, N)``, in at least float32.

        :param desc: ``(B, N, dim)`` descriptors of one image.
        """
        md = mask_dtype(self.compute_dtype)
        return keras.ops.sigmoid(keras.ops.cast(self.matchability(desc), md))[..., 0]

    def call(self, desc0: Any, desc1: Any, mask0: Optional[Any] = None,
             mask1: Optional[Any] = None) -> Tuple[Any, Any]:
        """Compute the ``(B, M + 1, N + 1)`` log assignment and the raw similarity."""
        md = mask_dtype(self.compute_dtype)
        quarter = float(self.dim) ** -0.25
        md0 = keras.ops.cast(self.final_proj(desc0), md) * quarter
        md1 = keras.ops.cast(self.final_proj(desc1), md) * quarter
        sim = keras.ops.einsum("bmd,bnd->bmn", md0, md1)

        z0 = keras.ops.cast(self.matchability(desc0), md)           # (B, M, 1)
        z1 = keras.ops.cast(self.matchability(desc1), md)           # (B, N, 1)

        real0, real1 = _real_mask(mask0, md), _real_mask(mask1, md)
        logits = sim
        if real0 is not None or real1 is not None:
            ones0 = keras.ops.ones_like(z0[..., 0], dtype="bool")
            ones1 = keras.ops.ones_like(z1[..., 0], dtype="bool")
            r0 = ones0 if real0 is None else real0
            r1 = ones1 if real1 is None else real1
            keep = keras.ops.logical_and(r0[:, :, None], r1[:, None, :])
            # rescue off: a row with no real column stays a flat finite row, not a spread
            # over padded columns; it is overwritten with 0 below either way
            logits = apply_attention_mask(sim, keep, rescue_axis=None)

        interior = _double_log_softmax(logits) + _certainties(z0, z1)   # (B, M, N)
        scores = _assemble_scores(interior, z0, z1)                      # (B, M+1, N+1)

        if real0 is not None or real1 is not None:
            one = keras.ops.ones_like(z0[:, :1, 0], dtype="bool")                       # (B, 1)
            full0 = keras.ops.concatenate([r0, one], axis=1)
            full1 = keras.ops.concatenate([r1, one], axis=1)
            keep_full = keras.ops.logical_and(full0[:, :, None], full1[:, None, :])
            scores = keras.ops.where(keep_full, scores, keras.ops.zeros_like(scores))
        return scores, sim

    def compute_output_shape(self, desc0_shape: Tuple[Any, ...],
                             desc1_shape: Tuple[Any, ...], mask0_shape: Optional[Any] = None,
                             mask1_shape: Optional[Any] = None) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """``((B, M + 1, N + 1), (B, M, N))``."""
        batch, m_pts = desc0_shape[0], desc0_shape[1]
        n_pts = desc1_shape[1]
        m_plus = None if m_pts is None else m_pts + 1
        n_plus = None if n_pts is None else n_pts + 1
        return (batch, m_plus, n_plus), (batch, m_pts, n_pts)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration."""
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


# ---------------------------------------------------------------------


def filter_matches(
        log_scores: Any,
        threshold: float,
        mask0: Optional[Any] = None,
        mask1: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Any]:
    """Hard mutual-nearest matches from a log assignment, per the LightGlue reference.

    Interface contract. ``log_scores`` is ``(B, M + 1, N + 1)`` as returned by
    :class:`MatchAssignment` (the dustbin row and column are ignored). Each real row and
    column takes its argmax over the interior; a pair is kept when both argmaxes agree
    (mutual) and ``exp(score) > threshold``. Optional ``mask0`` ``(B, M)`` and ``mask1``
    ``(B, N)`` (bool or 0/1) mark real keypoints; a padded keypoint is never matched and
    never chosen as a partner. Pure function, no variables, graph-safe.

    :param log_scores: ``(B, M + 1, N + 1)`` log scores.
    :param threshold: Match threshold on ``exp(score)``; strictly greater passes.
    :param mask0: Optional keep mask of image 0.
    :param mask1: Optional keep mask of image 1.
    :return: ``(matches0 (B, M) int32, matches1 (B, N) int32, matching_scores0 (B, M),
        matching_scores1 (B, N))``. ``matches*`` hold the partner index or ``-1``;
        scores are ``exp(score)`` for mutual pairs (also those below the threshold) and
        0 elsewhere, exactly as in the reference.
    """
    shape = keras.ops.shape(log_scores)
    m_pts, n_pts = shape[1] - 1, shape[2] - 1
    core = log_scores[:, :m_pts, :n_pts]
    real0, real1 = _real_mask(mask0, "float32"), _real_mask(mask1, "float32")
    if real0 is not None or real1 is not None:
        r0 = keras.ops.ones_like(core[:, :, 0], dtype="bool") if real0 is None else real0
        r1 = keras.ops.ones_like(core[:, 0, :], dtype="bool") if real1 is None else real1
        keep = keras.ops.logical_and(r0[:, :, None], r1[:, None, :])
        core = keras.ops.where(keep, core, mask_sentinel(core.dtype))
    else:
        r0 = r1 = None

    m0 = keras.ops.cast(keras.ops.argmax(core, axis=2), "int32")     # (B, M)
    m1 = keras.ops.cast(keras.ops.argmax(core, axis=1), "int32")     # (B, N)
    max0 = keras.ops.max(core, axis=2)
    max1 = keras.ops.max(core, axis=1)
    back0 = keras.ops.take_along_axis(m1, m0, axis=1)                # m1[m0[i]]
    back1 = keras.ops.take_along_axis(m0, m1, axis=1)                # m0[m1[j]]
    idx0 = keras.ops.arange(m_pts, dtype="int32")[None, :]
    idx1 = keras.ops.arange(n_pts, dtype="int32")[None, :]
    mutual0 = back0 == idx0
    mutual1 = back1 == idx1
    if r0 is not None:
        # a padded keypoint is never mutual; the partner of a real one is real by the
        # sentinel above except in an image with no real keypoint, which this closes
        mutual0 = keras.ops.logical_and(
            mutual0, keras.ops.logical_and(r0, keras.ops.take_along_axis(r1, m0, axis=1))
        )
        mutual1 = keras.ops.logical_and(
            mutual1, keras.ops.logical_and(r1, keras.ops.take_along_axis(r0, m1, axis=1))
        )

    zeros = keras.ops.zeros_like(max0)
    mscores0 = keras.ops.where(mutual0, keras.ops.exp(max0), zeros)
    mscores1 = keras.ops.where(
        mutual1, keras.ops.take_along_axis(mscores0, m1, axis=1), keras.ops.zeros_like(max1)
    )
    valid0 = keras.ops.logical_and(mutual0, mscores0 > threshold)
    valid1 = keras.ops.logical_and(mutual1, keras.ops.take_along_axis(valid0, m1, axis=1))
    minus_one0 = -keras.ops.ones_like(m0)
    minus_one1 = -keras.ops.ones_like(m1)
    return (
        keras.ops.where(valid0, m0, minus_one0),
        keras.ops.where(valid1, m1, minus_one1),
        mscores0,
        mscores1,
    )
