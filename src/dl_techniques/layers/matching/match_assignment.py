"""LightGlue soft assignment: log-space partial assignment with a dustbin.

:class:`MatchAssignment` turns the final descriptors of two images into an
``(M + 1, N + 1)`` matrix of log scores whose extra row and column are dustbins, "no
match". :func:`filter_matches` turns that matrix into hard mutual matches. Instead of
SuperGlue's Sinkhorn iterations, the interior is one row softmax plus one column
softmax, weighted by per-keypoint matchability::

    interior[i, j] = log_softmax_j(sim)[i, j] + log_softmax_i(sim)[i, j]
                     + logsigmoid(z0_i) + logsigmoid(z1_j)

The result is not a normalised distribution: each real row's total probability lies
between its dustbin mass and one, and likewise for columns. Padded rows and columns
come out as 0.0, which means nothing, so every consumer needs the keypoint masks.

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
    :return: ``(B, M, N)``. In probability space, row ``i`` of the first term and
        column ``j`` of the second each sum to one.
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
    dustbin_col = keras.ops.log_sigmoid(-z0)
    dustbin_row = keras.ops.transpose(keras.ops.log_sigmoid(-z1), (0, 2, 1))
    corner = keras.ops.zeros_like(dustbin_col[:, :1, :])
    top = keras.ops.concatenate([interior, dustbin_col], axis=2)
    bottom = keras.ops.concatenate([dustbin_row, corner], axis=2)
    return keras.ops.concatenate([top, bottom], axis=1)

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.matching.match_assignment")
class MatchAssignment(keras.layers.Layer):
    """Score every keypoint pair of two images, plus a dustbin for each keypoint.

    Both images share ``final_proj``, and the similarity ``sim`` is scaled by
    ``dim ** -0.5``, split as ``dim ** -0.25`` on each side. Scores are computed in at
    least float32 (``mask_dtype``): float32 under any half-precision policy, float64
    under a float64 policy. Padded keypoints (``mask`` 0) are removed from both
    softmaxes by :func:`apply_attention_mask`, so they take no mass from real ones.

    The reference has no mask; it slices the padding off first. Here, every output
    entry whose row is a padded image-0 keypoint or whose column is a padded image-1
    keypoint, dustbins included, is the finite constant ``0.0`` (log of probability
    one) with zero gradient. Consumers must use the masks. Entries of real rows and
    real columns, dustbins included, equal what the reference computes on the
    unpadded keypoints. A real row ``i`` sums, in probability space, to between
    ``sigmoid(-z0_i)`` and one, usually strictly below one; columns likewise.

    Assignment:

    .. code-block:: text

        desc0 [B, M, D]           desc1 [B, N, D]
               │                           │
               ▼                           ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ final_proj       │      │ final_proj       │
        │ (shared weights) │      │ (shared weights) │
        └──────────────────┘      └──────────────────┘
               │  md0 * D^-1/4             │  md1 * D^-1/4
               └─────────────┬─────────────┘
                             ▼
            sim = md0 md1^T [B, M, N] ──► returned as sim
                             │
                             ▼
            mask padded pairs (if masks)
                             │
                             ▼
            log_softmax over N + over M
                             │
                             ▼
            + logsigmoid(z0_i) + logsigmoid(z1_j)
                             │  interior [B, M, N]
                             ▼
            append dustbin row, column, corner
                             │  [B, M + 1, N + 1]
                             ▼
            zero padded rows and columns
                             │
                             ▼
            scores [B, M + 1, N + 1]

    Matchability logits feed both the interior and the dustbins:

    .. code-block:: text

        desc0 [B, M, D]      desc1 [B, N, D]
               │                    │
               ▼                    ▼
        ┌──────────────────────────────┐
        │ matchability  Dense(1)       │  shared
        └──────────────────────────────┘
               │                    │
               ▼                    ▼
        z0 [B, M, 1]         z1 [B, N, 1]

    Score matrix layout:

    .. code-block:: text

                      columns 0..N-1        column N
        rows 0..M-1   interior[i, j]        logsigmoid(-z0_i)
        row M         logsigmoid(-z1_j)     0

    Torch weight mapping (``Linear`` weights transpose into kernels, biases copy):

    .. code-block:: text

        torch (log_assignment.i.)     Keras
        matchability.{weight,bias}    matchability.{kernel,bias}
        final_proj.{weight,bias}      final_proj.{kernel,bias}

    :param dim: Descriptor width ``D``. Must be positive.
    :type dim: int
    :param **kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar final_proj: ``Dense(dim)`` applied to both images before the similarity.
    :ivar matchability: ``Dense(1)`` producing the per-keypoint matchability logit.

    Call arguments:
        desc0: ``(B, M, dim)``.
        desc1: ``(B, N, dim)``.
        mask0: Optional ``(B, M)`` keep mask, bool or 0/1, 1 = real keypoint.
        mask1: Optional ``(B, N)`` keep mask.

    Output:
        ``(scores, sim)``: ``scores`` ``(B, M + 1, N + 1)`` log scores, and ``sim``
        the scaled ``(B, M, N)`` similarity before masking, which the reference
        returns too.

    Example:

    .. code-block:: python

        layer = MatchAssignment(dim=64)
        scores, sim = layer(keras.random.normal((2, 30, 64)), keras.random.normal((2, 40, 64)))
        # scores: (2, 31, 41)
    """

    def __init__(self, dim: int, **kwargs: Any) -> None:
        """Validate ``dim`` and create the sublayers.

        Arguments are documented on the class.

        :raises ValueError: If ``dim`` is not positive.
        """
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
        """Build both sublayers explicitly. Returns at once if already built.

        :param desc0_shape: ``(batch, M, dim)``. An unknown last dimension is accepted.
        :param desc1_shape: ``(batch, N, dim)``. Checked only when given.
        :param mask0_shape: Accepted for the call signature, unused.
        :param mask1_shape: Accepted for the call signature, unused.
        :raises ValueError: If a given descriptor shape is not rank 3 with last dim
            ``dim``.
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
        """Return ``sigmoid(matchability(desc))`` in at least float32.

        No mask is applied.

        :param desc: ``(B, N, dim)`` descriptors of one image.
        :return: ``(B, N)`` matchability probabilities.
        """
        md = mask_dtype(self.compute_dtype)
        return keras.ops.sigmoid(keras.ops.cast(self.matchability(desc), md))[..., 0]

    def call(self, desc0: Any, desc1: Any, mask0: Optional[Any] = None,
             mask1: Optional[Any] = None) -> Tuple[Any, Any]:
        """Compute the ``(B, M + 1, N + 1)`` log assignment and the similarity.

        :param desc0: ``(B, M, dim)``.
        :param desc1: ``(B, N, dim)``.
        :param mask0: Optional ``(B, M)`` keep mask, 1 = real.
        :param mask1: Optional ``(B, N)`` keep mask, 1 = real. With one mask, the
            other image counts as all real.
        :return: ``(scores, sim)``, both in at least float32.
        """
        md = mask_dtype(self.compute_dtype)
        quarter = float(self.dim) ** -0.25
        md0 = keras.ops.cast(self.final_proj(desc0), md) * quarter
        md1 = keras.ops.cast(self.final_proj(desc1), md) * quarter
        sim = keras.ops.einsum("bmd,bnd->bmn", md0, md1)

        z0 = keras.ops.cast(self.matchability(desc0), md)
        z1 = keras.ops.cast(self.matchability(desc1), md)

        real0, real1 = _real_mask(mask0, md), _real_mask(mask1, md)
        logits = sim
        if real0 is not None or real1 is not None:
            ones0 = keras.ops.ones_like(z0[..., 0], dtype="bool")
            ones1 = keras.ops.ones_like(z1[..., 0], dtype="bool")
            r0 = ones0 if real0 is None else real0
            r1 = ones1 if real1 is None else real1
            keep = keras.ops.logical_and(r0[:, :, None], r1[:, None, :])
            # No rescue: a row with no real column stays flat and finite; its interior
            # is zeroed below either way.
            logits = apply_attention_mask(sim, keep, rescue_axis=None)

        interior = _double_log_softmax(logits) + _certainties(z0, z1)
        scores = _assemble_scores(interior, z0, z1)

        if real0 is not None or real1 is not None:
            # The appended True keeps the dustbin row and column of real keypoints.
            one = keras.ops.ones_like(z0[:, :1, 0], dtype="bool")
            full0 = keras.ops.concatenate([r0, one], axis=1)
            full1 = keras.ops.concatenate([r1, one], axis=1)
            keep_full = keras.ops.logical_and(full0[:, :, None], full1[:, None, :])
            scores = keras.ops.where(keep_full, scores, keras.ops.zeros_like(scores))
        return scores, sim

    def compute_output_shape(self, desc0_shape: Tuple[Any, ...],
                             desc1_shape: Tuple[Any, ...], mask0_shape: Optional[Any] = None,
                             mask1_shape: Optional[Any] = None) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Return ``((B, M + 1, N + 1), (B, M, N))``; ``desc1_shape`` is required."""
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
    """Extract hard mutual-nearest matches from a log assignment, as the reference does.

    ``log_scores`` is ``(B, M + 1, N + 1)`` as returned by :class:`MatchAssignment`;
    the dustbin row and column are ignored. Each row and column takes its argmax over
    the interior, and a pair is kept when both argmaxes agree (mutual) and
    ``exp(score) > threshold``. With masks, a padded keypoint is never matched and
    never chosen as a partner. Pass the masks given to :class:`MatchAssignment`:
    without them its padded entries, 0.0 or probability one, win the argmax. Pure
    function, no variables, graph-safe.

    :param log_scores: ``(B, M + 1, N + 1)`` log scores.
    :param threshold: Match threshold on ``exp(score)``; strictly greater passes.
    :param mask0: Optional ``(B, M)`` keep mask of image 0, bool or 0/1.
    :param mask1: Optional ``(B, N)`` keep mask of image 1, bool or 0/1. With one
        mask, the other image counts as all real.
    :return: ``(matches0 (B, M) int32, matches1 (B, N) int32, matching_scores0 (B, M),
        matching_scores1 (B, N))``. ``matches*`` hold the partner index or ``-1``.
        Scores, in the dtype of ``log_scores``, are ``exp(score)`` for every mutual
        pair, including those below the threshold, and 0 elsewhere, as in the
        reference.
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

    m0 = keras.ops.cast(keras.ops.argmax(core, axis=2), "int32")
    m1 = keras.ops.cast(keras.ops.argmax(core, axis=1), "int32")
    max0 = keras.ops.max(core, axis=2)
    max1 = keras.ops.max(core, axis=1)
    # back0[i] = m1[m0[i]] and back1[j] = m0[m1[j]].
    back0 = keras.ops.take_along_axis(m1, m0, axis=1)
    back1 = keras.ops.take_along_axis(m0, m1, axis=1)
    idx0 = keras.ops.arange(m_pts, dtype="int32")[None, :]
    idx1 = keras.ops.arange(n_pts, dtype="int32")[None, :]
    mutual0 = back0 == idx0
    mutual1 = back1 == idx1
    if r0 is not None:
        # The sentinel already keeps real rows on real partners, except when the
        # other image has no real keypoint; this check covers that case too.
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

# ---------------------------------------------------------------------