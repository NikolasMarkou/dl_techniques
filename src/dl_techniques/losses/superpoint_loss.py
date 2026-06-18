"""SuperPoint detector + descriptor loss functions.

This module implements the two training objectives from the SuperPoint paper
("SuperPoint: Self-Supervised Interest Point Detection and Description",
DeTone et al., CVPRW 2018):

1. :class:`SuperPointDetectorLoss` -- a softmax cross-entropy over the 65-class
   interest-point grid (an 8x8 spatial cell plus one "dustbin" / no-keypoint
   class), computed from raw logits (the detector head emits logits, per the
   repo convention "heads emit logits").

2. :class:`SuperPointDescriptorLoss` -- the bespoke hinge correspondence loss on
   descriptor similarity under a known homography. Positive (corresponding) cell
   pairs are pulled together up to a positive margin; negative (non-corresponding)
   pairs are pushed apart below a negative margin.

Both are serializable :class:`keras.losses.Loss` subclasses. The GLOBAL descriptor
loss weight (the famous ``lambda_d = 250`` balancing term from the paper) is applied
at COMPILE time by the trainer via ``loss_weights={"descriptors": 250.0}`` and is NOT
baked into the loss here; the per-positive-pair weight ``lambda_d`` (default 1.0)
IS a constructor argument.

All ops are ``keras.ops``-only and graph-safe (no raw ``tf.*``, no ``.numpy()``).
"""

import keras
from keras import ops
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# The detector grid has 8x8 = 64 spatial classes plus one dustbin class.
NUM_DETECTOR_CLASSES: int = 65


@keras.saving.register_keras_serializable()
class SuperPointDetectorLoss(keras.losses.Loss):
    """Interest-point detector loss: 65-class softmax cross-entropy from logits.

    **Intent**: Train the SuperPoint detector head to classify, for every H/8 x W/8
    cell, which of the 64 sub-pixel positions (within the 8x8 cell) contains an
    interest point, or the 65th "dustbin" class when the cell has no keypoint.

    **Formulation**:
    For each cell with integer label ``y in [0, 64]`` and logit vector
    ``z in R^65``, the loss is the standard sparse softmax cross-entropy

        L_cell = -log( softmax(z)[y] )

    reduced (mean) over all cells and the batch. ``from_logits=True`` because the
    detector head emits raw logits.

    Args:
        name: Loss instance name. Defaults to ``"superpoint_detector_loss"``.
        **kwargs: Forwarded to :class:`keras.losses.Loss` (e.g. ``reduction``).

    Input shapes:
        - ``y_true``: integer class labels per cell, shape ``(B, Hc, Wc)`` with
          values in ``[0, 64]`` (64 = dustbin / no-keypoint).
        - ``y_pred``: raw LOGITS, shape ``(B, Hc, Wc, 65)``.

    Output:
        Scalar loss (after the parent class reduction).

    Example:
        ```python
        loss_fn = SuperPointDetectorLoss()
        y_true = keras.random.randint((2, 8, 8), 0, 65)          # labels
        y_pred = keras.random.normal((2, 8, 8, 65))              # logits
        value = loss_fn(y_true, y_pred)
        ```
    """

    def __init__(
        self,
        name: str = "superpoint_detector_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        logger.info("SuperPointDetectorLoss initialized (65-class softmax CE from logits)")

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute per-cell sparse softmax cross-entropy from logits.

        Args:
            y_true: Integer labels ``(B, Hc, Wc)`` in ``[0, 64]``.
            y_pred: Raw logits ``(B, Hc, Wc, 65)``.

        Returns:
            Per-cell loss ``(B, Hc, Wc)``; the parent reduction yields the scalar.
        """
        # Sparse CE keeps memory low (no one-hot over 65 classes) and matches the
        # integer-label format. from_logits=True: the head emits raw logits.
        y_true = ops.cast(y_true, "int32")
        per_cell = keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )  # (B, Hc, Wc)
        return per_cell

    def get_config(self) -> Dict[str, Any]:
        """Return the base config (no extra hyperparameters)."""
        return super().get_config()


@keras.saving.register_keras_serializable()
class SuperPointDescriptorLoss(keras.losses.Loss):
    """Descriptor hinge correspondence loss (SuperPoint paper formulation).

    **Intent**: Make descriptors of CORRESPONDING cells (under a known homography
    relating an image and its warped copy) similar, while pushing descriptors of
    NON-corresponding cells apart, using a double hinge.

    **Formulation**:
    Given two coarse descriptor maps ``D`` (image) and ``D'`` (warped image) at
    H/8 resolution, flattened to ``N = Hc * Wc`` cells each with channel-vector
    ``d_i`` / ``d'_j``, and a correspondence indicator ``s_{ij} in {0, 1}``
    (1 iff cell ``i`` and warped cell ``j`` correspond under the homography):

        L = (1 / N^2) * sum_{i,j} [
              lambda_d * s_{ij} * max(0, m_pos - d_i . d'_j)
            + (1 - s_{ij}) * max(0, d_i . d'_j - m_neg)
            ]

    with positive margin ``m_pos`` (default 1.0), negative margin ``m_neg``
    (default 0.2), and per-positive-pair weight ``lambda_d`` (default 1.0).

    **Memory / cost**: the full ``N x N`` similarity matrix is formed
    (``(B, N, N)``). Cost is ``O(N^2)`` in both time and memory. At coarse
    resolution this is fine (e.g. an 8x8 grid -> ``N = 64`` -> a 64x64 matrix).
    At full feature resolution it would be prohibitive, so this loss is meant for
    the COARSE H/8 descriptor map only.

    **Keras Loss API note**: the true objective needs TWO descriptor maps plus a
    correspondence tensor, which does not fit the ``(y_true, y_pred)`` signature.
    The real computation is therefore exposed via :meth:`compute`. The standard
    ``call(y_true, y_pred)`` is a convenience adapter that treats ``y_pred`` as
    ``D``, ``y_true`` as ``D'``, and assumes an IDENTITY (diagonal) correspondence
    so the object remains usable as a plain Keras loss in smoke tests. For real
    training, call :meth:`compute` with the actual homography-derived
    correspondence.

    The GLOBAL balancing weight (250 in the paper) is applied at compile time by
    the trainer (``loss_weights=...``), NOT inside this loss.

    Args:
        positive_margin: ``m_pos``; target lower bound on positive-pair
            similarity. Defaults to 1.0.
        negative_margin: ``m_neg``; target upper bound on negative-pair
            similarity. Defaults to 0.2.
        lambda_d: Per-positive-pair weight inside the loss. Defaults to 1.0.
        name: Loss instance name. Defaults to ``"superpoint_descriptor_loss"``.
        **kwargs: Forwarded to :class:`keras.losses.Loss`.

    Input shapes (``compute``):
        - ``desc1``: ``(B, Hc, Wc, C)`` -- descriptor map of the image.
        - ``desc2``: ``(B, Hc, Wc, C)`` -- descriptor map of the warped image.
        - ``correspondence``: ``(B, N, N)`` with ``N = Hc * Wc``, values in
          ``{0, 1}`` (``s_{ij}``).

    Example:
        ```python
        loss_fn = SuperPointDescriptorLoss()
        d1 = keras.random.normal((2, 8, 8, 16))
        d2 = keras.random.normal((2, 8, 8, 16))
        corr = keras.ops.eye(64, batch_shape=(2,))   # diagonal correspondence
        value = loss_fn.compute(d1, d2, corr)
        ```
    """

    def __init__(
        self,
        positive_margin: float = 1.0,
        negative_margin: float = 0.2,
        lambda_d: float = 1.0,
        name: str = "superpoint_descriptor_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if positive_margin <= negative_margin:
            logger.warning(
                f"positive_margin ({positive_margin}) <= negative_margin "
                f"({negative_margin}); the hinge margins overlap, which is unusual."
            )
        if lambda_d < 0.0:
            raise ValueError(f"lambda_d must be non-negative, got {lambda_d}")

        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.lambda_d = lambda_d

        logger.info(
            f"SuperPointDescriptorLoss initialized: positive_margin={positive_margin}, "
            f"negative_margin={negative_margin}, lambda_d={lambda_d}"
        )

    def compute(
        self,
        desc1: keras.KerasTensor,
        desc2: keras.KerasTensor,
        correspondence: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the full double-sum hinge correspondence loss.

        Args:
            desc1: Image descriptor map ``(B, Hc, Wc, C)``.
            desc2: Warped-image descriptor map ``(B, Hc, Wc, C)``.
            correspondence: ``(B, N, N)`` indicator ``s_{ij}`` with ``N=Hc*Wc``.

        Returns:
            Scalar loss (mean over the batch of the per-pair-averaged double sum).
        """
        # Flatten the spatial grid to N = Hc * Wc cells, keeping channels.
        b = ops.shape(desc1)[0]
        c = ops.shape(desc1)[-1]
        d1 = ops.reshape(desc1, (b, -1, c))  # (B, N, C)
        d2 = ops.reshape(desc2, (b, -1, c))  # (B, N, C)

        # Pairwise similarity d_i . d'_j  ->  (B, N, N). O(N^2) memory.
        sim = ops.einsum("bnc,bmc->bnm", d1, d2)

        corr = ops.cast(correspondence, sim.dtype)

        # Positive term: pull corresponding pairs up to the positive margin.
        positive = corr * ops.maximum(0.0, self.positive_margin - sim)
        # Negative term: push non-corresponding pairs below the negative margin.
        negative = (1.0 - corr) * ops.maximum(0.0, sim - self.negative_margin)

        per_pair = self.lambda_d * positive + negative  # (B, N, N)

        # Average over the N^2 pairs (the 1 / N^2 factor), then mean over batch.
        per_sample = ops.mean(per_pair, axis=[1, 2])  # (B,)
        return ops.mean(per_sample)

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Convenience Keras-loss adapter with an IDENTITY correspondence.

        Treats ``y_pred`` as ``desc1`` and ``y_true`` as ``desc2`` and assumes a
        diagonal correspondence (cell ``i`` corresponds to cell ``i``). This makes
        the loss usable as a standard Keras loss in smoke tests; for real training
        use :meth:`compute` with the homography-derived correspondence.

        Args:
            y_true: Treated as ``desc2`` ``(B, Hc, Wc, C)``.
            y_pred: Treated as ``desc1`` ``(B, Hc, Wc, C)``.

        Returns:
            Scalar loss from :meth:`compute` with an identity correspondence.
        """
        desc1 = y_pred
        desc2 = y_true

        b = ops.shape(desc1)[0]
        hc = ops.shape(desc1)[1]
        wc = ops.shape(desc1)[2]
        n = hc * wc

        identity = ops.eye(n, dtype=desc1.dtype)  # (N, N)
        correspondence = ops.broadcast_to(
            ops.expand_dims(identity, 0), (b, n, n)
        )  # (B, N, N)

        return self.compute(desc1, desc2, correspondence)

    def get_config(self) -> Dict[str, Any]:
        """Serialize margins and lambda_d alongside the base config."""
        config = super().get_config()
        config.update(
            {
                "positive_margin": self.positive_margin,
                "negative_margin": self.negative_margin,
                "lambda_d": self.lambda_d,
            }
        )
        return config
