"""Person re-identification metric-learning losses (DeepSORT transcription).

Transcribes the three training modes of
https://github.com/nwojke/cosine_metric_learning (``losses.py``):

1. **cosine-softmax** (the default the DeepSORT descriptor trains with): the
   :class:`CosineClassifier` head already emits scaled cosine logits, so the
   objective is plain softmax cross-entropy over identities -- no wrapper
   class here; the trainer compiles stock sparse CE ``from_logits=True``.
2. :class:`SoftmarginTripletLoss` -- batch-hard softmargin triplet loss
   (Hermans et al., 2017): per anchor the hardest positive (max same-label
   distance, self included at distance 0) and hardest negative (min
   different-label distance) feed ``softplus(pos - neg)``.
3. :func:`magnet_loss_fn` -- unimodal magnet loss (Rippel et al., 2016)
   over batch class means with the shared-variance normalization.

Masking/shape notes: the triplet loss reduces to per-sample ``(B,)``
itself (PK batches guarantee every anchor has a positive). The magnet loss
is irreducibly BATCH-LEVEL (class means and the shared variance couple
every row), so it is a plain function, not a ``keras.losses.Loss``: wiring
it into ``compile`` would charge every row the batch aggregate, the same
trap ``goodhart_loss.py`` documents for its prior term.

All Keras paths are ``keras.ops``-only and graph-safe. The magnet function
is pure NumPy: it runs in custom loops and analysis, not in the graph.

References:
    - Wojke and Bewley, 2018. Deep Cosine Metric Learning for Person
      Re-identification. WACV 2018.
    - Hermans et al., 2017. In Defense of the Triplet Loss for Person
      Re-Identification. (https://arxiv.org/abs/1703.07737)
    - Rippel et al., 2016. Metric Learning With Adaptive Density
      Discrimination. ICLR 2016. (https://arxiv.org/abs/1511.06335)
"""

import keras
import numpy as np
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


def _pairwise_squared_distance(
    features: keras.KerasTensor, eps: float = 1e-5
) -> keras.KerasTensor:
    """Squared Euclidean distance matrix with the transcribed floor.

    Uses ``max(0, eps + sq)`` under the root (ref ``_pdist`` + eps), so
    rounding never feeds a negative to ``sqrt``.

    Args:
        features: Matrix ``(N, M)``.
        eps: Floor constant (ref: 1e-5).

    Returns:
        Distance matrix ``(N, N)``.
    """
    sq = keras.ops.sum(keras.ops.square(features), axis=1)
    squared = (
        -2.0 * keras.ops.matmul(features, keras.ops.transpose(features))
        + keras.ops.reshape(sq, (-1, 1))
        + keras.ops.reshape(sq, (1, -1))
    )
    return keras.ops.sqrt(keras.ops.maximum(0.0, eps + squared))


@register_dl_technique("dl_techniques.losses.reid_cosine_loss")
class SoftmarginTripletLoss(keras.losses.Loss):
    """Batch-hard softmargin triplet loss from L2-normalized features.

    **Intent**: Pull each anchor toward its hardest same-identity sample and
    push it from its hardest different-identity sample, with a soft margin:
    ``softplus(max_pos - min_neg)`` per anchor. Requires PK batches (every
    identity appears at least twice; the anchor itself counts as a
    distance-0 positive).

    Args:
        name: Loss instance name. Defaults to ``"softmargin_triplet_loss"``.
        **kwargs: Forwarded to :class:`keras.losses.Loss` (e.g. ``reduction``).

    Input shapes:
        - ``y_true``: integer identity labels ``(B,)``.
        - ``y_pred``: feature matrix ``(B, M)`` (L2-normalized upstream).

    Output:
        Per-sample loss ``(B,)``.
    """

    def __init__(
        self,
        name: str = "softmargin_triplet_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        logger.info("SoftmarginTripletLoss initialized (batch-hard, softplus margin)")

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the batch-hard softmargin triplet loss.

        Args:
            y_true: Identity labels ``(B,)``.
            y_pred: Features ``(B, M)``.

        Returns:
            Per-sample loss ``(B,)``.
        """
        labels = keras.ops.cast(keras.ops.reshape(y_true, (-1,)), "int32")
        distance_mat = _pairwise_squared_distance(
            keras.ops.cast(y_pred, "float32")
        )
        same = keras.ops.cast(
            keras.ops.equal(
                keras.ops.reshape(labels, (-1, 1)),
                keras.ops.reshape(labels, (1, -1)),
            ),
            "float32",
        )
        positive_distance = keras.ops.max(same * distance_mat, axis=1)
        almost_inf = keras.ops.full_like(distance_mat, 1e10)
        negative_distance = keras.ops.min(same * almost_inf + distance_mat, axis=1)
        return keras.ops.softplus(positive_distance - negative_distance)

    def get_config(self) -> Dict[str, Any]:
        """Return the base config (no extra hyperparameters)."""
        return super().get_config()


def magnet_loss_fn(
    features: np.ndarray, labels: np.ndarray, margin: float = 1.0
) -> Tuple[float, np.ndarray, float]:
    """Unimodal magnet loss over batch class means (transcribed).

    Computes per-class means from the batch, the shared variance around
    them, and the log-sum-exp density-discrimination objective. Returns the
    scalar loss plus the means and variance for monitoring, mirroring the
    reference return contract.

    Args:
        features: Matrix ``(N, M)`` float.
        labels: Identity per row ``(N,)`` int.
        margin: Margin hyperparameter (ref default 1.0).

    Returns:
        ``(loss, class_means (C, M), variance)``.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    y_mat = (labels[:, None] == unique[None, :]).astype(np.float64)
    counts = y_mat.sum(axis=0)
    class_means = (y_mat.T @ features) / np.maximum(counts[:, None], 1e-9)
    diff = features[:, None, :] - class_means[None, :, :]
    squared = np.sum(diff**2, axis=-1)
    variance = float(np.sum(y_mat * squared) / max(len(labels) - 1, 1))
    const = 1.0 / (-2.0 * (variance + 1e-4))
    linear = const * squared - y_mat * margin
    maxi = linear.max(axis=1, keepdims=True)
    exp_shifted = np.exp(linear - maxi)
    a = np.sum(y_mat * exp_shifted, axis=1)
    b = np.sum((1.0 - y_mat) * exp_shifted, axis=1)
    loss = np.maximum(0.0, -np.log(1e-4 + a / (1e-4 + b)))
    return float(np.mean(loss)), class_means.astype(np.float32), variance
