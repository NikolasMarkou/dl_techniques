"""Instance-embedding loss for mean-shift clustering.

Trains an embedding so that mean shift on it recovers instances: points of one
instance are pulled together and points of different instances are pushed apart
beyond a margin. The loss reads a point set per batch row, ``y_pred`` of shape
``(B, N, D)`` (typically the output of ``MeanShiftLayer``) with one instance id
per point in ``y_true`` of shape ``(B, N)``. Ids equal to ``ignore_label`` mark
points that take no part, such as padding or background. The pull term is the
mean squared distance over same-instance pairs, the push term is the mean
squared hinge ``max(0, margin - distance)`` over different-instance pairs, and
each mean divides by its pair count floored at one, so a row with a single
instance or no valid point costs zero rather than NaN. Every pair lives inside
one batch row, so the loss returns one value per row and ``sample_weight``
selects rows exactly. Cost is O(N^2) per row; rows with more than
``max_points`` points are subsampled deterministically.

References:
    - Cheng, 1995. Mean shift, mode seeking, and clustering. IEEE TPAMI 17(8).
    - Comaniciu and Meer, 2002. Mean shift: a robust approach toward feature
      space analysis. IEEE TPAMI 24(5).
    - De Brabandere, Neven and Van Gool, 2017. Semantic instance segmentation
      with a discriminative loss function. (https://arxiv.org/abs/1708.02551)
    - Kong and Fowlkes, 2018. Recurrent pixel embedding for instance grouping.
      CVPR 2018. (https://arxiv.org/abs/1712.08273)
"""

import math
import numbers
from typing import Any, Dict, Optional

import keras
import numpy as np

from dl_techniques.layers.mixtures.mean_shift_ops import pairwise_sq_dist
from dl_techniques.utils.dtype_policy import accumulation_dtype, stability_floor
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.validation import positive_float, positive_int

_SQRT_FLOOR = 1e-12


def _non_negative_float(value: Any, name: str) -> float:
    """Return ``value`` as a Python float after checking it is finite and >= 0.

    :param value: the caller's value.
    :type value: Any
    :param name: argument name for the error message.
    :type name: str
    :return: ``value`` as a plain ``float``.
    :rtype: float
    :raises ValueError: if ``value`` is a bool, not a real number, or not finite
        and non-negative.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"MeanShiftInstanceLoss: {name} must be a non-negative number, got {value!r}"
        )
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"MeanShiftInstanceLoss: {name} must be a non-negative finite number, "
            f"got {value}"
        )
    return value


@register_dl_technique("dl_techniques.losses.mean_shift_instance_loss")
class MeanShiftInstanceLoss(keras.losses.Loss):
    """Pull same-instance embeddings together and push different instances apart.

    For every batch row, over the valid points (id different from
    ``ignore_label``) and ordered pairs ``(i, j)`` with ``i != j``::

        pull = sum over same-id pairs  ||z_i - z_j||^2            / max(n_same, 1)
        push = sum over diff-id pairs  max(0, margin - ||z_i - z_j||)^2 / max(n_diff, 1)
        loss = pull_weight * pull + push_weight * push

    The loss is zero when every instance is a single point in space and distinct
    instances are at least ``margin`` apart. There is no cross-row term, so
    ``call`` returns shape ``(B,)`` and ``sample_weight`` zeroes whole rows.
    Memory is O(N^2) per row; when the static point count exceeds ``max_points``
    the rows are reduced to ``max_points`` evenly spaced points (the same indices
    on every call). With an unknown point count at trace time no subsampling is
    done, so keep ``N`` static for large point sets.

    .. code-block:: text

        y_true [B, N] ids        y_pred [B, N, D] embeddings
              |                          |
              v                          v
        valid = id != ignore      subsample to max_points (static N only)
              |                          |
              +------------+-------------+
                           v
              sq = pairwise_sq_dist(z, z)        [B, n, n]
                           |
              +------------+-------------+
              v                          v
        same-id valid pairs        diff-id valid pairs
        pull = mean sq             push = mean max(0, m - sqrt(sq))^2
              |                          |
              +------------+-------------+
                           v
              loss [B] = w_pull * pull + w_push * push

    :param bandwidth: Mean-shift bandwidth the embedding is trained for, a positive
        number. It only sets the default margin.
    :type bandwidth: float
    :param margin: Distance beyond which two different instances no longer repel,
        a positive number. ``None`` means ``2 * bandwidth``.
    :type margin: Optional[float]
    :param pull_weight: Weight of the same-instance term, non-negative.
    :type pull_weight: float
    :param push_weight: Weight of the different-instance term, non-negative.
    :type push_weight: float
    :param max_points: Largest number of points per row that is paired, a positive
        integer.
    :type max_points: int
    :param ignore_label: Instance id that marks a point as ignored.
    :type ignore_label: int
    :param name: Name of the loss instance.
    :type name: str
    :param kwargs: Forwarded to :class:`keras.losses.Loss` (for example ``reduction``).
    :type kwargs: Any

    :raises ValueError: for an invalid argument.
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        margin: Optional[float] = None,
        pull_weight: float = 1.0,
        push_weight: float = 1.0,
        max_points: int = 1024,
        ignore_label: int = -1,
        name: str = "mean_shift_instance_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        owner = "MeanShiftInstanceLoss"
        self.bandwidth = positive_float(bandwidth, "bandwidth", owner)
        self.margin = margin
        # DECISION plan-2026-10-29T104646-27f045c5/D-010: the margin defaults to 2 * bandwidth.
        # Do not hard-code a margin that ignores the bandwidth. See decisions.md.
        self.resolved_margin = (
            2.0 * self.bandwidth
            if margin is None
            else positive_float(margin, "margin", owner)
        )
        self.pull_weight = _non_negative_float(pull_weight, "pull_weight")
        self.push_weight = _non_negative_float(push_weight, "push_weight")
        self.max_points = positive_int(max_points, "max_points", owner)
        if isinstance(ignore_label, (bool, np.bool_)) or not isinstance(
            ignore_label, numbers.Integral
        ):
            raise ValueError(
                f"{owner}: ignore_label must be an integer, got {ignore_label!r}"
            )
        self.ignore_label = int(ignore_label)

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the per-row pull and push loss.

        :param y_true: Instance ids ``(B, N)``; a float dtype (``fit`` may pass one)
            is cast to ``int32``.
        :type y_true: keras.KerasTensor
        :param y_pred: Embeddings ``(B, N, D)``.
        :type y_pred: keras.KerasTensor
        :return: Per-row loss ``(B,)`` in at least float32.
        :rtype: keras.KerasTensor
        """
        z = keras.ops.convert_to_tensor(y_pred)
        acc = accumulation_dtype(getattr(z.dtype, "name", None) or str(z.dtype))
        z = keras.ops.cast(z, acc)
        ids = keras.ops.cast(y_true, "int32")

        n_points = z.shape[1]
        if n_points is not None and n_points > self.max_points:
            # DECISION plan-2026-10-29T104646-27f045c5/D-010: evenly spaced points, only for a static N.
            # Do not draw random indices (the value must repeat) or branch on a dynamic length.
            picked = (np.arange(self.max_points) * n_points) // self.max_points
            z = keras.ops.take(z, picked, axis=1)
            ids = keras.ops.take(ids, picked, axis=1)

        valid = keras.ops.not_equal(ids, self.ignore_label)
        pair = keras.ops.logical_and(
            keras.ops.expand_dims(valid, 2), keras.ops.expand_dims(valid, 1)
        )
        not_self = keras.ops.logical_not(
            keras.ops.eye(keras.ops.shape(ids)[1], dtype="bool")
        )
        pair = keras.ops.logical_and(pair, not_self)
        same_id = keras.ops.equal(
            keras.ops.expand_dims(ids, 2), keras.ops.expand_dims(ids, 1)
        )
        same_pair = keras.ops.logical_and(pair, same_id)
        diff_pair = keras.ops.logical_and(pair, keras.ops.logical_not(same_id))

        sq = pairwise_sq_dist(z, z)
        dist = keras.ops.sqrt(
            keras.ops.maximum(sq, stability_floor(acc, _SQRT_FLOOR))
        )
        hinge = keras.ops.square(
            keras.ops.maximum(self.resolved_margin - dist, 0.0)
        )
        zero = keras.ops.zeros_like(sq)

        n_same = keras.ops.sum(keras.ops.cast(same_pair, acc), axis=(1, 2))
        n_diff = keras.ops.sum(keras.ops.cast(diff_pair, acc), axis=(1, 2))
        pull = keras.ops.sum(keras.ops.where(same_pair, sq, zero), axis=(1, 2))
        push = keras.ops.sum(keras.ops.where(diff_pair, hinge, zero), axis=(1, 2))
        pull = pull / keras.ops.maximum(n_same, 1.0)
        push = push / keras.ops.maximum(n_diff, 1.0)
        return self.pull_weight * pull + self.push_weight * push

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments (``margin`` as given, ``None`` stays ``None``).

        :return: configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "bandwidth": self.bandwidth,
                "margin": self.margin,
                "pull_weight": self.pull_weight,
                "push_weight": self.push_weight,
                "max_points": self.max_points,
                "ignore_label": self.ignore_label,
            }
        )
        return config
