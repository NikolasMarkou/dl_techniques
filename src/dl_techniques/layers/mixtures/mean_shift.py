"""Differentiable mean shift on a point set, the :class:`MeanShiftLayer`, and a hard-assignment readout.

Mean shift moves every point to the kernel-weighted mean of the point set, so
repeated steps carry the points toward the modes of their kernel density estimate.
:class:`MeanShiftLayer` runs a fixed number of damped steps with the shared
operations of ``mean_shift_ops``; every step is a softmax, so the whole map is
differentiable and a loss on the shifted points reaches the inputs, the optional
projection and the bandwidth. The cluster labels are not part of the graph:
:func:`assign_clusters` reads them off the converged points in NumPy, once, at
evaluation time. Every logit and softmax runs in at least float32, also under
``mixed_float16``.

References:
    - Cheng, 1995. Mean shift, mode seeking, and clustering.
    - Comaniciu and Meer, 2002. Mean shift: a robust approach toward feature
      space analysis.
    - Carreira-Perpinan, 2015. A review of mean-shift algorithms for clustering.
      (https://arxiv.org/abs/1503.00687)
"""

import math
import numbers
import keras
import numpy as np
from typing import Any, ClassVar, Dict, FrozenSet, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.dtype_policy import accumulation_dtype
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.validation import positive_float, positive_int

from .mean_shift_ops import _check_kernel, mean_shift_iterate

# ---------------------------------------------------------------------

__all__ = ["MeanShiftLayer", "assign_clusters"]

_MIN_BANDWIDTH = 1e-2

_MAX_BANDWIDTH = 1e2

_BANDWIDTH_MARGIN = 1e-3

_LOWEST_BANDWIDTH = _MIN_BANDWIDTH * (1.0 + _BANDWIDTH_MARGIN)

_HIGHEST_BANDWIDTH = _MAX_BANDWIDTH * (1.0 - _BANDWIDTH_MARGIN)

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.mixtures.mean_shift")
class MeanShiftLayer(keras.layers.Layer):
    """
    Shift every point of a set toward its mode with ``num_steps`` mean-shift steps.

    Each step moves a point ``z`` to the kernel-weighted mean of the keys,
    damped by the step size ``eta``: ``z <- (1 - eta) z + eta * softmax(logits) @ keys``.
    Blurring keeps ``keys = z`` (the set contracts as a whole); non-blurring keeps
    ``keys`` at the (projected) input points, so each point climbs the fixed density
    of the inputs. The bandwidth ``h`` is one scalar, stored as ``log h`` when it is
    learnable and kept inside ``[1e-2, 1e2]``. With ``use_projection`` the points
    are first mapped by a bias-free ``Dense(projection_dim)``, so the density is
    estimated in a learned space. The layer holds no cluster count: read labels with
    :func:`assign_clusters`.

    Architecture:

    .. code-block:: text

        points [B, N, D]                  mask [B, N] (1 = keep)
                │
                ▼
        ┌──────────────────────────────┐
        │ projection: Dense(P), no bias│  only when use_projection
        └──────────────┬───────────────┘
                       ▼
        x [B, N, D']   (vMF: L2-normalise)       D' = P or D
                       │
                       ▼   z = x
        ┌─ repeat num_steps times ──────────────────────────────┐
        │  keys = z (blurring)  or  x (projection / inputs)      │
        │          │                                             │
        │          ▼                                             │
        │  logits = kernel_logits(z, keys, h)      [B, N, N]     │
        │          │       h = clip(exp(log_h), 1e-2, 1e2)       │
        │          ▼                                             │
        │  masked float32 softmax -> weights                     │
        │          │                                             │
        │          ▼                                             │
        │  z = (1 - eta) z + eta * weights @ keys                │
        │  vMF: z = normalise(z)                                 │
        └──────────────────────┬─────────────────────────────────┘
                               ▼
        shifted [B, N, D']   (+ last-step weights [B, N, N])

    The mask is a keep mask (1 or True keeps a point as a key): a masked point has
    exactly zero weight at every step, in every row. The shifted value of a masked
    point is still computed and finite; callers ignore it, and pass the same mask to
    :func:`assign_clusters` so such points get the label -1. A batch row with no kept
    point uses uniform weights, so its output is the plain mean of that row.
    ``num_steps`` is at least 1, as in ``MeanShiftAttention``; use the operations
    in ``mean_shift_ops`` for a zero-step identity. The layer is not a
    ``BaseMixtureLayer``: it has no cluster axis and no prototypes.

    :param num_steps: Number of mean-shift steps, a positive integer.
    :type num_steps: int
    :param bandwidth: Initial (or fixed) bandwidth ``h``, a positive number in
        ``[0.01001, 99.9]``: the clip range ``[1e-2, 1e2]`` less a relative margin of
        1e-3. At an end of the clip range the clip is active from the start and the
        learnable bandwidth gets exactly zero gradient, so the ends are refused.
        The bandwidth is the scale at which instances separate: points closer than
        about ``h`` merge. A blurring output collapses when ``h`` is large against
        the spacing of the data (every point moves to the global mean, so the
        output has no structure left and the loss signal vanishes); start near the
        spacing of the instances, not above it.
    :type bandwidth: float
    :param learnable_bandwidth: Train ``log h`` (one scalar weight), initialised at
        ``bandwidth``. When False the bandwidth is the constant ``bandwidth``.
    :type learnable_bandwidth: bool
    :param blurring: Move the keys together with the points.
    :type blurring: bool
    :param step_size: Damping ``eta`` in the open interval (0, 2).
    :type step_size: float
    :param kernel: ``"gaussian"`` or ``"vmf"`` (unit-sphere points).
    :type kernel: str
    :param use_projection: Map the points with a bias-free ``Dense(projection_dim)``
        before shifting. Requires ``projection_dim``.
    :type use_projection: bool
    :param projection_dim: Width ``P`` of the projection. Ignored (but stored)
        when ``use_projection`` is False.
    :type projection_dim: Optional[int]
    :param return_affinity: Also return the kernel weights of the last step,
        ``(B, N, N)`` with rows summing to 1.
    :type return_affinity: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    Input shape:
        ``(batch, N, D)`` with ``D`` known when the layer is built; call with an
        optional ``mask`` of shape ``(batch, N)``.

    Output shape:
        ``(batch, N, D')`` with ``D' = projection_dim`` when ``use_projection`` else
        ``D``; with ``return_affinity`` a tuple ``(shifted, affinity)`` where
        ``affinity`` is ``(batch, N, N)``.

    :raises ValueError: for an invalid argument.
    """

    # DECISION plan-2026-10-29T104646-27f045c5/D-008: a plain Layer with an empty legal set, not a BaseMixtureLayer.
    # Do not invent an output_mode or a prototype count to fit the base class. See decisions.md.
    VALID_OUTPUT_MODES: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(
            self,
            num_steps: int = 5,
            bandwidth: float = 1.0,
            learnable_bandwidth: bool = True,
            blurring: bool = True,
            step_size: float = 1.0,
            kernel: str = "gaussian",
            use_projection: bool = False,
            projection_dim: Optional[int] = None,
            return_affinity: bool = False,
            **kwargs: Any
    ) -> None:
        """Initialise the layer and create its optional projection."""
        super().__init__(**kwargs)

        owner = "MeanShiftLayer"
        self.num_steps = positive_int(num_steps, "num_steps", owner)
        self.bandwidth = positive_float(bandwidth, "bandwidth", owner)
        if not _LOWEST_BANDWIDTH <= self.bandwidth <= _HIGHEST_BANDWIDTH:
            raise ValueError(
                f"{owner}: bandwidth must lie in "
                f"[{_LOWEST_BANDWIDTH:.6g}, {_HIGHEST_BANDWIDTH:.6g}] (the clip range "
                f"[{_MIN_BANDWIDTH}, {_MAX_BANDWIDTH}] less a relative margin of "
                f"{_BANDWIDTH_MARGIN}), got {bandwidth!r}"
            )

        for name, value in (
                ("learnable_bandwidth", learnable_bandwidth),
                ("blurring", blurring),
                ("use_projection", use_projection),
                ("return_affinity", return_affinity),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{owner}: {name} must be a bool, got {value!r}")
        self.learnable_bandwidth = learnable_bandwidth
        self.blurring = blurring
        self.use_projection = use_projection
        self.return_affinity = return_affinity

        if (
                isinstance(step_size, bool)
                or not isinstance(step_size, numbers.Real)
                or not 0.0 < step_size < 2.0
        ):
            raise ValueError(
                f"{owner}: step_size must be a number in (0, 2), got {step_size!r}"
            )
        self.step_size = float(step_size)

        _check_kernel(kernel)
        self.kernel = kernel

        if use_projection:
            self.projection_dim = positive_int(projection_dim, "projection_dim", owner)
        else:
            self.projection_dim = projection_dim

        self.projection = None
        if self.use_projection:
            self.projection = keras.layers.Dense(
                self.projection_dim,
                use_bias=False,
                dtype=accumulation_dtype(self.compute_dtype),
                name="projection",
            )
        self.log_bandwidth = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the projection weights and the log-bandwidth weight.

        :param input_shape: shape ``(B, N, D)`` with ``D`` known.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: if the shape is not rank 3 or ``D`` is unknown.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"MeanShiftLayer expects inputs of shape (batch, N, D), got rank "
                f"{len(input_shape)}: {tuple(input_shape)}"
            )
        if input_shape[-1] is None:
            raise ValueError("The last dimension of the input must be defined.")

        if self.projection is not None:
            self.projection.build((None, None, input_shape[-1]))
        if self.learnable_bandwidth:
            self.log_bandwidth = self.add_weight(
                name="log_bandwidth",
                shape=(),
                initializer=keras.initializers.Constant(math.log(self.bandwidth)),
                trainable=True,
                dtype=self.variable_dtype,
                autocast=False,
            )
        super().build(input_shape)

    def _current_bandwidth(self, acc: str) -> Union[float, keras.KerasTensor]:
        """Return ``h`` clipped to ``[1e-2, 1e2]``: a tensor when learnable, else a float.

        :param acc: accumulation dtype name the kernel math runs in.
        :type acc: str
        :return: the bandwidth, a scalar.
        :rtype: Union[float, keras.KerasTensor]
        """
        if not self.learnable_bandwidth:
            return self.bandwidth
        h = keras.ops.exp(keras.ops.cast(self.log_bandwidth, acc))
        return keras.ops.clip(h, _MIN_BANDWIDTH, _MAX_BANDWIDTH)

    def call(
            self,
            points: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Shift the points toward their modes.

        :param points: tensor ``(B, N, D)``.
        :type points: keras.KerasTensor
        :param mask: ``None`` or a keep mask ``(B, N)``; masked points never act as
            keys. Their own shifted values are computed and should be ignored.
        :type mask: Optional[keras.KerasTensor]
        :return: ``shifted`` ``(B, N, D')`` in the compute dtype, or
            ``(shifted, affinity)`` with ``affinity`` ``(B, N, N)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        acc = accumulation_dtype(self.compute_dtype)
        if self.projection is not None:
            x = self.projection(points)
        else:
            x = points
        x = keras.ops.cast(x, acc)

        result = mean_shift_iterate(
            x,
            bandwidth=self._current_bandwidth(acc),
            num_steps=self.num_steps,
            step_size=self.step_size,
            blurring=self.blurring,
            mask=mask,
            kernel=self.kernel,
            return_weights=self.return_affinity,
        )
        if self.return_affinity:
            shifted, affinity = result
            return (
                keras.ops.cast(shifted, self.compute_dtype),
                keras.ops.cast(affinity, self.compute_dtype),
            )
        return keras.ops.cast(result, self.compute_dtype)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]:
        """Return the output shape without building the layer.

        :param input_shape: shape ``(B, N, D)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(B, N, D')``, or ``((B, N, D'), (B, N, N))`` with ``return_affinity``.
        :rtype: Union[tuple, Tuple[tuple, tuple]]
        """
        batch, count = input_shape[0], input_shape[1]
        width = self.projection_dim if self.use_projection else input_shape[-1]
        shifted = (batch, count, width)
        if self.return_affinity:
            return shifted, (batch, count, count)
        return shifted

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_steps": self.num_steps,
            "bandwidth": self.bandwidth,
            "learnable_bandwidth": self.learnable_bandwidth,
            "blurring": self.blurring,
            "step_size": self.step_size,
            "kernel": self.kernel,
            "use_projection": self.use_projection,
            "projection_dim": self.projection_dim,
            "return_affinity": self.return_affinity,
        })
        return config


def assign_clusters(
        shifted: Any, merge_radius: float, mask: Optional[Any] = None
) -> np.ndarray:
    """Label converged points by greedy radius merging (eval-only, NumPy).

    Points are visited in index order. A point takes the label of the first existing
    cluster center within ``merge_radius`` (Euclidean distance, ``<=``); if none is
    that close it opens a new cluster and becomes that cluster's center, so the
    centers are converged points, not means. The labels depend on the order of the
    points, and the result is not a partition into the true modes when clusters
    overlap. A point whose ``mask`` entry is false gets the label ``-1`` and neither
    opens nor joins a cluster, so the unmasked points are labelled exactly as if the
    masked ones were not there.

    This is a readout for evaluation and inspection. It is not differentiable, it
    runs outside any graph, and it must not be called inside a training step or a
    ``tf.function``.

    :param shifted: converged points ``(N, D)`` or ``(B, N, D)``, such as the output
        of :class:`MeanShiftLayer`.
    :type shifted: Any
    :param merge_radius: points closer than this to a cluster center join it,
        positive.
    :type merge_radius: float
    :param mask: ``None`` or a keep mask ``(N,)`` or ``(B, N)`` matching the leading
        axes of ``shifted`` (bool or 0/1, true keeps), such as the mask given to
        :class:`MeanShiftLayer`.
    :type mask: Optional[Any]
    :return: integer labels ``(N,)`` or ``(B, N)``, numbered from 0 per sample in
        order of first appearance among the kept points; ``-1`` for a masked point.
    :rtype: numpy.ndarray
    :raises ValueError: if ``merge_radius`` is not positive and finite, the points
        are not rank 2 or 3, or ``mask`` does not have the leading shape of the
        points.
    """
    radius = positive_float(merge_radius, "merge_radius", "assign_clusters")
    points = np.asarray(shifted, dtype=np.float64)
    if points.ndim not in (2, 3):
        raise ValueError(
            f"assign_clusters: shifted must have shape (N, D) or (B, N, D), got "
            f"rank {points.ndim}: {points.shape}"
        )
    if mask is None:
        kept = np.ones(points.shape[:-1], dtype=bool)
    else:
        kept = np.asarray(mask) != 0
        if kept.shape != points.shape[:-1]:
            raise ValueError(
                f"assign_clusters: mask must have shape {points.shape[:-1]} (the "
                f"leading axes of shifted), got {kept.shape}"
            )
    batched = points if points.ndim == 3 else points[None]
    batched_kept = kept if points.ndim == 3 else kept[None]

    labels = np.full(batched.shape[:2], -1, dtype=np.int64)
    for row, sample in enumerate(batched):
        centers = []
        for index, point in enumerate(sample):
            if not batched_kept[row, index]:
                continue
            if centers:
                gaps = np.linalg.norm(np.stack(centers) - point, axis=-1)
                near = np.flatnonzero(gaps <= radius)
            else:
                near = np.empty(0, dtype=np.int64)
            if near.size:
                labels[row, index] = near[0]
            else:
                centers.append(point)
                labels[row, index] = len(centers) - 1
    return labels if points.ndim == 3 else labels[0]
