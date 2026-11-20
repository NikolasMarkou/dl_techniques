r"""Geodesic normalization (GeoNorm), in :class:`GeoNorm`: the residual update of a
transformer sublayer taken as a step along a geodesic of the hypersphere, not as
``Norm(x + s)``.

Post-Norm computes ``Proj(x + alpha * s)``, a projected-gradient step: the sublayer
output ``s`` is an estimated gradient direction, the residual add leaves the
sphere, and the normalization projects back. Projection distorts the step and
puts kinks in the trajectory of the hidden state. GeoNorm instead projects ``s``
onto the tangent space at ``x`` and moves along the great circle through ``x`` in
that direction, which stays on the sphere of radius ``||x||`` by construction::

    v     = s - (x . s) / ||x||^2 * x                    tangent projection
    theta = alpha_k * min(||v|| / ||x||, clamp)          rotation angle
    out   = cos(theta) * x + ||x|| * sin(theta) * v / ||v||

Norms run per token over the last axis. ``out`` has the same norm as ``x`` and the
layer has no trainable weights. ``alpha_k = alpha * decay(k)`` follows the paper's
layer-wise step decay, with ``k`` the 0-based ``layer_idx`` of the transformer
block (both sublayers of one block share it)::

    harmonic   1 / (k + 1)          (the paper's best)
    sqrt       1 / sqrt(k + 1)
    linear     (T - k) / T          T = total_layers

The clamp (default ``pi / 4``, the paper's best) is applied to the raw ratio
``||v|| / ||x||`` BEFORE the decay, so under the harmonic schedule the largest
rotation at block ``k`` is ``alpha * clamp / (k + 1)``. ``clamp=None`` disables it.

This is a two-input layer, ``call([x, s], layer_idx)``, so it is not a key of
:func:`~dl_techniques.layers.norms.factory.create_normalization_layer`, whose
contract is one tensor in and one out. :class:`PolarWeightNorm` is the other
norm in this package that is not a factory type.

Differences from the paper's Appendix D listing, which does not run as printed:
its ``clamp`` constructor argument is ignored, its harmonic branch assigns to
``theta_attn`` so no decay applies, its default method name is misspelled, and it
creates integer ``scale`` and ``bias`` parameters although the text states that
GeoNorm adds no trainable parameters. This layer follows the text.

References:
    - Zheng et al., 2026. GeoNorm: Unify Pre-Norm and Post-Norm with Geodesic
      Optimization. (https://arxiv.org/abs/2601.22095)
"""

import math
import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.dtype_policy import stability_floor, statistics_dtype
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: The step-decay schedules the paper defines.
GEO_NORM_SCHEDULES = ("harmonic", "sqrt", "linear")

#: The paper's best clamp on the rotation angle (Appendix C: pi/2, pi/4, pi/8).
DEFAULT_CLAMP = math.pi / 4

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.norms.geo_norm")
class GeoNorm(keras.layers.Layer):
    """Geodesic update of a hidden state along a sublayer output, on the sphere.

    Replaces the Post-Norm step ``Norm(x + sublayer(x))``. ``call`` takes the
    state ``x`` and the sublayer output ``s`` as a two-element list and returns the
    state after one geodesic step. The layer owns no weights.

    :param schedule: Layer-wise step decay, one of ``'harmonic'``, ``'sqrt'``,
        ``'linear'``. Defaults to ``'harmonic'``.
    :type schedule: str
    :param alpha: Base step size ``alpha``, positive. ``alpha_k = alpha *
        decay(k)``. Defaults to 1.0.
    :type alpha: float
    :param clamp: Upper bound on the raw angle ``||v|| / ||x||`` in radians, applied
        before the decay, or ``None`` to disable it. Defaults to ``pi / 4``.
    :type clamp: Optional[float]
    :param total_layers: ``T``, the number of blocks. Required for
        ``schedule='linear'``, ignored otherwise.
    :type total_layers: Optional[int]
    :param epsilon: Floor on the radius ``||x||`` and on the tangent norm
        ``||v||``, applied as ``maximum(sum_of_squares, epsilon**2)``. Defaults to
        1e-6, the paper's Appendix D floor on the radius (``clamp(R, min=1e-6)``). The
        same listing floors the tangent norm at 1e-8; this layer floors both at
        ``epsilon``, which changes the result only for a step with
        ``||v|| < epsilon``, below the float32 resolution of ``x`` (measured in
        ``test_geo_norm.py``).
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    :type kwargs: Any

    :raises ValueError: If ``schedule`` is unknown, ``alpha`` or ``epsilon`` is not
        positive, ``clamp`` is not a positive number or ``None``, ``total_layers``
        is not a positive integer, or ``schedule='linear'`` has no
        ``total_layers``.

    Input shape:
        A list ``[x, s]`` of two tensors of identical shape ``(..., features)``.

    Output shape:
        The shape of ``x``, with the norm of ``x`` along the last axis.

    Example:
        >>> geo = GeoNorm()
        >>> x = keras.random.normal((2, 16, 64))
        >>> s = keras.random.normal((2, 16, 64))
        >>> y = geo([x, s], layer_idx=3)      # ||y|| == ||x|| per token

    Note:
        ``layer_idx`` is a call argument and is absent from ``get_config()``. A
        stack must pass its own depth, or every block runs at ``k = 0``, the largest
        step.
    """

    def __init__(
        self,
        schedule: str = "harmonic",
        alpha: float = 1.0,
        clamp: Optional[float] = DEFAULT_CLAMP,
        total_layers: Optional[int] = None,
        epsilon: float = 1e-6,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if schedule not in GEO_NORM_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {GEO_NORM_SCHEDULES}, got {schedule!r}"
            )
        if not alpha > 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if clamp is not None and not clamp > 0:
            raise ValueError(f"clamp must be positive or None, got {clamp}")
        if not epsilon > 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if total_layers is not None and (
            isinstance(total_layers, bool) or not isinstance(total_layers, int) or total_layers <= 0
        ):
            raise ValueError(f"total_layers must be a positive integer or None, got {total_layers!r}")
        if schedule == "linear" and total_layers is None:
            raise ValueError("schedule='linear' needs total_layers: its decay is (T - k) / T")

        self.schedule = schedule
        self.alpha = alpha
        self.clamp = clamp
        self.total_layers = total_layers
        self.epsilon = epsilon

        logger.debug(
            f"GeoNorm: schedule={schedule}, alpha={alpha}, clamp={clamp}, total_layers={total_layers}"
        )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Check that state and sublayer output agree. The layer has no weights.

        :param input_shape: ``[x_shape, s_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If it is not two shapes of equal rank and compatible
            static dimensions.
        """
        if self.built:
            return

        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"GeoNorm takes [x, sublayer_output], got input_shape={input_shape}"
            )
        x_shape, s_shape = (tuple(shape) for shape in input_shape)
        if len(x_shape) != len(s_shape) or any(
            a is not None and b is not None and a != b for a, b in zip(x_shape, s_shape)
        ):
            raise ValueError(
                f"state and sublayer output must have the same shape, got {x_shape} and {s_shape}"
            )

        super().build(input_shape)

    def decay(self, layer_idx: Union[int, keras.KerasTensor], dtype: str) -> keras.KerasTensor:
        """Step decay ``decay(k)`` for a 0-based block index ``k``.

        :param layer_idx: 0-based index of the transformer block.
        :type layer_idx: Union[int, keras.KerasTensor]
        :param dtype: dtype of the returned scalar.
        :type dtype: str
        :return: ``1/(k+1)``, ``1/sqrt(k+1)`` or ``(T-k)/T``.
        :rtype: keras.KerasTensor
        """
        k = keras.ops.cast(layer_idx, dtype)
        if self.schedule == "harmonic":
            return 1.0 / (k + 1.0)
        if self.schedule == "sqrt":
            return 1.0 / keras.ops.sqrt(k + 1.0)
        total = float(self.total_layers)
        return (total - k) / total

    def call(
        self,
        inputs: List[keras.KerasTensor],
        layer_idx: int = 0,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Take one geodesic step from ``x`` along the tangent part of ``s``.

        :param inputs: ``[x, s]``, the state and the sublayer output.
        :type inputs: List[keras.KerasTensor]
        :param layer_idx: 0-based index of the transformer block, selecting the
            step decay. Defaults to 0.
        :type layer_idx: int
        :param training: Unused; the layer behaves identically in both modes.
        :type training: Optional[bool]
        :return: The updated state, same shape and dtype as ``x``.
        :rtype: keras.KerasTensor
        """
        x, update = inputs
        # Static contract, re-asserted: build() sees only the first shapes to arrive.
        if len(x.shape) != len(update.shape) or any(
            a is not None and b is not None and a != b for a, b in zip(x.shape, update.shape)
        ):
            raise ValueError(
                f"state and sublayer output must have the same shape, got {tuple(x.shape)} and {tuple(update.shape)}"
            )
        out_dtype = x.dtype
        stat_dtype = statistics_dtype(out_dtype)
        x = keras.ops.cast(x, stat_dtype)
        update = keras.ops.cast(update, stat_dtype)

        # Floors act on the sum of squares: the gradient of sqrt at an exact zero is
        # not finite, and maximum() routes it to the constant floor instead.
        floor_sq = stability_floor(stat_dtype, self.epsilon ** 2)
        radius_sq = keras.ops.maximum(
            keras.ops.sum(x * x, axis=-1, keepdims=True), floor_sq
        )
        radius = keras.ops.sqrt(radius_sq)

        # Tangent projection: remove the component of s along x.
        along = keras.ops.sum(x * update, axis=-1, keepdims=True) / radius_sq
        tangent = update - along * x
        tangent_norm = keras.ops.sqrt(keras.ops.maximum(
            keras.ops.sum(tangent * tangent, axis=-1, keepdims=True), floor_sq
        ))

        ratio = tangent_norm / radius
        if self.clamp is not None:
            ratio = keras.ops.minimum(ratio, keras.ops.cast(self.clamp, stat_dtype))
        theta = self.alpha * self.decay(layer_idx, stat_dtype) * ratio

        output = (
            keras.ops.cos(theta) * x
            + radius * keras.ops.sin(theta) * (tangent / tangent_norm)
        )
        return keras.ops.cast(output, out_dtype)

    def compute_output_shape(
        self, input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Return the shape of ``x``.

        :param input_shape: ``[x_shape, s_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: ``x_shape``.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape[0])

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration, covering every constructor argument.

        ``layer_idx`` is a ``call`` argument and is absent.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "schedule": self.schedule,
            "alpha": self.alpha,
            "clamp": self.clamp,
            "total_layers": self.total_layers,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
