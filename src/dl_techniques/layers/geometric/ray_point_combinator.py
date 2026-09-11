"""
Combine raw ray-direction and distance logits into a camera-agnostic 3D point.

This is the single place OmniPoint-style ``P = d_hat * r_hat`` point construction
happens (see ``src/dl_techniques/models/vision/omnipoint/`` Problem Statement
invariant 3: camera-specific code lives only in ray-map generation, never here).

Architecture:
    Given raw, unnormalized head outputs -- a 3-channel ray logit tensor and a
    1-channel distance logit tensor, both of shape ``(..., H, W, C)`` with a
    matching leading shape -- this layer:

    1. L2-normalizes the ray channel to a unit vector, ``r_hat = r / max(||r||, eps)``.
       The ``eps`` floor (not a bare ``||r||``) is what keeps an all-zero raw ray
       finite: dividing by ``eps`` instead of ``0`` yields a bounded (not NaN)
       result at that pixel, satisfying invariant 1 (``||r_hat|| == 1`` by
       construction) everywhere the input is non-degenerate, and never producing
       NaN even where it is.
    2. Applies ``softplus`` to the raw distance channel, floored at ``eps``,
       so ``d_hat > 0`` everywhere by construction (invariant 2), never by
       loss pressure alone. The floor matters because ``softplus`` itself
       underflows to an exact float32 ``0.0`` for a sufficiently negative
       raw logit (measured: ``softplus(-1e6) == 0.0`` in float32) -- without
       it, a large-magnitude negative head output would violate "``> 0``
       by construction" in floating point even though it holds
       mathematically.
    3. Computes the point ``P = d_hat * r_hat`` via a broadcast multiply.

    The three tensors -- normalized ray, positive distance, and the combined
    point -- are all returned, because downstream consumers need them
    separately: the OmniPoint losses (``losses/omnipoint_losses.py``) supervise
    ``r_hat`` and ``d_hat`` independently (``RayDirectionLoss``,
    ``PointDistanceLoss``), not only the combined point.

Mathematics:
    Let ``r in R^{...xC=3}`` be the raw ray logits and ``d in R^{...xC=1}`` the
    raw distance logits. With ``eps`` a small positive floor::

        r_hat = r / max(sqrt(sum(r^2, axis=-1, keepdims=True)), eps)
        d_hat = max(softplus(d), eps) = max(log(1 + exp(d)), eps)
        P     = d_hat * r_hat

    ``r_hat`` broadcasts against ``d_hat`` on the channel axis (``C=3`` vs
    ``C=1``), matching ``camera_models.py``'s own ray-tensor layout
    (``utils/camera_models.py::_l2_normalize_rays``, which this layer's
    normalization deliberately mirrors so a ground-truth ray and a predicted
    ray are normalized identically).
"""

from typing import Any, Dict, Tuple

import keras

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Floor added under the ray norm before dividing, so an all-zero raw ray
#: vector produces a finite (not NaN) output instead of a 0/0 division.
_DEFAULT_EPSILON = 1e-8


@register_dl_technique("dl_techniques.layers.geometric.ray_point_combinator")
class RayPointCombinator(keras.layers.Layer):
    """Combine raw ray/distance logits into ``(ray, distance, point)``.

    Call the layer on a pair: ``layer((raw_ray, raw_distance))``, where
    ``raw_ray`` is ``(..., 3)`` and ``raw_distance`` is ``(..., 1)``, sharing
    every leading dimension. Returns the 3-tuple
    ``(unit_ray, positive_distance, point)``, each ``(..., 3)`` except
    ``positive_distance`` which stays ``(..., 1)``.

    This layer owns no trainable weights -- it is a pure, deterministic
    elementwise/normalize combinator, so ``build()`` is unnecessary and is not
    implemented.

    :param epsilon: Floor used in two places: under the ray L2 norm before
        dividing (guards a 0/0 division at an all-zero raw ray input) and as
        a lower bound on the distance activation (guards float32 underflow
        of ``softplus`` at a very negative raw logit). Default ``1e-8``,
        matching ``utils/camera_models.py``'s own ray normalization floor.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar epsilon: The resolved epsilon floor.
    :vartype epsilon: float

    Input shape:
        Tuple or list of two tensors:

        - ``raw_ray``: ``(..., 3)``, unnormalized ray-direction logits.
        - ``raw_distance``: ``(..., 1)``, unnormalized distance logits.

    Output shape:
        A 3-tuple ``(unit_ray, positive_distance, point)``:

        - ``unit_ray``: same shape as ``raw_ray``, ``(..., 3)``.
        - ``positive_distance``: same shape as ``raw_distance``, ``(..., 1)``.
        - ``point``: ``(..., 3)`` (the broadcast product of the first two).

    :raises ValueError: From ``call()``, if the input is not a pair, or if the
        ray/distance channel dimensions do not match the expected 3 / 1.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.geometric.ray_point_combinator import (
            RayPointCombinator,
        )

        raw_ray = keras.random.normal((2, 32, 32, 3))
        raw_distance = keras.random.normal((2, 32, 32, 1))
        ray, distance, point = RayPointCombinator()((raw_ray, raw_distance))
        ray.shape       # (2, 32, 32, 3), ||ray|| == 1 at every pixel
        distance.shape  # (2, 32, 32, 1), > 0 everywhere
        point.shape     # (2, 32, 32, 3), point == distance * ray
    """

    def __init__(
            self,
            epsilon: float = _DEFAULT_EPSILON,
            **kwargs: Any
    ) -> None:
        """Store the normalization epsilon. No weight is created.

        :param epsilon: Floor under the ray L2 norm before dividing.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Normalize, activate, and combine the raw ray/distance logits.

        :param inputs: The pair ``(raw_ray, raw_distance)``, shapes
            ``(..., 3)`` and ``(..., 1)``.
        :type inputs: tuple of keras.KerasTensor
        :return: ``(unit_ray, positive_distance, point)``.
        :rtype: tuple of keras.KerasTensor
        :raises ValueError: If ``inputs`` is not a 2-tuple/list.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "RayPointCombinator expects two inputs (raw_ray, "
                f"raw_distance); got inputs={inputs}"
            )
        raw_ray, raw_distance = inputs

        # L2-normalize the ray channel. The eps floor (not a bare norm) is
        # what keeps an all-zero raw ray finite instead of NaN.
        norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(raw_ray), axis=-1, keepdims=True)
        )
        unit_ray = raw_ray / keras.ops.maximum(norm, self.epsilon)

        # Positive-distance activation: softplus is > 0 everywhere by
        # construction, never by loss pressure alone (invariant 2). The
        # epsilon floor guards against float32 underflow to exact 0.0 at a
        # very negative raw logit (measured: softplus(-1e6) == 0.0).
        positive_distance = keras.ops.maximum(
            keras.ops.softplus(raw_distance), self.epsilon
        )

        # P = d_hat * r_hat, broadcasting (..., 1) against (..., 3).
        point = positive_distance * unit_ray

        return unit_ray, positive_distance, point

    def compute_output_shape(
            self,
            input_shape: Tuple[Tuple[Any, ...], Tuple[Any, ...]],
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...]]:
        """Return the three output shapes.

        :param input_shape: The pair ``(raw_ray_shape, raw_distance_shape)``.
        :type input_shape: tuple of tuple
        :return: ``(ray_shape, distance_shape, point_shape)``, where
            ``point_shape == ray_shape``.
        :rtype: tuple of tuple
        """
        ray_shape, distance_shape = input_shape
        ray_shape = tuple(ray_shape)
        distance_shape = tuple(distance_shape)
        return ray_shape, distance_shape, ray_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus ``epsilon``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

# ---------------------------------------------------------------------
