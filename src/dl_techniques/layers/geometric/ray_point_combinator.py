"""Combine raw ray and distance logits into a 3D point.

``RayPointCombinator`` takes a 3-channel ray logit tensor and a 1-channel
distance logit tensor and returns a unit ray, a positive distance, and their
product:

    r_hat = r / max(||r||, eps)
    d_hat = max(softplus(d), eps)
    P     = d_hat * r_hat

Both floors matter in floating point, not only on paper. Dividing by
``max(||r||, eps)`` rather than by ``||r||`` keeps an all-zero raw ray finite
instead of NaN, and flooring the softplus keeps the distance above zero for a
very negative logit, where ``softplus`` underflows to exact ``0.0`` in float32.
The normalization matches ``utils/camera_models.py``, so a predicted ray and a
ground-truth ray are normalized the same way.

All three tensors are returned, because the OmniPoint losses supervise the ray
and the distance separately rather than only the combined point. The layer
holds no weights and implements no ``build()``.
"""

import keras
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Floor used both under the ray norm and under the distance activation.
_DEFAULT_EPSILON = 1e-8

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.geometric.ray_point_combinator")
class RayPointCombinator(keras.layers.Layer):
    """Turn raw ray and distance logits into ``(ray, distance, point)``.

    Call the layer on a pair, ``layer((raw_ray, raw_distance))``, where
    ``raw_ray`` is ``(..., 3)`` and ``raw_distance`` is ``(..., 1)`` and every
    leading dimension matches. The ray is L2-normalized with an epsilon floor
    under the norm, the distance goes through a floored softplus, and the point
    is their broadcast product. The layer owns no trainable weights, so it is a
    pure elementwise and normalize combinator.

    Architecture:

    .. code-block:: text

            raw_ray [..., 3]              raw_distance [..., 1]
                    │                               │
                    ▼                               ▼
          ┌───────────────────┐           ┌───────────────────┐
          │ norm over last    │           │ softplus(d)       │
          │  axis, keepdims   │           │                   │
          └─────────┬─────────┘           └─────────┬─────────┘
                    ▼                               ▼
          ┌───────────────────┐           ┌───────────────────┐
          │ r / max(n, eps)   │           │ max(s, eps)       │
          └─────────┬─────────┘           └─────────┬─────────┘
            unit_ray [..., 3]          positive_distance [..., 1]
                    │                               │
                    ├───────────────┬───────────────┤
                    │               ▼               │
                    │     ┌───────────────────┐     │
                    │     │ d_hat * r_hat     │     │
                    │     └─────────┬─────────┘     │
                    ▼               ▼               ▼
                unit_ray          point      positive_distance
                [..., 3]         [..., 3]        [..., 1]

    ``call()`` returns them in the order
    ``(unit_ray, positive_distance, point)``.

    :param epsilon: Floor used in two places: under the ray L2 norm before
        dividing, which guards a 0/0 division at an all-zero raw ray, and as a
        lower bound on the distance activation, which guards float32 underflow
        of ``softplus``. Defaults to ``1e-8``, matching
        ``utils/camera_models.py``'s ray-normalization floor.
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
        - ``point``: ``(..., 3)``, the broadcast product of the first two.

    :raises ValueError: From ``call()``, if the input is not a pair. Channel
        counts are not checked; a wrong one surfaces as a broadcast error.

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
        """Store the epsilon floor. No weight is created.

        Arguments are documented on the class.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Normalize, activate, and combine the raw ray and distance logits.

        :param inputs: The pair ``(raw_ray, raw_distance)``, shapes
            ``(..., 3)`` and ``(..., 1)``.
        :type inputs: tuple of keras.KerasTensor
        :return: ``(unit_ray, positive_distance, point)``.
        :rtype: tuple of keras.KerasTensor
        :raises ValueError: If ``inputs`` is not a 2-tuple or 2-list.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "RayPointCombinator expects two inputs (raw_ray, "
                f"raw_distance); got inputs={inputs}"
            )
        raw_ray, raw_distance = inputs

        # The eps floor, rather than a bare norm, keeps an all-zero raw ray
        # finite instead of NaN.
        norm = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(raw_ray), axis=-1, keepdims=True)
        )
        unit_ray = raw_ray / keras.ops.maximum(norm, self.epsilon)

        # softplus underflows to exact 0.0 in float32 at a very negative logit,
        # so the floor is what makes the distance positive in practice.
        positive_distance = keras.ops.maximum(
            keras.ops.softplus(raw_distance), self.epsilon
        )

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