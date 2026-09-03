"""THERA ``linear_up`` frequency initializer for neural heat fields.

Provides :class:`LinearUpInitializer`, which draws the first-layer frequencies
of a SIREN-style neural heat field uniformly from a disk in frequency space.

Each of the ``N`` output units owns a 2D frequency vector ``(f_x, f_y)`` whose
polar form is::

    r     = pi * scale * sqrt(U_1),    U_1 ~ Uniform(0, 1)
    theta = 2 * pi * U_2,              U_2 ~ Uniform(0, 1)
    f_x   = r * cos(theta)
    f_y   = r * sin(theta)

The square root on the radius makes ``(f_x, f_y)`` uniform over the disk of
radius ``pi * scale``: the probability mass per unit area is constant, so
``r^2 / (pi*scale)^2 ~ Uniform(0, 1)``. That gives the initial frequencies an
isotropic, band-limited spread tied to the query scale, which keeps the SIREN
well conditioned at initialization.

The produced tensor has shape ``(2, N)`` (or ``(..., 2, N)``): row 0 holds the
``f_x`` components and row 1 the ``f_y`` components, matching the reference JAX
``linear_up``, which concatenates the x and y rows along ``axis=-2``.

Note:
    THERA's companion ``uniform_between(a, b)`` initializer is just
    ``keras.initializers.RandomUniform(minval=a, maxval=b)``. Use that stock
    initializer directly; no custom class is needed.

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields" (original JAX/Flax ``model/init.py::linear_up``).
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.initializers.linear_up_initializer")
class LinearUpInitializer(keras.initializers.Initializer):
    """Draw 2D heat-field frequencies uniformly over a disk of radius ``pi*scale``.

    Produces a ``(2, N)`` (or ``(..., 2, N)``) tensor of 2D frequency vectors.
    For each of the ``N`` columns a radius ``r = pi * scale * sqrt(U)`` and an
    angle ``theta = 2*pi*U`` are drawn from independent uniforms, then the
    column is set to ``(r*cos(theta), r*sin(theta))``.

    **Construction:**

    .. code-block:: text

        requested shape (..., 2, N)
                  │
          ┌───────┴────────┐
          ▼                ▼
        U_norm           U_theta        both Uniform(0, 1)
        [..., 1, N]      [..., 1, N]
          │                │
          ▼                ▼
        r = pi*scale     theta =
            *sqrt(U)       2*pi*U
          │                │
          └───────┬────────┘
                  │
          ┌───────┴────────┐
          ▼                ▼
        ┌────────────┐  ┌────────────┐
        │ r*cos(th)  │  │ r*sin(th)  │
        │   = f_x    │  │   = f_y    │
        └─────┬──────┘  └─────┬──────┘
              │ [..., 1, N]   │ [..., 1, N]
              └───────┬───────┘
                      ▼
              concatenate(axis=-2)
                      │
                      ▼
                [..., 2, N]   row 0 = f_x, row 1 = f_y

    The axis convention follows the reference JAX ``linear_up``: the
    second-to-last axis must have size 2 (the x/y split) and the last axis is
    the number of frequency vectors ``N``. Leading dimensions are supported by
    broadcasting the per-column draw across them.

    :param scale: Frequency-scale factor. The sampling disk has radius
        ``pi * scale``. Must be non-negative.
    :type scale: float
    :param seed: Optional integer seed. Two initializers with the same seed
        produce identical tensors.
    :type seed: int or None

    :ivar scale: The coerced frequency-scale factor.
    :vartype scale: float
    :ivar seed: The seed as passed by the caller.
    :vartype seed: int or None

    :raises ValueError: If ``scale`` is negative.

    Example:
        >>> init = LinearUpInitializer(scale=1.0, seed=0)
        >>> w = init((2, 64))
    """

    def __init__(
        self,
        scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Validate the scale and build the seed generator.

        :param scale: Frequency-scale factor; the disk radius is ``pi * scale``.
        :type scale: float
        :param seed: Optional integer seed.
        :type seed: int or None
        :raises ValueError: If ``scale`` is negative.
        """
        super().__init__()
        if scale < 0:
            raise ValueError(f"scale must be non-negative, got {scale}")
        self.scale = float(scale)
        self.seed = seed
        self._seed_generator = keras.random.SeedGenerator(seed)
        logger.debug(
            f"Initialized LinearUpInitializer(scale={self.scale}, seed={self.seed})"
        )

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, Any]] = None,
    ) -> Any:
        """Draw the frequency vectors.

        :param shape: Requested shape ``(..., 2, N)``; rank at least 2 with
            ``shape[-2] == 2``.
        :type shape: tuple of int
        :param dtype: Data type of the result. ``None`` falls back to
            ``keras.config.floatx()``.
        :type dtype: str or None
        :return: The frequency matrix, row 0 holding ``f_x`` and row 1 ``f_y``.
        :rtype: tensor
        :raises ValueError: If the shape has rank below 2 or ``shape[-2] != 2``.
        """
        if dtype is None:
            dtype = keras.config.floatx()
        if len(shape) < 2:
            raise ValueError(
                f"LinearUpInitializer requires rank >= 2 (got shape {shape}); "
                f"the second-to-last axis is the x/y split."
            )
        if shape[-2] != 2:
            raise ValueError(
                f"LinearUpInitializer requires shape[-2] == 2 (the x/y row "
                f"split), got shape {shape}."
            )

        n = shape[-1]
        leading = tuple(shape[:-2])
        # Draw on a (..., 1, N) tensor so the x and y rows stack along axis -2.
        draw_shape = leading + (1, n)

        pi = keras.ops.convert_to_tensor(3.141592653589793, dtype="float32")
        u_norm = keras.random.uniform(
            draw_shape, minval=0.0, maxval=1.0,
            dtype="float32", seed=self._seed_generator,
        )
        u_theta = keras.random.uniform(
            draw_shape, minval=0.0, maxval=1.0,
            dtype="float32", seed=self._seed_generator,
        )

        norm = pi * self.scale * keras.ops.sqrt(u_norm)
        theta = 2.0 * pi * u_theta
        # x and y are the (..., 1, N) f_x and f_y rows.
        x = norm * keras.ops.cos(theta)
        y = norm * keras.ops.sin(theta)
        # Stacking them gives the (..., 2, N) result.
        result = keras.ops.concatenate([x, y], axis=-2)

        return keras.ops.cast(result, dtype)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding ``scale`` and ``seed``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "scale": self.scale,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LinearUpInitializer":
        """Rebuild an initializer from a config dict.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: dict
        :return: A new initializer.
        :rtype: LinearUpInitializer
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        :return: A string naming the scale and seed.
        :rtype: str
        """
        return f"LinearUpInitializer(scale={self.scale}, seed={self.seed})"

# ---------------------------------------------------------------------
