"""THERA ``linear_up`` frequency initializer for neural heat fields (SIREN-style).

THERA (Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields)
initializes the *first-layer frequencies* of its thermal-activated SIREN field
with a 2D vector field drawn uniformly from a disk in frequency space. Each
output unit ``n`` (of ``N``) owns a 2D frequency vector ``(f_x, f_y)`` whose
polar form is

    r     = pi * scale * sqrt(U_1),    U_1 ~ Uniform(0, 1)
    theta = 2 * pi * U_2,              U_2 ~ Uniform(0, 1)
    f_x   = r * cos(theta)
    f_y   = r * sin(theta)

The ``sqrt`` on the radius makes ``(f_x, f_y)`` **uniform over the disk** of
radius ``pi * scale`` (uniform-on-disk sampling): the probability mass per unit
area is constant, so ``r^2 / (pi*scale)^2 ~ Uniform(0, 1)``. This gives the
heat-field's initial frequencies an isotropic, band-limited spread tied to the
query scale, which is what keeps the SIREN well-conditioned at initialization.

The produced tensor has shape ``(2, N)`` (or ``(..., 2, N)``): row 0 is the
``f_x`` components, row 1 is the ``f_y`` components, matching the reference JAX
``linear_up`` which concatenates the x and y rows along ``axis=-2``.

Note:
    THERA's companion ``uniform_between(a, b)`` initializer is simply
    ``keras.initializers.RandomUniform(minval=a, maxval=b)`` and needs no custom
    class — use that stock initializer directly.

Reference:
    Becker et al., "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with
    Neural Heat Fields" (original JAX/Flax ``model/init.py::linear_up``).
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LinearUpInitializer(keras.initializers.Initializer):
    """Initialize 2D heat-field frequencies uniformly over a disk of radius ``pi*scale``.

    Produces a ``(2, N)`` (or ``(..., 2, N)``) tensor of 2D frequency vectors.
    For each of the ``N`` columns a radius ``r = pi * scale * sqrt(U)`` and angle
    ``theta = 2*pi*U`` are drawn (``U ~ Uniform(0,1)`` independently), then the
    column is set to ``(r*cos(theta), r*sin(theta))`` with the x-component in
    row 0 and the y-component in row 1. The ``sqrt`` radius warp makes the
    samples uniform over the disk, so ``r^2 / (pi*scale)^2 ~ Uniform(0, 1)``.

    The required-axis convention follows the reference JAX ``linear_up``: the
    second-to-last axis must have size ``2`` (the x/y split), and the last axis
    is the number of frequency vectors ``N``. Leading dimensions are supported
    by broadcasting the per-column draw across them.

    Args:
        scale: Frequency-scale factor. The sampling disk has radius
            ``pi * scale``. Must be non-negative. Defaults to 1.0.
        seed: Optional integer seed for reproducibility. When set, two
            initializers with the same seed produce identical tensors.

    Example:
        >>> init = LinearUpInitializer(scale=1.0, seed=0)
        >>> w = init((2, 64))  # 64 frequency vectors; row 0 = f_x, row 1 = f_y

    Raises:
        ValueError: If ``scale`` is negative, or the requested ``shape`` does not
            have ``shape[-2] == 2``.
    """

    def __init__(
        self,
        scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
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
        # Draw the per-column radius/angle on a (..., 1, N) tensor so that the
        # x and y rows are stacked along axis -2 to produce the (..., 2, N) out.
        draw_shape = leading + (1, n)

        pi = ops.convert_to_tensor(3.141592653589793, dtype="float32")
        u_norm = keras.random.uniform(
            draw_shape, minval=0.0, maxval=1.0,
            dtype="float32", seed=self._seed_generator,
        )
        u_theta = keras.random.uniform(
            draw_shape, minval=0.0, maxval=1.0,
            dtype="float32", seed=self._seed_generator,
        )

        norm = pi * self.scale * ops.sqrt(u_norm)
        theta = 2.0 * pi * u_theta
        x = norm * ops.cos(theta)           # (..., 1, N) -> f_x row
        y = norm * ops.sin(theta)           # (..., 1, N) -> f_y row
        result = ops.concatenate([x, y], axis=-2)   # (..., 2, N)

        return ops.cast(result, dtype)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "scale": self.scale,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LinearUpInitializer":
        return cls(**config)

    def __repr__(self) -> str:
        return f"LinearUpInitializer(scale={self.scale}, seed={self.seed})"

# ---------------------------------------------------------------------
