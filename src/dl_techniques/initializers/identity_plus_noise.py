"""Near-identity initializer for square mixing matrices.

Provides :class:`IdentityPlusNoise`, which fills a square 2-D weight with the
identity matrix plus small Gaussian noise. Use it for a coupling or mixing
matrix that should start as an approximate no-op. ``WaveFieldAttention`` uses
it for its cross-head ``field_coupling`` matrix.
"""

import keras
from typing import Optional, Any, Dict
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.initializers.identity_plus_noise")
class IdentityPlusNoise(keras.initializers.Initializer):
    """Return ``eye(H) + RandomNormal(stddev, seed)`` for a square 2-D shape.

    **Construction:**

    .. code-block:: text

        requested shape (H, H)
                  │
          ┌───────┴────────┐
          ▼                ▼
        ┌────────┐   ┌──────────────┐
        │ eye(H) │   │ RandomNormal │  (skipped when stddev == 0)
        │        │   │ (0, stddev)  │
        └───┬────┘   └──────┬───────┘
            │  [H, H]       │  [H, H]
            └───────┬───────┘
                    ▼
                 ┌─────┐
                 │  +  │
                 └──┬──┘
                    ▼
                 [H, H]

    ``stddev=0`` returns the exact identity. The draw is reproducible under
    ``keras.utils.set_random_seed`` when ``seed`` is set, and also when
    ``seed=None`` and the global Keras seed is set.

    :param stddev: Standard deviation of the additive Gaussian noise.
    :type stddev: float
    :param seed: Optional integer seed for the noise draw.
    :type seed: int or None

    :ivar stddev: The coerced noise standard deviation.
    :vartype stddev: float
    :ivar seed: The seed as passed by the caller.
    :vartype seed: int or None

    Example:
        >>> init = IdentityPlusNoise(stddev=0.01, seed=0)
        >>> w = init((8, 8))
    """

    def __init__(self, stddev: float = 0.01, seed: Optional[int] = None) -> None:
        """Store the noise scale and seed.

        :param stddev: Standard deviation of the additive Gaussian noise.
        :type stddev: float
        :param seed: Optional integer seed for the noise draw.
        :type seed: int or None
        """
        self.stddev = float(stddev)
        self.seed = seed

    def __call__(self, shape, dtype=None):
        """Build the near-identity matrix.

        :param shape: Requested shape. Must be square and 2-D.
        :type shape: tuple of int
        :param dtype: Data type of the result. ``None`` means ``"float32"``.
        :type dtype: str or None
        :return: An ``(H, H)`` tensor holding the identity plus noise.
        :rtype: tensor
        :raises ValueError: If the shape is not a square 2-D shape.
        """
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"IdentityPlusNoise expects a square 2-D shape, got {shape}"
            )
        dtype = dtype or "float32"
        eye = keras.ops.eye(shape[0], dtype=dtype)
        if self.stddev == 0.0:
            return eye
        noise = keras.random.normal(
            shape, mean=0.0, stddev=self.stddev, dtype=dtype, seed=self.seed
        )
        return eye + noise

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding ``stddev`` and ``seed``.
        :rtype: dict
        """
        return {
            "stddev": self.stddev,
            "seed": self.seed
        }

# ---------------------------------------------------------------------
