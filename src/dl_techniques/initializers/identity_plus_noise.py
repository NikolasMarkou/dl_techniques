import keras
from keras import ops
from typing import Optional, Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class IdentityPlusNoise(keras.initializers.Initializer):
    """Initializer that returns ``eye(H) + RandomNormal(stddev, seed)``.

    Reproducible under ``keras.utils.set_random_seed`` when ``seed`` is set,
    or when ``seed=None`` and the global Keras seed is set. Preserves the
    invariant that ``stddev=0`` ⇒ exact identity.
    """

    def __init__(self, stddev: float = 0.01, seed: Optional[int] = None) -> None:
        self.stddev = float(stddev)
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"IdentityPlusNoise expects a square 2-D shape, got {shape}"
            )
        dtype = dtype or "float32"
        eye = ops.eye(shape[0], dtype=dtype)
        if self.stddev == 0.0:
            return eye
        noise = keras.random.normal(
            shape, mean=0.0, stddev=self.stddev, dtype=dtype, seed=self.seed
        )
        return eye + noise

    def get_config(self) -> Dict[str, Any]:
        return {
            "stddev": self.stddev,
            "seed": self.seed
        }

# ---------------------------------------------------------------------