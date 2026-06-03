"""Polar-coordinate weight initializer with exact per-vector norm control.

Inspired by PolarQuant (Han et al., 2025): a Gaussian vector's polar
representation has a magnitude (radius) that is statistically *independent* of
its direction (angles), and the directional distribution is exactly the
uniform distribution on the unit sphere (PolarQuant Lemma 2 -- the level-wise
``sin^k`` angle p.d.f.s are precisely those induced by a uniform direction).

``PolarInitializer`` exploits this to "sample in polar coordinates": it draws a
direction uniformly on the sphere (by normalizing a Gaussian) and sets the
radius to an *exact* user-specified value. This decouples norm from direction
and gives precise control over the per-vector norm at initialization -- e.g.
every neuron can start with identical weight-vector norm ("equinorm init"),
which is impossible with plain Gaussian/He/Glorot sampling (whose norms are
chi-distributed).

Sampling ``radius * gaussian / ||gaussian||`` is mathematically identical to
sampling the analytic level-wise angle p.d.f.s and running the inverse polar
transform, but works for arbitrary (non power-of-two) shapes with no extra
machinery, which is why it is implemented directly.
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PolarInitializer(keras.initializers.Initializer):
    """Initialize weights with an exact per-vector L2 norm and uniform direction.

    Every slice of the produced tensor taken along ``axis`` is set to have L2
    norm exactly ``gain * norm`` (or ``gain * sqrt(2)`` when ``norm is None``,
    matching the expected weight-vector energy of He-normal initialization), and
    a direction drawn uniformly on the unit sphere.

    Args:
        norm: Target L2 norm of each vector along ``axis``. If ``None`` a
            He-normal-equivalent energy of ``sqrt(2)`` is used. Defaults to None.
        axis: Axis along which each weight vector lies. For a Dense kernel of
            shape ``(fan_in, units)``, ``axis=0`` gives every output unit's
            weight vector the target norm. Defaults to 0.
        gain: Multiplicative scale applied to the target norm. Defaults to 1.0.
        seed: Optional integer seed for reproducibility.

    Example:
        >>> init = PolarInitializer(norm=1.0, axis=0, seed=0)  # unit-norm columns
        >>> w = init((64, 32))  # each of the 32 columns has ||.||_2 == 1
    """

    def __init__(
        self,
        norm: Optional[float] = None,
        axis: int = 0,
        gain: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if norm is not None and norm < 0:
            raise ValueError(f"norm must be non-negative, got {norm}")
        self.norm = None if norm is None else float(norm)
        self.axis = int(axis)
        self.gain = float(gain)
        self.seed = seed
        logger.debug(
            f"Initialized PolarInitializer(norm={self.norm}, axis={self.axis}, "
            f"gain={self.gain}, seed={self.seed})"
        )

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, Any]] = None,
    ) -> Any:
        if dtype is None:
            dtype = keras.config.floatx()
        ndim = len(shape)
        axis = self.axis if self.axis >= 0 else ndim + self.axis
        if not 0 <= axis < ndim:
            raise ValueError(f"axis {self.axis} out of range for shape {shape}")

        rng = np.random.default_rng(self.seed)
        w = rng.standard_normal(size=shape).astype("float32")
        norms = np.sqrt(np.sum(np.square(w), axis=axis, keepdims=True))
        directions = w / np.maximum(norms, 1e-12)

        target = self.gain * (np.sqrt(2.0) if self.norm is None else self.norm)
        result = (directions * target).astype("float32")
        return ops.cast(ops.convert_to_tensor(result), dtype)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "norm": self.norm,
            "axis": self.axis,
            "gain": self.gain,
            "seed": self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PolarInitializer":
        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"PolarInitializer(norm={self.norm}, axis={self.axis}, "
            f"gain={self.gain}, seed={self.seed})"
        )

# ---------------------------------------------------------------------
