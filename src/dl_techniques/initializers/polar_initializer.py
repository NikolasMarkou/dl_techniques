"""Polar-coordinate weight initializer with exact per-vector norm control.

A Gaussian vector's polar representation has a magnitude (radius) that is
statistically *independent* of its direction, and the directional distribution
is exactly the uniform distribution on the unit sphere. ``PolarInitializer``
exploits this to "sample in polar coordinates": it draws a direction uniformly
on the sphere (by normalizing a Gaussian) and sets the radius to a
user-specified value. This decouples norm from direction and gives precise
control over the per-vector norm at initialization -- e.g. every neuron can
start with an identical weight-vector norm ("equinorm init"), which plain
Gaussian/He/Glorot sampling cannot do (their norms are chi-distributed).

Sampling ``radius * gaussian / ||gaussian||`` is mathematically identical to
sampling the analytic level-wise angle p.d.f.s and running the inverse polar
transform, but works for arbitrary (non power-of-two) shapes with no extra
machinery, which is why it is implemented directly.

Which axes form a vector
------------------------
The norm is taken over the FAN-IN block of each output unit, i.e. every axis
except the last. For a ``Dense`` kernel ``(fan_in, units)`` that is axis 0; for
a ``Conv2D`` kernel ``(kh, kw, in_ch, out_ch)`` it is axes ``(0, 1, 2)``
jointly, because He variance is defined over ``kh * kw * in_ch``. Normalizing a
single axis of a conv kernel is NOT He-equivalent: measured on ``(3, 3, 64,
128)`` with ``axis=0``, each output unit accumulated a fan-in energy of 384
instead of 2 (192x) and the per-element std came out 13.9x above He's
``sqrt(2 / 576)``, which compounds to ~2.6e11 over ten such layers. Pass
``axis`` explicitly only when you want something other than the fan-in block.

What this does and does not guarantee
-------------------------------------
It guarantees that every fan-in vector has the same L2 norm. It does NOT give
dynamical isometry: fixing the norms controls only the DIAGONAL of ``W^T W``,
leaving the off-diagonals -- and hence the singular-value spectrum -- essentially
Marchenko-Pastur, the same as Gaussian init. Use
``keras.initializers.Orthogonal`` for a well-conditioned spectrum.

The benefit also shrinks as ``1 / sqrt(2 * fan_in)``: the relative spread of
He-normal column norms measures 17.9% at ``fan_in=16``, 8.9% at 64, 3.1% at 512
and 1.1% at 4096. Equinorm init removes exactly that spread, so it differs
materially from He only for narrow fan-ins -- which is an argument for the conv
case (small ``kh * kw * in_ch``), the one a scalar ``axis`` used to break.

"Exact" means exact to the floating-point resolution of the compute dtype:
measured max deviation from the target norm is 3.6e-07 in float32 and 4.5e-05
after a float16 cast, so under a ``mixed_float16`` policy the guarantee is two
orders of magnitude looser.

Seeding follows the Keras contract exactly: an initializer INSTANCE replays the
same tensor at every matching shape, whether or not you passed a seed (a
seedless instance resolves one integer seed at construction, drawn from the
global RNG state so ``keras.utils.set_random_seed`` controls it). Reusing one
instance across weights that play different architectural roles therefore starts
them bit-identical -- use ``dl_techniques.initializers.clone_initializer`` to get
an independent one.

References:
    - Han et al. (2025). *PolarQuant*. The polar decomposition fact used here
      (Lemma 2): the level-wise ``sin^k`` angle p.d.f.s are precisely those
      induced by a uniform direction. Note that PolarQuant is a KV-cache
      quantization paper -- it supports the decomposition, not this
      initialization scheme.
    - Salimans, T., & Kingma, D. P. (2016). *Weight Normalization: A Simple
      Reparameterization to Accelerate Training of Deep Neural Networks*. The
      closest prior art: it separates a magnitude ``g`` from a direction
      ``v / ||v||`` for exactly this reason.
    - Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). *Exact solutions to
      the nonlinear dynamics of learning in deep linear neural networks*. For
      the orthogonal alternative when the spectrum, not the norms, is what
      matters.
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

#: Expected weight-vector energy of He-normal init, used when ``norm is None``.
#: He gives ``std = sqrt(2 / fan_in)``, so ``E||w||^2 = fan_in * 2 / fan_in = 2``
#: independent of ``fan_in``; the target NORM is therefore ``sqrt(2)``.
HE_EQUIVALENT_NORM: float = float(np.sqrt(2.0))

#: Dtypes the normalization is carried out in directly. Anything narrower is
#: computed in float32 and cast, so the norm is not set in half precision.
_EXACT_COMPUTE_DTYPES = ("float32", "float64")

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.initializers.polar_initializer")
class PolarInitializer(keras.initializers.Initializer):
    """Initialize weights with an exact per-vector L2 norm and uniform direction.

    Each fan-in block of the produced tensor -- by default every axis except the
    last, taken jointly -- is set to have L2 norm exactly ``gain * norm`` (or
    ``gain * sqrt(2)`` when ``norm is None``, matching the expected weight-vector
    energy of He-normal initialization), with a direction drawn uniformly on the
    unit sphere.

    ``norm`` and ``gain`` are multiplicative and only ever appear as their
    product ``gain * norm``. ``gain`` scales the He-equivalent target too, so
    ``gain=2.0, norm=None`` targets a norm of ``2 * sqrt(2)``, i.e. FOUR times
    He's energy, not twice.

    Args:
        norm: Target L2 norm of each fan-in vector. Must be positive. If ``None``
            a He-normal-equivalent norm of ``sqrt(2)`` is used. Defaults to None.
        axis: Axis or axes over which each weight vector is taken. ``None`` (the
            default) means every axis except the last, which is the fan-in block
            of each output unit: axis 0 for a ``Dense`` kernel
            ``(fan_in, units)`` and axes ``(0, 1, 2)`` for a ``Conv2D`` kernel
            ``(kh, kw, in_ch, out_ch)``. An int or a sequence of ints selects the
            axes explicitly.
        gain: Positive multiplicative scale applied to the target norm.
            Defaults to 1.0.
        seed: Optional integer seed. Following the Keras contract, an instance
            replays the same tensor at every matching shape whether or not a seed
            is given; a seedless instance resolves a seed from the global RNG
            state at construction, so ``keras.utils.set_random_seed`` controls
            it. Use ``clone_initializer`` for an independent instance.

    Raises:
        ValueError: If ``norm`` or ``gain`` is not positive, if ``axis`` repeats
            an axis or falls outside the shape's rank, or if the shape is rank-1
            while ``axis`` is left at ``None`` (a rank-1 tensor has no fan-out
            axis, so "every axis except the last" is empty).

    Example:
        >>> init = PolarInitializer(norm=1.0, seed=0)   # unit-norm columns
        >>> w = init((64, 32))                          # each column has ||.||_2 == 1
        >>> conv = PolarInitializer()((3, 3, 64, 128))  # each of the 128 units
        >>> #                                             has fan-in energy 2 (He)
    """

    def __init__(
        self,
        norm: Optional[float] = None,
        axis: Optional[Union[int, Sequence[int]]] = None,
        gain: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if norm is not None and norm <= 0:
            raise ValueError(
                f"norm must be positive (norm=0 yields an all-zero, dead layer "
                f"with no symmetry breaking), got {norm}"
            )
        # Only |gain| ever reaches the result: the direction is already uniform,
        # so a sign flip is a no-op in distribution and a negative gain silently
        # realizes a norm of |gain| * norm.
        if gain <= 0:
            raise ValueError(f"gain must be positive, got {gain}")

        self.norm = None if norm is None else float(norm)
        self.axis = self._coerce_axis(axis)
        self.gain = float(gain)

        # Mirrors keras.initializers.RandomNormal: the CONFIG keeps whatever the
        # caller passed, while the resolved seed drives the draw. np.random is
        # seeded by keras.utils.set_random_seed, so a seedless instance is
        # reproducible under a global seed.
        self._init_seed = seed
        self.seed = seed if seed is not None else int(
            np.random.randint(0, 2 ** 31 - 1)
        )

        logger.debug(
            f"Initialized PolarInitializer(norm={self.norm}, axis={self.axis}, "
            f"gain={self.gain}, seed={self._init_seed})"
        )

    # -----------------------------------------------------------------
    # axis handling
    # -----------------------------------------------------------------

    @staticmethod
    def _coerce_axis(
        axis: Optional[Union[int, Sequence[int]]],
    ) -> Optional[Tuple[int, ...]]:
        """Coerce the ``axis`` argument to ``None`` or a tuple of ints.

        Args:
            axis: ``None``, an int, or a sequence of ints.

        Returns:
            ``None`` (meaning "every axis except the last") or a tuple of ints.

        Raises:
            ValueError: If a sequence is empty or repeats an axis.
        """
        if axis is None:
            return None
        if isinstance(axis, (int, np.integer)):
            return (int(axis),)

        coerced = tuple(int(a) for a in axis)
        if not coerced:
            raise ValueError("axis must not be an empty sequence")
        if len(set(coerced)) != len(coerced):
            raise ValueError(f"axis must not repeat an axis, got {coerced}")
        return coerced

    def _resolve_axes(self, shape: Sequence[int]) -> Tuple[int, ...]:
        """Resolve ``self.axis`` against a concrete shape.

        Args:
            shape: The kernel shape being initialized.

        Returns:
            The normalized, non-negative axes to reduce over.

        Raises:
            ValueError: If the shape is rank-1 under the default axis, or an axis
                is out of range or duplicated after wrapping.
        """
        ndim = len(shape)

        if self.axis is None:
            if ndim < 2:
                raise ValueError(
                    f"the default axis=None means 'every axis except the last' "
                    f"(the fan-in block of each output unit), which is empty for "
                    f"the rank-{ndim} shape {tuple(shape)}. Pass axis=0 "
                    f"explicitly to give a rank-1 tensor a single target norm."
                )
            return tuple(range(ndim - 1))

        resolved = tuple(a if a >= 0 else ndim + a for a in self.axis)
        for original, wrapped in zip(self.axis, resolved):
            if not 0 <= wrapped < ndim:
                raise ValueError(
                    f"axis {original} out of range for shape {tuple(shape)}"
                )
        if len(set(resolved)) != len(resolved):
            raise ValueError(
                f"axis {self.axis} resolves to duplicate axes {resolved} for "
                f"shape {tuple(shape)}"
            )
        return resolved

    # -----------------------------------------------------------------
    # call
    # -----------------------------------------------------------------

    def __call__(
        self,
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Draw a tensor whose every fan-in vector has the target norm.

        Args:
            shape: Shape of the tensor to initialize.
            dtype: Data type of the returned tensor. ``None`` falls back to
                ``keras.config.floatx()``.
            **kwargs: Additional arguments (unused).

        Returns:
            Tensor: The initialized tensor.

        Raises:
            ValueError: See :meth:`_resolve_axes`.
        """
        if dtype is None:
            dtype = keras.config.floatx()
        dtype = keras.backend.standardize_dtype(dtype)

        axes = self._resolve_axes(shape)

        # Set the norm in at least float32: doing it in float16 would leave the
        # "exact" guarantee at ~4.5e-05 instead of ~3.6e-07.
        compute_dtype = dtype if dtype in _EXACT_COMPUTE_DTYPES else "float32"

        # keras.random.normal (not np.random) so that the draw honours the Keras
        # global seed; the previous np.random.default_rng(None) bypassed
        # keras.utils.set_random_seed entirely. It also avoids materializing a
        # float64 numpy buffer twice the size of the result.
        w = keras.random.normal(
            shape=shape, mean=0.0, stddev=1.0, dtype=compute_dtype, seed=self.seed
        )

        norms = keras.ops.sqrt(
            keras.ops.sum(keras.ops.square(w), axis=axes, keepdims=True)
        )
        target = self.gain * (
            HE_EQUIVALENT_NORM if self.norm is None else self.norm
        )

        # A zero draw is a measure-zero event for a Gaussian; the clamp keeps it
        # finite, at the cost of that one vector realizing a norm of ~0.
        scale = target / keras.ops.maximum(norms, keras.backend.epsilon())
        result = w * keras.ops.cast(scale, w.dtype)

        return keras.ops.cast(result, dtype) if compute_dtype != dtype else result

    # -----------------------------------------------------------------
    # serialization
    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization.

        The ``seed`` written here is the one the caller passed, not the resolved
        draw seed, so a seedless initializer stays seedless across a round trip
        (the Keras convention).

        Returns:
            Dict containing the initializer configuration.
        """
        config = super().get_config()
        config.update({
            "norm": self.norm,
            "axis": self.axis,
            "gain": self.gain,
            "seed": self._init_seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PolarInitializer":
        """Create an initializer from configuration.

        Args:
            config: Configuration dictionary. ``axis`` may be an int (as written
                by earlier versions), a list, or ``None``.

        Returns:
            PolarInitializer instance.
        """
        return cls(**config)

    def __repr__(self) -> str:
        return (
            f"PolarInitializer(norm={self.norm}, axis={self.axis}, "
            f"gain={self.gain}, seed={self._init_seed})"
        )

# ---------------------------------------------------------------------
