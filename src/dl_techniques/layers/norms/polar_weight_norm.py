"""Polar weight reparameterization, after PolarQuant (Han et al., 2025).

This module repurposes PolarQuant's recursive Cartesian->polar transform -- the
paper uses it to *quantize* KV-cache vectors -- as a *trainable weight
reparameterization*. A weight vector ``w`` of dimension ``d`` (a power of two) is
represented by a single radius ``r = ||w||`` and ``d - 1`` angles organized into
``log2(d)`` hierarchical levels. The forward map (angles -> Cartesian) is smooth
and differentiable, so the angles can be optimized directly by gradient descent.

This generalizes Weight Normalization (Salimans & Kingma, 2016), which splits a
weight into ``(g = ||w||, v / ||v||)``: here the *direction* is itself given a
full hierarchical angular coordinate system rather than a free unit vector, and
the magnitude (radius) is an explicit, separately-regularizable parameter that
equals the exact per-unit weight norm.

Mathematical Foundation:
    A vector ``x`` of dimension ``d = 2^L`` has a bijective polar representation:
    a single radius ``r = ||x||`` plus ``d - 1`` angles organized into
    ``log2(d)`` hierarchical levels (PolarQuant Definition 1). The encode
    transform is a balanced binary tree that repeatedly pairs adjacent
    coordinates::

        (a, b) -> (atan2(b, a), sqrt(a^2 + b^2))

    collapsing ``d`` magnitudes to ``d / 2`` at each level until a single radius
    remains; the angles emitted along the way form the directional code. The
    decode transform inverts this with the symmetric expansion
    ``(r, psi) -> (r * cos psi, r * sin psi)`` (PolarQuant Algorithm 1). Both
    maps are smooth and backend-agnostic (``keras.ops``), operating on 2-D
    tensors ``(N, d)``.

Properties / Guarantees:
    - Exact per-unit norm. After build and after every optimizer step,
      ``||kernel[:, j]||_2 == |radius[j]|`` (verified to ~1e-7). Magnitude and
      direction are independently controllable -- e.g. apply a different
      regularizer or learning rate to ``radius`` vs ``angles``.
    - Drop-in initialization. ``build()`` samples a seed kernel from
      ``kernel_initializer``, encodes it, and stores the resulting
      ``(radius, angles)``, so a freshly built layer reproduces a standard
      ``Dense`` kernel exactly. Training then moves the polar parameters.
    - Any ``fan_in``. A non-power-of-two ``fan_in`` is internally zero-padded to
      the next power of two; the reconstructed direction is sliced back and
      renormalized, preserving the exact-norm guarantee (cost: up to ~2x
      redundant angle parameters when ``fan_in`` is not already a power of two).
    - Angular prior (optional). An ``angle_regularizer`` that pulls level >= 2
      angles toward ``pi / 4`` imposes a Gaussian-direction prior (PolarQuant
      Lemma 2: higher-level angles of a random Gaussian concentrate at
      ``pi / 4``).

Performance:
    The kernel is reconstructed (cos/sin tree) on *every* forward pass -- an
    ``O(units * d)`` overhead, negligible relative to the matmul for research
    use but not tuned for production inference throughput.

Usage:
    ```python
    import keras
    from dl_techniques.layers.norms import PolarWeightNorm

    inputs = keras.Input(shape=(256,))
    h = PolarWeightNorm(128, activation="relu")(inputs)
    out = PolarWeightNorm(10)(h)
    model = keras.Model(inputs, out)
    ```

    This module exposes ``polar_encode`` / ``polar_decode`` (the differentiable
    transform pair) and the ``PolarWeightNorm`` layer. A matching initializer,
    ``PolarInitializer`` (in ``dl_techniques/initializers/polar_initializer.py``),
    samples weights directly in polar coordinates for exact per-vector-norm
    control; see its own module docstring.

References:
    - PolarQuant: Quantizing KV Caches with Polar Transformation. Han, Kacham,
      Mirrokni, Karbasi, Zandieh. arXiv:2502.02617 (2025). The paper notes the
      transform's principles "extend beyond KV cache compression, offering
      potential applications in LLM weight quantization"; this module realizes a
      training-time variant of that idea.
    - Weight Normalization: A Simple Reparameterization to Accelerate Training of
      Deep Neural Networks. Salimans & Kingma. arXiv:1602.07868 (2016).
    - Tests: tests/test_layers/test_norms/test_polar_weight_norm.py,
      tests/test_initializers/test_polar_initializer.py
"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Recursive Cartesian <-> Polar transform (paper Definition 1 / Algorithm 1)
# ---------------------------------------------------------------------


def _next_power_of_two(n: int) -> int:
    """Smallest power of two >= ``n`` (for ``n >= 1``)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _level_sizes(d: int) -> List[int]:
    """Angle counts per level for dimension ``d``: ``[d/2, d/4, ..., 1]``.

    The list has ``log2(d)`` entries (empty when ``d == 1``) summing to ``d - 1``.
    """
    sizes: List[int] = []
    m = d
    while m > 1:
        m //= 2
        sizes.append(m)
    return sizes


def polar_encode(
    x: keras.KerasTensor,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Cartesian -> polar (paper ``Polar`` procedure, Algorithm 1).

    Args:
        x: 2-D tensor ``(N, d)`` with ``d`` a power of two.

    Returns:
        ``(radius, angles)`` where ``radius`` is ``(N, 1)`` and ``angles`` is
        ``(N, d - 1)`` (the concatenation of levels ``1..log2(d)``, sizes
        ``[d/2, d/4, ..., 1]``). For ``d == 1`` ``angles`` is ``(N, 0)``.

    The transform is bijective: :func:`polar_decode` inverts it exactly (up to
    floating-point error).
    """
    d = x.shape[-1]
    if d is None:
        raise ValueError("polar_encode requires a statically known last dim.")
    if not _is_power_of_two(d):
        raise ValueError(f"Last dim must be a power of two, got d={d}.")

    r = x
    angle_levels: List[keras.KerasTensor] = []
    m = d
    while m > 1:
        # Pair up adjacent coordinates: (..., m) -> (..., m/2, 2).
        pair = ops.reshape(r, (-1, m // 2, 2))
        a = pair[:, :, 0]  # r_{2j-1}
        b = pair[:, :, 1]  # r_{2j}
        angle_levels.append(ops.arctan2(b, a))  # psi in [0, 2pi) (lvl 1) / [0, pi/2]
        r = ops.sqrt(ops.square(a) + ops.square(b))  # new radius vector
        m //= 2

    radius = r  # (N, 1)
    if angle_levels:
        angles = ops.concatenate(angle_levels, axis=-1)  # (N, d-1)
    else:
        angles = radius[:, :0]  # (N, 0) for d == 1
    return radius, angles


def polar_decode(
    radius: keras.KerasTensor,
    angles: keras.KerasTensor,
) -> keras.KerasTensor:
    """Polar -> Cartesian (paper ``DeQuant`` procedure, Algorithm 1).

    Args:
        radius: ``(N, 1)`` top-level radius.
        angles: ``(N, d - 1)`` angle levels as produced by :func:`polar_encode`.

    Returns:
        ``(N, d)`` Cartesian reconstruction.
    """
    a_dim = angles.shape[-1]
    if a_dim is None:
        raise ValueError("polar_decode requires a statically known angle dim.")
    d = a_dim + 1

    # Split the flat angle vector back into levels [d/2, d/4, ..., 1].
    splits: List[keras.KerasTensor] = []
    start = 0
    for s in _level_sizes(d):
        splits.append(angles[:, start:start + s])
        start += s

    r = radius  # (N, 1) == r^(L)
    # Walk levels top-down (smallest level first) interleaving cos/sin children.
    for psi in reversed(splits):
        a = r * ops.cos(psi)
        b = r * ops.sin(psi)
        stacked = ops.stack([a, b], axis=-1)  # (N, m, 2)
        m = psi.shape[-1]
        r = ops.reshape(stacked, (-1, 2 * m))  # (N, 2m)
    return r


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PolarWeightNorm(keras.layers.Layer):
    """Dense layer with a polar-coordinate weight reparameterization.

    **Intent**: Give every output unit's weight vector an explicit, trainable
    magnitude (``radius``) and a hierarchical angular direction (``angles``),
    instead of a free Cartesian kernel. This generalizes Weight Normalization
    (``w = g * v / ||v||``) to a full ``log2(d)``-level angular coordinate
    system derived from PolarQuant's recursive polar transform. The radius is
    the *exact* L2 norm of each reconstructed weight column, and magnitude and
    direction are separate parameters that can be regularized / learning-rate
    scaled independently.

    **Architecture**:
    ```
    radius (units,)   angles (units, d-1)        d = next_pow2(fan_in)
          \\                  |
           \\         polar_decode(1, angles) -> unit dirs (units, d)
            \\                |
             \\        slice [:, :fan_in] + L2-renorm -> unit (units, fan_in)
              \\               |
               +--- * |radius| -> kernel^T (units, fan_in) -> transpose
                                 |
    Input(batch, ..., fan_in) --(matmul)--> (+bias) -> activation -> Output(batch, ..., units)
    ```

    The layer is initialized by sampling a seed kernel from ``kernel_initializer``
    and encoding it, so a freshly built ``PolarWeightNorm`` reproduces a standard
    ``Dense`` kernel exactly; training then moves the radius/angle parameters.

    Args:
        units: Positive integer, output dimensionality.
        activation: Optional activation (name, callable, or None).
        use_bias: Whether to add a bias vector. Defaults to True.
        kernel_initializer: Initializer for the *seed* kernel that is encoded
            into the initial radius/angles. Defaults to ``'glorot_uniform'``.
        bias_initializer: Initializer for the bias. Defaults to ``'zeros'``.
        radius_regularizer: Optional regularizer on the per-unit radius.
        angle_regularizer: Optional regularizer on the angle parameters (e.g. a
            custom regularizer pulling level >=2 angles toward pi/4 imposes a
            Gaussian-direction prior, per PolarQuant Lemma 2).
        bias_regularizer: Optional regularizer on the bias.
        epsilon: Small constant for the slice renormalization. Defaults to 1e-12.
        **kwargs: Passed to ``keras.layers.Layer``.

    Input shape:
        N-D tensor ``(batch, ..., fan_in)``. ``fan_in`` may be any positive
        integer; it is internally zero-padded to the next power of two.

    Output shape:
        N-D tensor ``(batch, ..., units)``.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Any]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, Any] = "glorot_uniform",
        bias_initializer: Union[str, Any] = "zeros",
        radius_regularizer: Optional[Union[str, Any]] = None,
        angle_regularizer: Optional[Union[str, Any]] = None,
        bias_regularizer: Optional[Union[str, Any]] = None,
        epsilon: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._validate_inputs(units, epsilon)

        self.units = int(units)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.radius_regularizer = keras.regularizers.get(radius_regularizer)
        self.angle_regularizer = keras.regularizers.get(angle_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.epsilon = float(epsilon)

        # Weight placeholders (created in build()).
        self.radius = None
        self.angles = None
        self.bias = None
        # Derived geometry (set in build()).
        self._fan_in: Optional[int] = None
        self._padded_dim: Optional[int] = None

        logger.debug(f"Initialized PolarWeightNorm(units={self.units})")

    @staticmethod
    def _validate_inputs(units: int, epsilon: float) -> None:
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if self.built:
            return

        fan_in = input_shape[-1]
        if fan_in is None:
            raise ValueError("The last dimension of the input must be defined.")
        fan_in = int(fan_in)
        d = _next_power_of_two(fan_in)
        self._fan_in = fan_in
        self._padded_dim = d

        # Sample a seed kernel and encode it -> initial (radius, angles), so the
        # layer starts equivalent to a Dense with the same kernel_initializer.
        seed_kernel = ops.convert_to_numpy(
            self.kernel_initializer((fan_in, self.units), dtype="float32")
        )  # (fan_in, units)
        cols = seed_kernel.T.astype("float32")  # (units, fan_in)
        norms = np.linalg.norm(cols, axis=1)  # (units,)
        safe = np.where(norms > 0.0, norms, 1.0)
        dirs = cols / safe[:, None]  # unit directions
        if d > fan_in:
            dirs = np.pad(dirs, ((0, 0), (0, d - fan_in)))
        _, angles0 = polar_encode(ops.convert_to_tensor(dirs.astype("float32")))
        angles0 = ops.convert_to_numpy(angles0)  # (units, d-1)
        radius0 = norms.astype("float32")  # (units,)

        self.radius = self.add_weight(
            name="radius",
            shape=(self.units,),
            initializer=lambda shape, dtype=None: ops.convert_to_tensor(
                radius0, dtype=dtype or "float32"
            ),
            trainable=True,
            regularizer=self.radius_regularizer,
        )
        self.angles = self.add_weight(
            name="angles",
            shape=(self.units, d - 1),
            initializer=lambda shape, dtype=None: ops.convert_to_tensor(
                angles0, dtype=dtype or "float32"
            ),
            trainable=True,
            regularizer=self.angle_regularizer,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
                regularizer=self.bias_regularizer,
            )

        logger.debug(
            f"Built PolarWeightNorm: fan_in={fan_in}, padded_dim={d}, "
            f"angles=(units={self.units}, {d - 1})"
        )
        super().build(input_shape)

    def _reconstruct_kernel(self) -> keras.KerasTensor:
        """Reconstruct the Cartesian kernel ``(fan_in, units)`` from polar params.

        Each column has exact L2 norm ``|radius[j]|``.
        """
        angles = ops.cast(self.angles, "float32")
        radius = ops.cast(self.radius, "float32")
        ones = ops.ones((self.units, 1), dtype="float32")
        full = polar_decode(ones, angles)  # (units, d), ~unit norm over d
        sliced = full[:, : self._fan_in]  # (units, fan_in)
        norm = ops.sqrt(ops.sum(ops.square(sliced), axis=-1, keepdims=True))
        unit = sliced / (norm + self.epsilon)  # exact unit over fan_in
        kernel_t = unit * radius[:, None]  # (units, fan_in), col-norm = |radius|
        return ops.transpose(kernel_t)  # (fan_in, units)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        inputs_fp32 = ops.cast(inputs, "float32")
        kernel = self._reconstruct_kernel()
        outputs = ops.matmul(inputs_fp32, kernel)
        if self.use_bias:
            outputs = ops.add(outputs, ops.cast(self.bias, "float32"))
        if self.activation is not None:
            outputs = self.activation(outputs)
        return ops.cast(outputs, inputs.dtype)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "radius_regularizer": keras.regularizers.serialize(self.radius_regularizer),
            "angle_regularizer": keras.regularizers.serialize(self.angle_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "epsilon": self.epsilon,
        })
        return config


# ---------------------------------------------------------------------
