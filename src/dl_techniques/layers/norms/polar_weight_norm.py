"""Polar weight reparameterization, after PolarQuant (Han et al., 2025).

PolarQuant introduced a recursive Cartesian-to-polar transform and used it to
quantize KV-cache vectors. This module reuses that transform as a trainable weight
reparameterization instead. A weight vector ``w`` of dimension ``d``, with ``d`` a
power of two, is stored as one radius ``r = ||w||`` plus ``d - 1`` angles arranged
in ``log2(d)`` hierarchical levels. The map from angles back to Cartesian is smooth
and differentiable, so the angles train by ordinary gradient descent.

This generalizes Weight Normalization (Salimans and Kingma, 2016), which splits a
weight into ``g = ||w||`` and ``v / ||v||``. Here the direction gets a full
hierarchical angular coordinate system rather than a free unit vector. The radius
is an explicit parameter that equals the exact per-unit weight norm, so it can
carry its own regularizer or learning rate.

Contents: :func:`polar_encode`, :func:`polar_decode` and the
:class:`PolarWeightNorm` layer. The layer is NOT registered with
``create_normalization_layer``; import it directly.

.. note::
    **Docstring style.** This file is Sphinx/reST like the rest of
    ``layers/norms/``. It was converted from Google ``Args:`` style in a knowing,
    recorded deviation from the "never convert a file wholesale" rule in
    ``src/dl_techniques/CLAUDE.md``. The reasoning: that rule exists to stop a
    blanket style claim being applied to a mixed tree, and this package is not
    mixed. Measured at the time, ``layers/norms/`` was 15 of 17 files Sphinx, this
    one file Google and zero files carrying both, so this file was a lone outlier
    rather than one sample from a mixed population. The ruling is
    ``plans/plan-2026-08-27T162655-0261908f/decisions.md`` D-002.

Mathematical foundation
-----------------------
A vector ``x`` of dimension ``d = 2^L`` has a bijective polar representation: one
radius ``r = ||x||`` plus ``d - 1`` angles in ``log2(d)`` levels (PolarQuant
Definition 1). Encoding is a balanced binary tree that repeatedly pairs adjacent
coordinates::

    (a, b) -> (atan2(b, a), sqrt(a^2 + b^2))

Each level halves the magnitude count, from ``d`` down to a single radius. The
angles emitted along the way are the directional code. Decoding inverts this with
the symmetric expansion ``(r, psi) -> (r cos psi, r sin psi)`` (PolarQuant
Algorithm 1). Both maps use ``keras.ops`` only, and both operate on 2-D tensors
shaped ``(N, d)``.

Measured round-trip error of ``polar_decode(*polar_encode(x))`` against ``x`` in
float32, on ``keras.random.normal((4, d), seed=0)``: ``1.490e-07`` at ``d = 2``,
``3.576e-07`` at ``d = 8``, ``7.153e-07`` at ``d = 64``. Those are sample maxima
over 4 rows, so they grow with the row count: the same three at
``keras.random.normal((1000, d), seed=0)`` are ``4.768e-07``, ``7.153e-07`` and
``1.192e-06``. The error stays at float32 rounding scale; the exact digits do
not reproduce at another row count.

Properties
----------
**Exact per-unit norm.** After build and after every optimizer step,
``||kernel[:, j]||_2`` equals ``|radius[j]|`` to float32 rounding. Measured
worst deviation ``2.384e-07`` over ``fan_in`` in {5, 7, 8, 16, 32, 64, 128}
crossed with ``units`` in {1, 2, 4, 6, 16, 64, 128}. The worst case moves around
that grid, so treat ``2.384e-07`` as the sweep's maximum, not a proven bound.
Magnitude and direction are separate parameters, so they can take different
regularizers or learning rates.

**Initialization matches Dense.** ``build()`` samples a seed kernel from
``kernel_initializer``, encodes it, and stores the resulting radius and angles. A
freshly built layer therefore reproduces the ``Dense`` kernel that the same
initializer and seed would have produced, to within float32 error. Measured
worst ``max|polar_kernel - dense_kernel| = 1.788e-07`` over ``fan_in`` in
{5, 7, 8, 16, 32, 64} crossed with ``units`` in {4, 6, 16, 64, 128}. It is not
bit-identical; do not assert equality.

**Any fan_in.** A non-power-of-two ``fan_in`` is zero-padded to the next power of
two. The reconstructed direction is sliced back to ``fan_in`` and renormalized,
which preserves the exact-norm property. The cost is up to about twice the angle
parameters. Measured angles per unit against ``fan_in - 1``: ratio ``1.000`` at
``fan_in = 32``, ``1.938`` at ``fan_in = 17``, ``1.969`` at ``fan_in = 33``.

**Angular prior, optional.** An ``angle_regularizer`` that pulls level 2 and
higher angles toward ``pi / 4`` imposes a Gaussian-direction prior. PolarQuant
Lemma 2 says higher-level angles of a random Gaussian concentrate there, and that
reproduces here. These are Monte-Carlo estimates, so the seed is part of the
claim. Measured on ``keras.random.normal((20000, 64), seed=0)``, mean angle per
level against ``pi / 4 = 0.78540``::

    level 1:  mean +0.00166   std 1.81373   range [-3.14158, +3.14158]
    level 2:  mean +0.78474   std 0.34192
    level 3:  mean +0.78555   std 0.24722
    level 4:  mean +0.78619   std 0.17694
    level 5:  mean +0.78693   std 0.12505
    level 6:  mean +0.78568   std 0.08754

Another seed moves the trailing digits: over seeds 0-9 the level-6 std alone
runs ``0.08730`` to ``0.08979``, a spread of about 3%. The concentration toward
``pi / 4``, and the narrowing with each level, are what hold at every draw.

Level 1 is the exception and is why the prior is scoped to level 2 and above. Its
angles come from ``atan2`` on signed coordinates and spread over ``[-pi, pi]``.
From level 2 on, both ``atan2`` arguments are radii and therefore non-negative, so
the angles land in ``[0, pi / 2]`` and tighten with each level.

Performance
-----------
The kernel is rebuilt by a cos/sin tree on EVERY forward pass. That is an
``O(units * d)`` overhead on top of the matmul. It is small for research use and
has not been tuned for production inference throughput.

References
----------
- PolarQuant: Quantizing KV Caches with Polar Transformation. Han, Kacham,
  Mirrokni, Karbasi, Zandieh. arXiv:2502.02617 (2025). The paper notes that the
  transform's principles "extend beyond KV cache compression, offering potential
  applications in LLM weight quantization". This module is a training-time
  variant of that idea.
- Weight Normalization: A Simple Reparameterization to Accelerate Training of
  Deep Neural Networks. Salimans and Kingma. arXiv:1602.07868 (2016).
- Tests: ``tests/test_layers/test_norms/test_polar_weight_norm.py`` and
  ``tests/test_initializers/test_polar_initializer.py``.
"""

import keras
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Recursive Cartesian <-> Polar transform (paper Definition 1 / Algorithm 1)
# ---------------------------------------------------------------------


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``.

    Used by ``PolarWeightNorm.build`` to pick the padded dimension ``d`` from a
    ``fan_in``. Measured: 1, 2, 4, 8, 8, 16, 16, 32 for inputs 1, 2, 3, 5, 8, 9,
    16, 17.

    :param n: A positive integer.
    :type n: int
    :return: The smallest power of two that is at least ``n``.
    :rtype: int
    :raises ValueError: If ``n`` is below 1. Measured: ``n=0`` raises
        ``ValueError: n must be >= 1, got 0``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return 1 << (n - 1).bit_length()


def _is_power_of_two(n: int) -> bool:
    """Report whether ``n`` is a power of two.

    Zero and negative values are ``False``, so this is safe to call on any int.
    ``polar_encode`` uses it to reject a last dimension the binary tree cannot
    pair.

    :param n: The integer to test.
    :type n: int
    :return: ``True`` when ``n >= 1`` and ``n`` has exactly one set bit.
    :rtype: bool
    """
    return n >= 1 and (n & (n - 1)) == 0


def _level_sizes(d: int) -> List[int]:
    """Return the angle count for each level of the tree at dimension ``d``.

    The list is ``[d/2, d/4, ..., 1]``. It has ``log2(d)`` entries and they sum to
    ``d - 1``, which is the total angle count. It is empty for ``d == 1``.
    Measured: ``d=8`` gives ``[4, 2, 1]``, ``d=64`` gives
    ``[32, 16, 8, 4, 2, 1]``.

    ``polar_decode`` uses this to split the flat angle vector back into levels, so
    the split must stay the exact inverse of the order ``polar_encode`` emits.

    :param d: The dimension, expected to be a power of two.
    :type d: int
    :return: Angle counts, largest level first.
    :rtype: List[int]
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
    """Convert Cartesian coordinates to the polar code (paper Algorithm 1, ``Polar``).

    Walks the binary tree bottom-up. At each level it pairs adjacent coordinates,
    emits ``atan2(b, a)`` for every pair, and replaces the pair by its magnitude.
    After ``log2(d)`` levels one radius is left.

    :func:`polar_decode` is the exact inverse, up to float32 error. Measured
    ``max|decode(encode(x)) - x|`` on ``keras.random.normal((4, d), seed=0)``:
    ``1.490e-07`` at ``d = 2``, ``3.576e-07`` at ``d = 8``, ``7.153e-07`` at
    ``d = 64``. Each is a maximum over 4 rows and grows with the row count; see
    the module docstring for the same triple at 1000 rows.

    Angle ranges differ by level, and the difference matters when you write an
    ``angle_regularizer``. Level 1 pairs signed input coordinates, so its angles
    span ``[-pi, pi]``: measured ``[-3.14158, +3.14158]`` on
    ``keras.random.normal((20000, 64), seed=0)``. Every later level pairs
    magnitudes, which are non-negative, so its angles land in ``[0, pi / 2]``:
    measured ``[+0.00294, +1.56986]`` at level 2 on that same seed-0 draw. The
    endpoints are sample extrema and do not reproduce across draws (``seed=7``
    gives ``[+0.00043, +1.56912]``); the ``[0, pi / 2]`` containment does.

    :param x: 2-D tensor ``(N, d)`` whose last dimension is a power of two and is
        statically known.
    :type x: keras.KerasTensor
    :return: ``(radius, angles)``. ``radius`` is ``(N, 1)``. ``angles`` is
        ``(N, d - 1)``, the concatenation of levels 1 to ``log2(d)`` with sizes
        ``[d/2, d/4, ..., 1]``. For ``d == 1`` ``angles`` is ``(N, 0)``.
    :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
    :raises ValueError: If the last dimension is dynamic. Message:
        ``polar_encode requires a statically known last dim.``
    :raises ValueError: If the last dimension is not a power of two. Measured:
        ``d=3`` raises ``Last dim must be a power of two, got d=3.``
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
        # `a` holds the odd-indexed radii r_{2j-1}, `b` the even-indexed r_{2j}.
        pair = keras.ops.reshape(r, (-1, m // 2, 2))
        a = pair[:, :, 0]
        b = pair[:, :, 1]
        # Level 1 sees signed coordinates and emits angles in [-pi, pi]. From
        # level 2 on, `a` and `b` are magnitudes, so the angles fall in [0, pi/2].
        angle_levels.append(keras.ops.arctan2(b, a))
        # The magnitude of each pair becomes the radius vector for the next level.
        r = keras.ops.sqrt(keras.ops.square(a) + keras.ops.square(b))
        m //= 2

    # After the loop `r` has been halved down to a single column: (N, 1).
    radius = r
    if angle_levels:
        # Levels are concatenated largest first, giving (N, d-1).
        angles = keras.ops.concatenate(angle_levels, axis=-1)
    else:
        # d == 1: no pairing happened, so emit an empty (N, 0) angle block by
        # slicing the radius rather than constructing a tensor of another dtype.
        angles = radius[:, :0]
    return radius, angles


def polar_decode(
    radius: keras.KerasTensor,
    angles: keras.KerasTensor,
) -> keras.KerasTensor:
    """Convert the polar code back to Cartesian (paper Algorithm 1, ``DeQuant``).

    Walks the tree top-down, smallest level first. At each level every radius
    splits into a ``(r cos psi, r sin psi)`` pair, doubling the width. After
    ``log2(d)`` levels the result is ``(N, d)``.

    ``d`` is inferred from the angle count as ``angles.shape[-1] + 1``. The
    ``radius`` argument is a free scale: passing ones yields unit-norm directions,
    which is what ``PolarWeightNorm._reconstruct_kernel`` does.

    :param radius: Top-level radius, shaped ``(N, 1)``.
    :type radius: keras.KerasTensor
    :param angles: Angle levels shaped ``(N, d - 1)``, in the order
        :func:`polar_encode` emits them. The last dimension must be statically
        known.
    :type angles: keras.KerasTensor
    :return: The Cartesian reconstruction, shaped ``(N, d)``.
    :rtype: keras.KerasTensor
    :raises ValueError: If the angle dimension is dynamic. Message:
        ``polar_decode requires a statically known angle dim.``
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

    # Start from the single top-level radius, shape (N, 1), the r^(L) of the paper.
    r = radius
    # Walk levels top-down, smallest level first, interleaving cos/sin children.
    for psi in reversed(splits):
        a = r * keras.ops.cos(psi)
        b = r * keras.ops.sin(psi)
        # Stack to (N, m, 2) so that the reshape below interleaves each pair's
        # cos child and sin child into adjacent positions.
        stacked = keras.ops.stack([a, b], axis=-1)
        m = psi.shape[-1]
        # Flatten the pair axis back in: (N, m, 2) becomes (N, 2m).
        r = keras.ops.reshape(stacked, (-1, 2 * m))
    return r


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PolarWeightNorm(keras.layers.Layer):
    """Dense layer whose kernel is stored in polar coordinates.

    Every output unit's weight vector gets an explicit trainable magnitude
    (``radius``) and a hierarchical angular direction (``angles``), instead of a
    free Cartesian kernel. This generalizes Weight Normalization
    (``w = g * v / ||v||``) to a full ``log2(d)``-level angular coordinate system
    taken from PolarQuant's recursive polar transform. The radius is the exact L2
    norm of each reconstructed weight column, so magnitude and direction can carry
    separate regularizers or learning rates.

    The kernel is rebuilt from ``radius`` and ``angles`` on every forward pass.
    There is no stored Cartesian kernel weight.

    **Architecture Overview:**

    .. code-block:: text

                          angles: (units, d - 1)
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ full   = polar_decode(ones((units, 1)), angles)         │
        │          cos/sin tree, log2(d) levels      (units, d)   │
        │ sliced = full[:, :fan_in]              (units, fan_in)  │
        │ unit   = sliced / (||sliced||_2 + epsilon)              │
        └───────────────────────────┬─────────────────────────────┘
                                    │ unit: (units, fan_in)
          radius: (units,) ────────►│
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ kernel_t = unit * radius[:, None]      (units, fan_in)  │
        │            each ROW has L2 norm |radius[j]|             │
        │ kernel   = transpose(kernel_t)         (fan_in, units)  │
        │            so each COLUMN carries |radius[j]|           │
        └───────────────────────────┬─────────────────────────────┘
                                    │ kernel: (fan_in, units)
          x: (batch, ..., fan_in) ─►│
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ out = matmul(cast(x, float32), kernel)                  │
        │ out = out + cast(bias, float32)   (optional; only when  │
        │                                    use_bias is True)    │
        │ out = activation(out)                                   │
        │ out = cast(out, x.dtype)                                │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
                      output: (batch, ..., units)

    ``d`` is ``next_pow2(fan_in)``. The top two boxes are
    ``_reconstruct_kernel``; the bottom box is ``call``. The forward pass runs in
    float32 whatever the compute dtype is, and casts back at the end. Measured
    under ``mixed_float16``: the variables stay float32 and the output comes back
    float16.

    ``build()`` samples a seed kernel from ``kernel_initializer`` and encodes it,
    so a fresh layer reproduces the ``Dense`` kernel that the same initializer and
    seed would give, to float32 rounding: measured worst ``1.788e-07`` over
    ``fan_in`` in {5, 7, 8, 16, 32, 64} crossed with ``units`` in
    {4, 6, 16, 64, 128}. Training then moves the radius and angle parameters.

    .. note::
        ``build()`` materializes the seed kernel on the host with NumPy to compute
        the initial radius and angles. That runs eagerly, which is what Keras
        ``build()`` always does under the Functional and Sequential APIs and under
        ``model.fit``, since build happens before tracing. The forward ``call()``
        is graph-safe. The one unsupported pattern is wrapping an UNBUILT instance
        directly in ``@tf.function``. Build the layer first, by calling it once
        eagerly or by using ``keras.Input``, before tracing.

    :param units: Output dimensionality. Must be a positive ``int``.
    :type units: int
    :param activation: Activation applied after the bias. Name, callable or
        ``None``. Defaults to ``None``, which Keras resolves to the identity.
    :type activation: Optional[Union[str, Any]]
    :param use_bias: Whether to add a bias vector. Defaults to ``True``.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the SEED kernel that is encoded
        into the initial radius and angles. It is not kept as a weight. Defaults
        to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, Any]
    :param bias_initializer: Initializer for the bias. Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, Any]
    :param radius_regularizer: Regularizer on the per-unit radius. Defaults to
        ``None``.
    :type radius_regularizer: Optional[Union[str, Any]]
    :param angle_regularizer: Regularizer on the angle parameters. A regularizer
        pulling level 2 and higher angles toward ``pi / 4`` imposes a
        Gaussian-direction prior; the module docstring has the measured level
        statistics. Defaults to ``None``.
    :type angle_regularizer: Optional[Union[str, Any]]
    :param bias_regularizer: Regularizer on the bias. Defaults to ``None``.
    :type bias_regularizer: Optional[Union[str, Any]]
    :param epsilon: Added to the slice norm before dividing, so a direction whose
        first ``fan_in`` components are all zero cannot divide by zero. Must be
        strictly positive. Defaults to 1e-12.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar units: The configured output width, stored as ``int(units)``.
    :vartype units: int
    :ivar activation: The resolved activation callable. Note that
        ``keras.activations.get(None)`` returns ``keras.activations.linear``, so
        this attribute is never ``None``.
    :vartype activation: Callable
    :ivar use_bias: The configured flag, stored exactly as passed.
    :vartype use_bias: bool
    :ivar epsilon: The configured epsilon, stored as ``float(epsilon)``.
    :vartype epsilon: float
    :ivar radius: Trainable ``(units,)`` per-unit magnitude, or ``None`` until
        ``build()`` runs.
    :vartype radius: Optional[keras.Variable]
    :ivar angles: Trainable ``(units, d - 1)`` angle block, or ``None`` until
        ``build()`` runs.
    :vartype angles: Optional[keras.Variable]
    :ivar bias: Trainable ``(units,)`` bias, or ``None`` when ``use_bias`` is
        ``False`` or ``build()`` has not run.
    :vartype bias: Optional[keras.Variable]

    :raises ValueError: If ``units`` is not a positive ``int``. Measured:
        ``units=0`` raises ``units must be a positive integer, got 0``, and
        ``units=3.0`` raises the same message with ``3.0``, because a ``float`` is
        rejected too. Raised in ``__init__``.
    :raises ValueError: If ``epsilon`` is not strictly positive. Measured:
        ``epsilon=0.0`` raises ``epsilon must be positive, got 0.0``. Raised in
        ``__init__``.
    :raises ValueError: If the last input dimension is dynamic. Measured:
        ``build((None, None))`` raises ``The last dimension of the input must be
        defined.`` Raised in ``build()``.

    Input shape:
        N-D tensor ``(batch, ..., fan_in)``. ``fan_in`` may be any positive
        integer. It is zero-padded internally to the next power of two.

    Output shape:
        N-D tensor ``(batch, ..., units)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import PolarWeightNorm

        x = keras.random.normal((8, 7))
        layer = PolarWeightNorm(4, activation="relu")
        y = layer(x)
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
        """Validate the configuration and store it.

        No weight is created here. ``radius``, ``angles`` and ``bias`` all need
        ``fan_in``, so they are created in ``build()``.

        :param units: Output dimensionality. Must be a positive ``int``.
        :type units: int
        :param activation: Activation applied after the bias.
        :type activation: Optional[Union[str, Any]]
        :param use_bias: Whether to add a bias vector.
        :type use_bias: bool
        :param kernel_initializer: Initializer for the seed kernel that is encoded
            into the initial radius and angles.
        :type kernel_initializer: Union[str, Any]
        :param bias_initializer: Initializer for the bias.
        :type bias_initializer: Union[str, Any]
        :param radius_regularizer: Regularizer on the per-unit radius.
        :type radius_regularizer: Optional[Union[str, Any]]
        :param angle_regularizer: Regularizer on the angle parameters.
        :type angle_regularizer: Optional[Union[str, Any]]
        :param bias_regularizer: Regularizer on the bias.
        :type bias_regularizer: Optional[Union[str, Any]]
        :param epsilon: Added to the slice norm before dividing. Must be strictly
            positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``units`` is not a positive ``int``.
        :raises ValueError: If ``epsilon`` is not strictly positive.
        """
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

        # Weight placeholders. All three are created in build(), which is the
        # first point at which fan_in is known.
        self.radius = None
        self.angles = None
        self.bias = None
        # Derived geometry, also set in build(). _padded_dim is
        # next_power_of_two(_fan_in) and fixes the angle count at _padded_dim - 1.
        self._fan_in: Optional[int] = None
        self._padded_dim: Optional[int] = None

        logger.debug(f"Initialized PolarWeightNorm(units={self.units})")

    @staticmethod
    def _validate_inputs(units: int, epsilon: float) -> None:
        """Reject an invalid ``units`` or ``epsilon`` at construction time.

        The ``units`` check is an ``isinstance(units, int)`` test, so a ``float``
        is rejected even when it is whole. Measured: ``units=3.0`` raises.

        :param units: The candidate output width.
        :type units: int
        :param epsilon: The candidate epsilon.
        :type epsilon: float
        :return: ``None``. Reports failure by raising.
        :rtype: None
        :raises ValueError: If ``units`` is not a positive ``int``.
        :raises ValueError: If ``epsilon`` is not strictly positive.
        """
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create ``radius``, ``angles`` and the optional ``bias``.

        Samples a seed kernel from ``kernel_initializer``, splits it into per-unit
        norms and unit directions, zero-pads the directions to ``_padded_dim`` when
        ``fan_in`` is not a power of two, and encodes them. The resulting radius
        and angle arrays become the initial values of the two trainable weights,
        which is what makes a fresh layer match a ``Dense`` built from the same
        initializer.

        This runs on the host with NumPy and is eager. See the class docstring's
        ``.. note::`` for what that rules out.

        :param input_shape: Shape of the input. Only the last entry, ``fan_in``, is
            read.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``None``.
        :rtype: None
        :raises ValueError: If the last dimension is ``None``.
        """
        if self.built:
            return

        fan_in = input_shape[-1]
        if fan_in is None:
            raise ValueError("The last dimension of the input must be defined.")
        fan_in = int(fan_in)
        d = _next_power_of_two(fan_in)
        self._fan_in = fan_in
        self._padded_dim = d

        # Sample a seed kernel and encode it into the initial (radius, angles), so
        # the layer starts equivalent to a Dense with the same kernel_initializer.
        # seed_kernel is (fan_in, units), matching Dense's kernel layout.
        seed_kernel = keras.ops.convert_to_numpy(
            self.kernel_initializer((fan_in, self.units), dtype="float32")
        )
        # Transpose to (units, fan_in) so each ROW is one output unit's weight
        # vector, which is the layout polar_encode expects.
        cols = seed_kernel.T.astype("float32")
        # norms is (units,), one L2 norm per output unit.
        norms = np.linalg.norm(cols, axis=1)
        # A zero column would divide by zero, so substitute 1.0 for the division
        # only. radius0 below still records the true norm, which stays 0.0.
        safe = np.where(norms > 0.0, norms, 1.0)
        # dirs is (units, fan_in), each row a unit-norm direction.
        dirs = cols / safe[:, None]
        if d > fan_in:
            dirs = np.pad(dirs, ((0, 0), (0, d - fan_in)))
        # The encoded radius is discarded: dirs is already unit norm, so it is 1.
        _, angles0 = polar_encode(keras.ops.convert_to_tensor(dirs.astype("float32")))
        # angles0 is (units, d-1), radius0 is (units,).
        angles0 = keras.ops.convert_to_numpy(angles0)
        radius0 = norms.astype("float32")

        self.radius = self.add_weight(
            name="radius",
            shape=(self.units,),
            initializer=lambda shape, dtype=None: keras.ops.convert_to_tensor(
                radius0, dtype=dtype or "float32"
            ),
            trainable=True,
            regularizer=self.radius_regularizer,
        )
        self.angles = self.add_weight(
            name="angles",
            shape=(self.units, d - 1),
            initializer=lambda shape, dtype=None: keras.ops.convert_to_tensor(
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
        """Rebuild the Cartesian kernel from ``radius`` and ``angles``.

        Decodes the angles at unit radius, slices the padding off, renormalizes
        the slice, then scales each row by its radius. Every column of the result
        has L2 norm ``|radius[j]|`` to float32 rounding; measured worst deviation
        ``2.384e-07`` over ``fan_in`` in {5, 7, 8, 16, 32, 64, 128} crossed with
        ``units`` in {1, 2, 4, 6, 16, 64, 128}.

        Runs on every forward pass. There is no cached kernel. The whole
        computation is done in float32, whatever the compute dtype is.

        :return: The kernel, shaped ``(fan_in, units)``, ready for ``matmul``.
        :rtype: keras.KerasTensor
        """
        angles = keras.ops.cast(self.angles, "float32")
        radius = keras.ops.cast(self.radius, "float32")
        ones = keras.ops.ones((self.units, 1), dtype="float32")
        # Decoding at radius 1 gives (units, d) rows of unit norm over all d.
        full = polar_decode(ones, angles)
        # Drop the zero padding: (units, fan_in). When fan_in < d these rows no
        # longer have unit norm, which is why the renormalization below is needed.
        sliced = full[:, : self._fan_in]
        norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(sliced), axis=-1, keepdims=True))
        # epsilon guards a row whose first fan_in components are all zero.
        unit = sliced / (norm + self.epsilon)
        # (units, fan_in), row L2 norm |radius[j]|.
        kernel_t = unit * radius[:, None]
        # Transpose to the Dense kernel layout (fan_in, units), so the rows above
        # become the columns whose norms the docstring quotes.
        return keras.ops.transpose(kernel_t)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Rebuild the kernel, then apply the dense transform.

        Casts the input to float32, reconstructs the kernel, matmuls, adds the
        bias when ``use_bias`` is ``True``, applies the activation and casts back
        to the input dtype. Measured under ``mixed_float16``: the input reaches
        ``call()`` as float16, so the output is float16 while the variables stay
        float32.

        The layer has no training-mode behaviour. ``training`` is accepted for
        interface compatibility and is not read.

        :param inputs: Input tensor shaped ``(batch, ..., fan_in)``.
        :type inputs: keras.KerasTensor
        :param training: Unused. Present for Keras call-signature compatibility.
        :type training: Optional[bool]
        :return: Output tensor shaped ``(batch, ..., units)``, in the input's
            dtype.
        :rtype: keras.KerasTensor
        """
        inputs_fp32 = keras.ops.cast(inputs, "float32")
        kernel = self._reconstruct_kernel()
        outputs = keras.ops.matmul(inputs_fp32, kernel)
        if self.use_bias:
            outputs = keras.ops.add(outputs, keras.ops.cast(self.bias, "float32"))
        # DECISION plan-2026-08-28T122601-61a91416/D-003: apply the activation
        # UNCONDITIONALLY. Do NOT "restore the safety check" by wrapping this in
        # `if self.activation is not None:` -- that guard's false arm is
        # UNREACHABLE. `__init__` (see the `self.activation` assignment above)
        # resolves it through `keras.activations.get(activation)`, which returns
        # `keras.activations.linear` for `None` -- measured on the pinned Keras
        # 3.8.0, and a true dtype-preserving identity at float16/32/64. The
        # guard therefore documented a `None` case that cannot occur and
        # contradicted this class's own `:ivar activation:` entry, which
        # already states the attribute is never `None`. Deletion measured
        # behaviour-preserving: forward output BIT-identical (uint32 view) at
        # BOTH `activation=None` and `activation='relu'`, before and after, plus
        # a matching `.keras` round-trip and `get_config()` at each.
        outputs = self.activation(outputs)
        return keras.ops.cast(outputs, inputs.dtype)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Replace the last dimension with ``units``.

        Every leading dimension is passed through unchanged, so a ``(None, 6, 7)``
        input with ``units=4`` gives ``(None, 6, 4)``.

        :param input_shape: Shape of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this layer.

        Serializes the activation, both initializers and all three regularizers.
        ``_fan_in`` and ``_padded_dim`` are NOT stored: they are derived from the
        input shape in ``build()``, so a reloaded layer recomputes them. Measured:
        a saved and reloaded model reproduces the original output exactly, max
        absolute difference ``0.0``.

        :return: The config dict, including the base ``keras.layers.Layer`` keys.
        :rtype: Dict[str, Any]
        """
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
