"""HKAN layer, :class:`HKANLayer`, and the float64 closed-form solve that fits it.

Each output is a linear combination of per-input blocks, and each block is a
weighted sum of ``m`` one-dimensional basis functions of one input. Instead of
backpropagation, both stages are fit by batched ridge or least-squares solves
in float64; the same weights also train under a stock optimizer. Basis names
map to ``KERAS_BASIS`` for ``call`` and ``NUMPY_BASIS`` for the solve. The two
tables must describe the same functions; the numpy forms are overflow-free
because hidden-layer inputs leave ``[0, 1]`` and the slope can reach 50. The
layer builds only under a float32 or float64 compute dtype.

.. code-block:: text

    phi[q, p] = sum_r coef[q, p, r] * g(slope * (x_p - centers[q, p, r]))
                + block_bias[q, p]
    h[q]      = sum_p mix[q, p] * phi[q, p] + bias[q]

``q`` indexes outputs, ``p`` inputs, ``r`` basis functions.

References:
    - Dudek and Rodak, 2025. HKAN: Hierarchical Kolmogorov-Arnold Network
      without Backpropagation. (https://arxiv.org/abs/2501.18199)
"""

import keras
import numbers
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

BASIS_NAMES: Tuple[str, ...] = (
    "sigmoid", "gaussian", "relu", "tanh", "softplus", "identity",
)
CENTER_MODES: Tuple[str, ...] = ("random", "equally_spaced", "data")

KERAS_BASIS: Dict[str, Callable[[Any], Any]] = {
    "sigmoid": lambda z: keras.ops.sigmoid(z),
    "gaussian": lambda z: keras.ops.exp(-keras.ops.square(z)),
    "relu": lambda z: keras.ops.relu(z),
    "tanh": lambda z: keras.ops.tanh(z),
    "softplus": lambda z: keras.ops.softplus(z),
    "identity": lambda z: z,
}

NUMPY_BASIS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": lambda z: np.exp(-np.logaddexp(0.0, -z)),
    "gaussian": lambda z: np.exp(-np.square(z)),
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": lambda z: np.tanh(z),
    "softplus": lambda z: np.logaddexp(0.0, z),
    "identity": lambda z: z,
}

#: Upper bound, in bytes, on the float64 feature tensor of one solve chunk
#: when the caller does not give a chunk size.
AUTO_CHUNK_BYTES: int = 256 * 1024 * 1024

#: Default ``slope`` of :class:`HKANLayer` and ``HKAN``. The reference code
#: uses 1.0, at which a sigmoid or tanh is almost linear over ``[0, 1]``
#: (decisions.md D-039).
DEFAULT_SLOPE: float = 5.0

#: Compute dtypes a layer builds under (the float32 and float64 policies).
SUPPORTED_COMPUTE_DTYPES: Tuple[str, ...] = ("float32", "float64")

#: Range, in float64 epsilons times max ``|y|``, at or under which a target is
#: constant (:func:`is_constant_target`). 4 covers one ulp off a constant at
#: any normal magnitude; 5 epsilons is not constant.
CONSTANT_TARGET_EPSILONS: float = 4.0

# DECISION plan-2026-09-30T082355-4d999dbc/D-035: tag stays non-zero and last.
# numpy drops trailing zero seeds, so a zero tag gives default_rng(seed). See decisions.md.
#: Last entry of every seed list of this package: the bytes of ``"HKAN"``.
SEED_DOMAIN_TAG: int = 0x484B414E

# ---------------------------------------------------------------------


def is_integer(value: Any) -> bool:
    """Tell whether ``value`` is an integer a size or a seed may be given as.

    True for a Python ``int`` and a numpy integer scalar (anything registered
    as ``numbers.Integral``); false for ``bool`` and every float, including an
    integral-valued one. Never raises. Used by the argument checks of
    :class:`HKANLayer` and ``HKAN``, which then store ``int(value)``.

    :param value: The candidate.
    :type value: Any
    :return: Whether it is an integer and not a ``bool``.
    :rtype: bool
    """
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, (bool, np.bool_))
    )


def is_constant_target(y: np.ndarray) -> bool:
    """Tell whether a regression target is constant up to float64 rounding.

    True when the range of ``y`` is at most ``CONSTANT_TARGET_EPSILONS``
    float64 epsilons times its largest magnitude. That includes an exact
    constant and a constant with one entry one ulp off. Used by
    :meth:`HKANLayer.solve_closed_form` and ``HKAN.fit_closed_form``. Never
    raises for a non-empty finite array; an empty one raises from ``np.max``.

    :param y: Target, shape ``(N,)``, N at least 1.
    :type y: np.ndarray
    :return: Whether ``y`` is constant.
    :rtype: bool
    """
    # DECISION plan-2026-09-30T082355-4d999dbc/D-051: constant is relative to max |y|.
    # Not max == min (one ulp breaks it), not a loose tolerance. See decisions.md.
    y = np.asarray(y, dtype=np.float64)
    scale = float(np.max(np.abs(y)))
    return bool(float(np.ptp(y)) <= CONSTANT_TARGET_EPSILONS * np.finfo(np.float64).eps * scale)


def solve_linear_float64(
        features: np.ndarray,
        target: np.ndarray,
        l2: float,
        fit_intercept: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a batch of ridge or least-squares problems onto one target, in float64.

    Every leading axis of ``features`` is a batch axis; the last two are
    ``(N, k)``. All problems share ``target``. With ``fit_intercept`` the
    columns and the target are centered first, so the penalty never touches
    the intercept, as in scikit-learn's ``Ridge`` and ``LinearRegression``.
    A positive ``l2`` solves the normal equations; otherwise each problem
    goes through ``np.linalg.lstsq`` with ``rcond=None``.

    :param features: Design matrices, shape ``(..., N, k)``, dtype float64.
    :type features: np.ndarray
    :param target: Regression target, shape ``(N,)``, dtype float64.
    :type target: np.ndarray
    :param l2: Ridge strength. ``0`` or below selects minimum-norm least
        squares.
    :type l2: float
    :param fit_intercept: Whether to fit an intercept by centering.
    :type fit_intercept: bool
    :return: ``(coef, intercept)`` of shapes ``(..., k)`` and ``(...)``. The
        intercept is all zeros when ``fit_intercept`` is ``False``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises TypeError: If either array is not float64.
    :raises numpy.linalg.LinAlgError: Propagated from the solver.
    """
    # DECISION plan-2026-09-30T082355-4d999dbc/D-005: the solve stays float64 numpy.
    # Do not cast to the model dtype; Gram conditions reach 1e18. See decisions.md.
    if features.dtype != np.float64 or target.dtype != np.float64:
        raise TypeError(
            "the closed-form solve runs in float64 only, got features "
            f"{features.dtype} and target {target.dtype}"
        )

    batch_shape = features.shape[:-2]
    n_rows, n_cols = features.shape[-2:]

    if fit_intercept:
        feature_mean = features.mean(axis=-2, keepdims=True)
        target_mean = target.mean()
        centered = features - feature_mean
        centered_target = target - target_mean
    else:
        centered = features
        centered_target = target

    if l2 > 0:
        transposed = np.swapaxes(centered, -1, -2)
        gram = np.matmul(transposed, centered)
        diagonal = np.arange(n_cols)
        gram[..., diagonal, diagonal] += l2
        rhs = np.matmul(transposed, centered_target)
        coef = np.linalg.solve(gram, rhs[..., None])[..., 0]
    else:
        # DECISION plan-2026-09-30T082355-4d999dbc/D-022: min-norm lstsq, rcond=None.
        # A bare-eps cutoff or the normal equations fail rank-deficient designs. See decisions.md.
        flat = centered.reshape((-1, n_rows, n_cols))
        coef = np.stack([
            np.linalg.lstsq(matrix, centered_target, rcond=None)[0]
            for matrix in flat
        ]).reshape(batch_shape + (n_cols,))

    if fit_intercept:
        intercept = target_mean - np.sum(feature_mean[..., 0, :] * coef, axis=-1)
    else:
        intercept = np.zeros(batch_shape, dtype=np.float64)
    return coef, intercept

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.hkan.hkan_layer")
class HKANLayer(keras.layers.Layer):
    """Map ``n_in`` inputs to ``units`` outputs through per-input basis blocks.

    The layer holds ``units * n_in`` one-dimensional blocks, each a weighted
    sum of ``num_basis`` basis functions of one input, and one linear
    combination of the blocks per output. Set the weights either with
    :meth:`solve_closed_form` followed by :meth:`assign_solution` (the
    paper's training), or with a stock optimizer on ``coef``, ``block_bias``,
    ``mix`` and ``bias``. ``centers`` never trains, and
    :meth:`assign_solution` does not write it.

    The ``identity`` basis is ``g(d) = d`` on the unscaled difference
    ``x_p - center``; it ignores ``slope``, as in the paper. With more than
    one basis function it is degenerate: every feature is ``x_p`` minus a
    constant, so ``num_basis`` only rescales the effective ridge strength.

    ``units``, ``num_basis``, ``seed`` and ``layer_index`` accept numpy
    integers as well as Python ones, reject ``bool`` and floats, and are
    stored as Python ``int``.

    Forward pass:

    .. code-block:: text

        x [B, n_in]
                 │
                 ▼
        ┌─────────────────────┐
        │ x_p - centers       │  centers non-trainable
        └─────────────────────┘
                 │  [B, n_out, n_in, m]
                 ▼
        times slope, then basis g
                 │
                 ▼
        ┌─────────────────────┐
        │ * coef, sum over r  │  + block_bias (if use_block_bias)
        └─────────────────────┘
                 │  phi [B, n_out, n_in]
                 ▼
        ┌─────────────────────┐
        │ * mix, sum over p   │  + bias (if use_bias)
        └─────────────────────┘
                 │
                 ▼
        h [B, n_out]

    identity multiplies by 1.0 instead of slope.

    Closed-form fit, per chunk of ``k`` outputs:

    .. code-block:: text

        x [N, n_in], y [N], centers [k, n_in, m]
                 │
                 ▼
        g(slope * (x_p - centers))
                 │  features [k, n_in, N, m]
                 ▼
        ┌─────────────────────┐
        │ solve per block     │  l2_block, use_block_bias
        └─────────────────────┘
                 │  coef, block_bias
                 ▼
        phi [k, n_in, N] ──► block_r2 [k, n_in]
                 │
                 ▼
        ┌─────────────────────┐
        │ solve per output    │  l2_mix, use_bias
        └─────────────────────┘
                 │  mix, bias
                 ▼
        output [N, k]

    Both solves regress onto the same ``y``.

    :param units: Number of outputs ``n_out``. Must be at least 1.
    :type units: int
    :param num_basis: Number of basis functions ``m`` per block. At least 1.
    :type num_basis: int
    :param basis: Basis function name, one of ``sigmoid``, ``gaussian``,
        ``relu``, ``tanh``, ``softplus``, ``identity``.
    :type basis: str
    :param slope: Multiplier applied to ``x_p - center`` before the basis.
        Must be finite and positive, and not a ``bool``. Ignored by
        ``identity``. The default is ``DEFAULT_SLOPE`` (5.0); the reference
        code's is 1.0.
    :type slope: float
    :param centers: How the centers are set, stored as ``centers_mode``.
        ``random``: uniform on ``[0, 1]``. ``equally_spaced``:
        ``linspace(0, 1, num_basis)`` for every block. ``data``: drawn later
        from the layer's own input by :meth:`sample_data_centers`; until
        then the weight holds a uniform draw.
    :type centers: str
    :param use_block_bias: Whether each block has an intercept.
    :type use_block_bias: bool
    :param use_bias: Whether the connecting stage has an intercept.
    :type use_bias: bool
    :param coef_initializer: Initializer for ``coef``. A gradient-only run
        starts from it; the closed-form solve overwrites it.
    :type coef_initializer: Union[str, keras.initializers.Initializer]
    :param mix_initializer: Initializer for ``mix``. ``None`` selects the
        constant ``1 / n_in``.
    :type mix_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param seed: Seed for the centers, ``None`` or a non-negative integer.
        ``None`` draws from the global numpy generator (which
        ``keras.utils.set_random_seed`` seeds). ``0`` is a seed like any
        other.
    :type seed: Optional[int]
    :param layer_index: Position of this layer in its model, non-negative;
        mixed into the seed so that two layers of one model draw different
        centers.
    :type layer_index: int
    :param **kwargs: Additional arguments for the ``keras.layers.Layer`` base.

    :ivar centers: Basis centers weight, shape ``(n_out, n_in, m)``,
        non-trainable. Not the ``centers`` argument.
    :ivar centers_mode: The ``centers`` argument.
    :ivar coef: Basis coefficients, shape ``(n_out, n_in, m)``.
    :ivar block_bias: Block intercepts, shape ``(n_out, n_in)``; ``None``
        when ``use_block_bias`` is ``False``.
    :ivar mix: Connecting weights, shape ``(n_out, n_in)``.
    :ivar bias: Connecting intercepts, shape ``(n_out,)``; ``None`` when
        ``use_bias`` is ``False``.

    Input shape:
        ``(batch_size, n_in)``.

    Output shape:
        ``(batch_size, units)``.
    """

    def __init__(
            self,
            units: int,
            num_basis: int = 10,
            basis: str = "sigmoid",
            slope: float = DEFAULT_SLOPE,
            centers: str = "random",
            use_block_bias: bool = True,
            use_bias: bool = True,
            coef_initializer: Union[str, keras.initializers.Initializer] = "random_normal",
            mix_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            seed: Optional[int] = None,
            layer_index: int = 0,
            **kwargs: Any,
    ) -> None:
        """Validate and store the arguments.

        Arguments are documented on the class.

        :raises ValueError: If ``units`` or ``num_basis`` is not a positive
            integer, ``basis`` or ``centers`` is not a known name, ``slope``
            is a ``bool``, non-finite or not positive, ``seed`` is neither
            ``None`` nor a non-negative integer, or ``layer_index`` is not a
            non-negative integer.
        """
        super().__init__(**kwargs)

        if not is_integer(units) or units < 1:
            raise ValueError(f"units must be a positive integer, got {units!r}")
        if not is_integer(num_basis) or num_basis < 1:
            raise ValueError(f"num_basis must be a positive integer, got {num_basis!r}")
        if basis not in BASIS_NAMES:
            raise ValueError(f"basis must be one of {BASIS_NAMES}, got {basis!r}")
        if isinstance(slope, bool) or not np.isfinite(slope) or slope <= 0:
            raise ValueError(f"slope must be finite and positive, got {slope!r}")
        if centers not in CENTER_MODES:
            raise ValueError(f"centers must be one of {CENTER_MODES}, got {centers!r}")
        if seed is not None and (not is_integer(seed) or seed < 0):
            raise ValueError(f"seed must be None or a non-negative integer, got {seed!r}")
        if not is_integer(layer_index) or layer_index < 0:
            raise ValueError(
                f"layer_index must be a non-negative integer, got {layer_index!r}"
            )

        self.units = int(units)
        self.num_basis = int(num_basis)
        self.basis = basis
        self.slope = float(slope)
        self.centers_mode = centers
        self.use_block_bias = bool(use_block_bias)
        self.use_bias = bool(use_bias)
        self.coef_initializer = keras.initializers.get(coef_initializer)
        self.mix_initializer = (
            None if mix_initializer is None
            else keras.initializers.get(mix_initializer)
        )
        self.seed = None if seed is None else int(seed)
        self.layer_index = int(layer_index)

        # DECISION plan-2026-09-30T082355-4d999dbc/D-018: identity ignores slope.
        # Scaling it would scale its ridge strength by 1 / slope**2. See decisions.md.
        self._effective_slope = 1.0 if basis == "identity" else self.slope

        self.centers = None
        self.coef = None
        self.block_bias = None
        self.mix = None
        self.bias = None

    # ------------------------------------------------------------------

    def _generator(self, stream: int) -> np.random.Generator:
        """Return the numpy generator of one random stream of this layer.

        :param stream: ``0`` for the build-time centers, ``1`` for the
            data-driven draw.
        :type stream: int
        :return: A generator seeded by ``(seed, layer_index, stream,
            SEED_DOMAIN_TAG)``. When ``seed`` is ``None``, a fresh seed is
            drawn from the global numpy generator on every call and
            ``stream`` has no effect. No seeded stream equals
            ``np.random.default_rng(seed)``.
        :rtype: np.random.Generator
        """
        if self.seed is not None:
            return np.random.default_rng(
                [self.seed, self.layer_index, stream, SEED_DOMAIN_TAG])
        return np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))

    def _initial_centers(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Compute the build-time centers in float64.

        ``equally_spaced`` gives ``linspace(0, 1, m)`` for every block;
        ``random`` and ``data`` both draw uniform on ``[0, 1]`` from stream 0.

        :param shape: ``(n_out, n_in, m)``.
        :type shape: Tuple[int, int, int]
        :return: The centers, shape ``shape``.
        :rtype: np.ndarray
        """
        if self.centers_mode == "equally_spaced":
            return np.broadcast_to(np.linspace(0.0, 1.0, shape[-1]), shape).copy()
        return self._generator(0).uniform(0.0, 1.0, size=shape)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the stacked weights of the layer and fix ``n_in``.

        Sets ``input_spec`` so every later input must have width ``n_in``.
        ``block_bias`` and ``bias`` are created only when their flag is set.

        :param input_shape: ``(batch_size, n_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 2, ``n_in`` is unknown,
            or the compute dtype is not float32 or float64.
        """
        if len(input_shape) != 2 or input_shape[-1] is None:
            raise ValueError(
                f"HKANLayer {self.name!r} needs an input of shape (batch, n_in) "
                f"with a known n_in, got {tuple(input_shape)}"
            )
        # DECISION plan-2026-09-30T082355-4d999dbc/D-037: float16 compute is refused.
        # Under mixed_float16 the forward pass drifts from the fit at slope 50. See decisions.md.
        compute_dtype = keras.backend.standardize_dtype(self.compute_dtype)
        if compute_dtype not in SUPPORTED_COMPUTE_DTYPES:
            raise ValueError(
                f"HKANLayer {self.name!r} does not support the compute dtype "
                f"{compute_dtype!r} (dtype policy {self.dtype_policy.name!r}). "
                "Supported dtype policies: 'float32' and 'float64'."
            )
        n_in = int(input_shape[-1])
        # Without this a (B, 1) input broadcasts against the centers silently.
        self.input_spec = keras.layers.InputSpec(ndim=2, axes={-1: n_in})
        block_shape = (self.units, n_in, self.num_basis)

        # Values come from the initializer: an assign after add_weight is lost when
        # a parent's call() triggers the build, and a tensor made here binds to the trace.
        def centers_initializer(shape: Any, dtype: Any = None) -> Any:
            return keras.ops.convert_to_tensor(
                self._initial_centers(tuple(shape)),
                dtype=dtype or self.variable_dtype,
            )

        self.centers = self.add_weight(
            name="centers", shape=block_shape,
            initializer=centers_initializer, trainable=False,
        )
        self.coef = self.add_weight(
            name="coef", shape=block_shape,
            initializer=clone_initializer(self.coef_initializer), trainable=True,
        )
        if self.use_block_bias:
            self.block_bias = self.add_weight(
                name="block_bias", shape=(self.units, n_in),
                initializer="zeros", trainable=True,
            )
        # DECISION plan-2026-09-30T082355-4d999dbc/D-007: mix starts at 1/n_in, not 0.
        # A zero mix or zero coef is a dead point for gradient training. See decisions.md.
        self.mix = self.add_weight(
            name="mix", shape=(self.units, n_in),
            initializer=(
                keras.initializers.Constant(1.0 / n_in)
                if self.mix_initializer is None
                else clone_initializer(self.mix_initializer)
            ),
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,),
                initializer="zeros", trainable=True,
            )

        super().build(input_shape)

    # ------------------------------------------------------------------

    def block_outputs(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Evaluate every block of the expanding stage.

        ``identity`` uses a slope of 1.0.

        :param inputs: Input tensor of shape ``(batch_size, n_in)``.
        :type inputs: keras.KerasTensor
        :return: ``phi`` of shape ``(batch_size, n_out, n_in)``.
        :rtype: keras.KerasTensor
        """
        difference = (
            keras.ops.expand_dims(keras.ops.expand_dims(inputs, 1), -1)
            - self.centers
        )
        features = KERAS_BASIS[self.basis](self._effective_slope * difference)
        phi = keras.ops.sum(features * self.coef, axis=-1)
        if self.use_block_bias:
            phi = phi + self.block_bias
        return phi

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the expanding stage and the connecting stage.

        :param inputs: Input tensor of shape ``(batch_size, n_in)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; the layer has no training-only behaviour.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, units)``.
        :rtype: keras.KerasTensor
        :raises ValueError: From ``input_spec``, when the input width is not
            the ``n_in`` the layer was built with.
        """
        outputs = keras.ops.sum(self.block_outputs(inputs) * self.mix, axis=-1)
        if self.use_bias:
            outputs = outputs + self.bias
        return outputs

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Return ``(batch_size, units)``.

        :param input_shape: ``(batch_size, n_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], self.units)

    # ------------------------------------------------------------------

    def sample_data_centers(self, x: np.ndarray) -> np.ndarray:
        """Draw data-driven centers from this layer's own input.

        For every block ``(q, p)`` independently, ``num_basis`` values are
        sampled with replacement from column ``p`` of ``x``. With a seed,
        every call draws the same rows (stream 1). Nothing is assigned; the
        caller assigns the result to :attr:`centers`.

        :param x: The layer's input, shape ``(N, n_in)``.
        :type x: np.ndarray
        :return: Centers of shape ``(units, n_in, num_basis)``, dtype of ``x``.
        :rtype: np.ndarray
        :raises ValueError: If ``x`` is not a non-empty rank-2 array.
        """
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] < 1:
            raise ValueError(
                f"HKANLayer {self.name!r} needs a non-empty (N, n_in) array to "
                f"draw centers from, got shape {x.shape}"
            )
        n_rows, n_in = x.shape
        rows = self._generator(1).integers(
            0, n_rows, size=(self.units, n_in, self.num_basis))
        return x[rows, np.arange(n_in)[None, :, None]]

    def solve_chunk_outputs(
            self, n_rows: int, n_in: int, chunk_size: Optional[int] = None,
    ) -> int:
        """Return how many outputs :meth:`solve_closed_form` solves at once.

        The float64 feature tensor of one chunk is
        ``outputs * n_in * n_rows * num_basis * 8`` bytes. A given
        ``chunk_size`` is used as it is, capped at ``units``, with no byte
        check. ``None`` picks the largest count whose tensor fits in
        ``AUTO_CHUNK_BYTES``, at least 1. ``HKAN.fit_closed_form`` sizes its
        forward check from the same budget.

        :param n_rows: Rows ``N`` of the layer input.
        :type n_rows: int
        :param n_in: Width of the layer input.
        :type n_in: int
        :param chunk_size: The caller's chunk size, or ``None``.
        :type chunk_size: Optional[int]
        :return: Outputs per chunk, between 1 and ``units``.
        :rtype: int
        :raises ValueError: If ``chunk_size`` is below 1.
        """
        if chunk_size is None:
            per_output = n_in * n_rows * self.num_basis * 8
            chunk_size = max(1, AUTO_CHUNK_BYTES // per_output)
        elif chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size!r}")
        return int(min(chunk_size, self.units))

    def solve_closed_form(
            self,
            x: np.ndarray,
            y: np.ndarray,
            l2_block: float,
            l2_mix: float,
            centers: np.ndarray,
            chunk_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Fit this layer's weights to ``y`` without gradients, in float64.

        Every block is a ridge regression of its ``m`` basis features onto
        ``y``; then every output is a regression of its ``n_in`` block outputs
        onto ``y``. Outputs are solved in chunks; see the closed-form diagram
        on the class. Nothing is assigned: pass the result to
        :meth:`assign_solution`.

        :param x: Layer input, shape ``(N, n_in)``, dtype float64.
        :type x: np.ndarray
        :param y: Target, shape ``(N,)``, dtype float64.
        :type y: np.ndarray
        :param l2_block: Ridge strength of the block regressions; ``0`` or
            below is minimum-norm least squares.
        :type l2_block: float
        :param l2_mix: Ridge strength of the connecting regressions; ``0`` or
            below is minimum-norm least squares.
        :type l2_mix: float
        :param centers: Centers to fit against, shape ``(units, n_in, m)``,
            dtype float64. Pass the values the ``centers`` weight holds, so
            that the fit and the forward pass use the same numbers.
        :type centers: np.ndarray
        :param chunk_size: Number of outputs solved at once. It bounds memory
            and does not change the result. ``None`` picks the largest chunk
            whose feature tensor fits in ``AUTO_CHUNK_BYTES``.
        :type chunk_size: Optional[int]
        :return: A dict with ``coef`` ``(units, n_in, m)``, ``block_bias``
            ``(units, n_in)``, ``mix`` ``(units, n_in)``, ``bias``
            ``(units,)`` (the two intercept arrays are zeros when their flag
            is off), ``output`` ``(N, units)`` (the float64 layer output on
            ``x``) and ``block_r2`` ``(units, n_in)`` (the coefficient of
            determination of each block's output against ``y``; ``0.0``
            everywhere when ``y`` is constant by :func:`is_constant_target`,
            where scikit-learn's ``r2_score`` gives ``1.0`` for an exact fit).
        :rtype: Dict[str, np.ndarray]
        :raises TypeError: If ``x``, ``y`` or ``centers`` is not float64.
        :raises ValueError: If ``x`` is not rank 2, ``y`` is not ``(N,)``,
            ``centers`` is not ``(units, n_in, m)``, ``chunk_size`` is below
            1, or the solution is not finite.
        :raises numpy.linalg.LinAlgError: Propagated from
            :func:`solve_linear_float64`.
        """
        for label, array in (("x", x), ("y", y), ("centers", centers)):
            if array.dtype != np.float64:
                raise TypeError(
                    f"HKANLayer {self.name!r}: the closed-form solve runs in "
                    f"float64 only, got {label} of dtype {array.dtype}"
                )
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError(
                f"HKANLayer {self.name!r}: expected x (N, n_in) and y (N,), "
                f"got {x.shape} and {y.shape}"
            )
        n_rows, n_in = x.shape
        if centers.shape != (self.units, n_in, self.num_basis):
            raise ValueError(
                f"HKANLayer {self.name!r}: centers must have shape "
                f"{(self.units, n_in, self.num_basis)}, got {centers.shape}"
            )
        chunk_size = self.solve_chunk_outputs(n_rows, n_in, chunk_size)

        basis = NUMPY_BASIS[self.basis]
        columns = x.T[None, :, :, None]
        total = float(np.sum(np.square(y - y.mean())))
        # total alone misreads 96 copies of 0.7 (rounding leaves 1e-30), hence the
        # range test; `total > 0` catches a range whose squares underflow to 0.
        constant_target = is_constant_target(y) or not total > 0

        coef = np.empty((self.units, n_in, self.num_basis), dtype=np.float64)
        block_bias = np.empty((self.units, n_in), dtype=np.float64)
        mix = np.empty((self.units, n_in), dtype=np.float64)
        bias = np.empty((self.units,), dtype=np.float64)
        block_r2 = np.empty((self.units, n_in), dtype=np.float64)
        output = np.empty((n_rows, self.units), dtype=np.float64)

        for start in range(0, self.units, chunk_size):
            stop = min(start + chunk_size, self.units)
            features = basis(
                self._effective_slope * (columns - centers[start:stop, :, None, :])
            )
            coef[start:stop], block_bias[start:stop] = solve_linear_float64(
                features, y, l2_block, self.use_block_bias)
            phi = (
                np.matmul(features, coef[start:stop, :, :, None])[..., 0]
                + block_bias[start:stop, :, None]
            )
            del features

            residual = np.sum(np.square(phi - y), axis=-1)
            # DECISION plan-2026-09-30T082355-4d999dbc/D-036: constant target gives R^2 0.0.
            # r2_score's 1.0 would mark every input as fully important. See decisions.md.
            if constant_target:
                block_r2[start:stop] = 0.0
            else:
                block_r2[start:stop] = 1.0 - residual / total

            stacked = np.swapaxes(phi, -1, -2)
            mix[start:stop], bias[start:stop] = solve_linear_float64(
                stacked, y, l2_mix, self.use_bias)
            output[:, start:stop] = (
                np.matmul(stacked, mix[start:stop, :, None])[..., 0]
                + bias[start:stop, None]
            ).T

        solution = {
            "coef": coef, "block_bias": block_bias, "mix": mix, "bias": bias,
            "output": output, "block_r2": block_r2,
        }
        self._require_finite(solution, ("coef", "block_bias", "mix", "bias", "output"))
        return solution

    def _require_finite(self, arrays: Dict[str, np.ndarray], names: Tuple[str, ...]) -> None:
        """Raise if any named array holds a non-finite value.

        :param arrays: Arrays by name.
        :type arrays: Dict[str, np.ndarray]
        :param names: The names to check.
        :type names: Tuple[str, ...]
        :raises ValueError: Naming this layer and the offending array.
        """
        for name in names:
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(
                    f"HKANLayer {self.name!r}: the closed-form solution has "
                    f"non-finite values in {name!r}; nothing was assigned. "
                    "Raise the ridge strength or reduce the slope."
                )

    def assign_solution(self, solution: Dict[str, np.ndarray]) -> None:
        """Write a solution of :meth:`solve_closed_form` into the weights.

        Writes ``coef`` and ``mix``, plus ``block_bias`` and ``bias`` when
        their flag is set. ``centers``, ``output`` and ``block_r2`` are not
        written. Values are cast to each weight's dtype; if any cast value is
        not finite, nothing is assigned.

        :param solution: The dict returned by :meth:`solve_closed_form`.
        :type solution: Dict[str, np.ndarray]
        :raises ValueError: If the layer is not built, or a cast value is not
            finite.
        :raises KeyError: If ``solution`` lacks a key that is written.
        """
        if not self.built:
            raise ValueError(
                f"HKANLayer {self.name!r} must be built before a solution is assigned"
            )
        targets = {"coef": self.coef, "mix": self.mix}
        if self.use_block_bias:
            targets["block_bias"] = self.block_bias
        if self.use_bias:
            targets["bias"] = self.bias
        with np.errstate(over="ignore"):
            cast = {
                name: np.asarray(solution[name]).astype(weight.dtype)
                for name, weight in targets.items()
            }
        self._require_finite(cast, tuple(cast))
        for name, weight in targets.items():
            weight.assign(cast[name])

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Every constructor argument; ``centers`` holds the mode
            string.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_basis": self.num_basis,
            "basis": self.basis,
            "slope": self.slope,
            "centers": self.centers_mode,
            "use_block_bias": self.use_block_bias,
            "use_bias": self.use_bias,
            "coef_initializer": keras.initializers.serialize(self.coef_initializer),
            "mix_initializer": (
                None if self.mix_initializer is None
                else keras.initializers.serialize(self.mix_initializer)
            ),
            "seed": self.seed,
            "layer_index": self.layer_index,
        })
        return config

# ---------------------------------------------------------------------
