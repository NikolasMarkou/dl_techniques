"""
One HKAN layer: a bank of one-dimensional basis-function regressors (the
expanding stage) followed by one linear combination per output (the connecting
stage), with a float64 closed-form solve for both.

For a layer with ``n_in`` inputs, ``n_out`` outputs and ``m`` basis functions:

.. code-block:: text

    block       phi[q, p](x_p) = sum_r coef[q, p, r] * g(slope * (x_p - centers[q, p, r]))
                                 + block_bias[q, p]
    connecting  h[q]           = sum_p mix[q, p] * phi[q, p] + bias[q]

``q`` indexes outputs, ``p`` inputs, ``r`` basis functions. ``centers`` is fixed
(non-trainable) in every training mode; ``coef``, ``block_bias``, ``mix`` and
``bias`` are ordinary trainable weights, so the same layer trains by the
closed-form solve of :meth:`HKANLayer.solve_closed_form` or by a stock
optimizer.

The basis ``g`` is chosen by name. Two tables map each name to a function: one
in ``keras.ops`` for ``call()`` and one in float64 numpy for the solve. They
must describe the same functions; the numpy table uses overflow-free forms
because hidden-layer inputs are not confined to ``[0, 1]`` and the slope can
be 50.

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

#: Compute dtypes a layer builds under (the float32 and float64 policies).
SUPPORTED_COMPUTE_DTYPES: Tuple[str, ...] = ("float32", "float64")

# DECISION plan-2026-09-30T082355-4d999dbc/D-035
# The tag is NON-ZERO and goes LAST in every seed list of this package. Do
# NOT drop it, set it to 0, or move it to the front: numpy ignores trailing
# zeros of a seed list, so `default_rng([seed, 0, 0])` IS `default_rng(seed)`
# and layer 0's "random" centers were bit for bit the first values a caller
# draws from `default_rng(seed)` (typically the training inputs). See
# decisions.md D-035.
#: Last entry of every seed list of this package: the bytes of ``"HKAN"``.
SEED_DOMAIN_TAG: int = 0x484B414E

# ---------------------------------------------------------------------


def is_integer(value: Any) -> bool:
    """Tell whether ``value`` is an integer a size or a seed may be given as.

    Interface contract (call sites: the argument validation of
    :class:`HKANLayer` and of ``HKAN``): true for a Python ``int`` and for a
    numpy integer scalar (anything registered as ``numbers.Integral``), false
    for ``bool`` and for every float, including an integral-valued one. It
    never raises. The caller stores ``int(value)``, so a config holds plain
    Python integers.

    :param value: The candidate.
    :type value: Any
    :return: Whether it is an integer and not a ``bool``.
    :rtype: bool
    """
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, (bool, np.bool_))
    )


def solve_linear_float64(
        features: np.ndarray,
        target: np.ndarray,
        l2: float,
        fit_intercept: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a batch of ridge or least-squares problems onto one target, in float64.

    Interface contract (call sites: the expanding stage and the connecting
    stage of :meth:`HKANLayer.solve_closed_form`): every leading axis of
    ``features`` is a batch axis; the last two are ``(N, k)``. All problems
    share ``target``. With ``fit_intercept`` the columns and the target are
    centered first, so the penalty never touches the intercept (the semantics
    of scikit-learn's ``Ridge`` and ``LinearRegression``).

    :param features: Design matrices, shape ``(..., N, k)``, dtype float64.
    :type features: np.ndarray
    :param target: Regression target, shape ``(N,)``, dtype float64.
    :type target: np.ndarray
    :param l2: Ridge strength. ``0`` selects minimum-norm least squares.
    :type l2: float
    :param fit_intercept: Whether to fit an intercept by centering.
    :type fit_intercept: bool
    :return: ``(coef, intercept)`` of shapes ``(..., k)`` and ``(...)``. The
        intercept is all zeros when ``fit_intercept`` is ``False``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises TypeError: If either array is not float64.
    :raises numpy.linalg.LinAlgError: Propagated from the solver; nothing is
        returned in that case.
    """
    # DECISION plan-2026-09-30T082355-4d999dbc/D-005
    # The solve is float64 and refuses anything else. Do NOT cast to the
    # model dtype here "for speed" and do NOT move it to keras.ops at the
    # compute dtype: the Gram matrices reach condition numbers of 1e6 to 1e18
    # and a float32 normal-equation solve was measured wrong by up to 8.7 in
    # the predictions. See decisions.md D-005.
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
        # DECISION plan-2026-09-30T082355-4d999dbc/D-022
        # At zero ridge the answer is the MINIMUM-NORM least-squares solution,
        # with the rank cut at eps * max(N, k) times the largest singular
        # value: numpy's rcond=None, and the cutoff scikit-learn's
        # LinearRegression passes to LAPACK (cond = max(X.shape) * eps).
        # Do NOT pass a bare machine epsilon as the cutoff: on a
        # rank-deficient design (an identity basis, whose features are all
        # x_p minus a constant, or a connecting stage whose block outputs are
        # the same fit) the near-null directions sit at 2e-15 to 1e-14 of the
        # largest singular value, survive a cutoff of 2.2e-16, and come back
        # as weights of order 1e13 where the minimum-norm answer is 0.2.
        # Do NOT solve the normal equations here either (np.linalg.solve
        # raises on duplicated centers and is off by 1e-3 in the predictions
        # at cond 1e15). See decisions.md D-022 (supersedes D-019) and D-005.
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
    """
    One HKAN layer: ``n_out * n_in`` one-dimensional blocks, then one linear
    combination per output.

    **Data flow**:

    .. code-block:: text

        x (B, n_in)
           │  slope * (x_p - centers[q, p, r])          (B, n_out, n_in, m)
           ▼
        basis g, weighted by coef, summed over r, + block_bias
           │                                             (B, n_out, n_in)  = phi
           ▼
        weighted by mix, summed over p, + bias
           │
           ▼
        h (B, n_out)

    Two ways to set the weights: :meth:`solve_closed_form` followed by
    :meth:`assign_solution` (the paper's training), or a stock optimizer on
    ``coef``, ``block_bias``, ``mix`` and ``bias``. ``centers`` never trains.

    The ``identity`` basis is ``g(d) = d`` on the unscaled difference
    ``x_p - center``: it ignores ``slope``, as in the paper, where the slope
    is a parameter of the non-linear basis functions only. With more than one
    basis function it is degenerate: every feature is ``x_p`` minus a
    constant, so ``num_basis`` only rescales the effective ridge strength.

    :param units: Number of outputs ``n_out``. Must be at least 1.
    :type units: int
    :param num_basis: Number of basis functions ``m`` per block. At least 1.
    :type num_basis: int
    :param basis: Basis function name, one of ``sigmoid``, ``gaussian``,
        ``relu``, ``tanh``, ``softplus``, ``identity``.
    :type basis: str
    :param slope: Multiplier applied to ``x_p - center`` before the basis.
        Must be finite and positive. Ignored by ``identity``.
    :type slope: float
    :param centers: How the centers are set. ``random``: uniform on
        ``[0, 1]``. ``equally_spaced``: ``linspace(0, 1, num_basis)`` for
        every block. ``data``: drawn later from the layer's own input by
        :meth:`sample_data_centers`; until then the weight holds a uniform
        draw as a placeholder.
    :type centers: str
    :param use_block_bias: Whether each block has an intercept.
    :type use_block_bias: bool
    :param use_bias: Whether the connecting stage has an intercept.
    :type use_bias: bool
    :param coef_initializer: Initializer for ``coef``. It is what a
        gradient-only run starts from; the closed-form solve overwrites it.
    :type coef_initializer: Union[str, keras.initializers.Initializer]
    :param mix_initializer: Initializer for ``mix``. ``None`` selects the
        constant ``1 / n_in``.
    :type mix_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param seed: Seed for the centers. ``None`` draws from the global numpy
        generator (which ``keras.utils.set_random_seed`` seeds). ``0`` is a
        seed like any other.
    :type seed: Optional[int]
    :param layer_index: Position of this layer in its model; mixed into the
        seed so that two layers of one model draw different centers.
    :type layer_index: int
    :param kwargs: Additional arguments for the ``keras.layers.Layer`` base.
    :raises ValueError: If an argument is out of range or a name is unknown.
        ``build`` raises it for a compute dtype other than float32 or
        float64, and ``call`` for an input whose width is not the ``n_in``
        the layer was built with.

    ``units``, ``num_basis``, ``seed`` and ``layer_index`` accept numpy
    integers as well as Python ones and are stored as Python ``int``.

    :ivar centers: Basis centers, shape ``(n_out, n_in, m)``, non-trainable.
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
            slope: float = 1.0,
            centers: str = "random",
            use_block_bias: bool = True,
            use_bias: bool = True,
            coef_initializer: Union[str, keras.initializers.Initializer] = "random_normal",
            mix_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            seed: Optional[int] = None,
            layer_index: int = 0,
            **kwargs: Any,
    ) -> None:
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

        # DECISION plan-2026-09-30T082355-4d999dbc/D-018
        # `identity` is g(d) = d on the UNSCALED difference. Do NOT multiply
        # it by `slope`: a model-wide slope of 50 with an identity top layer
        # (the authors' tutorial configuration) would then scale the top
        # layer's features by 50 and its ridge strength by 1/2500, which is a
        # different model from the paper's. See decisions.md D-018.
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
            SEED_DOMAIN_TAG)``, or by the global numpy generator when ``seed``
            is ``None``. No seeded stream of this package equals
            ``np.random.default_rng(seed)``.
        :rtype: np.random.Generator
        """
        if self.seed is not None:
            return np.random.default_rng(
                [self.seed, self.layer_index, stream, SEED_DOMAIN_TAG])
        return np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))

    def _initial_centers(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Compute the build-time centers in float64.

        :param shape: ``(n_out, n_in, m)``.
        :type shape: Tuple[int, int, int]
        :return: The centers, shape ``shape``.
        :rtype: np.ndarray
        """
        if self.centers_mode == "equally_spaced":
            return np.broadcast_to(np.linspace(0.0, 1.0, shape[-1]), shape).copy()
        return self._generator(0).uniform(0.0, 1.0, size=shape)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the stacked weights of the layer.

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
        # DECISION plan-2026-09-30T082355-4d999dbc/D-037
        # A float16 compute dtype is REFUSED here. Do NOT let it through
        # "because the variables are float32 under mixed_float16": the forward
        # pass then runs in float16, and at slope 50 it was measured up to
        # 2.3e-02 away from the fit's own train predictions on a fit whose
        # RMSE is 0.154 (0.126 on the reviewer's data), with no error. See
        # decisions.md D-037.
        compute_dtype = keras.backend.standardize_dtype(self.compute_dtype)
        if compute_dtype not in SUPPORTED_COMPUTE_DTYPES:
            raise ValueError(
                f"HKANLayer {self.name!r} does not support the compute dtype "
                f"{compute_dtype!r} (dtype policy {self.dtype_policy.name!r}). "
                "Supported dtype policies: 'float32' and 'float64'."
            )
        n_in = int(input_shape[-1])
        # Without this a (B, 1) input broadcasts against the centers of a
        # layer built for more columns and returns a (B, units) answer.
        self.input_spec = keras.layers.InputSpec(ndim=2, axes={-1: n_in})
        block_shape = (self.units, n_in, self.num_basis)

        # The value is produced INSIDE the initializer: an assign after
        # add_weight is discarded when the build is reached through a parent's
        # call(), and a tensor made here would bind to the tracing graph.
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
        # DECISION plan-2026-09-30T082355-4d999dbc/D-007
        # `mix` starts at 1/n_in and `coef` at a random draw. Do NOT
        # initialize both to zeros (or `mix` to zeros "because the solve
        # overwrites it"): d h / d coef carries `mix` and d h / d mix carries
        # `phi`, so a zero start is a dead point for the gradient-trained
        # modes. See decisions.md D-007.
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
        sampled with replacement from column ``p`` of ``x``. Nothing is
        assigned; the caller assigns the result to :attr:`centers`.

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
        onto ``y``. Nothing is assigned: pass the result to
        :meth:`assign_solution`.

        :param x: Layer input, shape ``(N, n_in)``, dtype float64.
        :type x: np.ndarray
        :param y: Target, shape ``(N,)``, dtype float64.
        :type y: np.ndarray
        :param l2_block: Ridge strength of the block regressions; ``0`` is
            minimum-norm least squares.
        :type l2_block: float
        :param l2_mix: Ridge strength of the connecting regressions.
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
            everywhere when ``y`` is constant, where scikit-learn's
            ``r2_score`` gives ``1.0`` for an exact fit).
        :rtype: Dict[str, np.ndarray]
        :raises TypeError: If ``x``, ``y`` or ``centers`` is not float64.
        :raises ValueError: If a shape is wrong, ``chunk_size`` is below 1,
            or the solution is not finite.
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
        if chunk_size is None:
            per_output = n_in * n_rows * self.num_basis * 8
            chunk_size = max(1, AUTO_CHUNK_BYTES // per_output)
        elif chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size!r}")

        basis = NUMPY_BASIS[self.basis]
        columns = x.T[None, :, :, None]
        total = float(np.sum(np.square(y - y.mean())))
        # "Constant" is max == min, exactly. The centered sum of squares
        # alone cannot tell: the mean of 96 copies of 0.7 is not 0.7, that
        # sum is 1e-30, and R^2 computed against it was measured at -2e+30.
        constant_target = not (np.ptp(y) > 0 and total > 0)

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
            # DECISION plan-2026-09-30T082355-4d999dbc/D-036
            # A CONSTANT target has no variance to explain: block R^2 is 0.0.
            # Do NOT return 1.0 for an exact fit there (scikit-learn's
            # `r2_score` convention): the importance of every input would
            # then read 1.0, "every input explains everything", for a target
            # no input explains. See decisions.md D-036.
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

        Values are cast to each weight's dtype. If the cast overflows, nothing
        is assigned.

        :param solution: The dict returned by :meth:`solve_closed_form`.
        :type solution: Dict[str, np.ndarray]
        :raises ValueError: If the layer is not built, or a cast value is not
            finite.
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

        :return: Every constructor argument.
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
