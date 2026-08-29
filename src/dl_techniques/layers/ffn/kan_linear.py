"""
Kolmogorov-Arnold Network (KAN) linear layer.

A more expressive alternative to the standard `Dense` layer, grounded in the
Kolmogorov-Arnold representation theorem. The theorem states that any
multivariate continuous function can be written as a finite composition of
univariate functions and addition. This layer is a practical, learnable
approximation of that statement.

A dense layer computes `y = activation(W @ x + b)`: one fixed activation after
one linear transformation. A KAN layer instead learns a separate univariate
activation `phi_ij` for every connection between input neuron `i` and output
neuron `j`, then sums them: `y_j = Σ_i phi_ij(x_i)`. The learning capacity
moves out of the linear weights and into the activations, which lets the
network fit richer relationships with fewer parameters and fewer layers.

The open question is how to parameterize `phi_ij(x)` so it stays
differentiable and cheap. Here each `phi_ij` is a B-spline: a piecewise
polynomial built as a linear combination of basis splines (`B_k`) over a grid:

`spline_ij(x) = Σ_k c_ijk * B_k(x)`

The coefficients `c_ijk` are the layer's main learnable parameters and set the
shape of the spline. The basis functions `B_k(x)` come from the Cox-de Boor
recursion.

To keep the optimization behaviour of a common activation, each `phi_ij` adds
a fixed base activation `b(x)` (SiLU by default) to the learnable spline, each
term with its own learnable scalar weight:

`phi_ij(x) = w_base_ij * b(x) + w_spline_ij * spline_ij(x)`

The base term carries global trends and the spline term carries local detail,
so the layer can adapt its shape to the data distribution.

References:
    - Liu, Z., Wang, Y., et al. (2024). "KAN: Kolmogorov-Arnold Networks."
      arXiv preprint arXiv:2404.19756.

"""

import keras
from typing import Tuple, Optional, Dict, Any, Union, Callable

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KANLinear(keras.layers.Layer):
    """Kolmogorov-Arnold Network (KAN) linear layer with learnable activations.

    This layer replaces the Dense operation ``activation(x @ W + b)`` with one
    learnable activation ``phi_ij`` per connection ``(i, j)``. Each ``phi_ij``
    is a weighted sum of a fixed base activation and a learnable B-spline:
    ``phi_ij(x) = w_base_ij * b(x) + w_spline_ij * sum_k(c_ijk * B_k(x))``.
    The outputs are summed over inputs, ``y_j = sum_i phi_ij(x_i)``. The basis
    functions ``B_k(x)`` come from the Cox-de Boor recursion on a knot grid
    that can be adapted to the data.

    **Architecture Overview:**

    .. code-block:: text

        Input  [..., input_features]
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
        ┌───────────────────┐    ┌──────────────────────────┐
        │ base path         │    │ spline path              │
        │ b(x_i)            │    │ sum_k c_ijk B_k(x_i)     │
        │ * base_scaler     │    │ * spline_scaler          │
        └─────────┬─────────┘    └────────────┬─────────────┘
                  └─────────────┬─────────────┘
                                ▼
                     phi_ij = base + spline
                                │
                                ▼
                     y_j = sum_i phi_ij(x_i)
                                │
                                ▼
        Output [..., features]

        Both paths run on every call. There is no branch and no
        flag that removes either one.

    **B-spline grid geometry:**

    .. code-block:: text

        x       [..., input_features]
                     │
                     ▼  Cox-de Boor recursion over `grid`
        basis   [..., input_features, n_basis]
                     │
                     ▼  einsum '...ik,iok->...io' with spline_weight
        spline  [..., input_features, features]

        With grid_size = G and spline_order = P:

          n_basis      = G + P
          grid_length  = G + 2*P + 1
          interior     = G + 1 knots spanning grid_range
          padding      = P extra knots at each end

        The defaults G=5, P=3 give n_basis=8 and grid_length=12.

        Weight shapes, with I = input_features, F = features:

          spline_weight  (I, F, n_basis)
          spline_scaler  (I, F)
          base_scaler    (I, F)
          grid           (grid_length,)   non-trainable

        The padding gives the recursion a full knot span at both
        ends. It is generated from the BOUNDARY spacing, not from
        a global step, so a non-uniform interior (the result of
        update_grid_from_samples) still extends consistently.

    :param features: Number of output features. Must be positive.
    :type features: int
    :param grid_size: Number of intervals in the B-spline grid. Higher values
        give the learnable activations finer resolution. Must be positive.
        Defaults to 5.
    :type grid_size: int
    :param spline_order: Degree of the B-spline basis functions. Higher orders
        are smoother (1=linear, 2=quadratic, 3=cubic). Must be >= 0.
        Defaults to 3.
    :type spline_order: int
    :param grid_range: ``(min, max)`` range for the INITIAL knot grid. ``min``
        must be strictly less than ``max``. Defaults to ``(-2.0, 2.0)``. The
        live grid can later be adapted to data by
        ``update_grid_from_samples()``, which rewrites the ``grid`` weight and
        leaves this attribute alone.
    :type grid_range: Tuple[float, float]
    :param activation: Base activation ``b(x)``. Defaults to ``'swish'``, which
        is SiLU.
    :type activation: Union[str, Callable]
    :param base_trainable: Whether ``base_scaler`` is trainable. Defaults to
        True.
    :type base_trainable: bool
    :param spline_trainable: Whether ``spline_scaler`` is trainable. Defaults
        to True.
    :type spline_trainable: bool
    :param kernel_initializer: Initializer for ``spline_weight``, the spline
        coefficients. Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param base_scaler_initializer: Initializer for ``base_scaler``, the
        residual path. Defaults to ``'ones'``, which reproduces the historical
        behaviour of this layer.
    :type base_scaler_initializer: Union[str, keras.initializers.Initializer]
    :param epsilon: Added to every knot-difference denominator in the Cox-de
        Boor recursion. Defaults to 1e-7.
    :type epsilon: float
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar features: The stored output width.
    :vartype features: int
    :ivar grid_size: The stored number of grid intervals.
    :vartype grid_size: int
    :ivar spline_order: The stored spline degree.
    :vartype spline_order: int
    :ivar grid_range: The INITIAL range, as passed. Never updated by
        ``update_grid_from_samples()``.
    :vartype grid_range: Tuple[float, float]
    :ivar base_activation_name: The activation argument exactly as passed.
    :vartype base_activation_name: Union[str, Callable]
    :ivar base_activation_fn: The resolved base activation callable.
    :vartype base_activation_fn: Callable
    :ivar base_trainable: Whether ``base_scaler`` is trainable.
    :vartype base_trainable: bool
    :ivar spline_trainable: Whether ``spline_scaler`` is trainable.
    :vartype spline_trainable: bool
    :ivar kernel_initializer: The resolved ``spline_weight`` initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar base_scaler_initializer: The resolved ``base_scaler`` initializer.
    :vartype base_scaler_initializer: keras.initializers.Initializer
    :ivar epsilon: The stored recursion epsilon.
    :vartype epsilon: float
    :ivar input_features: Last input axis, read in ``build()``. ``None`` before
        that.
    :vartype input_features: Optional[int]
    :ivar spline_weight: Spline coefficients, shape
        ``(input_features, features, grid_size + spline_order)``. Trainable.
    :vartype spline_weight: Optional[keras.Variable]
    :ivar spline_scaler: Per-connection spline weight, shape
        ``(input_features, features)``. Always initialized to ones; the paper
        gives no scheme for it.
    :vartype spline_scaler: Optional[keras.Variable]
    :ivar base_scaler: Per-connection base weight, shape
        ``(input_features, features)``.
    :vartype base_scaler: Optional[keras.Variable]
    :ivar grid: The knot sequence, shape
        ``(grid_size + 2 * spline_order + 1,)``. Non-trainable, held
        ``autocast=False``, and saved with the layer.
    :vartype grid: Optional[keras.Variable]

    :raises ValueError: If ``features`` or ``grid_size`` is not positive, if
        ``spline_order`` is negative, or if ``grid_range[0] >= grid_range[1]``.
    :raises ValueError: From ``build()``, if the input is rank 1 or the last
        input axis is undefined.
    :raises ValueError: From ``update_grid_from_samples()``, if the sample
        tensor is not 2D.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_features)``.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``features``.

    Example:
        .. code-block:: python

            layer = KANLinear(features=8)
            y = layer(keras.random.normal((4, 16)))
            y.shape                 # (4, 8)

    .. note::

        The Rigas et al. (2026) variance-controlled KAN init schemes are wired
        by passing the outputs of
        ``dl_techniques.initializers.create_kan_initializers(...)`` into the two
        weight slots: the ``residual`` initializer into ``base_scaler_initializer``
        and the ``spline`` initializer into ``kernel_initializer``. For example::

            res, spl = create_kan_initializers(grid_size=5, spline_order=3,
                                               scheme='power_law', seed=0)
            layer = KANLinear(features=8, kernel_initializer=spl,
                              base_scaler_initializer=res)

        See :class:`dl_techniques.initializers.KANInitializer` for the schemes.
    """

    def __init__(
            self,
            features: int,
            grid_size: int = 5,
            spline_order: int = 3,
            grid_range: Tuple[float, float] = (-2.0, 2.0),
            activation: Union[str, Callable] = 'swish',
            base_trainable: bool = True,
            spline_trainable: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            base_scaler_initializer: Union[str, keras.initializers.Initializer] = 'ones',
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and resolve the base activation.

        Every argument is documented on the class. No weight is created here:
        every weight shape depends on ``input_features``, which only
        ``build()`` knows, so the weight attributes start as ``None``.

        :raises ValueError: If ``features`` or ``grid_size`` is not positive,
            if ``spline_order`` is negative, or if
            ``grid_range[0] >= grid_range[1]``.
        """
        super().__init__(**kwargs)

        # Reject bad configuration before storing anything.
        if features <= 0:
            raise ValueError("Features must be a positive integer.")
        if grid_size <= 0:
            raise ValueError("Grid size must be a positive integer.")
        if spline_order < 0:
            raise ValueError("Spline order must be a non-negative integer.")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("Invalid grid range: min must be less than max.")

        # Store configuration
        self.features = features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.base_activation_name = activation
        self.base_trainable = base_trainable
        self.spline_trainable = spline_trainable
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.base_scaler_initializer = keras.initializers.get(base_scaler_initializer)
        self.epsilon = epsilon
        self.base_activation_fn = keras.activations.get(activation)

        # Attributes initialized in build()
        self.input_features: Optional[int] = None
        self.grid: Optional[keras.Variable] = None
        self.spline_weight: Optional[keras.Variable] = None
        self.spline_scaler: Optional[keras.Variable] = None
        self.base_scaler: Optional[keras.Variable] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the four weights and the B-spline knot grid.

        :param input_shape: Shape tuple of the input tensor. Must be at least
            2D, with a defined last axis.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` has fewer than 2 axes, or if its
            last axis is ``None``.
        """
        if self.built:
            return

        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        self.input_features = input_shape[-1]
        if self.input_features is None:
            raise ValueError("Input features dimension cannot be None.")

        # Number of B-spline basis functions.
        num_basis_fns = self.grid_size + self.spline_order

        # Spline coefficients (the control points).
        # Shape: (input_features, features, num_basis_fns)
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.input_features, self.features, num_basis_fns),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        # Per-connection spline weight. Always 'ones': the paper gives no
        # init scheme for it.
        # Shape: (input_features, features)
        self.spline_scaler = self.add_weight(
            name="spline_scaler",
            shape=(self.input_features, self.features),
            initializer='ones',
            trainable=self.spline_trainable,
        )

        # DECISION plan_2026-06-12_6cc7c378/D-003: base_scaler, shape
        # (input_features, features). Its default 'ones' initializer keeps this
        # weight byte-identical to the historical behaviour while still letting
        # a residual-path init scheme be wired in. Do NOT hard-code 'ones', and
        # do NOT route it through kernel_initializer, which is the spline path.
        # That plan directory is gone, so this comment is the record.
        self.base_scaler = self.add_weight(
            name="base_scaler",
            shape=(self.input_features, self.features),
            initializer=self.base_scaler_initializer,
            trainable=self.base_trainable,
        )

        # The knot sequence, which sets the shape of the basis functions. It
        # is a non-trainable weight, so it is saved and restored with the layer
        # but is never touched by gradient descent.
        # Size: grid_size + 1 interior knots + 2 * spline_order of padding
        grid_length = self.grid_size + 2 * self.spline_order + 1

        # The grid's INITIAL knots come from the `initializer` below.
        # WHAT NOT TO DO: do NOT restore
        #     self.grid = self.add_weight(..., initializer="zeros")
        #     self._set_grid_from_range(self.grid_range[0], self.grid_range[1])
        # Keras 3's StatelessScope records and DISCARDS an `.assign()` issued
        # during a build reached from a parent layer's call(), which is every
        # real model. Measured on CPU, keras 3.8.0, 2026-08-29: the shipped
        # spelling gives grid[0] = -4.399999618530273 on a direct build, a
        # functional parent and the factory path alike; the rejected spelling
        # gives 0.0 when built from inside a parent layer's call().
        # The RUNTIME writers `_set_grid_from_range` / `update_grid_from_samples`
        # assign from user code in a real scope and are fine; they must keep
        # working. Measured: `_set_grid_from_range(-1, 1)` moves grid[0] to
        # -2.1999998. Same DECISION in nbeats_blocks.py D-028.
        self.grid = self.add_weight(
            name="grid",
            shape=(grid_length,),
            initializer=lambda shape, dtype=None: keras.ops.cast(
                self._compute_grid_values(self.grid_range[0], self.grid_range[1]),
                dtype or self.dtype,
            ),
            trainable=False,
            dtype=self.dtype,
            # DECISION plan-2026-08-19T163559-499b6f0e/D-043: the grid is a
            # coordinate table, not an activation, so it is autocast=False. Do
            # NOT drop that: under mixed_float16 the Cox-de Boor recursion would
            # divide knot differences at half precision, where epsilon's 1e-7
            # default is subnormal (1.192093e-07). See decisions.md D-043.
            autocast=False,
        )

        super().build(input_shape)

    def _compute_grid_values(
        self, start: Union[float, keras.KerasTensor], stop: Union[float, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Compute the extended B-spline knot sequence values.

        :param start: Range minimum (Python float or scalar tensor).
        :type start: Union[float, keras.KerasTensor]
        :param stop: Range maximum (Python float or scalar tensor).
        :type stop: Union[float, keras.KerasTensor]
        :return: Tensor containing the complete knot sequence.
        :rtype: keras.KerasTensor
        """
        # Uniform interior knots across the requested range.
        # Shape: (grid_size + 1,)
        grid_points = keras.ops.linspace(
            start, stop, self.grid_size + 1, dtype=self.dtype
        )
        return self._extend_knots(grid_points)

    def _extend_knots(
        self, grid_points: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Pad an interior knot sequence with ``spline_order`` knots at each end.

        The interior sequence does not have to be uniformly spaced. Both
        extensions are generated from the BOUNDARY spacing (``t_1 - t_0`` on
        the left, ``t_n - t_{n-1}`` on the right), so a non-uniform,
        quantile-matched interior extends consistently at both ends.

        :param grid_points: Monotone interior knots of shape ``(grid_size + 1,)``.
        :type grid_points: keras.KerasTensor
        :return: Full knot sequence of shape ``(grid_size + 2 * spline_order + 1,)``.
        :rtype: keras.KerasTensor
        """
        h_left = grid_points[1] - grid_points[0]
        h_right = grid_points[-1] - grid_points[-2]

        start_indices = keras.ops.arange(-self.spline_order, 0, dtype=self.dtype)
        end_indices = keras.ops.arange(1, self.spline_order + 1, dtype=self.dtype)

        extended_knots_start = start_indices * h_left + grid_points[0]
        extended_knots_end = end_indices * h_right + grid_points[-1]

        return keras.ops.concatenate(
            [extended_knots_start, grid_points, extended_knots_end], axis=0
        )

    def _set_grid_from_range(
        self, start: Union[float, keras.KerasTensor], stop: Union[float, keras.KerasTensor]
    ) -> None:
        """Calculate and assign grid values to the state variable.

        This is the runtime grid-adaptation writer, used by
        ``update_grid_from_samples()``. Do not use it to supply the grid's
        initial value from ``build()``. An ``.assign()`` issued during Keras
        3's symbolic build pass is recorded by the surrounding
        ``StatelessScope`` and then discarded, which leaves every knot at zero.
        The initial knot sequence comes from the ``grid`` weight's
        ``initializer`` instead.

        :param start: Range minimum (Python float or scalar tensor).
        :type start: Union[float, keras.KerasTensor]
        :param stop: Range maximum (Python float or scalar tensor).
        :type stop: Union[float, keras.KerasTensor]
        """
        grid_values = self._compute_grid_values(start, stop)
        self.grid.assign(grid_values)

    def _compute_bspline_basis(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Compute B-spline basis functions using Cox-de Boor recursion formula.

        :param x: Input tensor of shape ``(..., input_features)``.
        :type x: keras.KerasTensor
        :return: Basis function values of shape ``(..., input_features, num_basis_fns)``.
        :rtype: keras.KerasTensor
        """
        # The recursion runs in the VARIABLE dtype, which is float32 under
        # `mixed_float16`, never in the compute dtype. `self.grid` is held
        # `autocast=False` and the input is lifted to match it. The basis is
        # cast back to `compute_dtype` at the layer boundary.
        compute_dtype = self.compute_dtype
        x = keras.ops.cast(x, self.dtype)

        # Add an axis so x broadcasts against the grid.
        # Shape after this: (..., input_features, 1)
        x = keras.ops.expand_dims(x, axis=-1)

        grid = self.grid

        # Base case k=0, the piecewise-constant basis:
        # B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1], else 0.
        # Slicing this way compares against every interval at once.
        grid_left = grid[:-1]
        grid_right = grid[1:]

        basis = keras.ops.cast(
            keras.ops.logical_and(x >= grid_left, x < grid_right),
            dtype=self.dtype,
        )

        # Raise the order one step at a time, per Cox-de Boor.
        for k in range(1, self.spline_order + 1):
            # Grid indices for term 1: t_i to t_{i+k}
            # Denominator: t_{i+k} - t_i
            d1 = grid[k:-1] - grid[:-(k + 1)]
            # Numerator: x - t_i
            n1 = x - grid[:-(k + 1)]

            # Term 1, with the stability epsilon in the denominator.
            term1 = keras.ops.divide(n1, d1 + self.epsilon) * basis[..., :-1]

            # Grid indices for term 2: t_{i+1} to t_{i+k+1}
            # Denominator: t_{i+k+1} - t_{i+1}
            d2 = grid[k + 1:] - grid[1:-k]
            # Numerator: t_{i+k+1} - x
            n2 = grid[k + 1:] - x

            # Term 2.
            term2 = keras.ops.divide(n2, d2 + self.epsilon) * basis[..., 1:]

            # Combine the two terms.
            basis = term1 + term2

        return keras.ops.cast(basis, compute_dtype)

    def call(
            self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass using learned activation functions.

        :param inputs: Input tensor of shape ``(batch_size, ..., input_features)``.
        :type inputs: keras.KerasTensor
        :param training: Unused, present for API consistency.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, ..., features)``.
        :rtype: keras.KerasTensor
        """
        # Path 1: the base activation.
        # Shape: (..., input_features)
        base_val = self.base_activation_fn(inputs)

        # Path 2: the B-spline.
        # Basis shape: (..., input_features, num_basis_fns)
        spline_basis = self._compute_bspline_basis(inputs)

        # Linear combination of the basis functions with the learned
        # coefficients. Tensor contraction:
        # spline_basis:  [..., i, k]
        # spline_weight: [i, o, k]
        # Result:        [..., i, o]
        spline_val = keras.ops.einsum(
            '...ik,iok->...io', spline_basis, self.spline_weight
        )

        # Combine the paths. base_val gains a trailing axis so it broadcasts
        # over the output features, then base_scaler (input_features, features)
        # scales it per connection.
        phi_base = keras.ops.expand_dims(base_val, axis=-1) * self.base_scaler

        # Scale the spline path per connection.
        phi_spline = spline_val * self.spline_scaler

        # phi is now the per-connection activation output.
        phi = phi_base + phi_spline

        # Aggregate over inputs: y_j = sum_i(phi_ij(x_i)).
        # That is the input_features axis, -2.
        output = keras.ops.sum(phi, axis=-2)

        return output

    def update_grid_from_samples(self, x: Union[keras.KerasTensor, Any]) -> None:
        """Adapt the B-spline knot grid to the empirical distribution of ``x``.

        It estimates per-feature quantile boundaries from a data batch,
        averages them across features, and assigns the resulting knot sequence,
        which is generally non-uniform, into the ``grid`` weight. The interior
        quantiles are kept, so this is a real quantile match and not a min/max
        range update. Everything is tensor-based, so it runs eagerly and under
        ``@tf.function``. ``grid_range`` is the configured INITIAL range and is
        not touched here. The ``grid`` weight is the adapted source of truth,
        and it persists across ``.keras`` save and load.

        :param x: Input data tensor of shape ``(batch_size, input_features)``.
        :type x: Union[keras.KerasTensor, Any]
        :raises ValueError: If input is not 2D.
        """
        x = keras.ops.convert_to_tensor(x, dtype=self.dtype)

        if len(keras.ops.shape(x)) != 2:
            raise ValueError("Input 'x' for grid update must be 2D (batch, features).")

        # Sort each feature column to find its distribution boundaries.
        # Shape: (batch_size, input_features)
        x_sorted = keras.ops.sort(x, axis=0)
        batch_size = keras.ops.shape(x)[0]

        # Pick grid_size + 1 evenly spaced ranks, which gives the quantiles.
        indices = keras.ops.cast(
            keras.ops.linspace(0, batch_size - 1, self.grid_size + 1),
            dtype="int32"
        )

        # Gather the values at those ranks.
        # Shape: (grid_size + 1, input_features)
        grid_points_per_feature = keras.ops.take(x_sorted, indices, axis=0)

        # Average across features to find a unified knot sequence for the layer.
        # Averaging a monotone-per-column matrix is monotone, so the result is a
        # valid (generally NON-uniform) knot sequence.
        # Shape: (grid_size + 1,)
        new_grid_points = keras.ops.mean(grid_points_per_feature, axis=1)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-074: keep the interior
        # quantiles. This used to call _set_grid_from_range(new[0], new[-1]),
        # which threw them away and rebuilt a UNIFORM grid between min and max.
        # Do NOT revert to that: on skewed data a uniform grid leaves most knots
        # where there are almost no samples. See decisions.md D-074.
        self.grid.assign(self._extend_knots(new_grid_points))

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape from input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.features
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "features": self.features,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "activation": keras.activations.serialize(self.base_activation_fn),
            "base_trainable": self.base_trainable,
            "spline_trainable": self.spline_trainable,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "base_scaler_initializer": keras.initializers.serialize(
                self.base_scaler_initializer
            ),
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
