"""
HKAN is a Kolmogorov-Arnold style regressor over ``(B, n_in)`` features whose
weights can be fitted layer by layer by least squares, with no gradient, and
whose same weights can also be trained by a stock optimizer.

A Kolmogorov-Arnold network replaces the fixed activation of an MLP by a
learned one-dimensional function on every edge. Training those functions by
backpropagation is slow and non-convex. HKAN removes the non-convex part by
fixing the non-linear parameters: every edge function is
``phi[q, p](x_p) = sum_r coef[q, p, r] * g(slope * (x_p - centers[q, p, r]))``
with the centers drawn once at random (uniformly, equally spaced, or from the
data) and never trained. What is left is linear in ``coef``, so each edge
function is one ridge regression of its basis features onto the target ``y``.
A layer then combines its edge functions per output,
``h[q] = sum_p mix[q, p] * phi[q, p]``, which is again one linear regression
onto ``y``. Layers are stacked: the outputs ``h`` of one layer are the inputs
of the next, and every stage of every layer regresses onto the SAME ``y``.
Depth therefore refines an approximation of ``y`` greedily; no stage is ever
optimized jointly with another.

The model is a list of :class:`HKANLayer`, one per paper layer, the last of
width 1. The output is ``(B, 1)``; a single target only, because every block
regresses onto one ``y``.

Deliberate choices, each with its reason:

- The closed-form training is the method :meth:`HKAN.fit_closed_form`, not an
  override of ``fit`` or ``train_step``. ``keras.Model.fit`` stays the stock
  gradient loop, so the two procedures can be used alone or one after the
  other (closed form first, then gradient fine-tuning of all stages jointly).
- The solve runs in float64 numpy whatever the model dtype, because the Gram
  matrices are badly conditioned (1e6 to 1e18 measured). The result is cast
  into the Keras weights. How far the model's forward pass then is from the
  float64 fit depends on the size of the fitted weights: about the largest
  weight times the resolution of the model dtype. Measured in float32: 6e-7
  at most on the authors' tutorial configuration (weights up to 13), but
  3.2e-03 RMS against a fit of RMSE 3.4e-08 with the hyperparameters of the
  paper's Table V row for TF1 on a smooth two-input probe target, not on TF1
  (connecting weights up to 6e3), and 2.6e+02 against 8.5e-02 at the
  reference code's defaults (slope 1, no ridge, coefficients up to 5e9).
  This constructor's defaults (``slope=5.0``, ``l2_block=0.01``,
  ``l2_mix=0.01``) are chosen so that the default model stays under 1e-3 of
  the target's standard deviation; the paper's and the reference code's
  behaviour is ``slope=1.0, l2_block=0.0, l2_mix=0.0``, passed explicitly.
  :meth:`HKAN.fit_closed_form` therefore runs the model once on the training
  rows, reports the gap and logs a warning when it is large.
- Intercepts are on by default in both stages (``use_block_bias``,
  ``use_bias``). The paper's equations have none; the authors' reference
  configuration fits them. Both forms are reachable.
- A hidden layer of width 1 is allowed.
- The centers are seeded (``seed``; ``0`` is a seed). With ``centers="data"``
  they are drawn inside :meth:`HKAN.fit_closed_form`, or by
  :meth:`HKAN.initialize_centers` for a gradient-only run; until one of the
  two runs the weights hold a uniform placeholder.
- The centers and the slope are not trainable in any mode.
- There is no variants table and no ``pretrained`` argument: the paper
  defines no named sizes and no weights are distributed.
- Supported dtype policies are float32 and float64, set globally or by the
  constructor's ``dtype`` argument, which every layer receives. Any other
  compute dtype (``mixed_float16``, ``float16``, ``bfloat16`` and
  ``mixed_bfloat16`` included) raises ``ValueError`` at build.

References:
    - Dudek and Rodak, 2025. HKAN: Hierarchical Kolmogorov-Arnold Network
      without Backpropagation. (https://arxiv.org/abs/2501.18199)
    - Liu et al., 2024. KAN: Kolmogorov-Arnold Networks.
      (https://arxiv.org/abs/2404.19756)
"""

import keras
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

from .hkan_layer import DEFAULT_SLOPE, HKANLayer, is_constant_target, is_integer

# ---------------------------------------------------------------------

# DECISION plan-2026-09-30T082355-4d999dbc/D-039
# The defaults of `slope`, `l2_block` and `l2_mix` are 5.0, 0.01 and 0.01. Do
# NOT restore the reference code's defaults (slope 1, no ridge on either
# stage) as DEFAULTS "for fidelity to the paper": with them the float32 model
# does not compute the fit it reports (fit RMSE 8.5e-02, model RMSE 2.6e+02
# with no hidden layer), and a block ridge alone does not repair it once there
# is a hidden layer, because the unpenalized connecting weights reach 2.5e+04
# at width 64. Do NOT drop the `l2_mix` default to 0 on its own for the same
# reason. The reference behaviour stays available by passing the three values
# explicitly. See decisions.md D-039 and D-034.
#: Default ridge strength of the block regressions.
DEFAULT_L2_BLOCK: float = 0.01
#: Default ridge strength of the connecting regressions.
DEFAULT_L2_MIX: float = 0.01

#: Rows per forward batch when :meth:`HKAN.initialize_centers` propagates
#: activations; the feature tensor of a layer is ``(B, n_out, n_in, m)``. The
#: forward check of :meth:`HKAN.fit_closed_form` sizes its batch from the
#: solve's byte budget instead (:meth:`HKAN._forward_batch_rows`).
_PROPAGATION_BATCH: int = 256

#: :meth:`HKAN.fit_closed_form` warns when the RMS difference between the
#: model's own forward pass and the float64 fit exceeds this fraction of the
#: target's standard deviation. A float32 forward pass of a well-scaled fit
#: sits near 1e-7 of it; 1e-3 is a gap a user would see in a reported RMSE.
FORWARD_DEVIATION_FRACTION: float = 1e-3

#: The limit used instead when the target is constant (``is_constant_target``):
#: an absolute RMS difference, about ten float32 steps at unit scale.
FORWARD_DEVIATION_FLOOR: float = 1e-6

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.hkan.model")
class HKAN(keras.Model):
    """
    Hierarchical Kolmogorov-Arnold Network for single-target regression.

    **Architecture**:

    .. code-block:: text

        x (B, n_in)
           │
           ▼
        HKANLayer 0   n_in            -> hidden_units[0]
           ▼
        HKANLayer 1   hidden_units[0] -> hidden_units[1]
           ▼
          ...
           ▼
        HKANLayer L   hidden_units[-1] -> 1
           │
           ▼
        y_hat (B, 1)

    Every per-layer setting (``num_basis``, ``basis``, ``slope``, ``centers``,
    ``l2_block``, ``l2_mix``) takes either one value for all layers or a
    sequence with one value per layer, ``len(hidden_units) + 1`` long.

    :param hidden_units: Widths of the hidden layers. Empty for a one-layer
        model. The output layer of width 1 is added automatically.
    :type hidden_units: Sequence[int]
    :param num_basis: Basis functions per block.
    :type num_basis: Union[int, Sequence[int]]
    :param basis: Basis name: ``sigmoid``, ``gaussian``, ``relu``, ``tanh``,
        ``softplus`` or ``identity``.
    :type basis: Union[str, Sequence[str]]
    :param slope: Multiplier of ``x_p - center`` inside the basis. Ignored by
        ``identity``. Default 5.0; the reference code's default is 1.0.
    :type slope: Union[float, Sequence[float]]
    :param centers: ``random``, ``equally_spaced`` or ``data``.
    :type centers: Union[str, Sequence[str]]
    :param l2_block: Ridge strength of the block regressions in
        :meth:`fit_closed_form`. Default 0.01. ``0``, the reference code's
        default, is minimum-norm least squares, which on a smooth basis
        returns coefficients a float32 model cannot carry
        (:meth:`fit_closed_form` warns when that happens). It has no effect
        on gradient training.
    :type l2_block: Union[float, Sequence[float]]
    :param l2_mix: Ridge strength of the connecting regressions in
        :meth:`fit_closed_form`. Default 0.01. The paper fits the connecting
        stage by plain least squares (``0``); on a wide layer the block
        outputs are nearly collinear and ``0`` gave connecting weights of
        4e2 to 5e8 in the measured cases. It has no effect on gradient
        training.
    :type l2_mix: Union[float, Sequence[float]]
    :param use_block_bias: Whether every block has an intercept.
    :type use_block_bias: bool
    :param use_bias: Whether every connecting stage has an intercept.
    :type use_bias: bool
    :param coef_initializer: Initializer of the block coefficients, the
        starting point of a gradient-only run.
    :type coef_initializer: Union[str, keras.initializers.Initializer]
    :param mix_initializer: Initializer of the connecting weights. ``None``
        selects the constant ``1 / n_in`` of each layer.
    :type mix_initializer: Optional[Union[str, keras.initializers.Initializer]]
    :param seed: Seed of the centers. ``None`` draws from the global numpy
        generator. ``0`` is a seed.
    :type seed: Optional[int]
    :param kwargs: Additional arguments for the ``keras.Model`` base. A
        ``dtype`` given here is the dtype policy of every layer too.
    :raises ValueError: If a width is not a positive integer, a per-layer
        sequence has the wrong length, a ridge strength is negative, or a
        layer rejects its setting. Building under a dtype policy other than
        float32 or float64 (global or given as ``dtype``) raises it too, and
        so does calling the model on an input of another width than it was
        built for.

    Integer arguments (``hidden_units`` entries, ``num_basis``, ``seed``) may
    be numpy integers; ``get_config`` returns plain Python numbers.

    :ivar hkan_layers: The layers, in order.
    :vartype hkan_layers: List[HKANLayer]

    Input shape:
        ``(batch_size, n_in)``.

    Output shape:
        ``(batch_size, 1)``.

    Example:
        >>> model = HKAN(hidden_units=(64,), num_basis=(23, 10),
        ...              basis=("tanh", "identity"), slope=50.0,
        ...              centers=("random", "data"), l2_block=(0.01, 0.1),
        ...              l2_mix=0.0, seed=0)  # the authors' tutorial configuration
        >>> diagnostics = model.fit_closed_form(x_train, y_train)
        >>> y_hat = model.predict(x_test)
        >>> # optional joint fine-tuning of all stages by gradient descent
        >>> model.compile(optimizer="adam", loss="mse")
        >>> model.fit(x_train, y_train, epochs=10)
    """

    def __init__(
            self,
            hidden_units: Sequence[int] = (),
            num_basis: Union[int, Sequence[int]] = 10,
            basis: Union[str, Sequence[str]] = "sigmoid",
            slope: Union[float, Sequence[float]] = DEFAULT_SLOPE,
            centers: Union[str, Sequence[str]] = "random",
            l2_block: Union[float, Sequence[float]] = DEFAULT_L2_BLOCK,
            l2_mix: Union[float, Sequence[float]] = DEFAULT_L2_MIX,
            use_block_bias: bool = True,
            use_bias: bool = True,
            coef_initializer: Union[str, keras.initializers.Initializer] = "random_normal",
            mix_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
            seed: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(hidden_units, (str, bytes)) or not isinstance(
                hidden_units, (list, tuple)):
            raise ValueError(
                f"hidden_units must be a list or tuple of integers, got {hidden_units!r}"
            )
        for width in hidden_units:
            if not is_integer(width) or width < 1:
                raise ValueError(
                    f"every entry of hidden_units must be a positive integer, "
                    f"got {hidden_units!r}"
                )
        self.hidden_units = [int(width) for width in hidden_units]
        self._units = self.hidden_units + [1]

        self.num_basis = num_basis
        self.basis = basis
        self.slope = slope
        self.centers = centers
        self.l2_block = l2_block
        self.l2_mix = l2_mix
        self.use_block_bias = bool(use_block_bias)
        self.use_bias = bool(use_bias)
        self.coef_initializer = keras.initializers.get(coef_initializer)
        self.mix_initializer = (
            None if mix_initializer is None
            else keras.initializers.get(mix_initializer)
        )
        self.seed = seed

        self._num_basis = self._per_layer("num_basis", num_basis)
        self._basis = self._per_layer("basis", basis)
        self._slope = self._per_layer("slope", slope)
        self._centers = self._per_layer("centers", centers)
        self._l2_block = [float(v) for v in self._per_layer("l2_block", l2_block)]
        self._l2_mix = [float(v) for v in self._per_layer("l2_mix", l2_mix)]
        for label, values in (("l2_block", self._l2_block), ("l2_mix", self._l2_mix)):
            for value in values:
                if not np.isfinite(value) or value < 0:
                    raise ValueError(
                        f"{label} must be finite and non-negative, got {value!r}"
                    )

        # A keras.Model-owned flat list keeps weight tracking through a
        # .keras save.
        # DECISION plan-2026-09-30T082355-4d999dbc/D-049
        # Every layer gets the MODEL's dtype policy. Do NOT drop `dtype=` below
        # "because the global policy already reaches the layers": a policy
        # given to the constructor (`HKAN(dtype="float64")`, the standard Keras
        # way) then never reached them. Measured: `dtype="float64"` left
        # float32 layers and the same 3.6e+07 forward deviation as float32,
        # and `dtype="mixed_float16"` built, fitted and predicted with no
        # error, inputs rounded to float16, past the refusal of D-037. See
        # decisions.md D-049.
        self.hkan_layers: List[HKANLayer] = [
            HKANLayer(
                units=self._units[index],
                num_basis=self._num_basis[index],
                basis=self._basis[index],
                slope=self._slope[index],
                centers=self._centers[index],
                use_block_bias=self.use_block_bias,
                use_bias=self.use_bias,
                coef_initializer=self.coef_initializer,
                mix_initializer=self.mix_initializer,
                seed=seed,
                layer_index=index,
                dtype=self.dtype_policy,
                name=f"hkan_layer_{index}",
            )
            for index in range(len(self._units))
        ]

        logger.info(
            f"Created HKAN: units {self._units}, num_basis {self._num_basis}, "
            f"basis {self._basis}, slope {self._slope}, centers {self._centers}, "
            f"seed {seed}"
        )

    def _per_layer(self, label: str, value: Any) -> List[Any]:
        """Expand a scalar-or-sequence setting to one value per layer.

        :param label: The argument name, for the error message.
        :type label: str
        :param value: One value, or a list or tuple with one value per layer.
        :type value: Any
        :return: A list of ``len(hidden_units) + 1`` values.
        :rtype: List[Any]
        :raises ValueError: If a sequence has the wrong length.
        """
        n_layers = len(self._units)
        if isinstance(value, (list, tuple)):
            if len(value) != n_layers:
                raise ValueError(
                    f"{label} has {len(value)} entries but the model has "
                    f"{n_layers} layers (len(hidden_units) + 1); got {value!r}"
                )
            return list(value)
        return [value] * n_layers

    def _forward_batch_rows(self, n_rows: int, chunk_size: Optional[int]) -> int:
        """Rows per batch of the forward check, from the solve's byte budget.

        For every layer, the batch whose feature tensor ``(B, n_out, n_in, m)``
        at the compute dtype is no larger than the float64 feature tensor of
        one solve chunk of that layer (``HKANLayer.solve_chunk_outputs``); the
        smallest over the layers, at least 1 and at most ``n_rows``.

        :param n_rows: Rows of the fit.
        :type n_rows: int
        :param chunk_size: The ``chunk_size`` the fit was given.
        :type chunk_size: Optional[int]
        :return: The batch size.
        :rtype: int
        """
        # DECISION plan-2026-09-30T082355-4d999dbc/D-050
        # The forward check's batch comes from the SAME byte budget as the
        # solve. Do NOT go back to a fixed row count (it was 256): the feature
        # tensor is (B, n_out, n_in, m) whatever `chunk_size` is, so a fixed
        # batch made `chunk_size` stop bounding memory. Measured (64 inputs,
        # width 1024, m 10, 300 rows, chunk_size 16, CPU): peak RSS 3322 MB
        # with the fixed batch against 745 MB with the check stubbed out. See
        # decisions.md D-050.
        rows = n_rows
        for layer in self.hkan_layers:
            n_in = int(layer.centers.shape[1])
            outputs = layer.solve_chunk_outputs(n_rows, n_in, chunk_size)
            solve_bytes = outputs * n_in * n_rows * layer.num_basis * 8
            row_bytes = (layer.units * n_in * layer.num_basis
                         * np.dtype(layer.compute_dtype).itemsize)
            rows = min(rows, max(1, solve_bytes // row_bytes))
        return int(rows)

    def _forward_numpy(self, x: np.ndarray, batch_rows: int) -> np.ndarray:
        """Run the built model on ``x`` in batches, at the model's own dtype.

        :param x: Inputs, shape ``(N, n_in)``.
        :type x: np.ndarray
        :param batch_rows: Rows per batch (:meth:`_forward_batch_rows`).
        :type batch_rows: int
        :return: The model output as float64, shape ``(N,)``.
        :rtype: np.ndarray
        """
        dtype = self.hkan_layers[0].centers.dtype
        return np.concatenate([
            keras.ops.convert_to_numpy(
                self(x[start:start + batch_rows].astype(dtype), training=False))
            for start in range(0, x.shape[0], batch_rows)
        ], axis=0)[:, 0].astype(np.float64)

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every layer in order, exactly the chain ``call()`` runs.

        :param input_shape: ``(batch_size, n_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 2 or ``n_in`` is unknown.
        """
        input_shape = tuple(input_shape)
        if len(input_shape) != 2 or input_shape[-1] is None:
            raise ValueError(
                "HKAN needs an input of shape (batch, n_in) with a known n_in, "
                f"got {input_shape}"
            )
        shape = input_shape
        for layer in self.hkan_layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the layers in order.

        :param inputs: Input tensor of shape ``(batch_size, n_in)``.
        :type inputs: keras.KerasTensor
        :param training: Passed to the layers, which ignore it.
        :type training: Optional[bool]
        :return: Predictions of shape ``(batch_size, 1)``.
        :rtype: keras.KerasTensor
        """
        outputs = inputs
        for layer in self.hkan_layers:
            outputs = layer(outputs, training=training)
        return outputs

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Return ``(batch_size, 1)``.

        :param input_shape: ``(batch_size, n_in)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], 1)

    # ------------------------------------------------------------------

    def _ensure_built(self, x: np.ndarray) -> None:
        """Build the model from ``x`` if needed, else check ``n_in``.

        :param x: Input array of shape ``(N, n_in)``.
        :type x: np.ndarray
        :raises ValueError: If ``x`` is not rank 2 and non-empty, or its
            width differs from the width the model was built with.
        """
        if x.ndim != 2 or x.shape[0] < 1:
            raise ValueError(f"x must be a non-empty (N, n_in) array, got shape {x.shape}")
        if not self.built:
            self.build((None, x.shape[1]))
            return
        n_in = self.hkan_layers[0].centers.shape[1]
        if x.shape[1] != n_in:
            raise ValueError(
                f"x has {x.shape[1]} columns but the model was built for {n_in}"
            )

    def fit_closed_form(
            self,
            x: np.ndarray,
            y: np.ndarray,
            chunk_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train every layer by least squares, without gradients.

        Layer by layer: the centers are drawn from the layer's input when its
        mode is ``data``; every block is fitted to ``y`` by ridge regression
        and every output by a linear regression of its blocks, in float64;
        the result is assigned into the layer's weights (cast to their
        dtype); the float64 layer output becomes the next layer's input. An
        unbuilt model is built from ``x`` first. If a layer's solution is not
        finite, a ``ValueError`` names it; earlier layers keep their new
        weights and that layer and the later ones keep their old ones.

        The returned train predictions are the float64 fit's own. The model
        holds the solution cast to its dtype, and its forward pass differs
        from the fit by about the weight magnitude times the resolution of
        that dtype: 1e-7 for weights of order 1 in float32, and arbitrarily
        much for the block coefficients of 1e7 to 1e11 that a zero ridge
        gives on a smooth basis, or for the connecting weights of 1e3 to 1e8
        that ``l2_mix = 0`` gives on a wide layer (4e2 to 5e8 in the measured
        cases). The fit therefore ends with one
        batched forward pass of the model on ``x`` and reports it
        (``forward_rmse``, ``forward_deviation_rms``). When the deviation
        exceeds ``FORWARD_DEVIATION_FRACTION`` of the target's standard
        deviation (``FORWARD_DEVIATION_FLOOR`` in absolute terms for a
        target that ``is_constant_target`` calls constant), or is not a
        finite number, a warning is logged; it is not an exception, because
        the float64 diagnostics are still correct.

        :param x: Training inputs, shape ``(N, n_in)``.
        :type x: np.ndarray
        :param y: Training target, shape ``(N,)`` or ``(N, 1)``.
        :type y: np.ndarray
        :param chunk_size: Number of outputs of a layer solved at once. It
            bounds memory and does not change the result. ``None`` picks it
            per layer. The forward check runs in batches sized from the same
            byte budget, so it bounds that pass's memory too.
        :type chunk_size: Optional[int]
        :return: ``layer_rmse``: list of the train RMSE of every layer's
            outputs against ``y`` (mean over the layer's outputs of the
            squared error, then the root); ``block_r2``: array
            ``(n_hidden_1, n_in)``, the coefficient of determination of every
            first-layer block against ``y``; ``importance``: array
            ``(n_in,)``, its mean over the outputs (paper equation 14), which
            is ``0.0`` for every input when ``y`` is constant (a departure
            from scikit-learn's ``r2_score``, which gives ``1.0`` for an exact
            fit of a constant); ``train_predictions``: float64 array ``(N,)``;
            ``forward_rmse``: the RMSE of the model's own forward pass on
            ``x`` against ``y``, the number a later ``predict`` reproduces;
            ``forward_deviation_rms``: the RMS difference between that forward
            pass and ``train_predictions``.
        :rtype: Dict[str, Any]
        :raises ValueError: If a shape is wrong, ``y`` has more than one
            column, an input is not finite, or a solution is not finite.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(
                "HKAN fits a single target: y must have shape (N,) or (N, 1), "
                f"got {y.shape}"
            )
        self._ensure_built(x)
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                f"x has {x.shape[0]} rows but y has {y.shape[0]}"
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("x and y must be finite")

        layer_rmse: List[float] = []
        block_r2 = None
        activations = x
        for index, layer in enumerate(self.hkan_layers):
            if layer.centers_mode == "data":
                layer.centers.assign(layer.sample_data_centers(activations))
            # DECISION plan-2026-09-30T082355-4d999dbc/D-020
            # The solve reads the centers BACK from the weight. Do NOT fit
            # against the float64 draw and store its cast: at slope 50 the
            # float32 rounding of a center moves the basis argument by about
            # 1e-6, and the fitted coefficients would then belong to centers
            # the model does not hold. See decisions.md D-020.
            stored_centers = np.asarray(
                keras.ops.convert_to_numpy(layer.centers), dtype=np.float64)
            solution = layer.solve_closed_form(
                activations, y,
                l2_block=self._l2_block[index], l2_mix=self._l2_mix[index],
                centers=stored_centers, chunk_size=chunk_size,
            )
            layer.assign_solution(solution)
            activations = solution["output"]
            layer_rmse.append(float(np.sqrt(np.mean(np.square(activations - y[:, None])))))
            if index == 0:
                block_r2 = solution["block_r2"]

        # DECISION plan-2026-09-30T082355-4d999dbc/D-034
        # The fit ends by running the MODEL on the training rows. Do NOT
        # remove this pass or report `layer_rmse` alone "because the solve is
        # exact": at zero ridge with a sigmoid basis the float64 fit had RMSE
        # 8.5e-02 while the float32 model it left behind had 2.6e+02, and
        # nothing said so. Do NOT turn the warning into an exception either:
        # the float64 diagnostics are correct and the float64 policy is a
        # valid remedy. See decisions.md D-034.
        train_predictions = activations[:, 0]
        forward = self._forward_numpy(x, self._forward_batch_rows(x.shape[0], chunk_size))
        forward_rmse = float(np.sqrt(np.mean(np.square(forward - y))))
        deviation = float(np.sqrt(np.mean(np.square(forward - train_predictions))))
        # is_constant_target decides "constant": the standard deviation of 96
        # copies of 0.7 is 1e-16, not 0, and a fraction of that always warns.
        spread = 0.0 if is_constant_target(y) else float(np.std(y))
        if spread > 0:
            limit = FORWARD_DEVIATION_FRACTION * spread
            limit_text = (
                f"{FORWARD_DEVIATION_FRACTION:g} of the target's standard "
                f"deviation {spread:.3e}")
        else:
            limit = FORWARD_DEVIATION_FLOOR
            limit_text = (
                f"the absolute floor {FORWARD_DEVIATION_FLOOR:g} used for a "
                "constant target")
        if not deviation <= limit:
            # Over every weight, with a default: a frozen model has no
            # trainable weights and max() of nothing raised here, after the
            # weights were assigned and before the warning (decisions.md D-050).
            largest = max(
                (float(np.abs(keras.ops.convert_to_numpy(weight)).max())
                 for weight in self.weights),
                default=0.0)
            logger.warning(
                "HKAN closed-form fit: the model does not compute the fit it "
                f"was given. The float64 fit has train RMSE {layer_rmse[-1]:.3e}; "
                f"the model's own forward pass ({self.hkan_layers[0].centers.dtype}) "
                f"has train RMSE {forward_rmse:.3e} and differs from the fit by "
                f"{deviation:.3e} RMS, above {limit_text}. The usual cause is "
                f"large weights (the largest here is {largest:.1e}) whose terms "
                "cancel in the float64 solve and not at the model's precision: "
                "block coefficients from a weak or zero l2_block, or connecting "
                "weights from l2_mix = 0 on a wide layer, whose block outputs "
                "are nearly collinear. Remedies: raise l2_block, raise l2_mix, "
                "or build the model under the float64 dtype policy, for "
                "example HKAN(..., dtype=\"float64\") (which narrows the gap by "
                "the ratio of the two precisions and does not close it for "
                "weights of 1e10 and above)."
            )

        logger.info(f"HKAN closed-form fit: per-layer train RMSE {layer_rmse}")
        return {
            "layer_rmse": layer_rmse,
            "block_r2": block_r2,
            "importance": block_r2.mean(axis=0),
            "train_predictions": train_predictions,
            "forward_rmse": forward_rmse,
            "forward_deviation_rms": deviation,
        }

    def initialize_centers(self, x: np.ndarray) -> None:
        """Draw the ``data`` centers without a closed-form fit.

        For a gradient-only run. Layer by layer, every layer whose mode is
        ``data`` draws its centers from its own current input: the first
        layer from ``x``, a deeper layer from the model's activations at its
        current weights, with the earlier layers' centers already drawn.
        Layers in another mode are left untouched. An unbuilt model is built
        from ``x`` first. Drawing deeper centers from activations at the
        initial weights is this implementation's choice; the paper defines
        the data draw only inside its layer-by-layer fit.

        :param x: Inputs to draw from, shape ``(N, n_in)``.
        :type x: np.ndarray
        :raises ValueError: If ``x`` has the wrong shape.
        """
        x = np.asarray(x)
        self._ensure_built(x)
        last_data_layer = max(
            (i for i, layer in enumerate(self.hkan_layers) if layer.centers_mode == "data"),
            default=-1,
        )
        activations = x.astype(self.hkan_layers[0].centers.dtype)
        for index in range(last_data_layer + 1):
            layer = self.hkan_layers[index]
            if layer.centers_mode == "data":
                layer.centers.assign(layer.sample_data_centers(activations))
            if index < last_data_layer:
                activations = np.concatenate([
                    keras.ops.convert_to_numpy(
                        layer(activations[start:start + _PROPAGATION_BATCH]))
                    for start in range(0, activations.shape[0], _PROPAGATION_BATCH)
                ], axis=0)

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        Fitted values are not in the config; they travel in the weights of
        the ``.keras`` archive.

        :return: Every constructor argument, numpy scalars converted to
            plain Python numbers.
        :rtype: Dict[str, Any]
        """
        def scalar(value: Any) -> Any:
            return value.item() if isinstance(value, np.generic) else value

        def plain(value: Any) -> Any:
            if isinstance(value, (list, tuple)):
                return [scalar(item) for item in value]
            return scalar(value)

        config = super().get_config()
        config.update({
            "hidden_units": list(self.hidden_units),
            "num_basis": plain(self.num_basis),
            "basis": plain(self.basis),
            "slope": plain(self.slope),
            "centers": plain(self.centers),
            "l2_block": plain(self.l2_block),
            "l2_mix": plain(self.l2_mix),
            "use_block_bias": self.use_block_bias,
            "use_bias": self.use_bias,
            "coef_initializer": keras.initializers.serialize(self.coef_initializer),
            "mix_initializer": (
                None if self.mix_initializer is None
                else keras.initializers.serialize(self.mix_initializer)
            ),
            "seed": plain(self.seed),
        })
        return config

# ---------------------------------------------------------------------


def create_hkan(
        hidden_units: Sequence[int] = (),
        input_dim: Optional[int] = None,
        **kwargs: Any,
) -> HKAN:
    """Create an :class:`HKAN`, optionally built for a known input width.

    :param hidden_units: Widths of the hidden layers.
    :type hidden_units: Sequence[int]
    :param input_dim: When given, the model is built for ``(None, input_dim)``
        so its weights exist before the first call.
    :type input_dim: Optional[int]
    :param kwargs: Any other :class:`HKAN` constructor argument. An unknown
        name raises from the constructor.
    :return: The model.
    :rtype: HKAN
    """
    model = HKAN(hidden_units=hidden_units, **kwargs)
    if input_dim is not None:
        model.build((None, input_dim))
    return model

# ---------------------------------------------------------------------
