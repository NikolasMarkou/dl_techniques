"""HKAN, a Kolmogorov-Arnold regressor fit layer by layer without gradients.

:class:`HKAN` stacks :class:`HKANLayer` blocks over ``(B, n_in)`` features
and returns one prediction per row, ``(B, 1)``. Instead of training learned
edge functions by backpropagation, it fixes the non-linear parameters (basis
centers and slope), so every block and every connecting stage is a linear
regression onto the target. :meth:`HKAN.fit_closed_form` solves them in
float64, layer by layer, all onto the same ``y``; ``keras.Model.fit`` stays
the stock gradient loop and can fine-tune the same weights afterwards. The
model fits one target only. Only the float32 and float64 dtype policies
build. The defaults ``slope=5.0`` and ``l2_block=l2_mix=0.01`` differ from
the reference code's ``1.0, 0.0, 0.0``. There is no variants table and no
``pretrained`` argument.

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

# DECISION plan-2026-09-30T082355-4d999dbc/D-039: defaults stay 5.0, 0.01, 0.01.
# The reference 1, 0, 0 leaves a float32 model far from its own fit. See decisions.md.
#: Default ridge strength of the block regressions.
DEFAULT_L2_BLOCK: float = 0.01
#: Default ridge strength of the connecting regressions.
DEFAULT_L2_MIX: float = 0.01

#: Rows per forward batch when :meth:`HKAN.initialize_centers` propagates
#: activations. The forward check of :meth:`HKAN.fit_closed_form` sizes its
#: batch from the solve's byte budget instead.
_PROPAGATION_BATCH: int = 256

#: :meth:`HKAN.fit_closed_form` warns when the RMS gap between the model's
#: forward pass and the float64 fit exceeds this fraction of the target's
#: standard deviation. A well-scaled float32 fit sits near 1e-7.
FORWARD_DEVIATION_FRACTION: float = 1e-3

#: Warning limit for a constant target (``is_constant_target``): this much
#: times ``max(1, max |y|)``, about ten float32 steps at the target's scale.
FORWARD_DEVIATION_FLOOR: float = 1e-6

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.hkan.model")
class HKAN(keras.Model):
    """Regress one target through a stack of :class:`HKANLayer` blocks.

    The model holds ``len(hidden_units) + 1`` layers, the last of width 1.
    Train it by :meth:`fit_closed_form`, by ``keras.Model.fit``, or by the
    first followed by the second; :meth:`fit_closed_form` is a separate
    method and does not override ``fit`` or ``train_step``. Centers and
    slopes never train. With ``centers="data"`` the centers are drawn inside
    :meth:`fit_closed_form`, or by :meth:`initialize_centers` for a
    gradient-only run; until then they hold a uniform placeholder.
    Intercepts are on by default in both stages; the paper's equations have
    none. A hidden layer of width 1 is allowed.

    Every per-layer setting (``num_basis``, ``basis``, ``slope``, ``centers``,
    ``l2_block``, ``l2_mix``) takes one value for all layers or a list or
    tuple with one value per layer, ``len(hidden_units) + 1`` long. Integer
    arguments may be numpy integers; ``get_config`` returns plain Python
    numbers and lists.

    Architecture:

    .. code-block:: text

        x [B, n_in]
                 │
                 ▼
        ┌─────────────────────┐
        │ hkan_layer_0        │  n_in -> hidden_units[0]
        └─────────────────────┘
                 │  [B, hidden_units[0]]
                 ▼
                ...
                 │  [B, hidden_units[-1]]
                 ▼
        ┌─────────────────────┐
        │ hkan_layer_L        │  hidden_units[-1] -> 1
        └─────────────────────┘
                 │
                 ▼
        y_hat [B, 1]

    L is len(hidden_units); with none, hkan_layer_0 maps n_in -> 1.

    Closed-form fit (:meth:`fit_closed_form`):

    .. code-block:: text

        x [N, n_in] as float64
                 │
                 ▼
        ┌─────────────────────┐
        │ hkan_layer_0        │◄── y, centers read from weight
        └─────────────────────┘
                 │  output [N, units of layer 0]
                 ├──► assign_solution, block_r2
                 ▼
                ...
                 │
                 ▼
        ┌─────────────────────┐
        │ hkan_layer_L        │◄── y
        └─────────────────────┘
                 │  train_predictions [N]
                 ▼
        model forward pass on x ──► forward_rmse, warning

    A data-mode layer draws its centers from its input before its solve.

    :param hidden_units: Widths of the hidden layers, a list or tuple of
        positive integers. Empty for a one-layer model. The output layer of
        width 1 is added automatically.
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
        :meth:`fit_closed_form`. Must be finite and non-negative. Default
        0.01. ``0``, the reference code's default, is minimum-norm least
        squares, which on a smooth basis returns coefficients a float32
        model cannot carry. No effect on gradient training.
    :type l2_block: Union[float, Sequence[float]]
    :param l2_mix: Ridge strength of the connecting regressions in
        :meth:`fit_closed_form`. Must be finite and non-negative. Default
        0.01. The paper uses ``0``; on a wide layer the block outputs are
        nearly collinear and ``0`` gives very large connecting weights. No
        effect on gradient training.
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
    :param seed: Seed of the centers, shared by every layer and mixed with
        its index. ``None`` draws from the global numpy generator. ``0`` is a
        seed.
    :type seed: Optional[int]
    :param **kwargs: Additional arguments for the ``keras.Model`` base. A
        ``dtype`` given here is the dtype policy of every layer too.

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
        """Validate the arguments and create one layer per stage.

        Arguments are documented on the class.

        :raises ValueError: If ``hidden_units`` is not a list or tuple of
            positive integers, a per-layer sequence has the wrong length,
            ``l2_block`` or ``l2_mix`` is not finite or is negative, or a
            layer rejects its setting.
        """
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

        # A model-owned flat list keeps weight tracking through a .keras save.
        # DECISION plan-2026-09-30T082355-4d999dbc/D-049: every layer gets dtype=self.dtype_policy.
        # Without it HKAN(dtype=...) skips the layers and their dtype check. See decisions.md.
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
        """Return rows per batch of the forward check, from the solve's byte budget.

        For each layer, the batch whose feature tensor ``(B, n_out, n_in, m)``
        at the compute dtype fits in the float64 feature tensor of one solve
        chunk (``HKANLayer.solve_chunk_outputs``). Returns the smallest over
        the layers, at least 1 and at most ``n_rows``. Needs a built model.

        :param n_rows: Rows of the fit.
        :type n_rows: int
        :param chunk_size: The ``chunk_size`` the fit was given.
        :type chunk_size: Optional[int]
        :return: The batch size.
        :rtype: int
        """
        # DECISION plan-2026-09-30T082355-4d999dbc/D-050: batch from the solve's byte budget.
        # A fixed row count stops chunk_size from bounding memory. See decisions.md.
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

        Calls the model with ``training=False``.

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
        :raises ValueError: If the input is not rank 2, ``n_in`` is unknown,
            or the compute dtype is not float32 or float64 (from
            :meth:`HKANLayer.build`).
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
        :raises ValueError: From the first layer's ``input_spec``, when the
            input width is not the one the model was built with.
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

        Layer by layer: a ``data``-mode layer draws its centers from its
        input; every block is fit to ``y`` by ridge regression and every
        output by a regression of its blocks, in float64; the result is cast
        into the layer's weights; the float64 layer output becomes the next
        layer's input. An unbuilt model is built from ``x`` first. If a
        layer's solution is not finite, ``ValueError`` names it. Earlier
        layers keep their new weights; that layer and later ones keep their
        old ``coef``, ``mix`` and intercepts, but a ``data``-mode layer has
        already overwritten its centers.

        The fit ends with one batched forward pass of the model on ``x``.
        The model holds the solution at its own dtype, so its output differs
        from the float64 fit by about the largest weight times that dtype's
        resolution. A warning is logged, not raised, when the RMS gap is not
        at most ``FORWARD_DEVIATION_FRACTION`` of the standard deviation of
        ``y``, or ``FORWARD_DEVIATION_FLOOR * max(1, max |y|)`` for a target
        that :func:`is_constant_target` calls constant. A NaN gap warns.

        :param x: Training inputs, shape ``(N, n_in)``.
        :type x: np.ndarray
        :param y: Training target, shape ``(N,)`` or ``(N, 1)``.
        :type y: np.ndarray
        :param chunk_size: Number of outputs of a layer solved at once. It
            bounds memory and does not change the result. ``None`` picks it
            per layer. The forward check sizes its batches from the same
            byte budget.
        :type chunk_size: Optional[int]
        :return: A dict with ``layer_rmse``, a list with each layer's train
            RMSE against ``y``, averaged over rows and the layer's outputs;
            ``block_r2``, array ``(units of layer 0, n_in)``, the R^2 of every
            first-layer block against ``y``; ``importance``, array
            ``(n_in,)``, its mean over the outputs (paper equation 14), all
            ``0.0`` when ``y`` is constant; ``train_predictions``, the float64
            fit's array ``(N,)``; ``forward_rmse``, the RMSE of the model's
            own forward pass, which a later ``predict`` reproduces; and
            ``forward_deviation_rms``, the RMS gap between that pass and
            ``train_predictions``.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``x`` is not a non-empty ``(N, n_in)`` array or
            its width differs from the built model, ``y`` is not ``(N,)`` or
            ``(N, 1)``, the row counts differ, an input is not finite,
            ``chunk_size`` is below 1, or a solution is not finite.
        :raises numpy.linalg.LinAlgError: Propagated from a layer's solve.
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
            # DECISION plan-2026-09-30T082355-4d999dbc/D-020: solve against centers read back.
            # Fitting the float64 draw fits centers the model does not hold. See decisions.md.
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

        # DECISION plan-2026-09-30T082355-4d999dbc/D-034: always run the model on x.
        # Warn, never raise: layer_rmse alone hides a model far from its fit. See decisions.md.
        train_predictions = activations[:, 0]
        forward = self._forward_numpy(x, self._forward_batch_rows(x.shape[0], chunk_size))
        forward_rmse = float(np.sqrt(np.mean(np.square(forward - y))))
        deviation = float(np.sqrt(np.mean(np.square(forward - train_predictions))))
        # np.std of 96 copies of 0.7 is 1e-16, not 0, hence is_constant_target.
        spread = 0.0 if is_constant_target(y) else float(np.std(y))
        if spread > 0:
            limit = FORWARD_DEVIATION_FRACTION * spread
            limit_text = (
                f"{FORWARD_DEVIATION_FRACTION:g} of the target's standard "
                f"deviation {spread:.3e}")
        else:
            limit = FORWARD_DEVIATION_FLOOR * max(1.0, float(np.max(np.abs(y))))
            limit_text = (
                f"the floor {limit:.3e} ({FORWARD_DEVIATION_FLOOR:g} times the "
                "target's scale) used for a constant target")
        if not deviation <= limit:
            # default=0.0 covers an empty weight list.
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

        For a gradient-only run. Each ``data``-mode layer draws its centers
        from its own current input: the first layer from ``x``, a deeper one
        from the activations at the current weights, after the earlier
        layers' centers are drawn. Other layers are left untouched, and
        propagation stops at the last ``data``-mode layer. An unbuilt model
        is built from ``x`` first. The paper defines the data draw only
        inside its layer-by-layer fit; the deeper-layer draw is this
        implementation's.

        :param x: Inputs to draw from, shape ``(N, n_in)``. Not checked for
            finiteness.
        :type x: np.ndarray
        :raises ValueError: If ``x`` is not a non-empty ``(N, n_in)`` array
            or its width differs from the built model.
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
            plain Python numbers and tuples to lists.
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
    :param **kwargs: Any other :class:`HKAN` constructor argument. An
        unknown name raises from the constructor.
    :return: The model.
    :rtype: HKAN
    :raises ValueError: If the constructor rejects an argument, or the build
        rejects ``input_dim`` or the dtype policy.
    """
    model = HKAN(hidden_units=hidden_units, **kwargs)
    if input_dim is not None:
        model.build((None, input_dim))
    return model

# ---------------------------------------------------------------------
