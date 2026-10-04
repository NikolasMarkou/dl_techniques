"""HarmonicDense: Euclidean-distance replacement for ``Dense + Softmax``.

Baek, Liu, Tyagi, Tegmark. "Harmonic Loss Trains Interpretable AI Models",
TMLR 2025.

A standard classifier head scores ``logit_i = x . w_i`` (dot product) and
normalizes with softmax. This layer instead scores by Euclidean distance::

    logit_i = -n * log ||x - w_i||_2

so each output unit is a prototype in input space and the prediction is the
nearest prototype under a harmonic (inverse-power) weighting. Because
``HarMax(d)_i == softmax(-n * log d)_i``, the ``"probs"`` mode is exactly a
:class:`dl_techniques.layers.activations.harmax.HarMax` applied to the
distances, while the ``"logits"`` mode is the numerically stable training
path for use with ``SparseCategoricalCrossentropy(from_logits=True)``.

Works on inputs of any rank ``(..., dim)``, e.g. an LM head on
``(batch, seq, dim)``. The layer owns one kernel ``(dim, units)`` and no
bias: a bias would be a per-class additive logit shift with no distance
interpretation.

References:
    - Baek et al., 2025. "Harmonic Loss Trains Interpretable AI Models".
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.structured_linear.harmonic_dense")
class HarmonicDense(keras.layers.Layer):
    """Dense head scored by distance to learned prototypes.

    Holds a kernel ``(dim, units)`` whose columns are the prototypes
    ``w_i``. Each forward pass computes squared distances
    ``d2_i = ||x - w_i||^2`` via the expanded form
    ``||x||^2 - 2 xW + ||w_i||^2`` (floored at ``epsilon``), then:

    - ``"distances"`` returns ``sqrt(d2)`` (feed into
      :class:`dl_techniques.layers.activations.harmax.HarMax`);
    - ``"logits"`` returns ``-0.5 * n * log(d2)`` (``== -n * log(d)``),
      for ``SparseCategoricalCrossentropy(from_logits=True)``;
    - ``"probs"`` returns ``softmax(logits)`` along the last axis, for
      ``SparseCategoricalCrossentropy()``.

    **Architecture Overview:**

    .. code-block:: text

                        x  [..., D]
                                │
                 ┌────────────────┼───────────────┐
                 │                │               │
                 ▼                ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │ ||x||^2       │ │ x @ kernel    │ │ ||w_j||^2    │
        │ (..., 1)      │ │ (..., units)  │ │ (1, units)   │
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                └────────────┬────┴────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ d2 = max(x2 - 2xW + w2, eps) │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │ output_mode branch           │
              │  distances -> sqrt(d2)       │
              │  logits    -> -0.5 n log(d2) │
              │  probs     -> softmax(logits)│
              └──────────────┬───────────────┘
                             ▼
                    y  [..., units]

    :param units: Positive integer, dimensionality of the output space.
    :type units: int
    :param n: Harmonic exponent. ``None`` (default) means
        ``sqrt(input_dim)``, the wide-layer heuristic from the paper. A
        number must be positive. The constructor value is what
        :meth:`get_config` writes; the resolved value lives on
        ``effective_n`` after :meth:`build`.
    :type n: Optional[float]
    :param output_mode: One of ``"probs"``, ``"logits"``, ``"distances"``.
        Defaults to ``"probs"``.
    :type output_mode: str
    :param epsilon: Floor for the squared distances before ``log``/``sqrt``.
        Must be positive. Defaults to 1e-8.
    :type epsilon: float
    :param kernel_initializer: Initializer for the ``(dim, units)`` kernel.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernel.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments passed to the Layer base
        class.

    :raises ValueError: If ``units`` is not a positive integer, if
        ``output_mode`` is unknown, if ``n`` is not ``None`` or positive,
        or if ``epsilon`` is not positive. Raised from ``__init__``.

    :ivar kernel: The prototype matrix, ``None`` until :meth:`build` runs.
    :vartype kernel: Optional[keras.Variable]
    :ivar effective_n: The resolved harmonic exponent, ``None`` until
        :meth:`build` runs.
    :vartype effective_n: Optional[float]
    """

    #: Every accepted ``output_mode`` spelling, checked in ``__init__``.
    _SUPPORTED_MODES: Tuple[str, ...] = ("probs", "logits", "distances")

    def __init__(
            self,
            units: int,
            n: Optional[float] = None,
            output_mode: str = "probs",
            epsilon: float = 1e-8,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and store it.

        No weight is created here; :meth:`build` creates the kernel once the
        input dimension is known.

        :param units: Positive integer, dimensionality of the output space.
        :type units: int
        :param n: Harmonic exponent or ``None`` for the ``sqrt(dim)``
            heuristic. Must be positive when given.
        :type n: Optional[float]
        :param output_mode: One of ``"probs"``, ``"logits"``,
            ``"distances"``. Defaults to ``"probs"``.
        :type output_mode: str
        :param epsilon: Floor for squared distances. Must be positive.
        :type epsilon: float
        :param kernel_initializer: Initializer for the kernel.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Regularizer for the kernel.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``units`` is not a positive integer, if
            ``output_mode`` is unknown, if ``n`` is not ``None`` or
            positive, or if ``epsilon`` is not positive.
        """
        super().__init__(**kwargs)
        if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")
        if output_mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Invalid output_mode '{output_mode}'. "
                f"Supported modes: {list(self._SUPPORTED_MODES)}"
            )
        if n is not None:
            n = float(n)
            if n <= 0:
                raise ValueError(f"n must be positive or None, got {n}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.units = units
        self.n = n
        self.output_mode = output_mode
        self.epsilon = float(epsilon)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel = None
        self.effective_n: Optional[float] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the ``(dim, units)`` prototype kernel.

        Resolves ``effective_n`` to ``float(n)`` when ``n`` was given, else
        ``sqrt(dim)``.

        :param input_shape: Shape of the input tensor. The last dimension
            must be defined.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last dimension of ``input_shape`` is
            ``None``.
        """
        if self.built:
            return
        dim = input_shape[-1]
        if dim is None:
            raise ValueError(
                "The last dimension of the input shape must be defined. "
                f"Received input_shape={input_shape}"
            )
        self.kernel = self.add_weight(
            name="kernel",
            shape=(dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        self.effective_n = float(self.n) if self.n is not None else float(dim) ** 0.5
        super().build(input_shape)

    def _squared_distances(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Return floored squared distances ``||x - w_i||^2``.

        Uses the expanded form ``||x||^2 - 2 xW + ||w||^2`` so inputs of any
        rank ``(..., dim)`` work through a single matmul.

        :param inputs: Input tensor shaped ``(..., dim)``.
        :type inputs: keras.KerasTensor
        :return: Squared distances shaped ``(..., units)``, floored at
            ``epsilon``.
        :rtype: keras.KerasTensor
        """
        x2 = keras.ops.sum(keras.ops.square(inputs), axis=-1, keepdims=True)
        w2 = keras.ops.sum(keras.ops.square(self.kernel), axis=0, keepdims=True)
        xw = keras.ops.matmul(inputs, self.kernel)
        return keras.ops.maximum(x2 - 2.0 * xw + w2, self.epsilon)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Score ``inputs`` against the prototypes per ``output_mode``.

        :param inputs: Input tensor shaped ``(..., dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor shaped ``(..., units)``: probabilities, logits, or
            raw distances according to ``output_mode``.
        :rtype: keras.KerasTensor
        """
        d2 = self._squared_distances(inputs)
        if self.output_mode == "distances":
            return keras.ops.sqrt(d2)
        logits = -0.5 * self.effective_n * keras.ops.log(d2)
        if self.output_mode == "logits":
            return logits
        return keras.ops.softmax(logits, axis=-1)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``tuple(input_shape[:-1]) + (units,)``.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        ``n`` is written back as passed to the constructor (``None`` stays
        ``None``); the resolved exponent is a build artifact, restored from
        the input dimension rather than from here.

        :return: The base Layer config plus ``units``, ``n``,
            ``output_mode``, ``epsilon``, and the serialized kernel
            initializer and regularizer.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "n": self.n,
            "output_mode": self.output_mode,
            "epsilon": self.epsilon,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
