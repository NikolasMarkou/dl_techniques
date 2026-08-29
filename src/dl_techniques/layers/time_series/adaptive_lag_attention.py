"""
Context-gated attention over a fixed set of lags.

This module holds ``AdaptiveLagAttentionLayer``. The layer makes a one-step
autoregressive forecast from a fixed set of past values, and lets a context
vector decide how much each past value counts.

Control and data are separate. A ``context_tensor``, usually from a deep
encoder such as an LSTM, is the controller. A ``lag_tensor`` holds the raw
historical values. The context produces two control signals:

1. **Attention weights.** A Dense layer with ``sigmoid`` turns the context
   into one weight per lag. Each weight says how much that historical value
   matters right now.

2. **Master gate.** A second Dense layer with ``sigmoid`` turns the same
   context into a single value. It scales the whole autoregressive
   contribution, so the layer can shut history off entirely.

The output is the weighted sum of the lags, scaled by the gate. A model can
learn to ignore history during anomalous periods (gate near 0), or to lean on
one seasonality by putting a high weight on its lag.

**Independent sigmoid attention.** Transformers use ``softmax``, which forces
``sum(weights) = 1`` and makes the lags compete. This layer uses ``sigmoid``
instead, so each weight is an independent value in ``(0, 1)``. Several lags
can be important at once, for example 7 days ago and 365 days ago. All of them
can also be near zero at the same time.

**Multiplicative gating.** The output is::

    output = g * (sum_i w_i * l_i)

where ``g`` is the master gate, ``w_i`` are the attention weights and ``l_i``
are the lag values.

References:
    Gating to control information flow is a foundational idea in modern deep
    learning, best known from LSTMs and GRUs.

    - Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
      Neural Computation.
      https://www.bioinf.jku.at/publications/older/2604.pdf
    - Cho, K., et al. (2014). Learning Phrase Representations using RNN
      Encoder-Decoder for Statistical Machine Translation. In EMNLP.
      https://arxiv.org/abs/1406.1078
"""

import keras
from keras import ops
from typing import List, Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.time_series.adaptive_lag_attention")
class AdaptiveLagAttentionLayer(keras.layers.Layer):
    """
    Weight a set of temporal lags from a context vector, then gate the result.

    The context tensor drives two Dense heads. One makes an independent
    sigmoid weight per lag. The other makes a single sigmoid master gate. The
    weights scale the lags, the result is summed, and the gate scales that
    sum. Sigmoid rather than softmax means the weights do not compete, so
    several lags can be important at once.

    The steps are:

    1. **Attention weights**: ``w = sigma(W_a * context + b_a)``, sigma is
       sigmoid, shape ``(batch, num_lags)``
    2. **Gate value**: ``g = sigma(W_g * context + b_g)``, shape ``(batch, 1)``
    3. **Weighted sum**: ``s = sum_i w_i * lag_i``, shape ``(batch,)``
    4. **Output**: ``output = g * s``, reshaped to ``(batch, 1)``

    **Architecture Overview:**

    .. code-block:: text

        Context (batch, context_dim)          Lags (batch, num_lags)
                │                                     │
                ├──────────────┐                      │
                ▼              ▼                      │
        ┌──────────────┐ ┌──────────┐                 │
        │Dense(num_lags│ │ Dense(1) │                 │
        │  sigmoid)    │ │ sigmoid  │                 │
        └──────┬───────┘ └────┬─────┘                 │
               │              │                       │
               ▼              │                       ▼
        Attention Weights     │              ┌────────────────┐
        (batch, num_lags)     │              │  Element-wise  │
               │              │              │  Multiply      │
               └──────────────┼──► w * lags ─┘
                              │         │
                              │         ▼
                              │    ops.sum(axis=-1)
                              │         │
                              │         ▼
                              │   Weighted Sum (batch,)
                              │         │
                              ▼         ▼
                         Gate (batch,1) │
                              │ squeeze │
                              ▼         ▼
                          (batch,)      │
                              │         │
                              ▼         ▼
                        ┌───────────────────┐
                        │  g * weighted_sum │
                        └─────────┬─────────┘
                                  │
                                  ▼
                        Output (batch, 1)

    The lag tensor is used as data only. No weights are attached to it, so
    the layer's parameter count depends on the context width, not on the lags.

    Input shape:
        A list of two tensors. Context ``(batch_size, context_dim)`` and lags
        ``(batch_size, num_lags)``. Both must be at least 2D.

    Output shape:
        2D tensor ``(batch_size, 1)``.

    Example:
        .. code-block:: python

            layer = AdaptiveLagAttentionLayer(num_lags=7)
            y = layer([context, lags])

    :param num_lags: Number of past values in the lag tensor. Must be a
        positive int and must equal ``lag_tensor.shape[-1]``.
    :type num_lags: int
    :param kernel_initializer: Initializer for both Dense kernels. Defaults to
        ``"glorot_uniform"``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for both Dense biases. Defaults to
        ``"zeros"``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for both Dense kernels.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for both Dense biases.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param activity_regularizer: Optional regularizer on the output. See the
        note in ``call()`` before using it.
    :type activity_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the Layer parent class.

    :raises ValueError: If ``num_lags`` is not a positive integer.
    :raises ValueError: If ``build`` or ``call`` gets anything other than a
        list of two entries.
    :raises ValueError: If the lag tensor's last axis is not ``num_lags``.

    :ivar attention_generator: Dense head making one sigmoid weight per lag.
    :vartype attention_generator: keras.layers.Dense
    :ivar gate_generator: Dense head making the single sigmoid master gate.
    :vartype gate_generator: keras.layers.Dense
    """

    def __init__(
        self,
        num_lags: int,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """
        Validate ``num_lags`` and create the two Dense sublayers.

        Both sublayers are created here and left unbuilt. ``build()`` builds
        them against the context shape.

        :param num_lags: Number of past values in the lag tensor.
        :type num_lags: int
        :param kernel_initializer: Initializer for kernel weights.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param bias_initializer: Initializer for bias vectors.
        :type bias_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Optional regularizer for kernel weights.
        :type kernel_regularizer: str or keras.regularizers.Regularizer or None
        :param bias_regularizer: Optional regularizer for bias vectors.
        :type bias_regularizer: str or keras.regularizers.Regularizer or None
        :param activity_regularizer: Optional regularizer on the output.
        :type activity_regularizer: str or keras.regularizers.Regularizer or None
        :param kwargs: Additional keyword arguments for the Layer parent class.

        :raises ValueError: If ``num_lags`` is not a positive integer.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(num_lags, int) or num_lags <= 0:
            raise ValueError(f"num_lags must be a positive integer, got {num_lags}")

        # Store ALL configuration parameters
        self.num_lags = num_lags
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Sublayer for attention weights: maps context -> independent weights
        self.attention_generator = keras.layers.Dense(
            units=self.num_lags,
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='attention_generator'
        )

        # Sublayer for master gate: maps context -> single gate value
        self.gate_generator = keras.layers.Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='gate_generator'
        )

        logger.debug(f"AdaptiveLagAttentionLayer initialized with num_lags={num_lags}")

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Validate the two input shapes and build both Dense sublayers.

        Both sublayers read the context, so both are built on
        ``context_shape``. The lag shape is only checked, never built against.
        Building explicitly means a ``.keras`` weight restore finds every
        variable already materialized.

        :param input_shape: A list of two shape tuples, for the context tensor
            and the lag tensor.
        :type input_shape: list[tuple[int or None, ...]]
        :raises ValueError: If ``input_shape`` is not a list of two shapes.
        :raises ValueError: If the context shape has fewer than 2 axes.
        :raises ValueError: If the lag shape's last axis is not ``num_lags``.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "This layer expects a list of two inputs: [context_tensor, lag_tensor]. "
                f"Received input_shape: {input_shape}"
            )

        context_shape, lag_shape = input_shape

        # Validate shapes
        if len(context_shape) < 2:
            raise ValueError(
                f"Context tensor must be at least 2D, got shape: {context_shape}"
            )

        if len(lag_shape) < 2 or lag_shape[-1] != self.num_lags:
            raise ValueError(
                f"The last dimension of the lag_tensor input ({lag_shape[-1] if len(lag_shape) >= 2 else 'unknown'}) "
                f"does not match `num_lags` ({self.num_lags})."
            )

        # Build sub-layers in computational order
        self.attention_generator.build(context_shape)
        self.gate_generator.build(context_shape)

        logger.debug(f"AdaptiveLagAttentionLayer built with context_shape={context_shape}, lag_shape={lag_shape}")

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Weight the lags from the context, sum them, then apply the gate.

        Note on ``activity_regularizer``: this method calls ``add_loss`` on the
        output itself, and Keras 3's ``Layer.__call__`` already applies the
        same ``activity_regularizer`` to the layer's output. With an
        ``activity_regularizer`` set, the penalty is therefore counted twice.
        Measured with ``L1(1.0)`` on a ``(2, 5)`` context and a ``(2, 4)`` lag
        tensor: ``len(layer.losses) == 2``, and both entries equal each other
        and a manual ``activity_regularizer(output)`` call. The value itself
        depends on the seed, so no figure is quoted here. This is a code
        defect, reported and left unfixed here.

        :param inputs: A list of two tensors. ``inputs[0]`` is the context of
            shape ``(batch_size, context_dim)``, ``inputs[1]`` is the lag
            tensor of shape ``(batch_size, num_lags)``.
        :type inputs: list[keras.KerasTensor]
        :param training: Whether the layer runs in training mode. Forwarded to
            both Dense sublayers.
        :type training: bool or None
        :return: Predicted value of shape ``(batch_size, 1)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``inputs`` is not a list of exactly two tensors.
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"Expected a list of two inputs, got {type(inputs)} with length "
                f"{len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
            )

        context_tensor, lag_tensor = inputs

        # 1. Generate independent attention weights from the context
        # Shape: (batch_size, num_lags)
        attention_weights = self.attention_generator(context_tensor, training=training)

        # 2. Generate the master gate value from the context
        # Shape: (batch_size, 1)
        gate = self.gate_generator(context_tensor, training=training)

        # 3. Compute the weighted sum of the lags
        # Shape: (batch_size,)
        weighted_sum_of_lags = ops.sum(attention_weights * lag_tensor, axis=-1)

        # 4. Apply the master gate
        # Shape: (batch_size,)
        gated_output = ops.squeeze(gate, axis=-1) * weighted_sum_of_lags

        # 5. Reshape for a consistent output shape
        # Shape: (batch_size, 1)
        output = ops.expand_dims(gated_output, axis=-1)

        # Apply activity regularization if specified
        if self.activity_regularizer is not None:
            self.add_loss(self.activity_regularizer(output))

        return output

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape from the context shape alone.

        Only the batch axis of the context is used. The lag shape is ignored.

        :param input_shape: A list of two shape tuples, for the context tensor
            and the lag tensor.
        :type input_shape: list[tuple[int or None, ...]]
        :return: The output shape ``(batch_size, 1)``.
        :rtype: tuple[int or None, ...]
        :raises ValueError: If ``input_shape`` is not a list of two shapes.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "Expected input_shape to be a list of two tuples, "
                f"got {type(input_shape)} with length "
                f"{len(input_shape) if hasattr(input_shape, '__len__') else 'unknown'}"
            )

        context_shape, _ = input_shape

        # Output shape: (batch_size, 1)
        return (context_shape[0], 1)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this layer.

        Initializers and regularizers are serialized to their dict form.

        :return: Configuration dictionary covering every ``__init__``
            parameter.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_lags": self.num_lags,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
