"""
Fuse contextual and autoregressive forecasts with a dynamic gating mechanism.

This layer implements a sophisticated fusion strategy for time series
forecasting, designed to combine the strengths of a deep, context-aware model
with a simple, dynamic autoregressive model. It is particularly effective for
time series that exhibit both complex, non-linear patterns and stable,
history-dependent behaviors.

The layer operates on the principle of a dynamically weighted mixture of
experts, where two specialized forecasting pathways are blended based on the
current input context.

1.  **The Contextual Pathway**: This path leverages a rich, latent
    representation of the time series (the ``context_tensor``), typically the
    output of a recurrent or attentional encoder. A dense layer transforms
    this context directly into a forecast. This pathway is responsible for
    capturing complex, non-linear relationships, incorporating exogenous
    features, and understanding the high-level "state" of the system.

2.  **The Autoregressive Pathway**: This path models the forecast as a function
    of recent past values (the ``lag_tensor``). Crucially, it is not a static
    autoregressive model. Instead, it uses a context-aware attention
    mechanism where the ``context_tensor`` generates weights for each lag. This
    allows the model to dynamically decide which past time steps are most
    relevant for the current prediction.

3.  **The Fusion Gate**: The core of the layer is a learned gating mechanism.
    The ``context_tensor`` is passed through a separate dense layer with a
    sigmoid activation to produce a scalar "fusion gate" value between 0 and 1.
    This gate determines the mixing proportion between the two pathways,
    allowing the model to learn a sophisticated switching strategy. For instance,
    it might learn to rely on the stable autoregressive path during normal
    periods (gate ~ 1) but switch to the more flexible contextual path during
    anomalous events or regime shifts (gate ~ 0).

**Foundational Mathematics:**

The layer's operation can be described as a context-dependent, gated linear
interpolation between two expert forecasts. Given a context vector ``c`` and a
lag vector ``l = [l_1, l_2, ..., l_n]``:

1.  Context Forecast: a direct, non-linear projection of the context.
        f_context = W_c * c + b_c

2.  Autoregressive Forecast: a contextually-weighted sum of lags.
        alpha = sigmoid(W_alpha * c + b_alpha)
        f_lag = W_l * (sum_i alpha_i * l_i) + b_l

3.  Fusion Gate: a learned interpolation coefficient.
        g = sigmoid(W_g * c + b_g)

4.  Final Output: the gated combination of the two forecasts.
        output = (1 - g) * f_context + g * f_lag

References:
    - Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory.
      Neural Computation.
      https://www.bioinf.jku.at/publications/older/2604.pdf
    - Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017).
      Language Modeling with Gated Convolutional Networks. In ICML.
      https://arxiv.org/abs/1612.08083
"""

import keras
from keras import ops
from typing import List, Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TemporalFusionLayer(keras.layers.Layer):
    """
    Fuse a context-based forecast with an attention-based autoregressive forecast.

    This layer implements a temporal fusion mechanism that combines deep
    contextual understanding with dynamic autoregressive modeling. Two parallel
    forecasting pathways are intelligently blended using a learned fusion gate
    derived from the context tensor.

    The mathematical operations are:

    1. Attention:  alpha = sigmoid(Dense_att(context))
    2. Gate:       g = sigmoid(Dense_gate(context))
    3. Context:    f_c = Dense_ctx(context)
    4. AR path:    f_l = Dense_lag(sum(alpha * lags))
    5. Fusion:     output = (1 - g) * f_c + g * f_l

    **Architecture Overview:**

    .. code-block:: text

        Inputs: [context_tensor, lag_tensor]
                     │                │
                     │                ▼
                     │    ┌──────────────────────┐
                     │    │  Lag Projector (opt.) │
                     │    └──────────┬───────────┘
                     │               │
                     ▼               │
        ┌────────────────────┐       │
        │   Attention Gen.   │       │
        │  Dense → sigmoid   │       │
        └────────┬───────────┘       │
                 │  alpha            │
                 └───────┐           │
                         ▼           ▼
                    ┌─────────────────────┐
                    │  Weighted Sum:      │
                    │  sum(alpha * lags)  │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Lag Forecaster    │
                    │      Dense          │
                    └─────────┬───────────┘
                              │ f_lag
        context_tensor        │
             │                │
             ├────────────────┤
             │                │
             ▼                │
        ┌──────────────┐     │
        │ Context       │     │
        │ Forecaster    │     │
        │  Dense        │     │
        └──────┬───────┘     │
               │ f_ctx       │
               │             │
               ▼             ▼
             ┌───────────────────┐
        g ──►│  Fusion Gate      │
             │  (1-g)*f_ctx      │
             │   + g * f_lag     │
             └────────┬──────────┘
                      ▼
               Output: (batch, output_dim)

    :param output_dim: Dimensionality of the final output forecast. Must be positive.
    :type output_dim: int
    :param num_lags: Number of past time series values (lags) to consider. Must be positive.
    :type num_lags: int
    :param project_lags: If True, an internal Dense layer transforms the raw lag
        values into a richer feature space before attention is applied.
    :type project_lags: bool
    :param kernel_initializer: Initializer for the kernel weights of all sublayers.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for the bias vectors of all sublayers.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for the kernel weights of all sublayers.
    :type kernel_regularizer: str or keras.regularizers.Regularizer, optional
    :param bias_regularizer: Optional regularizer for the bias vectors of all sublayers.
    :type bias_regularizer: str or keras.regularizers.Regularizer, optional
    :param activity_regularizer: Optional regularizer function for the output.
    :type activity_regularizer: str or keras.regularizers.Regularizer, optional
    :param kwargs: Additional keyword arguments for the Layer parent class.

    :raises ValueError: If ``output_dim`` or ``num_lags`` is not a positive integer.
    """

    def __init__(
        self,
        output_dim: int,
        num_lags: int,
        project_lags: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the TemporalFusionLayer.

        :param output_dim: Dimensionality of the final output forecast.
        :type output_dim: int
        :param num_lags: Number of lag values to consider.
        :type num_lags: int
        :param project_lags: Whether to project lags through a Dense layer.
        :type project_lags: bool
        :param kernel_initializer: Initializer for kernel weights.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param bias_initializer: Initializer for bias vectors.
        :type bias_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Optional regularizer for kernel weights.
        :type kernel_regularizer: str or keras.regularizers.Regularizer, optional
        :param bias_regularizer: Optional regularizer for bias vectors.
        :type bias_regularizer: str or keras.regularizers.Regularizer, optional
        :param activity_regularizer: Optional regularizer for the output.
        :type activity_regularizer: str or keras.regularizers.Regularizer, optional
        :param kwargs: Additional keyword arguments for the Layer parent class.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(num_lags, int) or num_lags <= 0:
            raise ValueError(f"num_lags must be a positive integer, got {num_lags}")

        # Store configuration parameters
        self.output_dim = output_dim
        self.num_lags = num_lags
        self.project_lags = project_lags
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # CREATE all sub-layers in __init__ (modern pattern)
        # --- Control Pathway Sublayers (driven by context) ---
        self.attention_generator = keras.layers.Dense(
            units=self.num_lags,
            activation='sigmoid',
            name='attention_generator',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        self.gate_generator = keras.layers.Dense(
            units=self.output_dim,
            activation='sigmoid',
            name='gate_generator',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # --- Contextual Forecast Pathway Sublayer ---
        self.context_forecaster = keras.layers.Dense(
            units=self.output_dim,
            name='context_forecaster',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # --- Autoregressive Forecast Pathway Sublayers ---
        if self.project_lags:
            # Optional layer to enrich the lag features
            self.lag_projector = keras.layers.Dense(
                units=self.num_lags,
                activation='relu',
                name='lag_projector',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )
        else:
            self.lag_projector = None

        # This layer creates the final AR forecast from the weighted sum
        self.lag_forecaster = keras.layers.Dense(
            units=self.output_dim,
            name='lag_forecaster',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        logger.debug(f"TemporalFusionLayer initialized with output_dim={output_dim}, num_lags={num_lags}")

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build the layer weights and sublayers based on input shape.

        Explicitly builds each sub-layer for robust serialization.

        :param input_shape: A list of two tuples representing the shapes of
            the context tensor and lag tensor inputs.
        :type input_shape: list of tuple

        :raises ValueError: If ``input_shape`` is not a list of two tensors.
        :raises ValueError: If the lag tensor's last dimension does not match ``num_lags``.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "TemporalFusionLayer expects a list of two inputs: "
                "[context_tensor, lag_tensor]. "
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

        # Build sub-layers in order they'll be used
        # All context-driven layers use the same context shape
        self.attention_generator.build(context_shape)
        self.gate_generator.build(context_shape)
        self.context_forecaster.build(context_shape)

        # Lag processing layers
        if self.lag_projector is not None:
            self.lag_projector.build(lag_shape)

        # The lag_forecaster receives a weighted sum, which is (batch_size, 1)
        weighted_sum_shape = (context_shape[0], 1)
        self.lag_forecaster.build(weighted_sum_shape)

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug(f"TemporalFusionLayer built with context_shape={context_shape}, lag_shape={lag_shape}")

    def call(self, inputs: List[keras.KerasTensor], training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass through the temporal fusion mechanism.

        :param inputs: A list containing two tensors: ``inputs[0]`` is the
            context tensor of shape ``(batch_size, context_dim)`` and
            ``inputs[1]`` is the lag tensor of shape ``(batch_size, num_lags)``.
        :type inputs: list of keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: bool, optional
        :return: Fused forecast tensor of shape ``(batch_size, output_dim)``.
        :rtype: keras.KerasTensor

        :raises ValueError: If ``inputs`` is not a list of two tensors.
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"Expected a list of two inputs, got {type(inputs)} with length {len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
            )

        context_tensor, lag_tensor = inputs

        # --- Pathway 1: Contextual Forecast ---
        context_forecast = self.context_forecaster(context_tensor, training=training)

        # --- Pathway 2: Autoregressive Forecast ---
        # Generate attention weights and the fusion gate from the context
        attention_weights = self.attention_generator(context_tensor, training=training)
        fusion_gate = self.gate_generator(context_tensor, training=training)

        # Optionally project lags into a richer feature space
        if self.lag_projector is not None:
            processed_lags = self.lag_projector(lag_tensor, training=training)
        else:
            processed_lags = lag_tensor

        # Calculate the weighted sum of lags (attention mechanism)
        weighted_sum = ops.sum(attention_weights * processed_lags, axis=-1, keepdims=True)
        lag_forecast = self.lag_forecaster(weighted_sum, training=training)

        # --- Pathway 3: Fusion ---
        # Blend the two forecasts using the learned gate
        final_forecast = (
            (1.0 - fusion_gate) * context_forecast +
            fusion_gate * lag_forecast
        )

        # Apply activity regularization if specified
        if self.activity_regularizer is not None:
            self.add_loss(self.activity_regularizer(final_forecast))

        return final_forecast

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: A list of two tuples representing the input shapes.
        :type input_shape: list of tuple
        :return: Output shape tuple.
        :rtype: tuple

        :raises ValueError: If ``input_shape`` is not a list of two tuples.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "Expected input_shape to be a list of two tuples, "
                f"got {type(input_shape)} with length {len(input_shape) if hasattr(input_shape, '__len__') else 'unknown'}"
            )

        context_shape, _ = input_shape

        # Convert to list for manipulation, then back to tuple
        output_shape = list(context_shape)
        output_shape[-1] = self.output_dim

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "num_lags": self.num_lags,
            "project_lags": self.project_lags,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
