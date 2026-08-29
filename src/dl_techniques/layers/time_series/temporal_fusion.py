"""
Blend a context forecast with an attention-weighted lag forecast.

This module holds ``TemporalFusionLayer``. The layer mixes two forecasts of
the same target. One comes from a learned context vector, the other from
recent past values of the series. A learned gate decides how much of each to
use, per output unit and per sample.

The two pathways are:

1.  **Contextual pathway.** A Dense layer maps the ``context_tensor`` straight
    to a forecast. The context is usually the output of a recurrent or
    attentional encoder. This path carries non-linear structure and exogenous
    features.

2.  **Autoregressive pathway.** Attention weights are generated from the
    context, one weight per lag. Those weights scale the ``lag_tensor``, the
    result is summed over lags, and a Dense layer turns that sum into a
    forecast. The weights come from the context, so the layer can change which
    lags matter from sample to sample.

The fusion gate is a third Dense head on the context, with a sigmoid. It
produces one value per output unit, not a single scalar. The output is a
per-unit interpolation between the two forecasts.

**Foundational Mathematics:**

Given a context vector ``c`` and a lag vector ``l = [l_1, ..., l_n]``:

1.  Context forecast, a direct projection of the context.
        f_context = W_c * c + b_c

2.  Autoregressive forecast, a contextually weighted sum of lags.
        alpha = sigmoid(W_alpha * c + b_alpha)
        s     = sum_i alpha_i * l_i
        f_lag = W_l * s + b_l

3.  Fusion gate, a vector of interpolation coefficients.
        g = sigmoid(W_g * c + b_g)

4.  Output, the gated combination of the two forecasts.
        output = (1 - g) * f_context + g * f_lag

Note that ``s`` is a single scalar per sample. Every lag is squeezed through
that one number before ``W_l`` sees it, so ``lag_forecaster``'s kernel has
shape ``(1, output_dim)``.

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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.temporal_fusion")
class TemporalFusionLayer(keras.layers.Layer):
    """
    Blend a context forecast with an attention-weighted lag forecast.

    Three Dense heads read the context tensor: one makes a forecast, one makes
    per-lag attention weights, one makes the fusion gate. A fourth Dense turns
    the attention-weighted lag sum into the second forecast. The gate then
    interpolates between the two.

    The steps are:

    1. Attention:  alpha = sigmoid(Dense_att(context))     [B, num_lags]
    2. Gate:       g     = sigmoid(Dense_gate(context))    [B, output_dim]
    3. Context:    f_c   = Dense_ctx(context)              [B, output_dim]
    4. AR path:    f_l   = Dense_lag(sum(alpha * lags))    [B, output_dim]
    5. Fusion:     out   = (1 - g) * f_c + g * f_l         [B, output_dim]

    The gate is a vector of length ``output_dim``, not a scalar. Each output
    unit gets its own mixing coefficient.

    **Architecture Overview:**

    .. code-block:: text

        context [B, Dc]                 lags [B, L]
             │                               │
             ├───────────┬────────────┐      │
             ▼           ▼            ▼      ▼
        ┌─────────┐ ┌─────────┐ ┌──────────────────┐
        │ Context │ │ Gate    │ │ Lag Pathway      │
        │ Fcst    │ │ Gen     │ │ (detail below)   │
        │ Dense O │ │ Dense O │ └────────┬─────────┘
        └────┬────┘ │ sigmoid │          │
             │      └────┬────┘          │
             │ f_ctx     │ g             │ f_lag
             │ [B, O]    │ [B, O]        │ [B, O]
             ▼           ▼               ▼
        ┌─────────────────────────────────────────┐
        │   (1 - g) * f_ctx  +  g * f_lag         │
        └────────────────────┬────────────────────┘
                             ▼
                      Output: [B, O]

    ``Dc`` is the context width, ``L`` is ``num_lags``, ``O`` is
    ``output_dim``. The lag pathway reads both inputs: the lags themselves and
    the context that weights them.

    **Lag Pathway Detail:**

    .. code-block:: text

           context [B, Dc]             lags [B, L]
                  │                          │
                  │                          ▼
                  │           ┌────────────────────────┐
                  │           │ Lag Projector          │
                  │           │ Dense L, relu          │
                  │           │ (optional)             │
                  │           └──────────────┬─────────┘
                  ▼                          │ [B, L]
        ┌───────────────────┐                │
        │ Attention Gen     │                │
        │ Dense L, sigmoid  │                │
        └─────────┬─────────┘                │
                  │ alpha [B, L]             │
                  └───────────►( * )◄────────┘
                                 │
                                 ▼
                      sum(axis=-1, keepdims)
                                 │ [B, 1]
                                 ▼
                      ┌─────────────────────┐
                      │ Lag Forecaster      │
                      │ Dense O             │
                      └──────────┬──────────┘
                                 ▼
                           f_lag: [B, O]

    ``project_lags=False`` drops the Lag Projector and the raw lags are
    multiplied by ``alpha`` directly.

    Input shape:
        A list of two tensors. Context ``(batch, context_dim)`` and lags
        ``(batch, num_lags)``. Both must be at least 2D.

    Output shape:
        The context shape with its last axis replaced by ``output_dim``, so
        ``(batch, output_dim)`` for a 2D context.

    Example:
        .. code-block:: python

            layer = TemporalFusionLayer(output_dim=1, num_lags=7)
            y = layer([context, lags])

    :param output_dim: Width of the final forecast, and of the gate. Must be
        a positive int.
    :type output_dim: int
    :param num_lags: Number of past values in the lag tensor. Must be a
        positive int and must equal ``lag_tensor.shape[-1]``.
    :type num_lags: int
    :param project_lags: If ``True``, a Dense layer with ``relu`` maps the raw
        lags to ``num_lags`` features before attention is applied.
    :type project_lags: bool
    :param kernel_initializer: Initializer for the kernels of all sublayers.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for the biases of all sublayers.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for all sublayer kernels.
    :type kernel_regularizer: str or keras.regularizers.Regularizer, optional
    :param bias_regularizer: Optional regularizer for all sublayer biases.
    :type bias_regularizer: str or keras.regularizers.Regularizer, optional
    :param activity_regularizer: Optional regularizer on the output. See the
        note in ``call()`` before using it.
    :type activity_regularizer: str or keras.regularizers.Regularizer, optional
    :param kwargs: Additional keyword arguments for the Layer parent class.

    :raises ValueError: If ``output_dim`` or ``num_lags`` is not a positive
        integer.

    :ivar attention_generator: Dense head making per-lag weights from context.
    :vartype attention_generator: keras.layers.Dense
    :ivar gate_generator: Dense head making the ``output_dim``-wide gate.
    :vartype gate_generator: keras.layers.Dense
    :ivar context_forecaster: Dense head making the contextual forecast.
    :vartype context_forecaster: keras.layers.Dense
    :ivar lag_projector: Optional Dense on the raw lags, ``None`` when
        ``project_lags`` is ``False``.
    :vartype lag_projector: keras.layers.Dense or None
    :ivar lag_forecaster: Dense mapping the ``[B, 1]`` weighted lag sum to the
        autoregressive forecast.
    :vartype lag_forecaster: keras.layers.Dense
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
        Validate the arguments and create the five Dense sublayers.

        The sublayers are created here and left unbuilt. ``build()`` builds
        them. ``lag_projector`` is created only when ``project_lags`` is
        ``True``; otherwise it is ``None``.

        :param output_dim: Width of the final forecast and of the gate.
        :type output_dim: int
        :param num_lags: Number of lag values the layer expects.
        :type num_lags: int
        :param project_lags: Whether to run the raw lags through a Dense layer
            before applying attention.
        :type project_lags: bool
        :param kernel_initializer: Initializer for all sublayer kernels.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param bias_initializer: Initializer for all sublayer biases.
        :type bias_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Optional regularizer for sublayer kernels.
        :type kernel_regularizer: str or keras.regularizers.Regularizer, optional
        :param bias_regularizer: Optional regularizer for sublayer biases.
        :type bias_regularizer: str or keras.regularizers.Regularizer, optional
        :param activity_regularizer: Optional regularizer on the output.
        :type activity_regularizer: str or keras.regularizers.Regularizer, optional
        :param kwargs: Additional keyword arguments for the Layer parent class.

        :raises ValueError: If ``output_dim`` or ``num_lags`` is not a
            positive integer.
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
        Validate the two input shapes and build every sublayer.

        The three context-driven heads are built on ``context_shape``. The
        optional projector is built on ``lag_shape``. ``lag_forecaster`` is
        built on ``(batch, 1)``, because the weighted lag sum keeps one
        element on the last axis.

        Each sublayer is built explicitly so that a ``.keras`` weight restore
        finds every variable already materialized.

        :param input_shape: A list of two shape tuples, for the context tensor
            and the lag tensor.
        :type input_shape: list of tuple

        :raises ValueError: If ``input_shape`` is not a list of two shapes.
        :raises ValueError: If the context shape has fewer than 2 axes.
        :raises ValueError: If the lag shape's last axis is not ``num_lags``.
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

        logger.debug(f"TemporalFusionLayer built with context_shape={context_shape}, lag_shape={lag_shape}")

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor], training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Run both forecast pathways and blend them with the gate.

        Note on ``activity_regularizer``: this method calls ``add_loss`` on the
        output itself, and Keras 3 also applies the same regularizer in
        ``Layer.__call__`` (``keras/src/layers/layer.py:925-928``). With an
        ``activity_regularizer`` set, the penalty is therefore counted twice.
        Measured with ``L1(1.0)`` on a ``(2, 5)`` context and a ``(2, 4)`` lag
        tensor: ``len(layer.losses) == 2``, both entries ``2.6033520698547363``.
        This is a code defect, reported and left unfixed here.

        :param inputs: A list of two tensors. ``inputs[0]`` is the context of
            shape ``(batch_size, context_dim)``, ``inputs[1]`` is the lag
            tensor of shape ``(batch_size, num_lags)``.
        :type inputs: list of keras.KerasTensor
        :param training: Whether the layer runs in training mode. Forwarded to
            every Dense sublayer.
        :type training: bool, optional
        :return: Fused forecast of shape ``(batch_size, output_dim)``.
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
        Compute the output shape from the context shape alone.

        The lag shape is ignored. The context shape is returned with its last
        axis replaced by ``output_dim``.

        :param input_shape: A list of two shape tuples, for the context tensor
            and the lag tensor.
        :type input_shape: list of tuple
        :return: The context shape with ``output_dim`` as its last axis.
        :rtype: tuple

        :raises ValueError: If ``input_shape`` is not a list of two shapes.
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
        Return the constructor arguments needed to rebuild this layer.

        Initializers and regularizers are serialized to their dict form.

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
