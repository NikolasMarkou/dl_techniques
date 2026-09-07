"""FiLMLayer, a Feature-wise Linear Modulation layer.

FiLMLayer is a Keras layer that takes a list of two tensors, a content
tensor and a style vector, and returns the content tensor with a
per-channel scale and shift applied:

    output = content * (scale_factor + gamma) + beta

Gamma and beta come from separate Dense projections of the style vector,
each with its own width, activation, initializer and constraint. Layer
normalization and dropout can run on the style vector first.
`modulation_mode` decides which projections are built at all, so
'multiplicative' drops beta and 'additive' drops both gamma and
`scale_factor`. Inputs are passed as `[content, style]`. The content
channel count must be static. `epsilon` reaches only the optional layer
normalization. A `gamma_units` or `beta_units` that differs from the
channel count adds a second linear Dense to return the projection to the
channel width; that extra Dense takes no bias, regularizer or constraint.

References:
    - Perez et al., 2018. FiLM: Visual reasoning with a general conditioning
      layer. (https://doi.org/10.1609/aaai.v32i1.11671)
    - De Vries et al., 2017. Modulating early visual processing by language.
      (https://papers.nips.cc/paper/7237)
    - Dumoulin et al., 2018. Feature-wise transformations.
      (https://doi.org/10.23915/distill.00011)
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.regularization.film")
class FiLMLayer(keras.layers.Layer):
    """
    Modulate a content tensor with per-channel scale and shift from a style vector.

    Projects the style vector to gamma and beta, then applies
    ``output = content * (scale_factor + gamma) + beta``. ``modulation_mode``
    selects which half runs: 'multiplicative' applies only the scale,
    'additive' applies only the shift, and 'both' applies the scale first.
    The projection for a disabled half is never created.

    Architecture:

    .. code-block:: text

        content [B, ..., C]                     style [B, S]
              │                                       │
              │                                       ▼
              │                             ┌───────────────────┐
              │                             │ layer norm        │ (optional)
              │                             └───────────────────┘
              │                                       │
              │                                       ▼
              │                             ┌───────────────────┐
              │                             │ dropout           │ (optional)
              │                             └───────────────────┘
              │                                       │ [B, S]
              │                             ┌─────────┴─────────┐
              │                             ▼                   ▼
              │                       ┌──────────┐        ┌──────────┐
              │                       │ gamma    │        │ beta     │
              │                       │ dense    │        │ dense    │
              │                       └──────────┘        └──────────┘
              │                             │ [B, gu]           │ [B, bu]
              │                             ▼                   ▼
              │                       ┌──────────┐        ┌──────────┐
              │                       │ channel  │        │ channel  │ (opt)
              │                       │ dense    │        │ dense    │
              │                       └──────────┘        └──────────┘
              │                             │ [B, C]            │ [B, C]
              ▼                             │                   │
        ┌─────────────────────┐             │                   │
        │ * (scale + gamma)   │◄────────────┘                   │
        └─────────────────────┘  ('multiplicative', 'both')     │
              │ [B, ..., C]                                     │
              ▼                                                 │
        ┌─────────────────────┐                                 │
        │ + beta              │◄────────────────────────────────┘
        └─────────────────────┘  ('additive', 'both')
              │
              ▼
        output [B, ..., C]

    In 'additive' mode neither gamma nor ``scale_factor`` is applied.

    Broadcast shape:

    .. code-block:: text

        content rank 4  ->  gamma, beta reshaped to [B, 1, 1, C]
        content rank 3  ->  gamma, beta reshaped to [B, 1, C]
        content rank 2  ->  gamma, beta reshaped to [B, C]

    The batch entry is read from the projected tensor's static shape.

    Input shape:
        List of two tensors. Content ``(batch, ..., channels)`` with a static
        channel count, and style ``(batch, style_dim)``.

    Output shape:
        Same as the content tensor.

    :param gamma_units: Output units for gamma projection. If None, uses content
        channels.
    :type gamma_units: Optional[int]
    :param beta_units: Output units for beta projection. If None, uses content
        channels.
    :type beta_units: Optional[int]
    :param gamma_activation: Activation function for gamma projection. Defaults to
        'tanh'.
    :type gamma_activation: Union[str, Callable]
    :param beta_activation: Activation function for beta projection. Defaults to
        'linear'.
    :type beta_activation: Union[str, Callable]
    :param use_bias: Whether to use bias in the gamma and beta projections. The
        channel projections never use bias.
    :type use_bias: bool
    :param scale_factor: Base scaling factor applied before gamma modulation.
        Ignored when ``modulation_mode`` is 'additive'. Defaults to 1.0.
    :type scale_factor: float
    :param projection_dropout_rate: Dropout rate applied to style vector before
        projection. A value of 0.0 creates no dropout layer. Defaults to 0.0.
    :type projection_dropout_rate: float
    :param use_layer_norm: Whether to apply LayerNormalization to style vector.
        Defaults to False.
    :type use_layer_norm: bool
    :param gamma_kernel_initializer: Initializer for gamma projection weights.
    :type gamma_kernel_initializer: Union[str, keras.initializers.Initializer]
    :param beta_kernel_initializer: Initializer for beta projection weights.
    :type beta_kernel_initializer: Union[str, keras.initializers.Initializer]
    :param gamma_bias_initializer: Initializer for gamma projection bias.
    :type gamma_bias_initializer: Union[str, keras.initializers.Initializer]
    :param beta_bias_initializer: Initializer for beta projection bias.
    :type beta_bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the gamma and beta projection weights.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the gamma and beta projection biases.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activity_regularizer: Regularizer for the gamma and beta projection
        outputs. It is also assigned to the base Layer slot of the same name.
    :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param gamma_constraint: Constraint for gamma projection weights.
    :type gamma_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param beta_constraint: Constraint for beta projection weights.
    :type beta_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param modulation_mode: Strategy for applying modulation ('multiplicative',
        'additive', 'both'). Defaults to 'both'.
    :type modulation_mode: Literal['multiplicative', 'additive', 'both']
    :param epsilon: Epsilon for the optional style LayerNormalization. Has no
        effect when ``use_layer_norm`` is False. Defaults to 1e-8.
    :type epsilon: float
    :param kwargs: Additional arguments for Layer base class.

    :ivar num_channels: Content channel count, read from the content shape in
        ``build()``.
    :vartype num_channels: Optional[int]

    :raises ValueError: If ``projection_dropout_rate`` is outside ``[0, 1)``.
    :raises ValueError: If ``modulation_mode`` is not one of the three names.
    :raises ValueError: If ``epsilon`` is not positive.
    """

    def __init__(
            self,
            gamma_units: Optional[int] = None,
            beta_units: Optional[int] = None,
            gamma_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'tanh',
            beta_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'linear',
            use_bias: bool = True,
            scale_factor: float = 1.0,
            projection_dropout_rate: float = 0.0,
            use_layer_norm: bool = False,
            gamma_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            beta_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            gamma_bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            beta_bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            gamma_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
            beta_constraint: Optional[Union[str, keras.constraints.Constraint]] = None,
            modulation_mode: Literal['multiplicative', 'additive', 'both'] = 'both',
            epsilon: float = 1e-8,
            **kwargs: Any
    ) -> None:
        """Store the configuration and create the style-preprocessing sub-layers."""
        super().__init__(**kwargs)

        self.gamma_units = gamma_units
        self.beta_units = beta_units
        self.gamma_activation = gamma_activation
        self.beta_activation = beta_activation
        self.use_bias = use_bias
        self.scale_factor = scale_factor
        self.projection_dropout_rate = projection_dropout_rate
        self.use_layer_norm = use_layer_norm
        self.gamma_kernel_initializer = keras.initializers.get(gamma_kernel_initializer)
        self.beta_kernel_initializer = keras.initializers.get(beta_kernel_initializer)
        self.gamma_bias_initializer = keras.initializers.get(gamma_bias_initializer)
        self.beta_bias_initializer = keras.initializers.get(beta_bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.modulation_mode = modulation_mode
        self.epsilon = epsilon

        if projection_dropout_rate < 0.0 or projection_dropout_rate >= 1.0:
            raise ValueError(f"projection_dropout_rate must be in [0, 1), got {projection_dropout_rate}")

        if modulation_mode not in ['multiplicative', 'additive', 'both']:
            raise ValueError(f"modulation_mode must be one of ['multiplicative', 'additive', 'both'], "
                             f"got {modulation_mode}")

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # The projections need the content channel count, so build() creates them.
        self.gamma_projection: Optional[keras.layers.Dense] = None
        self.beta_projection: Optional[keras.layers.Dense] = None
        self.gamma_channel_projection: Optional[keras.layers.Dense] = None
        self.beta_channel_projection: Optional[keras.layers.Dense] = None
        self.layer_norm: Optional[keras.layers.LayerNormalization] = None
        self.dropout: Optional[keras.layers.Dropout] = None
        self.num_channels: Optional[int] = None

        # Norm and dropout keep the style shape, so they need nothing from build().
        self._create_sublayers()

    def _create_sublayers(self) -> None:
        """Create the optional style normalization and dropout layers."""
        if self.use_layer_norm:
            self.layer_norm = keras.layers.LayerNormalization(
                epsilon=self.epsilon,
                name=f"{self.name}_style_norm"
            )

        if self.projection_dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                self.projection_dropout_rate,
                name=f"{self.name}_style_dropout"
            )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer's weights and sub-layers.

        :param input_shape: List of two shapes: [content_shape, style_shape].
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If ``input_shape`` is not a list of two shapes.
        :raises ValueError: If the content channel count is ``None``.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                f"FiLMLayer expects input_shape to be a list of 2 shapes, "
                f"got {type(input_shape)} with length {len(input_shape) if isinstance(input_shape, list) else 'N/A'}"
            )

        content_shape, style_shape = input_shape
        self.num_channels = content_shape[-1]

        if self.num_channels is None:
            raise ValueError("Content tensor must have a known number of channels")

        gamma_proj_units = self.gamma_units or self.num_channels
        beta_proj_units = self.beta_units or self.num_channels

        if self.modulation_mode in ['multiplicative', 'both']:
            self.gamma_projection = keras.layers.Dense(
                gamma_proj_units,
                activation=self.gamma_activation,
                use_bias=self.use_bias,
                kernel_initializer=self.gamma_kernel_initializer,
                bias_initializer=self.gamma_bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.gamma_constraint,
                name=f"{self.name}_gamma_projection"
            )

            # Gamma has to reach the channel width before it can broadcast.
            if gamma_proj_units != self.num_channels:
                self.gamma_channel_projection = keras.layers.Dense(
                    self.num_channels,
                    activation='linear',
                    use_bias=False,
                    kernel_initializer='glorot_uniform',
                    name=f"{self.name}_gamma_channel_proj"
                )

        if self.modulation_mode in ['additive', 'both']:
            self.beta_projection = keras.layers.Dense(
                beta_proj_units,
                activation=self.beta_activation,
                use_bias=self.use_bias,
                kernel_initializer=self.beta_kernel_initializer,
                bias_initializer=self.beta_bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.beta_constraint,
                name=f"{self.name}_beta_projection"
            )

            # Beta has to reach the channel width before it can broadcast.
            if beta_proj_units != self.num_channels:
                self.beta_channel_projection = keras.layers.Dense(
                    self.num_channels,
                    activation='linear',
                    use_bias=False,
                    kernel_initializer='glorot_uniform',
                    name=f"{self.name}_beta_channel_proj"
                )

        # Explicit builds so every weight exists before a load_weights call.
        if self.layer_norm is not None:
            self.layer_norm.build(style_shape)

        if self.dropout is not None:
            self.dropout.build(style_shape)

        # Norm and dropout preserve the shape, so both projections take it as is.
        processed_style_shape = style_shape

        if self.gamma_projection is not None:
            self.gamma_projection.build(processed_style_shape)
            if self.gamma_channel_projection is not None:
                gamma_shape = self.gamma_projection.compute_output_shape(processed_style_shape)
                self.gamma_channel_projection.build(gamma_shape)

        if self.beta_projection is not None:
            self.beta_projection.build(processed_style_shape)
            if self.beta_channel_projection is not None:
                beta_shape = self.beta_projection.compute_output_shape(processed_style_shape)
                self.beta_channel_projection.build(beta_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the configurable FiLM transformation.

        :param inputs: List containing [content_tensor, style_vector].
        :type inputs: List[keras.KerasTensor]
        :param training: Whether the layer is in training mode. Reaches the
            optional style normalization and dropout.
        :type training: Optional[bool]
        :return: The modulated content tensor.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``inputs`` does not hold exactly two tensors.
        """
        if len(inputs) != 2:
            raise ValueError(f"FiLMLayer expects 2 inputs, got {len(inputs)}")

        content_tensor, style_vector = inputs

        processed_style = style_vector

        if self.layer_norm is not None:
            processed_style = self.layer_norm(processed_style, training=training)

        if self.dropout is not None:
            processed_style = self.dropout(processed_style, training=training)

        # In 'additive' mode the content passes through to the shift untouched.
        modulated_content = content_tensor

        if self.modulation_mode in ['multiplicative', 'both'] and self.gamma_projection is not None:
            gamma = self.gamma_projection(processed_style)

            if self.gamma_channel_projection is not None:
                gamma = self.gamma_channel_projection(gamma)

            # One singleton axis per non-batch, non-channel content axis.
            gamma_shape = [gamma.shape[0]] + [1] * (len(content_tensor.shape) - 2) + [gamma.shape[-1]]
            gamma = ops.reshape(gamma, gamma_shape)

            modulated_content = modulated_content * (self.scale_factor + gamma)

        if self.modulation_mode in ['additive', 'both'] and self.beta_projection is not None:
            beta = self.beta_projection(processed_style)

            if self.beta_channel_projection is not None:
                beta = self.beta_channel_projection(beta)

            # One singleton axis per non-batch, non-channel content axis.
            beta_shape = [beta.shape[0]] + [1] * (len(content_tensor.shape) - 2) + [beta.shape[-1]]
            beta = ops.reshape(beta, beta_shape)

            modulated_content = modulated_content + beta

        return modulated_content

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as content tensor).

        :param input_shape: List of [content_shape, style_shape].
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape matching content tensor.
        :rtype: Tuple[Optional[int], ...]
        """
        content_shape, _ = input_shape
        return content_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for layer serialization.

        :return: Configuration dictionary. Regularizers and constraints that
            were left unset serialize as ``None``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'gamma_units': self.gamma_units,
            'beta_units': self.beta_units,
            'gamma_activation': keras.activations.serialize(keras.activations.get(self.gamma_activation)),
            'beta_activation': keras.activations.serialize(keras.activations.get(self.beta_activation)),
            'use_bias': self.use_bias,
            'scale_factor': self.scale_factor,
            'projection_dropout_rate': self.projection_dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'gamma_kernel_initializer': keras.initializers.serialize(self.gamma_kernel_initializer),
            'beta_kernel_initializer': keras.initializers.serialize(self.beta_kernel_initializer),
            'gamma_bias_initializer': keras.initializers.serialize(self.gamma_bias_initializer),
            'beta_bias_initializer': keras.initializers.serialize(self.beta_bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer) if self.bias_regularizer else None,
            'activity_regularizer': keras.regularizers.serialize(
                self.activity_regularizer) if self.activity_regularizer else None,
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint) if self.gamma_constraint else None,
            'beta_constraint': keras.constraints.serialize(self.beta_constraint) if self.beta_constraint else None,
            'modulation_mode': self.modulation_mode,
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
