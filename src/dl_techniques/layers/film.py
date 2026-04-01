"""
Feature-wise Linear Modulation (FiLM) Layer.

Applies learned affine transformations to feature maps based on conditioning
vectors, enabling conditional generation, style transfer, and multi-modal
learning. The FiLM transformation is FiLM(x, z) = (lambda + gamma) * x + beta,
where gamma, beta = f(z) are learned projections from the conditioning vector z.

References:
    1. Perez, E., et al. (2018). "FiLM: Visual reasoning with a general
       conditioning layer." AAAI. https://doi.org/10.1609/aaai.v32i1.11671
    2. De Vries, H., et al. (2017). "Modulating early visual processing by
       language." NeurIPS. https://papers.nips.cc/paper/7237
    3. Dumoulin, V., et al. (2018). "Feature-wise transformations." Distill.
       https://doi.org/10.23915/distill.00011
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Literal

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FiLMLayer(keras.layers.Layer):
    """
    Highly configurable Feature-wise Linear Modulation (FiLM) Layer.

    Applies an affine transformation to content tensors based on style/condition
    vectors. Learns projections from style vectors to generate per-channel scaling
    (gamma) and shifting (beta) parameters: output = content * (scale_factor + gamma)
    + beta. Supports multiplicative-only, additive-only, or combined modulation modes,
    with optional layer normalization and dropout on the conditioning vector.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────┐  ┌──────────────────────┐
        │Content [B, H, W, C]  │  │Style Vector [B, S]   │
        └──────────┬───────────┘  └──────────┬───────────┘
                   │                         ▼
                   │              ┌──────────────────────┐
                   │              │ LayerNorm (optional)  │
                   │              │ Dropout (optional)    │
                   │              └──────────┬───────────┘
                   │                 ┌───────┴───────┐
                   │                 ▼               ▼
                   │          ┌───────────┐   ┌───────────┐
                   │          │Dense_gamma│   │Dense_beta │
                   │          │(act, bias)│   │(act, bias)│
                   │          └─────┬─────┘   └─────┬─────┘
                   │                ▼               ▼
                   │          Gamma [B,1,1,C]  Beta [B,1,1,C]
                   │                │               │
                   ▼                ▼               ▼
        ┌─────────────────────────────────────────────────┐
        │  Output = Content * (lambda + Gamma) + Beta     │
        └──────────────────────┬──────────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────────┐
        │  Output [B, H, W, C]                            │
        └─────────────────────────────────────────────────┘

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
    :param use_bias: Whether to use bias in projection layers.
    :type use_bias: bool
    :param scale_factor: Base scaling factor applied before gamma modulation.
        Defaults to 1.0.
    :type scale_factor: float
    :param projection_dropout: Dropout rate applied to style vector before
        projection. Defaults to 0.0.
    :type projection_dropout: float
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
    :param kernel_regularizer: Regularizer for projection layer weights.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for projection layer biases.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activity_regularizer: Regularizer for projection layer outputs.
    :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param gamma_constraint: Constraint for gamma projection weights.
    :type gamma_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param beta_constraint: Constraint for beta projection weights.
    :type beta_constraint: Optional[Union[str, keras.constraints.Constraint]]
    :param modulation_mode: Strategy for applying modulation ('multiplicative',
        'additive', 'both'). Defaults to 'both'.
    :type modulation_mode: Literal['multiplicative', 'additive', 'both']
    :param epsilon: Small constant for numerical stability. Defaults to 1e-8.
    :type epsilon: float
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            gamma_units: Optional[int] = None,
            beta_units: Optional[int] = None,
            gamma_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'tanh',
            beta_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'linear',
            use_bias: bool = True,
            scale_factor: float = 1.0,
            projection_dropout: float = 0.0,
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
        """Initialize highly configurable FiLM layer."""
        super().__init__(**kwargs)

        # Store all configuration parameters
        self.gamma_units = gamma_units
        self.beta_units = beta_units
        self.gamma_activation = gamma_activation
        self.beta_activation = beta_activation
        self.use_bias = use_bias
        self.scale_factor = scale_factor
        self.projection_dropout = projection_dropout
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

        # Validate inputs
        if projection_dropout < 0.0 or projection_dropout >= 1.0:
            raise ValueError(f"projection_dropout must be in [0, 1), got {projection_dropout}")

        if modulation_mode not in ['multiplicative', 'additive', 'both']:
            raise ValueError(f"modulation_mode must be one of ['multiplicative', 'additive', 'both'], "
                             f"got {modulation_mode}")

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Sub-layers to be created in build()
        self.gamma_projection: Optional[keras.layers.Dense] = None
        self.beta_projection: Optional[keras.layers.Dense] = None
        self.gamma_channel_projection: Optional[keras.layers.Dense] = None
        self.beta_channel_projection: Optional[keras.layers.Dense] = None
        self.layer_norm: Optional[keras.layers.LayerNormalization] = None
        self.dropout: Optional[keras.layers.Dropout] = None
        self.num_channels: Optional[int] = None

        # Create sub-layers during initialization (Modern Keras 3 pattern)
        self._create_sublayers()

    def _create_sublayers(self) -> None:
        """Create all sub-layers during initialization."""
        # Optional preprocessing layers
        if self.use_layer_norm:
            self.layer_norm = keras.layers.LayerNormalization(
                epsilon=self.epsilon,
                name=f"{self.name}_style_norm"
            )

        if self.projection_dropout > 0.0:
            self.dropout = keras.layers.Dropout(
                self.projection_dropout,
                name=f"{self.name}_style_dropout"
            )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer's weights and sub-layers.

        :param input_shape: List of two shapes: [content_shape, style_shape].
        :type input_shape: List[Tuple[Optional[int], ...]]
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

        # Determine projection dimensions
        gamma_proj_units = self.gamma_units or self.num_channels
        beta_proj_units = self.beta_units or self.num_channels

        # Create main projection layers
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

            # Additional projection if units don't match channels
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

            # Additional projection if units don't match channels
            if beta_proj_units != self.num_channels:
                self.beta_channel_projection = keras.layers.Dense(
                    self.num_channels,
                    activation='linear',
                    use_bias=False,
                    kernel_initializer='glorot_uniform',
                    name=f"{self.name}_beta_channel_proj"
                )

        # Build all sub-layers explicitly for robust serialization
        if self.layer_norm is not None:
            self.layer_norm.build(style_shape)

        if self.dropout is not None:
            self.dropout.build(style_shape)

        # Build projection layers
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
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: The modulated content tensor.
        :rtype: keras.KerasTensor
        """
        if len(inputs) != 2:
            raise ValueError(f"FiLMLayer expects 2 inputs, got {len(inputs)}")

        content_tensor, style_vector = inputs

        # Preprocess style vector
        processed_style = style_vector

        if self.layer_norm is not None:
            processed_style = self.layer_norm(processed_style, training=training)

        if self.dropout is not None:
            processed_style = self.dropout(processed_style, training=training)

        # Initialize modulated content
        modulated_content = content_tensor

        # Apply multiplicative modulation
        if self.modulation_mode in ['multiplicative', 'both'] and self.gamma_projection is not None:
            gamma = self.gamma_projection(processed_style)

            # Optional channel projection
            if self.gamma_channel_projection is not None:
                gamma = self.gamma_channel_projection(gamma)

            # Reshape for broadcasting: (batch, 1, 1, ..., channels)
            gamma_shape = [gamma.shape[0]] + [1] * (len(content_tensor.shape) - 2) + [gamma.shape[-1]]
            gamma = ops.reshape(gamma, gamma_shape)

            # Apply multiplicative modulation
            modulated_content = modulated_content * (self.scale_factor + gamma)

        # Apply additive modulation
        if self.modulation_mode in ['additive', 'both'] and self.beta_projection is not None:
            beta = self.beta_projection(processed_style)

            # Optional channel projection
            if self.beta_channel_projection is not None:
                beta = self.beta_channel_projection(beta)

            # Reshape for broadcasting: (batch, 1, 1, ..., channels)
            beta_shape = [beta.shape[0]] + [1] * (len(content_tensor.shape) - 2) + [beta.shape[-1]]
            beta = ops.reshape(beta, beta_shape)

            # Apply additive modulation
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

        :return: Configuration dictionary.
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
            'projection_dropout': self.projection_dropout,
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
