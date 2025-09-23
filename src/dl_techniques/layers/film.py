"""
Feature-wise Linear Modulation (FiLM) Layer

This module provides a highly configurable implementation of Feature-wise Linear Modulation,
a powerful technique for conditional neural networks that enables dynamic feature modulation
based on auxiliary conditioning information.

Feature-wise Linear Modulation applies learned affine transformations to feature maps based
on conditioning vectors, enabling sophisticated conditional generation, style transfer, and
multi-modal learning applications. The technique has proven particularly effective in visual
reasoning, conditional image synthesis, and neural style transfer tasks.

**Key Features:**
- **Multi-modal modulation**: Supports multiplicative, additive, or combined modulation strategies
- **Flexible architecture**: Configurable projection dimensions, activations, and preprocessing
- **Advanced regularization**: Comprehensive support for weight constraints and regularizers
- **Numerical robustness**: Configurable epsilon and scale factors for stable training
- **Production ready**: Full serialization support and comprehensive error handling

**Mathematical Foundation:**
The FiLM transformation applies learnable per-channel affine transformations:

    γᵢ, βᵢ = fᵢ(z)  for i ∈ {1, ..., C}

    FiLM(x, z) = γ ⊙ x + β

where x ∈ ℝᴮˣᴴˣᵂˣᶜ is the content tensor, z ∈ ℝᴮˣˢ is the conditioning vector,
f: ℝˢ → ℝᶜˣ² is the learned projection function, and ⊙ denotes element-wise multiplication.

This implementation extends the basic formulation with:
- Configurable base scaling: FiLM(x, z) = (λ + γ) ⊙ x + β
- Modal flexibility: Support for γ-only, β-only, or combined modulation
- Style preprocessing: Optional normalization and dropout on conditioning vectors

**References:**
1. Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018).
   "FiLM: Visual reasoning with a general conditioning layer."
   Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).
   https://doi.org/10.1609/aaai.v32i1.11671

2. De Vries, H., Strub, F., Mary, J., Larochelle, H., Pietquin, O., & Courville, A. C. (2017).
   "Modulating early visual processing by language."
   Advances in Neural Information Processing Systems, 30.
   https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language

3. Dumoulin, V., Perez, E., Schucher, N., Strub, F., De Vries, H., Courville, A., & Bengio, Y. (2018).
   "Feature-wise transformations."
   Distill, 3(7), e11.
   https://doi.org/10.23915/distill.00011

4. Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2019).
   "Self-attention generative adversarial networks."
   International Conference on Machine Learning (ICML).
   [Shows FiLM usage in modern generative models]

5. Park, T., Liu, M. Y., Wang, T. C., & Zhu, J. Y. (2019).
   "Semantic image synthesis with spatially-adaptive normalization."
   Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
   https://doi.org/10.1109/CVPR.2019.00244
   [SPADE - Related spatially-adaptive modulation technique]

**Implementation Notes:**
- Built following Modern Keras 3 patterns for robust serialization
- Supports arbitrary tensor shapes beyond 4D (e.g., 3D, 5D tensors)
- Memory efficient broadcasting for large feature maps
- Thread-safe and suitable for distributed training
- Comprehensive type hints and Sphinx-compatible documentation
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Literal

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FiLMLayer(keras.layers.Layer):
    """
    Highly configurable Feature-wise Linear Modulation (FiLM) Layer.

    Applies an affine transformation to content tensors based on style/condition vectors.
    This layer learns projections from style vectors to generate per-channel scaling (gamma)
    and shifting (beta) parameters for conditional feature modulation. Supports extensive
    customization of architectures, activations, regularization, and projection strategies.

    **Intent**: Enable sophisticated conditional generation and style transfer by modulating
    feature maps based on auxiliary condition vectors. Commonly used in conditional GANs,
    style transfer networks, and conditional image synthesis where content needs to be
    dynamically adjusted based on style or semantic conditions.

    **Architecture**:
    ```
    Content Tensor(B, H, W, C) + Style Vector(B, S)
                        ↓
    Style → [Optional: Norm] → Dense_γ(units, activation) → Gamma(B, 1, 1, C)
         ↘ [Optional: Norm] → Dense_β(units, activation) → Beta(B, 1, 1, C)
                        ↓
    Output = Content * (scale_factor + Gamma) + Beta
    ```

    **Mathematical Operations**:
    1. **Gamma projection**: γ = f_γ(norm(style)) where f_γ is configurable projection
    2. **Beta projection**: β = f_β(norm(style)) where f_β is configurable projection
    3. **Modulation**: output = content ⊙ (λ + γ) + β, where λ is base scale factor

    Args:
        gamma_units: Output units for gamma projection. If None, uses content channels.
        beta_units: Output units for beta projection. If None, uses content channels.
        gamma_activation: Activation function for gamma projection. Can be string or callable.
        beta_activation: Activation function for beta projection. Can be string or callable.
        use_bias: Whether to use bias in projection layers.
        scale_factor: Base scaling factor applied before gamma modulation.
        projection_dropout: Dropout rate applied to style vector before projection.
        use_layer_norm: Whether to apply LayerNormalization to style vector.
        gamma_kernel_initializer: Initializer for gamma projection weights.
        beta_kernel_initializer: Initializer for beta projection weights.
        gamma_bias_initializer: Initializer for gamma projection bias.
        beta_bias_initializer: Initializer for beta projection bias.
        kernel_regularizer: Regularizer applied to projection layer weights.
        bias_regularizer: Regularizer applied to projection layer biases.
        activity_regularizer: Regularizer applied to projection layer outputs.
        gamma_constraint: Constraint applied to gamma projection weights.
        beta_constraint: Constraint applied to beta projection weights.
        modulation_mode: Strategy for applying modulation ('multiplicative', 'additive', 'both').
        epsilon: Small constant for numerical stability.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        List of two tensors:
        - content_tensor: (batch_size, height, width, channels) or (batch_size, ..., channels)
        - style_vector: (batch_size, style_features)

    Output shape:
        Same as content_tensor: (batch_size, height, width, channels) or (batch_size, ..., channels)

    Attributes:
        gamma_projection: Dense layer for scaling parameters
        beta_projection: Dense layer for shifting parameters
        layer_norm: Optional LayerNormalization for style preprocessing
        dropout: Optional Dropout for style regularization
        num_channels: Number of channels being modulated

    Example:
        ```python
        # Basic usage
        film = FiLMLayer()
        content = keras.random.normal((2, 64, 64, 128))
        style = keras.random.normal((2, 256))
        output = film([content, style])

        # Highly configured version
        film = FiLMLayer(
            gamma_activation='tanh',
            beta_activation='linear',
            use_layer_norm=True,
            projection_dropout=0.1,
            scale_factor=1.5,
            kernel_regularizer='l2',
            modulation_mode='both'
        )

        # Different projection dimensions
        film = FiLMLayer(
            gamma_units=64,  # Project to 64 units first
            beta_units=64,
            gamma_activation='gelu'
        )
        ```

    Note:
        When gamma_units or beta_units differ from content channels, an additional
        projection layer maps to the correct channel dimension. This enables
        dimensionality reduction or expansion in the modulation pathway.
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
        """
        Build the layer's weights and sub-layers.

        Args:
            input_shape: List of two shapes: [content_shape, style_shape].
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
        """
        Apply the configurable FiLM transformation.

        Args:
            inputs: List containing [content_tensor, style_vector].
            training: Whether the layer is in training mode.

        Returns:
            The modulated content tensor.
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
        """Compute output shape (same as content tensor)."""
        content_shape, _ = input_shape
        return content_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for layer serialization."""
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
