"""
# Capsule Networks (CapsNet) Implementation

A comprehensive Keras-based implementation of Capsule Networks as proposed by
Sabour et al. in "Dynamic Routing Between Capsules" (2017).

## Architecture Overview

```
Input --> Conv Layers --> Primary Capsule --> CapsuleBlock
                                               |
                                               +--> RoutingCapsule
                                               |      |
                                               |      +--> Dynamic Routing
                                               |
                                               +--> (Optional)
                                                      |
                                                      +--> Dropout
                                                      |
                                                      +--> LayerNorm
```

## Core Components

### 1. Primary Capsule Layer
Transforms traditional CNN features into capsule vectors.

### 2. Routing Capsule Layer
Implements dynamic routing mechanism between capsule layers.

### 3. Capsule Block
A complete block combining routing capsules with regularization.

## Dynamic Routing Algorithm

The routing process iteratively refines the connection strengths between input
and output capsules through an agreement mechanism.

## References

Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules.
In Advances in Neural Information Processing Systems (pp. 3856-3866).
"""

import keras
from keras import ops
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .activations.squash import SquashLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PrimaryCapsule(keras.layers.Layer):
    """Primary Capsule Layer implementation.

    This layer implements the primary capsule layer which converts regular CNN features
    into capsule format. It consists of a convolutional layer followed by reshaping
    and squashing operations.

    **Intent**: Transform traditional convolutional feature maps into capsule representation,
    where each capsule encodes the instantiation parameters of a specific feature type.
    This is the bridge between standard CNN architectures and capsule-based processing.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    Conv2D(filters=num_capsules × dim_capsules, kernel_size, strides)
           ↓
    Reshape([batch, num_spatial_locations, num_capsules, dim_capsules])
           ↓
    Reshape([batch, num_spatial_locations × num_capsules, dim_capsules])
           ↓
    SquashLayer(axis=-1)  ← Non-linear activation preserving direction
           ↓
    Output(shape=[batch, num_capsules_total, dim_capsules])
    ```

    **Mathematical Operation**:
        ```
        conv_out = Conv2D(inputs)                    # Shape: [B, H', W', C']
        capsules = Reshape(conv_out)                 # Shape: [B, N, D]
        output = Squash(capsules, axis=-1)           # Shape: [B, N, D]
        ```

    Where:
    - B = batch_size
    - H', W' = spatial dimensions after convolution
    - C' = num_capsules × dim_capsules
    - N = num_spatial_locations × num_capsules (total capsules)
    - D = dim_capsules

    Args:
        num_capsules: Integer, number of primary capsules to create per spatial location.
            Must be positive. For example, 32 capsules means 32 different feature types
            are encoded at each spatial position.
        dim_capsules: Integer, dimension of each capsule's output vector. Must be positive.
            Typical values are 8 or 16. Higher dimensions allow capsules to encode more
            information about their detected features.
        kernel_size: Integer or tuple of integers, size of the convolutional kernel.
            Controls the receptive field size for capsule extraction.
        strides: Integer or tuple of integers, stride length for convolution.
            Controls spatial downsampling. Defaults to 1.
        padding: String, type of padding for convolution ('valid' or 'same').
            'valid' reduces spatial size, 'same' preserves it. Defaults to 'valid'.
        kernel_initializer: String or Initializer, initializer for the conv kernel.
            Defaults to 'he_normal'.
        kernel_regularizer: Optional regularizer function for the conv kernel.
            Helps prevent overfitting in the capsule extraction stage.
        use_bias: Boolean, whether to include bias terms in convolution. Defaults to True.
        squash_axis: Integer, axis along which to apply squashing operation.
            Should be -1 to squash along capsule dimension. Defaults to -1.
        squash_epsilon: Float, epsilon for numerical stability in squashing. Optional.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: ``(batch_size, height, width, channels)``

    Output shape:
        3D tensor with shape: ``(batch_size, num_spatial_capsules * num_capsules, dim_capsules)``
        where ``num_spatial_capsules`` depends on the input size and convolution parameters.

    Attributes:
        conv: Conv2D layer that extracts capsule features from input.
        squash_layer: SquashLayer that applies non-linear squashing activation.

    Example:
        ```python
        # Create primary capsules from feature maps
        primary_caps = PrimaryCapsule(
            num_capsules=32,
            dim_capsules=8,
            kernel_size=9,
            strides=2
        )

        # Process feature maps to capsules
        feature_maps = keras.random.normal((32, 20, 20, 256))
        capsules = primary_caps(feature_maps)
        print(capsules.shape)  # (32, 1152, 8) - 6*6*32 = 1152 capsules
        ```

    Raises:
        ValueError: If num_capsules or dim_capsules is not positive.
        ValueError: If padding is not 'valid' or 'same'.
        ValueError: If input shape is not 4D during build.

    Note:
        The number of output capsules depends on the spatial dimensions after convolution.
        With valid padding and stride > 1, spatial size decreases significantly.
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = 1,
        padding: str = "valid",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        squash_axis: int = -1,
        squash_epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if dim_capsules <= 0:
            raise ValueError(f"dim_capsules must be positive, got {dim_capsules}")
        if padding not in ["valid", "same"]:
            raise ValueError(f"padding must be one of 'valid' or 'same', got {padding}")

        # Store configuration
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.squash_axis = squash_axis
        self.squash_epsilon = squash_epsilon

        # CREATE sub-layers in __init__ (modern Keras 3 pattern)
        self.conv = keras.layers.Conv2D(
            filters=self.num_capsules * self.dim_capsules,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name="primary_conv"
        )

        self.squash_layer = SquashLayer(
            axis=self.squash_axis,
            epsilon=self.squash_epsilon,
            name="primary_squash"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of input tensor as tuple.
        """
        # Validate input shape
        if len(input_shape) != 4:  # [batch_size, height, width, channels]
            raise ValueError(f"Expected 4D input shape [batch, height, width, channels], got {input_shape}")

        # BUILD sub-layers explicitly (critical for serialization)
        self.conv.build(input_shape)

        # Compute conv output shape to build squash layer
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        h, w = conv_output_shape[1], conv_output_shape[2]
        num_spatial_capsules = h * w

        # Build squash layer with reshaped dimensions
        squash_input_shape = (
            input_shape[0],
            num_spatial_capsules * self.num_capsules,
            self.dim_capsules
        )
        self.squash_layer.build(squash_input_shape)

        super().build(input_shape)

        logger.info(f"Built PrimaryCapsule layer: {self.num_capsules} capsules, "
                   f"{self.dim_capsules} dimensions each")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for primary capsule layer.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels].
            training: Boolean, whether in training mode.

        Returns:
            Output tensor of shape [batch_size, num_capsules_total, dim_capsules].
        """
        batch_size = ops.shape(inputs)[0]

        # Apply convolution
        conv_output = self.conv(inputs, training=training)

        # Reshape to capsule format
        # Calculate dimensions based on the output of the convolutional layer
        h, w = ops.shape(conv_output)[1], ops.shape(conv_output)[2]
        num_spatial_capsules = h * w

        # Reshape to [batch_size, num_spatial_capsules * num_capsules, dim_capsules]
        capsules = ops.reshape(
            conv_output,
            [batch_size, num_spatial_capsules * self.num_capsules, self.dim_capsules]
        )

        # Apply squashing
        return self.squash_layer(capsules, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Shape of output tensor as tuple.
        """
        # Compute the shape after convolution
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        h, w = conv_output_shape[1], conv_output_shape[2]
        num_spatial_capsules = h * w

        # Shape after reshaping and squashing
        return (
            input_shape[0],  # batch size
            num_spatial_capsules * self.num_capsules,
            self.dim_capsules
        )

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "squash_axis": self.squash_axis,
            "squash_epsilon": self.squash_epsilon,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RoutingCapsule(keras.layers.Layer):
    """Capsule layer with dynamic routing between capsules.

    Implements the iterative dynamic routing mechanism that determines how lower-level
    capsules should contribute to higher-level capsules. This is the key innovation
    of Capsule Networks that replaces max-pooling with a more sophisticated routing
    mechanism based on agreement between predictions and outputs.

    **Intent**: Enable dynamic, learned routing of information between capsule layers
    through an iterative agreement mechanism. Unlike fixed connections in traditional
    neural networks, routing coefficients are computed on-the-fly based on the
    agreement between capsule predictions and outputs, allowing the network to
    discover part-whole relationships.

    **Architecture** (per routing iteration):
    ```
    Input Capsules(shape=[batch, num_input_capsules, input_dim])
           ↓
    Transform via W: û_j|i = W_ij × u_i  [Prediction vectors]
           ↓
    Initialize: b_ij = 0  [Routing logits]
           ↓
    ┌─────────────────────── Routing Iteration ───────────────────────┐
    │                                                                 │
    │  Softmax: c_ij = softmax(b_ij)  ← [Routing coefficients]        │
    │           ↓                                                     │
    │  Weighted Sum: s_j = Σ_i c_ij × û_j|i                           │
    │           ↓                                                     │
    │  Add Bias: s_j = s_j + bias_j (if use_bias=True)                │
    │           ↓                                                     │
    │  Squash: v_j = squash(s_j)  ← [Output capsule activation]       │
    │           ↓                                                     │
    │  Agreement: a_ij = û_j|i · v_j  [Dot product measures match]    │
    │           ↓                                                     │
    │  Update: b_ij ← b_ij + a_ij  [Strengthen agreeing connections]  │
    │           ↓                                                     │
    └──────────────────────── Repeat 3 times ─────────────────────────┘
           ↓
    Output Capsules(shape=[batch, num_capsules, dim_capsules])
    ```

    **Mathematical Formulation**:

    For each routing iteration:

    .. math::
        c_{ij} = \\text{softmax}(b_{ij}) \\quad \\text{(routing coefficients)}

    .. math::
        s_j = \\sum_i c_{ij} \\hat{u}_{j|i} \\quad \\text{(weighted sum)}

    .. math::
        v_j = \\text{squash}(s_j) \\quad \\text{(output capsule)}

    .. math::
        b_{ij} \\leftarrow b_{ij} + \\hat{u}_{j|i} \\cdot v_j \\quad \\text{(update logits)}

    **Key Innovation**:
    The routing coefficients c_ij are not learned parameters but are computed
    dynamically for each input based on the agreement between predicted and
    actual capsule activations. This allows the network to route information
    flexibly based on the input content.

    Args:
        num_capsules: Integer, number of output capsules to produce. Must be positive.
            This is typically much smaller than num_input_capsules (e.g., 10 for digit
            classification from 1152 primary capsules).
        dim_capsules: Integer, dimension of each output capsule's vector. Must be positive.
            Larger dimensions allow capsules to encode more detailed instantiation
            parameters. Typical values are 8-32.
        routing_iterations: Integer, number of dynamic routing iterations. The paper recommends 3
            iterations as a good balance between performance and computational cost.
            Must be positive. Defaults to 3.
        kernel_initializer: String or Initializer, initializer for the transformation matrices W that transform
            input capsule predictions to output capsule space. Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional regularizer function applied to the transformation matrices.
            Helps prevent overfitting in the transformation weights.
        use_bias: Boolean, whether to include bias terms in the routing computation.
            Adds a learned bias to the weighted sum before squashing. Defaults to True.
        squash_axis: Integer, axis along which to apply the squashing non-linearity. Default is -2
            to squash along the capsule dimension in the intermediate representation.
        squash_epsilon: Optional float, small constant for numerical stability in the squashing function.
        **kwargs: Additional keyword arguments passed to the Layer base class.

    Input shape:
        3D tensor with shape: ``(batch_size, num_input_capsules, input_dim_capsules)``

    Output shape:
        3D tensor with shape: ``(batch_size, num_capsules, dim_capsules)``

    Attributes:
        W: Transformation weight matrix of shape [1, num_input_capsules, num_capsules,
            dim_capsules, input_dim_capsules]. Projects input capsules to output space.
        bias: Optional bias vector of shape [1, 1, num_capsules, dim_capsules, 1].
        squash_layer: SquashLayer instance for non-linear capsule activation.

    Example:
        ```python
        # Route from 1152 primary capsules (8D) to 10 digit capsules (16D)
        routing_layer = RoutingCapsule(
            num_capsules=10,
            dim_capsules=16,
            routing_iterations=3
        )
        primary_caps = keras.random.normal((32, 1152, 8))
        digit_caps = routing_layer(primary_caps)
        print(digit_caps.shape)  # (32, 10, 16)

        # Lower-dimensional routing for efficiency
        routing_layer = RoutingCapsule(
            num_capsules=10,
            dim_capsules=8,
            routing_iterations=2,  # Fewer iterations for speed
            use_bias=False         # No bias for regularization
        )
        ```

    Raises:
        ValueError: If num_capsules, dim_capsules, or routing_iterations is not positive.
        ValueError: If input shape is not 3D during build.

    Note:
        The computational complexity is O(routing_iterations × num_input_capsules ×
        num_capsules × dim_capsules), which can be significant for large capsule layers.
        Consider using fewer routing iterations or smaller capsule dimensions for
        efficiency when working with large-scale problems.

    References:
        Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules.
        In Advances in Neural Information Processing Systems (pp. 3856-3866).
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        routing_iterations: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        squash_axis: int = -2,
        squash_epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_capsules <= 0:
            raise ValueError(f"num_capsules must be positive, got {num_capsules}")
        if dim_capsules <= 0:
            raise ValueError(f"dim_capsules must be positive, got {dim_capsules}")
        if routing_iterations <= 0:
            raise ValueError(f"routing_iterations must be positive, got {routing_iterations}")

        # Store configuration
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iterations = routing_iterations
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.squash_axis = squash_axis
        self.squash_epsilon = squash_epsilon

        # CREATE squashing layer in __init__ (modern Keras 3 pattern)
        self.squash_layer = SquashLayer(
            axis=self.squash_axis,
            epsilon=self.squash_epsilon,
            name="routing_squash"
        )

        # Weight variables - created in build()
        self.W = None
        self.bias = None

        # Shape attributes - set in build()
        self.num_input_capsules = None
        self.input_dim_capsules = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build layer weights based on input shape.

        Args:
            input_shape: Shape of input tensor [batch_size, num_input_capsules, input_dim_capsules].
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape [batch, num_capsules, dim_capsules], got {input_shape}")

        # Extract dimensions from input shape
        self.num_input_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]

        # Create weight matrix for transformations between input and output capsules
        # Shape: [1, num_input_capsules, num_capsules, dim_capsules, input_dim_capsules]
        self.W = self.add_weight(
            shape=[1, self.num_input_capsules, self.num_capsules,
                   self.dim_capsules, self.input_dim_capsules],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="capsule_transformation_weights"
        )

        # Create bias term if requested
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[1, 1, self.num_capsules, self.dim_capsules, 1],
                initializer="zeros",
                trainable=True,
                name="capsule_bias"
            )

        # BUILD squashing layer - it needs output shape
        squash_input_shape = (input_shape[0], 1, self.num_capsules, self.dim_capsules, 1)
        self.squash_layer.build(squash_input_shape)

        super().build(input_shape)

        logger.info(f"Built RoutingCapsule layer: {self.num_input_capsules} -> {self.num_capsules} capsules, "
                   f"{self.routing_iterations} routing iterations")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass implementing dynamic routing between capsules.

        Args:
            inputs: Input tensor of shape [batch_size, num_input_capsules, input_dim_capsules].
            training: Boolean, whether in training mode.

        Returns:
            Output tensor of shape [batch_size, num_capsules, dim_capsules].
        """
        batch_size = ops.shape(inputs)[0]

        # Prepare inputs by expanding dimensions for matrix multiplication
        # [batch_size, num_input_capsules, input_dim_capsules] ->
        # [batch_size, num_input_capsules, 1, 1, input_dim_capsules]
        inputs_expanded = ops.expand_dims(ops.expand_dims(inputs, axis=-1), axis=2)

        # Tile inputs for efficient broadcasting with weights
        # [batch_size, num_input_capsules, 1, 1, input_dim_capsules] ->
        # [batch_size, num_input_capsules, num_capsules, 1, input_dim_capsules]
        inputs_tiled = ops.tile(
            inputs_expanded,
            [1, 1, self.num_capsules, 1, 1]
        )

        # Calculate predicted output vectors (u_hat) through transformation
        # [1, num_input_capsules, num_capsules, dim_capsules, input_dim_capsules] *
        # [batch_size, num_input_capsules, num_capsules, 1, input_dim_capsules] ->
        # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1]
        u_hat = ops.matmul(self.W, inputs_tiled)

        # Initialize routing logits (b) to zero
        b = ops.zeros([batch_size, self.num_input_capsules, self.num_capsules, 1, 1])

        # Perform iterative dynamic routing
        for i in range(self.routing_iterations):
            # Convert logits to routing weights using softmax
            # Shape: [batch_size, num_input_capsules, num_capsules, 1, 1]
            c = keras.activations.softmax(b, axis=2)

            # Weighted sum of predictions to get capsule outputs
            # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] *
            # [batch_size, num_input_capsules, num_capsules, 1, 1] ->
            # [batch_size, 1, num_capsules, dim_capsules, 1]
            s = ops.sum(c * u_hat, axis=1, keepdims=True)

            # Add bias if requested
            if self.use_bias and self.bias is not None:
                s += self.bias

            # Apply squashing non-linearity
            v = self.squash_layer(s, training=training)

            # For all but the last iteration, update routing logits
            if i < self.routing_iterations - 1:
                # Tile output capsules to match input capsules for agreement calculation
                # [batch_size, 1, num_capsules, dim_capsules, 1] ->
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1]
                v_tiled = ops.tile(v, [1, self.num_input_capsules, 1, 1, 1])

                # Calculate agreement between predictions and outputs
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] *
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] ->
                # [batch_size, num_input_capsules, num_capsules, 1, 1]
                agreement = ops.sum(
                    u_hat * v_tiled,
                    axis=-2,
                    keepdims=True
                )

                # Update routing logits based on agreement
                b += agreement

        # Final output: shape [batch_size, num_capsules, dim_capsules]
        return ops.reshape(v, (batch_size, self.num_capsules, self.dim_capsules))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Shape of output tensor as tuple.
        """
        return (input_shape[0], self.num_capsules, self.dim_capsules)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routing_iterations": self.routing_iterations,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "squash_axis": self.squash_axis,
            "squash_epsilon": self.squash_epsilon,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsuleBlock(keras.layers.Layer):
    """A complete capsule block with optional dropout and normalization.

    This block combines a RoutingCapsule layer with optional regularization
    techniques like dropout and layer normalization to create a robust
    capsule processing unit suitable for production use.

    **Intent**: Provide a modular, reusable capsule processing unit that combines
    dynamic routing with modern regularization techniques. This block can be
    stacked to create deeper capsule networks while maintaining training stability
    through optional normalization and preventing overfitting through dropout.

    **Architecture**:
    ```
    Input Capsules(shape=[batch, num_input_capsules, input_dim])
           ↓
    RoutingCapsule(num_capsules, dim_capsules, routing_iterations)
           ├─→ Transform: W × input_capsules
           ├─→ Dynamic Routing: 3 iterations (default)
           └─→ Squash activation
           ↓
    [Optional] Dropout(dropout_rate) ← Applied during training only
           ↓                            Randomly zeros elements
           ↓
    [Optional] LayerNormalization ← Normalizes across capsule dimension
           ↓                         Stabilizes training
           ↓
    Output Capsules(shape=[batch, num_capsules, dim_capsules])
    ```

    **Processing Flow**:
    ```
    x = RoutingCapsule(inputs)           # Core capsule routing
    if dropout_rate > 0:
        x = Dropout(x, training=training)  # Regularization
    if use_layer_norm:
        x = LayerNorm(x)                  # Normalization
    return x
    ```

    Args:
        num_capsules: Integer, number of output capsules. Must be positive.
            This defines the cardinality of the capsule representation at this level.
        dim_capsules: Integer, dimension of each output capsule's vector. Must be positive.
            Higher dimensions allow more expressive capsule representations.
        routing_iterations: Integer, number of routing iterations. Must be positive. Defaults to 3.
            More iterations allow better routing convergence but increase computation.
        dropout_rate: Float, dropout rate. Must be in [0.0, 1.0). 0.0 means no dropout. Defaults to 0.0.
            Typical values are 0.2-0.5 for regularization. Applied only during training.
        use_layer_norm: Boolean, whether to use layer normalization. Defaults to False.
            Recommended for deeper networks to stabilize training and gradients.
        kernel_initializer: String or Initializer, initializer for the transformation matrices.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional regularizer function for the transformation matrices.
            L1/L2 regularization can be applied to routing weights for better generalization.
        use_bias: Boolean, whether to use biases in routing. Defaults to True.
            Bias can help the model learn better capsule activations.
        squash_axis: Integer, axis along which to apply squashing operation. Defaults to -2.
            Should match the axis used in RoutingCapsule for consistency.
        squash_epsilon: Optional float, epsilon for numerical stability in squashing.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: ``(batch_size, num_input_capsules, input_dim_capsules)``

    Output shape:
        3D tensor with shape: ``(batch_size, num_capsules, dim_capsules)``

    Attributes:
        capsule_layer: RoutingCapsule instance performing dynamic routing.
        dropout: Optional Dropout layer for regularization.
        layer_norm: Optional LayerNormalization for training stability.

    Example:
        ```python
        # Create a capsule block with regularization
        capsule_block = CapsuleBlock(
            num_capsules=10,
            dim_capsules=16,
            routing_iterations=3,
            dropout_rate=0.2,
            use_layer_norm=True
        )

        # Process capsules
        primary_caps = keras.random.normal((32, 1152, 8))
        digit_caps = capsule_block(primary_caps, training=True)
        print(digit_caps.shape)  # (32, 10, 16)

        # Minimal block without regularization (for small datasets)
        simple_block = CapsuleBlock(
            num_capsules=5,
            dim_capsules=8,
            routing_iterations=2,
            dropout_rate=0.0,        # No dropout
            use_layer_norm=False     # No normalization
        )

        # Heavy regularization block (for large datasets prone to overfitting)
        regularized_block = CapsuleBlock(
            num_capsules=20,
            dim_capsules=32,
            routing_iterations=3,
            dropout_rate=0.5,        # Strong dropout
            use_layer_norm=True,     # With normalization
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    Raises:
        ValueError: If dropout_rate is not in [0.0, 1.0).
        TypeError: If use_layer_norm is not boolean.

    Note:
        When using dropout, ensure training=True is passed during training and
        training=False during inference. Layer normalization is applied after
        dropout for better training dynamics.
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        routing_iterations: int = 3,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        squash_axis: int = -2,
        squash_epsilon: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dropout_rate < 0.0 or dropout_rate >= 1.0:
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")
        if not isinstance(use_layer_norm, bool):
            raise TypeError(f"use_layer_norm must be boolean, got {type(use_layer_norm)}")

        # Store configuration parameters
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iterations = routing_iterations
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.squash_axis = squash_axis
        self.squash_epsilon = squash_epsilon

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        self.capsule_layer = RoutingCapsule(
            num_capsules=self.num_capsules,
            dim_capsules=self.dim_capsules,
            routing_iterations=self.routing_iterations,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            squash_axis=self.squash_axis,
            squash_epsilon=self.squash_epsilon,
            name="block_capsules"
        )

        # Optional regularization layers
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name="block_dropout")
        else:
            self.dropout = None

        if self.use_layer_norm:
            self.layer_norm = keras.layers.LayerNormalization(
                axis=-1,
                name="block_norm"
            )
        else:
            self.layer_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of input tensor.
        """
        # BUILD all sub-layers explicitly (critical for serialization)
        self.capsule_layer.build(input_shape)

        # Compute output shape for building optional layers
        output_shape = self.capsule_layer.compute_output_shape(input_shape)

        if self.dropout is not None:
            self.dropout.build(output_shape)

        if self.layer_norm is not None:
            self.layer_norm.build(output_shape)

        super().build(input_shape)

        logger.info(f"Built CapsuleBlock: {self.num_capsules} capsules, "
                   f"dropout={self.dropout_rate}, layer_norm={self.use_layer_norm}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the capsule block.

        Args:
            inputs: Input tensor of shape [batch_size, num_input_capsules, input_dim_capsules].
            training: Boolean, whether in training mode.

        Returns:
            Processed output tensor of shape [batch_size, num_capsules, dim_capsules].
        """
        # Process through capsule layer
        x = self.capsule_layer(inputs, training=training)

        # Apply optional regularization
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.layer_norm is not None:
            x = self.layer_norm(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Shape of output tensor as tuple.
        """
        return self.capsule_layer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routing_iterations": self.routing_iterations,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "squash_axis": self.squash_axis,
            "squash_epsilon": self.squash_epsilon,
        })
        return config

# ---------------------------------------------------------------------