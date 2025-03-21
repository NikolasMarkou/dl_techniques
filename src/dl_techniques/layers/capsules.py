"""
Modern Keras implementation of Capsule Networks (CapsNet).

This module provides a Keras-based implementation of Capsule Networks,
following the architecture proposed in the paper "Dynamic Routing Between Capsules"
by Sabour et al.

Key Differences Between Capsules and Traditional Neurons:

1. Input/Output Nature:
   - Traditional Neuron: Takes scalar inputs (x_i) and produces scalar output (h)
   - Capsule: Takes vector inputs (u_i) and produces vector output (v_j)

2. Transformation Process:
   a) Linear Transformation:
      - Traditional: Simple scalar multiplication and bias addition (a_ji = w_ij*x_i + b_j)
      - Capsule: Matrix multiplication with weight matrix W_ij and bias B_j (û_j|i = W_ij*u_i + B_j)

   b) Weighting & Summation:
      - Traditional: Scalar weighted sum with learned weights (z_j = Σ w_ij*x_i)
      - Capsule: Vector weighted sum using coupling coefficients c_ij (s_j = Σ c_ij*û_j|i)
        * Coupling coefficients are dynamically computed through routing algorithm

3. Activation Function:
   - Traditional: Element-wise non-linearity (sigmoid, tanh, ReLU)
   - Capsule: "Squash" function that preserves vector orientation while normalizing length:
     squash(s) = ||s||² / (1 + ||s||²) * (s / ||s||)
     * Ensures output vector length is in [0,1] representing probability
     * Preserves vector orientation to maintain feature representation

4. Information Encoding:
   - Traditional: Activity scalar represents feature presence
   - Capsule: Vector properties encode feature attributes:
     * Vector length: Probability of feature existence
     * Vector orientation: Feature properties/parameters

5. Key Implementation Aspects:
   - Dynamic Routing: Iterative process to determine coupling coefficients
   - Agreement Mechanism: Vectors "vote" for higher-level features
   - Hierarchical Structure: Lower-level capsules feed into higher-level ones

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017).
      Dynamic routing between capsules. In Advances in Neural
      Information Processing Systems (pp. 3856-3866).
"""

from __future__ import annotations

import keras
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any, List, Callable, TypeVar, Type, cast

# Type variables for improved type hinting
T = TypeVar('T', bound='BaseCapsuleLayer')

# Constants for numerical stability
EPSILON = 1e-9

def length(vectors: tf.Tensor) -> tf.Tensor:
    """Compute length of capsule vectors.

    Args:
        vectors: Capsule vectors of shape [..., dim_capsule]

    Returns:
        Length of vectors with shape [...]
    """
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + EPSILON)


def squash(vectors: tf.Tensor, axis: int = -1, epsilon: float = EPSILON) -> tf.Tensor:
    """Apply squashing non-linearity to vectors (capsules).

    The squashing function ensures that:
    1. Short vectors get shrunk to almost zero length
    2. Long vectors get shrunk to a length slightly below 1
    3. Vector orientation is preserved

    Args:
        vectors: Input tensor to be squashed
        axis: Axis along which to compute the norm
        epsilon: Small constant for numerical stability

    Returns:
        Squashed vectors with norm between 0 and 1
    """
    # Handle edge case of zero vectors
    if tf.reduce_all(tf.equal(vectors, 0)):
        return vectors

    squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1.0 + squared_norm)
    unit_vector = vectors / safe_norm
    return scale * unit_vector


def margin_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    margin: float = 0.9,
    downweight: float = 0.5
) -> tf.Tensor:
    """Implements the margin loss for capsule networks.

    As described in the paper, the margin loss is similar to cross-entropy but
    uses separate margins for positive and negative classes.

    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted capsule lengths
        margin: Margin for positive class separation
        downweight: Weight for negative class terms

    Returns:
        Margin loss value
    """
    # Validate inputs
    if not tf.is_tensor(y_true) or not tf.is_tensor(y_pred):
        raise TypeError("Both y_true and y_pred must be tensors")
    if y_true.shape[-1] != y_pred.shape[-1]:
        raise ValueError(f"Last dimension of y_true and y_pred must match. Got {y_true.shape[-1]} and {y_pred.shape[-1]}")

    # Calculate positive and negative class losses
    positive_loss = y_true * tf.square(tf.maximum(0., margin - y_pred))
    negative_loss = downweight * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    # Combine losses
    L = positive_loss + negative_loss

    # Reduce to scalar loss
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


# ======================================================================
# Core Capsule Layers
# ======================================================================

class BaseCapsuleLayer(keras.layers.Layer):
    """Base class for all capsule layers.

    This base class provides common functionality for all capsule layers,
    including configuration handling and squashing operations.

    Args:
        kernel_initializer: Initializer for the transformation matrices
        kernel_regularizer: Regularizer function for the transformation matrices
        use_bias: Whether to use biases in routing
        name: Optional name for the layer
        **kwargs: Additional keyword arguments for the Layer base class
    """

    def __init__(
        self,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Store configuration
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self._built = False  # Track if layer has been built

    def squash_vectors(self, vectors: tf.Tensor, axis: int = -1) -> tf.Tensor:
        """Apply squashing operation to vectors.

        Args:
            vectors: Input tensor to be squashed
            axis: Axis along which to compute the norm

        Returns:
            Squashed vectors with norm between 0 and 1
        """
        return squash(vectors, axis=axis, epsilon=EPSILON)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
        """
        config = super().get_config()
        config.update({
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias
        })
        return config

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            New layer instance
        """
        # Handle special cases of serialized objects
        if "kernel_initializer" in config:
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        # Create new instance
        return cls(**config)


class PrimaryCapsule(BaseCapsuleLayer):
    """Primary Capsule Layer implementation.

    This layer implements the primary capsule layer which converts regular CNN features
    into capsule format. It consists of a convolutional layer followed by reshaping
    and squashing operations.

    Args:
        num_capsules: Number of primary capsules to create
        dim_capsules: Dimension of each capsule's output vector
        kernel_size: Size of the convolutional kernel
        strides: Stride length for convolution
        padding: Type of padding for convolution ('valid' or 'same')
        kernel_initializer: Initializer for the conv kernel
        kernel_regularizer: Regularizer function for the conv kernel
        use_bias: Whether to include bias terms
        name: Optional name for the layer
        **kwargs: Additional keyword arguments for the base class
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
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
            name=name,
            **kwargs
        )

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

        # Convolutional layer to generate capsule outputs
        self.conv = keras.layers.Conv2D(
            filters=num_capsules * dim_capsules,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias
        )

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of input tensor
        """
        # Validate input shape
        if len(input_shape) != 4:  # [batch_size, height, width, channels]
            raise ValueError(f"Expected 4D input shape [batch, height, width, channels], got {input_shape}")

        # Build the convolutional layer
        self.conv.build(input_shape)
        self._built = True

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass for primary capsule layer.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch_size, num_capsules, dim_capsules]
        """
        batch_size = tf.shape(inputs)[0]

        # Apply convolution
        conv_output = self.conv(inputs, training=training)

        # Reshape to capsule format
        # Calculate dimensions based on the output of the convolutional layer
        h, w = tf.shape(conv_output)[1], tf.shape(conv_output)[2]
        num_spatial_capsules = h * w

        # Reshape to [batch_size, h*w * num_capsules, dim_capsules]
        capsules = tf.reshape(
            conv_output,
            [batch_size, num_spatial_capsules * self.num_capsules, self.dim_capsules]
        )

        # Apply squashing
        return self.squash_vectors(capsules)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        # Compute the shape after convolution
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        h, w = conv_output_shape[1], conv_output_shape[2]
        num_spatial_capsules = h * w

        # Shape after reshaping and squashing
        return tf.TensorShape([
            input_shape[0],  # batch size
            num_spatial_capsules * self.num_capsules,
            self.dim_capsules
        ])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding
        })
        return config


class RoutingCapsule(BaseCapsuleLayer):
    """Capsule layer with dynamic routing.

    Implements the dynamic routing mechanism between capsules as described in the paper.
    This unified implementation replaces both DigitCaps and CapsuleLayer from the
    original code.

    Args:
        num_capsules: Number of output capsules
        dim_capsules: Dimension of each output capsule's vector
        routing_iterations: Number of routing iterations
        kernel_initializer: Initializer for the transformation matrices
        kernel_regularizer: Regularizer function for the transformation matrices
        use_bias: Whether to use biases in routing
        name: Optional name for the layer
        **kwargs: Additional keyword arguments for the base class
    """

    def __init__(
        self,
        num_capsules: int,
        dim_capsules: int,
        routing_iterations: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
            name=name,
            **kwargs
        )

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

        # These will be set in build()
        self.W = None
        self.bias = None
        self.num_input_capsules = None
        self.input_dim_capsules = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer weights based on input shape.

        Args:
            input_shape: Shape of input tensor [batch_size, num_input_capsules, input_dim_capsules]
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

        self._built = True
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass implementing dynamic routing between capsules.

        Args:
            inputs: Input tensor of shape [batch_size, num_input_capsules, input_dim_capsules]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch_size, num_capsules, dim_capsules]
        """
        if not self._built:
            raise ValueError("Layer must be built before being called. "
                             "Call build() or call the layer on a tensor first.")

        # Verify input shape matches expectations
        batch_size = tf.shape(inputs)[0]
        input_shape = tf.shape(inputs)

        if input_shape[1] != self.num_input_capsules or input_shape[2] != self.input_dim_capsules:
            tf.debugging.assert_equal(
                input_shape[1],
                self.num_input_capsules,
                message=f"Expected input_shape[1]={self.num_input_capsules}, got {input_shape[1]}"
            )
            tf.debugging.assert_equal(
                input_shape[2],
                self.input_dim_capsules,
                message=f"Expected input_shape[2]={self.input_dim_capsules}, got {input_shape[2]}"
            )

        # Prepare inputs by expanding dimensions for matrix multiplication
        # [batch_size, num_input_capsules, input_dim_capsules] ->
        # [batch_size, num_input_capsules, 1, 1, input_dim_capsules]
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, axis=-1), axis=2)

        # Tile inputs for efficient broadcasting with weights
        # [batch_size, num_input_capsules, 1, 1, input_dim_capsules] ->
        # [batch_size, num_input_capsules, num_capsules, 1, input_dim_capsules]
        inputs_tiled = tf.tile(
            inputs_expanded,
            [1, 1, self.num_capsules, 1, 1]
        )

        # Calculate predicted output vectors (u_hat) through transformation
        # [1, num_input_capsules, num_capsules, dim_capsules, input_dim_capsules] *
        # [batch_size, num_input_capsules, num_capsules, 1, input_dim_capsules] ->
        # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1]
        u_hat = tf.matmul(self.W, inputs_tiled, transpose_b=True)

        # Initialize routing logits (b) to zero
        b = tf.zeros([batch_size, self.num_input_capsules, self.num_capsules, 1, 1])

        # Perform iterative dynamic routing
        for i in range(self.routing_iterations):
            # Convert logits to routing weights using softmax
            # Shape: [batch_size, num_input_capsules, num_capsules, 1, 1]
            c = tf.nn.softmax(b, axis=2)

            # Weighted sum of predictions to get capsule outputs
            # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] *
            # [batch_size, num_input_capsules, num_capsules, 1, 1] ->
            # [batch_size, 1, num_capsules, dim_capsules, 1]
            s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)

            # Add bias if requested
            if self.use_bias and self.bias is not None:
                s += self.bias

            # Apply squashing non-linearity
            v = self.squash_vectors(s, axis=-2)

            # For all but the last iteration, update routing logits
            if i < self.routing_iterations - 1:
                # Tile output capsules to match input capsules for agreement calculation
                # [batch_size, 1, num_capsules, dim_capsules, 1] ->
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1]
                v_tiled = tf.tile(v, [1, self.num_input_capsules, 1, 1, 1])

                # Calculate agreement between predictions and outputs
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] *
                # [batch_size, num_input_capsules, num_capsules, dim_capsules, 1] ->
                # [batch_size, num_input_capsules, num_capsules, 1, 1]
                agreement = tf.reduce_sum(
                    u_hat * v_tiled,
                    axis=-2,
                    keepdims=True
                )

                # Update routing logits based on agreement
                b += agreement

        # Final output: shape [batch_size, num_capsules, dim_capsules]
        return tf.squeeze(v, axis=[1, -1])

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        return tf.TensorShape([input_shape[0], self.num_capsules, self.dim_capsules])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routing_iterations": self.routing_iterations
        })
        return config


class CapsuleBlock(BaseCapsuleLayer):
    """
    A complete capsule block with optional dropout and normalization.

    This block combines a RoutingCapsule layer with optional regularization
    techniques like dropout and layer normalization.

    Args:
        num_capsules: Number of output capsules
        dim_capsules: Dimension of each output capsule's vector
        routing_iterations: Number of routing iterations
        dropout_rate: Dropout rate (0.0 means no dropout)
        use_layer_norm: Whether to use layer normalization
        kernel_initializer: Initializer for the transformation matrices
        kernel_regularizer: Regularizer function for the transformation matrices
        use_bias: Whether to use biases in routing
        name: Optional name for the layer
        **kwargs: Additional keyword arguments for the base class
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
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
            name=name,
            **kwargs
        )

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

        # Create layers
        self.capsule_layer = RoutingCapsule(
            num_capsules=num_capsules,
            dim_capsules=dim_capsules,
            routing_iterations=routing_iterations,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias,
            name=f"{name}_capsules" if name else None
        )

        # Optional regularization layers
        self.dropout = None
        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(dropout_rate)

        self.layer_norm = None
        if use_layer_norm:
            self.layer_norm = keras.layers.LayerNormalization(
                axis=-1,
                name=f"{name}_norm" if name else None
            )

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of input tensor
        """
        # Build the capsule layer
        self.capsule_layer.build(input_shape)

        # Build layer normalization if used
        if self.use_layer_norm and self.layer_norm is not None:
            output_shape = self.capsule_layer.compute_output_shape(input_shape)
            self.layer_norm.build(output_shape)

        self._built = True
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through the capsule block.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Processed output tensor
        """
        # Process through capsule layer
        x = self.capsule_layer(inputs, training=training)

        # Apply optional regularization
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        return self.capsule_layer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routing_iterations": self.routing_iterations,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm
        })
        return config
