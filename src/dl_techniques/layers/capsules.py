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
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union, Literal, Dict, Any, List, Callable

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
    L = y_true * tf.square(tf.maximum(0., margin - y_pred)) + \
        downweight * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
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
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias

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
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iterations = routing_iterations

    def build(self, input_shape: tf.TensorShape):
        """Build layer weights based on input shape.

        Args:
            input_shape: Shape of input tensor [batch_size, num_input_capsules, input_dim_capsules]
        """
        # Extract dimensions from input shape
        self.num_input_capsules = input_shape[-2]
        self.input_dim_capsules = input_shape[-1]

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

        self.built = True

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass implementing dynamic routing between capsules.

        Args:
            inputs: Input tensor of shape [batch_size, num_input_capsules, input_dim_capsules]
            training: Whether in training mode

        Returns:
            Output tensor of shape [batch_size, num_capsules, dim_capsules]
        """
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
        u_hat = tf.reduce_sum(
            self.W * inputs_tiled,
            axis=-1,
            keepdims=True
        )

        # Initialize routing logits (b) to zero
        batch_size = tf.shape(inputs)[0]
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
            if self.use_bias:
                s += self.bias

            # Apply squashing non-linearity
            v = self.squash_vectors(s)

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
        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None

        if use_layer_norm:
            self.layer_norm = keras.layers.LayerNormalization(
                axis=-1,
                name=f"{name}_norm" if name else None
            )
        else:
            self.layer_norm = None

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


class CapsNet(keras.Model):
    """
    Complete Capsule Network model.

    This model implements a customizable capsule network with
    convolutional feature extraction, primary capsules, and routing capsules.

    Args:
        num_classes: Number of output classes
        routing_iterations: Number of routing iterations
        conv_filters: List of numbers of filters for convolutional layers
        primary_capsules: Number of primary capsules
        primary_capsule_dim: Dimension of primary capsule vectors
        digit_capsule_dim: Dimension of digit/class capsule vectors
        kernel_initializer: Initializer for transformation matrices
        kernel_regularizer: Regularizer for transformation matrices
        name: Optional name for the model
    """

    def __init__(
        self,
        num_classes: int,
        routing_iterations: int = 3,
        conv_filters: List[int] = [256, 256],
        primary_capsules: int = 32,
        primary_capsule_dim: int = 8,
        digit_capsule_dim: int = 16,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Store configuration
        self.num_classes = num_classes
        self.routing_iterations = routing_iterations
        self.conv_filters = conv_filters
        self.primary_capsules = primary_capsules
        self.primary_capsule_dim = primary_capsule_dim
        self.digit_capsule_dim = digit_capsule_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Feature extraction layers
        self.feature_layers = []
        for i, filters in enumerate(conv_filters):
            self.feature_layers.append(
                keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=9 if i == 0 else 5,
                    strides=1,
                    padding="valid",
                    activation="relu",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    name=f"conv_{i+1}"
                )
            )

        # Primary capsule layer
        self.primary_caps = PrimaryCapsule(
            num_capsules=primary_capsules,
            dim_capsules=primary_capsule_dim,
            kernel_size=9,
            strides=2,
            padding="valid",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="primary_caps"
        )

        # Digit capsule layer
        self.digit_caps = RoutingCapsule(
            num_capsules=num_classes,
            dim_capsules=digit_capsule_dim,
            routing_iterations=routing_iterations,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="digit_caps"
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> Dict[str, tf.Tensor]:
        """Forward pass through the complete capsule network.

        Args:
            inputs: Input images of shape [batch_size, height, width, channels]
            training: Whether in training mode

        Returns:
            Dictionary with digit_caps (raw capsule outputs) and length (class probabilities)
        """
        # Feature extraction
        x = inputs
        for layer in self.feature_layers:
            x = layer(x, training=training)

        # Primary capsules
        primary_caps_output = self.primary_caps(x, training=training)

        # Digit capsules
        digit_caps_output = self.digit_caps(primary_caps_output, training=training)

        # Calculate capsule lengths (class probabilities)
        lengths = length(digit_caps_output)

        return {
            "digit_caps": digit_caps_output,
            "length": lengths
        }

    def train_step(self, data):
        """Custom training step with margin loss.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dictionary of metrics
        """
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = margin_loss(y, y_pred["length"])

        # Update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred["length"])

        # Return metrics
        metrics = {"loss": loss}
        metrics.update({m.name: m.result() for m in self.metrics})
        return metrics

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing model configuration
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "routing_iterations": self.routing_iterations,
            "conv_filters": self.conv_filters,
            "primary_capsules": self.primary_capsules,
            "primary_capsule_dim": self.primary_capsule_dim,
            "digit_capsule_dim": self.digit_capsule_dim,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create model from configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            New model instance
        """
        return cls(**config)


def build_capsnet(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    routing_iterations: int = 3,
    conv_filters: List[int] = [256, 256],
    primary_capsules: int = 32,
    primary_capsule_dim: int = 8,
    digit_capsule_dim: int = 16,
    kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    learning_rate: float = 0.001
) -> keras.Model:
    """Build and compile a CapsNet model.

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of classes
        routing_iterations: Number of routing iterations
        conv_filters: List of filter counts for convolutional layers
        primary_capsules: Number of primary capsules
        primary_capsule_dim: Dimension of primary capsule vectors
        digit_capsule_dim: Dimension of digit/class capsule vectors
        kernel_initializer: Initializer for weights
        kernel_regularizer: Regularizer for weights
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled CapsNet model
    """
    # Create model
    model = CapsNet(
        num_classes=num_classes,
        routing_iterations=routing_iterations,
        conv_filters=conv_filters,
        primary_capsules=primary_capsules,
        primary_capsule_dim=primary_capsule_dim,
        digit_capsule_dim=digit_capsule_dim,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )

    # Build model with input shape
    model.build(input_shape=(None,) + input_shape)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=margin_loss,
        metrics=['accuracy']
    )

    return model


def example_usage():
    """Example usage of the CapsNet model."""
    # Example parameters
    input_shape = (28, 28, 1)  # MNIST
    num_classes = 10
    kernel_regularizer = keras.regularizers.l2(0.0005)

    # Build model
    model = build_capsnet(
        input_shape=input_shape,
        num_classes=num_classes,
        kernel_initializer="he_normal",
        kernel_regularizer=kernel_regularizer
    )

    # Show model summary
    model.summary()

    # Save model (optional)
    model.save("capsnet_mnist.keras")

    return model


if __name__ == "__main__":
    example_usage()