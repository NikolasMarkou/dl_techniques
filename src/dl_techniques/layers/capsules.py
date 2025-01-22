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

This architecture enables CapsNets to better preserve spatial relationships and hierarchical relationships
in the data, addressing limitations of traditional CNNs such as viewpoint invariance and spatial
relationship understanding.
"""

import keras
import numpy as np
from typing import Optional, Tuple, Union, Literal
import tensorflow as tf


# Constants
EPSILON = 1e-9


class SquashActivation(keras.layers.Layer):
    """Squashing activation function for capsule networks.

    Implements the squashing function as described in the CapsNet paper:
    v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||

    This layer applies the non-linear squashing function to ensure that short vectors
    get shrunk to almost zero length and long vectors get shrunk to a length slightly
    below 1.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply squashing activation to input tensor.

        Args:
            inputs: Input tensor of shape [..., num_capsules, dim_capsules, 1]
                or [..., num_capsules, dim_capsules]

        Returns:
            Squashed tensor with same shape as input
        """
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=-2, keepdims=True)
        scale = squared_norm / (1.0 + squared_norm) / tf.sqrt(squared_norm + EPSILON)
        return scale * inputs


class PrimaryCaps(keras.layers.Layer):
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
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias

        self.conv = keras.layers.Conv2D(
            filters=num_capsules * dim_capsules,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias
        )
        self.squash = SquashActivation()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass for primary capsule layer.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels]

        Returns:
            Output tensor of shape [batch_size, num_capsules, dim_capsules]
        """
        batch_size = tf.shape(inputs)[0]

        # Apply convolution
        conv_output = self.conv(inputs)

        # Reshape to capsule format
        conv_shape = tf.shape(conv_output)
        capsules = tf.reshape(
            conv_output,
            [batch_size, -1, self.dim_capsules]
        )

        # Apply squashing
        return self.squash(capsules)

    def get_config(self) -> dict:
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
            "padding": self.padding,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias
        })
        return config


class DigitCaps(keras.layers.Layer):
    """Digit Capsule Layer with dynamic routing.

    This layer implements the digit capsules that receive inputs from the primary
    capsules and produce the final output capsules. It uses dynamic routing
    between capsules as described in the paper.

    Args:
        num_capsules: Number of digit capsules (usually number of classes)
        dim_capsules: Dimension of each capsule's output vector
        routing_iterations: Number of routing iterations
        kernel_initializer: Initializer for the transformation matrices
        kernel_regularizer: Regularizer function for the transformation matrices
        use_bias: Whether to use biases in routing
    """

    def __init__(
            self,
            num_capsules: int,
            dim_capsules: int,
            routing_iterations: int = 3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routing_iterations = routing_iterations
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.squash = SquashActivation()

    def build(self, input_shape: tf.TensorShape):
        """Build layer weights based on input shape.

        Args:
            input_shape: Shape of input tensor
        """
        input_dim_capsules = input_shape[-1]
        num_input_capsules = input_shape[-2]

        self.W = self.add_weight(
            shape=[1, num_input_capsules, self.num_capsules,
                   self.dim_capsules, input_dim_capsules],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="digit_capsule_kernel"
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=[1, 1, self.num_capsules, self.dim_capsules, 1],
                initializer="zeros",
                name="digit_capsule_bias"
            )

        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass implementing dynamic routing between capsules.

        Args:
            inputs: Input tensor of shape [batch_size, num_capsules, dim_capsules]

        Returns:
            Output tensor of shape [batch_size, num_capsules, dim_capsules]
        """
        # Prepare inputs
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, axis=-1), axis=2)
        inputs_tiled = tf.tile(inputs_expanded, [1, 1, self.num_capsules, 1, 1])

        # Calculate predicted output vectors
        u_hat = tf.reduce_sum(self.W * inputs_tiled, axis=-1, keepdims=True)

        # Initialize routing logits
        batch_size = tf.shape(inputs)[0]
        b = tf.zeros([batch_size, inputs.shape[1], self.num_capsules, 1, 1])

        # Iterative routing
        for i in range(self.routing_iterations):
            c = tf.nn.softmax(b, axis=2)

            if i == self.routing_iterations - 1:
                s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
                if self.use_bias:
                    s = s + self.bias
                v = self.squash(s)
            else:
                s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
                if self.use_bias:
                    s = s + self.bias
                v = self.squash(s)
                v_tiled = tf.tile(v, [1, inputs.shape[1], 1, 1, 1])
                agreement = tf.reduce_sum(u_hat * v_tiled, axis=-2, keepdims=True)
                b = b + agreement

        return tf.squeeze(v, axis=[1, -1])

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
        """
        config = super().get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
            "routing_iterations": self.routing_iterations,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias
        })
        return config


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


def length(vectors: tf.Tensor) -> tf.Tensor:
    """Compute length of capsule vectors.

    Args:
        vectors: Capsule vectors of shape [..., dim_capsule]

    Returns:
        Length of vectors
    """
    return tf.sqrt(tf.reduce_sum(tf.square(vectors), axis=-1) + EPSILON)