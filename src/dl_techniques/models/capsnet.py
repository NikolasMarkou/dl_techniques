"""
CapsNet (Capsule Network) Model Implementation

Reference:
    'Dynamic Routing Between Capsules'
    Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton
    Paper: https://arxiv.org/abs/1710.09829

Architecture Overview:
    1. Input Layer: Takes images of shape (height, width, channels)
    2. Conv1: Initial convolutional layer for feature extraction
    3. PrimaryCaps: Converts conventional CNN features to capsule format
    4. DigitCaps: Final capsule layer with dynamic routing
    5. Decoder Network (optional): For reconstruction-based regularization

Key Components:
    - Convolutional Layer:
        * First layer with ReLU activation
        * Traditional feature extraction
        * Filters: 256, Kernel: 9x9

    - Primary Capsules:
        * Transition from Conv to Capsule representation
        * Multiple capsule units (default: 32)
        * Each capsule outputs 8D vector
        * Uses squashing non-linearity

    - Digit Capsules:
        * Final layer implementing dynamic routing
        * One capsule per class
        * 16D output vectors per capsule
        * Length represents probability of class

    - Reconstruction Network (Optional):
        * Acts as a regularizer
        * Decoder: FC(512) -> FC(1024) -> FC(pixels)
        * Reconstructs input from the DigitCaps representation

Training Features:
    - Custom margin loss for capsule networks
    - Optional reconstruction loss (weighted at 0.0005)
    - Dynamic routing between capsules
    - Support for masking and custom training steps

Usage Example:
    ```python
    model = CapsNet(
        num_classes=10,
        input_shape=(28, 28, 1),
        reconstruction=True
    )
    model.compile(
        optimizer='adam',
        metrics=['accuracy']
    )
    ```
"""

import keras
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any, Union

from dl_techniques.layers.capsules import (
    DigitCaps, PrimaryCaps, margin_loss, length)


class CapsNet(keras.Model):
    """Complete CapsNet model implementation.

    This class implements the full CapsNet architecture as described in the paper
    "Dynamic Routing Between Capsules" using modern Keras practices.

    Args:
        num_classes: Number of output classes
        conv_filters: Number of filters in initial conv layer
        primary_dims: Dimension of primary capsules
        primary_caps: Number of primary capsules
        digit_dims: Dimension of digit capsules
        routing_iterations: Number of routing iterations
        kernel_regularizer: Optional regularizer for all kernels
        reconstruction: Whether to include the reconstruction network
        input_shape: Shape of input images (height, width, channels)
    """

    def __init__(
            self,
            num_classes: int = 10,
            conv_filters: int = 256,
            primary_dims: int = 8,
            primary_caps: int = 32,
            digit_dims: int = 16,
            routing_iterations: int = 3,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            reconstruction: bool = True,
            input_shape: Tuple[int, int, int] = (28, 28, 1),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.primary_dims = primary_dims
        self.primary_caps = primary_caps
        self.digit_dims = digit_dims
        self.routing_iterations = routing_iterations
        self.reconstruction = reconstruction
        self.input_shape = input_shape

        # Initial convolutional layer
        self.conv1 = keras.layers.Conv2D(
            filters=conv_filters,
            kernel_size=9,
            padding='valid',
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )

        # Primary capsules
        self.primary_caps = PrimaryCaps(
            num_capsules=primary_caps,
            dim_capsules=primary_dims,
            kernel_size=9,
            strides=2,
            padding='valid',
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )

        # Digit capsules
        self.digit_caps = DigitCaps(
            num_capsules=num_classes,
            dim_capsules=digit_dims,
            routing_iterations=routing_iterations,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer
        )

        if reconstruction:
            # Reconstruction layers
            self.decoder = keras.Sequential([
                keras.layers.Dense(
                    512, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=kernel_regularizer
                ),
                keras.layers.Dense(
                    1024, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=kernel_regularizer
                ),
                keras.layers.Dense(
                    tf.math.reduce_prod(input_shape),
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    kernel_regularizer=kernel_regularizer
                ),
                keras.layers.Reshape(input_shape)
            ])

    def call(
            self,
            inputs: tf.Tensor,
            training: bool = False,
            mask: Optional[tf.Tensor] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass of the CapsNet model.

        Args:
            inputs: Input tensor of shape [batch_size, height, width, channels]
            training: Whether in training mode
            mask: Optional mask tensor for reconstruction

        Returns:
            If reconstruction is enabled:
                Tuple of (digit capsule outputs, reconstructed images)
            Otherwise:
                Digit capsule outputs
        """
        # Conv layer
        x = self.conv1(inputs)

        # Primary capsules
        primary_caps_output = self.primary_caps(x)

        # Digit capsules
        digit_caps_output = self.digit_caps(primary_caps_output)

        # Calculate capsule lengths (predictions)
        predictions = length(digit_caps_output)

        if not self.reconstruction:
            return predictions

        # Mask for reconstruction
        if mask is not None:
            # Use provided mask
            masked = tf.multiply(digit_caps_output, tf.expand_dims(mask, -1))
        else:
            # Use predicted classes
            mask = tf.one_hot(tf.argmax(predictions, axis=1), depth=self.num_classes)
            masked = tf.multiply(digit_caps_output, tf.expand_dims(mask, -1))

        # Flatten masked capsules
        decoder_input = tf.reshape(masked, [-1, self.num_classes * self.digit_dims])

        # Reconstruction
        reconstructed = self.decoder(decoder_input)

        return predictions, reconstructed

    def train_step(
            self,
            data: Tuple[tf.Tensor, tf.Tensor]
    ) -> Dict[str, float]:
        """Custom training step implementation.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dictionary of metric results for this batch
        """
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            if self.reconstruction:
                predictions, reconstructed = self(x, training=True, mask=y)
                # Calculate losses
                margin_loss_value = margin_loss(y, predictions)
                reconstruction_loss = tf.reduce_mean(
                    tf.square(x - reconstructed)
                )
                total_loss = margin_loss_value + 0.0005 * reconstruction_loss
            else:
                predictions = self(x, training=True)
                total_loss = margin_loss(y, predictions)

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        if self.reconstruction:
            results['reconstruction_loss'] = reconstruction_loss

        return results

    def test_step(
            self,
            data: Tuple[tf.Tensor, tf.Tensor]
    ) -> Dict[str, float]:
        """Custom test step implementation.

        Args:
            data: Tuple of (inputs, targets)

        Returns:
            Dictionary of metric results for this batch
        """
        x, y = data

        # Forward pass
        if self.reconstruction:
            predictions, reconstructed = self(x, training=False, mask=y)
            # Calculate losses
            margin_loss_value = margin_loss(y, predictions)
            reconstruction_loss = tf.reduce_mean(
                tf.square(x - reconstructed)
            )
            total_loss = margin_loss_value + 0.0005 * reconstruction_loss
        else:
            predictions = self(x, training=False)
            total_loss = margin_loss(y, predictions)

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        if self.reconstruction:
            results['reconstruction_loss'] = reconstruction_loss

        return results

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing model configuration
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "conv_filters": self.conv_filters,
            "primary_dims": self.primary_dims,
            "primary_caps": self.primary_caps,
            "digit_dims": self.digit_dims,
            "routing_iterations": self.routing_iterations,
            "reconstruction": self.reconstruction,
            "input_shape": self.input_shape
        })
        return config
