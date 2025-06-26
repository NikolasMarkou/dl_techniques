"""
Keras-compliant implementation of Capsule Networks (CapsNet).

This module provides a fully Keras-compliant implementation of the Capsule Network
architecture that works with standard Keras training workflows (compile/fit).
The custom training logic is integrated into the model's train_step and test_step methods.

Architecture Overview:
    1. Feature Extraction: Conv2D layers for initial feature extraction
    2. Primary Capsules: Convert conventional CNN features to capsule format
    3. Routing Capsules: Final capsule layer with dynamic routing
    4. Decoder Network (optional): For reconstruction-based regularization

Features:
    - Full Keras compliance with compile/fit workflow
    - Custom train_step and test_step for capsule-specific losses
    - Integrated margin loss and reconstruction loss
    - Customizable architecture with flexible parameters
    - Comprehensive metrics and logging

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017).
      Dynamic routing between capsules. In Advances in Neural
      Information Processing Systems (pp. 3856-3866).
"""

import os
import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any, List

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import length
from dl_techniques.losses.capsule_margin_loss import capsule_margin_loss
from dl_techniques.layers.capsules import PrimaryCapsule, RoutingCapsule

class CapsuleAccuracy(keras.metrics.Metric):
    """Custom accuracy metric for capsule networks based on capsule lengths."""

    def __init__(self, name: str = "capsule_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None):
        """Update accuracy state based on capsule lengths.

        Args:
            y_true: One-hot encoded true labels.
            y_pred: Dictionary containing 'length' key with capsule lengths.
            sample_weight: Optional sample weights.
        """
        if isinstance(y_pred, dict) and "length" in y_pred:
            lengths = y_pred["length"]
        else:
            lengths = y_pred

        y_true_classes = ops.argmax(y_true, axis=1)
        y_pred_classes = ops.argmax(lengths, axis=1)

        matches = ops.cast(ops.equal(y_true_classes, y_pred_classes), self.dtype)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            matches = ops.multiply(matches, sample_weight)

        self.total.assign_add(ops.sum(matches))
        self.count.assign_add(ops.cast(ops.size(matches), self.dtype))

    def result(self):
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


@keras.saving.register_keras_serializable()
class CapsNet(keras.Model):
    """Keras-compliant Capsule Network model.

    This model implements the full CapsNet architecture with integrated training logic
    that works seamlessly with Keras compile/fit workflow.

    Args:
        num_classes: Number of output classes.
        routing_iterations: Number of routing iterations for capsule routing.
        conv_filters: List of filter numbers for convolutional layers.
        primary_capsules: Number of primary capsules.
        primary_capsule_dim: Dimension of primary capsule vectors.
        digit_capsule_dim: Dimension of digit/class capsule vectors.
        reconstruction: Whether to include reconstruction network.
        input_shape: Shape of input images (height, width, channels).
        decoder_architecture: List of hidden layer sizes for decoder network.
        kernel_initializer: Initializer for convolutional weights.
        kernel_regularizer: Regularizer for convolutional weights.
        use_batch_norm: Whether to use batch normalization after convolutions.
        positive_margin: Positive margin for margin loss (m^+).
        negative_margin: Negative margin for margin loss (m^-).
        downweight: Downweight parameter for negative class loss (λ).
        reconstruction_weight: Weight for reconstruction loss component.
        name: Optional name for the model.
        **kwargs: Additional keyword arguments for the base Model class.

    Raises:
        ValueError: If any parameter is invalid or inconsistent.
    """

    def __init__(
        self,
        num_classes: int,
        routing_iterations: int = 3,
        conv_filters: List[int] = [256, 256],
        primary_capsules: int = 32,
        primary_capsule_dim: int = 8,
        digit_capsule_dim: int = 16,
        reconstruction: bool = True,
        input_shape: Optional[Tuple[int, int, int]] = None,
        decoder_architecture: List[int] = [512, 1024],
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_batch_norm: bool = True,
        positive_margin: float = 0.9,
        negative_margin: float = 0.1,
        downweight: float = 0.5,
        reconstruction_weight: float = 0.01,
        name: Optional[str] = "capsnet",
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Validate inputs
        self._validate_parameters(
            num_classes, routing_iterations, primary_capsules,
            primary_capsule_dim, digit_capsule_dim, reconstruction, input_shape
        )

        # Store configuration
        self.num_classes = num_classes
        self.routing_iterations = routing_iterations
        self.conv_filters = conv_filters.copy()
        self.primary_capsules = primary_capsules
        self.primary_capsule_dim = primary_capsule_dim
        self.digit_capsule_dim = digit_capsule_dim
        self.reconstruction = reconstruction
        self._input_shape = input_shape
        self.decoder_architecture = decoder_architecture.copy()
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = self._process_regularizer(kernel_regularizer)
        self.use_batch_norm = use_batch_norm

        # Loss configuration
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.downweight = downweight
        self.reconstruction_weight = reconstruction_weight

        # Initialize layer containers
        self.conv_layers = []
        self.batch_norm_layers = []
        self.activation_layers = []
        self.primary_caps = None
        self.digit_caps = None
        self.decoder = None

        # Build status
        self._layers_built = False

    def _validate_parameters(
        self,
        num_classes: int,
        routing_iterations: int,
        primary_capsules: int,
        primary_capsule_dim: int,
        digit_capsule_dim: int,
        reconstruction: bool,
        input_shape: Optional[Tuple[int, int, int]]
    ) -> None:
        """Validate initialization parameters."""
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if routing_iterations <= 0:
            raise ValueError(f"routing_iterations must be positive, got {routing_iterations}")
        if primary_capsules <= 0:
            raise ValueError(f"primary_capsules must be positive, got {primary_capsules}")
        if primary_capsule_dim <= 0:
            raise ValueError(f"primary_capsule_dim must be positive, got {primary_capsule_dim}")
        if digit_capsule_dim <= 0:
            raise ValueError(f"digit_capsule_dim must be positive, got {digit_capsule_dim}")
        if reconstruction and input_shape is None:
            logger.warning(
                "Reconstruction enabled but input_shape not provided. "
                "Decoder will be created during build if input shape is available."
            )

    def _process_regularizer(
        self,
        regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    ) -> Optional[keras.regularizers.Regularizer]:
        """Process and validate regularizer parameter."""
        if regularizer is None:
            return None
        if isinstance(regularizer, str):
            regularizer_map = {
                "l1": keras.regularizers.L1(0.01),
                "l2": keras.regularizers.L2(0.01),
                "l1_l2": keras.regularizers.L1L2(l1=0.01, l2=0.01)
            }
            if regularizer.lower() in regularizer_map:
                return regularizer_map[regularizer.lower()]
            else:
                return keras.regularizers.get(regularizer)
        return keras.regularizers.get(regularizer)

    def build(self, input_shape: Tuple[Optional[int], int, int, int]) -> None:
        """Build the model layers based on input shape."""
        if self._layers_built:
            return

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape [batch, height, width, channels], got {input_shape}"
            )

        logger.info(f"Building CapsNet with input shape: {input_shape}")

        # Store input shape for reconstruction if not provided during init
        if self.reconstruction and self._input_shape is None:
            self._input_shape = tuple(input_shape[1:])

        # Build feature extraction layers
        self._build_feature_extraction()

        # Build capsule layers
        self._build_capsule_layers()

        # Build decoder if needed
        if self.reconstruction and self._input_shape is not None:
            self._build_decoder()

        self._layers_built = True
        super().build(input_shape)

    def _build_feature_extraction(self) -> None:
        """Build convolutional feature extraction layers."""
        for i, filters in enumerate(self.conv_filters):
            # Convolutional layer
            conv_layer = keras.layers.Conv2D(
                filters=filters,
                kernel_size=9 if i == 0 else 5,  # First layer uses 9x9, others use 5x5
                strides=1,
                padding="valid",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"conv_{i+1}"
            )
            self.conv_layers.append(conv_layer)

            # Batch normalization (optional)
            if self.use_batch_norm:
                bn_layer = keras.layers.BatchNormalization(name=f"bn_{i+1}")
                self.batch_norm_layers.append(bn_layer)
            else:
                self.batch_norm_layers.append(None)

            # Activation
            activation_layer = keras.layers.ReLU(name=f"relu_{i+1}")
            self.activation_layers.append(activation_layer)

    def _build_capsule_layers(self) -> None:
        """Build primary and routing capsule layers."""
        # Primary capsule layer
        self.primary_caps = PrimaryCapsule(
            num_capsules=self.primary_capsules,
            dim_capsules=self.primary_capsule_dim,
            kernel_size=9,
            strides=2,
            padding="valid",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="primary_caps"
        )

        # Digit/class capsule layer
        self.digit_caps = RoutingCapsule(
            num_capsules=self.num_classes,
            dim_capsules=self.digit_capsule_dim,
            routing_iterations=self.routing_iterations,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="digit_caps"
        )

    def _build_decoder(self) -> None:
        """Build reconstruction decoder network."""
        if self._input_shape is None:
            raise ValueError("Cannot build decoder without input_shape")

        decoder_layers = []

        # Hidden layers
        for i, units in enumerate(self.decoder_architecture):
            decoder_layers.append(
                keras.layers.Dense(
                    units=units,
                    activation="relu",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"decoder_hidden_{i+1}"
                )
            )

        # Calculate output size
        flattened_size = int(self._input_shape[0] * self._input_shape[1] * self._input_shape[2])

        # Output layer
        decoder_layers.append(
            keras.layers.Dense(
                units=flattened_size,
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="decoder_output"
            )
        )

        # Reshape to original input shape
        decoder_layers.append(
            keras.layers.Reshape(
                target_shape=self._input_shape,
                name="decoder_reshape"
            )
        )

        # Create sequential decoder
        self.decoder = keras.Sequential(decoder_layers, name="reconstruction_decoder")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass through the capsule network."""
        # Validate input
        if len(inputs.shape) != 4:
            raise ValueError(f"Expected 4D input [batch, height, width, channels], got shape {inputs.shape}")

        # Feature extraction
        x = inputs
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if self.use_batch_norm and self.batch_norm_layers[i] is not None:
                x = self.batch_norm_layers[i](x, training=training)
            x = self.activation_layers[i](x)

        # Primary capsules
        primary_caps_output = self.primary_caps(x)

        # Digit capsules
        digit_caps_output = self.digit_caps(primary_caps_output)

        # Calculate capsule lengths (class probabilities)
        lengths = length(digit_caps_output)

        # Prepare results
        results = {
            "digit_caps": digit_caps_output,
            "length": lengths
        }

        # Handle reconstruction if enabled
        if self.reconstruction and self.decoder is not None:
            reconstructed = self._reconstruct(digit_caps_output, lengths, mask)
            results["reconstructed"] = reconstructed

        return results

    def _reconstruct(
        self,
        digit_caps: keras.KerasTensor,
        lengths: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Perform reconstruction using the decoder network."""
        if mask is not None:
            # Validate mask shape
            if mask.shape[-1] != self.num_classes:
                raise ValueError(
                    f"Mask shape mismatch. Expected last dimension {self.num_classes}, "
                    f"got {mask.shape[-1]}"
                )
            # Use provided mask (one-hot encoded labels)
            reconstruction_mask = mask
        else:
            # Use predicted classes for reconstruction
            reconstruction_mask = ops.one_hot(ops.argmax(lengths, axis=1), num_classes=self.num_classes)

        # Mask digit capsules
        masked_caps = ops.multiply(digit_caps, ops.expand_dims(reconstruction_mask, -1))

        # Flatten for decoder input
        decoder_input = ops.reshape(masked_caps, (-1, self.num_classes * self.digit_capsule_dim))

        # Generate reconstruction
        return self.decoder(decoder_input)

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom training step with margin loss and reconstruction loss."""
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(x, training=True, mask=y)

            # Calculate margin loss
            margin_loss_value = ops.mean(capsule_margin_loss(
                outputs["length"],  # y_pred
                y,                  # y_true
                self.downweight,
                self.positive_margin,
                self.negative_margin
            ))

            # Initialize total loss with margin loss
            total_loss = margin_loss_value
            reconstruction_loss_value = ops.convert_to_tensor(0.0, dtype=total_loss.dtype)

            # Add reconstruction loss if applicable
            if self.reconstruction and "reconstructed" in outputs:
                reconstruction_loss_value = ops.mean(ops.square(x - outputs["reconstructed"]))
                total_loss += self.reconstruction_weight * reconstruction_loss_value

            # Add regularization losses
            if self.losses:
                total_loss += ops.sum(self.losses)

        # Calculate gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics manually - avoiding deprecated compiled_metrics
        for metric in self.metrics:
            if isinstance(metric, CapsuleAccuracy):
                metric.update_state(y, outputs)
            else:
                # For other metrics, try to update with appropriate format
                try:
                    metric.update_state(y, outputs["length"])
                except:
                    # If that fails, skip this metric
                    continue

        # Return metrics
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.result()

        results.update({
            "loss": total_loss,
            "margin_loss": margin_loss_value,
            "reconstruction_loss": reconstruction_loss_value
        })

        return results

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom test step with margin loss and reconstruction loss."""
        x, y = data

        # Forward pass
        outputs = self(x, training=False, mask=y)

        # Calculate margin loss
        margin_loss_value = ops.mean(capsule_margin_loss(
            outputs["length"],  # y_pred
            y,                  # y_true
            self.downweight,
            self.positive_margin,
            self.negative_margin
        ))

        # Initialize total loss with margin loss
        total_loss = margin_loss_value
        reconstruction_loss_value = ops.convert_to_tensor(0.0, dtype=total_loss.dtype)

        # Add reconstruction loss if applicable
        if self.reconstruction and "reconstructed" in outputs:
            reconstruction_loss_value = ops.mean(ops.square(x - outputs["reconstructed"]))
            total_loss += self.reconstruction_weight * reconstruction_loss_value

        # Add regularization losses
        if self.losses:
            total_loss += ops.sum(self.losses)

        # Update metrics manually - avoiding deprecated compiled_metrics
        for metric in self.metrics:
            if isinstance(metric, CapsuleAccuracy):
                metric.update_state(y, outputs)
            else:
                # For other metrics, try to update with appropriate format
                try:
                    metric.update_state(y, outputs["length"])
                except:
                    # If that fails, skip this metric
                    continue

        # Return metrics
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.result()

        results.update({
            "loss": total_loss,
            "margin_loss": margin_loss_value,
            "reconstruction_loss": reconstruction_loss_value
        })

        return results

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "routing_iterations": self.routing_iterations,
            "conv_filters": self.conv_filters,
            "primary_capsules": self.primary_capsules,
            "primary_capsule_dim": self.primary_capsule_dim,
            "digit_capsule_dim": self.digit_capsule_dim,
            "reconstruction": self.reconstruction,
            "input_shape": self._input_shape,
            "decoder_architecture": self.decoder_architecture,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "positive_margin": self.positive_margin,
            "negative_margin": self.negative_margin,
            "downweight": self.downweight,
            "reconstruction_weight": self.reconstruction_weight
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CapsNet":
        """Create model from configuration."""
        # Handle serialized objects
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def save_model(
        self,
        filepath: str,
        overwrite: bool = True,
        save_format: str = "keras"
    ) -> None:
        """Save the model to a file."""
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save model
        self.save(filepath, overwrite=overwrite, save_format=save_format)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "CapsNet":
        """Load a saved model."""
        custom_objects = {
            "CapsNet": cls,
            "PrimaryCapsule": PrimaryCapsule,
            "RoutingCapsule": RoutingCapsule,
            "capsule_margin_loss": capsule_margin_loss,
            "length": length,
            "CapsuleAccuracy": CapsuleAccuracy
        }

        model = keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info(f"Model loaded from {filepath}")
        return model

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional information."""
        super().summary(**kwargs)
        logger.info(f"CapsNet Configuration:")
        logger.info(f"  - Classes: {self.num_classes}")
        logger.info(f"  - Routing iterations: {self.routing_iterations}")
        logger.info(f"  - Conv filters: {self.conv_filters}")
        logger.info(f"  - Primary capsules: {self.primary_capsules} x {self.primary_capsule_dim}D")
        logger.info(f"  - Digit capsules: {self.num_classes} x {self.digit_capsule_dim}D")
        logger.info(f"  - Reconstruction: {self.reconstruction}")
        logger.info(f"  - Batch normalization: {self.use_batch_norm}")
        logger.info(f"  - Margins: +{self.positive_margin}, -{self.negative_margin}")
        logger.info(f"  - Downweight: {self.downweight}")
        logger.info(f"  - Reconstruction weight: {self.reconstruction_weight}")


# Helper function to create and compile CapsNet
def create_capsnet(
    num_classes: int,
    input_shape: Tuple[int, int, int],
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs
) -> CapsNet:
    """Create and compile a CapsNet model.

    Args:
        num_classes: Number of output classes.
        input_shape: Shape of input images.
        optimizer: Optimizer name or instance.
        learning_rate: Learning rate for optimizer.
        **kwargs: Additional arguments for CapsNet.

    Returns:
        Compiled CapsNet model.
    """
    model = CapsNet(
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    # Handle optimizer
    if isinstance(optimizer, str):
        optimizer = keras.optimizers.get(optimizer)
        optimizer.learning_rate = learning_rate

    # Compile model with dummy loss (we handle loss in train_step/test_step)
    model.compile(
        optimizer=optimizer,
        loss=None,  # We handle loss computation in train_step/test_step
        metrics=[CapsuleAccuracy()]
    )

    return model