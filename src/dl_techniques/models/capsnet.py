"""
Modern Keras implementation of Capsule Networks (CapsNet) - Architecture Only.

This module provides a clean Keras-based implementation of the Capsule Network
architecture, following the paper "Dynamic Routing Between Capsules" by Sabour et al.
The training logic is separated into a dedicated trainer class.

Architecture Overview:
    1. Feature Extraction: Conv2D layers for initial feature extraction
    2. Primary Capsules: Convert conventional CNN features to capsule format
    3. Routing Capsules: Final capsule layer with dynamic routing
    4. Decoder Network (optional): For reconstruction-based regularization

Features:
    - Clean separation of architecture and training logic
    - Customizable architecture with flexible parameters
    - Optional reconstruction decoder for regularization
    - Full support for model serialization and loading
    - Comprehensive type hints and documentation

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017).
      Dynamic routing between capsules. In Advances in Neural
      Information Processing Systems (pp. 3856-3866).
"""

import os
import keras
from keras import ops
from typing import Optional, Tuple, Union, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.capsules import PrimaryCapsule, RoutingCapsule
from dl_techniques.utils.tensors import length
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CapsNet(keras.Model):
    """Complete Capsule Network model implementation.

    This model implements the full CapsNet architecture as described in the paper
    "Dynamic Routing Between Capsules" with optional reconstruction network.
    Training logic is handled separately to maintain clean separation of concerns.

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
        """Validate initialization parameters.

        Args:
            num_classes: Number of output classes.
            routing_iterations: Number of routing iterations.
            primary_capsules: Number of primary capsules.
            primary_capsule_dim: Dimension of primary capsule vectors.
            digit_capsule_dim: Dimension of digit capsule vectors.
            reconstruction: Whether reconstruction is enabled.
            input_shape: Input shape for reconstruction.

        Raises:
            ValueError: If any parameter is invalid.
        """
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
        """Process and validate regularizer parameter.

        Args:
            regularizer: Regularizer specification.

        Returns:
            Processed regularizer instance or None.
        """
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
        """Build the model layers based on input shape.

        Args:
            input_shape: Shape of input tensor (batch, height, width, channels).

        Raises:
            ValueError: If input shape is invalid.
        """
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
                name=f"conv_{i + 1}"
            )
            self.conv_layers.append(conv_layer)

            # Batch normalization (optional)
            if self.use_batch_norm:
                bn_layer = keras.layers.BatchNormalization(name=f"bn_{i + 1}")
                self.batch_norm_layers.append(bn_layer)
            else:
                self.batch_norm_layers.append(None)

            # Activation
            activation_layer = keras.layers.ReLU(name=f"relu_{i + 1}")
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
                    name=f"decoder_hidden_{i + 1}"
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
        """Forward pass through the capsule network.

        Args:
            inputs: Input images of shape [batch_size, height, width, channels].
            training: Whether in training mode.
            mask: Optional mask for reconstruction (one-hot labels).

        Returns:
            Dictionary containing:
                - 'digit_caps': Digit capsule outputs
                - 'length': Capsule lengths (class probabilities)
                - 'reconstructed': Reconstructed images (if reconstruction enabled)

        Raises:
            ValueError: If input shape is incorrect or mask shape is invalid.
        """
        # Validate input
        input_shape = ops.shape(inputs)
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
        """Perform reconstruction using the decoder network.

        Args:
            digit_caps: Digit capsule outputs.
            lengths: Capsule lengths.
            mask: Optional one-hot mask for reconstruction.

        Returns:
            Reconstructed images.

        Raises:
            ValueError: If mask shape is invalid.
        """
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

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], int, int, int]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute output shapes based on input shape.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Dictionary of output shapes for each output.

        Raises:
            ValueError: If input shape is invalid.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape [batch, height, width, channels], got {input_shape}")

        batch_size = input_shape[0]

        output_shapes = {
            "digit_caps": (batch_size, self.num_classes, self.digit_capsule_dim),
            "length": (batch_size, self.num_classes)
        }

        # Add reconstruction shape if applicable
        if self.reconstruction and self._input_shape is not None:
            output_shapes["reconstructed"] = (batch_size,) + self._input_shape

        return output_shapes

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing model configuration.
        """
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
            "use_batch_norm": self.use_batch_norm
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CapsNet":
        """Create model from configuration.

        Args:
            config: Model configuration dictionary.

        Returns:
            New model instance.
        """
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
        """Save the model to a file.

        Args:
            filepath: Path to save the model.
            overwrite: Whether to overwrite existing file.
            save_format: Format to save the model ('keras' or 'tf').
        """
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save model
        self.save(filepath, overwrite=overwrite, save_format=save_format)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "CapsNet":
        """Load a saved model.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded model instance.
        """
        from dl_techniques.losses.capsule_margin_loss import capsule_margin_loss

        custom_objects = {
            "CapsNet": cls,
            "PrimaryCapsule": PrimaryCapsule,
            "RoutingCapsule": RoutingCapsule,
            "capsule_margin_loss": capsule_margin_loss,
            "length": length
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

# ---------------------------------------------------------------------
