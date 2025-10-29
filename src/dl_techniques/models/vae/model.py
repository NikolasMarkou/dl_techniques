"""
Variational Autoencoder (VAE) Model Implementation
========================================================

A complete implementation of the ResNet-based VAE architecture using modern Keras 3 patterns.
VAE learns latent representations through variational inference with proper reparameterization
trick, KL divergence regularization, and generative capabilities.

Based on: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
https://arxiv.org/abs/1312.6114

Key Features:
------------
- ResNet-based encoder and decoder with residual connections
- Proper reparameterization trick for gradient flow
- KL divergence regularization with configurable weighting
- Support for multiple VAE variants for different image sizes
- Custom training loop with reconstruction + KL losses
- Numerical stability measures and gradient clipping
- Complete serialization support with modern Keras 3 patterns
- Production-ready implementation with comprehensive validation

Architecture Concept:
-------------------
VAE learns to encode images into a latent space following a prior distribution
(typically standard normal), then decodes from this space to reconstruct images.
The reparameterization trick enables gradient-based optimization through sampling.

Mathematical Foundation:
- Encoder: q(z|x) - Approximate posterior distribution
- Decoder: p(x|z) - Likelihood distribution
- Prior: p(z) = N(0, I) - Standard normal prior
- Loss: L = E[log p(x|z)] - β * KL(q(z|x)||p(z))
- Reparameterization: z = μ + σ * ε, where ε ~ N(0, I)

Model Variants:
--------------
- VAE-Micro: depths=2, filters=[16, 32], latent_dim=32 (small images)
- VAE-Small: depths=2, filters=[32, 64], latent_dim=64 (MNIST, CIFAR-10)
- VAE-Medium: depths=3, filters=[32, 64, 128], latent_dim=128 (larger images)
- VAE-Large: depths=3, filters=[64, 128, 256], latent_dim=256 (high-res images)
- VAE-XLarge: depths=4, filters=[64, 128, 256, 512], latent_dim=512 (very high-res)

Usage Examples:
-------------
```python
# MNIST VAE
model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=64)

# CIFAR-10 VAE
model = VAE.from_variant("medium", input_shape=(32, 32, 3), latent_dim=128)

# High-resolution VAE
model = VAE.from_variant("large", input_shape=(128, 128, 3), latent_dim=256)

# Custom VAE
model = create_vae(input_shape=(64, 64, 3), latent_dim=128, kl_loss_weight=0.01)
```
"""

import keras
import tensorflow as tf
from keras import layers, ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.sampling import Sampling

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VAE(keras.Model):
    """ResNet-based Variational Autoencoder using modern Keras 3 patterns.

    VAE learns latent representations through variational inference with
    ResNet-based encoder and decoder networks. Uses reparameterization trick
    for proper gradient flow and includes comprehensive numerical stability measures.

    Args:
        latent_dim: Integer, dimensionality of the latent space.
        input_shape: Tuple of integers, shape of input images (H, W, C).
        depths: Integer, number of depth levels in the encoder/decoder.
        steps_per_depth: Integer, number of residual blocks per depth level.
        filters: List of integers, filter counts for each depth level.
        kl_loss_weight: Float, weight for KL divergence loss term.
        kernel_initializer: String or initializer, weight initialization method.
        kernel_regularizer: Regularizer for convolutional weights.
        use_batch_norm: Boolean, whether to use batch normalization.
        use_bias: Boolean, whether to use bias terms.
        dropout_rate: Float, dropout rate for regularization.
        activation: String or callable, activation function.
        final_activation: String, activation for final reconstruction layer.
        name: String, name of the model.
        **kwargs: Additional arguments for keras.Model.

    Example:
        >>> # Create VAE for MNIST
        >>> model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=64)
        >>>
        >>> # Custom VAE configuration
        >>> model = VAE(
        ...     latent_dim=128,
        ...     input_shape=(64, 64, 3),
        ...     depths=3,
        ...     filters=[32, 64, 128],
        ...     kl_loss_weight=0.01
        ... )
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "micro": {
            "depths": 2,
            "steps_per_depth": 1,
            "filters": [16, 32],
            "default_latent_dim": 32,
            "kl_loss_weight": 0.01,
        },
        "small": {
            "depths": 2,
            "steps_per_depth": 1,
            "filters": [32, 64],
            "default_latent_dim": 64,
            "kl_loss_weight": 0.01,
        },
        "medium": {
            "depths": 3,
            "steps_per_depth": 1,
            "filters": [32, 64, 128],
            "default_latent_dim": 128,
            "kl_loss_weight": 0.005,
        },
        "large": {
            "depths": 3,
            "steps_per_depth": 2,
            "filters": [64, 128, 256],
            "default_latent_dim": 256,
            "kl_loss_weight": 0.005,
        },
        "xlarge": {
            "depths": 4,
            "steps_per_depth": 2,
            "filters": [64, 128, 256, 512],
            "default_latent_dim": 512,
            "kl_loss_weight": 0.001,
        },
    }

    # Architecture constants
    DEFAULT_ACTIVATION = "leaky_relu"
    DEFAULT_FINAL_ACTIVATION = "sigmoid"
    DEFAULT_INITIALIZER = "he_normal"

    def __init__(
        self,
        latent_dim: int,
        input_shape: Tuple[int, int, int],
        depths: int = 2,
        steps_per_depth: int = 1,
        filters: Optional[List[int]] = None,
        kl_loss_weight: float = 0.01,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_batch_norm: bool = True,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        activation: str = "leaky_relu",
        final_activation: str = "sigmoid",
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        # Validate inputs
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if depths <= 0:
            raise ValueError(f"depths must be positive, got {depths}")
        if steps_per_depth <= 0:
            raise ValueError(
                f"steps_per_depth must be positive, got {steps_per_depth}"
            )
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        # Set default filters if not provided
        if filters is None:
            filters = [32 * (2**i) for i in range(depths)]

        if len(filters) != depths:
            raise ValueError(
                f"Filters array length {len(filters)} must equal depths {depths}"
            )

        # Store configuration
        self.latent_dim = latent_dim
        self._input_shape = input_shape
        self.depths = depths
        self.steps_per_depth = steps_per_depth
        self.filters = filters
        self.kl_loss_weight = kl_loss_weight
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.final_activation = final_activation

        # Validate input dimensions
        height, width, channels = input_shape
        if height < 8 or width < 8:
            raise ValueError(
                f"Input dimensions must be at least 8x8, got {height}x{width}"
            )

        # Initialize metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        # Build the model using functional API
        inputs = keras.Input(shape=input_shape, name="input")
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, name=name or "vae", **kwargs)

        # Create a reusable decoder model from the main graph. This allows
        # self.decode() to reuse the trained decoder weights.
        decoder_input = self.get_layer("vae_sampling").output
        decoder_output = self.output["reconstruction"]
        self.decoder = keras.Model(decoder_input, decoder_output, name="decoder")

        logger.info(
            f"Created VAE model for input {input_shape} with latent_dim={latent_dim}, "
            f"depths={depths}, filters={filters}"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """Build the complete VAE model architecture.

        Args:
            inputs: Input tensor

        Returns:
            Dictionary containing all VAE outputs
        """
        # Build encoder
        z_mean, z_log_var = self._build_encoder(inputs)

        # Build sampling layer
        sampling_layer = Sampling(name="vae_sampling")
        z = sampling_layer([z_mean, z_log_var])

        # Build decoder
        reconstruction = self._build_decoder(z)

        return {
            "z": z,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction,
        }

    def _build_encoder(
        self, inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Build the encoder network with ResNet blocks.

        Args:
            inputs: Input tensor

        Returns:
            Tuple of (z_mean, z_log_var) tensors
        """
        x = inputs

        # Initial conv layer
        x = layers.Conv2D(
            filters=self.filters[0],
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="encoder_stem_conv",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(center=self.use_bias, name="encoder_stem_bn")(
                x
            )
        x = layers.Activation(self.activation, name="encoder_stem_activation")(x)

        # Encoder blocks with downsampling
        for depth in range(self.depths):
            x = self._build_encoder_stage(x, depth)

        # Global pooling and latent projection
        x = layers.GlobalAveragePooling2D(name="encoder_global_pool")(x)

        # Latent space projection
        z_mean = layers.Dense(
            units=self.latent_dim,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer="zeros",
            kernel_regularizer=self.kernel_regularizer,
            name="encoder_z_mean",
        )(x)

        z_log_var = layers.Dense(
            units=self.latent_dim,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=keras.initializers.Constant(
                -2.0
            ),  # Initialize to small variance
            kernel_regularizer=self.kernel_regularizer,
            name="encoder_z_log_var",
        )(x)

        return z_mean, z_log_var

    def _build_encoder_stage(
        self, x: keras.KerasTensor, stage_idx: int
    ) -> keras.KerasTensor:
        """Build a single encoder stage with downsampling and residual blocks.

        Args:
            x: Input tensor
            stage_idx: Index of the stage

        Returns:
            Output tensor from the stage
        """
        num_filters = self.filters[stage_idx]

        # Downsampling layer
        x = layers.Conv2D(
            filters=num_filters,
            kernel_size=2,
            strides=2,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"encoder_downsample_{stage_idx}",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(
                center=self.use_bias, name=f"encoder_downsample_bn_{stage_idx}"
            )(x)
        x = layers.Activation(
            self.activation, name=f"encoder_downsample_activation_{stage_idx}"
        )(x)

        # Residual blocks
        for step in range(self.steps_per_depth):
            x = self._build_residual_block(
                x, num_filters, f"encoder_{stage_idx}_{step}"
            )

        return x

    def _build_residual_block(
        self, x: keras.KerasTensor, filters: int, prefix: str
    ) -> keras.KerasTensor:
        """Build a residual block.

        Args:
            x: Input tensor
            filters: Number of filters
            prefix: Name prefix for layers

        Returns:
            Output tensor with residual connection
        """
        residual = x

        # First convolution
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{prefix}_conv_1",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(center=self.use_bias, name=f"{prefix}_bn_1")(
                x
            )
        x = layers.Activation(self.activation, name=f"{prefix}_activation_1")(x)

        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate, name=f"{prefix}_dropout")(x)

        # Second convolution
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"{prefix}_conv_2",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(center=self.use_bias, name=f"{prefix}_bn_2")(
                x
            )

        # Residual connection
        x = layers.Add(name=f"{prefix}_add")([x, residual])
        x = layers.Activation(self.activation, name=f"{prefix}_activation_final")(x)

        return x

    def _build_decoder(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Build the decoder network with ResNet blocks.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed image tensor
        """
        # Calculate feature map size after all downsampling
        height, width, channels = self._input_shape
        feature_height = height // (2**self.depths)
        feature_width = width // (2**self.depths)

        # Ensure minimum size
        feature_height = max(feature_height, 1)
        feature_width = max(feature_width, 1)

        # Project latent to feature map
        x = layers.Dense(
            units=feature_height * feature_width * self.filters[-1],
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="decoder_projection",
        )(z)

        x = layers.Reshape(
            (feature_height, feature_width, self.filters[-1]), name="decoder_reshape"
        )(x)

        # Decoder stages with upsampling
        for depth in range(self.depths - 1, -1, -1):
            x = self._build_decoder_stage(x, depth)

        # Final output layer
        x = layers.Conv2D(
            filters=channels,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=self.final_activation,
            use_bias=self.use_bias,
            kernel_regularizer=keras.regularizers.L1(1e-6),
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer="zeros",
            name="decoder_output",
        )(x)

        # Ensure exact shape matching
        if x.shape[1:] != self._input_shape:
            # Resize to exact input shape if needed
            target_height, target_width = self._input_shape[:2]
            x = layers.Resizing(
                height=target_height,
                width=target_width,
                interpolation="bilinear",
                name="decoder_resize",
            )(x)

        return x

    def _build_decoder_stage(
        self, x: keras.KerasTensor, stage_idx: int
    ) -> keras.KerasTensor:
        """Build a single decoder stage with upsampling and residual blocks.

        Args:
            x: Input tensor
            stage_idx: Index of the stage

        Returns:
            Output tensor from the stage
        """
        num_filters = self.filters[stage_idx]

        # Upsampling layer
        x = layers.UpSampling2D(
            size=(2, 2), interpolation="nearest", name=f"decoder_upsample_{stage_idx}"
        )(x)

        # Convolution after upsampling
        x = layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"decoder_conv_{stage_idx}",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(
                center=self.use_bias, name=f"decoder_bn_{stage_idx}"
            )(x)
        x = layers.Activation(self.activation, name=f"decoder_activation_{stage_idx}")(
            x
        )

        # Residual blocks
        for step in range(self.steps_per_depth):
            x = self._build_residual_block(
                x, num_filters, f"decoder_{stage_idx}_{step}"
            )

        return x

    @classmethod
    def from_variant(
        cls,
        variant: str,
        input_shape: Tuple[int, int, int],
        latent_dim: Optional[int] = None,
        **kwargs,
    ) -> "VAE":
        """Create a VAE model from a predefined variant.

        Args:
            variant: String, one of "micro", "small", "medium", "large", "xlarge"
            input_shape: Tuple, input image shape (H, W, C)
            latent_dim: Integer, latent dimension. If None, uses variant default
            **kwargs: Additional arguments passed to the constructor

        Returns:
            VAE model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # MNIST VAE
            >>> model = VAE.from_variant("small", input_shape=(28, 28, 1), latent_dim=64)
            >>> # CIFAR-10 VAE
            >>> model = VAE.from_variant("medium", input_shape=(32, 32, 3), latent_dim=128)
            >>> # High-resolution VAE
            >>> model = VAE.from_variant("large", input_shape=(128, 128, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        # Use provided latent_dim or variant default
        if latent_dim is None:
            latent_dim = config["default_latent_dim"]

        logger.info(f"Creating VAE-{variant.upper()} model")
        logger.info(f"Input shape: {input_shape}, Latent dim: {latent_dim}")

        return cls(
            latent_dim=latent_dim,
            input_shape=input_shape,
            depths=config["depths"],
            steps_per_depth=config["steps_per_depth"],
            filters=config["filters"],
            kl_loss_weight=config["kl_loss_weight"],
            **kwargs,
        )

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return metrics tracked by the model."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encode(
        self, inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode inputs to latent parameters.

        Args:
            inputs: Input tensor to encode

        Returns:
            Tuple of (z_mean, z_log_var) tensors
        """
        outputs = self(inputs, training=False)
        return outputs["z_mean"], outputs["z_log_var"]

    def decode(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Decode latent samples to reconstructions.

        Args:
            z: Latent tensor to decode

        Returns:
            Reconstructed tensor
        """
        return self.decoder(z)

    def sample(self, num_samples: int) -> keras.KerasTensor:
        """Generate samples from the latent space.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated samples tensor
        """
        z = keras.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)

    def train_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom training step with VAE losses.

        Args:
            data: Training data (can be tuple or single tensor)

        Returns:
            Dictionary of loss values
        """
        # Handle different data formats
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Validate input shape
        if x.shape[1:] != self._input_shape:
            logger.warning(
                f"Input shape {x.shape} doesn't match expected {self._input_shape}"
            )

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(x, training=True)
            reconstruction = outputs["reconstruction"]

            # Validate reconstruction shape
            if reconstruction.shape != x.shape:
                raise ValueError(
                    f"Reconstruction shape {reconstruction.shape} "
                    f"doesn't match input {x.shape}"
                )

            # Compute losses
            reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)
            kl_loss = self._compute_kl_loss(outputs["z_mean"], outputs["z_log_var"])

            # Total loss
            total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

            # Add regularization losses
            if self.losses:
                total_loss += ops.sum(self.losses)

        # Compute and clip gradients
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(total_loss, trainable_weights)
        gradients = [
            ops.clip(grad, -1.0, 1.0) if grad is not None else None
            for grad in gradients
        ]

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom test step with VAE losses.

        Args:
            data: Test data (can be tuple or single tensor)

        Returns:
            Dictionary of loss values
        """
        # Handle different data formats
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Forward pass
        outputs = self(x, training=False)
        reconstruction = outputs["reconstruction"]

        # Compute losses
        reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)
        kl_loss = self._compute_kl_loss(outputs["z_mean"], outputs["z_log_var"])
        total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

        # Add regularization losses
        if self.losses:
            total_loss += ops.sum(self.losses)

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def _compute_reconstruction_loss(
        self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute reconstruction loss with numerical stability.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Reconstruction loss value
        """
        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
            )

        # Flatten for loss computation
        y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
        y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

        # Clip predictions to avoid log(0)
        y_pred_clipped = ops.clip(y_pred_flat, 1e-7, 1.0 - 1e-7)

        # Binary crossentropy for better numerical stability
        reconstruction_loss = ops.mean(
            keras.losses.binary_crossentropy(y_true_flat, y_pred_clipped)
        )

        return reconstruction_loss

    def _compute_kl_loss(
        self, z_mean: keras.KerasTensor, z_log_var: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute KL divergence loss with numerical stability.

        Args:
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution

        Returns:
            KL divergence loss value
        """
        # Clip log variance to prevent numerical issues
        z_log_var_clipped = ops.clip(z_log_var, -20.0, 20.0)

        # Compute KL divergence: KL(q||p) = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * ops.sum(
            1.0 + z_log_var_clipped - ops.square(z_mean) - ops.exp(z_log_var_clipped),
            axis=1,
        )

        # Take mean across batch
        kl_loss = ops.mean(kl_loss)

        return kl_loss

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = {
            "latent_dim": self.latent_dim,
            "input_shape": self._input_shape,
            "depths": self.depths,
            "steps_per_depth": self.steps_per_depth,
            "filters": self.filters,
            "kl_loss_weight": self.kl_loss_weight,
            "kernel_initializer": keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "final_activation": self.final_activation,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VAE":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            VAE model instance
        """
        # Deserialize complex objects
        if config.get("kernel_initializer"):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        # Convert input_shape from list back to tuple
        if "input_shape" in config and isinstance(config["input_shape"], list):
            config["input_shape"] = tuple(config["input_shape"])

        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info("VAE configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Latent dimension: {self.latent_dim}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Steps per depth: {self.steps_per_depth}")
        logger.info(f"  - Filters: {self.filters}")
        logger.info(f"  - KL loss weight: {self.kl_loss_weight}")
        logger.info(f"  - Total parameters: {self.count_params():,}")


# ---------------------------------------------------------------------


def create_vae(
    input_shape: Tuple[int, int, int],
    latent_dim: int,
    variant: str = "small",
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs,
) -> VAE:
    """Convenience function to create and compile VAE models.

    Args:
        input_shape: Tuple representing (height, width, channels) of input
        latent_dim: Integer, dimensionality of the latent space
        variant: String, model variant ("micro", "small", "medium", "large", "xlarge")
        optimizer: String name or optimizer instance. Default is "adam"
        learning_rate: Float, learning rate for optimizer. Default is 0.001
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Compiled VAE model ready for training

    Example:
        >>> # MNIST VAE
        >>> model = create_vae(input_shape=(28, 28, 1), latent_dim=64, variant="small")
        >>>
        >>> # CIFAR-10 VAE
        >>> model = create_vae(input_shape=(32, 32, 3), latent_dim=128, variant="medium")
        >>>
        >>> # Custom learning rate
        >>> model = create_vae(
        ...     input_shape=(64, 64, 3),
        ...     latent_dim=256,
        ...     variant="large",
        ...     learning_rate=0.0005
        ... )
    """
    # Create the model
    model = VAE.from_variant(
        variant=variant, input_shape=input_shape, latent_dim=latent_dim, **kwargs
    )

    # Set up optimizer
    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, "learning_rate"):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    # Compile the model
    model.compile(optimizer=optimizer_instance)

    # Validate the model works
    test_input = keras.random.uniform((2,) + input_shape)
    test_output = model(test_input, training=False)

    # Validate outputs
    assert (
        test_output["reconstruction"].shape == test_input.shape
    ), "Reconstruction shape mismatch"
    assert test_output["z_mean"].shape == (
        2,
        latent_dim,
    ), "z_mean shape mismatch"
    assert test_output["z_log_var"].shape == (
        2,
        latent_dim,
    ), "z_log_var shape mismatch"

    logger.info(f"Created VAE-{variant.upper()} for input shape {input_shape}")
    logger.info(f"Latent dim: {latent_dim}, Parameters: {model.count_params():,}")

    return model


def create_vae_from_config(
    config: Dict[str, Any],
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
) -> VAE:
    """Create VAE from configuration dictionary.

    Args:
        config: Configuration dictionary containing VAE parameters
        optimizer: String name or optimizer instance
        learning_rate: Float, learning rate for optimizer

    Returns:
        Compiled VAE model

    Example:
        >>> config = {
        ...     "latent_dim": 128,
        ...     "input_shape": (64, 64, 3),
        ...     "depths": 3,
        ...     "filters": [32, 64, 128],
        ...     "kl_loss_weight": 0.01
        ... }
        >>> model = create_vae_from_config(config)
    """
    # Create the model
    model = VAE(**config)

    # Set up optimizer
    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, "learning_rate"):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    # Compile the model
    model.compile(optimizer=optimizer_instance)

    logger.info(f"Created VAE from config with latent_dim={config['latent_dim']}")

    return model
