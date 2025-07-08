"""
Keras Spatial Variational Autoencoder (Spatial VAE) for Image Data.

This module provides a novel VAE implementation where instead of projecting the
entire image to a single latent vector, each spatial location in the encoder's
feature space gets its own latent representation. This allows for more
fine-grained spatial control and better preservation of spatial structure.

In a traditional VAE:
- Image → Encoder → Single latent vector (z_mean, z_log_var) → Decoder → Reconstructed image

In a Spatial VAE:
- Image → Encoder → Spatial latent maps (H×W×latent_dim for mean and log_var) → Decoder → Reconstructed image

This approach enables:
- Spatial locality preservation
- Fine-grained spatial manipulation
- Better spatial understanding
- More interpretable latent representations

Typical Usage:
    >>> from dl_techniques.models.spatial_vae import create_spatial_vae
    >>>
    >>> # Create spatial VAE for MNIST
    >>> spatial_vae = create_spatial_vae(
    ...     input_shape=(28, 28, 1),
    ...     latent_dim=8,
    ...     encoder_filters=[32, 64],
    ...     decoder_filters=[64, 32],
    ...     spatial_latent_size=(7, 7)  # 7x7 spatial latent grid
    ... )
    >>>
    >>> # Train the model
    >>> spatial_vae.fit(train_dataset, epochs=50, validation_data=val_dataset)
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any, List

from dl_techniques.utils.logger import logger


# Export the classes so they can be imported for model loading
__all__ = ['SpatialSampling', 'SpatialVAE', 'create_spatial_vae']


@keras.saving.register_keras_serializable()
class SpatialSampling(keras.layers.Layer):
    """Spatial sampling layer for Spatial VAE.

    This layer performs reparameterization trick on spatial feature maps,
    sampling from each spatial location independently.

    Args:
        seed: Integer, random seed for sampling.
        **kwargs: Additional keyword arguments for Layer base class.
    """

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seed = seed

    def call(self, inputs: List[keras.KerasTensor], training: Optional[bool] = None) -> keras.KerasTensor:
        """Perform spatial reparameterization trick.

        Args:
            inputs: List containing [z_mean, z_log_var] spatial tensors.
            training: Whether in training mode.

        Returns:
            Sampled spatial latent tensor.
        """
        z_mean, z_log_var = inputs
        batch_size = ops.shape(z_mean)[0]
        spatial_dims = ops.shape(z_mean)[1:-1]  # (H, W)
        latent_dim = ops.shape(z_mean)[-1]

        # Sample epsilon with same spatial structure
        epsilon = keras.random.normal(
            shape=(batch_size,) + tuple(spatial_dims) + (latent_dim,),
            seed=self.seed if training else None
        )

        # Reparameterization: z = mean + std * epsilon
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({"seed": self.seed})
        return config


@keras.saving.register_keras_serializable()
class SpatialVAE(keras.Model):
    """Spatial Variational Autoencoder (Spatial VAE) model.

    This VAE variant maintains spatial structure in the latent space by encoding
    each spatial location in the feature map independently, rather than collapsing
    to a single latent vector.

    Args:
        latent_dim: Integer, dimensionality of the latent space at each spatial location.
        spatial_latent_size: Tuple of integers (H, W) specifying the spatial dimensions
            of the latent representation. If None, will be inferred from encoder output.
        encoder_filters: List of integers specifying the number of filters for
            each encoder convolutional layer. Defaults to [32, 64].
        decoder_filters: List of integers specifying the number of filters for
            each decoder transposed convolutional layer. Defaults to [64, 32].
        kl_loss_weight: Float, weight for the KL divergence loss term. Defaults to 1.0.
        input_shape: Optional tuple specifying the input image shape (H, W, C).
        kernel_initializer: String name or initializer instance for conv layers.
        kernel_regularizer: String name or regularizer instance for conv layers.
        use_batch_norm: Boolean, whether to use batch normalization layers.
        dropout_rate: Float between 0 and 1, dropout rate for regularization.
        activation: String or callable, activation function for hidden layers.
        name: Optional string name for the model.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        Dictionary containing:
        - 'reconstruction': 4D tensor with same shape as input
        - 'z_mean': 4D tensor with shape `(batch_size, spatial_H, spatial_W, latent_dim)`
        - 'z_log_var': 4D tensor with shape `(batch_size, spatial_H, spatial_W, latent_dim)`

    Example:
        >>> # Create a Spatial VAE for CIFAR-10
        >>> spatial_vae = SpatialVAE(
        ...     latent_dim=16,
        ...     input_shape=(32, 32, 3),
        ...     encoder_filters=[32, 64, 128],
        ...     decoder_filters=[128, 64, 32],
        ...     spatial_latent_size=(8, 8)
        ... )
        >>>
        >>> # Train the model
        >>> spatial_vae.compile(optimizer='adam')
        >>> spatial_vae.fit(train_data, epochs=50)
    """

    def __init__(
        self,
        latent_dim: int,
        spatial_latent_size: Optional[Tuple[int, int]] = None,
        encoder_filters: List[int] = None,
        decoder_filters: List[int] = None,
        kl_loss_weight: float = 1.0,
        input_shape: Optional[Tuple[int, int, int]] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        activation: Union[str, callable] = "leaky_relu",
        name: Optional[str] = "spatial_vae",
        **kwargs: Any
    ) -> None:
        """Initialize the Spatial VAE model."""
        super().__init__(name=name, **kwargs)

        # Store configuration
        self.latent_dim = latent_dim
        self.spatial_latent_size = spatial_latent_size
        self.encoder_filters = encoder_filters or [32, 64]
        self.decoder_filters = decoder_filters or [64, 32]
        self.kl_loss_weight = kl_loss_weight
        self._input_shape_arg = input_shape
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Core layers
        self.encoder = None
        self.spatial_z_mean_conv = None
        self.spatial_z_log_var_conv = None
        self.spatial_sampling_layer = None
        self.decoder = None

        # Shape information
        self._encoder_output_shape = None
        self._build_input_shape = None

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        logger.info(f"Initialized Spatial VAE with latent_dim={latent_dim}, "
                   f"spatial_latent_size={spatial_latent_size}, "
                   f"encoder_filters={self.encoder_filters}")

        # Build immediately if input_shape is provided
        if self._input_shape_arg is not None:
            self.build((None,) + tuple(self._input_shape_arg))

    def build(self, input_shape: Tuple) -> None:
        """Build the Spatial VAE architecture."""
        if self.built:
            return

        self._build_input_shape = input_shape
        if len(input_shape) > 1:
            self._input_shape_arg = tuple(input_shape[1:])

        # Build encoder
        if self.encoder is None:
            self._build_encoder()

        # Calculate encoder output shape
        if self._encoder_output_shape is None:
            self._calculate_encoder_output_shape()

        # Build spatial latent projection layers
        if self.spatial_z_mean_conv is None:
            self.spatial_z_mean_conv = keras.layers.Conv2D(
                filters=self.latent_dim,
                kernel_size=1,
                padding="same",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="spatial_z_mean_conv"
            )

        if self.spatial_z_log_var_conv is None:
            self.spatial_z_log_var_conv = keras.layers.Conv2D(
                filters=self.latent_dim,
                kernel_size=1,
                padding="same",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="spatial_z_log_var_conv"
            )

        # Build sampling layer
        if self.spatial_sampling_layer is None:
            self.spatial_sampling_layer = SpatialSampling(seed=42, name="spatial_sampling")

        # Build decoder
        if self.decoder is None:
            self._build_decoder()

        # Build all sublayers properly
        if self._input_shape_arg is not None:
            dummy_input = ops.zeros((1,) + tuple(self._input_shape_arg))

            # Build encoder
            encoder_features = self.encoder(dummy_input)

            # Build spatial convolution layers
            _ = self.spatial_z_mean_conv(encoder_features)
            _ = self.spatial_z_log_var_conv(encoder_features)

            # Build decoder
            dummy_latent = ops.zeros((1,) + self.spatial_latent_size + (self.latent_dim,))
            _ = self.decoder(dummy_latent)

        super().build(input_shape)
        logger.info(f"Spatial VAE built successfully with input shape: {input_shape}")

    def _build_encoder(self) -> None:
        """Build the encoder network."""
        encoder_layers = []

        for i, filters in enumerate(self.encoder_filters):
            # Convolutional layer
            encoder_layers.append(
                keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"encoder_conv_{i}"
                )
            )

            # Batch normalization
            if self.use_batch_norm:
                encoder_layers.append(
                    keras.layers.BatchNormalization(name=f"encoder_bn_{i}")
                )

            # Activation
            if self.activation == "leaky_relu":
                encoder_layers.append(keras.layers.LeakyReLU(name=f"encoder_act_{i}"))
            else:
                encoder_layers.append(
                    keras.layers.Activation(self.activation, name=f"encoder_act_{i}")
                )

            # Dropout
            if self.dropout_rate > 0:
                encoder_layers.append(
                    keras.layers.Dropout(self.dropout_rate, name=f"encoder_dropout_{i}")
                )

        self.encoder = keras.Sequential(encoder_layers, name="encoder")

    def _calculate_encoder_output_shape(self) -> None:
        """Calculate the shape after encoder conv layers."""
        if self._input_shape_arg is None:
            logger.warning("Cannot calculate encoder output shape: input_shape_arg is None")
            return

        height, width, channels = self._input_shape_arg
        current_height, current_width = height, width

        logger.info(f"Starting shape calculation from: {self._input_shape_arg}")

        # Calculate shape after each conv layer (stride=2)
        for i, filters in enumerate(self.encoder_filters):
            prev_height, prev_width = current_height, current_width
            current_height = (current_height + 1) // 2
            current_width = (current_width + 1) // 2
            logger.info(f"Conv layer {i}: {prev_height}x{prev_width} -> {current_height}x{current_width}")

        last_filters = self.encoder_filters[-1]
        self._encoder_output_shape = (current_height, current_width, last_filters)

        # Set spatial latent size if not provided
        if self.spatial_latent_size is None:
            self.spatial_latent_size = (current_height, current_width)
            logger.info(f"Auto-set spatial_latent_size to: {self.spatial_latent_size}")

        logger.info(f"Final encoder output shape: {self._encoder_output_shape}")

    def _build_decoder(self) -> None:
        """Build the decoder network."""
        if self._encoder_output_shape is None:
            raise ValueError("Encoder output shape not calculated. Build encoder first.")

        decoder_layers = []

        # Start with spatial latent input (H, W, latent_dim)
        # First, project to higher channel dimension if needed
        if self.latent_dim != self.encoder_filters[-1]:
            decoder_layers.append(
                keras.layers.Conv2D(
                    filters=self.encoder_filters[-1],
                    kernel_size=1,
                    padding="same",
                    activation="relu",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name="decoder_input_projection"
                )
            )

        # Transposed convolutional layers
        for i, filters in enumerate(self.decoder_filters):
            decoder_layers.append(
                keras.layers.UpSampling2D(size=(2,2))
            )
            decoder_layers.append(
                keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"decoder_conv_transpose_{i}"
                )
            )

            # Batch normalization
            if self.use_batch_norm:
                decoder_layers.append(
                    keras.layers.BatchNormalization(name=f"decoder_bn_{i}")
                )

            # Activation
            if self.activation == "leaky_relu":
                decoder_layers.append(keras.layers.LeakyReLU(name=f"decoder_act_{i}"))
            else:
                decoder_layers.append(
                    keras.layers.Activation(self.activation, name=f"decoder_act_{i}")
                )

            # Dropout
            if self.dropout_rate > 0:
                decoder_layers.append(
                    keras.layers.Dropout(self.dropout_rate, name=f"decoder_dropout_{i}")
                )

        # Final output layer
        decoder_layers.append(
            keras.layers.Conv2DTranspose(
                filters=self._input_shape_arg[-1],
                kernel_size=3,
                strides=1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                name="decoder_output"
            )
        )

        self.decoder = keras.Sequential(decoder_layers, name="decoder")

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return list of metrics tracked by the model."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass through the Spatial VAE.

        Args:
            inputs: Input images tensor.
            training: Whether in training mode.

        Returns:
            Dictionary containing reconstruction, z_mean, and z_log_var.
        """
        # Encoder pass - outputs spatial feature maps
        encoder_features = self.encoder(inputs, training=training)

        # Project to spatial latent parameters
        z_mean = self.spatial_z_mean_conv(encoder_features, training=training)
        z_log_var = self.spatial_z_log_var_conv(encoder_features, training=training)

        # Sample from spatial latent distribution
        z = self.spatial_sampling_layer([z_mean, z_log_var], training=training)

        # Decoder pass
        reconstruction = self.decoder(z, training=training)

        return {
            "reconstruction": reconstruction,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

    def encode(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode inputs to spatial latent parameters.

        Args:
            inputs: Input images tensor.

        Returns:
            Tuple of (z_mean, z_log_var) spatial tensors.
        """
        if not self.built:
            self.build(inputs.shape)

        encoder_features = self.encoder(inputs, training=False)
        z_mean = self.spatial_z_mean_conv(encoder_features, training=False)
        z_log_var = self.spatial_z_log_var_conv(encoder_features, training=False)
        return z_mean, z_log_var

    def decode(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Decode spatial latent samples to reconstructions.

        Args:
            z: Spatial latent samples tensor.

        Returns:
            Reconstructed images tensor.
        """
        if not self.built:
            raise ValueError("Model must be built before decoding.")

        return self.decoder(z, training=False)

    def sample(self, num_samples: int) -> keras.KerasTensor:
        """Generate samples from the spatial latent space.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Generated images tensor.
        """
        if not self.built:
            raise ValueError("Model must be built before sampling.")

        z = keras.random.normal(
            shape=(num_samples,) + self.spatial_latent_size + (self.latent_dim,)
        )
        return self.decode(z)

    def train_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom training step for Spatial VAE."""
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)

            # Reconstruction loss
            reconstruction_loss = self._compute_reconstruction_loss(x, outputs["reconstruction"])

            # Spatial KL divergence loss
            kl_loss = self._compute_spatial_kl_loss(outputs["z_mean"], outputs["z_log_var"])

            # Total loss
            total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

            # Add regularization losses
            if self.losses:
                total_loss += ops.sum(self.losses)

        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom test step for Spatial VAE."""
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        outputs = self(x, training=False)

        # Compute losses
        reconstruction_loss = self._compute_reconstruction_loss(x, outputs["reconstruction"])
        kl_loss = self._compute_spatial_kl_loss(outputs["z_mean"], outputs["z_log_var"])
        total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

        # Add regularization losses
        if self.losses:
            total_loss += ops.sum(self.losses)

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def _compute_reconstruction_loss(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute reconstruction loss (MSE)."""
        y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
        y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

        per_sample_mse = ops.mean(ops.square(y_true_flat - y_pred_flat), axis=1)
        reconstruction_loss = ops.mean(per_sample_mse)
        return reconstruction_loss

    def _compute_spatial_kl_loss(
        self,
        z_mean: keras.KerasTensor,
        z_log_var: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute spatial KL divergence loss.

        Args:
            z_mean: Spatial latent mean tensor (B, H, W, latent_dim).
            z_log_var: Spatial latent log variance tensor (B, H, W, latent_dim).

        Returns:
            KL divergence loss scalar.
        """
        # Compute KL divergence for each spatial location and latent dimension
        kl_per_location = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))

        # Sum over latent dimensions and spatial locations, then average over batch
        kl_loss = ops.mean(ops.sum(kl_per_location, axis=[1, 2, 3]))
        return kl_loss

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "spatial_latent_size": self.spatial_latent_size,
            "encoder_filters": self.encoder_filters,
            "decoder_filters": self.decoder_filters,
            "kl_loss_weight": self.kl_loss_weight,
            "input_shape": self._input_shape_arg,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpatialVAE":
        """Create Spatial VAE from configuration."""
        # Deserialize initializers and regularizers
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "kernel_regularizer" in config and isinstance(config["kernel_regularizer"], dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)


def create_spatial_vae(
    input_shape: Tuple[int, int, int],
    latent_dim: int,
    spatial_latent_size: Optional[Tuple[int, int]] = None,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs
) -> SpatialVAE:
    """Create and compile a Spatial VAE model.

    Args:
        input_shape: Tuple specifying input image shape (height, width, channels).
        latent_dim: Integer, dimensionality of the latent space at each spatial location.
        spatial_latent_size: Optional tuple (H, W) for spatial latent dimensions.
        optimizer: String name or optimizer instance.
        learning_rate: Float, learning rate for the optimizer.
        **kwargs: Additional arguments passed to SpatialVAE constructor.

    Returns:
        Compiled Spatial VAE model ready for training.

    Example:
        >>> # Create a Spatial VAE for CIFAR-10
        >>> spatial_vae = create_spatial_vae(
        ...     input_shape=(32, 32, 3),
        ...     latent_dim=16,
        ...     encoder_filters=[32, 64, 128],
        ...     decoder_filters=[128, 64, 32],
        ...     spatial_latent_size=(8, 8),
        ...     kl_loss_weight=0.5
        ... )
    """
    # Create the model
    model = SpatialVAE(
        latent_dim=latent_dim,
        spatial_latent_size=spatial_latent_size,
        input_shape=input_shape,
        **kwargs
    )

    # Set up optimizer
    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, 'learning_rate'):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    # Compile the model
    model.compile(optimizer=optimizer_instance)

    # Ensure the model is built and summary is meaningful
    if not model.built:
        model.build((None,) + input_shape)

    logger.info(f"Created and compiled Spatial VAE with input_shape={input_shape}, "
               f"latent_dim={latent_dim}, spatial_latent_size={spatial_latent_size}")

    return model