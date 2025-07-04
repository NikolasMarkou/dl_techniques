"""
Keras Variational Autoencoder (VAE) for Image Data.

This module provides a flexible and fully Keras-compliant implementation of a
Variational Autoencoder (VAE), specifically designed for image data using a
convolutional architecture.

A VAE is a generative model that learns a probabilistic mapping from a
high-dimensional input space (e.g., images) to a low-dimensional, continuous
latent space, and a corresponding mapping back to the input space.

The model consists of two main components:

1. **Encoder**: A neural network (here, convolutional) that takes an input image
   and outputs the parameters—mean (`z_mean`) and log-variance (`z_log_var`)—of
   a Gaussian distribution in the latent space.
2. **Decoder**: A neural network (here, transposed convolutional) that takes a
   sample `z` from the latent distribution and attempts to reconstruct the
   original input image.

The training process optimizes a two-part loss function:

- **Reconstruction Loss**: Measures how well the decoder reconstructs the input.
  This implementation uses Mean Squared Error.
- **Kullback-Leibler (KL) Divergence**: A regularization term that forces the
  learned latent distributions to be close to a standard normal distribution
  (mean 0, variance 1). This encourages a smooth and well-structured latent
  space suitable for generation.

Typical Usage:
    >>> # Import the factory function
    >>> from dl_techniques.models.vae import create_vae
    >>>
    >>> # Define model parameters
    >>> input_shape = (28, 28, 1)  # e.g., for MNIST
    >>> latent_dim = 16
    >>>
    >>> # Create and compile the model
    >>> vae = create_vae(
    ...     input_shape=input_shape,
    ...     latent_dim=latent_dim,
    ...     encoder_filters=[32, 64],
    ...     decoder_filters=[64, 32],
    ...     kl_loss_weight=0.5,
    ...     learning_rate=0.001
    ... )
    >>>
    >>> # Print the model summary
    >>> vae.summary()
    >>>
    >>> # Train the model
    >>> vae.fit(train_dataset, epochs=20, validation_data=val_dataset)
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.sampling import Sampling

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VAE(keras.Model):
    """Keras-compliant Variational Autoencoder (VAE) model.

    This class implements a convolutional Variational Autoencoder that can be used
    for learning latent representations of image data and generating new samples.

    The model uses a convolutional encoder to map input images to a latent space
    and a deconvolutional decoder to reconstruct images from latent samples.

    Args:
        latent_dim: Integer, dimensionality of the latent space.
        encoder_filters: List of integers specifying the number of filters for
            each encoder convolutional layer. Defaults to [32, 64].
        decoder_filters: List of integers specifying the number of filters for
            each decoder transposed convolutional layer. Defaults to [64, 32].
        kl_loss_weight: Float, weight for the KL divergence loss term in the
            total loss. Defaults to 1.0.
        input_shape: Optional tuple specifying the input image shape
            (height, width, channels). If provided, the model will be built
            immediately. If None, the model will be built on first call.
        kernel_initializer: String name or initializer instance for conv layers.
            Defaults to "he_normal".
        kernel_regularizer: String name or regularizer instance for conv layers.
            Defaults to None.
        use_batch_norm: Boolean, whether to use batch normalization layers.
            Defaults to True.
        dropout_rate: Float between 0 and 1, dropout rate for regularization.
            Defaults to 0.0 (no dropout).
        activation: String or callable, activation function for hidden layers.
            Defaults to "leaky_relu".
        name: Optional string name for the model. Defaults to "vae".
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        Dictionary containing:
        - 'reconstruction': 4D tensor with same shape as input
        - 'z_mean': 2D tensor with shape `(batch_size, latent_dim)`
        - 'z_log_var': 2D tensor with shape `(batch_size, latent_dim)`

    Example:
        >>> # Create a VAE for MNIST-like data
        >>> vae = VAE(
        ...     latent_dim=16,
        ...     input_shape=(28, 28, 1),
        ...     encoder_filters=[32, 64],
        ...     decoder_filters=[64, 32]
        ... )
        >>>
        >>> # Compile and train
        >>> vae.compile(optimizer='adam')
        >>> vae.fit(train_data, epochs=50)
        >>>
        >>> # Generate new samples
        >>> latent_samples = keras.random.normal((10, 16))
        >>> generated = vae.decode(latent_samples)
    """

    def __init__(
        self,
        latent_dim: int,
        encoder_filters: List[int] = None,
        decoder_filters: List[int] = None,
        kl_loss_weight: float = 1.0,
        input_shape: Optional[Tuple[int, int, int]] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        activation: Union[str, callable] = "leaky_relu",
        name: Optional[str] = "vae",
        **kwargs: Any
    ) -> None:
        """Initialize the VAE model.

        Args:
            latent_dim: Dimensionality of the latent space.
            encoder_filters: List of filter counts for encoder layers.
            decoder_filters: List of filter counts for decoder layers.
            kl_loss_weight: Weight for KL divergence loss.
            input_shape: Input image shape (H, W, C).
            kernel_initializer: Weight initializer for conv layers.
            kernel_regularizer: Weight regularizer for conv layers.
            use_batch_norm: Whether to use batch normalization.
            dropout_rate: Dropout rate for regularization.
            activation: Activation function for hidden layers.
            name: Model name.
            **kwargs: Additional arguments for Model base class.
        """
        super().__init__(name=name, **kwargs)

        # Store configuration
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters or [32, 64]
        self.decoder_filters = decoder_filters or [64, 32]
        self.kl_loss_weight = kl_loss_weight
        self._input_shape_arg = input_shape
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Core layers that will be built in build()
        self.encoder = None
        self.decoder = None
        self.sampling_layer = None

        # Shape information for decoder
        self._encoder_output_shape = None
        self._build_input_shape = None

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        logger.info(f"Initialized VAE with latent_dim={latent_dim}, "
                   f"encoder_filters={self.encoder_filters}, "
                   f"decoder_filters={self.decoder_filters}")

        # If input_shape is provided, build the model immediately.
        # This makes the model usable right after instantiation without
        # needing to see data first.
        if self._input_shape_arg is not None:
            # `build` expects a batch dimension, so we add `None`.
            self.build((None,) + tuple(self._input_shape_arg))

    def build(self, input_shape: Tuple) -> None:
        """Build the VAE architecture.

        Args:
            input_shape: Shape of input tensor including batch dimension.
        """
        if self.built:
            return

        self._build_input_shape = input_shape
        # Ensure _input_shape_arg is a tuple without batch dimension
        if len(input_shape) > 1:
             self._input_shape_arg = tuple(input_shape[1:])

        # Only build if not already built during initialization
        if self.encoder is None:
            # Build encoder
            self._build_encoder()

        # Calculate encoder output shape if not already done
        if self._encoder_output_shape is None:
            self._calculate_encoder_output_shape()

        # Only build if not already built during initialization
        if self.sampling_layer is None:
            # Build sampling layer
            self.sampling_layer = Sampling(seed=42, name="vae_sampling")

        # Only build if not already built during initialization
        if self.decoder is None:
            # Build decoder
            self._build_decoder()

        # Call parent build after setting up our components
        super().build(input_shape)

        logger.info(f"VAE built successfully with input shape: {input_shape}")

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

        # Flatten and dense layers for latent parameters
        encoder_layers.extend([
            keras.layers.Flatten(name="encoder_flatten"),
            keras.layers.Dense(
                self.latent_dim * 2,  # Output both mean and log_var
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="encoder_latent_dense"
            )
        ])

        self.encoder = keras.Sequential(encoder_layers, name="encoder")

    def _calculate_encoder_output_shape(self) -> None:
        """Calculate the shape after encoder conv layers (before flatten)."""
        if self._input_shape_arg is None:
            logger.warning("Cannot calculate encoder output shape: input_shape_arg is None")
            return

        # Calculate shape after each conv layer (stride=2)
        height, width, channels = self._input_shape_arg
        current_height, current_width = height, width

        logger.info(f"Starting shape calculation from: {self._input_shape_arg}")
        logger.info(f"Encoder filters: {self.encoder_filters}")

        # Calculate shape after each conv layer (stride=2)
        for i, _ in enumerate(self.encoder_filters):
            prev_height, prev_width = current_height, current_width
            current_height = (current_height + 1) // 2  # Ceiling division for stride=2
            current_width = (current_width + 1) // 2
            logger.info(f"Conv layer {i}: {prev_height}x{prev_width} -> {current_height}x{current_width}")

        # Store the shape before flattening
        last_filters = self.encoder_filters[-1]
        self._encoder_output_shape = (current_height, current_width, last_filters)

        logger.info(f"Final encoder output shape before flatten: {self._encoder_output_shape}")

    def _build_decoder(self) -> None:
        """Build the decoder network."""
        if self._encoder_output_shape is None:
            raise ValueError("Encoder output shape not calculated. Build encoder first.")

        decoder_layers = []

        # Dense layer to reshape from latent space
        decoder_layers.append(
            keras.layers.Dense(
                units=self._encoder_output_shape[0] *
                      self._encoder_output_shape[1] *
                      self._encoder_output_shape[2],
                activation="relu",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="decoder_dense"
            )
        )

        # Reshape to feature map
        decoder_layers.append(
            keras.layers.Reshape(
                target_shape=self._encoder_output_shape,
                name="decoder_reshape"
            )
        )

        # Transposed convolutional layers
        for i, filters in enumerate(self.decoder_filters):
            decoder_layers.append(
                keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=3,
                    strides=2,
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
                filters=self._input_shape_arg[-1],  # Output channels
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
        """Forward pass through the VAE.

        Args:
            inputs: Input images tensor.
            training: Whether in training mode.

        Returns:
            Dictionary containing reconstruction, z_mean, and z_log_var.
        """
        # Encoder pass
        encoder_output = self.encoder(inputs, training=training)

        # Split into mean and log variance
        z_mean = encoder_output[:, :self.latent_dim]
        z_log_var = encoder_output[:, self.latent_dim:]
        z = self.sampling_layer([z_mean, z_log_var], training=training)

        # Decoder pass
        reconstruction = self.decoder(z, training=training)

        return {
            "reconstruction": reconstruction,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
        }

    def encode(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode inputs to latent parameters.

        Args:
            inputs: Input images tensor.

        Returns:
            Tuple of (z_mean, z_log_var) tensors.
        """
        if not self.built:
            self.build(inputs.shape)

        encoder_output = self.encoder(inputs, training=False)
        z_mean = encoder_output[:, :self.latent_dim]
        z_log_var = encoder_output[:, self.latent_dim:]
        return z_mean, z_log_var

    def decode(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Decode latent samples to reconstructions.

        Args:
            z: Latent samples tensor.

        Returns:
            Reconstructed images tensor.
        """
        if not self.built:
            raise ValueError("Model must be built before decoding. Call model.build() or encode some data first.")

        return self.decoder(z, training=False)

    def sample(self, num_samples: int) -> keras.KerasTensor:
        """Generate samples from the latent space.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Generated images tensor.
        """
        if not self.built:
            raise ValueError("Model must be built before sampling. Provide input_shape during initialization or call model.build().")

        z = keras.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)

    def train_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom training step for VAE.

        Args:
            data: Training data (can be tuple or single tensor).

        Returns:
            Dictionary of metric values.
        """
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            outputs = self(x, training=True)

            # Reconstruction loss
            reconstruction_loss = self._compute_reconstruction_loss(x, outputs["reconstruction"])

            # KL divergence loss
            kl_loss = self._compute_kl_loss(outputs["z_mean"], outputs["z_log_var"])

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
        """Custom test step for VAE.

        Args:
            data: Test data (can be tuple or single tensor).

        Returns:
            Dictionary of metric values.
        """
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        outputs = self(x, training=False)

        # Compute losses
        reconstruction_loss = self._compute_reconstruction_loss(x, outputs["reconstruction"])
        kl_loss = self._compute_kl_loss(outputs["z_mean"], outputs["z_log_var"])
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
        """Compute reconstruction loss (MSE).

        Args:
            y_true: True images.
            y_pred: Reconstructed images.

        Returns:
            Reconstruction loss scalar.
        """
        # The original implementation summed squared errors (SSE per sample), which
        # creates a loss value that is dependent on image size. This can unbalance
        # the total loss w.r.t the KL divergence. Using the Mean Squared Error
        # (MSE per sample) normalizes the loss and typically leads to more
        # stable training.
        y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
        y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

        per_sample_mse = ops.mean(ops.square(y_true_flat - y_pred_flat), axis=1)
        reconstruction_loss = ops.mean(per_sample_mse)
        return reconstruction_loss

    def _compute_kl_loss(
        self,
        z_mean: keras.KerasTensor,
        z_log_var: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute KL divergence loss.

        Args:
            z_mean: Latent mean tensor.
            z_log_var: Latent log variance tensor.

        Returns:
            KL divergence loss scalar.
        """
        kl_loss = -0.5 * ops.mean(
            ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                axis=1
            )
        )
        return kl_loss

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
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
    def from_config(cls, config: Dict[str, Any]) -> "VAE":
        """Create VAE from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            VAE instance.
        """
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

# ---------------------------------------------------------------------

def create_vae(
    input_shape: Tuple[int, int, int],
    latent_dim: int,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs
) -> VAE:
    """Create and compile a VAE model.

    This is a factory function that creates a VAE model with the specified
    parameters and compiles it with the given optimizer.

    Args:
        input_shape: Tuple specifying input image shape (height, width, channels).
        latent_dim: Integer, dimensionality of the latent space.
        optimizer: String name or optimizer instance. Defaults to "adam".
        learning_rate: Float, learning rate for the optimizer. Defaults to 0.001.
        **kwargs: Additional arguments passed to VAE constructor.

    Returns:
        Compiled VAE model ready for training.

    Example:
        >>> # Create a VAE for CIFAR-10 like data
        >>> vae = create_vae(
        ...     input_shape=(32, 32, 3),
        ...     latent_dim=128,
        ...     encoder_filters=[32, 64, 128],
        ...     decoder_filters=[128, 64, 32],
        ...     kl_loss_weight=0.5,
        ...     learning_rate=0.001
        ... )
        >>>
        >>> # Train the model
        >>> history = vae.fit(
        ...     train_dataset,
        ...     epochs=100,
        ...     validation_data=val_dataset
        ... )
    """
    # Create the model
    model = VAE(
        latent_dim=latent_dim,
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

    # Ensure the model is built (already handled in __init__ if input_shape is provided)
    if not model.built:
        model.build((None,) + input_shape)

    logger.info(f"Created and compiled VAE with input_shape={input_shape}, "
               f"latent_dim={latent_dim}")

    return model

# ---------------------------------------------------------------------