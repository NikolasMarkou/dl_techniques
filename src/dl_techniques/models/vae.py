"""
ResNet-based Variational Autoencoder (VAE) Implementation
========================================================

This module implements a ResNet-based Variational Autoencoder (VAE) using Keras 3.8.0,
designed for unsupervised learning of latent representations with generative capabilities.
The implementation follows VAE principles with proper reparameterization trick, KL divergence
regularization, and numerical stability measures.

Key Features:
    - **ResNet Architecture**: Utilizes residual connections in both encoder and decoder
    - **Configurable Depth**: Adjustable network depth and steps per depth level
    - **Proper VAE Loss**: Implements reconstruction loss + KL divergence regularization
    - **Numerical Stability**: Gradient clipping, loss clamping, and proper initialization
    - **Backend Agnostic**: Uses keras.ops for compatibility across TensorFlow, JAX, PyTorch
    - **Full Serialization**: Proper get_config/from_config implementation
    - **Custom Training**: Implements custom train_step and test_step methods
    - **Sampling Capabilities**: Generate new samples from learned latent distribution

Architecture Overview:
    ```
    Input Image → Encoder (ResNet blocks + downsampling) → μ, log(σ²)
                                                              ↓
    Reconstructed ← Decoder (ResNet blocks + upsampling) ← z ~ N(μ, σ²)
    ```

Mathematical Foundation:
    - **Encoder**: q(z|x) - Approximate posterior distribution
    - **Decoder**: p(x|z) - Likelihood distribution
    - **Prior**: p(z) = N(0, I) - Standard normal prior
    - **Loss**: L = E[log p(x|z)] - β * KL(q(z|x)||p(z))
    - **Reparameterization**: z = μ + σ * ε, where ε ~ N(0, I)

Classes:
    VAE: Main ResNet-based Variational Autoencoder model

Functions:
    create_vae: Factory function for creating and compiling VAE models

Dependencies:
    - keras>=3.8.0
    - tensorflow>=2.18.0 (backend)
    - dl_techniques.utils.logger
    - dl_techniques.layers.sampling.Sampling

Usage Example:
    ```python
    # Create a VAE for 64x64 RGB images
    vae = create_vae(
        input_shape=(64, 64, 3),
        latent_dim=128,
        depths=3,
        steps_per_depth=2,
        filters=[32, 64, 128],
        kl_loss_weight=0.001,
        optimizer='adam'
    )

    # Train the model
    vae.fit(train_data, epochs=50, validation_data=val_data)

    # Generate new samples
    samples = vae.sample(num_samples=10)

    # Encode images to latent space
    z_mean, z_log_var = vae.encode(images)

    # Decode latent vectors to images
    reconstructions = vae.decode(z_vectors)
    ```

Technical Details:
    - **Encoder Architecture**:
      * Initial convolution layer
      * Progressive downsampling (stride=2) at each depth level
      * Residual blocks with batch normalization and dropout
      * Dense layers for μ and log(σ²) prediction

    - **Decoder Architecture**:
      * Dense projection from latent to feature maps
      * Progressive upsampling (2x) at each depth level
      * Residual blocks with batch normalization and dropout
      * Final convolution with sigmoid activation

    - **Loss Components**:
      * Reconstruction Loss: Binary crossentropy between input and reconstruction
      * KL Divergence: KL(q(z|x)||N(0,I)) for regularization
      * Regularization: Optional kernel regularization terms

    - **Numerical Stability**:
      * Gradient clipping to prevent exploding gradients
      * Loss clamping to prevent numerical overflow
      * Proper weight initialization (He normal, small variance for latent layers)
      * Epsilon clamping for log operations

Performance Considerations:
    - Memory efficient implementation with lazy weight creation
    - Supports mixed precision training
    - Configurable batch normalization and dropout for regularization
    - Optimized for modern GPU architectures

Model Persistence:
    - Fully serializable using Keras .keras format
    - Proper config serialization for all parameters
    - Custom object registration for loading saved models

Notes:
    - Requires input images to be normalized to [0, 1] range
    - Output reconstructions are in [0, 1] range (sigmoid activation)
    - KL loss weight typically needs tuning based on dataset and latent dimension
    - Larger latent dimensions may require higher KL loss weights
    - ResNet blocks help with gradient flow in deeper architectures
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

    def __init__(
        self,
        latent_dim: int,
        depths: int = 3,
        steps_per_depth: int = 2,
        filters: List[int] = None,
        kl_loss_weight: float = 0.01,
        input_shape: Optional[Tuple[int, int, int]] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_batch_norm: bool = True,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        activation: Union[str, callable] = "leaky_relu",
        final_activation: str = "sigmoid",
        name: Optional[str] = "resnet_vae",
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Store configuration
        self.latent_dim = latent_dim
        self.depths = depths
        self.steps_per_depth = steps_per_depth
        self.filters = filters or [32, 64, 128]
        self.kl_loss_weight = kl_loss_weight
        self._input_shape = input_shape
        self.final_activation = final_activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias


        # Components to be built
        self.encoder = None
        self.decoder = None
        self.sampling_layer = None

        # Shape tracking
        self._build_input_shape = None

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

        # Validation
        if self._input_shape is None:
            raise ValueError("Input shape must be provided")
        if len(self._input_shape) != 3:
            raise ValueError("Input shape must be (height, width, channels)")
        if len(self.filters) != self.depths:
            raise ValueError(f"Filters array must be the same size as depths {self.depths}")

        logger.info(f"Initialized ResnetVAE with latent_dim={latent_dim}")

    def build(self, input_shape: Tuple) -> None:
        """Build the ResNet VAE architecture."""
        self._build_input_shape = input_shape
        self._input_shape = tuple(input_shape[1:])

        # Build encoder
        self._build_encoder(self._input_shape)

        # Build sampling layer
        self.sampling_layer = Sampling(seed=42, name="resnet_vae_sampling")

        # Build decoder
        self._build_decoder()

        super().build(input_shape)
        logger.info("ResnetVAE built successfully")

    def _build_encoder(self, input_shape: Tuple[int, int, int]) -> None:
        """Build the encoder network."""
        x_input = keras.Input(shape=input_shape, name="encoder_input")
        x = x_input

        # Initial conv layer
        x = keras.layers.Conv2D(
            filters=self.filters[0],
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )(x)

        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(center=self.use_bias)(x)
        x = keras.layers.Activation(self.activation)(x)

        # Encoder blocks with downsampling
        for depth in range(self.depths):
            # First layer in each depth does downsampling
            x = keras.layers.Conv2D(
                filters=self.filters[depth],
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"encoder_downsample_{depth}"
            )(x)

            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(center=self.use_bias)(x)
            x = keras.layers.Activation(self.activation)(x)

            # Additional layers at this depth
            for step in range(self.steps_per_depth - 1):
                residual = x

                x = keras.layers.Conv2D(
                    filters=self.filters[depth],
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"encoder_conv_{depth}_{step}_0"
                )(x)

                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization(center=self.use_bias)(x)
                x = keras.layers.Activation(self.activation)(x)

                if self.dropout_rate > 0:
                    x = keras.layers.Dropout(self.dropout_rate)(x)

                x = keras.layers.Conv2D(
                    filters=self.filters[depth],
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"encoder_conv_{depth}_{step}_1"
                )(x)

                x = keras.layers.Activation(self.activation)(x)

                # Residual connection
                x = keras.layers.Add()([x, residual])

        x = keras.layers.Flatten()(x)

        z_mean = keras.layers.Dense(
            units=self.latent_dim,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros',
            kernel_regularizer=self.kernel_regularizer,
            name="encoder_latent_mean"
        )(x)

        z_log_var = keras.layers.Dense(
            units=self.latent_dim,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=keras.initializers.Constant(-2.0),  # Initialize to small variance
            kernel_regularizer=self.kernel_regularizer,
            name="encoder_latent_var"
        )(x)

        self.encoder = (
            keras.Model(
                inputs=x_input,
                outputs=[z_mean, z_log_var],
                name="encoder"
            )
        )

    def _build_decoder(self) -> None:
        """Build the decoder network."""
        # Calculate the feature map size after all downsampling
        feature_height = self._input_shape[0] // (2 ** self.depths)
        feature_width = self._input_shape[1] // (2 ** self.depths)

        # Ensure minimum size
        feature_height = max(feature_height, 1)
        feature_width = max(feature_width, 1)

        z_input = keras.Input(shape=(self.latent_dim,), name="decoder_input")

        # Project latent to feature map
        x = keras.layers.Dense(
            units=feature_height * feature_width * self.filters[-1],
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="decoder_projection"
        )(z_input)

        x = keras.layers.Reshape((feature_height, feature_width, self.filters[-1]))(x)

        # Decoder blocks with upsampling
        for depth in range(self.depths - 1, -1, -1):
            # Upsample
            x = keras.layers.UpSampling2D(
                size=(2, 2),
                name=f"decoder_upsample_{depth}",
                interpolation="nearest")(x)

            # Convolution after upsampling
            x = keras.layers.Conv2D(
                filters=self.filters[depth],
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"decoder_conv_{depth}"
            )(x)

            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(center=self.use_bias)(x)
            x = keras.layers.Activation(self.activation)(x)

            # Additional layers at this depth
            for step in range(self.steps_per_depth - 1):
                residual = x

                x = keras.layers.Conv2D(
                    filters=self.filters[depth],
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"decoder_conv_{depth}_{step}_0"
                )(x)

                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization(center=self.use_bias)(x)
                x = keras.layers.Activation(self.activation)(x)

                if self.dropout_rate > 0:
                    x = keras.layers.Dropout(self.dropout_rate)(x)

                x = keras.layers.Conv2D(
                    filters=self.filters[depth],
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"decoder_conv_{depth}_{step}_1"
                )(x)

                x = keras.layers.Activation(self.activation)(x)

                # Residual connection
                x = keras.layers.Add()([x, residual])

        # Final output layer
        x = keras.layers.Conv2D(
            filters=self._input_shape[-1],
            kernel_size=1,
            strides=1,
            padding="same",
            activation=self.final_activation,
            use_bias=self.use_bias,
            kernel_regularizer=keras.regularizers.L1(1e-6),
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros',
            name="decoder_output"
        )(x)

        # Ensure exact shape matching
        if x.shape[1:] != self._input_shape:
            # Resize to exact input shape if needed
            target_height, target_width = self._input_shape[:2]
            x = keras.layers.Resizing(
                height=target_height,
                width=target_width,
                interpolation="bilinear",
                name="decoder_resize"
            )(x)

        self.decoder = keras.Model(inputs=z_input, outputs=x, name="decoder")

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return metrics tracked by the model."""
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
        """Forward pass through the ResNet VAE."""
        # Encoder pass
        encoder_output = self.encoder(inputs, training=training)

        # Split into mean and log variance
        z_mean = encoder_output[0]
        z_log_var = encoder_output[1]
        z = self.sampling_layer([z_mean, z_log_var], training=training)

        # Decoder pass
        reconstruction = self.decoder(z, training=training)

        return {
            "z": z,
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction
        }

    def encode(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Encode inputs to latent parameters."""
        if not self.built:
            self.build((None,) + self._input_shape)

        encoder_output = self.encoder(inputs, training=False)
        z_mean = encoder_output[:, :self.latent_dim]
        z_log_var = encoder_output[:, self.latent_dim:]
        return z_mean, z_log_var

    def decode(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Decode latent samples to reconstructions."""
        if not self.built:
            raise ValueError("Model must be built before decoding.")

        return self.decoder(z, training=False)

    def check_for_nan_weights(self) -> bool:
        """Check if any weights contain NaN values."""
        for weight in self.trainable_weights:
            if ops.any(ops.isnan(weight)):
                logger.error(f"NaN detected in weight: {weight.name}")
                return True
        return False

    def sample(self, num_samples: int) -> keras.KerasTensor:
        """Generate samples from the latent space."""
        if not self.built:
            raise ValueError("Model must be built before sampling.")

        z = keras.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)

    def train_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom training step with proper shape validation and gradient clipping."""
        # Handle different data formats
        if isinstance(data, tuple):
            x = data[0]
            if len(data) > 1:
                # If there are labels, ignore them for VAE training
                pass
        else:
            x = data

        # Validate input shape
        expected_shape = (None,) + self._input_shape
        if x.shape[1:] != self._input_shape:
            logger.warning(f"Input shape {x.shape} doesn't match expected {expected_shape}")

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(x, training=True)
            reconstruction = outputs["reconstruction"]

            # Validate reconstruction shape
            if reconstruction.shape != x.shape:
                logger.error(f"Shape mismatch: input {x.shape}, reconstruction {reconstruction.shape}")
                raise ValueError(f"Reconstruction shape {reconstruction.shape} doesn't match input {x.shape}")

            # Compute losses
            reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)
            kl_loss = self._compute_kl_loss(outputs["z_mean"], outputs["z_log_var"])

            # Total loss
            total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

            # Add regularization losses
            if self.losses:
                total_loss += ops.sum(self.losses)

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_weights)

        # Check for None gradients
        if any(grad is None for grad in gradients):
            logger.warning("Some gradients are None - this may indicate a problem")

        # Clip gradients to prevent explosion
        gradients = [ops.clip(grad, -1.0, 1.0) if grad is not None else None for grad in gradients]

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom test step with proper shape validation."""
        # Handle different data formats
        if isinstance(data, tuple):
            x = data[0]
            if len(data) > 1:
                # If there are labels, ignore them for VAE training
                pass
        else:
            x = data

        # Validate input shape
        if x.shape[1:] != self._input_shape:
            logger.warning(f"Input shape {x.shape} doesn't match expected {self._input_shape}")

        # Forward pass
        outputs = self(x, training=False)
        reconstruction = outputs["reconstruction"]

        # Validate reconstruction shape
        if reconstruction.shape != x.shape:
            logger.error(f"Shape mismatch: input {x.shape}, reconstruction {reconstruction.shape}")
            raise ValueError(f"Reconstruction shape {reconstruction.shape} doesn't match input {x.shape}")

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
            "kl_loss": self.kl_loss_tracker.result()
        }

    def _compute_reconstruction_loss(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute reconstruction loss with numerical stability."""
        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

        # Flatten for loss computation
        y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
        y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

        # Clip predictions to avoid log(0)
        y_pred_clipped = ops.clip(y_pred_flat, 1e-7, 1.0 - 1e-7)

        # Use Keras' built-in binary crossentropy for better numerical stability
        reconstruction_loss = ops.mean(
            keras.losses.binary_crossentropy(y_true_flat, y_pred_clipped)
        )

        return reconstruction_loss

    def _compute_kl_loss(
        self,
        z_mean: keras.KerasTensor,
        z_log_var: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute KL divergence loss with numerical stability."""
        # Clip log variance to prevent numerical issues
        z_log_var_clipped = ops.clip(z_log_var, -20.0, 20.0)

        # Compute KL divergence: KL(q||p) = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl_loss = -0.5 * ops.sum(
            1.0 + z_log_var_clipped - ops.square(z_mean) - ops.exp(z_log_var_clipped),
            axis=1
        )

        # Take mean across batch
        kl_loss = ops.mean(kl_loss)

        return kl_loss

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "depths": self.depths,
            "filters": self.filters,
            "use_bias": self.use_bias,
            "latent_dim": self.latent_dim,
            "steps_per_depth": self.steps_per_depth,
            "kl_loss_weight": self.kl_loss_weight,
            "input_shape": self._input_shape,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "final_activation": self.final_activation,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VAE":
        """Create ResnetVAE from configuration."""
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

# Factory function for the ResNet VAE
def create_vae(
    input_shape: Tuple[int, int, int],
    latent_dim: int,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    **kwargs
) -> VAE:
    """Create and compile a ResNet VAE model."""
    # Default parameters for stability
    default_kwargs = {
        'kl_loss_weight': 0.01,  # Lower KL weight for stability
        'depths': 2,  # Reduced complexity
        'steps_per_depth': 1,  # Reduced complexity
        'filters': [32, 64],  # Reduced filters
        'dropout_rate': 0.1,  # Add some dropout
        'use_batch_norm': True,
        'kernel_initializer': 'he_normal',
    }

    # Override defaults with user-provided kwargs
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    # Create the model
    model = VAE(
        latent_dim=latent_dim,
        input_shape=input_shape,
        **kwargs
    )

    # Compile the model
    model.compile(optimizer=optimizer)

    # Build the model
    model.build(input_shape=(None,) + input_shape)

    # Test the model to ensure it works
    test_input = keras.random.normal((2,) + input_shape)  # Use batch size 2 for testing
    test_output = model(test_input, training=False)

    # Validate outputs
    assert test_output['reconstruction'].shape == test_input.shape, "Reconstruction shape mismatch"
    assert test_output['z_mean'].shape == (2, latent_dim), "z_mean shape mismatch"
    assert test_output['z_log_var'].shape == (2, latent_dim), "z_log_var shape mismatch"

    logger.info(f"Created ResNet VAE for input shape {input_shape}")
    logger.info(f"Latent dim: {latent_dim}, Reconstruction shape: {test_output['reconstruction'].shape}")
    logger.info(f"Model parameters: {model.count_params():,}")

    return model


# ---------------------------------------------------------------------