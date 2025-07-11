"""
Hierarchical Variational Autoencoder (HVAE) Implementation
========================================================

This module implements a Hierarchical Variational Autoencoder (HVAE) using Keras 3.8.0,
designed for learning hierarchical latent representations with multiple levels of abstraction.
The implementation follows the hierarchical VAE principles with proper reparameterization trick,
multi-level KL divergence regularization, and hierarchical generation capabilities.

Key Features:
    - **Hierarchical Architecture**: Multiple levels of latent variables with dependencies
    - **Multi-scale Representation**: Captures both global and local features
    - **Configurable Depth**: Adjustable number of hierarchical levels
    - **Proper HVAE Loss**: Reconstruction loss + multi-level KL divergence regularization
    - **Numerical Stability**: Gradient clipping, loss clamping, and proper initialization
    - **Backend Agnostic**: Uses keras.ops for compatibility across TensorFlow, JAX, PyTorch
    - **Full Serialization**: Proper get_config/from_config implementation
    - **Custom Training**: Implements custom train_step and test_step methods
    - **Hierarchical Sampling**: Generate samples from learned hierarchical distributions

Architecture Overview:
    ```
    Input → Hidden → Level 1 Latents (z₁, z₂) → Level 2 Latents (z₃, z₄) → Decoder → Output
    ```

Mathematical Foundation:
    - **Level 1 Encoder**: q(z₁,z₂|x) - First level latent distributions
    - **Level 2 Encoder**: q(z₃|z₁), q(z₄|z₂) - Higher level latent distributions
    - **Decoder**: p(x|z₁,z₂,z₃,z₄) - Hierarchical reconstruction distribution
    - **Prior**: p(z₃)=N(0,I), p(z₄)=N(0,I) - Priors for top-level latents
    - **Loss**: L = E[log p(x|z)] - β₁*KL(q(z₁,z₂|x)||p(z₁,z₂)) - β₂*KL(q(z₃,z₄|z₁,z₂)||p(z₃,z₄))

Classes:
    HierarchicalVAE: Main Hierarchical Variational Autoencoder model

Functions:
    create_hvae: Factory function for creating and compiling HVAE models

Dependencies:
    - keras>=3.8.0
    - tensorflow>=2.18.0 (backend)
    - dl_techniques.utils.logger
    - dl_techniques.layers.sampling.Sampling
    - dl_techniques.layers.gaussian_pyramid.GaussianPyramid

Usage Example:
    ```python
    # Create a HVAE for 64x64 RGB images
    hvae = create_hvae(
        input_shape=(64, 64, 3),
        latent_dims=[32, 16],  # Two level 1 latent variables
        num_levels=2,
        kl_loss_weights=[0.01, 0.005],  # Different weights for each level
        use_pyramid_downsampling=True,
        optimizer='adam'
    )

    # Train the model
    hvae.fit(train_data, epochs=50, validation_data=val_data)

    # Generate hierarchical samples
    samples = hvae.sample(num_samples=10)

    # Encode images to hierarchical latent space
    latent_dict = hvae.encode(images)  # Returns dict with all latent levels

    # Decode hierarchical latent vectors to images
    reconstructions = hvae.decode(latent_dict)
    ```
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
from dl_techniques.layers.gaussian_pyramid import GaussianPyramid

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalVAE(keras.Model):
    """Hierarchical Variational Autoencoder with multiple levels of latent variables.

    This model implements a hierarchical VAE where latent variables are organized
    in multiple levels, with higher levels conditioning on lower levels. This
    architecture is particularly effective for learning structured representations
    where different levels capture different aspects of the data.

    Args:
        latent_dims: List of integers specifying the dimensionality of each level 1 latent variable.
            For example, [32, 16] creates two level 1 latent variables with dimensions 32 and 16.
        num_levels: Integer, number of hierarchical levels (minimum 2).
        kl_loss_weights: List of floats, KL loss weights for each level.
        input_shape: Tuple specifying the input shape (height, width, channels).
        hidden_dims: List of integers, dimensions for hidden layers.
        use_pyramid_downsampling: Boolean, whether to use Gaussian pyramid for downsampling.
        kernel_initializer: Weight initializer for convolutional layers.
        kernel_regularizer: Regularizer for convolutional layers.
        use_batch_norm: Boolean, whether to use batch normalization.
        use_bias: Boolean, whether to use bias in layers.
        dropout_rate: Float, dropout rate for regularization.
        activation: Activation function to use.
        name: Optional name for the model.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        latent_dims: List[int],
        num_levels: int = 2,
        kl_loss_weights: List[float] = None,
        input_shape: Optional[Tuple[int, int, int]] = None,
        hidden_dims: List[int] = None,
        use_pyramid_downsampling: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_batch_norm: bool = True,
        use_bias: bool = True,
        dropout_rate: float = 0.1,
        activation: Union[str, callable] = "leaky_relu",
        name: Optional[str] = "hierarchical_vae",
        **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if num_levels < 2:
            raise ValueError("num_levels must be at least 2 for hierarchical VAE")
        if len(latent_dims) < 2:
            raise ValueError("latent_dims must have at least 2 dimensions")
        if input_shape is None:
            raise ValueError("input_shape must be provided")
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (height, width, channels)")

        # Store configuration
        self.latent_dims = latent_dims
        self.num_levels = num_levels
        self.kl_loss_weights = kl_loss_weights or [1.0 / (i + 1) for i in range(num_levels)]
        self._input_shape = input_shape
        self.hidden_dims = hidden_dims or [64, 128, 256]
        self.use_pyramid_downsampling = use_pyramid_downsampling
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Ensure we have the right number of KL weights
        if len(self.kl_loss_weights) != num_levels:
            logger.warning(f"Adjusting KL loss weights to match {num_levels} levels")
            self.kl_loss_weights = [1.0 / (i + 1) for i in range(num_levels)]

        # Components to be built
        self.hidden_encoder = None
        self.level_encoders = []
        self.level_decoders = []
        self.sampling_layers = []
        self.pyramid_layer = None
        self.final_decoder = None

        # Shape tracking
        self._build_input_shape = None
        self._num_level1_latents = len(latent_dims)  # Number of level 1 latent variables
        self._num_level2_latents = len(latent_dims)  # Same number for level 2

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_trackers = [
            keras.metrics.Mean(name=f"kl_loss_level_{i}")
            for i in range(num_levels)
        ]

        logger.info(f"Initialized HVAE with {num_levels} levels, latent dims: {latent_dims}")

    def build(self, input_shape: Tuple) -> None:
        """Build the hierarchical VAE architecture."""
        self._build_input_shape = input_shape
        self._input_shape = tuple(input_shape[1:])

        # Build components
        self._build_hidden_encoder()
        self._build_level_encoders()
        self._build_sampling_layers()
        self._build_decoders()

        if self.use_pyramid_downsampling:
            self._build_pyramid_layer()

        super().build(input_shape)
        logger.info("Hierarchical VAE built successfully")

    def _build_hidden_encoder(self) -> None:
        """Build the initial hidden encoder."""
        x_input = keras.Input(shape=self._input_shape, name="hidden_encoder_input")
        x = x_input

        # Initial convolution
        x = keras.layers.Conv2D(
            filters=self.hidden_dims[0],
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="hidden_stem_conv"
        )(x)

        if self.use_batch_norm:
            x = keras.layers.BatchNormalization(center=self.use_bias)(x)
        x = keras.layers.Activation(self.activation)(x)

        # Progressive downsampling
        for i, hidden_dim in enumerate(self.hidden_dims[1:], 1):
            x = keras.layers.Conv2D(
                filters=hidden_dim,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"hidden_conv_{i}"
            )(x)

            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(center=self.use_bias)(x)
            x = keras.layers.Activation(self.activation)(x)

            if self.dropout_rate > 0:
                x = keras.layers.Dropout(self.dropout_rate)(x)

        # Global average pooling
        x = keras.layers.GlobalAveragePooling2D()(x)

        # Dense layers
        x = keras.layers.Dense(
            units=256,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="hidden_dense"
        )(x)

        if self.dropout_rate > 0:
            x = keras.layers.Dropout(self.dropout_rate)(x)

        self.hidden_encoder = keras.Model(
            inputs=x_input,
            outputs=x,
            name="hidden_encoder"
        )

    def _build_level_encoders(self) -> None:
        """Build encoders for each hierarchical level."""
        self.level_encoders = []

        # Level 1 encoders (from hidden features)
        for i in range(self._num_level1_latents):
            # Create encoder for this latent variable
            hidden_input = keras.Input(shape=(256,), name=f"level1_encoder_{i}_input")

            # Mean and log variance for this latent variable
            z_mean = keras.layers.Dense(
                units=self.latent_dims[i],
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer="zeros",
                kernel_regularizer=self.kernel_regularizer,
                name=f"level1_z{i}_mean"
            )(hidden_input)

            z_log_var = keras.layers.Dense(
                units=self.latent_dims[i],
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=keras.initializers.Constant(-2.0),
                kernel_regularizer=self.kernel_regularizer,
                name=f"level1_z{i}_log_var"
            )(hidden_input)

            encoder = keras.Model(
                inputs=hidden_input,
                outputs=[z_mean, z_log_var],
                name=f"level1_encoder_{i}"
            )
            self.level_encoders.append(encoder)

        # Level 2 encoders (from level 1 latents)
        for i in range(self._num_level2_latents):
            latent_input = keras.Input(
                shape=(self.latent_dims[i],),
                name=f"level2_encoder_{i}_input"
            )

            # Dense layer for processing
            x = keras.layers.Dense(
                units=64,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"level2_dense_{i}"
            )(latent_input)

            if self.dropout_rate > 0:
                x = keras.layers.Dropout(self.dropout_rate)(x)

            # Higher level latent dimensions (reduced)
            higher_dim = max(self.latent_dims[i] // 2, 4)

            z_mean = keras.layers.Dense(
                units=higher_dim,
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer="zeros",
                kernel_regularizer=self.kernel_regularizer,
                name=f"level2_z{i}_mean"
            )(x)

            z_log_var = keras.layers.Dense(
                units=higher_dim,
                use_bias=self.use_bias,
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=keras.initializers.Constant(-2.0),
                kernel_regularizer=self.kernel_regularizer,
                name=f"level2_z{i}_log_var"
            )(x)

            encoder = keras.Model(
                inputs=latent_input,
                outputs=[z_mean, z_log_var],
                name=f"level2_encoder_{i}"
            )
            self.level_encoders.append(encoder)

    def _build_sampling_layers(self) -> None:
        """Build sampling layers for each latent variable."""
        self.sampling_layers = []

        # Total number of latent variables across all levels
        total_latents = self._num_level1_latents + self._num_level2_latents

        for i in range(total_latents):
            sampling_layer = Sampling(seed=42 + i, name=f"sampling_layer_{i}")
            self.sampling_layers.append(sampling_layer)

    def _build_decoders(self) -> None:
        """Build hierarchical decoders."""
        # Calculate combined latent dimension
        level1_dims = sum(self.latent_dims)  # Sum of all level 1 latent dimensions
        level2_dims = sum(max(self.latent_dims[i] // 2, 4) for i in range(self._num_level2_latents))  # Sum of all level 2 latent dimensions
        combined_latent_dim = level1_dims + level2_dims

        # Combined latent input
        combined_input = keras.Input(shape=(combined_latent_dim,), name="decoder_input")

        # Calculate feature map dimensions
        feature_height = self._input_shape[0] // (2 ** (len(self.hidden_dims) - 1))
        feature_width = self._input_shape[1] // (2 ** (len(self.hidden_dims) - 1))
        feature_height = max(feature_height, 1)
        feature_width = max(feature_width, 1)

        # Project to feature map
        x = keras.layers.Dense(
            units=feature_height * feature_width * self.hidden_dims[-1],
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="decoder_projection"
        )(combined_input)

        x = keras.layers.Reshape(
            (feature_height, feature_width, self.hidden_dims[-1])
        )(x)

        # Progressive upsampling
        for i, hidden_dim in enumerate(reversed(self.hidden_dims[:-1])):
            x = keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

            x = keras.layers.Conv2D(
                filters=hidden_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"decoder_conv_{i}"
            )(x)

            if self.use_batch_norm:
                x = keras.layers.BatchNormalization(center=self.use_bias)(x)
            x = keras.layers.Activation(self.activation)(x)

            if self.dropout_rate > 0:
                x = keras.layers.Dropout(self.dropout_rate)(x)

        # Final output layer
        x = keras.layers.Conv2D(
            filters=self._input_shape[-1],
            kernel_size=3,
            strides=1,
            padding="same",
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.L1(1e-6),
            name="decoder_output"
        )(x)

        # Ensure exact shape matching
        if x.shape[1:] != self._input_shape:
            target_height, target_width = self._input_shape[:2]
            x = keras.layers.Resizing(
                height=target_height,
                width=target_width,
                interpolation="bilinear",
                name="decoder_resize"
            )(x)

        self.final_decoder = keras.Model(
            inputs=combined_input,
            outputs=x,
            name="final_decoder"
        )

    def _build_pyramid_layer(self) -> None:
        """Build Gaussian pyramid layer for multi-scale processing."""
        self.pyramid_layer = GaussianPyramid(
            levels=3,
            kernel_size=(5, 5),
            sigma=1.0,
            scale_factor=2,
            padding="same",
            name="gaussian_pyramid"
        )

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return metrics tracked by the model."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ] + self.kl_loss_trackers

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass through the hierarchical VAE."""
        # Hidden encoding
        hidden_features = self.hidden_encoder(inputs, training=training)

        # Level 1 encoding and sampling
        level1_latents = []
        level1_means = []
        level1_log_vars = []

        for i, encoder in enumerate(self.level_encoders[:self._num_level1_latents]):
            z_mean, z_log_var = encoder(hidden_features, training=training)
            z = self.sampling_layers[i]([z_mean, z_log_var], training=training)

            level1_latents.append(z)
            level1_means.append(z_mean)
            level1_log_vars.append(z_log_var)

        # Level 2 encoding and sampling
        level2_latents = []
        level2_means = []
        level2_log_vars = []

        for i, (encoder, z1) in enumerate(zip(
            self.level_encoders[self._num_level1_latents:],
            level1_latents
        )):
            z_mean, z_log_var = encoder(z1, training=training)
            z = self.sampling_layers[self._num_level1_latents + i](
                [z_mean, z_log_var], training=training
            )

            level2_latents.append(z)
            level2_means.append(z_mean)
            level2_log_vars.append(z_log_var)

        # Combine all latents for decoding
        all_latents = level1_latents + level2_latents
        combined_latents = ops.concatenate(all_latents, axis=1)

        # Decode
        reconstruction = self.final_decoder(combined_latents, training=training)

        return {
            "reconstruction": reconstruction,
            "level1_means": level1_means,
            "level1_log_vars": level1_log_vars,
            "level2_means": level2_means,
            "level2_log_vars": level2_log_vars,
            "level1_latents": level1_latents,
            "level2_latents": level2_latents,
            "combined_latents": combined_latents
        }

    def encode(self, inputs: keras.KerasTensor) -> Dict[str, List[keras.KerasTensor]]:
        """Encode inputs to hierarchical latent representations."""
        if not self.built:
            self.build((None,) + self._input_shape)

        outputs = self(inputs, training=False)
        return {
            "level1_means": outputs["level1_means"],
            "level1_log_vars": outputs["level1_log_vars"],
            "level2_means": outputs["level2_means"],
            "level2_log_vars": outputs["level2_log_vars"],
            "level1_latents": outputs["level1_latents"],
            "level2_latents": outputs["level2_latents"]
        }

    def decode(self, latent_dict: Dict[str, List[keras.KerasTensor]]) -> keras.KerasTensor:
        """Decode hierarchical latent representations to reconstructions."""
        if not self.built:
            raise ValueError("Model must be built before decoding.")

        # Combine latents
        all_latents = latent_dict["level1_latents"] + latent_dict["level2_latents"]
        combined_latents = ops.concatenate(all_latents, axis=1)

        return self.final_decoder(combined_latents, training=False)

    def sample(self, num_samples: int) -> keras.KerasTensor:
        """Generate samples from the hierarchical latent space."""
        if not self.built:
            raise ValueError("Model must be built before sampling.")

        # Sample from each level
        level1_samples = []
        level2_samples = []

        # Level 1 samples
        for i in range(self._num_level1_latents):
            z = keras.random.normal(shape=(num_samples, self.latent_dims[i]))
            level1_samples.append(z)

        # Level 2 samples
        for i in range(self._num_level2_latents):
            higher_dim = max(self.latent_dims[i] // 2, 4)
            z = keras.random.normal(shape=(num_samples, higher_dim))
            level2_samples.append(z)

        # Combine and decode
        all_samples = level1_samples + level2_samples
        combined_samples = ops.concatenate(all_samples, axis=1)

        return self.final_decoder(combined_samples, training=False)

    def train_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom training step for hierarchical VAE."""
        # Handle data
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Validate input shape
        if x.shape[1:] != self._input_shape:
            logger.warning(f"Input shape {x.shape} doesn't match expected {self._input_shape}")

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(x, training=True)
            reconstruction = outputs["reconstruction"]

            # Compute reconstruction loss
            reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)

            # Compute KL losses for each level
            kl_losses = []

            # Level 1 KL losses
            for i, (mean, log_var) in enumerate(zip(
                outputs["level1_means"], outputs["level1_log_vars"]
            )):
                kl_loss = self._compute_kl_loss(mean, log_var)
                kl_losses.append(kl_loss)

            # Level 2 KL losses
            for i, (mean, log_var) in enumerate(zip(
                outputs["level2_means"], outputs["level2_log_vars"]
            )):
                kl_loss = self._compute_kl_loss(mean, log_var)
                kl_losses.append(kl_loss)

            # Total loss
            total_loss = reconstruction_loss
            for i, kl_loss in enumerate(kl_losses):
                level_idx = 0 if i < len(outputs["level1_means"]) else 1
                total_loss += self.kl_loss_weights[level_idx] * kl_loss

            # Add regularization losses
            if self.losses:
                total_loss += ops.sum(self.losses)

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.trainable_weights)
        gradients = [ops.clip(grad, -1.0, 1.0) if grad is not None else None for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        for i, kl_loss in enumerate(kl_losses):
            level_idx = 0 if i < len(outputs["level1_means"]) else 1
            self.kl_loss_trackers[level_idx].update_state(kl_loss)

        # Return metrics
        result = {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
        for i, tracker in enumerate(self.kl_loss_trackers):
            result[f"kl_loss_level_{i}"] = tracker.result()

        return result

    def test_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom test step for hierarchical VAE."""
        # Handle data
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Forward pass
        outputs = self(x, training=False)
        reconstruction = outputs["reconstruction"]

        # Compute losses
        reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)

        # Compute KL losses for each level
        kl_losses = []

        # Level 1 KL losses
        for mean, log_var in zip(outputs["level1_means"], outputs["level1_log_vars"]):
            kl_loss = self._compute_kl_loss(mean, log_var)
            kl_losses.append(kl_loss)

        # Level 2 KL losses
        for mean, log_var in zip(outputs["level2_means"], outputs["level2_log_vars"]):
            kl_loss = self._compute_kl_loss(mean, log_var)
            kl_losses.append(kl_loss)

        # Total loss
        total_loss = reconstruction_loss
        for i, kl_loss in enumerate(kl_losses):
            level_idx = 0 if i < len(outputs["level1_means"]) else 1
            total_loss += self.kl_loss_weights[level_idx] * kl_loss

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        for i, kl_loss in enumerate(kl_losses):
            level_idx = 0 if i < len(outputs["level1_means"]) else 1
            self.kl_loss_trackers[level_idx].update_state(kl_loss)

        # Return metrics
        result = {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
        for i, tracker in enumerate(self.kl_loss_trackers):
            result[f"kl_loss_level_{i}"] = tracker.result()

        return result

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

        # Use binary crossentropy
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

        # Compute KL divergence
        kl_loss = -0.5 * ops.sum(
            1.0 + z_log_var_clipped - ops.square(z_mean) - ops.exp(z_log_var_clipped),
            axis=1
        )

        return ops.mean(kl_loss)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "latent_dims": self.latent_dims,
            "num_levels": self.num_levels,
            "kl_loss_weights": self.kl_loss_weights,
            "input_shape": self._input_shape,
            "hidden_dims": self.hidden_dims,
            "use_pyramid_downsampling": self.use_pyramid_downsampling,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HierarchicalVAE":
        """Create HVAE from configuration."""
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
# Factory function
# ---------------------------------------------------------------------

def create_hvae(
    input_shape: Tuple[int, int, int],
    latent_dims: List[int] = None,
    num_levels: int = 2,
    kl_loss_weights: List[float] = None,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    **kwargs
) -> HierarchicalVAE:
    """Create and compile a Hierarchical VAE model.

    Args:
        input_shape: Tuple specifying input shape (height, width, channels).
        latent_dims: List of latent dimensions for each level 1 latent variable.
        num_levels: Number of hierarchical levels.
        kl_loss_weights: KL loss weights for each level.
        optimizer: Optimizer for training.
        **kwargs: Additional arguments for HierarchicalVAE.

    Returns:
        Compiled HierarchicalVAE model.
    """
    # Default parameters
    if latent_dims is None:
        latent_dims = [32, 16]  # Two level 1 latent variables

    if kl_loss_weights is None:
        kl_loss_weights = [0.01, 0.005]  # Lower weight for higher levels

    # Default kwargs
    default_kwargs = {
        'hidden_dims': [64, 128, 256],
        'use_pyramid_downsampling': True,
        'dropout_rate': 0.1,
        'use_batch_norm': True,
        'kernel_initializer': 'he_normal',
    }

    # Override defaults with user-provided kwargs
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    # Create model
    model = HierarchicalVAE(
        latent_dims=latent_dims,
        num_levels=num_levels,
        kl_loss_weights=kl_loss_weights,
        input_shape=input_shape,
        **kwargs
    )

    # Compile model
    model.compile(optimizer=optimizer)

    # Build model
    model.build(input_shape=(None,) + input_shape)

    # Test model
    test_input = keras.random.normal((2,) + input_shape)
    test_output = model(test_input, training=False)

    # Validate outputs
    assert test_output['reconstruction'].shape == test_input.shape, "Reconstruction shape mismatch"
    assert len(test_output['level1_means']) == len(latent_dims), "Level 1 means count mismatch"
    assert len(test_output['level2_means']) == len(latent_dims), "Level 2 means count mismatch"

    logger.info(f"Created Hierarchical VAE for input shape {input_shape}")
    logger.info(f"Latent dims: {latent_dims}, Levels: {num_levels}")
    logger.info(f"Model parameters: {model.count_params():,}")

    return model