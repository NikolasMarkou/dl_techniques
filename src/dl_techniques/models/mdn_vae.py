"""
MDN-VAE: Variational Autoencoder with Mixture Density Network
=============================================================

This module implements a ResNet-based Variational Autoencoder (VAE) that uses
a Mixture Density Network (MDN) for the latent distribution, allowing for
multi-modal latent representations.

Key Differences from Standard VAE:
    - **Multi-modal Latent Space**: Uses mixture of Gaussians instead of single Gaussian
    - **Flexible Posterior**: Can capture complex, multi-modal posterior distributions
    - **Enhanced Expressiveness**: Better suited for datasets with distinct clusters
    - **Adaptive Complexity**: Mixture components can specialize for different data modes

Architecture:
    ```
    Input → Encoder → MDN Layer → {μ₁...μₖ, σ₁...σₖ, π₁...πₖ}
                                           ↓
    Output ← Decoder ← z ~ Σᵢ πᵢ N(μᵢ, σᵢ²)
    ```

Mathematical Foundation:
    - **Posterior**: q(z|x) = Σᵢ πᵢ(x) N(z | μᵢ(x), σᵢ²(x))
    - **Prior**: p(z) = N(0, I)
    - **KL Divergence**: Approximated using Monte Carlo sampling
    - **Loss**: L = E[log p(x|z)] - β * KL(q(z|x)||p(z))
"""

import keras
import numpy as np
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.mdn_layer import MDNLayer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SamplingMDN(keras.layers.Layer):
    """Sampling layer for MDN-based latent distributions.

    This layer samples from a mixture of Gaussians distribution defined by
    the MDN parameters.
    """

    def __init__(self,
                 num_mixtures: int,
                 latent_dim: int,
                 seed: Optional[int] = None,
                 **kwargs):
        self.num_mixtures = num_mixtures
        self.latent_dim = latent_dim
        super().__init__(**kwargs)
        self.seed = seed

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        batch_size = input_shape[0]
        return batch_size, self.latent_dim

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Sample from the mixture distribution.

        Args:
            inputs: mdn_params
            training: Whether in training mode

        Returns:
            Sampled latent vectors
        """
        mdn_params = inputs

        # Extract mixture parameters
        batch_size = ops.shape(mdn_params)[0]

        # Calculate split points
        mu_end = self.num_mixtures * self.latent_dim
        sigma_end = mu_end + (self.num_mixtures * self.latent_dim)

        # Split parameters
        out_mu = mdn_params[..., :mu_end]
        out_sigma = mdn_params[..., mu_end:sigma_end]
        out_pi = mdn_params[..., sigma_end:]

        # Reshape means and sigmas
        out_mu = ops.reshape(out_mu, [batch_size, self.num_mixtures, self.latent_dim])
        out_sigma = ops.reshape(out_sigma, [batch_size, self.num_mixtures, self.latent_dim])

        # Ensure numerical stability
        out_sigma = ops.maximum(out_sigma, 1e-6)

        # Convert logits to probabilities
        mix_weights = keras.activations.softmax(out_pi, axis=-1)

        if training:
            # During training, use reparameterization trick with Gumbel-Softmax
            # for differentiable component selection

            # Sample component selection using Gumbel-Max trick
            gumbel_noise = -ops.log(-ops.log(
                keras.random.uniform(ops.shape(out_pi), seed=self.seed) + 1e-10
            ))
            log_mix_weights = ops.log(mix_weights + 1e-10)
            selected_logits = log_mix_weights + gumbel_noise

            # Soft selection using softmax (differentiable)
            selection_weights = keras.activations.softmax(selected_logits / 0.1, axis=-1)
            selection_weights = ops.expand_dims(selection_weights, -1)

            # Weighted combination of all components (soft selection)
            selected_mu = ops.sum(out_mu * selection_weights, axis=1)
            selected_sigma = ops.sum(out_sigma * selection_weights, axis=1)
        else:
            # During inference, use hard selection
            selected_components = ops.argmax(mix_weights, axis=-1)
            one_hot = ops.one_hot(selected_components, num_classes=self.num_mixtures)
            one_hot_expanded = ops.expand_dims(one_hot, -1)

            selected_mu = ops.sum(out_mu * one_hot_expanded, axis=1)
            selected_sigma = ops.sum(out_sigma * one_hot_expanded, axis=1)

        # Sample from selected Gaussian
        epsilon = keras.random.normal(
            shape=(batch_size, self.latent_dim),
            seed=self.seed
        )
        z = selected_mu + selected_sigma * epsilon

        return z

    def get_config(self):
        config = super().get_config()
        config.update({"seed": self.seed})
        return config


@keras.saving.register_keras_serializable()
class MDN_VAE(keras.Model):
    """Variational Autoencoder with Mixture Density Network latent distribution."""

    def __init__(
            self,
            latent_dim: int,
            num_mixtures: int = 5,
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
            name: Optional[str] = "mdn_vae",
            **kwargs: Any
    ) -> None:
        """Initialize MDN-VAE.

        Args:
            latent_dim: Dimension of latent space
            num_mixtures: Number of Gaussian mixtures in latent distribution
            depths: Number of depth levels in ResNet architecture
            steps_per_depth: Number of residual blocks per depth
            filters: List of filter counts for each depth
            kl_loss_weight: Weight for KL divergence term
            input_shape: Input image shape (H, W, C)
            kernel_initializer: Weight initializer
            kernel_regularizer: Weight regularizer
            use_batch_norm: Whether to use batch normalization
            use_bias: Whether to use bias terms
            dropout_rate: Dropout rate
            activation: Activation function
            name: Model name
        """
        super().__init__(name=name, **kwargs)

        # Store configuration
        self.latent_dim = latent_dim
        self.num_mixtures = num_mixtures
        self.depths = depths
        self.steps_per_depth = steps_per_depth
        self.filters = filters or [32, 64, 128]
        self.kl_loss_weight = kl_loss_weight
        self._input_shape = input_shape
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias

        # Components to be built
        self.encoder = None
        self.decoder = None
        self.mdn_layer = None
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

        logger.info(f"Initialized MDN-VAE with latent_dim={latent_dim}, num_mixtures={num_mixtures}")

    def build(self, input_shape: Tuple) -> None:
        """Build the MDN-VAE architecture."""
        self._build_input_shape = input_shape
        self._input_shape = tuple(input_shape[1:])

        # Build encoder with MDN output
        self._build_encoder(self._input_shape)

        # Build MDN layer
        self.mdn_layer = MDNLayer(
            output_dimension=self.latent_dim,
            num_mixtures=self.num_mixtures,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="latent_mdn"
        )

        # Build sampling layer
        self.sampling_layer = SamplingMDN(
            seed=42,
            latent_dim=self.latent_dim,
            num_mixtures=self.num_mixtures,
            name="mdn_sampling"
        )

        # Build decoder (same as original VAE)
        self._build_decoder()

        super().build(input_shape)
        logger.info("MDN-VAE built successfully")

    def _build_encoder(self, input_shape: Tuple[int, int, int]) -> None:
        """Build the encoder network that outputs features for MDN layer."""
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
            # Downsampling
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

            # Residual blocks
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
                x = keras.layers.Add()([x, residual])

        # Flatten for MDN input
        x = keras.layers.Flatten()(x)

        # Additional dense layer before MDN
        x = keras.layers.Dense(
            units=256,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="encoder_projection"
        )(x)

        self.encoder = keras.Model(
            inputs=x_input,
            outputs=x,
            name="encoder"
        )

    def _build_decoder(self) -> None:
        """Build the decoder network (same as original VAE)."""
        # Calculate feature map size after downsampling
        feature_height = self._input_shape[0] // (2 ** self.depths)
        feature_width = self._input_shape[1] // (2 ** self.depths)
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
                interpolation="nearest"
            )(x)

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

            # Residual blocks
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
                x = keras.layers.Add()([x, residual])

        # Final output layer
        x = keras.layers.Conv2D(
            filters=self._input_shape[-1],
            kernel_size=1,
            strides=1,
            padding="same",
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_regularizer=keras.regularizers.L1(1e-6),
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer='zeros',
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
        """Forward pass through the MDN-VAE."""
        # Encoder pass
        features = self.encoder(inputs, training=training)

        # MDN layer to get mixture parameters
        mdn_params = self.mdn_layer(features, training=training)

        # Sample from mixture distribution
        z = self.sampling_layer(
            mdn_params,
            training=training
        )

        # Decoder pass
        reconstruction = self.decoder(z, training=training)

        return {
            "mdn_params": mdn_params,
            "z": z,
            "reconstruction": reconstruction
        }

    def encode(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Encode inputs to MDN parameters."""
        if not self.built:
            self.build((None,) + self._input_shape)

        features = self.encoder(inputs, training=False)
        mdn_params = self.mdn_layer(features, training=False)
        return mdn_params

    def decode(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Decode latent samples to reconstructions."""
        if not self.built:
            raise ValueError("Model must be built before decoding.")

        return self.decoder(z, training=False)

    def sample(self, num_samples: int) -> keras.KerasTensor:
        """Generate samples from the prior (standard normal)."""
        if not self.built:
            raise ValueError("Model must be built before sampling.")

        z = keras.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)

    def train_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom training step."""
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(x, training=True)
            reconstruction = outputs["reconstruction"]
            mdn_params = outputs["mdn_params"]
            z = outputs["z"]

            # Compute losses
            reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)
            kl_loss = self._compute_mdn_kl_loss(mdn_params, z)

            # Total loss
            total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

            # Add regularization losses
            if self.losses:
                total_loss += ops.sum(self.losses)

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.trainable_weights)
        gradients = [ops.clip(grad, -1.0, 1.0) if grad is not None else None
                     for grad in gradients]
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
        """Custom test step."""
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Forward pass
        outputs = self(x, training=False)
        reconstruction = outputs["reconstruction"]
        mdn_params = outputs["mdn_params"]
        z = outputs["z"]

        # Compute losses
        reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)
        kl_loss = self._compute_mdn_kl_loss(mdn_params, z)
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
        """Compute reconstruction loss."""
        y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
        y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

        y_pred_clipped = ops.clip(y_pred_flat, 1e-7, 1.0 - 1e-7)

        reconstruction_loss = ops.mean(
            keras.losses.binary_crossentropy(y_true_flat, y_pred_clipped)
        )

        return reconstruction_loss

    def _compute_mdn_kl_loss(
            self,
            mdn_params: keras.KerasTensor,
            z_samples: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute KL divergence for mixture distribution using Monte Carlo approximation.

        KL(q(z|x)||p(z)) ≈ E_q[log q(z|x) - log p(z)]
        where q(z|x) is the mixture distribution and p(z) is N(0,I)
        """
        batch_size = ops.shape(mdn_params)[0]

        # Extract mixture parameters
        mu, sigma, pi_logits = self.mdn_layer.split_mixture_params(mdn_params)

        # Convert to log probabilities for numerical stability
        log_pi = ops.log(keras.activations.softmax(pi_logits, axis=-1) + 1e-10)

        # Expand z_samples for broadcasting with mixture components
        z_expanded = ops.expand_dims(z_samples, 1)  # [batch, 1, latent_dim]

        # Compute log probability of z under each mixture component
        # log N(z | μᵢ, σᵢ²) = -0.5 * [log(2π) + log(σᵢ²) + (z-μᵢ)²/σᵢ²]
        log_2pi = ops.log(2.0 * np.pi)
        log_sigma_sq = 2.0 * ops.log(sigma + 1e-10)

        diff_sq = ops.square(z_expanded - mu) / (ops.square(sigma) + 1e-10)
        log_probs_components = -0.5 * (log_2pi + log_sigma_sq + diff_sq)

        # Sum over dimensions
        log_probs_components = ops.sum(log_probs_components, axis=-1)  # [batch, num_mix]

        # Add log mixture weights
        log_probs_weighted = log_pi + log_probs_components  # [batch, num_mix]

        # Log-sum-exp for numerical stability
        max_log_prob = ops.max(log_probs_weighted, axis=-1, keepdims=True)
        log_q_z = max_log_prob + ops.log(
            ops.sum(ops.exp(log_probs_weighted - max_log_prob), axis=-1, keepdims=True)
        )
        log_q_z = ops.squeeze(log_q_z, axis=-1)

        # Log probability under prior p(z) = N(0, I)
        log_p_z = -0.5 * (self.latent_dim * log_2pi + ops.sum(ops.square(z_samples), axis=-1))

        # KL divergence: E_q[log q(z|x) - log p(z)]
        kl_loss = ops.mean(log_q_z - log_p_z)

        return kl_loss

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "num_mixtures": self.num_mixtures,
            "depths": self.depths,
            "filters": self.filters,
            "use_bias": self.use_bias,
            "steps_per_depth": self.steps_per_depth,
            "kl_loss_weight": self.kl_loss_weight,
            "input_shape": self._input_shape,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MDN_VAE":
        """Create MDN-VAE from configuration."""
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
# Factory function for the MDN-VAE
# ---------------------------------------------------------------------

def create_mdn_vae(
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        num_mixtures: int = 5,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        **kwargs
) -> MDN_VAE:
    """Create and compile an MDN-VAE model.

    Args:
        input_shape: Input image shape (H, W, C)
        latent_dim: Dimension of latent space
        num_mixtures: Number of Gaussian mixtures
        optimizer: Optimizer to use
        **kwargs: Additional arguments for MDN_VAE

    Returns:
        Compiled MDN-VAE model
    """
    # Default parameters for stability
    default_kwargs = {
        'kl_loss_weight': 0.001,  # Lower KL weight for mixture distribution
        'depths': 2,
        'steps_per_depth': 1,
        'filters': [32, 64],
        'dropout_rate': 0.1,
        'use_batch_norm': True,
        'kernel_initializer': 'he_normal',
    }

    # Override defaults with user-provided kwargs
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    # Create the model
    model = MDN_VAE(
        latent_dim=latent_dim,
        num_mixtures=num_mixtures,
        input_shape=input_shape,
        **kwargs
    )

    # Compile the model
    model.compile(optimizer=optimizer)

    # Build the model
    model.build(input_shape=(None,) + input_shape)

    # Test the model
    test_input = keras.random.normal((2,) + input_shape)
    test_output = model(test_input, training=False)

    # Validate outputs
    assert test_output['reconstruction'].shape == test_input.shape, "Reconstruction shape mismatch"
    assert test_output['z'].shape == (2, latent_dim), "Latent shape mismatch"

    # Calculate expected MDN param size
    expected_mdn_size = (2 * latent_dim * num_mixtures) + num_mixtures
    assert test_output['mdn_params'].shape[-1] == expected_mdn_size, "MDN params shape mismatch"

    logger.info(f"Created MDN-VAE for input shape {input_shape}")
    logger.info(f"Latent dim: {latent_dim}, Num mixtures: {num_mixtures}")
    logger.info(f"Model parameters: {model.count_params():,}")

    return model


# ---------------------------------------------------------------------
