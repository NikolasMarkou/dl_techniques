"""
Hierarchical Variational Autoencoder (HVAE) Implementation
=========================================================

This module implements a Hierarchical Variational Autoencoder using Keras 3.8.0,
designed for multi-scale image processing with Laplacian pyramid decomposition.
The implementation processes images at multiple resolutions and learns hierarchical
representations that can be combined to reconstruct the original image.

Key Features:
    - **Multi-Scale Processing**: Processes images at multiple resolution levels
    - **Laplacian Pyramid**: Uses Laplacian decomposition for scale separation
    - **Hierarchical Reconstruction**: Combines reconstructions from multiple scales
    - **Gaussian Downsampling**: Non-learnable Gaussian filtering for downsampling
    - **Bilinear Upsampling**: Non-learnable bilinear interpolation for upsampling
    - **Configurable Levels**: Adjustable number of hierarchy levels
    - **ResNet-based VAE**: Uses existing VAE architecture for encoders/decoders
    - **Proper Serialization**: Full get_config/from_config implementation

Architecture Overview:
    ```
    Input Image (Level 0)
        ↓ (Gaussian downsample)
    Level 1 Image
        ↓ (Gaussian downsample)
    Level 2 Image
        ↓ (Gaussian downsample)
    ...
    Bottom Level (Level N-1)

    Laplacian[0] = Level 0 - Upsample(Level 1)
    Laplacian[1] = Level 1 - Upsample(Level 2)
    ...
    Laplacian[N-2] = Level N-2 - Upsample(Level N-1)
    Bottom = Level N-1 (no Laplacian)

    Each Laplacian[i] → VAE_Encoder[i] → z[i] → VAE_Decoder[i] → Decoder_Output[i]
    Bottom → VAE_Encoder[N-1] → z[N-1] → VAE_Decoder[N-1] → Decoder_Output[N-1]

    Deep Supervision with Hierarchical Reconstruction:
    Intermediate_Rec[N-1] = Decoder_Output[N-1]
    Intermediate_Rec[N-2] = Decoder_Output[N-2] + Upsample(Intermediate_Rec[N-1])
    ...
    Intermediate_Rec[0] = Decoder_Output[0] + Upsample(Intermediate_Rec[1])

    Loss[i] = ReconstructionLoss(Intermediate_Rec[i], Gaussian_Pyramid[i])
    Total_Loss = Sum(Loss[i]) + KL_Loss
    ```

Mathematical Foundation:
    - **Laplacian Pyramid**: L[i] = G[i] - Upsample(G[i+1])
    - **Gaussian Pyramid**: G[i+1] = Downsample(G[i])
    - **Hierarchical VAE**: Each level has its own latent space z[i] ~ N(μ[i], σ[i])
    - **Deep Supervision**: Loss[i] = ReconstructionLoss(Intermediate_Rec[i], G[i])
    - **Hierarchical Reconstruction**: Intermediate_Rec[i] = Decoder_Output[i] + Upsample(Intermediate_Rec[i+1])
    - **Total Loss**: Sum(Loss[i]) + β * Sum(KL(q(z[i]|L[i])||p(z[i])))

Classes:
    GaussianDownsample: Non-learnable Gaussian filtering and downsampling layer
    HVAE: Main Hierarchical Variational Autoencoder model

Functions:
    create_hvae: Factory function for creating and compiling HVAE models

Dependencies:
    - keras>=3.8.0
    - tensorflow>=2.18.0 (backend)
    - dl_techniques.utils.logger
    - dl_techniques.layers.sampling.Sampling
    - Existing VAE architecture (provided)

Usage Example:
    ```python
    # Create HVAE for 128x128 RGB images with 3 levels
    hvae = create_hvae(
        input_shape=(128, 128, 3),
        num_levels=3,
        latent_dims=[64, 32, 16],  # Different latent dims per level
        kl_loss_weight=0.01,
        optimizer='adam'
    )

    # Train the model
    hvae.fit(train_data, epochs=50, validation_data=val_data)

    # Generate samples
    samples = hvae.sample(num_samples=10)

    # Encode to hierarchical latent space
    z_means, z_log_vars = hvae.encode(images)

    # Decode from hierarchical latent space
    reconstructions = hvae.decode(z_samples)
    ```

Technical Details:
    - **Gaussian Downsampling**: 5x5 Gaussian kernel with σ=1.0
    - **Bilinear Upsampling**: 2x2 upsampling with bilinear interpolation
    - **Power of 2 Constraint**: Input dimensions must be divisible by 2^num_levels
    - **Laplacian Computation**: Difference between adjacent pyramid levels
    - **Deep Supervision**: Multi-level loss supervision at each pyramid level
    - **Hierarchical Loss**: Sum of reconstruction losses at all pyramid levels plus KL losses

Performance Considerations:
    - Memory efficient with separate VAE models per level
    - Supports different latent dimensions per level
    - Configurable VAE parameters per level
    - Proper gradient flow through hierarchical structure

Model Persistence:
    - Fully serializable using Keras .keras format
    - Proper config serialization for all hierarchical parameters
    - Custom object registration for loading saved models

Notes:
    - Input image dimensions must be powers of 2 for proper pyramid construction
    - Each level can have different VAE configurations
    - Gaussian downsampling is fixed (non-learnable)
    - Upsampling uses bilinear interpolation (non-learnable)
    - Bottom level processes the lowest resolution without Laplacian
"""

import keras
import numpy as np
from keras import ops
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any, List

from dl_techniques.utils.logger import logger
from dl_techniques.layers.sampling import Sampling
from .vae import VAE


@keras.saving.register_keras_serializable()
class GaussianDownsample(keras.layers.Layer):
    """
    Non-learnable Gaussian filtering and downsampling layer.

    This layer applies a fixed 5x5 Gaussian kernel followed by 2x2 subsampling.
    The Gaussian kernel is not trainable and provides smooth downsampling.

    Args:
        sigma: Standard deviation of the Gaussian kernel.
        kernel_size: Size of the Gaussian kernel (must be odd).
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        kernel_size: int = 5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kernel_size = kernel_size

        # Validate kernel size
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        # Create Gaussian kernel
        self.gaussian_kernel = None
        self._build_gaussian_kernel()

    def _build_gaussian_kernel(self) -> None:
        """Build the fixed Gaussian kernel."""
        # Create 1D Gaussian kernel
        kernel_1d = np.zeros(self.kernel_size)
        center = self.kernel_size // 2

        for i in range(self.kernel_size):
            x = i - center
            kernel_1d[i] = np.exp(-(x**2) / (2 * self.sigma**2))

        # Normalize
        kernel_1d = kernel_1d / np.sum(kernel_1d)

        # Create 2D kernel by outer product
        kernel_2d = np.outer(kernel_1d, kernel_1d)

        # Store as constant (non-trainable)
        self.gaussian_kernel = kernel_2d.astype(np.float32)

    def build(self, input_shape: Tuple) -> None:
        """Build the layer."""
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply Gaussian filtering and downsampling."""
        # Get input shape
        batch_size = ops.shape(inputs)[0]
        height = ops.shape(inputs)[1]
        width = ops.shape(inputs)[2]
        channels = ops.shape(inputs)[3]

        # Create depthwise convolution kernel
        # Shape: (kernel_size, kernel_size, channels, 1)
        kernel = ops.expand_dims(self.gaussian_kernel, axis=-1)
        kernel = ops.expand_dims(kernel, axis=-1)
        kernel = ops.repeat(kernel, channels, axis=2)

        # Apply depthwise convolution (Gaussian filtering)
        filtered = ops.nn.depthwise_conv(
            inputs,
            kernel,
            strides=1,
            padding="same"
        )

        # Subsample by 2 (take every other pixel)
        downsampled = filtered[:, ::2, ::2, :]

        return downsampled

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """Compute output shape after downsampling."""
        input_shape_list = list(input_shape)
        input_shape_list[1] = input_shape_list[1] // 2  # Height
        input_shape_list[2] = input_shape_list[2] // 2  # Width
        return tuple(input_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "sigma": self.sigma,
            "kernel_size": self.kernel_size,
        })
        return config


@keras.saving.register_keras_serializable()
class HVAE(keras.Model):
    """
    Hierarchical Variational Autoencoder (HVAE) model.

    This model processes images at multiple scales using Laplacian pyramid decomposition.
    Each level has its own VAE encoder and decoder, and the final reconstruction is
    formed by hierarchically combining the outputs.

    Args:
        num_levels: Number of hierarchy levels (pyramid levels).
        latent_dims: List of latent dimensions for each level.
        input_shape: Shape of input images (height, width, channels).
        kl_loss_weight: Weight for KL divergence loss.
        vae_config: Configuration for VAE models at each level.
        gaussian_sigma: Standard deviation for Gaussian downsampling.
        **kwargs: Additional keyword arguments for the Model base class.
    """

    def __init__(
        self,
        num_levels: int,
        latent_dims: List[int],
        input_shape: Tuple[int, int, int],
        kl_loss_weight: float = 0.01,
        vae_config: Optional[Dict[str, Any]] = None,
        gaussian_sigma: float = 1.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.num_levels = num_levels
        self.latent_dims = latent_dims
        self._input_shape = input_shape
        self.kl_loss_weight = kl_loss_weight
        self.vae_config = vae_config or {}
        self.gaussian_sigma = gaussian_sigma

        # Validation
        if len(latent_dims) != num_levels:
            raise ValueError(f"Number of latent dimensions ({len(latent_dims)}) must match number of levels ({num_levels})")

        # Check power of 2 constraint
        height, width, channels = input_shape
        if height % (2 ** (num_levels - 1)) != 0 or width % (2 ** (num_levels - 1)) != 0:
            raise ValueError(f"Input dimensions must be divisible by 2^{num_levels-1} for {num_levels} levels")

        # Components to be built
        self.gaussian_downsample = GaussianDownsample(sigma=gaussian_sigma)
        self.vaes = []  # List of VAE models for each level

        # Build input shape for each level
        self.level_shapes = []
        current_shape = input_shape
        for i in range(num_levels):
            self.level_shapes.append(current_shape)
            if i < num_levels - 1:  # Don't downsample the last level
                current_shape = (current_shape[0] // 2, current_shape[1] // 2, current_shape[2])

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        logger.info(f"Initialized HVAE with {num_levels} levels")

    def build(self, input_shape: Tuple) -> None:
        """Build the HVAE architecture."""
        # Create VAE models for each level
        for i in range(self.num_levels):
            level_shape = self.level_shapes[i]
            latent_dim = self.latent_dims[i]

            # Create VAE configuration for this level
            vae_config = self.vae_config.copy()
            vae_config.update({
                'input_shape': level_shape,
                'latent_dim': latent_dim,
                'name': f'vae_level_{i}'
            })

            # Create VAE model
            vae = VAE(**vae_config)
            vae.build(input_shape=(None,) + level_shape)
            self.vaes.append(vae)

        super().build(input_shape)
        logger.info("HVAE built successfully")

    def _create_gaussian_pyramid(self, image: keras.KerasTensor) -> List[keras.KerasTensor]:
        """Create Gaussian pyramid by iterative downsampling."""
        pyramid = [image]
        current = image

        for i in range(self.num_levels - 1):
            # Downsample using Gaussian filter
            current = self.gaussian_downsample(current)
            pyramid.append(current)

        return pyramid

    def _create_laplacian_pyramid(self, gaussian_pyramid: List[keras.KerasTensor]) -> List[keras.KerasTensor]:
        """Create Laplacian pyramid from Gaussian pyramid."""
        laplacian_pyramid = []

        for i in range(self.num_levels - 1):
            # Current level
            current = gaussian_pyramid[i]

            # Next level (smaller)
            next_level = gaussian_pyramid[i + 1]

            # Upsample next level to current size
            target_height = ops.shape(current)[1]
            target_width = ops.shape(current)[2]
            upsampled = ops.image.resize(
                next_level,
                (target_height, target_width),
                interpolation="bilinear"
            )

            # Compute Laplacian (difference)
            laplacian = current - upsampled
            laplacian_pyramid.append(laplacian)

        # Add the bottom level (no Laplacian)
        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    def _encode_levels(self, laplacian_pyramid: List[keras.KerasTensor], training: Optional[bool] = None) -> Tuple[List[keras.KerasTensor], List[keras.KerasTensor]]:
        """Encode each level of the Laplacian pyramid."""
        z_means = []
        z_log_vars = []

        for i, level_input in enumerate(laplacian_pyramid):
            # Encode with level-specific VAE
            outputs = self.vaes[i](level_input, training=training)
            z_means.append(outputs['z_mean'])
            z_log_vars.append(outputs['z_log_var'])

        return z_means, z_log_vars

    def _decode_levels(self, z_samples: List[keras.KerasTensor], training: Optional[bool] = None) -> List[keras.KerasTensor]:
        """Decode each level latent sample."""
        reconstructions = []

        for i, z in enumerate(z_samples):
            # Decode with level-specific VAE
            reconstruction = self.vaes[i].decode(z)
            reconstructions.append(reconstruction)

        return reconstructions

    def _reconstruct_hierarchically(self, level_reconstructions: List[keras.KerasTensor]) -> keras.KerasTensor:
        """Reconstruct final image by combining level reconstructions."""
        # Start from the bottom level (finest scale)
        reconstruction = level_reconstructions[-1]

        # Work backwards through levels
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample current reconstruction
            target_height = ops.shape(level_reconstructions[i])[1]
            target_width = ops.shape(level_reconstructions[i])[2]
            upsampled = ops.image.resize(
                reconstruction,
                (target_height, target_width),
                interpolation="bilinear"
            )

            # Add current level reconstruction
            reconstruction = level_reconstructions[i] + upsampled

        return reconstruction

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
    ) -> Dict[str, Union[keras.KerasTensor, List[keras.KerasTensor]]]:
        """Forward pass through the HVAE."""
        # Create pyramids
        gaussian_pyramid = self._create_gaussian_pyramid(inputs)
        laplacian_pyramid = self._create_laplacian_pyramid(gaussian_pyramid)

        # Encode each level
        z_means, z_log_vars = self._encode_levels(laplacian_pyramid, training=training)

        # Sample from latent distributions
        z_samples = []
        for i in range(self.num_levels):
            sampling_layer = Sampling(seed=42 + i)
            z = sampling_layer([z_means[i], z_log_vars[i]], training=training)
            z_samples.append(z)

        # Decode each level
        level_reconstructions = self._decode_levels(z_samples, training=training)

        # Hierarchical reconstruction
        final_reconstruction = self._reconstruct_hierarchically(level_reconstructions)

        return {
            'z_means': z_means,
            'z_log_vars': z_log_vars,
            'z_samples': z_samples,
            'level_reconstructions': level_reconstructions,
            'reconstruction': final_reconstruction,
            'gaussian_pyramid': gaussian_pyramid,
            'laplacian_pyramid': laplacian_pyramid
        }

    def encode(self, inputs: keras.KerasTensor) -> Tuple[List[keras.KerasTensor], List[keras.KerasTensor]]:
        """Encode inputs to hierarchical latent parameters."""
        if not self.built:
            self.build((None,) + self._input_shape)

        # Create pyramids
        gaussian_pyramid = self._create_gaussian_pyramid(inputs)
        laplacian_pyramid = self._create_laplacian_pyramid(gaussian_pyramid)

        # Encode each level
        z_means, z_log_vars = self._encode_levels(laplacian_pyramid, training=False)

        return z_means, z_log_vars

    def decode(self, z_samples: List[keras.KerasTensor]) -> keras.KerasTensor:
        """Decode hierarchical latent samples to reconstruction."""
        if not self.built:
            raise ValueError("Model must be built before decoding.")

        # Decode each level
        level_reconstructions = self._decode_levels(z_samples, training=False)

        # Hierarchical reconstruction
        final_reconstruction = self._reconstruct_hierarchically(level_reconstructions)

        return final_reconstruction

    def sample(self, num_samples: int) -> keras.KerasTensor:
        """Generate samples from the hierarchical latent space."""
        if not self.built:
            raise ValueError("Model must be built before sampling.")

        # Sample from each level's latent space
        z_samples = []
        for i in range(self.num_levels):
            z = keras.random.normal(shape=(num_samples, self.latent_dims[i]))
            z_samples.append(z)

        return self.decode(z_samples)

    def train_step(self, data) -> Dict[str, keras.KerasTensor]:
        """Custom training step for hierarchical VAE."""
        # Handle different data formats
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(x, training=True)
            reconstruction = outputs['reconstruction']
            z_means = outputs['z_means']
            z_log_vars = outputs['z_log_vars']

            # Compute reconstruction loss
            reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)

            # Compute KL loss for each level
            kl_loss = 0.0
            for i in range(self.num_levels):
                level_kl = self._compute_kl_loss(z_means[i], z_log_vars[i])
                kl_loss += level_kl

            # Total loss
            total_loss = reconstruction_loss + self.kl_loss_weight * kl_loss

            # Add regularization losses
            if self.losses:
                total_loss += ops.sum(self.losses)

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_weights)

        # Clip gradients
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
        """Custom test step for hierarchical VAE."""
        # Handle different data formats
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Forward pass
        outputs = self(x, training=False)
        reconstruction = outputs['reconstruction']
        z_means = outputs['z_means']
        z_log_vars = outputs['z_log_vars']

        # Compute reconstruction loss
        reconstruction_loss = self._compute_reconstruction_loss(x, reconstruction)

        # Compute KL loss for each level
        kl_loss = 0.0
        for i in range(self.num_levels):
            level_kl = self._compute_kl_loss(z_means[i], z_log_vars[i])
            kl_loss += level_kl

        # Total loss
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
        # Flatten for loss computation
        y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
        y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

        # Clip predictions to avoid log(0)
        y_pred_clipped = ops.clip(y_pred_flat, 1e-7, 1.0 - 1e-7)

        # Binary crossentropy loss
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

        # KL divergence: KL(q||p) = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
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
            "num_levels": self.num_levels,
            "latent_dims": self.latent_dims,
            "input_shape": self._input_shape,
            "kl_loss_weight": self.kl_loss_weight,
            "vae_config": self.vae_config,
            "gaussian_sigma": self.gaussian_sigma,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HVAE":
        """Create HVAE from configuration."""
        return cls(**config)


def create_hvae(
    input_shape: Tuple[int, int, int],
    num_levels: int,
    latent_dims: List[int],
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    kl_loss_weight: float = 0.01,
    vae_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> HVAE:
    """
    Create and compile a Hierarchical VAE model.

    Args:
        input_shape: Shape of input images (height, width, channels).
        num_levels: Number of hierarchy levels.
        latent_dims: List of latent dimensions for each level.
        optimizer: Optimizer for training.
        kl_loss_weight: Weight for KL divergence loss.
        vae_config: Configuration for VAE models at each level.
        **kwargs: Additional arguments for HVAE.

    Returns:
        Compiled HVAE model.
    """
    # Default VAE configuration
    default_vae_config = {
        'depths': 2,
        'steps_per_depth': 1,
        'use_batch_norm': True,
        'dropout_rate': 0.1,
        'kernel_initializer': 'he_normal',
    }

    # Override defaults with user config
    if vae_config is not None:
        default_vae_config.update(vae_config)

    # Create the model
    model = HVAE(
        num_levels=num_levels,
        latent_dims=latent_dims,
        input_shape=input_shape,
        kl_loss_weight=kl_loss_weight,
        vae_config=default_vae_config,
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
    assert len(test_output['z_means']) == num_levels, "Number of z_means mismatch"
    assert len(test_output['z_log_vars']) == num_levels, "Number of z_log_vars mismatch"

    logger.info(f"Created HVAE for input shape {input_shape}")
    logger.info(f"Levels: {num_levels}, Latent dims: {latent_dims}")
    logger.info(f"Reconstruction shape: {test_output['reconstruction'].shape}")
    logger.info(f"Model parameters: {model.count_params():,}")

    return model