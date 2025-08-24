"""
Sampling Layer for Variational Autoencoders (VAEs).

This module defines the `Sampling` layer, a custom Keras layer that implements
the reparameterization trick. This technique is a cornerstone of Variational
Autoencoders, enabling the model to be trained via backpropagation despite
involving a random sampling step.

The layer takes the mean (mu) and the log-variance (log_var) of a latent
Gaussian distribution (as predicted by a VAE's encoder) and generates a
sample `z` from this distribution. The reparameterization trick expresses
the sample `z` as a deterministic function of `mu`, `log_var`, and an
auxiliary random noise variable `epsilon` (sampled from a standard normal
distribution), according to the formula:

    z = mu + exp(0.5 * log_var) * epsilon

By reformulating the sampling process this way, the gradients can flow
backwards from the decoder's loss, through the sample `z`, and to the
encoder's parameters (`mu` and `log_var`), allowing for end-to-end training.
"""

import keras
from keras import ops
from typing import Tuple, Any, Dict, Optional, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Sampling(keras.layers.Layer):
    """
    Uses reparameterization trick to sample from a Normal distribution.

    This layer implements the reparameterization trick commonly used in Variational
    Autoencoders (VAEs). It takes as input the mean and log variance of a latent
    distribution and returns a sample from that distribution using the reparameterization
    trick: z = mean + std * epsilon, where epsilon is sampled from a standard normal
    distribution.

    The reparameterization trick allows gradients to flow through the sampling operation
    by expressing the random variable as a deterministic function of the parameters and
    an auxiliary noise variable. This is essential for training VAEs end-to-end via
    gradient descent.

    Mathematical formulation:
        z = μ + σ * ε
        where σ = exp(0.5 * log_var) and ε ~ N(0, I)

    Args:
        seed: Optional integer seed for random number generation. If None, uses
            random seed for each call. Providing a seed ensures reproducible sampling.
            Defaults to None.
        **kwargs: Additional keyword arguments to pass to the Layer base class.

    Input shape:
        A list/tuple of two tensors:
        - z_mean: Tensor of shape `(batch_size, latent_dim)` representing the mean
        - z_log_var: Tensor of shape `(batch_size, latent_dim)` representing the
          log variance

    Output shape:
        Tensor of shape `(batch_size, latent_dim)` representing samples from the
        latent distribution.

    Returns:
        A tensor containing samples from the Normal distribution parameterized by
        the input mean and log variance.

    Example:
        ```python
        # Basic usage in VAE encoder
        encoded = encoder_layers(inputs)
        z_mean = keras.layers.Dense(latent_dim, name="z_mean")(encoded)
        z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(encoded)

        # Sample from the latent distribution
        z = Sampling()([z_mean, z_log_var])

        # With deterministic seed for reproducibility
        z_reproducible = Sampling(seed=42)([z_mean, z_log_var])

        # In a complete VAE model
        inputs = keras.Input(shape=(784,))
        encoded = keras.layers.Dense(256, activation='relu')(inputs)
        z_mean = keras.layers.Dense(2)(encoded)
        z_log_var = keras.layers.Dense(2)(encoded)
        z = Sampling()([z_mean, z_log_var])
        decoded = keras.layers.Dense(784, activation='sigmoid')(z)
        vae = keras.Model(inputs, decoded)
        ```

    Note:
        This layer does not create any trainable parameters. It only performs
        the sampling operation using the input mean and log variance tensors.
        The layer is stateless except for the optional random seed.

    Raises:
        ValueError: If inputs don't contain exactly 2 tensors or if tensor
            shapes are incompatible.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Sampling layer.

        Args:
            seed: Optional integer seed for random number generation. If provided,
                ensures reproducible sampling across calls.
            **kwargs: Additional keyword arguments for the Layer parent class.
        """
        super().__init__(**kwargs)

        # Validate seed parameter
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be an integer or None, got {type(seed)}")

        # Store configuration
        self.seed = seed

        logger.debug(f"Initialized Sampling layer with seed={seed}")

    def build(self, input_shape: Union[Tuple[Tuple, ...], List[Tuple]]) -> None:
        """
        Build the layer and validate input shapes.

        Args:
            input_shape: Tuple or list of two shape tuples for z_mean and z_log_var tensors.

        Raises:
            ValueError: If input_shape doesn't contain exactly 2 tensors or if the
                tensor shapes are incompatible.
        """
        # Validate input structure
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                f"Sampling layer expects exactly 2 inputs (z_mean, z_log_var), "
                f"but received {len(input_shape) if isinstance(input_shape, (tuple, list)) else 'invalid'} inputs."
            )

        z_mean_shape, z_log_var_shape = input_shape

        # Validate tensor dimensions
        if len(z_mean_shape) != len(z_log_var_shape):
            raise ValueError(
                f"z_mean and z_log_var must have the same number of dimensions. "
                f"Got z_mean: {len(z_mean_shape)}, z_log_var: {len(z_log_var_shape)}"
            )

        # Validate shapes are compatible (excluding batch dimension)
        if z_mean_shape[1:] != z_log_var_shape[1:]:
            raise ValueError(
                f"z_mean and z_log_var must have the same shape except for batch dimension. "
                f"Got z_mean: {z_mean_shape}, z_log_var: {z_log_var_shape}"
            )

        # Validate minimum dimensions
        if len(z_mean_shape) < 2:
            raise ValueError(
                f"Input tensors must be at least 2D (batch_size, latent_dim), "
                f"got shapes: z_mean={z_mean_shape}, z_log_var={z_log_var_shape}"
            )

        logger.debug(f"Built Sampling layer with input shapes: {input_shape}")
        super().build(input_shape)

    def call(
        self,
        inputs: Union[Tuple[keras.KerasTensor, keras.KerasTensor], List[keras.KerasTensor]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply reparameterization trick to sample from Normal distribution.

        Args:
            inputs: Tuple or list containing (z_mean, z_log_var) tensors.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                included for consistency with Layer interface.

        Returns:
            Sampled tensor from the latent distribution using the reparameterization trick.

        Raises:
            ValueError: If inputs doesn't contain exactly 2 tensors.
        """
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(
                f"Sampling layer call expects exactly 2 inputs, got {len(inputs) if isinstance(inputs, (tuple, list)) else 'invalid'}"
            )

        z_mean, z_log_var = inputs

        # Get the shape of z_mean for sampling epsilon
        mean_shape = ops.shape(z_mean)

        # Sample epsilon from standard normal distribution
        # Use the same shape as z_mean
        epsilon = keras.random.normal(
            shape=mean_shape,
            seed=self.seed
        )

        # Apply reparameterization trick: z = mean + std * epsilon
        # Note: std = exp(0.5 * log_var) = sqrt(var)
        std = ops.exp(0.5 * z_log_var)
        z_sample = z_mean + std * epsilon

        return z_sample

    def compute_output_shape(self, input_shape: Union[Tuple[Tuple, ...], List[Tuple]]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Tuple or list of two shape tuples for input tensors.

        Returns:
            Output shape tuple (same as z_mean shape).
        """
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(f"Expected 2 input shapes, got {len(input_shape) if isinstance(input_shape, (tuple, list)) else 'invalid'}")

        z_mean_shape, _ = input_shape
        return tuple(z_mean_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration including all parameters
            needed to reconstruct the layer.
        """
        config = super().get_config()
        config.update({
            "seed": self.seed,
        })
        return config

# ---------------------------------------------------------------------