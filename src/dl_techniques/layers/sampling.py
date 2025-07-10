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

Typical Usage:
  >>> # Define the latent dimension
  >>> latent_dim = 2
  >>>
  >>> # Assume `encoder_output` is the output of the main encoder network
  >>> # e.g., encoder_output = keras.layers.Dense(16, activation="relu")(input_tensor)
  >>>
  >>> # Project the encoder output into mean and log variance vectors
  >>> z_mean = keras.layers.Dense(latent_dim, name="z_mean")(encoder_output)
  >>> z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(encoder_output)
  >>>
  >>> # Use the Sampling layer to get a sample from the latent space
  >>> z = Sampling()([z_mean, z_log_var])
  >>>
  >>> # The sampled `z` tensor can then be passed to the decoder
  >>> # e.g., decoded_output = decoder(z)
  >>> # vae_model = keras.Model(inputs=input_tensor, outputs=decoded_output)
"""

import keras
from keras import ops
from typing import Tuple, Any, Dict, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Sampling(keras.layers.Layer):
    """Uses reparameterization trick to sample from a Normal distribution.

    This layer implements the reparameterization trick commonly used in Variational
    Autoencoders (VAEs). It takes as input the mean and log variance of a latent
    distribution and returns a sample from that distribution using the reparameterization
    trick: z = mean + std * epsilon, where epsilon is sampled from a standard normal
    distribution.

    The reparameterization trick allows gradients to flow through the sampling operation
    by expressing the random variable as a deterministic function of the parameters and
    an auxiliary noise variable.

    Args:
        seed: Optional integer seed for random number generation. If None, uses
            random seed for each call. Defaults to None.
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
        >>> # In a VAE encoder
        >>> encoded = encoder_layers(inputs)
        >>> z_mean = keras.layers.Dense(latent_dim, name="z_mean")(encoded)
        >>> z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(encoded)
        >>> # Sample from the latent distribution
        >>> z = Sampling()([z_mean, z_log_var])
        >>>
        >>> # With deterministic seed for reproducibility
        >>> z = Sampling(seed=42)([z_mean, z_log_var])
    """

    def __init__(
            self,
            seed: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the Sampling layer.

        Args:
            seed: Optional integer seed for random number generation.
            **kwargs: Additional keyword arguments for the Layer parent class.
        """
        super().__init__(**kwargs)
        self.seed = seed

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.debug(f"Initialized Sampling layer with seed={seed}")

    def build(self, input_shape: Union[Tuple[Tuple, Tuple], list]) -> None:
        """Build the layer and validate input shapes.

        Args:
            input_shape: Tuple or list of two shape tuples for z_mean and z_log_var tensors.

        Raises:
            ValueError: If input_shape doesn't contain exactly 2 tensors or if the
                tensor shapes are incompatible.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Validate input shapes
        if len(input_shape) != 2:
            raise ValueError(
                f"Sampling layer expects exactly 2 inputs (z_mean, z_log_var), "
                f"but received {len(input_shape)} inputs."
            )

        z_mean_shape, z_log_var_shape = input_shape

        if len(z_mean_shape) != len(z_log_var_shape):
            raise ValueError(
                f"z_mean and z_log_var must have the same number of dimensions. "
                f"Got z_mean: {len(z_mean_shape)}, z_log_var: {len(z_log_var_shape)}"
            )

        if z_mean_shape[1:] != z_log_var_shape[1:]:
            raise ValueError(
                f"z_mean and z_log_var must have the same shape except for batch dimension. "
                f"Got z_mean: {z_mean_shape}, z_log_var: {z_log_var_shape}"
            )

        logger.debug(f"Built Sampling layer with input shapes: {input_shape}")
        super().build(input_shape)

    def call(
            self,
            inputs: Union[Tuple[keras.KerasTensor, keras.KerasTensor], list],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply reparameterization trick to sample from Normal distribution.

        Args:
            inputs: Tuple or list containing (z_mean, z_log_var) tensors.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                included for consistency.

        Returns:
            Sampled tensor from the latent distribution using the reparameterization trick.

        Raises:
            ValueError: If inputs doesn't contain exactly 2 tensors.
        """
        if len(inputs) != 2:
            raise ValueError(
                f"Sampling layer call expects exactly 2 inputs, got {len(inputs)}"
            )

        z_mean, z_log_var = inputs

        # Sample epsilon from standard normal distribution
        epsilon = keras.random.normal(
            shape=ops.shape(z_mean),
            seed=self.seed
        )

        # Apply reparameterization trick: z = mean + std * epsilon
        # Note: std = exp(0.5 * log_var)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape: Union[Tuple[Tuple, Tuple], list]) -> Tuple:
        """Compute the output shape of the layer.

        Args:
            input_shape: Tuple or list of two shape tuples for input tensors.

        Returns:
            Output shape tuple (same as z_mean shape).
        """
        z_mean_shape, _ = input_shape
        return tuple(z_mean_shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "seed": self.seed,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        This method is needed for proper model saving and loading.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
