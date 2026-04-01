"""
Sample from a latent Normal distribution using the reparameterization trick.

This layer is the central component that enables the training of Variational
Autoencoders (VAEs) through gradient-based optimization. In a VAE, the
encoder network learns to map an input `x` to the parameters of a posterior
distribution `q_φ(z|x)`, typically a diagonal Gaussian. This layer's role is
to draw a sample `z` from this learned distribution, which is then passed to
the decoder.

The primary challenge in this architecture is that the sampling operation is
stochastic and thus non-differentiable. Standard backpropagation cannot flow
through a random node, which would prevent the training of the encoder. This
layer resolves this issue by implementing the "reparameterization trick."

The core mathematical insight is to re-express the random variable `z` as a
deterministic function of the distribution's parameters (mean `μ` and
standard deviation `σ`) and an auxiliary, parameter-independent random
variable `ε`. For a Gaussian distribution, this is formulated as:

`z = μ + σ * ε`,  where `ε ~ N(0, I)`

Here, `μ` and `log_var` (from which `σ` is derived as `exp(0.5 * log_var)`)
are the outputs of the encoder network. The randomness is sourced entirely
from `ε`, which is sampled from a fixed standard normal distribution. The
transformation that produces `z` is now a simple, deterministic computation.

This reformulation makes the entire model differentiable with respect to the
encoder's parameters `φ`. The gradient of the loss function can flow from the
decoder, through the sampled `z`, back to `μ` and `σ`, and finally to the
weights of the encoder. This allows the VAE to be trained end-to-end using
standard stochastic gradient descent, jointly optimizing the encoder and
decoder.

References:
    - Kingma & Welling, 2013. Auto-Encoding Variational Bayes.
      (https://arxiv.org/abs/1312.6114)
    - Rezende, Mohamed, & Wierstra, 2014. Stochastic Backpropagation and
      Approximate Inference in Deep Generative Models.
      (https://arxiv.org/abs/1401.4082)

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
    """Sample from a latent Normal distribution via the reparameterisation trick.

    This stateless layer takes ``(z_mean, z_log_var)`` from a VAE encoder
    and produces a differentiable sample
    ``z = mu + exp(0.5 * log_var) * epsilon`` where
    ``epsilon ~ N(0, I)``. Because the randomness is isolated in the
    auxiliary variable ``epsilon``, gradients can flow through ``z`` back
    to the encoder parameters, enabling end-to-end training of
    Variational Autoencoders.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐   ┌──────────────────┐
        │  z_mean      │   │  z_log_var       │
        │  [B, D]      │   │  [B, D]          │
        └──────┬───────┘   └────────┬─────────┘
               │                    │
               │         ┌─────────┴─────────┐
               │         │ std = exp(0.5 *   │
               │         │       log_var)    │
               │         └─────────┬─────────┘
               │                   │
               │    ┌──────────────┤
               │    │   epsilon    │
               │    │   ~ N(0, I)  │
               │    └──────┬───────┘
               │           │
               ▼           ▼
        ┌──────────────────────────────────┐
        │  z = z_mean + std * epsilon      │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, D]                   │
        └──────────────────────────────────┘

    :param seed: Optional integer seed for reproducible sampling.
    :type seed: Optional[int]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
        self,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the Sampling layer."""
        super().__init__(**kwargs)

        # Validate seed parameter
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be an integer or None, got {type(seed)}")

        # Store configuration
        self.seed = seed

        logger.debug(f"Initialized Sampling layer with seed={seed}")

    def build(self, input_shape: Union[Tuple[Tuple, ...], List[Tuple]]) -> None:
        """Validate input shapes and build the layer.

        :param input_shape: List of two shape tuples for ``z_mean``
            and ``z_log_var``.
        :type input_shape: Union[Tuple[Tuple, ...], List[Tuple]]"""
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
        """Apply the reparameterisation trick to sample ``z``.

        :param inputs: Tuple of ``(z_mean, z_log_var)`` tensors.
        :type inputs: Union[Tuple, List]
        :param training: Training flag (unused).
        :type training: Optional[bool]
        :return: Sampled latent tensor with same shape as ``z_mean``.
        :rtype: keras.KerasTensor"""
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
        """Compute the output shape (same as ``z_mean``).

        :param input_shape: List of two input shape tuples.
        :type input_shape: Union[Tuple[Tuple, ...], List[Tuple]]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(f"Expected 2 input shapes, got {len(input_shape) if isinstance(input_shape, (tuple, list)) else 'invalid'}")

        z_mean_shape, _ = input_shape
        return tuple(z_mean_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            "seed": self.seed,
        })
        return config

# ---------------------------------------------------------------------