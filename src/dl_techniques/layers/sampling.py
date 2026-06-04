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

This file now hosts two sibling reparameterization samplers. ``Sampling``
draws from a Gaussian *ball* (``z = mu + sigma * epsilon``), the classic VAE
posterior sample. ``HypersphereSampling`` instead draws from a thin *shell*
of a (scaled) unit hypersphere: the direction is obtained from the encoder
mean plus Gaussian noise, L2-normalized onto the sphere, and the radius is a
thin Gaussian shell centered at a constructor ``radius`` with per-sample
thickness derived from ``z_log_var``. Both layers share the same stateless
5-method Keras-3 skeleton, raw-int seed convention, and serialization keys.

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
        └──────┬───────┘   └───────┬──────────┘
               │                   │
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


@keras.saving.register_keras_serializable()
class HypersphereSampling(keras.layers.Layer):
    """Sample from a thin Gaussian shell of a scaled unit hypersphere.

    This stateless layer is a sibling of :class:`Sampling`. Instead of drawing
    from a Gaussian *ball* it draws from a thin *shell* of a (scaled) unit
    hypersphere, decoupling *direction* from *radius*:

    - The direction ``u`` comes from the encoder mean ``z_mean`` plus Gaussian
      noise ``epsilon ~ N(0, I)``, L2-normalized onto the unit sphere. The
      ``z_mean`` information is preserved in the *direction* of the sample.
    - The radius ``r`` is a thin Gaussian shell centered at the constructor
      ``radius`` with per-sample thickness ``exp(0.5 * log_var)``, where
      ``log_var`` is a single scalar per sample (shape ``[B, 1]``).

    The sample is ``z = r * u``. Because randomness is isolated in the
    auxiliary variables ``epsilon`` and ``eta``, gradients flow through ``z``
    back to BOTH ``z_mean`` (through ``u``) and ``z_log_var`` (through ``r``),
    enabling end-to-end training. Sampling direction via Gaussian noise plus
    L2-normalization is the classic Marsaglia/Muller method for drawing a
    point uniformly from the surface of a sphere.

    **Degenerate direction.** When ``z_mean + epsilon`` is exactly the zero
    vector, the unit direction is deterministically resolved to the canonical
    basis vector ``e_0 = [1, 0, ..., 0]`` (not a zero / NaN / eps-floored
    near-zero), giving a well-defined, gradient-stable output ``z = r * e_0``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐   ┌──────────────────┐
        │  z_mean      │   │  z_log_var       │
        │  [B, D]      │   │  [B, 1]          │
        └──────┬───────┘   └───────┬──────────┘
               │                   │
        ┌──────┴───────┐    ┌───────┴──────────┐
        │  epsilon     │    │  eta             │
        │  ~ N(0, I)   │    │  ~ N(0, 1)       │
        └──────┬───────┘    └───────┬──────────┘
               │                    │
        ┌──────┴───────┐    ┌───────┴──────────┐
        │ g = z_mean   │    │ r = radius +     │
        │   + epsilon  │    │  exp(.5*lv)*eta  │
        └──────┬───────┘    └───────┬──────────┘
               │                    │
        ┌──────┴───────┐            │
        │ u = g/||g||  │            │
        │ (zero->e_0)  │            │
        └──────┬───────┘            │
               ▼                    ▼
        ┌──────────────────────────────────┐
        │  z = r * u                       │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, D]                   │
        └──────────────────────────────────┘

    References:
        - Marsaglia, 1972. Choosing a Point from the Surface of a Sphere.
          Annals of Mathematical Statistics, 43(2), 645-646.
        - Muller, 1959. A Note on a Method for Generating Points Uniformly on
          N-Dimensional Spheres. Communications of the ACM, 2(4), 19-20.
        - Davidson, Falorsi, De Cao, Kipf, & Tomczak, 2018. Hyperspherical
          Variational Auto-Encoders. (https://arxiv.org/abs/1804.00891)

    :param radius: Positive radius of the hypersphere shell center. Must be
        ``> 0``.
    :type radius: float
    :param seed: Optional integer seed for reproducible sampling.
    :type seed: Optional[int]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
        self,
        radius: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the HypersphereSampling layer."""
        super().__init__(**kwargs)

        # Validate radius parameter
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")

        # Validate seed parameter
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be an integer or None, got {type(seed)}")

        # Store configuration
        self.radius = radius
        self.seed = seed

        logger.debug(f"Initialized HypersphereSampling layer with radius={radius}, seed={seed}")

    def build(self, input_shape: Union[Tuple[Tuple, ...], List[Tuple]]) -> None:
        """Validate input shapes and build the layer.

        :param input_shape: List of two shape tuples for ``z_mean``
            and ``z_log_var``.
        :type input_shape: Union[Tuple[Tuple, ...], List[Tuple]]"""
        # Validate input structure
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                f"HypersphereSampling layer expects exactly 2 inputs (z_mean, z_log_var), "
                f"but received {len(input_shape) if isinstance(input_shape, (tuple, list)) else 'invalid'} inputs."
            )

        z_mean_shape, z_log_var_shape = input_shape

        # Validate minimum dimensions
        if len(z_mean_shape) < 2 or len(z_log_var_shape) < 2:
            raise ValueError(
                f"Input tensors must be at least 2D (batch_size, latent_dim), "
                f"got shapes: z_mean={z_mean_shape}, z_log_var={z_log_var_shape}"
            )

        # Validate tensor ranks match
        if len(z_mean_shape) != len(z_log_var_shape):
            raise ValueError(
                f"z_mean and z_log_var must have the same number of dimensions. "
                f"Got z_mean: {len(z_mean_shape)}, z_log_var: {len(z_log_var_shape)}"
            )

        # Validate z_log_var carries a single scalar variance per sample
        if z_log_var_shape[-1] != 1:
            raise ValueError(
                f"z_log_var last dimension must be exactly 1 (single scalar "
                f"variance per sample), got shape: {z_log_var_shape}"
            )

        logger.debug(f"Built HypersphereSampling layer with input shapes: {input_shape}")
        super().build(input_shape)

    def call(
        self,
        inputs: Union[Tuple[keras.KerasTensor, keras.KerasTensor], List[keras.KerasTensor]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Sample ``z`` from the thin hypersphere shell.

        :param inputs: Tuple of ``(z_mean, z_log_var)`` tensors, shaped
            ``[B, D]`` and ``[B, 1]`` respectively.
        :type inputs: Union[Tuple, List]
        :param training: Training flag (unused).
        :type training: Optional[bool]
        :return: Sampled latent tensor with same shape as ``z_mean``.
        :rtype: keras.KerasTensor"""
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(
                f"HypersphereSampling layer call expects exactly 2 inputs, got {len(inputs) if isinstance(inputs, (tuple, list)) else 'invalid'}"
            )

        z_mean, z_log_var = inputs

        # Small floor to guard the normalization against division by zero.
        eps0 = 1e-12

        # Sample the auxiliary noise. The raw int seed is passed directly to
        # keras.random.normal (NOT a SeedGenerator) so that same-seed -> same
        # output, matching the sibling Sampling layer's save/load contract.
        # DECISION plan_2026-06-04_a114f829/D-002
        epsilon = keras.random.normal(shape=ops.shape(z_mean), seed=self.seed)  # [B, D]
        eta = keras.random.normal(shape=ops.shape(z_log_var), seed=self.seed)   # [B, 1]

        # Raw direction = encoder mean + Gaussian noise.
        g = z_mean + epsilon  # [B, D]
        norm = ops.sqrt(ops.sum(ops.square(g), axis=-1, keepdims=True))  # [B, 1]

        # Degenerate-direction handling: when g is exactly the zero vector the
        # unit direction resolves to the canonical basis vector e_0 =
        # [1, 0, ..., 0] (a fixed, gradient-stable target), NOT a zero / NaN /
        # arbitrary eps-floored near-zero. Do NOT replace this with a bare
        # ops.normalize (it cannot emit e_0 on the zero row).
        # DECISION plan_2026-06-04_a114f829/D-004
        batch = ops.shape(g)[0]
        dim = ops.shape(g)[-1]
        e_0 = ops.one_hot(ops.zeros((batch,), dtype="int32"), dim)  # [B, D], 1 at col 0
        u = ops.where(norm < eps0, e_0, g / ops.maximum(norm, eps0))  # [B, D]

        # Thin Gaussian shell radius centered at self.radius.
        r = self.radius + ops.exp(0.5 * z_log_var) * eta  # [B, 1]

        # Scale the unit direction by the per-sample radius (broadcast [B,1]*[B,D]).
        z_sample = r * u  # [B, D]

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
            "radius": self.radius,
            "seed": self.seed,
        })
        return config

# ---------------------------------------------------------------------