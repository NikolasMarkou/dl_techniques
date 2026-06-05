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

import math
import keras
# ``tf.math.bessel_i0e`` is the SINGLE raw-TensorFlow primitive used in this
# module: the order-0, exponentially-scaled modified Bessel function. It is
# stable and differentiable, and ``keras.ops`` has no Bessel function of any
# order. Every other operation below stays on ``keras.ops``. See D-001.
import tensorflow as tf
from keras import ops
from typing import Tuple, Any, Dict, Literal, Optional, Union, List

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
    - The radius ``r`` is a thin, strictly-positive shell of thickness
      ``shell_thickness * radius`` around the constructor ``radius``,
      per-sample modulated by ``exp(0.5 * log_var)``:
      ``r = radius * (1 + shell_thickness * exp(0.5 * log_var) * eta)``, then
      floored at ``radius * 0.05`` (5% of radius) so ``r`` is always positive.
      Here ``log_var`` is a single scalar per sample (shape ``[B, 1]``) and is
      clipped to ``[-20, 6]`` to cap ``exp`` blow-ups. This replaces the old
      ``radius + exp(0.5 * log_var) * eta`` shell whose std collapsed to
      ``radius`` (the radius-variance KL pulls ``log_var -> 0``), which let
      ~10% of samples take a negative radius (antipode flip) and ~15% land
      at/near the origin -> latents that were NOT on the sphere.

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
        ┌──────┴───────┐    ┌───────────────────────────────┐
        │ g = z_mean   │    │ r = radius * (1 + thickness *  │
        │   + epsilon  │    │   exp(.5*lv)*eta); floor 5%    │
        └──────┬───────┘    └───────────────┬───────────────┘
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
    :param shell_thickness: Positive relative thickness of the radius shell.
        The per-sample radius is ``radius * (1 + shell_thickness *
        exp(0.5 * log_var) * eta)`` floored at ``radius * 0.05``, so the shell
        std at ``log_var = 0`` is ``shell_thickness * radius``. Must be ``> 0``.
        Default ``0.1`` (a genuinely thin shell).
    :type shell_thickness: float
    :param seed: Optional integer seed for reproducible sampling.
    :type seed: Optional[int]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
        self,
        radius: float = 1.0,
        shell_thickness: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the HypersphereSampling layer."""
        super().__init__(**kwargs)

        # Validate radius parameter
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")

        # Validate shell_thickness parameter
        if shell_thickness <= 0:
            raise ValueError(f"shell_thickness must be > 0, got {shell_thickness}")

        # Validate seed parameter
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be an integer or None, got {type(seed)}")

        # Store configuration
        self.radius = radius
        self.shell_thickness = shell_thickness
        self.seed = seed

        logger.debug(
            f"Initialized HypersphereSampling layer with radius={radius}, "
            f"shell_thickness={shell_thickness}, seed={seed}"
        )

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

        # DECISION plan_2026-06-04_7ff8ea8b/D-004: thin, strictly-positive radius shell.
        # The old `radius + exp(0.5*lv)*eta` had shell std ~= radius (KL pulls rlv->0),
        # giving ~10% negative radii / ~15% near-origin samples => latents off the sphere.
        z_log_var_clipped = ops.clip(z_log_var, -20.0, 6.0)   # cap exp() blowups
        shell = self.shell_thickness * ops.exp(0.5 * z_log_var_clipped) * eta   # [B,1]
        r = self.radius * (1.0 + shell)
        r = ops.maximum(r, self.radius * 0.05)   # strictly positive floor (5% of radius)

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
            "shell_thickness": self.shell_thickness,
            "seed": self.seed,
        })
        return config

# ---------------------------------------------------------------------
# von Mises-Fisher (vMF) KL Numerics
# ---------------------------------------------------------------------
#
# Closed-form KL divergence between a vMF posterior ``vMF(mu, kappa)`` and the
# uniform prior on the unit sphere ``S^{dim-1}`` (= vMF with ``kappa = 0``),
# per Davidson et al. 2018 (https://arxiv.org/abs/1804.00891):
#
#     KL = kappa * A_{dim/2}(kappa) + log C_dim(kappa) - log C_dim(0)
#
# where ``A_nu(kappa) = I_nu(kappa) / I_{nu-1}(kappa)`` is a ratio of modified
# Bessel functions of the first kind and ``C_dim`` is the vMF normalizer. The
# KL depends only on ``kappa`` and ``dim`` (NOT on the mean direction ``mu``).
#
# These module-level helpers back the ``vmf`` VAE mode's analytic KL term and
# are validated against ``scipy.special.ive`` in ``tests/test_layers/
# test_sampling.py`` (the SC1 gate; max abserr ~6e-6).


def _bessel_ratio_cf(
        kappa: keras.KerasTensor,
        order: float,
        n_extra: int = 64,
) -> keras.KerasTensor:
    """Compute the Bessel ratio ``A_order(kappa) = I_order / I_{order-1}``.

    Uses the continued-fraction / downward (Miller) recurrence

        ``r_n = 1 / (2 * n / kappa + r_{n+1})``,  seed ``r_N = 0``,

    recursing from ``N = order + n_extra`` down to ``r_order``. This is stable
    for ALL real ``order`` (including the half-integer orders of the odd-``dim``
    path) and all ``kappa``, and is fully differentiable in ``keras.ops``
    arithmetic with no special-function call.

    # DECISION plan_2026-06-04_6196678d/D-001: continued-fraction (downward
    # Miller) Bessel ratio, NEVER upward recurrence (upward is unstable for
    # nu>=kappa: relerr 1e2-1e5 + impossible negative ratios at latent_dim
    # 16/32; verified vs scipy). Do NOT "simplify" this to an upward
    # recurrence or a bessel_i0e/i1e ratio. See decisions.md D-001.

    Args:
        kappa: Concentration tensor, shape ``[B, 1]`` or ``[B]``, ``> 0``.
        order: Bessel order ``nu`` of the numerator (Python float; may be
            half-integer for odd ``dim``).
        n_extra: Number of extra downward-recurrence steps above ``order``
            before seeding ``r = 0``. The default of 64 gives float32
            convergence across the trained ``kappa`` range.

    Returns:
        The ratio ``I_order(kappa) / I_{order-1}(kappa)``, same shape as
        ``kappa``.
    """
    r = ops.zeros_like(kappa)
    n = float(order) + float(n_extra)
    while n >= float(order) - 1e-6:
        r = 1.0 / ((2.0 * n) / kappa + r)
        n -= 1.0
    return r


def _log_iv(
        kappa: keras.KerasTensor,
        order: float,
) -> keras.KerasTensor:
    """Compute ``log I_order(kappa)`` stably by telescoping CF ratios.

    Starts from a base order whose log-Bessel has a closed form and telescopes
    upward via ``log I_j = log I_{j-1} + log A_j(kappa)``:

    - Integer ``order`` -> base order 0, with
      ``log I_0(kappa) = kappa + log(bessel_i0e(kappa))``.
    - Half-integer ``order`` (odd-``dim`` path) -> base order 1/2, with
      ``I_{1/2}(kappa) = sqrt(2 / (pi * kappa)) * sinh(kappa)`` and a stable
      ``log sinh(kappa) = kappa + log1p(-exp(-2 kappa)) - log 2``.

    The single ``bessel_i0e`` call (integer path) is the only raw-TF primitive
    in this module (D-001).

    Args:
        kappa: Concentration tensor, ``> 0``.
        order: Bessel order ``nu`` (Python float; integer or half-integer).

    Returns:
        ``log I_order(kappa)``, same shape as ``kappa``.
    """
    is_half = abs(order - round(order)) > 1e-6
    if not is_half:
        # Integer order: telescope from log I_0.
        log_i = kappa + ops.log(tf.math.bessel_i0e(kappa))
        base = 0.0
    else:
        # Half-integer order: telescope from log I_{1/2}.
        log_sinh = kappa + ops.log1p(-ops.exp(-2.0 * kappa)) - math.log(2.0)
        log_i = 0.5 * ops.log(2.0 / (math.pi * kappa)) + log_sinh
        base = 0.5

    steps = int(round(order - base))
    j = base + 1.0
    for _ in range(steps):
        log_i = log_i + ops.log(_bessel_ratio_cf(kappa, j))
        j += 1.0
    return log_i


def vmf_kl_divergence(
        kappa: keras.KerasTensor,
        dim: int,
) -> keras.KerasTensor:
    """Per-row KL of ``vMF(mu, kappa)`` from the uniform sphere prior.

    Computes ``KL(vMF(mu, kappa) || Uniform(S^{dim-1}))``, which depends on
    ``kappa`` and ``dim`` only (NOT on the mean direction ``mu``):

        ``KL = kappa * A_{dim/2}(kappa) + log C_dim(kappa) - log C_dim(0)``

    with ``log C_dim(kappa) = (dim/2 - 1) log kappa - (dim/2) log(2 pi)
    - log I_{dim/2 - 1}(kappa)`` and the build-time scalar
    ``log C_dim(0) = -log 2 - (dim/2) log pi + lgamma(dim/2)``.

    The Bessel terms use the stable continued-fraction ratio (D-001) and the
    telescoping log-normalizer, supporting all ``dim`` (even AND odd) via the
    half-integer base case in :func:`_log_iv`.

    Args:
        kappa: Concentration tensor, shape ``[B, 1]`` or ``[B]``. Values are
            clipped to ``[1e-6, 1e4]`` for numerical safety (the ``kappa -> 0``
            uniform limit gives ``KL -> 0``).
        dim: Latent dimensionality ``m`` (Python int; the model passes
            ``self.latent_dim``). Loop bounds are static so the graph unrolls
            cleanly.

    Returns:
        Per-row KL divergence, same shape as ``kappa``.
    """
    m = int(dim)
    nu = m / 2.0

    # Guard log(kappa) at the uniform (kappa -> 0) limit and cap large kappa.
    kappa_safe = ops.maximum(kappa, 1e-6)
    kappa_safe = ops.minimum(kappa_safe, 1e4)

    # A_{m/2}(kappa) = I_{m/2}(kappa) / I_{m/2 - 1}(kappa).
    A = _bessel_ratio_cf(kappa_safe, nu)

    log_Cm = (
        (m / 2.0 - 1.0) * ops.log(kappa_safe)
        - (m / 2.0) * math.log(2.0 * math.pi)
        - _log_iv(kappa_safe, nu - 1.0)
    )
    # log C_dim(0): a build-time Python scalar.
    log_Cm0 = (
        -math.log(2.0)
        - (m / 2.0) * math.log(math.pi)
        + math.lgamma(m / 2.0)
    )

    return kappa_safe * A + log_Cm - log_Cm0

# ---------------------------------------------------------------------
# von Mises-Fisher (vMF) Reparameterized Sampler
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VMFSampling(keras.layers.Layer):
    """Sample from a von Mises-Fisher posterior on the unit hypersphere.

    This stateless layer is the third sibling of :class:`Sampling` and
    :class:`HypersphereSampling`. It draws a differentiable reparameterized
    sample from the von Mises-Fisher distribution ``vMF(mu_hat, kappa)`` on the
    UNIT sphere ``S^{D-1}`` (no radius scaling -- vMF is unit-sphere by
    definition). ``mu_hat`` is the L2-normalized encoder mean ``z_mean`` and
    ``kappa`` is a strictly-positive scalar concentration ``[B, 1]``. As
    ``kappa -> 0`` the sample tends to the uniform distribution on the sphere;
    as ``kappa -> inf`` it concentrates at ``mu_hat``. This is the true S-VAE
    posterior of Davidson et al. (2018), in contrast to the thin-shell
    :class:`HypersphereSampling` (which has no directional concentration).

    **Sampling algorithm (Wood 1994 rejection + Householder).**

    1. Draw an axial coordinate ``w in [-1, 1]`` (the cosine of the angle to
       ``mu_hat``) via the Ulrich/Wood Beta-envelope rejection scheme. To avoid
       a dynamic ``while_loop`` the layer draws ``rejection_oversample`` (K)
       candidate ``w`` per row in one batched shot, accepts the FIRST valid
       candidate per row, and falls back to the mode ``w = 1`` in the
       vanishingly-rare event that all K candidates reject.
    2. Draw a tangent direction ``v`` uniformly on ``S^{D-2}`` orthogonal to
       the canonical pole ``e_0`` (first coordinate zeroed) and form the
       canonical sample ``z_can = w * e_0 + sqrt(1 - w^2) * v`` (concentrated at
       ``e_0``).
    3. Rotate ``e_0 -> mu_hat`` by a Householder reflection so the sample
       concentrates at the encoder direction. The degenerate ``mu_hat == e_0``
       case is handled by the identity branch; a degenerate ``z_mean`` (norm
       below ``eps``) falls back to ``e_0`` (the shared ``sampling.py`` idiom).

    The output ``z`` is unit-norm for every row (``||z|| ~ 1`` to float32
    precision).

    **Gradient (v1 approximation).** The Naesseth et al. (2017) implicit
    reparameterization correction is OMITTED in this v1 sampler (see
    ``# DECISION plan_2026-06-04_6196678d/D-004``). The ``kappa`` gradient still
    flows through the Wood envelope parameter ``b(kappa)`` inside ``w_cand``,
    and the ``z_mean`` gradient flows through ``mu_hat`` in the Householder
    reflection. In the VAE the dominant ``kappa`` learning signal is the
    analytic vMF KL (:func:`vmf_kl_divergence`); the omitted correction only
    biases the Monte-Carlo gradient of the reconstruction term and was found
    minor for moderate ``kappa`` by Davidson et al.

    **GPU / XLA limitation.** The rejection sampler draws ``keras.random.beta``,
    which lowers to ``StatelessRandomGammaV3`` — an op with no XLA-GPU kernel in
    TF 2.18. A model containing ``VMFSampling`` must therefore run with XLA
    disabled on GPU: compile with ``jit_compile=False`` (the :class:`VAE` model
    forces this for ``sampling_type='vmf'``) or use an eager call rather than a
    ``predict()``/``fit()`` path that XLA-compiles the beta draw.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐   ┌──────────────────┐
        │  z_mean      │   │  kappa           │
        │  [B, D]      │   │  [B, 1]  (> 0)   │
        └──────┬───────┘   └───────┬──────────┘
               │                   │
        ┌──────┴───────┐    ┌──────┴───────────────────────┐
        │ mu_hat =     │    │ Wood Beta-envelope rejection │
        │  z_mean/||.||│    │  K candidates -> w in [-1,1] │
        │ (zero->e_0)  │    │  (accept-first, w=1 mode FB) │
        └──────┬───────┘    └──────────────┬───────────────┘
               │                           │
               │            ┌──────────────┴───────────────┐
               │            │ z_can = w*e_0 + sqrt(1-w^2)*v │
               │            │  v ~ U(S^{D-2}) ⊥ e_0         │
               │            └──────────────┬───────────────┘
               │                           │
        ┌──────┴───────────────────────────┴───────────────┐
        │ Householder reflect e_0 -> mu_hat  =>  z (unit)   │
        └──────────────────────────┬───────────────────────┘
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │  Output [B, D],  ||z|| ~ 1                        │
        └──────────────────────────────────────────────────┘

    References:
        - Davidson, Falorsi, De Cao, Kipf, & Tomczak, 2018. Hyperspherical
          Variational Auto-Encoders. (https://arxiv.org/abs/1804.00891)
        - Wood, 1994. Simulation of the von Mises Fisher distribution.
          Communications in Statistics - Simulation and Computation, 23(1).
        - Ulrich, 1984. Computer Generation of Distributions on the m-Sphere.
          Applied Statistics, 33(2), 158-163.
        - Naesseth, Ruiz, Linderman, & Blei, 2017. Reparameterization Gradients
          through Acceptance-Rejection Sampling Algorithms. (AISTATS) -- the
          gradient correction OMITTED in this v1.

    :param rejection_oversample: Number ``K`` of candidate axial weights drawn
        per row in one batched shot (fixed-K Wood rejection, no
        ``while_loop``). Must be a positive integer. Default ``32`` (all-reject
        probability ~3e-16 at the trained kappa range).
    :type rejection_oversample: int
    :param seed: Optional integer seed for reproducible sampling. The raw int
        is reused across the Beta/uniform/normal draws (same convention as the
        sibling samplers' save/load contract).
    :type seed: Optional[int]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
        self,
        rejection_oversample: int = 32,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the VMFSampling layer."""
        super().__init__(**kwargs)

        # Validate rejection_oversample parameter
        if not isinstance(rejection_oversample, int) or isinstance(
            rejection_oversample, bool
        ) or rejection_oversample <= 0:
            raise ValueError(
                f"rejection_oversample must be a positive integer, "
                f"got {rejection_oversample!r}"
            )

        # Validate seed parameter
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be an integer or None, got {type(seed)}")

        # Store configuration
        self.rejection_oversample = rejection_oversample
        self.seed = seed

        logger.debug(
            f"Initialized VMFSampling layer with "
            f"rejection_oversample={rejection_oversample}, seed={seed}"
        )

    def build(self, input_shape: Union[Tuple[Tuple, ...], List[Tuple]]) -> None:
        """Validate input shapes and build the layer.

        :param input_shape: List of two shape tuples for ``z_mean`` ``[B, D]``
            and ``kappa`` ``[B, 1]``.
        :type input_shape: Union[Tuple[Tuple, ...], List[Tuple]]"""
        # Validate input structure
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                f"VMFSampling layer expects exactly 2 inputs (z_mean, kappa), "
                f"but received {len(input_shape) if isinstance(input_shape, (tuple, list)) else 'invalid'} inputs."
            )

        z_mean_shape, kappa_shape = input_shape

        # Validate minimum dimensions
        if len(z_mean_shape) < 2 or len(kappa_shape) < 2:
            raise ValueError(
                f"Input tensors must be at least 2D (batch_size, latent_dim), "
                f"got shapes: z_mean={z_mean_shape}, kappa={kappa_shape}"
            )

        # Validate tensor ranks match
        if len(z_mean_shape) != len(kappa_shape):
            raise ValueError(
                f"z_mean and kappa must have the same number of dimensions. "
                f"Got z_mean: {len(z_mean_shape)}, kappa: {len(kappa_shape)}"
            )

        # Validate kappa carries a single scalar concentration per sample
        if kappa_shape[-1] != 1:
            raise ValueError(
                f"kappa last dimension must be exactly 1 (single scalar "
                f"concentration per sample), got shape: {kappa_shape}"
            )

        # vMF is undefined on S^0; require latent_dim D >= 2.
        if z_mean_shape[-1] is not None and z_mean_shape[-1] < 2:
            raise ValueError(
                f"z_mean last dimension (latent_dim) must be >= 2 "
                f"(vMF is undefined on S^0), got shape: {z_mean_shape}"
            )

        logger.debug(f"Built VMFSampling layer with input shapes: {input_shape}")
        super().build(input_shape)

    def call(
        self,
        inputs: Union[Tuple[keras.KerasTensor, keras.KerasTensor], List[keras.KerasTensor]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Sample ``z`` from ``vMF(mu_hat, kappa)`` on the unit sphere.

        :param inputs: Tuple of ``(z_mean, kappa)`` tensors, shaped ``[B, D]``
            and ``[B, 1]`` respectively.
        :type inputs: Union[Tuple, List]
        :param training: Training flag (unused).
        :type training: Optional[bool]
        :return: Sampled unit-norm latent tensor with same shape as ``z_mean``.
        :rtype: keras.KerasTensor"""
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(
                f"VMFSampling layer call expects exactly 2 inputs, got {len(inputs) if isinstance(inputs, (tuple, list)) else 'invalid'}"
            )

        z_mean, kappa = inputs

        eps0 = 1e-12
        D = int(z_mean.shape[-1])
        B = ops.shape(z_mean)[0]
        K = self.rejection_oversample

        # Unit mean direction mu_hat from z_mean, with the canonical e_0
        # fallback on a degenerate (near-zero) encoder mean (shared idiom).
        norm = ops.sqrt(ops.sum(ops.square(z_mean), axis=-1, keepdims=True))
        e0 = ops.one_hot(ops.zeros((B,), "int32"), D)                 # [B, D]
        mu = ops.where(norm < eps0, e0, z_mean / ops.maximum(norm, eps0))
        kap = ops.maximum(kappa, 1e-6)   # [B, 1], keep > 0; NOT stop_gradient

        # DECISION plan_2026-06-04_6196678d/D-002: fixed-K (no while_loop) Wood
        # 1994 vMF rejection sampler. Draw K candidate axial weights per row in
        # ONE batched shot via the Beta envelope, accept the FIRST valid
        # candidate per row, fall back to the mode w=1 if all K reject
        # (P~3e-16 at K=32). Do NOT replace this with a dynamic keras.ops
        # while_loop (harder to keep differentiable + XLA-clean). See D-002.
        # Ulrich (1984) / Wood (1994) exact acceptance form. The envelope mode
        # is x0 = (1-b)/(1+b) and the log-acceptance offset is
        # c = kappa*x0 + (D-1)*log(1 - x0^2); accept on
        # kappa*w + (D-1)*log(1 - x0*w) >= c + log(u). This is the
        # E[w] == A_{D/2}(kappa)-faithful variant (verified vs scipy's
        # vonmises_fisher to ~2e-4); do NOT substitute the
        # d = 4ab/(1+b) - (D-1)log(D-1) / log(1-(1-b)w) variant, which is
        # systematically under-concentrated by ~0.05 in E[w].
        m1 = float(D - 1)
        s = ops.sqrt(4.0 * ops.square(kap) + m1 * m1)
        b = (-2.0 * kap + s) / m1                                     # [B, 1]
        x0 = (1.0 - b) / (1.0 + b)                                    # [B, 1]
        c = kap * x0 + m1 * ops.log(ops.maximum(1.0 - x0 * x0, eps0))  # [B, 1]

        # K candidate axial coords w in one batched shot. beta(shape, a, b).
        epsb = keras.random.beta((B, K), m1 / 2.0, m1 / 2.0, seed=self.seed)  # [B,K]
        w_cand = (1.0 - (1.0 + b) * epsb) / (1.0 - (1.0 - b) * epsb)  # [B,K]
        uni = keras.random.uniform((B, K), seed=self.seed)            # [B,K]
        accept = (
            kap * w_cand + m1 * ops.log(ops.maximum(1.0 - x0 * w_cand, eps0)) - c
        ) >= ops.log(uni)                                            # [B,K] bool
        acc_f = ops.cast(accept, w_cand.dtype)
        any_acc = ops.max(acc_f, axis=1, keepdims=True) > 0.0        # [B,1]
        first = ops.argmax(acc_f, axis=1)                            # [B] first True
        w_sel = ops.take_along_axis(
            w_cand, ops.expand_dims(first, 1), axis=1
        )                                                            # [B,1]
        w = ops.where(any_acc, w_sel, ops.ones_like(w_sel))         # mode fallback
        w = ops.clip(w, -1.0, 1.0)                                  # guard sqrt

        # DECISION plan_2026-06-04_6196678d/D-004: the Naesseth-2017 implicit
        # reparameterization correction is OMITTED in this v1. The kappa
        # gradient flows through b(kappa) in w_cand above and the z_mean
        # gradient through mu in the Householder reflection below; the dominant
        # kappa signal is the analytic KL. Do NOT add the correction here
        # without re-evaluating stability (see decisions.md D-004).

        # Tangent direction v ~ uniform on S^{D-2}, orthogonal to e_0.
        t = keras.random.normal((B, D), seed=self.seed)
        zero_first = ops.concatenate(
            [ops.zeros((B, 1)), ops.ones((B, D - 1))], axis=1
        )
        t = t * zero_first
        v = t / ops.maximum(
            ops.sqrt(ops.sum(ops.square(t), axis=-1, keepdims=True)), eps0
        )                                                            # [B,D], v[...,0]=0
        z_can = w * e0 + ops.sqrt(ops.maximum(1.0 - ops.square(w), 0.0)) * v

        # Householder reflection e_0 -> mu_hat (swaps the two unit vectors):
        # H x = x - 2 (uhat . x) uhat, uhat = (e_0 - mu) / ||e_0 - mu||.
        uvec = e0 - mu
        unorm = ops.sqrt(ops.sum(ops.square(uvec), axis=-1, keepdims=True))
        uhat = uvec / ops.maximum(unorm, eps0)
        z_ref = z_can - 2.0 * ops.sum(uhat * z_can, axis=-1, keepdims=True) * uhat
        z = ops.where(unorm < eps0, z_can, z_ref)   # mu == e_0 -> identity

        return z

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
            "rejection_oversample": self.rejection_oversample,
            "seed": self.seed,
        })
        return config

# ---------------------------------------------------------------------
# Sampling Layer Factory (inline)
# ---------------------------------------------------------------------
#
# A registry-driven factory for the two reparameterization samplers defined
# above. It mirrors the canonical ``sequence_pooling/factory.py`` 4-function
# surface (validate -> merge defaults -> filter to ctor params -> inject name
# -> ``cls(**params)``; unknown-type errors name the available types).
#
# DECISION plan_2026-06-04_d4ef81f1/D-001: this factory is placed INLINE in
# sampling.py and NOT promoted to a ``sampling/`` package with a sibling
# ``factory.py``. Do NOT "tidy" it into a package: ``sampling.py`` is a
# top-level module imported as ``from dl_techniques.layers.sampling import
# Sampling`` by >=3 sites, and promotion would break every such caller for no
# functional gain (the two samplers have only 1-2 ctor params each). The
# inline placement has a repo precedent (``sparse_autoencoder.py``). See
# decisions.md D-001.

# ---------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------

SamplingType = Literal["gaussian", "hypersphere", "vmf"]
"""
Type alias for supported reparameterization-sampler mechanisms.

This literal type provides IDE autocompletion and type checking for valid
sampler types supported by the factory.
"""

# ---------------------------------------------------------------------
# Sampling Layer Registry
# ---------------------------------------------------------------------

SAMPLING_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gaussian": {
        "class": Sampling,
        "description": (
            "Gaussian-ball reparameterization (z=mu+exp(0.5*log_var)*eps)"
        ),
        "required_params": [],
        "optional_params": {
            "seed": None,
        },
        "use_case": "standard VAE diagonal-Gaussian posterior",
    },

    "hypersphere": {
        "class": HypersphereSampling,
        "description": (
            "Thin-shell hypersphere reparameterization "
            "(z=r*normalize(z_mean+eps))"
        ),
        "required_params": [],
        "optional_params": {
            "radius": 1.0,
            "seed": None,
        },
        "use_case": (
            "hyperspherical-latent VAE; direction on unit sphere, scalar "
            "radius shell"
        ),
    },

    "vmf": {
        "class": VMFSampling,
        "description": (
            "von Mises-Fisher reparameterized sampler (Wood 1994 rejection "
            "+ Householder; unit sphere, no radius)"
        ),
        "required_params": [],
        "optional_params": {
            "rejection_oversample": 32,
            "seed": None,
        },
        "use_case": (
            "true vMF S-VAE; directional posterior on the unit sphere"
        ),
    },
}
"""
Registry of reparameterization-sampler implementations with metadata.

Each entry contains:
    - class: The actual layer class implementation.
    - description: Technical description of the sampling mechanism.
    - required_params: List of mandatory parameters for instantiation.
    - optional_params: Dict of optional parameters with default values.
    - use_case: Scenarios and applications where this sampler excels.
"""


# ---------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------

def get_sampling_info() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve metadata for all available sampler types.

    Provides per-type metadata including the technical description, parameter
    specifications, and use cases for every supported sampling mechanism.

    Returns:
        A dictionary mapping each sampler type to its metadata (description,
        required_params, optional_params, use_case). Each entry is a shallow
        copy so callers cannot mutate the registry.
    """
    return {
        sampling_type: info.copy()
        for sampling_type, info in SAMPLING_REGISTRY.items()
    }


def validate_sampling_config(
        sampling_type: str,
        **kwargs: Any
) -> None:
    """
    Validate sampler configuration parameters.

    Performs type-existence checking, required-parameter completeness, and
    light value-range validation on any numeric parameters that are present.

    Args:
        sampling_type: The sampler type to validate against.
        **kwargs: Parameter dictionary to validate for the specified type.

    Raises:
        ValueError: If sampling_type is not supported, required parameters are
            missing, or a provided parameter value violates its constraint.
    """
    if sampling_type not in SAMPLING_REGISTRY:
        available_types = sorted(SAMPLING_REGISTRY.keys())
        raise ValueError(
            f"Unknown sampling type '{sampling_type}'. "
            f"Available types: {available_types}"
        )

    info = SAMPLING_REGISTRY[sampling_type]
    required = info["required_params"]
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Required parameters for '{sampling_type}' are missing: "
            f"{missing}. Required: {required}, Provided: "
            f"{list(kwargs.keys())}"
        )

    # Validate positive-float parameters: a provided radius must be > 0.
    if "radius" in kwargs and kwargs["radius"] <= 0:
        raise ValueError(
            f"Parameter 'radius' must be > 0, got {kwargs['radius']}"
        )

    logger.debug(
        f"Validation successful for '{sampling_type}' with parameters: "
        f"{kwargs}"
    )


def create_sampling_layer(
        sampling_type: SamplingType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating reparameterization-sampler layers.

    Provides a centralized, type-safe way to instantiate any sampler layer in
    this module, with parameter validation, default-value handling, and
    detailed error reporting. Dispatch is pure registry lookup (no if/elif on
    type).

    Args:
        sampling_type: The type of sampler to create (``'gaussian'`` or
            ``'hypersphere'``).
        name: Optional name for the layer instance.
        **kwargs: Type-specific parameters for the sampler layer. See
            ``get_sampling_info()`` for per-type parameter specs.

    Returns:
        A fully configured and instantiated sampler layer.

    Raises:
        ValueError: If sampling_type is invalid, required parameters are
            missing, parameter values are out of range, or layer construction
            fails.
        TypeError: If parameter types are incompatible with the target class.
    """
    try:
        # Validate configuration before proceeding
        validate_sampling_config(sampling_type, **kwargs)

        # Get layer information and class
        info = SAMPLING_REGISTRY[sampling_type]
        sampling_class = info["class"]

        # Merge user parameters with defaults (user wins)
        params = info["optional_params"].copy()
        params.update(kwargs)

        # Filter parameters to match the constructor signature
        valid_param_names = set(info["required_params"]) | set(
            info["optional_params"].keys()
        )
        final_params = {
            k: v for k, v in params.items() if k in valid_param_names
        }

        # Add name if provided
        if name:
            final_params["name"] = name

        logger.info(
            f"Creating '{sampling_type}' sampling layer "
            f"({sampling_class.__name__}) with parameters: {final_params}"
        )

        # Instantiate the sampler layer
        return sampling_class(**final_params)

    except (TypeError, ValueError) as e:
        # Provide detailed error context, including the available-type hint.
        info = SAMPLING_REGISTRY.get(sampling_type)
        if info:
            class_name = info["class"].__name__
            error_msg = (
                f"Failed to create '{sampling_type}' sampling layer "
                f"({class_name}). "
                f"Required parameters: {info['required_params']}. "
                f"Provided parameters: {list(kwargs.keys())}. "
                f"Please verify parameter compatibility. Original error: {e}"
            )
        else:
            error_msg = (
                f"Failed to create sampling layer. "
                f"Unknown type '{sampling_type}'. "
                f"Available types: {sorted(get_sampling_info().keys())}. "
                f"Error: {e}"
            )

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_sampling_from_config(
        config: Dict[str, Any]
) -> keras.layers.Layer:
    """
    Create a sampler layer from a configuration dictionary.

    Convenience function for instantiating sampler layers from dictionary-based
    configurations, useful for loading architectures from JSON/YAML files,
    hyperparameter optimization, and configuration-driven model building.

    Args:
        config: Configuration dictionary containing a ``'type'`` key specifying
            the sampler type and additional keys for layer-specific parameters.

    Returns:
        Instantiated and configured sampler layer.

    Raises:
        ValueError: If config is missing the required ``'type'`` key.
        TypeError: If config parameter types are invalid.
    """
    config_copy = config.copy()
    try:
        sampling_type = config_copy.pop("type")
    except KeyError as e:
        available_keys = list(config.keys()) if config else []
        raise ValueError(
            f"Configuration dictionary must include a 'type' key specifying "
            f"the sampling layer type. Available keys in config: "
            f"{available_keys}. "
            f"Valid types: {sorted(SAMPLING_REGISTRY.keys())}"
        ) from e

    logger.debug(f"Creating sampling layer from config: {config}")
    return create_sampling_layer(sampling_type, **config_copy)

# ---------------------------------------------------------------------