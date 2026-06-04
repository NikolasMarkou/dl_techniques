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

SamplingType = Literal["gaussian", "hypersphere"]
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