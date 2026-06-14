"""
A differentiable Gaussian Mixture Model (GMM) clustering layer for deep networks.

This layer embeds a soft-clustering mechanism based on a Gaussian Mixture Model
directly into a neural network, enabling end-to-end gradient-based training of
the mixture parameters (component means, diagonal covariances, and mixing
weights). It provides a fully differentiable, probabilistic alternative to hard
K-means: instead of assigning each input to a single nearest centroid, it
computes posterior responsibilities under a learned mixture of diagonal
Gaussians.

Architecture and Core Concepts:

Each of the ``K`` components is parameterized by a mean ``mu_k``, a diagonal
covariance ``Sigma_k = diag(var_k)`` (stored in log-space for positivity), and a
mixing logit. The responsibility (soft assignment) of an input ``x_i`` to
component ``k`` is the posterior probability under the mixture.

Key mechanisms include:

1.  **Probabilistic Soft Assignments (E-step):** The layer evaluates the
    diagonal-Gaussian log-density of each input under every component, adds the
    log mixing weights, and normalizes via a (temperature-scaled) softmax to
    obtain responsibilities. This is the exact GMM posterior at ``temperature=1``
    and is differentiable with respect to all mixture parameters.

2.  **Fully Differentiable Parameters:** Means, log-variances, and mixing logits
    are standard trainable weights optimized by the host model's optimizer
    through the main loss together with the responsibilities. There is no
    non-differentiable hard re-estimation step.

3.  **Isometric-Kernel Regularization:** To counteract degenerate, highly
    anisotropic components and to encourage well-conditioned, spherical
    (isometric) Gaussian kernels, an auxiliary loss penalizes the dispersion of
    each component's per-dimension log-variances around their per-component mean.
    Driving this dispersion to zero pushes every covariance toward an isotropic
    form ``Sigma_k = sigma_k^2 * I``, under which the Mahalanobis metric reduces
    to a scaled Euclidean (isometric) metric.

Mathematical Foundation:

    The diagonal-Gaussian log-density of ``x_i`` under component ``k`` is:
    ``log N(x_i | mu_k, Sigma_k) =
        -0.5 * ( sum_d (x_id - mu_kd)^2 / var_kd + sum_d log var_kd + D log(2*pi) )``

    The responsibility is the temperature-scaled posterior:
    ``r_ik = softmax_k( (log pi_k + log N(x_i | mu_k, Sigma_k)) / tau )``

    The isometric-kernel regularization term is:
    ``L_iso = lambda * mean_k mean_d ( log var_kd - mean_d' log var_kd' )^2``

References:

    -   Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*,
        Chapter 9 (Mixture Models and EM).
    -   The end-to-end learning of a probabilistic codebook is related to
        methods such as Vector-Quantized / Gaussian-Mixture Variational
        Autoencoders (van den Oord, A., et al., 2017; Dilokthanakul, N., et al.,
        2016).
    -   Penalizing anisotropy to promote spherical, well-conditioned components
        is a diversity/conditioning regularizer used to avoid covariance collapse
        in deep mixture models.
"""

import keras
import numpy as np
from typing import Optional, Union, Literal, List, Any, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...utils.logger import logger
from ...initializers.orthonormal_initializer import OrthonormalInitializer

# ---------------------------------------------------------------------

# Type aliases for better readability
OutputMode = Literal['assignments', 'mixture']
TensorShape = Union[Tuple[int, ...], List[int]]
Axis = Union[int, List[int]]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GMMLayer(keras.layers.Layer):
    """Differentiable Gaussian Mixture Model layer with isometric-kernel regularization.

    This layer implements a fully differentiable soft-clustering mechanism based on
    a mixture of ``K`` diagonal Gaussians. For each input it computes posterior
    responsibilities ``r_ik = softmax_k((log pi_k + log N(x_i | mu_k, Sigma_k)) / tau)``
    where ``Sigma_k = diag(var_k)``. Means, log-variances, and mixing logits are
    trainable weights optimized end-to-end. An auxiliary isometric-kernel loss
    penalizes per-component anisotropy, driving each covariance toward an isotropic
    (spherical) form ``Sigma_k = sigma_k^2 * I``.

    **Architecture Overview:**

    .. code-block:: text

        +-------------------------------------+
        |   Input (arbitrary shape)           |
        +----------------+--------------------+
                         |
                         v
        +-------------------------------------+
        |  Reshape for clustering             |
        |  (flatten cluster_axis dims)        |
        +----------------+--------------------+
                         |
                         v
        +-------------------------------------+
        |  Diagonal-Gaussian log-densities    |
        |  log N(x | mu_k, Sigma_k)           |
        +----------------+--------------------+
                         |
                         v
        +-------------------------------------+
        |  Responsibilities:                  |
        |  softmax((log pi + log N) / tau)    |
        +--------+--------------+-------------+
                 |              |
        (isometric loss)        |
                 v              |
        +-----------------+     |
        | add_loss:       |     |
        | anisotropy      |     |
        | penalty         |     |
        +-----------------+     |
                                v
        +-------------------------------------+
        |  Output: responsibilities or        |
        |  reconstructed mixture (means)      |
        +-------------------------------------+

    :param n_components: Number of mixture components (K). Must be positive.
    :type n_components: int
    :param temperature: Softmax temperature for responsibilities. ``1.0`` yields
        the exact GMM posterior; lower values sharpen assignments. Must be positive.
        Defaults to 1.0.
    :type temperature: float
    :param isometric_regularizer_strength: Weight ``lambda`` of the isometric-kernel
        loss penalizing per-component log-variance dispersion. Must be non-negative.
        Defaults to 0.01.
    :type isometric_regularizer_strength: float
    :param variance_floor: Lower bound applied to component variances for numerical
        stability. Must be positive. Defaults to 1e-3.
    :type variance_floor: float
    :param output_mode: Output type: ``'assignments'`` for responsibilities or
        ``'mixture'`` for reconstructed inputs using component means. Defaults to
        ``'assignments'``.
    :type output_mode: str
    :param cluster_axis: Axis or axes to perform clustering on. Negative values are
        supported. Defaults to -1.
    :type cluster_axis: Union[int, List[int]]
    :param mean_initializer: Initializer for component means. Supports
        ``'orthonormal'``. Defaults to ``'orthonormal'``.
    :type mean_initializer: Union[str, keras.initializers.Initializer]
    :param log_variance_initializer: Initializer for per-dimension log-variances.
        Defaults to ``'zeros'`` (unit variance).
    :type log_variance_initializer: Union[str, keras.initializers.Initializer]
    :param mean_regularizer: Optional regularizer for component means. Defaults to None.
    :type mean_regularizer: Optional[keras.regularizers.Regularizer]
    :param random_seed: Random seed for initialization. Defaults to None.
    :type random_seed: Optional[int]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If n_components is not positive.
    :raises ValueError: If temperature is not positive.
    :raises ValueError: If isometric_regularizer_strength is negative.
    :raises ValueError: If variance_floor is not positive.
    :raises ValueError: If output_mode is not ``'assignments'`` or ``'mixture'``.
    """

    def __init__(
        self,
        n_components: int,
        temperature: float = 1.0,
        isometric_regularizer_strength: float = 0.01,
        variance_floor: float = 1e-3,
        output_mode: OutputMode = 'assignments',
        cluster_axis: Axis = -1,
        mean_initializer: Union[str, keras.initializers.Initializer] = 'orthonormal',
        log_variance_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        mean_regularizer: Optional[keras.regularizers.Regularizer] = None,
        random_seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Input validation
        self._validate_init_args(
            n_components, temperature, isometric_regularizer_strength,
            variance_floor, output_mode
        )

        # Store ALL configuration parameters
        self.n_components = n_components
        self.temperature = temperature
        self.isometric_regularizer_strength = isometric_regularizer_strength
        self.variance_floor = variance_floor
        self.output_mode = output_mode
        self.cluster_axis = [cluster_axis] if isinstance(cluster_axis, int) else list(cluster_axis)
        # DECISION plan_2026-06-14_8c7365d0/D-005: serialize the ORIGINAL (pre-build)
        # cluster_axis, not the build()-mutated positive form. build() rewrites negative
        # axes to positive against input_rank (_setup_cluster_axes), so serializing
        # self.cluster_axis would bake in a rank-specific value -> cross-rank reload picks
        # the wrong logical axis. Stash the constructor value here and emit it in get_config.
        self._cluster_axis_arg = list(self.cluster_axis)
        # DECISION plan_2026-06-08_57a975d1/D-002: do NOT replace this with a bare
        # keras.initializers.get(mean_initializer). 'orthonormal' is not a registered
        # keras alias (OrthonormalInitializer registers as Custom>OrthonormalInitializer),
        # so get('orthonormal') raises. Keep the string and let build() resolve it
        # (build handles both the string and an Initializer instance). See D-001.
        if isinstance(mean_initializer, str) and mean_initializer.lower() == 'orthonormal':
            self.mean_initializer = mean_initializer
        else:
            self.mean_initializer = keras.initializers.get(mean_initializer)
        self.log_variance_initializer = keras.initializers.get(log_variance_initializer)
        self.mean_regularizer = keras.regularizers.get(mean_regularizer)
        self.random_seed = random_seed

        # Initialize attribute placeholders - weights created in build()
        self.means: Optional[keras.Variable] = None
        self.log_variances: Optional[keras.Variable] = None
        self.mixture_logits: Optional[keras.Variable] = None
        self.input_rank: Optional[int] = None
        self.feature_dims: Optional[int] = None
        self.non_feature_dims: Optional[List[int]] = None
        self.original_shape: Optional[List[int]] = None

    def _validate_init_args(
        self,
        n_components: int,
        temperature: float,
        isometric_regularizer_strength: float,
        variance_floor: float,
        output_mode: str
    ) -> None:
        """Validate initialization arguments.

        :param n_components: Number of mixture components.
        :type n_components: int
        :param temperature: Softmax temperature.
        :type temperature: float
        :param isometric_regularizer_strength: Isometric-kernel loss weight.
        :type isometric_regularizer_strength: float
        :param variance_floor: Lower bound on component variances.
        :type variance_floor: float
        :param output_mode: Output mode string.
        :type output_mode: str
        :raises ValueError: If any argument is invalid.
        """
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError(f"n_components must be a positive integer, got {n_components}")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if isometric_regularizer_strength < 0:
            raise ValueError(
                f"isometric_regularizer_strength must be non-negative, "
                f"got {isometric_regularizer_strength}"
            )
        if not isinstance(variance_floor, (int, float)) or variance_floor <= 0:
            raise ValueError(f"variance_floor must be positive, got {variance_floor}")
        if output_mode not in ['assignments', 'mixture']:
            raise ValueError(
                f"output_mode must be 'assignments' or 'mixture', got {output_mode}"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights.

        :param input_shape: Shape of input tensor as tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is invalid or incompatible with cluster_axis.
        """
        # Store input information
        self.input_rank = len(input_shape)
        self.original_shape = list(input_shape)

        # Normalize and validate cluster axes
        self._setup_cluster_axes()

        # Compute dimensions
        self.feature_dims = self._compute_feature_dims(input_shape)
        self.non_feature_dims = self._compute_non_feature_dims()

        # Initialize component means using add_weight
        self._initialize_means()

        # Initialize per-dimension log-variances (diagonal covariance, log-space)
        self.log_variances = self.add_weight(
            name="log_variances",
            shape=(self.n_components, self.feature_dims),
            initializer=self.log_variance_initializer,
            trainable=True,
            dtype=self.dtype
        )

        # Initialize mixing logits (uniform mixture at start)
        self.mixture_logits = self.add_weight(
            name="mixture_logits",
            shape=(self.n_components,),
            initializer="zeros",
            trainable=True,
            dtype=self.dtype
        )

        # Call parent build at the end
        super().build(input_shape)

    def _setup_cluster_axes(self) -> None:
        """Setup and validate cluster axes.

        :raises ValueError: If cluster axes are invalid.
        """
        # DECISION plan_2026-06-14_7384c2e3/D-003: re-derive from the ORIGINAL constructor
        # value (_cluster_axis_arg), not in-place on self.cluster_axis. This makes build()
        # idempotent -- a second build() re-normalizes from the stable source instead of
        # double-shifting an already-positive axis (which would corrupt cluster_axis).
        # Convert negative axes to positive
        self.cluster_axis = [
            axis if axis >= 0 else self.input_rank + axis
            for axis in self._cluster_axis_arg
        ]

        # Validate axes
        if not all(0 <= axis < self.input_rank for axis in self.cluster_axis):
            raise ValueError(
                f"Invalid cluster_axis: {self.cluster_axis} for input rank {self.input_rank}"
            )

        # Sort axes for consistent processing
        self.cluster_axis.sort()

    def _compute_feature_dims(self, input_shape: Tuple[Optional[int], ...]) -> int:
        """Compute total feature dimensions.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Product of dimensions along cluster axes.
        :rtype: int
        :raises ValueError: If input shape is invalid.
        """
        try:
            return int(np.prod([input_shape[axis] for axis in self.cluster_axis]))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid input shape {input_shape} for cluster axes {self.cluster_axis}"
            ) from e

    def _compute_non_feature_dims(self) -> List[int]:
        """Compute non-feature dimensions.

        :return: List of axes not used for clustering.
        :rtype: List[int]
        """
        return [i for i in range(self.input_rank) if i not in self.cluster_axis]

    def _initialize_means(self) -> None:
        """Initialize component-mean variables with the appropriate initializer."""
        # Handle orthonormal initialization specially
        initializer_name = getattr(
            self.mean_initializer,
            '__class__',
            type(self.mean_initializer)
        ).__name__

        if (initializer_name == 'OrthonormalInitializer' or
            (isinstance(self.mean_initializer, str) and
             self.mean_initializer.lower() == 'orthonormal')):

            if self.n_components <= self.feature_dims:
                initializer = OrthonormalInitializer(seed=self.random_seed)
            else:
                logger.warning(
                    f"n_components ({self.n_components}) > feature_dims ({self.feature_dims}), "
                    "falling back to glorot_normal initializer"
                )
                initializer = keras.initializers.GlorotNormal(seed=self.random_seed)
        else:
            initializer = self.mean_initializer

        # Create means weight
        self.means = self.add_weight(
            name="means",
            shape=(self.n_components, self.feature_dims),
            initializer=initializer,
            regularizer=self.mean_regularizer,
            trainable=True,
            dtype=self.dtype
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute shape of layer output.

        :param input_shape: Shape of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output tensor shape.
        :rtype: Tuple[Optional[int], ...]
        """
        if self.output_mode == 'assignments':
            # Normalize axes LOCALLY from the original constructor value + input rank
            # rather than reading self.cluster_axis (which build() mutates negative->
            # positive and which may not be normalized yet during functional-API tracing,
            # when compute_output_shape is called BEFORE build). Mirrors _setup_cluster_axes.
            rank = len(input_shape)
            axes = sorted(
                ax if ax >= 0 else rank + ax for ax in self._cluster_axis_arg
            )
            output_shape = list(input_shape)

            # Handle multiple clustering axes
            if len(axes) > 1:
                # Replace clustered dimensions with n_components
                # Remove extra axes in reverse order to preserve indices
                for axis in reversed(axes[1:]):
                    output_shape.pop(axis)
                output_shape[axes[0]] = self.n_components
            else:
                output_shape[axes[0]] = self.n_components

            return tuple(output_shape)

        # For mixture mode, output shape matches input
        return tuple(input_shape)

    def _reshape_for_clustering(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape input tensor for clustering operations.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Reshaped tensor with shape ``(batch * non_feature_dims, feature_dims)``.
        :rtype: keras.KerasTensor
        """
        # Optimize for common case of single axis at end
        if len(self.cluster_axis) == 1 and self.cluster_axis[0] == self.input_rank - 1:
            return keras.ops.reshape(inputs, [-1, self.feature_dims])

        # General case requires transpose
        perm = self.non_feature_dims + self.cluster_axis
        transposed = keras.ops.transpose(inputs, perm)
        return keras.ops.reshape(transposed, [-1, self.feature_dims])

    def _effective_variances(self) -> keras.KerasTensor:
        """Compute floored component variances from log-variance parameters.

        :return: Variance tensor of shape ``(n_components, feature_dims)``.
        :rtype: keras.KerasTensor
        """
        return keras.ops.maximum(
            keras.ops.exp(self.log_variances),
            self.variance_floor
        )

    def _log_gaussian_density(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Compute diagonal-Gaussian log-densities of inputs under each component.

        :param inputs: Input tensor of shape ``(batch, features)``.
        :type inputs: keras.KerasTensor
        :return: Log-density tensor of shape ``(batch, n_components)``.
        :rtype: keras.KerasTensor
        """
        # Floored variances and matching log-determinant term
        variances = self._effective_variances()  # (n_components, features)
        log_variances = keras.ops.log(variances)  # (n_components, features)

        # Broadcast: (batch, 1, features) against (1, n_components, features)
        expanded_inputs = keras.ops.expand_dims(inputs, axis=1)
        expanded_means = keras.ops.expand_dims(self.means, axis=0)
        expanded_variances = keras.ops.expand_dims(variances, axis=0)

        # Mahalanobis term: sum_d (x - mu)^2 / var  ->  (batch, n_components)
        squared_diff = keras.ops.square(expanded_inputs - expanded_means)
        mahalanobis = keras.ops.sum(squared_diff / expanded_variances, axis=-1)

        # Log-determinant term: sum_d log var  ->  (n_components,)
        log_det = keras.ops.sum(log_variances, axis=-1)

        # Normalization constant D * log(2*pi)
        norm_const = float(self.feature_dims) * float(np.log(2.0 * np.pi))

        # log N(x | mu, Sigma) = -0.5 * (mahalanobis + log_det + D log(2*pi))
        log_density = -0.5 * (mahalanobis + keras.ops.expand_dims(log_det, axis=0) + norm_const)

        return log_density

    def _responsibilities(self, log_density: keras.KerasTensor) -> keras.KerasTensor:
        """Compute temperature-scaled posterior responsibilities.

        :param log_density: Log-density tensor of shape ``(batch, n_components)``.
        :type log_density: keras.KerasTensor
        :return: Responsibility probabilities of shape ``(batch, n_components)``.
        :rtype: keras.KerasTensor
        """
        # Log mixing weights via stable log-softmax  ->  (n_components,)
        log_mixing = keras.ops.log_softmax(self.mixture_logits, axis=-1)

        # Joint log-probability: log pi_k + log N(x | k)  ->  (batch, n_components)
        log_joint = keras.ops.expand_dims(log_mixing, axis=0) + log_density

        # Temperature-scaled posterior (exact GMM E-step at temperature == 1)
        return keras.ops.softmax(log_joint / self.temperature, axis=-1)

    def _isometric_regularization_loss(self) -> keras.KerasTensor:
        """Compute the isometric-kernel regularization loss.

        Penalizes the per-component dispersion of log-variances across feature
        dimensions; minimizing it drives each covariance toward an isotropic
        (spherical) form ``Sigma_k = sigma_k^2 * I``.

        :return: Scalar regularization loss.
        :rtype: keras.KerasTensor
        """
        log_variances = keras.ops.log(self._effective_variances())  # (n_components, features)

        # Per-component mean log-variance  ->  (n_components, 1)
        mean_log_variance = keras.ops.mean(log_variances, axis=-1, keepdims=True)

        # Mean squared deviation from the per-component mean (anisotropy measure)
        anisotropy = keras.ops.mean(
            keras.ops.square(log_variances - mean_log_variance)
        )

        return self.isometric_regularizer_strength * anisotropy

    def _reshape_output(self, output: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape clustering output to match desired output shape.

        :param output: Output tensor from clustering.
        :type output: keras.KerasTensor
        :return: Reshaped output tensor.
        :rtype: keras.KerasTensor
        """
        if self.output_mode == 'assignments':
            output_shape = list(self.original_shape)

            # Handle multiple clustering axes
            if len(self.cluster_axis) > 1:
                # Remove extra axes in reverse order to preserve indices
                for axis in reversed(self.cluster_axis[1:]):
                    output_shape.pop(axis)
                output_shape[self.cluster_axis[0]] = self.n_components
            else:
                output_shape[self.cluster_axis[0]] = self.n_components

            # Set batch dimension to -1 for dynamic reshaping
            output_shape[0] = -1

        else:  # output_mode == 'mixture'
            output_shape = list(self.original_shape)
            output_shape[0] = -1

        return keras.ops.reshape(output, output_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass performing differentiable GMM soft clustering.

        :param inputs: Input tensor with arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode. The isometric-kernel
            regularization loss is registered during training.
        :type training: Optional[bool]
        :return: Output tensor based on output_mode.
        :rtype: keras.KerasTensor
        """
        # Cast inputs to layer dtype for numerical stability
        inputs = keras.ops.cast(inputs, self.dtype)

        # Reshape input for clustering
        reshaped_inputs = self._reshape_for_clustering(inputs)

        # Compute log-densities and posterior responsibilities
        log_density = self._log_gaussian_density(reshaped_inputs)
        responsibilities = self._responsibilities(log_density)

        # Register isometric-kernel regularization during training.
        # ``training is True``: graph-safe identity check (symbolic/None/False skip the
        # add_loss without coercing a tensor to bool). Canonical repo idiom.
        if training is True and self.isometric_regularizer_strength > 0:
            self.add_loss(self._isometric_regularization_loss())

        # Compute output based on mode
        if self.output_mode == 'assignments':
            output = responsibilities
        else:  # output_mode == 'mixture'
            # Reconstruct inputs as responsibility-weighted component means
            output = keras.ops.matmul(responsibilities, self.means)

        # Reshape output to match desired shape
        return self._reshape_output(output)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "n_components": self.n_components,
            "temperature": self.temperature,
            "isometric_regularizer_strength": self.isometric_regularizer_strength,
            "variance_floor": self.variance_floor,
            "output_mode": self.output_mode,
            # DECISION plan_2026-06-14_8c7365d0/D-005: serialize the pre-build axis.
            "cluster_axis": self._cluster_axis_arg,
            "mean_initializer": (
                self.mean_initializer if isinstance(self.mean_initializer, str)
                else keras.initializers.serialize(self.mean_initializer)
            ),
            "log_variance_initializer": keras.initializers.serialize(self.log_variance_initializer),
            "mean_regularizer": keras.regularizers.serialize(self.mean_regularizer),
            "random_seed": self.random_seed
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GMMLayer":
        """Create a layer instance from its serialized configuration.

        :param config: Configuration dictionary produced by ``get_config``.
        :type config: Dict[str, Any]
        :return: Reconstructed layer instance.
        :rtype: GMMLayer
        """
        config = dict(config)
        if "mean_initializer" in config and not isinstance(config["mean_initializer"], str):
            config["mean_initializer"] = keras.initializers.deserialize(config["mean_initializer"])
        if "log_variance_initializer" in config:
            config["log_variance_initializer"] = keras.initializers.deserialize(
                config["log_variance_initializer"]
            )
        if "mean_regularizer" in config:
            config["mean_regularizer"] = keras.regularizers.deserialize(config["mean_regularizer"])
        return cls(**config)

    @property
    def component_means(self) -> Optional[keras.KerasTensor]:
        """Get current component means.

        :return: Tensor of shape ``(n_components, feature_dims)`` or None if not built.
        :rtype: Optional[keras.KerasTensor]
        """
        return self.means

    @property
    def component_variances(self) -> Optional[keras.KerasTensor]:
        """Get current (floored) component diagonal variances.

        :return: Tensor of shape ``(n_components, feature_dims)`` or None if not built.
        :rtype: Optional[keras.KerasTensor]
        """
        if self.log_variances is None:
            return None
        return self._effective_variances()

    @property
    def mixture_weights(self) -> Optional[keras.KerasTensor]:
        """Get current normalized mixing weights.

        :return: Tensor of shape ``(n_components,)`` or None if not built.
        :rtype: Optional[keras.KerasTensor]
        """
        if self.mixture_logits is None:
            return None
        return keras.ops.softmax(self.mixture_logits, axis=-1)

    def reset_parameters(
        self,
        new_means: Optional[keras.KerasTensor] = None
    ) -> None:
        """Reset mixture parameters to new values or reinitialize.

        :param new_means: Optional tensor of shape ``(n_components, feature_dims)``.
            If None, means are reinitialized with small random values.
        :type new_means: Optional[keras.KerasTensor]
        :raises ValueError: If new_means has wrong shape or layer is not built.
        """
        if not self.built:
            raise ValueError("Layer must be built before resetting parameters")

        if new_means is not None:
            expected_shape = (self.n_components, self.feature_dims)
            if tuple(new_means.shape) != expected_shape:
                raise ValueError(
                    f"new_means must have shape {expected_shape}, "
                    f"got {tuple(new_means.shape)}"
                )
            self.means.assign(new_means)
        else:
            # Generate fresh random values to ensure distinct means
            new_values = keras.random.normal(
                shape=(self.n_components, self.feature_dims),
                dtype=self.dtype,
                seed=self.random_seed
            ) * 0.1  # Small scale for stability
            self.means.assign(new_values)

        # Reset covariances to unit variance and mixing weights to uniform
        self.log_variances.assign(keras.ops.zeros_like(self.log_variances))
        self.mixture_logits.assign(keras.ops.zeros_like(self.mixture_logits))

# ---------------------------------------------------------------------
