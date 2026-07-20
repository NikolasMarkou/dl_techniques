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
from ...utils.tensors import resolve_training_factor
from ...initializers.orthonormal_initializer import OrthonormalInitializer
from .base import BaseMixtureLayer

# ---------------------------------------------------------------------

# Type aliases for better readability
OutputMode = Literal['assignments', 'mixture']
CovarianceType = Literal['diagonal', 'low_rank']
TensorShape = Union[Tuple[int, ...], List[int]]
Axis = Union[int, List[int]]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GMMLayer(BaseMixtureLayer):
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

        ┌─────────────────────────────────────┐
        │   Input (arbitrary shape)           │
        │   Cast to variable_dtype (float32)  │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  Reshape for clustering             │
        │  (flatten cluster_axis dims)        │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  Diagonal-Gaussian log-densities    │
        │  log N(x | mu_k, Sigma_k)           │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  Responsibilities:                  │
        │  softmax((log pi + log N) / tau)    │
        └────────┬──────────────┬─────────────┘
                 │              │ (training only)
                 │              ▼
                 │    ┌──────────────────────┐
                 │    │  add_loss:           │
                 │    │  isometric reg.      │
                 │    └──────────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────────┐
        │  Compute output                     │
        │  (assignments or mixture)           │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  Reshape output                     │
        │  Cast to compute_dtype              │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  Output: assignments or mixture     │
        └─────────────────────────────────────┘

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
    :param covariance_type: Covariance parameterization. ``'diagonal'`` (the default)
        uses ``Sigma_k = diag(var_k)``, exactly as before. ``'low_rank'`` uses the
        low-rank-plus-diagonal **factor-analysis** form
        ``Sigma_k = diag(var_k) + U_k U_k^T``, where ``U_k`` is the additional
        ``covariance_factors`` weight of shape ``(K, D, covariance_rank)``. Only
        ``'low_rank'`` can represent correlated features. Note the diagonal
        ``var_k`` is free per-dimension, which is what makes this factor analysis
        rather than probabilistic PCA (PPCA is the constrained special case
        ``var_k = sigma_k^2 * 1``, which this layer does not enforce).
    :type covariance_type: str
    :param covariance_rank: Rank ``R`` of the low-rank factor ``U_k``. Must be a
        positive integer. Ignored (but still serialized) when
        ``covariance_type='diagonal'``. Defaults to 1.
    :type covariance_rank: int
    :param factor_initializer: Initializer for the low-rank factor ``U_k``. Defaults
        to ``'glorot_uniform'``. Note that ``'zeros'`` makes ``U_k`` a dead weight:
        the Woodbury correction is quadratic in ``U_k`` near zero, so its gradient
        vanishes identically at ``U_k = 0``.
    :type factor_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If n_components is not positive.
    :raises ValueError: If temperature is not positive.
    :raises ValueError: If isometric_regularizer_strength is negative.
    :raises ValueError: If variance_floor is not positive.
    :raises ValueError: If output_mode is not ``'assignments'`` or ``'mixture'``.
    :raises ValueError: If covariance_type is not ``'diagonal'`` or ``'low_rank'``.
    :raises ValueError: If covariance_rank is not a positive integer.
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
        covariance_type: CovarianceType = 'diagonal',
        covariance_rank: int = 1,
        factor_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Input validation
        self._validate_init_args(
            n_components, temperature, isometric_regularizer_strength,
            variance_floor, output_mode, covariance_type, covariance_rank
        )

        # Store ALL configuration parameters
        self.n_components = n_components
        self.temperature = temperature
        self.isometric_regularizer_strength = isometric_regularizer_strength
        self.variance_floor = variance_floor
        self.output_mode = output_mode
        self.covariance_type = covariance_type
        self.covariance_rank = covariance_rank
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
        self.factor_initializer = keras.initializers.get(factor_initializer)
        self.mean_regularizer = keras.regularizers.get(mean_regularizer)
        self.random_seed = random_seed

        # DECISION plan-2026-07-20T141712-e03557c8/D-002: announce the reframed regularizer
        # semantics instead of silently changing a default. Do NOT "fix" this by making
        # isometric_regularizer_strength default to 0.0 when covariance_type is non-
        # diagonal -- deriving one kwarg's default from another kwarg's VALUE is the
        # exact untraceable-bug shape rejected as Option C. Severity is info, not
        # warning: under the factor-analysis reading (Sigma = diag(d) + U U^T) the
        # combination is legitimate -- the diagonal/low-rank split is non-identifiable,
        # so penalizing the diagonal's anisotropy pushes anisotropy into U, where the
        # model can represent it. Warning here would cry wolf on a default-valued config.
        if self.covariance_type != 'diagonal' and self.isometric_regularizer_strength > 0:
            logger.info(
                f"GMMLayer: covariance_type='{self.covariance_type}' with "
                f"isometric_regularizer_strength={self.isometric_regularizer_strength} > 0. "
                "The isometric-kernel regularizer reads log_variances only, so under a "
                "non-diagonal covariance it regularizes the RESIDUAL DIAGONAL toward "
                "isotropy, NOT the total covariance -- the low-rank factor stays free to "
                "carry anisotropy. Under the factor-analysis structure "
                "Sigma = diag(d) + U U^T the diagonal/low-rank split is non-identifiable, "
                "so this pushes anisotropy into U by construction rather than removing it. "
                "Pass isometric_regularizer_strength=0.0 to disable it entirely."
            )

        # Initialize attribute placeholders - weights created in build()
        self.means: Optional[keras.Variable] = None
        self.log_variances: Optional[keras.Variable] = None
        self.mixture_logits: Optional[keras.Variable] = None
        self.covariance_factors: Optional[keras.Variable] = None
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
        output_mode: str,
        covariance_type: str,
        covariance_rank: int
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
        :param covariance_type: Covariance parameterization string.
        :type covariance_type: str
        :param covariance_rank: Rank of the low-rank covariance factor.
        :type covariance_rank: int
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
        if covariance_type not in ['diagonal', 'low_rank']:
            raise ValueError(
                f"covariance_type must be 'diagonal' or 'low_rank', got {covariance_type}"
            )
        if not isinstance(covariance_rank, int) or isinstance(covariance_rank, bool) \
                or covariance_rank < 1:
            raise ValueError(
                f"covariance_rank must be a positive integer, got {covariance_rank}"
            )

    # DECISION plan-2026-07-20T141712-e03557c8/D-007: this property is a pure NAMING seam
    # (self.n_components vs KMeansLayer's self.n_clusters), not a semantic merge. Do NOT
    # rename it to a shared public attribute -- that would break get_config() keys, the
    # registry params, and the byte-unchanged __init__ signature requirement (I2/A5).
    @property
    def _n_prototypes(self) -> int:
        """Prototype count seam read by ``_ClusterAxisMixin`` (see BaseMixtureLayer).

        :return: Number of mixture components.
        :rtype: int
        """
        return self.n_components

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
            dtype=self.dtype,
            autocast=False  # mixed-precision: keep float32 for the density math
        )

        # Initialize mixing logits (uniform mixture at start)
        self.mixture_logits = self.add_weight(
            name="mixture_logits",
            shape=(self.n_components,),
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
            autocast=False  # mixed-precision: keep float32 for the density math
        )

        # Low-rank covariance factor U_k, created ONLY for covariance_type='low_rank'.
        # Under 'diagonal' the weight must not exist at all, so existing checkpoints
        # and weight counts stay byte-compatible (invariant I1).
        if self.covariance_type == 'low_rank':
            if self.covariance_rank >= self.feature_dims:
                logger.warning(
                    f"covariance_rank ({self.covariance_rank}) >= feature_dims "
                    f"({self.feature_dims}); the low-rank parameterization is still valid "
                    "but no longer economical -- it costs more parameters than a dense "
                    "covariance would."
                )
            self.covariance_factors = self.add_weight(
                name="covariance_factors",
                shape=(self.n_components, self.feature_dims, self.covariance_rank),
                initializer=self.factor_initializer,
                trainable=True,
                dtype=self.dtype,
                autocast=False  # mixed-precision: matrix solves are MORE fp16-fragile
            )

        # Call parent build at the end
        super().build(input_shape)

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

        # Create means weight.
        # Mixed-precision: autocast=False keeps the parameter in variable_dtype (float32)
        # inside call() under a mixed_float16 policy. The diagonal-Gaussian density
        # (exp/log/softmax/division) is numerically unsafe in float16, so the forward is
        # computed in float32 and the OUTPUT is cast to compute_dtype on return. Without
        # this, the autocast float16 weight mismatches the float32 inputs
        # (InvalidArgumentError: Sub half vs float).
        self.means = self.add_weight(
            name="means",
            shape=(self.n_components, self.feature_dims),
            initializer=initializer,
            regularizer=self.mean_regularizer,
            trainable=True,
            dtype=self.dtype,
            autocast=False
        )

    def _effective_variances(self) -> keras.KerasTensor:
        """Compute floored component variances from log-variance parameters.

        :return: Variance tensor of shape ``(n_components, feature_dims)``.
        :rtype: keras.KerasTensor
        """
        return keras.ops.maximum(
            keras.ops.exp(self.log_variances),
            self.variance_floor
        )

    def _assemble_log_density(
        self,
        mahalanobis: keras.KerasTensor,
        log_det: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Combine a Mahalanobis term and a log-determinant into a Gaussian log-density.

        Shared by both covariance parameterizations, which differ only in HOW the two
        inputs are computed -- the final assembly and the ``D log(2*pi)`` normalization
        constant are identical for any ``Sigma``.

        :param mahalanobis: Quadratic form ``(x-mu)^T Sigma^-1 (x-mu)``, shape
            ``(batch, n_components)``.
        :type mahalanobis: keras.KerasTensor
        :param log_det: Per-component ``log det(Sigma_k)``, shape ``(n_components,)``.
        :type log_det: keras.KerasTensor
        :return: Log-density tensor of shape ``(batch, n_components)``.
        :rtype: keras.KerasTensor
        """
        # Normalization constant D * log(2*pi)
        norm_const = float(self.feature_dims) * float(np.log(2.0 * np.pi))

        # log N(x | mu, Sigma) = -0.5 * (mahalanobis + log_det + D log(2*pi))
        return -0.5 * (mahalanobis + keras.ops.expand_dims(log_det, axis=0) + norm_const)

    def _log_gaussian_density(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Compute Gaussian log-densities of inputs under each component.

        Dispatches on ``self.covariance_type``, a plain Python string fixed at
        construction time. This is a trace-time branch on configuration, not a branch
        on a tensor VALUE, so it is graph-safe.

        :param inputs: Input tensor of shape ``(batch, features)``.
        :type inputs: keras.KerasTensor
        :return: Log-density tensor of shape ``(batch, n_components)``.
        :rtype: keras.KerasTensor
        """
        if self.covariance_type == 'low_rank':
            return self._log_gaussian_density_low_rank(inputs)
        return self._log_gaussian_density_diagonal(inputs)

    def _log_gaussian_density_diagonal(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
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

        return self._assemble_log_density(mahalanobis, log_det)

    def _log_gaussian_density_low_rank(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Compute low-rank-plus-diagonal Gaussian log-densities under each component.

        Implements the factor-analysis covariance ``Sigma_k = diag(d_k) + U_k U_k^T``
        (``d_k`` free per-dimension, NOT the isotropic PPCA special case
        ``d_k = sigma_k^2 * 1``) via the matrix determinant
        lemma and the Woodbury identity, so nothing of size ``(D, D)`` is ever
        materialized. With ``M_k = I_R + U_k^T diag(1/d_k) U_k`` (the ``(R, R)``
        capacitance matrix):

        - ``log det(Sigma_k) = sum_d log d_kd + log det(M_k)``
        - ``(x-mu)^T Sigma_k^-1 (x-mu) = z^T (x-mu) - ||M_k^-1/2 U_k^T z||^2``,
          where ``z = (x-mu) / d_k``.

        # DECISION plan-2026-07-20T141712-e03557c8/D-003: derive BOTH quantities from a single
        # keras.ops.cholesky(M_k). Do NOT switch this to keras.ops.slogdet: (a) slogdet's
        # `sign` output silently yields a plausible-but-wrong log-density if Sigma is ever
        # non-PD, and (b) its batching over a leading axis works empirically here while its
        # own docstring says the input "must be 2D and square" -- an undocumented behavior.
        # Cholesky is safe by construction: variance_floor keeps d_k > 0, so
        # M_k = I + PSD is always PD (invariant I6) and the sign question cannot arise.
        # The correct symbol is keras.ops.linalg.solve_triangular; a top-level
        # keras.ops.triangular_solve does NOT exist.
        # Its batched RHS convention -- a (K,R,R) factor against a (K,R,B) right-hand side
        # with a LEADING batch axis -- is why `w` is transposed to (K,R,B) and back. That
        # convention was the one unverified claim in the plan (assumption A2) and was
        # confirmed empirically before this code was written; a wrong convention would NOT
        # raise, it would silently return a wrong density. The dense-oracle test
        # (test_low_rank_matches_dense_reference) is the standing guard on it -- if that
        # test ever fails, do NOT loosen its tolerance.

        :param inputs: Input tensor of shape ``(batch, features)``.
        :type inputs: keras.KerasTensor
        :return: Log-density tensor of shape ``(batch, n_components)``.
        :rtype: keras.KerasTensor
        """
        ops = keras.ops

        variances = self._effective_variances()                  # (K, D), floored > 0
        factors = self.covariance_factors                        # (K, D, R)

        # Centered inputs: (batch, 1, D) - (1, K, D)  ->  (batch, K, D)
        diff = ops.expand_dims(inputs, axis=1) - ops.expand_dims(self.means, axis=0)

        # z = diag(1/d) (x - mu)  ->  (batch, K, D)
        z = diff / ops.expand_dims(variances, axis=0)

        # Diagonal part of the quadratic form: (x-mu)^T diag(1/d) (x-mu)  ->  (batch, K)
        diagonal_quadratic = ops.sum(z * diff, axis=-1)

        # Capacitance matrix M = I_R + U^T diag(1/d) U  ->  (K, R, R). Always PD.
        scaled_factors = factors / ops.expand_dims(variances, axis=-1)   # (K, D, R)
        capacitance = ops.eye(
            self.covariance_rank, dtype=self.variable_dtype
        ) + ops.matmul(ops.transpose(factors, (0, 2, 1)), scaled_factors)

        # Single factorization serving BOTH the log-determinant and the quadratic form.
        cholesky_factor = ops.cholesky(capacitance)               # (K, R, R), lower

        # Woodbury correction: w = U^T z  ->  (batch, K, R)
        projected = ops.einsum('bkd,kdr->bkr', z, factors)

        # solve_triangular wants a leading batch axis on BOTH operands: (K,R,R) vs (K,R,B).
        solved = ops.linalg.solve_triangular(
            cholesky_factor, ops.transpose(projected, (1, 2, 0)), lower=True
        )                                                          # (K, R, batch)
        correction = ops.transpose(ops.sum(ops.square(solved), axis=1))   # (batch, K)

        mahalanobis = diagonal_quadratic - correction              # (batch, K)

        # log det(Sigma) = sum_d log d + 2 * sum_r log L_rr  ->  (K,)
        log_det = ops.sum(ops.log(variances), axis=-1) + 2.0 * ops.sum(
            ops.log(ops.diagonal(cholesky_factor, axis1=-2, axis2=-1)), axis=-1
        )

        return self._assemble_log_density(mahalanobis, log_det)

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
        dimensions. What that dispersion MEASURES depends on ``covariance_type``:

        - **``'diagonal'``** -- ``log_variances`` fully determines ``Sigma_k``, so
          minimizing this loss drives the TOTAL covariance toward an isotropic
          (spherical) form ``Sigma_k = sigma_k^2 * I``.
        - **``'low_rank'``** -- ``Sigma_k = diag(d_k) + U_k U_k^T``, and this loss
          reads ``d_k`` only. It therefore regularizes the RESIDUAL DIAGONAL toward
          isotropy and leaves ``U_k`` free to carry anisotropy. Total-covariance
          anisotropy is NOT regularized under this mode.

        Under ``'low_rank'`` the structure is **factor analysis** -- ``d_k`` is a free
        per-dimension diagonal, not a scalar multiple of the identity. (Probabilistic
        PCA is the constrained special case ``d_k = sigma_k^2 * 1``; this layer cannot
        produce it, because this loss only *softly* penalizes the dispersion of
        ``log d_k`` and never enforces isotropy.) That distinction is what makes the
        regularizer and the low-rank factor complementary **by construction** rather
        than by analogy: in factor analysis the split between the diagonal term and the
        low-rank term is **non-identifiable** -- the same ``Sigma_k`` can be expressed
        with anisotropy residing in either term. Penalizing the diagonal's anisotropy
        therefore does not remove anisotropy from the model; it *relocates* it into
        ``U_k``, which is precisely the term with the capacity to represent it.

        # DECISION plan-2026-07-20T141712-e03557c8/D-002: the loss BODY is deliberately
        # identical under both modes -- only this docstring and the __init__ notice
        # distinguish them. Do NOT "fix" the low-rank case by adding a term over
        # U_k's Gram spectrum (the semantically complete Option B): that needs an
        # eigendecomposition or SVD per component per step, a new numerical-stability
        # surface, and a second coefficient -- a permanent complexity charge levied to
        # solve a problem no caller has. Do NOT instead zero the default strength when
        # covariance_type is non-diagonal (Option C); see the __init__ anchor. Do NOT
        # describe this structure as PPCA (Sigma = sigma^2 I + U U^T) -- that was the
        # original, WRONG justification, corrected at step 6. d_k is free per-dimension,
        # so the family is FACTOR ANALYSIS; PPCA is the isotropic-diagonal special case
        # this layer never reaches, since the loss only softly penalizes dispersion of
        # log d_k at default strength 0.01. The complementarity argument does not rest
        # on the PPCA analogy and survives without it: in factor analysis the
        # diagonal/low-rank split is non-identifiable, so penalizing the diagonal's
        # anisotropy PUSHES anisotropy into U rather than removing it from the model.
        # Complementary by construction, not by resemblance to a related model.
        # NOTE the pre-registered falsification (covariance_factors gradients suppressed
        # >10x versus a strength=0.0 control) is TAUTOLOGICAL and cannot fire: this loss
        # reads log_variances only, so d(iso)/dU is a structural zero. See D-006. The
        # standing check is a regression guard against a future Option-B edit, NOT
        # evidence for this reframe.

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
        # Cast inputs to variable_dtype (float32) so the density math runs in full
        # precision and matches the autocast=False weights under a mixed_float16 policy.
        # The output is cast back to compute_dtype before returning. Under the default
        # float32 policy this is a no-op.
        inputs = keras.ops.cast(inputs, self.variable_dtype)

        # Reshape input for clustering
        reshaped_inputs = self._reshape_for_clustering(inputs)

        # Compute log-densities and posterior responsibilities
        log_density = self._log_gaussian_density(reshaped_inputs)
        responsibilities = self._responsibilities(log_density)

        # Register isometric-kernel regularization during training.
        # DECISION plan_2026-06-14_5e80bd3e/D-001: gate on a graph-safe training factor so
        # the loss fires for a symbolic training=True tensor (custom @tf.function loop) and
        # is a zero contribution under symbolic-False, never coercing a tensor to a bool.
        # python-True keeps the exact unmasked add_loss; the symbolic path multiplies by the
        # 0/1 factor.
        if self.isometric_regularizer_strength > 0:
            # variable_dtype factor so the masked loss term stays float32-consistent
            # under a mixed_float16 policy.
            training_factor = resolve_training_factor(training, self.variable_dtype)
            if training_factor is not None:
                loss = self._isometric_regularization_loss()
                self.add_loss(
                    loss if isinstance(training_factor, float)
                    else training_factor * loss
                )

        # Compute output based on mode
        if self.output_mode == 'assignments':
            output = responsibilities
        else:  # output_mode == 'mixture'
            # Reconstruct inputs as responsibility-weighted component means
            output = keras.ops.matmul(responsibilities, self.means)

        # Reshape, then cast to compute_dtype so the layer emits the policy's compute
        # dtype (float16 under mixed precision; no-op under float32).
        return keras.ops.cast(self._reshape_output(output), self.compute_dtype)

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
            "random_seed": self.random_seed,
            "covariance_type": self.covariance_type,
            "covariance_rank": self.covariance_rank,
            "factor_initializer": keras.initializers.serialize(self.factor_initializer),
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
        if "factor_initializer" in config:
            config["factor_initializer"] = keras.initializers.deserialize(
                config["factor_initializer"]
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

        Restores every trainable parameter to its initial state:

        - ``means`` — set to ``new_means`` if given, else small random values.
        - ``log_variances`` — zeroed, i.e. unit variances (isotropic).
        - ``mixture_logits`` — zeroed, i.e. uniform mixing weights.
        - ``covariance_factors`` — under ``covariance_type='low_rank'`` ONLY,
          re-drawn from ``factor_initializer``. Absent under ``'diagonal'``,
          where the weight does not exist.

        .. note::
           Under ``'low_rank'`` this method is **not deterministic** unless
           ``factor_initializer`` carries a seed, because re-running the
           initializer draws fresh values. This matches how the weight was
           created in :meth:`build`.

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

        # DECISION plan-2026-07-20T141712-e03557c8/D-005: re-RUN factor_initializer here.
        # Do NOT "simplify" this to zeros_like(self.covariance_factors) to match the two
        # lines above. U = 0 makes the Woodbury correction quadratic in U about the
        # origin, so dL/dU vanishes identically and the weight is permanently untrainable
        # -- the same reason factor_initializer defaults to 'glorot_uniform' and not
        # 'zeros'. "Reset" means "return to the layer's INITIAL state", and the initial
        # state of this weight is the initializer's output, not the additive identity.
        if self.covariance_factors is not None:
            self.covariance_factors.assign(
                self.factor_initializer(
                    shape=(self.n_components, self.feature_dims, self.covariance_rank),
                    dtype=self.covariance_factors.dtype,
                )
            )

# ---------------------------------------------------------------------
