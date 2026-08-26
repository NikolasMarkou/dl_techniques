"""
A differentiable K-means clustering layer for deep networks.

This layer embeds a clustering mechanism directly into a neural network as an
**EMA codebook with differentiable assignments** — the VQ-VAE-EMA scheme. Hard
assignment is replaced by a temperature-softmax, so the layer's OUTPUT is
differentiable and gradients flow back into the inputs (and thus into the
encoder before it). The centroids themselves are NOT gradient-trained: they are
``trainable=False`` and are moved only by the in-``call`` momentum/EMA update,
which no optimizer and no ``add_loss`` term touches.

The design incorporates several modern techniques to ensure stable and
effective training within a deep learning context, such as soft assignments,
momentum-based updates, and a novel centroid repulsion mechanism.

Architecture and Core Concepts:

The core of the layer's differentiability lies in its use of "soft
assignments." Instead of assigning each input vector to the single closest
centroid (a non-differentiable `argmin` operation), the layer computes a
probability distribution over all centroids.

Key mechanisms include:

1.  **Soft Assignments:** The layer calculates the squared Euclidean distance
    from an input vector to each of the `K` centroids. These distances are
    then passed through a temperature-controlled softmax function. The
    `temperature` parameter controls the "softness" of the assignment: lower
    temperatures produce sharper, more confident distributions (approaching a
    one-hot encoding), while higher temperatures result in smoother, more
    uncertain assignments.

2.  **EMA Centroid Updates:** During training the centroids are moved toward
    the assignment-weighted average of the batch's input vectors, via an
    exponential moving average applied in-place inside ``call``. This is a
    stochastic, gradient-free stand-in for the discrete re-averaging step of
    standard K-means; it is not backpropagated through, and a centroid that
    owns no mass in the batch is held still (see ``_MIN_CLUSTER_MASS``).

3.  **Momentum and Repulsion:** To stabilize training, the layer includes two
    additional forces. A momentum term smooths the centroid updates over
    time, preventing drastic oscillations. More importantly, a "repulsion
    force" is applied between centroids. This force actively pushes centroids
    apart if their pairwise distance falls below a predefined threshold,
    counteracting the common failure mode of "centroid collapse" where
    multiple centroids converge to the same point in the feature space. This
    encourages the centroids to span the data manifold more effectively.

Mathematical Foundation:

    The soft assignment probability `a_ij` of an input vector `x_i` to a
    centroid `c_j` is calculated as:
    `a_ij = softmax(-||x_i - c_j||² / τ)_j`
    where `τ` is the temperature.

    The per-step delta for a centroid `c_j` combines a mass-gated data pull with a
    repulsive push from the other centroids:
    `u_j = alive_j * ( (Σ_i a_ij * x_i) / (Σ_i a_ij) - c_j ) + Σ_{k≠j} Repel(c_j, c_k)`
    The COMBINED delta — repulsion included — is then smoothed by the momentum
    buffer, and only the smoothed value is applied. See the class docstring for the
    exact three-line form.

References:

    This layer's design synthesizes ideas from the broader field of deep
    clustering and representation learning. While K-means is a classical
    algorithm, its integration into neural networks in a differentiable manner is
    a more recent development.

    -   The concept of soft assignments is related to fuzzy c-means clustering
        and is a common technique in differentiable clustering.
    -   The end-to-end learning of a "codebook" or dictionary of centroids is a
        central idea in methods like Vector-Quantized Variational Autoencoders
        (VQ-VAE), as introduced by van den Oord, A., et al. (2017).
    -   The use of repulsion or other diversity-promoting regularizers on the
        centroids is a technique employed to prevent codebook collapse in such
        models.

"""

import keras
from typing import Optional, Union, Any, Tuple, Dict, ClassVar, FrozenSet

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...utils.logger import logger
from ...utils.tensors import resolve_training_factor, pairwise_squared_distance
from .base import (
    Axis,
    BaseMixtureLayer,
    OutputMode,
    resolve_initializer_arg,
    resolve_prototype_initializer,
)

# ---------------------------------------------------------------------

# ``OutputMode`` and ``Axis`` are shared with ``GMMLayer`` and are imported from
# ``base.py`` above.

# Soft assignments never give a cluster EXACTLY zero mass, so "dead" needs a threshold.
# Mass is a sum of responsibilities in units of whole data points, so 1e-3 reads as
# "this cluster owns less than a thousandth of one point's vote" -- an absolute floor,
# deliberately NOT scaled by batch size, so the same centroid is judged alive or dead
# identically at batch 8 and at batch 8192.
_MIN_CLUSTER_MASS = 1e-3

# Scale of the fresh normal draw used by `reset_centroids()` when no explicit
# centroids are supplied. Small on purpose: a reset seeds centroids near the origin
# and lets the EMA update carry them out to the data, rather than starting them at
# unit scale in a direction the data may not occupy.
_RESET_CENTROID_SCALE = 0.1

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class KMeansLayer(BaseMixtureLayer):
    """Differentiable K-means layer with momentum and centroid repulsion.

    An EMA codebook with differentiable assignments. The soft assignment
    probability of input ``x_i`` to centroid ``c_j`` is
    ``a_ij = softmax(-||x_i - c_j||^2 / tau)_j`` — differentiable, so gradients
    reach the inputs through the output. The centroids are ``trainable=False``
    and are moved only by this in-place update, run inside ``call`` when
    training (``target_j`` is the assignment-weighted mean of the batch,
    ``alive_j`` is 0 for a centroid with no mass)::

        u   = alive * (target - c) + repulsion(c)
        m   = momentum * m + (1 - momentum) * u
        c  += centroid_lr * m

    Note the nesting: repulsion enters ``u`` and is therefore smoothed INSIDE the
    momentum EMA, not added on top of it.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │   Input (arbitrary shape)           │
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
        │  Compute ||x - c||^2 distances      │
        └────────────────┬────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────┐
        │  Soft assignments:                  │
        │  softmax(-distances / tau)          │
        └────────┬──────────────┬─────────────┘
                 │              │
        (training only)         │
                 ▼              │
        ┌─────────────────┐     │
        │ Update centroids│     │
        │ + momentum      │     │
        │ + repulsion     │     │
        └─────────────────┘     │
                                ▼
        ┌─────────────────────────────────────┐
        │  Output: assignments or mixture     │
        └─────────────────────────────────────┘

    **Two limitations, both measured by reading the shipped code:**

    * *Masks are ignored.* The layer declares no ``supports_masking`` and no
      ``compute_mask``, and nothing in the package does. A padded timestep is a
      real point to the assignment softmax and contributes its full weight to the
      EMA target, so padding drags the centroids. Strip padding before this layer.
    * *Single replica only.* ``_update_centroids`` writes the codebook with
      ``assign``/``assign_add`` inside ``call``, which under ``tf.distribute`` is
      replica-local: each replica sees only its own shard, and the centroids
      diverge across replicas with no all-reduce to reconcile them.

    :param n_clusters: Number of clusters (K). Must be positive.
    :type n_clusters: int
    :param temperature: Softmax temperature for assignments. Lower values create
        harder assignments. Must be positive. Defaults to 0.1.
    :type temperature: float
    :param momentum: Momentum coefficient for centroid updates. Must be in [0, 1).
        Defaults to 0.9.
    :type momentum: float
    :param centroid_lr: Learning rate for centroid updates. Must be in (0, 1].
        Defaults to 0.1.
    :type centroid_lr: float
    :param repulsion_strength: Strength of repulsive force between centroids.
        Must be non-negative. Defaults to 0.1.
    :type repulsion_strength: float
    :param min_distance: Minimum desired distance between centroids. Must be positive.
        Defaults to 1.0.
    :type min_distance: float
    :param output_mode: Output type: ``'assignments'`` for cluster probabilities or
        ``'mixture'`` for reconstructed inputs using centroids. Defaults to ``'assignments'``.
    :type output_mode: str
    :param cluster_axis: Axis or axes to perform clustering on. Negative values
        are supported. Defaults to -1.
    :type cluster_axis: Union[int, List[int]]
    :param centroid_initializer: Initializer for centroids. Supports ``'orthonormal'``.
        Defaults to ``'orthonormal'``.
    :type centroid_initializer: Union[str, keras.initializers.Initializer]
    :param centroid_regularizer: **Inert — accepted, serialized, and never applied.**
        Centroids are ``trainable=False`` and Keras collects regularizer losses from
        trainable weights only, so ``layer.losses`` stays empty however this is set
        (measured on Keras 3.8.0: an ``L2(1.0)`` evaluating to 4.02 on the centroids
        contributes 0 terms). Setting it emits a warning. It is kept so existing
        configs still deserialize. Contrast ``GMMLayer.mean_regularizer``, which is
        live: GMM's ``means`` are trainable, so its penalty does reach ``layer.losses``.
        Defaults to None.
    :type centroid_regularizer: Optional[keras.regularizers.Regularizer]
    :param random_seed: Random seed for initialization. Defaults to None.
    :type random_seed: Optional[int]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If n_clusters is not positive.
    :raises ValueError: If temperature is not positive.
    :raises ValueError: If momentum is not in [0, 1).
    :raises ValueError: If centroid_lr is not in (0, 1].
    :raises ValueError: If repulsion_strength is negative.
    :raises ValueError: If min_distance is not positive.
    :raises ValueError: If output_mode is not ``'assignments'`` or ``'mixture'``.
    """

    #: The legal ``output_mode`` values for this layer, declared once on the class that
    #: owns them. ``mixtures.factory.validate_mixture_config`` reads this attribute off
    #: ``MIXTURE_REGISTRY[type]['class']`` instead of carrying its own copy.
    VALID_OUTPUT_MODES: ClassVar[FrozenSet[str]] = frozenset({'assignments', 'mixture'})

    def __init__(
        self,
        n_clusters: int,
        temperature: float = 0.1,
        momentum: float = 0.9,
        centroid_lr: float = 0.1,
        repulsion_strength: float = 0.1,
        min_distance: float = 1.0,
        output_mode: OutputMode = 'assignments',
        cluster_axis: Axis = -1,
        centroid_initializer: Union[str, keras.initializers.Initializer] = 'orthonormal',
        centroid_regularizer: Optional[keras.regularizers.Regularizer] = None,
        random_seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Input validation
        self._validate_init_args(
            n_clusters, temperature, momentum, centroid_lr,
            repulsion_strength, min_distance, output_mode
        )

        # Store ALL configuration parameters
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.momentum = momentum
        self.centroid_lr = centroid_lr
        self.repulsion_strength = repulsion_strength
        self.min_distance = min_distance
        self.output_mode = output_mode
        self._init_cluster_axis(cluster_axis)
        self.centroid_initializer = resolve_initializer_arg(centroid_initializer)
        self.centroid_regularizer = keras.regularizers.get(centroid_regularizer)
        # DECISION plan-2026-08-26T061816-c515641a/D-015: keep the inert parameter, warn
        # once. Do NOT delete `centroid_regularizer` -- it is in `get_config()`, so every
        # saved config carrying it would fail to deserialize. Do NOT flip `centroids` to
        # trainable to "make it work" (H-4 / D-002: an optimizer would then double-update
        # them alongside the EMA). Re-measured on Keras 3.8.0: `layer.losses == []` while
        # the same regularizer evaluates to 4.02 on those centroids -- the penalty is
        # never collected, so removing this warning restores a silent no-op.
        if self.centroid_regularizer is not None:
            logger.warning(
                "centroid_regularizer has no effect: centroids are trainable=False "
                "(they are updated by the internal EMA, not by an optimizer), and Keras "
                "collects regularizer losses from trainable weights only, so this "
                "penalty never reaches layer.losses. It is kept for serialization "
                "compatibility. GMMLayer's mean_regularizer IS live, because its means "
                "are trainable."
            )
        self.random_seed = random_seed

        # Initialize attribute placeholders - weights created in build()
        # R6: input_rank/feature_dims/non_feature_dims/original_shape are set by
        # BaseMixtureLayer.__init__ (called via super() above) — not re-declared here.
        self.centroids: Optional[keras.Variable] = None
        self.centroid_momentum: Optional[keras.Variable] = None

    def _validate_init_args(
        self,
        n_clusters: int,
        temperature: float,
        momentum: float,
        centroid_lr: float,
        repulsion_strength: float,
        min_distance: float,
        output_mode: str
    ) -> None:
        """Validate initialization arguments.

        :param n_clusters: Number of clusters.
        :type n_clusters: int
        :param temperature: Softmax temperature.
        :type temperature: float
        :param momentum: Momentum coefficient.
        :type momentum: float
        :param centroid_lr: Centroid learning rate.
        :type centroid_lr: float
        :param repulsion_strength: Repulsion force strength.
        :type repulsion_strength: float
        :param min_distance: Minimum distance between centroids.
        :type min_distance: float
        :param output_mode: Output mode string.
        :type output_mode: str
        :raises ValueError: If any argument is invalid.
        """
        # DECISION plan-2026-08-26T061816-c515641a/D-008: the `isinstance(n_clusters, bool)`
        # clause is NOT redundant with the `isinstance(n_clusters, int)` clause -- in Python
        # `isinstance(True, int)` is True, so a YAML/JSON config `n_clusters: true` reaches here
        # as the integer 1. Removing it restores the measured pre-fix failure: construction
        # succeeds and `build()` dies far away with
        # `ValueError: Cannot convert '(True, 8)' to a shape`. Mirrors gmm.py's n_components.
        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool) or n_clusters < 1:
            raise ValueError(f"n_clusters must be a positive integer, got {n_clusters}")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if not 0 < centroid_lr <= 1:
            raise ValueError(f"centroid_lr must be in (0, 1], got {centroid_lr}")
        if repulsion_strength < 0:
            raise ValueError(f"repulsion_strength must be non-negative, got {repulsion_strength}")
        if min_distance <= 0:
            raise ValueError(f"min_distance must be positive, got {min_distance}")
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {sorted(self.VALID_OUTPUT_MODES)}, "
                f"got '{output_mode}'"
            )

    # DECISION plan-2026-07-20T141712-e03557c8/D-007: this property is a pure NAMING seam
    # (self.n_clusters vs GMMLayer's self.n_components), not a semantic merge. Do NOT
    # rename it to a shared public attribute -- that would break get_config() keys, the
    # registry params, and the byte-unchanged __init__ signature requirement (I2/A5).
    @property
    def _n_prototypes(self) -> int:
        """Prototype count seam read by ``_ClusterAxisMixin`` (see BaseMixtureLayer).

        :return: Number of centroids.
        :rtype: int
        """
        return self.n_clusters

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights.

        :param input_shape: Shape of input tensor as tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is invalid or incompatible with cluster_axis.
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions, got {len(input_shape)}"
            )

        # Store input information
        self.input_rank = len(input_shape)
        self.original_shape = list(input_shape)

        # Normalize and validate cluster axes
        self._setup_cluster_axes()

        # Compute dimensions
        self.feature_dims = self._compute_feature_dims(input_shape)
        self.non_feature_dims = self._compute_non_feature_dims()

        # Initialize centroids using add_weight
        self._initialize_centroids()

        # Initialize momentum buffer with zeros.
        # Mixed-precision: autocast=False keeps the EMA buffer in variable_dtype (float32)
        # so the momentum assign/assign_add stays full precision (see centroids below).
        self.centroid_momentum = self.add_weight(
            name="centroid_momentum",
            shape=(self.n_clusters, self.feature_dims),
            initializer="zeros",
            trainable=False,
            dtype=self.dtype,
            autocast=False
        )

        # Call parent build at the end
        super().build(input_shape)

    def _initialize_centroids(self) -> None:
        """Initialize centroid variables with appropriate initializer."""
        initializer = resolve_prototype_initializer(
            self.centroid_initializer,
            count=self.n_clusters,
            count_name='n_clusters',
            feature_dims=self.feature_dims,
            seed=self.random_seed,
        )

        # Create centroids weight.
        # Mixed-precision: autocast=False keeps the centroids in variable_dtype (float32)
        # inside call() under a mixed_float16 policy. The distance / temperature-softmax
        # math runs in float32 (matching the float32 inputs cast) and the output is cast to
        # compute_dtype on return. Without this, the autocast float16 weight mismatches the
        # float32 inputs (InvalidArgumentError: Sub half vs float).
        # DECISION plan-2026-07-21T080009-845927c7/D-002: centroids are trainable=False.
        # They are updated ONLY by the internal EMA/momentum/repulsion mechanism in
        # call() (_update_centroids) — the VQ-VAE-EMA scheme this layer's docstring
        # describes as REPLACING gradient-based re-averaging. Do NOT restore
        # trainable=True: an optimizer would then apply a SECOND (gradient) update to
        # centroids per step, double-updating them alongside the EMA step. This mirrors
        # centroid_momentum (also trainable=False).
        self.centroids = self.add_weight(
            name="centroids",
            shape=(self.n_clusters, self.feature_dims),
            initializer=initializer,
            regularizer=self.centroid_regularizer,
            trainable=False,
            dtype=self.dtype,
            autocast=False
        )

    def _compute_distances(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Compute squared Euclidean distances to centroids.

        :param inputs: Input tensor of shape ``(batch, features)``.
        :type inputs: keras.KerasTensor
        :return: Distances tensor of shape ``(batch, n_clusters)``.
        :rtype: keras.KerasTensor
        """
        # R3 (D-002): shared pairwise squared-distance helper. inputs (batch, features)
        # x centroids (n_clusters, features) -> (batch, n_clusters). Numerically
        # identical to the prior inline expand-axis-1/0 broadcast.
        return pairwise_squared_distance(inputs, self.centroids)

    def _soft_assignments(self, distances: keras.KerasTensor) -> keras.KerasTensor:
        """Compute soft cluster assignments using temperature-scaled softmax.

        :param distances: Distance tensor of shape ``(batch, n_clusters)``.
        :type distances: keras.KerasTensor
        :return: Assignment probabilities of shape ``(batch, n_clusters)``.
        :rtype: keras.KerasTensor
        """
        # Scale distances by temperature
        scaled_distances = -distances / self.temperature

        # Apply stable softmax
        return keras.ops.softmax(scaled_distances, axis=-1)

    def _compute_repulsion_forces(self) -> keras.KerasTensor:
        """Compute repulsive forces between centroids to prevent collapse.

        :return: Tensor of shape ``(n_clusters, feature_dims)`` containing repulsion vectors.
        :rtype: keras.KerasTensor
        """
        # Compute pairwise differences between centroids
        # Shape: (n_clusters, n_clusters, feature_dims)
        centroid_diffs = (keras.ops.expand_dims(self.centroids, axis=1) -
                         keras.ops.expand_dims(self.centroids, axis=0))

        # Compute squared distances
        # Shape: (n_clusters, n_clusters)
        squared_distances = keras.ops.sum(keras.ops.square(centroid_diffs), axis=-1)

        # Add small epsilon to prevent division by zero on diagonal
        distances = keras.ops.sqrt(squared_distances + keras.backend.epsilon())

        # Compute repulsion strength based on distance
        # Uses soft thresholding with min_distance
        # Shape: (n_clusters, n_clusters)
        repulsion_weights = keras.ops.maximum(
            0.0,
            1.0 - distances / self.min_distance
        )

        # Scale repulsion by strength parameter and distance
        # Shape: (n_clusters, n_clusters, 1)
        repulsion_scale = keras.ops.expand_dims(
            self.repulsion_strength * repulsion_weights / (distances + keras.backend.epsilon()),
            axis=-1
        )

        # Compute repulsion vectors
        # Shape: (n_clusters, n_clusters, feature_dims)
        repulsion_vectors = repulsion_scale * centroid_diffs

        # Sum repulsion from all other centroids
        # Shape: (n_clusters, feature_dims)
        total_repulsion = keras.ops.sum(repulsion_vectors, axis=1)

        return total_repulsion

    def _update_centroids(
        self,
        inputs: keras.KerasTensor,
        assignments: keras.KerasTensor,
        factor: Any = 1.0
    ) -> None:
        """Update centroids using soft assignments with momentum and repulsion.

        :param inputs: Input tensor of shape ``(batch, features)``.
        :type inputs: keras.KerasTensor
        :param assignments: Soft assignment probabilities of shape ``(batch, n_clusters)``.
        :type assignments: keras.KerasTensor
        :param factor: Training factor from ``resolve_training_factor``. The python
            float ``1.0`` (python ``training=True``) takes the exact unmasked path;
            a 0/1 scalar tensor (symbolic training) masks the update so a runtime-False
            flag is a true no-op.
        :type factor: Any
        """
        # Compute weighted sum of points
        # Shape: (n_clusters, features)
        sum_weighted_points = keras.ops.transpose(
            keras.ops.matmul(keras.ops.transpose(inputs), assignments)
        )

        # Compute sum of weights for normalization
        # Shape: (n_clusters,)
        sum_weights = keras.ops.sum(assignments, axis=0, keepdims=True)

        # Per-cluster mass, in units of whole data points.
        # Shape: (n_clusters, 1)
        cluster_mass = keras.ops.transpose(sum_weights)

        # Compute target centroids from data
        # Shape: (n_clusters, features)
        target_centroids = sum_weighted_points / (
            cluster_mass + keras.backend.epsilon()
        )

        # DECISION plan-2026-08-26T061816-c515641a/D-006: gate the DATA-DRIVEN pull by
        # cluster mass. `target_centroids` is 0/epsilon -> the ORIGIN for a centroid that
        # owns no responsibility, so an ungated update drags every dead centroid toward
        # the data manifold and it merges with the live ones (measured: separation
        # 56.46 -> 0.34 in 60 steps, ending closer to a live centroid than the live
        # centroids are to each other). Multiplying by `alive` makes a zero-mass
        # centroid's data term exactly zero, leaving it to repulsion alone. Do NOT gate
        # `repulsion_forces` as well -- repulsion is what separates two centroids that
        # are BOTH alive and coincident, and masking it would reinstate the collapse this
        # layer's repulsion term exists to prevent. Do NOT fold the mask into
        # `target_centroids` (e.g. `alive * target`) either: that pulls a dead centroid
        # to the origin via the `- self.centroids` term instead of freezing it.
        # Shape: (n_clusters, 1), exactly 1.0 for every cluster with real mass, so this
        # is a bit-exact no-op on the all-alive path (pinned by SC-6).
        alive = keras.ops.cast(cluster_mass > _MIN_CLUSTER_MASS, cluster_mass.dtype)

        # Compute repulsion forces
        repulsion_forces = self._compute_repulsion_forces()

        # Combine data-driven update with repulsion
        update = alive * (target_centroids - self.centroids) + repulsion_forces

        # Update momentum buffer
        new_momentum = (self.momentum * self.centroid_momentum +
                       (1.0 - self.momentum) * update)

        if isinstance(factor, float):
            # python training=True fast path: exact, unmasked (factor is always 1.0).
            self.centroid_momentum.assign(new_momentum)
            self.centroids.assign_add(self.centroid_lr * self.centroid_momentum)
        else:
            # Symbolic-tensor path: mask both writes by the 0/1 factor so a runtime
            # training=False leaves momentum and centroids unchanged (true no-op).
            masked_momentum = self.centroid_momentum + factor * (
                new_momentum - self.centroid_momentum
            )
            self.centroid_momentum.assign(masked_momentum)
            self.centroids.assign_add(factor * self.centroid_lr * masked_momentum)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass performing differentiable K-means clustering.

        :param inputs: Input tensor with arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode. Centroid updates
            only occur during training.
        :type training: Optional[bool]
        :return: Output tensor based on output_mode.
        :rtype: keras.KerasTensor
        """
        # Cast inputs to variable_dtype (float32) so the distance / softmax math runs in
        # full precision and matches the autocast=False centroids under a mixed_float16
        # policy. The output is cast back to compute_dtype before returning. Under the
        # default float32 policy this is a no-op.
        inputs = keras.ops.cast(inputs, self.variable_dtype)

        # Reshape input for clustering
        reshaped_inputs, leading_dims = self._reshape_for_clustering(inputs)

        # Compute distances and assignments
        distances = self._compute_distances(reshaped_inputs)
        assignments = self._soft_assignments(distances)

        # DECISION plan-2026-07-21T080009-845927c7/D-003: compute the output BEFORE the centroid
        # update. `_update_centroids` mutates self.centroids in place via assign_add, so
        # reading self.centroids AFTER it (as the old order did) reconstructed the
        # 'mixture' output from POST-update centroids while `assignments` came from the
        # PRE-update centroids — an intra-call inconsistency. Both must use the same
        # (pre-update) centroid state. Do NOT move this block back below _update_centroids.
        if self.output_mode == 'assignments':
            output = assignments
        else:  # output_mode == 'mixture'
            # Reconstruct inputs using weighted centroids (pre-update, consistent with
            # the assignments computed above).
            output = keras.ops.matmul(assignments, self.centroids)

        # DECISION plan_2026-06-14_5e80bd3e/D-001: gate the EMA update on a graph-safe
        # training factor (None -> skip; 1.0 -> exact python-True path; 0/1 tensor ->
        # masked symbolic path). This fires the update for a symbolic training=True tensor
        # (custom @tf.function loop) AND keeps a symbolic False a true no-op, without ever
        # coercing a tensor to a python bool. Supersedes the prior `if training is True:`
        # gate which silently skipped the symbolic case.
        # variable_dtype factor so the masked centroid update stays float32-consistent
        # under a mixed_float16 policy (matches the autocast=False weights).
        training_factor = resolve_training_factor(training, self.variable_dtype)
        if training_factor is not None:
            self._update_centroids(reshaped_inputs, assignments, training_factor)

        # Reshape, then cast to compute_dtype so the layer emits the policy's compute
        # dtype (float16 under mixed precision; no-op under float32).
        return keras.ops.cast(self._reshape_output(output, leading_dims), self.compute_dtype)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "n_clusters": self.n_clusters,
            "temperature": self.temperature,
            "momentum": self.momentum,
            "centroid_lr": self.centroid_lr,
            "repulsion_strength": self.repulsion_strength,
            "min_distance": self.min_distance,
            "output_mode": self.output_mode,
            # DECISION plan_2026-06-14_8c7365d0/D-005: serialize the pre-build axis.
            "cluster_axis": self._cluster_axis_arg,
            "centroid_initializer": (
                self.centroid_initializer if isinstance(self.centroid_initializer, str)
                else keras.initializers.serialize(self.centroid_initializer)
            ),
            # Always serialize (returns None for a None regularizer) for uniformity
            # with GMMLayer.get_config.
            "centroid_regularizer": keras.regularizers.serialize(self.centroid_regularizer),
            "random_seed": self.random_seed
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KMeansLayer":
        """Create a layer instance from its serialized configuration.

        :param config: Configuration dictionary produced by ``get_config``.
        :type config: Dict[str, Any]
        :return: Reconstructed layer instance.
        :rtype: KMeansLayer
        """
        config = dict(config)
        if "centroid_initializer" in config and not isinstance(config["centroid_initializer"], str):
            config["centroid_initializer"] = keras.initializers.deserialize(
                config["centroid_initializer"]
            )
        if "centroid_regularizer" in config:
            config["centroid_regularizer"] = keras.regularizers.deserialize(
                config["centroid_regularizer"]
            )
        return cls(**config)

    @property
    def cluster_centers(self) -> Optional[keras.KerasTensor]:
        """Get current cluster centers.

        :return: Tensor of shape ``(n_clusters, feature_dims)`` or None if not built.
        :rtype: Optional[keras.KerasTensor]
        """
        return self.centroids

    def reset_centroids(self, new_centroids: Optional[keras.KerasTensor] = None) -> None:
        """Reset centroids to new values or reinitialize.

        :param new_centroids: Optional tensor of shape ``(n_clusters, feature_dims)``.
            If None, centroids are reinitialized using random values.
        :type new_centroids: Optional[keras.KerasTensor]
        :raises ValueError: If new_centroids has wrong shape or layer is not built.
        """
        if not self.built:
            raise ValueError("Layer must be built before resetting centroids")

        if new_centroids is not None:
            expected_shape = (self.n_clusters, self.feature_dims)
            if tuple(new_centroids.shape) != expected_shape:
                raise ValueError(
                    f"new_centroids must have shape {expected_shape}, "
                    f"got {tuple(new_centroids.shape)}"
                )
            self.centroids.assign(new_centroids)
        else:
            # Generate fresh random values to ensure different centroids.
            # DECISION plan-2026-07-20T160907-7de371a1/D-009: deliberately UNSEEDED.
            # Do NOT pass `seed=self.random_seed` here for symmetry with
            # `GMMLayer.reset_parameters()`. `keras.random.normal(seed=<int>)` is
            # STATELESS, so a fixed integer seed returns the identical draw on every
            # call -- repeated no-arg resets on a seeded layer then produce bit-identical
            # centroids (measured max|a-b| = 0.0 across three calls), which defeats the
            # method's only purpose: escaping a collapsed centroid configuration
            # mid-training. This was added and reverted once (D-009).
            #
            # CORRECTION (D-009 pass-2 review): an earlier version of this comment
            # said "reproducibility already lives at the layer level". That is only
            # HALF true and must not be read as "a whole-run seeding protocol covers
            # this call". BUILD-time reproducibility does hold -- `random_seed` governs
            # build() init, and two KMeansLayer(random_seed=42) build to identical
            # centroids. RESET-time reproducibility holds under NO protocol at all:
            # a bare keras.random.normal(seed=None) is not covered by
            # keras.utils.set_random_seed() in this Keras/TF version, so build+reset
            # under a fixed global seed gives different centroids on every run
            # (verified: max|a-b| = 0.318, 3/3 runs differ). Callers who need
            # deterministic resets do not have them today.
            #
            # This is also not the binary the earlier comment implied. A
            # `keras.random.SeedGenerator(self.random_seed)` held on the layer would
            # satisfy BOTH constraints -- it advances state per call, so repeated
            # resets still re-draw, while a given layer replays the same sequence
            # across processes. That is a recorded FUTURE OPTION, not implemented
            # here: it adds a serialized stateful variable to the layer and so is a
            # change to the save/load contract, which belongs in its own plan.
            # Until then the honest statement is: unseeded, re-draws correctly,
            # NOT reproducible.
            new_values = keras.random.normal(
                shape=(self.n_clusters, self.feature_dims),
                dtype=self.dtype
            ) * _RESET_CENTROID_SCALE
            self.centroids.assign(new_values)

        # Reset momentum buffer
        self.centroid_momentum.assign(keras.ops.zeros_like(self.centroid_momentum))

# ---------------------------------------------------------------------
