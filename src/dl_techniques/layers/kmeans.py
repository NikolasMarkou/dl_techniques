"""
A differentiable K-means clustering layer for deep networks.

This layer embeds a clustering mechanism directly into a neural network,
enabling end-to-end training of cluster centroids. It provides a
differentiable alternative to the traditional K-means algorithm by
substituting its discrete, non-differentiable operations (hard assignment,
centroid recalculation) with continuous, gradient-based approximations.

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

2.  **Differentiable Centroid Updates:** During training, the centroids are
    updated based on these soft assignments. The new position for each
    centroid is calculated as a weighted average of all input vectors, where
    the weights are the assignment probabilities. This replaces the discrete
    re-averaging step in standard K-means with a smooth, differentiable
    operation.

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

    The update to a centroid `c_j` is conceptually a combination of a data-driven
    pull, a repulsive push from other centroids, and momentum:
    `Δc_j ∝ ( (Σ_i a_ij * x_i) / (Σ_i a_ij) - c_j ) + Σ_{k≠j} Repel(c_j, c_k)`
    This update is then smoothed using a momentum buffer before being applied,
    ensuring stable convergence.

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
import numpy as np
from typing import Optional, Union, Literal, List, Any, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..initializers.orthonormal_initializer import OrthonormalInitializer

# ---------------------------------------------------------------------

# Type aliases for better readability
OutputMode = Literal['assignments', 'mixture']
TensorShape = Union[Tuple[int, ...], List[int]]
Axis = Union[int, List[int]]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class KMeansLayer(keras.layers.Layer):
    """Differentiable K-means layer with momentum and centroid repulsion.

    This layer implements a differentiable version of K-means clustering using
    soft assignments via temperature-controlled softmax, momentum-based centroid
    updates, and repulsive forces between centroids to prevent collapse. The
    soft assignment probability of input ``x_i`` to centroid ``c_j`` is
    ``a_ij = softmax(-||x_i - c_j||^2 / tau)_j``. Centroids are updated as
    ``c_new = c + alpha * (momentum_update + repulsion_forces)``.

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
        ┌────────────────┐      │
        │ Update centroids│     │
        │ + momentum      │     │
        │ + repulsion     │     │
        └────────────────┘      │
                                ▼
        ┌─────────────────────────────────────┐
        │  Output: assignments or mixture     │
        └─────────────────────────────────────┘

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
    :param centroid_regularizer: Optional regularizer for centroids. Defaults to None.
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
        self.cluster_axis = [cluster_axis] if isinstance(cluster_axis, int) else cluster_axis
        self.centroid_initializer = keras.initializers.get(centroid_initializer)
        self.centroid_regularizer = centroid_regularizer
        self.random_seed = random_seed

        # Initialize attribute placeholders - weights created in build()
        self.centroids: Optional[keras.Variable] = None
        self.centroid_momentum: Optional[keras.Variable] = None
        self.input_rank: Optional[int] = None
        self.feature_dims: Optional[int] = None
        self.non_feature_dims: Optional[List[int]] = None
        self.original_shape: Optional[List[int]] = None

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
        if not isinstance(n_clusters, int) or n_clusters < 1:
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

        # Initialize centroids using add_weight
        self._initialize_centroids()

        # Initialize momentum buffer with zeros
        self.centroid_momentum = self.add_weight(
            name="centroid_momentum",
            shape=(self.n_clusters, self.feature_dims),
            initializer="zeros",
            trainable=False,
            dtype=self.dtype
        )

        # Call parent build at the end
        super().build(input_shape)

    def _setup_cluster_axes(self) -> None:
        """Setup and validate cluster axes.

        :raises ValueError: If cluster axes are invalid.
        """
        # Convert negative axes to positive
        self.cluster_axis = [
            axis if axis >= 0 else self.input_rank + axis
            for axis in self.cluster_axis
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

    def _initialize_centroids(self) -> None:
        """Initialize centroid variables with appropriate initializer."""
        # Handle orthonormal initialization specially
        initializer_name = getattr(
            self.centroid_initializer,
            '__class__',
            type(self.centroid_initializer)
        ).__name__

        if (initializer_name == 'OrthonormalInitializer' or
            (isinstance(self.centroid_initializer, str) and
             self.centroid_initializer.lower() == 'orthonormal')):

            if self.n_clusters <= self.feature_dims:
                initializer = OrthonormalInitializer(seed=self.random_seed)
            else:
                logger.warning(
                    f"n_clusters ({self.n_clusters}) > feature_dims ({self.feature_dims}), "
                    "falling back to glorot_normal initializer"
                )
                initializer = keras.initializers.GlorotNormal(seed=self.random_seed)
        else:
            initializer = self.centroid_initializer

        # Create centroids weight
        self.centroids = self.add_weight(
            name="centroids",
            shape=(self.n_clusters, self.feature_dims),
            initializer=initializer,
            regularizer=self.centroid_regularizer,
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
            output_shape = list(input_shape)

            # Handle multiple clustering axes
            if len(self.cluster_axis) > 1:
                # Replace clustered dimensions with n_clusters
                # Remove extra axes in reverse order to preserve indices
                for axis in reversed(self.cluster_axis[1:]):
                    output_shape.pop(axis)
                output_shape[self.cluster_axis[0]] = self.n_clusters
            else:
                output_shape[self.cluster_axis[0]] = self.n_clusters

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

    def _compute_distances(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Compute squared Euclidean distances to centroids.

        :param inputs: Input tensor of shape ``(batch, features)``.
        :type inputs: keras.KerasTensor
        :return: Distances tensor of shape ``(batch, n_clusters)``.
        :rtype: keras.KerasTensor
        """
        # Use broadcasting for memory efficiency
        expanded_inputs = keras.ops.expand_dims(inputs, axis=1)  # (batch, 1, features)
        expanded_centroids = keras.ops.expand_dims(self.centroids, axis=0)  # (1, n_clusters, features)

        # Compute squared Euclidean distances
        distances = keras.ops.sum(
            keras.ops.square(expanded_inputs - expanded_centroids),
            axis=-1
        )

        return distances

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
        assignments: keras.KerasTensor
    ) -> None:
        """Update centroids using soft assignments with momentum and repulsion.

        :param inputs: Input tensor of shape ``(batch, features)``.
        :type inputs: keras.KerasTensor
        :param assignments: Soft assignment probabilities of shape ``(batch, n_clusters)``.
        :type assignments: keras.KerasTensor
        """
        # Compute weighted sum of points
        # Shape: (n_clusters, features)
        sum_weighted_points = keras.ops.transpose(
            keras.ops.matmul(keras.ops.transpose(inputs), assignments)
        )

        # Compute sum of weights for normalization
        # Shape: (n_clusters,)
        sum_weights = keras.ops.sum(assignments, axis=0, keepdims=True)

        # Compute target centroids from data
        # Shape: (n_clusters, features)
        target_centroids = sum_weighted_points / (
            keras.ops.transpose(sum_weights) + keras.backend.epsilon()
        )

        # Compute repulsion forces
        repulsion_forces = self._compute_repulsion_forces()

        # Combine data-driven update with repulsion
        update = (target_centroids - self.centroids) + repulsion_forces

        # Update momentum buffer
        new_momentum = (self.momentum * self.centroid_momentum +
                       (1.0 - self.momentum) * update)
        self.centroid_momentum.assign(new_momentum)

        # Apply momentum update with learning rate
        self.centroids.assign_add(self.centroid_lr * self.centroid_momentum)

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
                output_shape[self.cluster_axis[0]] = self.n_clusters
            else:
                output_shape[self.cluster_axis[0]] = self.n_clusters

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
        """Forward pass performing differentiable K-means clustering.

        :param inputs: Input tensor with arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode. Centroid updates
            only occur during training.
        :type training: Optional[bool]
        :return: Output tensor based on output_mode.
        :rtype: keras.KerasTensor
        """
        # Cast inputs to layer dtype for numerical stability
        inputs = keras.ops.cast(inputs, self.dtype)

        # Reshape input for clustering
        reshaped_inputs = self._reshape_for_clustering(inputs)

        # Compute distances and assignments
        distances = self._compute_distances(reshaped_inputs)
        assignments = self._soft_assignments(distances)

        # Update centroids during training
        if training:
            self._update_centroids(reshaped_inputs, assignments)

        # Compute output based on mode
        if self.output_mode == 'assignments':
            output = assignments
        else:  # output_mode == 'mixture'
            # Reconstruct inputs using weighted centroids
            output = keras.ops.matmul(assignments, self.centroids)

        # Reshape output to match desired shape
        return self._reshape_output(output)

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
            "cluster_axis": self.cluster_axis,
            "centroid_initializer": keras.initializers.serialize(self.centroid_initializer),
            "centroid_regularizer": (keras.regularizers.serialize(self.centroid_regularizer)
                                   if self.centroid_regularizer else None),
            "random_seed": self.random_seed
        })
        return config

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
            # Generate fresh random values to ensure different centroids
            new_values = keras.random.normal(
                shape=(self.n_clusters, self.feature_dims),
                dtype=self.dtype
            ) * 0.1  # Small scale for stability
            self.centroids.assign(new_values)

        # Reset momentum buffer
        self.centroid_momentum.assign(keras.ops.zeros_like(self.centroid_momentum))

# ---------------------------------------------------------------------