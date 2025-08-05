"""
Differentiable K-Means Layer for Neural Networks.

This module implements a differentiable K-means clustering layer that can be seamlessly
integrated into neural network architectures. Unlike traditional K-means, this implementation
uses soft assignments, momentum-based centroid updates, and repulsive forces to enable
stable end-to-end training through backpropagation.

Mathematical Background
-----------------------

Traditional K-means clustering partitions data points into K clusters by minimizing:
    J = Σᵢ Σⱼ wᵢⱼ ||xᵢ - cⱼ||²

Where:
- xᵢ are data points
- cⱼ are cluster centroids
- wᵢⱼ ∈ {0,1} are hard assignments

The differentiable version replaces hard assignments with soft assignments:
    wᵢⱼ = softmax(-||xᵢ - cⱼ||² / τ)

Where τ is the temperature parameter controlling assignment softness.

Key Features
------------

1. **Soft Assignments**: Uses temperature-scaled softmax for differentiable clustering
2. **Momentum Updates**: Incorporates momentum for stable centroid learning
3. **Centroid Repulsion**: Prevents centroid collapse through inter-centroid repulsive forces
4. **Flexible Output**: Supports both assignment probabilities and mixture reconstructions
5. **Multi-axis Clustering**: Can cluster along arbitrary tensor dimensions
6. **Orthonormal Initialization**: Uses orthogonal initialization when possible for better convergence

Centroid Update Mechanism
-------------------------

The centroids are updated using a combination of:

1. **Data-driven updates**: Moving centroids toward their assigned points
2. **Momentum**: Smoothing updates over time steps
3. **Repulsive forces**: Preventing centroids from collapsing to the same location

The update rule is:
    vₜ₊₁ = β·vₜ + (1-β)·(c_target - cₜ + F_repulsion)
    cₜ₊₁ = cₜ + α·vₜ₊₁

Where:
- β is momentum coefficient
- α is learning rate
- F_repulsion prevents centroid collapse

Repulsive Forces
----------------

To prevent centroids from collapsing, repulsive forces are computed as:
    F_repulsion = Σₖ≠ⱼ strength × max(0, 1 - ||cⱼ - cₖ||/d_min) × (cⱼ - cₖ)/||cⱼ - cₖ||

This ensures centroids maintain minimum separation while learning from data.

Usage Examples
--------------

Basic Feature Clustering:
    >>> # Cluster feature vectors into 10 groups
    >>> kmeans_layer = KMeansLayer(n_clusters=10, temperature=0.1)
    >>> assignments = kmeans_layer(features)  # Shape: (..., 10)

Mixture Reconstruction:
    >>> # Reconstruct inputs using learned centroids
    >>> kmeans_layer = KMeansLayer(n_clusters=5, output_mode='mixture')
    >>> reconstructed = kmeans_layer(inputs)  # Same shape as inputs

Multi-dimensional Clustering:
    >>> # Cluster spatial locations in images
    >>> spatial_kmeans = KMeansLayer(
    ...     n_clusters=16,
    ...     cluster_axis=[1, 2],  # Height and width dimensions
    ...     temperature=0.05
    ... )
    >>> spatial_assignments = spatial_kmeans(image_features)

Integration in Neural Networks:
    >>> model = keras.Sequential([
    ...     keras.layers.Dense(256, activation='relu'),
    ...     KMeansLayer(n_clusters=32, temperature=0.1),
    ...     keras.layers.Dense(10, activation='softmax')
    ... ])

Applications
------------

- **Representation Learning**: Learning discrete latent representations
- **Attention Mechanisms**: Soft attention over learned prototypes
- **Vector Quantization**: Differentiable vector quantization for compression
- **Mixture Models**: Learning mixture components in end-to-end fashion
- **Regularization**: Encouraging clustering structure in embeddings

References
----------

- Lloyd, S. (1982). Least squares quantization in PCM. IEEE Transactions on Information Theory.
- Xie, J. et al. (2016). Unsupervised deep embedding for clustering analysis. ICML.
- Yang, B. et al. (2017). Towards k-means-friendly spaces: Simultaneous deep learning and clustering. ICML.
- Caron, M. et al. (2018). Deep clustering for unsupervised learning of visual features. ECCV.

Author: DL-Techniques Team
License: MIT
"""

import keras
import numpy as np
from keras import ops
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
    """A differentiable K-means layer with momentum and centroid repulsion.

    This layer implements a differentiable version of K-means clustering using soft
    assignments, momentum, and repulsive forces between centroids to prevent collapse.

    The layer performs clustering on specified axes and can output either soft assignments
    or mixture representations based on the learned centroids.

    Args:
        n_clusters (int): Number of clusters (K in K-means).
        temperature (float, optional): Softmax temperature for assignments. Defaults to 0.1.
        momentum (float, optional): Momentum coefficient for centroid updates. Defaults to 0.9.
        centroid_lr (float, optional): Learning rate for centroid updates. Defaults to 0.1.
        repulsion_strength (float, optional): Strength of the repulsive force between centroids. Defaults to 0.1.
        min_distance (float, optional): Minimum desired distance between centroids. Defaults to 1.0.
        output_mode (str, optional): Output type: 'assignments' or 'mixture'. Defaults to 'assignments'.
        cluster_axis (Union[int, List[int]], optional): Axis or axes to perform clustering on. Defaults to -1.
        centroid_initializer (Union[str, keras.initializers.Initializer], optional):
            Initializer for centroids. Defaults to 'orthonormal'.
        centroid_regularizer (Optional[keras.regularizers.Regularizer], optional):
            Regularizer for centroids. Defaults to None.
        random_seed (Optional[int], optional): Random seed for initialization. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Raises:
        ValueError: If any of the input parameters are invalid.

    Example:
        >>> # Basic usage for feature clustering
        >>> layer = KMeansLayer(n_clusters=10, temperature=0.1)
        >>> output = layer(input_tensor)  # Shape: (batch, ..., 10)

        >>> # Mixture output mode
        >>> layer = KMeansLayer(n_clusters=5, output_mode='mixture')
        >>> reconstructed = layer(input_tensor)  # Same shape as input
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
        """Initialize the KMeansLayer."""
        super().__init__(**kwargs)

        # Input validation
        self._validate_init_args(
            n_clusters, temperature, momentum, centroid_lr,
            repulsion_strength, min_distance, output_mode
        )

        # Store configuration
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

        # Initialize state variables (set in build)
        self.centroids: Optional[keras.Variable] = None
        self.centroid_momentum: Optional[keras.Variable] = None
        self.input_rank: Optional[int] = None
        self.feature_dims: Optional[int] = None
        self.non_feature_dims: Optional[List[int]] = None
        self.original_shape: Optional[List[int]] = None
        self._built_input_shape: Optional[Tuple[int, ...]] = None

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

        Args:
            n_clusters: Number of clusters
            temperature: Softmax temperature
            momentum: Momentum coefficient
            centroid_lr: Centroid learning rate
            repulsion_strength: Repulsion force strength
            min_distance: Minimum distance between centroids
            output_mode: Output mode string

        Raises:
            ValueError: If any argument is invalid
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

    def build(self, input_shape: TensorShape) -> None:
        """Build the layer weights.

        Args:
            input_shape: Shape of input tensor as tuple or list

        Raises:
            ValueError: If input shape is invalid
        """
        # Store input information
        self._built_input_shape = tuple(input_shape)
        self.input_rank = len(input_shape)
        self.original_shape = list(input_shape)

        # Normalize and validate cluster axes
        self._setup_cluster_axes()

        # Compute dimensions
        self.feature_dims = self._compute_feature_dims(input_shape)
        self.non_feature_dims = self._compute_non_feature_dims()

        # Initialize centroids
        self._initialize_centroids()

        # Initialize momentum buffer with zeros
        self.centroid_momentum = self.add_weight(
            name="centroid_momentum",
            shape=(self.n_clusters, self.feature_dims),
            initializer="zeros",
            trainable=False,
            dtype=self.dtype
        )

        super().build(input_shape)

    def _setup_cluster_axes(self) -> None:
        """Setup and validate cluster axes.

        Raises:
            ValueError: If cluster axes are invalid
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

    def _compute_feature_dims(self, input_shape: TensorShape) -> int:
        """Compute total feature dimensions.

        Args:
            input_shape: Input tensor shape

        Returns:
            Product of dimensions along cluster axes

        Raises:
            ValueError: If input shape is invalid
        """
        try:
            return int(np.prod([input_shape[axis] for axis in self.cluster_axis]))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid input shape {input_shape} for cluster axes {self.cluster_axis}"
            ) from e

    def _compute_non_feature_dims(self) -> List[int]:
        """Compute non-feature dimensions.

        Returns:
            List of axes not used for clustering
        """
        return [i for i in range(self.input_rank) if i not in self.cluster_axis]

    def _initialize_centroids(self) -> None:
        """Initialize centroid variables with appropriate initializer."""
        # Handle orthonormal initialization specially
        initializer_name = getattr(self.centroid_initializer, '__class__', type(self.centroid_initializer)).__name__

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

        self.centroids = self.add_weight(
            name="centroids",
            shape=(self.n_clusters, self.feature_dims),
            initializer=initializer,
            regularizer=self.centroid_regularizer,
            trainable=True,
            dtype=self.dtype
        )

    def compute_output_shape(self, input_shape: TensorShape) -> Tuple[int, ...]:
        """Compute shape of layer output.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Output tensor shape
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

        Args:
            inputs: Input tensor

        Returns:
            Reshaped tensor with shape (batch * non_feature_dims, feature_dims)
        """
        # Optimize for common case of single axis at end
        if len(self.cluster_axis) == 1 and self.cluster_axis[0] == self.input_rank - 1:
            return ops.reshape(inputs, [-1, self.feature_dims])

        # General case requires transpose
        perm = self.non_feature_dims + self.cluster_axis
        transposed = ops.transpose(inputs, perm)
        return ops.reshape(transposed, [-1, self.feature_dims])

    def _compute_distances(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Compute squared Euclidean distances to centroids.

        Args:
            inputs: Input tensor of shape (batch, features)

        Returns:
            Distances tensor of shape (batch, n_clusters)
        """
        # Use broadcasting for memory efficiency
        expanded_inputs = ops.expand_dims(inputs, axis=1)  # (batch, 1, features)
        expanded_centroids = ops.expand_dims(self.centroids, axis=0)  # (1, n_clusters, features)

        # Compute squared Euclidean distances
        distances = ops.sum(
            ops.square(expanded_inputs - expanded_centroids),
            axis=-1
        )

        return distances

    def _soft_assignments(self, distances: keras.KerasTensor) -> keras.KerasTensor:
        """Compute soft cluster assignments using temperature-scaled softmax.

        Args:
            distances: Distance tensor of shape (batch, n_clusters)

        Returns:
            Assignment probabilities of shape (batch, n_clusters)
        """
        # Scale distances by temperature
        scaled_distances = -distances / self.temperature

        # Apply stable softmax
        return ops.softmax(scaled_distances, axis=-1)

    def _compute_repulsion_forces(self) -> keras.KerasTensor:
        """Compute repulsive forces between centroids to prevent collapse.

        Returns:
            Tensor of shape (n_clusters, feature_dims) containing repulsion vectors
        """
        # Compute pairwise differences between centroids
        # Shape: (n_clusters, n_clusters, feature_dims)
        centroid_diffs = (ops.expand_dims(self.centroids, axis=1) -
                         ops.expand_dims(self.centroids, axis=0))

        # Compute squared distances
        # Shape: (n_clusters, n_clusters)
        squared_distances = ops.sum(ops.square(centroid_diffs), axis=-1)

        # Add small epsilon to prevent division by zero on diagonal
        distances = ops.sqrt(squared_distances + keras.backend.epsilon())

        # Compute repulsion strength based on distance
        # Uses soft thresholding with min_distance
        # Shape: (n_clusters, n_clusters)
        repulsion_weights = ops.maximum(
            0.0,
            1.0 - distances / self.min_distance
        )

        # Scale repulsion by strength parameter and distance
        # Shape: (n_clusters, n_clusters, 1)
        repulsion_scale = ops.expand_dims(
            self.repulsion_strength * repulsion_weights / (distances + keras.backend.epsilon()),
            axis=-1
        )

        # Compute repulsion vectors
        # Shape: (n_clusters, n_clusters, feature_dims)
        repulsion_vectors = repulsion_scale * centroid_diffs

        # Sum repulsion from all other centroids
        # Shape: (n_clusters, feature_dims)
        total_repulsion = ops.sum(repulsion_vectors, axis=1)

        return total_repulsion

    def _update_centroids(
        self,
        inputs: keras.KerasTensor,
        assignments: keras.KerasTensor
    ) -> None:
        """Update centroids using soft assignments with momentum and repulsion.

        Args:
            inputs: Input tensor of shape (batch, features)
            assignments: Soft assignment probabilities of shape (batch, n_clusters)
        """
        # Compute weighted sum of points
        # Shape: (n_clusters, features)
        sum_weighted_points = ops.transpose(
            ops.matmul(ops.transpose(inputs), assignments)
        )

        # Compute sum of weights for normalization
        # Shape: (n_clusters,)
        sum_weights = ops.sum(assignments, axis=0, keepdims=True)

        # Compute target centroids from data
        # Shape: (n_clusters, features)
        target_centroids = sum_weighted_points / (
            ops.transpose(sum_weights) + keras.backend.epsilon()
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

        Args:
            output: Output tensor from clustering

        Returns:
            Reshaped output tensor
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

        return ops.reshape(output, output_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor
            training: Boolean indicating training mode. If None, uses the global training mode.

        Returns:
            Output tensor based on output_mode:
            - 'assignments': Soft cluster assignments
            - 'mixture': Reconstructed tensor using cluster centroids
        """
        # Determine training mode
        if training is None:
            training = False  # Default to inference mode for simplicity

        # Cast inputs to layer dtype for numerical stability
        inputs = ops.cast(inputs, self.dtype)

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
            output = ops.matmul(assignments, self.centroids)

        # Reshape output to match desired shape
        return self._reshape_output(output)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration
        """
        return {
            "input_shape": self._built_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Dictionary containing build configuration
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @property
    def cluster_centers(self) -> Optional[keras.KerasTensor]:
        """Get current cluster centers.

        Returns:
            Tensor of shape (n_clusters, feature_dims) containing cluster centroids,
            or None if layer hasn't been built yet
        """
        return self.centroids

    def reset_centroids(self, new_centroids: Optional[keras.KerasTensor] = None) -> None:
        """Reset centroids to new values or reinitialize.

        Args:
            new_centroids: Optional tensor of shape (n_clusters, feature_dims).
                If None, centroids are reinitialized using random values.

        Raises:
            ValueError: If new_centroids has wrong shape or layer isn't built
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
        self.centroid_momentum.assign(ops.zeros_like(self.centroid_momentum))

# ---------------------------------------------------------------------
