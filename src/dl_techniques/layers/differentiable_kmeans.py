import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Union, Literal, List

OutputMode = Literal['assignments', 'mixture']

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class DifferentiableKMeansLayer(keras.layers.Layer):
    """
    A differentiable K-means layer that can operate on arbitrary tensor shapes.

    The layer implements a differentiable version of K-means clustering using soft assignments,
    which allows for end-to-end training in neural networks. The softmax temperature parameter
    controls how "soft" the assignments are - lower values make assignments more discrete
    (closer to regular k-means), while higher values make them softer.

    Features:
    - Can cluster on any axis or combination of axes
    - Handles arbitrary input shapes
    - Two output modes: soft assignments or centroid mixtures
    - Trainable centroids
    - Configurable softmax temperature

    Example shapes:
    - Input (batch, features) → cluster on features axis
    - Input (batch, height, width, channels) → cluster on channels or spatial dimensions
    - Input (batch, sequence, features) → cluster on sequence or feature dimension

    Args:
        n_clusters: Number of clusters (K in K-means)
        temperature: Softmax temperature for assignments (default: 0.1)
                    Lower values → harder assignments
                    Higher values → softer assignments
        output_mode: Type of output to return (default: 'assignments')
                    'assignments': returns soft cluster assignments
                    'mixture': returns weighted mixture of centroids
        cluster_axis: Which axis/axes to perform clustering on (default: -1)
                     Can be single axis (int) or multiple axes (List[int])
        random_seed: Optional random seed for centroid initialization
    """

    def __init__(
            self,
            n_clusters: int,
            temperature: float = 0.1,
            output_mode: OutputMode = 'assignments',
            cluster_axis: Union[int, List[int]] = -1,
            random_seed: Optional[int] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Validate inputs
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if output_mode not in ['assignments', 'mixture']:
            raise ValueError("output_mode must be either 'assignments' or 'mixture'")

        # Store configuration
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.output_mode = output_mode
        self.cluster_axis = cluster_axis if isinstance(cluster_axis, list) else [cluster_axis]
        self.random_seed = random_seed

        # Will be set in build()
        self.centroids = None
        self.input_rank = None
        self.feature_dims = None
        self.non_feature_dims = None
        self.original_shape = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Initialize layer variables and compute shapes."""
        # Store input information
        self.input_rank = len(input_shape)
        self.original_shape = list(input_shape)

        # Convert negative axes to positive and sort
        self.cluster_axis = [
            axis if axis >= 0 else self.input_rank + axis
            for axis in self.cluster_axis
        ]
        self.cluster_axis.sort()

        # Compute feature dimensions
        self.feature_dims = np.prod([
            input_shape[axis] for axis in self.cluster_axis
        ]).astype(int)

        # Compute non-feature dimensions
        self.non_feature_dims = [
            i for i in range(self.input_rank)
            if i not in self.cluster_axis
        ]

        # Initialize centroids with Glorot/Xavier initialization
        self.centroids = self.add_weight(
            name="centroids",
            shape=(self.n_clusters, self.feature_dims),
            initializer=keras.initializers.GlorotNormal(seed=self.random_seed),
            trainable=True
        )

    def compute_output_shape(self, input_shape):
        """Compute the output shape based on input shape and configuration."""
        if self.output_mode == 'assignments':
            output_shape = list(input_shape)

            # For multiple clustering axes, collapse them into one
            if len(self.cluster_axis) > 1:
                for axis in reversed(self.cluster_axis[1:]):
                    output_shape.pop(axis)
                output_shape[self.cluster_axis[0]] = self.n_clusters
            else:
                output_shape[self.cluster_axis[0]] = self.n_clusters

            return tuple(output_shape)
        else:  # output_mode == 'mixture'
            return input_shape

    def _reshape_for_clustering(self, inputs: tf.Tensor) -> tf.Tensor:
        """Reshape input tensor for clustering operations."""
        # For single axis clustering at the end, we can reshape directly
        if len(self.cluster_axis) == 1 and self.cluster_axis[0] == self.input_rank - 1:
            return tf.reshape(inputs, [-1, self.feature_dims])

        # Otherwise, we need to transpose first
        else:
            # Move clustering dimensions to the end
            perm = self.non_feature_dims + self.cluster_axis
            transposed = tf.transpose(inputs, perm)

            # Flatten all non-clustering dimensions into batch dimension
            return tf.reshape(transposed, [-1, self.feature_dims])

    def _reshape_output(self, output: tf.Tensor) -> tf.Tensor:
        """Reshape clustering output back to desired shape."""
        input_shape = self.original_shape

        if self.output_mode == 'assignments':
            # Start with original shape
            output_shape = list(input_shape)

            # If we're clustering on multiple axes, first collapse them into one
            if len(self.cluster_axis) > 1:
                # Remove all but the first cluster axis, working backwards
                for axis in reversed(self.cluster_axis[1:]):
                    output_shape.pop(axis)
                # Replace the first cluster axis with n_clusters
                output_shape[self.cluster_axis[0]] = self.n_clusters
            else:
                # Single axis clustering - simply replace with n_clusters
                output_shape[self.cluster_axis[0]] = self.n_clusters

            # Replace None in the batch dimension with -1 for dynamic reshaping
            output_shape[0] = -1

        else:  # output_mode == 'mixture'
            # For mixture mode, output shape is same as input shape
            output_shape = list(input_shape)
            # Replace None in the batch dimension with -1 for dynamic reshaping
            output_shape[0] = -1

        return tf.reshape(output, output_shape)

    def _compute_distances(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute squared Euclidean distances to centroids."""
        # Expand dimensions for broadcasting
        expanded_inputs = tf.expand_dims(inputs, axis=1)  # (batch, 1, features)
        expanded_centroids = tf.expand_dims(self.centroids, axis=0)  # (1, clusters, features)

        # Compute distances
        distances = tf.reduce_sum(
            tf.square(expanded_inputs - expanded_centroids),
            axis=-1
        )  # (batch, clusters)

        return distances

    def _soft_assignments(self, distances: tf.Tensor) -> tf.Tensor:
        """Compute soft cluster assignments using temperature-scaled softmax."""
        scaled_distances = -distances / self.temperature
        assignments = tf.nn.softmax(scaled_distances, axis=-1)
        return assignments

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the layer."""
        # Reshape input for clustering
        reshaped_inputs = self._reshape_for_clustering(inputs)

        # Compute distances and assignments
        distances = self._compute_distances(reshaped_inputs)
        assignments = self._soft_assignments(distances)

        if training:
            # Update centroids using soft assignments
            sum_weighted_points = tf.transpose(
                tf.matmul(tf.transpose(reshaped_inputs), assignments)
            )
            sum_weights = tf.reduce_sum(assignments, axis=0, keepdims=True)
            new_centroids = sum_weighted_points / (tf.transpose(sum_weights) + 1e-7)
            self.centroids.assign(new_centroids)

        # Compute output based on mode
        if self.output_mode == 'assignments':
            output = assignments
        else:  # output_mode == 'mixture'
            output = tf.matmul(assignments, self.centroids)

        # Reshape output to match desired shape
        return self._reshape_output(output)

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "n_clusters": self.n_clusters,
            "temperature": self.temperature,
            "output_mode": self.output_mode,
            "cluster_axis": self.cluster_axis,
            "random_seed": self.random_seed
        })
        return config

# ---------------------------------------------------------------------
