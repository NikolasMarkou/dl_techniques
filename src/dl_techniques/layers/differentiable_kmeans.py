"""
DifferentiableKMeans Implementation Guide and Use Cases
====================================================

This implementation provides a differentiable K-means clustering layer that can be
integrated into neural networks for end-to-end training. Here's a comprehensive
guide on its applications and usage patterns.

Core Features
------------
1. Differentiable clustering with learnable centroids
2. Soft assignments using temperature-controlled softmax
3. Support for arbitrary tensor shapes and clustering axes
4. Two output modes: cluster assignments or centroid mixtures
5. Built-in monitoring and visualization tools
6. Comprehensive metrics for clustering quality

Primary Use Cases
---------------

1. Feature Learning and Dimensionality Reduction:
   - Unsupervised feature extraction
   - Learning compact representations
   - Data compression
   Example:
   ```python
   kmeans = DifferentiableKMeansLayer(
       n_clusters=10,
       output_mode='mixture'
   )
   features = kmeans(input_data)
   ```

2. Semantic Segmentation:
   - Image segmentation
   - Object detection preprocessing
   - Region proposal generation
   Example:
   ```python
   kmeans = DifferentiableKMeansLayer(
       n_clusters=5,
       cluster_axis=[1, 2]  # Spatial dimensions
   )
   segments = kmeans(image_features)
   ```

3. Sequence Clustering:
   - Time series analysis
   - Text document clustering
   - Behavioral pattern detection
   Example:
   ```python
   kmeans = DifferentiableKMeansLayer(
       n_clusters=8,
       cluster_axis=1  # Sequence dimension
   )
   sequence_clusters = kmeans(temporal_data)
   ```

4. Attention Mechanism:
   - Soft attention based on cluster assignments
   - Memory-efficient attention alternatives
   - Prototype learning
   Example:
   ```python
   attention_weights = kmeans(query_vectors, temperature=0.5)
   attended_features = tf.matmul(attention_weights, value_vectors)
   ```

5. Online Clustering:
   - Streaming data analysis
   - Real-time pattern detection
   - Adaptive clustering
   Example:
   ```python
   model = keras.Sequential([
       kmeans,
       keras.layers.Dense(output_dim)
   ])
   model.compile(optimizer='adam', loss='mse')
   ```

Advanced Applications
-------------------

1. Multi-Scale Clustering:
   ```python
   def create_multiscale_clustering(input_shape, scales=[10, 5, 3]):
       inputs = keras.Input(shape=input_shape)
       outputs = []
       for n_clusters in scales:
           kmeans = DifferentiableKMeansLayer(n_clusters=n_clusters)
           outputs.append(kmeans(inputs))
       return keras.Model(inputs, outputs)
   ```

2. Hierarchical Clustering:
   ```python
   def create_hierarchical_clustering(input_shape, hierarchy=[32, 16, 8]):
       inputs = keras.Input(shape=input_shape)
       x = inputs
       for n_clusters in hierarchy:
           kmeans = DifferentiableKMeansLayer(n_clusters=n_clusters)
           x = kmeans(x)
       return keras.Model(inputs, x)
   ```

3. Feature Extraction Pipeline:
   ```python
   def create_feature_extractor(input_shape):
       inputs = keras.Input(shape=input_shape)
       x = keras.layers.Dense(256, activation='relu')(inputs)
       kmeans = DifferentiableKMeansLayer(n_clusters=10)
       clusters = kmeans(x)
       features = keras.layers.Concatenate()([x, clusters])
       return keras.Model(inputs, features)
   ```

Best Practices
-------------

1. Temperature Tuning:
   - Start with temperature=0.1
   - Lower values (0.01-0.05) for harder assignments
   - Higher values (0.5-1.0) for softer assignments
   - Adjust based on task requirements

2. Number of Clusters:
   - Start with sqrt(n_samples) as rule of thumb
   - Use elbow method or silhouette analysis
   - Consider computational constraints
   - Monitor cluster distributions

3. Training Strategy:
   - Use appropriate learning rates (0.001-0.0001)
   - Monitor clustering metrics
   - Consider curriculum learning
   - Use callbacks for visualization

4. Performance Optimization:
   - Batch size optimization
   - GPU memory management
   - Efficient data preprocessing
   - Caching strategies

Common Pitfalls and Solutions
---------------------------

1. Cluster Collapse:
   - Use distribution penalty in loss
   - Monitor cluster sizes
   - Adjust temperature
   - Initialize centroids properly

2. Gradient Issues:
   - Use gradient clipping
   - Adjust learning rate
   - Monitor gradient norms
   - Use robust loss functions

3. Memory Management:
   - Batch size tuning
   - Efficient tensor operations
   - Memory-efficient attention
   - GPU optimization

Integration Patterns
------------------

1. With CNNs:
   ```python
   def create_cnn_clustering():
       cnn = keras.applications.ResNet50(include_top=False)
       kmeans = DifferentiableKMeansLayer(n_clusters=10)
       return keras.Sequential([cnn, kmeans])
   ```

2. With Transformers:
   ```python
   def create_transformer_clustering():
       transformer = TransformerBlock(...)
       kmeans = DifferentiableKMeansLayer(n_clusters=10)
       return keras.Sequential([transformer, kmeans])
   ```

3. With AutoEncoders:
   ```python
   def create_clustering_autoencoder():
       encoder = create_encoder(...)
       decoder = create_decoder(...)
       kmeans = DifferentiableKMeansLayer(n_clusters=10)
       return encoder, kmeans, decoder
   ```

Monitoring and Debugging
----------------------

1. Key Metrics:
   - Silhouette score
   - Inertia
   - Cluster distribution
   - Assignment entropy

2. Visualization:
   - t-SNE plots
   - Cluster size distribution
   - Assignment heatmaps
   - Training metrics

3. Debugging Tools:
   - Gradient checking
   - Cluster stability analysis
   - Memory profiling
   - Performance optimization

Production Considerations
-----------------------

1. Model Export:
   - Save in .keras format
   - Version control
   - Documentation
   - Testing pipeline

2. Deployment:
   - Resource requirements
   - Batch size optimization
   - Inference optimization
   - Monitoring setup

3. Maintenance:
   - Retraining strategy
   - Performance monitoring
   - Quality metrics
   - Update procedures

Example Applications
------------------

1. Image Segmentation:
   - Document layout analysis
   - Medical image segmentation
   - Satellite image analysis
   - Object detection

2. Text Analysis:
   - Document clustering
   - Topic modeling
   - Semantic analysis
   - Text summarization

3. Time Series:
   - Financial data analysis
   - Sensor data clustering
   - Behavioral patterns
   - Anomaly detection

4. Recommendation Systems:
   - User clustering
   - Item categorization
   - Collaborative filtering
   - Content organization

For detailed implementation examples and advanced usage patterns,
refer to the accompanying code and documentation.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Union, Literal, List, Any, Tuple, Dict
from keras.api.layers import Layer
from keras.api import initializers
from keras.api import backend

# ---------------------------------------------------------------------

# Type aliases for better readability
OutputMode = Literal['assignments', 'mixture']
TensorShape = Union[tf.TensorShape, List[int]]
Axis = Union[int, List[int]]


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class DifferentiableKMeansLayer(Layer):
    """A differentiable K-means layer with momentum and centroid repulsion.

    This layer implements a differentiable version of K-means clustering using soft
    assignments, momentum, and repulsive forces between centroids to prevent collapse.

    Args:
        n_clusters: int
            Number of clusters (K in K-means)
        temperature: float, optional (default=0.1)
            Softmax temperature for assignments
        momentum: float, optional (default=0.9)
            Momentum coefficient for centroid updates
        centroid_lr: float, optional (default=0.1)
            Learning rate for centroid updates
        repulsion_strength: float, optional (default=0.1)
            Strength of the repulsive force between centroids
        min_distance: float, optional (default=1.0)
            Minimum desired distance between centroids
        output_mode: str, optional (default='assignments')
            Output type: 'assignments' or 'mixture'
        cluster_axis: Union[int, List[int]], optional (default=-1)
            Axis or axes to perform clustering on
        random_seed: Optional[int], optional (default=None)
            Random seed for initialization
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
            random_seed: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the layer."""
        super().__init__(**kwargs)

        # Input validation
        self._validate_init_args(
            n_clusters, temperature, momentum, centroid_lr,
            repulsion_strength, min_distance, output_mode
        )

        # Store configuration
        self.n_clusters = n_clusters
        self.temperature = tf.constant(temperature, dtype=self.dtype)
        self.momentum = tf.constant(momentum, dtype=self.dtype)
        self.centroid_lr = tf.constant(centroid_lr, dtype=self.dtype)
        self.repulsion_strength = tf.constant(repulsion_strength, dtype=self.dtype)
        self.min_distance = tf.constant(min_distance, dtype=self.dtype)
        self.output_mode = output_mode
        self.cluster_axis = [cluster_axis] if isinstance(cluster_axis, int) else cluster_axis
        self.random_seed = random_seed

        # Initialize state variables (set in build)
        self.centroids: Optional[tf.Variable] = None
        self.centroid_momentum: Optional[tf.Variable] = None
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
        """Validate initialization arguments."""
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
        """Build the layer.

        Args:
            input_shape: Shape of input tensor

        Raises:
            ValueError: If input shape is invalid
        """
        # Store input information
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

    def _setup_cluster_axes(self) -> None:
        """Setup and validate cluster axes."""
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
            int: Product of dimensions along cluster axes
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
        """Initialize centroid variables."""
        initializer = initializers.GlorotNormal(seed=self.random_seed)
        self.centroids = self.add_weight(
            name="centroids",
            shape=(self.n_clusters, self.feature_dims),
            initializer=initializer,
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
                # Remove extra axes in reverse order to preserve indices
                for axis in reversed(self.cluster_axis[1:]):
                    output_shape.pop(axis)
                output_shape[self.cluster_axis[0]] = self.n_clusters
            else:
                output_shape[self.cluster_axis[0]] = self.n_clusters

            return tuple(output_shape)

        # For mixture mode, output shape matches input
        return input_shape

    def _reshape_for_clustering(self, inputs: tf.Tensor) -> tf.Tensor:
        """Reshape input tensor for clustering operations.

        Args:
            inputs: Input tensor

        Returns:
            Reshaped tensor with shape (batch * non_feature_dims, feature_dims)
        """
        # Optimize for common case of single axis at end
        if len(self.cluster_axis) == 1 and self.cluster_axis[0] == self.input_rank - 1:
            return tf.reshape(inputs, [-1, self.feature_dims])

        # General case requires transpose
        perm = self.non_feature_dims + self.cluster_axis
        transposed = tf.transpose(inputs, perm)
        return tf.reshape(transposed, [-1, self.feature_dims])

    def _compute_distances(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute squared Euclidean distances to centroids.

        Args:
            inputs: Input tensor of shape (batch, features)

        Returns:
            Distances tensor of shape (batch, n_clusters)
        """
        # Use broadcasting for memory efficiency
        expanded_inputs = tf.expand_dims(inputs, axis=1)
        expanded_centroids = tf.expand_dims(self.centroids, axis=0)

        # Compute squared Euclidean distances
        distances = tf.reduce_sum(
            tf.square(expanded_inputs - expanded_centroids),
            axis=-1
        )

        return distances

    def _soft_assignments(self, distances: tf.Tensor) -> tf.Tensor:
        """Compute soft cluster assignments.

        Args:
            distances: Distance tensor of shape (batch, n_clusters)

        Returns:
            Assignment probabilities of shape (batch, n_clusters)
        """
        # Scale distances by temperature
        scaled_distances = -distances / self.temperature

        # Apply stable softmax
        max_distances = tf.reduce_max(scaled_distances, axis=-1, keepdims=True)
        exp_distances = tf.exp(scaled_distances - max_distances)
        sum_exp_distances = tf.reduce_sum(exp_distances, axis=-1, keepdims=True)

        return exp_distances / (sum_exp_distances + backend.epsilon())

    def _compute_repulsion_forces(self) -> tf.Tensor:
        """Compute repulsive forces between centroids.

        Returns:
            Tensor of shape (n_clusters, feature_dims) containing repulsion vectors
        """
        # Compute pairwise distances between centroids
        # Shape: (n_clusters, n_clusters, feature_dims)
        centroid_diffs = tf.expand_dims(self.centroids, 1) - tf.expand_dims(self.centroids, 0)

        # Compute squared distances
        # Shape: (n_clusters, n_clusters)
        squared_distances = tf.reduce_sum(tf.square(centroid_diffs), axis=-1)

        # Add epsilon to prevent division by zero in diagonal
        distances = tf.sqrt(squared_distances + backend.epsilon())

        # Compute repulsion strength based on distance
        # Uses soft thresholding with min_distance
        # Shape: (n_clusters, n_clusters)
        repulsion_weights = tf.maximum(
            0.0,
            1.0 - distances / self.min_distance
        )

        # Scale repulsion by strength parameter and distance
        # Shape: (n_clusters, n_clusters, 1)
        repulsion_scale = tf.expand_dims(
            self.repulsion_strength * repulsion_weights / (distances + backend.epsilon()),
            axis=-1
        )

        # Compute repulsion vectors
        # Shape: (n_clusters, n_clusters, feature_dims)
        repulsion_vectors = repulsion_scale * centroid_diffs

        # Sum repulsion from all other centroids
        # Shape: (n_clusters, feature_dims)
        total_repulsion = tf.reduce_sum(repulsion_vectors, axis=1)

        return total_repulsion

    def _update_centroids(
            self,
            inputs: tf.Tensor,
            assignments: tf.Tensor
    ) -> None:
        """Update centroids using soft assignments with momentum and repulsion.

        Args:
            inputs: Input tensor
            assignments: Soft assignment probabilities
        """
        # Compute weighted sum of points
        sum_weighted_points = tf.transpose(
            tf.matmul(tf.transpose(inputs), assignments)
        )

        # Compute sum of weights
        sum_weights = tf.reduce_sum(assignments, axis=0, keepdims=True)

        # Compute target centroids from data
        target_centroids = sum_weighted_points / (
                tf.transpose(sum_weights) + backend.epsilon()
        )

        # Compute repulsion forces
        repulsion_forces = self._compute_repulsion_forces()

        # Combine data-driven update with repulsion
        update = (target_centroids - self.centroids) + repulsion_forces

        # Update momentum buffer
        self.centroid_momentum.assign(
            self.momentum * self.centroid_momentum +
            (1.0 - self.momentum) * update
        )

        # Apply momentum update with learning rate
        self.centroids.assign_add(self.centroid_lr * self.centroid_momentum)

    def _reshape_output(self, output: tf.Tensor) -> tf.Tensor:
        """Reshape clustering output to match desired shape.

        Args:
            output: Output tensor from clustering

        Returns:
            Reshaped output tensor
        """
        if self.output_mode == 'assignments':
            output_shape = list(self.original_shape)

            # Handle multiple clustering axes
            if len(self.cluster_axis) > 1:
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

        return tf.reshape(output, output_shape)

    def call(
            self,
            inputs: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor
            training: Boolean indicating training mode

        Returns:
            Output tensor
        """
        # Cast inputs to layer dtype
        inputs = tf.cast(inputs, self.dtype)

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
            output = tf.matmul(assignments, self.centroids)

        # Reshape output to match desired shape
        return self._reshape_output(output)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "n_clusters": self.n_clusters,
            "temperature": float(self.temperature),
            "momentum": float(self.momentum),
            "centroid_lr": float(self.centroid_lr),
            "repulsion_strength": float(self.repulsion_strength),
            "min_distance": float(self.min_distance),
            "output_mode": self.output_mode,
            "cluster_axis": self.cluster_axis,
            "random_seed": self.random_seed
        })
        return config


# ---------------------------------------------------------------------


