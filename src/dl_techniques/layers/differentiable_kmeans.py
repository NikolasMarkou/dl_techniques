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
from dataclasses import dataclass
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from typing import Optional, Union, Literal, List, Any, Tuple, Dict

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.python import keras
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------

# Type aliases for better readability
OutputMode = Literal['assignments', 'mixture']
TensorShape = Union[tf.TensorShape, List[int]]
Axis = Union[int, List[int]]


# ---------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class DifferentiableKMeansLayer(layers.Layer):
    """A differentiable K-means layer that can operate on arbitrary tensor shapes.

    This layer implements a differentiable version of K-means clustering using soft
    assignments, enabling end-to-end training in neural networks. The softmax
    temperature parameter controls the "softness" of assignments.

    Args:
        n_clusters: int
            Number of clusters (K in K-means)
        temperature: float, optional (default=0.1)
            Softmax temperature for assignments. Lower values produce harder
            assignments, higher values produce softer assignments.
        output_mode: str, optional (default='assignments')
            Type of output to return:
            - 'assignments': returns soft cluster assignments
            - 'mixture': returns weighted mixture of centroids
        cluster_axis: Union[int, List[int]], optional (default=-1)
            Axis or axes to perform clustering on. Can be a single axis (int)
            or multiple axes (List[int])
        random_seed: Optional[int], optional (default=None)
            Random seed for centroid initialization

    Input shape:
        N-D tensor with shape: (batch_size, ..., feature_dim)
        The feature dimension(s) specified by cluster_axis will be used for clustering.

    Output shape:
        If output_mode == 'assignments':
            Same as input shape but with cluster_axis dimensions replaced by n_clusters
        If output_mode == 'mixture':
            Same as input shape

    Raises:
        ValueError: If invalid arguments are provided

    Example:
        ```python
        # Cluster on feature dimension
        layer = DifferentiableKMeansLayer(n_clusters=10, temperature=0.1)
        input_tensor = tf.random.normal((32, 100))  # (batch_size, features)
        output_tensor = layer(input_tensor)  # (batch_size, n_clusters)
        ```
    """

    def __init__(
            self,
            n_clusters: int,
            temperature: float = 0.1,
            output_mode: OutputMode = 'assignments',
            cluster_axis: Axis = -1,
            random_seed: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the layer."""
        super().__init__(**kwargs)

        # Input validation
        self._validate_init_args(n_clusters, temperature, output_mode)

        # Store configuration
        self.n_clusters = n_clusters
        self.temperature = tf.constant(temperature, dtype=self.dtype)
        self.output_mode = output_mode
        self.cluster_axis = [cluster_axis] if isinstance(cluster_axis, int) else cluster_axis
        self.random_seed = random_seed

        # Initialize state variables (set in build)
        self.centroids: Optional[tf.Variable] = None
        self.input_rank: Optional[int] = None
        self.feature_dims: Optional[int] = None
        self.non_feature_dims: Optional[List[int]] = None
        self.original_shape: Optional[List[int]] = None

    def _validate_init_args(
            self,
            n_clusters: int,
            temperature: float,
            output_mode: str
    ) -> None:
        """Validate initialization arguments.

        Args:
            n_clusters: Number of clusters
            temperature: Softmax temperature
            output_mode: Output mode string

        Raises:
            ValueError: If any arguments are invalid
        """
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError(f"n_clusters must be a positive integer, got {n_clusters}")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
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

    def _update_centroids(
            self,
            inputs: tf.Tensor,
            assignments: tf.Tensor
    ) -> None:
        """Update centroids using soft assignments.

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

        # Update centroids with weighted average
        new_centroids = sum_weighted_points / (
                tf.transpose(sum_weights) + backend.epsilon()
        )

        self.centroids.assign(new_centroids)

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
        """Get layer configuration.

        Returns:
            Dictionary containing configuration
        """
        config = super().get_config()
        config.update({
            "n_clusters": self.n_clusters,
            "temperature": float(self.temperature),
            "output_mode": self.output_mode,
            "cluster_axis": self.cluster_axis,
            "random_seed": self.random_seed
        })
        return config


# ---------------------------------------------------------------------


