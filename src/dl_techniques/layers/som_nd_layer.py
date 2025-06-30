import keras
import tensorflow as tf
from typing import Tuple, Optional, Union, Dict, Any, Callable

@keras.saving.register_keras_serializable()
class SOMLayer(keras.layers.Layer):
    """N-Dimensional Self-Organizing Map (SOM) layer.

    This layer implements a Self-Organizing Map (SOM), an unsupervised neural
    network that maps high-dimensional input data onto a lower-dimensional
    discretized grid. It functions as a content-addressable memory that
    preserves the topological properties of the input space.

    The SOM organizes neurons in an N-dimensional grid (e.g., 1D line,
    2D plane, 3D cube). For each input vector, a "Best Matching Unit" (BMU)
    is found, and the weights of the BMU and its neighbors are updated to
    move closer to the input vector. This process, repeated over many
    iterations, results in a topologically ordered map where similar inputs
    activate nearby neurons on the grid.

    This implementation is fully vectorized for efficient training on modern
    hardware and does not use Python loops for weight updates. The layer's
    weights are not updated via backpropagation and must be trained using
    a custom training loop or by repeatedly calling the layer on input data
    in training mode.

    Args:
        grid_shape: Tuple of integers, the shape of the SOM neuron grid.
            For example, `(10, 10)` for a 2D grid or `(5, 5, 5)` for a 3D grid.
        input_dim: Integer, the dimensionality of the input data vectors.
        initial_learning_rate: Float, the starting learning rate for weight
            updates. Defaults to 0.1.
        decay_function: Optional callable that takes the current iteration and
            max iterations and returns a new learning rate. If `None`, a
            linear decay is used. Defaults to `None`.
        sigma: Float, the initial radius of the neighborhood function.
            Defaults to `1.0`.
        neighborhood_function: String, the type of neighborhood function to use.
            Can be either `'gaussian'` or `'bubble'`. Defaults to `'gaussian'`.
        weights_initializer: String or `keras.initializers.Initializer`.
            Initialization method for the SOM weights. Supports standard Keras
            initializers as well as special strings `'random'` (uniform in
            [0, 1]) and `'sample'` (falls back to `'random'`).
            Defaults to `'random'`.
        regularizer: Optional `keras.regularizers.Regularizer` function applied
            to the weights. Defaults to `None`.
        name: Optional string, the name of the layer.

    Input shape:
        A 2D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
        A tuple of two tensors:
        - `bmu_indices`: `(batch_size, len(grid_shape))` containing integer
          coordinates of the BMU for each input.
        - `quantization_errors`: `(batch_size,)` containing the Euclidean
          distance between each input and its BMU's weights.

    Example:
        >>> # Create a 10x10 SOM for 784-dimensional MNIST data
        >>> som_layer = SOMLayer(grid_shape=(10, 10), input_dim=784)
        >>> input_data = tf.random.uniform((128, 784))
        >>>
        >>> # Forward pass to find BMUs and update weights
        >>> bmu_coords, q_error = som_layer(input_data, training=True)
        >>> print(bmu_coords.shape, q_error.shape)
        (128, 2) (128,)
        >>>
        >>> # Get the learned weight map
        >>> weights_map = som_layer.get_weights_map()
        >>> print(weights_map.shape)
        (10, 10, 784)
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        input_dim: int,
        initial_learning_rate: float = 0.1,
        decay_function: Optional[Callable] = None,
        sigma: float = 1.0,
        neighborhood_function: str = 'gaussian',
        weights_initializer: Union[str, keras.initializers.Initializer] = 'random',
        regularizer: Optional[keras.regularizers.Regularizer] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        if not all(isinstance(d, int) and d > 0 for d in grid_shape):
            raise ValueError("`grid_shape` must be a tuple of positive integers.")
        if input_dim <= 0:
            raise ValueError("`input_dim` must be a positive integer.")
        if neighborhood_function not in ['gaussian', 'bubble']:
            raise ValueError("`neighborhood_function` must be 'gaussian' or 'bubble'.")

        self.grid_shape = grid_shape
        self.grid_dim = len(grid_shape)
        self.input_dim = input_dim
        self.initial_learning_rate = initial_learning_rate
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function

        # Store raw initializer/regularizer for serialization
        self._weights_initializer_config = weights_initializer
        self._regularizer_config = regularizer

        # Handle special string initializers vs. Keras standard initializers
        if isinstance(weights_initializer, str) and weights_initializer in ['random', 'sample']:
            self.weights_initializer = None  # Use custom logic in build
        else:
            self.weights_initializer = keras.initializers.get(weights_initializer)
        self.regularizer = keras.regularizers.get(regularizer)

        self.decay_function = decay_function or (
            lambda x, max_iter: self.initial_learning_rate * (1 - x / max_iter)
        )

        self.iterations = self.add_weight(
            name="iterations", shape=(), dtype=tf.float32, trainable=False,
            initializer=tf.zeros_initializer()
        )
        self.max_iterations = self.add_weight(
            name="max_iterations", shape=(), dtype=tf.float32, trainable=False,
            initializer=lambda shape, dtype: tf.constant(1000.0, dtype=dtype)
        )
        self.grid_positions = self._initialize_grid_positions()

    def build(self, input_shape: Tuple) -> None:
        """Build the layer's weights."""
        if len(input_shape) != 2 or input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch_size, {self.input_dim}), "
                f"but received input_shape={input_shape}"
            )

        if self.weights_initializer is None:
            # Handle special strings 'random' and 'sample'.
            # 'sample' requires input data, which isn't available here, so it
            # falls back to random uniform, matching original behavior.
            initializer = lambda shape, dtype: tf.random.uniform(
                shape=shape, minval=0.0, maxval=1.0, seed=42, dtype=dtype
            )
        else:
            initializer = self.weights_initializer

        self.weights_map = self.add_weight(
            name="som_weights",
            shape=(*self.grid_shape, self.input_dim),
            initializer=initializer,
            regularizer=self.regularizer,
            trainable=False  # Weights are updated manually
        )
        self.built = True

    def _initialize_grid_positions(self) -> tf.Tensor:
        """Initialize the N-dimensional grid positions for the SOM."""
        coord_ranges = [tf.range(d, dtype=tf.float32) for d in self.grid_shape]
        mesh_coords = tf.meshgrid(*coord_ranges, indexing='ij')
        position_grid = tf.stack(mesh_coords, axis=-1)
        return position_grid

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Find BMUs and, if training, update weights."""
        bmu_indices, quantization_errors = self._find_bmu(inputs)

        if training:
            self._update_weights(inputs, bmu_indices)
            self.iterations.assign_add(tf.cast(tf.shape(inputs)[0], tf.float32))

        return bmu_indices, quantization_errors

    def _find_bmu(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Find the Best Matching Unit (BMU) for each input vector."""
        flat_weights = tf.reshape(self.weights_map, [-1, self.input_dim])

        # Use squared Euclidean distance for efficiency
        squared_distances = tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, 1) - tf.expand_dims(flat_weights, 0)),
            axis=2
        )

        bmu_flat_indices = tf.argmin(squared_distances, axis=1)
        bmu_indices = tf.unravel_index(bmu_flat_indices, self.grid_shape)
        bmu_indices = tf.stack(bmu_indices, axis=1)

        quantization_errors = tf.gather(
            tf.sqrt(squared_distances), bmu_flat_indices, batch_dims=1
        )
        return tf.cast(bmu_indices, tf.int32), quantization_errors

    def _update_weights(self, inputs: tf.Tensor, bmu_indices: tf.Tensor) -> None:
        """Update the SOM weights using a vectorized learning rule."""
        current_learning_rate = self.decay_function(self.iterations, self.max_iterations)
        current_sigma = self.sigma * (1.0 - self.iterations / self.max_iterations)
        current_sigma = tf.maximum(current_sigma, 1e-4) # Prevent division by zero

        bmu_coords = tf.cast(bmu_indices, dtype=tf.float32)

        # 1. Expand dimensions for broadcasting
        bmu_coords_expanded = tf.reshape(
            bmu_coords, [tf.shape(inputs)[0]] + [1] * self.grid_dim + [self.grid_dim]
        )
        grid_pos_expanded = tf.expand_dims(self.grid_positions, axis=0)

        # 2. Compute neighborhood values for all neurons for each BMU in the batch
        squared_dist_to_bmus = tf.reduce_sum(
            tf.square(grid_pos_expanded - bmu_coords_expanded), axis=-1
        )

        if self.neighborhood_function == 'gaussian':
            neighborhood = tf.exp(-squared_dist_to_bmus / (2 * tf.square(current_sigma)))
        else: # 'bubble'
            dist_to_bmus = tf.sqrt(squared_dist_to_bmus)
            neighborhood = tf.cast(dist_to_bmus <= current_sigma, tf.float32)

        # 3. Compute the weight update delta for each input
        neighborhood_expanded = tf.expand_dims(neighborhood, axis=-1)
        inputs_expanded = tf.reshape(
            inputs, [tf.shape(inputs)[0]] + [1] * self.grid_dim + [self.input_dim]
        )
        delta_per_input = neighborhood_expanded * (inputs_expanded - tf.expand_dims(self.weights_map, 0))

        # 4. Sum the deltas over the batch and apply learning rate
        weight_update = current_learning_rate * tf.reduce_sum(delta_per_input, axis=0)
        self.weights_map.assign_add(weight_update)

    def get_weights_map(self) -> tf.Tensor:
        """Return the current weights organized as an N-D map."""
        return self.weights_map

    def get_config(self) -> Dict[str, Any]:
        """Return the layer's configuration for serialization."""
        config = super().get_config()
        config.update({
            'grid_shape': self.grid_shape,
            'input_dim': self.input_dim,
            'initial_learning_rate': self.initial_learning_rate,
            'sigma': self.sigma,
            'neighborhood_function': self.neighborhood_function,
            'weights_initializer': self._weights_initializer_config,
            'regularizer': self._regularizer_config,
        })
        return config
