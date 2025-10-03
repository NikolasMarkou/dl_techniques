"""
N-Dimensional Self-Organizing Map (SOM) layer implementation for Keras.

A Self-Organizing Map is an unsupervised neural network that performs dimensionality
reduction through competitive learning and topological organization. This implementation
supports N-dimensional grids (1D, 2D, 3D, etc.) and maps high-dimensional input data 
onto a lower-dimensional discretized grid, preserving topological relationships between 
input patterns.

SOMs function as associative memory structures where:
  - Each neuron represents a prototype vector in the input space
  - Competitive learning determines which neuron "wins" for each input
  - Neighborhood functions create topological organization
  - Similar inputs activate nearby regions in the grid
  - The grid preserves distance relationships from the input space

Learning Process:
----------------
1. For each input vector:
   a. Find the Best Matching Unit (BMU) - the neuron with weights closest to the input
   b. Update the BMU and its neighbors to move them closer to the input
   c. The neighborhood size and learning rate decrease over time

Grid Structure:
--------------
The SOM organizes neurons in an N-dimensional grid:
- 1D: Linear chain of neurons (5,) -> 5 neurons in a line
- 2D: Rectangular grid (10, 10) -> 100 neurons in a 10x10 plane  
- 3D: Cubic grid (5, 5, 5) -> 125 neurons in a 5x5x5 cube
- etc.

Each neuron has coordinates in the grid and a weight vector with the same
dimensionality as the input.

Competitive Learning:
-------------------
For an input x, the best matching unit (BMU) is the neuron with weight vector w_bmu that
minimizes the Euclidean distance:

    BMU = argmin_i ||x - w_i||^2

Neighborhood Functions:
---------------------
1. Gaussian neighborhood:
   h_i = exp(-d^2/(2*sigma^2))
   where d = distance from neuron i to the BMU

2. Bubble neighborhood:
   h_i = 1 if d <= sigma, 0 otherwise
   where d = distance from neuron i to the BMU

Weight Update:
------------
For each neuron i, the weight update is:

    w_i(t+1) = w_i(t) + α(t) * h_i(t) * (x - w_i(t))

Where:
    - α(t) is the learning rate at time t
    - h_i(t) is the neighborhood function value for neuron i at time t
    - Both α(t) and h_i(t) decrease over time, leading to fine-tuning

Applications:
-----------
- Dimensionality reduction and visualization
- Clustering and pattern recognition
- Feature extraction and organization
- Associative memory systems
- Anomaly detection
- Topological data analysis

Memory Properties:
----------------
SOMs function as content-addressable memory where:
- Each neuron stores a prototype pattern
- Similar inputs activate the same or nearby memory locations
- The topology preserves relationships between stored memories
- Retrieval works by finding the best matching memory unit
- The memory has generalization capabilities for novel inputs

Notes
-----
- Weight vectors are updated manually during training, not through backpropagation
- Both learning rate and neighborhood radius decrease over training iterations
- The SOM preserves topological relationships, not exact distances
- The output provides BMU coordinates (grid position) and quantization error
- This implementation is fully vectorized for efficient training on modern hardware

Examples
--------
>>> # Create a 10x10 SOM for 784-dimensional MNIST digits
>>> som_layer = SOMLayer(grid_shape=(10, 10), input_dim=784,
...                     initial_learning_rate=0.5, sigma=2.0)
>>>
>>> # Forward pass (finding BMUs)
>>> bmu_indices, quant_errors = som_layer(input_data, training=True)
>>>
>>> # The grid can be visualized to see the organization of the memory space
>>> weights_grid = som_layer.get_weights_map()

>>> # Create a 1D SOM for time series clustering
>>> som_1d = SOMLayer(grid_shape=(50,), input_dim=100)
>>>
>>> # Create a 3D SOM for complex data organization
>>> som_3d = SOMLayer(grid_shape=(8, 8, 8), input_dim=512)

References
----------
[1] Kohonen, T. (1982). Self-organized formation of topologically correct feature
       maps. Biological Cybernetics, 43(1), 59-69.

[2] Kohonen, T. (1990). The self-organizing map. Proceedings of the IEEE, 78(9),
       1464-1480.

[3] Kohonen, T. (2001). Self-Organizing Maps. Springer Series in Information
       Sciences, Vol. 30, Springer, Berlin.

[4] Ultsch, A., & Siemon, H. P. (1990). Kohonen's Self Organizing Feature Maps for
       Exploratory Data Analysis. In Proceedings of International Neural Networks
       Conference (INNC).

[5] Vesanto, J., & Alhoniemi, E. (2000). Clustering of the self-organizing map.
       IEEE Transactions on Neural Networks, 11(3), 586-600.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Union, Dict, Any, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SOMLayer(keras.layers.Layer):
    """
    N-Dimensional Self-Organizing Map (SOM) layer implementation for Keras.

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

    Parameters
    ----------
    grid_shape : Tuple[int, ...]
        The shape of the SOM neuron grid. For example, `(10, 10)` for a 2D grid 
        or `(5, 5, 5)` for a 3D grid.
    input_dim : int
        The dimensionality of the input data vectors.
    initial_learning_rate : float, optional
        The starting learning rate for weight updates. Defaults to 0.1.
    decay_function : Callable, optional
        Optional callable that takes the current iteration and max iterations 
        and returns a new learning rate. If `None`, a linear decay is used. 
        Defaults to `None`.
    sigma : float, optional
        The initial radius of the neighborhood function. Defaults to 1.0.
    neighborhood_function : str, optional
        The type of neighborhood function to use. Can be either `'gaussian'` 
        or `'bubble'`. Defaults to `'gaussian'`.
    weights_initializer : str or keras.initializers.Initializer, optional
        Initialization method for the SOM weights. Supports standard Keras
        initializers as well as special strings `'random'` (uniform in [0, 1]) 
        and `'sample'` (falls back to `'random'`). Defaults to `'random'`.
    regularizer : keras.regularizers.Regularizer, optional
        Optional regularizer function applied to the weights. Defaults to `None`.
    name : str, optional
        The name of the layer. Defaults to `None`.
    **kwargs : Any
        Additional keyword arguments for the base layer.

    Input shape:
        A 2D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
        A tuple of two tensors:
        - `bmu_indices`: `(batch_size, len(grid_shape))` containing integer
          coordinates of the BMU for each input.
        - `quantization_errors`: `(batch_size,)` containing the Euclidean
          distance between each input and its BMU's weights.

    Returns
    -------
    Tuple[keras.KerasTensor, keras.KerasTensor]
        A tuple containing:
        - BMU coordinates of shape (batch_size, len(grid_shape))
        - Quantization error of shape (batch_size,)

    Raises
    ------
    ValueError
        If `grid_shape` contains non-positive integers.
    ValueError
        If `input_dim` is not a positive integer.
    ValueError
        If `initial_learning_rate` is not positive.
    ValueError
        If `sigma` is not positive.
    ValueError
        If `neighborhood_function` is not 'gaussian' or 'bubble'.

    Example
    -------
    >>> # Create a 10x10 SOM for 784-dimensional MNIST data
    >>> som_layer = SOMLayer(grid_shape=(10, 10), input_dim=784)
    >>> input_data = keras.random.uniform((128, 784))
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
            **kwargs: Any
    ) -> None:
        """Initialize the SOM layer."""
        super().__init__(name=name, **kwargs)

        # Validation
        if not all(isinstance(d, int) and d > 0 for d in grid_shape):
            raise ValueError("`grid_shape` must be a tuple of positive integers.")
        if input_dim <= 0:
            raise ValueError("`input_dim` must be a positive integer.")
        if initial_learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {initial_learning_rate}")
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        if neighborhood_function not in ['gaussian', 'bubble']:
            raise ValueError(f"Neighborhood function must be 'gaussian' or 'bubble', got {neighborhood_function}")

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

        # Define custom decay function if none provided
        if decay_function is None:
            self.decay_function = lambda x, max_iter: self.initial_learning_rate * (1 - x / max_iter)
        else:
            self.decay_function = decay_function

        # Initialize weights to None (will be created in build)
        self.weights_map = None
        self.iterations = None
        self.max_iterations = None
        self.grid_positions = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple) -> None:
        """
        Build the SOM layer by initializing the weight vectors.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Convert input_shape to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Verify input shape
        if len(input_shape_list) != 2 or input_shape_list[-1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch_size, {self.input_dim}), "
                f"but received input_shape={input_shape}"
            )

        # Training iterations counter
        self.iterations = self.add_weight(
            name="iterations",
            shape=(),
            dtype="float32",
            initializer="zeros",
            trainable=False
        )

        self.max_iterations = self.add_weight(
            name="max_iterations",
            shape=(),
            dtype="float32",
            initializer=keras.initializers.Constant(1000.0),
            trainable=False
        )

        # Handle weight initialization
        if self.weights_initializer is None:
            # Handle special strings 'random' and 'sample'.
            # 'sample' requires input data, which isn't available here, so it
            # falls back to random uniform, matching original behavior.
            initializer = keras.initializers.RandomUniform(minval=0.0, maxval=1.0, seed=42)
        else:
            initializer = self.weights_initializer

        self.weights_map = self.add_weight(
            name="som_weights",
            shape=(*self.grid_shape, self.input_dim),
            initializer=initializer,
            regularizer=self.regularizer,
            trainable=False  # Weights are updated manually
        )

        # Initialize grid positions
        self.grid_positions = self._initialize_grid_positions()

        super().build(input_shape)

        logger.info(f"Built SOMLayer with grid_shape={self.grid_shape}, input_dim={self.input_dim}")

    def _initialize_grid_positions(self) -> keras.KerasTensor:
        """
        Initialize the N-dimensional grid positions for the SOM.

        Returns
        -------
        keras.KerasTensor
            A tensor of shape (*grid_shape, len(grid_shape)) containing
            the coordinates of each neuron in the N-dimensional grid.
        """
        coord_ranges = [ops.cast(ops.arange(d), "float32") for d in self.grid_shape]
        mesh_coords = ops.meshgrid(*coord_ranges, indexing='ij')
        position_grid = ops.stack(mesh_coords, axis=-1)
        return position_grid

    def call(self,
             inputs: keras.KerasTensor,
             training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass for the SOM layer.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Boolean indicating whether the layer should behave in
            training mode or inference mode.

        Returns
        -------
        Tuple[keras.KerasTensor, keras.KerasTensor]
            A tuple containing:
            - BMU coordinates of shape (batch_size, len(grid_shape))
            - Quantization error of shape (batch_size,)
        """
        # Find the Best Matching Units (BMUs) for each input
        bmu_indices, quantization_errors = self._find_bmu(inputs)

        # If in training mode, update the weights
        if training:
            self._update_weights(inputs, bmu_indices)
            self.iterations.assign_add(ops.cast(ops.shape(inputs)[0], "float32"))

        # Apply regularization if specified
        if self.regularizer is not None:
            self.add_loss(self.regularizer(self.weights_map))

        return bmu_indices, quantization_errors

    def _find_bmu(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Find the Best Matching Unit (BMU) for each input vector.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        Tuple[keras.KerasTensor, keras.KerasTensor]
            A tuple containing:
            - BMU coordinates of shape (batch_size, len(grid_shape))
            - Quantization error of shape (batch_size,)
        """
        # Reshape weights to [total_neurons, input_dim]
        flat_weights = ops.reshape(self.weights_map, (-1, self.input_dim))

        # Compute distances between inputs and all neurons
        # We use squared Euclidean distance for efficiency
        squared_distances = ops.sum(
            ops.square(ops.expand_dims(inputs, 1) - ops.expand_dims(flat_weights, 0)),
            axis=2
        )

        # Find the index of the minimum distance (BMU)
        bmu_flat_indices = ops.argmin(squared_distances, axis=1)

        # Convert flat indices to N-dimensional grid coordinates
        bmu_indices = ops.unravel_index(bmu_flat_indices, self.grid_shape)
        bmu_indices = ops.stack(bmu_indices, axis=1)

        # Compute the quantization error (minimum distance)
        min_distances = ops.min(squared_distances, axis=1)
        quantization_errors = ops.sqrt(min_distances)

        return ops.cast(bmu_indices, "int32"), quantization_errors

    def _update_weights(self, inputs: keras.KerasTensor, bmu_indices: keras.KerasTensor) -> None:
        """
        Update the SOM weights using a vectorized learning rule.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        bmu_indices : keras.KerasTensor
            BMU coordinates of shape (batch_size, len(grid_shape)).
        """
        # Update learning rate and sigma based on iteration
        current_learning_rate = self.decay_function(self.iterations, self.max_iterations)
        current_sigma = self.sigma * (1.0 - self.iterations / self.max_iterations)
        current_sigma = ops.maximum(current_sigma, 1e-4)  # Prevent division by zero

        bmu_coords = ops.cast(bmu_indices, dtype="float32")

        # Expand dimensions for broadcasting
        bmu_coords_expanded = ops.reshape(
            bmu_coords, [ops.shape(inputs)[0]] + [1] * self.grid_dim + [self.grid_dim]
        )
        grid_pos_expanded = ops.expand_dims(self.grid_positions, axis=0)

        # Compute neighborhood values for all neurons for each BMU in the batch
        squared_dist_to_bmus = ops.sum(
            ops.square(grid_pos_expanded - bmu_coords_expanded), axis=-1
        )

        if self.neighborhood_function == 'gaussian':
            neighborhood = ops.exp(-squared_dist_to_bmus / (2 * ops.square(current_sigma)))
        else:  # 'bubble'
            dist_to_bmus = ops.sqrt(squared_dist_to_bmus)
            neighborhood = ops.cast(dist_to_bmus <= current_sigma, "float32")

        # Compute the weight update delta for each input
        neighborhood_expanded = ops.expand_dims(neighborhood, axis=-1)
        inputs_expanded = ops.reshape(
            inputs, [ops.shape(inputs)[0]] + [1] * self.grid_dim + [self.input_dim]
        )
        delta_per_input = neighborhood_expanded * (inputs_expanded - ops.expand_dims(self.weights_map, 0))

        # Sum the deltas over the batch and apply learning rate
        weight_update = current_learning_rate * ops.sum(delta_per_input, axis=0)
        self.weights_map.assign_add(weight_update)

    def get_weights_map(self) -> keras.KerasTensor:
        """
        Get the current weights organized as an N-dimensional map.

        Returns
        -------
        keras.KerasTensor
            The SOM weights as an N-dimensional grid of shape 
            (*grid_shape, input_dim).
        """
        return self.weights_map

    def compute_output_shape(self, input_shape: Tuple) -> Tuple[Tuple, Tuple]:
        """
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor.

        Returns
        -------
        Tuple[Tuple, Tuple]
            A tuple containing:
            - BMU coordinates shape: (batch_size, len(grid_shape))
            - Quantization error shape: (batch_size,)
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)
        batch_size = input_shape_list[0]

        bmu_shape = tuple([batch_size, self.grid_dim])
        error_shape = tuple([batch_size])

        return bmu_shape, error_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for the layer.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary for the layer.
        """
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

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for the layer.

        Returns
        -------
        Dict[str, Any]
            Build configuration dictionary.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from a configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Build configuration dictionary.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SOMLayer":
        """
        Create a layer from its configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        SOMLayer
            A new SOMLayer instance.
        """
        # Handle complex object deserialization
        if 'weights_initializer' in config:
            if isinstance(config['weights_initializer'], dict):
                config['weights_initializer'] = keras.initializers.deserialize(
                    config['weights_initializer']
                )
        if 'regularizer' in config:
            if isinstance(config['regularizer'], dict):
                config['regularizer'] = keras.regularizers.deserialize(
                    config['regularizer']
                )

        return cls(**config)

# ---------------------------------------------------------------------