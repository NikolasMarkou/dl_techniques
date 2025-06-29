"""
Self-Organizing Map (SOM) layer implementation for Keras.

A Self-Organizing Map is an unsupervised neural network that performs dimensionality
reduction through competitive learning and topological organization. The layer maps
high-dimensional input data onto a lower-dimensional grid of neurons, preserving
topological relationships between input patterns.

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
The SOM organizes neurons in a 2D grid:

    y
    ^
    |  (0,0)-----(0,1)-----(0,2)---...
    |    |         |         |
    |    |         |         |
    |  (1,0)-----(1,1)-----(1,2)---...
    |    |         |         |
    |    |         |         |
    |  (2,0)-----(2,1)-----(2,2)---...
    |    :         :         :
    |
    +------------------------------------> x

Each neuron (i,j) has a weight vector w_ij with the same dimensionality as the input.

Competitive Learning:
-------------------
For an input x, the best matching unit (BMU) is the neuron with weight vector w_bmu that
minimizes the Euclidean distance:

    BMU = argmin_ij ||x - w_ij||^2

Neighborhood Functions:
---------------------
1. Gaussian neighborhood:
   h_ij = exp(-d^2/(2*sigma^2))
   where d = distance from neuron (i,j) to the BMU

2. Bubble neighborhood:
   h_ij = 1 if d <= sigma, 0 otherwise
   where d = distance from neuron (i,j) to the BMU

The neighborhood function creates a "bubble" of activation around the BMU:

         Gaussian             |              Bubble
                              |
    ●●●●●●●●●●●●●●●           |            ●●●●●●●●
    ●●●●●●●●●●●●●●●           |            ●●●●●●●●
    ●●●●●●●●●●●●●●●           |            ●●●●●●●●
    ●●●●●●●○●●●●●●●           |            ●●●○○●●●
    ●●●●●○○○○○●●●●●           |            ●●●○○●●●
    ●●●●○○○○○○○●●●●           |            ●●●●●●●●
    ●●●○○○○X○○○○●●●           |            ●●●●●●●●
    ●●●●○○○○○○○●●●●           |            ●●●●●●●●
    ●●●●●○○○○○●●●●●           |
    ●●●●●●●○●●●●●●●           |     X = BMU, ○ = Activated neighbors
    ●●●●●●●●●●●●●●●           |     ● = Non-activated neurons
    ●●●●●●●●●●●●●●●           |

Weight Update:
------------
For each neuron (i,j), the weight update is:

    w_ij(t+1) = w_ij(t) + α(t) * h_ij(t) * (x - w_ij(t))

Where:
    - α(t) is the learning rate at time t
    - h_ij(t) is the neighborhood function value for neuron (i,j) at time t
    - Both α(t) and h_ij(t) decrease over time, leading to fine-tuning

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

Examples
--------
>>> # Create a 10x10 SOM for 784-dimensional MNIST digits
>>> som_layer = SOM2dLayer(map_size=(10, 10), input_dim=784,
...                     initial_learning_rate=0.5, sigma=2.0)
>>>
>>> # Forward pass (finding BMUs)
>>> bmu_indices, quant_errors = som_layer(input_data, training=True)
>>>
>>> # The grid can be visualized to see the organization of the memory space
>>> weights_grid = som_layer.get_weights_as_grid()

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
import tensorflow as tf
from typing import Tuple, Optional, Union, Dict, Any, Callable

@keras.utils.register_keras_serializable()
class SOM2dLayer(keras.layers.Layer):
    """
    Self-Organizing Map layer implementation for Keras.

    This layer implements a Self-Organizing Map (SOM) which can be used
    as a memory structure to map high-dimensional input data onto
    a lower-dimensional grid. The SOM preserves the topological properties
    of the input space.

    Parameters
    ----------
    map_size : Tuple[int, int]
        Size of the SOM grid (height, width).
    input_dim : int
        Dimensionality of the input data.
    initial_learning_rate : float, optional
        Initial learning rate for weight updates. Defaults to 0.1.
    decay_function : Callable, optional
        Learning rate decay function. Defaults to None.
    sigma : float, optional
        Initial neighborhood radius. Defaults to 1.0.
    neighborhood_function : str, optional
        Type of neighborhood function to use ('gaussian' or 'bubble').
        Defaults to 'gaussian'.
    weights_initializer : str or keras.initializers.Initializer, optional
        Initialization method for weights. Defaults to 'random'.
    regularizer : keras.regularizers.Regularizer, optional
        Regularizer function applied to the weights. Defaults to None.
    name : str, optional
        Name of the layer. Defaults to None.
    """

    def __init__(
            self,
            map_size: Tuple[int, int],
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
        """Initialize the SOM layer."""
        super(SOM2dLayer, self).__init__(name=name, **kwargs)

        self.map_size = map_size
        self.grid_height, self.grid_width = map_size
        self.input_dim = input_dim
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer

        # Define custom decay function if none provided
        if decay_function is None:
            self.decay_function = lambda x, max_iter: self.initial_learning_rate * (1 - x / max_iter)
        else:
            self.decay_function = decay_function

        # Training iterations counter
        self.iterations = self.add_weight(
            name="iterations",
            shape=(),
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        self.max_iterations = self.add_weight(
            name="max_iterations",
            shape=(),
            dtype=tf.float32,
            initializer=lambda shape, dtype: tf.constant(1000.0, dtype=dtype),
            trainable=False
        )

        # Initialize grid positions
        self.grid_positions = self._initialize_grid_positions()

    def build(self, input_shape: Tuple) -> None:
        """
        Build the SOM layer by initializing the weight vectors.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor.
        """
        # Verify input shape
        if input_shape[1] != self.input_dim:
            raise ValueError(f"Expected input shape with dimension {self.input_dim}, "
                             f"but got input shape with dimension {input_shape[1]}")

        # Initialize weights based on the chosen initializer
        if isinstance(self.weights_initializer, str):
            if self.weights_initializer == 'random':
                initial_weights = tf.random.uniform(
                    shape=(self.grid_height, self.grid_width, self.input_dim),
                    minval=0.0, maxval=1.0, seed=42
                )
            elif self.weights_initializer == 'sample':
                # This would initialize using random samples from input data
                # But we can't access input data here, so use random init
                initial_weights = tf.random.uniform(
                    shape=(self.grid_height, self.grid_width, self.input_dim),
                    minval=0.0, maxval=1.0, seed=42
                )
            else:
                raise ValueError(f"Unsupported weights_initializer: {self.weights_initializer}")

            # Create the weight variable
            self.weights_map = self.add_weight(
                name="som_weights",
                shape=(self.grid_height, self.grid_width, self.input_dim),
                initializer=lambda shape, dtype: initial_weights,
                regularizer=self.regularizer,
                trainable=False  # SOM weights are updated manually, not through backprop
            )
        else:
            # Use the provided initializer
            self.weights_map = self.add_weight(
                name="som_weights",
                shape=(self.grid_height, self.grid_width, self.input_dim),
                initializer=self.weights_initializer,
                regularizer=self.regularizer,
                trainable=False
            )

        super(SOM2dLayer, self).build(input_shape)

    def _initialize_grid_positions(self) -> tf.Tensor:
        """
        Initialize the 2D grid positions for the SOM.

        Returns
        -------
        tf.Tensor
            A tensor of shape (grid_height, grid_width, 2) containing
            the (y, x) coordinates of each neuron in the grid.
        """
        # Create a meshgrid of positions
        x_coords = tf.range(self.grid_width, dtype=tf.float32)
        y_coords = tf.range(self.grid_height, dtype=tf.float32)

        # Create a meshgrid
        y_grid, x_grid = tf.meshgrid(y_coords, x_coords, indexing='ij')

        # Stack to get (y, x) coordinates for each position
        position_grid = tf.stack([y_grid, x_grid], axis=-1)

        return position_grid

    def call(self, inputs: tf.Tensor, training: bool = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass for the SOM layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Boolean indicating whether the layer should behave in
            training mode or inference mode.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple containing:
            - BMU coordinates of shape (batch_size, 2)
            - Quantization error of shape (batch_size,)
        """
        # Find the Best Matching Units (BMUs) for each input
        bmu_indices, quantization_errors = self._find_bmu(inputs)

        # If in training mode, update the weights
        if training:
            self._update_weights(inputs, bmu_indices)
            self.iterations.assign_add(1.0)

        return bmu_indices, quantization_errors

    def _find_bmu(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Find the Best Matching Unit (BMU) for each input.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple containing:
            - BMU coordinates of shape (batch_size, 2)
            - Quantization error of shape (batch_size,)
        """
        # Reshape weights to [grid_height * grid_width, input_dim]
        flat_weights = tf.reshape(self.weights_map, [-1, self.input_dim])

        # Compute distances between inputs and all neurons
        # We use squared Euclidean distance for efficiency
        expanded_inputs = tf.expand_dims(inputs, 1)  # [batch_size, 1, input_dim]
        expanded_weights = tf.expand_dims(flat_weights, 0)  # [1, grid_height * grid_width, input_dim]

        # Compute squared distances
        squared_diff = tf.square(
            expanded_inputs - expanded_weights)  # [batch_size, grid_height * grid_width, input_dim]
        squared_distances = tf.reduce_sum(squared_diff, axis=2)  # [batch_size, grid_height * grid_width]

        # Find the index of the minimum distance (BMU)
        bmu_flat_indices = tf.argmin(squared_distances, axis=1)  # [batch_size]

        # Convert flat indices to 2D grid coordinates
        bmu_y = tf.cast(bmu_flat_indices // self.grid_width, tf.int32)
        bmu_x = tf.cast(bmu_flat_indices % self.grid_width, tf.int32)
        bmu_indices = tf.stack([bmu_y, bmu_x], axis=1)  # [batch_size, 2]

        # Compute the quantization error (minimum distance)
        quantization_errors = tf.gather(
            tf.sqrt(tf.reshape(squared_distances, [tf.shape(inputs)[0], -1])),
            bmu_flat_indices,
            batch_dims=1
        )

        return bmu_indices, quantization_errors

    def _update_weights(self, inputs: tf.Tensor, bmu_indices: tf.Tensor) -> None:
        """
        Update the weights of the SOM using the Kohonen learning rule.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, input_dim).
        bmu_indices : tf.Tensor
            BMU coordinates of shape (batch_size, 2).
        """
        # Update learning rate and sigma based on iteration
        current_learning_rate = self.decay_function(
            self.iterations, self.max_iterations
        )

        # Sigma also decreases with iterations
        current_sigma = self.sigma * (1.0 - self.iterations / self.max_iterations)
        current_sigma = tf.maximum(current_sigma, 0.01)  # Don't let it go to zero

        # Process each input in the batch
        for i in range(tf.shape(inputs)[0]):
            # Get the current input and its BMU
            current_input = inputs[i]
            bmu_coord = tf.cast(bmu_indices[i], tf.float32)

            # Compute neighborhood values for all neurons
            if self.neighborhood_function == 'gaussian':
                # Calculate squared distances from all neurons to BMU
                squared_dist = tf.reduce_sum(
                    tf.square(self.grid_positions - tf.reshape(bmu_coord, [1, 1, 2])),
                    axis=2
                )

                # Compute Gaussian neighborhood
                neighborhood = tf.exp(-squared_dist / (2 * tf.square(current_sigma)))
                neighborhood = tf.expand_dims(neighborhood, axis=-1)  # [grid_height, grid_width, 1]

            elif self.neighborhood_function == 'bubble':
                # Calculate distances from all neurons to BMU
                dist = tf.sqrt(tf.reduce_sum(
                    tf.square(self.grid_positions - tf.reshape(bmu_coord, [1, 1, 2])),
                    axis=2
                ))

                # Compute bubble neighborhood (1 within radius, 0 outside)
                neighborhood = tf.cast(dist <= current_sigma, tf.float32)
                neighborhood = tf.expand_dims(neighborhood, axis=-1)  # [grid_height, grid_width, 1]

            else:
                raise ValueError(f"Unsupported neighborhood_function: {self.neighborhood_function}")

            # Compute the weight update
            weighted_input = tf.reshape(current_input, [1, 1, self.input_dim])
            weight_update = current_learning_rate * neighborhood * (weighted_input - self.weights_map)

            # Apply the update
            self.weights_map.assign_add(weight_update)

    def get_weights_as_grid(self) -> tf.Tensor:
        """
        Get the current weights organized as a grid.

        Returns
        -------
        tf.Tensor
            The SOM weights as a grid of shape (grid_height, grid_width, input_dim).
        """
        return self.weights_map

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for the layer.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary for the layer.
        """
        config = super(SOM2dLayer, self).get_config()
        config.update({
            'map_size': self.map_size,
            'input_dim': self.input_dim,
            'initial_learning_rate': self.initial_learning_rate,
            'sigma': self.sigma,
            'neighborhood_function': self.neighborhood_function,
            'weights_initializer': self.weights_initializer
        })
        return config