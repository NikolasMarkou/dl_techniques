"""
Self-Organizing Map (SOM) layer implementation for Keras.

This file should be saved as: dl_techniques/layers/som_2d_layer.py

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
from keras import ops
from typing import Tuple, Optional, Union, Dict, Any, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
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
        Initialization method for weights. Defaults to 'random_uniform'.
    regularizer : keras.regularizers.Regularizer, optional
        Regularizer function applied to the weights. Defaults to None.
    name : str, optional
        Name of the layer. Defaults to None.
    **kwargs : Any
        Additional keyword arguments for the base layer.
    """

    def __init__(
        self,
        map_size: Tuple[int, int],
        input_dim: int,
        initial_learning_rate: float = 0.1,
        decay_function: Optional[Callable] = None,
        sigma: float = 1.0,
        neighborhood_function: str = 'gaussian',
        weights_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
        regularizer: Optional[keras.regularizers.Regularizer] = None,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the SOM layer."""
        super().__init__(name=name, **kwargs)

        self.map_size = map_size
        self.grid_height, self.grid_width = map_size
        self.input_dim = input_dim
        self.initial_learning_rate = initial_learning_rate
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function
        self.weights_initializer = keras.initializers.get(weights_initializer)
        self.regularizer = keras.regularizers.get(regularizer)

        # Validate inputs
        if self.grid_height <= 0 or self.grid_width <= 0:
            raise ValueError(f"Map size must be positive, got {map_size}")
        if self.input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {input_dim}")
        if self.initial_learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {initial_learning_rate}")
        if self.sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        if self.neighborhood_function not in ['gaussian', 'bubble']:
            raise ValueError(f"Neighborhood function must be 'gaussian' or 'bubble', got {neighborhood_function}")

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
        if input_shape_list[-1] != self.input_dim:
            raise ValueError(f"Expected input shape with dimension {self.input_dim}, "
                           f"but got input shape with dimension {input_shape_list[-1]}")

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

        # Initialize SOM weights
        self.weights_map = self.add_weight(
            name="som_weights",
            shape=(self.grid_height, self.grid_width, self.input_dim),
            initializer=self.weights_initializer,
            regularizer=self.regularizer,
            trainable=False  # SOM weights are updated manually, not through backprop
        )

        # Initialize grid positions
        self.grid_positions = self._initialize_grid_positions()

        super().build(input_shape)

        logger.info(f"Built SOM2dLayer with map_size={self.map_size}, input_dim={self.input_dim}")

    def _initialize_grid_positions(self) -> keras.KerasTensor:
        """
        Initialize the 2D grid positions for the SOM.

        Returns
        -------
        keras.KerasTensor
            A tensor of shape (grid_height, grid_width, 2) containing
            the (y, x) coordinates of each neuron in the grid.
        """
        # Create coordinate ranges
        x_coords = ops.cast(ops.arange(self.grid_width), "float32")
        y_coords = ops.cast(ops.arange(self.grid_height), "float32")

        # Create meshgrid using numpy-style operations
        x_grid = ops.broadcast_to(
            ops.reshape(x_coords, (1, self.grid_width)),
            (self.grid_height, self.grid_width)
        )
        y_grid = ops.broadcast_to(
            ops.reshape(y_coords, (self.grid_height, 1)),
            (self.grid_height, self.grid_width)
        )

        # Stack to get (y, x) coordinates for each position
        position_grid = ops.stack([y_grid, x_grid], axis=-1)

        return position_grid

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
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
            - BMU coordinates of shape (batch_size, 2)
            - Quantization error of shape (batch_size,)
        """
        # Find the Best Matching Units (BMUs) for each input
        bmu_indices, quantization_errors = self._find_bmu(inputs)

        # If in training mode, update the weights
        if training:
            self._update_weights(inputs, bmu_indices)
            # Update iterations counter
            self.iterations.assign_add(1.0)

        # Apply regularization if specified
        if self.regularizer is not None:
            self.add_loss(self.regularizer(self.weights_map))

        return bmu_indices, quantization_errors

    def _find_bmu(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Find the Best Matching Unit (BMU) for each input.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        Tuple[keras.KerasTensor, keras.KerasTensor]
            A tuple containing:
            - BMU coordinates of shape (batch_size, 2)
            - Quantization error of shape (batch_size,)
        """
        # Reshape weights to [grid_height * grid_width, input_dim]
        flat_weights = ops.reshape(self.weights_map, (-1, self.input_dim))

        # Compute distances between inputs and all neurons
        # We use squared Euclidean distance for efficiency
        expanded_inputs = ops.expand_dims(inputs, 1)  # [batch_size, 1, input_dim]
        expanded_weights = ops.expand_dims(flat_weights, 0)  # [1, grid_height * grid_width, input_dim]

        # Compute squared distances
        squared_diff = ops.square(expanded_inputs - expanded_weights)
        squared_distances = ops.sum(squared_diff, axis=2)

        # Find the index of the minimum distance (BMU)
        bmu_flat_indices = ops.argmin(squared_distances, axis=1)

        # Convert flat indices to 2D grid coordinates
        bmu_y = ops.cast(bmu_flat_indices // self.grid_width, "int32")
        bmu_x = ops.cast(bmu_flat_indices % self.grid_width, "int32")
        bmu_indices = ops.stack([bmu_y, bmu_x], axis=1)

        # Compute the quantization error (minimum distance)
        batch_size = ops.shape(inputs)[0]
        quantization_errors = ops.sqrt(
            ops.take_along_axis(
                squared_distances,
                ops.expand_dims(bmu_flat_indices, axis=1),
                axis=1
            )
        )
        quantization_errors = ops.squeeze(quantization_errors, axis=1)

        return bmu_indices, quantization_errors

    def _update_weights(self, inputs: keras.KerasTensor, bmu_indices: keras.KerasTensor) -> None:
        """
        Update the weights of the SOM using the Kohonen learning rule.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        bmu_indices : keras.KerasTensor
            BMU coordinates of shape (batch_size, 2).
        """
        # Update learning rate based on iteration
        current_learning_rate = self.decay_function(
            self.iterations, self.max_iterations
        )

        # Sigma also decreases with iterations
        current_sigma = self.sigma * (1.0 - self.iterations / self.max_iterations)
        current_sigma = ops.maximum(current_sigma, 0.01)  # Don't let it go to zero

        # For SOM, we typically process one sample at a time to maintain proper learning
        # In batch processing, we'll use the first sample in the batch for weight updates
        # This is a common approach in SOM implementations

        current_input = inputs[0]  # Use first sample in batch
        bmu_coord = ops.cast(bmu_indices[0], "float32")  # Use corresponding BMU

        # Compute neighborhood values for all neurons
        if self.neighborhood_function == 'gaussian':
            # Calculate squared distances from all neurons to BMU
            bmu_expanded = ops.reshape(bmu_coord, (1, 1, 2))
            squared_dist = ops.sum(
                ops.square(self.grid_positions - bmu_expanded),
                axis=2
            )

            # Compute Gaussian neighborhood
            neighborhood = ops.exp(-squared_dist / (2 * ops.square(current_sigma)))
            neighborhood = ops.expand_dims(neighborhood, axis=-1)

        elif self.neighborhood_function == 'bubble':
            # Calculate distances from all neurons to BMU
            bmu_expanded = ops.reshape(bmu_coord, (1, 1, 2))
            dist = ops.sqrt(ops.sum(
                ops.square(self.grid_positions - bmu_expanded),
                axis=2
            ))

            # Compute bubble neighborhood (1 within radius, 0 outside)
            neighborhood = ops.cast(dist <= current_sigma, "float32")
            neighborhood = ops.expand_dims(neighborhood, axis=-1)

        # Compute the weight update
        weighted_input = ops.reshape(current_input, (1, 1, self.input_dim))
        weight_update = current_learning_rate * neighborhood * (weighted_input - self.weights_map)

        # Apply the update
        self.weights_map.assign_add(weight_update)

    def get_weights_as_grid(self) -> keras.KerasTensor:
        """
        Get the current weights organized as a grid.

        Returns
        -------
        keras.KerasTensor
            The SOM weights as a grid of shape (grid_height, grid_width, input_dim).
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
            - BMU coordinates shape: (batch_size, 2)
            - Quantization error shape: (batch_size,)
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)
        batch_size = input_shape_list[0]

        bmu_shape = tuple([batch_size, 2])
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
            'map_size': self.map_size,
            'input_dim': self.input_dim,
            'initial_learning_rate': self.initial_learning_rate,
            'sigma': self.sigma,
            'neighborhood_function': self.neighborhood_function,
            'weights_initializer': keras.initializers.serialize(self.weights_initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
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
    def from_config(cls, config: Dict[str, Any]) -> "SOM2dLayer":
        """
        Create a layer from its configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        SOM2dLayer
            A new SOM2dLayer instance.
        """
        # Deserialize complex objects
        if 'weights_initializer' in config:
            config['weights_initializer'] = keras.initializers.deserialize(
                config['weights_initializer']
            )
        if 'regularizer' in config:
            config['regularizer'] = keras.regularizers.deserialize(
                config['regularizer']
            )

        return cls(**config)

# ---------------------------------------------------------------------
