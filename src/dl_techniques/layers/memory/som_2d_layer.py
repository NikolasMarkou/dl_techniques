"""
Self-Organizing Map (SOM) 2D layer implementation.

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
from typing import Tuple, Optional, Union, Dict, Any, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .som_nd_layer import SOMLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SOM2dLayer(SOMLayer):
    """
    2D Self-Organizing Map layer for competitive learning and topological organization.

    Specializes the general N-Dimensional ``SOMLayer`` for 2D grids, mapping
    high-dimensional input data onto a rectangular ``(H, W)`` neuron grid.
    For each input ``x``, the Best Matching Unit is found via
    ``BMU = argmin_{i,j} ||x - w_{i,j}||^2``, then the BMU and its neighbors
    are updated: ``w_{i,j} <- w_{i,j} + alpha * h_{i,j} * (x - w_{i,j})``
    where ``h_{i,j} = exp(-d^2/(2*sigma^2))`` for Gaussian neighborhood.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │             SOM2dLayer                   │
        │                                          │
        │  Input(batch, input_dim)                 │
        │         │                                │
        │         ▼                                │
        │  Distance: ||x - w_{i,j}||^2             │
        │         │                                │
        │         ▼                                │
        │  BMU = argmin_{i,j} (distances)          │
        │         │                                │
        │         ▼                                │
        │  Neighborhood: h = exp(-d^2 / 2*sigma^2) │
        │         │                                │
        │         ▼                                │
        │  Update: w += alpha * h * (x - w)        │
        │         │                                │
        │         ▼                                │
        │  Output: BMU_coords(batch,2),            │
        │          quant_errors(batch,)            │
        └──────────────────────────────────────────┘

    :param map_size: Shape of the 2D SOM grid ``(height, width)``. Must contain
        exactly 2 positive integers.
    :type map_size: Tuple[int, int]
    :param input_dim: Dimensionality of the input data. Must be positive.
    :type input_dim: int
    :param initial_learning_rate: Initial learning rate for weight updates.
        Defaults to 0.1.
    :type initial_learning_rate: float
    :param decay_function: Learning rate decay function. If None, uses linear decay.
        Defaults to None.
    :type decay_function: Optional[Callable]
    :param sigma: Initial neighborhood radius in grid coordinates. Defaults to 1.0.
    :type sigma: float
    :param neighborhood_function: Type of neighborhood function
        (``'gaussian'`` or ``'bubble'``). Defaults to ``'gaussian'``.
    :type neighborhood_function: str
    :param weights_initializer: Initialization method for neuron weights.
        Defaults to ``'random_uniform'``.
    :type weights_initializer: Union[str, keras.initializers.Initializer]
    :param regularizer: Regularizer applied to neuron weights. Defaults to None.
    :type regularizer: Optional[keras.regularizers.Regularizer]
    :param name: Name of the layer. Defaults to None.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the base Layer class.
    :type kwargs: Any
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
        """Initialize the 2D SOM layer with validation and configuration setup."""
        # Validate the 2D-specific input
        if not (isinstance(map_size, (tuple, list)) and len(map_size) == 2):
            raise ValueError(f"map_size must be a tuple of exactly 2 integers, got {map_size}")

        if not all(isinstance(dim, int) and dim > 0 for dim in map_size):
            raise ValueError(f"map_size must contain positive integers, got {map_size}")

        # Initialize the parent SOMLayer with grid_shape from map_size
        super().__init__(
            grid_shape=map_size,
            input_dim=input_dim,
            initial_learning_rate=initial_learning_rate,
            decay_function=decay_function,
            sigma=sigma,
            neighborhood_function=neighborhood_function,
            weights_initializer=weights_initializer,
            regularizer=regularizer,
            name=name,
            **kwargs
        )

        # Store map_size for backward compatibility and clear interface
        self.map_size = tuple(map_size)  # Ensure it's a tuple

    def get_weights_as_grid(self) -> keras.KerasTensor:
        """
        Get the current neuron weights organized as a 2D grid for visualization.

        Alias for ``get_weights_map()`` providing backward compatibility and
        intuitive naming for 2D SOMs.

        :return: SOM weights reshaped as ``(height, width, input_dim)``.
        :rtype: keras.KerasTensor
        """
        return self.get_weights_map()

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Uses ``map_size`` instead of ``grid_shape`` for backward compatibility
        with existing 2D SOM implementations.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        # Get the base configuration from parent SOMLayer
        config = super().get_config()

        # Replace 'grid_shape' with 'map_size' for 2D layer compatibility
        if 'grid_shape' in config:
            config['map_size'] = self.map_size
            del config['grid_shape']

        # Ensure proper serialization of initializer and regularizer
        # The parent class already handles this robustly, but we maintain
        # explicit serialization for clarity and debugging
        if hasattr(self, '_weights_initializer_config'):
            config['weights_initializer'] = keras.initializers.serialize(
                keras.initializers.get(self._weights_initializer_config)
            )

        if hasattr(self, '_regularizer_config') and self._regularizer_config is not None:
            config['regularizer'] = keras.regularizers.serialize(
                keras.regularizers.get(self._regularizer_config)
            )

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SOM2dLayer':
        """
        Create layer instance from configuration dictionary.

        :param config: Dictionary containing layer configuration parameters.
        :type config: Dict[str, Any]
        :return: SOM2dLayer instance reconstructed from the configuration.
        :rtype: SOM2dLayer
        """
        # Handle initializer deserialization
        if 'weights_initializer' in config:
            config['weights_initializer'] = keras.initializers.deserialize(
                config['weights_initializer']
            )

        # Handle regularizer deserialization
        if 'regularizer' in config and config['regularizer'] is not None:
            config['regularizer'] = keras.regularizers.deserialize(
                config['regularizer']
            )

        return cls(**config)

# ---------------------------------------------------------------------