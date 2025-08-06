"""
Hierarchical Memory System using Multiple Self-Organizing Map Layers.

A hierarchical memory system that organizes information at multiple levels of abstraction
using a cascade of Self-Organizing Maps (SOMs). The system processes input data through
multiple levels, from abstract (small grids capturing broad patterns) to fine-grained
(large grids capturing detailed patterns).

The hierarchy works as follows:
- Level 1 (Abstract): Small grid size, captures broad categories and patterns
- Level 2 (Intermediate): Medium grid size, captures more detailed patterns
- Level N (Fine-grained): Large grid size, captures fine-detailed patterns

Each level in the hierarchy processes the same input data but with different grid
resolutions, allowing the system to capture multi-scale patterns and organize
memory at different levels of granularity.

Architecture:
------------
Input → Level 1 SOM (abstract, small grid)
     → Level 2 SOM (intermediate, medium grid)
     → Level N SOM (fine-grained, large grid)

Memory Organization:
------------------
- Abstract levels capture general categories and broad patterns
- Fine-grained levels capture specific details and nuanced patterns
- The hierarchy preserves both global structure and local details
- Each level maintains its own topological organization

Applications:
-----------
- Multi-scale pattern recognition
- Hierarchical clustering and classification
- Content-addressable memory with multiple levels of detail
- Associative memory systems with abstraction hierarchies
- Feature extraction at multiple scales

References
----------
[1] Kohonen, T. (2001). Self-Organizing Maps. Springer.
[2] Miikkulainen, R. (1990). Hierarchical feature maps. In Parallel Models of
    Associative Memory (pp. 143-186).
[3] Ultsch, A. (2003). Emergent self-organizing feature maps used for prediction
    and classification of multivariate data. In Emergent Neural Computational
    Architectures Based on Neuroscience (pp. 301-317).
"""

import keras
from typing import Tuple, Optional, Union, Dict, Any, Callable, List

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..som_nd_layer import SOMLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalMemorySystem(keras.layers.Layer):
    """
    Hierarchical memory system using multiple Self-Organizing Map layers.

    Creates a hierarchy of SOM layers that process input data at multiple levels
    of abstraction, from abstract (small grids) to fine-grained (large grids).
    Each level captures different scales of patterns in the input data, providing
    a multi-resolution memory organization.

    The system organizes memory hierarchically where:
    - Abstract levels (smaller grids) capture broad, general patterns
    - Fine-grained levels (larger grids) capture detailed, specific patterns
    - Each level maintains topological organization of its input space
    - All levels process the same input but at different resolutions

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input data vectors.
    levels : int, optional
        The number of hierarchy levels in the memory system. Defaults to 3.
    grid_dimensions : int, optional
        The number of dimensions for each SOM grid (e.g., 2 for 2D grids).
        Defaults to 2.
    base_grid_size : int, optional
        The grid size for the most abstract level (Level 1). Subsequent levels
        will have larger grids based on the expansion factor. Defaults to 5.
    grid_expansion_factor : float, optional
        Factor by which the grid size increases at each level. For example,
        with base_grid_size=5 and expansion_factor=2.0:
        Level 1: 5x5, Level 2: 10x10, Level 3: 20x20. Defaults to 2.0.
    initial_learning_rate : float, optional
        The starting learning rate for all SOM layers. Defaults to 0.1.
    decay_function : Callable, optional
        Optional callable for learning rate decay. Applied to all SOM layers.
        Defaults to None (uses linear decay).
    sigma : float, optional
        The initial neighborhood radius for all SOM layers. Defaults to 1.0.
    neighborhood_function : str, optional
        The neighborhood function type ('gaussian' or 'bubble') for all SOM
        layers. Defaults to 'gaussian'.
    weights_initializer : str or keras.initializers.Initializer, optional
        Weight initialization method for all SOM layers. Defaults to 'random'.
    regularizer : keras.regularizers.Regularizer, optional
        Optional regularizer applied to all SOM layers. Defaults to None.
    name : str, optional
        Name of the layer. Defaults to None.
    **kwargs : Any
        Additional keyword arguments for the base layer.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
        A tuple containing results from each hierarchy level:
        - bmu_indices_list: List of (batch_size, grid_dimensions) tensors
        - quantization_errors_list: List of (batch_size,) tensors

    Returns
    -------
    Tuple[List[keras.KerasTensor], List[keras.KerasTensor]]
        A tuple containing:
        - List of BMU coordinate tensors, one per level
        - List of quantization error tensors, one per level

    Raises
    ------
    ValueError
        If input_dim is not a positive integer.
    ValueError
        If levels is less than 1.
    ValueError
        If grid_dimensions is less than 1.
    ValueError
        If base_grid_size is less than 1.
    ValueError
        If grid_expansion_factor is not positive.

    Example
    -------
    >>> # Create a 3-level hierarchical memory system
    >>> hierarchical_memory = HierarchicalMemorySystem(
    ...     input_dim=784,  # For MNIST
    ...     levels=3,
    ...     base_grid_size=5,
    ...     grid_expansion_factor=2.0
    ... )
    >>>
    >>> # Process input data
    >>> input_data = keras.random.uniform((128, 784))
    >>> bmu_coords_list, q_errors_list = hierarchical_memory(input_data, training=True)
    >>>
    >>> # Level 1 (abstract): 5x5 grid
    >>> print(bmu_coords_list[0].shape)  # (128, 2)
    >>> # Level 2 (intermediate): 10x10 grid
    >>> print(bmu_coords_list[1].shape)  # (128, 2)
    >>> # Level 3 (fine-grained): 20x20 grid
    >>> print(bmu_coords_list[2].shape)  # (128, 2)
    """

    def __init__(
            self,
            input_dim: int,
            levels: int = 3,
            grid_dimensions: int = 2,
            base_grid_size: int = 5,
            grid_expansion_factor: float = 2.0,
            initial_learning_rate: float = 0.1,
            decay_function: Optional[Callable] = None,
            sigma: float = 1.0,
            neighborhood_function: str = 'gaussian',
            weights_initializer: Union[str, keras.initializers.Initializer] = 'random',
            regularizer: Optional[keras.regularizers.Regularizer] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the hierarchical memory system."""
        super().__init__(name=name, **kwargs)

        # Validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if levels < 1:
            raise ValueError(f"levels must be at least 1, got {levels}")
        if grid_dimensions < 1:
            raise ValueError(f"grid_dimensions must be at least 1, got {grid_dimensions}")
        if base_grid_size < 1:
            raise ValueError(f"base_grid_size must be at least 1, got {base_grid_size}")
        if grid_expansion_factor <= 0:
            raise ValueError(f"grid_expansion_factor must be positive, got {grid_expansion_factor}")

        # Store configuration
        self.input_dim = input_dim
        self.levels = levels
        self.grid_dimensions = grid_dimensions
        self.base_grid_size = base_grid_size
        self.grid_expansion_factor = grid_expansion_factor
        self.initial_learning_rate = initial_learning_rate
        self.decay_function = decay_function
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer

        # Will be initialized in build()
        self.som_layers = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple) -> None:
        """
        Build the hierarchical memory system by creating all SOM layers.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Create SOM layers for each hierarchy level
        self.som_layers = []

        for level in range(self.levels):
            # Calculate grid size for this level
            current_grid_size = int(self.base_grid_size * (self.grid_expansion_factor ** level))
            grid_shape = tuple([current_grid_size] * self.grid_dimensions)

            # Create SOM layer for this level
            som_layer = SOMLayer(
                grid_shape=grid_shape,
                input_dim=self.input_dim,
                initial_learning_rate=self.initial_learning_rate,
                decay_function=self.decay_function,
                sigma=self.sigma,
                neighborhood_function=self.neighborhood_function,
                weights_initializer=self.weights_initializer,
                regularizer=self.regularizer,
                name=f"som_level_{level + 1}"
            )

            # Build the SOM layer
            som_layer.build(input_shape)
            self.som_layers.append(som_layer)

        super().build(input_shape)

        logger.info(f"Built HierarchicalMemorySystem with {self.levels} levels, "
                    f"grid sizes: {[som.grid_shape for som in self.som_layers]}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[List[keras.KerasTensor], List[keras.KerasTensor]]:
        """
        Forward pass through the hierarchical memory system.

        Processes the input through all hierarchy levels, from abstract to
        fine-grained. Each level finds its Best Matching Units (BMUs) and
        updates its weights if in training mode.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Boolean indicating whether to perform weight updates.

        Returns
        -------
        Tuple[List[keras.KerasTensor], List[keras.KerasTensor]]
            A tuple containing:
            - List of BMU coordinate tensors from each level
            - List of quantization error tensors from each level
        """
        bmu_indices_list = []
        quantization_errors_list = []

        # Process input through each hierarchy level
        for level_idx, som_layer in enumerate(self.som_layers):
            bmu_indices, quantization_errors = som_layer(inputs, training=training)

            bmu_indices_list.append(bmu_indices)
            quantization_errors_list.append(quantization_errors)

        return bmu_indices_list, quantization_errors_list

    def get_level_weights(self, level: int) -> keras.KerasTensor:
        """
        Get the weight map for a specific hierarchy level.

        Parameters
        ----------
        level : int
            The hierarchy level (0-indexed) to get weights for.

        Returns
        -------
        keras.KerasTensor
            The weight map for the specified level.

        Raises
        ------
        ValueError
            If level is out of range.
        """
        if level < 0 or level >= self.levels:
            raise ValueError(f"Level must be between 0 and {self.levels - 1}, got {level}")

        return self.som_layers[level].get_weights_map()

    def get_all_weights(self) -> List[keras.KerasTensor]:
        """
        Get weight maps for all hierarchy levels.

        Returns
        -------
        List[keras.KerasTensor]
            List of weight maps, one for each hierarchy level.
        """
        return [som_layer.get_weights_map() for som_layer in self.som_layers]

    def get_grid_shapes(self) -> List[Tuple[int, ...]]:
        """
        Get the grid shapes for all hierarchy levels.

        Returns
        -------
        List[Tuple[int, ...]]
            List of grid shapes, one for each hierarchy level.
        """
        return [som_layer.grid_shape for som_layer in self.som_layers]

    def compute_output_shape(self, input_shape: Tuple) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Compute the output shape of the hierarchical memory system.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor.

        Returns
        -------
        Tuple[List[Tuple], List[Tuple]]
            A tuple containing:
            - List of BMU coordinate shapes for each level
            - List of quantization error shapes for each level
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)
        batch_size = input_shape_list[0]

        bmu_shapes = []
        error_shapes = []

        for level in range(self.levels):
            bmu_shape = tuple([batch_size, self.grid_dimensions])
            error_shape = tuple([batch_size])

            bmu_shapes.append(bmu_shape)
            error_shapes.append(error_shape)

        return bmu_shapes, error_shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for the hierarchical memory system.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'levels': self.levels,
            'grid_dimensions': self.grid_dimensions,
            'base_grid_size': self.base_grid_size,
            'grid_expansion_factor': self.grid_expansion_factor,
            'initial_learning_rate': self.initial_learning_rate,
            'decay_function': self.decay_function,
            'sigma': self.sigma,
            'neighborhood_function': self.neighborhood_function,
            'weights_initializer': self.weights_initializer,
            'regularizer': self.regularizer,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for the hierarchical memory system.

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
    def from_config(cls, config: Dict[str, Any]) -> "HierarchicalMemorySystem":
        """
        Create a hierarchical memory system from its configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        HierarchicalMemorySystem
            A new HierarchicalMemorySystem instance.
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
