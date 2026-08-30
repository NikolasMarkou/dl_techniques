"""
N-dimensional Self-Organizing Map (SOM) layer for Keras.

A SOM is an unsupervised network that maps high-dimensional vectors onto a
discrete grid of neurons. Each neuron holds one prototype vector. For an input
`x` the layer finds the closest prototype -- the Best Matching Unit (BMU) --
and pulls the BMU and its grid neighbours toward `x`. Nearby grid positions end
up holding similar prototypes, so the grid preserves the topology of the input
space. This module supports any grid rank: `(50,)` is a chain of 50 neurons,
`(10, 10)` a 10x10 plane, `(8, 8, 8)` a 512-neuron cube.

The layer trains itself. Weights are updated in `call()` with `assign_add`, not
by backpropagation, and the weight map is a non-trainable variable.

The rule is Kohonen's:

    BMU  = argmin_i ||x - w_i||^2
    w_i <- w_i + alpha(t) * h_i(t) * (x - w_i)

`alpha(t)` is the learning rate and `h_i(t)` the neighbourhood value for neuron
`i`. Both shrink as training proceeds, so early steps move large regions of the
grid and later steps fine-tune. Two neighbourhood shapes are available:

    gaussian:  h_i = exp(-d^2 / (2 * sigma^2))
    bubble:    h_i = 1 if d <= sigma else 0

where `d` is the grid distance from neuron `i` to the BMU.

Read as memory, a SOM is content-addressable. Similar inputs retrieve the same
or a nearby slot, and a novel input still retrieves its closest prototype. That
makes the layer useful for clustering, visualization, feature organization and
anomaly detection.

The layer returns BMU grid coordinates and the quantization error (the distance
to the winning prototype), not a reconstruction.

Example:
    >>> # A 10x10 SOM over 784-dimensional MNIST digits.
    >>> som_layer = SOMLayer(grid_shape=(10, 10), input_dim=784,
    ...                     initial_learning_rate=0.5, sigma=2.0)
    >>> bmu_indices, quant_errors = som_layer(input_data, training=True)
    >>> weights_grid = som_layer.get_weights_map()
    >>>
    >>> # A 1D SOM for time-series clustering.
    >>> som_1d = SOMLayer(grid_shape=(50,), input_dim=100)
    >>>
    >>> # A 3D SOM.
    >>> som_3d = SOMLayer(grid_shape=(8, 8, 8), input_dim=512)

References:
    [1] Kohonen, T. (1982). Self-organized formation of topologically correct
        feature maps. Biological Cybernetics, 43(1), 59-69.
    [2] Kohonen, T. (1990). The self-organizing map. Proceedings of the IEEE,
        78(9), 1464-1480.
    [3] Kohonen, T. (2001). Self-Organizing Maps. Springer Series in
        Information Sciences, Vol. 30, Springer, Berlin.
    [4] Ultsch, A., & Siemon, H. P. (1990). Kohonen's Self Organizing Feature
        Maps for Exploratory Data Analysis. In Proceedings of International
        Neural Networks Conference (INNC).
    [5] Vesanto, J., & Alhoniemi, E. (2000). Clustering of the self-organizing
        map. IEEE Transactions on Neural Networks, 11(3), 586-600.
"""

import keras
from typing import Tuple, Optional, Union, Dict, Any, Callable
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.memory.som_nd_layer")
class SOMLayer(keras.layers.Layer):
    """
    N-dimensional Self-Organizing Map layer.

    Maps input vectors onto a grid of prototype neurons by competitive
    learning. Every forward pass finds each input's Best Matching Unit and
    returns its grid coordinates plus the quantization error. When
    ``training=True`` the layer also moves the BMU and its neighbours toward
    the input.

    The weight map is a NON-TRAINABLE variable. It is updated inside ``call()``
    with ``assign_add``, so an optimizer never sees it and no gradient flows
    into it. A regularizer, if given, is still applied via ``add_loss()``.

    **Architecture Overview:**

    .. code-block:: text

        inputs (batch, input_dim)  -- INPUT tensor, not a weight
                 │
                 ▼
        ┌────────────────────────────────────────────────┐
        │ squared distance to every neuron               │
        │ reads weights_map (*grid_shape, input_dim)     │
        └────────────────────────────────────────────────┘
                 │  squared_distances (batch, num_neurons)
                 ├─────────────────────────────┐
                 ▼                             ▼
        ┌─────────────────────────┐  ┌─────────────────────────┐
        │ argmin over neurons     │  │ sqrt of min distance    │
        │ bmu_indices             │  │ quantization_errors     │
        │ (batch, grid_dim) int32 │  │ (batch,)                │
        └─────────────────────────┘  └─────────────────────────┘
                 │                             │
                 ├──────────────┐              │
                 │              ▼              │
                 │   ┌───────────────────────────────────┐
                 │   │ neighbourhood h around the BMU    │
                 │   │ (batch, *grid_shape)  (training)  │
                 │   └───────────────────────────────────┘
                 │              │
                 │              ▼
                 │   ┌───────────────────────────────────┐
                 │   │ weights_map.assign_add(           │
                 │   │   lr * mean(h * (x - w), axis=0)) │
                 │   │ iterations.assign_add(batch size) │
                 │   └───────────────────────────────────┘
                 │
                 │   add_loss(regularizer(weights_map))   (optional)
                 │
                 ▼                             ▼
                 bmu_indices                   quantization_errors
                 (batch, grid_dim) int32       (batch,) float
                 └──────────── return ─────────┘

    **Neighbourhood Function:**

    .. code-block:: text

        squared_dist_to_bmus (batch, *grid_shape)
        current_sigma (scalar, decayed)
                 │
                 ▼
          neighborhood_function
                 │
           ┌─────┴──────────────────┐
           │                        │
        'gaussian'               'bubble'
           │                        │
           ▼                        ▼
        ┌──────────────────┐   ┌──────────────────────┐
        │ exp(-d2 /        │   │ sqrt(d2) <= sigma    │
        │    (2 * sigma^2))│   │ cast to float32      │
        └──────────────────┘   └──────────────────────┘
           │                        │
           └───────────┬────────────┘
                       ▼
                neighborhood (batch, *grid_shape)

    Input shape:
        2D tensor of shape ``(batch_size, input_dim)``.

    Output shape:
        Tuple of two tensors: ``(batch_size, grid_dim)`` int32 BMU
        coordinates, and ``(batch_size,)`` float quantization errors.

    Example:
        >>> som = SOMLayer(grid_shape=(10, 10), input_dim=784, sigma=2.0)
        >>> bmu_coords, quant_errors = som(x, training=True)

    :param grid_shape: Shape of the neuron grid, e.g. ``(10, 10)`` for 2D or
        ``(5, 5, 5)`` for 3D. All entries must be positive integers.
    :type grid_shape: Tuple[int, ...]
    :param input_dim: Width of each input vector.
    :type input_dim: int
    :param initial_learning_rate: Starting learning rate for the weight
        update. Must be positive. Defaults to 0.1.
    :type initial_learning_rate: float
    :param decay_function: Callable ``(iterations, max_iterations) -> rate``.
        If None, linear decay from ``initial_learning_rate`` to 0 is used.
        Defaults to None.
    :type decay_function: Optional[Callable]
    :param sigma: Starting neighbourhood radius, in grid units. Must be
        positive. Defaults to 1.0.
    :type sigma: float
    :param neighborhood_function: ``'gaussian'`` or ``'bubble'``. Defaults to
        ``'gaussian'``.
    :type neighborhood_function: str
    :param weights_initializer: Initializer for the neuron weight map. The
        string ``'sample'`` is rejected; see ``__init__``. Defaults to
        ``'random_uniform'``.
    :type weights_initializer: Union[str, keras.initializers.Initializer]
    :param regularizer: Optional regularizer applied to the weight map through
        ``add_loss()``. Defaults to None.
    :type regularizer: Optional[keras.regularizers.Regularizer]
    :param name: Layer name. Defaults to None.
    :type name: Optional[str]
    :param kwargs: Forwarded to ``keras.layers.Layer.__init__``.
    :type kwargs: Any

    :ivar grid_dim: Rank of the grid, ``len(grid_shape)``.
    :vartype grid_dim: int
    :ivar num_neurons: Total neuron count, ``prod(grid_shape)``.
    :vartype num_neurons: int
    :ivar weights_map: Non-trainable variable of shape
        ``(*grid_shape, input_dim)``. Created in ``build()``.
    :vartype weights_map: keras.Variable
    :ivar iterations: Non-trainable float32 scalar counting the input vectors
        seen with ``training=True``.
    :vartype iterations: keras.Variable
    :ivar max_iterations: Non-trainable float32 scalar, the horizon the decay
        schedules are expressed against. Initialized to 1000.0.
    :vartype max_iterations: keras.Variable
    :ivar grid_positions: Plain tensor of shape ``(*grid_shape, grid_dim)``
        holding each neuron's grid coordinates. Not a variable.
    :vartype grid_positions: keras.KerasTensor
    """

    def __init__(
            self,
            grid_shape: Tuple[int, ...],
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
        """
        Validate the configuration and store it. Weights are created in
        ``build()``.

        See the class docstring for the meaning of every parameter.

        :raises ValueError: If ``grid_shape`` holds a non-positive or
            non-integer entry, ``input_dim`` is not positive,
            ``initial_learning_rate`` is not positive, ``sigma`` is not
            positive, or ``neighborhood_function`` is not ``'gaussian'`` or
            ``'bubble'``.
        :raises NotImplementedError: If ``weights_initializer`` is the string
            ``'sample'``.
        """
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
        self.num_neurons = int(np.prod(self.grid_shape))
        self.input_dim = input_dim
        self.initial_learning_rate = initial_learning_rate
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function

        # Keep the raw arguments so get_config() round-trips them.
        self._decay_function_config = decay_function
        self._weights_initializer_config = weights_initializer
        self._regularizer_config = regularizer

        # 'sample' (data-sample init) was advertised but never implemented; it
        # silently fell back to a seeded RandomUniform. Raise instead, so the
        # caller picks a real initializer.
        if isinstance(weights_initializer, str) and weights_initializer == 'sample':
            raise NotImplementedError(
                "'sample' initializer (data-sample init) is not implemented; pass "
                "a Keras initializer instance or a standard string such as "
                "'random_uniform' instead."
            )
        self.weights_initializer = keras.initializers.get(weights_initializer)
        self.regularizer = keras.regularizers.get(regularizer)

        # Default schedule: linear decay to zero at max_iterations.
        if decay_function is None:
            self.decay_function = lambda x, max_iter: self.initial_learning_rate * (1 - x / max_iter)
        else:
            self.decay_function = decay_function

        # Created in build().
        self.weights_map = None
        self.iterations = None
        self.max_iterations = None
        self.grid_positions = None

        # Set in build(), read by get_build_config().
        self._build_input_shape = None

    def build(self, input_shape: Tuple) -> None:
        """
        Create the weight map, the two counters and the grid coordinates.

        :param input_shape: Shape of the input tensor. Must be
            ``(batch_size, input_dim)``.
        :type input_shape: Tuple
        :raises ValueError: If ``input_shape`` is not rank 2 or its last axis
            is not ``input_dim``.
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

        # Counts the input vectors seen with training=True.
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

        # The dead 'sample' fallback was removed; __init__ raises instead.
        initializer = self.weights_initializer

        # trainable=False: call() updates this map with assign_add, so it must
        # stay out of the optimizer's hands.
        self.weights_map = self.add_weight(
            name="som_weights",
            shape=(*self.grid_shape, self.input_dim),
            initializer=initializer,
            regularizer=self.regularizer,
            trainable=False
        )

        # Initialize grid positions
        self.grid_positions = self._initialize_grid_positions()

        logger.info(f"Built SOMLayer with grid_shape={self.grid_shape}, input_dim={self.input_dim}")

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def _initialize_grid_positions(self) -> keras.KerasTensor:
        """
        Build the neuron coordinate grid.

        Returns a plain tensor, not a variable. That matters under mixed
        precision: Keras autocasts variables read inside ``call()`` but leaves
        plain tensors alone.

        :return: Tensor of shape ``(*grid_shape, grid_dim)`` holding each
            neuron's integer grid coordinates as float32.
        :rtype: keras.KerasTensor
        """
        coord_ranges = [keras.ops.cast(keras.ops.arange(d), "float32") for d in self.grid_shape]
        mesh_coords = keras.ops.meshgrid(*coord_ranges, indexing='ij')
        position_grid = keras.ops.stack(mesh_coords, axis=-1)
        return position_grid

    def call(self,
             inputs: keras.KerasTensor,
             training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Find each input's BMU, and update the map when training.

        With ``training=True`` this also mutates ``weights_map`` and
        ``iterations``. The return value is the same either way.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether to run the competitive weight update.
            Defaults to None, which does not update.
        :type training: Optional[bool]
        :return: ``(bmu_indices, quantization_errors)`` of shapes
            ``(batch_size, grid_dim)`` int32 and ``(batch_size,)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Find the Best Matching Units (BMUs) for each input
        bmu_indices, quantization_errors = self._find_bmu(inputs)

        # If in training mode, update the weights
        if training:
            self._update_weights(inputs, bmu_indices)
            self.iterations.assign_add(keras.ops.cast(keras.ops.shape(inputs)[0], "float32"))

        # Apply regularization if specified
        if self.regularizer is not None:
            self.add_loss(self.regularizer(self.weights_map))

        return bmu_indices, quantization_errors

    def _find_bmu(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Find the Best Matching Unit for each input vector.

        Compares every input against every neuron at once, so the intermediate
        distance tensor is ``(batch_size, num_neurons)``.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: ``(bmu_indices, quantization_errors)``. ``bmu_indices`` is
            int32 of shape ``(batch_size, grid_dim)``; ``quantization_errors``
            is the Euclidean distance to the winning neuron, shape
            ``(batch_size,)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Reshape weights to [total_neurons, input_dim]
        flat_weights = keras.ops.reshape(self.weights_map, (-1, self.input_dim))

        # Compute distances between inputs and all neurons
        # We use squared Euclidean distance for efficiency
        squared_distances = keras.ops.sum(
            keras.ops.square(keras.ops.expand_dims(inputs, 1) - keras.ops.expand_dims(flat_weights, 0)),
            axis=2
        )

        # Find the index of the minimum distance (BMU)
        bmu_flat_indices = keras.ops.argmin(squared_distances, axis=1)

        # Convert flat indices to N-dimensional grid coordinates
        bmu_indices = keras.ops.unravel_index(bmu_flat_indices, self.grid_shape)
        bmu_indices = keras.ops.stack(bmu_indices, axis=1)

        # Compute the quantization error (minimum distance)
        min_distances = keras.ops.min(squared_distances, axis=1)
        quantization_errors = keras.ops.sqrt(min_distances)

        return keras.ops.cast(bmu_indices, "int32"), quantization_errors

    def _update_weights(self, inputs: keras.KerasTensor, bmu_indices: keras.KerasTensor) -> None:
        """
        Apply one Kohonen update to the whole map.

        Decays the learning rate and sigma against ``iterations`` and
        ``max_iterations``, builds the neighbourhood around each BMU, then
        adds ``lr * mean_over_batch(h * (x - w))`` to ``weights_map``.

        Both decayed quantities are floored. The default schedule is linear
        and unbounded downward, so once ``iterations`` passes
        ``max_iterations`` the learning rate would go negative and every
        update would push neurons AWAY from their inputs. Sigma is floored at
        1e-4 to keep the Gaussian denominator non-zero.

        The whole update runs in float32; see the D-062 note below.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param bmu_indices: BMU grid coordinates of shape
            ``(batch_size, grid_dim)``.
        :type bmu_indices: keras.KerasTensor
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-062
        # This update runs entirely in float32. MEASURED: without the casts,
        # `call(training=True)` under `mixed_float16` raises `TypeError: ...
        # got tf.float32 != tf.float16`. Do NOT cast the grid to float16
        # instead: distant neurons underflow to 0.0. See decisions.md D-062.
        geom_dtype = "float32"
        inputs = keras.ops.cast(inputs, geom_dtype)
        current_learning_rate = self.decay_function(self.iterations, self.max_iterations)
        current_learning_rate = keras.ops.cast(
            keras.ops.maximum(current_learning_rate, 0.0), geom_dtype)
        current_sigma = self.sigma * (1.0 - self.iterations / self.max_iterations)
        # Prevent division by zero.
        current_sigma = keras.ops.cast(keras.ops.maximum(current_sigma, 1e-4), geom_dtype)

        bmu_coords = keras.ops.cast(bmu_indices, dtype=geom_dtype)

        # Expand dimensions for broadcasting
        bmu_coords_expanded = keras.ops.reshape(
            bmu_coords, [keras.ops.shape(inputs)[0]] + [1] * self.grid_dim + [self.grid_dim]
        )
        grid_pos_expanded = keras.ops.expand_dims(self.grid_positions, axis=0)

        # Compute neighborhood values for all neurons for each BMU in the batch
        squared_dist_to_bmus = keras.ops.sum(
            keras.ops.square(grid_pos_expanded - bmu_coords_expanded), axis=-1
        )

        # Two neighbourhood shapes. 'gaussian' falls off smoothly; 'bubble' is
        # a hard 0/1 cut at current_sigma. __init__ allows nothing else, so the
        # else branch is 'bubble'.
        if self.neighborhood_function == 'gaussian':
            neighborhood = keras.ops.exp(-squared_dist_to_bmus / (2 * keras.ops.square(current_sigma)))
        else:
            dist_to_bmus = keras.ops.sqrt(squared_dist_to_bmus)
            neighborhood = keras.ops.cast(dist_to_bmus <= current_sigma, geom_dtype)

        # Compute the weight update delta for each input
        neighborhood_expanded = keras.ops.expand_dims(neighborhood, axis=-1)
        inputs_expanded = keras.ops.reshape(
            inputs, [keras.ops.shape(inputs)[0]] + [1] * self.grid_dim + [self.input_dim]
        )
        weights_map = keras.ops.cast(self.weights_map, geom_dtype)
        delta_per_input = neighborhood_expanded * (
            inputs_expanded - keras.ops.expand_dims(weights_map, 0))

        # AVERAGE the deltas over the batch, then apply the learning rate.
        #
        # Kohonen's rule w += eta*h*(x-w) is stable for eta*h <= 1. Summing
        # over a batch of B gives an effective coefficient of eta * sum_b h_b,
        # which for B=32 and eta=0.1 exceeds 1 for any neuron near several
        # BMUs; the update then overshoots past the inputs and oscillates.
        # Averaging bounds the coefficient by eta at any batch size.
        weight_update = current_learning_rate * keras.ops.mean(delta_per_input, axis=0)
        self.weights_map.assign_add(weight_update)

    def get_weights_map(self) -> keras.KerasTensor:
        """
        Return the neuron prototypes laid out on the grid.

        :return: The weight map of shape ``(*grid_shape, input_dim)``.
        :rtype: keras.KerasTensor
        """
        return self.weights_map

    def compute_output_shape(self, input_shape: Tuple) -> Tuple[Tuple, Tuple]:
        """
        Compute the shapes of the two output tensors.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple
        :return: ``((batch_size, grid_dim), (batch_size,))``.
        :rtype: Tuple[Tuple, Tuple]
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)
        batch_size = input_shape_list[0]

        bmu_shape = tuple([batch_size, self.grid_dim])
        error_shape = tuple([batch_size])

        return bmu_shape, error_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this layer.

        ``weights_initializer`` and ``regularizer`` are serialized from the
        raw values passed to ``__init__``, not from the resolved objects.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'grid_shape': self.grid_shape,
            'input_dim': self.input_dim,
            'initial_learning_rate': self.initial_learning_rate,
            'decay_function': self._decay_function_config,
            'sigma': self.sigma,
            'neighborhood_function': self.neighborhood_function,
            'weights_initializer': keras.initializers.serialize(
                keras.initializers.get(self._weights_initializer_config)
            ),
            'regularizer': (
                keras.regularizers.serialize(
                    keras.regularizers.get(self._regularizer_config)
                )
                if self._regularizer_config is not None
                else None
            ),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Return the input shape ``build()`` was called with.

        :return: Build configuration dictionary with one key,
            ``"input_shape"``.
        :rtype: Dict[str, Any]
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Rebuild the layer's weights from a saved build configuration.

        Does nothing if the saved ``"input_shape"`` is None, which happens
        when the layer was never built before saving.

        :param config: Build configuration dictionary from
            ``get_build_config()``.
        :type config: Dict[str, Any]
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SOMLayer":
        """
        Create a layer from a configuration dictionary.

        Deserializes ``weights_initializer`` and ``regularizer`` back into
        Keras objects before calling the constructor.

        :param config: Configuration dictionary from ``get_config()``.
        :type config: Dict[str, Any]
        :return: A new ``SOMLayer``.
        :rtype: SOMLayer
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
