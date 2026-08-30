"""
2D Self-Organizing Map (SOM) layer for Keras.

This module holds `SOM2dLayer`, a thin subclass of the N-dimensional
`SOMLayer` in `som_nd_layer.py`. It fixes the grid rank at 2 and renames the
grid argument to `map_size`. Everything else -- weight creation, the Best
Matching Unit search, the Kohonen update, the decay schedules and the
serialization of the shared arguments -- lives in `SOMLayer`. Read
`som_nd_layer.py` for how the map actually learns.

The recommended entry point is the factory function
``dl_techniques.layers.memory.create_som_2d(map_size, input_dim, **kwargs)``,
which delegates here. The class stays public and Keras-registered for three
reasons:

* downstream `isinstance(x, SOM2dLayer)` checks keep working,
* the 2D-specific `get_weights_as_grid()` alias is preserved,
* `get_config()` emits `map_size` rather than `grid_shape`, so `.keras` files
  written before the factory existed still load.

The additive-factory rationale is the `D-002` anchor at the top of
`factory.py`. Its owning plan directory, `plan_2026-05-13_8c1dc6fd`, no longer
exists, so that anchor is where the reasoning is kept.

A SOM maps high-dimensional vectors onto a grid of neurons. Each neuron
`(i, j)` holds a prototype vector `w_ij` as wide as the input. For an input
`x` the closest prototype wins -- that neuron is the Best Matching Unit --
and it and its grid neighbours are pulled toward `x`. Similar inputs end up
activating nearby grid cells, so the grid preserves the topology of the input
space. Read as memory, that makes the layer content-addressable: a novel
input still retrieves its closest prototype.

**Grid Layout:**

.. code-block:: text

    y
    ^
    |  (0,0)-----(0,1)-----(0,2)---...
    |    |         |         |
    |  (1,0)-----(1,1)-----(1,2)---...
    |    |         |         |
    |  (2,0)-----(2,1)-----(2,2)---...
    |    :         :         :
    +----------------------------------> x

`map_size` is `(height, width)`, so `map_size=(10, 10)` is a 100-neuron
square and the returned BMU coordinates are `(row, column)` pairs.

The layer returns BMU grid coordinates and the quantization error, not a
reconstruction. It trains itself: the weight map is a non-trainable variable
updated inside `call()`, never by an optimizer.

Example:
    >>> # A 10x10 SOM over 784-dimensional MNIST digits.
    >>> som_layer = SOM2dLayer(map_size=(10, 10), input_dim=784,
    ...                        initial_learning_rate=0.5, sigma=2.0)
    >>> bmu_indices, quant_errors = som_layer(input_data, training=True)
    >>>
    >>> # The grid can be plotted to inspect the organized memory space.
    >>> weights_grid = som_layer.get_weights_as_grid()

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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .som_nd_layer import SOMLayer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.memory.som_2d_layer")
class SOM2dLayer(SOMLayer):
    """
    2D Self-Organizing Map layer.

    A ``SOMLayer`` with its grid rank fixed at 2. ``map_size`` is validated as
    a 2-tuple of positive integers and passed to ``SOMLayer`` as
    ``grid_shape``; every other argument is forwarded unchanged. This class
    has no ``build()`` and no ``call()`` of its own, so the forward pass, the
    weight map and the competitive update are the parent's. See
    ``SOMLayer`` in ``som_nd_layer.py`` for that graph and for the argument
    rules it enforces, including the rejected ``'sample'`` initializer and the
    floors applied to the decayed learning rate and sigma.

    What this subclass adds is three things: the ``get_weights_as_grid()``
    alias, a ``get_config()`` that emits ``map_size`` in place of
    ``grid_shape``, and a ``from_config()`` that reads it back.

    **Architecture Overview:**

    .. code-block:: text

        map_size = (height, width)
                 │  renamed to grid_shape
                 ▼
        ┌────────────────────────────────────────────────┐
        │ SOM2dLayer.__init__                            │
        │ checks map_size is 2 positive integers         │
        │ super().__init__(grid_shape=map_size, ...)     │
        │ stores self.map_size                           │
        └────────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────────────────────────────────┐
        │ SOMLayer  (parent, som_nd_layer.py)            │
        │ owns build(), call(), weights_map and the      │
        │ competitive update -- see its own diagram      │
        └────────────────────────────────────────────────┘
                 │
                 ▼
        return (bmu_indices, quantization_errors)
                (batch, 2) int32   (batch,) float

        added here, off the forward path:
          get_weights_as_grid() ──► parent get_weights_map()
          get_config()          ──► 'map_size' key
          from_config()         ──► cls(**config)

    Input shape:
        2D tensor of shape ``(batch_size, input_dim)``.

    Output shape:
        Tuple of two tensors: ``(batch_size, 2)`` int32 BMU ``(row, column)``
        coordinates, and ``(batch_size,)`` float quantization errors.

    Example:
        >>> som = SOM2dLayer(map_size=(10, 10), input_dim=784, sigma=2.0)
        >>> bmu_coords, quant_errors = som(x, training=True)
        >>> grid = som.get_weights_as_grid()

    :param map_size: Grid shape as ``(height, width)``. Must be exactly 2
        positive integers.
    :type map_size: Tuple[int, int]
    :param input_dim: Width of each input vector. Must be positive.
    :type input_dim: int
    :param initial_learning_rate: Starting learning rate for the weight
        update. Defaults to 0.1.
    :type initial_learning_rate: float
    :param decay_function: Callable ``(iterations, max_iterations) -> rate``.
        If None, linear decay is used. Defaults to None.
    :type decay_function: Optional[Callable]
    :param sigma: Starting neighbourhood radius, in grid units. Defaults
        to 1.0.
    :type sigma: float
    :param neighborhood_function: ``'gaussian'`` or ``'bubble'``. Defaults to
        ``'gaussian'``.
    :type neighborhood_function: str
    :param weights_initializer: Initializer for the neuron weight map.
        Defaults to ``'random_uniform'``.
    :type weights_initializer: Union[str, keras.initializers.Initializer]
    :param regularizer: Optional regularizer applied to the weight map.
        Defaults to None.
    :type regularizer: Optional[keras.regularizers.Regularizer]
    :param name: Layer name. Defaults to None.
    :type name: Optional[str]
    :param kwargs: Forwarded to ``SOMLayer.__init__``.
    :type kwargs: Any

    :ivar map_size: The grid shape as a tuple, kept so ``get_config()`` can
        emit it. ``SOMLayer.grid_shape`` holds the same pair.
    :vartype map_size: Tuple[int, int]
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
        """
        Validate ``map_size`` and hand every argument to ``SOMLayer``.

        Weights are created by the parent's ``build()``. See the class
        docstring for the meaning of every parameter.

        :raises ValueError: If ``map_size`` is not a tuple or list of length
            2, or holds a non-integer or non-positive entry. The parent
            raises for the remaining arguments.
        """
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

        # Kept so get_config() can emit 'map_size'; the parent already holds
        # the same pair as grid_shape.
        self.map_size = tuple(map_size)

    def get_weights_as_grid(self) -> keras.KerasTensor:
        """
        Return the neuron weight map as a 2D grid.

        An alias for the inherited ``SOMLayer.get_weights_map()``. It exists
        for the older 2D-only name and returns the same tensor.

        :return: Weight map of shape ``(height, width, input_dim)``.
        :rtype: keras.KerasTensor
        """
        return self.get_weights_map()

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config, with ``grid_shape`` renamed to ``map_size``.

        Takes the parent's config, which already serializes the initializer
        and the regularizer, then swaps the ``grid_shape`` key for
        ``map_size``. That keeps ``.keras`` files written before this class
        gained an N-D parent loadable.

        :return: Config dictionary keyed on ``map_size``, not ``grid_shape``.
        :rtype: Dict[str, Any]
        """
        # The parent already serializes the initializer and the regularizer,
        # so nothing here needs to touch them.
        config = super().get_config()

        # Replace 'grid_shape' with 'map_size' for 2D layer compatibility
        if 'grid_shape' in config:
            config['map_size'] = self.map_size
            del config['grid_shape']

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SOM2dLayer':
        """
        Rebuild a layer from a config dictionary.

        Turns the serialized ``weights_initializer`` and ``regularizer``
        entries back into objects, then calls the constructor. ``config``
        carries ``map_size``, which is the constructor's own argument name,
        so no key renaming is needed here.

        :param config: Config dictionary, as produced by ``get_config()``.
        :type config: Dict[str, Any]
        :return: A new layer built from ``config``.
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