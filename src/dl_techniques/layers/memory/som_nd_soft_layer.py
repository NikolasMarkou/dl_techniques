"""
Differentiable soft Self-Organizing Map (Soft SOM) layer for Keras.

A classical SOM picks one winning neuron per input and edits its weights in
place. That argmax is not differentiable, so a classical SOM cannot sit in
the middle of a network trained by gradient descent. This layer replaces the
winner-take-all step with a softmax over the whole grid, which makes the
whole forward pass differentiable.

The practical difference from `SOMLayer` in `som_nd_layer.py` is where the
learning happens. `SOMLayer` owns a NON-trainable weight map and edits it
inside `call()` with `assign_add`; no optimizer is involved. `SoftSOMLayer`
owns a TRAINABLE weight map and never assigns to it. Its prototypes move
only because an optimizer applies gradients, from the enclosing model's loss
and from the losses this layer adds itself. Do not read the update rule of
one class into the other.

The forward pass is three steps:

1. Squared Euclidean distance from the input to every prototype, giving a
   tensor of shape `(batch, *grid_shape)`.
2. Softmax of the negative distances, so a close prototype gets a high
   weight. `temperature` sets the sharpness: low is nearly one-hot, high is
   nearly uniform. Two modes are available, see below.
3. Soft reconstruction `y = sum_i a_i * w_i`, a weighted average of all
   prototypes. This is the layer output and has the same shape as the input.

The two softmax modes are not interchangeable.

    global (use_per_dimension_softmax=False)
        One softmax over all prod(grid_shape) neurons.
        a_i = softmax(-||x - w_i||^2 / tau)

    per-dimension (use_per_dimension_softmax=True, the default)
        One independent softmax along each grid axis, then the elementwise
        product across axes, then renormalization. The joint assignment is
        FACTORIZED:
        a_{i1..id} ~ prod_k softmax_axis_k(-||x - w||^2 / tau)
        This is not a joint softmax. It biases the map toward axis-aligned,
        separable prototypes.

With `training=True` the layer can add up to three losses. Each has its own
gate, and they are not all independent:

    reconstruction MSE   use_reconstruction_loss AND
                         reconstruction_weight > 0
    topological          topological_weight > 0
                         (there is no on/off flag for this one)
    sharpness entropy    sharpness_weight > 0
                         (requires use_per_dimension_softmax; the global
                         path produces no per-axis softmaxes, so the
                         constructor REFUSES that combination outright
                         rather than accepting a weight it cannot honour)

The topological loss is what makes the grid map-like: it rewards nearby grid
positions for having correlated activation patterns across the batch, using
`exp(-d / topological_sigma)` as the neighbourhood kernel.

Uses: a differentiable clustering or bottleneck layer, a topologically
ordered feature extractor, a regularizer on an autoencoder latent space, or
a continuous content-addressable memory.

Example:
    >>> # Standalone: an 8x8 grid over 64-dimensional vectors.
    >>> som = SoftSOMLayer(grid_shape=(8, 8), input_dim=64, temperature=0.5)
    >>> reconstruction = som(x)          # same shape as x
    >>> assignments = som.get_soft_assignments(x)   # (batch, 8, 8)
    >>>
    >>> # As a topological bottleneck inside an autoencoder.
    >>> h = keras.layers.Dense(128, activation='relu')(encoder_input)
    >>> z = SoftSOMLayer(grid_shape=(16, 16), input_dim=128,
    ...                  reconstruction_weight=1.0,
    ...                  topological_weight=0.5)(h)
    >>>
    >>> # Sharper assignments, or the global softmax path instead.
    >>> sharp = SoftSOMLayer(grid_shape=(5, 5), input_dim=32,
    ...                      temperature=0.01)
    >>> flat = SoftSOMLayer(grid_shape=(5, 5), input_dim=32,
    ...                     use_per_dimension_softmax=False)

    The layer's internal losses reach the enclosing model automatically
    through `add_loss`, so nothing has to be wired up at compile time.

References:
    [1] Kohonen, T. (1982). Self-organized formation of topologically correct
        feature maps. Biological Cybernetics, 43(1), 59-69.
    [2] Ritter, H., & Schulten, K. (1988). Convergence properties of
        Kohonen's topology conserving maps: fluctuations, stability, and
        dimension selection. Biological Cybernetics, 60(1), 59-71.
    [3] Soft assignment as a differentiable stand-in for hard clustering is
        used in many forms across the deep learning literature; this
        implementation follows that general idea rather than one paper.
"""

import keras
from typing import Tuple, Optional, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.memory.som_nd_soft_layer")
class SoftSOMLayer(keras.layers.Layer):
    """
    Differentiable soft Self-Organizing Map layer.

    Learns a grid of prototype vectors and returns a soft reconstruction of
    its input, ``y = sum_i a_i * w_i``, where the assignment weights ``a``
    come from a softmax over the negative squared distances to the
    prototypes. Every operation is differentiable.

    This layer trains by BACKPROPAGATION. ``weights_map`` is a trainable
    variable and ``call()`` never assigns to it; an optimizer moves it. That
    is the opposite of ``SOMLayer`` in ``som_nd_layer.py``, whose weight map
    is non-trainable and is edited in place by a competitive update. Nothing
    here picks a single winner.

    Two assignment modes are available and they are not equivalent. The
    global mode runs one softmax over all ``prod(grid_shape)`` neurons. The
    per-dimension mode (the default) runs one softmax per grid axis and takes
    the product, so the joint assignment is factorized and the map is biased
    toward axis-aligned prototypes.

    **Architecture Overview:**

    .. code-block:: text

        inputs (batch, input_dim)  -- INPUT tensor, not a weight
                 │
                 ▼
        ┌────────────────────────────────────────────────┐
        │ squared distance to every prototype            │
        │ reads weights_map (*grid_shape, input_dim)     │
        └────────────────────────────────────────────────┘
                 │  squared_distances (batch, *grid_shape)
                 ▼
          use_per_dimension_softmax
                 │
           ┌─────┴────────────────────┐
          True                      False
           │                          │
           ▼                          ▼
        ┌───────────────────────┐  ┌───────────────────────┐
        │ softmax(-d / tau)     │  │ flatten the grid      │
        │ once per grid axis    │  │ softmax(-d / tau)     │
        │ product across axes   │  │ over all neurons      │
        │ renormalize           │  │ reshape back          │
        │ keeps dim_softmaxes   │  │ dim_softmaxes = None  │
        └───────────────────────┘  └───────────────────────┘
           │                          │
           └─────────────┬────────────┘
                         ▼
             soft_assignments (batch, *grid_shape)
                         │
                         ▼
        ┌────────────────────────────────────────────────┐
        │ y = sum over the grid of a_i * w_i             │
        │ reconstruction (batch, input_dim)              │
        └────────────────────────────────────────────────┘
                         │
                         ▼
        training=True only; add_loss, each gated alone:
          ├─► MSE(inputs, reconstruction)      (optional)
          │     use_reconstruction_loss and weight > 0
          ├─► topological(soft_assignments)    (optional)
          │     topological_weight > 0
          └─► sharpness(dim_softmaxes)         (optional)
                sharpness_weight > 0; only ever the
                per-dim path, since __init__ refuses
                a positive weight on the global path
                         │
                         ▼
             return reconstruction (batch, input_dim)

    Input shape:
        2D tensor of shape ``(batch_size, input_dim)``.

    Output shape:
        2D tensor of shape ``(batch_size, input_dim)`` -- the same shape as
        the input.

    Example:
        >>> som = SoftSOMLayer(grid_shape=(8, 8), input_dim=64,
        ...                    temperature=0.5)
        >>> reconstruction = som(x)
        >>> assignments = som.get_soft_assignments(x)

    Note:
        The sharpness loss reads the per-axis softmaxes, which only the
        per-dimension path produces. Setting ``sharpness_weight > 0`` while
        ``use_per_dimension_softmax=False`` is therefore unsatisfiable, and
        ``__init__`` raises ``ValueError`` on it. It used to be accepted and
        then silently ignored, which let a configuration train a lie.

    :param grid_shape: Shape of the neuron grid, e.g. ``(10, 10)`` for 2D.
        All entries must be positive integers.
    :type grid_shape: Tuple[int, ...]
    :param input_dim: Width of each input vector. Must be positive.
    :type input_dim: int
    :param temperature: Softmax temperature ``tau``. Lower gives sharper,
        more one-hot assignments. Must be positive. Defaults to 1.0.
    :type temperature: float
    :param use_per_dimension_softmax: Use the factorized per-axis softmax
        instead of one global softmax. Defaults to True.
    :type use_per_dimension_softmax: bool
    :param use_reconstruction_loss: Add the internal MSE reconstruction
        loss. Also needs ``reconstruction_weight > 0``. Defaults to True.
    :type use_reconstruction_loss: bool
    :param reconstruction_weight: Multiplier on the reconstruction loss.
        Must be non-negative; 0 disables it. Defaults to 1.0.
    :type reconstruction_weight: float
    :param topological_weight: Multiplier on the topological preservation
        loss. Must be non-negative; 0 disables it, and there is no separate
        on/off flag. Defaults to 0.1.
    :type topological_weight: float
    :param topological_sigma: Length scale of the neighbourhood kernel
        ``h(d) = exp(-d / sigma)`` inside the topological loss. Larger means
        broader neighbourhoods. Must be positive. Defaults to 1.0, which
        reproduces the earlier fixed-scale ``exp(-d)`` behaviour.
    :type topological_sigma: float
    :param sharpness_weight: Multiplier on the entropy sharpness loss. Must
        be non-negative, and must be 0.0 unless
        ``use_per_dimension_softmax`` is True. Defaults to 0.0, which
        disables it.
    :type sharpness_weight: float
    :param kernel_initializer: Initializer for the prototype weight map.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer on the weight map. Defaults to
        ``keras.regularizers.L2(1e-5)``, not None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Forwarded to ``keras.layers.Layer.__init__``.
    :type kwargs: Any

    :ivar grid_dim: Rank of the grid, ``len(grid_shape)``.
    :vartype grid_dim: int
    :ivar weights_map: Trainable variable of shape
        ``(*grid_shape, input_dim)`` holding one prototype per neuron.
        Created in ``build()``.
    :vartype weights_map: keras.Variable
    :ivar grid_positions: Plain tensor of shape ``(*grid_shape, grid_dim)``
        holding each neuron's grid coordinates, used by the topological
        loss. Not a variable.
    :vartype grid_positions: keras.KerasTensor
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        input_dim: int,
        temperature: float = 1.0,
        use_per_dimension_softmax: bool = True,
        use_reconstruction_loss: bool = True,
        reconstruction_weight: float = 1.0,
        topological_weight: float = 0.1,
        topological_sigma: float = 1.0,
        sharpness_weight: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-5),
        **kwargs: Any
    ) -> None:
        """
        Validate and store the configuration. Weights are created in
        ``build()``.

        See the class docstring for the meaning of every parameter.

        :raises ValueError: If ``grid_shape`` holds a non-positive or
            non-integer entry; if ``input_dim`` or ``temperature`` is not
            positive; if ``topological_sigma`` is not positive; if
            ``reconstruction_weight``, ``topological_weight`` or
            ``sharpness_weight`` is negative; or if ``sharpness_weight`` is
            positive while ``use_per_dimension_softmax`` is False, which is
            unsatisfiable because only the per-dimension path produces the
            per-axis softmaxes that loss reads.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not all(isinstance(d, int) and d > 0 for d in grid_shape):
            raise ValueError("grid_shape must contain positive integers.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        if topological_weight < 0:
            raise ValueError("topological_weight must be non-negative.")
        if topological_sigma <= 0:
            raise ValueError("topological_sigma must be positive.")
        if sharpness_weight < 0:
            raise ValueError("sharpness_weight must be non-negative.")
        # DECISION plan-2026-08-30T063229-ccd6ad17/D-016
        # Refuse the combination; before this it was accepted and the weight
        # silently ignored. Do NOT soften to a warning and do NOT wire the loss
        # into the global path -- a config that trains a lie is worse than one
        # that refuses to build. Safe to narrow: 134 archives carry no such
        # config, and both trainers hardcode or default the other corner.
        # See decisions.md D-016 and D-010.
        if sharpness_weight > 0 and not use_per_dimension_softmax:
            raise ValueError(
                "sharpness_weight > 0 requires use_per_dimension_softmax=True. "
                "The sharpness loss is the entropy of the per-axis softmaxes, "
                "which only the per-dimension path produces; the global softmax "
                "path has none, so the weight could never take effect. Got "
                f"sharpness_weight={sharpness_weight} with "
                "use_per_dimension_softmax=False. Either set "
                "sharpness_weight=0.0 to keep the global softmax, or set "
                "use_per_dimension_softmax=True to keep the sharpness loss."
            )

        # Store ALL configuration parameters for serialization
        self.grid_shape = grid_shape
        self.grid_dim = len(grid_shape)
        self.input_dim = input_dim
        self.temperature = temperature
        self.use_per_dimension_softmax = use_per_dimension_softmax
        self.use_reconstruction_loss = use_reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        self.topological_weight = topological_weight
        self.topological_sigma = topological_sigma
        self.sharpness_weight = sharpness_weight

        # Store initializers and regularizers for serialization
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Initialize weight attributes - actual weights created in build()
        self.weights_map = None
        self.grid_positions = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the trainable prototype map and the grid coordinates.

        :param input_shape: Shape of the input tensor. Must be
            ``(batch_size, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 2, its last axis
            is None, or its last axis is not ``input_dim``.
        """
        # Verify input shape compatibility
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape (batch_size, input_dim), got {input_shape}")

        if input_shape[-1] is None:
            raise ValueError("Input dimension must be defined, got None")

        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got input shape with "
                f"last dimension={input_shape[-1]}"
            )

        # trainable=True: this map is moved by the optimizer, through the
        # gradients of the soft reconstruction. SOMLayer's map is the opposite
        # -- non-trainable, and edited in place by a competitive update.
        self.weights_map = self.add_weight(
            name="som_weights",
            shape=(*self.grid_shape, self.input_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Create grid positions for topological regularization
        self.grid_positions = self._create_grid_positions()

        logger.info(
            f"Built SoftSOMLayer with grid_shape={self.grid_shape}, "
            f"input_dim={self.input_dim}, trainable weights: {self.weights_map.shape}"
        )

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def _create_grid_positions(self) -> keras.KerasTensor:
        """
        Build the neuron coordinate grid used by the topological loss.

        Returns a plain tensor, not a variable, so mixed precision leaves it
        alone.

        :return: Tensor of shape ``(*grid_shape, grid_dim)`` holding each
            neuron's integer grid coordinates as float32.
        :rtype: keras.KerasTensor
        """
        # Create coordinate ranges for each dimension
        coord_ranges = [keras.ops.cast(keras.ops.arange(d), "float32") for d in self.grid_shape]

        # Create meshgrid and stack into position tensor
        mesh_coords = keras.ops.meshgrid(*coord_ranges, indexing='ij')
        position_grid = keras.ops.stack(mesh_coords, axis=-1)

        return position_grid

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Assign the input softly across the grid and reconstruct it.

        No weight is modified here. With ``training=True`` up to three
        regularization losses are added through ``add_loss``; see the class
        docstring for which flag gates which loss. The return value is the
        same either way.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether to add the regularization losses. Defaults
            to None, which adds none.
        :type training: Optional[bool]
        :return: Soft reconstruction of shape ``(batch_size, input_dim)``.
        :rtype: keras.KerasTensor
        """
        # Compute soft assignments and get intermediate softmax results for regularization
        soft_assignments, dim_softmaxes = self._compute_soft_assignments(inputs)

        # Perform soft reconstruction
        reconstruction = self._soft_reconstruction(soft_assignments)

        # Add regularization losses during training
        if training:
            if self.use_reconstruction_loss and self.reconstruction_weight > 0:
                recon_loss = self._reconstruction_loss(inputs, reconstruction)
                self.add_loss(self.reconstruction_weight * recon_loss)

            if self.topological_weight > 0:
                topo_loss = self._topological_loss(soft_assignments)
                self.add_loss(self.topological_weight * topo_loss)

            # Add sharpness loss for encouraging decisive assignments
            if (self.sharpness_weight > 0 and
                self.use_per_dimension_softmax and
                dim_softmaxes is not None):
                sharp_loss = self._sharpness_loss(dim_softmaxes)
                self.add_loss(self.sharpness_weight * sharp_loss)

        return reconstruction

    def _compute_soft_assignments(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, Optional[list]]:
        """
        Turn distances to the prototypes into soft assignments.

        Broadcasts the input against the whole weight map to get squared
        distances of shape ``(batch_size, *grid_shape)``, then routes to
        whichever softmax mode is configured.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: ``(soft_assignments, dim_softmaxes)``. The assignments have
            shape ``(batch_size, *grid_shape)``. ``dim_softmaxes`` is the
            list of per-axis softmaxes on the per-dimension path, and None on
            the global path.
        :rtype: Tuple[keras.KerasTensor, Optional[list]]
        """
        # Compute squared distances from inputs to all neurons
        # inputs: (batch_size, input_dim)
        # weights_map: (*grid_shape, input_dim)

        # Expand inputs to broadcast with weights_map
        expanded_inputs = inputs
        for _ in range(self.grid_dim):
            expanded_inputs = keras.ops.expand_dims(expanded_inputs, axis=1)

        # Expand weights for broadcasting
        expanded_weights = keras.ops.expand_dims(self.weights_map, axis=0)

        # Compute squared distances: (batch_size, *grid_shape)
        squared_distances = keras.ops.sum(
            keras.ops.square(expanded_inputs - expanded_weights),
            axis=-1
        )

        # Convert distances to soft assignments
        if self.use_per_dimension_softmax:
            return self._per_dimension_softmax(squared_distances)
        else:
            global_assignments = self._global_softmax(squared_distances)
            return global_assignments, None

    def _per_dimension_softmax(self, distances: keras.KerasTensor) -> Tuple[keras.KerasTensor, list]:
        """
        Softmax along each grid axis, then combine multiplicatively.

        Runs one softmax per grid axis over the negative distances, takes
        the elementwise product of those marginals, and renormalizes over
        the spatial axes. The result is a factorized joint assignment, not a
        joint softmax.

        :param distances: Squared distances of shape
            ``(batch_size, *grid_shape)``.
        :type distances: keras.KerasTensor
        :return: ``(combined, dim_softmaxes)``. ``combined`` has shape
            ``(batch_size, *grid_shape)`` and sums to 1 over the grid;
            ``dim_softmaxes`` holds one tensor per grid axis, in axis order.
        :rtype: Tuple[keras.KerasTensor, list]
        """
        # Apply softmax along each grid dimension independently
        dim_softmaxes = []

        for dim_idx in range(self.grid_dim):
            # Apply softmax along current dimension (dim_idx + 1 due to batch dimension)
            spatial_axis = dim_idx + 1

            # Apply softmax with negative distances (closer = higher probability)
            dim_softmax = keras.ops.softmax(-distances / self.temperature, axis=spatial_axis)
            dim_softmaxes.append(dim_softmax)

        # Combine dimension-wise softmaxes multiplicatively
        combined = dim_softmaxes[0]
        for i in range(1, len(dim_softmaxes)):
            combined = combined * dim_softmaxes[i]

        # Normalize to ensure probabilities sum to 1 across spatial dimensions
        spatial_axes = list(range(1, self.grid_dim + 1))
        total = keras.ops.sum(combined, axis=spatial_axes, keepdims=True)
        combined = combined / (total + 1e-8)

        return combined, dim_softmaxes

    def _global_softmax(self, distances: keras.KerasTensor) -> keras.KerasTensor:
        """
        One softmax over every neuron in the grid.

        Flattens the spatial axes, softmaxes the negative distances across
        all neurons, then reshapes back to the grid.

        :param distances: Squared distances of shape
            ``(batch_size, *grid_shape)``.
        :type distances: keras.KerasTensor
        :return: Soft assignments of shape ``(batch_size, *grid_shape)``,
            summing to 1 over the whole grid.
        :rtype: keras.KerasTensor
        """
        # Flatten spatial dimensions for global softmax
        batch_size = keras.ops.shape(distances)[0]
        flat_distances = keras.ops.reshape(distances, (batch_size, -1))

        # Apply global softmax over all neurons
        flat_softmax = keras.ops.softmax(-flat_distances / self.temperature, axis=1)

        # Reshape back to original grid shape
        return keras.ops.reshape(flat_softmax, (batch_size,) + self.grid_shape)

    def _soft_reconstruction(self, soft_assignments: keras.KerasTensor) -> keras.KerasTensor:
        """
        Average the prototypes, weighted by the soft assignments.

        Computes ``y = sum_i a_i * w_i`` by summing over every grid axis.

        :param soft_assignments: Assignment weights of shape
            ``(batch_size, *grid_shape)``.
        :type soft_assignments: keras.KerasTensor
        :return: Reconstruction of shape ``(batch_size, input_dim)``.
        :rtype: keras.KerasTensor
        """
        # soft_assignments: (batch_size, *grid_shape)
        # weights_map: (*grid_shape, input_dim)

        # Expand assignments for element-wise multiplication
        expanded_assignments = keras.ops.expand_dims(soft_assignments, axis=-1)

        # Expand weights for broadcasting
        expanded_weights = keras.ops.expand_dims(self.weights_map, axis=0)

        # Compute weighted prototype vectors
        weighted_neurons = expanded_assignments * expanded_weights

        # Sum over all spatial dimensions to get reconstruction
        spatial_axes = list(range(1, self.grid_dim + 1))
        reconstruction = keras.ops.sum(weighted_neurons, axis=spatial_axes)

        return reconstruction

    def _reconstruction_loss(
        self,
        inputs: keras.KerasTensor,
        reconstruction: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Mean squared error between the input and its reconstruction.

        Pushes the prototypes to form a codebook that can represent the
        data.

        :param inputs: Original input of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param reconstruction: Reconstruction of the same shape.
        :type reconstruction: keras.KerasTensor
        :return: Scalar MSE, averaged over batch and features.
        :rtype: keras.KerasTensor
        """
        mse_loss = keras.ops.mean(keras.ops.square(inputs - reconstruction))
        return mse_loss

    def _topological_loss(self, soft_assignments: keras.KerasTensor) -> keras.KerasTensor:
        """
        Reward nearby neurons for activating together.

        Builds a neighbourhood kernel ``exp(-d / topological_sigma)`` over
        the pairwise grid distances, correlates the batch-centred assignment
        patterns of every neuron pair, and returns the NEGATIVE mean of the
        product. Minimizing it therefore raises the correlation between
        spatially close neurons, which is what makes the grid map-like.

        :param soft_assignments: Assignment weights of shape
            ``(batch_size, *grid_shape)``.
        :type soft_assignments: keras.KerasTensor
        :return: Scalar loss. Typically negative.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(soft_assignments)[0]

        # Flatten grid positions and assignments for pairwise computations
        total_neurons = keras.ops.prod(keras.ops.convert_to_tensor(self.grid_shape))
        flat_positions = keras.ops.reshape(self.grid_positions, (total_neurons, self.grid_dim))
        flat_assignments = keras.ops.reshape(soft_assignments, (batch_size, total_neurons))

        # Compute pairwise spatial distances between all grid positions
        position_diff = keras.ops.expand_dims(flat_positions, axis=1) - keras.ops.expand_dims(flat_positions, axis=0)
        position_distances = keras.ops.sqrt(keras.ops.sum(keras.ops.square(position_diff), axis=-1) + 1e-8)

        # Create neighborhood weights (exponential decay with distance).
        # NOTE: scaling controlled by `topological_sigma` (default 1.0 preserves
        # legacy `exp(-d)` behavior numerically).
        neighborhood_weights = keras.ops.exp(-position_distances / self.topological_sigma)

        # Compute activation correlations between neurons across batch
        normalized_assignments = flat_assignments - keras.ops.mean(flat_assignments, axis=0, keepdims=True)
        assignment_correlations = keras.ops.matmul(
            keras.ops.transpose(normalized_assignments),
            normalized_assignments
        ) / keras.ops.cast(keras.ops.maximum(batch_size - 1, 1), "float32")

        # Topological loss: promote correlation between spatially close neurons
        topo_loss = -keras.ops.mean(neighborhood_weights * assignment_correlations)

        return topo_loss

    def _sharpness_loss(self, dim_softmaxes: list) -> keras.KerasTensor:
        """
        Average entropy of the per-axis assignments.

        Computes ``H(p) = -sum(p * log(p))`` along each grid axis and
        averages. Minimizing it pushes the assignments toward one-hot.

        ``dim_softmaxes`` is never empty: this method is only called on the
        per-dimension path, which always returns one tensor per grid axis,
        and ``grid_shape`` is validated non-empty in ``__init__``. The
        constructor now refuses ``sharpness_weight > 0`` on the global path,
        so there is no configuration in which this runs with nothing to
        average.

        :param dim_softmaxes: One softmax tensor per grid axis, in axis
            order, as returned by ``_per_dimension_softmax``. Non-empty.
        :type dim_softmaxes: list
        :return: Scalar mean entropy across the grid axes.
        :rtype: keras.KerasTensor
        """
        total_entropy = keras.ops.convert_to_tensor(0.0, dtype="float32")

        for dim_idx, softmax_tensor in enumerate(dim_softmaxes):
            # The spatial axis along which softmax was computed
            spatial_axis = dim_idx + 1

            # Compute entropy: H(p) = -Σ(p * log(p))
            # Add small epsilon for numerical stability
            log_probs = keras.ops.log(softmax_tensor + 1e-9)
            entropy = -keras.ops.sum(softmax_tensor * log_probs, axis=spatial_axis)

            # Average entropy across batch and remaining spatial dimensions
            total_entropy += keras.ops.mean(entropy)

        # Return average entropy across all spatial dimensions
        return total_entropy / len(dim_softmaxes)

    def get_weights_map(self) -> keras.KerasTensor:
        """
        Return the learned prototype map.

        :return: Trainable weight map of shape
            ``(*grid_shape, input_dim)``.
        :rtype: keras.KerasTensor
        :raises RuntimeError: If the layer has not been built yet.
        """
        if self.weights_map is None:
            raise RuntimeError("Layer must be built before accessing weights_map")
        return self.weights_map

    def get_soft_assignments(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Return the soft assignments for an input, without reconstructing.

        Useful for inspecting where data lands on the grid: the argmax over
        the grid axes is the closest thing this layer has to a BMU.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Assignments of shape ``(batch_size, *grid_shape)``, summing
            to 1 over the grid.
        :rtype: keras.KerasTensor
        """
        soft_assignments, _ = self._compute_soft_assignments(inputs)
        return soft_assignments

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which equals the input shape.

        The output is a reconstruction of the input, so nothing changes.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape, as a tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config needed to rebuild this layer.

        Includes every constructor argument, with the initializer and the
        regularizer serialized.

        :return: Config dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'grid_shape': self.grid_shape,
            'input_dim': self.input_dim,
            'temperature': self.temperature,
            'use_per_dimension_softmax': self.use_per_dimension_softmax,
            'use_reconstruction_loss': self.use_reconstruction_loss,
            'reconstruction_weight': self.reconstruction_weight,
            'topological_weight': self.topological_weight,
            'topological_sigma': self.topological_sigma,
            'sharpness_weight': self.sharpness_weight,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------