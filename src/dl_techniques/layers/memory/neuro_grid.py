"""
Differentiable n-dimensional memory grid with probabilistic addressing.

A classical Self-Organizing Map picks one winning cell per input. That argmax
is not differentiable, so a classical SOM cannot sit inside a network trained
by gradient descent. `NeuroGrid` replaces the winner search with a soft
lookup. Every cell contributes to the output, weighted by a learned
probability.

The layer owns two things. A grid of learnable latent vectors, of shape
`(d1, d2, ..., dn, latent_dim)`. And one `Dense` projection per grid
dimension, which turns the input into an address.

The forward pass is four steps:

1. Projection. `Dense` layer `i` maps the input to `d_i` logits. There is no
   activation on it; the softmax is applied separately so the temperature can
   be applied first.

2. Temperature softmax, `P_i = softmax(logits_i / T)`. A low temperature gives
   a sharp, nearly one-hot address; a high one gives a diffuse address. `T` is
   a weight, trainable only when `learnable_temperature=True`.

3. Outer product, `P_joint = P_1 (x) P_2 (x) ... (x) P_n`, of shape
   `(batch, d1, ..., dn)`. The joint address is FACTORIZED: it is a product of
   per-axis distributions, not one softmax over the whole grid.

4. Soft lookup, `output = sum P_joint * grid_weights`, giving
   `(batch, latent_dim)`.

Factorizing the address is what keeps the layer cheap. A grid of
`d1 * ... * dn` cells is addressed by `sum d_i` logits rather than by
`prod d_i` of them. The idea comes from Product Key Memories.

Gradients reach both the grid vectors and the projections, so the memory
content and the addressing logic are learned together. Similar inputs end up
addressing nearby cells, which is where the SOM-like topological behaviour
comes from.

Both 2-D `(batch, input_dim)` and 3-D `(batch, seq_len, input_dim)` inputs
work. A 3-D input is flattened to `(batch * seq_len, input_dim)`, run through
the same grid, then reshaped back, so each token is addressed on its own.

Beyond the forward pass the layer exposes read-only analysis helpers: the
per-dimension and joint addressing probabilities, grid utilisation counts,
best matching units, and six input quality measures derived from how sharp
the addressing is.

References:
    - Kohonen, T. (1990). The Self-Organizing Map. *Proceedings of the IEEE*.
      (Conceptual foundation for topological data representation).
    - Graves, A., et al. (2014). Neural Turing Machines. *arXiv preprint*.
      (Pioneered differentiable addressing for external memory).
    - Lample, G., et al. (2019). Large Memory Layers with Product Keys. *NeurIPS*.
      (Inspiration for compositional, factorized addressing).
    - Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network.
      *arXiv preprint*. (Popularized the use of temperature in softmax).
"""

import keras
import numpy as np
from typing import List, Tuple, Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from dl_techniques.initializers import clone_initializer
from dl_techniques.initializers.hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

# Subscript alphabet for the ``_soft_lookup`` einsum, one letter per grid axis.
# The equation also spends ``'b'`` on the batch axis and ``'z'`` on the latent
# axis, so neither may appear here: the run is ``'i'``..``'y'``, 17 letters.
GRID_EINSUM_SUBSCRIPTS: str = 'ijklmnopqrstuvwxy'


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.memory.neuro_grid")
class NeuroGrid(keras.layers.Layer):
    """
    Differentiable N-dimensional memory grid with probabilistic addressing.

    Learns a grid of latent vectors and returns a weighted average of them.
    One ``Dense`` projection per grid dimension turns the input into a
    probability distribution over that axis,
    ``P_i = softmax(Dense_i(x) / T)``. The outer product of those
    distributions is the joint address ``P_joint``, and the output is
    ``sum P_joint * grid_weights``. Nothing picks a single winner, so every
    step is differentiable.

    The address is FACTORIZED. It is a product of per-axis distributions, not
    one softmax over the whole grid, which is what makes a large grid cheap to
    address. That is the opposite trade-off from ``SOMLayer`` in
    ``som_nd_layer.py``, which searches for a single best matching unit and
    updates it in place without an optimizer.

    Both 2-D ``(batch, input_dim)`` and 3-D ``(batch, seq_len, input_dim)``
    inputs work. A 3-D input is flattened over the token axis, addressed, then
    reshaped back, so tokens do not interact.

    **Architecture Overview:**

    .. code-block:: text

        inputs -- INPUT tensor, not a weight
        (batch, input_dim) or (batch, seq_len, input_dim)
                             │
                             ▼
                        input rank
                             │
                  ┌──────────┴──────────┐
                 3-D                   2-D
                  │                     │
                  ▼                     │
        ┌───────────────────┐           │
        │ reshape to        │           │
        │ (batch*seq_len,   │           │
        │  input_dim)       │           │
        └─────────┬─────────┘           │
                  └──────────┬──────────┘
                             ▼
                  inputs_2d (B, input_dim)
                             │
                             ▼
        ┌─────────────────────────────────────────┐
        │ per-dimension projection tower          │
        │ n Dense + softmax, see sub-block below  │
        └────────────────────┬────────────────────┘
                             │  P_1 .. P_n, each (B, d_i)
                             ├─► entropy add_loss     (optional)
                             │     strength > 0 and training
                             ▼
        ┌─────────────────────────────────────────┐
        │ outer product, + epsilon, renormalize   │
        │ joint_prob (B, d1, ..., dn)             │
        └────────────────────┬────────────────────┘
                             ▼
        ┌─────────────────────────────────────────┐
        │ soft lookup, weighted sum over the grid │
        │ reads grid_weights (d1..dn, latent_dim) │
        │ einsum, or flat matmul when n_dims > 6  │
        └────────────────────┬────────────────────┘
                             │  output_2d (B, latent_dim)
                             ▼
                        input rank
                             │
                  ┌──────────┴──────────┐
                 3-D                   2-D
                  │                     │
                  ▼                     │
        ┌───────────────────┐           │
        │ reshape to        │           │
        │ (batch, seq_len,  │           │
        │  latent_dim)      │           │
        └─────────┬─────────┘           │
                  └──────────┬──────────┘
                             ▼
        output (batch, latent_dim) or
               (batch, seq_len, latent_dim)

    **Per-Dimension Projection Tower:**

    .. code-block:: text

        inputs_2d (B, input_dim) -- INPUT tensor, not a weight
                                │
               ┌────────────────┼────────────────┐
               │                │                │
               ▼                ▼                ▼
         ┌───────────┐    ┌───────────┐    ┌───────────┐
         │ Dense d1  │    │ Dense d2  │    │ Dense dn  │
         │ no activ. │    │ no activ. │    │ no activ. │
         └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
               │ logits         │  ...           │
               ▼                ▼                ▼
         ┌───────────┐    ┌───────────┐    ┌───────────┐
         │ divide by │    │ divide by │    │ divide by │
         │ T + eps   │    │ T + eps   │    │ T + eps   │
         │ softmax   │    │ softmax   │    │ softmax   │
         └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
               │                │                │
               ▼                ▼                ▼
            P_1 (B,d1)      P_2 (B,d2)      P_n (B,dn)

        The columns run in a Python loop, not in parallel on device.
        T is one temperature weight shared by every column. It is
        trainable only when learnable_temperature=True (optional);
        otherwise it is a fixed non-trainable weight.

    Input shape:
        2D tensor ``(batch_size, input_dim)`` or 3D tensor
        ``(batch_size, seq_len, input_dim)``.

    Output shape:
        ``(batch_size, latent_dim)`` or ``(batch_size, seq_len,
        latent_dim)`` -- the input rank is preserved and only the last
        axis changes.

    Example:
        >>> grid = NeuroGrid(grid_shape=[10, 8], latent_dim=32)
        >>> y = grid(x)                       # (batch, 32)
        >>> probs = grid.get_addressing_probabilities(x)['joint']
        >>>
        >>> # Sharper addressing, and let the temperature be learned.
        >>> sharp = NeuroGrid(grid_shape=[10, 8], latent_dim=32,
        ...                   temperature=0.1,
        ...                   learnable_temperature=True)
        >>>
        >>> # Push the addressing toward one cell during training.
        >>> sparse = NeuroGrid(grid_shape=[10, 8], latent_dim=32,
        ...                    entropy_regularizer_strength=0.01)

    Note:
        ``grid_regularizer`` defaults to a
        ``SoftOrthonormalConstraintRegularizer``, not to None, so the grid
        vectors are pushed apart unless you pass something else. The
        ``grid_initializer`` default is likewise an
        ``OrthogonalHypersphereInitializer`` instance.

    Note:
        The entropy loss is added only when
        ``entropy_regularizer_strength > 0`` AND ``training`` is true. It
        raises nothing when you set the strength and call the layer outside
        training; it simply does not fire.

    :param grid_shape: List of integers defining grid dimensions, e.g.
        ``[10, 8, 6]`` for a 10x8x6 grid. All values must be positive.
    :type grid_shape: Union[List[int], Tuple[int, ...]]
    :param latent_dim: Dimensionality of each grid latent vector
        (output feature size). Must be positive.
    :type latent_dim: int
    :param use_bias: Whether dense projection layers use bias.
    :type use_bias: bool
    :param temperature: Initial softmax temperature; lower values
        yield sharper addressing. Must be positive.
    :type temperature: float
    :param learnable_temperature: Whether temperature is trainable.
    :type learnable_temperature: bool
    :param entropy_regularizer_strength: Strength of entropy
        regularisation encouraging sharper distributions. Non-negative.
    :type entropy_regularizer_strength: float
    :param kernel_initializer: Initializer for projection Dense kernels. Each
        projection receives its OWN copy, so the per-axis kernels are drawn
        independently rather than all taking the same draw.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for projection Dense biases. Copied
        per projection on the same terms as ``kernel_initializer``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param grid_initializer: Initializer for the grid latent vectors.
    :type grid_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for Dense kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for Dense biases.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param grid_regularizer: Optional regularizer for grid weights.
    :type grid_regularizer: Optional[keras.regularizers.Regularizer]
    :param epsilon: Small positive constant added to the temperature before
        dividing, to the joint probability before renormalising, and inside
        every logarithm. Must be positive.
    :type epsilon: float
    :param kwargs: Forwarded to ``keras.layers.Layer.__init__``.
    :type kwargs: Any

    :ivar n_dims: Rank of the grid, ``len(grid_shape)``.
    :vartype n_dims: int
    :ivar total_grid_size: Number of cells, ``prod(grid_shape)``.
    :vartype total_grid_size: int
    :ivar initial_temperature: The ``temperature`` argument, kept for
        serialization. The live value lives in the ``temperature`` weight.
    :vartype initial_temperature: float
    :ivar projection_layers: One ``Dense`` per grid dimension, created in
        ``__init__`` and built in ``build()``.
    :vartype projection_layers: List[keras.layers.Dense]
    :ivar grid_weights: Trainable variable of shape
        ``(d1, ..., dn, latent_dim)`` holding one latent vector per cell.
        Created in ``build()``.
    :vartype grid_weights: keras.Variable
    :ivar temperature: Scalar weight holding the softmax temperature.
        Trainable and constrained non-negative when
        ``learnable_temperature=True``, otherwise non-trainable. Created in
        ``build()``.
    :vartype temperature: keras.Variable
    :ivar input_is_3d: Whether ``build()`` saw a rank-3 input shape. Recorded
        for inspection; ``call()`` re-derives the rank from its own argument
        and does not read this. Not serialized.
    :vartype input_is_3d: Optional[bool]
    """

    def __init__(
            self,
            grid_shape: Union[List[int], Tuple[int, ...]],
            latent_dim: int,
            use_bias: bool = False,
            temperature: float = 1.0,
            learnable_temperature: bool = False,
            entropy_regularizer_strength: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            grid_initializer: Union[str, keras.initializers.Initializer] = OrthogonalHypersphereInitializer(),
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            grid_regularizer: Optional[keras.regularizers.Regularizer] = SoftOrthonormalConstraintRegularizer(0.1, 0.0, 0.001),
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        """
        Validate the configuration and create the projection layers.

        The grid and the temperature weight are created later, in ``build()``,
        because their shapes depend on the input width.

        See the class docstring for the meaning of every parameter.

        :raises ValueError: If ``grid_shape`` is empty or holds a non-positive
            entry; or if ``latent_dim``, ``temperature`` or ``epsilon`` is not
            positive; or if ``entropy_regularizer_strength`` is negative.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not grid_shape or len(grid_shape) == 0:
            raise ValueError("grid_shape cannot be empty")
        if any(dim <= 0 for dim in grid_shape):
            raise ValueError(f"All grid dimensions must be positive, got {grid_shape}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if entropy_regularizer_strength < 0:
            raise ValueError(f"entropy_regularizer_strength must be non-negative, got {entropy_regularizer_strength}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration
        self.grid_shape = tuple(grid_shape)
        self.latent_dim = latent_dim
        self.initial_temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.entropy_regularizer_strength = entropy_regularizer_strength
        self.epsilon = epsilon
        self.n_dims = len(self.grid_shape)
        self.use_bias = use_bias
        self.total_grid_size = int(np.prod(self.grid_shape))

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.grid_initializer = keras.initializers.get(grid_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.grid_regularizer = keras.regularizers.get(grid_regularizer)

        # Create projection layers in __init__
        self.projection_layers = []
        for i, dim_size in enumerate(self.grid_shape):
            # No activation here. The softmax is applied in call() so the
            # temperature can divide the logits first.
            layer = keras.layers.Dense(
                units=dim_size,
                use_bias=self.use_bias,
                activation=None,
                # One INDEPENDENT initializer copy per projection. A single
                # Initializer instance re-emits the same tensor at every
                # matching shape, which would start all N projections
                # bit-identical. Same pattern as baseline_ntm.py.
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'projection_{i}'
            )
            self.projection_layers.append(layer)

        # Grid weights and temperature created in build()
        self.grid_weights = None
        self.temperature = None
        # Set in build() from the input rank.
        self.input_is_3d = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the grid, the temperature weight, and build the projections.

        The projections are built against ``(None, input_dim)`` so the same
        built layers serve 2-D and 3-D inputs.

        :param input_shape: Shape of the input tensor. Must be rank 2 or
            rank 3, with a defined last axis.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 2 or 3, or if its last
            axis is None.
        """
        if len(input_shape) < 2 or len(input_shape) > 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {input_shape}")

        # For 3D inputs (transformer mode), we use the last dimension (embed_dim)
        # For 2D inputs (traditional mode), we use the last dimension (input_dim)
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension (input/embedding dimension) must be defined")

        # Store input shape info for call method
        self.input_is_3d = len(input_shape) == 3

        # Build projection layers - they work on the last dimension regardless of 2D/3D.
        # The leading axis is left undefined because a 3-D input is flattened
        # to (batch * seq_len, input_dim) before it reaches these layers.
        projection_input_shape = (None, input_dim)
        for layer in self.projection_layers:
            layer.build(projection_input_shape)

        # Create learnable temperature parameter
        if self.learnable_temperature:
            self.temperature = self.add_weight(
                name='temperature',
                shape=(),
                initializer=keras.initializers.Constant(self.initial_temperature),
                # Keeps the learned temperature from going negative.
                constraint=keras.constraints.NonNeg(),
                trainable=True
            )
        else:
            # Fixed temperature as non-trainable weight
            self.temperature = self.add_weight(
                name='temperature',
                shape=(),
                initializer=keras.initializers.Constant(self.initial_temperature),
                trainable=False
            )

        # Create grid weights: (d1, d2, ..., dn, latent_dim)
        grid_weight_shape = self.grid_shape + (self.latent_dim,)
        self.grid_weights = self.add_weight(
            name='grid_weights',
            shape=grid_weight_shape,
            initializer=self.grid_initializer,
            regularizer=self.grid_regularizer,
            trainable=True
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Project, form the joint address, and read the grid.

        A rank-3 input is flattened over the token axis first and reshaped
        back at the end. The entropy loss is added only when
        ``entropy_regularizer_strength > 0`` and ``training`` is true.

        :param inputs: Input tensor, rank 2 or rank 3.
        :type inputs: keras.KerasTensor
        :param training: Whether the caller is in training mode. Gates the
            entropy loss.
        :type training: Optional[bool]
        :return: Output tensor with the same rank as the input and
            ``latent_dim`` on its last axis.
        :rtype: keras.KerasTensor
        """
        original_shape = keras.ops.shape(inputs)
        input_rank = len(inputs.shape)

        # Handle 3D inputs by reshaping to 2D for processing
        if input_rank == 3:
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            # Reshape to (batch_size * seq_len, embed_dim) for processing
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
        else:
            inputs_2d = inputs

        # Get temperature-controlled probability distributions for each dimension
        probabilities = []
        total_entropy_loss = 0.0

        for layer in self.projection_layers:
            # logits: (batch_size [* seq_len], dim_i)
            logits = layer(inputs_2d, training=training)
            # Apply learnable temperature-controlled softmax for sharper/smoother addressing
            scaled_logits = logits / (self.temperature + self.epsilon)
            prob = keras.ops.softmax(scaled_logits, axis=-1)
            probabilities.append(prob)

            # Add entropy regularization to encourage sharper probabilities
            if self.entropy_regularizer_strength > 0.0 and training:
                # Compute entropy: -sum(p * log(p))
                entropy = -keras.ops.sum(prob * keras.ops.log(prob + self.epsilon), axis=-1)
                # Average entropy across batch and add as regularization loss
                avg_entropy = keras.ops.mean(entropy)
                entropy_loss = self.entropy_regularizer_strength * avg_entropy
                total_entropy_loss += entropy_loss

        # Add entropy regularization loss if enabled
        if self.entropy_regularizer_strength > 0.0 and training:
            self.add_loss(total_entropy_loss)

        # Compute joint probability using efficient outer product
        joint_prob = self._compute_joint_probability(probabilities)

        # Perform soft lookup with numerical stability
        output_2d = self._soft_lookup(joint_prob)

        # Restore original shape for 3D inputs (transformer mode)
        if input_rank == 3:
            output = keras.ops.reshape(output_2d, (batch_size, seq_len, self.latent_dim))
        else:
            output = output_2d

        return output

    def _compute_joint_probability(self, probabilities: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Combine the per-axis distributions into one joint distribution.

        Takes the outer product one axis at a time by broadcasting, adds
        ``epsilon``, then renormalises so each row sums to 1.

        :param probabilities: One probability tensor ``(batch, d_i)`` per grid
            dimension, in grid order.
        :type probabilities: List[keras.KerasTensor]
        :return: Joint probability tensor ``(batch, d1, ..., dn)``.
        :rtype: keras.KerasTensor
        """
        # Start with first probability: (batch, d1)
        joint_prob = probabilities[0]

        # Sequentially compute outer products with numerical stability
        for i, prob in enumerate(probabilities[1:], 1):
            # Add new axis for broadcasting: joint_prob becomes (batch, d1, ..., di-1, 1)
            joint_prob = keras.ops.expand_dims(joint_prob, axis=-1)

            # Add axes to prob for proper broadcasting: (batch, 1, ..., 1, di)
            for _ in range(i):
                # On the first pass prob is still (batch, di) and the new axis
                # goes in front; afterwards it goes before the last axis.
                if len(prob.shape) == 2:
                    prob = keras.ops.expand_dims(prob, axis=1)
                else:
                    prob = keras.ops.expand_dims(prob, axis=-2)

            # Element-wise multiplication gives outer product
            joint_prob = joint_prob * prob

        # Add small epsilon for numerical stability
        joint_prob = joint_prob + self.epsilon

        # Renormalize to ensure probabilities sum to 1
        prob_sum = keras.ops.sum(
            joint_prob,
            axis=tuple(range(1, len(joint_prob.shape))),
            keepdims=True
        )
        joint_prob = joint_prob / (prob_sum + self.epsilon)

        return joint_prob

    def _soft_lookup(self, joint_prob: keras.KerasTensor) -> keras.KerasTensor:
        """
        Read the grid, weighting every cell by its joint probability.

        Builds an einsum equation such as ``'bijk,ijkz->bz'`` from the grid
        rank. Grids of rank above 6 take a flat ``matmul`` instead, which
        computes the same sum.

        :param joint_prob: Joint probability tensor ``(batch, d1, ..., dn)``.
        :type joint_prob: keras.KerasTensor
        :return: Weighted sum tensor ``(batch, latent_dim)``.
        :rtype: keras.KerasTensor
        """
        # Create einsum equation for the weighted sum
        # joint_prob: (batch, d1, d2, ..., dn)
        # grid_weights: (d1, d2, ..., dn, latent_dim)
        # output: (batch, latent_dim)

        # Build einsum equation dynamically based on grid dimensions
        batch_idx = 'b'
        # Grid axes take one letter each from GRID_EINSUM_SUBSCRIPTS. Do NOT go
        # back to `chr(ord('i') + j)` capped by a number: the run 'i'..'z' is 18
        # characters but its 18th IS 'z', which this equation already spends on
        # the latent axis, so 18 emits 'ijklmnopqrstuvwxyz,ijklmnopqrstuvwxyzz->bz'
        # and TensorFlow raises InvalidArgumentError on the repeated 'z'. The
        # honest maximum is 17 and it lives in the constant, where a test can
        # address it. Only n_dims <= 6 reaches here, so the slice never truncates.
        grid_indices = GRID_EINSUM_SUBSCRIPTS[:self.n_dims]
        latent_idx = 'z'

        # joint_prob indices: batch + grid dimensions
        joint_indices = batch_idx + grid_indices

        # grid_weights indices: grid dimensions + latent
        grid_indices_with_latent = grid_indices + latent_idx

        # output indices: batch + latent
        output_indices = batch_idx + latent_idx

        # Einsum equation: e.g., 'bijk,ijkz->bz' for 3D grid
        equation = f"{joint_indices},{grid_indices_with_latent}->{output_indices}"

        # For very high dimensional grids, fall back to manual computation:
        # einsum gets slow and unreliable past a handful of index letters.
        if self.n_dims > 6:
            # Reshape for manual computation
            batch_size = keras.ops.shape(joint_prob)[0]
            joint_flat = keras.ops.reshape(joint_prob, (batch_size, self.total_grid_size))
            grid_flat = keras.ops.reshape(self.grid_weights, (self.total_grid_size, self.latent_dim))
            output = keras.ops.matmul(joint_flat, grid_flat)
        else:
            # Use einsum for lower dimensional grids
            output = keras.ops.einsum(equation, joint_prob, self.grid_weights)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Replace the last axis with ``latent_dim`` and keep the rest.

        :param input_shape: Input shape tuple, rank 2 or rank 3.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple of the same rank.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 2 or rank 3.
        """
        if len(input_shape) == 2:
            # 2D input: (batch_size, input_dim) → (batch_size, latent_dim)
            return (input_shape[0], self.latent_dim)
        elif len(input_shape) == 3:
            # 3D input: (batch_size, seq_len, embed_dim) → (batch_size, seq_len, latent_dim)
            return (input_shape[0], input_shape[1], self.latent_dim)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this layer.

        ``input_is_3d`` is not included: ``build()`` derives it from the
        input shape.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'use_bias': self.use_bias,
            'grid_shape': list(self.grid_shape),
            'latent_dim': self.latent_dim,
            'temperature': self.initial_temperature,
            'learnable_temperature': self.learnable_temperature,
            'entropy_regularizer_strength': self.entropy_regularizer_strength,
            'epsilon': self.epsilon,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'grid_initializer': keras.initializers.serialize(self.grid_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'grid_regularizer': keras.regularizers.serialize(self.grid_regularizer),
        })
        return config

    def get_grid_weights(self) -> keras.KerasTensor:
        """
        Return the grid variable itself, for analysis or visualisation.

        This is the live variable, not a copy.

        :return: Grid weights ``(d1, d2, ..., dn, latent_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the layer has not been built.
        """
        if self.grid_weights is None:
            raise ValueError("Layer must be built before accessing grid weights")
        return self.grid_weights

    def get_addressing_probabilities(
            self,
            inputs: keras.KerasTensor
    ) -> Dict[str, Union[List[keras.KerasTensor], keras.KerasTensor]]:
        """
        Recompute the addressing distributions without running the lookup.

        Repeats the projection and softmax steps of ``call()``, then the outer
        product. A rank-3 input is flattened over the token axis and the
        results are NOT reshaped back, so every returned tensor has a leading
        axis of ``batch * seq_len``. No loss is added.

        :param inputs: Input tensor, rank 2 or rank 3.
        :type inputs: keras.KerasTensor
        :return: Dictionary with ``'individual'`` (a list of per-dimension
            probability tensors ``(B, d_i)``), ``'joint'`` (one tensor
            ``(B, d1, ..., dn)``) and ``'entropy'`` (a list of per-dimension
            entropy tensors ``(B,)``).
        :rtype: Dict[str, Union[List[keras.KerasTensor], keras.KerasTensor]]
        :raises ValueError: If the layer has not been built.
        """
        if not self.built:
            raise ValueError("Layer must be built before getting probabilities")

        # Handle input reshaping for 3D inputs
        input_rank = len(inputs.shape)
        if input_rank == 3:
            original_shape = keras.ops.shape(inputs)
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
        else:
            inputs_2d = inputs

        # Get individual probabilities (recompute to match forward pass)
        probabilities = []
        entropies = []

        for layer in self.projection_layers:
            logits = layer(inputs_2d)
            scaled_logits = logits / (self.temperature + self.epsilon)
            prob = keras.ops.softmax(scaled_logits, axis=-1)
            probabilities.append(prob)

            # Compute entropy as uncertainty measure
            entropy = -keras.ops.sum(prob * keras.ops.log(prob + self.epsilon), axis=-1)
            entropies.append(entropy)

        # Compute joint probability
        joint_prob = self._compute_joint_probability(probabilities)

        return {
            'individual': probabilities,
            'joint': joint_prob,
            'entropy': entropies
        }

    def get_grid_utilization(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """
        Report how much of the grid a batch of inputs actually uses.

        Counts, per flattened cell, how many inputs peak there, and sums the
        joint probability mass landing on each cell.

        Every item lands in exactly one cell, so ``activation_counts`` totals
        the number of items scored and ``utilization_rate`` is that count over
        the same total: it sums to 1.0 and no entry exceeds 1.0. A rank-3 input
        is scored PER TOKEN, so "item" means ``batch * seq_len`` there and
        ``batch`` at rank 2. Both rates are therefore comparable across ranks.

        :param inputs: Input tensor, rank 2 or rank 3.
        :type inputs: keras.KerasTensor
        :return: Dictionary with ``'activation_counts'`` and
            ``'total_activation'``, both of shape ``(total_grid_size,)``, and
            ``'utilization_rate'`` of the same shape.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If the layer has not been built.
        """
        if not self.built:
            raise ValueError("Layer must be built before computing utilization")

        prob_info = self.get_addressing_probabilities(inputs)
        joint_prob = prob_info['joint']

        # Find most activated position per input
        joint_prob_flat = keras.ops.reshape(joint_prob, (keras.ops.shape(joint_prob)[0], -1))
        max_positions = keras.ops.argmax(joint_prob_flat, axis=-1)

        # Fix: Replace scatter_update with proper counting using one-hot
        activation_counts = keras.ops.zeros((self.total_grid_size,))

        # Convert positions to one-hot and sum
        one_hot_positions = keras.ops.one_hot(max_positions, self.total_grid_size)
        activation_counts = keras.ops.sum(one_hot_positions, axis=0)

        # Total activations per position (sum of all probabilities)
        total_activation = keras.ops.sum(joint_prob_flat, axis=0)

        # Utilization rate (normalized). The divisor is the leading axis of the
        # FLATTENED joint probability, i.e. the number of items actually counted:
        # `batch` at rank 2 but `batch * seq_len` at rank 3. Using
        # `shape(inputs)[0]` instead divides a token count by a row count and
        # yields "rates" above 1.0 on any input with seq_len > 1.
        total_items = keras.ops.cast(keras.ops.shape(joint_prob_flat)[0], 'float32')
        utilization_rate = activation_counts / (total_items + self.epsilon)

        return {
            'activation_counts': activation_counts,
            'total_activation': total_activation,
            'utilization_rate': utilization_rate
        }

    def find_best_matching_units(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """
        Find the single most probable grid cell for each input.

        Note the two key names read backwards from what you might expect.
        ``'bmu_coordinates'`` is the FLAT argmax index and
        ``'bmu_indices'`` is the unravelled per-axis coordinate.

        :param inputs: Input tensor, rank 2 or rank 3. A rank-3 input is
            flattened over the token axis, so the leading axis of every
            returned tensor is ``batch * seq_len``.
        :type inputs: keras.KerasTensor
        :return: Dictionary with ``'bmu_indices'`` of shape ``(B, n_dims)``,
            ``'bmu_probabilities'`` of shape ``(B,)``, and
            ``'bmu_coordinates'`` of shape ``(B,)``.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If the layer has not been built.
        """
        if not self.built:
            raise ValueError("Layer must be built before finding BMUs")

        prob_info = self.get_addressing_probabilities(inputs)
        joint_prob = prob_info['joint']

        # Find maximum probability positions
        joint_prob_flat = keras.ops.reshape(joint_prob, (keras.ops.shape(joint_prob)[0], -1))
        bmu_coordinates = keras.ops.argmax(joint_prob_flat, axis=-1)
        bmu_probabilities = keras.ops.max(joint_prob_flat, axis=-1)

        # Convert flat indices back to n-dimensional indices
        bmu_indices = []
        remaining = bmu_coordinates

        for dim_size in reversed(self.grid_shape):
            dim_indices = remaining % dim_size
            bmu_indices.append(dim_indices)
            remaining = remaining // dim_size

        # Reverse to get correct order and stack
        bmu_indices = keras.ops.stack(list(reversed(bmu_indices)), axis=-1)

        return {
            'bmu_indices': bmu_indices,
            'bmu_probabilities': bmu_probabilities,
            'bmu_coordinates': bmu_coordinates
        }

    def set_temperature(self, new_temperature: float) -> None:
        """
        Overwrite the temperature weight in place.

        Useful for annealing the addressing from diffuse to sharp during
        training. The assignment happens even when
        ``learnable_temperature=False``.

        :param new_temperature: New temperature value. Must be positive.
        :type new_temperature: float
        :raises ValueError: If ``new_temperature`` is not positive, or if the
            layer has not been built.
        """
        if new_temperature <= 0:
            raise ValueError(f"temperature must be positive, got {new_temperature}")
        if self.temperature is None:
            raise ValueError("Layer must be built before setting temperature")

        # Update the temperature weight value
        self.temperature.assign(new_temperature)

    def get_current_temperature(self) -> float:
        """
        Read the temperature weight back as a Python float.

        :return: Current temperature.
        :rtype: float
        :raises ValueError: If the layer has not been built.
        """
        if self.temperature is None:
            raise ValueError("Layer must be built before getting temperature")

        return float(keras.ops.convert_to_numpy(self.temperature))

    def compute_input_quality(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """
        Score how cleanly each input addresses the grid.

        The premise is that a sharp, confident address means the input looks
        like something the grid has learned, and a diffuse one means it does
        not. Six measures are returned:

        - ``addressing_confidence`` -- peak joint probability, in ``[0, 1]``.
        - ``addressing_entropy`` -- entropy of the joint distribution. Lower
          is sharper. Not normalised.
        - ``dimension_consistency`` -- mean per-axis peak probability, in
          ``[0, 1]``.
        - ``grid_coherence`` -- ``1 / (1 + var(joint))``, in ``(0, 1]``.
        - ``uncertainty`` -- mean per-axis entropy. Not normalised.
        - ``overall_quality`` -- weighted mix of the five, in ``[0, 1]``.
          The two entropies are first inverted and clipped against their
          theoretical maxima, ``log(total_grid_size)`` and
          ``log(max(grid_shape))``.

        For a rank-2 input each measure has shape ``(batch,)``. For a rank-3
        input each is computed per token and reshaped to
        ``(batch, seq_len)``.

        :param inputs: Input tensor, rank 2 or rank 3.
        :type inputs: keras.KerasTensor
        :return: Dictionary of the six measures above.
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If the layer has not been built.
        """
        if not self.built:
            raise ValueError("Layer must be built before computing quality")

        original_shape = keras.ops.shape(inputs)
        input_rank = len(inputs.shape)

        # Handle 3D inputs by reshaping to 2D for processing
        if input_rank == 3:
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
            effective_batch_size = batch_size * seq_len
        else:
            inputs_2d = inputs
            effective_batch_size = keras.ops.shape(inputs)[0]

        prob_info = self.get_addressing_probabilities(inputs_2d)
        individual_probs = prob_info['individual']
        joint_prob = prob_info['joint']
        entropies = prob_info['entropy']

        # 1. Addressing Confidence: Maximum probability in joint distribution
        joint_prob_flat = keras.ops.reshape(joint_prob, (effective_batch_size, -1))
        addressing_confidence = keras.ops.max(joint_prob_flat, axis=-1)

        # 2. Addressing Entropy: Entropy of joint probability distribution (lower = better quality)
        joint_entropy = -keras.ops.sum(
            joint_prob_flat * keras.ops.log(joint_prob_flat + self.epsilon),
            axis=-1
        )

        # 3. Dimension Consistency: How sharp/consistent are individual dimensions
        dimension_sharpness = []
        for prob in individual_probs:
            # Higher max probability = more consistent/sharp addressing
            max_prob = keras.ops.max(prob, axis=-1)
            dimension_sharpness.append(max_prob)

        # Average sharpness across all dimensions
        dimension_consistency = keras.ops.mean(keras.ops.stack(dimension_sharpness, axis=-1), axis=-1)

        # 4. Grid Coherence: Measure based on probability distribution spread
        # Lower spread (higher concentration) indicates better mapping to grid structure
        prob_variance = keras.ops.var(joint_prob_flat, axis=-1)
        # Invert so higher values mean better coherence
        grid_coherence = 1.0 / (1.0 + prob_variance)

        # 5. Combined Uncertainty: Average of individual dimension entropies
        avg_dimension_entropy = keras.ops.mean(keras.ops.stack(entropies, axis=-1), axis=-1)
        uncertainty = avg_dimension_entropy

        # 6. Overall Quality Score: Composite measure (0-1 scale)
        # Normalize and combine multiple factors.
        # These three are already in [0, 1] and are renamed, not rescaled:
        # two are peak probabilities and one is 1 / (1 + variance).
        confidence_norm = addressing_confidence
        consistency_norm = dimension_consistency
        coherence_norm = grid_coherence

        # Entropy-based terms (invert and normalize to 0-1)
        max_joint_entropy = keras.ops.log(keras.ops.cast(self.total_grid_size, 'float32'))
        entropy_quality = 1.0 - (joint_entropy / (max_joint_entropy + self.epsilon))
        entropy_quality = keras.ops.clip(entropy_quality, 0.0, 1.0)

        max_dim_entropy = keras.ops.log(keras.ops.cast(keras.ops.max(keras.ops.array(self.grid_shape)), 'float32'))
        uncertainty_quality = 1.0 - (uncertainty / (max_dim_entropy + self.epsilon))
        uncertainty_quality = keras.ops.clip(uncertainty_quality, 0.0, 1.0)

        # Weighted combination of quality factors. The five weights sum to 1.0
        # and every term is in [0, 1], so overall_quality is in [0, 1] too.
        # The two entropy-derived terms carry 0.40 of the total between them.
        overall_quality = (
                0.25 * confidence_norm +
                0.25 * entropy_quality +
                0.20 * consistency_norm +
                0.15 * coherence_norm +
                0.15 * uncertainty_quality
        )

        # Reshape results back to match input format for 3D inputs
        if input_rank == 3:
            # Reshape from (batch * seq_len,) to (batch, seq_len)
            addressing_confidence = keras.ops.reshape(addressing_confidence, (batch_size, seq_len))
            joint_entropy = keras.ops.reshape(joint_entropy, (batch_size, seq_len))
            dimension_consistency = keras.ops.reshape(dimension_consistency, (batch_size, seq_len))
            grid_coherence = keras.ops.reshape(grid_coherence, (batch_size, seq_len))
            uncertainty = keras.ops.reshape(uncertainty, (batch_size, seq_len))
            overall_quality = keras.ops.reshape(overall_quality, (batch_size, seq_len))

        return {
            'addressing_confidence': addressing_confidence,
            'addressing_entropy': joint_entropy,
            'dimension_consistency': dimension_consistency,
            'grid_coherence': grid_coherence,
            'uncertainty': uncertainty,
            'overall_quality': overall_quality
        }

    def get_quality_statistics(self, inputs: keras.KerasTensor) -> Dict[str, float]:
        """
        Summarise ``compute_input_quality`` over a whole batch.

        Runs ``compute_input_quality`` and reduces each of its six measures to
        mean, std, min, max and median, giving 30 floats. The reduction is
        over every element, so for a rank-3 input it pools across tokens as
        well as across the batch.

        The numbers are for monitoring: watching the mean drift, or the std
        widen, tells you the incoming data has changed. A gap between mean and
        median points at outliers. There are no built-in thresholds; pick them
        from your own data.

        This converts to NumPy, so it does not belong inside a compiled
        training step.

        :param inputs: Input tensor, rank 2 or rank 3.
        :type inputs: keras.KerasTensor
        :return: Dictionary keyed ``'{measure}_{stat}'`` with ``stat`` one of
            ``mean``, ``std``, ``min``, ``max``, ``median``. Values are Python
            floats.
        :rtype: Dict[str, float]
        :raises ValueError: If the layer has not been built.
        """
        quality_measures = self.compute_input_quality(inputs)

        statistics = {}
        for measure_name, measure_values in quality_measures.items():
            measure_np = keras.ops.convert_to_numpy(measure_values)
            statistics[f"{measure_name}_mean"] = float(np.mean(measure_np))
            statistics[f"{measure_name}_std"] = float(np.std(measure_np))
            statistics[f"{measure_name}_min"] = float(np.min(measure_np))
            statistics[f"{measure_name}_max"] = float(np.max(measure_np))
            statistics[f"{measure_name}_median"] = float(np.median(measure_np))

        return statistics

    def filter_by_quality_threshold(
            self,
            inputs: keras.KerasTensor,
            quality_threshold: float = 0.5,
            quality_measure: str = 'overall_quality'
    ) -> Dict[str, keras.KerasTensor]:
        """
        Partition the scored items into a high-quality and a low-quality set.

        Scores every item with ``compute_input_quality``, compares one chosen
        measure against the threshold, and gathers the two subsets. The
        comparison is ``>=`` for the high half and ``<`` for the low half, so
        the two sets are disjoint and, for finite scores, cover every item.

        The unit being partitioned is the unit ``compute_input_quality``
        scores. For a rank-2 input that is the ``batch`` rows; for a rank-3
        input it is the ``batch * seq_len`` TOKENS, which are scored
        individually. Both returned subsets are therefore rank 2, shaped
        ``(n_selected, input_dim)`` -- a rank-3 input comes back flattened to
        its token axis, not re-assembled into sequences, because an arbitrary
        threshold does not select whole sequences. ``high_quality_mask`` and
        ``quality_scores`` keep the shape ``compute_input_quality`` gave them,
        so the mask is the map back to the original layout.

        :param inputs: Input tensor, rank 2 or rank 3.
        :type inputs: keras.KerasTensor
        :param quality_threshold: Cut-off applied to the chosen measure.
        :type quality_threshold: float
        :param quality_measure: Which key of ``compute_input_quality`` to
            threshold on.
        :type quality_measure: str
        :return: Dictionary with ``'high_quality_inputs'``,
            ``'low_quality_inputs'`` (both ``(n_selected, input_dim)``),
            ``'high_quality_mask'`` and ``'quality_scores'`` (both shaped like
            the measure).
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If ``quality_measure`` is not one of the six
            measure names, or if the layer has not been built.
        """
        quality_measures = self.compute_input_quality(inputs)

        if quality_measure not in quality_measures:
            raise ValueError(f"Unknown quality measure: {quality_measure}")

        quality_scores = quality_measures[quality_measure]
        high_quality_mask = quality_scores >= quality_threshold
        low_quality_mask = quality_scores < quality_threshold

        # Flatten to the ITEM axis that compute_input_quality scores: rank-2
        # rows stay as they are, rank-3 tokens collapse from (batch, seq_len)
        # to (batch * seq_len). Gathering rank-3 input along axis 0 with the
        # mask's row indices would gather whole sequences, repeatedly.
        items = keras.ops.reshape(inputs, (-1, keras.ops.shape(inputs)[-1]))
        high_quality_mask_flat = keras.ops.reshape(high_quality_mask, (-1,))
        low_quality_mask_flat = keras.ops.reshape(low_quality_mask, (-1,))

        # DECISION plan-2026-08-30T063229-ccd6ad17/D-014
        # A single-argument keras.ops.where returns a LIST of per-axis index
        # tensors, so index it [0], never [:, 0] -- that slice raised TypeError
        # at EVERY rank. The flatten above is not optional either: [0] alone
        # gathers whole sequences at rank 3. See decisions.md D-014.
        high_quality_indices = keras.ops.where(high_quality_mask_flat)[0]
        low_quality_indices = keras.ops.where(low_quality_mask_flat)[0]

        high_quality_inputs = keras.ops.take(items, high_quality_indices, axis=0)
        low_quality_inputs = keras.ops.take(items, low_quality_indices, axis=0)

        return {
            'high_quality_inputs': high_quality_inputs,
            'low_quality_inputs': low_quality_inputs,
            'high_quality_mask': high_quality_mask,
            'quality_scores': quality_scores
        }

# ---------------------------------------------------------------------
