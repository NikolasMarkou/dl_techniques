"""
Rigid simplex projection with learnable rotation and bounded scaling.

This layer embodies the principle of constrained geometric parameterization, a
design paradigm that fixes the geometry of a weight matrix and lets training
adjust only how the input is presented to it. The core idea is to replace a
freely learned linear map with a frozen Equiangular Tight Frame, so the layer
performs template matching against a rigid simplex of maximally separated
directions rather than discovering arbitrary filter shapes. The learnable
degrees of freedom collapse from a full `input_dim x units` matrix to a rotation
and a single scalar, which constrains what the layer can express while
guaranteeing properties that an unconstrained matrix must learn by luck.

The frozen matrix is a regular simplex centered at the origin, which is the
canonical ETF. Its defining property is maximum repulsion: `N` dimensions admit
only `N` mutually orthogonal vectors, so fitting `N+1` symmetric directions into
the same space forces them past orthogonality into mutual obtuseness. For unit
rows `v_i`:

`v_i . v_i = 1`
`v_i . v_j = -1/N`   for `i != j`

The corresponding Gram matrix carries ones on its diagonal and `-1/N` elsewhere,
which is the theoretical minimum coherence achievable for `N+1` vectors. Three
consequences follow, each of which matters for a network:

1.  **Maximal diversity.** Because off-diagonal coherence is at its lower bound,
    no two output directions are more aligned than geometry permits, so no pair
    of units is redundant by construction.
2.  **Rank deficiency and equilibrium.** With `N+1` rows in `N`-dimensional
    space the matrix has rank `N`, and the rows are linearly dependent in the
    strongest possible way: `sum_i v_i = 0`. The frame is in exact equilibrium,
    with no net direction favored.
3.  **Isometry and whiteness.** The covariance matrix is a scaled identity,
    `V^T V = ((N+1)/N) I`, so the map neither stretches nor shrinks signal
    magnitude and its columns are perfectly uncorrelated. This is the
    initialization property that gradient-preserving schemes approximate; here
    it holds exactly and is never degraded by training, since the frame is
    non-trainable.

Architecturally, the forward pass composes three stages:

`output = s * (x R) S`

where `S` is the frozen simplex of shape `(input_dim, units)`, `R` is a
trainable `(input_dim, input_dim)` rotation, and `s` is a scalar. Because the
simplex is fixed, alignment is entirely the rotation's job: `R` learns which
directions of the input space should map onto which vertices of the simplex.
Orthogonality of `R` is enforced softly, by an auxiliary penalty rather than a
hard reparameterization:

`L_ortho = lambda * ||R^T R - I||^2`

A soft constraint is the pragmatic choice here because hard orthogonalization
would require a matrix decomposition or a Cayley/exponential map on every step,
and small departures from orthogonality are harmless: they perturb the isometry
of the composite map slightly rather than breaking it. The scale `s` is held
inside `[scale_min, scale_max]` by a value-range constraint applied after each
update, which restores the one degree of magnitude freedom the frozen frame
removes while preventing it from growing without bound and defeating the purpose
of the fixed geometry.

When `units` exceeds the `input_dim + 1` vertices a simplex provides, the frame
is tiled and truncated to the requested width. This necessarily repeats
directions, so the equiangular and isometry guarantees hold only within each
tile rather than across the full output.

References:
    - Papyan et al., 2020. Prevalence of Neural Collapse during the Terminal
      Phase of Deep Learning Training. (https://arxiv.org/abs/2008.08186)
    - Strohmer and Heath, 2003. Grassmannian Frames with Applications to Coding
      and Communication. Applied and Computational Harmonic Analysis 14(3).
    - Saxe et al., 2014. Exact Solutions to the Nonlinear Dynamics of Learning
      in Deep Linear Networks. (https://arxiv.org/abs/1312.6120)
    - Bansal et al., 2018. Can We Gain More from Orthogonality Regularizations
      in Training Deep Networks? (https://arxiv.org/abs/1810.09102)

"""

import keras
import numpy as np
from keras import ops
from keras import initializers
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RigidSimplexLayer(keras.layers.Layer):
    """Project inputs onto a rigid Simplex with learnable rotation and scaling.

    The layer maintains a frozen Equiangular Tight Frame (Simplex) weight
    matrix whose rows are maximally separated unit vectors
    (``v_i . v_j = -1/N`` for ``i != j``). Rather than learning arbitrary
    filters, the network learns only a rotation matrix ``R`` (softly
    constrained towards orthogonality via
    ``Loss = lambda * ||R^T R - I||^2``) and a bounded global scale
    ``s in [scale_min, scale_max]``. The forward pass is:
    ``output = s * (x @ R) @ Simplex``.
    This forces the model to perform template matching against a fixed
    crystal structure rather than discovering arbitrary filter shapes.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [..., input_dim]          │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  x @ rotation_kernel             │
        │  [input_dim, input_dim]          │
        │  (trainable, soft ortho loss)    │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  rotated @ static_simplex        │
        │  [input_dim, units]              │
        │  (NON-TRAINABLE, frozen ETF)     │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  * global_scale                  │
        │  (bounded [scale_min, scale_max])│
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Output [..., units]             │
        └──────────────────────────────────┘

    :param units: Dimensionality of the output space (Simplex projections).
    :type units: int
    :param scale_min: Minimum allowed scaling factor.
    :type scale_min: float
    :param scale_max: Maximum allowed scaling factor.
    :type scale_max: float
    :param orthogonality_penalty: Weight for the orthogonality
        regularisation loss on the rotation kernel.
    :type orthogonality_penalty: float
    :param rotation_initializer: Initializer for the rotation matrix.
    :type rotation_initializer: Union[str, initializers.Initializer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
            self,
            units: int,
            scale_min: float = 0.5,
            scale_max: float = 2.0,
            orthogonality_penalty: float = 1e-4,
            rotation_initializer: Union[str, initializers.Initializer] = 'identity',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if scale_min >= scale_max:
            raise ValueError(
                f"scale_min ({scale_min}) must be less than scale_max ({scale_max})"
            )
        if orthogonality_penalty < 0:
            raise ValueError(
                f"orthogonality_penalty must be non-negative, got {orthogonality_penalty}"
            )

        # Store configuration
        self.units = units
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.orthogonality_penalty = orthogonality_penalty
        self.rotation_initializer = initializers.get(rotation_initializer)

        # Weight attributes - created in build()
        self.static_simplex = None
        self.rotation_kernel = None
        self.global_scale = None
        self._input_dim = None

    def _create_simplex_matrix(
            self,
            input_dim: int,
            output_dim: int
    ) -> np.ndarray:
        """Generate a centred, normalised Simplex weight matrix.

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param output_dim: Number of Simplex projections.
        :type output_dim: int
        :return: Weight matrix ``(input_dim, output_dim)`` as float32.
        :rtype: np.ndarray"""
        dimensions = input_dim
        matrix = np.identity(dimensions, dtype=np.float32)

        # Calculate the last point to be equidistant from all others
        # This creates a regular simplex in N dimensions
        last_point = np.ones((1, dimensions), dtype=np.float32) * \
                     ((1.0 + np.sqrt(dimensions + 1.0)) / dimensions)

        matrix = np.vstack([matrix, last_point])

        # Center points at origin
        mean_m = np.mean(matrix, axis=0)
        matrix = matrix - mean_m

        # Normalize to unit vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        matrix = matrix / norms

        # Transpose to get (input_dim, N+1) shape
        W = matrix.T

        # Tile or slice to match requested output_dim
        current_cols = W.shape[1]

        if output_dim > current_cols:
            tile_factor = int(np.ceil(output_dim / current_cols))
            W = np.tile(W, (1, tile_factor))

        W = W[:, :output_dim]

        return W.astype(np.float32)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create Simplex, rotation kernel, and scale weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        self._input_dim = input_dim

        # 1. Static Simplex (frozen weights - geometry remains rigid)
        simplex_weights = self._create_simplex_matrix(input_dim, self.units)
        self.static_simplex = self.add_weight(
            name='static_simplex',
            shape=(input_dim, self.units),
            initializer=initializers.Constant(simplex_weights),
            trainable=False,
            dtype=self.dtype,
        )

        # 2. Trainable rotation matrix (learns optimal input alignment)
        self.rotation_kernel = self.add_weight(
            name='rotation_kernel',
            shape=(input_dim, input_dim),
            initializer=self.rotation_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        # 3. Bounded scaling factor
        self.global_scale = self.add_weight(
            name='global_scale',
            shape=(1,),
            initializer=initializers.Constant(1.0),
            constraint=ValueRangeConstraint(min_value=self.scale_min, max_value=self.scale_max),
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass: rotate, project onto Simplex, and scale.

        :param inputs: Input tensor ``(batch, ..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor ``(batch, ..., units)``.
        :rtype: keras.KerasTensor"""
        # 1. Add orthogonality regularization loss (soft constraint for rotation)
        # R^T * R should approximate Identity for valid rotation
        r_t_r = ops.matmul(
            ops.transpose(self.rotation_kernel),
            self.rotation_kernel
        )
        identity = ops.eye(self._input_dim, dtype=self.dtype)
        ortho_loss = ops.mean(ops.square(r_t_r - identity))
        self.add_loss(self.orthogonality_penalty * ortho_loss)

        # 2. Rotate inputs to align with Simplex
        rotated_inputs = ops.matmul(inputs, self.rotation_kernel)

        # 3. Project onto static Simplex
        outputs = ops.matmul(rotated_inputs, self.static_simplex)

        # 4. Apply bounded scaling
        outputs = outputs * self.global_scale

        return outputs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape from input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension replaced by ``units``.
        :rtype: Tuple[Optional[int], ...]"""
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'units': self.units,
            'scale_min': self.scale_min,
            'scale_max': self.scale_max,
            'orthogonality_penalty': self.orthogonality_penalty,
            'rotation_initializer': initializers.serialize(self.rotation_initializer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RigidSimplexLayer':
        """Create a layer instance from a configuration dictionary.

        :param config: Configuration from ``get_config()``.
        :type config: Dict[str, Any]
        :return: New ``RigidSimplexLayer`` instance.
        :rtype: RigidSimplexLayer"""
        if 'rotation_initializer' in config:
            config['rotation_initializer'] = initializers.deserialize(
                config['rotation_initializer']
            )
        return cls(**config)

# ---------------------------------------------------------------------
