"""RigidSimplexLayer, a projection onto a frozen Equiangular Tight Frame.

RigidSimplexLayer is a Keras layer that maps `[..., input_dim]` to
`[..., units]`. Its projection matrix is a fixed regular simplex frame:
`input_dim + 1` unit vectors in `input_dim` dimensions, every pair at inner
product `-1/input_dim`. That matrix never trains. The learnable parts are a
square rotation `R` and a scalar `s`, giving

    output = s * (x @ R) @ Simplex

with `R` pulled toward orthogonality by a penalty on the mean squared
entries of `R^T R - I`. The penalty is added through `add_loss` on every
call, at inference as well as in training. The frame carries only
`input_dim + 1` distinct vectors, so `units` past that count repeats
columns and `units` below it drops them. The scale starts at 1.0 and is
clipped into `[scale_min, scale_max]` after the first update. The last
input dimension must be static.

References:
    - Papyan et al., 2020. Prevalence of Neural Collapse during the Terminal
      Phase of Deep Learning Training. (https://arxiv.org/abs/2008.08186)
    - Strohmer and Heath, 2003. Grassmannian Frames with Applications to
      Coding and Communication. (Applied and Computational Harmonic Analysis 14(3))
    - Saxe et al., 2014. Exact Solutions to the Nonlinear Dynamics of
      Learning in Deep Linear Networks. (https://arxiv.org/abs/1312.6120)
    - Bansal et al., 2018. Can We Gain More from Orthogonality
      Regularizations in Training Deep Networks? (https://arxiv.org/abs/1810.09102)
"""

import keras
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.structured_linear.rigid_simplex_layer")
class RigidSimplexLayer(keras.layers.Layer):
    """
    Project inputs onto a rigid simplex with a learnable rotation and scale.

    Holds a non-trainable simplex frame of shape ``(input_dim, units)`` whose
    columns are unit vectors at pairwise inner product ``-1/input_dim``. The
    layer learns a rotation ``R`` and a bounded scalar ``s``, so the composite
    map is ``output = s * (x @ R) @ Simplex``. ``R`` is not reparameterized;
    it is pushed toward orthogonality by an auxiliary loss equal to
    ``orthogonality_penalty * mean((R^T R - I)^2)``.

    Architecture:

    .. code-block:: text

        input [..., input_dim]
              │
              ▼
        ┌──────────────────────────┐
        │ rotation_kernel matmul   ├──► ortho penalty ──► add_loss
        │ [input_dim, input_dim]   │    (trainable R)
        └──────────────────────────┘
              │ [..., input_dim]
              ▼
        ┌──────────────────────────┐
        │ static_simplex matmul    │    (frozen etf)
        │ [input_dim, units]       │
        └──────────────────────────┘
              │ [..., units]
              ▼
        ┌──────────────────────────┐
        │ global_scale multiply    │    (clipped scalar)
        │ [1]                      │
        └──────────────────────────┘
              │
              ▼
        output [..., units]

    The penalty branch runs on every call, training or not.

    Frame width:

    .. code-block:: text

        vertices = input_dim + 1 distinct unit vectors

        units <  vertices        columns dropped
                                 angles hold, frame no longer tight
        units == k * vertices    exact frame, operator = k*(N+1)/N * I
        otherwise                columns tiled then cut
                                 repeated columns sit at coherence 1

    Input shape:
        N-D tensor ``(..., input_dim)``. The last dimension must be static.

    Output shape:
        Same rank as the input, with the last dimension set to ``units``.

    Note:
        ``global_scale`` is initialized at 1.0 and the range constraint is
        applied after an optimizer update, so a range that excludes 1.0
        leaves the first forward pass outside the bounds.

    :param units: Dimensionality of the output space (Simplex projections).
    :type units: int
    :param scale_min: Minimum allowed scaling factor.
    :type scale_min: float
    :param scale_max: Maximum allowed scaling factor. Must exceed ``scale_min``.
    :type scale_max: float
    :param orthogonality_penalty: Weight for the orthogonality regularisation loss on the rotation kernel.
    :type orthogonality_penalty: float
    :param rotation_initializer: Initializer for the rotation matrix. Defaults
        to ``'identity'``, which starts the layer as the bare frame projection.
    :type rotation_initializer: Union[str, initializers.Initializer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :ivar static_simplex: Frozen frame of shape ``(input_dim, units)``.
    :vartype static_simplex: keras.Variable
    :ivar rotation_kernel: Trainable rotation of shape ``(input_dim, input_dim)``.
    :vartype rotation_kernel: keras.Variable
    :ivar global_scale: Trainable scalar of shape ``(1,)``, range constrained.
    :vartype global_scale: keras.Variable

    :raises ValueError: If ``units`` is not positive.
    :raises ValueError: If ``scale_min`` is not less than ``scale_max``.
    :raises ValueError: If ``orthogonality_penalty`` is negative.
    """

    def __init__(
            self,
            units: int,
            scale_min: float = 0.5,
            scale_max: float = 2.0,
            orthogonality_penalty: float = 1e-4,
            rotation_initializer: Union[str, keras.initializers.Initializer] = 'identity',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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

        self.units = units
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.orthogonality_penalty = orthogonality_penalty
        self.rotation_initializer = keras.initializers.get(rotation_initializer)

        # Every shape below depends on the input width, so nothing exists yet.
        self.static_simplex = None
        self.rotation_kernel = None
        self.global_scale = None
        self._input_dim = None

    def _create_simplex_matrix(
            self,
            input_dim: int,
            output_dim: int
    ) -> np.ndarray:
        """
        Generate a centred, normalised Simplex weight matrix.

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param output_dim: Number of Simplex projections.
        :type output_dim: int
        :return: Weight matrix ``(input_dim, output_dim)`` as float32, columns
            drawn from the ``input_dim + 1`` simplex vertices.
        :rtype: np.ndarray
        """

        dimensions = input_dim
        matrix = np.identity(dimensions, dtype=np.float32)

        # This value places the extra vertex equidistant from the identity rows.
        last_point = np.ones((1, dimensions), dtype=np.float32) * \
                     ((1.0 + np.sqrt(dimensions + 1.0)) / dimensions)

        matrix = np.vstack([matrix, last_point])

        # Centering makes the vertices sum to zero, which sets coherence to -1/N.
        mean_m = np.mean(matrix, axis=0)
        matrix = matrix - mean_m

        # Norms are clamped so a degenerate vertex cannot divide by zero.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        matrix = matrix / norms

        # Vertices become columns so that x @ W projects onto them.
        W = matrix.T

        # A width other than a multiple of N+1 repeats or discards vertices.
        current_cols = W.shape[1]

        if output_dim > current_cols:
            tile_factor = int(np.ceil(output_dim / current_cols))
            W = np.tile(W, (1, tile_factor))

        W = W[:, :output_dim]

        return W.astype(np.float32)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create Simplex, rotation kernel, and scale weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last dimension of ``input_shape`` is ``None``."""
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        self._input_dim = input_dim

        # Registered as a weight, not a constant, so it saves and loads with the model.
        simplex_weights = self._create_simplex_matrix(input_dim, self.units)
        self.static_simplex = self.add_weight(
            name='static_simplex',
            shape=(input_dim, self.units),
            initializer=keras.initializers.Constant(simplex_weights),
            trainable=False,
            dtype=self.dtype,
        )

        self.rotation_kernel = self.add_weight(
            name='rotation_kernel',
            shape=(input_dim, input_dim),
            initializer=self.rotation_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        # The constraint runs after updates, so 1.0 may start outside the range.
        self.global_scale = self.add_weight(
            name='global_scale',
            shape=(1,),
            initializer=keras.initializers.Constant(1.0),
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
        """
        Forward pass: rotate, project onto Simplex, and scale.

        :param inputs: Input tensor ``(batch, ..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag. Accepted for API symmetry; the
            orthogonality loss is added regardless of its value.
        :type training: Optional[bool]
        :return: Output tensor ``(batch, ..., units)``.
        :rtype: keras.KerasTensor
        """

        # The loss is a mean of squared entries, so it does not scale with input_dim.
        r_t_r = keras.ops.matmul(
            keras.ops.transpose(self.rotation_kernel),
            self.rotation_kernel
        )
        identity = keras.ops.eye(self._input_dim, dtype=self.dtype)
        ortho_loss = keras.ops.mean(keras.ops.square(r_t_r - identity))
        self.add_loss(self.orthogonality_penalty * ortho_loss)

        rotated_inputs = keras.ops.matmul(inputs, self.rotation_kernel)

        outputs = keras.ops.matmul(rotated_inputs, self.static_simplex)

        outputs = outputs * self.global_scale

        return outputs

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape from input shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension replaced by ``units``.
        :rtype: Tuple[Optional[int], ...]
        """

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """

        config = super().get_config()
        config.update({
            'units': self.units,
            'scale_min': self.scale_min,
            'scale_max': self.scale_max,
            'orthogonality_penalty': self.orthogonality_penalty,
            'rotation_initializer': keras.initializers.serialize(self.rotation_initializer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RigidSimplexLayer':
        """
        Create a layer instance from a configuration dictionary.

        :param config: Configuration from ``get_config()``.
        :type config: Dict[str, Any]
        :return: New ``RigidSimplexLayer`` instance.
        :rtype: RigidSimplexLayer
        """

        if 'rotation_initializer' in config:
            config['rotation_initializer'] = keras.initializers.deserialize(
                config['rotation_initializer']
            )
        return cls(**config)

# ---------------------------------------------------------------------
