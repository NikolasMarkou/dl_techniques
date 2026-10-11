"""RigidSimplexLayer, a projection onto a frozen Equiangular Tight Frame.

RigidSimplexLayer is a Keras layer that maps `[..., input_dim]` to
`[..., units]`. Its projection matrix is a fixed frame of unit vectors that
never trains; the learnable parts are a square rotation `R` and a scalar `s`,
giving

    output = s * (x @ R) @ Frame

with `R` pulled toward orthogonality by a penalty on the mean squared
entries of `R^T R - I`. The scale starts at 1.0 and is clipped into
`[scale_min, scale_max]` after the first update. The last input dimension
must be static.

**Frame width — the `frame` argument.** A single regular simplex holds
exactly `input_dim + 1` unit vectors in `input_dim` dimensions, every pair at
inner product `-1/input_dim`. Two constructions fill a wider `units`:

- `frame='crystal'` (the default) packs several *rotated* simplexes onto the
  sphere with a greedy minimax-angular-separation heuristic, so every one of
  the `units` columns is a distinct direction.
- `frame='tiled'` repeats one simplex's vertices. It is exact at or below
  `input_dim + 1`, and **degenerate beyond it**: measured at
  `input_dim=16, units=100` it yields 17 distinct columns of 100, with 490
  column pairs at inner product exactly `1.0` — 83% of the outputs are exact
  duplicates of another. The mode is retained so configurations written
  against it keep producing their original values; it is not a codebook.

Migration note: `frame` was added after the layer shipped. `get_config()`
always emits it, but a config **lacking** the key is reconstructed as
`frame='tiled'`, because that is what such a config was built with. Old
`.keras` archives are unaffected either way — the frame is a saved weight and
is restored bit-exactly, not rebuilt.

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
from dl_techniques.layers.structured_linear.crystal_frames import (
    crystal_frame,
    tiled_simplex_frame
)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tensors import canonical_binary_input_shape

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
        otherwise                see the `frame` argument below

    Input shape:
        N-D tensor ``(..., input_dim)``. The last dimension must be static.

    Output shape:
        Same rank as the input, with the last dimension set to ``units``.

    Note:
        ``global_scale`` is initialized at 1.0 and the range constraint is
        applied after an optimizer update, so a range that excludes 1.0
        leaves the first forward pass outside the bounds.

    :param units: Dimensionality of the output space (frame projections).
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
    :param frame: How the frozen frame is built once ``units`` exceeds the
        ``input_dim + 1`` a single simplex can supply. ``'crystal'`` packs
        rotated simplexes for ``units`` distinct directions; ``'tiled'``
        repeats one simplex's vertices and is exact only at or below
        ``input_dim + 1``. Defaults to ``'crystal'``.
    :type frame: str
    :param frame_seed: Seed for the crystal frame's rotation sampling. ``None``
        (the default) draws a fresh frame on every build; pass an integer for
        a reproducible one. Ignored by ``frame='tiled'``, which is
        deterministic.
    :type frame_seed: Optional[int]
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
    :raises ValueError: If ``frame`` is not one of ``'crystal'`` or ``'tiled'``.
    """

    #: The two accepted values of the ``frame`` argument.
    FRAME_MODES = ('crystal', 'tiled')

    def __init__(
            self,
            units: int,
            scale_min: float = 0.5,
            scale_max: float = 2.0,
            orthogonality_penalty: float = 1e-4,
            rotation_initializer: Union[str, keras.initializers.Initializer] = 'identity',
            frame: str = 'crystal',
            frame_seed: Optional[int] = None,
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
        if frame not in self.FRAME_MODES:
            raise ValueError(
                f"frame must be one of {list(self.FRAME_MODES)}, got {frame!r}"
            )

        self.units = units
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.orthogonality_penalty = orthogonality_penalty
        self.rotation_initializer = keras.initializers.get(rotation_initializer)
        self.frame = frame
        self.frame_seed = frame_seed

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
        Generate the frozen frame, dispatching on the ``frame`` mode.

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param output_dim: Number of frame projections.
        :type output_dim: int
        :return: Weight matrix ``(input_dim, output_dim)`` as float32 with
            unit-norm columns.
        :rtype: np.ndarray
        """

        if self.frame == 'tiled':
            return tiled_simplex_frame(input_dim, output_dim)

        return crystal_frame(
            dimensions=input_dim,
            width=output_dim,
            seed=self.frame_seed
        )

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
        Forward pass: rotate, project onto the frozen frame, and scale.

        :param inputs: Input tensor ``(batch, ..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag. The orthogonality penalty is added
            unless this is exactly ``False``, so an inference forward leaves
            ``self.losses`` empty; an eager call or a ``fit`` step passes
            ``None`` and still receives it.
        :type training: Optional[bool]
        :return: Output tensor ``(batch, ..., units)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the last dimension of ``inputs`` disagrees with
            the width this layer was built for.
        """

        # The build-time contract is re-asserted here: build() only ever sees
        # the shape that arrived first, so a contract checked there alone is
        # checked exactly once.
        if inputs.shape[-1] != self._input_dim:
            raise ValueError(
                f"RigidSimplexLayer was built for input_dim={self._input_dim} "
                f"but received input_dim={inputs.shape[-1]}"
            )

        if training is not False and self.orthogonality_penalty > 0.0:
            # The loss is a mean of squared entries, so it does not scale with
            # input_dim.
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

        :param input_shape: Shape of the input. A shape that arrives as a list
            is canonicalized first: ``list(input_shape)`` over ``[[None, 8]]``
            iterates the OUTER list, so the last-element rewrite would replace
            the batch axis and return ``(units,)`` instead of ``(None, units)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension replaced by ``units``.
        :rtype: Tuple[Optional[int], ...]
        """

        output_shape = list(canonical_binary_input_shape(input_shape))
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
            'frame': self.frame,
            'frame_seed': self.frame_seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RigidSimplexLayer':
        """
        Create a layer instance from a configuration dictionary.

        ``frame`` postdates this layer, so a config written before it existed
        carries no ``frame`` key and was necessarily built by tiling. Such a
        config is reconstructed as ``frame='tiled'`` -- the values it was built
        to produce -- rather than silently picking up the new default. The
        caller's dict is copied, not edited in place.

        :param config: Configuration from ``get_config()``.
        :type config: Dict[str, Any]
        :return: New ``RigidSimplexLayer`` instance.
        :rtype: RigidSimplexLayer
        """

        config = dict(config)

        if 'frame' not in config:
            logger.warning(
                "RigidSimplexLayer config predates the 'frame' argument and is "
                "reconstructed with frame='tiled', the construction it was built "
                "with. That frame repeats one simplex's vertices, so a width "
                "beyond input_dim + 1 yields duplicated columns at inner product "
                "1.0. Pass frame='crystal' explicitly to opt into packed frames."
            )
            config['frame'] = 'tiled'

        # rotation_initializer needs no handling here: __init__ runs it through
        # keras.initializers.get, which accepts both a string and a serialized
        # config dict. Deserializing in from_config would be both redundant and
        # a second site to keep in step with __init__.
        return cls(**config)

# ---------------------------------------------------------------------
