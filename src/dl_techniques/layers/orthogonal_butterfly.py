"""Learnable exactly-orthogonal layer via a butterfly of 2x2 Givens rotations.

This is the operator-valued sibling of the polar weight reparameterization
(`layers/norms/polar_weight_norm.py`, after PolarQuant, Han et al. 2025). The
polar transform organizes a vector's angles into a ``log2(d)``-level binary tree
of coordinate pairs; here the *same hierarchical pairing* is used to build a
structured **orthogonal** ``d x d`` linear map: ``log2(d)`` stages, each applying
``d/2`` independent 2x2 rotations to coordinate pairs separated by a doubling
stride (the Cooley-Tukey / FFT butterfly access pattern).

The map is exactly orthogonal by construction (each 2x2 rotation is orthogonal
and the rotations in a stage act on disjoint coordinate pairs), needs no matrix
inverse, no Cayley/expm, and no soft penalty. Cost is ``O(d log d)`` versus
``O(d^2)`` for a dense orthogonal layer, with ``(d/2) * log2(d)`` angle
parameters per block. With ``angle_initializer='zeros'`` the layer is the
identity, which makes it a stable drop-in / residual building block.

A single butterfly block spans only the FFT-structured subset of ``SO(d)``;
``num_blocks > 1`` composes several blocks to recover expressivity.

Note: unlike `PolarWeightNorm`, this layer requires the feature dimension to be
a power of two and does NOT pad -- padding cannot preserve orthogonality on the
original subspace.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.utils.logger import logger


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _butterfly_apply(
    x: keras.KerasTensor,
    angles: keras.KerasTensor,
    d: int,
    num_blocks: int,
    levels: int,
    inverse: bool = False,
) -> keras.KerasTensor:
    """Apply the butterfly orthogonal transform to ``x`` of shape ``(N, d)``.

    Args:
        x: ``(N, d)`` input, ``d`` a power of two.
        angles: ``(num_blocks, levels, d/2)`` rotation angles.
        d: feature dimension (static).
        num_blocks: number of stacked butterfly blocks (static).
        levels: ``log2(d)`` (static).
        inverse: if True apply the inverse transform ``W^{-1} = W^T``. Since
            ``W = R_last ... R_first`` (composition of orthogonal stages), the
            inverse reverses the block/stage order and transposes each 2x2
            rotation (``R(theta)^{-1} = R(-theta)``).

    Returns:
        ``(N, d)`` transformed tensor with ``||output|| == ||input||`` per row.
        ``_butterfly_apply(_butterfly_apply(x, ...), ..., inverse=True) == x``.
    """
    block_iter = range(num_blocks - 1, -1, -1) if inverse else range(num_blocks)
    for block in block_iter:
        stage_iter = range(levels - 1, -1, -1) if inverse else range(levels)
        for s in stage_iter:
            stride = 1 << s
            g = d // (2 * stride)
            xr = ops.reshape(x, (-1, g, 2, stride))  # partners are `stride` apart
            a = xr[:, :, 0, :]  # (N, g, stride)
            b = xr[:, :, 1, :]
            theta = ops.reshape(angles[block, s, :], (g, stride))  # (g, stride)
            cos_t = ops.cos(theta)
            sin_t = ops.sin(theta)
            if inverse:  # R(-theta) = transpose of the forward 2x2 rotation
                a_rot = a * cos_t + b * sin_t
                b_rot = -a * sin_t + b * cos_t
            else:
                a_rot = a * cos_t - b * sin_t
                b_rot = a * sin_t + b * cos_t
            xr = ops.stack([a_rot, b_rot], axis=2)  # (N, g, 2, stride)
            x = ops.reshape(xr, (-1, d))
    return x


@keras.saving.register_keras_serializable()
class OrthogonalButterfly(keras.layers.Layer):
    """Structured, exactly-orthogonal ``d x d`` linear layer (butterfly Givens).

    **Intent**: Provide a cheap (``O(d log d)``), exactly norm-preserving linear
    transform whose only parameters are rotation angles -- useful for normalizing
    flows (zero log-det), orthogonal RNN recurrence (no exploding/vanishing
    norms), and invertible/lossless blocks. It is the operator form of the polar
    transform's hierarchical angle tree.

    **Architecture**:
    ```
    Input(batch, ..., d)        d = 2^L
        |
        for block in [0..num_blocks):
          for stage s in [0..L):  stride = 2^s
            reshape -> (.., d/(2*stride), 2, stride)   # pair partners stride apart
            [a; b] -> [a cosθ - b sinθ ; a sinθ + b cosθ]   # d/2 2x2 rotations
        |
        (+ optional bias)
    Output(batch, ..., d)
    ```

    Exactly orthogonal for any angle values; ``angle_initializer='zeros'`` gives
    the identity.

    Args:
        num_blocks: Number of stacked butterfly blocks. More blocks => more
            expressive (one block spans only the FFT-structured subset of SO(d)).
            Defaults to 1.
        use_bias: Add a bias after the rotation (breaks pure linearity but not
            the rotation's orthogonality). Defaults to False.
        angle_initializer: Initializer for the rotation angles. Defaults to
            ``'zeros'`` (identity transform).
        angle_regularizer: Optional regularizer on the angles.
        bias_initializer: Initializer for the bias. Defaults to ``'zeros'``.
        bias_regularizer: Optional regularizer on the bias.
        **kwargs: Passed to ``keras.layers.Layer``.

    Input shape:
        N-D tensor ``(batch, ..., d)`` with ``d`` a power of two.

    Output shape:
        Same as input: ``(batch, ..., d)``.

    Raises:
        ValueError: if the last input dimension is not a power of two.
    """

    def __init__(
        self,
        num_blocks: int = 1,
        use_bias: bool = False,
        angle_initializer: Union[str, Any] = "zeros",
        angle_regularizer: Optional[Union[str, Any]] = None,
        bias_initializer: Union[str, Any] = "zeros",
        bias_regularizer: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._validate_inputs(num_blocks)

        self.num_blocks = int(num_blocks)
        self.use_bias = use_bias
        self.angle_initializer = keras.initializers.get(angle_initializer)
        self.angle_regularizer = keras.regularizers.get(angle_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.angles = None
        self.bias = None
        self._dim: Optional[int] = None
        self._levels: Optional[int] = None

        logger.debug(f"Initialized OrthogonalButterfly(num_blocks={self.num_blocks})")

    @staticmethod
    def _validate_inputs(num_blocks: int) -> None:
        if not isinstance(num_blocks, int) or num_blocks <= 0:
            raise ValueError(f"num_blocks must be a positive integer, got {num_blocks}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        d = input_shape[-1]
        if d is None:
            raise ValueError("The last dimension of the input must be defined.")
        d = int(d)
        if not _is_power_of_two(d):
            raise ValueError(
                f"OrthogonalButterfly requires a power-of-two feature dim, got {d}."
            )
        levels = d.bit_length() - 1
        self._dim = d
        self._levels = levels

        if levels > 0:
            self.angles = self.add_weight(
                name="angles",
                shape=(self.num_blocks, levels, d // 2),
                initializer=self.angle_initializer,
                trainable=True,
                regularizer=self.angle_regularizer,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(d,),
                initializer=self.bias_initializer,
                trainable=True,
                regularizer=self.bias_regularizer,
            )

        logger.debug(f"Built OrthogonalButterfly: dim={d}, levels={levels}")
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        inverse: bool = False,
    ) -> keras.KerasTensor:
        """Apply the transform (``inverse=False``) or its exact inverse.

        With a bias, the forward map is ``y = W x + b`` and the inverse is
        ``x = W^T (y - b)`` (bias subtracted before the inverse rotation).
        """
        inputs_fp32 = ops.cast(inputs, "float32")
        orig_shape = ops.shape(inputs_fp32)
        x = ops.reshape(inputs_fp32, (-1, self._dim))  # flatten leading dims
        if inverse and self.use_bias:
            x = ops.subtract(x, ops.cast(self.bias, "float32"))
        if self._levels > 0:
            x = _butterfly_apply(
                x,
                ops.cast(self.angles, "float32"),
                self._dim,
                self.num_blocks,
                self._levels,
                inverse=inverse,
            )
        if self.use_bias and not inverse:
            x = ops.add(x, ops.cast(self.bias, "float32"))
        x = ops.reshape(x, orig_shape)
        return ops.cast(x, inputs.dtype)

    def inverse(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Convenience alias for ``call(inputs, inverse=True)``."""
        return self.call(inputs, inverse=True)

    def log_det_jacobian(
        self,
        inputs: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Log-determinant of the Jacobian: exactly ``0`` (orthogonal map).

        Returns a tensor of zeros with shape ``inputs.shape[:-1]`` (one scalar
        per transformed vector), the standard contribution of an orthogonal
        flow step to a change-of-variables log-likelihood.
        """
        return ops.zeros(ops.shape(inputs)[:-1], dtype=inputs.dtype)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_blocks": self.num_blocks,
            "use_bias": self.use_bias,
            "angle_initializer": keras.initializers.serialize(self.angle_initializer),
            "angle_regularizer": keras.regularizers.serialize(self.angle_regularizer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
