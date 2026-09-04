"""
Orthogonal butterfly layer, built by the ``OrthogonalButterfly`` class.

A dense orthogonal ``d x d`` matrix costs ``O(d^2)`` parameters and needs a
matrix inverse, Cayley map, or soft penalty to stay orthogonal during
training. This layer instead composes ``log2(d)`` stages of ``d/2``
independent 2x2 Givens rotations, in the Cooley-Tukey / FFT butterfly access
pattern also used by the polar weight reparameterization in
``norms/polar_weight_norm.py``. Every rotation is orthogonal and acts on a
disjoint coordinate pair, so the composed map is exactly orthogonal for any
angle values, at ``O(d log d)`` cost. The transform is exactly invertible
(``W^-1 = W^T``, via ``call(x, inverse=True)`` or the ``inverse()`` alias)
and its log-det-Jacobian is exactly zero, which makes it a cheap normalizing-flow
step. The feature dimension must be a power of two; a non-power-of-two
dimension raises ``ValueError`` rather than being padded. With the default
``angle_initializer='zeros'`` the layer starts as the identity map.

References:
    - Butterfly / Givens parameterizations of orthogonal matrices via the
      Cooley-Tukey factorization (cf. butterfly / Kaleidoscope matrices,
      Dao et al.).
    - The recursive coordinate-pairing tree of PolarQuant
      (arXiv:2502.02617); see norms/polar_weight_norm.py.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tensors import is_power_of_two


def _butterfly_apply(
    x: keras.KerasTensor,
    angles: keras.KerasTensor,
    d: int,
    num_blocks: int,
    levels: int,
    inverse: bool = False,
) -> keras.KerasTensor:
    """Apply the butterfly orthogonal transform to ``x`` of shape ``(N, d)``.

    :param x: ``(N, d)`` input, ``d`` a power of two.
    :param angles: ``(num_blocks, levels, d/2)`` rotation angles.
    :param d: Feature dimension (static).
    :param num_blocks: Number of stacked butterfly blocks (static).
    :param levels: ``log2(d)`` (static).
    :param inverse: If True, apply the inverse transform ``W^{-1} = W^T`` by
        reversing the block/stage order and transposing each 2x2 rotation
        (``R(theta)^{-1} = R(-theta)``).
    :return: ``(N, d)`` transformed tensor with ``||output|| == ||input||``
        per row.
    """
    block_iter = range(num_blocks - 1, -1, -1) if inverse else range(num_blocks)
    for block in block_iter:
        stage_iter = range(levels - 1, -1, -1) if inverse else range(levels)
        for s in stage_iter:
            stride = 1 << s
            g = d // (2 * stride)
            # Partners are `stride` apart.
            xr = ops.reshape(x, (-1, g, 2, stride))
            a = xr[:, :, 0, :]
            b = xr[:, :, 1, :]
            theta = ops.reshape(angles[block, s, :], (g, stride))
            cos_t = ops.cos(theta)
            sin_t = ops.sin(theta)
            if inverse:
                # R(-theta): transpose of the forward 2x2 rotation.
                a_rot = a * cos_t + b * sin_t
                b_rot = -a * sin_t + b * cos_t
            else:
                a_rot = a * cos_t - b * sin_t
                b_rot = a * sin_t + b * cos_t
            xr = ops.stack([a_rot, b_rot], axis=2)
            x = ops.reshape(xr, (-1, d))
    return x


@register_dl_technique("dl_techniques.layers.structured_linear.orthogonal_butterfly")
class OrthogonalButterfly(keras.layers.Layer):
    """Structured, exactly-orthogonal ``d x d`` linear layer (butterfly Givens).

    One block spans only the FFT-structured subset of ``SO(d)``; passing
    ``num_blocks > 1`` composes several blocks to recover expressivity.

    Architecture:

    .. code-block:: text

        Input [batch, ..., d]          d = 2^L
            │
            ▼
        ┌──────────────────────────────────────┐
        │ for block in 0..num_blocks:           │
        │   for stage s in 0..L: stride = 2^s   │
        │     pair partners `stride` apart      │
        │     [a;b] -> [a cos0 - b sin0 ;        │
        │               a sin0 + b cos0]        │
        │     (d/2 disjoint 2x2 rotations)      │
        └──────────────────┬─────────────────────┘
                            ▼
                    (+ bias, optional)
                            │
                            ▼
        Output [batch, ..., d]

    :param num_blocks: Number of stacked butterfly blocks. Defaults to 1.
    :type num_blocks: int
    :param use_bias: Add a bias after the rotation. Breaks pure linearity but
        not the rotation's orthogonality. Defaults to False.
    :type use_bias: bool
    :param angle_initializer: Initializer for the rotation angles. Defaults
        to ``'zeros'`` (identity transform).
    :param angle_regularizer: Optional regularizer on the angles.
    :param bias_initializer: Initializer for the bias. Defaults to ``'zeros'``.
    :param bias_regularizer: Optional regularizer on the bias.
    :param kwargs: Passed to ``keras.layers.Layer``.

    Input shape:
        N-D tensor ``(batch, ..., d)`` with ``d`` a power of two.

    Output shape:
        Same as input: ``(batch, ..., d)``.

    :raises ValueError: If the last input dimension is not a power of two.
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
        if not is_power_of_two(d):
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
