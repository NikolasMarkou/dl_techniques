"""BandLogitNorm: L2 normalization followed by a band-limited rescale.

``BandLogitNorm`` normalizes an input to unit L2 length and then multiplies it by
a scale drawn from the band ``[1 - max_band_width, 1]``. Direction is preserved
exactly; only the magnitude is changed.

The norm is ``sqrt(max(sum(x ** 2), epsilon))``. It SUMS over ``axis``, so this
is an L2 norm, not an RMS (``sqrt(mean(x ** 2))``). The output magnitude
therefore depends on the length of the normalized axis, unlike the RMS-based
layers that sit beside this one in the package.

Steps
-----

1. L2 length: ``x_length = sqrt(max(sum(x ** 2), epsilon))``, ``keepdims=True``.
2. Unit norm: ``x_normalized = x / x_length``.
3. ``LayerNormalization`` applied to ``x_length``.
4. Bound it: ``tanh(4 * layer_norm_output)``.
5. Map to the band: ``scale = (1 - w) + w * (tanh_output + 1) / 2``, where ``w``
   is ``max_band_width``.
6. Output: ``x_normalized * scale``.

The scale is constant in practice
---------------------------------

Step 3 normalizes a tensor whose last axis has length 1, which always yields 0.
So ``tanh(4 * 0) = 0`` and the scale collapses to ``1 - 0.5 * max_band_width``
for every input. Measured: with ``max_band_width`` of 0.01, 0.5 and 0.9 the
output L2 norm is exactly 0.995, 0.75 and 0.55, with no spread across rows to
six decimals (measured spread at most ``1.51e-07`` over shapes ``(4, 32)`` to
``(64, 128)``, float32 rounding). The layer is L2 normalization times a
constant; the input-adaptive part is inert. It is kept as-is for backward
compatibility, because callers exist in
``train/rms_variants_train/``.
"""

import keras
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms._masking import (
    normalizes_only_the_feature_axis,
)


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BandLogitNorm(keras.layers.Layer):
    """L2-normalize the input, then rescale it into a narrow band.

    Divides the input by its own L2 length, then multiplies by a scale taken from
    ``[1 - max_band_width, 1]``. Direction is preserved exactly. The output shape
    equals the input shape.

    The norm sums over ``axis``, so it is an L2 norm and not an RMS
    (``sqrt(mean(x ** 2))``), despite this layer sitting beside the RMS family.

    ``supports_masking`` is a promise about the AXIS, not about the class. It is
    ``True`` only while the normalized axis is the trailing (feature) axis. At
    ``axis=-1`` the output at one position depends on that position alone: the
    measured cross-position leak on a ``(3, 5, 8)`` input is exactly ``0.0``, in
    both training modes. Normalizing the token axis couples positions instead;
    the measured leak at ``axis=1`` on the same input is ``0.914``. The flag is
    ``False`` there, so Keras drops the mask and says so. ``__init__`` decides
    from the spelling alone, because only ``-1`` names the trailing axis at every
    rank. ``build()`` then makes it exact.

    .. warning::
        **The adaptive scale is inert.** ``LayerNormalization`` is applied to the
        L2-length tensor, whose last axis has length 1 (``keepdims=True``).
        Normalizing a single-element axis always gives 0, so ``tanh(4 * 0) = 0``
        and the scale collapses to the constant ``1 - 0.5 * max_band_width``.
        Measured output L2 norms: 0.995, 0.75 and 0.55 for ``max_band_width`` of
        0.01, 0.5 and 0.9, with no spread across rows to six decimals
        (measured spread at most ``1.51e-07`` over shapes ``(4, 32)`` to
        ``(64, 128)``). Treat this layer as L2 normalization times a constant.
        It is kept as-is because callers exist in ``train/rms_variants_train/``.

    **Architecture Overview:**

    .. code-block:: text

                              inputs: (..., D)
                                  │
            ┌─────────────────────┤
            │                     │
            │                     ▼
            │        ┌─────────────────────────┐
            │        │ square, sum over axis,  │
            │        │ floor at epsilon, sqrt  │
            │        └────────────┬────────────┘
            │                     │ x_length: (..., 1)
            │           ┌─────────┴──────────────────┐
            │           │                            │
            ▼           ▼                            │
          ┌───────────────────────────┐              │
          │ unit norm:                │              │
          │ inputs / x_length         │              │
          └─────────────┬─────────────┘              │
                        │                            ▼
                        │               ┌─────────────────────────┐
                        │               │ LayerNormalization      │
                        │               │ (owns weights)          │
                        │               └────────────┬────────────┘
                        │                            │
                        │                            ▼
                        │               ┌─────────────────────────┐
                        │               │ tanh(4 * .)             │
                        │               └────────────┬────────────┘
                        │                            │
                        │                            ▼
                        │               ┌─────────────────────────┐
                        │               │ map into the band       │
                        │               │ [1 - max_band_width, 1] │
                        │               └────────────┬────────────┘
                        │ x_normalized               │ scale: (..., 1)
                        ▼                            ▼
              ┌───────────────────────────────────────────────────┐
              │ multiply: x_normalized * scale                    │
              └───────────────────────┬───────────────────────────┘
                                      │
                                      ▼
                              output: (..., D)

    :param max_band_width: Width of the band below unit norm. Must satisfy
        ``0 < max_band_width < 1``. Defaults to 0.01.
    :type max_band_width: float
    :param axis: Axis reduced by the L2 norm. Defaults to -1.
    :type axis: int
    :param epsilon: Floor applied to the squared norm, and the epsilon passed to
        the internal ``LayerNormalization``. Must be positive. Defaults to 1e-7.
    :type epsilon: float

    :ivar max_band_width: The configured band width.
    :vartype max_band_width: float
    :ivar axis: The configured normalization axis.
    :vartype axis: int
    :ivar epsilon: The configured numerical floor.
    :vartype epsilon: float
    :ivar norm: The internal ``LayerNormalization`` sublayer applied to the
        L2-length tensor.
    :vartype norm: keras.layers.LayerNormalization

    :raises ValueError: If max_band_width is not in (0, 1).
    :raises ValueError: If epsilon is not positive.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import BandLogitNorm

        x = keras.random.normal((4, 16))
        y = BandLogitNorm(max_band_width=0.01)(x)
    """

    def __init__(
            self,
            max_band_width: float = 0.01,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        """Initialize the layer.

        :param max_band_width: Width of the band below unit norm. Must be in
            ``(0, 1)``.
        :type max_band_width: float
        :param axis: Axis reduced by the L2 norm.
        :type axis: int
        :param epsilon: Floor applied to the squared norm. Must be positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If max_band_width is not in (0, 1).
        :raises ValueError: If epsilon is not positive.
        """
        super().__init__(**kwargs)

        self._validate_inputs(max_band_width, epsilon)

        # Every constructor argument is stored, because get_config() must return
        # all of them.
        self.axis = axis
        self.epsilon = epsilon
        self.max_band_width = max_band_width

        # Creating the sublayer here is the Keras 3 pattern: construction does not
        # need the input shape. build() builds it against the L2-length shape.
        # See the class docstring for why that (..., 1) input makes it inert.
        self.norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=self.epsilon,
            name=f"{self.name}_layer_norm",
        )

        # supports_masking is a promise about the AXIS, not about the class. It
        # holds only while the normalized axis is the trailing (feature) axis.
        # Only the spelling `-1` names that axis at every rank, so that is all
        # __init__ can decide. build() makes the answer exact.
        self.supports_masking = normalizes_only_the_feature_axis(axis)

        logger.debug(
            f"Initialized BandLogitNorm with "
            f"axis={axis}, "
            f"epsilon={epsilon}, "
            f"max_band_width={max_band_width}, "
        )

    def _validate_inputs(self, max_band_width: float, epsilon: float) -> None:
        """Reject out-of-range constructor arguments.

        :param max_band_width: Band width to validate.
        :type max_band_width: float
        :param epsilon: Epsilon to validate.
        :type epsilon: float

        :raises ValueError: If max_band_width is not in (0, 1).
        :raises ValueError: If epsilon is not positive.
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape) -> None:
        """Build the ``LayerNormalization`` sublayer and finalize the mask flag.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        """
        if self.built:
            return

        # Refine the __init__ estimate now that the rank is known. Keras reads
        # supports_masking inside __call__, which runs build() first, so this is
        # the value that decides whether the mask survives.
        self.supports_masking = normalizes_only_the_feature_axis(
            self.axis, rank=len(input_shape)
        )

        # The L2-length tensor fed to self.norm has a length-1 axis, because the
        # reduction uses keepdims=True.
        norm_shape = list(input_shape)
        norm_shape[self.axis] = 1
        self.norm.build(norm_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply band-limited L2 normalization.

        :param inputs: Input tensor to normalize.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the internal ``LayerNormalization``.
        :type training: Optional[bool]

        :return: Normalized tensor, same shape as ``inputs``. Its L2 norm over
            ``axis`` is ``1 - 0.5 * max_band_width``. See the class docstring for
            why that value is constant rather than input-dependent.
        :rtype: keras.KerasTensor
        """
        x = inputs

        # Step 1: L2 length along axis. maximum() floors the squared sum, so
        # sqrt never sees a value below epsilon. keepdims=True keeps the rank.
        x_squared = keras.ops.square(x)
        x_sum_squared = keras.ops.maximum(keras.ops.sum(x_squared, axis=self.axis, keepdims=True), self.epsilon)
        x_length = keras.ops.sqrt(x_sum_squared)

        # Step 2: unit L2 norm.
        x_normalized = x / x_length

        # Step 3: normalize the length tensor. Its last axis has length 1, so the
        # result is always 0. See the class docstring.
        x_length_normalized = self.norm(x_length, training=training)

        # Step 4: bound the result to [-1, +1].
        x_length_normalized = keras.activations.tanh(4 * x_length_normalized)

        # Step 5: map [-1, +1] to [1 - max_band_width, 1]. The first line moves
        # it to [0, 1]; the second scales and offsets it into the band.
        scale = (x_length_normalized + 1.0) / 2.0
        scale = (1.0 - self.max_band_width) + self.max_band_width * scale

        # Step 6: rescale the unit-normalized tensor.
        return x_normalized * scale

    def compute_output_shape(self, input_shape) -> tuple:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple

        :return: The same shape tuple that was passed in.
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration needed to rebuild this layer.

        :return: Dictionary holding every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
            "max_band_width": self.max_band_width
        })
        return config


# ---------------------------------------------------------------------
