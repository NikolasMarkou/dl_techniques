"""DynamicTanh (DyT): a normalization-free replacement for LayerNormalization.

``DynamicTanh`` computes ``weight * tanh(alpha * x) + bias``, where ``alpha`` is
a single learnable scalar and ``weight`` and ``bias`` are per-feature learnable
vectors. There is no mean, no variance and no reduction of any kind.

The layer comes from "Transformers without Normalization" (Zhu et al., CVPR
2025), https://arxiv.org/abs/2503.10622. The paper's claim is that a saturating
scalar nonlinearity reproduces what LayerNormalization does for a Transformer
without computing statistics.

Because the transform is elementwise, one position's value never reaches another
position's output. Measured cross-token leak on ``(3, 5, 8)`` and ``(4, 5, 8)``
inputs, at ``axis=-1``, ``axis=1`` and ``axis=2``: exactly ``0.0`` in every case.
That is why ``supports_masking`` is set ``True`` unconditionally here, while the
RMS-family layers in this package have to decide it from the axis.
"""

import keras
from typing import Optional, Union, Dict, Any, List, Tuple
from keras import ops, constraints, initializers, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DynamicTanh(keras.layers.Layer):
    """Apply ``weight * tanh(alpha * x) + bias`` with a learnable scalar ``alpha``.

    A drop-in replacement for ``LayerNormalization`` in Transformers. ``alpha``
    is one scalar weight for the whole layer; ``weight`` and ``bias`` have the
    shape of the axes named by ``axis``. The output has the same shape as the
    input.

    The transform is elementwise, so the layer computes no statistics and mixes
    no positions. Measured cross-token leak on ``(3, 5, 8)`` and ``(4, 5, 8)``
    inputs, at ``axis=-1``, ``axis=1`` and ``axis=2``: exactly ``0.0`` in every
    case. ``supports_masking`` is therefore ``True`` for every axis spelling.

    ``axis`` does not choose a reduction, because there is no reduction. It
    chooses which dimensions get their own ``weight`` and ``bias`` entry.
    Measured on a ``(2, 4, 8)`` input: ``axis=-1`` gives ``weight.shape=(8,)``
    broadcast as ``(1, 1, 8)``, and ``axis=1`` gives ``(4,)`` broadcast as
    ``(1, 4, 1)``.

    **Architecture Overview:**

    .. code-block:: text

                      input: x  (batch, ..., D)
                                  │
                                  ▼
          ┌───────────────────────────────────────────────────┐
          │ scaled = alpha * x                                │
          │ alpha is one scalar weight                        │
          └───────────────────────┬───────────────────────────┘
                                  │
                                  ▼
          ┌───────────────────────────────────────────────────┐
          │ tanh_outputs = tanh(scaled)                       │
          └───────────────────────┬───────────────────────────┘
                                  │ in (-1, 1)
                                  ▼
          ┌───────────────────────────────────────────────────┐
          │ weight and bias reshaped to                       │
          │ _broadcast_shape, fixed in build()                │
          └───────────────────────┬───────────────────────────┘
                                  │
                                  ▼
          ┌───────────────────────────────────────────────────┐
          │ output = tanh_outputs * weight                    │
          │          + bias                                   │
          └───────────────────────┬───────────────────────────┘
                                  │
                                  ▼
                      output: (batch, ..., D)

    :param axis: Axis or axes that get their own ``weight`` and ``bias`` entry.
        Defaults to -1, the feature axis. A list is accepted; the entries are
        checked against the input rank in ``build()``.
    :type axis: Union[int, List[int]]
    :param alpha_init_value: Initial value of the learnable ``alpha``. Must be
        strictly positive; see the ``:raises:`` note below, which explains why
        this check is checkpoint-visible. The paper suggests 0.6-0.8 for
        attention normalization and 0.1-0.2 for FFN and final decoder
        normalization. Defaults to 0.5.
    :type alpha_init_value: float
    :param kernel_initializer: Initializer for ``weight``. Defaults to
        ``'ones'``.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for ``bias``. Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for ``weight``.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for ``bias``.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kernel_constraint: Optional constraint for ``weight``.
    :type kernel_constraint: Optional[constraints.Constraint]
    :param bias_constraint: Optional constraint for ``bias``.
    :type bias_constraint: Optional[constraints.Constraint]
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar axis: The axis list built in ``__init__``. A scalar ``axis`` becomes a
        one-element list. ``build()`` never mutates it.
    :vartype axis: List[int]
    :ivar alpha_init_value: The configured initial alpha, stored as a float.
    :vartype alpha_init_value: float
    :ivar alpha: The scalar learnable weight, or None until ``build()`` runs.
    :vartype alpha: Optional[keras.Variable]
    :ivar weight: The per-feature scale, or None until ``build()`` runs.
    :vartype weight: Optional[keras.Variable]
    :ivar bias: The per-feature offset, or None until ``build()`` runs.
    :vartype bias: Optional[keras.Variable]

    :raises ValueError: If ``alpha_init_value`` is not an int or a float.
    :raises ValueError: If ``alpha_init_value`` is not strictly positive. **This
        check is checkpoint-visible.** ``get_config()`` writes
        ``alpha_init_value``, so a ``.keras`` model saved by a version that
        accepted a non-positive value no longer deserializes. Measured: a
        functional model holding ``DynamicTanh(alpha_init_value=-0.5)``, saved at
        commit ``a8a042f53``, now fails with a ``TypeError`` from the
        deserializer whose root cause is this ``ValueError``. The break was
        accepted after measuring how narrow it is: a repo-wide grep finds zero
        call sites passing a non-positive value, and only the CONFIG value is
        checked. A checkpoint whose LEARNED ``alpha`` weight is negative still
        loads and reproduces its outputs exactly (measured: restored
        ``alpha = -0.5``, ``max|delta| = 0.0``). See ``decisions.md`` D-014 of
        ``plan-2026-08-25T195813-d5a035ab``.
    :raises ValueError: If an entry of ``axis`` is out of bounds for the input
        rank. Raised in ``build()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import DynamicTanh

        x = keras.random.normal((4, 16, 64))
        y = DynamicTanh(alpha_init_value=0.5)(x)
    """

    def __init__(
        self,
        axis: Union[int, List[int]] = -1,
        alpha_init_value: float = 0.5,
        kernel_initializer: Union[str, initializers.Initializer] = 'ones',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs: Any
    ) -> None:
        """Validate ``alpha_init_value`` and store the configuration.

        No weight is created here. ``alpha``, ``weight`` and ``bias`` need the
        input rank, so they are created in ``build()``.

        :param axis: Axis or axes that get their own ``weight`` and ``bias``.
        :type axis: Union[int, List[int]]
        :param alpha_init_value: Strictly positive initial alpha.
        :type alpha_init_value: float
        :param kernel_initializer: Initializer for ``weight``.
        :type kernel_initializer: Union[str, initializers.Initializer]
        :param bias_initializer: Initializer for ``bias``.
        :type bias_initializer: Union[str, initializers.Initializer]
        :param kernel_regularizer: Optional regularizer for ``weight``.
        :type kernel_regularizer: Optional[regularizers.Regularizer]
        :param bias_regularizer: Optional regularizer for ``bias``.
        :type bias_regularizer: Optional[regularizers.Regularizer]
        :param kernel_constraint: Optional constraint for ``weight``.
        :type kernel_constraint: Optional[constraints.Constraint]
        :param bias_constraint: Optional constraint for ``bias``.
        :type bias_constraint: Optional[constraints.Constraint]
        :param kwargs: Forwarded to ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``alpha_init_value`` is not a number.
        :raises ValueError: If ``alpha_init_value`` is not strictly positive.
        """
        super().__init__(**kwargs)

        # Validate alpha initialization value
        if not isinstance(alpha_init_value, (int, float)):
            raise ValueError(f"alpha_init_value must be a number, got {type(alpha_init_value)}")
        # Sign check mirrored from validate_normalization_config's 'dynamic_tanh'
        # branch in factory.py (same message). A non-positive alpha flips the
        # transform's sign, and alpha == 0 makes the layer the constant-zero map
        # tanh(0 * x); the factory has always refused both.
        #
        # DECISION plan-2026-08-25T195813-d5a035ab/D-014
        # Checkpoint-visible, accepted knowingly: get_config() writes alpha_init_value,
        # so an archive holding a non-positive one fails to LOAD with a TypeError whose
        # root cause is this ValueError (measured at a8a042f53). Do NOT loosen it to
        # construction-only or special-case from_config; only the INIT value is checked,
        # so a trained-negative alpha WEIGHT still loads. See decisions.md D-014 and
        # tests/test_layers/test_norms/test_the_negative_alpha_init_is_checkpoint_visible.py
        if alpha_init_value <= 0:
            raise ValueError("alpha_init_value must be a positive number")

        # Store ALL configuration parameters. self.axis keeps the constructor value
        # verbatim (never mutated by build) so get_config is build-state-independent;
        # the build-normalized (positive) axes live in self._norm_axis.
        self.axis = list(axis) if isinstance(axis, (list, tuple)) else [axis]
        self._norm_axis: Optional[List[int]] = None
        # Static broadcast tuple for weight/bias, derived once in build(). Kept
        # None here so the `if self.built: return` early exit in build() can
        # never leave it undefined.
        self._broadcast_shape: Optional[Tuple[int, ...]] = None
        self.alpha_init_value = float(alpha_init_value)

        # Store serializable initializers/regularizers/constraints
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Initialize weight attributes - created in build()
        self.alpha = None
        self.weight = None
        self.bias = None

        # The transform is elementwise, so every output slot depends only on the
        # same input slot. Measured leak at axis -1, 1 and 2 on a rank-3 input:
        # exactly 0.0. The flag is honest at every axis, so unlike the RMS family
        # this layer does not have to refine it in build().
        self.supports_masking = True

        logger.debug(
            f"Initialized DynamicTanh with "
            f"axis={axis}, "
            f"alpha_init_value={alpha_init_value}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create ``alpha``, ``weight`` and ``bias``, and fix the broadcast shape.

        Resolves every entry of ``axis`` against the input rank, sizes ``weight``
        and ``bias`` from the resolved axes, and precomputes the broadcast tuple
        ``call()`` reshapes them to.

        :param input_shape: Shape tuple of the input tensor. The batch dimension
            may be None.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If an entry of ``axis`` is out of bounds for the
            input rank.
        """
        if self.built:
            return

        ndims = len(input_shape)

        # Validate and normalize axes
        normalized_axis = []
        for ax in self.axis:
            if ax >= ndims or ax < -ndims:
                raise ValueError(
                    f"Axis {ax} is out of bounds for tensor of dimension {ndims}"
                )
            # Convert negative axes to positive
            normalized_ax = ndims + ax if ax < 0 else ax
            normalized_axis.append(normalized_ax)

        self._norm_axis = normalized_axis

        # Calculate parameter shape for weight and bias
        param_shape = tuple(input_shape[ax] for ax in self._norm_axis)

        # Create layer's own weights
        # Alpha: one learnable scalar for the whole layer, hence shape=().
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=lambda shape, dtype: ops.cast(self.alpha_init_value, dtype),
            trainable=True,
            dtype=self.dtype
        )

        # Weight: affine transformation scaling
        self.weight = self.add_weight(
            name="weight",
            shape=param_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )

        # Bias: affine transformation offset
        self.bias = self.add_weight(
            name="bias",
            shape=param_shape,
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype
        )

        # Static broadcast tuple: every value is known here (param_shape comes
        # from input_shape), so call() never has to re-derive it from a dynamic
        # per-call shape query. Built by axis index, not by the order the axes
        # were written, so it stays identical to the previous construction.
        axis_to_size = dict(zip(self._norm_axis, param_shape))
        self._broadcast_shape = tuple(
            axis_to_size[i] if i in axis_to_size else 1 for i in range(ndims)
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply ``weight * tanh(alpha * inputs) + bias``.

        :param inputs: Input tensor. Any shape whose rank matches the one the
            layer was built on.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag. Unused; the layer behaves the same
            in both modes and the argument is kept for API compatibility.
        :type training: Optional[bool]

        :return: Transformed tensor, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # Step 1: Scale inputs by learnable alpha
        scaled_inputs = self.alpha * inputs

        # Step 2: Apply hyperbolic tangent
        tanh_outputs = ops.tanh(scaled_inputs)

        # Step 3: Apply affine transformation with proper broadcasting.
        # The reshape is required. A naive `tanh_outputs * self.weight` raises
        # InvalidArgumentError for any non-trailing axis: measured at axis=1 on a
        # (2, 8, 16) input, where weight has shape (8,). On CPU the message is
        # `Incompatible shapes: [2,8,16] vs. [8]`; on GPU the same op reports
        # `required broadcastable shapes`. The reshape target is the static tuple
        # computed in build().
        weight_broadcasted = ops.reshape(self.weight, self._broadcast_shape)
        bias_broadcasted = ops.reshape(self.bias, self._broadcast_shape)

        # Final affine transformation
        outputs = tanh_outputs * weight_broadcasted + bias_broadcasted

        return outputs

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this layer.

        ``axis`` is returned as the normalized list built in ``__init__``, which
        never changes after construction, so the config does not depend on
        whether ``build()`` has run.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'alpha_init_value': self.alpha_init_value,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config

# ---------------------------------------------------------------------
