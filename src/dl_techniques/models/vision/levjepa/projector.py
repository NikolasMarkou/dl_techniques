"""``LeVJEPAProjector``: the SIGReg projection head.

Ports the LeVJEPA PyTorch reference's ``Projector`` class: ``Dense ->
BatchNorm -> GELU -> Dense``, projecting encoder embeddings into the space
SIGReg's uniformity statistic is computed in.

The PyTorch reference flattens every leading axis to ``(-1, input_dim)``
before ``BatchNorm1d`` and reshapes back afterwards, since ``BatchNorm1d``
only accepts rank-2 input. Keras' ``BatchNormalization(axis=-1)`` has no
such restriction: it treats every axis but the last as a batch axis
regardless of rank, so this port applies it directly with no reshape.

References:
    - LeVJEPA PyTorch reference, ``module.py::Projector`` (pasted transcript;
      no public arXiv id in this plan's context).
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# DECISION plan-2026-09-03T113223-2a714a91/D-014: no reshape-around BatchNorm.
# Measured: BatchNormalization(axis=-1) on a rank-3 input matches the reference's
# reshape-to-2D-then-back path exactly (max abs diff 0.0). See decisions.md.


@register_dl_technique("dl_techniques.models.levjepa.projector")
class LeVJEPAProjector(keras.layers.Layer):
    """SIGReg projection head: ``Dense -> BatchNorm -> GELU -> Dense``.

    Projects encoder embeddings into the space SIGReg's uniformity statistic
    is computed in. The input dimension is inferred from ``build()``'s
    ``input_shape`` (Keras-idiomatic), rather than requiring an explicit
    ``input_dim`` constructor argument as the PyTorch reference does.

    Architecture:

    .. code-block:: text

        x [..., input_dim]
            |
        Dense(hidden_dim)
            |
        BatchNormalization(axis=-1)
            |
        GELU
            |
        Dense(output_dim or input_dim)
            |
        y [..., output_dim]

    :param hidden_dim: Width of the hidden ``Dense`` layer. Must be positive.
        Defaults to ``2048``, matching the reference.
    :type hidden_dim: int
    :param output_dim: Width of the output ``Dense`` layer. ``None``
        (default) uses the inferred input dimension, matching the
        reference's ``output_dim = output_dim or input_dim``.
    :type output_dim: Optional[int]
    :param kernel_initializer: Initializer for both ``Dense`` kernels.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for both ``Dense`` biases. Defaults
        to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for both kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for both biases.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar fc1: ``Dense(hidden_dim)``.
    :ivar norm: ``BatchNormalization(axis=-1)``.
    :ivar act: ``Activation('gelu')``.
    :ivar fc2: ``Dense(output_dim or input_dim)``, built lazily in ``build()``
        once ``input_dim`` is known.

    Input shape:
        ``(..., input_dim)`` -- any rank ``>= 2``.

    Output shape:
        ``(..., output_dim)``, where ``output_dim`` defaults to
        ``input_dim`` when not given.

    :raises ValueError: If ``hidden_dim`` is not positive, or ``output_dim``
        is given and not positive. Raised from ``__init__``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.models.vision.levjepa.projector import LeVJEPAProjector

        projector = LeVJEPAProjector(hidden_dim=2048)
        x = keras.random.normal((3, 8, 384))  # (views, batch, dim)
        projector(x).shape
        # (3, 8, 384)
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        output_dim: Optional[int] = None,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and create every sub-layer except the
        final ``Dense``, which is deferred to ``build()`` because its width
        may depend on the inferred input dimension.

        :param hidden_dim: Width of the hidden ``Dense`` layer.
        :type hidden_dim: int
        :param output_dim: Width of the output ``Dense`` layer, or ``None``
            to use the inferred input dimension.
        :type output_dim: Optional[int]
        :param kernel_initializer: Initializer for both ``Dense`` kernels.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param bias_initializer: Initializer for both ``Dense`` biases.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Optional kernel regularizer.
        :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
        :param bias_regularizer: Optional bias regularizer.
        :type bias_regularizer: Optional[keras.regularizers.Regularizer]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If the configuration is invalid.
        """
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim is not None and output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim) if output_dim is not None else None
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.fc1 = keras.layers.Dense(
            self.hidden_dim,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="fc1",
        )
        self.norm = keras.layers.BatchNormalization(axis=-1, name="norm")
        self.act = keras.layers.Activation("gelu", name="act")
        # `fc2` is created in `build()` once the resolved output width
        # (`output_dim` or the inferred input dim) is known.
        self.fc2: Optional[keras.layers.Dense] = None

        logger.info(
            f"Initialized LeVJEPAProjector with hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer in computational order.

        :param input_shape: Shape of the input, ``(..., input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` has rank ``< 2`` or its last
            dimension is unknown.
        """
        if self.built:
            return

        if len(input_shape) < 2:
            raise ValueError(f"Expected an input of rank >= 2, got {input_shape}")
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("The last dimension of the input must be known, got None")

        resolved_output_dim = self.output_dim if self.output_dim is not None else input_dim

        self.fc1.build(input_shape)
        hidden_shape = tuple(input_shape[:-1]) + (self.hidden_dim,)
        self.norm.build(hidden_shape)
        self.fc2 = keras.layers.Dense(
            resolved_output_dim,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="fc2",
        )
        self.fc2.build(hidden_shape)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Apply ``Dense -> BatchNorm -> GELU -> Dense``.

        No reshape-around is performed -- see the module-level ``DECISION``
        note: Keras' ``BatchNormalization(axis=-1)`` is numerically
        equivalent, at any rank, to the reference's reshape-to-2D-then-back.

        :param inputs: Input tensor, ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag.
        :type training: Optional[bool]
        :return: Output tensor, ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        """
        x = self.fc1(inputs, training=training)
        x = self.norm(x, training=training)
        x = self.act(x)
        return self.fc2(x, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape.

        :param input_shape: Shape of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(..., output_dim)``, where ``output_dim`` defaults to
            ``input_shape[-1]`` when not explicitly set.
        :rtype: Tuple[Optional[int], ...]
        """
        resolved_output_dim = (
            self.output_dim if self.output_dim is not None else input_shape[-1]
        )
        return tuple(input_shape[:-1]) + (resolved_output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        :return: Dictionary holding every ``__init__`` parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------
