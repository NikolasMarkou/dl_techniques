"""
``LeVJEPAProjector``: the SIGReg projection head.

Ports the LeVJEPA PyTorch reference's ``Projector`` class (``module.py``)
verbatim in structure:

.. code-block:: python

    class Projector(nn.Module):
        def __init__(self, input_dim, hidden_dim=2048, output_dim=None,
                     norm_layer=nn.BatchNorm1d, act_layer=nn.GELU):
            output_dim = output_dim or input_dim
            norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), norm, act_layer(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            shape = x.shape
            x = x.reshape(-1, shape[-1])
            x = self.net(x)
            return x.reshape(*shape[:-1], x.shape[-1])

The reference flattens every leading axis to ``(-1, input_dim)`` before
``BatchNorm1d`` and reshapes back afterwards, because ``nn.BatchNorm1d``
only accepts rank-2 (or rank-3 with a channel-second layout) input. Keras'
``BatchNormalization(axis=-1)`` has no such rank restriction -- it already
treats every axis except the last as a batch axis for its per-channel
statistics. MEASURED (bash-tool run, ``.venv/bin/python3``): applying
``BatchNormalization(axis=-1)`` directly to a rank-3 tensor ``(4, 5, 8)``
versus reshaping to ``(20, 8)``, calling the SAME layer, and reshaping the
result back produced an EXACTLY IDENTICAL output (``max abs diff == 0.0``)
and identical ``moving_mean``/``moving_variance`` updates. So the explicit
reshape-around is unneeded complexity for this port and is deliberately
NOT implemented -- see the DECISION note below.

Architecture:
    .. code-block:: text

        x [..., input_dim]
          │
        Dense(hidden_dim)
          │
        BatchNormalization(axis=-1)
          │
        GELU
          │
        Dense(output_dim or input_dim)
          ▼
        y [..., output_dim]

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

# DECISION plan-2026-09-03T113223-2a714a91/D-014
# The reference's `Projector.forward` reshapes every leading axis to
# `(-1, input_dim)` before `nn.BatchNorm1d` and reshapes back afterwards,
# because `nn.BatchNorm1d` cannot consume a bare rank-3 tensor without that
# rearrangement. MEASURED (bash-tool run): Keras' `BatchNormalization(axis=-1)`
# applied DIRECTLY to a rank-3 input is numerically IDENTICAL (max abs diff
# 0.0, including moving-stat updates) to the reshape-to-2D-then-back path,
# because both compute statistics over every axis except the last regardless
# of rank. WHAT NOT TO DO: do not add a `reshape(-1, dim) -> ... -> reshape
# back` wrapper here "to match the PyTorch reference's rank handling" -- that
# reshape is a PyTorch/`BatchNorm1d`-API workaround with no Keras counterpart
# to work around, and adding it back would be unneeded complexity (KISS) with
# zero numerical effect. See decisions.md D-014.


@register_dl_technique("dl_techniques.models.levjepa.projector")
class LeVJEPAProjector(keras.layers.Layer):
    """SIGReg projection head: ``Dense -> BatchNorm -> GELU -> Dense``.

    Projects encoder embeddings into the space SIGReg's uniformity statistic
    is computed in. The input dimension is inferred from ``build()``'s
    ``input_shape`` (Keras-idiomatic), rather than requiring an explicit
    ``input_dim`` constructor argument as the PyTorch reference does.

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
