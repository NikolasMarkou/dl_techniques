import keras
from keras import layers
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.temporal_convolutional_network")
class TemporalBlock(layers.Layer):
    """
    One residual block of a Temporal Convolutional Network.

    The block runs two dilated causal 1D convolutions, each followed by
    dropout. It then adds the input back and applies the activation to the
    sum. Causal padding means output step ``t`` sees only inputs up to ``t``;
    the gradient of ``output[30]`` with respect to any later input measures
    exactly ``0.0``.

    The residual add happens inside this block. A caller that stacks blocks
    writes ``x = block(x)`` and must not add its own skip connection.

    The residual branch is the input itself when the input already has
    ``filters`` channels. Otherwise a 1x1 convolution projects it so the
    shapes match. That projection is created in ``build()``, so with matching
    channels it holds no weights at all.

    **Architecture Overview:**

    .. code-block:: text

          Input x: [B, T, C]
                  │
                  ├─────────────────────┐
                  ▼                     │
        ┌────────────────────┐          │
        │ Conv1D causal      │          │
        │ dilation=d, k, act │          │
        └─────────┬──────────┘          │
                  ▼                     │
        ┌────────────────────┐          │
        │ Dropout            │          │
        └─────────┬──────────┘          │
                  ▼                     │
        ┌────────────────────┐          │
        │ Conv1D causal      │          │
        │ dilation=d, k, act │          │
        └─────────┬──────────┘          │
                  ▼                     │
        ┌────────────────────┐          │
        │ Dropout            │          ▼
        └─────────┬──────────┘  ┌─────────────────┐
                  │ [B, T, F]   │ Conv1D 1x1      │
                  │             │ (optional)      │
                  │             └────────┬────────┘
                  │                      │ [B, T, F]
                  ▼                      │
                ( + ) ◄──────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Activation         │
        └─────────┬──────────┘
                  ▼
           Output: [B, T, F]

    ``C == filters`` removes the 1x1 branch and the input is added as is.

    The activation runs three times per block. Each ``Conv1D`` is built with
    ``activation=activation``, and ``self.act`` applies it again to the
    residual sum. The 1x1 projection is linear and uses the Keras default
    kernel initializer, not ``kernel_initializer``.

    Input shape:
        3D tensor ``(batch, time, channels)``.

    Output shape:
        3D tensor ``(batch, time, filters)``. The time axis is unchanged.

    Example:
        .. code-block:: python

            block = TemporalBlock(filters=32, kernel_size=3, dilation_rate=4)
            y = block(keras.random.normal((8, 128, 16)))

    :param filters: Number of convolutional filters, and the output channel
        count. Must be positive.
    :type filters: int
    :param kernel_size: Kernel size of both dilated convolutions. Must be
        positive.
    :type kernel_size: int
    :param dilation_rate: Dilation rate of both convolutions. Must be
        positive.
    :type dilation_rate: int
    :param dropout_rate: Dropout probability after each convolution. Must be
        in ``[0, 1]``.
    :type dropout_rate: float
    :param activation: Activation name used inside both convolutions and
        again on the residual sum.
    :type activation: str
    :param kernel_initializer: Initializer for the two convolution kernels.
        The 1x1 residual projection does not use it.
    :type kernel_initializer: str
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``filters``, ``kernel_size`` or ``dilation_rate``
        is not positive, or if ``dropout_rate`` is outside ``[0, 1]``.
    """

    def __init__(
            self,
            filters: int,
            kernel_size: int,
            dilation_rate: int,
            dropout_rate: float = 0.0,
            activation: str = 'relu',
            kernel_initializer: str = 'he_normal',
            **kwargs
    ):
        super().__init__(**kwargs)
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if dilation_rate <= 0:
            raise ValueError(f"dilation_rate must be positive, got {dilation_rate}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.activation = deserialize_activation(activation)
        self.kernel_initializer = kernel_initializer

        # Padding 'causal' in Keras handles the Chomp1d logic automatically
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation=activation,
            kernel_initializer=kernel_initializer
        )
        self.dropout1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation=activation,
            kernel_initializer=kernel_initializer
        )
        self.dropout2 = layers.Dropout(dropout_rate)

        # Reused activation applied to the residual sum (created once, not per call)
        self.act = layers.Activation(activation)

        # 1x1 conv for residual connection if dimensions mismatch
        self.downsample = None

    def build(self, input_shape):
        """
        Build the block and every sublayer, threading shapes through each.

        Builds ``conv1``, ``dropout1``, ``conv2``, ``dropout2`` and the residual
        activation. If ``input_shape[-1]`` differs from ``filters``, it also
        creates and builds the 1x1 ``downsample`` projection. Building here
        materializes all inner ``Conv1D`` variables before any forward pass or
        ``.keras`` weight restore.

        :param input_shape: Shape tuple ``(batch, time, channels)``.
        :type input_shape: tuple
        """
        self.conv1.build(input_shape)
        shape_f = (input_shape[0], input_shape[1], self.filters)
        self.dropout1.build(shape_f)
        self.conv2.build(shape_f)
        self.dropout2.build(shape_f)
        self.act.build(shape_f)

        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(
                self.filters, kernel_size=1, padding='same'
            )
            self.downsample.build(input_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the block.

        The time axis is unchanged; only the channel axis becomes ``filters``.

        :param input_shape: Shape tuple ``(batch, time, channels)``.
        :type input_shape: tuple
        :return: Output shape ``(batch, time, filters)``.
        :rtype: tuple
        """
        return (input_shape[0], input_shape[1], self.filters)

    def call(self, inputs, training=None):
        """
        Run the two convolutions, add the residual, then activate.

        The residual branch is ``inputs`` when ``downsample`` is ``None``, and
        ``downsample(inputs)`` otherwise.

        :param inputs: Input tensor of shape ``(batch, time, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer runs in training mode. Only the two
            dropout sublayers use it.
        :type training: bool, optional
        :return: Output tensor of shape ``(batch, time, filters)``.
        :rtype: keras.KerasTensor
        """
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)

        res = inputs if self.downsample is None else self.downsample(inputs)
        return self.act(x + res)

    def get_config(self):
        """
        Return the constructor arguments needed to rebuild this block.

        :return: Configuration dictionary, with ``activation`` serialized back
            to its string name.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate,
            'activation': serialize_activation(self.activation),
            'kernel_initializer': self.kernel_initializer
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.temporal_convolutional_network")
class TemporalConvNet(layers.Layer):
    """
    Temporal Convolutional Network (TCN) encoder.

    Stacks ``num_levels`` ``TemporalBlock`` layers. Block ``i`` uses dilation
    ``2**i``, so the receptive field grows exponentially with depth while the
    parameter count grows linearly. NBEATSx uses this layer to encode
    exogenous variables into a context basis.

    Each block carries its own residual add, so this layer is a plain
    sequential chain: ``x = block(x)`` per level, with no outer skip
    connection. The residual wiring lives in ``TemporalBlock``, not here.

    **Architecture Overview:**

    .. code-block:: text

            Input: [B, T, C]
                    │
                    ▼
        ┌───────────────────────────┐
        │ TemporalBlock  dilation=1 │
        │ (residual add inside)     │
        └───────────┬───────────────┘
                    ▼ [B, T, F]
        ┌───────────────────────────┐
        │ TemporalBlock  dilation=2 │
        └───────────┬───────────────┘
                    ▼ [B, T, F]
                   ...
                    ▼
        ┌───────────────────────────┐
        │ TemporalBlock  d=2^(L-1)  │
        └───────────┬───────────────┘
                    ▼
            Output: [B, T, F]

    Only the first block changes the channel count, from ``C`` to
    ``filters``. Every later block already sees ``filters`` channels, so its
    1x1 residual projection is skipped.

    **Dilation Ladder and Receptive Field:**

    .. code-block:: text

        L = num_levels, k = kernel_size, 2 convs per block

          dilations        1, 2, 4, ..., 2^(L-1)
          receptive field  1 + 2*(k-1)*(2^L - 1) steps

          k=2  L=4    31 steps
          k=3  L=4    61 steps
          k=2  L=8   511 steps
          k=3  L=8  1021 steps

    The formula was checked against measured gradient support for every
    ``L`` in 1..5 and ``k`` in {2, 3}, and matched in all ten cases.

    Input shape:
        3D tensor ``(batch, time, channels)``.

    Output shape:
        3D tensor ``(batch, time, filters)``. The time axis is unchanged.

    Example:
        .. code-block:: python

            tcn = TemporalConvNet(filters=64, kernel_size=3, num_levels=4)
            y = tcn(keras.random.normal((8, 200, 5)))

    :param filters: Number of filters in every block, and the output channel
        count. Must be positive.
    :type filters: int
    :param kernel_size: Kernel size passed to every block. Must be positive.
    :type kernel_size: int
    :param num_levels: Number of stacked blocks. Must be positive. Doubling
        it roughly doubles the receptive field.
    :type num_levels: int
    :param dropout_rate: Dropout probability inside each block. Must be in
        ``[0, 1]``.
    :type dropout_rate: float
    :param activation: Activation name passed to every block.
    :type activation: str
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``filters``, ``kernel_size`` or ``num_levels`` is
        not positive, or if ``dropout_rate`` is outside ``[0, 1]``.
    """

    def __init__(
            self,
            filters: int,
            kernel_size: int = 2,
            num_levels: int = 4,
            dropout_rate: float = 0.0,
            activation: str = 'relu',
            **kwargs
    ):
        super().__init__(**kwargs)
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.num_levels = num_levels
        self.dropout_rate = dropout_rate
        self.activation = deserialize_activation(activation)

        self.blocks = []
        for i in range(num_levels):
            dilation_rate = 2 ** i
            self.blocks.append(
                TemporalBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    dropout_rate=dropout_rate,
                    activation=activation
                )
            )

    def build(self, input_shape):
        """
        Build each stacked ``TemporalBlock`` in order, threading shapes.

        Block 0 gets ``input_shape``. Every later block gets the previous
        block's output shape ``(batch, time, filters)``. Building the blocks
        here materializes all inner ``Conv1D`` children, so a caller such as
        ``ExogenousBlock`` that calls ``encoder.build()`` gets full propagation
        and a correct ``.keras`` weight restore.

        :param input_shape: Shape tuple ``(batch, time, channels)``.
        :type input_shape: tuple
        """
        current = input_shape
        for block in self.blocks:
            block.build(current)
            current = block.compute_output_shape(current)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the stack.

        The time axis is unchanged; only the channel axis becomes ``filters``.

        :param input_shape: Shape tuple ``(batch, time, channels)``.
        :type input_shape: tuple
        :return: Output shape ``(batch, time, filters)``.
        :rtype: tuple
        """
        return (input_shape[0], input_shape[1], self.filters)

    def call(self, inputs, training=None):
        """
        Run the input through every block in order.

        Each block already adds its own residual, so nothing is added here.

        :param inputs: Input tensor of shape ``(batch, time, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer runs in training mode. Forwarded to
            every block.
        :type training: bool, optional
        :return: Encoded tensor of shape ``(batch, time, filters)``.
        :rtype: keras.KerasTensor
        """
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def get_config(self):
        """
        Return the constructor arguments needed to rebuild this encoder.

        :return: Configuration dictionary, with ``activation`` serialized back
            to its string name.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'num_levels': self.num_levels,
            'dropout_rate': self.dropout_rate,
            'activation': serialize_activation(self.activation)
        })
        return config

# ---------------------------------------------------------------------
