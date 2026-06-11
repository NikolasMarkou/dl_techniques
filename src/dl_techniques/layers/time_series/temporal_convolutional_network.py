import keras
from keras import layers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TemporalBlock(layers.Layer):
    """
    A single residual block for the Temporal Convolutional Network.

    Consists of two dilated causal 1D convolutions with activation and
    dropout, followed by a residual connection. If the input channel count
    differs from ``filters``, a 1x1 convolution is used to match dimensions
    before the residual addition.

    **Architecture Overview:**

    .. code-block:: text

        Input x: (batch, time, channels)
                │
                ├──────────────────────────────┐
                ▼                              │ (residual)
        ┌───────────────────────┐              │
        │  Causal Conv1D        │              │
        │  (dilation=d, k=k)    │              │
        └───────────┬───────────┘              │
                    ▼                          │
        ┌───────────────────────┐              │
        │      Dropout          │              │
        └───────────┬───────────┘              │
                    ▼                          │
        ┌───────────────────────┐              │
        │  Causal Conv1D        │              │
        │  (dilation=d, k=k)    │              │
        └───────────┬───────────┘              │
                    ▼                          │
        ┌───────────────────────┐              │
        │      Dropout          │              │
        └───────────┬───────────┘              │
                    ▼                          │
                  ( + ) ◄──────────────────────┘
                    │       (1x1 Conv if needed)
                    ▼
        ┌───────────────────────┐
        │     Activation        │
        └───────────┬───────────┘
                    ▼
              Output: (batch, time, filters)

    :param filters: Number of convolutional filters (output channels).
    :type filters: int
    :param kernel_size: Size of the convolutional kernel.
    :type kernel_size: int
    :param dilation_rate: Dilation rate for the causal convolutions.
    :type dilation_rate: int
    :param dropout_rate: Dropout probability applied after each convolution.
    :type dropout_rate: float
    :param activation: Activation function name for convolutions and residual output.
    :type activation: str
    :param kernel_initializer: Initializer for the convolutional kernels.
    :type kernel_initializer: str
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        self.activation = activation
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
        Build the layer and all sublayers, threading shapes through each.

        Explicitly builds ``conv1``, ``dropout1``, ``conv2``, ``dropout2``, the
        residual activation, and (conditionally) the ``downsample`` projection so
        that every inner ``Conv1D`` materializes its variables at build time —
        before any forward pass or ``.keras`` weight restore.

        :param input_shape: Shape tuple of the input tensor.
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
        Compute the output shape of the temporal block.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape ``(batch, time, filters)``.
        :rtype: tuple
        """
        return (input_shape[0], input_shape[1], self.filters)

    def call(self, inputs, training=None):
        """
        Forward pass through the temporal block.

        :param inputs: Input tensor of shape ``(batch, time, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
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
        Return the layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer
        })
        return config


@keras.saving.register_keras_serializable()
class TemporalConvNet(layers.Layer):
    """
    Temporal Convolutional Network (TCN) encoder.

    Stacks multiple ``TemporalBlock`` layers with exponentially increasing
    dilation rates (1, 2, 4, ..., 2^{num_levels-1}) to achieve a large
    receptive field while keeping the parameter count compact. Used in
    NBEATSx to encode exogenous variables into a context basis.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, time, channels)
                    │
                    ▼
        ┌───────────────────────────┐
        │  TemporalBlock            │
        │  (dilation=1)             │
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────────────┐
        │  TemporalBlock            │
        │  (dilation=2)             │
        └───────────┬───────────────┘
                    ▼
                   ...
                    ▼
        ┌───────────────────────────┐
        │  TemporalBlock            │
        │  (dilation=2^{L-1})       │
        └───────────┬───────────────┘
                    ▼
        Output: (batch, time, filters)

    :param filters: Number of convolutional filters in each block.
    :type filters: int
    :param kernel_size: Kernel size for dilated convolutions.
    :type kernel_size: int
    :param num_levels: Number of stacked temporal blocks.
    :type num_levels: int
    :param dropout_rate: Dropout probability within each block.
    :type dropout_rate: float
    :param activation: Activation function name used in each block.
    :type activation: str
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        self.activation = activation

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
        Build each stacked ``TemporalBlock`` in sequence, threading shapes.

        Block 0 receives ``input_shape``; each subsequent block receives the
        previous block's output shape ``(batch, time, filters)``. Building each
        block here materializes all inner ``Conv1D`` children at build time so
        that ``encoder.build()`` (e.g. from ``ExogenousBlock``) propagates fully
        and ``.keras`` weight restore lands correctly.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
        current = input_shape
        for block in self.blocks:
            block.build(current)
            current = block.compute_output_shape(current)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the temporal convolutional network.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape ``(batch, time, filters)``.
        :rtype: tuple
        """
        return (input_shape[0], input_shape[1], self.filters)

    def call(self, inputs, training=None):
        """
        Forward pass through the stacked temporal blocks.

        :param inputs: Input tensor of shape ``(batch, time, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
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
        Return the layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'num_levels': self.num_levels,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        })
        return config
