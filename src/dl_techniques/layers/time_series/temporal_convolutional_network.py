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

        # 1x1 conv for residual connection if dimensions mismatch
        self.downsample = None

    def build(self, input_shape):
        """
        Build the layer, creating a 1x1 projection if input channels differ from filters.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(
                self.filters, kernel_size=1, padding='same'
            )
        super().build(input_shape)

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
        return layers.Activation(self.activation)(x + res)

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
