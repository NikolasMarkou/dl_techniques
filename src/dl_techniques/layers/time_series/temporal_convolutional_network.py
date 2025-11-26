import keras
from keras import layers

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TemporalBlock(layers.Layer):
    """
    A single residual block for the Temporal Convolutional Network.

    Consists of two dilated 1D convolutions with weight normalization,
    activation, and dropout.
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
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(
                self.filters, kernel_size=1, padding='same'
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)

        res = inputs if self.downsample is None else self.downsample(inputs)
        return layers.Activation(self.activation)(x + res)

    def get_config(self):
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
    Temporal Convolutional Network (TCN) Encoder.

    Used in NBEATSx to encode exogenous variables into a context basis.
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
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'num_levels': self.num_levels,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        })
        return config