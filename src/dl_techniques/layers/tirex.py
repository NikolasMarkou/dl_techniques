"""
TiRex-inspired time series forecasting components for Keras.

This module implements time series forecasting layers inspired by the TiRex architecture,
adapted to work with Keras and our project's available components.
"""

import keras
from keras import ops
from typing import Optional, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .norms import RMSNorm
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ResidualBlock(keras.layers.Layer):
    """
    Residual block with linear transformations and ReLU activation.

    This layer applies a residual connection around a two-layer MLP with ReLU activation.

    Args:
        hidden_dim: Integer, dimensionality of the hidden layer.
        output_dim: Integer, dimensionality of the output space.
        dropout_rate: Float, dropout rate for regularization. Defaults to 0.0.
        activation: String or callable, activation function. Defaults to 'relu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            dropout_rate: float = 0.0,
            activation: Union[str, callable] = 'relu',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Layers will be initialized in build()
        self.hidden_layer = None
        self.output_layer = None
        self.residual_layer = None
        self.dropout = None
        self.activation_fn = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the layer weights."""
        self._build_input_shape = input_shape
        input_dim = input_shape[-1]

        # Hidden transformation
        self.hidden_layer = keras.layers.Dense(
            self.hidden_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="hidden_layer"
        )

        # Output transformation
        self.output_layer = keras.layers.Dense(
            self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_layer"
        )

        # Residual connection
        self.residual_layer = keras.layers.Dense(
            self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="residual_layer"
        )

        # Dropout
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

        # Build sublayers
        self.hidden_layer.build(input_shape)

        # Calculate intermediate shape after hidden layer
        hidden_output_shape = list(input_shape)
        hidden_output_shape[-1] = self.hidden_dim
        self.output_layer.build(tuple(hidden_output_shape))

        self.residual_layer.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass with residual connection."""
        # Main path
        hidden = self.hidden_layer(inputs, training=training)
        if self.dropout is not None:
            hidden = self.dropout(hidden, training=training)
        output = self.output_layer(hidden, training=training)

        # Residual path
        residual = self.residual_layer(inputs, training=training)

        # Combine
        return output + residual

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class StandardScaler(keras.layers.Layer):
    """
    Standard scaling layer for time series normalization.

    Applies z-score normalization: (x - mean) / std to the input along the last dimension.
    Handles NaN values by replacing them with nan_replacement before computation.

    Args:
        epsilon: Float, small value to prevent division by zero. Defaults to 1e-5.
        nan_replacement: Float, value to replace NaN values with. Defaults to 0.0.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            epsilon: float = 1e-5,
            nan_replacement: float = 0.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.nan_replacement = nan_replacement

    def call(self, inputs, training=None):
        """Apply standard scaling."""
        # Replace NaN values
        x = ops.where(ops.isnan(inputs), self.nan_replacement, inputs)

        # Compute mean and std along the last dimension
        mean = ops.mean(x, axis=-1, keepdims=True)
        variance = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
        std = ops.sqrt(variance + self.epsilon)

        # Handle case where std is zero by using absolute mean + epsilon
        std = ops.where(ops.equal(std, 0.0), ops.abs(mean) + self.epsilon, std)

        # Apply normalization
        normalized = (x - mean) / std

        # Store scaling parameters for potential inverse transform
        self.last_mean = mean
        self.last_std = std

        return normalized

    def inverse_transform(self, scaled_inputs):
        """Inverse transform the scaled data back to original scale."""
        if not hasattr(self, 'last_mean') or not hasattr(self, 'last_std'):
            logger.warning("No scaling parameters available for inverse transform")
            return scaled_inputs

        return scaled_inputs * self.last_std + self.last_mean

    def compute_output_shape(self, input_shape):
        """Output shape is same as input shape."""
        return input_shape

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "nan_replacement": self.nan_replacement,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PatchEmbedding(keras.layers.Layer):
    """
    Patch embedding layer for time series data.

    Converts time series into patches and embeds them into a higher dimensional space.
    Supports overlapping patches through stride parameter.

    Args:
        patch_size: Integer, size of each patch.
        embed_dim: Integer, embedding dimension.
        stride: Integer, stride for patch extraction. If None, uses patch_size (non-overlapping).
        padding: String, padding mode ('same', 'valid', or 'causal'). Defaults to 'causal'.
        use_bias: Boolean, whether to use bias in the embedding layer.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            patch_size: int,
            embed_dim: int,
            stride: Optional[int] = None,
            padding: str = 'causal',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride if stride is not None else patch_size
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Embedding layer will be initialized in build()
        self.embedding = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the embedding layer."""
        self._build_input_shape = input_shape

        # Create 1D convolution layer for patch embedding
        self.embedding = keras.layers.Conv1D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="patch_embedding"
        )

        self.embedding.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Convert inputs to patches and embed them."""
        # Handle NaN values by replacing with zeros
        x = ops.where(ops.isnan(inputs), 0.0, inputs)

        # Apply patch embedding
        embedded = self.embedding(x, training=training)

        return embedded

    def compute_output_shape(self, input_shape):
        """Compute output shape after patch embedding."""
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        if self.padding == 'valid':
            output_len = (seq_len - self.patch_size) // self.stride + 1
        elif self.padding == 'same':
            output_len = (seq_len + self.stride - 1) // self.stride
        else:  # causal
            output_len = seq_len // self.stride

        return (batch_size, output_len, self.embed_dim)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "stride": self.stride,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class QuantileHead(keras.layers.Layer):
    """
    Quantile prediction head for probabilistic forecasting.

    Predicts multiple quantiles simultaneously for uncertainty quantification.

    Args:
        num_quantiles: Integer, number of quantiles to predict.
        output_length: Integer, length of the forecast horizon.
        hidden_dim: Integer, hidden dimension for the prediction head.
        dropout_rate: Float, dropout rate for regularization.
        use_bias: Boolean, whether to use bias in layers.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            num_quantiles: int,
            output_length: int,
            hidden_dim: int = 256,
            dropout_rate: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_quantiles = num_quantiles
        self.output_length = output_length
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Layers will be initialized in build()
        self.projection = None
        self.dropout = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the quantile prediction head."""
        self._build_input_shape = input_shape

        # Project to quantile predictions
        self.projection = keras.layers.Dense(
            self.num_quantiles * self.output_length,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="quantile_projection"
        )

        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

        self.projection.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Predict quantiles."""
        x = inputs

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Project to quantile predictions
        quantile_preds = self.projection(x, training=training)

        # Reshape to [batch_size, num_quantiles, output_length]
        batch_size = ops.shape(inputs)[0]
        quantiles = ops.reshape(
            quantile_preds,
            (batch_size, self.num_quantiles, self.output_length)
        )

        return quantiles

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.num_quantiles, self.output_length)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_quantiles": self.num_quantiles,
            "output_length": self.output_length,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MixedSequentialBlock(keras.layers.Layer):
    """
    Mixed sequential block combining LSTM and self-attention mechanisms.

    This block can operate as either an LSTM block, a Transformer block, or a hybrid
    depending on the configuration.

    Args:
        embed_dim: Integer, embedding dimension.
        num_heads: Integer, number of attention heads (for transformer mode).
        lstm_units: Integer, number of LSTM units (for LSTM mode).
        ff_dim: Integer, feed-forward dimension.
        block_type: String, type of block ('lstm', 'transformer', or 'mixed').
        dropout_rate: Float, dropout rate for regularization.
        use_layer_norm: Boolean, whether to use layer normalization.
        activation: String or callable, activation function for feed-forward layers.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            lstm_units: Optional[int] = None,
            ff_dim: Optional[int] = None,
            block_type: str = 'mixed',
            dropout_rate: float = 0.1,
            use_layer_norm: bool = True,
            activation: Union[str, callable] = 'relu',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lstm_units = lstm_units if lstm_units is not None else embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 4
        self.block_type = block_type
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.activation = activation

        # Validate block type
        if block_type not in ['lstm', 'transformer', 'mixed']:
            raise ValueError(f"block_type must be one of ['lstm', 'transformer', 'mixed'], got: {block_type}")

        # Layers will be initialized in build()
        self.lstm_layer = None
        self.attention_layer = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None
        self.ff_layer1 = None
        self.ff_layer2 = None
        self.dropout1 = None
        self.dropout2 = None
        self.dropout3 = None
        self.projection = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the mixed sequential block."""
        self._build_input_shape = input_shape

        # Initialize layers based on block type
        if self.block_type in ['lstm', 'mixed']:
            self.lstm_layer = keras.layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name="lstm"
            )

        if self.block_type in ['transformer', 'mixed']:
            self.attention_layer = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                name="attention"
            )

        # Normalization layers
        if self.use_layer_norm:
            self.norm1 = RMSNorm(name="norm1")
            self.norm2 = RMSNorm(name="norm2")
            if self.block_type == 'mixed':
                self.norm3 = RMSNorm(name="norm3")

        # Feed-forward layers
        self.ff_layer1 = keras.layers.Dense(
            self.ff_dim,
            activation=self.activation,
            name="ff1"
        )
        self.ff_layer2 = keras.layers.Dense(
            self.embed_dim,
            name="ff2"
        )

        # Dropout layers
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)
        if self.block_type == 'mixed':
            self.dropout3 = keras.layers.Dropout(self.dropout_rate)

        # Projection layer for LSTM output to match embed_dim
        if self.block_type in ['lstm', 'mixed'] and self.lstm_units != self.embed_dim:
            self.projection = keras.layers.Dense(self.embed_dim, name="lstm_projection")

        # Build sublayers
        if self.lstm_layer is not None:
            self.lstm_layer.build(input_shape)
            lstm_output_shape = list(input_shape)
            lstm_output_shape[-1] = self.lstm_units
            if self.projection is not None:
                self.projection.build(tuple(lstm_output_shape))

        if self.attention_layer is not None:
            self.attention_layer.build(input_shape, input_shape)

        if self.norm1 is not None:
            self.norm1.build(input_shape)
        if self.norm2 is not None:
            self.norm2.build(input_shape)
        if self.norm3 is not None:
            self.norm3.build(input_shape)

        # FF layers
        self.ff_layer1.build(input_shape)
        ff1_output_shape = list(input_shape)
        ff1_output_shape[-1] = self.ff_dim
        self.ff_layer2.build(tuple(ff1_output_shape))

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        """Forward pass through the mixed sequential block."""
        x = inputs

        # LSTM processing
        if self.block_type in ['lstm', 'mixed']:
            if self.norm1 is not None:
                lstm_input = self.norm1(x, training=training)
            else:
                lstm_input = x

            lstm_output = self.lstm_layer(lstm_input, training=training, mask=mask)

            # Project LSTM output if needed
            if self.projection is not None:
                lstm_output = self.projection(lstm_output, training=training)

            lstm_output = self.dropout1(lstm_output, training=training)

            if self.block_type == 'lstm':
                x = x + lstm_output
            else:  # mixed
                x = x + lstm_output

        # Attention processing
        if self.block_type in ['transformer', 'mixed']:
            norm_layer = self.norm2 if self.block_type == 'transformer' else self.norm3
            dropout_layer = self.dropout2 if self.block_type == 'transformer' else self.dropout3

            if norm_layer is not None:
                attn_input = norm_layer(x, training=training)
            else:
                attn_input = x

            attn_output = self.attention_layer(
                attn_input, attn_input,
                training=training,
                attention_mask=mask
            )
            attn_output = dropout_layer(attn_output, training=training)
            x = x + attn_output

        # Feed-forward processing
        if self.norm2 is not None and self.block_type != 'mixed':
            ff_input = self.norm2(x, training=training)
        elif self.norm2 is not None:  # mixed case
            ff_input = self.norm2(x, training=training)
        else:
            ff_input = x

        ff_output = self.ff_layer1(ff_input, training=training)
        ff_output = self.ff_layer2(ff_output, training=training)

        final_dropout = self.dropout2 if self.block_type != 'mixed' else self.dropout2
        ff_output = final_dropout(ff_output, training=training)

        return x + ff_output

    def compute_output_shape(self, input_shape):
        """Output shape is same as input shape."""
        return input_shape

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "lstm_units": self.lstm_units,
            "ff_dim": self.ff_dim,
            "block_type": self.block_type,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "activation": self.activation,
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
