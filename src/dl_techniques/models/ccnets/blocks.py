"""Shared CCNet building-block layers.

This module collects the generic, reusable Keras 3 layers used to assemble
the Explainer / Reasoner / Producer networks across all CCNet task
architectures (MNIST, CIFAR-100, text, ...). They are deliberately
task-agnostic and carry no CCNet-specific contract:

- :class:`FiLMLayer`: Feature-wise Linear Modulation that conditions a
  spatial content tensor on a style/explanation vector.
- :class:`ConvBlock`: Conv2D + BatchNorm + GoLU activation, with optional
  spatial pooling.
- :class:`DenseBlock`: Dense + (optional) BatchNorm + GoLU activation, with
  optional dropout.

All three are registered as Keras serializable so models built from them can
round-trip through ``model.save`` / ``keras.models.load_model``.
"""

import keras

from dl_techniques.layers.activations.golu import GoLU


@keras.saving.register_keras_serializable()
class FiLMLayer(keras.layers.Layer):
    """Feature-wise Linear Modulation (FiLM) with optional orthonormal regularization."""

    def __init__(self, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_regularizer = kernel_regularizer
        self.gamma_projection = None
        self.beta_projection = None
        self.num_channels = None

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(f"FiLMLayer expects 2 input shapes, got {input_shape}")

        content_shape, style_shape = input_shape
        self.num_channels = content_shape[-1]
        if self.num_channels is None:
            raise ValueError("Content tensor must have known channel count")

        self.gamma_projection = keras.layers.Dense(
            self.num_channels, activation='tanh',
            kernel_regularizer=self.kernel_regularizer,
            name=f"{self.name}_gamma_projection"
        )
        self.beta_projection = keras.layers.Dense(
            self.num_channels, activation='linear',
            name=f"{self.name}_beta_projection"
        )
        self.gamma_projection.build(style_shape)
        self.beta_projection.build(style_shape)
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError(f"FiLMLayer expects 2 inputs, got {len(inputs)}")
        content_tensor, style_vector = inputs
        gamma = self.gamma_projection(style_vector)
        beta = self.beta_projection(style_vector)
        gamma = keras.ops.expand_dims(keras.ops.expand_dims(gamma, 1), 1)
        beta = keras.ops.expand_dims(keras.ops.expand_dims(beta, 1), 1)
        return content_tensor * (1.0 + gamma) + beta

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config['kernel_regularizer'] = (
            keras.regularizers.serialize(self.kernel_regularizer)
            if self.kernel_regularizer else None
        )
        return config


@keras.saving.register_keras_serializable()
class ConvBlock(keras.layers.Layer):
    """Conv2D + BatchNorm + GoLU + optional Pooling block."""

    def __init__(
        self, filters, kernel_size=3, strides=1, padding="same",
        activation="relu", use_pooling=False, pool_size=2, pool_type="max",
        kernel_regularizer=None, **kwargs
    ):
        super().__init__(**kwargs)
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if pool_type not in ["max", "avg"]:
            raise ValueError(f"pool_type must be 'max' or 'avg', got {pool_type}")

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.kernel_regularizer = kernel_regularizer

        self.conv = keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides,
            padding=padding, kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_conv"
        )
        self.bn = keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.act = GoLU(name=f"{self.name}_activation")

        if use_pooling:
            pool_cls = keras.layers.MaxPooling2D if pool_type == "max" else keras.layers.AveragePooling2D
            self.pool = pool_cls(pool_size=pool_size, name=f"{self.name}_pool")
        else:
            self.pool = None

    def build(self, input_shape):
        self.conv.build(input_shape)
        conv_shape = self.conv.compute_output_shape(input_shape)
        self.bn.build(conv_shape)
        self.act.build(conv_shape)
        if self.pool is not None:
            self.pool.build(conv_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

    def compute_output_shape(self, input_shape):
        shape = self.conv.compute_output_shape(input_shape)
        if self.pool is not None:
            shape = self.pool.compute_output_shape(shape)
        return shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters, 'kernel_size': self.kernel_size,
            'strides': self.strides, 'padding': self.padding,
            'activation': self.activation, 'use_pooling': self.use_pooling,
            'pool_size': self.pool_size, 'pool_type': self.pool_type,
            'kernel_regularizer': (
                keras.regularizers.serialize(self.kernel_regularizer)
                if self.kernel_regularizer else None
            ),
        })
        return config


@keras.saving.register_keras_serializable()
class DenseBlock(keras.layers.Layer):
    """Dense + BatchNorm + GoLU + optional Dropout block."""

    def __init__(
        self, units, activation="relu", dropout_rate=0.0,
        use_batch_norm=True, kernel_regularizer=None, **kwargs
    ):
        super().__init__(**kwargs)
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")

        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.kernel_regularizer = kernel_regularizer

        self.dense = keras.layers.Dense(
            units=units, kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_dense"
        )
        self.bn = keras.layers.BatchNormalization(name=f"{self.name}_bn") if use_batch_norm else None
        self.act = GoLU(name=f"{self.name}_activation")
        self.dropout = keras.layers.Dropout(rate=dropout_rate, name=f"{self.name}_dropout") if dropout_rate > 0.0 else None

    def build(self, input_shape):
        self.dense.build(input_shape)
        dense_shape = self.dense.compute_output_shape(input_shape)
        if self.bn is not None:
            self.bn.build(dense_shape)
        self.act.build(dense_shape)
        if self.dropout is not None:
            self.dropout.build(dense_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        if self.bn is not None:
            x = self.bn(x, training=training)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units, 'activation': self.activation,
            'dropout_rate': self.dropout_rate, 'use_batch_norm': self.use_batch_norm,
            'kernel_regularizer': (
                keras.regularizers.serialize(self.kernel_regularizer)
                if self.kernel_regularizer else None
            ),
        })
        return config
