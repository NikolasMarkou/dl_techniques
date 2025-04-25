import keras
from keras import ops
from keras.api.layers import Dense, Layer
from typing import Optional, Union, Any, Dict, Tuple, List

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.layers.layer_scale import LearnableMultiplier


@keras.saving.register_keras_serializable()
class OrthonomalRegularizer(keras.regularizers.Regularizer):
    """
    Orthonormal regularization that enforces W^T * W ≈ I.

    Args:
        factor: Float, strength of the orthonormal regularization.
    """

    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, x):
        # Get shape information
        if len(x.shape) != 2:
            return 0.0

        # Compute W^T * W - I
        wt_w = ops.matmul(ops.transpose(x), x)
        identity = ops.eye(wt_w.shape[0], wt_w.shape[1])
        diff = wt_w - identity
        # Compute Frobenius norm squared
        ortho_loss = ops.sum(ops.square(diff))
        return self.factor * ortho_loss

    def get_config(self):
        return {'factor': self.factor}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable()
class CenteringNormalization(Layer):
    """
    Layer that performs centering normalization (subtract mean only).

    Args:
        axis: Integer or list of integers, the axis to normalize along.
        epsilon: Small float added for numerical stability.
    """

    def __init__(self, axis=-1, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis if isinstance(axis, list) else [axis]
        self.epsilon = epsilon
        self._build_input_shape = None

    def build(self, input_shape):
        self._build_input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs):
        # Calculate mean along specified axes
        mean = ops.mean(inputs, axis=self.axis, keepdims=True)
        return inputs - mean

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'epsilon': self.epsilon
        })
        return config

    def get_build_config(self):
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config):
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class LogitNormalization(Layer):
    """
    Layer that performs L2 normalization.

    Args:
        axis: Integer or list of integers, the axis to normalize along.
        epsilon: Small float added to norm to avoid dividing by zero.
    """

    def __init__(self, axis=-1, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self._build_input_shape = None

    def build(self, input_shape):
        self._build_input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs):
        # Calculate L2 norm
        square_sum = ops.sum(ops.square(inputs), axis=self.axis, keepdims=True)
        norm = ops.sqrt(square_sum + self.epsilon)
        return inputs / norm

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'epsilon': self.epsilon
        })
        return config

    def get_build_config(self):
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config):
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

@keras.saving.register_keras_serializable()
class OrthoCenterBlock(keras.layers.Layer):
    """
    A custom layer that implements:
    Dense Layer → Soft Orthonormal Regularization → Centering Normalization → Logit Normalization → Constrained Scale [0,1]

    This block combines orthonormal-regularized weights with mean-centering normalization,
    avoiding the redundancy of full layer normalization while preserving variance information.
    The final logit normalization projects to the unit hypersphere, and constrained scales
    provide sparse, interpretable feature attention.

    Args:
        units: Integer, dimensionality of the output space.
        activation: Activation function to use.
        use_bias: Boolean, whether the layer uses a bias vector.
        ortho_reg_factor: Float, strength of the orthonormal regularization.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Additional regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias vector.
        scale_initial_value: Float, initial value for the scale parameters.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            units: int,
            activation: Optional[Union[str, callable]] = None,
            use_bias: bool = True,
            ortho_reg_factor: float = 0.1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            scale_initial_value: float = 0.5,
            **kwargs: Any
    ):
        super().__init__(**kwargs)

        # Store configuration parameters
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.ortho_reg_factor = ortho_reg_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.scale_initial_value = scale_initial_value

        # These will be initialized in build
        self.dense = None
        self.centering = None
        self.logit_norm = None
        self.constrained_scale = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the layer and its sublayers."""
        # Store for serialization
        self._build_input_shape = input_shape

        # Create orthonormal regularizer
        ortho_reg = OrthonomalRegularizer(factor=self.ortho_reg_factor)

        # Combine with existing regularizer if present
        combined_regularizer = self.kernel_regularizer
        if combined_regularizer is not None:
            # This is a workaround since we can't easily combine regularizer objects
            old_reg = combined_regularizer

            def combined_fn(x):
                return old_reg(x) + ortho_reg(x)

            combined_regularizer = combined_fn
        else:
            combined_regularizer = ortho_reg

        # Create sublayers
        self.dense = Dense(
            units=self.units,
            activation=None,  # We'll apply activation separately
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=combined_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # For 2D inputs (batch, features), center across feature dimension
        # For 4D inputs (batch, height, width, channels), center across all but batch
        self.centering = CenteringNormalization(axis=-1)
        self.logit_norm = LogitNormalization(axis=-1)
        self.constrained_scale = \
            LearnableMultiplier(
                multiplier_type="CHANNEL",
                initializer=keras.initializers.Constant(self.scale_initial_value),
                regularizer=keras.regularizers.L1(1e-5),
                constraint=ValueRangeConstraint(min_value=0.0, max_value=1.0)
            )

        # Build all sublayers
        self.dense.build(input_shape)

        # Calculate intermediate shapes for building other layers
        dense_output_shape = self.dense.compute_output_shape(input_shape)

        self.centering.build(dense_output_shape)
        center_output_shape = self.centering.compute_output_shape(dense_output_shape)

        self.logit_norm.build(center_output_shape)
        logit_output_shape = self.logit_norm.compute_output_shape(center_output_shape)

        self.constrained_scale.build(logit_output_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward computation with all normalization steps."""
        # Apply dense layer
        z = self.dense(inputs)

        # Apply centering normalization
        z_centered = self.centering(z)

        # Apply logit normalization
        z_logit = self.logit_norm(z_centered)

        # Apply constrained scale
        outputs = self.constrained_scale(z_logit)

        # Apply activation if specified
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return self.dense.compute_output_shape(input_shape)

    def get_config(self):
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "ortho_reg_factor": self.ortho_reg_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "scale_initial_value": self.scale_initial_value
        })
        return config

    def get_build_config(self):
        """Get the config needed to build the layer from a config."""
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config):
        """Build the layer from a config created with get_build_config."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

