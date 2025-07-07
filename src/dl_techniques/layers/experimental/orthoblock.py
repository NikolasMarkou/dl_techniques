import keras
from typing import Optional, Union, Any
from keras.api.layers import Dense, LayerNormalization

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .layer_scale import LearnableMultiplier
from ..constraints.value_range_constraint import ValueRangeConstraint
from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class OrthoBlock(keras.layers.Layer):
    """
    A layer implementing: Dense → Orthonormal Reg → {Norm} → Constrained Scale → Activation.

    This block learns a decorrelated feature representation by enforcing an orthonormal
    constraint on the dense layer's weights. It then uses Layer Normalization to
    stabilize the activations. Finally, a constrained learnable scale acts as a
    sparse, interpretable feature gate before the final activation.

    Args:
        units: Integer, dimensionality of the output space.
        activation: Activation function to use.
        use_bias: Boolean, whether the dense layer uses a bias vector.
        ortho_reg_factor: Float, strength of the orthonormal regularization.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer applied to the kernel weights (e.g., L2).
        bias_regularizer: Regularizer applied to the bias vector.
        scale_initial_value: Float, initial value for the feature scale parameters.
        **kwargs: Additional keyword arguments for the Layer base class.
    """
    def __init__(
            self,
            units: int,
            activation: Optional[Union[str, callable]] = None,
            use_bias: bool = True,
            ortho_reg_factor: float = 0.01,
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

        # Sublayers to be initialized in build()
        self.dense = None
        self.ortho_reg = None
        self.norm = None  # Renamed for clarity
        self.constrained_scale = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the layer and its sublayers."""
        self._build_input_shape = input_shape

        # The layer's internal regularizer, applied manually in `call()`.
        self.ortho_reg = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=self.ortho_reg_factor
        )

        # The Dense sublayer, correctly receiving the user's regularizer.
        self.dense = Dense(
            units=self.units,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="ortho_dense"
        )

        # Use LayerNormalization, disabling its internal scale to avoid redundancy.
        self.norm = LayerNormalization(center=True, scale=False, name="layer_norm")

        self.constrained_scale = LearnableMultiplier(
            multiplier_type="CHANNEL",
            initializer=keras.initializers.Constant(self.scale_initial_value),
            regularizer=keras.regularizers.L1(1e-5),
            constraint=ValueRangeConstraint(min_value=0.0, max_value=1.0),
            name="constrained_scale"
        )

        # Build all sublayers sequentially
        self.dense.build(input_shape)
        dense_output_shape = self.dense.compute_output_shape(input_shape)
        self.norm.build(dense_output_shape)
        norm_output_shape = self.norm.compute_output_shape(dense_output_shape)
        self.constrained_scale.build(norm_output_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward computation with all steps, including adding the orthonormal loss."""
        z = self.dense(inputs)

        # Apply the internal orthonormal loss using `self.add_loss()`.
        if self.ortho_reg_factor > 0:
            ortho_loss = self.ortho_reg(self.dense.kernel)
            self.add_loss(ortho_loss)

        # Apply the normalization and scaling pipeline
        z_norm = self.norm(z)
        z_scaled = self.constrained_scale(z_norm)

        # Apply final activation. This is the correct placement.
        if self.activation is not None:
            outputs = self.activation(z_scaled)
        else:
            outputs = z_scaled

        return outputs

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return self.dense.compute_output_shape(input_shape)

    def get_config(self):
        """Returns the layer's configuration for serialization."""
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
        """Returns the config needed to build the layer from a config."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Builds the layer from a config created with get_build_config."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
