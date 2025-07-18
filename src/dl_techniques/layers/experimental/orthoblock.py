"""
OrthoBlock: A Structured Layer for Learning Decorrelated Features.

This module implements a composite Keras layer designed to learn feature
representations that are decorrelated, well-scaled, and interpretable. It
achieves this by enforcing a specific, mathematically-motivated computational
flow, making it a powerful alternative to a standard `Dense` layer in many
deep learning models.

The layer is built on the principle that structured feature extraction can lead
to better generalization, stability, and model insight.

Computational Flow:
The `OrthoBlock` processes inputs through a four-stage pipeline:

1.  **Orthogonally Regularized Dense Projection:**
    - A standard linear transformation (`z = xW + b`) where the weight matrix `W`
      is regularized to have approximately orthonormal columns.
    - **Math:** The regularizer adds a penalty proportional to `||WᵀW - I||²` to
      the loss, encouraging the columns of `W` to form an orthonormal basis.
    - **Purpose:** This promotes feature decorrelation, as the input is projected
      onto a set of orthogonal axes. It also improves gradient flow and
      training stability due to the norm-preserving properties of orthogonal
      transformations.

2.  **BandRMS Normalization:**
    - The output of the dense layer is normalized using `BandRMS`.
    - **Purpose:** This stabilizes the activations by constraining their Root Mean
      Square (RMS) value to a learnable range, preventing the feature magnitudes
      from exploding or vanishing.

3.  **Constrained Learnable Scaling:**
    - The normalized activations are element-wise multiplied by a learnable vector
      `s`, where each element `s_i` is constrained to the range `[0, 1]`.
    - **Purpose:** This acts as a **learnable feature gate**. The model can learn
      to "turn off" less important features by driving their corresponding scale
      factor towards zero, or "pass through" important ones by driving the scale
      towards one. This adds interpretability and enables automatic feature
      selection.

4.  **Final Activation:**
    - A standard non-linear activation function is applied.
"""

import keras
from typing import Optional, Union, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.layer_scale import LearnableMultiplier
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class OrthoBlock(keras.layers.Layer):
    """Orthogonal Block layer with constrained feature scaling.

    This layer combines an orthogonally regularized dense projection with
    BandRMS normalization and a constrained learnable feature gate. It is
    designed to improve training stability and model interpretability.

    Key Benefits:
    - **Feature Decorrelation:** The orthonormal regularization on the dense
      kernel encourages the learned features to be independent.
    - **Training Stability:** The combination of an isometric-like projection
      and robust activation normalization stabilizes gradient updates.
    - **Interpretability:** The constrained scaling vector acts as a feature
      gate, providing insight into which learned features are most impactful
      for the model's task.

    Args:
        units: Integer, dimensionality of the output space (number of neurons).
        activation: Activation function to use. If `None`, no activation is applied.
            Can be string name of activation or callable.
        use_bias: Boolean, whether the dense layer uses a bias vector.
        ortho_reg_factor: Float, strength of the orthonormal regularization applied
            to the dense layer weights. Higher values enforce stronger orthogonality.
        kernel_initializer: Initializer for the dense layer kernel weights matrix.
            If `None`, the default initializer ("glorot_uniform") will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the default
            initializer ("zeros") will be used.
        bias_regularizer: Optional regularizer for the bias vector.
        scale_initial_value: Float, initial value for the constrained feature scale
            parameters. Should be between 0.0 and 1.0.
        **kwargs: Additional keyword arguments for the Layer base class (e.g., name, dtype).

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common case would be a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.

    Returns:
        A tensor representing the result of the orthogonal block computation.

    Raises:
        ValueError: If `units` is not a positive integer.
        ValueError: If `ortho_reg_factor` is negative.
        ValueError: If `scale_initial_value` is not between 0.0 and 1.0.

    Example:
        >>> # Basic usage
        >>> x = keras.Input(shape=(128,))
        >>> y = OrthoBlock(units=64, activation='relu')(x)
        >>> model = keras.Model(inputs=x, outputs=y)

        >>> # With custom regularization
        >>> ortho_layer = OrthoBlock(
        ...     units=32,
        ...     activation='gelu',
        ...     ortho_reg_factor=0.02,
        ...     scale_initial_value=0.3
        ... )
        >>> output = ortho_layer(input_tensor)

    Notes:
        - The orthonormal regularization encourages the weight matrix to have
          orthogonal columns, which can help with feature decorrelation.
        - The BandRMS normalization helps stabilize training by constraining
          the L2 norm of activations.
        - The constrained scaling acts as a learnable feature gate, where values
          close to 0 effectively "turn off" features.
        - This layer is particularly useful in scenarios where interpretable
          and decorrelated features are desired.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        ortho_reg_factor: float = 0.01,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        scale_initial_value: float = 0.5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate input parameters
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")
        if not isinstance(ortho_reg_factor, (int, float)) or ortho_reg_factor < 0:
            raise ValueError(f"ortho_reg_factor must be non-negative, got {ortho_reg_factor}")
        if not isinstance(scale_initial_value, (int, float)) or not (0.0 <= scale_initial_value <= 1.0):
            raise ValueError(f"scale_initial_value must be between 0.0 and 1.0, got {scale_initial_value}")

        # Store configuration parameters
        activation_str = activation if activation is not None else "linear"
        self.units = units
        self.activation = keras.activations.get(activation_str)
        self.use_bias = use_bias
        self.ortho_reg_factor = ortho_reg_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.scale_initial_value = scale_initial_value

        # Create orthonormal regularizer
        self.ortho_reg = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=self.ortho_reg_factor
        )
        # Sublayers - to be initialized in build()
        self.dense = None
        self.norm = None
        self.constrained_scale = None
        self._build_input_shape = None

        logger.debug(f"Initialized OrthoBlock with {units} units and ortho_reg_factor={ortho_reg_factor}")

    def build(self, input_shape):
        """Build the layer and its sublayers.

        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples,
                indicating the input shape of the layer.
        """
        logger.debug(f"Building OrthoBlock with input_shape: {input_shape}")
        self._build_input_shape = input_shape

        # Build Dense sublayer with orthonormal regularization
        self.dense = keras.layers.Dense(
            units=self.units,
            activation=None,  # Explicitly set to None (not linear)
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.ortho_reg,
            bias_regularizer=self.bias_regularizer,
            name="ortho_dense"
        )

        # Build BandRMS normalization layer
        self.norm = BandRMS(name="band_rms")

        # Build constrained scale layer
        self.constrained_scale = LearnableMultiplier(
            multiplier_type="CHANNEL",
            initializer=keras.initializers.Constant(self.scale_initial_value),
            regularizer=keras.regularizers.L1(1e-5),
            constraint=ValueRangeConstraint(min_value=0.0, max_value=1.0),
            name="constrained_scale"
        )

        # Build all sublayers sequentially to ensure proper shape propagation
        self.dense.build(input_shape)
        dense_output_shape = self.dense.compute_output_shape(input_shape)

        self.norm.build(dense_output_shape)
        norm_output_shape = self.norm.compute_output_shape(dense_output_shape)

        self.constrained_scale.build(norm_output_shape)

        super().build(input_shape)
        logger.debug("OrthoBlock build completed successfully")

    def call(self, inputs, training=None):
        """Forward computation through the orthogonal block.

        Args:
            inputs: Input tensor or list/tuple of input tensors.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying the full orthogonal block computation.
        """
        # Dense projection with orthonormal regularization
        z = self.dense(inputs, training=training)

        # BandRMS normalization to stabilize activations
        z_norm = self.norm(z, training=training)

        # Constrained scaling for feature gating
        z_scaled = self.constrained_scale(z_norm, training=training)

        # Apply final activation function
        outputs = self.activation(z_scaled)

        return outputs

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple.
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Replace last dimension with units
        output_shape_list = input_shape_list[:-1] + [self.units]

        # Return as tuple for consistency
        return tuple(output_shape_list)

    def get_config(self):
        """Returns the layer's configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "ortho_reg_factor": self.ortho_reg_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "scale_initial_value": self.scale_initial_value
        })
        return config

    def get_build_config(self):
        """Returns the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Builds the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])
            logger.debug("OrthoBlock rebuilt from config successfully")

# ---------------------------------------------------------------------
