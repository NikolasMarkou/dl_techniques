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

2.  **RMS Normalization:**
    - The output of the dense layer is normalized using `RMSNorm`.
    - **Purpose:** This stabilizes the activations by constraining their Root Mean
      Square (RMS) value, preventing the feature magnitudes
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
from typing import Optional, Union, Any, Tuple

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.layer_scale import LearnableMultiplier
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint
from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class OrthoBlock(keras.layers.Layer):
    """Orthogonal Block layer with constrained feature scaling.

    This layer combines an orthogonally regularized dense projection with
    RMSNorm normalization and a constrained learnable feature gate. It is
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
            Must be positive.
        activation: Activation function to use. If `None`, no activation is applied.
            Can be string name of activation or callable. Defaults to None.
        use_bias: Boolean, whether the dense layer uses a bias vector.
            Defaults to True.
        ortho_reg_factor: Float, strength of the orthonormal regularization applied
            to the dense layer weights. Higher values enforce stronger orthogonality.
            Must be non-negative. Defaults to 0.01.
        kernel_initializer: Initializer for the dense layer kernel weights matrix.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the bias vector. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for the kernel weights.
            Defaults to None.
        bias_regularizer: Optional regularizer for the bias vector.
            Defaults to None.
        scale_initial_value: Float, initial value for the constrained scale
            parameters. Should be between 0.0 and 1.0. Defaults to 0.5.
        **kwargs: Additional keyword arguments for the Layer base class
            (e.g., name, dtype).

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
        ```python
        # Basic usage
        inputs = keras.Input(shape=(128,))
        outputs = OrthoBlock(units=64, activation='relu')(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # With custom regularization
        ortho_layer = OrthoBlock(
            units=32,
            activation='gelu',
            ortho_reg_factor=0.02,
            scale_initial_value=0.3,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        output = ortho_layer(input_tensor)
        ```

    Notes:
        - The orthonormal regularization encourages the weight matrix to have
          orthogonal columns, which can help with feature decorrelation.
        - The RMSNorm normalization helps stabilize training by constraining
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
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
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
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.ortho_reg_factor = ortho_reg_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.scale_initial_value = scale_initial_value

        # CREATE orthonormal regularizer
        self.ortho_reg = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=self.ortho_reg_factor
        )

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.dense = keras.layers.Dense(
            units=self.units,
            activation=None,  # No activation in dense layer - applied at the end
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.ortho_reg,
            bias_regularizer=self.bias_regularizer,
            name="ortho_dense"
        )

        # RMS normalization layer
        self.norm = RMSNorm(
            axis=-1,
            use_scale=False,  # We use our own constrained scaling
            name="norm_rms"
        )

        # Constrained scale layer
        self.constrained_scale = LearnableMultiplier(
            multiplier_type="CHANNEL",
            initializer=keras.initializers.Constant(self.scale_initial_value),
            regularizer=keras.regularizers.L1(1e-5),
            constraint=ValueRangeConstraint(min_value=0.0, max_value=1.0),
            name="constrained_scale"
        )

        logger.debug(f"Initialized OrthoBlock with {units} units and ortho_reg_factor={ortho_reg_factor}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its sub-layers.

        This method explicitly builds each sub-layer to ensure proper
        serialization and weight management following modern Keras 3 patterns.

        Args:
            input_shape: Shape tuple (tuple of integers), indicating the
                input shape of the layer.
        """
        logger.debug(f"Building OrthoBlock with input_shape: {input_shape}")

        # BUILD sub-layers in computational order to ensure proper shape propagation
        self.dense.build(input_shape)
        dense_output_shape = self.dense.compute_output_shape(input_shape)

        self.norm.build(dense_output_shape)
        norm_output_shape = self.norm.compute_output_shape(dense_output_shape)

        self.constrained_scale.build(norm_output_shape)

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug("OrthoBlock build completed successfully")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward computation through the orthogonal block.

        Args:
            inputs: Input tensor with shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying the full orthogonal block computation
            with shape (..., units).
        """
        # Step 1: Dense projection with orthonormal regularization
        z = self.dense(inputs, training=training)

        # Step 2: RMS normalization to stabilize activations
        z_norm = self.norm(z, training=training)

        # Step 3: Constrained scaling for learnable feature gating
        z_scaled = self.constrained_scale(z_norm, training=training)

        # Step 4: Apply final activation function
        outputs = self.activation(z_scaled)

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple where the last dimension is replaced with units.
        """
        # Convert to list for manipulation
        output_shape = list(input_shape)

        # Replace last dimension with units
        output_shape[-1] = self.units

        return tuple(output_shape)

    def get_config(self) -> dict[str, Any]:
        """Returns the layer's configuration for serialization.

        Returns:
            Dictionary containing ALL constructor parameters needed for
            layer reconstruction.
        """
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
            "scale_initial_value": self.scale_initial_value,
        })
        return config

# ---------------------------------------------------------------------