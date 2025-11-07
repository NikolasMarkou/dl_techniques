"""
Learn decorrelated and gated features through a structured pipeline.

This layer provides a mathematically-motivated alternative to a standard Dense
layer by enforcing a structured computational flow. The core philosophy is to
decouple the feature transformation into three distinct, interpretable steps:
a rotation to create decorrelated features, a normalization step to stabilize
activations, and a learned gating mechanism for feature selection. This
structured approach is designed to improve gradient flow, enhance model
stability, and provide insights into feature importance.

Architecturally, the layer implements a four-stage pipeline:
1.  **Orthogonal Projection:** An initial linear transformation is applied,
    but its weight matrix `W` is heavily regularized to be orthogonal. This
    encourages the columns of `W` to form an orthonormal basis.
2.  **Magnitude Stabilization:** The resulting activations are normalized using
    Root Mean Square Normalization (ZeroCenteredRMSNorm). This step controls the
    magnitude of the feature vectors without the centering operation of
    Layer Normalization, preserving the directional information from the
    orthogonal projection.
3.  **Non-Linearity:** A standard activation function is applied.
4.  **Feature Gating:** A learnable scaling vector `s`, with its values
    constrained to the range `[0, 1]`, is applied element-wise.

The foundational mathematical principle is the enforcement of orthogonality on
the projection matrix `W`. An orthogonal transformation is a rotation (or
roto-reflection) that preserves the geometric structure of the input data,
such as distances and angles. This property is enforced via a soft
regularization term that penalizes the matrix `W` based on its deviation from
orthonormality:

`Loss_ortho = λ * ||W^T W - I||²_F`

where `I` is the identity matrix and `||.||²_F` is the squared Frobenius
norm. By minimizing this loss, the columns of `W` are encouraged to be
mutually orthogonal and have a norm of one. This has two critical benefits:
it ensures that the learned features are decorrelated, capturing unique and
non-redundant information, and it promotes stable gradient flow by preventing
the transformation from exploding or vanishing gradient magnitudes.

The final scaling step, `s`, acts as a set of learnable gates. Because each
`s_i` is constrained to `[0, 1]`, the model can learn to selectively "turn
off" features by driving their corresponding `s_i` towards zero, or "pass
them through" by driving `s_i` towards one. This provides a form of automatic,
differentiable feature selection and offers a degree of interpretability, as
the final values of `s` can be inspected to understand the relative
importance of the learned features.

References:
    - Saxe et al., 2013. Exact solutions to the nonlinear dynamics of
      learning in deep linear neural networks (for orthogonal initialization).
    - Cisse et al., 2017. Parseval Networks: Improving Robustness to
      Adversarial Examples (for orthogonal regularization).
    - Zhang & Sennrich, 2019. Root Mean Square Layer Normalization.

"""

import keras
from typing import Optional, Union, Any, Tuple, Dict, Callable

# ---------------------------------------------------------------------
# Framework-specific imports
# ---------------------------------------------------------------------

from .layer_scale import LearnableMultiplier
from .norms.zero_centered_rms_norm import ZeroCenteredRMSNorm
from ..constraints.value_range_constraint import ValueRangeConstraint
from ..regularizers.binary_preference import BinaryPreferenceRegularizer
from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from ..initializers.hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class OrthoBlock(keras.layers.Layer):
    """
    Structured feature learning block with orthogonal regularization and constrained scaling.

    This composite layer implements a four-stage pipeline designed to learn decorrelated,
    well-scaled, and interpretable feature representations. It combines orthogonally
    regularized linear projection, RMS normalization, learnable feature gating, and
    configurable activation to create a structured alternative to standard Dense layers.

    **Intent**: Provide a mathematically-motivated building block that encourages
    feature decorrelation, training stability, and model interpretability through
    constrained feature scaling that acts as learnable feature gates.

    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           ↓
    Dense(units) + OrthonormalRegularizer
           ↓
    ZeroCenteredRMSNorm(stabilize activations)
           ↓
    Activation(user-specified)
           ↓
    ConstrainedScale([0,1] learnable gates)
           ↓
    Output(shape=[..., units])
    ```

    **Computational Flow**:

    1. **Orthogonally Regularized Dense Projection**:
       - Linear transformation: z = xW + b where W has orthonormal regularization
       - Math: Regularizer adds penalty ∝ ||W^T W - I||² encouraging orthogonal columns
       - Purpose: Promotes feature decorrelation and improves gradient flow stability

    2. **RMS Normalization**:
       - Root Mean Square normalization to constrain activation magnitudes
       - Purpose: Prevents feature explosion/vanishing, stabilizes training dynamics

    3. **Constrained Learnable Scaling**:
       - Element-wise multiplication by learnable vector s ∈ [0,1]^units
       - Purpose: Acts as learnable feature gates - model learns feature importance

    4. **Final Activation**:
       - Standard non-linear activation function applied to scaled features

    **Key Benefits**:
    - Feature decorrelation through orthonormal weight regularization
    - Training stability via RMS normalization and orthogonal transformations
    - Interpretability through constrained scaling vectors (feature importance)
    - Automatic feature selection when scale factors approach zero

    Args:
        units: Integer, dimensionality of the output space (number of neurons).
            Must be positive. This determines both the output size and the number
            of feature gates in the scaling layer.
        activation: Optional activation function. Can be string name ('relu', 'gelu'),
            callable, or None for linear activation. Applied after all other operations.
            Defaults to None.
        use_bias: Boolean, whether the dense layer uses a bias vector.
            When True, adds learnable bias term after linear transformation.
            Defaults to True.
        ortho_reg_factor: Float, strength of orthonormal regularization applied
            to dense layer weights. Higher values enforce stronger orthogonality
            but may slow convergence. Must be non-negative. Defaults to 0.01.
        kernel_initializer: Initializer for the dense layer weight matrix.
            String name or Initializer instance. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the bias vector. Only used when
            use_bias=True. Defaults to 'zeros'.
        bias_regularizer: Optional regularizer for the bias vector.
            Defaults to None.
        scale_initial_value: Float, initial value for constrained scale parameters.
            Must be between 0.0 and 1.0. Higher values start with more "open" gates.
            Defaults to 0.5 for balanced initialization.
        **kwargs: Additional keyword arguments for Layer base class (name, dtype, etc.).

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        Most common: 2D tensor with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        Same rank as input, but last dimension becomes `units`.

    Attributes:
        dense: Dense layer with orthonormal regularization for feature projection.
        norm: ZeroCenteredRMSNorm layer for activation stabilization.
        constrained_scale: LearnableMultiplier with [0,1] constraints for feature gating.
        ortho_reg: SoftOrthonormalConstraintRegularizer for weight matrix regularization.

    Example:
        ```python
        # Basic usage - decorrelated features with ReLU activation
        inputs = keras.Input(shape=(128,))
        outputs = OrthoBlock(units=64, activation='relu')(inputs)
        model = keras.Model(inputs, outputs)

        # Custom regularization for stronger orthogonality
        ortho_layer = OrthoBlock(
            units=32,
            activation='gelu',
            ortho_reg_factor=0.02,        # Stronger orthogonal regularization
            scale_initial_value=0.3,      # Start with more closed gates
        )

        # In a deep network for feature decorrelation
        inputs = keras.Input(shape=(784,))
        x = keras.layers.Dense(512, activation='relu')(inputs)
        x = OrthoBlock(units=256, activation='gelu')(x)  # Decorrelated features
        x = keras.layers.Dense(128, activation='relu')(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        # Access learned feature importance after training
        scale_weights = ortho_layer.constrained_scale.get_weights()[0]
        print(f"Feature importance scores: {scale_weights}")  # Values in [0,1]
        ```

    Raises:
        ValueError: If units is not a positive integer.
        ValueError: If ortho_reg_factor is negative.
        ValueError: If scale_initial_value is not between 0.0 and 1.0.

    Note:
        The constrained scaling vector provides interpretability - values close to 0
        indicate features the model considers less important, while values close to 1
        indicate important features. This can be inspected post-training for model
        analysis and feature selection insights.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]]] = None,
        use_bias: bool = True,
        ortho_reg_factor: float = 0.01,
        kernel_initializer: Union[str, keras.initializers.Initializer] = OrthogonalHypersphereInitializer(),
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        scale_initial_value: float = 0.5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate input parameters with clear error messages
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")
        if not isinstance(ortho_reg_factor, (int, float)) or ortho_reg_factor < 0:
            raise ValueError(f"ortho_reg_factor must be non-negative, got {ortho_reg_factor}")
        if not isinstance(scale_initial_value, (int, float)) or not (0.0 <= scale_initial_value <= 1.0):
            raise ValueError(f"scale_initial_value must be between 0.0 and 1.0, got {scale_initial_value}")

        # Store ALL configuration parameters for serialization
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.ortho_reg_factor = ortho_reg_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.scale_initial_value = scale_initial_value

        # CREATE orthonormal regularizer
        self.ortho_reg = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=self.ortho_reg_factor,
            l1_coefficient=1e-5,
            use_matrix_scaling=True
        )

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        # Dense layer with orthonormal regularization
        self.dense = keras.layers.Dense(
            units=self.units,
            activation=None,  # Activation applied separately at the end
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.ortho_reg,  # Orthonormal regularization
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            name="ortho_dense"
        )


        # RMS normalization for activation stabilization
        self.norm = ZeroCenteredRMSNorm(
            axis=-1,
            use_scale=False,  # We handle scaling separately with constraints
            name="zero_centered_rms_norm"
        )

        # Constrained learnable scaling for feature gating
        self.constrained_scale = LearnableMultiplier(
            multiplier_type="CHANNEL",
            initializer=keras.initializers.Constant(self.scale_initial_value),
            regularizer=BinaryPreferenceRegularizer(multiplier=1e-4),  # Encourage sparsity
            constraint=ValueRangeConstraint(min_value=0.0, max_value=1.0),
            name="constrained_scale"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization
        following the modern Keras 3 pattern.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Validate input shape
        if input_shape[-1] is None:
            raise ValueError("Last dimension of input must be defined for OrthoBlock")

        # BUILD sub-layers in computational order for proper shape propagation
        self.dense.build(input_shape)
        dense_output_shape = self.dense.compute_output_shape(input_shape)

        self.norm.build(dense_output_shape)
        norm_output_shape = self.norm.compute_output_shape(dense_output_shape)

        self.constrained_scale.build(norm_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward computation through the orthogonal block pipeline.

        Applies the four-stage computation: orthogonally regularized dense projection,
        RMS normalization, constrained scaling, and final activation.

        Args:
            inputs: Input tensor with shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor after applying the full orthogonal block computation
            with shape (..., units).
        """
        # Stage 1: Dense projection with orthonormal regularization
        z = self.dense(inputs, training=training)

        # Stage 2: RMS normalization for activation stabilization
        z_norm = self.norm(z, training=training)

        # Stage 3: Constrained scaling for learnable feature gating
        outputs = self.constrained_scale(z_norm, training=training)

        # Stage 4: Apply activation function
        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            outputs = outputs

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

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

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration for serialization.

        Returns ALL constructor parameters needed for layer reconstruction.

        Returns:
            Dictionary containing complete layer configuration.
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
            "scale_initial_value": self.scale_initial_value,
        })
        return config

# ---------------------------------------------------------------------