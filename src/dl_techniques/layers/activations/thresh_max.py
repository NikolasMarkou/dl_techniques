"""
A sparse softmax variant using differentiable confidence thresholding.

This layer provides a modification of the standard softmax function designed
to produce sparse probability distributions. Standard softmax often assigns
non-zero probabilities to all classes, even those that are highly
unlikely. ThreshMax encourages sparsity by effectively zeroing out the
probabilities of classes whose scores fall below a confidence-based
threshold.

Architectural Overview:
    The core principle is to transform the output of a standard softmax
    based on each class's "confidence" relative to a uniform distribution.
    The process involves:
    1.  A standard softmax is applied to the input logits to obtain `p`.
    2.  A confidence threshold is established as `τ = 1/N`.
    3.  A smooth, differentiable step function `S` is applied to the
        confidence difference `d = p - τ`.
    4.  **Gating Operation**: The mask `S(d)` is applied *multiplicatively*
        to the original probabilities `p`. This prevents "Rank Collapse" by
        preserving the relative magnitude of the confident classes.
    5.  The resulting values are renormalized to sum to one.

    Optimization Note: Previous versions included explicit handling for
    maximum entropy (uniform) inputs. Mathematical analysis confirms that
    the gated sum maintains a safe lower bound (approx 0.5) even in worst-case
    uniform scenarios, making explicit fallback logic unnecessary.

Mathematical Foundation:
    1. Standard Softmax: p = softmax(x)
    2. Confidence Gate:  g = 0.5 * (tanh(slope * (p - 1/N)) + 1)
    3. Gated Probs:      p_gated = p * g
    4. Renormalization:  p_sparse = p_gated / sum(p_gated)

References:
    -   **Sparse Softmax Variants:** Similar to Sparsemax.
    -   **Confidence Thresholding:** Pruning connections in attention.
    -   **Differentiable Relaxations:** Smooth approximation of hard steps.

"""

import keras
from keras import ops, initializers, regularizers, constraints
from typing import Optional, Any, Tuple, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.regularizers.l2_custom import L2_custom
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint


# ---------------------------------------------------------------------
# Keras layer implementation
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ThreshMax(keras.layers.Layer):
    """ThreshMax activation layer with learnable sparsity.

    A drop-in replacement for Softmax that produces sparse probability distributions
    while preserving the relative rank of confident classes. Supports an optional
    trainable slope parameter to learn the optimal sparsity threshold.

    Mathematical formulation:
        1. y_soft = softmax(x)
        2. gate = differentiable_step(y_soft - 1/N, slope)
        3. y_gated = y_soft * gate
        4. y_final = y_gated / sum(y_gated)

    Args:
        axis: Integer, axis along which to apply normalization. Defaults to -1.
        slope: Float, initial steepness of the step function. Defaults to 10.0.
        epsilon: Float, numerical stability constant. Defaults to 1e-12.
        trainable_slope: Boolean, if True, the slope is learned. Defaults to False.
        slope_initializer: Initializer for slope (if trainable).
        slope_regularizer: Regularizer for slope. Defaults to negative L2 to
            encourage sharpening over time.
        slope_constraint: Constraint for slope. Defaults to [1, 50.0].
    """

    def __init__(
            self,
            axis: int = -1,
            slope: float = 10.0,
            epsilon: float = 1e-12,
            trainable_slope: bool = False,
            slope_initializer: Union[str, initializers.Initializer] = "ones",
            slope_regularizer: Optional[Union[str, regularizers.Regularizer]] = L2_custom(-1e-4),
            slope_constraint: Optional[Union[str, constraints.Constraint]] = ValueRangeConstraint(1.0, 50.0),
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if slope <= 0:
            raise ValueError(f"slope must be positive, got {slope}")

        self.axis = axis
        self.slope_initial_value = float(slope)
        self.epsilon = float(epsilon)
        self.trainable_slope = trainable_slope
        self.slope_initializer = initializers.get(slope_initializer)
        self.slope_regularizer = regularizers.get(slope_regularizer)
        self.slope_constraint = constraints.get(slope_constraint)
        self.slope_weight = None

        logger.info(
            f"Initialized ThreshMax(axis={axis}, slope={slope}, "
            f"trainable_slope={trainable_slope})"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer weights if trainable_slope is True.
        
        Args:
            input_shape: Shape of the input tensor.
        """
        init = self.slope_initializer
        # Use slope_initial_value if using default "ones" initializer
        is_default_ones = (
                isinstance(init, initializers.Ones) or
                (isinstance(init, str) and init == 'ones') or
                (hasattr(init, 'get_config') and init.get_config().get('class_name') == 'Ones')
        )

        if is_default_ones and self.slope_initial_value != 1.0:
            init = initializers.Constant(self.slope_initial_value)

        self.slope_weight = self.add_weight(
            name='slope',
            shape=(),
            initializer=init,
            regularizer=self.slope_regularizer,
            constraint=self.slope_constraint,
            trainable=self.trainable_slope
        )

        super().build(input_shape)

    @staticmethod
    def _differentiable_step(
            x: keras.KerasTensor,
            slope: Union[float, keras.KerasTensor] = 1.0,
            shift: Union[float, keras.KerasTensor] = 0.0
    ) -> keras.KerasTensor:
        """Approximates a Heaviside step function using a scaled and shifted tanh.

        The formula is: f(x) = (tanh(slope * (x - shift)) + 1) / 2

        Args:
            x: Input tensor.
            slope: Controls steepness. Higher = sharper step.
            shift: Center point where output is 0.5.

        Returns:
            Tensor with values smoothly transitioning from 0 to 1.
        """
        # Cast scalar inputs to tensor if x is a tensor for consistent broadcasting
        if isinstance(slope, (int, float)):
            slope = ops.convert_to_tensor(slope, dtype=x.dtype)
        if isinstance(shift, (int, float)):
            shift = ops.convert_to_tensor(shift, dtype=x.dtype)

        scaled_shifted_x = slope * (x - shift)
        return (ops.tanh(scaled_shifted_x) + 1.0) / 2.0

    def _compute_threshmax(
            self,
            x: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Internal computation for ThreshMax activation.

        Optimized logic:
        1. Computes Softmax.
        2. Calculates soft gate based on deviation from uniform distribution.
        3. Applies multiplicative gating (Rank Preservation).
        4. Renormalizes.

        Args:
            x: Input logits.

        Returns:
            Sparse probability distribution summing to 1.
        """
        # Step 1: Compute standard softmax
        y_soft = keras.activations.softmax(x, axis=self.axis)

        # Step 2: Compute confidence difference from uniform probability
        num_classes = ops.shape(x)[self.axis]
        uniform_prob = 1.0 / ops.cast(num_classes, x.dtype)
        confidence_diff = y_soft - uniform_prob

        # Step 3: Compute soft gating mask
        # This creates a value in [0, 1] based on confidence
        gate = self._differentiable_step(confidence_diff, slope=self.slope_weight, shift=0.0)

        # Step 4: Apply gating mask to original probabilities (Multiplicative)
        # This prevents Rank Collapse by preserving relative probability magnitudes
        y_stepped = y_soft * gate

        # Step 5: Renormalize
        # The sum of y_stepped is theoretically lower-bounded around 0.5 (for uniform inputs)
        # and higher for peaked inputs, so explicit degenerate case handling is dead code.
        total_sum = ops.sum(y_stepped, axis=self.axis, keepdims=True)
        return y_stepped / (total_sum + self.epsilon)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply ThreshMax activation to inputs.
        
        Args:
            inputs: Input tensor containing logits.
            training: Boolean indicating training mode (unused).
            
        Returns:
            Output tensor with sparse probability distributions.
        """
        return self._compute_threshmax(inputs)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.
        
        Args:
            input_shape: Shape of the input tensor.
            
        Returns:
            Shape of the output tensor (same as input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.
        
        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'slope': self.slope_initial_value,
            'epsilon': self.epsilon,
            'trainable_slope': self.trainable_slope,
            'slope_initializer': initializers.serialize(self.slope_initializer),
            'slope_regularizer': regularizers.serialize(self.slope_regularizer),
            'slope_constraint': constraints.serialize(self.slope_constraint),
        })
        return config

    def __repr__(self) -> str:
        """Get string representation of the layer.
        
        Returns:
            String representation.
        """
        mode = "learnable" if self.trainable_slope else "fixed"
        return (f"ThreshMax(axis={self.axis}, slope={self.slope_initial_value}, "
                f"mode='{mode}', name='{self.name}')")


# ---------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------


def thresh_max(
        x: keras.KerasTensor,
        axis: int = -1,
        slope: Union[float, keras.KerasTensor] = 10.0,
        epsilon: float = 1e-12
) -> keras.KerasTensor:
    """Functional interface for ThreshMax activation.

    Args:
        x: Input tensor containing logits.
        axis: The axis along which the softmax normalization is applied.
        slope: Controls the steepness of the differentiable step function.
        epsilon: Small value for numerical stability.

    Returns:
        Output tensor with sparse probability distributions.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if isinstance(slope, (int, float)) and slope <= 0:
        raise ValueError(f"slope must be positive, got {slope}")

    layer = ThreshMax(axis=axis, slope=slope, epsilon=epsilon, trainable_slope=False)
    return layer(x)

# ---------------------------------------------------------------------
