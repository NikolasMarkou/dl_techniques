"""
A learnable, differentiable approximation of a step function.

This layer addresses a fundamental limitation of the Heaviside step function,
whose discontinuous nature and zero-everywhere derivative (except at the
origin) make it incompatible with gradient-based optimization methods. It
provides a smooth, fully differentiable surrogate by employing a transformed
hyperbolic tangent (`tanh`) function. The key innovation is that the two
defining characteristics of the step—its location and its steepness—are
promoted to trainable parameters, allowing a network to learn optimal soft
gating or thresholding behaviors.

Architectural and Mathematical Underpinnings:

The layer's operation is defined by a single mathematical transformation:

`f(x) = 0.5 * (tanh(slope * (x - shift)) + 1)`

Each component of this equation has a distinct conceptual role:

1.  **Hyperbolic Tangent (`tanh`)**: This function serves as the core
    differentiable switch. It naturally maps any real-valued input to the
    range `[-1, 1]`, exhibiting a steep, sigmoidal ("S"-shaped) transition
    centered around zero.

2.  **Learnable `shift` Parameter**: This parameter controls the horizontal
    position of the transition. By learning an optimal `shift`, the network
    determines the threshold or activation point at which the soft step
    occurs, effectively deciding "where" the gate should be placed.

3.  **Learnable `slope` Parameter**: This parameter controls the steepness of
    the transition. A larger `slope` value will cause the `tanh` function to
    more closely approximate a true, hard step function, leading to a more
    binary decision. Conversely, a smaller `slope` results in a gentler,
    softer transition. This allows the network to learn the "hardness" of
    the decision boundary required for the task.

4.  **Output Scaling (`0.5 * (... + 1)`)**: This is a simple linear
    transformation that rescales the `tanh` output from its native `[-1, 1]`
    range to the `[0, 1]` range. This makes the output directly interpretable
    as a probabilistic gate or a soft mask.

This functional form is a generalization of the gating mechanisms that are
central to modern recurrent neural networks, providing a versatile tool for
implementing conditional computation and attention in a variety of contexts.

References:
    - Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical
      Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
      (Provides a prominent example of using sigmoid-like gating functions).
"""

import keras
from typing import Optional, Union, Any, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.regularizers.l2_custom import L2_custom
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DifferentiableStep(keras.layers.Layer):
    """
    A learnable, differentiable step function, configurable for scalar or per-axis operation.

    This layer implements a soft, fully differentiable step function using a scaled
    and shifted hyperbolic tangent (tanh). The steepness (`slope`) and center point
    (`shift`) of the step are trainable parameters. It can operate in two modes:
    1.  **Scalar Mode (`axis=None`)**: A single `slope` and `shift` are learned and
        applied to the entire input tensor.
    2.  **Per-Axis Mode (`axis` is an int)**: A vector of `slope` and `shift` values
        is learned, one for each element along the specified axis, enabling
        per-feature or per-channel thresholding.

    **Intent**: Provide a flexible, learnable gating or soft thresholding mechanism.
    Useful for tasks requiring learned binary decisions, feature selection, or
    conditional computation where gradients must flow and different thresholds may be
    needed for different features.

    **Architecture**:
    The layer has no sub-layers and directly applies a mathematical transformation.
    ```
    Input(shape=[...])
           ↓
    f(x) = 0.5 * (tanh(slope * (x - shift)) + 1)
           ↓
    Output(shape=[...])
    ```

    **Mathematical Operation**:
        `output = 0.5 * (tanh(slope * (inputs - shift)) + 1)`

    Where `slope` and `shift` are either scalars or vectors that broadcast to the input shape.

    Args:
        axis: The axis over which to apply per-feature stepping. If `None`, a single
            scalar `slope` and `shift` are used for the entire input. If an integer,
            a separate `slope` and `shift` are learned for each index along this axis.
            Defaults to -1 (typically the feature/channel axis).
        slope_initializer: Initializer for the `slope` parameter(s). Can be a string
            name ('ones', 'zeros') or an Initializer instance. Defaults to 'ones'.
        shift_initializer: Initializer for the `shift` parameter(s). Can be a string
            name or an Initializer instance. Defaults to 'zeros'.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        N-D tensor of any shape. If `axis` is specified, the input must have a
        defined dimension for that axis.

    Output shape:
        N-D tensor with the same shape as the input.

    Attributes:
        slope: Trainable weight(s) controlling the steepness of the step. Shape is
            `()` if `axis` is `None`, otherwise it's broadcastable to the input shape
            along the specified `axis`.
        shift: Trainable weight(s) controlling the center point of the step. Shape
            is determined by the `axis` parameter, same as `slope`.

    Example:
        ```python
        # Per-feature gating (default axis=-1)
        # Learns a different threshold for each of the 64 features.
        inputs = keras.Input(shape=(64,))
        gated_features = DifferentiableStep(axis=-1)(inputs)
        model = keras.Model(inputs, gated_features)

        # Scalar gating
        # Learns one threshold for the entire input tensor.
        image_input = keras.Input(shape=(28, 28, 1))
        # Use a single gate for all pixels
        scalar_gate_layer = DifferentiableStep(axis=None)
        attention_mask = scalar_gate_layer(image_input)

        # Initialize with a steeper slope for per-channel gating
        image_input_rgb = keras.Input(shape=(28, 28, 3))
        channel_gate = DifferentiableStep(
            axis=-1, # One gate per channel (R, G, B)
            slope_initializer=keras.initializers.Constant(10.0),
            shift_initializer=keras.initializers.Constant(0.5)
        )(image_input_rgb)
        ```
    """

    def __init__(
            self,
            axis: Optional[int] = -1,
            slope_initializer: Union[str, keras.initializers.Initializer] = 'ones',
            shift_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            shift_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = keras.regularizers.L2(1e-3), # incentivize center -> 0
            shift_constraint: Optional[Union[str, keras.constraints.Constraint]] = ValueRangeConstraint(min_value=-1, max_value=+1), # clip it to be sure
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if axis is not None and not isinstance(axis, int):
            raise TypeError(f"Expected `axis` to be an int or None, but got: {axis}")

        # Store ALL configuration for serialization
        self.axis = axis
        self.slope_initializer = keras.initializers.get(slope_initializer)
        self.shift_initializer = keras.initializers.get(shift_initializer)
        self.shift_regularizer = keras.regularizers.get(shift_regularizer)
        self.shift_constraint = keras.constraints.get(shift_constraint)

        # Initialize weight attributes - they will be created in build()
        self.slope = None
        self.shift = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's trainable weights based on the `axis` configuration.
        """
        if self.axis is None:
            # Scalar mode: weights have no shape
            param_shape = ()
        else:
            # Per-axis mode: weights have a shape that broadcasts to the input
            rank = len(input_shape)

            # Handle negative axis
            axis = self.axis if self.axis >= 0 else rank + self.axis
            if axis < 0 or axis >= rank:
                raise ValueError(
                    f"Invalid axis: {self.axis}. It is out of bounds for an "
                    f"input of rank {rank}."
                )

            # Shape for broadcasting, e.g., [1, 1, channels, 1] for an image
            param_shape = [1] * rank
            if input_shape[axis] is None:
                raise ValueError(
                    f"The dimension for axis {axis} must be defined, but it is None. "
                    f"Input shape received: {input_shape}"
                )
            param_shape[axis] = input_shape[axis]
            param_shape = tuple(param_shape)

        # Create the layer's weights with the determined shape
        self.slope = self.add_weight(
            name='slope',
            shape=param_shape,
            initializer=self.slope_initializer,
            # reasonable range 1: normal tanh, 10: heavy side
            constraint=ValueRangeConstraint(min_value=+1.0, max_value=+10.0),
            # incentivizes the slope to increase -> become heavy side
            regularizer=L2_custom(-1e-3),
            trainable=True,
        )

        self.shift = self.add_weight(
            name='shift',
            shape=param_shape,
            initializer=self.shift_initializer,
            constraint=self.shift_constraint,
            regularizer=self.shift_regularizer,
            trainable=True,
        )

        # Always call the parent's build() method at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass computation applying the soft step function.
        Broadcasting handles both scalar and per-axis cases automatically.
        """
        scaled_shifted_x = self.slope * (inputs - self.shift)
        return (keras.ops.tanh(scaled_shifted_x) + 1.0) / 2.0

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.
        Since this is an element-wise operation, the output shape is
        identical to the input shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration for serialization.
        """
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'slope_initializer': keras.initializers.serialize(self.slope_initializer),
            'shift_initializer': keras.initializers.serialize(self.shift_initializer),
            'shift_regularizer': keras.regularizers.serialize(self.shift_regularizer),
            'shift_constraint': keras.constraints.serialize(self.shift_constraint),
        })
        return config

# ---------------------------------------------------------------------
