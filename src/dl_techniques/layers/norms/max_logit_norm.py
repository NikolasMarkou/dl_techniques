"""
MaxLogit Normalization Implementations for Out-of-Distribution Detection.

This module implements variants of MaxLogit normalization for out-of-distribution detection:
1. Basic MaxLogit normalization
2. Decoupled MaxLogit (DML) normalization that separates cosine and L2 components
3. DML+ with dedicated models for cosine and norm components

The implementations follow the paper:
"Decoupling MaxLogit for Out-of-Distribution Detection"

Mathematical Background:
-----------------------
MaxLogit normalization improves out-of-distribution (OOD) detection by normalizing
logits using their L2 norm. This separates the magnitude information from the
direction information in the logit space.

Key Benefits:
- **Better OOD Detection**: Separates in-distribution from out-of-distribution samples
- **Interpretable Components**: Decouples cosine similarity and magnitude components
- **Improved Calibration**: Provides better uncertainty estimates
- **Training Stability**: L2 normalization prevents logit explosion

References:
[1] "Decoupling MaxLogit for Out-of-Distribution Detection"
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MaxLogitNorm(keras.layers.Layer):
    """Basic MaxLogit normalization layer for out-of-distribution detection.

    Applies L2 normalization to logits to separate magnitude and direction
    components, improving OOD detection. The layer computes:
    ``output = inputs / ||inputs||_2``, where the L2 norm is taken along
    the specified axis with epsilon for numerical stability.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────┐
        │    Input Logits (x)     │
        │   shape: (B, C)         │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  L2 Norm along axis     │
        │  norm = √(Σ(x²) + ε)    │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Normalize: x / norm    │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Normalized Logits      │
        │  shape: (B, C)          │
        └─────────────────────────┘

    :param axis: Axis along which to normalize. Typically -1 for the class
        dimension. Defaults to -1.
    :type axis: int
    :param epsilon: Small constant for numerical stability. Must be positive.
        Defaults to 1e-7.
    :type epsilon: float

    :raises ValueError: If epsilon is not positive.
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Initialize the MaxLogitNorm layer.

        :param axis: Axis along which to normalize.
        :type axis: int
        :param epsilon: Small constant for numerical stability. Must be positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for the Layer base class.
        :type kwargs: Any

        :raises ValueError: If epsilon is not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.axis = axis
        self.epsilon = epsilon

        logger.debug(f"Initialized MaxLogitNorm with axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, epsilon: float) -> None:
        """Validate initialization parameters.

        :param epsilon: Small constant for numerical stability.
        :type epsilon: float

        :raises ValueError: If epsilon is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply MaxLogit normalization.

        :param inputs: Input logits tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether the layer should behave in
            training mode or inference mode. Not used in this layer.
        :type training: Optional[bool]

        :return: Tensor with L2 normalized logits along the specified axis.
        :rtype: keras.KerasTensor
        """
        # Cast inputs to computation dtype for numerical stability
        inputs = ops.cast(inputs, self.compute_dtype)

        # Compute L2 norm with numerical stability
        squared = ops.square(inputs)
        norm = ops.sqrt(
            ops.sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # L2 normalize
        return inputs / norm

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape tuple (same as input shape).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config


@keras.saving.register_keras_serializable()
class DecoupledMaxLogit(keras.layers.Layer):
    """Decoupled MaxLogit (DML) normalization layer.

    Separates MaxLogit into cosine similarity and L2 norm components with
    learnable weighting. The decomposition computes:
    ``normalized = inputs / ||inputs||_2``,
    ``max_cosine = max(normalized)``,
    ``max_norm = max(||inputs||_2)``,
    ``output = constant × max_cosine + max_norm``.
    This allows analysis of which component drives OOD detection.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │      Input Logits (x)        │
        │      shape: (B, C)           │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   L2 Norm: √(Σ(x²) + ε)      │
        └──────┬───────────────┬───────┘
               │               │
               ▼               ▼
        ┌──────────────┐ ┌─────────────┐
        │ Normalize:   │ │ Max Norm:   │
        │ x / norm     │ │ max(norm)   │
        └──────┬───────┘ └──────┬──────┘
               │                │
               ▼                │
        ┌──────────────┐        │
        │ Max Cosine:  │        │
        │ max(x/norm)  │        │
        └──────┬───────┘        │
               │                │
               ▼                ▼
        ┌──────────────────────────────┐
        │  Combine:                    │
        │  c × max_cosine + max_norm   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Output: (combined,          │
        │    max_cosine, max_norm)     │
        └──────────────────────────────┘

    :param constant: Weight between cosine and L2 components. Must be positive.
        Defaults to 1.0.
    :type constant: float
    :param axis: Axis along which to normalize. Defaults to -1.
    :type axis: int
    :param epsilon: Small constant for numerical stability. Must be positive.
        Defaults to 1e-7.
    :type epsilon: float

    :raises ValueError: If constant or epsilon is not positive.
    """

    def __init__(
        self,
        constant: float = 1.0,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Initialize the DecoupledMaxLogit layer.

        :param constant: Weight between cosine and L2 components. Must be positive.
        :type constant: float
        :param axis: Axis along which to normalize.
        :type axis: int
        :param epsilon: Small constant for numerical stability. Must be positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for the Layer base class.
        :type kwargs: Any

        :raises ValueError: If constant or epsilon is not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(constant, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.constant = constant
        self.axis = axis
        self.epsilon = epsilon

        logger.debug(f"Initialized DecoupledMaxLogit with constant={constant}, axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, constant: float, epsilon: float) -> None:
        """Validate initialization parameters.

        :param constant: Weight between cosine and L2 components.
        :type constant: float
        :param epsilon: Small constant for numerical stability.
        :type epsilon: float

        :raises ValueError: If constant or epsilon is not positive.
        """
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Apply decoupled MaxLogit normalization.

        :param inputs: Input logits tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether the layer should behave in
            training mode or inference mode. Not used in this layer.
        :type training: Optional[bool]

        :return: Tuple of (combined score, MaxCosine component, MaxNorm component).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        inputs = ops.cast(inputs, self.compute_dtype)

        # Compute L2 norm with numerical stability
        squared = ops.square(inputs)
        norm = ops.sqrt(
            ops.sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Compute normalized features (cosine)
        normalized = inputs / norm

        # Get maximum cosine similarity (remove keepdims for final output)
        max_cosine = ops.max(normalized, axis=self.axis)

        # Get maximum norm (squeeze to remove keepdims)
        max_norm = ops.squeeze(ops.max(norm, axis=self.axis), axis=-1)

        # Combine with learned weight
        output = self.constant * max_cosine + max_norm

        return output, max_cosine, max_norm

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Tuple of output shapes for (combined, max_cosine, max_norm).
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Remove the axis dimension
        if self.axis == -1 or self.axis == len(input_shape_list) - 1:
            output_shape = tuple(input_shape_list[:-1])
        else:
            output_shape_list = input_shape_list[:self.axis] + input_shape_list[self.axis + 1:]
            output_shape = tuple(output_shape_list)

        # All three outputs have the same shape
        return (output_shape, output_shape, output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config


@keras.saving.register_keras_serializable()
class DMLPlus(keras.layers.Layer):
    """DML+ layer for separate focal and center OOD detection models.

    Designed for specialized models optimized for different components of
    decoupled MaxLogit. The focal model computes ``MaxCosine = max(x / ||x||_2)``
    for similarity-based OOD detection, while the center model computes
    ``MaxNorm = max(||x||_2)`` for magnitude-based detection. Combining both
    in an ensemble yields improved OOD detection performance.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │      Input Logits (x)        │
        │      shape: (B, C)           │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   L2 Norm: √(Σ(x²) + ε)      │
        └──────┬───────────────┬───────┘
               │               │
               ▼               ▼
        ┌──────────────┐ ┌─────────────┐
        │ Normalize:   │ │  L2 Norm    │
        │ x / norm     │ │  (keepdims) │
        └──────┬───────┘ └──────┬──────┘
               │                │
               ▼                ▼
        ┌──────────────┐ ┌─────────────┐
        │ [focal]      │ │ [center]    │
        │ max(x/norm)  │ │ max(norm),  │
        │ → MaxCosine  │ │ norm_factor │
        └──────────────┘ └─────────────┘

    :param model_type: Type of model, either ``"focal"`` (returns MaxCosine) or
        ``"center"`` (returns MaxNorm and norm factor).
    :type model_type: Literal["focal", "center"]
    :param axis: Axis along which to normalize. Defaults to -1.
    :type axis: int
    :param epsilon: Small constant for numerical stability. Must be positive.
        Defaults to 1e-7.
    :type epsilon: float

    :raises ValueError: If model_type is not ``"focal"`` or ``"center"``.
    :raises ValueError: If epsilon is not positive.
    """

    def __init__(
        self,
        model_type: Literal["focal", "center"],
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Initialize the DMLPlus layer.

        :param model_type: Type of model (``"focal"`` or ``"center"``).
        :type model_type: Literal["focal", "center"]
        :param axis: Axis along which to normalize.
        :type axis: int
        :param epsilon: Small constant for numerical stability. Must be positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for the Layer base class.
        :type kwargs: Any

        :raises ValueError: If model_type is invalid or epsilon is not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(model_type, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.model_type = model_type
        self.axis = axis
        self.epsilon = epsilon

        logger.debug(f"Initialized DMLPlus with model_type={model_type}, axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, model_type: str, epsilon: float) -> None:
        """Validate initialization parameters.

        :param model_type: Type of model (``"focal"`` or ``"center"``).
        :type model_type: str
        :param epsilon: Small constant for numerical stability.
        :type epsilon: float

        :raises ValueError: If model_type is invalid or epsilon is not positive.
        """
        if model_type not in ["focal", "center"]:
            raise ValueError(f"model_type must be 'focal' or 'center', got {model_type}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Apply DML+ normalization based on model type.

        :param inputs: Input logits tensor.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether the layer should behave in
            training mode or inference mode. Not used in this layer.
        :type training: Optional[bool]

        :return: For focal model: MaxCosine score tensor. For center model:
            tuple of (MaxNorm score, normalization factor).
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        inputs = ops.cast(inputs, self.compute_dtype)

        # Compute L2 norm with numerical stability
        squared = ops.square(inputs)
        norm = ops.sqrt(
            ops.sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Compute normalized features
        normalized = inputs / norm

        if self.model_type == "focal":
            # Focal model returns MaxCosine
            return ops.max(normalized, axis=self.axis)
        else:
            # Center model returns MaxNorm and norm factor
            max_norm = ops.squeeze(ops.max(norm, axis=self.axis), axis=-1)
            return max_norm, norm

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape(s) depending on model type.
        :rtype: Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]]
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Remove the axis dimension for reduced shape
        if self.axis == -1 or self.axis == len(input_shape_list) - 1:
            reduced_shape = tuple(input_shape_list[:-1])
        else:
            reduced_shape_list = input_shape_list[:self.axis] + input_shape_list[self.axis + 1:]
            reduced_shape = tuple(reduced_shape_list)

        if self.model_type == "focal":
            return reduced_shape
        else:
            # Center model returns (max_norm, norm_factor)
            return (reduced_shape, input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "model_type": self.model_type,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
