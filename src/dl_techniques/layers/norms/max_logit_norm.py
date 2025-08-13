"""
MaxLogit Normalization Implementations for Out-of-Distribution Detection.

This module implements variants of MaxLogit normalization for out-of-distribution detection:
1. Basic MaxLogit normalization
2. Decoupled MaxLogit (DML) normalization that separates cosine and L2 components
3. DML+ with dedicated models for cosine and norm components

The implementations follow the paper:
"Decoupling MaxLogit for Out-of-Distribution Detection"
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MaxLogitNorm(keras.layers.Layer):
    """Basic MaxLogit normalization layer for out-of-distribution detection.

    Applies L2 normalization and cosine similarity to logits for better OOD detection.
    This is the base implementation that normalizes logits using their L2 norm.

    The layer computes: output = inputs / ||inputs||_2

    Args:
        axis: Integer, the axis along which to normalize. Defaults to -1 (last axis).
        epsilon: Float, small constant for numerical stability. Must be positive.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary. The most common use case is a 2D input with shape
        `(batch_size, features)`.

    Output shape:
        Same as input shape.

    Returns:
        Tensor with L2 normalized logits along the specified axis.

    Raises:
        ValueError: If epsilon is not positive.

    Example:
        >>> layer = MaxLogitNorm(axis=-1, epsilon=1e-7)
        >>> logits = keras.random.normal((2, 10))
        >>> normalized = layer(logits)
        >>> print(normalized.shape)
        (2, 10)
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._validate_inputs(epsilon)
        self.axis = axis
        self.epsilon = epsilon
        self._build_input_shape = None

    def _validate_inputs(self, epsilon: float) -> None:
        """Validate initialization parameters.

        Args:
            epsilon: Small constant for numerical stability.

        Raises:
            ValueError: If epsilon is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape):
        """Build the layer (no weights to create).

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        self._build_input_shape = input_shape
        super().build(input_shape)

    def call(
        self,
        inputs,
        training: Optional[bool] = None
    ):
        """Apply MaxLogit normalization.

        Args:
            inputs: Input logits tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer.

        Returns:
            Tensor with L2 normalized logits.
        """
        # Cast inputs to computation dtype
        inputs = ops.cast(inputs, self.compute_dtype)

        # Compute L2 norm with numerical stability
        squared = ops.square(inputs)
        norm = ops.sqrt(
            ops.sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # L2 normalize
        return inputs / norm

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DecoupledMaxLogit(keras.layers.Layer):
    """Decoupled MaxLogit (DML) normalization layer.

    Separates MaxLogit into cosine similarity and L2 norm components
    with learnable weighting between them. This allows better understanding
    of which component contributes to out-of-distribution detection.

    The layer computes:
    - normalized = inputs / ||inputs||_2
    - max_cosine = max(normalized)
    - max_norm = max(||inputs||_2)
    - output = constant * max_cosine + max_norm

    Args:
        constant: Float, weight between cosine and L2 components. Must be positive.
        axis: Integer, the axis along which to normalize. Defaults to -1.
        epsilon: Float, small constant for numerical stability. Must be positive.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary. The most common use case is a 2D input with shape
        `(batch_size, features)`.

    Output shape:
        If axis=-1 and input shape is (batch_size, features), output shapes are:
        - Combined output: (batch_size,)
        - Max cosine: (batch_size,)
        - Max norm: (batch_size,)

    Returns:
        Tuple of three tensors:
        - Combined score (constant * max_cosine + max_norm)
        - MaxCosine component
        - MaxNorm component

    Raises:
        ValueError: If constant or epsilon is not positive.

    Example:
        >>> layer = DecoupledMaxLogit(constant=1.0, axis=-1, epsilon=1e-7)
        >>> logits = keras.random.normal((2, 10))
        >>> combined, max_cos, max_norm = layer(logits)
        >>> print(combined.shape, max_cos.shape, max_norm.shape)
        (2,) (2,) (2,)
    """

    def __init__(
        self,
        constant: float = 1.0,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._validate_inputs(constant, epsilon)
        self.constant = constant
        self.axis = axis
        self.epsilon = epsilon
        self._build_input_shape = None

    def _validate_inputs(self, constant: float, epsilon: float) -> None:
        """Validate initialization parameters.

        Args:
            constant: Weight between cosine and L2 components.
            epsilon: Small constant for numerical stability.

        Raises:
            ValueError: If constant or epsilon is not positive.
        """
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape):
        """Build the layer (no weights to create).

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        self._build_input_shape = input_shape
        super().build(input_shape)

    def call(
        self,
        inputs,
        training: Optional[bool] = None
    ) -> Tuple:
        """Apply decoupled MaxLogit normalization.

        Args:
            inputs: Input logits tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer.

        Returns:
            Tuple of three tensors:
            - Combined score (constant * max_cosine + max_norm)
            - MaxCosine component
            - MaxNorm component
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

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Tuple of output shapes for (combined, max_cosine, max_norm).
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
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DMLPlus(keras.layers.Layer):
    """DML+ implementation for separate focal and center models.

    This layer is designed to be used in separate models optimized for different
    components of the decoupled MaxLogit. The focal model is optimized for MaxCosine
    while the center model is optimized for MaxNorm.

    Args:
        model_type: String, either "focal" or "center" indicating the model type.
        axis: Integer, the axis along which to normalize. Defaults to -1.
        epsilon: Float, small constant for numerical stability. Must be positive.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary. The most common use case is a 2D input with shape
        `(batch_size, features)`.

    Output shape:
        - For focal model: Reduced by one dimension along the specified axis.
        - For center model: Tuple of (reduced_shape, input_shape) for
          (max_norm, norm_factor).

    Returns:
        - For focal model: MaxCosine score tensor.
        - For center model: Tuple of (MaxNorm score, normalization factor).

    Raises:
        ValueError: If model_type is not "focal" or "center", or if epsilon is not positive.

    Example:
        >>> focal_layer = DMLPlus(model_type="focal", axis=-1)
        >>> center_layer = DMLPlus(model_type="center", axis=-1)
        >>> logits = keras.random.normal((2, 10))
        >>> max_cosine = focal_layer(logits)
        >>> max_norm, norm_factor = center_layer(logits)
        >>> print(max_cosine.shape, max_norm.shape, norm_factor.shape)
        (2,) (2,) (2, 1)
    """

    def __init__(
        self,
        model_type: Literal["focal", "center"],
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._validate_inputs(model_type, epsilon)
        self.model_type = model_type
        self.axis = axis
        self.epsilon = epsilon
        self._build_input_shape = None

    def _validate_inputs(self, model_type: str, epsilon: float) -> None:
        """Validate initialization parameters.

        Args:
            model_type: Type of model ("focal" or "center").
            epsilon: Small constant for numerical stability.

        Raises:
            ValueError: If model_type is invalid or epsilon is not positive.
        """
        if model_type not in ["focal", "center"]:
            raise ValueError(f"model_type must be 'focal' or 'center', got {model_type}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape):
        """Build the layer (no weights to create).

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        self._build_input_shape = input_shape
        super().build(input_shape)

    def call(
        self,
        inputs,
        training: Optional[bool] = None
    ) -> Union[tuple, Any]:
        """Apply DML+ normalization based on model type.

        Args:
            inputs: Input logits tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer.

        Returns:
            For focal model: MaxCosine score tensor.
            For center model: Tuple of (MaxNorm score, normalization factor).
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

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape(s) depending on model type.
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
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "model_type": self.model_type,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
