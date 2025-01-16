"""
MaxLogit Normalization Implementations.

This module implements variants of MaxLogit normalization for out-of-distribution detection:
1. Basic MaxLogit normalization
2. Decoupled MaxLogit (DML) normalization that separates cosine and L2 components
3. DML+ with dedicated models for cosine and norm components

The implementations follow the paper:
"Decoupling MaxLogit for Out-of-Distribution Detection"
"""

import tensorflow as tf
from keras import Layer
from typing import Optional, Tuple, Dict, Any, Union, Literal


class MaxLogitNorm(Layer):
    """
    Basic MaxLogit normalization layer.

    Applies L2 normalization and cosine similarity to logits for better OOD detection.
    This is the base implementation before decoupling.
    """

    def __init__(
            self,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        """
        Initialize MaxLogit normalization.

        Args:
            axis: Dimension along which to normalize
            epsilon: Small constant for numerical stability
        """
        super().__init__(**kwargs)
        self._validate_inputs(epsilon)
        self.axis = axis
        self.epsilon = epsilon

    def _validate_inputs(self, epsilon: float) -> None:
        """Validate initialization parameters."""
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Apply MaxLogit normalization.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode (unused)

        Returns:
            Normalized logits
        """
        # Cast inputs to float
        inputs = tf.cast(inputs, self.compute_dtype)

        # Compute L2 norm
        squared = tf.square(inputs)
        norm = tf.sqrt(
            tf.reduce_sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # L2 normalize
        return inputs / norm

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


class DecoupledMaxLogit(Layer):
    """
    Decoupled MaxLogit (DML) normalization layer.

    Separates MaxLogit into cosine similarity and L2 norm components
    with learnable weighting between them.
    """

    def __init__(
            self,
            constant: float = 1.0,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        """
        Initialize DML normalization.

        Args:
            constant: Weight between cosine and L2 components
            axis: Dimension along which to normalize
            epsilon: Small constant for numerical stability
        """
        super().__init__(**kwargs)
        self._validate_inputs(constant, epsilon)
        self.constant = constant
        self.axis = axis
        self.epsilon = epsilon

    def _validate_inputs(self, constant: float, epsilon: float) -> None:
        """Validate initialization parameters."""
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Apply decoupled MaxLogit normalization.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode

        Returns:
            Tuple of:
            - Normalized logits
            - MaxCosine component
            - MaxNorm component
        """
        inputs = tf.cast(inputs, self.compute_dtype)

        # Compute L2 norm
        squared = tf.square(inputs)
        norm = tf.sqrt(
            tf.reduce_sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Compute normalized features (cosine)
        normalized = inputs / norm

        # Get maximum cosine similarity
        max_cosine = tf.reduce_max(normalized, axis=self.axis)

        # Get norm
        max_norm = tf.reduce_max(norm, axis=self.axis)

        # Combine with learned weight
        output = self.constant * max_cosine + max_norm

        return output, max_cosine, max_norm

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


class DMLPlus:
    """
    DML+ implementation that uses separate models for cosine and norm components.

    This class manages two models:
    1. Focal model optimized for MaxCosine
    2. Center model optimized for MaxNorm
    """

    def __init__(
            self,
            model_type: Literal["focal", "center"],
            axis: int = -1,
            epsilon: float = 1e-7
    ):
        """
        Initialize DML+ model.

        Args:
            model_type: Whether this is the focal or center model
            axis: Dimension along which to normalize
            epsilon: Small constant for numerical stability
        """
        self.model_type = model_type
        self.axis = axis
        self.epsilon = epsilon

    def __call__(
            self,
            inputs: tf.Tensor,
            training: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Apply DML+ normalization based on model type.

        Args:
            inputs: Input logits tensor
            training: Whether in training mode

        Returns:
            For focal model: MaxCosine score
            For center model: (MaxNorm score, normalization factor)
        """
        inputs = tf.cast(inputs, tf.float32)

        # Compute L2 norm
        squared = tf.square(inputs)
        norm = tf.sqrt(
            tf.reduce_sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Compute normalized features
        normalized = inputs / norm

        if self.model_type == "focal":
            # Focal model returns MaxCosine
            return tf.reduce_max(normalized, axis=self.axis)

        else:
            # Center model returns MaxNorm and norm factor
            max_norm = tf.reduce_max(norm, axis=self.axis)
            return max_norm, norm
