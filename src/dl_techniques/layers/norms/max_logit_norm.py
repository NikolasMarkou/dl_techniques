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
    """
    Basic MaxLogit normalization layer for out-of-distribution detection.

    Applies L2 normalization to logits for better OOD detection by separating
    the magnitude and direction components of logits. This helps distinguish
    between in-distribution and out-of-distribution samples.

    The layer computes:

    .. math::
        \\text{output} = \\frac{\\text{inputs}}{||\\text{inputs}||_2}

    Args:
        axis: Axis along which to normalize. Typically -1 for the class dimension.
            Defaults to -1.
        epsilon: Small constant for numerical stability. Must be positive.
            Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary. The most common use case is a 2D input with shape
        ``(batch_size, features)``.

    Output shape:
        Same as input shape.

    Raises:
        ValueError: If epsilon is not positive.

    Example:
        Basic usage for OOD detection:

        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.max_logit_norm import MaxLogitNorm

            # Standard classification model with MaxLogit normalization
            inputs = keras.Input(shape=(784,))
            x = keras.layers.Dense(256, activation='relu')(inputs)
            logits = keras.layers.Dense(10)(x)  # Raw logits

            # Apply MaxLogit normalization for OOD detection
            normalized_logits = MaxLogitNorm(axis=-1)(logits)
            outputs = keras.layers.Softmax()(normalized_logits)

            model = keras.Model(inputs, outputs)

        Integration with uncertainty estimation:

        .. code-block:: python

            def create_ood_detector(num_classes):
                inputs = keras.Input(shape=(input_dim,))

                # Feature extraction
                features = keras.layers.Dense(512, activation='relu')(inputs)
                logits = keras.layers.Dense(num_classes)(features)

                # MaxLogit normalization for OOD detection
                normalized_logits = MaxLogitNorm()(logits)
                probabilities = keras.layers.Softmax()(normalized_logits)

                # The max probability can be used as confidence score
                return keras.Model(inputs, [probabilities, normalized_logits])

        Custom epsilon for numerical stability:

        .. code-block:: python

            # For mixed precision training, might need larger epsilon
            max_logit = MaxLogitNorm(axis=-1, epsilon=1e-6)
            normalized = max_logit(logits)

    Note:
        - This layer performs only computation on inputs without creating any weights
        - Useful for improving out-of-distribution detection performance
        - Can be combined with temperature scaling for calibration
        - Works well with mixed precision training
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MaxLogitNorm layer.

        Args:
            axis: Axis along which to normalize.
            epsilon: Small constant for numerical stability. Must be positive.
            **kwargs: Additional keyword arguments for the Layer base class.

        Raises:
            ValueError: If epsilon is not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.axis = axis
        self.epsilon = epsilon

        logger.debug(f"Initialized MaxLogitNorm with axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, epsilon: float) -> None:
        """
        Validate initialization parameters.

        Args:
            epsilon: Small constant for numerical stability.

        Raises:
            ValueError: If epsilon is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply MaxLogit normalization.

        Args:
            inputs: Input logits tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer.

        Returns:
            Tensor with L2 normalized logits along the specified axis.
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
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        Returns:
            Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config


@keras.saving.register_keras_serializable()
class DecoupledMaxLogit(keras.layers.Layer):
    """
    Decoupled MaxLogit (DML) normalization layer.

    Separates MaxLogit into cosine similarity and L2 norm components with learnable
    weighting between them. This allows better understanding of which component
    contributes to out-of-distribution detection.

    The layer computes:
    - normalized = inputs / ||inputs||_2
    - max_cosine = max(normalized)
    - max_norm = max(||inputs||_2)
    - output = constant × max_cosine + max_norm

    Args:
        constant: Weight between cosine and L2 components. Must be positive.
            Defaults to 1.0.
        axis: Axis along which to normalize. Defaults to -1.
        epsilon: Small constant for numerical stability. Must be positive.
            Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary. The most common use case is a 2D input with shape
        ``(batch_size, features)``.

    Output shape:
        If axis=-1 and input shape is (batch_size, features), all output shapes are:
        ``(batch_size,)`` for each of the three returned tensors.

    Returns:
        Tuple of three tensors:
        - Combined score (constant × max_cosine + max_norm)
        - MaxCosine component
        - MaxNorm component

    Raises:
        ValueError: If constant or epsilon is not positive.

    Example:
        Basic usage for decoupled OOD detection:

        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.max_logit_norm import DecoupledMaxLogit

            # Create model with decoupled MaxLogit
            inputs = keras.Input(shape=(784,))
            x = keras.layers.Dense(256, activation='relu')(inputs)
            logits = keras.layers.Dense(10)(x)

            # Get separate cosine and norm components
            combined, max_cos, max_norm = DecoupledMaxLogit(constant=1.0)(logits)

            # Use different components for different purposes
            model = keras.Model(inputs, [combined, max_cos, max_norm])

        Analysis of OOD detection components:

        .. code-block:: python

            def analyze_ood_components():
                # Test with different weighting
                dml_layer = DecoupledMaxLogit(constant=0.5, axis=-1)

                # Separate analysis of cosine vs norm contribution
                combined, cosine, norm = dml_layer(test_logits)

                # cosine: measures similarity to training distribution
                # norm: measures confidence/magnitude
                # combined: balanced score for OOD detection

                return combined, cosine, norm

        Custom weighting for specific datasets:

        .. code-block:: python

            # For datasets where cosine similarity is more important
            dml_cosine_heavy = DecoupledMaxLogit(constant=2.0)

            # For datasets where norm is more informative
            dml_norm_heavy = DecoupledMaxLogit(constant=0.5)

    Note:
        - Returns three separate components for detailed analysis
        - Constant parameter controls the balance between cosine and norm
        - Useful for understanding which component drives OOD detection
        - Can be used to train specialized models for each component
    """

    def __init__(
        self,
        constant: float = 1.0,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """
        Initialize the DecoupledMaxLogit layer.

        Args:
            constant: Weight between cosine and L2 components. Must be positive.
            axis: Axis along which to normalize.
            epsilon: Small constant for numerical stability. Must be positive.
            **kwargs: Additional keyword arguments for the Layer base class.

        Raises:
            ValueError: If constant or epsilon is not positive.
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
        """
        Validate initialization parameters.

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

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Apply decoupled MaxLogit normalization.

        Args:
            inputs: Input logits tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer.

        Returns:
            Tuple of three tensors:
            - Combined score (constant × max_cosine + max_norm)
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

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

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
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        Returns:
            Dictionary containing all constructor arguments.
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
    """
    DML+ implementation for separate focal and center models.

    This layer is designed to be used in separate models optimized for different
    components of the decoupled MaxLogit. The focal model is optimized for MaxCosine
    while the center model is optimized for MaxNorm.

    Args:
        model_type: Type of model, either "focal" or "center".
            - "focal": Returns MaxCosine component for similarity-based OOD detection
            - "center": Returns MaxNorm component for magnitude-based OOD detection
        axis: Axis along which to normalize. Defaults to -1.
        epsilon: Small constant for numerical stability. Must be positive.
            Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary. The most common use case is a 2D input with shape
        ``(batch_size, features)``.

    Output shape:
        - For focal model: Reduced by one dimension along the specified axis.
        - For center model: Tuple of (reduced_shape, input_shape) for
          (max_norm, norm_factor).

    Returns:
        - For focal model: MaxCosine score tensor.
        - For center model: Tuple of (MaxNorm score, normalization factor).

    Raises:
        ValueError: If model_type is not "focal" or "center".
        ValueError: If epsilon is not positive.

    Example:
        Separate models for focal and center components:

        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.max_logit_norm import DMLPlus

            # Create separate models for different components
            def create_focal_model():
                inputs = keras.Input(shape=(784,))
                x = keras.layers.Dense(256, activation='relu')(inputs)
                logits = keras.layers.Dense(10)(x)

                # Focal model optimized for cosine similarity
                max_cosine = DMLPlus(model_type="focal")(logits)
                return keras.Model(inputs, max_cosine)

            def create_center_model():
                inputs = keras.Input(shape=(784,))
                x = keras.layers.Dense(256, activation='relu')(inputs)
                logits = keras.layers.Dense(10)(x)

                # Center model optimized for norm magnitude
                max_norm, norm_factor = DMLPlus(model_type="center")(logits)
                return keras.Model(inputs, [max_norm, norm_factor])

        Training specialized models:

        .. code-block:: python

            # Train focal model on cosine-based objectives
            focal_model = create_focal_model()
            focal_model.compile(optimizer='adam', loss='mse')

            # Train center model on norm-based objectives
            center_model = create_center_model()
            center_model.compile(optimizer='adam', loss=['mse', 'mae'])

        Ensemble inference:

        .. code-block:: python

            def ensemble_ood_detection(inputs):
                # Get components from specialized models
                max_cosine = focal_model(inputs)
                max_norm, _ = center_model(inputs)

                # Combine for final OOD score
                combined_score = 0.7 * max_cosine + 0.3 * max_norm
                return combined_score

    Note:
        - Designed for training separate specialized models
        - Focal model focuses on directional similarity
        - Center model focuses on magnitude information
        - Can be combined in ensemble for improved OOD detection
    """

    def __init__(
        self,
        model_type: Literal["focal", "center"],
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """
        Initialize the DMLPlus layer.

        Args:
            model_type: Type of model ("focal" or "center").
            axis: Axis along which to normalize.
            epsilon: Small constant for numerical stability. Must be positive.
            **kwargs: Additional keyword arguments for the Layer base class.

        Raises:
            ValueError: If model_type is invalid or epsilon is not positive.
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
        """
        Validate initialization parameters.

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

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Apply DML+ normalization based on model type.

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

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

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
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        Returns:
            Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "model_type": self.model_type,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
